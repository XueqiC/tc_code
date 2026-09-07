"""Blocked Fisher geometry and an implicit, realizable parameter decoder.

H = diag(F) + eps I; F is nonnegative, unnormalised unless its saved provenance
explicitly says otherwise. Nothing estimates Fisher from held-out deltas.
For training delta columns D, diagonalise the SMALL Gram matrix D^T H D and
store A = V Lambda^(-1/2). The decoder is C = D A, so C^T H C = I.
Q = H^(1/2) C is Euclidean orthonormal in whitened coordinates. Calling Q
itself H-orthonormal before applying H^(-1/2) would apply the metric twice.

Only training-fold deltas may be passed to build_basis. Their amplitude is
preserved. Inputs may be read-only numpy memmaps; no d-by-d or d-by-r basis is
allocated. Workspace is O(block_size * number_of_sources + sources^2).
Vector outputs accept writable float64 memmaps. The saved coefficient matrix
and immutable delta references implement the decoder without a random sketch.
Residual norms use H, and are computed directly (not by subtracting nearly
equal squared norms). All parameter vectors use deltas.parameter_layout order.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .deltas import BLOCK_SIZE, canonical_hash, parameter_layout, write_json_exclusive


def _array(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().float().numpy() if value.dtype == torch.bfloat16 else value.detach().cpu().numpy()
    return np.asanyarray(value)


def _vector(value, size):
    if np.shape(value) != (size,):
        raise ValueError(f"expected a flat vector of length {size}")


def _output(out, size):
    if out is None:
        return np.empty(size, dtype=np.float64)
    _vector(out, size)
    if out.dtype != np.float64 or not out.flags.writeable:
        raise ValueError("out must be a writable float64 vector")
    return out


@dataclass
class DiagonalMetric:
    diagonal: np.ndarray
    layout_hash: str
    eps: float
    version: str
    fisher_hash: str
    block_size: int = BLOCK_SIZE

    def __post_init__(self):
        if self.block_size <= 0 or self.diagonal.ndim != 1 or not self.layout_hash:
            raise ValueError("invalid metric shape, block size or layout hash")
        if not np.isfinite(self.eps) or self.eps <= 0 or not self.version:
            raise ValueError("positive finite damping and a Fisher version are required")
        for sl in self.blocks():
            if not np.isfinite(self.diagonal[sl]).all() or np.any(self.diagonal[sl] <= 0):
                raise ValueError("H must be finite and strictly positive")

    @property
    def size(self):
        return len(self.diagonal)

    def blocks(self):
        for start in range(0, self.size, self.block_size):
            yield slice(start, min(start + self.block_size, self.size))

    def sqrt_delta(self, delta, out=None):
        """Return H^(1/2) delta, preserving its magnitude."""
        _vector(delta, self.size)
        out = _output(out, self.size)
        if np.shares_memory(out, self.diagonal):
            raise ValueError("output must not overwrite H")
        for sl in self.blocks():
            block = np.asarray(delta[sl], dtype=np.float64)
            if not np.isfinite(block).all():
                raise ValueError("nonfinite delta")
            out[sl] = np.sqrt(self.diagonal[sl]) * block
        return out

    def norm(self, delta):
        _vector(delta, self.size)
        total = 0.0
        for sl in self.blocks():
            block = np.asarray(delta[sl], dtype=np.float64)
            if not np.isfinite(block).all():
                raise ValueError("nonfinite delta")
            total += float((block * self.diagonal[sl]) @ block)
        return float(np.sqrt(total))


def from_fisher(fisher, layout, *, eps: float, layout_hash: str | None = None,
                version: str = "fisher-v1", block_size: int = BLOCK_SIZE, out=None):
    """Load/align named Fisher vectors, or a flat vector WITH its layout hash.

    ``layout`` is a model or parameter_layout(model). Accepted saved formats:
    .npy flat vector (requires layout_hash), .npz named tensors or keys
    ``fisher`` and ``layout_hash``, and .pt/.pth weights-only tensor mappings.
    Named values may have the parameter shape or be flat; missing/extra names,
    a mismatched hash, wrong lengths, negative Fisher and NaN/Inf are errors.
    ``out`` can be a float64 memmap for H's diagonal.
    """
    import hashlib

    entries = parameter_layout(layout) if hasattr(layout, "named_parameters") else layout
    expected_hash = canonical_hash(entries)
    if not np.isfinite(eps) or eps <= 0 or block_size <= 0:
        raise ValueError("eps and block_size must be positive")
    if isinstance(fisher, (str, Path)):
        path = Path(fisher)
        if path.suffix == ".npy":
            fisher = np.load(path, mmap_mode="r", allow_pickle=False)
        elif path.suffix == ".npz":
            with np.load(path, allow_pickle=False) as data:
                fisher = {key: data[key] for key in data.files}
        elif path.suffix in {".pt", ".pth"}:
            fisher = torch.load(path, map_location="cpu", weights_only=True)
        else:
            raise ValueError("unsupported Fisher format")
    if isinstance(fisher, Mapping) and "fisher" in fisher and "layout_hash" in fisher:
        saved_hash = str(np.asarray(fisher["layout_hash"]).item())
        if layout_hash is not None and saved_hash != layout_hash:
            raise ValueError("conflicting Fisher layout hashes")
        layout_hash, fisher = saved_hash, fisher["fisher"]
    if layout_hash is not None and layout_hash != expected_hash:
        raise ValueError("Fisher layout hash mismatch")
    size = sum(entry["numel"] for entry in entries)
    diagonal = _output(out, size)
    if isinstance(fisher, Mapping):
        if set(fisher) != {e["name"] for e in entries}:
            raise ValueError("Fisher parameter names do not match the layout")
        vectors = []
        for entry in entries:
            value = _array(fisher[entry["name"]])
            if value.shape not in {tuple(entry["shape"]), (entry["numel"],)}:
                raise ValueError(f"Fisher shape mismatch: {entry['name']}")
            vectors.append((entry["offset"], value.reshape(-1)))
    else:
        if layout_hash is None:
            raise ValueError("flat Fisher requires its saved layout hash")
        fisher = _array(fisher)
        _vector(fisher, size)
        vectors = [(0, fisher)]
    digest = hashlib.sha256()
    for offset, vector in vectors:
        for start in range(0, len(vector), block_size):
            block = np.asarray(vector[start:start + block_size], dtype=np.float64)
            if not np.isfinite(block).all() or np.any(block < 0):
                raise ValueError("Fisher must be finite and nonnegative")
            digest.update(block.astype("<f8", copy=False).tobytes())
            diagonal[offset + start:offset + start + len(block)] = block + eps
    return DiagonalMetric(diagonal, expected_hash, eps, version, digest.hexdigest(), block_size)


def whiten(delta, H: DiagonalMetric, out=None):
    return H.sqrt_delta(delta, out)


@dataclass
class Basis:
    deltas: tuple
    H: DiagonalMetric
    coefficients: np.ndarray
    eigenvalues: np.ndarray
    version: str
    rtol: float
    atol: float

    @property
    def rank(self):
        return self.coefficients.shape[1]

    def _columns(self, sl):
        if not self.deltas:
            return np.empty((sl.stop - sl.start, 0), dtype=np.float64)
        block = np.column_stack([np.asarray(d[sl], dtype=np.float64) for d in self.deltas])
        if not np.isfinite(block).all():
            raise ValueError("nonfinite training delta")
        return block

    def encode(self, delta):
        """Return (x = C^T H delta, ||delta - Cx||_H), including held-out residual."""
        _vector(delta, self.H.size)
        x = np.zeros(self.rank)
        for sl in self.H.blocks():
            vector = np.asarray(delta[sl], dtype=np.float64)
            if not np.isfinite(vector).all():
                raise ValueError("nonfinite delta")
            x += self.coefficients.T @ (self._columns(sl).T @ (self.H.diagonal[sl] * vector))
        return x, self._residual(delta, x)[1]

    def decode(self, x, residual=None, out=None):
        """Apply C implicitly, optionally adding a supplied orthogonal residual.

        Use residual(delta) to obtain that residual. This function does not
        discard or silently reproject caller-provided residuals.
        """
        x = np.asarray(x, dtype=np.float64)
        _vector(x, self.rank)
        if not np.isfinite(x).all():
            raise ValueError("nonfinite coordinates")
        if residual is not None:
            _vector(residual, self.H.size)
        out = _output(out, self.H.size)
        if np.shares_memory(out, self.H.diagonal):
            raise ValueError("output must not overwrite H")
        if any(np.shares_memory(out, d) for d in self.deltas):
            raise ValueError("output must not overwrite training deltas")
        weights = self.coefficients @ x
        for sl in self.H.blocks():
            block = self._columns(sl) @ weights
            if residual is not None:
                block += residual[sl]
            if not np.isfinite(block).all():
                raise ValueError("nonfinite decoded delta")
            out[sl] = block
        return out

    def _residual(self, delta, x, out=None):
        weights, norm2 = self.coefficients @ x, 0.0
        for sl in self.H.blocks():
            block = np.asarray(delta[sl], dtype=np.float64) - self._columns(sl) @ weights
            norm2 += float((block * self.H.diagonal[sl]) @ block)
            if out is not None:
                out[sl] = block
        return out, float(np.sqrt(norm2))

    def residual(self, delta, out=None):
        """Return (orthogonal residual vector, H norm); supports a memmap output."""
        x, _ = self.encode(delta)
        out = _output(out, self.H.size)
        if np.shares_memory(out, self.H.diagonal):
            raise ValueError("output must not overwrite H")
        if any(np.shares_memory(out, d) for d in self.deltas):
            raise ValueError("output must not overwrite training deltas")
        return self._residual(delta, x, out)

    def save(self, path, delta_references):
        """Save small decoder coefficients and ordered immutable delta references.

        References must contain path and content_hash for each training delta;
        reload those vectors and the matching Fisher before reconstructing Basis.
        """
        if len(delta_references) != len(self.deltas) or any(
                not r.get("path") or not r.get("content_hash") for r in delta_references):
            raise ValueError("one path/content_hash reference per training delta is required")
        write_json_exclusive(path, dict(version=self.version, layout_hash=self.H.layout_hash,
                             fisher_hash=self.H.fisher_hash, fisher_version=self.H.version,
                             eps=self.H.eps, rtol=self.rtol, atol=self.atol,
                             coefficients=self.coefficients.tolist(), rank=self.rank,
                             eigenvalues=self.eigenvalues.tolist(), deltas=delta_references))


def build_basis(deltas: list, H: DiagonalMetric, *, rtol=1e-10, atol=0.0,
                version="basis-v1") -> Basis:
    """Fit the uncentered training span via a blocked small Gram eigensystem.

    Thresholds apply to Gram eigenvalues; zero/repeated/dependent updates are
    valid, including an entirely empty rank-zero span. Inputs remain referenced
    and must be treated as immutable throughout the decoder's lifetime.
    """
    if not np.isfinite([rtol, atol]).all() or rtol < 0 or atol < 0 or not version:
        raise ValueError("invalid basis tolerances/version")
    for delta in deltas:
        _vector(delta, H.size)
    basis = Basis(tuple(deltas), H, np.empty((len(deltas), 0)), np.empty(0), version, rtol, atol)
    gram = np.zeros((len(deltas), len(deltas)), dtype=np.float64)
    for sl in H.blocks():
        block = basis._columns(sl)
        gram += block.T @ (H.diagonal[sl, None] * block)
    if not len(deltas):
        return basis
    if not np.isfinite(gram).all():
        raise ValueError("Fisher Gram matrix overflow")
    values, vectors = np.linalg.eigh((gram + gram.T) * 0.5)
    cutoff = max(atol, rtol * max(0.0, float(values[-1])))
    keep = values > cutoff
    basis.eigenvalues = values[keep][::-1]
    basis.coefficients = vectors[:, keep][:, ::-1] / np.sqrt(basis.eigenvalues)
    return basis


def encode(delta, basis: Basis):
    return basis.encode(delta)


def decode(x, basis: Basis, residual=None, out=None):
    return basis.decode(x, residual, out)
