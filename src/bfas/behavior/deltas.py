"""Lossless replay of differences in ordered, trainable parameter coordinates.

The layout is ``named_parameters()`` order, including BOTH LoRA A and B (or
all trainables of a synthetic model), flattened in contiguous C order. Frozen
weights must come from the same checkpoint. Parameter storage may be fp32,
fp16 or bf16; differences are computed after promotion, never in bf16.

Ordinary local updates have exactly representable fp32 differences. For large
changes where fp32 subtraction loses bits, an additional fp32 roundoff tensor
is saved. Replay evaluates base + scale * difference, then its scaled roundoff
in fp64 before casting once to parameter dtype. Unit-scale replay is checked
bit for bit at capture time; unrepresentable/nonfinite updates are rejected.
The primary values and roundoff together describe the actual factor update.

``ell`` belongs to the scorer, not these coordinates. Cache callers must put
the shared gradient/evaluation ``ell`` option (``full_sequence`` or
``length_normalized``), scoring precision and decoding settings in eval_config.
Artifacts are immutable: existing destinations are never overwritten.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import tempfile

import numpy as np
import torch


FORMAT_VERSION = "behavior-delta-v1"
BLOCK_SIZE = 65536


def canonical_hash(value) -> str:
    """SHA256 of strict, canonical JSON (NaN and opaque objects are errors)."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def content_key(*, model_id: str, content_hashes: Mapping, preprocessing: Mapping,
                parameter_layout: str, fingerprint_version: str,
                basis_version: str, eval_config: Mapping) -> str:
    """Include content identities, never just trajectory IDs, in a cache key.

    ``content_hashes`` should include checkpoint, state/prompt, teacher and
    student text hashes. Preprocessing requires prompt_cap and prompt_side
    (head/tail means which portion is RETAINED, matching appworld_train).
    Include tokenizer/template versions and response caps there as applicable.
    """
    if preprocessing.get("prompt_side") not in {"head", "tail"}:
        raise ValueError("preprocessing requires prompt_side=head or tail")
    if not isinstance(preprocessing.get("prompt_cap"), int) or preprocessing["prompt_cap"] <= 0:
        raise ValueError("preprocessing requires a positive prompt_cap")
    if not all((model_id, content_hashes, parameter_layout, fingerprint_version, basis_version)):
        raise ValueError("cache identity fields must not be empty")
    return canonical_hash(dict(schema=FORMAT_VERSION, model_id=model_id,
                               content_hashes=dict(content_hashes),
                               preprocessing=dict(preprocessing), layout_hash=parameter_layout,
                               fingerprint_version=fingerprint_version,
                               basis_version=basis_version, eval_config=dict(eval_config)))


def parameter_layout(model) -> list[dict]:
    """Return explicit ordered names, shapes, storage dtypes and flat offsets."""
    result, offset = [], 0
    for name, p in model.named_parameters():
        if p.requires_grad:
            result.append(dict(name=name, shape=list(p.shape), dtype=str(p.dtype),
                               stride=list(p.stride()), numel=p.numel(), offset=offset))
            offset += p.numel()
    if not result:
        raise ValueError("model has no trainable parameters")
    return result


def layout_hash(model) -> str:
    return canonical_hash(parameter_layout(model))


def tensor_state_hash(state: Mapping[str, torch.Tensor]) -> str:
    """Hash exact tensor bytes, names, shapes and dtypes, with bounded copies."""
    digest = hashlib.sha256()
    for name, tensor in state.items():
        digest.update(canonical_hash([name, list(tensor.shape), str(tensor.dtype)]).encode())
        flat = tensor.detach().reshape(-1)
        for start in range(0, flat.numel(), BLOCK_SIZE):
            raw = flat[start:start + BLOCK_SIZE].cpu().contiguous().view(torch.uint8)
            digest.update(raw.numpy().tobytes())
    return digest.hexdigest()


def file_hash(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_exclusive(path, value) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with open(path, "x", encoding="utf-8") as handle:
        handle.write(payload)


@dataclass
class Delta:
    tensors: dict[str, torch.Tensor]
    manifest: dict
    roundoff: dict[str, torch.Tensor] = field(default_factory=dict)

    def flat(self, out=None):
        """Flatten in manifest order; an optional float64 memmap avoids RAM output."""
        size = sum(entry["numel"] for entry in self.manifest["layout"])
        if out is None:
            out = np.empty(size, dtype=np.float64)
        if out.shape != (size,) or out.dtype != np.float64:
            raise ValueError("out must be a float64 vector of layout size")
        for entry in self.manifest["layout"]:
            name, offset, n = entry["name"], entry["offset"], entry["numel"]
            high = self.tensors[name].reshape(-1)
            low = self.roundoff.get(name)
            low = low.reshape(-1) if low is not None else None
            for start in range(0, n, BLOCK_SIZE):
                stop = min(start + BLOCK_SIZE, n)
                out[offset + start:offset + stop] = high[start:stop].numpy()
                if low is not None:
                    out[offset + start:offset + stop] += low[start:stop].numpy()
        return out


def _capture_device_delta(model, base_state, layout, base_hash, clipping_stats, metadata):
    """Subtract/check on device, then transfer one packed fp32 delta to host.

    The rare roundoff coordinates preserve the existing lossless replay format.
    No frozen weights or updated trainable snapshot crosses to CPU.
    """
    params = dict(model.named_parameters())
    device = params[layout[0]["name"]].device
    size = sum(e["numel"] for e in layout)
    high = torch.empty(size, dtype=torch.float32, device=device)
    low = torch.zeros_like(high)
    norm2 = torch.zeros((), dtype=torch.float64, device=device)
    valid = torch.ones((), dtype=torch.bool, device=device)
    for entry in layout:
        name, offset, n = entry["name"], entry["offset"], entry["numel"]
        p, b = params[name].detach(), base_state[name].detach()
        if p.shape != b.shape or p.dtype != b.dtype:
            raise ValueError(f"base shape/dtype mismatch: {name}")
        if p.device != device or b.device != device:
            raise ValueError("device delta capture requires a colocated base snapshot")
        if p.dtype not in {torch.float32, torch.float16, torch.bfloat16}:
            raise ValueError("exact fp32 delta storage supports fp32/fp16/bf16 parameters")
        pf, bf = p.reshape(-1), b.reshape(-1)
        # Larger GPU blocks avoid thousands of tiny CPU/OpenMP operations while
        # bounding temporary fp64 workspace, including on consumer GPUs.
        for start in range(0, n, 16 * BLOCK_SIZE):
            stop = min(start + 16 * BLOCK_SIZE, n)
            sl = slice(offset + start, offset + stop)
            updated, initial = pf[start:stop].double(), bf[start:stop].double()
            high[sl] = (updated - initial).float()
            replay = initial + high[sl].double()
            low[sl] = (updated - replay).float()
            reproduced = (replay + low[sl].double()).to(p.dtype)
            valid &= torch.isfinite(high[sl]).all() & torch.isfinite(low[sl]).all()
            valid &= (reproduced.view(torch.uint8) == pf[start:stop].contiguous().view(torch.uint8)).all()
            diff = high[sl].double() + low[sl].double()
            norm2 += torch.dot(diff, diff)
    if not bool(valid):
        raise ValueError("nonfinite delta or delta cannot replay exactly")
    has_low = [bool(v) for v in torch.stack([low[e["offset"]:e["offset"] + e["numel"]].ne(0).any()
                                           for e in layout]).tolist()]
    chunks = [high] + [low[e["offset"]:e["offset"] + e["numel"]] for e, keep in zip(layout, has_low) if keep]
    packed = (torch.cat(chunks) if len(chunks) > 1 else high).cpu()
    tensors, roundoff, cursor = {}, {}, size
    for entry, keep in zip(layout, has_low):
        name, offset, n = entry["name"], entry["offset"], entry["numel"]
        tensors[name] = packed[offset:offset + n].view(entry["shape"])
        if keep:
            roundoff[name] = packed[cursor:cursor + n].view(entry["shape"])
            cursor += n
    manifest = dict(format=FORMAT_VERSION, layout=layout, layout_hash=canonical_hash(layout),
                    value_dtype="float32", base_hash=base_hash, update_norm=math.sqrt(float(norm2)),
                    clipping_stats=clipping_stats if clipping_stats is not None else
                    {"status": "not_provided", "count": None}, metadata=metadata or {})
    return Delta(tensors, manifest, roundoff)


def capture_delta(model, base_state: Mapping, *, clipping_stats=None, metadata=None, base_hash=None) -> Delta:
    """Capture trainables against an independent snapshot (not state_dict views)."""
    layout = parameter_layout(model)
    if next(p for p in model.parameters() if p.requires_grad).device.type == "cuda":
        # Standalone callers may supply a CPU base; the resident driver supplies
        # both its verified hash and device snapshot, with no per-source upload.
        base_hash = base_hash or tensor_state_hash(base_state)
        params = dict(model.named_parameters())
        device_base = {n: t.to(params[n].device) for n, t in base_state.items()}
        return _capture_device_delta(model, device_base, layout, base_hash, clipping_stats, metadata)
    tensors, roundoff, base, norm2 = {}, {}, {}, 0.0
    params = dict(model.named_parameters())
    for entry in layout:
        name = entry["name"]
        p = params[name].detach().cpu()
        b = base_state[name].detach().cpu()
        if p.shape != b.shape or p.dtype != b.dtype:
            raise ValueError(f"base shape/dtype mismatch: {name}")
        if p.dtype not in {torch.float32, torch.float16, torch.bfloat16}:
            raise ValueError("exact fp32 delta storage supports fp32/fp16/bf16 parameters")
        base[name] = b
        high = torch.empty_like(p, dtype=torch.float32, memory_format=torch.contiguous_format)
        low = torch.zeros_like(high)
        pf, bf, hf, lf = (t.reshape(-1) for t in (p, b, high, low))
        for start in range(0, p.numel(), BLOCK_SIZE):
            sl = slice(start, start + BLOCK_SIZE)
            updated, initial = pf[sl].double(), bf[sl].double()
            hf[sl] = (updated - initial).float()
            replay = initial + hf[sl].double()
            lf[sl] = (updated - replay).float()
            reproduced = (replay + lf[sl].double()).to(p.dtype)
            if not torch.isfinite(hf[sl]).all() or not torch.isfinite(lf[sl]).all():
                raise ValueError(f"nonfinite delta: {name}")
            if not torch.equal(reproduced.view(torch.uint8), pf[sl].contiguous().view(torch.uint8)):
                raise ValueError(f"delta cannot replay exactly: {name}")
            diff = hf[sl].double() + lf[sl].double()
            norm2 += float(torch.dot(diff, diff))
        tensors[name] = high
        if torch.count_nonzero(low):
            roundoff[name] = low
    manifest = dict(format=FORMAT_VERSION, layout=layout, layout_hash=canonical_hash(layout),
                    value_dtype="float32", base_hash=base_hash or tensor_state_hash(base),
                    update_norm=math.sqrt(norm2),
                    clipping_stats=clipping_stats if clipping_stats is not None else
                    {"status": "not_provided", "count": None}, metadata=metadata or {})
    return Delta(tensors, manifest, roundoff)


def _manifest_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".manifest.json")


def save_delta(model, base_state, path, *, clipping_stats=None, metadata=None, delta=None) -> Delta:
    """Write .npz or .safetensors and a sibling <filename>.manifest.json."""
    if delta is None:
        delta = capture_delta(model, base_state, clipping_stats=clipping_stats, metadata=metadata)
    path = Path(path)
    if path.suffix not in {".npz", ".safetensors"}:
        raise ValueError("delta path must end in .npz or .safetensors")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or _manifest_path(path).exists():
        raise FileExistsError(path)
    arrays, entries = {}, []
    for i, entry in enumerate(delta.manifest["layout"]):
        name, key = entry["name"], f"delta_{i}"
        arrays[key] = delta.tensors[name]
        record = {"name": name, "key": key}
        if name in delta.roundoff:
            record["roundoff_key"] = f"roundoff_{i}"
            arrays[record["roundoff_key"]] = delta.roundoff[name]
        entries.append(record)
    # Copy a completed temporary payload under exclusive creation. An interrupted
    # two-file write has no valid manifest and load_delta refuses it.
    with tempfile.TemporaryDirectory(dir=path.parent) as directory:
        temp = Path(directory) / path.name
        if path.suffix == ".npz":
            np.savez(temp, **{k: v.numpy() for k, v in arrays.items()})
        else:
            from safetensors.torch import save_file
            save_file(arrays, str(temp))
        with open(temp, "rb") as src, open(path, "xb") as dst:
            for chunk in iter(lambda: src.read(1024 * 1024), b""):
                dst.write(chunk)
    delta.manifest.update(storage=path.suffix[1:], tensors=entries, payload_hash=file_hash(path))
    write_json_exclusive(_manifest_path(path), delta.manifest)
    return delta


def load_delta(path) -> Delta:
    """Validate payload checksum, layout, tensor coordinates, dtype and finiteness."""
    path = Path(path)
    manifest = json.loads(_manifest_path(path).read_text())
    if manifest["format"] != FORMAT_VERSION or manifest["payload_hash"] != file_hash(path):
        raise ValueError("delta format/checksum mismatch")
    if canonical_hash(manifest["layout"]) != manifest["layout_hash"]:
        raise ValueError("delta layout hash mismatch")
    if manifest["storage"] == "npz":
        with np.load(path, allow_pickle=False) as archive:
            arrays = {k: torch.from_numpy(archive[k].copy()) for k in archive.files}
    elif manifest["storage"] == "safetensors":
        from safetensors.torch import load_file
        arrays = load_file(str(path))
    else:
        raise ValueError("unsupported delta storage")
    tensors, roundoff, used = {}, {}, set()
    if len(manifest["tensors"]) != len(manifest["layout"]):
        raise ValueError("delta tensor count mismatch")
    for entry, stored in zip(manifest["layout"], manifest["tensors"]):
        if entry["name"] != stored["name"] or entry["name"] in tensors:
            raise ValueError("delta parameter order mismatch")
        for field, target in (("key", tensors), ("roundoff_key", roundoff)):
            if field not in stored:
                continue
            key = stored[field]
            value = arrays[key]
            if list(value.shape) != entry["shape"] or value.dtype != torch.float32:
                raise ValueError("delta shape/dtype mismatch")
            if not torch.isfinite(value).all() or key in used:
                raise ValueError("nonfinite/duplicate delta tensor")
            target[entry["name"]] = value
            used.add(key)
    if used != set(arrays):
        raise ValueError("unexpected delta tensors")
    return Delta(tensors, manifest, roundoff)


def apply_delta(model, delta: Delta, scale: float = 1.0):
    """Add to a fresh base in place; reject wrong layout or an already updated base."""
    if not math.isfinite(scale):
        raise ValueError("scale must be finite")
    if layout_hash(model) != delta.manifest["layout_hash"]:
        raise ValueError("model/delta layout mismatch")
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    if tensor_state_hash(params) != delta.manifest["base_hash"]:
        raise ValueError("apply_delta requires the exact fresh base parameters")
    if scale == 0:
        return model
    # Validate all additions before mutation, so overflow cannot leave a partial model.
    for mutate in (False, True):
        with torch.no_grad():
            for name, p in params.items():
                high, low = delta.tensors[name], delta.roundoff.get(name)
                # LoRA factors are contiguous. Copy back also supports strided test parameters.
                current = p.detach().cpu().contiguous().reshape(-1)
                result = current.clone() if mutate else None
                for start in range(0, p.numel(), BLOCK_SIZE):
                    sl = slice(start, start + BLOCK_SIZE)
                    value = current[sl].double() + scale * high.reshape(-1)[sl].double()
                    if low is not None:
                        value += scale * low.reshape(-1)[sl].double()
                    value = value.to(p.dtype)
                    if not torch.isfinite(value).all():
                        raise ValueError(f"scaled delta overflow: {name}")
                    if mutate:
                        result[sl] = value
                if mutate:
                    p.copy_(result.reshape(p.shape))
    return model


def _flat_block(vector, start, block_size):
    block = vector[start:start + block_size]
    if isinstance(block, torch.Tensor):
        block = block.detach().to(device="cpu", dtype=torch.float64).numpy()
    block = np.asarray(block, dtype=np.float64)
    if not np.isfinite(block).all():
        raise ValueError("nonfinite first-order input")
    return block


def _dot(left, right, block_size=BLOCK_SIZE) -> float:
    if np.shape(left) != np.shape(right) or len(np.shape(left)) != 1:
        raise ValueError("expected aligned flat vectors")
    total = 0.0
    for start in range(0, len(left), block_size):
        a = _flat_block(left, start, block_size)
        b = _flat_block(right, start, block_size)
        total += float(a @ b)
    return total


def first_order_diff(gT, gS, delta) -> float:
    """Predict change of ell_T - ell_S, with gradients of the SAME named ell."""
    if np.shape(gT) != np.shape(gS) or np.shape(gT) != np.shape(delta) or len(np.shape(delta)) != 1:
        raise ValueError("expected aligned flat vectors")
    total = 0.0
    for start in range(0, len(delta), BLOCK_SIZE):
        gradient = _flat_block(gT, start, BLOCK_SIZE) - _flat_block(gS, start, BLOCK_SIZE)
        total += float(gradient @ _flat_block(delta, start, BLOCK_SIZE))
    return total


def first_order_teacher(gT, delta) -> float:
    """Predict change of ell_T (no student-gradient subtraction)."""
    return _dot(gT, delta)
