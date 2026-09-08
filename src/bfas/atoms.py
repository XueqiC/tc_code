"""Capability-atom discovery: sparse dictionary learning on event fingerprints (CRCD spec §3.3, §6.1).

Model.  Given n event fingerprints psi_i in R^d (rows of ``Psi``, shape (n, d)), fit a dictionary
U in R^{d x K} with unit-norm columns (the *atoms*) and sparse compositions z_i in R^K:

    min_{U,Z}  1/(2n) ||Psi - Z U^T||_F^2 + (lambda_z / n) ||Z||_1      s.t. ||u_k||_2 = 1.

``lambda_z`` is the per-event lasso penalty, i.e. every event solves
1/2 ||psi_i - U z_i||^2 + lambda_z ||z_i||_1, which keeps the grid n-invariant (it equals the spec's
lambda with an extra 1/n on the L1 term; the spec's literal lambda is lambda_z / n).

Layout convention: ``Psi`` is (n, d) with one event per row, ``U`` is (d, K), ``Z`` is (n, K),
so Psi ~= Z @ U.T.  The math in the spec uses the transposed (d x n) layout; z_i is row i of Z.

Solver: alternating minimisation.  Z-step = FISTA (soft-thresholding, step 1/L with L = sigma_max(U)^2),
U-step = block coordinate descent over atoms with the exact constrained minimiser for each column
(for a fixed Z the objective in u_k is an isotropic quadratic, so the unit-sphere minimiser is the
normalised unconstrained minimiser; Mairal et al. 2010 Alg. 2 with an equality instead of a ball
constraint).  Dead atoms (never used by the current code) are re-seeded from the worst-reconstructed
event, deterministically.  Everything is seeded and pure numpy.

Baselines with the same input and dimensionality: PCA/SVD (``fit_pca``), hard k-means dictionary
(``fit_kmeans``), random unit-norm dictionary (``random_dictionary``).

Coverage (§6.1): ``reconstruction_coverage(U, Psi_held) = 1 - min_Z ||Psi_held - Z U^T||_F^2 / ||Psi_held||_F^2``
(unregularised least squares, i.e. projection onto span(U)).  Model selection (``select_model``) uses
the held-out *sparse-coded* reconstruction at the candidate lambda, because the span-only coverage is
independent of lambda.  Atom relevance zbar_ki = |z_ki| (``atom_relevance``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

__all__ = [
    "DictionaryFit",
    "fit",
    "encode",
    "least_squares_codes",
    "reconstruction_coverage",
    "sparse_reconstruction_error",
    "per_event_sparse_errors",
    "objective",
    "select_model",
    "load_fingerprints_json",
    "load_features",
    "fit_pca",
    "encode_pca",
    "fit_kmeans",
    "encode_kmeans",
    "random_dictionary",
    "atom_relevance",
    "transfer_prediction",
    "split_events",
    "load_npz",
    "DEFAULT_KS",
    "DEFAULT_LAMBDAS",
]

DEFAULT_KS: tuple[int, ...] = (4, 8, 16, 32)
DEFAULT_LAMBDAS: tuple[float, ...] = (0.0, 0.01, 0.02, 0.05, 0.1, 0.2)


# ----------------------------------------------------------------------------------------------
# core pieces
# ----------------------------------------------------------------------------------------------
def _as2d(Psi) -> np.ndarray:
    Psi = np.asarray(Psi, dtype=np.float64)
    if Psi.ndim == 1:
        Psi = Psi[None, :]
    if Psi.ndim != 2:
        raise ValueError(f"Psi must be 2-d (n, d); got shape {Psi.shape}")
    return Psi


def _soft_threshold(x: np.ndarray, t: float) -> np.ndarray:
    return np.sign(x) * np.maximum(np.abs(x) - t, 0.0)


def _normalise_columns(U: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(U, axis=0)
    norms = np.where(norms < eps, 1.0, norms)
    return U / norms


def _lipschitz(U: np.ndarray) -> float:
    s = np.linalg.norm(U, ord=2)
    return float(max(s * s, 1e-12))


def encode(U, Psi_new, lambda_z: float, n_iter: int = 500, tol: float = 1e-7, Z0=None) -> np.ndarray:
    """Sparse codes for new fingerprints: argmin_z 1/2||psi - U z||^2 + lambda_z ||z||_1 via FISTA.

    Deterministic (zero initialisation unless ``Z0`` is given, fixed iteration schedule).
    Returns Z with shape (m, K).  lambda_z == 0 gives an (iterative) least-squares solution.
    """
    U = np.asarray(U, dtype=np.float64)
    Psi_new = _as2d(Psi_new)
    m, K = Psi_new.shape[0], U.shape[1]
    if lambda_z < 0:
        raise ValueError("lambda_z must be >= 0")
    L = _lipschitz(U)
    step = 1.0 / L
    thr = lambda_z * step
    Z = np.zeros((m, K)) if Z0 is None else np.array(Z0, dtype=np.float64, copy=True)
    Y = Z.copy()
    t = 1.0
    UtU = U.T @ U
    PsiU = Psi_new @ U
    for _ in range(n_iter):
        grad = Y @ UtU - PsiU
        Z_new = _soft_threshold(Y - step * grad, thr)
        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        Y = Z_new + ((t - 1.0) / t_new) * (Z_new - Z)
        diff = np.linalg.norm(Z_new - Z)
        Z, t = Z_new, t_new
        if diff <= tol * max(1.0, np.linalg.norm(Z)):
            break
    return Z


def least_squares_codes(U, Psi) -> np.ndarray:
    """min_Z ||Psi - Z U^T||_F^2 in closed form (projection onto span(U)); shape (n, K)."""
    U = np.asarray(U, dtype=np.float64)
    Psi = _as2d(Psi)
    Z, *_ = np.linalg.lstsq(U, Psi.T, rcond=None)
    return Z.T


def reconstruction_coverage(U, Psi_held) -> float:
    """1 - min_Z ||Psi_held - Z U^T||_F^2 / ||Psi_held||_F^2  (spec §6.1); in [0, 1]."""
    Psi_held = _as2d(Psi_held)
    denom = float(np.sum(Psi_held * Psi_held))
    if denom <= 0:
        return float("nan")
    Z = least_squares_codes(U, Psi_held)
    resid = Psi_held - Z @ np.asarray(U, dtype=np.float64).T
    cov = 1.0 - float(np.sum(resid * resid)) / denom
    return float(min(1.0, max(0.0, cov)))


def sparse_reconstruction_error(U, Psi_held, lambda_z: float) -> float:
    """Relative reconstruction error ||Psi - Z(lambda) U^T||_F^2 / ||Psi||_F^2 with the *sparse* code.

    Unlike ``reconstruction_coverage`` this depends on lambda, so it can select lambda on held-out data.
    """
    Psi_held = _as2d(Psi_held)
    Z = encode(U, Psi_held, lambda_z)
    resid = Psi_held - Z @ np.asarray(U, dtype=np.float64).T
    denom = float(np.sum(Psi_held * Psi_held))
    return float(np.sum(resid * resid)) / denom if denom > 0 else float("nan")


def objective(U, Z, Psi, lambda_z: float) -> float:
    """1/(2n)||Psi - Z U^T||_F^2 + (lambda_z/n)||Z||_1."""
    Psi = _as2d(Psi)
    n = Psi.shape[0]
    resid = Psi - np.asarray(Z) @ np.asarray(U).T
    return float(np.sum(resid * resid) / (2.0 * n) + lambda_z * np.sum(np.abs(Z)) / n)


def _init_dictionary(Psi: np.ndarray, K: int, rng: np.random.Generator, init: str) -> np.ndarray:
    d = Psi.shape[1]
    if init == "random":
        U = rng.standard_normal((d, K))
    elif init == "samples":
        idx = rng.choice(Psi.shape[0], size=K, replace=Psi.shape[0] < K)
        U = Psi[idx].T.copy()
        U = U + 1e-6 * rng.standard_normal(U.shape)  # break exact duplicates deterministically
    else:
        raise ValueError(f"unknown init {init!r}")
    return _normalise_columns(U)


def _update_dictionary(U: np.ndarray, Z: np.ndarray, Psi: np.ndarray, sweeps: int = 2,
                       dead_tol: float = 1e-10) -> tuple[np.ndarray, int]:
    """Block coordinate descent on the atoms with the exact unit-sphere column minimiser."""
    A = Z.T @ Z            # (K, K)
    B = Psi.T @ Z          # (d, K)
    U = U.copy()
    K = U.shape[1]
    n_dead = 0
    for _ in range(sweeps):
        for k in range(K):
            if A[k, k] <= dead_tol:
                continue
            u = U[:, k] + (B[:, k] - U @ A[:, k]) / A[k, k]
            nrm = np.linalg.norm(u)
            if nrm > 1e-12:
                U[:, k] = u / nrm
    dead = np.where(np.diag(A) <= dead_tol)[0]
    if dead.size:
        resid = Psi - Z @ U.T
        order = np.argsort(-np.linalg.norm(resid, axis=1))
        for j, k in enumerate(dead):
            r = resid[order[j % len(order)]]
            nrm = np.linalg.norm(r)
            if nrm > 1e-12:
                U[:, k] = r / nrm
                n_dead += 1
    return U, n_dead


@dataclass
class DictionaryFit:
    U: np.ndarray                    # (d, K) unit-norm atoms
    Z: np.ndarray                    # (n, K) sparse compositions of the training events
    K: int
    lambda_z: float
    seed: int
    objective_history: list[float] = field(default_factory=list)
    n_iter: int = 0
    converged: bool = False
    init: str = "random"

    @property
    def nnz_per_event(self) -> np.ndarray:
        return (np.abs(self.Z) > 1e-10).sum(axis=1)

    @property
    def atom_usage(self) -> np.ndarray:
        """Fraction of training events on which each atom is active."""
        return (np.abs(self.Z) > 1e-10).mean(axis=0)

    def summary(self) -> dict:
        return {
            "K": int(self.K), "lambda_z": float(self.lambda_z), "seed": int(self.seed),
            "n_iter": int(self.n_iter), "converged": bool(self.converged), "init": self.init,
            "objective": float(self.objective_history[-1]) if self.objective_history else None,
            "mean_nnz_per_event": float(self.nnz_per_event.mean()),
            "median_nnz_per_event": float(np.median(self.nnz_per_event)),
            "sparsity": float(1.0 - (np.abs(self.Z) > 1e-10).mean()),
            "atoms_used": int((self.atom_usage > 0).sum()),
            "atom_usage": [float(x) for x in self.atom_usage],
        }


def fit(Psi, K: int, lambda_z: float, seed: int = 0, n_iter: int = 200, tol: float = 1e-6,
        init: str = "best", fista_iter: int = 200, sweeps: int = 2) -> DictionaryFit:
    """Alternating sparse dictionary learning (see module docstring).  Deterministic given ``seed``.

    init: ``random`` (Gaussian unit-norm atoms), ``samples`` (random training events; can get stuck
    on atom mixtures), or ``best`` (default: run both with the same seed, keep the lower objective).
    """
    Psi = _as2d(Psi)
    n, d = Psi.shape
    if K < 1:
        raise ValueError("K must be >= 1")
    if init == "best":
        runs = [fit(Psi, K, lambda_z, seed=seed, n_iter=n_iter, tol=tol, init=i, fista_iter=fista_iter,
                    sweeps=sweeps) for i in ("random", "samples")]
        best = min(runs, key=lambda r: r.objective_history[-1])
        best.init = "random" if best is runs[0] else "samples"
        return best
    rng = np.random.default_rng(seed)
    U = _init_dictionary(Psi, K, rng, init)
    Z = np.zeros((n, K))
    hist: list[float] = []
    converged = False
    it = 0
    for it in range(1, n_iter + 1):
        Z = encode(U, Psi, lambda_z, n_iter=fista_iter, Z0=Z)
        U, _ = _update_dictionary(U, Z, Psi, sweeps=sweeps)
        hist.append(objective(U, Z, Psi, lambda_z))
        if len(hist) >= 2 and abs(hist[-2] - hist[-1]) <= tol * max(abs(hist[-2]), 1e-12):
            converged = True
            break
    # final, fully converged codes for the returned dictionary
    Z = encode(U, Psi, lambda_z, n_iter=1000, Z0=Z)
    hist.append(objective(U, Z, Psi, lambda_z))
    return DictionaryFit(U=U, Z=Z, K=K, lambda_z=lambda_z, seed=seed, objective_history=hist,
                         n_iter=it, converged=converged)


# ----------------------------------------------------------------------------------------------
# model selection
# ----------------------------------------------------------------------------------------------
def split_events(n: int, dev_frac: float = 0.3, seed: int = 0, groups: Sequence | None = None):
    """Deterministic train/dev index split.  With ``groups`` the split is by group (events sharing a
    prompt stay together), otherwise by event."""
    rng = np.random.default_rng(seed)
    if groups is None:
        perm = rng.permutation(n)
        n_dev = max(1, int(round(dev_frac * n)))
        dev = np.sort(perm[:n_dev])
        train = np.sort(perm[n_dev:])
        return train, dev
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    perm = rng.permutation(len(uniq))
    n_dev = max(1, int(round(dev_frac * len(uniq))))
    dev_groups = set(uniq[perm[:n_dev]].tolist())
    dev = np.array([i for i in range(n) if groups[i] in dev_groups])
    train = np.array([i for i in range(n) if groups[i] not in dev_groups])
    return train, dev


def per_event_sparse_errors(U, Psi_held, lambda_z: float) -> np.ndarray:
    """Relative sparse-coded reconstruction error of each held-out event, ||psi - U z||^2/||psi||^2."""
    Psi_held = _as2d(Psi_held)
    Z = encode(U, Psi_held, lambda_z)
    resid = Psi_held - Z @ np.asarray(U, dtype=np.float64).T
    denom = np.maximum(np.sum(Psi_held * Psi_held, axis=1), 1e-12)
    return np.sum(resid * resid, axis=1) / denom


def select_model(Psi, Ks: Iterable[int] = DEFAULT_KS, lambdas: Iterable[float] = DEFAULT_LAMBDAS,
                 seed: int = 0, dev_frac: float = 0.3, groups: Sequence | None = None,
                 criterion: str = "sparse_recon_1se", fit_kwargs: dict | None = None) -> dict:
    """Grid over K and lambda_z; fit on train events, score on dev events.

    criterion:
      * ``sparse_recon_1se`` (default): among cells whose mean held-out sparse-coded reconstruction
        error is within one standard error of the best cell, pick the sparsest (fewest active atoms
        per event, then smallest K).  The lasso "one-standard-error rule"; label-free.
      * ``sparse_recon``: plain argmin of held-out sparse-coded error (in practice lambda -> 0, i.e.
        the PCA-equivalent dictionary; kept for reference).
      * ``coverage``: held-out span coverage (§6.1; higher is better) — depends on K only, lambda
        ties broken towards the sparser model.
    Uses only fingerprints, never transfer/test labels.  Returns the grid and the chosen cell.
    """
    Psi = _as2d(Psi)
    train, dev = split_events(Psi.shape[0], dev_frac=dev_frac, seed=seed, groups=groups)
    fit_kwargs = dict(fit_kwargs or {})
    grid = []
    for K in Ks:
        for lam in lambdas:
            f = fit(Psi[train], int(K), float(lam), seed=seed, **fit_kwargs)
            cov = reconstruction_coverage(f.U, Psi[dev])
            errs = per_event_sparse_errors(f.U, Psi[dev], float(lam))
            train_cov = reconstruction_coverage(f.U, Psi[train])
            grid.append({"K": int(K), "lambda_z": float(lam), "dev_coverage": cov,
                         "dev_sparse_recon_error": float(errs.mean()),
                         "dev_sparse_recon_se": float(errs.std(ddof=1) / np.sqrt(len(errs))) if len(errs) > 1 else 0.0,
                         "train_coverage": train_cov,
                         "mean_nnz_per_event": float(f.nnz_per_event.mean()),
                         "atoms_used": int((f.atom_usage > 0).sum()), "converged": f.converged,
                         "n_iter": f.n_iter})
    argmin = min(grid, key=lambda g: (g["dev_sparse_recon_error"], g["K"], -g["lambda_z"]))
    if criterion == "sparse_recon":
        best = argmin
    elif criterion == "sparse_recon_1se":
        bound = argmin["dev_sparse_recon_error"] + argmin["dev_sparse_recon_se"]
        cand = [g for g in grid if g["dev_sparse_recon_error"] <= bound]
        best = min(cand, key=lambda g: (g["mean_nnz_per_event"], g["K"], -g["lambda_z"]))
    elif criterion == "coverage":
        best = max(grid, key=lambda g: (round(g["dev_coverage"], 6), g["lambda_z"], -g["K"]))
    else:
        raise ValueError(f"unknown criterion {criterion!r}")
    return {"criterion": criterion, "seed": int(seed), "dev_frac": float(dev_frac),
            "n_train": int(len(train)), "n_dev": int(len(dev)), "train_idx": train.tolist(),
            "dev_idx": dev.tolist(), "grid": grid, "best": best, "argmin": argmin}


# ----------------------------------------------------------------------------------------------
# baselines with the same input and dimensionality
# ----------------------------------------------------------------------------------------------
def fit_pca(Psi, K: int, center: bool = False) -> dict:
    """PCA / truncated SVD with K orthonormal components (no sparsity).

    ``center=False`` is the truncated SVD of Psi itself (same reconstruction formula as the
    dictionary, no mean term); ``center=True`` is classical PCA.  Returns U (d, K), Z (n, K), mean.
    """
    Psi = _as2d(Psi)
    mean = Psi.mean(axis=0) if center else np.zeros(Psi.shape[1])
    X = Psi - mean
    _, s, Vt = np.linalg.svd(X, full_matrices=False)
    K = min(K, Vt.shape[0])
    U = Vt[:K].T.copy()
    # sign convention: largest-magnitude entry of each component positive (deterministic)
    for k in range(K):
        j = np.argmax(np.abs(U[:, k]))
        if U[j, k] < 0:
            U[:, k] *= -1
    Z = X @ U
    total = float(np.sum(s * s))
    return {"U": U, "Z": Z, "mean": mean, "K": K, "singular_values": s[:K],
            "explained_variance_ratio": (s[:K] ** 2 / total) if total > 0 else s[:K] * 0}


def encode_pca(pca: dict, Psi_new) -> np.ndarray:
    return (_as2d(Psi_new) - pca["mean"]) @ pca["U"]


def fit_kmeans(Psi, K: int, seed: int = 0, n_iter: int = 100) -> dict:
    """Hard-assignment dictionary: k-means++ init + Lloyd's; atoms = unit-norm centroids,
    Z = one-hot times the projection coefficient <psi_i, u_k>."""
    Psi = _as2d(Psi)
    n = Psi.shape[0]
    K = min(K, n)
    rng = np.random.default_rng(seed)
    centers = np.empty((K, Psi.shape[1]))
    centers[0] = Psi[rng.integers(n)]
    d2 = np.sum((Psi - centers[0]) ** 2, axis=1)
    for k in range(1, K):
        p = d2 / d2.sum() if d2.sum() > 0 else np.full(n, 1.0 / n)
        centers[k] = Psi[rng.choice(n, p=p)]
        d2 = np.minimum(d2, np.sum((Psi - centers[k]) ** 2, axis=1))
    labels = np.zeros(n, dtype=int)
    for _ in range(n_iter):
        dist = ((Psi[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new = np.argmin(dist, axis=1)
        for k in range(K):
            m = new == k
            if m.any():
                centers[k] = Psi[m].mean(axis=0)
        if np.array_equal(new, labels):
            break
        labels = new
    U = _normalise_columns(centers.T.copy())
    return {"U": U, "labels": labels, "Z": encode_kmeans({"U": U}, Psi), "K": K, "centers": centers}


def encode_kmeans(km: dict, Psi_new) -> np.ndarray:
    Psi_new = _as2d(Psi_new)
    U = km["U"]
    sims = Psi_new @ U
    lab = np.argmax(sims, axis=1)
    Z = np.zeros_like(sims)
    Z[np.arange(len(lab)), lab] = sims[np.arange(len(lab)), lab]
    return Z


def random_dictionary(d: int, K: int, seed: int = 0) -> np.ndarray:
    """Random unit-norm dictionary (d, K)."""
    rng = np.random.default_rng(seed)
    return _normalise_columns(rng.standard_normal((d, K)))


# ----------------------------------------------------------------------------------------------
# relevance and transfer prediction
# ----------------------------------------------------------------------------------------------
def atom_relevance(Z) -> np.ndarray:
    """zbar_ki = |z_ki|; shape (n, K).  Average over events for per-atom relevance."""
    return np.abs(np.asarray(Z, dtype=np.float64))


def transfer_prediction(U, Z_source, Z_target) -> np.ndarray:
    """First-order transfer prediction P[i, j] = z_j^T U^T U z_i for source rows i and target rows j
    (spec §3.4, without the step size eta).  Shape (n_source, n_target)."""
    U = np.asarray(U, dtype=np.float64)
    G = U.T @ U
    return np.asarray(Z_source) @ G @ np.asarray(Z_target).T


# ----------------------------------------------------------------------------------------------
# I/O
# ----------------------------------------------------------------------------------------------
def load_npz(path) -> dict:
    """Load a fingerprint file with keys ``psi`` (n, d) and ``state_hash`` (n,); other keys are
    passed through (e.g. category, cluster, traj, response_len)."""
    with np.load(Path(path), allow_pickle=True) as f:
        out = {k: f[k] for k in f.files}
    if "psi" not in out or "state_hash" not in out:
        raise KeyError(f"{path}: expected keys 'psi' and 'state_hash', found {sorted(out)}")
    out["psi"] = np.asarray(out["psi"], dtype=np.float64)
    out["state_hash"] = np.asarray(out["state_hash"]).astype(str)
    if out["psi"].shape[0] != out["state_hash"].shape[0]:
        raise ValueError("psi and state_hash have different lengths")
    return out


def load_fingerprints_json(path, key: str = "a3") -> dict:
    """Load per-event features from a ``crcd_fingerprints_v*.json`` (``gate.events[*].<key>``).
    Returns psi (n, d) and the event metadata (task_id, traj, category, cluster, dU, u_minus)."""
    import json

    doc = json.loads(Path(path).read_text())
    events = doc["gate"]["events"]
    psi = np.asarray([e[key] for e in events], dtype=np.float64)
    meta = [{k: v for k, v in e.items() if k != key} for e in events]
    return {"psi": psi, "meta": meta, "feature_key": key, "gate_k": doc["gate"].get("k")}


def load_features(path, key: str = "a3") -> dict:
    """Dispatch on suffix: ``.npz`` (keys psi, state_hash, ...) or the fingerprints json."""
    path = Path(path)
    if path.suffix == ".npz":
        return load_npz(path)
    return load_fingerprints_json(path, key=key)
