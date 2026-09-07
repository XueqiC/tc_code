"""CPU P2 model competition, with source-group nested cross-validation.

X has shape (n_sources, k), M has shape (m_probes, n_sources), R is (m, k).
Inputs must be amplitude-preserving codes in fixed H-orthonormal coordinates,
or H-whitened parameter differences. They must NOT be a globally learned PCA,
centred matrix, or random sketch. Each fit constructs its span from training X
only; no centring or coordinate standardisation changes the H-metric penalty.
An externally supplied decoder C must satisfy C.T H C = I. Fold directions are
decoded through C @ fold_basis, never through an inverse random projection.

B2 solves the stated reduced-rank ridge objective exactly on complete training
columns. For Y=M.T, augment Z=[X; sqrt(lambda) I], T=[Y; 0]. If Z=L D V.T,
the rank-K least-squares solution is W=V D^-1 (L.T T)_K, R=W.T. Truncating
the ordinary ridge *coefficient* is generally wrong: truncation belongs in
the augmented fitted-data geometry. Zero singular values use the minimum-norm
solution. The zero augmentation encodes the penalty; it NEVER imputes NA labels.

NA policy: exclude entirely unobserved training probes; use only source columns
complete on the remaining probes. Report every exclusion; abstain (NA) when no
complete columns exist. This conservative closed-form policy avoids claiming
the augmented solution optimises a masked matrix-completion objective.

All reported predictions are for new sources on the same fixed probes. Group
bootstrap intervals describe OOF prediction uncertainty, not retraining-seed
uncertainty. Optional two-way bootstrap also resamples entire probe groups.

Hierarchical fits use one global rank-1 block, family rank-1/2 blocks, and an
optional residual block. By default their sum is a single response operator R;
source factors select training subsets only, never prediction-time gates.
The old source-gated estimator remains available in explicit 'gated' mode.
Each block uses the same H geometry
and lambda grid. The sum of their rank budgets is the matched flat B2 budget;
the block budget is explicitly not an equal-parameter-count claim.
Teacher and student targets are independent. Combined fits remove a rank-1
student response fitted anew inside every training partition before selection.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class FitConfig:
    ranks: tuple[int, ...] = (1, 2, 4)
    lambdas: tuple[float, ...] = (0.0, 0.01, 0.1, 1.0)
    inner_splits: int = 3
    permutation_seeds: tuple[int, ...] = (11, 29, 47)
    bootstrap_samples: int = 200
    bootstrap_seed: int = 0
    confidence: float = 0.95
    two_way_bootstrap: bool = False
    model: str = "flat"
    family_ranks: tuple[int, ...] = (1, 2)
    min_family_sources: int = 8
    residual_rank: int = 0
    hierarchical_mode: str = "operator"
    error_quantile: float = 0.95

    def __post_init__(self) -> None:
        if not self.ranks or any(k not in (1, 2, 4) for k in self.ranks):
            raise ValueError("ranks must be a nonempty subset of {1, 2, 4}")
        if not self.lambdas or any(not np.isfinite(x) or x < 0 for x in self.lambdas):
            raise ValueError("lambdas must be a fixed nonempty finite nonnegative grid")
        if self.inner_splits < 2 or self.bootstrap_samples < 0 or not 0 < self.confidence < 1:
            raise ValueError("invalid CV/bootstrap configuration")
        if len(set(self.permutation_seeds)) != len(self.permutation_seeds):
            raise ValueError("permutation seeds must be distinct")
        if self.model not in {"flat", "hierarchical"}:
            raise ValueError("model must be flat or hierarchical")
        if self.hierarchical_mode not in {"operator", "gated"}:
            raise ValueError("hierarchical_mode must be operator or gated")
        if not np.isfinite(self.error_quantile) or not 0 <= self.error_quantile <= 1:
            raise ValueError("error_quantile must be between zero and one")
        if not self.family_ranks or any(k not in (1, 2) for k in self.family_ranks):
            raise ValueError("family_ranks must be a nonempty subset of {1, 2}")
        if (not isinstance(self.min_family_sources, int) or self.min_family_sources < 1
                or not isinstance(self.residual_rank, int) or self.residual_rank < 0):
            raise ValueError("min_family_sources must be positive and residual_rank nonnegative integers")


def _matrix(value, name: str, *, finite: bool = False) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.ndim != 2 or np.isinf(result).any() or (finite and not np.isfinite(result).all()):
        raise ValueError(f"{name} must be a 2D array {'of finite values' if finite else 'with finite values or NA'}")
    return result


def _rank(s: np.ndarray, shape: tuple) -> int:
    return int(np.sum(s > (s[0] * max(shape) * np.finfo(float).eps))) if len(s) else 0


def training_span(X: np.ndarray) -> np.ndarray:
    """Uncentred orthonormal span of training updates, columns in code space."""
    X = _matrix(X, "X", finite=True)
    _, s, vt = np.linalg.svd(X, full_matrices=False)
    return vt[:_rank(s, X.shape)].T


def reduced_rank_ridge(X, M, rank: int, ridge: float) -> np.ndarray:
    """Global optimum of ||M-R X.T||² + ridge ||R||², rank(R)<=rank.

    Complete data only. See module docstring for the augmented-data derivation.
    The orthogonal residual outside col(Z) is constant; Eckart--Young applies
    to L.T T, and multiplication by invertible D preserves the rank constraint.
    """
    X, M = _matrix(X, "X", finite=True), _matrix(M, "M", finite=True)
    if X.shape[0] != M.shape[1] or rank < 1 or not np.isfinite(ridge) or ridge < 0:
        raise ValueError("incompatible response axes, rank, or ridge penalty")
    k, m = X.shape[1], M.shape[0]
    if k == 0 or X.shape[0] == 0:
        return np.zeros((m, k))
    Z = np.concatenate([X, np.sqrt(ridge) * np.eye(k)], axis=0)
    T = np.concatenate([M.T, np.zeros((k, m))], axis=0)
    left, s, vt = np.linalg.svd(Z, full_matrices=False)
    keep = _rank(s, Z.shape)
    if keep == 0:
        return np.zeros((m, k))
    target = left[:, :keep].T @ T
    a, sigma, bt = np.linalg.svd(target, full_matrices=False)
    r = min(rank, len(sigma))
    truncated = (a[:, :r] * sigma[:r]) @ bt[:r]
    return ((vt[:keep].T / s[:keep]) @ truncated).T


def _ridge(X: np.ndarray, M: np.ndarray, ridge: float) -> np.ndarray:
    return reduced_rank_ridge(X, M, max(1, min(X.shape[1], M.shape[0])), ridge)


def response_modes(R, C=None) -> dict:
    """R=A Sigma V.T => U=C V (parameter), B=A Sigma (probe).

    R @ V == B and R @ (V @ a) == B @ a. C may be a (d,k) matrix
    or a decoder callable taking a (k,K) matrix; no optional GPU module is
    imported here. With C omitted U is unavailable, never mislabeled as V.
    """
    R = _matrix(R, "R")
    valid = np.isfinite(R).all(axis=1)
    a, s, vt = np.linalg.svd(R[valid], full_matrices=False)
    r = _rank(s, R[valid].shape)
    V = vt[:r].T
    B = np.full((R.shape[0], r), np.nan)
    B[valid] = a[:, :r] * s[:r]
    U = None if C is None else (C(V) if callable(C) else np.asarray(C) @ V)
    return {"V": V, "B": B, "U": U, "singular_values": s[:r],
            "orientation": "R=A Sigma V.T; U=C V parameter directions; B=A Sigma probe response modes"}


def _metadata_slice(metadata: Mapping | None, indices) -> dict:
    return {key: np.asarray(value)[indices] for key, value in (metadata or {}).items()}


def _baseline_features(X, kind: str, metadata: Mapping, categories: Sequence[str]) -> np.ndarray:
    norm = np.linalg.norm(X, axis=1)
    if kind == "norm":
        return norm[:, None]
    if kind == "category":
        labels = np.asarray(metadata["category"], dtype=str)
        return (labels[:, None] == np.asarray(categories)[None, :]) * norm[:, None]
    if kind == "length":
        lengths = np.asarray(metadata["length"], dtype=float)
        if lengths.shape != norm.shape or not np.isfinite(lengths).all() or (lengths < 0).any():
            raise ValueError("length must contain finite nonnegative source lengths")
        return (norm * lengths)[:, None]
    raise ValueError(f"unknown baseline feature kind {kind}")


@dataclass
class ResponseModel:
    name: str
    R: np.ndarray
    basis: np.ndarray
    rank: int
    ridge: float
    requested_rank: int
    diagnostics: dict = field(default_factory=dict)
    cv_scores: list = field(default_factory=list)
    feature_kind: str | None = None
    categories: tuple[str, ...] = ()
    feature_scale: np.ndarray | None = None

    def predict(self, X, metadata: Mapping | None = None) -> np.ndarray:
        X = _matrix(X, "X", finite=True)
        if X.shape[1] != self.basis.shape[0]:
            raise ValueError("prediction code dimension differs from training codes")
        if self.feature_kind:
            features = _baseline_features(X, self.feature_kind, metadata or {}, self.categories)
            return self.R @ (features / self.feature_scale).T
        return self.R @ X.T

    def modes(self, C=None) -> dict:
        if self.feature_kind:
            raise ValueError("metadata baselines do not define linear parameter directions")
        return response_modes(self.R, C)


class ResponsePredictor:
    """A fixed linear response map: prediction(delta) = R C.T H delta.

    Accept a fitted flat/hierarchical model or its (probe, code) R, and optionally
    bind the ``whiten.Basis`` used to encode its training data. Hierarchies always
    export the sum of their operators, including when fitted in legacy mode;
    metadata baselines cannot be used. R is copied to freeze this prediction map.
    The basis must use the same coordinates as R. No torch import is needed here.

    Batch X and deltas have rows for updates; predictions have columns for them.
    Single vectors return a probe vector. ``D_codes`` instead stores source codes
    as columns (k, n), so B_hat = R @ D_codes and recipe(a) = R @ (D_codes @ a).
    Updates outside span(C) are also defined: their H-orthogonal part maps to zero.
    """

    def __init__(self, model, basis=None):
        if getattr(model, "feature_kind", None):
            raise ValueError("metadata baselines do not define a response operator")
        self.R = _matrix(getattr(model, "R", model), "R").copy()
        self.basis = basis
        if basis is not None:
            self._check_basis(basis)

    def _check_basis(self, basis):
        if basis.rank != self.R.shape[1]:
            raise ValueError("basis rank differs from response code dimension")

    def predict_from_codes(self, X):
        X = np.asarray(X, dtype=float)
        single = X.ndim == 1
        X = _matrix(X[None, :] if single else X, "X", finite=True)
        if X.shape[1] != self.R.shape[1]:
            raise ValueError("prediction code dimension differs from response operator")
        prediction = self.R @ X.T
        return prediction[:, 0] if single else prediction

    def predict_from_deltas(self, deltas, basis=None):
        basis = self.basis if basis is None else basis
        if basis is None:
            raise ValueError("a Basis is required to encode parameter updates")
        self._check_basis(basis)
        # Iterate over existing vectors/memmaps; never materialize a dense decoder.
        shape = getattr(deltas, "shape", None)
        single = ((shape is not None and len(shape) == 1) or
                  (isinstance(deltas, (list, tuple)) and len(deltas) > 0 and np.isscalar(deltas[0])))
        if single:
            return self.predict_from_codes(basis.encode(deltas)[0])
        if np.isscalar(deltas) or (shape is not None and len(shape) != 2):
            raise ValueError("deltas must be a parameter vector or an iterable of parameter vectors")
        if shape is not None and shape[1] != basis.H.size:
            raise ValueError("deltas must have one column per parameter")
        codes = [basis.encode(delta)[0] for delta in deltas]
        X = np.stack(codes) if codes else np.empty((0, basis.rank))
        return self.predict_from_codes(X)

    def predict_recipe(self, a, D_codes):
        D_codes = _matrix(D_codes, "D_codes", finite=True)
        a = np.asarray(a, dtype=float)
        if (D_codes.shape[0] != self.R.shape[1] or a.shape != (D_codes.shape[1],)
                or not np.isfinite(a).all()):
            raise ValueError("D_codes must be (code, source) and a a finite source vector")
        return self.predict_from_codes(D_codes @ a)


def permute_within_strata(strata, seed: int) -> np.ndarray:
    labels = np.asarray(strata, dtype=str)
    if labels.ndim != 1:
        raise ValueError("strata must be a 1D list of fixed category/dose keys")
    rng, permutation = np.random.default_rng(seed), np.arange(len(labels))
    for label in sorted(set(labels)):
        indices = np.flatnonzero(labels == label)
        permutation[indices] = rng.permutation(indices)
    return permutation


def fit_model(X, M, *, name: str = "B2", rank: int = 2, ridge: float = 0.1,
              metadata: Mapping | None = None, strata=None, permutation_seed: int | None = None) -> ResponseModel:
    """Fit on training data only. This function never takes held-out responses."""
    X, M = _matrix(X, "X", finite=True), _matrix(M, "M")
    if X.shape[0] != M.shape[1] or not X.shape[0] or not M.shape[0]:
        raise ValueError("X=(n,k) and M=(m,n) must have nonempty matching axes")
    if rank < 1 or not np.isfinite(ridge) or ridge < 0:
        raise ValueError("rank and ridge must be valid")
    metadata = metadata or {}
    n, k = X.shape
    span = training_span(X)
    diagnostics = {"preprocessing": "training-only uncentred span; no coordinate scaling/intercept",
                   "missing_policy": "complete sources on observed training probes"}
    if name.startswith("B3"):
        if permutation_seed is None:
            raise ValueError("B3 requires a fixed permutation seed")
        strata = np.repeat("all", n) if strata is None else strata
        if len(strata) != n:
            raise ValueError("strata length differs from sources")
        perm = permute_within_strata(strata, permutation_seed)
        M = M[:, perm]
        diagnostics.update(permutation=perm, permutation_seed=permutation_seed,
                           permutation_moved_fraction=float(np.mean(perm != np.arange(n))),
                           permutation_strata=np.asarray(strata, dtype=str))
    observed = np.isfinite(M).any(axis=1)
    complete = np.isfinite(M[observed]).all(axis=0) if observed.any() else np.zeros(n, dtype=bool)
    diagnostics.update(training_sources=n, complete_training_sources=int(complete.sum()),
                       excluded_training_sources=np.flatnonzero(~complete),
                       unobserved_training_probes=np.flatnonzero(~observed),
                       training_span_dimension=span.shape[1])
    kind = name.removeprefix("B0-") if name.startswith("B0-") else None
    categories, scale = (), None
    basis = span
    if kind:
        categories = tuple(sorted(set(map(str, metadata["category"])))) if kind == "category" else ()
        features = _baseline_features(X, kind, metadata, categories)
        scale = np.sqrt(np.mean(features ** 2, axis=0))
        scale = np.where(scale > 0, scale, 1.0)
        features = features / scale
        R = np.full((M.shape[0], features.shape[1]), np.nan)
    else:
        if name == "B0":
            common = X.mean(axis=0)
            if np.linalg.norm(common) > np.finfo(float).eps * max(1.0, np.linalg.norm(X)):
                basis = (common / np.linalg.norm(common))[:, None]
            else:
                basis = span[:, :1]
        elif name == "B1":
            basis = span[:, :rank]
        elif name not in {"B-null", "B2"} and not name.startswith("B3"):
            raise ValueError(f"unknown model {name}")
        features = X @ basis
        R = np.full((M.shape[0], k), np.nan)
    if name == "B-null":
        R[:] = 0
    elif complete.any() and observed.any():
        inputs, targets = features[complete], M[np.ix_(observed, complete)]
        local = (reduced_rank_ridge(inputs, targets, rank, ridge)
                 if name == "B2" or name.startswith("B3") else _ridge(inputs, targets, ridge))
        R[observed] = local if kind else local @ basis.T
    valid = np.isfinite(R).all(axis=1)
    singular = np.linalg.svd(R[valid], compute_uv=False)
    return ResponseModel(name, R, basis, _rank(singular, R[valid].shape), ridge, rank,
                         diagnostics, feature_kind=kind, categories=categories, feature_scale=scale)


def grouped_folds(groups, folds) -> list[tuple[np.ndarray, np.ndarray]]:
    """Validate supplied fold labels or explicit {train: [...], test: [...]} folds."""
    groups = np.asarray(groups, dtype=str)
    n = len(groups)
    if groups.ndim != 1 or n == 0:
        raise ValueError("source groups must be a nonempty 1D array")
    if len(folds) and isinstance(folds[0], Mapping):
        result = []
        for fold in folds:
            arrays = [np.asarray(fold[key]) for key in ("train", "test")]
            if any(a.ndim != 1 or not np.issubdtype(a.dtype, np.integer) for a in arrays):
                raise ValueError("explicit fold indices must be integer vectors")
            result.append(tuple(arrays))
    else:
        labels = np.asarray(folds, dtype=str)
        if labels.shape != (n,):
            raise ValueError("one fixed fold label is required per source")
        result = [(np.flatnonzero(labels != label), np.flatnonzero(labels == label))
                  for label in sorted(set(labels))]
    seen = np.zeros(n, dtype=int)
    group_test_fold = {}
    for f, (train, test) in enumerate(result):
        for indices in (train, test):
            if not len(indices) or len(np.unique(indices)) != len(indices) or (indices < 0).any() or (indices >= n).any():
                raise ValueError("empty, duplicate, or out-of-range fold indices")
        if set(groups[train]) & set(groups[test]):
            raise ValueError("parent-task group leakage between train and test")
        for group in groups[test]:
            if group in group_test_fold and group_test_fold[group] != f:
                raise ValueError("one parent-task group appears in multiple test folds")
            group_test_fold[group] = f
        seen[test] += 1
    if not np.all(seen == 1):
        raise ValueError("each source must be held out exactly once")
    return result


def _inner_folds(groups, count: int):
    groups = np.asarray(groups, dtype=str)
    unique = sorted(set(groups))
    if len(unique) < 2:
        raise ValueError("inner grouped CV requires at least two training groups")
    mapping = {g: i % min(count, len(unique)) for i, g in enumerate(unique)}
    return grouped_folds(groups, [mapping[g] for g in groups])


def select_and_fit(X, M, groups, *, name: str, config: FitConfig,
                   metadata: Mapping | None = None, strata=None, permutation_seed=None,
                   candidate_ranks=None, inner_targets=None) -> ResponseModel:
    """Nested selection of rank and lambda; even inner bases see training X only."""
    X, M, groups = np.asarray(X), np.asarray(M), np.asarray(groups)
    if name == "B-null":
        return fit_model(X, M, name=name, rank=1, ridge=0)
    ranks = config.ranks if name in {"B1", "B2"} or name.startswith("B3") else (1,)
    if candidate_ranks is not None:
        ranks = candidate_ranks
    splits, scores = _inner_folds(groups, config.inner_splits), []
    for rank in ranks:
        for ridge in config.lambdas:
            total, count, fold_mses = 0.0, 0, []
            for j, (train, valid) in enumerate(splits):
                train_M, valid_M = inner_targets[j] if inner_targets is not None else (M[:, train], M[:, valid])
                model = fit_model(X[train], train_M, name=name, rank=rank, ridge=ridge,
                                  metadata=_metadata_slice(metadata, train),
                                  strata=None if strata is None else np.asarray(strata)[train],
                                  permutation_seed=permutation_seed)
                pred = model.predict(X[valid], _metadata_slice(metadata, valid))
                mask = np.isfinite(pred) & np.isfinite(valid_M)
                error = float(np.sum((pred[mask] - valid_M[mask]) ** 2))
                total, count = total + error, count + int(mask.sum())
                fold_mses.append(error / mask.sum() if mask.any() else np.nan)
            scores.append({"rank": rank, "lambda": ridge, "mse": total / count if count else np.nan,
                           "valid_cells": count, "fold_mse": fold_mses})
    available = [s for s in scores if np.isfinite(s["mse"])]
    # No evaluable inner labels: preserve a diagnostic abstention, never tune on outer labels.
    best = min(available, key=lambda s: (s["mse"], s["rank"], s["lambda"])) if available else scores[0]
    model = fit_model(X, M, name=name, rank=best["rank"], ridge=best["lambda"],
                      metadata=metadata, strata=strata, permutation_seed=permutation_seed)
    if not available:
        model.R[:] = np.nan
        model.rank = 0
        model.diagnostics["selection_status"] = "no evaluable inner validation responses; abstained"
    model.cv_scores = scores
    return model


def _factor_labels(metadata, n):
    values = (metadata or {}).get("source_factor")
    if values is None or np.asarray(values).shape != (n,):
        raise ValueError("hierarchical models require one manifest source_factor per source")
    if any(not isinstance(v, (str, np.str_)) or not v for v in values):
        raise ValueError("source_factor labels must be nonempty strings")
    return np.asarray(values, dtype=str)


def _factor_pool(labels, minimum):
    mapping = {f: f if np.sum(labels == f) >= minimum else "other" for f in sorted(set(labels))}
    return mapping, np.asarray([mapping[f] for f in labels])


@dataclass
class HierarchicalResponseModel:
    """Block-structured R = R_global + sum(R_family) + R_residual.

    Each family block acts on its training code span through R_family=B V.T.
    Operator predictions need no source labels. Explicit legacy 'gated' mode
    applies only the labelled block (unknowns use 'other' if present); it does
    not satisfy parameter-response consistency and must not define recipes.
    """
    global_model: ResponseModel
    family_models: dict[str, ResponseModel]
    residual_model: ResponseModel | None
    factor_mapping: dict[str, str]
    diagnostics: dict = field(default_factory=dict)
    name: str = "hierarchical"
    mode: str = "operator"

    @property
    def R(self):
        return sum(model.R for model in self.blocks().values())

    @property
    def rank(self):
        return sum(model.rank for model in self.blocks().values())

    @property
    def requested_rank(self):
        return sum(model.requested_rank for model in self.blocks().values())

    def blocks(self):
        return {"global": self.global_model,
                **{f"family:{f}": model for f, model in self.family_models.items()},
                **({"residual": self.residual_model} if self.residual_model is not None else {})}

    def predict_levels(self, X, metadata=None, *, mode=None):
        X = _matrix(X, "X", finite=True)
        mode = self.mode if mode is None else mode
        if mode not in {"operator", "gated"}:
            raise ValueError("mode must be operator or gated")
        global_prediction = self.global_model.predict(X)
        family_prediction = np.zeros_like(global_prediction)
        if mode == "operator":
            for model in self.family_models.values():
                family_prediction += model.predict(X)
        else:
            labels = _factor_labels(metadata, len(X))
            pooled = np.asarray([self.factor_mapping.get(f, "other") for f in labels])
            for f, model in self.family_models.items():
                use = pooled == f
                if use.any():
                    family_prediction[:, use] = model.predict(X[use])
        residual_prediction = (self.residual_model.predict(X) if self.residual_model is not None
                               else np.zeros_like(global_prediction))
        return {"global": global_prediction, "family": family_prediction, "residual": residual_prediction}

    def predict(self, X, metadata=None, *, mode=None):
        mode = self.mode if mode is None else mode
        if mode == "operator":
            return ResponsePredictor(self).predict_from_codes(_matrix(X, "X", finite=True))
        return sum(self.predict_levels(X, metadata, mode=mode).values())

    def modes(self, C=None):
        return {name: model.modes(C) for name, model in self.blocks().items()}


def _select_block(candidates, validation):
    """Select a rank/lambda using pooled inner validation SSE, never outer labels."""
    scores = []
    for rank, ridge in candidates:
        total, count, fold_mse = 0.0, 0, []
        for train_X, train_M, valid_X, valid_M in validation:
            model = fit_model(train_X, train_M, rank=rank, ridge=ridge)
            P = model.predict(valid_X)
            mask = np.isfinite(P) & np.isfinite(valid_M)
            error = float(np.sum((P[mask] - valid_M[mask]) ** 2))
            total += error
            count += int(mask.sum())
            fold_mse.append(error / mask.sum() if mask.any() else np.nan)
        scores.append({"rank": rank, "lambda": ridge, "mse": total / count if count else np.nan,
                       "valid_cells": count, "fold_mse": fold_mse})
    available = [s for s in scores if np.isfinite(s["mse"])]
    best = min(available, key=lambda s: (s["mse"], s["rank"], s["lambda"])) if available else scores[0]
    return best, scores, bool(available)


def fit_hierarchical(X, M, groups, *, config=None, metadata=None, source_factors=None,
                     inner_targets=None, mode=None) -> HierarchicalResponseModel:
    """Greedy global rank-1, family rank-{1,2}, optional residual rank-K fit.

    Each level selects lambda from the same frozen grid by grouped inner CV.
    Upstream blocks and rare-family pooling are rebuilt on each inner training
    set before residuals are formed. Earlier selected hyperparameters are held
    fixed while selecting the next level. No centring or free intercept is used.
    In default operator mode, family labels select training subsets, but every
    fitted family operator acts on all codes. Residual targets subtract that
    same operator sum in both training and validation. ``mode='gated'`` retains
    the legacy label-dependent estimator for comparisons only.
    """
    config = config or FitConfig(model="hierarchical")
    mode = config.hierarchical_mode if mode is None else mode
    if mode not in {"operator", "gated"}:
        raise ValueError("mode must be operator or gated")
    X, M = _matrix(X, "X", finite=True), _matrix(M, "M")
    if M.shape[1] != len(X) or np.shape(groups) != (len(X),):
        raise ValueError("hierarchical response/group axes differ from X")
    metadata = dict(metadata or {})
    if source_factors is not None:
        metadata["source_factor"] = source_factors
    labels = _factor_labels(metadata, len(X))
    mapping, pooled = _factor_pool(labels, config.min_family_sources)
    splits = _inner_folds(groups, config.inner_splits)
    global_model = select_and_fit(X, M, groups, name="B2", config=config,
                                  candidate_ranks=(1,), inner_targets=inner_targets)
    remaining = M - global_model.predict(X)
    # Cache only inner-training fits; validation responses are used for scoring.
    inner = []
    for j, (train, valid) in enumerate(splits):
        train_M, valid_M = inner_targets[j] if inner_targets is not None else (M[:, train], M[:, valid])
        model = fit_model(X[train], train_M, rank=1, ridge=global_model.ridge)
        pool_map, train_labels = _factor_pool(labels[train], config.min_family_sources)
        inner.append({"train": train, "valid": valid,
                      "train_M": train_M - model.predict(X[train]),
                      "valid_M": valid_M - model.predict(X[valid]),
                      "train_labels": train_labels,
                      "valid_labels": np.asarray([pool_map.get(f, "other") for f in labels[valid]]),
                      "mapping": pool_map})
    family_models = {}
    for f in sorted(set(pooled)):
        validation = []
        for fold in inner:
            train_use, valid_use = fold["train_labels"] == f, fold["valid_labels"] == f
            if train_use.any() and valid_use.any():
                validation.append((X[fold["train"]][train_use], fold["train_M"][:, train_use],
                                   X[fold["valid"]][valid_use], fold["valid_M"][:, valid_use]))
        best, scores, available = _select_block(
            [(k, ridge) for k in config.family_ranks for ridge in config.lambdas], validation)
        use = pooled == f
        model = fit_model(X[use], remaining[:, use], rank=best["rank"], ridge=best["lambda"])
        model.cv_scores = scores
        if not available:
            # Retain the existing abstention policy for unidentifiable selections.
            model.R[:] = np.nan
            model.rank = 0
            model.diagnostics["selection_status"] = "no evaluable inner family responses; abstained"
        family_models[f] = model
    hierarchy = HierarchicalResponseModel(global_model, family_models, None, mapping, mode=mode)
    remaining -= hierarchy.predict_levels(X, metadata)["family"]
    if config.residual_rank:
        validation = []
        for fold in inner:
            train, valid = fold["train"], fold["valid"]
            train_M, valid_M = fold["train_M"].copy(), fold["valid_M"].copy()
            for f in sorted(set(fold["train_labels"])):
                # An inner-only 'other' pool uses a predeclared grid fallback.
                selected = family_models.get(f)
                rank = selected.requested_rank if selected is not None else config.family_ranks[0]
                ridge = selected.ridge if selected is not None else config.lambdas[0]
                use, test_use = fold["train_labels"] == f, fold["valid_labels"] == f
                block = fit_model(X[train][use], fold["train_M"][:, use], rank=rank, ridge=ridge)
                if mode == "operator":
                    train_M -= block.predict(X[train])
                    valid_M -= block.predict(X[valid])
                else:
                    train_M[:, use] -= block.predict(X[train][use])
                    valid_M[:, test_use] -= block.predict(X[valid][test_use])
            validation.append((X[train], train_M, X[valid], valid_M))
        best, scores, available = _select_block([(config.residual_rank, r) for r in config.lambdas], validation)
        residual = fit_model(X, remaining, rank=config.residual_rank, ridge=best["lambda"])
        residual.cv_scores = scores
        if not available:
            residual.R[:] = np.nan
            residual.rank = 0
            residual.diagnostics["selection_status"] = "no evaluable inner residual responses; abstained"
        hierarchy.residual_model = residual
    hierarchy.diagnostics = {
        "preprocessing": "sequential training-only H-metric reduced-rank ridge; no intercept",
        "factor_mapping": mapping, "family_training_sources": {f: int(np.sum(pooled == f)) for f in family_models},
        "inner_factor_mappings": [f["mapping"] for f in inner],
        "mode": mode,
        "unknown_factor_policy": ("labels ignored; all blocks act through R" if mode == "operator" else
                                  "training other pool if present, otherwise global + residual only"),
        "total_rank_budget": hierarchy.requested_rank, "total_fitted_rank": hierarchy.rank,
        "rank_definition": "sum of block ranks (upper bound on the rank of their operator sum)"}
    return hierarchy


def _ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ordered = values[order]
    starts = np.r_[0, np.flatnonzero(ordered[1:] != ordered[:-1]) + 1]
    ends = np.r_[starts[1:], len(values)]
    result = np.empty(len(values), dtype=float)
    for start, end in zip(starts, ends):
        result[order[start:end]] = (start + end - 1) / 2 + 1
    return result


def spearman(actual, predicted) -> float:
    actual, predicted = np.asarray(actual, dtype=float).ravel(), np.asarray(predicted, dtype=float).ravel()
    valid = np.isfinite(actual) & np.isfinite(predicted)
    if valid.sum() < 2 or np.ptp(actual[valid]) == 0 or np.ptp(predicted[valid]) == 0:
        return float("nan")
    return float(np.corrcoef(_ranks(actual[valid]), _ranks(predicted[valid]))[0, 1])


def auroc(labels, scores) -> float:
    labels, scores = np.asarray(labels, dtype=float).ravel(), np.asarray(scores, dtype=float).ravel()
    valid = np.isfinite(labels) & np.isfinite(scores)
    labels, scores = labels[valid], scores[valid]
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("AUROC labels must be binary or NA")
    positives, negatives = int(np.sum(labels == 1)), int(np.sum(labels == 0))
    if positives == 0 or negatives == 0:
        return float("nan")
    return float((_ranks(scores)[labels == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def _finite_mean(values) -> float:
    values = np.asarray(values, dtype=float)
    return float(np.mean(values[np.isfinite(values)])) if np.isfinite(values).any() else float("nan")


def _summary(M: np.ndarray, P: np.ndarray) -> dict:
    valid = np.isfinite(M) & np.isfinite(P)
    truth, predicted = M[valid], P[valid]
    mse = float(np.mean((truth - predicted) ** 2)) if len(truth) else float("nan")
    zero_mse = float(np.mean(truth ** 2)) if len(truth) else float("nan")
    nonzero = truth != 0
    # Full net benefit is defined only if every fixed probe is evaluable.
    complete = valid.all(axis=0)
    net, predicted_net = M[:, complete].sum(axis=0), P[:, complete].sum(axis=0)
    return {"mse": mse, "zero_mse": zero_mse, "improvement_vs_zero": zero_mse - mse,
            "relative_improvement_vs_zero": 1 - mse / zero_mse if zero_mse > 0 else np.nan,
            "spearman": spearman(truth, predicted), "sign_auroc": auroc(truth > 0, predicted),
            "damage_auroc": auroc(truth < 0, -predicted),
            "nonzero_sign_auroc": auroc(truth[nonzero] > 0, predicted[nonzero]),
            "nonzero_coverage": float(np.mean(nonzero)) if len(truth) else np.nan,
            "net_benefit_spearman": spearman(net, predicted_net),
            "net_benefit_sources": int(complete.sum()), "evaluated_cells": int(valid.sum())}


def evaluate(M, predictions, *, source_groups=None, probe_groups=None,
             source_factors=None, probe_factors=None) -> dict:
    """Per-source (over probes) and pooled metrics, including undefined reasons.

    Sign AUROC is positive vs nonpositive, retaining zeros. Damage AUROC is
    negative vs nonnegative. A separate nonzero-only sign metric is explicitly
    labelled with coverage. Net benefit = sum(max(M,0)) - sum(max(-M,0)); damage
    has unit weight. Partial observed sums are labelled and never rank as totals.
    """
    M, P = _matrix(M, "M"), _matrix(predictions, "predictions")
    if M.shape != P.shape:
        raise ValueError("response/prediction shapes differ")
    source_groups = np.asarray(source_groups if source_groups is not None else np.arange(M.shape[1]), dtype=str)
    probe_groups = np.asarray(probe_groups if probe_groups is not None else np.arange(M.shape[0]), dtype=str)
    if source_groups.shape != (M.shape[1],) or probe_groups.shape != (M.shape[0],):
        raise ValueError("group axes differ from response matrix")
    valid, observed = np.isfinite(M) & np.isfinite(P), np.isfinite(M)
    per_source = [_summary(M[:, i:i + 1], P[:, i:i + 1]) for i in range(M.shape[1])]
    report = _summary(M, P)
    report["per_source"] = per_source
    report["macro_source_spearman"] = _finite_mean([x["spearman"] for x in per_source])
    report["macro_source_sign_auroc"] = _finite_mean([x["sign_auroc"] for x in per_source])
    report["metric_source_counts"] = {key: sum(np.isfinite(s[key]) for s in per_source)
                                      for key in ("mse", "spearman", "sign_auroc", "damage_auroc")}
    effective_source_groups = len(set(source_groups[observed.any(axis=0)]))
    effective_probe_groups = len(set(probe_groups[observed.any(axis=1)]))
    report["counts"] = {
        "sources": M.shape[1], "independent_sources": effective_source_groups,
        "source_groups": effective_source_groups, "probe_groups": effective_probe_groups,
        "total_source_groups": len(set(source_groups)), "total_probe_groups": len(set(probe_groups)),
        "probes": M.shape[0], "sources_with_observations": int(observed.any(axis=0).sum()),
        "sources_with_any_change": int(((M != 0) & observed).any(axis=0).sum()),
        "source_groups_with_any_change": len(set(source_groups[((M != 0) & observed).any(axis=0)])),
        "changed_cells": int(((M != 0) & observed).sum()),
        "observed_cells": int(observed.sum()), "missing_cells": int((~observed).sum()),
        "missing_predictions_on_observed": int((observed & ~valid).sum()),
        "observed_fraction": float(observed.mean()),
    }
    values = M[observed]
    report["fractions"] = {name: float(np.mean(mask)) if len(values) else np.nan
                           for name, mask in (("positive", values > 0), ("negative", values < 0), ("zero", values == 0))}
    report["net_benefit"] = []
    for i in range(M.shape[1]):
        ok = valid[:, i]
        positive = float(np.maximum(M[ok, i], 0).sum()) if ok.any() else np.nan
        damage = float(np.maximum(-M[ok, i], 0).sum()) if ok.any() else np.nan
        report["net_benefit"].append({"source_index": i, "probe_coverage": float(ok.mean()),
                                      "observed_positive": positive, "observed_damage": damage,
                                      "observed_net": positive - damage,
                                      "actual": float(M[:, i].sum()) if ok.all() else np.nan,
                                      "predicted": float(P[:, i].sum()) if ok.all() else np.nan})
    report["source_ranking"] = sorted((i for i in range(M.shape[1]) if valid[:, i].all()),
                                       key=lambda i: (-report["net_benefit"][i]["predicted"], i))
    report["degenerate_rows"] = []
    for axis, array, pred in (("probe", M, P), ("source", M.T, P.T)):
        for i, (row, estimate) in enumerate(zip(array, pred)):
            mask = np.isfinite(row) & np.isfinite(estimate)
            y, p = row[mask], estimate[mask]
            reasons = []
            if not len(y):
                reasons.append("no paired observations")
            else:
                if len(y) < 2 or np.ptp(y) == 0:
                    reasons.append("constant/insufficient response: Spearman NA")
                if len(y) < 2 or np.ptp(p) == 0:
                    reasons.append("constant/insufficient prediction: Spearman NA")
                if len(set(y > 0)) < 2:
                    reasons.append("one sign class: sign AUROC NA")
            if reasons:
                report["degenerate_rows"].append({"axis": axis, "index": i, "n": len(y), "reasons": reasons})
    if source_factors is not None or probe_factors is not None:
        factors = {}
        for axis, labels, size in (("source_factor", source_factors, M.shape[1]),
                                   ("probe_factor", probe_factors, M.shape[0])):
            if labels is None:
                continue
            labels = np.asarray(labels, dtype=str)
            if labels.shape != (size,):
                raise ValueError(f"{axis} axis differs from response matrix")
            factors[axis] = {}
            for label in sorted(set(labels)):
                rows = np.flatnonzero(labels == label) if axis == "probe_factor" else np.arange(M.shape[0])
                cols = np.flatnonzero(labels == label) if axis == "source_factor" else np.arange(M.shape[1])
                ix = np.ix_(rows, cols)
                entry = _summary(M[ix], P[ix])
                complete = valid[ix].all(axis=0)
                eligible = cols[complete]
                net = P[np.ix_(rows, eligible)].sum(axis=0)
                entry["source_ranking"] = eligible[np.argsort(-net, kind="stable")].tolist()
                entry["actual_net"] = M[np.ix_(rows, eligible)].sum(axis=0)
                entry["predicted_net"] = net
                entry["source_indices"] = eligible
                factors[axis][label] = entry
        report["per_factor"] = factors
    return report


def explained_levels(M, levels):
    """Sequential held-out SSE reductions on identical cells; negatives retained.

    The denominator is uncentred response energy (variance about the required
    zero origin), consistent with improvement_vs_zero and the no-intercept fit.
    """
    valid = np.isfinite(M)
    for P in levels.values():
        valid &= np.isfinite(P)
    target, prediction = M[valid], np.zeros(int(valid.sum()))
    energy = float(np.sum(target ** 2))
    previous, report = energy, {}
    for name, P in levels.items():
        prediction += P[valid]
        sse = float(np.sum((target - prediction) ** 2))
        report[name] = {"incremental_explained_variance": (previous - sse) / energy if energy > 0 else np.nan,
                        "cumulative_explained_variance": 1 - sse / energy if energy > 0 else np.nan,
                        "sse_reduction": previous - sse, "remaining_sse": sse,
                        "evaluated_cells": int(valid.sum()), "zero_energy": energy,
                        "denominator": "held-out squared response about zero"}
        previous = sse
    return report


def group_bootstrap(M, predictions: Mapping[str, np.ndarray], source_groups, *,
                    probe_groups=None, samples=200, seed=0, confidence=0.95, two_way=False,
                    candidate="B2") -> dict:
    """Paired OOF group bootstrap; all models and contrasts use the same cells."""
    M = _matrix(M, "M")
    source_groups = np.asarray(source_groups, dtype=str)
    probe_groups = np.asarray(probe_groups if probe_groups is not None else np.arange(M.shape[0]), dtype=str)
    if source_groups.shape != (M.shape[1],) or probe_groups.shape != (M.shape[0],):
        raise ValueError("bootstrap group axes differ from M")
    if samples < 0 or not 0 < confidence < 1:
        raise ValueError("invalid bootstrap settings")
    common = np.isfinite(M)
    for P in predictions.values():
        if np.shape(P) != M.shape:
            raise ValueError("prediction shape differs from M")
        common &= np.isfinite(P)
    target = np.where(common, M, np.nan)
    sg, pg = sorted(set(source_groups)), sorted(set(probe_groups))
    keys = ("mse", "improvement_vs_zero", "relative_improvement_vs_zero", "spearman", "sign_auroc",
            "damage_auroc", "net_benefit_spearman")
    draws = {name: {key: [] for key in keys} for name in predictions}
    contrasts = {f"{candidate}_vs_{name}": [] for name in predictions
                 if name not in {candidate, "B-null"}} if candidate in predictions else {}
    rng = np.random.default_rng(seed)
    enough = len(sg) >= 2 and (not two_way or len(pg) >= 2)
    for _ in range(samples if enough else 0):
        cols = np.concatenate([np.flatnonzero(source_groups == g) for g in rng.choice(sg, len(sg), replace=True)])
        rows = (np.concatenate([np.flatnonzero(probe_groups == g) for g in rng.choice(pg, len(pg), replace=True)])
                if two_way else np.arange(M.shape[0]))
        ix = np.ix_(rows, cols)
        stats = {name: _summary(target[ix], P[ix]) for name, P in predictions.items()}
        for name, summary in stats.items():
            for key in keys:
                draws[name][key].append(summary[key])
        for name in predictions:
            if f"{candidate}_vs_{name}" in contrasts:
                contrasts[f"{candidate}_vs_{name}"].append(stats[name]["mse"] - stats[candidate]["mse"])
    def interval(values):
        values = np.asarray(values)
        values = values[np.isfinite(values)]
        bounds = np.quantile(values, [(1 - confidence) / 2, (1 + confidence) / 2]) if len(values) >= 2 else [np.nan, np.nan]
        return {"low": float(bounds[0]), "high": float(bounds[1]), "valid_resamples": len(values), "requested_resamples": samples}
    return {"models": {name: {key: interval(values) for key, values in series.items()} for name, series in draws.items()},
            "paired_mse_improvement": {name: interval(values) for name, values in contrasts.items()},
            "method": "paired source-group bootstrap" + (" + independent probe-group bootstrap" if two_way else " (fixed probes)"),
            "seed": seed, "confidence": confidence,
            "limitation": "OOF predictions held fixed; not independent training seeds or full algorithm retraining"}


@dataclass
class CrossFitResult:
    predictions: dict[str, np.ndarray]
    folds: list[dict]
    metrics: dict
    confidence_intervals: dict
    diagnostics: dict
    level_predictions: dict[str, np.ndarray] = field(default_factory=dict)
    target_matrix: np.ndarray | None = None
    error_bounds: dict[str, np.ndarray] = field(default_factory=dict)


def residual_error_bounds(residuals, *, quantile=0.95):
    """Per-probe empirical quantiles of |grouped held-out residual|, shape (m, 1).

    Input is (probe, held-out source), with one OOF prediction per source from
    grouped cross-fitting. Finite cells are pooled equally within each probe;
    this does not give groups equal weight. NA cells are excluded, and probes
    without any observed residual retain NA (never a zero error allowance).
    The column broadcasts over a new pool's sources as E_ji in recipe solvers.

    This is an EMPIRICAL QUANTILE, NOT A CONFIDENCE BOUND or simultaneous error
    guarantee. In particular it does not certify new-source coverage or account
    for model selection on these same OOF errors. Use separate calibration data
    to evaluate coverage; infeasibility under E does not prove true infeasibility.
    """
    residuals = _matrix(residuals, "residuals")
    if not np.isfinite(quantile) or not 0 <= quantile <= 1:
        raise ValueError("quantile must be between zero and one")
    bounds = np.full((len(residuals), 1), np.nan)
    for j, row in enumerate(residuals):
        valid = np.isfinite(row)
        if valid.any():
            bounds[j, 0] = np.quantile(np.abs(row[valid]), quantile)
    return bounds


def cross_fit(X, M, source_groups, folds, *, config: FitConfig | None = None,
              probe_groups=None, metadata: Mapping | None = None, strata=None,
              probe_factors=None, student_response=None) -> CrossFitResult:
    config = config or FitConfig()
    X, M = _matrix(X, "X", finite=True), _matrix(M, "M")
    if X.shape[0] != M.shape[1]:
        raise ValueError("X=(sources,codes), M=(probes,sources) axes do not match")
    source_groups = np.asarray(source_groups, dtype=str)
    if source_groups.shape != (X.shape[0],):
        raise ValueError("one parent-task group required per source")
    splits = grouped_folds(source_groups, folds)
    metadata = metadata or {}
    if any(np.asarray(v).shape != (X.shape[0],) for v in metadata.values()):
        raise ValueError("metadata must have one scalar per source")
    hierarchical = config.model == "hierarchical"
    factors = _factor_labels(metadata, len(X)) if hierarchical else metadata.get("source_factor")
    strata = np.asarray(strata if strata is not None else metadata.get(
        "category", factors if factors is not None else np.repeat("all", X.shape[0])), dtype=str)
    if strata.shape != (X.shape[0],):
        raise ValueError("one permutation stratum required per source")
    names = ["B-null", "B0", "B0-norm"]
    names += ["B0-" + key for key in ("category", "length") if key in metadata]
    names += ["B1", "B2"] + (["hierarchical", "B2-matched"] if hierarchical else [])
    names += [f"B3-seed{seed}" for seed in config.permutation_seeds]
    candidate = "hierarchical" if hierarchical else "B2"
    predictions, fitted = {name: np.full_like(M, np.nan) for name in names}, []
    levels = {level: np.full_like(M, np.nan) for level in ("global", "family", "residual")} if hierarchical else {}
    target = M.copy()
    common_predictions, common_folds = np.full_like(M, np.nan), []
    if student_response is not None:
        student_response = _matrix(student_response, "student_response")
        if student_response.shape != M.shape:
            raise ValueError("student response axes must match the combined response")
    for f, (train, test) in enumerate(splits):
        train_M, inner_targets = M[:, train], None
        if student_response is not None:
            common = select_and_fit(X[train], student_response[:, train], source_groups[train],
                                    name="B2", config=config, candidate_ranks=(1,))
            train_M = train_M + common.predict(X[train])
            common_predictions[:, test] = common.predict(X[test])
            target[:, test] = M[:, test] + common_predictions[:, test]
            modes = common.modes()
            common_folds.append({"fold": f, "train": train, "test": test, "R": common.R,
                                 "modes": modes, "lambda": common.ridge, "cv_scores": common.cv_scores,
                                 "heldout_coordinate": X[test] @ modes["V"]})
            inner_targets = []
            for inner_train, inner_valid in _inner_folds(source_groups[train], config.inner_splits):
                a, b = train[inner_train], train[inner_valid]
                nuisance = select_and_fit(X[a], student_response[:, a], source_groups[a],
                                         name="B2", config=config, candidate_ranks=(1,))
                inner_targets.append((M[:, a] + nuisance.predict(X[a]), M[:, b] + nuisance.predict(X[b])))
        models = {}
        for name in names:
            seed = int(name.removeprefix("B3-seed")) if name.startswith("B3-seed") else None
            if hierarchical and (name == "hierarchical" or seed is not None):
                fit_M, fit_inner = train_M, inner_targets
                if seed is not None:
                    permutation = permute_within_strata(strata[train], seed)
                    fit_M = train_M[:, permutation]
                    # Permute independently within each inner training partition.
                    # In particular, no outer training permutation crosses inner folds.
                    fit_inner = []
                    for j, (a, b) in enumerate(_inner_folds(source_groups[train], config.inner_splits)):
                        am, bm = inner_targets[j] if inner_targets is not None else (train_M[:, a], train_M[:, b])
                        fit_inner.append((am[:, permute_within_strata(strata[train][a], seed)], bm))
                model = fit_hierarchical(X[train], fit_M, source_groups[train], config=config,
                                         metadata=_metadata_slice(metadata, train), inner_targets=fit_inner)
                model.name = name
                if seed is not None:
                    model.diagnostics.update(permutation=permutation, permutation_seed=seed,
                        permutation_moved_fraction=float(np.mean(permutation != np.arange(len(train)))),
                        permutation_strata=strata[train])
            else:
                matched_rank = models["hierarchical"].requested_rank if name == "B2-matched" else None
                model = select_and_fit(X[train], train_M, source_groups[train],
                                       name="B2" if matched_rank is not None else name,
                                       config=config, metadata=_metadata_slice(metadata, train),
                                       strata=strata[train], permutation_seed=seed,
                                       candidate_ranks=(matched_rank,) if matched_rank is not None else None,
                                       inner_targets=inner_targets)
                if matched_rank is not None:
                    model.name = name
                    model.diagnostics["matched_total_rank_budget"] = matched_rank
            models[name] = model
            predictions[name][:, test] = model.predict(X[test], _metadata_slice(metadata, test))
            if name == "hierarchical":
                for level, P in model.predict_levels(X[test], _metadata_slice(metadata, test)).items():
                    levels[level][:, test] = P
        basis = training_span(X[train])
        residuals = np.linalg.norm(X[test] - (X[test] @ basis) @ basis.T, axis=1)
        fitted.append({"fold": f, "train": train, "test": test, "models": models,
                       "heldout_projection_residual_norm": residuals})
    common = np.isfinite(target)
    for prediction in predictions.values():
        common &= np.isfinite(prediction)
    fair_M = np.where(common, target, np.nan)
    metrics = {name: evaluate(fair_M, prediction, source_groups=source_groups, probe_groups=probe_groups,
                             source_factors=factors, probe_factors=probe_factors)
               for name, prediction in predictions.items()}
    ci = group_bootstrap(target, predictions, source_groups, probe_groups=probe_groups,
                         samples=config.bootstrap_samples, seed=config.bootstrap_seed,
                         confidence=config.confidence, two_way=config.two_way_bootstrap, candidate=candidate)
    robustness = {}
    for name in ("B0", "B1"):
        fold_deltas = [_summary(fair_M[:, f["test"]], predictions[name][:, f["test"]])["mse"] -
                       _summary(fair_M[:, f["test"]], predictions[candidate][:, f["test"]])["mse"] for f in fitted]
        leave_group_out = []
        for group in sorted(set(source_groups)):
            keep = source_groups != group
            leave_group_out.append(_summary(fair_M[:, keep], predictions[name][:, keep])["mse"] -
                                   _summary(fair_M[:, keep], predictions[candidate][:, keep])["mse"])
        robustness[name] = {"fold_improvements": fold_deltas, "leave_one_group_out_improvements": leave_group_out}
    diagnostics = {"candidate": candidate, "observations": evaluate(target, np.zeros_like(M), source_groups=source_groups,
                                              probe_groups=probe_groups)["counts"],
                   "common_evaluation_cells": int(common.sum()),
                   "observed_cells_excluded_from_comparison": int((np.isfinite(target) & ~common).sum()),
                   "null_on_all_observed": _summary(target, np.zeros_like(M)),
                   "robustness": robustness, "strata": strata,
                   "permutation_moved_fraction": {name: [f["models"][name].diagnostics["permutation_moved_fraction"]
                                                         for f in fitted] for name in names if name.startswith("B3")},
                   "scope": "new sources, fixed known probes; no unseen-probe prediction claim"}
    if hierarchical:
        diagnostics["explained_variance_by_level"] = explained_levels(fair_M, levels)
        diagnostics["matched_rank_comparison"] = [
            {"fold": f["fold"], "total_rank_budget": f["models"]["hierarchical"].requested_rank,
             "hierarchical_fitted_rank": f["models"]["hierarchical"].rank,
             "flat_fitted_rank": f["models"]["B2-matched"].rank,
             "hierarchical_mse": _summary(fair_M[:, f["test"]], predictions["hierarchical"][:, f["test"]])["mse"],
             "flat_B2_mse": _summary(fair_M[:, f["test"]], predictions["B2-matched"][:, f["test"]])["mse"]}
            for f in fitted]
        diagnostics["matched_rank_note"] = "equal sum-of-block rank budget; flat effective rank is capped by code/probe dimensions"
    if student_response is not None:
        diagnostics["student_common_mode"] = {
            "definition": "combined = teacher - student + fitted_student_common; one student erasure coordinate",
            "raw_margin_prediction": {name: P - common_predictions for name, P in predictions.items()},
            "prediction": common_predictions, "folds": common_folds,
            "fitting": "student-only rank-1 reduced-rank ridge; removed anew in each inner training split"}
    errors = {name: residual_error_bounds(target - P, quantile=config.error_quantile)
              for name, P in predictions.items()}
    diagnostics["estimation_error"] = {
        "quantile": config.error_quantile,
        "method": "per-probe empirical quantile of absolute grouped cross-fitting residuals",
        "is_confidence_bound": False,
        "finite_residual_counts": {name: np.sum(np.isfinite(target) & np.isfinite(P), axis=1)
                                   for name, P in predictions.items()},
        "limitation": "no simultaneous or new-source coverage guarantee; unobserved probes remain NA"}
    return CrossFitResult(predictions, fitted, metrics, ci, diagnostics, levels, target, errors)


def cross_fit_targets(X, teacher, student, source_groups, folds, **kwargs):
    """Independent teacher/student fits plus a combined target without erasure.

    Teacher may be a saved teacher log-likelihood or margin delta. Combined
    subtraction requires comparable teacher/student log-likelihood definitions;
    a teacher margin matrix should only be fitted independently.
    """
    teacher, student = _matrix(teacher, "teacher"), _matrix(student, "student")
    if teacher.shape != student.shape:
        raise ValueError("teacher/student response axes differ")
    return {"teacher": cross_fit(X, teacher, source_groups, folds, **kwargs),
            "student": cross_fit(X, student, source_groups, folds, **kwargs),
            "combined": cross_fit(X, teacher - student, source_groups, folds,
                                  student_response=student, **kwargs)}


@dataclass(frozen=True)
class DecisionConfig:
    """All scientific success/power thresholds must be frozen before fitting."""
    min_source_groups: int
    min_probe_groups: int
    min_changed_sources: int
    min_observed_fraction: float
    max_ci_width: float
    min_zero_improvement: float
    min_baseline_improvement: float
    min_permutation_gap: float
    min_net_benefit_spearman: float
    min_leave_group_out_improvement: float
    min_fold_win_fraction: float
    min_permutation_moved_fraction: float

    def __post_init__(self):
        if any(not np.isfinite(v) for v in self.__dict__.values()):
            raise ValueError("decision thresholds must be finite")
        for key in ("min_observed_fraction", "min_fold_win_fraction", "min_permutation_moved_fraction"):
            if not 0 <= getattr(self, key) <= 1:
                raise ValueError(f"{key} must be in [0, 1]")
        if self.max_ci_width < 0 or min(self.min_source_groups, self.min_probe_groups, self.min_changed_sources) < 0:
            raise ValueError("power/count thresholds cannot be negative")


def stage_decision(result: CrossFitResult | Mapping, config: DecisionConfig | None, *,
                   measurement_stable: bool | None = None, measurement_resolved: bool | None = None,
                   response_kind: str = "behavior") -> dict:
    """Conservative gate using grouped intervals, permutation and influence checks.

    A GO-candidate authorises a P3 review, not a claim that atoms are established.
    Missing frozen thresholds or measurement evidence is INCONCLUSIVE.
    """
    def decision(label, reason):
        return {"decision": label, "reason": reason}
    if config is None:
        return decision("INCONCLUSIVE", "scientific thresholds were not supplied/frozen")
    if measurement_stable is not True or measurement_resolved is not True:
        return decision("INCONCLUSIVE", "measurement stability/resolution has not been demonstrated")
    if response_kind != "behavior":
        return decision("NO-GO", "likelihood-only improvement cannot pass the behaviour gate")
    data = result.__dict__ if isinstance(result, CrossFitResult) else result
    metrics, ci, diagnostics = data["metrics"], data["confidence_intervals"], data["diagnostics"]
    candidate = diagnostics.get("candidate", "B2")
    counts = metrics[candidate]["counts"]
    if (counts["source_groups"] < config.min_source_groups or counts["probe_groups"] < config.min_probe_groups
            or counts["sources_with_any_change"] < config.min_changed_sources
            or counts["observed_fraction"] < config.min_observed_fraction):
        return decision("INCONCLUSIVE", "insufficient independent groups, changed sources, or response coverage")
    zero = ci["models"][candidate]["improvement_vs_zero"]
    pairs = ci["paired_mse_improvement"]
    required = [zero] + [pairs[f"{candidate}_vs_{name}"] for name in ("B0", "B1")]
    finite = lambda v: v is not None and bool(np.isfinite(v))
    if any(not finite(v["low"]) or not finite(v["high"]) or v["high"] - v["low"] > config.max_ci_width for v in required):
        return decision("INCONCLUSIVE", "grouped uncertainty is undefined or exceeds the frozen width limit")
    controls = [key for key in pairs if key.startswith(f"{candidate}_vs_B3")]
    perm_ok = bool(controls) and all(finite(pairs[key]["low"]) and pairs[key]["low"] >= config.min_permutation_gap for key in controls)
    perm_ok = perm_ok and all(min(values) >= config.min_permutation_moved_fraction
                              for values in diagnostics["permutation_moved_fraction"].values())
    robust = True
    for name in ("B0", "B1"):
        info = diagnostics["robustness"][name]
        fold = np.asarray(info["fold_improvements"], dtype=float)
        leave = np.asarray(info["leave_one_group_out_improvements"], dtype=float)
        robust &= (np.isfinite(fold).all() and np.isfinite(leave).all()
                   and np.mean(fold >= config.min_baseline_improvement) >= config.min_fold_win_fraction
                   and np.min(leave) >= config.min_leave_group_out_improvement)
    if (zero["low"] >= config.min_zero_improvement
            and all(pairs[f"{candidate}_vs_{name}"]["low"] >= config.min_baseline_improvement for name in ("B0", "B1"))
            and finite(metrics[candidate]["net_benefit_spearman"])
            and metrics[candidate]["net_benefit_spearman"] >= config.min_net_benefit_spearman and perm_ok and robust):
        return decision("GO-candidate", f"{candidate} exceeds zero/simple baselines with grouped support, failed permutation controls, and distributed gains")
    for name in ("B0", "B1"):
        baseline = ci["models"][name]["improvement_vs_zero"]
        if finite(baseline["low"]) and baseline["low"] >= config.min_zero_improvement and pairs[f"{candidate}_vs_{name}"]["high"] < config.min_baseline_improvement:
            return decision("SIMPLIFY", f"{name} predicts useful structure; {candidate} has no meaningful incremental gain within the frozen bound")
    if zero["high"] < config.min_zero_improvement:
        return decision("NO-GO", f"resolved measurements bound {candidate} below the minimum meaningful prediction improvement")
    return decision("INCONCLUSIVE", "the frozen criteria do not establish gain, simplification, or futility; inspect permutation and group influence")
