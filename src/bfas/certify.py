"""Block IV — certification primitives (spec §6, METHOD.md §6).

Three certifiable quantities, each with an explicit finite-sample guarantee:

1. **Representation coverage** of a capability dictionary ``U`` (d, K) on held-out
   fingerprints: per-event ``c_i = 1 - min_z ||psi_i - U z||^2 / ||psi_i||^2``
   (= ``||U U^T psi||^2 / ||psi||^2`` for orthonormal ``U``); the set-level spec
   quantity ``1 - min_Z ||Psi - U Z||_F^2 / ||Psi||_F^2`` and percentile-bootstrap CIs
   over events, with a random-basis and a self-fitted-PCA reference and a group-aware
   cross-fitted estimate.

2. **Atom-wise residual certificate**: ``J_k = sum_i w_ki v_i`` with
   ``w_ki = zbar_ki / sum_i zbar_ki`` and ``v_i in [0, 1]`` (``mirror.JEstimator``).
   For fixed weights and independent bounded ``v_i``, Hoeffding gives
   ``P(J_k - E J_k <= -t) <= exp(-2 t^2 / sum_i w_ki^2)``; solving for ``t`` at level
   ``delta / K`` (Bonferroni) yields simultaneous lower confidence bounds
   ``LCB[J_k]`` and ``UCB[r_k] = [rho_k - LCB[J_k]]_+`` over all atoms.
   ``n_k = sum_i zbar_ki`` is the loading-weighted event count and
   ``n_eff,k = 1 / sum_i w_ki^2`` the Hoeffding-effective count.

3. **Held-out risk** on never-trained prompts: exact Clopper–Pearson binomial
   bounds on failure rates, an exact McNemar test for paired student/base
   differences, per-category Bonferroni-simultaneous UCBs, and a category-level
   risk–coverage curve (deploy categories whose UCB <= tau).

All functions are CPU-only numpy/scipy; nothing here trains, tunes or touches the
teacher.  Everything is deterministic given the seed arguments.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import numpy as np
from scipy.stats import beta as _beta
from scipy.stats import binom as _binom

__all__ = [
    "clopper_pearson",
    "mcnemar_exact",
    "paired_difference_ci",
    "hoeffding_weighted_lcb",
    "hoeffding_effective_n",
    "bonferroni",
    "AtomCertificate",
    "atom_certificate",
    "lambda_trajectory_summary",
    "coverage_per_event",
    "coverage_set",
    "bootstrap_ci",
    "random_basis_coverage",
    "self_pca_coverage",
    "crossfit_coverage",
    "category_risk_table",
    "risk_coverage_curve",
]


# ----------------------------------------------------------------------------------------------
# exact binomial bounds
# ----------------------------------------------------------------------------------------------
def clopper_pearson(k: int, n: int, alpha: float = 0.05, side: str = "two") -> tuple[float, float]:
    """Exact (Clopper–Pearson) confidence bounds for a binomial proportion.

    ``side='two'`` returns the central 1-alpha interval; ``'upper'`` returns
    ``(0, UCB)`` with a one-sided 1-alpha upper bound; ``'lower'`` returns ``(LCB, 1)``.
    Guarantee: the interval contains the true p with probability >= 1 - alpha for every
    p (conservative, exact for all n; k ~ Binomial(n, p) i.i.d. trials).
    """
    if n <= 0:
        return (0.0, 1.0)
    if not 0 <= k <= n:
        raise ValueError(f"k={k} must lie in [0, n={n}]")
    if side == "two":
        a_lo = a_hi = alpha / 2.0
    elif side == "upper":
        a_lo, a_hi = None, alpha
    elif side == "lower":
        a_lo, a_hi = alpha, None
    else:
        raise ValueError("side must be 'two', 'upper' or 'lower'")
    lo = 0.0 if (a_lo is None or k == 0) else float(_beta.ppf(a_lo, k, n - k + 1))
    hi = 1.0 if (a_hi is None or k == n) else float(_beta.ppf(1.0 - a_hi, k + 1, n - k))
    return (lo, hi)


def mcnemar_exact(b: int, c: int, mid_p: bool = False) -> dict[str, float]:
    """Exact McNemar test on discordant pairs.

    ``b`` = pairs where the first system is correct and the second wrong,
    ``c`` = pairs where the first is wrong and the second correct.  Under H0 the
    discordant counts are Binomial(b + c, 1/2); the two-sided exact p-value is
    ``2 * min(P[X <= min(b,c)], P[X >= max(b,c)])`` capped at 1.  With ``mid_p`` the
    point probability is halved (less conservative, still valid asymptotically).
    """
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "n_discordant": 0, "p_value": 1.0, "statistic": 0.0}
    m = min(b, c)
    cdf = float(_binom.cdf(m, n, 0.5))
    if mid_p:
        cdf -= 0.5 * float(_binom.pmf(m, n, 0.5))
    p = min(1.0, 2.0 * cdf)
    stat = (b - c) ** 2 / n
    return {"b": b, "c": c, "n_discordant": n, "p_value": p, "statistic": float(stat)}


def paired_difference_ci(n: int, b: int, c: int, alpha: float = 0.05) -> dict[str, float]:
    """Wald CI for the paired difference of proportions ``(b - c) / n`` (first minus
    second) with variance ``(b + c - (b - c)^2 / n) / n^2``.  Asymptotic; reported
    alongside the exact McNemar p-value, not used for any guarantee claim."""
    if n <= 0:
        return {"diff": float("nan"), "lo": float("nan"), "hi": float("nan")}
    diff = (b - c) / n
    var = max(0.0, (b + c - (b - c) ** 2 / n)) / n ** 2
    from scipy.stats import norm

    z = float(norm.ppf(1 - alpha / 2))
    half = z * math.sqrt(var)
    return {"diff": diff, "lo": diff - half, "hi": diff + half, "z": z}


# ----------------------------------------------------------------------------------------------
# Hoeffding / Bonferroni for weighted means of bounded variables
# ----------------------------------------------------------------------------------------------
def bonferroni(delta: float, m: int) -> float:
    """Per-test level for ``m`` simultaneous one-sided bounds at family level ``delta``."""
    if m <= 0:
        raise ValueError("m must be positive")
    return float(delta) / float(m)


def _normalised_weights(weights: Sequence[float]) -> np.ndarray:
    w = np.abs(np.asarray(weights, dtype=np.float64))
    s = w.sum()
    if s <= 0:
        return w * np.nan
    return w / s


def hoeffding_effective_n(weights: Sequence[float]) -> float:
    """``1 / sum_i w_i^2`` for normalised weights; equals n for uniform weights."""
    w = _normalised_weights(weights)
    if np.any(np.isnan(w)):
        return 0.0
    return float(1.0 / np.sum(w * w))


def hoeffding_weighted_lcb(values: Sequence[float], weights: Sequence[float], delta: float,
                           lo: float = 0.0, hi: float = 1.0) -> dict[str, float]:
    """One-sided Hoeffding lower confidence bound for a weighted mean of independent
    variables with ``lo <= v_i <= hi`` and *fixed* normalised weights ``w_i``:

        P( sum_i w_i v_i - E[sum_i w_i v_i] <= -t ) <= exp(-2 t^2 / ((hi-lo)^2 sum_i w_i^2))

    so ``LCB = mean - (hi - lo) * sqrt( ln(1/delta) * sum_i w_i^2 / 2 )`` holds with
    probability >= 1 - delta.  Returns the mean, the half-width and the bound (clipped
    to ``lo``).  Weights that are all zero give NaN (atom unresolved).
    """
    v = np.asarray(values, dtype=np.float64)
    w = _normalised_weights(weights)
    if v.shape != w.shape:
        raise ValueError("values and weights must have the same length")
    if np.any(np.isnan(w)) or len(v) == 0:
        return {"mean": float("nan"), "half_width": float("nan"), "lcb": float("nan"),
                "n_eff": 0.0}
    if np.any(v < lo - 1e-12) or np.any(v > hi + 1e-12):
        raise ValueError("values fall outside [lo, hi]; Hoeffding bound invalid")
    mean = float(np.sum(w * v))
    s2 = float(np.sum(w * w))
    half = (hi - lo) * math.sqrt(max(0.0, math.log(1.0 / delta)) * s2 / 2.0)
    return {"mean": mean, "half_width": half, "lcb": max(lo, mean - half),
            "n_eff": float(1.0 / s2)}


@dataclass
class AtomCertificate:
    k: int
    J: float
    rho: float
    r: float
    lcb_J: float
    ucb_r: float
    n_weighted: float          # n_k = sum_i zbar_ki
    n_eff: float               # 1 / sum_i w_ki^2
    n_events: int              # events with zbar_ki > 0
    half_width: float
    resolved: bool
    lambda_final: float | None = None
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        d = {"k": self.k, "J": self.J, "rho": self.rho, "r": self.r, "LCB_J": self.lcb_J,
             "UCB_r": self.ucb_r, "n_weighted": self.n_weighted, "n_eff": self.n_eff,
             "n_events": self.n_events, "half_width": self.half_width,
             "resolved": self.resolved, "lambda_final": self.lambda_final}
        d.update(self.extra)
        return d


def atom_certificate(Z: np.ndarray, values: Sequence[float], rho: Sequence[float] | float,
                     delta: float = 0.05, min_n_eff: float = 5.0,
                     lambda_final: Sequence[float] | None = None,
                     value_range: tuple[float, float] = (0.0, 1.0)) -> list[AtomCertificate]:
    """Simultaneous atom-wise certificate ``(J_k, rho_k, r_k, LCB[J_k], UCB[r_k], n_k)``.

    ``Z`` (n, K) are the per-event relevances ``zbar_ki >= 0`` (the same weights the
    trainer's ``JEstimator`` uses), ``values`` the per-event bounded quantities ``v_i``.
    Bonferroni over the K atoms: each LCB at level ``delta / K``, so all K bounds hold
    jointly with probability >= 1 - delta *under the independence assumption on v_i*.
    Atoms with ``n_eff < min_n_eff`` are marked unresolved (bound still reported).
    """
    Z = np.abs(np.asarray(Z, dtype=np.float64))
    v = np.asarray(values, dtype=np.float64)
    n, K = Z.shape
    if len(v) != n:
        raise ValueError("values length must match Z rows")
    rho_v = np.broadcast_to(np.asarray(rho, dtype=np.float64), (K,))
    d_k = bonferroni(delta, K)
    out: list[AtomCertificate] = []
    lo, hi = value_range
    for k in range(K):
        zk = Z[:, k]
        est = hoeffding_weighted_lcb(v, zk, d_k, lo, hi)
        J = est["mean"]
        r = float(max(rho_v[k] - J, 0.0)) if not math.isnan(J) else float("nan")
        ucb_r = float(max(rho_v[k] - est["lcb"], 0.0)) if not math.isnan(J) else float("nan")
        out.append(AtomCertificate(
            k=k, J=J, rho=float(rho_v[k]), r=r, lcb_J=est["lcb"], ucb_r=ucb_r,
            n_weighted=float(zk.sum()), n_eff=est["n_eff"], n_events=int((zk > 0).sum()),
            half_width=est["half_width"],
            resolved=bool(est["n_eff"] >= min_n_eff and not math.isnan(J)),
            lambda_final=None if lambda_final is None else float(lambda_final[k]),
        ))
    return out


def lambda_trajectory_summary(lams: np.ndarray, cap: float | np.ndarray | None = None) -> list[dict]:
    """Per-atom summary of a (T, K) multiplier trajectory."""
    L = np.asarray(lams, dtype=np.float64)
    T, K = L.shape
    capv = None if cap is None else np.broadcast_to(np.asarray(cap, dtype=np.float64), (K,))
    out = []
    for k in range(K):
        col = L[:, k]
        rec = {"k": k, "init": float(col[0]), "final": float(col[-1]), "max": float(col.max()),
               "min": float(col.min()), "mean": float(col.mean()),
               "n_increase": int((np.diff(col) > 1e-12).sum()),
               "n_decrease": int((np.diff(col) < -1e-12).sum())}
        if capv is not None:
            at_cap = col >= capv[k] - 1e-9
            rec["frac_steps_at_cap"] = float(at_cap.mean())
            first = np.argmax(at_cap) if at_cap.any() else -1
            rec["first_step_at_cap"] = int(first)
        out.append(rec)
    return out


# ----------------------------------------------------------------------------------------------
# representation coverage
# ----------------------------------------------------------------------------------------------
def _orthonormal_basis(U: np.ndarray) -> np.ndarray:
    """Orthonormal basis of span(U) (QR); makes ``||Q Q^T psi||^2`` the exact
    least-squares projection energy for any (possibly non-orthonormal) dictionary."""
    Q, _ = np.linalg.qr(np.asarray(U, dtype=np.float64))
    return Q


def coverage_per_event(U: np.ndarray, Psi: np.ndarray) -> np.ndarray:
    """``c_i = ||P_U psi_i||^2 / ||psi_i||^2`` in [0, 1] for each row of ``Psi``."""
    Q = _orthonormal_basis(U)
    Psi = np.asarray(Psi, dtype=np.float64)
    if Psi.ndim == 1:
        Psi = Psi[None, :]
    proj = Psi @ Q
    num = np.sum(proj * proj, axis=1)
    den = np.sum(Psi * Psi, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        c = np.where(den > 0, num / den, np.nan)
    return np.clip(c, 0.0, 1.0)


def coverage_set(U: np.ndarray, Psi: np.ndarray) -> float:
    """Spec §6.1: ``1 - min_Z ||Psi - U Z||_F^2 / ||Psi||_F^2`` (energy-weighted over rows)."""
    Q = _orthonormal_basis(U)
    Psi = np.asarray(Psi, dtype=np.float64)
    den = float(np.sum(Psi * Psi))
    if den <= 0:
        return float("nan")
    proj = Psi @ Q
    return float(min(1.0, max(0.0, np.sum(proj * proj) / den)))


def bootstrap_ci(x: Sequence[float], n_boot: int = 2000, alpha: float = 0.05, seed: int = 0,
                 stat=np.mean) -> dict[str, float]:
    """Percentile bootstrap CI of ``stat`` over i.i.d. resamples of ``x`` (rows = events)."""
    x = np.asarray(x, dtype=np.float64)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return {"n": 0, "point": float("nan"), "lo": float("nan"), "hi": float("nan")}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    boots = np.array([stat(x[i]) for i in idx])
    return {"n": int(len(x)), "point": float(stat(x)),
            "lo": float(np.quantile(boots, alpha / 2)), "hi": float(np.quantile(boots, 1 - alpha / 2)),
            "n_boot": int(n_boot), "alpha": float(alpha)}


def random_basis_coverage(Psi: np.ndarray, K: int, n_rep: int = 20, seed: int = 0) -> dict[str, float]:
    """Reference: mean per-event coverage under ``n_rep`` random orthonormal K-bases
    (expected ``K / d`` for isotropic directions)."""
    Psi = np.asarray(Psi, dtype=np.float64)
    d = Psi.shape[1]
    rng = np.random.default_rng(seed)
    means, sets = [], []
    for _ in range(n_rep):
        U = rng.standard_normal((d, K))
        means.append(float(np.nanmean(coverage_per_event(U, Psi))))
        sets.append(coverage_set(U, Psi))
    return {"K_over_d": K / d, "per_event_mean": float(np.mean(means)),
            "per_event_mean_sd": float(np.std(means)), "set": float(np.mean(sets)),
            "n_rep": n_rep}


def self_pca_coverage(Psi: np.ndarray, K: int) -> dict[str, float]:
    """Reference (optimistic, in-sample): uncentred PCA-K fitted on ``Psi`` itself."""
    Psi = np.asarray(Psi, dtype=np.float64)
    _, s, Vt = np.linalg.svd(Psi, full_matrices=False)
    K = min(K, Vt.shape[0])
    U = Vt[:K].T
    c = coverage_per_event(U, Psi)
    return {"per_event_mean": float(np.nanmean(c)), "set": coverage_set(U, Psi), "K": int(K)}


def crossfit_coverage(Psi: np.ndarray, K: int, groups: Sequence | None = None, n_folds: int = 5,
                      seed: int = 0) -> dict:
    """Group-aware K-fold: fit uncentred PCA-K on the training folds, score the held-out
    fold; every event is scored exactly once by a basis that never saw its group.
    Returns the pooled per-event coverages (for a bootstrap CI) and fold set-coverages."""
    Psi = np.asarray(Psi, dtype=np.float64)
    n = Psi.shape[0]
    g = np.arange(n) if groups is None else np.asarray(groups)
    uniq = np.unique(g)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(uniq))
    fold_of_group = {uniq[perm[i]]: i % n_folds for i in range(len(uniq))}
    fold = np.array([fold_of_group[x] for x in g])
    per_event = np.full(n, np.nan)
    fold_sets = []
    for f in range(n_folds):
        tr, te = fold != f, fold == f
        if te.sum() == 0 or tr.sum() < K:
            continue
        _, _, Vt = np.linalg.svd(Psi[tr], full_matrices=False)
        U = Vt[:K].T
        per_event[te] = coverage_per_event(U, Psi[te])
        fold_sets.append(coverage_set(U, Psi[te]))
    return {"per_event": per_event, "fold_set": fold_sets, "n_folds": n_folds,
            "n_groups": int(len(uniq)), "K": int(K)}


# ----------------------------------------------------------------------------------------------
# category risk and selective deployment
# ----------------------------------------------------------------------------------------------
def category_risk_table(per_category: Mapping[str, tuple[int, int]], alpha: float = 0.05,
                        simultaneous: bool = True) -> dict[str, dict]:
    """``per_category[c] = (n_fail, n)`` -> per-category failure rate with one-sided
    Clopper–Pearson UCB.  With ``simultaneous=True`` the per-category level is
    ``alpha / C`` (Bonferroni) so all UCBs hold jointly with probability >= 1 - alpha."""
    C = len(per_category)
    a = alpha / C if (simultaneous and C > 0) else alpha
    out = {}
    for c, (f, n) in per_category.items():
        _, ucb = clopper_pearson(f, n, a, side="upper")
        lo2, hi2 = clopper_pearson(f, n, alpha, side="two")
        out[c] = {"n": int(n), "n_fail": int(f), "risk": (f / n if n else float("nan")),
                  "UCB": ucb, "ci95_lo": lo2, "ci95_hi": hi2, "level_per_category": a}
    return out


def risk_coverage_curve(table: Mapping[str, dict], taus: Iterable[float],
                        realised: Mapping[str, tuple[int, int]] | None = None) -> list[dict]:
    """For each threshold ``tau`` deploy the categories whose UCB <= tau.  Reports the
    item-level coverage, the certified risk of the deployed pool
    ``sum_c n_c UCB_c / sum_c n_c`` (<= tau; valid jointly with the table's guarantee),
    the in-sample empirical risk of the deployed pool and, if ``realised`` is given
    (``{category: (n_fail, n)}`` on an untouched set), the realised failure rate on the
    deployed categories of that set with its own exact two-sided CI."""
    n_total = sum(v["n"] for v in table.values())
    rows = []
    for tau in taus:
        dep = [c for c, v in table.items() if v["UCB"] <= tau]
        n_dep = sum(table[c]["n"] for c in dep)
        f_dep = sum(table[c]["n_fail"] for c in dep)
        cert = (sum(table[c]["n"] * table[c]["UCB"] for c in dep) / n_dep) if n_dep else float("nan")
        rec = {"tau": float(tau), "deployed": sorted(dep), "n_deployed": int(n_dep),
               "coverage": (n_dep / n_total if n_total else float("nan")),
               "certified_risk": cert, "empirical_risk_in_sample": (f_dep / n_dep if n_dep else float("nan")),
               "max_UCB_deployed": (max(table[c]["UCB"] for c in dep) if dep else float("nan"))}
        if realised is not None:
            rn = sum(realised[c][1] for c in dep if c in realised)
            rf = sum(realised[c][0] for c in dep if c in realised)
            lo, hi = clopper_pearson(rf, rn) if rn else (float("nan"), float("nan"))
            rec.update({"realised_n": int(rn), "realised_fail": int(rf),
                        "realised_risk": (rf / rn if rn else float("nan")),
                        "realised_ci95": [lo, hi],
                        "realised_within_nominal": (bool(rf / rn <= tau) if rn else None),
                        "realised_coverage_of_calibration": (rn / sum(v[1] for v in realised.values())
                                                             if realised else float("nan"))})
        rows.append(rec)
    return rows
