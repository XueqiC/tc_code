"""CPU unit tests for the Block IV certification primitives (src/bfas/certify.py)."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bfas import certify  # noqa: E402


# ---------------------------------------------------------------- Clopper–Pearson
def test_clopper_pearson_known_values():
    # zero failures out of 60: one-sided 95% upper = 1 - 0.05^(1/60)
    _, ucb = certify.clopper_pearson(0, 60, 0.05, side="upper")
    assert ucb == pytest.approx(1 - 0.05 ** (1 / 60), rel=1e-9)
    # two-sided 95% for k=0: upper = 1 - 0.025^(1/n)
    lo, hi = certify.clopper_pearson(0, 60, 0.05, side="two")
    assert lo == 0.0 and hi == pytest.approx(1 - 0.025 ** (1 / 60), rel=1e-9)
    # k = n: lower = 0.025^(1/n), upper = 1
    lo, hi = certify.clopper_pearson(10, 10, 0.05)
    assert hi == 1.0 and lo == pytest.approx(0.025 ** (1 / 10), rel=1e-9)
    # textbook: 5/20 -> (0.0866, 0.4910)
    lo, hi = certify.clopper_pearson(5, 20, 0.05)
    assert lo == pytest.approx(0.0866, abs=2e-4) and hi == pytest.approx(0.4910, abs=2e-4)
    assert certify.clopper_pearson(0, 0) == (0.0, 1.0)


def test_clopper_pearson_coverage_monte_carlo():
    rng = np.random.default_rng(0)
    for p in (0.05, 0.3, 0.7):
        n, reps = 40, 3000
        ks = rng.binomial(n, p, size=reps)
        cover = [certify.clopper_pearson(int(k), n, 0.10)[0] <= p <= certify.clopper_pearson(int(k), n, 0.10)[1]
                 for k in ks]
        assert np.mean(cover) >= 0.90 - 0.02          # conservative interval
        upper_ok = [p <= certify.clopper_pearson(int(k), n, 0.10, side="upper")[1] for k in ks]
        assert np.mean(upper_ok) >= 0.90 - 0.02


# ---------------------------------------------------------------- McNemar
def test_mcnemar_exact():
    r = certify.mcnemar_exact(10, 10)
    assert r["p_value"] == pytest.approx(1.0)
    r = certify.mcnemar_exact(10, 0)
    assert r["p_value"] == pytest.approx(2 * 0.5 ** 10, rel=1e-9)
    assert certify.mcnemar_exact(0, 0)["p_value"] == 1.0
    # symmetric in (b, c)
    assert certify.mcnemar_exact(3, 12)["p_value"] == certify.mcnemar_exact(12, 3)["p_value"]
    # mid-p is smaller than the exact p
    assert certify.mcnemar_exact(3, 12, mid_p=True)["p_value"] < certify.mcnemar_exact(3, 12)["p_value"]


def test_mcnemar_type1_error_monte_carlo():
    rng = np.random.default_rng(1)
    rejections = 0
    reps = 4000
    for _ in range(reps):
        n = 30
        b = rng.binomial(n, 0.5)
        rejections += certify.mcnemar_exact(int(b), int(n - b))["p_value"] < 0.05
    assert rejections / reps <= 0.05 + 0.01


def test_paired_difference_ci():
    r = certify.paired_difference_ci(100, 20, 10)
    assert r["diff"] == pytest.approx(0.10)
    assert r["lo"] < 0.10 < r["hi"]
    r0 = certify.paired_difference_ci(100, 15, 15)
    assert r0["diff"] == 0.0 and r0["lo"] < 0 < r0["hi"]


# ---------------------------------------------------------------- Hoeffding / Bonferroni
def test_hoeffding_uniform_weights_matches_classic():
    v = np.linspace(0, 1, 50)
    w = np.ones(50)
    r = certify.hoeffding_weighted_lcb(v, w, 0.05)
    assert r["n_eff"] == pytest.approx(50)
    assert r["half_width"] == pytest.approx(math.sqrt(math.log(20) / (2 * 50)))
    assert r["lcb"] == pytest.approx(v.mean() - r["half_width"])


def test_hoeffding_range_scaling_and_validation():
    v = np.array([0.0, 2.0, 4.0])
    r = certify.hoeffding_weighted_lcb(v, [1, 1, 1], 0.1, lo=0.0, hi=4.0)
    r1 = certify.hoeffding_weighted_lcb(v / 4, [1, 1, 1], 0.1)
    assert r["half_width"] == pytest.approx(4 * r1["half_width"])
    with pytest.raises(ValueError):
        certify.hoeffding_weighted_lcb([1.5], [1.0], 0.1)
    z = certify.hoeffding_weighted_lcb([0.5, 0.5], [0.0, 0.0], 0.1)
    assert math.isnan(z["lcb"]) and z["n_eff"] == 0.0


def test_hoeffding_weighted_bound_holds_monte_carlo():
    """Weighted mean of independent Bernoulli(p_i): the LCB must undershoot the true
    weighted mean at least (1 - delta) of the time."""
    rng = np.random.default_rng(2)
    n, delta = 60, 0.1
    w = rng.exponential(size=n)
    p = rng.uniform(0.2, 0.9, size=n)
    truth = float(np.sum(w / w.sum() * p))
    ok = 0
    reps = 3000
    for _ in range(reps):
        v = rng.binomial(1, p)
        ok += certify.hoeffding_weighted_lcb(v, w, delta)["lcb"] <= truth
    assert ok / reps >= 1 - delta


def test_bonferroni_simultaneous_atoms():
    """K atoms with shared events: all K LCBs hold jointly >= 1 - delta of the time."""
    rng = np.random.default_rng(3)
    n, K, delta = 80, 8, 0.1
    Z = rng.uniform(size=(n, K)) * (rng.uniform(size=(n, K)) < 0.5)
    p = rng.uniform(0.3, 0.9, size=n)
    W = Z / Z.sum(axis=0, keepdims=True)
    truth = W.T @ p
    joint_ok = 0
    reps = 1500
    for _ in range(reps):
        v = rng.binomial(1, p)
        cert = certify.atom_certificate(Z, v, rho=1.0, delta=delta, min_n_eff=1.0)
        joint_ok += all(c.lcb_J <= truth[c.k] for c in cert)
    assert joint_ok / reps >= 1 - delta
    assert certify.bonferroni(0.05, 10) == pytest.approx(0.005)


def test_atom_certificate_fields():
    Z = np.array([[1.0, 0.0], [1.0, 0.0], [0.5, 0.0]])
    v = [1.0, 0.0, 1.0]
    cert = certify.atom_certificate(Z, v, rho=[0.9, 0.9], delta=0.05, min_n_eff=2.0)
    a, b = cert
    assert a.J == pytest.approx((1 + 0 + 0.5) / 2.5)
    assert a.n_weighted == pytest.approx(2.5) and a.n_events == 3
    assert a.r == pytest.approx(max(0.9 - a.J, 0.0))
    assert a.ucb_r == pytest.approx(max(0.9 - a.lcb_J, 0.0)) and a.ucb_r >= a.r
    assert math.isnan(b.J) and not b.resolved and b.n_events == 0
    d = a.as_dict()
    assert {"k", "J", "rho", "r", "LCB_J", "UCB_r", "n_weighted", "n_eff", "resolved"} <= set(d)


def test_lambda_trajectory_summary():
    lam = np.array([[0.5, 0.5], [1.0, 0.5], [5.0, 0.2], [5.0, 0.0]])
    s = certify.lambda_trajectory_summary(lam, cap=5.0)
    assert s[0]["final"] == 5.0 and s[0]["first_step_at_cap"] == 2 and s[0]["frac_steps_at_cap"] == 0.5
    assert s[1]["n_decrease"] == 2 and s[1]["n_increase"] == 0 and s[1]["frac_steps_at_cap"] == 0.0


# ---------------------------------------------------------------- coverage
def test_coverage_identities():
    rng = np.random.default_rng(4)
    d, K, n = 32, 4, 50
    U, _ = np.linalg.qr(rng.standard_normal((d, K)))
    Psi_in = rng.standard_normal((n, K)) @ U.T             # lies in span(U)
    assert np.allclose(certify.coverage_per_event(U, Psi_in), 1.0)
    assert certify.coverage_set(U, Psi_in) == pytest.approx(1.0)
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    Psi_out = rng.standard_normal((n, d - K)) @ Q[:, K:].T
    Uc = Q[:, :K]
    assert np.allclose(certify.coverage_per_event(Uc, Psi_out), 0.0, atol=1e-10)
    # non-orthonormal dictionary spanning the same space gives the same coverage
    M = U @ rng.standard_normal((K, K))
    Psi = rng.standard_normal((n, d))
    assert np.allclose(certify.coverage_per_event(M, Psi), certify.coverage_per_event(U, Psi))
    # set coverage equals the energy-weighted mean of per-event coverages
    c = certify.coverage_per_event(U, Psi)
    e = np.sum(Psi * Psi, axis=1)
    assert certify.coverage_set(U, Psi) == pytest.approx(float(np.sum(c * e) / e.sum()))


def test_random_basis_reference_and_bootstrap():
    rng = np.random.default_rng(5)
    d, K, n = 64, 8, 400
    Psi = rng.standard_normal((n, d))
    ref = certify.random_basis_coverage(Psi, K, n_rep=10, seed=0)
    assert ref["per_event_mean"] == pytest.approx(K / d, abs=0.02)
    ci = certify.bootstrap_ci(certify.coverage_per_event(rng.standard_normal((d, K)), Psi), n_boot=300)
    assert ci["lo"] <= ci["point"] <= ci["hi"]
    self_fit = certify.self_pca_coverage(Psi, K)
    assert self_fit["set"] >= ref["set"]


def test_crossfit_coverage_group_aware():
    rng = np.random.default_rng(6)
    d, K = 40, 3
    B, _ = np.linalg.qr(rng.standard_normal((d, K)))
    n = 90
    Psi = rng.standard_normal((n, K)) @ B.T + 0.05 * rng.standard_normal((n, d))
    groups = np.arange(n) // 3
    cf = certify.crossfit_coverage(Psi, K, groups=groups, n_folds=5, seed=0)
    assert np.isfinite(cf["per_event"]).all() and cf["n_groups"] == 30
    assert np.nanmean(cf["per_event"]) > 0.9       # planted 3-d structure is recovered out-of-fold
    # a group is never split across folds: scoring with n_folds == n_groups still works
    cf2 = certify.crossfit_coverage(Psi, K, groups=groups, n_folds=30, seed=1)
    assert np.isfinite(cf2["per_event"]).all()


# ---------------------------------------------------------------- risk / coverage
def test_category_table_and_risk_coverage_curve():
    per = {"a": (0, 100), "b": (10, 100), "c": (50, 100)}
    tab = certify.category_risk_table(per, alpha=0.05, simultaneous=True)
    assert tab["a"]["level_per_category"] == pytest.approx(0.05 / 3)
    assert tab["a"]["UCB"] == pytest.approx(1 - (0.05 / 3) ** (1 / 100))
    assert tab["a"]["UCB"] < tab["b"]["UCB"] < tab["c"]["UCB"]
    curve = certify.risk_coverage_curve(tab, [0.05, 0.2, 0.7],
                                        realised={"a": (1, 20), "b": (3, 20), "c": (9, 20)})
    assert curve[0]["deployed"] == ["a"] and curve[0]["coverage"] == pytest.approx(1 / 3)
    assert curve[1]["deployed"] == ["a", "b"] and curve[2]["deployed"] == ["a", "b", "c"]
    for row in curve:
        if row["n_deployed"]:
            assert row["certified_risk"] <= row["tau"] + 1e-12
            assert row["empirical_risk_in_sample"] <= row["certified_risk"] + 1e-12
            assert row["realised_ci95"][0] <= row["realised_risk"] <= row["realised_ci95"][1]
    # coverage is monotone in tau
    assert [r["coverage"] for r in curve] == sorted(r["coverage"] for r in curve)
