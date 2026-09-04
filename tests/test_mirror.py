"""Numerical checks for the CRCD mirror target (spec §5.3 limiting behaviours)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bfas import mirror  # noqa: E402


def _event(score_t=-1.2, score_s=-0.4, q_t=1.0, q_s=1 / 6, extra=()):
    cands = [
        mirror.Candidate("T", "teacher", q_t, logp0=score_t * 10, length=10),
        mirror.Candidate("S", "student", q_s, logp0=score_s * 10, length=10),
    ]
    for n, (sc, qu) in enumerate(extra):
        cands.append(mirror.Candidate(f"X{n}", f"x{n}", qu, logp0=sc * 10, length=10))
    return mirror.Event("ev", "prompt", cands)


def test_log_odds_identity():
    ev = _event()
    s0, u = ev.scores0(), ev.utilities()
    for pressure, eta in [(0.0, 1.0), (0.7, 0.25), (3.0, 0.1), (12.0, 2.0)]:
        q = mirror.mirror_target(s0, u, pressure, eta)
        lhs = np.log(q[0] / q[1])
        rhs = mirror.log_odds(s0, u, pressure, eta, 0, 1)
        assert lhs == pytest.approx(rhs, abs=1e-10)
        assert rhs == pytest.approx((s0[0] - s0[1]) + pressure * (u[0] - u[1]) / eta)


def test_large_lambda_concentrates_on_higher_utility():
    ev = _event(score_t=-3.0, score_s=-0.1)  # base strongly prefers y^S
    s0, u = ev.scores0(), ev.utilities()
    q = mirror.mirror_target(s0, u, pressure=50.0, eta=0.25)
    assert q[0] > 0.999 and mirror.target_stats(q, ev)["entropy"] < 1e-2
    q_mid = mirror.mirror_target(s0, u, pressure=1.0, eta=0.25)
    assert 0.0 < q_mid[0] < q[0]


def test_zero_lambda_recovers_base_restricted_to_candidates():
    ev = _event(extra=[(-0.8, 0.5)])
    s0, u = ev.scores0(), ev.utilities()
    q = mirror.mirror_target(s0, u, pressure=0.0, eta=0.25)
    ref = np.exp(s0) / np.exp(s0).sum()
    np.testing.assert_allclose(q, ref, atol=1e-12)
    np.testing.assert_allclose(q, mirror.restricted_policy(s0), atol=1e-12)


def test_low_utility_teacher_gets_less_mass_than_student():
    # equal base scores, teacher branch worse than student branch: no negative
    # example rule, the utility term alone moves mass toward y^S.
    ev = _event(score_t=-1.0, score_s=-1.0, q_t=0.2, q_s=0.8)
    s0, u = ev.scores0(), ev.utilities()
    for pressure in (0.5, 2.0, 10.0):
        q = mirror.mirror_target(s0, u, pressure, eta=0.25)
        assert q[0] < q[1]
    q_hi = mirror.mirror_target(s0, u, 10.0, eta=0.25)
    assert q_hi[1] > 0.999
    assert ev.best_index() == 1


def test_nearly_satisfied_atom_small_log_odds_change():
    ev = _event()
    s0, u = ev.scores0(), ev.utilities()
    base = mirror.log_odds(s0, u, 0.0, 0.25, 0, 1)
    small = mirror.log_odds(s0, u, 0.05, 0.25, 0, 1)
    large = mirror.log_odds(s0, u, 5.0, 0.25, 0, 1)
    assert abs(small - base) < 0.2 < abs(large - base)


def test_capability_pressure_normalisation_and_global_mirror():
    rng = np.random.default_rng(0)
    Z = rng.random((5, 3))
    W = mirror.normalized_loadings(Z, eps=0.0)
    np.testing.assert_allclose(W.sum(axis=0), 1.0)
    lam = np.array([1.0, 0.0, 2.0])
    Lam = mirror.capability_pressure(W, lam)
    np.testing.assert_allclose(Lam, W[:, 0] + 2 * W[:, 2])
    # K=1, zbar=1 -> every event carries lambda/n (global mirror).
    Wg = mirror.normalized_loadings(np.ones((4, 1)), eps=0.0)
    np.testing.assert_allclose(mirror.capability_pressure(Wg, np.array([2.0])), 0.5)
    # sign of loadings is irrelevant
    np.testing.assert_allclose(mirror.normalized_loadings(-Z), W)


def test_dual_update_satisfied_atom_goes_to_zero_violated_rises_and_caps():
    dual = mirror.DualState(lam=[0.5, 0.5], rho=[0.3, 0.9], eps=0.0,
                            kappa=2.0, d=[1.0, 1.0], eta_dual=1.0)
    J = np.array([0.6, 0.2])  # atom 0 satisfied, atom 1 violated
    for _ in range(20):
        rec = dual.update(J)
    assert dual.lam[0] == 0.0
    assert dual.lam[1] == pytest.approx(2.0)  # clamped at kappa d
    np.testing.assert_allclose(rec["r"], [0.0, 0.7])
    np.testing.assert_allclose(rec["xi"], [0.0, 0.7])  # slack only where clamp binds
    assert list(rec["satisfied"]) == [True, False]
    # monotone rise while unclamped
    dual2 = mirror.DualState(lam=[0.0], rho=[0.9], eps=0.0, kappa=100.0, eta_dual=0.5)
    prev = 0.0
    for _ in range(5):
        lam = dual2.update(np.array([0.2]))["lambda_after"][0]
        assert lam > prev
        prev = lam
    assert prev == pytest.approx(5 * 0.5 * 0.7)


def test_J_estimator_and_event_values():
    ev = _event(extra=[(-0.8, 0.5)])
    p = np.array([0.2, 0.5, 0.3])
    assert mirror.event_J_value(p, ev, "best_mass") == pytest.approx(0.2)
    assert mirror.event_J_value(p, ev, "expected_utility") == pytest.approx(
        0.2 * 1.0 + 0.5 / 6 + 0.3 * 0.5)
    Z = np.array([[1.0, 0.0], [1.0, 3.0], [0.0, 1.0]])
    J = mirror.atom_J_from_events(Z, [0.2, 0.8, 0.5])
    np.testing.assert_allclose(J, [0.5, (3 * 0.8 + 0.5) / 4])
    est = mirror.JEstimator(2)
    assert np.isnan(est.value()).all()
    np.testing.assert_allclose(est.value(fallback=np.array([0.1, 0.2])), [0.1, 0.2])


def test_cmd_loss_gradient_is_p_minus_q_and_zero_at_target():
    q = np.array([0.7, 0.2, 0.1])
    scores = torch.tensor(np.log(q), dtype=torch.float32, requires_grad=True)
    loss = mirror.cmd_loss(q, scores)
    assert loss.item() == pytest.approx(0.0, abs=1e-9)
    scores2 = torch.tensor([0.0, 0.0, 0.0], requires_grad=True)
    loss2 = mirror.cmd_loss(q, scores2)
    loss2.backward()
    p = np.full(3, 1 / 3)
    np.testing.assert_allclose(scores2.grad.numpy(), p - q, atol=1e-6)
    assert loss2.item() == pytest.approx(float((q * np.log(q / p)).sum()))


def test_token_kl_zero_for_identical_logits_and_positive_otherwise():
    torch.manual_seed(0)
    lt = torch.randn(1, 6, 11)
    mask = torch.tensor([[False, False, True, True, True, True]])
    assert float(mirror.token_kl_theta_base(lt, lt.clone(), mask)) == pytest.approx(0.0, abs=1e-7)
    assert float(mirror.token_kl_theta_base(lt, lt + torch.randn_like(lt), mask)) > 0


def test_displacement_penalty_projector_and_zero_at_init():
    torch.manual_seed(0)
    a = torch.nn.Parameter(torch.zeros(8, 4))
    b = torch.nn.Parameter(torch.zeros(6, 4))
    # all penalties snapshot theta_0 now (at "init"), before any displacement
    pen = mirror.DisplacementPenalty([("b", b), ("a", a)], fisher=1.0, U=None, dim=16, seed=0)
    pen_z = mirror.DisplacementPenalty([("a", a), ("b", b)], fisher=1.0,
                                       U=np.zeros((16, 2)), dim=16, seed=0)
    pen_f = mirror.DisplacementPenalty([("a", a), ("b", b)],
                                       fisher={"a": torch.full((8, 4), 4.0),
                                               "b": torch.ones(6, 4)},
                                       dim=16, seed=0)
    assert pen().item() == 0.0
    with torch.no_grad():
        a.add_(torch.randn_like(a))
    full = pen().item()
    assert full > 0
    # sketch is norm preserving in expectation: same order of magnitude
    assert 0.1 * float(a.pow(2).sum()) < full < 10 * float(a.pow(2).sum())
    # U = zeros -> full displacement penalty (same as U=None)
    assert pen_z().item() == pytest.approx(full)
    # diagonal Fisher scales the whitened displacement
    assert pen_f().item() == pytest.approx(4 * full, rel=1e-5)
    # U spanning the current sketch direction removes the penalty entirely
    sk = pen.sketch().detach().numpy()[:, None]
    pen.set_subspace(sk)
    assert pen().item() == pytest.approx(0.0, abs=1e-6 * full)
    pen.set_subspace(None)
    assert pen().item() == pytest.approx(full)
    # differentiable
    pen().backward()
    assert a.grad is not None and float(a.grad.abs().sum()) > 0


def test_candidates_from_row_utility_modes():
    row = {"task_id": "t", "turn_index": 0, "teacher": "oracle_gt", "prompt": "p",
           "response": "gt", "_rejected": "bad", "_event_u_plus": 0.83,
           "_event_u_minus": 0.17, "_traj": "t#ev1"}
    ev = mirror.candidates_from_row(row, "auto")
    assert ev.event_id == "t#ev1" and ev.meta["utility_mode"] == "gt_teacher"
    assert ev.utilities().tolist() == [1.0, 0.17]
    ev2 = mirror.candidates_from_row(row, "stored")
    assert ev2.utilities().tolist() == [0.83, 0.17]
    row["turn_index"] = 2
    assert mirror.candidates_from_row(row, "auto").meta["utility_mode"] == "stored"
