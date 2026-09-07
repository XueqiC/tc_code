"""CPU mathematical checks for feasible recipes and finite-dose controllers."""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.bfas.behavior import recipe  # noqa: E402
from src.bfas.behavior.recipe import (  # noqa: E402
    FeasibleSet, compare_global_kl, distill_recipe, find_recipe, finite_step_bound,
    iterate_atoms, positive_direction_exists, trust_region_qp,
)


def toy(**kwargs):
    # Columns: demo 1 improves missing by +1 but retains by -1;
    # demo 2 has missing effect 0 and retention effect +1.
    return FeasibleSet([[1., 0.], [-1., 1.]], R=[1], M=[0], r=[1.], **kwargs)


def test_recipes_use_one_response_operator_for_their_realized_parameter_updates():
    from src.bfas.behavior.lowrank import ResponsePredictor, fit_model
    from src.bfas.behavior.whiten import DiagonalMetric, build_basis

    H = DiagonalMetric(np.array([2., 3., 4.]), "layout", .1, "v1", "fisher")
    basis = build_basis(list(np.eye(3)[:2]), H)
    C = np.column_stack([basis.decode(x) for x in np.eye(2)])
    codes = np.array([[.5, .5], [1., 0.], [0., 1.]])
    D = C @ codes.T
    model = fit_model(codes, np.array([[1., 0.], [-1., 1.]]) @ codes.T, rank=2, ridge=0.)
    predictor = ResponsePredictor(model, basis)
    B_hat = predictor.predict_from_deltas(D.T)
    np.testing.assert_allclose(B_hat, model.R @ C.T @ np.diag(H.diagonal) @ D, atol=1e-12)
    fs = FeasibleSet(B_hat, [1], [0], [1.], X=codes)
    found, distilled = find_recipe(fs), distill_recipe(fs, lam=.1)
    assert found["success"] and distilled["success"]
    for result, key in ((found, "a"), (distilled, "w")):
        np.testing.assert_allclose(result["b"], predictor.predict_from_deltas(D @ result[key]), atol=1e-12)
        np.testing.assert_allclose(result["b"], predictor.predict_recipe(result[key], codes.T), atol=1e-12)
    np.testing.assert_allclose(predictor.predict_recipe([1., 0., 0.], codes.T),
                               predictor.predict_recipe([0., .5, .5], codes.T), atol=1e-12)


@pytest.mark.parametrize("E", [0., np.zeros(2), np.zeros((2, 1)), np.zeros((2, 2))])
def test_zero_error_recovers_original_recipe_and_distillation(E):
    fs = toy(X=np.eye(2))
    for solve in (find_recipe, distill_recipe):
        original = solve(fs)
        zero = solve(fs, E=E)
        direct = solve(fs.B_hat, fs.R, fs.M, fs.r, X=fs.X, E=E)
        for result in (zero, direct):
            assert result["success"] and result["objective"] == pytest.approx(original["objective"])
            np.testing.assert_allclose(result["b"], original["b"], atol=1e-12)
            np.testing.assert_allclose(result.get("a", result.get("w")),
                                       original.get("a", original.get("w")), atol=1e-12)
            assert result["feasibility_report"]["gain_loss"] == 0.
            assert result["feasibility_report"]["retention_gain_loss"] == 0.
    result = positive_direction_exists(fs.B_M, fs.B_R, fs.r, E=E)
    assert result["exists"] and result["gain"] == pytest.approx(.5)


def test_conservative_sets_and_max_gains_tighten_monotonically():
    grid = np.column_stack([np.linspace(0., 1., 101), np.linspace(1., 0., 101)])
    previous_membership = np.ones(len(grid), dtype=bool)
    gains = []
    for e in (0., .05, .2, .6, 1.1):
        fs = toy(E=e)
        membership = np.array([fs.contains(a) for a in grid])
        assert np.all(~membership | previous_membership)
        previous_membership = membership
        result = find_recipe(fs)
        report = result["feasibility_report"]
        gains.append(report["conservative"]["value"])
        if result["success"]:
            np.testing.assert_allclose(fs.conservative_B_R @ result["a"],
                                       (fs.B_R - e) @ result["a"])
            assert np.min(fs.conservative_B_R @ result["a"]) >= -1e-8
        else:
            assert result["status"] == "infeasible" and result["infeasible"]
    assert np.all(np.diff(gains) <= 1e-8)
    assert gains[0] == pytest.approx(.5) and gains[-1] == 0.


def test_small_error_preserves_complementarity_and_reports_gain_shrinkage():
    fs = toy(E=.1)
    values = []
    for columns in ([0], [1], [0, 1]):
        result = find_recipe(fs.B_hat[:, columns], fs.R, fs.M, fs.r, E=fs.E[:, columns])
        values.append(result["feasibility_report"]["conservative"]["value"])
    assert values == pytest.approx([0., 0., .35])
    result = find_recipe(fs)
    np.testing.assert_allclose(result["a"], [.45, .55])
    np.testing.assert_allclose(result["conservative_b"], [.35, 0.], atol=1e-12)
    report = result["feasibility_report"]
    assert report["nominal"]["max_gain"] == pytest.approx(.5)
    assert report["retention_only"]["max_gain"] == pytest.approx(.45)
    assert report["conservative"]["max_gain"] == pytest.approx(.35)
    assert report["gain_loss"] == pytest.approx(.15)
    assert report["retention_gain_loss"] == pytest.approx(.05)
    assert report["relative_gain_loss"] == pytest.approx(.3)
    assert not report["infeasible"] and not report["became_infeasible"]
    direction = positive_direction_exists(fs.B_M, fs.B_R, fs.r, E=fs.E)
    assert direction["success"] and direction["exists"] and fs.contains(direction["w"])
    assert direction["gain"] == pytest.approx(.35)
    distilled = distill_recipe(fs, eps=.8, lam=.1)
    assert distilled["success"] and distilled["simplex_feasible"]
    assert distilled["conservative_weighted_gain"] > 0
    assert distilled["effective_dose"] <= .8 + 1e-8
    assert np.min(distilled["conservative_retention_change"]) >= -1e-8
    assert distilled["w"][0] > 0 and distilled["w"][1] > 0


def test_large_error_flags_empty_simplex_while_zero_distillation_is_feasible():
    fs = toy(E=1.1)
    found = find_recipe(fs)
    assert not found["success"] and found["infeasible"] and found["a"] is None
    assert found["status"] == "infeasible"
    report = found["feasibility_report"]
    assert report["became_infeasible"] and report["infeasible"]
    assert report["conservative"]["max_gain"] is None and report["gain_loss"] == pytest.approx(.5)
    direction = positive_direction_exists(fs.B_M, fs.B_R, fs.r, E=fs.E)
    assert direction["success"] and not direction["exists"] and direction["infeasible"]
    certificate = direction["certificate"]
    assert certificate["type"] == "farkas"
    np.testing.assert_allclose(certificate["source_bounds"],
                               fs.conservative_B_R.T @ certificate["retention_multipliers"])
    assert np.all(certificate["source_bounds"] <= -1 + 1e-8)
    distilled = distill_recipe(fs)
    assert distilled["success"] and distilled["feasible"] and not distilled["simplex_feasible"]
    assert distilled["simplex_status"] == "infeasible"
    np.testing.assert_allclose(distilled["w"], 0., atol=1e-8)
    np.testing.assert_allclose(distilled["residual_gap"], fs.r)


def test_conservative_gap_and_retention_use_per_entry_errors_and_preserve_nominal_outputs():
    E = np.array([[.1, .03], [.2, .05]])
    fs = toy(E=E, X=np.eye(2))
    original_E = E.copy()
    result = distill_recipe(fs, eps=.7, lam=.2)
    assert result["success"] and result["kkt"]["verified"]
    w = result["w"]
    np.testing.assert_allclose(result["b"], fs.B_hat @ w)
    np.testing.assert_allclose(result["predicted_gain"], fs.B_M @ w)
    np.testing.assert_allclose(result["conservative_gain"], (fs.B_M - E[fs.M]) @ w)
    np.testing.assert_allclose(result["conservative_retention_change"], (fs.B_R - E[fs.R]) @ w)
    gap = np.maximum(fs.r - (fs.B_M - E[fs.M]) @ w, 0)
    np.testing.assert_allclose(result["residual_gap"], gap)
    assert result["objective"] == pytest.approx(.5 * gap @ gap + .1 * w @ fs.K @ w)
    assert result["metric_source"] == "codes"
    # r overrides and repeated atom solves must retain E without double subtraction.
    found = find_recipe(fs, r=[.5])
    assert found["success"] and fs.contains(found["a"])
    assert found["objective"] == pytest.approx(.5 * fs.conservative_B_M @ found["a"])
    sequence = iterate_atoms(fs, k=3, eta=.5)
    for i, (a, _, _) in enumerate(sequence["atoms"]):
        assert fs.contains(a)
        np.testing.assert_allclose(sequence["gap_history"][i + 1],
                                   np.maximum(sequence["gap_history"][i] - .5 * fs.conservative_B_M @ a, 0))
    np.testing.assert_array_equal(fs.E, original_E)
    np.testing.assert_array_equal(E, original_E)


def test_error_quantiles_broadcast_per_probe_and_missing_error_is_not_zero():
    from src.bfas.behavior.lowrank import residual_error_bounds

    E = residual_error_bounds([[.01, -.03, .05], [-.1, .02, .2]], quantile=.5)
    fs = toy(E=E)
    np.testing.assert_allclose(fs.E, [[.03, .03], [.1, .1]])
    np.testing.assert_allclose(toy(E=E[:, 0]).E, fs.E)
    assert find_recipe(fs)["success"] and fs.feasibility_report()["gain_loss"] > 0
    missing = residual_error_bounds([[np.nan], [.1]])
    with pytest.raises(ValueError, match="finite"):
        toy(E=missing)


@pytest.mark.parametrize("E", [-.1, np.nan, np.inf, [0., -.1], [0.], [0., 0., 0.],
                              [[.1, .2]], np.zeros((2, 3)), np.zeros((2, 2, 1))])
def test_invalid_error_bounds_are_explicit_errors(E):
    for operation in (lambda: toy(E=E), lambda: find_recipe(toy(), E=E),
                      lambda: distill_recipe(toy(), E=E),
                      lambda: positive_direction_exists(toy().B_M, toy().B_R, [1.], E=E)):
        with pytest.raises(ValueError, match="E"):
            operation()


def test_two_demonstrations_have_superadditive_value_and_complementary_recipe():
    fs = toy(X=np.array([[2., 0.], [0., 3.]]))
    values = []
    for columns in ([0], [1], [0, 1]):
        result = find_recipe(fs.B_hat[:, columns], R=[1], M=[0], r=[1.])
        # V permits the no-update action. Demo 1's normalized simplex is empty.
        values.append(result["objective"] if result["success"] else 0.)
    assert values == pytest.approx([0., 0., .5])
    result = find_recipe(fs)
    assert result["success"] and fs.contains(result["a"])
    np.testing.assert_allclose(result["a"], [.5, .5])
    np.testing.assert_allclose(result["u"], [1., 1.5])
    np.testing.assert_allclose(result["b"], [.5, 0.])
    np.testing.assert_array_equal(result["active_retention"], [1])
    assert result["support_size"] == result["sparsity_bound"] == 2
    np.testing.assert_allclose(fs.K, fs.X @ fs.X.T)


@pytest.mark.parametrize("seed", range(30))
def test_random_vertices_obey_active_independent_constraint_sparsity(seed):
    rng = np.random.default_rng(seed)
    n, q = int(rng.integers(3, 12)), int(rng.integers(0, 5))
    retention = rng.normal(size=(q, n))
    retention -= retention.mean(axis=1, keepdims=True)  # Uniform mixture is feasible.
    if q > 1:
        retention[-1] = retention[0]  # Dependent constraints must not inflate rank.
    B = np.vstack([rng.normal(size=(2, n)) + 1, retention])
    fs = FeasibleSet(B, np.arange(2, 2 + q), [0, 1], [.8, 1.2])
    result = find_recipe(fs)
    assert result["success"] and fs.contains(result["a"])
    assert result["support_size"] <= len(result["active_retention"]) + 1
    assert result["support_size"] <= result["active_rank"] + 1
    assert not result["sparsity_violation"]
    assert result["objective"] >= fs.r @ (fs.B_M @ np.full(n, 1 / n)) - 1e-8
    assert result["upper_bound"] == pytest.approx(result["objective"], abs=1e-7)
    assert np.min(result["dual_slack"]) >= -1e-7


def test_dense_degenerate_lp_is_reported_and_crossed_back_to_vertex(monkeypatch):
    real_lp = recipe._lp
    calls = []

    def dense_first(*args, **kwargs):
        result = real_lp(*args, **kwargs)
        if not calls:
            result.x = np.full(4, .25)  # An optimal nonvertex solver output.
        calls.append(result)
        return result

    monkeypatch.setattr(recipe, "_lp", dense_first)
    with pytest.warns(RuntimeWarning, match="sparsity"):
        result = find_recipe(np.ones((1, 4)), [], [0], [1.])
    assert result["success"] and result["degeneracy_detected"]
    assert result["tie_break_used"] and not result["sparsity_violation"]
    assert result["support_size"] == 1 and result["objective"] == pytest.approx(1.)


def test_greedy_atoms_follow_residual_and_stop_at_no_positive_direction():
    fs = toy(X=np.eye(2))
    original = fs.r.copy()
    result = iterate_atoms(fs, k=8, eta=1.)
    assert result["success"] and result["status"] == "gap_closed"
    assert len(result["atoms"]) == 2 and result["support_sizes"] == [2, 2]
    np.testing.assert_allclose(result["gap_history"].ravel(), [1., .5, 0.])
    np.testing.assert_array_equal(fs.r, original)
    for a, u, b in result["atoms"]:
        np.testing.assert_allclose(u, fs.X.T @ a)
        np.testing.assert_allclose(b, fs.B_hat @ a)
    assert iterate_atoms(fs, k=0)["atoms"] == []
    flat = iterate_atoms([[0., -1.]], [], [0], [1.])
    assert flat["status"] == "no_positive_direction" and flat["atoms"] == []
    impossible = iterate_atoms([[1.], [-1.]], [1], [0], [1.])
    assert not impossible["success"] and impossible["status"] == "infeasible"


def test_greedy_multigap_residual_uses_positive_part_including_new_damage():
    B = np.array([[2., 0.], [-1., 1.]])
    result = iterate_atoms(B, [], [0, 1], [2., .1], k=2, eta=.25)
    residual = np.array([2., .1])
    for i, (a, _, _) in enumerate(result["atoms"]):
        residual = np.maximum(residual - .25 * B @ a, 0)
        np.testing.assert_allclose(result["gap_history"][i + 1], residual)
    assert result["gap_history"][1, 1] > .1


def test_positive_direction_certificates_feasible_nonpositive_and_empty():
    fs = toy()
    positive = positive_direction_exists(fs.B_M, fs.B_R, fs.r)
    assert positive["success"] and positive["exists"] and positive["feasible"]
    assert fs.contains(positive["w"]) and positive["gain"] == pytest.approx(.5)
    negative = positive_direction_exists(-fs.B_M, fs.B_R, fs.r)
    assert negative["exists"] is False and negative["feasible"]
    assert negative["upper_bound"] <= 1e-8
    cert = negative["certificate"]
    c = -fs.B_M.T @ fs.r
    assert np.all(cert["retention_multipliers"] >= 0)
    assert np.all(c + fs.B_R.T @ cert["retention_multipliers"] <= cert["upper_bound"] + 1e-8)
    impossible = positive_direction_exists([[1., 2.]], [[-1., -2.]], [1.])
    assert impossible["exists"] is False and not impossible["feasible"]
    cert = impossible["certificate"]
    assert cert["type"] == "farkas" and np.all(cert["retention_multipliers"] >= 0)
    assert np.all(cert["source_bounds"] <= -1 + 1e-8)
    zero = positive_direction_exists([[0., 0.]], np.empty((0, 2)), [1.])
    assert zero["exists"] is False and zero["feasible"]
    empty = positive_direction_exists(np.empty((1, 0)), np.empty((0, 0)), [1.])
    assert empty["exists"] is False and empty["certificate"]["type"] == "empty_simplex"


def test_distillation_respects_retention_and_gram_dose_and_reduces_gap():
    fs = toy(K=np.diag([4., 1.]), X=np.eye(2))
    result = distill_recipe(fs, eps=.8, lam=.1)
    assert result["success"] and result["kkt"]["verified"]
    assert not result["fallback"] and result["metric_source"] == "provided"
    assert np.min(result["w"]) >= -1e-8
    assert np.min(result["predicted_retention_change"]) >= -1e-8
    assert result["effective_dose"] <= .8 + 1e-8
    np.testing.assert_allclose(result["w"], np.full(2, .8 / np.sqrt(5)), atol=1e-6)
    assert result["objective"] < result["initial_objective"]
    assert np.linalg.norm(result["residual_gap"]) < np.linalg.norm(fs.r)
    assert result["effective_dose"] == pytest.approx(np.sqrt(result["w"] @ fs.K @ result["w"]))
    np.testing.assert_allclose(result["predicted_gain"], fs.B_M @ result["w"])
    np.testing.assert_array_equal(result["active_retention"], [1])
    np.testing.assert_array_equal(result["active_set"], [0, 1])


@pytest.mark.parametrize("lam", [0., .1, 2.])
@pytest.mark.parametrize("E", [0., .5])
def test_distill_unconstrained_scalar_matches_closed_form(lam, E):
    result = distill_recipe([[2.]], [], [0], [1.], K=[[3.]], eps=10., lam=lam, E=E)
    assert result["success"]
    gain = 2 - E
    if lam:
        np.testing.assert_allclose(result["w"], [gain / (gain ** 2 + 3 * lam)], atol=1e-6)
    else:
        # The one-sided loss has an entire optimal plateau after closing r.
        assert result["w"][0] >= 1 / gain - 1e-7 and result["objective"] < 1e-12


def test_distill_singular_gram_zero_dose_and_empty_simplex_are_distinct():
    singular = FeasibleSet([[1., 1.]], [], [0], [1.], X=np.array([[1.], [-1.]]))
    result = distill_recipe(singular, eps=0.)
    assert result["success"] and result["kkt"]["verified"]
    assert result["w"].sum() >= 1. - 1e-7
    assert result["objective"] < 1e-12
    assert result["effective_dose"] < 1e-8
    zero = distill_recipe(toy(), eps=0.)
    assert zero["success"] and np.array_equal(zero["w"], [0., 0.])
    impossible_recipe = distill_recipe([[1.], [-1.]], [1], [0], [1.], eps=1.)
    assert impossible_recipe["success"] and impossible_recipe["feasible"]
    np.testing.assert_allclose(impossible_recipe["w"], [0.], atol=1e-8)
    no_sources = distill_recipe(np.empty((1, 0)), [], [0], [1.])
    assert no_sources["success"] and no_sources["w"].size == 0


@pytest.mark.parametrize("nonfinite", [False, True])
def test_distill_reports_solver_failure_with_feasible_fallback(monkeypatch, nonfinite):
    def failed(fun, x0, **kwargs):
        return SimpleNamespace(x=np.full_like(x0, np.nan if nonfinite else 0.), success=False,
                               message="forced solver failure", nit=0)
    monkeypatch.setattr(recipe, "minimize", failed)
    result = distill_recipe(toy())
    assert not result["success"] and result["fallback"]
    assert result["status"] == "solver_failed" and result["feasible"]
    assert len(result["attempts"]) == 2 and not result["kkt"]["verified"]
    np.testing.assert_array_equal(result["w"], [0., 0.])


def test_qp_solver_failure_returns_verified_feasible_fallback_diagnostics(monkeypatch):
    def failed(fun, x0, **kwargs):
        return SimpleNamespace(x=np.full_like(x0, np.nan), success=False, message="forced NaN", nit=0)
    monkeypatch.setattr(recipe, "minimize", failed)
    result = trust_region_qp([1.], [[1.]], np.empty((0, 1)), (1., []), 0., 0., .5, 0.)
    assert not result["success"] and result["status"] == "solver_failed"
    assert result["feasible"] and result["fallback"] and len(result["attempts"]) == 2
    assert not result["kkt"]["verified"]
    assert result["kkt"]["primal_violation"] == 0.
    np.testing.assert_array_equal(result["x"], [0.])


@pytest.mark.parametrize("eps", [1e-2, 1e-5, 1e-8])
def test_small_dose_is_checked_in_norm_units(eps):
    distilled = distill_recipe([[1.]], [], [0], [1.], eps=eps)
    shaped = trust_region_qp([1.], [[1.]], np.empty((0, 1)), (1., []), 0., 0., eps, 0.)
    for result in (distilled, shaped):
        assert result["success"] and result["effective_dose"] <= eps + 1e-9


@pytest.mark.parametrize("E", [0., np.array([[.1, .03], [.2, .05]])])
def test_distill_analytic_gradients_match_finite_differences(monkeypatch, E):
    real_minimize = recipe.minimize
    def checked(fun, x0, *, jac, constraints, **kwargs):
        point = np.array([.12, .23])
        delta = 1e-6 * np.eye(2)
        expected = np.array([(fun(point + dx) - fun(point - dx)) / 2e-6 for dx in delta])
        np.testing.assert_allclose(jac(point), expected, atol=1e-8)
        for constraint in constraints:
            expected = np.column_stack([(constraint["fun"](point + dx) - constraint["fun"](point - dx)) / 2e-6 for dx in delta])
            np.testing.assert_allclose(constraint["jac"](point), expected, atol=1e-8)
        return real_minimize(fun, x0, jac=jac, constraints=constraints, **kwargs)
    monkeypatch.setattr(recipe, "minimize", checked)
    assert distill_recipe(toy(E=E), lam=.2, eps=.6)["success"]


def test_shaping_beats_global_scaling_at_matched_damage_on_separated_modes():
    result = compare_global_kl([2., 1.], [[0., 1.]], [[-1., 0.]], 1.,
                               alpha=0., tau=.5, rho=10., perp_norm=1.)
    assert result["success"] and result["matched_damage"] and result["strictly_better"]
    np.testing.assert_allclose(result["shaped"]["x"], [.5, 1.], atol=1e-7)
    np.testing.assert_allclose(result["global_x"], [.5, .25], atol=1e-7)
    assert result["global_damage"] == pytest.approx(result["shaped"]["damage"])
    assert result["gain_improvement"] == pytest.approx(.75)
    assert result["global_perp_norm"] == pytest.approx(.25)


def test_no_damaging_component_ties_global_scaling():
    result = compare_global_kl([0., 1.], [[0., 1.]], [[-1., 0.]], 1.,
                               alpha=0., tau=.5, rho=10., perp_norm=0.)
    assert result["success"] and result["matched_damage"] and result["dominates"]
    assert not result["strictly_better"]
    np.testing.assert_allclose(result["shaped"]["x"], result["global_x"], atol=1e-7)


def test_qp_active_ball_and_kkt_multipliers_match_projection_closed_form():
    x0, gain = np.array([2., 1.]), np.array([1., 2.])
    alpha, rho, perp = .4, 1.5, .5
    shifted = x0 + alpha / 2 * gain
    radius = np.sqrt(rho ** 2 - perp ** 2)
    result = trust_region_qp(x0, gain[None, :], np.empty((0, 2)), (1., []), alpha, 0., rho, perp)
    assert result["success"] and result["kkt"]["verified"]
    np.testing.assert_allclose(result["x"], radius * shifted / np.linalg.norm(shifted), atol=1e-6)
    mu = result["multipliers"]["ball"]
    assert mu == pytest.approx(np.linalg.norm(shifted) / radius - 1, abs=1e-6)
    np.testing.assert_allclose(2 * (result["x"] - x0) - alpha * gain + 2 * mu * result["x"], 0, atol=1e-6)
    assert result["effective_dose"] == pytest.approx(rho)


def test_qp_multiple_signed_retention_modes_and_hinge_kkt():
    minus = np.array([[-1., 0.], [1., 0.], [0., 1.]])
    q = np.array([2., 3., 1.])
    result = trust_region_qp([2., 1.], [[0., 1.]], minus,
                              {"plus": [1.], "minus": q}, 0., .6, 10., 0.)
    assert result["success"] and result["kkt"]["verified"]
    np.testing.assert_allclose(result["x"], [.3, 1.], atol=1e-7)
    assert result["damage"] == pytest.approx(.6)
    mu = result["multipliers"]
    np.testing.assert_allclose(mu["hinge"] + mu["epigraph"], mu["damage"] * q, atol=1e-7)
    np.testing.assert_allclose(2 * (result["x"] - [2., 1.]) - minus.T @ mu["hinge"]
                               + 2 * mu["ball"] * result["x"], 0, atol=1e-7)


def test_qp_zero_damage_budget_infeasible_perp_and_singleton():
    zero_damage = trust_region_qp([2., 1.], [[0., 1.]], [[-1., 0.]], 1., 0., 0., 2., 0.)
    assert zero_damage["success"]
    np.testing.assert_allclose(zero_damage["x"], [0., 1.], atol=1e-7)
    impossible = trust_region_qp([1.], [[1.]], [[-1.]], 1., 0., 0., 1., 2.)
    assert not impossible["success"] and impossible["status"] == "infeasible"
    assert impossible["x"] is None
    singleton = trust_region_qp([1.], [[1.]], [[-1.]], 1., 0., 0., 1., 1.)
    assert singleton["success"] and singleton["multipliers"]["ball"] is None
    np.testing.assert_array_equal(singleton["x"], [0.])


def test_pareto_helper_reports_counterexample_instead_of_claiming_universal_dominance():
    # Both gains load on x1. Closest-point shaping shrinks x1 more aggressively
    # than the globally scaled ray; the proximity objective is not max gain.
    result = compare_global_kl([2., 1.], [[1., 0.]], [[-2., -1.]], 1.,
                               alpha=0., tau=1., rho=10., perp_norm=0.)
    assert result["success"] and result["matched_damage"]
    # Projection onto 2*x1+x2<=1 gives (.4,.2), exactly the global ray here.
    assert abs(result["gain_improvement"]) < 1e-7
    unequal = compare_global_kl([1., 2.], [[1., 0.]], [[-2., -1.]], 1.,
                                alpha=0., tau=1., rho=10., perp_norm=0.)
    assert unequal["success"] and unequal["matched_damage"]
    assert unequal["gain_improvement"] < 0 and not unequal["dominates"]


def test_global_comparison_flags_unattainable_matched_damage():
    result = compare_global_kl([0., 0.], [[1., 0.]], [[-1., 0.]], 1.,
                               alpha=1., tau=1., rho=2., perp_norm=0.)
    assert result["success"] and result["shaped"]["damage"] > .1
    assert not result["matched_damage"] and not result["dominates"]


def test_finite_step_bound_monotone_and_satisfies_quadratic_remainder():
    budgets = np.linspace(0., 2., 30)
    for gain in (-2., 0., 3.):
        bounds = finite_step_bound(gain, 2., budgets)
        assert np.all(np.diff(bounds) >= 0)
        np.testing.assert_allclose(gain * bounds - bounds ** 2, -budgets, atol=1e-12)
    assert np.all(np.diff(finite_step_bound(np.linspace(-2., 2., 30), 1., .1)) > 0)
    assert np.all(np.diff(finite_step_bound(.2, np.linspace(.1, 2., 30), .1)) < 0)
    assert finite_step_bound(-2., 0., .4) == pytest.approx(.2)
    assert np.isinf(finite_step_bound(0., 0., 0.))
    assert np.isinf(finite_step_bound(1., 0., .4))
    assert finite_step_bound(0., 2., 0.) == 0.
    assert finite_step_bound(-1e12, 1., 1.) == pytest.approx(1e-12, rel=1e-12)


@pytest.mark.parametrize("operation", [
    lambda: FeasibleSet([[np.nan]], [], [0], [1.]),
    lambda: FeasibleSet([[1.]], [], [0], [-1.]),
    lambda: FeasibleSet([[1.]], [0], [0], [1.]),
    lambda: FeasibleSet([[1.]], [], [0, 0], [1., 1.]),
    lambda: FeasibleSet([[1.]], [], [2], [1.]),
    lambda: FeasibleSet([[1.]], [], [0.], [1.]),
    lambda: toy(K=[[1., 0.], [0., -1.]]),
    lambda: toy(K=[[1., 2.], [0., 1.]]),
    lambda: toy(X=np.ones((3, 2))),
    lambda: distill_recipe(toy(), eps=-1.),
    lambda: distill_recipe(toy(), lam=-1.),
    lambda: iterate_atoms(toy(), eta=0.),
    lambda: iterate_atoms(toy(), k=-1),
    lambda: find_recipe(toy(), tol=0.),
    lambda: finite_step_bound(1., -1., .1),
    lambda: finite_step_bound(1., 1., -.1),
])
def test_invalid_inputs_are_explicit_errors(operation):
    with pytest.raises(ValueError):
        operation()


def test_import_and_solve_without_torch_dependency():
    code = """
import sys
sys.modules['torch'] = None
from src.bfas.behavior.recipe import FeasibleSet, find_recipe, distill_recipe
fs = FeasibleSet([[1., 0.], [-1., 1.]], [1], [0], [1.])
assert find_recipe(fs)['success']
assert distill_recipe(fs)['success']
assert sys.modules['torch'] is None
"""
    completed = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
