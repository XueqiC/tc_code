"""CPU synthetic checks for response provenance, nested fitting, and the CLI."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.bfas.behavior.lowrank import (  # noqa: E402
    DecisionConfig, FitConfig, ResponsePredictor, auroc, cross_fit, cross_fit_targets, evaluate, fit_hierarchical, fit_model,
    group_bootstrap, grouped_folds, permute_within_strata, reduced_rank_ridge,
    residual_error_bounds, response_modes, spearman, stage_decision,
)
from src.bfas.behavior.response import (  # noqa: E402
    OutcomeRecord, ProbeRecord, assemble_matrix, content_hash, read_responses,
    validate_split, write_json, write_responses,
)


def operator_data():
    # Family structure is present in the update itself. No labels are required
    # to predict an unseen mixture containing several family directions.
    contrasts = np.array(np.meshgrid(*[[-1., 1.]] * 3)).reshape(3, -1).T
    X = np.tile(contrasts, (12, 1))
    factors = np.tile(np.repeat(["a", "b", "silent"], len(contrasts)), 4)
    X[factors != "a", 1] = 0
    X[factors != "b", 2] = 0
    R = np.diag([5., 2., 1.])
    groups = np.repeat(np.arange(4), 3 * len(contrasts))
    return X, R @ X.T, factors, groups, R


@pytest.mark.parametrize("hierarchical", [False, True])
def test_response_predictor_realized_update_identity_for_all_codes_and_deltas(hierarchical):
    from src.bfas.behavior.whiten import DiagonalMetric, build_basis

    H = DiagonalMetric(np.array([1., 2., 3., 4., 5.]), "layout", .1, "v1", "fisher", 2)
    basis = build_basis(list(np.eye(5)[:3]), H)
    C = np.column_stack([basis.decode(x) for x in np.eye(basis.rank)])
    X, M, labels, groups, true_R = operator_data()
    if hierarchical:
        model = fit_hierarchical(X, M, groups, source_factors=labels,
                                  config=FitConfig(model="hierarchical", lambdas=(0.,),
                                                   family_ranks=(1,), min_family_sources=2,
                                                   residual_rank=1))
        assert model.mode == "operator"
    else:
        model = fit_model(X, M, rank=3, ridge=0.)
    predictor = ResponsePredictor(model, basis)
    np.testing.assert_allclose(predictor.R, true_R, atol=1e-12)
    # Identical realized updates with different sparse/nonnegative recipes.
    codes = np.array([[.2, .4, -.1], [.7, -.3, .9]])
    D_codes = np.column_stack([codes[0], codes[0] + codes[1], codes[0] - codes[1]])
    D = C @ D_codes
    a, b = np.array([1., 0., 0.]), np.array([0., .5, .5])
    np.testing.assert_allclose(D @ a, D @ b, atol=1e-12)
    B_hat = predictor.predict_from_deltas(D.T, basis)
    np.testing.assert_allclose(predictor.predict_from_deltas(iter(D.T)), B_hat)
    np.testing.assert_allclose(B_hat, model.R @ C.T @ np.diag(H.diagonal) @ D, atol=1e-12)
    np.testing.assert_allclose(B_hat, predictor.predict_from_codes(D_codes.T), atol=1e-12)
    for weights in (a, b, np.array([-.5, 3., 2.])):
        np.testing.assert_allclose(predictor.predict_recipe(weights, D_codes),
                                   predictor.predict_from_deltas(D @ weights), atol=1e-12)
        np.testing.assert_allclose(B_hat @ weights, predictor.predict_recipe(weights, D_codes), atol=1e-12)
    np.testing.assert_allclose(predictor.predict_recipe(a, D_codes),
                               predictor.predict_recipe(b, D_codes), atol=1e-12)
    # Arbitrary held-out updates include a component outside the fitted span.
    deltas = np.random.default_rng(51).normal(size=(7, H.size))
    np.testing.assert_allclose(predictor.predict_from_deltas(deltas),
                               model.R @ C.T @ np.diag(H.diagonal) @ deltas.T, atol=1e-12)
    np.testing.assert_array_equal(predictor.predict_from_deltas(np.eye(5)[4]), np.zeros(3))
    assert predictor.predict_from_deltas([]).shape == (3, 0)
    with pytest.raises(ValueError, match="parameter"):
        predictor.predict_from_deltas(np.empty((0, H.size + 1)))


def test_operator_hierarchy_ignores_prediction_labels_and_is_linear():
    X, M, labels, groups, _ = operator_data()
    model = fit_hierarchical(X, M, groups, source_factors=labels,
                              config=FitConfig(model="hierarchical", lambdas=(0.,),
                                               family_ranks=(1,), min_family_sources=2))
    assert model.mode == FitConfig().hierarchical_mode == "operator"
    updates = np.repeat([[1., 2., 3.]], 4, axis=0)
    expected = model.R @ updates.T
    for metadata in (None, {"source_factor": ["a", "b", "silent", "unknown"]},
                     {"source_factor": ["unknown", "silent", "a", "b"]}):
        np.testing.assert_allclose(model.predict(updates, metadata), expected, atol=1e-12)
        np.testing.assert_allclose(sum(model.predict_levels(updates, metadata).values()), expected, atol=1e-12)
    np.testing.assert_allclose(model.predict(updates[:1] * 2), 2 * model.predict(updates[:1]))
    blocks = model.blocks()
    np.testing.assert_allclose(model.R, sum(block.R for block in blocks.values()))
    assert np.linalg.norm(blocks["family:a"].R) > 1
    assert np.linalg.norm(blocks["family:b"].R) > .5
    assert model.diagnostics["mode"] == "operator"
    # Explicit legacy gating is available but cannot leak into the predictor.
    model.mode = "gated"
    gated = model.predict(updates, {"source_factor": ["a", "b", "silent", "unknown"]})
    assert not np.allclose(gated[:, 0], gated[:, 1])
    np.testing.assert_allclose(ResponsePredictor(model).predict_from_codes(updates), expected)
    np.testing.assert_allclose(model.predict(updates, mode="operator"), expected)


def test_predictor_rejects_metadata_baselines_and_invalid_axes():
    X = np.eye(2)
    with pytest.raises(ValueError, match="metadata baselines"):
        ResponsePredictor(fit_model(X, X, name="B0-norm"))
    predictor = ResponsePredictor(np.ones((3, 2)))
    for operation in (lambda: predictor.predict_from_codes(np.zeros((2, 3))),
                      lambda: predictor.predict_from_codes([0., np.nan]),
                      lambda: predictor.predict_recipe([1.], np.ones((3, 1))),
                      lambda: predictor.predict_recipe([np.inf], np.ones((2, 1))),
                      lambda: predictor.predict_from_deltas([1., 2.]),
                      lambda: ResponsePredictor(np.ones((2, 2)), SimpleNamespace(rank=3))):
        with pytest.raises(ValueError):
            operation()


def test_residual_error_bounds_are_empirical_quantiles_preserving_missing_probes():
    residuals = np.array([[-1., 2., -3., np.nan], [np.nan] * 4, [0., 4., -8., 12.]])
    for q in (0., .5, .9, 1.):
        E = residual_error_bounds(residuals, quantile=q)
        assert E.shape == (3, 1) and np.isnan(E[1, 0])
        assert E[0, 0] == pytest.approx(np.quantile([1., 2., 3.], q))
        assert E[2, 0] == pytest.approx(np.quantile([0., 4., 8., 12.], q))
    assert np.isnan(residual_error_bounds(np.empty((2, 0)))).all()
    for q in (-.1, 1.1, np.nan, np.inf):
        with pytest.raises(ValueError):
            residual_error_bounds(residuals, quantile=q)
        with pytest.raises(ValueError):
            FitConfig(error_quantile=q)


@pytest.mark.parametrize("hierarchical", [False, True])
def test_grouped_cross_fit_reports_quantiles_of_heldout_residuals(hierarchical):
    X, M, labels, groups, _ = operator_data()
    M = M + np.random.default_rng(72).normal(scale=.2, size=M.shape)
    config = FitConfig(model="hierarchical" if hierarchical else "flat", ranks=(1, 2),
                       family_ranks=(1,), min_family_sources=2, lambdas=(.1,),
                       error_quantile=.75, permutation_seeds=(), bootstrap_samples=0)
    result = cross_fit(X, M, groups, groups, config=config, metadata={"source_factor": labels})
    for name, predictions in result.predictions.items():
        np.testing.assert_allclose(result.error_bounds[name],
                                   np.quantile(np.abs(M - predictions), .75, axis=1)[:, None])
    if hierarchical:
        for fold in result.folds:
            model = fold["models"]["hierarchical"]
            np.testing.assert_allclose(result.predictions["hierarchical"][:, fold["test"]],
                                       model.R @ X[fold["test"]].T)
    diagnostics = result.diagnostics["estimation_error"]
    assert diagnostics["quantile"] == .75 and diagnostics["is_confidence_bound"] is False
    np.testing.assert_array_equal(diagnostics["finite_residual_counts"]["B2"], M.shape[1])


def planted(seed=8, n=96):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 10))
    X[:, :8] *= 5  # high-variance update directions have no behavioural effect
    R = np.zeros((12, 10))
    R[:, -2:] = rng.normal(size=(12, 2))
    M = R @ X.T + rng.normal(scale=0.01, size=(12, n))
    groups = np.array([f"task-{i // 2:03}" for i in range(n)])
    folds = np.arange(n) // 2 % 4
    return X, M, R, groups, folds


@pytest.fixture(scope="module")
def competition():
    X, M, R, groups, folds = planted()
    config = FitConfig(lambdas=(0, 0.01, 0.1), permutation_seeds=(11, 29), bootstrap_samples=30)
    result = cross_fit(X, M, groups, folds, config=config,
                       probe_groups=[f"probe-{i // 3}" for i in range(len(M))],
                       metadata={"category": np.where(np.arange(len(X)) % 2, "a", "b"),
                                 "length": np.arange(len(X)) % 7 + 10})
    return X, M, R, groups, folds, config, result


def test_planted_b2_recovers_and_beats_common_and_pca(competition):
    X, M, R, _, _, _, result = competition
    assert result.metrics["B2"]["mse"] < 0.001
    for baseline in ("B0", "B1", "B-null"):
        assert result.metrics["B2"]["mse"] < 0.01 * result.metrics[baseline]["mse"]
    assert result.metrics["B2"]["net_benefit_spearman"] > 0.99
    for fold in result.folds:
        fitted = fold["models"]["B2"]
        np.testing.assert_allclose(fitted.R, R, atol=0.01)
        assert fitted.rank in (2, 4)
        assert fitted.basis.shape[0] == X.shape[1]
    assert result.confidence_intervals["paired_mse_improvement"]["B2_vs_B1"]["low"] > 0


def test_permuted_mapping_loses_predictive_power(competition):
    *_, result = competition
    for seed in (11, 29):
        metric = result.metrics[f"B3-seed{seed}"]
        assert metric["mse"] > 100 * result.metrics["B2"]["mse"]
        assert metric["relative_improvement_vs_zero"] < 0.15
        for fold in result.folds:
            d = fold["models"][f"B3-seed{seed}"].diagnostics
            strata = d["permutation_strata"]
            np.testing.assert_array_equal(strata[d["permutation"]], strata)
    p = permute_within_strata(["a", "a", "b", "b", "singleton"], 3)
    assert p[-1] == 4


def test_heldout_response_changes_cannot_leak_into_fit(competition):
    X, M, _, groups, folds, config, original = competition
    # Match the first fit's metadata. Only fold 0 should be invariant: labels
    # in that fold legitimately become training labels in the other fits.
    metadata = {"category": np.where(np.arange(len(X)) % 2, "a", "b"),
                "length": np.arange(len(X)) % 7 + 10}
    changed = M.copy()
    test = original.folds[0]["test"]
    changed[:, test] = np.random.default_rng(101).normal(size=(len(M), len(test))) * 10000
    changed[0, test[0]] = np.nan
    modified = cross_fit(X, changed, groups, folds, config=FitConfig(**{**asdict(config), "bootstrap_samples": 0}),
                         metadata=metadata)
    for name, model in original.folds[0]["models"].items():
        other = modified.folds[0]["models"][name]
        assert model.rank == other.rank and model.ridge == other.ridge
        np.testing.assert_array_equal(model.basis, other.basis)
        np.testing.assert_array_equal(model.R, other.R)
        assert model.cv_scores == other.cv_scores
        np.testing.assert_array_equal(original.predictions[name][:, test], modified.predictions[name][:, test])
        if not model.feature_kind:
            np.testing.assert_array_equal(model.modes()["V"], other.modes()["V"])
            np.testing.assert_array_equal(model.modes()["B"], other.modes()["B"])


def test_training_span_excludes_heldout_input_direction():
    X, M, _, groups, folds = planted(n=32)
    X[folds != 0, -1] = 0
    result = cross_fit(X, M, groups, folds, config=FitConfig(ranks=(1,), lambdas=(0.1,),
                       permutation_seeds=(), bootstrap_samples=0))
    first = result.folds[0]
    np.testing.assert_allclose(first["models"]["B2"].basis[-1], 0, atol=1e-14)
    np.testing.assert_allclose(first["models"]["B2"].R[:, -1], 0, atol=1e-14)
    np.testing.assert_allclose(first["heldout_projection_residual_norm"], np.abs(X[folds == 0, -1]))


def test_zero_input_is_zero_response_in_all_models(competition):
    X, _, _, _, _, _, result = competition
    zeros = np.zeros((3, X.shape[1]))
    for fold in result.folds:
        for model in fold["models"].values():
            predicted = model.predict(zeros, {"category": ["a", "b", "unseen"], "length": [3, 10, 42]})
            np.testing.assert_array_equal(predicted, np.zeros_like(predicted))


def test_correct_reduced_rank_ridge_not_naive_coefficient_truncation():
    rng = np.random.default_rng(66)
    X = rng.normal(size=(45, 5)) * [0.2, 0.5, 2, 5, 15]
    M = rng.normal(size=(7, 5)) @ X.T + rng.normal(size=(7, 45))
    ridge, rank = 7.0, 2
    R = reduced_rank_ridge(X, M, rank, ridge)
    unrestricted = np.linalg.solve(X.T @ X + ridge * np.eye(X.shape[1]), X.T @ M.T).T
    u, s, vt = np.linalg.svd(unrestricted, full_matrices=False)
    naive = (u[:, :rank] * s[:rank]) @ vt[:rank]
    objective = lambda coeff: np.sum((M - coeff @ X.T) ** 2) + ridge * np.sum(coeff ** 2)
    assert objective(R) < objective(naive) - 1
    # Independent completion-of-squares derivation in the Gram metric.
    eig, vectors = np.linalg.eigh(X.T @ X + ridge * np.eye(X.shape[1]))
    root = (vectors * np.sqrt(eig)) @ vectors.T
    u, s, vt = np.linalg.svd(unrestricted @ root, full_matrices=False)
    expected = np.linalg.solve(root, ((u[:, :rank] * s[:rank]) @ vt[:rank]).T).T
    np.testing.assert_allclose(R, expected, rtol=1e-10, atol=1e-10)
    assert np.linalg.matrix_rank(R, tol=1e-8) == rank


def test_singular_design_and_zero_penalty():
    X = np.array([[1, 2, 0], [2, 4, 0], [-1, -2, 0]], dtype=float)
    M = np.array([[1, 2, -1], [3, 6, -3]], dtype=float)
    R = reduced_rank_ridge(X, M, 1, 0)
    np.testing.assert_allclose(R @ X.T, M, atol=1e-12)
    np.testing.assert_array_equal(reduced_rank_ridge(np.zeros_like(X), M, 2, 0), np.zeros((2, 3)))


def test_svd_orientation_parameter_right_probe_left():
    rng = np.random.default_rng(3)
    C, _ = np.linalg.qr(rng.normal(size=(17, 5)))
    R = rng.normal(size=(8, 2)) @ rng.normal(size=(2, 5))
    modes = response_modes(R, C)
    assert modes["U"].shape == (17, 2)
    assert modes["B"].shape == (8, 2)
    np.testing.assert_allclose(modes["U"], C @ modes["V"])
    np.testing.assert_allclose(R @ modes["V"], modes["B"], atol=1e-12)
    np.testing.assert_allclose(modes["B"] @ modes["V"].T, R, atol=1e-12)
    a = np.array([0.3, -0.7])
    np.testing.assert_allclose(R @ (C.T @ (modes["U"] @ a)), modes["B"] @ a, atol=1e-12)
    np.testing.assert_allclose(response_modes(R, lambda v: C @ v)["U"], modes["U"])
    assert response_modes(R)["U"] is None


def test_na_metrics_stay_na_and_zero_changes_remain_visible(tmp_path):
    M = np.array([[0, np.nan, 1], [0, np.nan, 1], [0, np.nan, np.nan]])
    report = evaluate(M, np.zeros_like(M), source_groups=["a", "b", "c"], probe_groups=["p", "p", "q"])
    assert np.isnan(report["spearman"])
    assert np.isnan(report["per_source"][0]["sign_auroc"])
    assert np.isnan(report["per_source"][1]["mse"])
    assert np.isnan(report["net_benefit"][2]["actual"])
    assert report["source_ranking"] == [0]
    assert report["counts"]["missing_cells"] == 4
    assert report["counts"]["independent_sources"] == 2  # missing group contributes no effective source
    assert report["fractions"]["zero"] == 3 / 5
    assert report["counts"]["sources_with_any_change"] == 1
    assert report["degenerate_rows"]
    empty = evaluate(np.full((2, 3), np.nan), np.zeros((2, 3)))
    assert np.isnan(empty["mse"]) and np.isnan(empty["sign_auroc"])
    assert np.isnan(spearman([1, 1], [1, 2]))
    assert np.isnan(auroc([1, 1], [0, 1]))
    assert auroc([0, 1], [0, 0]) == 0.5  # tied predictions with both classes are defined
    path = tmp_path / "metrics.json"
    write_json(path, report)
    saved = json.loads(path.read_text())
    assert saved["spearman"] is None and "NaN" not in path.read_text()


def test_missing_training_labels_are_excluded_not_imputed():
    X, M, _, _, _ = planted(n=32)
    M[0, 0] = np.nan
    M[-1] = np.nan
    model = fit_model(X, M, rank=2, ridge=0)
    assert model.diagnostics["excluded_training_sources"].tolist() == [0]
    assert model.diagnostics["unobserved_training_probes"].tolist() == [len(M) - 1]
    expected = fit_model(X[1:], M[:-1, 1:], rank=2, ridge=0)
    np.testing.assert_allclose(model.R[:-1], expected.R, atol=1e-10)
    assert np.isnan(model.predict(X)[-1]).all()
    missing = M.copy()
    missing[0, ::2], missing[1, 1::2] = np.nan, np.nan
    assert np.isnan(fit_model(X, missing).predict(X)).all()


def test_group_bootstrap_resamples_groups_and_pairs_controls():
    M = np.array([[1, 1, -2, -2]], dtype=float)
    P = {"B2": M.copy(), "B0": np.zeros_like(M), "B1": np.zeros_like(M)}
    intervals = group_bootstrap(M, P, ["a", "a", "b", "b"], samples=50, seed=8)
    delta = intervals["paired_mse_improvement"]["B2_vs_B0"]
    assert delta["low"] >= 1 and delta["high"] <= 4
    one = group_bootstrap(M, P, ["a"] * 4, samples=10)
    assert np.isnan(one["models"]["B2"]["mse"]["low"])
    two_way = group_bootstrap(np.tile(M, (3, 1)), {k: np.tile(v, (3, 1)) for k, v in P.items()},
                              ["a", "a", "b", "b"], probe_groups=["p", "q", "q"],
                              samples=10, two_way=True)
    assert two_way["models"]["B2"]["mse"]["valid_resamples"] == 10


def test_group_folds_reject_leakage_and_invalid_coverage():
    with pytest.raises(ValueError, match="group leakage"):
        grouped_folds(["a", "a", "b", "b"], [0, 1, 0, 1])
    with pytest.raises(ValueError, match="exactly once"):
        grouped_folds(["a", "b", "c"], [{"train": [0, 1], "test": [2]}])
    with pytest.raises(ValueError, match="integer"):
        grouped_folds(["a", "b"], [{"train": [0.5], "test": [1]}])


def probe(pid="p"):
    return ProbeRecord(pid, "probe-parent", "probe-trajectory", content_hash("state"),
                       content_hash("target"), content_hash("student"), "bfcl", "dev", "probe-group")


def outcome(source="s", seed=0, **kwargs):
    defaults = dict(source_id=source, run_id="run-" + source, probe_id="p", eval_seed=seed,
                    raw_output_hash=content_hash("raw"), checker_version="v1", base_outcome=0, updated_outcome=1)
    return OutcomeRecord(**{**defaults, **kwargs})


def test_response_jsonl_manifest_and_seed_na_roundtrip(tmp_path):
    records = [outcome(seed=0), outcome(seed=1), outcome("other", seed=0),
               outcome("other", seed=1, status="failed", error_type="timeout", updated_outcome=None)]
    path = tmp_path / "responses.jsonl"
    original = write_responses(path, [probe()], records, {"eval_seeds": [0, 1], "run_id": "demo"})
    probes, recovered, manifest = read_responses(path)
    assert probes == [probe()] and recovered == records and manifest == original
    response = assemble_matrix(recovered, source_ids=["s", "other", "absent"], probe_ids=["p"], eval_seeds=[0, 1])
    assert response.M[0, 0] == 1
    assert np.isnan(response.M[0, 1:]).all()
    assert response.valid_repeats.tolist() == [[2, 1, 0]]
    assert response.diagnostics["missing_records"] == 2
    write_responses(path, probes, recovered, {"eval_seeds": [0, 1], "run_id": "demo"}, resume=True)
    with pytest.raises(FileExistsError):
        write_responses(path, probes, recovered, {"run_id": "different"}, resume=True)
    path.write_text(path.read_text().replace('"checker_version":"v1"', '"checker_version":"v2"'))
    with pytest.raises(ValueError, match="hash mismatch"):
        read_responses(path)


def test_outcome_validation_and_split_overlap():
    assert outcome(updated_outcome=0).difference == 0
    with pytest.raises(ValueError, match="difference"):
        outcome(difference=0)
    with pytest.raises(ValueError, match="duplicate"):
        assemble_matrix([outcome(), outcome()], source_ids=["s"], probe_ids=["p"])
    with pytest.raises(ValueError, match="multiple update runs"):
        assemble_matrix([outcome(), outcome(seed=1, run_id="different")], source_ids=["s"], probe_ids=["p"])
    with pytest.raises(ValueError, match="inconsistent paired base"):
        assemble_matrix([outcome(), outcome("other", base_outcome=1)], source_ids=["s", "other"], probe_ids=["p"])
    with pytest.raises(ValueError, match="parent_task_id"):
        validate_split([{"parent_task_id": "probe-parent"}], [probe()])
    assert probe().content_key != ProbeRecord(**{**asdict(probe()), "target_hash": content_hash("changed")}).content_key


def decision_config(**overrides):
    values = dict(min_source_groups=4, min_probe_groups=2, min_changed_sources=8,
                  min_observed_fraction=0.8, max_ci_width=10, min_zero_improvement=0.01,
                  min_baseline_improvement=0.01, min_permutation_gap=0.01,
                  min_net_benefit_spearman=0.5, min_leave_group_out_improvement=0.01,
                  min_fold_win_fraction=0.75, min_permutation_moved_fraction=0.5)
    return DecisionConfig(**{**values, **overrides})


def test_stage_decision_requires_measurement_and_frozen_thresholds(competition):
    *_, result = competition
    assert stage_decision(result, None)["decision"] == "INCONCLUSIVE"
    cfg = decision_config()
    assert stage_decision(result, cfg)["decision"] == "INCONCLUSIVE"
    assert stage_decision(result, cfg, measurement_stable=True, measurement_resolved=True)["decision"] == "GO-candidate"
    assert stage_decision(result, cfg, measurement_stable=True, measurement_resolved=True,
                          response_kind="likelihood")["decision"] == "NO-GO"
    assert stage_decision(result, decision_config(min_source_groups=1000), measurement_stable=True,
                          measurement_resolved=True)["decision"] == "INCONCLUSIVE"


def test_stage_simplify_and_nogo_use_configured_bounds(competition):
    *_, result = competition
    # Isolate gate policy from refitting: move interval bounds, not thresholds
    # inferred from results. Other power/measurement checks remain satisfied.
    payload = json.loads(json.dumps({"metrics": result.metrics, "confidence_intervals": result.confidence_intervals,
                                     "diagnostics": result.diagnostics}, default=lambda x: x.tolist()))
    payload["confidence_intervals"]["models"]["B0"]["improvement_vs_zero"].update(low=0.2, high=0.3)
    payload["confidence_intervals"]["paired_mse_improvement"]["B2_vs_B0"].update(low=-0.002, high=0.002)
    assert stage_decision(payload, decision_config(), measurement_stable=True, measurement_resolved=True)["decision"] == "SIMPLIFY"
    payload["confidence_intervals"]["models"]["B0"]["improvement_vs_zero"].update(low=-0.2, high=0)
    payload["confidence_intervals"]["models"]["B2"]["improvement_vs_zero"].update(low=-0.1, high=0)
    assert stage_decision(payload, decision_config(), measurement_stable=True, measurement_resolved=True)["decision"] == "NO-GO"


def cli(*args):
    return subprocess.run([sys.executable, str(ROOT / "tools/behavior_atom_experiment.py"), *map(str, args)],
                          cwd=ROOT, capture_output=True, text=True, timeout=60)


@pytest.mark.parametrize("extension", ["json", "npz"])
def test_fit_cli_end_to_end_dry_run_real_resume_report(tmp_path, extension):
    X, M, _, groups, folds = planted(n=32)
    data = {"X": X, "M": M, "source_groups": groups, "folds": folds,
            "probe_groups": np.array(["p", "q"] * 6)}
    data_path = tmp_path / f"responses.{extension}"
    if extension == "json":
        write_json(data_path, data)
    else:
        np.savez(data_path, **data)
    config = tmp_path / "config.json"
    write_json(config, {"run_id": "synthetic", "data": data_path.name,
                        "fit": {"ranks": [1, 2], "lambdas": [0, 0.1], "permutation_seeds": [11], "bootstrap_samples": 5}})
    out = tmp_path / "results"
    arguments = ["fit", "--config", config, "--output-dir", out]
    dry = cli(*arguments, "--dry-run")
    assert dry.returncode == 0, dry.stderr
    assert json.loads(dry.stdout)["new_teacher_tokens"] == 0
    assert not out.exists()
    real = cli(*arguments)
    assert real.returncode == 0, real.stderr
    report = json.loads((out / "stage_report.json").read_text())
    assert report["decision"] == "INCONCLUSIVE"  # no measured pilot or frozen thresholds
    assert report["metrics"]["B2"]["mse"] < report["metrics"]["B1"]["mse"]
    assert report["costs"]["new_teacher_tokens"] == 0
    before = (out / "fit.manifest.json").read_bytes()
    resumed = cli(*arguments, "--resume")
    assert resumed.returncode == 0, resumed.stderr
    assert (out / "fit.manifest.json").read_bytes() == before
    assert cli(*arguments).returncode != 0
    rendered = cli("report", "--config", config, "--manifest", out / "fit_result.json", "--output-dir", out)
    assert rendered.returncode == 0, rendered.stderr
    assert (out / "report.md").exists()
    assert cli("fit", "--config", config, "--output-dir", out, "--resume", "--dry-run").returncode != 0
    changed = json.loads(config.read_text())
    changed["fit"]["lambdas"] = [0, 100]
    config.write_text(json.dumps(changed))
    assert cli(*arguments, "--resume").returncode != 0


def test_fit_cli_jsonl_aligns_explicit_source_ids(tmp_path):
    rng = np.random.default_rng(6)
    X = rng.normal(size=(16, 3))
    probes = [ProbeRecord(**{**asdict(probe()), "probe_id": f"p{i}"}) for i in range(4)]
    records = [outcome(f"s{i}", probe_id=p.probe_id, updated_outcome=float(X[i, 0] > 0))
               for i in range(16) for p in probes]
    responses = tmp_path / "paired.jsonl"
    write_responses(responses, probes, records, {"eval_seeds": [0]})
    codes = tmp_path / "codes.npz"
    np.savez(codes, X=X, source_ids=[f"s{i}" for i in range(16)],
             source_groups=[f"g{i}" for i in range(16)], folds=np.arange(16) % 4)
    descriptor = tmp_path / "dataset.json"
    write_json(descriptor, {"responses": responses.name, "codes": codes.name})
    config = tmp_path / "config.json"
    write_json(config, {"data": descriptor.name, "fit": {"ranks": [1], "lambdas": [0.1],
                                                       "permutation_seeds": [], "bootstrap_samples": 0}})
    real = cli("fit", "--config", config, "--output-dir", tmp_path / "out")
    assert real.returncode == 0, real.stderr
    fit = json.loads((tmp_path / "out/fit_result.json").read_text())
    assert fit["source_ids"] == [f"s{i}" for i in range(16)]
    assert fit["response_diagnostics"]["missing_records"] == 0


def test_collect_refuses_without_pilot_pass_and_future_stages_are_plan_only(tmp_path):
    config, manifest = tmp_path / "config.json", tmp_path / "manifest.json"
    write_json(config, {"protocol": {"micro_update": {"model_id": "cpu-stub"}, "eval_seeds": [0]}})
    write_json(manifest, {})
    for extra in ([], ["--dry-run"]):
        refusal = cli("collect", "--config", config, "--manifest", manifest, *extra)
        assert refusal.returncode != 0 and "pilot pass" in refusal.stderr
    for stage in ("intervene", "distill"):
        assert cli(stage, "--config", config).returncode != 0
        planned = cli(stage, "--config", config, "--dry-run")
        assert planned.returncode == 0 and json.loads(planned.stdout)["new_teacher_tokens"] == 0


def test_pilot_dry_run_costs_without_importing_optional_runner(tmp_path, monkeypatch, capsys):
    spec = importlib.util.spec_from_file_location("behavior_cli_test", ROOT / "tools/behavior_atom_experiment.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.importlib, "import_module", lambda *a, **k: pytest.fail("dry-run imported optional runner"))
    config, manifest = tmp_path / "config.json", tmp_path / "manifest.json"
    write_json(config, {"protocol": {"micro_update": {"model_id": "stub"}, "eval_seeds": [7]},
                        "cost_estimate": {"gpu_hours_per_update": 0.1, "gpu_seconds_per_rollout": 2,
                                          "environment_seconds_per_rollout": 3}})
    write_json(manifest, {"sources": [{"source_id": "s", "parent_task_id": "s-parent",
                                      "trajectory_id": "s-traj", "state_hash": "s-hash"}], "probes": [asdict(probe())]})
    assert module.main(["pilot", "--config", str(config), "--manifest", str(manifest), "--dry-run"]) == 0
    planned = json.loads(capsys.readouterr().out)
    assert planned["number_of_models"] == 2 and planned["max_rollouts"] == 2
    assert planned["cost_estimate"]["environment_seconds"] == 6
    assert planned["cost_estimate"]["gpu_hours"] == pytest.approx(0.1 + 4 / 3600)


def test_pilot_hash_gate_binds_protocol_and_evidence(tmp_path):
    from tools.behavior_atom_experiment import frozen_protocol, require_pilot_pass
    config = {"_base": str(tmp_path), "protocol": {"micro_update": {"model_id": "stub", "steps": 3}},
              "pilot_pass": "pass.json"}
    protocol = frozen_protocol(config)
    evidence = tmp_path / "evidence.json"
    write_json(evidence, {"changed_probes": 3})
    write_json(tmp_path / "pass.json", {"status": "pass", "passed": True,
                "measurement_stable": True, "measurement_resolved": True,
                "frozen_config_hash": content_hash(protocol), "artifact_hashes": {evidence.name: content_hash(evidence.read_bytes())}})
    assert require_pilot_pass(config, protocol, {})["passed"]
    changed = {**protocol, "micro_update": {"model_id": "stub", "steps": 4}}
    with pytest.raises(ValueError, match="frozen config hash"):
        require_pilot_pass(config, changed, {})
    evidence.write_text("{}")
    with pytest.raises(ValueError, match="evidence artifact"):
        require_pilot_pass(config, protocol, {})


def fake_collection_runtime(monkeypatch):
    from tools import behavior_atom_experiment as module
    calls, fail_source = [], {"id": None}

    def runner(factory, rows, cfg):
        calls.append(cfg)
        assert factory is None and rows[0]["response"] == "existing supervision"
        return SimpleNamespace(delta=None, manifest={"status": "complete", "run_id": cfg["run_id"],
                                                    "source_ids": [cfg["source_id"]], "new_teacher_tokens": 0})

    def evaluator(*, update, source, probes, eval_seeds, protocol):
        if fail_source["id"] == source["source_id"]:
            raise RuntimeError("simulated interrupted evaluation")
        return [outcome(source["source_id"], seed=seed, probe_id=p["probe_id"], run_id=update.manifest["run_id"])
                for p in probes for seed in eval_seeds]

    def assessor(*, outcomes, updates, protocol):
        return {"passed": True, "measurement_stable": True, "measurement_resolved": True,
                "metrics": {"changed_records": sum(r.difference != 0 for r in outcomes)}}

    original = module.importlib.import_module

    def imported(name):
        if name == "src.bfas.behavior.microupdate":
            return SimpleNamespace(run_micro_update=runner)
        if name == "fake_cpu_runtime":
            return SimpleNamespace(evaluator=evaluator, assessor=assessor)
        return original(name)

    monkeypatch.setattr(module.importlib, "import_module", imported)
    protocol = {"micro_update": {"model_id": "fake-cpu", "steps": 3}, "eval_seeds": [7],
                "evaluator": "fake_cpu_runtime:evaluator", "pilot_assessor": "fake_cpu_runtime:assessor"}
    sources = [{"source_id": f"s{i}", "parent_task_id": f"parent{i}", "trajectory_id": f"trajectory{i}",
                "state_hash": f"hash{i}", "rows": [{"response": "existing supervision", "_rejected": "student"}]}
               for i in range(2)]
    return module, calls, fail_source, protocol, {"sources": sources, "probes": [asdict(probe())]}


def test_pilot_collect_use_lazy_runner_and_measured_pass(tmp_path, monkeypatch):
    module, calls, _, protocol, data = fake_collection_runtime(monkeypatch)
    data_path, pilot_config = tmp_path / "data.json", tmp_path / "pilot.json"
    write_json(data_path, data)
    write_json(pilot_config, {"run_id": "fake", "data": data_path.name, "protocol": protocol})
    pilot_out, collect_out = tmp_path / "pilot", tmp_path / "collect"
    assert module.main(["pilot", "--config", str(pilot_config), "--output-dir", str(pilot_out)]) == 0
    assert len(calls) == 2 and (pilot_out / "pilot_pass.json").is_file()
    collect_config = tmp_path / "collect.json"
    write_json(collect_config, {"run_id": "fake", "data": data_path.name, "protocol": protocol,
                               "pilot_pass": str(pilot_out / "pilot_pass.json")})
    args = ["collect", "--config", str(collect_config), "--output-dir", str(collect_out)]
    assert module.main(args) == 0
    assert len(calls) == 4 and all(c["steps"] == 3 for c in calls)
    assert module.main([*args, "--resume"]) == 0
    assert len(calls) == 4
    probes, outcomes, manifest = read_responses(collect_out / "responses.jsonl")
    assert len(outcomes) == 2 and manifest["provenance"]["eval_seeds"] == [7]
    assert all(o.difference == 1 for o in outcomes)


def test_partial_collect_resumes_completed_sources_without_retraining(tmp_path, monkeypatch):
    module, calls, failure, protocol, data = fake_collection_runtime(monkeypatch)
    config = {"_base": str(tmp_path), "protocol": protocol}
    frozen = module.frozen_protocol(config)
    failure["id"] = "s1"
    out = tmp_path / "partial"
    with pytest.raises(RuntimeError, match="interrupted"):
        module._collect("pilot", config, frozen, data, out, False)
    failure["id"] = None
    result = module._collect("pilot", config, frozen, data, out, True)
    assert result["status"] == "complete"
    assert [c["source_id"] for c in calls] == ["s0", "s1", "s1"]
    assert calls[-1]["resume"] is True  # delegate completed delta replay to the runner
    assert (out / "pilot_pass.json").exists()


def test_gate_accepts_serialized_na_without_turning_it_into_zero(competition, tmp_path):
    *_, result = competition
    path = tmp_path / "gate.json"
    payload = {"metrics": result.metrics, "confidence_intervals": result.confidence_intervals,
               "diagnostics": result.diagnostics}
    # Copy via the strict serializer, then intentionally make a required bound NA.
    write_json(path, payload)
    saved = json.loads(path.read_text())
    saved["confidence_intervals"]["models"]["B2"]["improvement_vs_zero"]["low"] = None
    assert stage_decision(saved, decision_config(), measurement_stable=True,
                          measurement_resolved=True)["decision"] == "INCONCLUSIVE"


def hierarchical_planted(*, residual=False):
    # Balanced contrast sources make global/family factors identifiable in every
    # training fold, including inner folds. Parent groups span all three families.
    contrasts = np.array(np.meshgrid(*[[-1., 1.]] * 5)).reshape(5, -1).T
    X = np.tile(contrasts, (12, 1))
    factors = np.tile(np.repeat(["api-a", "api-b", "silent"], len(contrasts)), 4)
    folds = np.repeat(np.arange(4), 3 * len(contrasts))
    groups = np.array([f"parent-{f}" for f in folds])
    global_R = np.zeros((6, 5))
    global_R[0, 0] = 5
    blocks = {f: np.zeros_like(global_R) for f in set(factors)}
    blocks["api-a"][1, 1] = 2
    blocks["api-a"][2, 2] = 1
    blocks["api-b"][3, 3] = 2
    M = global_R @ X.T
    for f, R in blocks.items():
        M[:, factors == f] += R @ X[factors == f].T
    if residual:
        # Family rank-1 leaves a shared second direction for the residual level.
        M[4] += 0.4 * X[:, 4]
    return X, M, factors, groups, folds, global_R, blocks


def hierarchical_config(**overrides):
    # These original fixtures plant label-dependent responses for identical
    # codes. Preserve their legacy-estimator checks with an explicit opt-in.
    return FitConfig(**{**dict(model="hierarchical", ranks=(1, 2, 4), family_ranks=(1, 2),
                              lambdas=(0., 0.1), min_family_sources=8, inner_splits=3,
                              permutation_seeds=(), bootstrap_samples=0,
                              hierarchical_mode="gated"), **overrides})


@pytest.fixture(scope="module")
def hierarchy_competition():
    X, M, factors, groups, folds, global_R, blocks = hierarchical_planted()
    result = cross_fit(X, M, groups, folds, config=hierarchical_config(permutation_seeds=(11,), bootstrap_samples=20),
                       metadata={"source_factor": factors}, probe_factors=["boundary", "a", "a", "b", "none", "none"])
    return X, M, factors, groups, folds, global_R, blocks, result


def test_hierarchical_recovers_global_and_correct_family_blocks(hierarchy_competition):
    X, M, factors, _, _, global_R, blocks, result = hierarchy_competition
    assert result.metrics["hierarchical"]["mse"] < 1e-20
    assert result.metrics["hierarchical"]["mse"] < result.metrics["B2-matched"]["mse"] * 1e-8
    for fold in result.folds:
        model = fold["models"]["hierarchical"]
        np.testing.assert_allclose(model.global_model.R, global_R, atol=1e-10)
        for f, R in blocks.items():
            np.testing.assert_allclose(model.family_models[f].R, R, atol=1e-10)
        assert np.linalg.norm(model.family_models["silent"].R) < 1e-10
        assert model.family_models["api-a"].requested_rank == 2
        assert fold["models"]["B2-matched"].requested_rank == model.requested_rank
        assert all(block.ridge in (0., 0.1) for block in model.blocks().values())
        for name, modes in model.modes(np.eye(X.shape[1])).items():
            np.testing.assert_allclose(modes["B"] @ modes["V"].T, model.blocks()[name].R, atol=1e-12)
            np.testing.assert_allclose(modes["U"].T @ modes["U"], np.eye(modes["U"].shape[1]), atol=1e-12)
    levels = result.diagnostics["explained_variance_by_level"]
    assert levels["global"]["incremental_explained_variance"] > 0.8
    assert levels["family"]["incremental_explained_variance"] > 0.01
    assert levels["family"]["cumulative_explained_variance"] == pytest.approx(1)
    assert levels["residual"]["sse_reduction"] == 0
    np.testing.assert_allclose(sum(result.level_predictions.values()), result.predictions["hierarchical"])
    # The balanced design has many exact ties; roundoff in near-zero block
    # predictions breaks those ties in the existing (exact-tie) rank metric.
    assert result.metrics["hierarchical"]["per_factor"]["probe_factor"]["a"]["net_benefit_spearman"] > 0.8
    assert result.metrics["B3-seed11"]["mse"] > 0.1
    assert "hierarchical_vs_B2-matched" in result.confidence_intervals["paired_mse_improvement"]
    assert stage_decision(result, decision_config(), measurement_stable=True,
                          measurement_resolved=True)["decision"] == "GO-candidate"
    # Every factor sees the same code. Only its own block is applied.
    model = result.folds[0]["models"]["hierarchical"]
    probe_X = np.ones((4, X.shape[1]))
    P = model.predict(probe_X, {"source_factor": ["api-a", "api-b", "silent", "unknown"]})
    np.testing.assert_allclose(P[:, 0], (global_R + blocks["api-a"]) @ probe_X[0], atol=1e-10)
    np.testing.assert_allclose(P[:, 1], (global_R + blocks["api-b"]) @ probe_X[1], atol=1e-10)
    np.testing.assert_allclose(P[:, 2:], np.tile(global_R @ probe_X[0], (2, 1)).T, atol=1e-10)
    for block in result.folds[0]["models"].values():
        np.testing.assert_array_equal(block.predict(np.zeros_like(probe_X),
                                      {"source_factor": ["api-a", "api-b", "silent", "unknown"]}), np.zeros((6, 4)))


def test_hierarchical_residual_level_predicts_remaining_signal():
    X, M, factors, groups, folds, *_ = hierarchical_planted(residual=True)
    result = cross_fit(X, M, groups, folds, config=hierarchical_config(family_ranks=(1,), residual_rank=2),
                       metadata={"source_factor": factors})
    level = result.diagnostics["explained_variance_by_level"]["residual"]
    assert level["incremental_explained_variance"] > 0.005
    assert result.metrics["hierarchical"]["mse"] < np.mean((M - result.level_predictions["global"] - result.level_predictions["family"]) ** 2)
    assert all(f["models"]["hierarchical"].residual_model.requested_rank == 2 for f in result.folds)


def test_hierarchical_pools_rare_families_from_training_only():
    X, M, factors, groups, folds, *_ = hierarchical_planted()
    labels = factors.copy()
    for f in range(4):
        use = np.flatnonzero((folds == f) & (factors == "silent"))
        labels[use[:3]] = "rare-a"
        labels[use[3:6]] = "rare-b"
    model = fit_hierarchical(X[folds != 0], M[:, folds != 0], groups[folds != 0],
                             source_factors=labels[folds != 0], config=hierarchical_config(min_family_sources=10))
    assert model.factor_mapping["rare-a"] == model.factor_mapping["rare-b"] == "other"
    assert model.diagnostics["family_training_sources"]["other"] == 18
    np.testing.assert_allclose(model.predict(X[:1], {"source_factor": ["unseen"]}),
                               model.predict(X[:1], {"source_factor": ["rare-a"]}))
    with pytest.raises(ValueError, match="source_factor"):
        model.predict(X[:1])


def test_hierarchical_missing_responses_remain_na():
    X, M, factors, groups, _, *_ = hierarchical_planted()
    M[-1] = np.nan
    M[0, 0] = np.nan
    model = fit_hierarchical(X, M, groups, source_factors=factors, config=hierarchical_config(residual_rank=1))
    assert model.global_model.diagnostics["excluded_training_sources"].tolist() == [0]
    assert model.family_models["api-a"].diagnostics["excluded_training_sources"].tolist() == [0]
    P = model.predict(X, {"source_factor": factors})
    assert np.isnan(P[-1]).all() and np.isfinite(P[:-1]).all()
    # No family may manufacture labels when no source has a complete response.
    M[0, ::2], M[1, 1::2] = np.nan, np.nan
    abstained = fit_hierarchical(X, M, groups, source_factors=factors, config=hierarchical_config())
    assert np.isnan(abstained.predict(X, {"source_factor": factors})).all()


@pytest.mark.parametrize("seed", [2, 7, 19])
def test_hierarchical_has_no_advantage_on_pure_noise(seed):
    X, _, factors, groups, folds, *_ = hierarchical_planted()
    M = np.random.default_rng(seed).normal(size=(6, len(X)))
    result = cross_fit(X, M, groups, folds, config=hierarchical_config(lambdas=(0., 0.1, 1.)),
                       metadata={"source_factor": factors})
    assert result.metrics["hierarchical"]["improvement_vs_zero"] < 0
    assert result.metrics["hierarchical"]["mse"] >= result.metrics["B2-matched"]["mse"]


def test_hierarchical_heldout_labels_and_student_common_mode_do_not_leak(hierarchy_competition):
    X, M, factors, groups, folds, _, _, original = hierarchy_competition
    test = original.folds[0]["test"]
    changed = M.copy()
    changed[:, test] = np.random.default_rng(91).normal(size=(len(M), len(test))) * 1e4
    changed[0, test[0]] = np.nan
    cfg = hierarchical_config(permutation_seeds=(11,))
    modified = cross_fit(X, changed, groups, folds, config=cfg, metadata={"source_factor": factors})
    for name in ("hierarchical", "B3-seed11"):
        before, after = original.folds[0]["models"][name], modified.folds[0]["models"][name]
        assert before.factor_mapping == after.factor_mapping
        for key, model in before.blocks().items():
            other = after.blocks()[key]
            assert model.cv_scores == other.cv_scores
            assert model.ridge == other.ridge and model.requested_rank == other.requested_rank
            np.testing.assert_array_equal(model.R, other.R)
        np.testing.assert_array_equal(original.predictions[name][:, test], modified.predictions[name][:, test])
    # A student-only mode lies in a code direction independent of teacher modes.
    student = np.arange(1., 7.)[:, None] * X[:, 4]
    sides = cross_fit_targets(X, M, student, groups, folds, config=hierarchical_config(),
                              metadata={"source_factor": factors})
    np.testing.assert_array_equal(sides["teacher"].predictions["hierarchical"], original.predictions["hierarchical"])
    np.testing.assert_allclose(sides["combined"].target_matrix, M, atol=1e-10)
    np.testing.assert_allclose(sides["combined"].predictions["hierarchical"], original.predictions["hierarchical"], atol=1e-10)
    common = sides["combined"].diagnostics["student_common_mode"]
    np.testing.assert_allclose(common["prediction"], student, atol=1e-10)
    assert all(f["heldout_coordinate"].shape[1] == 1 for f in common["folds"])
    changed_student = student.copy()
    changed_student[:, test] += 1e4
    combined = cross_fit(X, changed - changed_student, groups, folds, student_response=changed_student,
                         config=hierarchical_config(), metadata={"source_factor": factors})
    np.testing.assert_array_equal(combined.predictions["hierarchical"][:, test],
                                   sides["combined"].predictions["hierarchical"][:, test])
    for level, model in sides["combined"].folds[0]["models"]["hierarchical"].blocks().items():
        other = combined.folds[0]["models"]["hierarchical"].blocks()[level]
        np.testing.assert_array_equal(model.R, other.R)
        assert model.cv_scores == other.cv_scores


def test_hierarchical_cli_targets_report_and_schema_labels(tmp_path):
    X, M, factors, groups, folds, *_ = hierarchical_planted()
    student = np.arange(1., 7.)[:, None] * X[:, 4]
    dataset, config, out = tmp_path / "data.json", tmp_path / "config.json", tmp_path / "fit"
    write_json(dataset, {"X": X, "M_teacher": M, "M_student": student,
                         "source_groups": groups, "folds": folds, "source_factor": factors,
                         "probe_factor": ["boundary", "a", "a", "b", "other", "other"],
                         "probe_family": ["all", "api-a", "api-a", "api-b", "other", "other"],
                         "probe_margin_base": [0.1] * 6})
    write_json(config, {"data": dataset.name, "fit": {"ranks": [1], "lambdas": [0., 0.1],
                                                     "permutation_seeds": [], "bootstrap_samples": 0}})
    result = cli("fit", "--config", config, "--output-dir", out, "--model", "hierarchical", "--target", "all",
                 "--min-family-sources", 8, "--family-ranks", 1, 2, "--residual-rank", 0)
    assert result.returncode == 0, result.stderr
    saved = json.loads((out / "fit_result.json").read_text())
    assert set(saved["targets"]) == {"teacher", "student", "combined"}
    assert saved["fit_config"]["model"] == "hierarchical"
    assert saved["probe_margin_base"] == [0.1] * 6
    assert saved["source_factor"] == factors.tolist()
    assert "family:api-a" in saved["folds"][0]["models"]["hierarchical"]["blocks"]
    with np.load(out / "fit_arrays.npz", allow_pickle=False) as arrays:
        assert "target_combined_student_common_prediction" in arrays.files
        np.testing.assert_allclose(arrays["target_combined_target_matrix"], M, atol=1e-10)
    rendered = cli("report", "--config", config, "--manifest", out / "fit_result.json", "--output-dir", out,
                   "--model", "hierarchical", "--target", "all")
    assert rendered.returncode == 0, rendered.stderr
    text = (out / "report.md").read_text()
    assert "matched total rank" in text and "incremental held-out explained variance" in text
    assert "Student common mode" in text and "net-benefit Spearman" in text
    p = replace(probe(), probe_factor="boundary", probe_family="calls", probe_margin_base=-0.02)
    record = outcome(source_factor="calls")
    path = tmp_path / "labels.jsonl"
    write_responses(path, [p], [record], {"eval_seeds": [0]})
    recovered_probes, recovered_outcomes, _ = read_responses(path)
    assert recovered_probes == [p] and recovered_outcomes == [record]
    with pytest.raises(ValueError, match="probe_margin_base"):
        replace(p, probe_margin_base=np.nan)
    from tools.behavior_atom_experiment import _jsonl_dataset, _fit_inputs
    recovered = _jsonl_dataset(path, {"source_ids": ["s"]}, {}, {})
    assert recovered["source_factor"] == ["calls"]
    assert recovered["probe_factor"] == ["boundary"]
    assert recovered["probe_family"] == ["calls"]
    assert recovered["probe_margin_base"] == [-0.02]
    # Manifest rows can have a different order from the saved matrix axes.
    source_ids = [f"s{i}" for i in range(len(X))]
    aligned = _fit_inputs({"X": X, "M": M, "source_groups": groups, "folds": folds,
                           "source_ids": source_ids,
                           "sources": [{"source_id": sid, "source_factor": f}
                                       for sid, f in reversed(list(zip(source_ids, factors)))]},
                          {"fit": {"model": "hierarchical"}})
    assert aligned["metadata"]["source_factor"] == factors.tolist()
