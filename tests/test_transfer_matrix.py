"""CPU-only tests of the transfer-matrix bookkeeping (tools/bfcl_transfer_matrix.py) and of the
evaluation statistics (tools/bfcl_transfer_matrix_eval.py) on synthetic data."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))

import bfcl_transfer_matrix as tm  # noqa: E402
import bfcl_transfer_matrix_eval as te  # noqa: E402


# ----------------------------------------------------------------------------------------------
# measurement bookkeeping
# ----------------------------------------------------------------------------------------------
def test_row_order_is_seeded_permutation():
    a, b = tm.row_order(50, 0), tm.row_order(50, 0)
    assert np.array_equal(a, b)
    assert sorted(a.tolist()) == list(range(50))
    assert not np.array_equal(a, tm.row_order(50, 1))


def test_make_batches_covers_all_and_respects_budget():
    rng = np.random.default_rng(0)
    lengths = rng.integers(20, 500, size=97).tolist()
    batches = tm.make_batches(lengths, token_budget=2000, max_batch=16)
    seen = sorted(k for b in batches for k in b)
    assert seen == list(range(97))
    for b in batches:
        assert len(b) <= 16
        assert max(lengths[k] for k in b) * len(b) <= 2000 or len(b) == 1
    # lengths are non-decreasing across the batch sequence (length-sorted)
    flat = [lengths[k] for b in batches for k in b]
    assert flat == sorted(flat)
    assert batches == tm.make_batches(lengths, token_budget=2000, max_batch=16)  # deterministic


def test_margins_and_record_row_sign():
    n = 4
    state = tm.empty_state(n, steps=3, order=tm.row_order(n))
    state["base_logp_T"] = np.array([-10, -20, -30, -40], dtype=np.float32)
    state["base_logp_S"] = np.array([-50, -60, -70, -80], dtype=np.float32)
    state["n_tok_T"] = np.array([10, 10, 10, 10], dtype=np.int32)
    state["n_tok_S"] = np.array([25, 30, 35, 40], dtype=np.int32)
    # after updating event 1: teacher of event 1 goes up by 5 nats, student of event 1 down by 5
    lpT = state["base_logp_T"].copy(); lpS = state["base_logp_S"].copy()
    lpT[1] += 5.0; lpS[1] -= 5.0
    tm.record_row(state, 1, lpT, lpS, seconds=1.5, losses=[0.7, 0.6, 0.5])
    assert state["done"][1] and not state["done"][0]
    assert state["T_sum"][1, 1] == pytest.approx(10.0)
    assert state["T_mean"][1, 1] == pytest.approx(5.0 / 10 + 5.0 / 30)
    assert np.all(state["T_mean"][1, [0, 2, 3]] == 0)
    assert np.isnan(state["T_mean"][0]).all()
    assert state["row_seconds"][1] == pytest.approx(1.5)
    assert np.allclose(state["loss_trace"][1], [0.7, 0.6, 0.5])


def test_save_load_resume_roundtrip(tmp_path):
    n = 5
    state = tm.empty_state(n, steps=2, order=tm.row_order(n))
    state["base_logp_T"] = -np.arange(1, n + 1, dtype=np.float32)
    state["base_logp_S"] = -2 * np.arange(1, n + 1, dtype=np.float32)
    state["n_tok_T"] = np.full(n, 3, dtype=np.int32); state["n_tok_S"] = np.full(n, 6, dtype=np.int32)
    rng = np.random.default_rng(1)
    for i in (2, 4):
        tm.record_row(state, i, state["base_logp_T"] + rng.normal(size=n).astype(np.float32),
                      state["base_logp_S"] + rng.normal(size=n).astype(np.float32), 0.1, [0.6, 0.5])
    path = tmp_path / "m.npz"
    tm.save_state(state, path, {"steps": 2, "lr": 5e-6, "subset": None})
    loaded, meta = tm.load_state(path)
    assert meta["steps"] == 2 and meta["subset"] is None
    assert np.array_equal(loaded["done"], state["done"])
    np.testing.assert_array_equal(loaded["T_mean"], state["T_mean"])
    with np.load(path) as f:
        assert "self_effect" in f.files
        assert f["self_effect"][2] == state["T_mean"][2, 2]
    # resume: rows not done are those in planned order with done False
    planned = loaded["row_order"]
    todo = [int(i) for i in planned if not loaded["done"][i]]
    assert set(todo) == {0, 1, 3}


def test_projected_total():
    assert tm.projected_total(np.array([10.0, 20.0, 30.0]), 3, 100, 50.0) == pytest.approx(50 + 20 * 100)


# ----------------------------------------------------------------------------------------------
# evaluation statistics
# ----------------------------------------------------------------------------------------------
def test_centring_removes_main_effects():
    rng = np.random.default_rng(0)
    n = 30
    T = rng.normal(size=(n, n)) + rng.normal(size=(n, 1)) * 3 + rng.normal(size=(1, n)) * 2
    mask = ~np.eye(n, dtype=bool)
    Tr = te.centre(T, mask, axis=1)
    Tc = te.centre(T, mask, axis=0)
    Td = te.double_centre(T, mask)
    assert np.allclose(np.where(mask, Tr, 0).sum(axis=1), 0, atol=1e-9)
    assert np.allclose(np.where(mask, Tc, 0).sum(axis=0), 0, atol=1e-9)
    assert np.allclose(np.where(mask, Td, 0).sum(axis=1), 0, atol=1e-6)
    assert np.allclose(np.where(mask, Td, 0).sum(axis=0), 0, atol=1e-6)


def test_auroc_matches_definition():
    scores = np.array([0.1, 0.4, 0.35, 0.8, 0.7])
    y = np.array([0, 0, 1, 1, 1])
    # pairs (pos, neg): (0.35 > 0.1) 1, (0.35 > 0.4) 0, (0.8 > both) 2, (0.7 > both) 2 -> 5/6
    assert te.auroc(scores, y) == pytest.approx(5 / 6)
    assert np.isnan(te.auroc(scores, np.ones(5, dtype=int)))
    assert te.auroc(np.arange(6), np.array([0, 0, 0, 1, 1, 1])) == 1.0


def test_ndcg_at_k():
    rel = np.array([0.0, 1.0, 0.5, 0.0])
    assert te.ndcg(np.array([0, 3, 2, 1]), rel, 2) == pytest.approx(1.0)      # perfect ordering
    worse = te.ndcg(np.array([3, 0, 1, 2]), rel, 2)                            # top-2 = [0, 3] -> 0 relevance
    assert worse == pytest.approx(0.0)


def test_symmetry_and_rank_stats():
    rng = np.random.default_rng(0)
    n = 40
    u = rng.normal(size=(n, 1))
    S = u @ u.T                                                 # symmetric rank-1
    S[np.diag_indices(n)] = np.nan                              # diagonal ignored
    done = np.ones(n, dtype=bool)
    sym = te.symmetry_stats(S, done)
    assert sym["pearson_T_ij_vs_T_ji"] == pytest.approx(1.0)
    assert sym["spearman_T_ij_vs_T_ji"] == pytest.approx(1.0)
    A = rng.normal(size=(n, n)); A[np.diag_indices(n)] = np.nan
    sym2 = te.symmetry_stats(A, done)
    assert abs(sym2["pearson_T_ij_vs_T_ji"]) < 0.4
    spec = te.spectrum_stats(S, done)
    assert spec["frac_norm_rank"][0] > 0.85   # diagonal is imputed with the row mean, so not exactly rank 1
    spec2 = te.spectrum_stats(A, done)
    assert spec2["frac_norm_rank"][0] < 0.3
    assert spec2["participation_ratio"] > spec["participation_ratio"]


def test_evaluate_predictor_recovers_planted_structure():
    rng = np.random.default_rng(0)
    n = 60
    psi = rng.normal(size=(n, 8)); psi /= np.linalg.norm(psi, axis=1, keepdims=True)
    T = psi @ psi.T + 0.05 * rng.normal(size=(n, n))
    done = np.ones(n, dtype=bool)
    mask = te.offdiag_mask(done, n)
    good = te.evaluate(psi @ psi.T, T, mask)
    bad = te.evaluate(rng.normal(size=(n, n)), T, mask)
    assert good["spearman_offdiag"] > 0.9 > abs(bad["spearman_offdiag"])
    assert good["spearman_doublecentred"] > 0.9
    assert good["auroc_sign"] > 0.95 and abs(bad["auroc_sign"] - 0.5) < 0.1
    assert good["ndcg@5"] > 0.9 > bad["ndcg@5"]
    # partially measured rows: only measured rows enter
    done2 = done.copy(); done2[:30] = False
    mask2 = te.offdiag_mask(done2, n)
    assert mask2.sum() == 30 * (n - 1)
    assert te.evaluate(psi @ psi.T, T, mask2)["n_pairs"] == 30 * (n - 1)


def test_group_bootstrap_and_permutation_run():
    rng = np.random.default_rng(0)
    n = 24
    psi = rng.normal(size=(n, 4))
    T = psi @ psi.T + 0.1 * rng.normal(size=(n, n))
    done = np.ones(n, dtype=bool)
    groups = np.array([f"p{k // 2}" for k in range(n)])       # two events per prompt
    S = {"good": psi @ psi.T, "rand": rng.normal(size=(n, n))}
    ci = te.group_bootstrap(S, T, done, groups, n_boot=20, seed=0)
    assert ci["good"]["spearman_offdiag"]["lo"] > ci["rand"]["spearman_offdiag"]["hi"]
    perm = te.permutation_test(S, T, done, n_perm=20, seed=0)
    assert perm["good"]["spearman_offdiag"]["p"] <= 1 / 21 + 1e-9
    assert perm["rand"]["spearman_offdiag"]["p"] > 0.05
