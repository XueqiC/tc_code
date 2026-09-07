"""tools/acquisition_replay.py: CPU unit tests of the greedy residual-reduction-per-token rule and
of the budget-filling selection rules (synthetic pool; no tokenizer, no model)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

_spec = importlib.util.spec_from_file_location("acquisition_replay", ROOT / "tools" / "acquisition_replay.py")
ar = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ar)


def _synthetic(n: int = 24, K: int = 4, seed: int = 0):
    rng = np.random.default_rng(seed)
    Z = rng.random((n, K))
    Z /= Z.sum(axis=1, keepdims=True)
    psi = rng.standard_normal((n, 8))
    psi /= np.linalg.norm(psi, axis=1, keepdims=True)
    kernel = np.maximum(psi @ psi.T, 0.0)
    cost = rng.integers(5, 200, size=n).astype(float)
    r0 = rng.random(K) * 0.6 + 0.2
    rows = [{"task_id": f"t{i // 3}", "turn_index": 0 if i % 5 else 2, "_traj": f"t{i // 3}#ev1",
             "prompt": f"p{i}", "response": f"T{i}", "_rejected": f"S{i}", "teacher": "teacher_authored_gt",
             "_event_u_plus": 1.0, "_event_u_minus": 0.2, "_event_dU": float((i % 6 + 1) / 6),
             "_seed_category": ["simple_python", "live_multiple", "oos_parallel"][i % 3]} for i in range(n)]
    return rows, Z, kernel, cost, r0


@pytest.mark.parametrize("policy", ar.POLICIES)
def test_budget_respected_for_every_policy(policy):
    rows, Z, kernel, cost, r0 = _synthetic()
    for frac in (0.1, 0.25, 0.5, 0.75):
        B = frac * cost.sum()
        sel = ar.select(policy, B, rows, cost, Z, r0, kernel)
        assert len(set(sel)) == len(sel)
        assert cost[sel].sum() <= B + 1e-9
        # greedy-with-skip: no unselected row still fits
        rest = [i for i in range(len(rows)) if i not in set(sel)]
        assert all(cost[i] + cost[sel].sum() > B + 1e-9 for i in rest), (policy, frac)


def test_greedy_is_deterministic_and_ties_break_by_cost_then_row():
    rows, Z, kernel, cost, r0 = _synthetic()
    a = ar.greedy_residual(Z, r0, kernel, cost, 0.4 * cost.sum())
    b = ar.greedy_residual(Z, r0, kernel, cost, 0.4 * cost.sum())
    assert a[0] == b[0] and a[1] == b[1]
    # identical value & cost everywhere -> the lower row index is picked first
    n, K = 6, 3
    Zt = np.full((n, K), 1.0 / K)
    ct = np.full(n, 10.0)
    sel, _ = ar.greedy_residual(Zt, np.full(K, 0.5), None, ct, 35.0)
    assert sel == [0, 1, 2]
    # identical value, different cost -> cheaper first
    ct2 = np.array([10.0, 5.0, 10.0, 5.0, 10.0, 10.0])
    sel2, _ = ar.greedy_residual(Zt, np.full(K, 0.5), None, ct2, 20.0)
    assert sel2 == [1, 3, 0]


def test_overlap_discount_lowers_a_duplicate():
    """Two rows with identical loadings and cost: one is a fingerprint duplicate of an already
    selected row (kernel 1), the other is orthogonal (kernel 0).  With the overlap term the
    orthogonal one must be picked next; without it the tie falls to the lower row index (the
    duplicate)."""
    K = 3
    Z = np.array([[0.6, 0.2, 0.2],   # row 0: selected first (highest value)
                  [0.2, 0.4, 0.4],   # row 1: duplicate of row 0 in fingerprint space
                  [0.2, 0.4, 0.4]])  # row 2: same loadings, orthogonal fingerprint
    kernel = np.array([[1.0, 1.0, 0.0],
                       [1.0, 1.0, 0.0],
                       [0.0, 0.0, 1.0]])
    cost = np.array([1.0, 1.0, 1.0])
    r0 = np.array([0.9, 0.5, 0.5])
    with_overlap, trace = ar.greedy_residual(Z, r0, kernel, cost, 2.0)
    without, _ = ar.greedy_residual(Z, r0, None, cost, 2.0)
    assert with_overlap[0] == 0 and without[0] == 0
    assert with_overlap[1] == 2, with_overlap
    assert without[1] == 1, without
    # the risk trace is monotone and the recorded reduction matches the replay
    assert trace[0]["risk_after"] < trace[0]["risk_before"]
    rep = ar.replay_residual(Z, r0, with_overlap, 0.25)
    assert rep["risk_after"] == pytest.approx(trace[-1]["risk_after"])
    assert rep["risk_reduction"] > 0


def test_residual_step_and_value_formula():
    r = np.array([0.5, 0.2])
    z = np.array([0.8, 0.2])
    eta = 0.25
    expected = float(sum(rk ** 2 * (1 - (1 - eta * zk) ** 2) for rk, zk in zip(r, z)))
    assert ar.predicted_reduction_step(r, z, eta) == pytest.approx(expected)
    # a row that loads only on atoms with zero residual is worth nothing
    assert ar.predicted_reduction_step(np.array([0.0, 0.7]), np.array([1.0, 0.0]), eta) == 0.0


def test_du_filter_equals_random_when_all_positive_and_filters_otherwise():
    rows, Z, kernel, cost, r0 = _synthetic()
    B = 0.5 * cost.sum()
    assert ar.select("du_positive_filter", B, rows, cost, Z, r0, kernel) == \
        ar.select("random", B, rows, cost, Z, r0, kernel)
    rows2 = [dict(r) for r in rows]
    for i in range(0, len(rows2), 2):
        rows2[i]["_event_dU"] = 0.0
    sel = ar.select("du_positive_filter", B, rows2, cost, Z, r0, kernel)
    assert sel and all(i % 2 == 1 for i in sel)


def test_consequential_first_orders_by_abs_du_then_cost():
    rows, Z, kernel, cost, r0 = _synthetic()
    sel = ar.select("consequential_first", cost.sum(), rows, cost, Z, r0, kernel)
    du = [abs(rows[i]["_event_dU"]) for i in sel]
    assert du == sorted(du, reverse=True)
    for a, b in zip(sel, sel[1:]):
        if rows[a]["_event_dU"] == rows[b]["_event_dU"]:
            assert (cost[a], a) <= (cost[b], b)


def test_first_divergence_puts_low_turns_first():
    rows, Z, kernel, cost, r0 = _synthetic()
    sel = ar.select("first_divergence", cost.sum(), rows, cost, Z, r0, kernel)
    turns = [int(rows[i]["turn_index"]) for i in sel]
    assert turns == sorted(turns)


def test_row_costs_share_sums_to_task_cost():
    rows, *_ = _synthetic(n=6)
    rows[5]["teacher"] = ar.DEMO_TEACHER
    rows[5]["task_id"] = "parallel_80"
    gen = {f"t{k}": {"content_tokens": 100.0 * (k + 1), "copies": 1, "estimate": 130.0 * (k + 1)} for k in range(2)}
    costs = ar.row_costs(rows, gen, {"parallel_80": 465})
    t0 = [c for c in costs if c["task_id"] == "t0"]
    assert len(t0) == 3 and sum(c["share"] for c in t0) == pytest.approx(130.0)
    assert costs[5]["kind"] == "demo_exact" and costs[5]["share"] == 465.0
    assert costs[5]["task_cost_campaign_amortised"] == pytest.approx(1_233_607 / 34)
    with pytest.raises(KeyError):
        ar.row_costs([dict(rows[0], task_id="unknown")], gen, {})


def test_atom_residuals_from_base_values():
    Z = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
    v = np.array([1.0, 0.0, 0.5])
    J0, r0 = ar.atom_residuals(Z, v)
    assert J0 == pytest.approx([(1.0 + 0.25) / 1.5, (0.0 + 0.25) / 1.5])
    assert r0 == pytest.approx(1.0 - J0)


@pytest.mark.skipif(not (ROOT / "results/analysis/acquisition_replay_bfcl.json").is_file(), reason="replay not run")
def test_real_replay_json_budgets_and_pools():
    import json
    d = json.loads((ROOT / "results/analysis/acquisition_replay_bfcl.json").read_text())
    for pol, budgets in d["results"].items():
        for tag, s in budgets.items():
            if tag.startswith("b"):
                assert s["tokens_share"] <= s["budget"] + 1e-6, (pol, tag)
            else:
                assert s["n_rows"] <= s["budget"] + 1e-6, (pol, tag)
    for key, rel in d["pools"].items():
        pol, tag = key.rsplit("_", 1)
        n = sum(1 for l in (ROOT / rel).open() if l.strip())
        assert n == d["results"][pol][tag]["n_rows"], key
