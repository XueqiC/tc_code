"""CPU-only purchase, rendering, and selection oracles."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import random
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/"src"), str(ROOT)]

from bfas.rtd.baselines.paper_data import (EMPTY_THOUGHT, STUDENT, TeacherRow, deployed_row,
    first_thought, frozen_purchase, load_purchased, select_smartad)
from bfas.rtd.bank_build import seal_v11, state_record
from bfas.rtd.persistence import digest
from bfas.rtd.transport import FullState


def row(package="a", task="task", target="ACTION: take apple", index=0, benchmark="alfworld"):
    return TeacherRow(package, task, "parent", index,
                      "<bos><|turn>model\n"+EMPTY_THOUGHT, target, benchmark)


def test_purchase_frozen_order_charges_failure_and_stops_not_knapsack():
    ids = ["a", "b", "c", "d"]
    order = ids.copy()
    random.Random(0).shuffle(order)
    assert order == ["c", "a", "b", "d"]
    # c is a failed attempt, a is positive, b blocks, d would fit but is not examined.
    costs, calls = dict(c=4, a=2, b=5, d=1), []
    def lookup(q):
        calls.append(q)
        return costs[q]
    purchase = frozen_purchase(reversed(ids), lookup, 30, ".25")
    assert purchase["B"] == 8 and purchase["B_exact"] == "7.50"
    assert purchase["purchased_package_ids"] == ["c", "a"]
    assert purchase["teacher_tokens_charged"] == 6
    assert purchase["blocked_next"] == dict(package_id="b", tokens=5)
    assert calls == ["c", "a", "b"]


@pytest.mark.parametrize("method", ["smartad", "sad", "kang", "gad"])
def test_purchase_order_is_independent_of_method(method):
    result = frozen_purchase(["d", "b", "a", "c"], lambda _: 1, 12, .25)
    assert result["purchase_seed"] == 0
    assert result["purchase_order"] == ["c", "a", "b", "d"]
    assert result["teacher_tokens_charged"] == result["B"] == 3


@pytest.mark.parametrize("fraction", [0, -1, 1.1, "NaN", "Infinity"])
def test_reject_bad_budget(fraction):
    with pytest.raises(ValueError):
        frozen_purchase(["a"], lambda _: 1, 10, fraction)


def test_zero_cost_and_exact_cap():
    result = frozen_purchase(["a", "b"], lambda q: 0 if q == "a" else 4, 16, .25)
    assert result["purchased_package_ids"] == ["a", "b"]
    assert result["teacher_tokens_charged"] == 4 and result["blocked_next"] is None
    with pytest.raises(ValueError):
        frozen_purchase(["a", "a"], lambda _: 1, 10, .25)
    with pytest.raises(ValueError):
        frozen_purchase(["a"], lambda _: True, 10, .25)


def make_bank(tmp_path):
    records, payloads = [], {}
    for name, cost, usable in [("good", 20, True), ("bad", 3, False), ("good2", 10, True)]:
        qid = digest(name)
        state = FullState.create({"task_id": name}, [{"role": "user", "content": name}],
                                 "<bos><|turn>model\n"+EMPTY_THOUGHT, "parent")
        records.append(state_record(qid, state, "demo_attempt", cost, "exact",
                                    unavailable=None if usable else "failed"))
        payloads[qid] = dict(cost=cost, cost_confidence="exact",
            historical_response=dict(teacher="openai/gpt-5.6-luna-FC", verified=usable, tokens_spent=cost),
            provenance=dict(task_id=name, rendering_student=STUDENT),
            behaviors=[dict(state=asdict(state), text="answer<turn|>\n")] if usable else [])
    bank = tmp_path/"bank"
    seal_v11(bank, records, payloads, benchmark="bfcl", student=STUDENT, public={}, audit={}, inputs={})
    return bank


def test_real_certificate_and_failed_inventory(tmp_path):
    bank = make_bank(tmp_path)
    before = {p: p.read_bytes() for p in bank.rglob("*.json")}
    purchase, rows = load_purchased(bank, "bfcl", 1)
    expected = frozen_purchase([digest(x) for x in ("good", "bad", "good2")],
        lambda q: json.loads((bank/"sealed"/(q+".json")).read_text())["cost"], 30, 1)
    assert purchase["teacher_tokens_charged"] == expected["teacher_tokens_charged"]
    assert purchase["usable_cost_basis"] == 30  # bad's 3 tokens are outside denominator
    assert all(r.task_id != "bad" for r in rows)
    assert sum(c["tokens"] for c in purchase["charges"]) == purchase["teacher_tokens_charged"]
    assert all(p.read_bytes() == data for p, data in before.items())
    assert all(not r.target.startswith(EMPTY_THOUGHT) and r.prompt.endswith(EMPTY_THOUGHT) for r in rows)
    with pytest.raises(ValueError, match="absolute"):
        load_purchased(Path("relative-bank"), "bfcl", 1)


def test_bank_tamper_rejected(tmp_path):
    bank = make_bank(tmp_path)
    order = sorted(digest(x) for x in ("good", "bad", "good2"))
    random.Random(0).shuffle(order)
    path = bank/"sealed"/(order[0]+".json")
    payload = json.loads(path.read_text())
    payload["cost"] = 0
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="integrity"):
        load_purchased(bank, "bfcl", 1)


def test_deployment_boundary_preserves_bytes():
    source = row()
    deployed = deployed_row(source)
    assert deployed.prompt+deployed.target == source.prompt+source.target
    assert deployed.prompt.endswith(EMPTY_THOUGHT)
    assert deployed == source
    assert deployed_row(deployed) == deployed


def test_smartad_groups_trajectory_by_task_and_normalizes_length():
    rows = [row("a", target="short"), row("b", target="long", index=0),
            row("b", target="long", index=1), row("c", task="another", target="short")]
    def nll(r):
        return (4., 1) if r.target == "short" else (12., 10)
    selected, audit = select_smartad(rows, nll)
    assert [r.package_id for r in selected] == ["b", "b", "c"]
    assert audit["base_student_mean_nll"] == {"a": 4., "b": 1.2, "c": 4.}


def test_kang_prefix_only_first_turn_and_only_owned_content():
    rows = [deployed_row(row(index=i, target=f"take apple {i}")) for i in range(2)]
    prefixed = first_thought(rows)
    assert "THOUGHT:" in prefixed[0].target and "take apple 1" in prefixed[0].target
    assert prefixed[1] == rows[1]
    assert all(a.prompt == b.prompt for a, b in zip(rows, prefixed))


def test_cli_prepare_manifest_has_no_worker_or_model_call(tmp_path, monkeypatch):
    from tools import baseline_run
    bank = make_bank(tmp_path)
    def forbidden(*args, **kwargs):
        pytest.fail("prepare must not start workers or load models")
    monkeypatch.setattr(baseline_run.subprocess, "run", forbidden)
    out = tmp_path/"run"
    assert baseline_run.main(["--method", "gad", "--benchmark", "bfcl", "--bank", str(bank),
                             "--budget-fraction", "1", "--run-dir", str(out), "--prepare-only"]) == 0
    manifest = json.loads((out/"manifest.json").read_text())
    assert manifest["gpu_hours"] == manifest["new_teacher_calls"] == 0
    assert manifest["hyperparameters"]["student_steps"] == 24
    assert manifest["hyperparameters"]["lora_rank"] == 16
    assert manifest["hyperparameters"]["lora_alpha"] == 32
    assert manifest["status"] == "prepared"
    assert not (out/"metrics.json").exists()
