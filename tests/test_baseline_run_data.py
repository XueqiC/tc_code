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


@pytest.mark.parametrize("cap,ids,spent,blocked", [
    (0, [], 0, dict(package_id="c", tokens=4)),
    (4, ["c"], 4, dict(package_id="a", tokens=2)),
    (6, ["c", "a"], 6, dict(package_id="b", tokens=5)),
    (10, ["c", "a"], 6, dict(package_id="b", tokens=5)),
    (11, ["c", "a", "b"], 11, dict(package_id="d", tokens=1)),
    (12, ["c", "a", "b", "d"], 12, None),
    (20, ["c", "a", "b", "d"], 12, None),
])
def test_absolute_cap_is_frozen_prefix_independent_of_bank_basis(cap, ids, spent, blocked):
    costs = dict(c=4, a=2, b=5, d=1)
    for basis in (None, 0, 1, 100_000):
        calls = []
        def lookup(q):
            calls.append(q)
            return costs[q]
        result = frozen_purchase(reversed(costs), lookup, basis, budget_tokens=cap)
        assert result["B"] == result["budget_tokens"] == cap
        assert result["B_exact"] == str(cap) and result["rounding"] == "none"
        assert result["budget_fraction"] is None
        assert result["purchase_order"] == ["c", "a", "b", "d"]
        assert result["purchased_package_ids"] == ids
        assert result["teacher_tokens_charged"] == spent <= cap
        assert result["remaining_tokens"] == cap - spent
        assert result["blocked_next"] == blocked
        assert calls == ids + ([blocked["package_id"]] if blocked else [])


def test_absolute_zero_cap_can_purchase_zero_cost_package():
    result = frozen_purchase(["a", "b"], lambda q: 0 if q == "a" else 1, budget_tokens=0)
    assert result["purchased_package_ids"] == ["a"]
    assert result["teacher_tokens_charged"] == 0
    assert result["blocked_next"] == dict(package_id="b", tokens=1)


@pytest.mark.parametrize("cap", [-1, 1.5, "10", True])
def test_reject_noninteger_or_negative_absolute_cap(cap):
    with pytest.raises(ValueError, match="nonnegative integer"):
        frozen_purchase(["a"], lambda _: 1, budget_tokens=cap)


@pytest.mark.parametrize("kwargs", [{}, dict(fraction=".25", budget_tokens=10)])
def test_purchase_requires_exactly_one_budget(kwargs):
    with pytest.raises(ValueError, match="exactly one"):
        frozen_purchase(["a"], lambda _: 1, 100, **kwargs)


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


def test_absolute_cap_charges_failed_inventory_above_usable_basis(tmp_path):
    bank = make_bank(tmp_path)
    before = {p: p.read_bytes() for p in bank.rglob("*.json")}
    purchase, rows = load_purchased(bank, "bfcl", budget_tokens=33)
    assert purchase["B"] == purchase["teacher_tokens_charged"] == 33
    assert purchase["usable_cost_basis"] == 30
    assert set(purchase["purchased_package_ids"]) == {digest(x) for x in ("good", "bad", "good2")}
    failed = next(c for c in purchase["charges"] if c["package_id"] == digest("bad"))
    assert failed["tokens"] == 3 and failed["usable"] is False
    assert failed["unavailable_reason"] == "failed"
    assert {r.task_id for r in rows} == {"good", "good2"}
    assert sum(c["tokens"] for c in purchase["charges"]) == 33
    assert all(p.read_bytes() == data for p, data in before.items())


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


def test_smoke_cli_pins_training_and_evaluation_caps(tmp_path, monkeypatch):
    from tools import baseline_run
    bank = make_bank(tmp_path)
    def forbidden(*a, **kw):
        pytest.fail("smoke prepare must not load models or call services")
    monkeypatch.setattr(baseline_run.subprocess, "run", forbidden)
    out = tmp_path/"smoke"
    assert baseline_run.main(["--method", "smartad", "--benchmark", "bfcl", "--bank", str(bank),
        "--budget-tokens", "33", "--run-dir", str(out), "--smoke", "--prepare-only"]) == 0
    manifest = json.loads((out/"manifest.json").read_text())
    assert manifest["smoke"] and manifest["config"]["smoke"]
    assert manifest["hyperparameters"]["student_steps"] == 2
    assert manifest["evaluation_protocol"]["tasks"] == 3
    assert manifest["evaluation_protocol"]["official_full"] is False
    assert manifest["config"]["training_device"] == "cuda:0"
    assert manifest["config"]["score_position_chunk_size"] == 32
    assert 1 <= manifest["config"]["cpu_threads"] <= 4


@pytest.mark.parametrize("budget", [[], ["--budget-tokens", "10", "--budget-fraction", ".25"]])
def test_cli_requires_exactly_one_budget(tmp_path, budget):
    from tools import baseline_run
    with pytest.raises(SystemExit) as error:
        baseline_run.arguments(["--method", "sad", "--benchmark", "bfcl", "--bank", str(tmp_path),
                                "--run-dir", str(tmp_path/"run"), *budget])
    assert error.value.code == 2


@pytest.mark.parametrize("value", ["", "1.5", "-1", "10,", ",10", "10,,20", "10,no", "10,10"])
def test_cli_rejects_invalid_token_levels(tmp_path, value):
    from tools import baseline_run
    with pytest.raises(SystemExit) as error:
        baseline_run.arguments(["--method", "sad", "--benchmark", "bfcl", "--bank", str(tmp_path),
                                "--run-dir", str(tmp_path/"run"), "--budget-tokens", value])
    assert error.value.code == 2


@pytest.mark.parametrize("levels", [(0,), (33,), (0, 15, 30, 33), (33, 15)])
def test_cli_absolute_prepare_layout_and_manifests(tmp_path, monkeypatch, capsys, levels):
    from tools import baseline_run
    bank = make_bank(tmp_path)
    before = {p: p.read_bytes() for p in bank.rglob("*.json")}
    def forbidden(*args, **kwargs):
        pytest.fail("prepare must not start workers or load models")
    monkeypatch.setattr(baseline_run.subprocess, "run", forbidden)
    prefix = tmp_path/"results"/"curve"
    assert baseline_run.main(["--method", "gad", "--benchmark", "bfcl", "--bank", str(bank),
        "--budget-tokens", ",".join(map(str, levels)), "--run-dir", str(prefix), "--prepare-only"]) == 0
    summaries = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    expected_dirs = [prefix.with_name(f"curve_B{cap}") if len(levels) > 1 else prefix for cap in levels]
    assert set(prefix.parent.iterdir()) == set(expected_dirs)
    assert [s["run_dir"] for s in summaries] == list(map(str, expected_dirs))
    assert [s["B"] for s in summaries] == list(levels)
    manifests = []
    for cap, directory, summary in zip(levels, expected_dirs, summaries):
        manifest = json.loads((directory/"manifest.json").read_text())
        rows = json.loads((directory/"purchased_rows.json").read_text())
        expected, expected_rows = load_purchased(bank, "bfcl", budget_tokens=cap)
        assert manifest["B"] == manifest["budget_tokens"] == cap
        assert manifest["budget_fraction"] is None
        assert manifest["teacher_tokens_charged"] == summary["teacher_tokens_charged"] <= cap
        assert manifest["purchased_package_ids"] == expected["purchased_package_ids"]
        assert manifest["charges"] == expected["charges"]
        assert manifest["teacher_tokens_charged"] == sum(c["tokens"] for c in manifest["charges"])
        assert rows == [asdict(r) for r in expected_rows]
        assert manifest["rows_hash"] == digest(rows)
        assert manifest["gpu_hours"] == manifest["new_teacher_calls"] == 0
        assert manifest["status"] == "prepared"
        assert set(p.name for p in directory.iterdir()) == {"manifest.json", "purchased_rows.json"}
        manifests.append(manifest)
    for manifest in manifests[1:]:
        for key in ("purchase_order", "purchase_order_hash", "config", "hyperparameters", "evaluation_protocol"):
            assert manifest[key] == manifests[0][key]
    ordered = sorted(manifests, key=lambda m: m["B"])
    for lower, higher in zip(ordered, ordered[1:]):
        ids = lower["purchased_package_ids"]
        assert higher["purchased_package_ids"][:len(ids)] == ids
    assert all(p.read_bytes() == data for p, data in before.items())


def test_cli_curve_refuses_existing_later_level_before_any_work(tmp_path, monkeypatch):
    from tools import baseline_run
    bank = make_bank(tmp_path)
    prefix = tmp_path/"curve"
    occupied = tmp_path/"curve_B33"
    occupied.mkdir()
    marker = occupied/"manifest.json"
    marker.write_text("existing run")
    def forbidden(*args, **kwargs):
        pytest.fail("directory collisions must be found before starting workers")
    monkeypatch.setattr(baseline_run.subprocess, "run", forbidden)
    with pytest.raises(FileExistsError, match="fresh"):
        baseline_run.main(["--method", "sad", "--benchmark", "bfcl", "--bank", str(bank),
            "--budget-tokens", "15,33", "--run-dir", str(prefix)])
    assert not (tmp_path/"curve_B15").exists()
    assert marker.read_text() == "existing run"


@pytest.mark.parametrize("budget,levels", [
    (["--budget-tokens", "33,30"], (33, 30)),
    (["--budget-tokens", "33"], (33,)),
    (["--budget-fraction", "1"], (30,)),
])
def test_cli_run_dispatches_each_budget_to_both_workers_on_cpu(tmp_path, monkeypatch, budget, levels):
    from tools import baseline_run
    from bfas.rtd.baselines import paper_evaluation
    bank = make_bank(tmp_path)
    prefix = tmp_path/"curve"
    calls = []
    # Exercise the real worker CLI/manifest dispatch with CPU stubs for model work.
    def record_worker(phase, directory, manifest):
        calls.append((phase, directory, manifest["B"]))
    monkeypatch.setattr(baseline_run, "train_worker",
                        lambda directory, manifest: record_worker("train", directory, manifest))
    monkeypatch.setattr(paper_evaluation, "evaluate_run",
                        lambda root, directory, manifest: record_worker("evaluate", directory, manifest))
    def fake_subprocess(command, log):
        worker = baseline_run.arguments(command[2:])
        assert worker._phase in ("train", "evaluate")
        if budget[0] == "--budget-tokens":
            assert worker.budget_tokens == [levels[len(calls)//2]]
            assert worker.budget_fraction is None
        else:
            assert worker.budget_fraction == "1" and worker.budget_tokens is None
        assert baseline_run.main(command[2:]) == 0
    monkeypatch.setattr(baseline_run, "run_worker", fake_subprocess)
    assert baseline_run.main(["--method", "sad", "--benchmark", "bfcl", "--bank", str(bank),
                             "--run-dir", str(prefix), *budget]) == 0
    directories = [prefix.with_name(f"curve_B{cap}") if len(levels) > 1 else prefix for cap in levels]
    assert calls == [(phase, directory, cap) for cap, directory in zip(levels, directories)
                     for phase in ("train", "evaluate")]
    for directory in directories:
        manifest = json.loads((directory/"manifest.json").read_text())
        assert manifest["status"] == "complete"
        assert manifest["gpu_hours"] == 0
        assert (directory/"train.log").exists() and (directory/"evaluate.log").exists()
