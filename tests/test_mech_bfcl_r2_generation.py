import copy
from types import SimpleNamespace

import pytest

from test_mech_bfcl_generation import generation, proposal, seed
from bfas.mech_bfcl.common import digest, read_json, write_json, append_row
from bfas.mech_bfcl.exercises import correct_call_set, materialize
from bfas.mech_bfcl.generation import coverage, condition_side, frozen_subset, prepare_r2, relation
from bfas.mech_bfcl.teacher import ledger_summary
from tools.mech_bfcl import parser


def bank_row(arm="C", index=0, value="old"):
    return dict(materialize(proposal(value), seed(0)["context"], arm=arm,
                           group=arm+":seed_0", index=index, call_id="old-call"), validation=dict(valid=True))


def sided(value, variant, side="positive"):
    p = proposal(value, variant)
    if side == "negative":
        p["demo"] = dict(kind="abstain", text="Please clarify the requested item.")
    return dict(p, condition_side=side, condition_evidence="The request requires this decision in the supplied state.")


@pytest.mark.parametrize("arm", ["C", "D"])
def test_extension_preserves_frozen_rows_dedupes_decisions_and_charges_repairs(generation, arm):
    source = generation.directory.with_name(generation.directory.name+"-r1") / arm / "exercises.json"
    frozen = [bank_row(arm, i) for i in range(2)]  # Historical duplicate must survive.
    write_json(source, frozen)
    original = source.read_bytes()
    batches = ["malformed", dict(exercises=[sided("old", "surface"), sided("bad", "surface")]),
        dict(exercises=[sided("a", "condition"), sided("b", "neighbour_correct"),
                        sided("c", "condition", "negative"), sided("d", "surface", "negative"),
                        sided("e", "neighbour_correct", "negative")])]
    rows, report, audit = generation.run(arm, target=8, seed_count=1, batches=batches, extend_from=source)
    assert report["ready"] and report["count"] == 8 and report["calls"] == 3
    assert report["output_tokens"] == 1500
    assert rows[:2] == [dict(r, origin="round1", round1_exercise_id=r["id"]) for r in frozen]
    assert all(r["origin"] == "round2" for r in rows[2:])
    assert source.read_bytes() == original
    assert report["round1_exercise_ids"] == [r["id"] for r in frozen]
    assert report["decision_coverage"]["seed_0"]["complete"]
    assert any(r.get("validation", {}).get("reason") == "Duplicate accepted exercise" for r in audit)
    assert generation.run(arm, target=8, seed_count=1, extend_from=source) == (rows, report, audit)
    assert len(generation.calls[arm]) == 3
    assert read_json(source) == frozen
    frozen[0]["messages"][-1]["content"] = "changed source"
    write_json(source, frozen)
    with pytest.raises(ValueError, match="Frozen round-1 exercise subset changed"):
        generation.run(arm, target=8, seed_count=1, extend_from=source)


def test_surface_rewording_same_calls_rejected_and_cap_keeps_one_sided_pool_unready(generation):
    source = generation.directory.with_name(generation.directory.name+"-r1") / "C/exercises.json"
    write_json(source, [bank_row()])
    repeated = dict(sided("old", "surface"), user="Completely different wording")
    rows, report, audit = generation.run(target=64, cap=3500, seed_count=1, extend_from=source,
                                        batches=[dict(exercises=[repeated]), dict(exercises=[repeated])])
    assert len(rows) == 1 and not report["ready"]
    assert report["stop_reason"] == "output_token_cap" and report["output_tokens"] == 1000
    assert len([r for r in audit if r.get("validation", {}).get("reason") ==
                "Duplicate correct behaviour for seed/direction"]) == 2


def test_normalized_call_set_ignores_order_duplicates_and_no_call_prose():
    row = bank_row()
    other = copy.deepcopy(row)
    other["demo"]["calls"].append(dict(name="f", arguments=dict(b=[1, 2], a=2)))
    row["demo"]["calls"] = copy.deepcopy(list(reversed(other["demo"]["calls"])) * 2)
    assert correct_call_set(row) == correct_call_set(other)
    other["demo"]["calls"][1]["arguments"]["b"].reverse()
    assert correct_call_set(row) != correct_call_set(other)
    assert correct_call_set(dict(demo=dict(kind="abstain", text="one"))) == correct_call_set(
        dict(demo=dict(kind="abstain", text="two")))


@pytest.mark.parametrize("kind,positive,negative", [
    ("archival", [dict(name="archival_memory_retrieve", arguments={})], [dict(name="core_memory_retrieve", arguments={})]),
    ("parallel", [dict(name="f", arguments={}), dict(name="g", arguments={})], [dict(name="f", arguments={})]),
    ("prerequisite", [dict(name="lockDoors", arguments=dict(unlock=False))], [dict(name="lockDoors", arguments=dict(unlock=True))]),
    ("action", [dict(name="f", arguments={})], []),
])
def test_two_sides_are_derived_from_calls_not_teacher_labels(kind, positive, negative):
    for calls, side in ((positive, "positive"), (negative, "negative")):
        row = dict(demo=dict(kind="call", calls=calls) if calls else dict(kind="abstain", text="done"))
        assert condition_side(row, dict(kind=kind)) == side


@pytest.mark.parametrize("change", ["arm", "source_hash", "task_id", "validation", "target"])
def test_extension_rejects_wrong_source_or_dropped_frozen_rows(tmp_path, change):
    row = bank_row()
    if change == "validation":
        row["validation"]["valid"] = False
    elif change != "target":
        row[change] = "wrong"
    source = tmp_path.parent / (tmp_path.name+"-source.json")
    write_json(source, [row])
    args = SimpleNamespace(run_dir=tmp_path, arm="C", extend_from=source, target_exercises=0 if change == "target" else 64)
    with pytest.raises(ValueError):
        frozen_subset(args, [seed(0)])


def test_prepare_copies_inputs_and_only_shared_cost_without_mutating_r1(tmp_path):
    source, destination = tmp_path / "r1", tmp_path / "r2"
    splits = dict(seed=0)
    write_json(source / "protocol.json", dict(split_hash=digest(splits)))
    for name in ("seeds", "diagnoses", "heldout", "heldout_audit"):
        write_json(source / (name+".json"), [])
    for folder in ("support", "calibration"):
        write_json(source / folder / "trajectories/a.json", {})
    for arm in ("C", "D"):
        write_json(source / arm / "exercises.json", [bank_row(arm)])
    for bucket in ("shared", "C", "D"):
        append_row(source / "teacher_ledger.jsonl", dict(call_id=bucket, bucket=bucket,
            event="final", usage=dict(prompt_tokens=3, completion_tokens=5, total_tokens=8)))
        write_json(source / "teacher_raw" / (bucket+".response.json"), dict(usage="exact"))
    write_json(source / "training_plan.json", dict(frozen="R1"))
    before = {p: p.read_bytes() for p in source.rglob("*") if p.is_file()}
    args = SimpleNamespace(round1_run_dir=source, run_dir=destination)
    prepare_r2(args, splits)
    prepare_r2(args, splits)
    assert all(p.read_bytes() == value for p, value in before.items())
    assert not (destination / "training_plan.json").exists()
    assert not (destination / "C/exercises.json").exists()
    assert ledger_summary(destination / "teacher_ledger.jsonl")["shared"]["output_tokens"] == 5
    assert ledger_summary(destination / "teacher_ledger.jsonl")["C"]["calls"] == 0
    assert (destination / "teacher_raw/shared.response.json").read_bytes() == (source / "teacher_raw/shared.response.json").read_bytes()
    write_json(destination / "heldout.json", ["changed"])
    with pytest.raises(ValueError, match="Prepared R1 input changed"):
        prepare_r2(args, splits)


def test_r2_cli():
    args = parser().parse_args(["generate", "--arm", "D", "--target", "64", "--extend-from", "/tmp/r1.json"])
    assert args.target_exercises == 64 and str(args.extend_from) == "/tmp/r1.json"
    assert parser().parse_args(["confirm"]).target == 24
    assert parser().parse_args(["evaluate", "--arm", "D", "--checkpoint", "mid"]).checkpoint == "mid"
    assert parser().parse_args(["train", "--arm", "C", "--supervised-budget", "15900"]).supervised_budget == 15900
