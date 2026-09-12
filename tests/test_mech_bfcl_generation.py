"""Exercise the complete generation stage with CPU-only teacher/student stubs."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import appworld_teacher as shared
from bfas.mech_bfcl import pipeline
from bfas.mech_bfcl.common import bind_run, read_json, write_json
from bfas.mech_bfcl.exercises import exercise_fingerprint, materialize
from bfas.mech_bfcl.teacher import Teacher, ledger_summary
from tools.mech_bfcl import parser


def seed(index):
    context = dict(task_id=f"task_{index}", parent_id=f"parent_{index}",
        category="simple_python", frame_id=f"task_{index}:0",
        messages=[dict(role="user", content="original")],
        functions=[dict(name="lookup")], snapshot={}, snapshot_error=None, involved_classes=[])
    return dict(seed_id=f"seed_{index}", context=context, failure_frame=dict(response="wrong"))


def proposal(value, variant="surface", *, user=None):
    return dict(user=user or f"Find item {value}", variant=variant,
        demo=dict(kind="call", calls=[dict(name="lookup", arguments=dict(color=value))]))


@pytest.fixture
def generation(monkeypatch, tmp_path):
    config = shared.TeacherConfig(name="openai/gpt-5.6-luna", model="gpt-5.6-luna",
        endpoint="https://api.openai.com/v1/chat/completions", api_key="stub-key",
        backend="openai_api", service_tier="flex")
    calls, probes = {"C": [], "D": []}, []
    seeds = [seed(i) for i in range(5)]
    responses = lambda index: dict(exercises=[
        proposal(f"{index}_{j}", ("condition", "surface", "neighbour_correct")[j % 3])
        for j in range(4)])

    def transport(config, messages, **kwargs):
        assert kwargs["retries"] == 0
        assert kwargs["max_completion_tokens"] == pipeline.GENERATION_CALL_TOKENS
        payload = json.loads(messages[1]["content"])
        arm = "C" if payload["student_gap"] is None else "D"
        value = responses(len(calls[arm]))
        text = value if isinstance(value, str) else json.dumps(value)
        calls[arm].append(copy.deepcopy(messages))
        kwargs["response_callback"](dict(http_status=200, data=dict(
            model="gpt-5.6-luna-stub", service_tier="flex",
            usage=dict(prompt_tokens=100, completion_tokens=500, total_tokens=600),
            choices=[dict(finish_reason="stop", message=dict(content=text))])))
        return text

    def student_reply(args, row, tokenizer):
        probes.append(row["id"])
        return ("wrong" if row["messages"][-1]["content"] == "failed neighbour" else "right"), {}

    validator = SimpleNamespace(
        validate=lambda row: dict(valid=row["messages"][-1]["content"] != "invalid"),
        score=lambda row, response: dict(correct=response == "right"))
    monkeypatch.setattr(pipeline, "training_seeds", lambda *args: seeds)
    monkeypatch.setattr(pipeline, "Teacher", lambda directory: Teacher(directory, config, transport))
    monkeypatch.setattr(pipeline, "OfficialValidator", lambda: validator)
    monkeypatch.setattr(pipeline, "check_server", lambda *args: None)
    monkeypatch.setattr(pipeline, "heldout", lambda *args: [])
    monkeypatch.setattr(pipeline, "native_pair", lambda *args: ("prompt", "target"))
    monkeypatch.setattr(pipeline, "student_reply", student_reply)
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *args, **kwargs: None)))

    def run(arm="C", target=64, cap=24000, *, batches=None, seed_count=5):
        nonlocal responses
        if batches is not None:
            responses = lambda index: batches[index]
        seeds[:] = [seed(i) for i in range(seed_count)]
        args = SimpleNamespace(run_dir=tmp_path, arm=arm, target_exercises=target,
                               max_output_tokens=cap, tokenizer="stub")
        splits = dict(seed=0)
        bind_run(tmp_path, splits)
        write_json(tmp_path / "diagnoses.json", [dict(seed_id=s["seed_id"], valid=True,
                   diagnosis=dict(hypothesis="private gap")) for s in seeds])
        pipeline.generate(args, splits)
        return (read_json(tmp_path / arm / "exercises.json"),
                read_json(tmp_path / arm / "generation.json"),
                read_json(tmp_path / arm / "generation_audit.json"))

    return SimpleNamespace(run=run, directory=tmp_path, calls=calls, probes=probes)


@pytest.mark.parametrize("target,cap", [(64, 24000), (7, 30000)])
def test_target_reached_with_symmetric_rotation_and_resume(generation, target, cap):
    banks = {}
    for arm in ("C", "D"):
        rows, report, audit = generation.run(arm, target, cap)
        banks[arm] = rows
        expected_calls = (target + 3) // 4
        assert len(rows) == target
        assert len({r["id"] for r in rows}) == target
        assert report["ready"] and not report["below_target"]
        assert report["stop_reason"] == "target_reached"
        assert report["calls"] == expected_calls
        assert report["output_tokens"] == 500 * expected_calls
        assert {r["variant"] for r in rows} == pipeline.VARIANTS
        assert report["condition_action_changes"]
        if target % 4:
            assert audit[-1]["unused_proposals"] == 4 - target % 4
        replay = generation.run(arm, target, cap)
        assert replay == (rows, report, audit)
        assert len(generation.calls[arm]) == expected_calls
    assert read_json(generation.directory / "protocol.json")["budgets"] == dict(shared=8000, C=cap, D=cap)
    assert [exercise_fingerprint(r) for r in banks["C"]] == [exercise_fingerprint(r) for r in banks["D"]]
    diversity = []
    for index, (c, d) in enumerate(zip(generation.calls["C"], generation.calls["D"])):
        assert c[0] == d[0]
        cp, dp = (json.loads(messages[1]["content"]) for messages in (c, d))
        assert cp.pop("student_gap") is None
        assert dp.pop("student_gap")["hypothesis"] == dict(hypothesis="private gap")
        assert cp == dp
        assert cp["context"]["task_id"] == f"task_{index % 5}"
        if index >= 5:
            assert cp["accepted_exercises"]
        diversity.append(cp["diversity_instruction"])
    assert len(set(diversity)) == len(diversity)


def test_each_seed_gets_new_diversity_focus_when_rotation_has_four_seeds(generation):
    generation.run(seed_count=4)
    by_seed = {}
    for messages in generation.calls["C"]:
        payload = json.loads(messages[1]["content"])
        # Ignore the batch number: the substantive instruction must also change.
        instruction = payload["diversity_instruction"].split(": ", 1)[1]
        by_seed.setdefault(payload["context"]["task_id"], []).append(instruction)
    assert len(by_seed) == 4
    assert all(len(instructions) == len(set(instructions)) == 4 for instructions in by_seed.values())


@pytest.mark.parametrize("arm", ["C", "D"])
@pytest.mark.parametrize("cap,expected_calls", [(2999, 0), (4000, 3)])
def test_cap_stops_before_full_reservation_and_replays(generation, arm, cap, expected_calls):
    # Repeated valid output cannot fill the target; every duplicate is charged.
    batches = [dict(exercises=[proposal("blue")])] * expected_calls
    rows, report, audit = generation.run(arm, cap=cap, batches=batches, seed_count=1)
    assert len(rows) == bool(expected_calls)
    assert report["stop_reason"] == "output_token_cap"
    assert report["below_target"] and not report["ready"]
    assert report["calls"] == expected_calls
    assert report["output_tokens"] == 500 * expected_calls
    assert report["output_tokens"] + report["call_output_tokens"] > cap
    assert len(generation.calls[arm]) == expected_calls
    assert generation.run(arm, cap=cap, seed_count=1) == (rows, report, audit)
    assert len(generation.calls[arm]) == expected_calls


@pytest.mark.parametrize("arm", ["C", "D"])
def test_rejections_and_duplicates_are_charged_before_target(generation, arm):
    blue = proposal("blue")
    batches = ["[]", dict(exercises="bad list"),
        dict(exercises=[proposal("invalid", user="invalid")]),
        dict(exercises=[blue, dict(blue, variant="condition")]),
        dict(exercises=[dict(blue, variant="neighbour_correct"), proposal("green", "condition")]),
        dict(exercises=[proposal("red", "neighbour_correct"), proposal("unused")])]
    rows, report, audit = generation.run(arm, target=3, batches=batches, seed_count=1)
    assert len(rows) == len({exercise_fingerprint(r) for r in rows}) == 3
    assert report["calls"] == 6 and report["output_tokens"] == 3000
    assert report["ready"] and report["stop_reason"] == "target_reached"
    duplicates = [r for r in audit if r.get("validation", {}).get("reason") == "Duplicate accepted exercise"]
    assert len(duplicates) == 2
    assert all(r["validation"]["duplicate_of"] == rows[0]["id"] for r in duplicates)
    assert len(generation.probes) == 1  # A duplicate neighbour needs no student probe.
    assert audit[-1]["unused_proposals"] == 1
    assert all(r["validation"]["valid"] for r in rows)
    assert ledger_summary(generation.directory / "teacher_ledger.jsonl")[arm]["output_tokens"] == 3000


@pytest.mark.parametrize("arm", ["C", "D"])
def test_variant_coverage_accumulates_across_rounds_with_neighbour_check(generation, arm):
    batches = [dict(exercises=[proposal("blue"), proposal("green", "condition")]),
        dict(exercises=[proposal("red", "neighbour_correct", user="failed neighbour")]),
        dict(exercises=[proposal("red", "neighbour_correct")])]
    rows, report, audit = generation.run(arm, target=3, cap=4000, batches=batches, seed_count=1)
    assert report["ready"] and report["calls"] == 3
    assert report["coverage"]["seed_0"] == dict(condition=1, surface=1, neighbour_correct=1)
    assert report["missing_variant_groups"] == []
    assert rows[-1]["base_neighbour_check"]["correct"]
    assert any(r.get("validation", {}).get("reason") == "Base was not right on neighbour candidate" for r in audit)


@pytest.mark.parametrize("condition_changes", [False, True])
def test_d_readiness_requires_changed_condition_action(generation, condition_changes):
    batches = [dict(exercises=[proposal("blue"),
        proposal("green" if condition_changes else "blue", "condition", user="changed condition"),
        proposal("red", "neighbour_correct")])]
    _, report, _ = generation.run("D", target=3, batches=batches, seed_count=1)
    assert report["ready"] == condition_changes
    assert report["condition_action_changes"] == condition_changes


@pytest.mark.parametrize("change", [dict(target=65), dict(cap=23000), dict(seed_count=4)])
def test_c_and_d_cannot_change_frozen_loop_settings(generation, change):
    generation.run("C")
    with pytest.raises(ValueError, match="new --run-dir"):
        generation.run("D", **change)
    assert not generation.calls["D"]


def test_legacy_generation_requires_new_run_directory(generation):
    legacy = generation.directory / "C" / "generation.json"
    write_json(legacy, dict(count=22))
    before = legacy.read_bytes()
    with pytest.raises(ValueError, match="old protocol; use a new --run-dir"):
        generation.run("C")
    assert legacy.read_bytes() == before
    assert not (generation.directory / "generation_protocol.json").exists()
    assert not generation.calls["C"]


def test_fingerprint_ignores_labels_but_distinguishes_executor_state():
    c = materialize(proposal("blue"), seed(0)["context"], arm="C", group="c", index=0, call_id="one")
    d = dict(c, id="new", arm="D", cost_call_id="two", variant="condition", source_frame="other")
    assert exercise_fingerprint(c) == exercise_fingerprint(d)
    d["snapshot"] = dict(lookup=dict(available=False))
    assert exercise_fingerprint(c) != exercise_fingerprint(d)


def test_generation_cli_defaults_and_new_run_directory(tmp_path):
    args = parser().parse_args(["generate", "--arm", "C", "--run-dir", str(tmp_path)])
    assert args.run_dir == tmp_path
    assert args.target_exercises == 64 and args.max_output_tokens == 24000
    args = parser().parse_args(["generate", "--arm", "D", "--target-exercises", "96", "--max-output-tokens", "30000"])
    assert args.target_exercises == 96 and args.max_output_tokens == 30000
    for flag in ("--target-exercises", "--max-output-tokens"):
        with pytest.raises(SystemExit):
            parser().parse_args(["generate", "--arm", "C", flag, "0"])
