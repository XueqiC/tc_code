import copy
import io
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import appworld_teacher as shared
from bfas.mech_bfcl import confirmation, pipeline, stats
from bfas.mech_bfcl.common import append_row, digest, read_json, write_json
from bfas.mech_bfcl.teacher import Teacher, ledger_summary


@pytest.fixture
def confirmation_run(monkeypatch, tmp_path):
    splits = dict(seed=0, support=["support"], calibration=["leak", "b", "c", "d"], evaluation=["full"],
        items={tid: dict(parent_id=parent, category="simple_python", source_hash=digest(dict(id=tid)))
               for tid, parent in [("support", "diagnosed"), ("leak", "diagnosed"), ("b", "pb"), ("c", "pb"),
                                   ("d", "pd"), ("full", "pf")]})
    write_json(tmp_path / "seeds.json", [dict(context=dict(task_id="support", parent_id="diagnosed"))])
    heldout = [dict(id=f"held-{layer}", task_id="b", parent_id="pb", category="simple_python", layer=layer,
                    generation_group="heldout") for layer in (1, 2)]
    write_json(tmp_path / "heldout.json", heldout)
    captured = [dict(task_id=tid, frame_id=tid+":0", messages=[dict(role="user", content="original")],
                    functions=[dict(name="lookup")], snapshot={}, snapshot_error=None, involved_classes=[])
                for tid in splits["calibration"]]
    monkeypatch.setattr(confirmation, "frames", lambda directory: copy.deepcopy(captured))
    monkeypatch.setattr(confirmation, "native_pair", lambda *args: ("prompt", "target"))
    calls, values = [], {}

    def proposed(user, side):
        return dict(user=user, demo=dict(kind="call", calls=[dict(name="lookup", arguments=dict(x=1))])
                    if side == "positive" else dict(kind="abstain", text="Please specify an item."),
                    condition_side=side, condition_evidence="Supported by the request.")

    default = dict(anchor=proposed("Find item one", "positive"),
                   condition=proposed("Which item should I select?", "negative"),
                   surface=proposed("Look up the first item", "positive"))
    config = shared.TeacherConfig(name="openai/gpt-5.6-luna", model="gpt-5.6-luna",
        endpoint="https://api.openai.com/v1/chat/completions", api_key="stub", backend="openai_api", service_tier="flex")

    def transport(config, messages, **kwargs):
        assert (tmp_path / "confirmation_plan.json").exists()  # Pre-register before purchase.
        payload = json.loads(messages[1]["content"])
        tid = payload["context"]["task_id"]
        assert payload["context"]["parent_id"] != "diagnosed"
        calls.append((tid, kwargs["max_completion_tokens"]))
        text = json.dumps(values.get(tid, default))
        kwargs["response_callback"](dict(http_status=200, data=dict(model="stub-luna", service_tier="flex",
            usage=dict(prompt_tokens=100, completion_tokens=200, total_tokens=300),
            choices=[dict(finish_reason="stop", message=dict(content=text))])))
        return text

    append_row(tmp_path / "teacher_ledger.jsonl", dict(call_id="r1", bucket="shared", event="final",
               usage=dict(prompt_tokens=100, completion_tokens=7000, total_tokens=7100)))
    teacher = Teacher(tmp_path, config, transport)
    validator = SimpleNamespace(validate=lambda row: dict(valid=row["messages"][-1]["content"] != "invalid"),
                                score=lambda row, response: dict(correct=response == "right"))
    args = SimpleNamespace(run_dir=tmp_path, target=6, seed=0, tokenizer="stub", base_url="http://stub/v1",
                           served_model="base", arm="base", repeat="main", train_seed=0, checkpoint="end")
    def build():
        return confirmation.build(args, splits, teacher, validator, None)
    return SimpleNamespace(args=args, splits=splits, calls=calls, values=values, default=default,
                           validator=validator, build=build, heldout=heldout, teacher=teacher)


def test_confirmation_preregistered_independent_parents_shared_cost_and_reuse(confirmation_run):
    b = confirmation_run
    path = b.args.run_dir / "heldout.json"
    original = path.read_bytes()
    result = b.build()
    assert len(result["items"]) == 6 and len(result["pairs"]) == 4
    assert all(p["valid"] for p in result["pairs"])
    assert len({r["parent_id"] for r in result["items"]}) == 2
    assert all(r["parent_id"] != "diagnosed" for r in result["items"])
    assert b.calls == [(tid, 500) for tid, _ in b.calls] and len(b.calls) == 2
    assert ledger_summary(b.teacher.path)["shared"]["output_tokens"] == 7400
    assert path.read_bytes() == original
    assert b.build() == result and len(b.calls) == 2
    b.args.target = 7
    with pytest.raises(ValueError, match="preregistration changed"):
        b.build()


def test_confirmation_invalid_pairs_become_unpaired_items_without_replacement(confirmation_run):
    b = confirmation_run
    for tid in ("b", "c", "d"):
        value = copy.deepcopy(b.default)
        value["condition"] = None
        value["surface"]["demo"]["calls"][0]["arguments"]["x"] = 2  # Invalid invariant relation.
        b.values[tid] = value
    result = b.build()
    assert result["count"] == 4 and result["below_target"]
    assert not any(p["valid"] for p in result["pairs"])
    scored = [dict(r, correct=True) for r in result["items"]]
    metrics = confirmation.metrics(scored, result)
    assert metrics["valid_pairs"] == 0 and metrics["both_sides_correct_rate"] is None
    assert len(metrics["unpaired_items"]) == 4
    assert len(b.calls) == 2  # No replacement or uncharged repair call.


@pytest.mark.parametrize("change", ["invalid", "cue", "wrong_side"])
def test_confirmation_uses_exercise_validator_and_rejects_cues_and_fake_sides(confirmation_run, change):
    b = confirmation_run
    for tid in ("b", "c", "d"):
        value = copy.deepcopy(b.default)
        value["anchor"]["user"] = "invalid" if change == "invalid" else "This is a drill" if change == "cue" else "normal"
        if change == "wrong_side":
            value["anchor"]["condition_side"] = "negative"
        b.values[tid] = value
    result = b.build()
    assert len(result["items"]) == 4
    assert all(r["confirmation_role"] != "anchor" for r in result["items"])
    assert not any(p["valid"] for p in result["pairs"])
    assert ledger_summary(b.teacher.path)["shared"]["output_tokens"] == 7400


def test_confirmation_checks_all_training_sources_again_at_evaluation(confirmation_run):
    b = confirmation_run
    result = b.build()
    leaked = result["items"][0]
    write_json(b.args.run_dir / "D/exercises.json", [leaked])
    with pytest.raises(ValueError, match="source isolation"):
        confirmation.validate_set(result, b.args.run_dir, b.splits)


@pytest.mark.parametrize("bad", [False, "null", "missing", "http"])
def test_probability_is_teacher_forced_target_only_with_explicit_unavailable(monkeypatch, bad):
    monkeypatch.setattr(confirmation, "native_pair", lambda *a: ("prompt", "target"))
    tokenizer = SimpleNamespace(encode=lambda s, **kw: [1, 2] if s == "prompt" else [3, 4, 5])
    calls = []
    def urlopen(request, **kwargs):
        body = json.loads(request.data)
        calls.append(body)
        assert body["prompt"] == [1, 2, 3, 4, 5] and body["echo"] and body["logprobs"] == 1
        if bad == "http":
            raise OSError("logprobs unsupported")
        raw = dict(choices=[dict(logprobs=dict(token_logprobs=[None, -9, -1, -2, None if bad == "null" else -3, -8]))])
        return io.StringIO(json.dumps({} if bad == "missing" else raw))
    monkeypatch.setattr(confirmation.urllib.request, "urlopen", urlopen)
    args = SimpleNamespace(served_model="mech-D-mid", seed=0, base_url="http://stub/v1")
    result = confirmation.target_probability(args, dict(demo=dict(calls=[{}])), tokenizer)
    assert result["available"] == (not bad)
    assert result["diagnostic_only"]
    if not bad:
        assert result["log_probability"] == -6 and result["probability"] == pytest.approx(math.exp(-6))
        assert result["target_tokens"] == 3 and result["mean_token_log_probability"] == -2


def test_pair_metrics_and_probability_deltas_do_not_change_correctness(confirmation_run):
    specification = confirmation_run.build()
    base = [dict(r, correct=False, target_probability=dict(available=True, probability=0.1, log_probability=-2.3))
            for r in specification["items"]]
    after = [dict(r, correct=r["confirmation_role"] != "surface",
                  target_probability=dict(available=True, probability=0.3, log_probability=-1.2)) for r in base]
    result = confirmation.metrics(after, specification, base)
    assert result["accuracy"] == 4/6
    assert result["both_sides_correct_rate"] == .5
    assert not result["unpaired_items"]
    assert all(d["probability_delta"] == pytest.approx(.2) for d in result["target_probability_deltas"])


@pytest.fixture
def evaluated_confirmation(confirmation_run, monkeypatch):
    b = confirmation_run
    specification = b.build()
    entries = {tid: dict(id=tid) for tid in b.splits["items"]}
    monkeypatch.setattr(pipeline, "inventory", lambda: (None, entries))
    monkeypatch.setattr(pipeline, "check_server", lambda args, arm: dict(id=args.served_model))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **kw: None)))
    monkeypatch.setattr(pipeline, "OfficialValidator", lambda: b.validator)
    calls = []
    def reply(args, exercise, tokenizer):
        calls.append(exercise["id"])
        return "right", {}
    monkeypatch.setattr(pipeline, "student_reply", reply)
    monkeypatch.setattr(confirmation, "target_probability", lambda args, row, tokenizer: dict(
        available=True, probability=.1 if args.arm == "base" else .2, log_probability=-2 if args.arm == "base" else -1))
    def official(args, ids, destination, adapter, splits, evaluation_scope):
        write_json(destination / "cost.json", dict(output_tokens=10))
        return [dict(id=tid, parent_id=splits["items"][tid]["parent_id"], category="simple_python", layer=3,
                     correct=True, evaluation_scope=evaluation_scope) for tid in ids]
    monkeypatch.setattr(pipeline, "official_run", official)
    b.calls_student = calls
    b.specification = specification
    return b


def test_evaluation_mid_paths_resume_and_heldout_unchanged(evaluated_confirmation):
    b = evaluated_confirmation
    for arm, checkpoint in (("base", "end"), ("D", "mid")):
        b.args.arm, b.args.checkpoint = arm, checkpoint
        b.args.served_model = "base" if arm == "base" else "mech-D-mid"
        pipeline.evaluate(b.args, b.splits)
        path = b.args.run_dir / "evaluation" / ("base" if arm == "base" else "D-mid") / "main"
        result = read_json(path / "confirmation.json")
        assert result["metrics"]["both_sides_correct_rate"] == 1
        assert read_json(path / "protocol.json")["checkpoint"] == checkpoint
        assert {r["layer"] for r in read_json(path / "items.json")} == {1, 2, 3}
        count = len(b.calls_student)
        pipeline.evaluate(b.args, b.splits)
        assert len(b.calls_student) == count
    assert read_json(b.args.run_dir / "heldout.json") == b.heldout
    b.specification["items"][0]["messages"][-1]["content"] = "changed"
    write_json(b.args.run_dir / "confirmation.json", b.specification)
    with pytest.raises(ValueError, match="Evaluation protocol/artifact changed"):
        pipeline.evaluate(b.args, b.splits)


def test_combined_report_mid_end_r1_seeds_confirmation_and_readonly_source(evaluated_confirmation):
    b = evaluated_confirmation
    for arm, checkpoint in (("base", "end"), ("C", "end"), ("D", "end"), ("C", "mid"), ("D", "mid")):
        b.args.arm, b.args.checkpoint = arm, checkpoint
        b.args.served_model = arm+("-mid" if checkpoint == "mid" else "")
        pipeline.evaluate(b.args, b.splits)
        if arm != "base":
            write_json(b.args.run_dir / arm / "training/metrics.json", dict(train_seed=0,
                supervised_tokens=15900, optimizer_steps=32, checkpoints=dict(mid=dict(
                    supervised_tokens=8000, optimizer_steps=16, path="adapter_mid"))))
    source = b.args.run_dir / "r1"
    for name in ("base", "C", "D", "C-s1", "D-s1"):
        arm = name.split("-")[0]
        write_json(source / "evaluation" / name / "main/items.json",
                   read_json(b.args.run_dir / "evaluation" / arm / "main/items.json"))
        if arm != "base":
            folder = "training_s1" if "-s1" in name else "training"
            write_json(source / arm / folder / "metrics.json", dict(train_seed=1 if "-s1" in name else 0,
                       supervised_tokens=15900, optimizer_steps=32))
    before = {p: p.read_bytes() for p in source.rglob("*") if p.is_file()}
    b.args.round1_run_dir = source
    stats.report(b.args, b.splits)
    report = read_json(b.args.run_dir / "report.json")
    rows = {r["variant"]: r for r in report["mechanism_table"]}
    assert {"R2/C-mid", "R2/D-mid", "R2/C", "R2/D", "R1/C", "R1/D", "R1/C-s1", "R1/D-s1"} <= rows.keys()
    assert rows["R2/C-mid"]["supervised_tokens"] == 8000
    assert rows["R2/C"]["supervised_tokens"] == 15900
    assert rows["R2/D"]["confirmation"]["valid_pairs"] == 4
    assert not rows["R1/D"]["confirmation"]
    assert "R1_C_s1_vs_R2_C_mid" in report["intervals"]
    assert report["intervals"]["confirm_base_vs_D_mid"]["pairs"]["independent_parents"] == 2
    assert report["intervals"]["confirm_base_vs_D_mid"]["items"]["items"] == 6
    assert all(p.read_bytes() == value for p, value in before.items())
    assert not (source / "paired").exists()
