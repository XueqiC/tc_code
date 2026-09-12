import copy
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bfas.mech_bfcl.harness import multiturn_snapshot, pack, unpack, restore_multiturn_context


def test_random_snapshot_is_lossless_and_does_not_share_mutable_rng():
    rng = random.Random(42)
    rng.random()
    restored = unpack(json.loads(json.dumps(pack(dict(_random=rng)))))
    assert restored["_random"] is not rng
    assert [restored["_random"].random() for _ in range(10)] == [rng.random() for _ in range(10)]


@pytest.fixture
def replay(monkeypatch):
    util = SimpleNamespace()
    calls = []

    def execute(code, initial, classes, model, tid, long_context=False):
        key = f"{model}_{tid}_Fake_instance"
        if not hasattr(util, key):
            setattr(util, key, SimpleNamespace(value=initial["Fake"]["value"], _random=random.Random(12)))
        instance = getattr(util, key)
        outputs = []
        for call in code:
            calls.append((model, call, long_context))
            instance.value = int(call.split("=")[1].rstrip(")"))
            outputs.append(json.dumps(dict(value=instance.value)))
        return outputs, dict(Fake=instance)

    util.execute_multi_turn_func_call = execute
    monkeypatch.setitem(sys.modules, "bfcl_eval.eval_checker.multi_turn_eval", SimpleNamespace(multi_turn_utils=util))
    monkeypatch.setitem(sys.modules, "bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker", SimpleNamespace(
        state_checker=lambda a, b: dict(valid=a["Fake"].value == b["Fake"].value)))
    monkeypatch.setitem(sys.modules, "bfcl_eval.model_handler.utils", SimpleNamespace(
        convert_to_function_call=lambda rows: [f"{name}(x={args['x']})" for row in rows for name, args in row.items()]))
    entry = dict(id="multi_turn_long_context_1", initial_config=dict(Fake=dict(value=0)), involved_classes=["Fake"],
                 question=[[dict(role="user", content="first")], [dict(role="user", content="second")]])
    def action(x):
        return dict(role="assistant", tool_calls=[dict(function=dict(name="advance", arguments=dict(x=x)))])
    frame = dict(task_id=entry["id"], messages=[dict(role="user", content="first"), action(1),
        dict(role="tool", content='{"value":1}'), dict(role="user", content="second"), action(2),
        dict(role="tool", content='{"value":2}')])
    return SimpleNamespace(util=util, calls=calls, entry=entry, frame=frame, truth=[["advance(x=1)"], ["advance(x=99)"]])


def test_reconstructs_prior_gold_then_current_prefix_without_future_gold(replay):
    b = replay
    original = copy.deepcopy(b.frame)
    snapshot, provenance = multiturn_snapshot(b.frame, b.entry, b.truth)
    assert unpack(snapshot["Fake"])["value"] == 2
    assert provenance["turn_index"] == 1
    assert provenance["prior_ground_truth_calls"] == [["advance(x=1)"]]
    assert provenance["current_turn_calls"] == ["advance(x=2)"]
    assert all(c[2] for c in b.calls)
    assert not any("99" in c[1] for c in b.calls)
    assert b.frame == original
    assert not any(k.endswith("_instance") for k in vars(b.util))


@pytest.mark.parametrize("mismatch", ["prior_state", "current_output", "question", "turn", "unsafe_call"])
def test_snapshot_exclusions_preserve_a_precise_reason_and_clean_instances(replay, mismatch):
    b = replay
    if mismatch == "prior_state":
        b.truth[0] = ["advance(x=5)"]
    elif mismatch == "current_output":
        b.frame["messages"][-1]["content"] = '{"value":5}'
    elif mismatch == "question":
        b.frame["messages"][0]["content"] = "invented question"
    elif mismatch == "turn":
        b.truth = []
    else:
        b.frame["messages"][1]["tool_calls"][0]["function"]["name"] = "__import__('os').system"
    with pytest.raises(ValueError):
        multiturn_snapshot(b.frame, b.entry, b.truth)
    assert not any(k.endswith("_instance") for k in vars(b.util))


def test_legacy_context_repair_loads_official_initial_and_gold_without_mutation(replay, monkeypatch):
    b = replay
    monkeypatch.setitem(sys.modules, "bfcl_eval.utils", SimpleNamespace(
        load_dataset_entry=lambda category: [b.entry],
        load_ground_truth_entry=lambda category: [dict(id=b.entry["id"], ground_truth=b.truth)]))
    context = dict(task_id=b.entry["id"], snapshot=None, snapshot_error="Unserializable executor state: Random",
                   involved_classes=["Fake"])
    repaired = restore_multiturn_context(context, b.frame)
    assert repaired["snapshot_error"] is None and repaired["snapshot_reconstruction"]
    assert context["snapshot"] is None
    b.truth[0] = ["advance(x=5)"]
    failed = restore_multiturn_context(context, b.frame)
    assert "Prior student history diverges" in failed["snapshot_error"]
    assert failed["snapshot"] is None
