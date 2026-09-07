"""State replay unit tests and an explicit two-package real CPU integration test."""
from copy import deepcopy
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time
import warnings

import pytest

from bfas.rtd.benchmarks import alfworld_bank as bank
from bfas.rtd.benchmarks.alfworld_state import (
    Observation, canonical_hash, parent_hash, reconstruct_package, replay_commands,
    validate_full_state,
)
from bfas.rtd.benchmarks.alfworld_support import (
    BoundedEnvBridge, EnvironmentUnavailable, FrozenRenderer, RealStepper,
    adapter, environment_identity, prompt_messages, verify_package,
)
from bfas.rtd.transport import FullState


class FakeStepper:
    def __init__(self, request, *, fault=None):
        self.request, self.fault = request, fault
        self.closed = False
        self.calls = []
        self.observations = []

    def observation(self, i):
        text = ["Your task is to: put apple on table.\n" + "x" * 2100,
                "Full observation " + "y" * 500, "Task complete."][i]
        commands = [("go to table 1",), ("put apple 1 on table 1",), ()][i]
        obs = Observation(i + (1 if self.fault == "order" and i else 0), text, commands,
                          "0" * 64 if self.fault == "world" else self.request["world_hash"],
                          done=i == 2, won=i == 2 and self.fault != "lost")
        self.observations.append(asdict(obs))
        return obs

    def reset(self, request):
        self.calls.append("reset")
        if self.fault == "timeout":
            raise EnvironmentUnavailable("fake worker timeout")
        return 0, self.observation(0)

    def step(self, cursor, command):
        self.calls.append(command)
        return cursor + 1, self.observation(cursor + 1)

    def close(self):
        self.closed = True


def render(request, history):
    return "identical policy prompt"  # state equality must include everything else


@pytest.fixture
def request_state():
    files = {n: canonical_hash(n) for n in ("game.tw-pddl", "traj_data.json", "initial_state.pddl")}
    return dict(benchmark="alfworld", split="train", task_id="pick_and_place_simple-Apple-None-Table-1/trial-1",
                goal="put apple on table.", world_files=files, world_hash=canonical_hash(files),
                environment_hash="a" * 64, max_episode_steps=40, state_kind="train_reset_request")


def make_payload(request):
    commands = ["go to table 1", "put apple 1 on table 1"]
    result = replay_commands(request, commands, FakeStepper(request), render)
    turns = [dict(target=c, prompt="historical bare deployment prompt", context=prompt_messages(
        request, json.loads(result.states[i].history_json), react=False)) for i, c in enumerate(commands)]
    row = dict(task_id=request["task_id"], timestamp="2026-01-01", attempt_index=0,
               verified=True, tokens_spent=40, demo=dict(turns=turns))
    q = bank.query_id("b" * 64, row, 1)
    return dict(query_id=q, provenance=dict(task_id=request["task_id"], ledger_sha256="b" * 64,
                                           timestamp=row["timestamp"], attempt_index=0, line=1),
                historical_response=row, raw_ledger_line=json.dumps(row),
                request_state_hash=canonical_hash(request), status="candidate", payload_kind="extracted_teacher_commands",
                success=True, cost=40, cost_confidence="estimated", cost_basis="estimated",
                usage=dict(estimated_output_tokens=40), commands=commands, behaviors=[],
                dependencies=[], exclusion_reasons=[], unavailable_reason=None)


def test_full_history_and_equal_prompts_do_not_merge_states(request_state):
    p = make_payload(request_state)
    result = replay_commands(request_state, p["commands"], FakeStepper(request_state), render)
    again = replay_commands(request_state, p["commands"], FakeStepper(request_state), render)
    assert result == again and result.won and result.done
    assert len({s.state_hash for s in result.states}) == 3
    assert len({s.prompt for s in result.states}) == 1
    history = json.loads(result.states[-1].history_json)
    assert len(history[0]["content"]) > 2000 and len(history[2]["content"]) > 300
    assert [h["role"] for h in history] == ["user", "assistant", "tool", "assistant", "tool"]
    with pytest.raises(ValueError, match="full state mismatch"):
        result.states[0].assert_matches(result.states[1])
    other_request = {**request_state, "task_id": request_state["task_id"].replace("trial-1", "trial-2")}
    other = replay_commands(other_request, p["commands"], FakeStepper(other_request), render)
    with pytest.raises(ValueError, match="full state mismatch"):
        result.states[0].assert_matches(other.states[0])
    lost = replay_commands(request_state, p["commands"], FakeStepper(request_state, fault="lost"), render)
    assert [s.state_hash for s in lost.states] == [s.state_hash for s in result.states]
    assert "won" not in result.states[-1].history_json


@pytest.mark.parametrize("field,value", [("goal", ""), ("world_hash", "0" * 64), ("world_files", {})])
def test_missing_goal_and_wrong_world_hash_rejected_before_environment(request_state, field, value):
    request_state[field] = value
    stepper = FakeStepper(request_state)
    with pytest.raises(ValueError):
        replay_commands(request_state, ["look"], stepper, render)
    assert stepper.calls == []


@pytest.mark.parametrize("fault,match", [("order", "out-of-order"), ("world", "world hash")])
def test_bad_stepper_observations_rejected(request_state, fault, match):
    with pytest.raises(ValueError, match=match):
        replay_commands(request_state, ["go to table 1", "put apple 1 on table 1"],
                        FakeStepper(request_state, fault=fault), render)


def test_out_of_order_full_history_rejected_after_rehash(request_state):
    p = make_payload(request_state)
    state = replay_commands(request_state, p["commands"], FakeStepper(request_state), render).states[-1]
    history = json.loads(state.history_json)
    history[1], history[3] = history[3], history[1]
    forged = FullState.create(request_state, history, state.prompt, state.parent_hash)
    with pytest.raises(ValueError, match="out-of-order"):
        validate_full_state(forged, expected_world_hash=request_state["world_hash"])


def test_teacher_prefix_requires_package_ownership_and_inner_fold_before_reset(request_state):
    p = make_payload(request_state)
    for owned, parents, match in [((), {parent_hash(request_state["task_id"])}, "owned"),
                                   ({p["query_id"]}, (), "inner fold")]:
        with pytest.raises(ValueError, match=match):
            reconstruct_package(p, request_state, None, render, owned_ids=owned, inner_parent_hashes=parents)
    p["dependencies"] = ["not-owned"]
    with pytest.raises(ValueError, match="owned"):
        reconstruct_package(p, request_state, None, render, owned_ids={p["query_id"]},
                            inner_parent_hashes={parent_hash(request_state["task_id"])})


@pytest.mark.parametrize("fault", [None, "lost", "order", "world", "timeout", "context", "goal", "hash"])
def test_package_verification_fails_closed_and_always_cleans_up(request_state, fault):
    p = make_payload(request_state)
    worker = FakeStepper(request_state, fault=fault)
    if fault == "context":
        p["historical_response"]["demo"]["turns"][1]["context"][0]["content"] += "diverged"
    elif fault == "goal":
        request_state["goal"] = ""
    elif fault == "hash":
        del request_state["world_hash"]
    verified = verify_package(p, request_state, worker, render)
    assert worker.closed
    assert verified["status"] == ("usable" if fault is None else "unavailable")
    if fault is not None:
        assert verified["unavailable_reason"] and verified["behaviors"] == []
    if fault == "lost":
        assert verified["verification"]["won"] is False
    if fault == "timeout":
        assert verified["verification"]["won"] is None  # infrastructure is not reward zero
    if fault == "context":
        assert len(verified["verification"]["states"]) == 2  # retain partial divergent evidence
    if fault is None:
        assert len(verified["behaviors"]) == 2 and verified["verification"]["success_reproduced"]


def test_renderer_preserves_goal_and_existing_windows_without_truncating_state(request_state):
    history = [dict(role="user", index=0, content="reset", admissible=["look"], done=False)]
    for i in range(1, 11):
        history.extend([dict(role="assistant", index=i, content=f"command-{i}"),
                        dict(role="tool", index=i, content=f"feedback-{i}-" + "x" * 2100,
                             admissible=["look"], done=False)])
    before = deepcopy(history)
    messages = prompt_messages(request_state, history)
    assert history == before
    text = messages[-1]["content"]
    assert request_state["goal"] in text
    assert "> command-1\n" not in text and "> command-2\n" not in text
    assert "> command-3\n" in text and "> command-10\n" in text
    assert "feedback-10-" + "x" * 288 in text


def test_real_stepper_checks_world_before_starting_worker(request_state, tmp_path, monkeypatch):
    monkeypatch.setattr(adapter, "DATA", tmp_path)
    world = tmp_path / "train" / request_state["task_id"]
    world.mkdir(parents=True)
    for name in request_state["world_files"]:
        (world / name).write_text("wrong content")
    worker = RealStepper(environment_hash=request_state["environment_hash"])
    with pytest.raises(ValueError, match="wrong world hash"):
        worker.reset(request_state)
    assert worker.bridge is None


def test_bridge_deadline_cleans_worker_and_uses_explicit_cpu_data_roots(tmp_path, monkeypatch):
    # Fake Python worker executable: no ALFWorld invocation in this unit test.
    script = tmp_path / "silent-worker"
    script.write_text(f"#!{sys.executable}\nimport os,time,json\n"
                      f"open({str(tmp_path / 'environment.json')!r},'w').write(json.dumps(dict(cwd=os.getcwd(),"
                      "cuda=os.environ['CUDA_VISIBLE_DEVICES'],data=os.environ['ALFWORLD_DATA'],pid=os.getpid())))\n"
                      "time.sleep(10)\n")
    script.chmod(0o700)
    monkeypatch.setattr(adapter, "ENV_PYTHON", script)
    started = time.monotonic()
    with pytest.raises(EnvironmentUnavailable, match="timeout"):
        BoundedEnvBridge("game/trial", timeout=0.3)
    assert time.monotonic() - started < 4
    values = json.loads((tmp_path / "environment.json").read_text())
    assert values["cuda"] == "" and values["data"] == str(adapter.DATA.parent)
    assert values["cwd"] != str(adapter.ROOT) and not Path(values["cwd"]).exists()
    assert not Path(f"/proc/{values['pid']}").exists()


# Keep the integration marker selectable with -m without changing shared pytest
# configuration. Suppress only its declaration warning, not runtime warnings.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", pytest.PytestUnknownMarkWarning)
    integration = pytest.mark.integration


@integration
@pytest.mark.skipif(os.environ.get("RTD_ALFWORLD_REAL_TESTS") != "1",
                    reason="opt in with RTD_ALFWORLD_REAL_TESTS=1; real read-only assets")
def test_real_environment_two_packages_replay_deterministically(tmp_path, monkeypatch):
    """Skip only unavailable prerequisites/worker startup, never a replay mismatch."""
    from tools.rtd_alfworld_verify import TOKENIZER
    root = adapter.ROOT
    if not adapter.ENV_PYTHON.is_file() or not (TOKENIZER / "tokenizer.json").is_file():
        pytest.skip("real ALFWorld environment Python or local frozen tokenizer unavailable; no download attempted")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    try:
        environment = environment_identity(root, TOKENIZER)
    except FileNotFoundError as exc:
        pytest.skip(f"real ALFWorld prerequisites unavailable: {exc}")
    rows = [(i, json.loads(line)) for i, line in enumerate((root / bank.LEDGER).read_text().splitlines(), 1)]
    selected = [(i, r) for i, r in rows if r["verified"]][:2]
    assert len(selected) == 2
    renderer = FrozenRenderer(TOKENIZER)
    sources = bank._Sources(root)
    for line, row in selected:
        from bfas.rtd.benchmarks.alfworld_caps import CapConfiguration
        request = bank._reset_request(sources, row["task_id"], CapConfiguration(), environment["environment_hash"])
        p = make_payload(request)
        # Use complete real archived commands and contexts, no synthetic substitutions.
        q = bank.query_id(bank.file_hash(root / bank.LEDGER), row, line)
        p.update(query_id=q, historical_response=row, raw_ledger_line=json.dumps(row),
                 commands=[t["target"] for t in row["demo"]["turns"]], cost=row["tokens_spent"])
        p["provenance"].update(task_id=row["task_id"])
        first = verify_package(p, request, RealStepper(environment_hash=environment["environment_hash"]), renderer)
        if first["verification"]["error_type"] == "EnvironmentUnavailable" and not first["verification"]["states"]:
            pytest.skip(f"real ALFWorld worker unavailable: {first['unavailable_reason']}")
        assert first["status"] == "usable", first["unavailable_reason"]
        second = verify_package(p, request, RealStepper(environment_hash=environment["environment_hash"]), renderer)
        assert second["status"] == "usable", second["unavailable_reason"]
        assert first["verification"]["transcript_hash"] == second["verification"]["transcript_hash"]
        assert first["verification"]["states"] == second["verification"]["states"]
