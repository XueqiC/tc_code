from __future__ import annotations

import json
import os
import sys
import threading
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bfas.adapter import (
    BenchmarkAdapter,
    CollectionArtifacts,
    Demo,
    Rollout,
    TaskRef,
    Turn,
)
from bfas.adapters.alfworld import ALFWorldAdapter
import bfas.adapters.alfworld as alfworld_module
from bfas.audit import AuditError, audit_pool
from bfas.protocol import AdaptiveSampler, make_support_split, support_size
from bfas.adapters.bfcl import BFCLAdapter, extract_verdicts
import bfas.adapters.bfcl as bfcl
from bfas import run as bfas_run


class SyntheticAdapter(BenchmarkAdapter):
    name = "synthetic"

    def task_pool(self) -> list[TaskRef]:
        return [TaskRef("call", "calls"), TaskRef("prose", "prose")]

    def official_eval_split_disjoint(self) -> bool:
        return True

    def rollout(self, policy, task_ids, temperature, guided_demos=None):
        return []

    def teacher_demo(self, task_ids, attempts):
        return {}

    def evaluate(self, policy_ref, out_dir):
        return {}

    def serving_probe(self) -> None:
        return None

    def generation_suffix(self) -> str:
        return "<GEN>"

    def target_policy(self, category: str) -> str:
        return "call_required" if category == "calls" else "prose_ok"

    def rerender(self, row) -> str:
        return row["_render_context"]


def valid_row() -> dict[str, Any]:
    prompt = "task<GEN>"
    return {
        "task_id": "call",
        "category": "calls",
        "prompt": prompt,
        "response": '<tool_call>\n{"name":"f"}\n</tool_call>',
        "_render_context": prompt,
        "_task_phat": 0.5,
    }


def test_split_determinism_sizing_and_calibration() -> None:
    tasks = [TaskRef(f"task-{i:04d}", f"cat-{i % 4}") for i in range(2000)]
    first = make_support_split(tasks)
    second = make_support_split(reversed(tasks))
    assert first == second
    assert len(first.support) == 100
    assert len(first.demand) == 80
    assert len(first.calibration) == 20
    assert set(first.demand).isdisjoint(first.calibration)
    assert set(first.demand) | set(first.calibration) == set(first.support)
    assert support_size(20) == 20
    assert support_size(100) == 50
    assert support_size(10_000) == 250


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda row: row.update(messages=[]), "forbidden messages"),
        (lambda row: row.update(prompt="task<WRONG>"), "generation suffix"),
        (lambda row: row.update(response="plain prose"), "requires a call"),
    ],
)
def test_pool_audit_rejects_contract_violations(mutation, message: str) -> None:
    row = valid_row()
    mutation(row)
    with pytest.raises(AuditError, match=message):
        audit_pool([row], SyntheticAdapter())


def test_pool_audit_rejects_prompt_over_training_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = valid_row()
    row["prompt"] = row["_render_context"] = "long-task-context<GEN>"
    monkeypatch.setenv("AW_MAX_PROMPT_TOKENS", "10")

    with pytest.raises(
        AuditError,
        match=(
            r"longest row 0 task_id='call' has 22 tokens.*"
            r"AW_MAX_PROMPT_TOKENS=10.*required cap >= 22"
        ),
    ):
        audit_pool(
            [row], SyntheticAdapter(), prompt_token_length=lambda prompt: len(prompt)
        )


def _write_score(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_bfcl_verdicts_require_present_positive_summary(tmp_path: Path) -> None:
    _write_score(
        tmp_path / "model/non_live/BFCL_v4_alpha_score.json",
        [
            {"accuracy": 0.5, "correct_count": 1, "total_count": 2},
            {"id": "alpha-fail", "valid": False},
        ],
    )
    _write_score(
        tmp_path / "model/live/BFCL_v4_beta_score.json",
        [{"accuracy": 1.0, "correct_count": 2, "total_count": 2}],
    )
    verdicts = extract_verdicts(tmp_path, {
        "alpha-pass": "alpha",
        "alpha-fail": "alpha",
        "beta-1": "beta",
        "beta-2": "beta",
        "missing-1": "missing",
    })
    assert verdicts == {
        "alpha-pass": True,
        "alpha-fail": False,
        "beta-1": True,
        "beta-2": True,
        "missing-1": False,
    }


def test_bfcl_generate_uses_harness_owned_vllm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[list[str], dict[str, Any]]] = []

    def run(command, **kwargs):
        calls.append((list(command), kwargs))

    monkeypatch.setattr(bfcl.subprocess, "run", run)
    monkeypatch.setenv("GPU_UTIL", "0.73")
    monkeypatch.setenv("PATH", "/original/bin")
    monkeypatch.setenv("LOCAL_SERVER_ENDPOINT", "old-endpoint")

    BFCLAdapter(port=9123)._run_generate(
        ["--model", bfcl.MODEL_NAME, "--test-category", "all"],
        "Qwen/Qwen3.5-4B",
    )

    assert len(calls) == 1
    command, kwargs = calls[0]
    assert command[:2] == [str(bfcl.BFCL_BIN), "generate"]
    assert command[-6:] == [
        "--backend", "vllm",
        "--num-gpus", "1",
        "--gpu-memory-utilization", "0.73",
    ]
    assert "--skip-server-setup" not in command
    assert "--local-model-path" not in command
    env = kwargs["env"]
    assert "LOCAL_SERVER_ENDPOINT" not in env
    assert env["LOCAL_SERVER_PORT"] == "9123"
    assert env["VLLM_USE_FLASHINFER_SAMPLER"] == "0"
    assert env["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    assert env["PATH"] == f"{bfcl.VLLM_BIN_DIR}{os.pathsep}/original/bin"
    assert kwargs["cwd"] == bfcl.BFCL_ROOT
    assert kwargs["check"] is True


def test_bfcl_trained_checkpoint_is_merged_rebuilt_and_cleaned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    checkpoint = project / "results/bfas/bfcl/arm_s0/checkpoint"
    checkpoint.mkdir(parents=True)
    (checkpoint / "model.safetensors").write_bytes(b"trained")
    merged = checkpoint.parent / "hub_merged"
    merged.mkdir()
    (merged / "stale.txt").write_text("old")
    leaderboard = project / "leaderboard"
    calls: list[tuple[list[str], dict[str, Any]]] = []

    monkeypatch.setattr(bfcl, "ROOT", project)
    monkeypatch.setattr(bfcl, "BFCL_ROOT", leaderboard)
    monkeypatch.setattr(bfcl, "BFCL_BIN", project / "bfcl")
    monkeypatch.setattr(bfcl, "VLLM_BIN_DIR", project / "vllm-bin")
    monkeypatch.setattr(bfcl, "MERGE_EXPORT", project / "tools/merge.py")

    def run(command, **kwargs):
        command = list(command)
        calls.append((command, kwargs))
        if command[0] == str(project / ".venv/bin/python"):
            out = Path(command[command.index("--out") + 1])
            out.mkdir(parents=True)
            (out / "model.safetensors.index.json").write_text("{}")
        else:
            local_path = Path(command[command.index("--local-model-path") + 1])
            assert local_path == merged
            assert (local_path / "model.safetensors.index.json").is_file()
            assert not (local_path / "stale.txt").exists()

    monkeypatch.setattr(bfcl.subprocess, "run", run)

    BFCLAdapter()._run_generate(["--model", bfcl.MODEL_NAME], checkpoint)

    merge_command, merge_kwargs = calls[0]
    assert merge_command == [
        str(project / ".venv/bin/python"),
        str(project / "tools/merge.py"),
        "--adapter", str(checkpoint.resolve()),
        "--out", str(merged),
        "--verify",
    ]
    assert merge_kwargs == {"cwd": project, "check": True}
    assert not merged.exists()
    trashed = list((project / "_trash").glob("hub_merged_*"))
    assert len(trashed) == 1
    assert (trashed[0] / "stale.txt").read_text() == "old"


def test_bfcl_serving_lane_does_not_start_pipeline_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unexpected_server(*args, **kwargs):
        raise AssertionError("pipeline server must not be created for BFCL")

    monkeypatch.setattr(bfas_run, "VLLMServer", unexpected_server)
    with bfas_run.serving_lane(
        BFCLAdapter(), "policy", "0", 8901, tmp_path / "server.log"
    ):
        pass


def test_beta_posterior_adaptive_stopping() -> None:
    sampler = AdaptiveSampler(["solved", "failed"])
    for _ in range(3):
        sampler.observe("solved", True)
        sampler.observe("failed", False)
    assert "solved" not in sampler.active()
    assert "failed" in sampler.active()
    while "failed" in sampler.active():
        sampler.observe("failed", False)
    assert sampler.rounds() == {"solved": 3, "failed": 8}
    assert sampler.p_hats()["solved"] == pytest.approx(4 / 5)
    assert sampler.p_hats()["failed"] == pytest.approx(1 / 10)


def test_collection_cache_round_trip_preserves_dataclasses(tmp_path: Path) -> None:
    deployment = Turn("deploy<GEN>", "open door", [{"role": "user"}])
    guided = Rollout(
        "task",
        True,
        (Turn("guided<GEN>", "open door", [{"role": "user"}]),),
        {
            "checker_verified": True,
            "deployment_turns": [deployment],
            "mu_fields": [{"_mu_nll_sum": 1.25, "_mu_ntok": 2}],
        },
    )
    demo = Demo(
        "task",
        (Turn("teacher<GEN>", "open door", [{"role": "user"}]),),
        "open door",
        {"attempt": 1},
    )
    artifacts = CollectionArtifacts(
        unguided_rollouts=(Rollout("task", False, (), {"checker_verified": False}),),
        p_hats={"task": 0.25},
        unguided_rounds={"task": 3},
        teacher_demos={"task": demo},
        guided_rollouts=(guided,),
        guided_rounds={"task": 2},
        checker_summary={"category": {"verified": 1, "total": 2}},
    )

    bfas_run._write_collection(tmp_path, artifacts)
    loaded = bfas_run._load_collection(tmp_path)

    assert isinstance(loaded, CollectionArtifacts)
    assert isinstance(loaded.teacher_demos["task"], Demo)
    assert isinstance(loaded.teacher_demos["task"].turns[0], Turn)
    cached_guided = loaded.guided_rollouts[0]
    assert isinstance(cached_guided, Rollout)
    assert isinstance(cached_guided.raw["deployment_turns"][0], Turn)
    assert cached_guided.raw["mu_fields"] == [
        {"_mu_nll_sum": 1.25, "_mu_ntok": 2}
    ]


def test_collection_cache_reuses_and_refreshes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = 0

    def collect(*args, **kwargs):
        nonlocal calls
        calls += 1
        return CollectionArtifacts(
            unguided_rollouts=(),
            p_hats={"task": float(calls)},
            unguided_rounds={},
            teacher_demos={},
            guided_rollouts=(),
            guided_rounds={},
            checker_summary={},
        )

    monkeypatch.setattr(bfas_run, "_collect_seed_artifacts", collect)
    monkeypatch.delenv("BFAS_REFRESH_COLLECT", raising=False)
    adapter = SyntheticAdapter()
    first = bfas_run._cached_collection(
        adapter, "policy", ["task"], "0", 8900, tmp_path / "collect_s0"
    )
    second = bfas_run._cached_collection(
        adapter, "policy", ["task"], "0", 8900, tmp_path / "collect_s0"
    )
    assert calls == 1
    assert first.p_hats == second.p_hats == {"task": 1.0}

    monkeypatch.setenv("BFAS_REFRESH_COLLECT", "1")
    refreshed = bfas_run._cached_collection(
        adapter, "policy", ["task"], "0", 8900, tmp_path / "collect_s0"
    )
    assert calls == 2
    assert refreshed.p_hats == {"task": 2.0}


def test_phase_cache_resumes_after_partial_demo_collection(
    tmp_path: Path,
) -> None:
    class ResumeAdapter(SyntheticAdapter):
        def __init__(self, crash_on: str | None = None):
            self.crash_on = crash_on
            self.demo_calls: list[str] = []
            self.rollout_phases: list[str] = []

        def task_pool(self) -> list[TaskRef]:
            return [TaskRef("a", "tasks"), TaskRef("b", "tasks")]

        def rollout(self, policy, task_ids, temperature, guided_demos=None):
            guided = guided_demos is not None
            self.rollout_phases.append("guided" if guided else "unguided")
            return [
                Rollout(
                    task_id,
                    guided,
                    (Turn("prompt", "response"),),
                    {"checker_verified": guided},
                )
                for task_id in task_ids
            ]

        def teacher_demo(self, task_ids, attempts):
            task_id = task_ids[0]
            self.demo_calls.append(task_id)
            if task_id == self.crash_on:
                raise SystemExit("simulated process crash")
            return {
                task_id: Demo(
                    task_id,
                    (Turn("teacher", "response"),),
                    "response",
                    {"checker_verified": True},
                )
            }

    cache_dir = tmp_path / "collect_s0"
    first = ResumeAdapter(crash_on="b")
    with pytest.raises(SystemExit, match="simulated process crash"):
        bfas_run._collect_seed_artifacts(
            first, "policy", ["a", "b"], "0", 8900, cache_dir
        )

    assert (cache_dir / "unguided.json").is_file()
    assert not (cache_dir / "guided.json").exists()
    partial_demos = json.loads((cache_dir / "demos.json").read_text())
    assert partial_demos["completed_task_ids"] == ["a"]
    assert partial_demos["complete"] is False

    resumed = ResumeAdapter()
    artifacts = bfas_run._collect_seed_artifacts(
        resumed, "policy", ["a", "b"], "0", 8900, cache_dir
    )

    assert resumed.demo_calls == ["b"]
    assert resumed.rollout_phases and set(resumed.rollout_phases) == {"guided"}
    assert set(artifacts.teacher_demos) == {"a", "b"}
    assert (cache_dir / "guided.json").is_file()

    fully_cached = ResumeAdapter()
    bfas_run._collect_seed_artifacts(
        fully_cached, "policy", ["a", "b"], "0", 8900, cache_dir
    )
    assert fully_cached.demo_calls == []
    assert fully_cached.rollout_phases == []


def test_demo_failures_are_isolated_and_logged(capsys: pytest.CaptureFixture[str]) -> None:
    class FailingAdapter(SyntheticAdapter):
        def teacher_demo(self, task_ids, attempts):
            task_id = task_ids[0]
            if task_id == "bad":
                raise RuntimeError("broken episode")
            return {
                task_id: Demo(task_id, (), "worked", {"checker_verified": True})
            }

    results: dict[str, Demo | None] = {}
    bfas_run._collect_teacher_demo_phase(
        FailingAdapter(), ["bad", "good"], 2, results.__setitem__
    )

    assert results["bad"] is None
    assert isinstance(results["good"], Demo)
    assert "[bfas][demo] task=bad failed: broken episode" in capsys.readouterr().out


def test_alfworld_teacher_react_prompt_and_command_only_worked_example(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import appworld_teacher
    import bfas.adapters.alfworld as alfworld

    task_type = "pick_heat_then_place_in_recep"
    task_id = f"{task_type}-Apple-None-DiningTable-1/trial"
    action = "heat apple 1 with microwave 1"
    reply = (
        "THOUGHT: I should inspect the room first.\n"
        "ACTION: look\n"
        "THOUGHT: The history shows I already hold the apple, so heat it.\n"
        f"ACTION: {action}"
    )
    teacher_messages: list[list[dict[str, str]]] = []

    class Bridge:
        def __init__(self, split: str, received_task_id: str):
            assert split == "train"
            assert received_task_id == task_id

        @staticmethod
        def _read() -> dict[str, Any]:
            return {
                "observation": "You are holding apple 1 beside microwave 1.",
                "admissible": ["look", action],
            }

        @staticmethod
        def step(command: str) -> dict[str, Any]:
            assert command == action
            return {
                "observation": "The apple is now hot.",
                "admissible": ["look"],
                "done": True,
                "won": True,
            }

        @staticmethod
        def close() -> None:
            return None

    def generate_reply(config, messages, temperature=None):
        teacher_messages.append(messages)
        return reply

    adapter = ALFWorldAdapter()
    adapter._tokenizer = object()
    monkeypatch.setattr(alfworld, "_EnvBridge", Bridge)
    monkeypatch.setattr(
        adapter,
        "_render",
        lambda messages: json.dumps(messages, sort_keys=True),
    )
    monkeypatch.setattr(
        appworld_teacher, "load_teacher_config", lambda name: object()
    )
    monkeypatch.setattr(appworld_teacher, "generate_reply", generate_reply)

    episode = adapter.teacher_episode(task_id, attempt_index=0, temperature=0.0)

    assert episode.verified is True
    assert episode.response_texts == (reply,)
    assert episode.demo is not None
    assert episode.demo.worked_example == action
    assert "THOUGHT" not in episode.demo.worked_example
    assert "ACTION:" not in episode.demo.worked_example
    assert teacher_messages[0][0] == {
        "role": "system",
        "content": alfworld.TEACHER_REACT_INSTRUCTION,
    }
    teacher_prompt = teacher_messages[0][1]["content"]
    assert alfworld.TEACHER_REACT_EXAMPLES[task_type] in teacher_prompt
    assert teacher_prompt.count("Canonical worked example") == 1
    assert alfworld.TEACHER_REACT_EXAMPLES["pick_cool_then_place_in_recep"] not in teacher_prompt
    assert alfworld.TEACHER_REACT_INSTRUCTION not in episode.demo.turns[0].prompt
    assert "Canonical worked example" not in episode.demo.turns[0].prompt


def test_alfworld_teacher_quota_waits_probes_and_retries(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import appworld_teacher
    import bfas.adapters.alfworld as alfworld

    adapter = ALFWorldAdapter()
    adapter._tokenizer = object()
    episode_calls = 0
    sleeps: list[float] = []
    probe_calls = 0

    monkeypatch.setenv("BFAS_TEACHER_QUOTA_WAIT_S", "7")
    monkeypatch.setenv("BFAS_TEACHER_QUOTA_DEADLINE_S", "20")
    monkeypatch.setattr(
        appworld_teacher, "load_teacher_config", lambda name: object()
    )

    def generate_reply(config, messages, temperature=None):
        nonlocal probe_calls
        probe_calls += 1
        return "OK"

    def episode(*args, **kwargs):
        nonlocal episode_calls
        episode_calls += 1
        if episode_calls == 1:
            raise appworld_teacher.TeacherAPIError(
                "chat completion returned HTTP 429 after 8 attempts"
            )
        return Rollout(
            "task",
            True,
            (Turn("prompt", "command"),),
            {"checker_verified": True},
        )

    monkeypatch.setattr(appworld_teacher, "generate_reply", generate_reply)
    monkeypatch.setattr(alfworld.time, "sleep", sleeps.append)
    monkeypatch.setattr(adapter, "_episode", episode)

    demos = adapter.teacher_demo(["task"], attempts=1)

    assert set(demos) == {"task"}
    assert episode_calls == 2
    assert probe_calls == 1
    assert sleeps == [7.0]
    assert (
        "[bfas][demo] teacher quota exhausted, sleeping 7s"
        in capsys.readouterr().out
    )


def test_alfworld_teacher_quota_deadline_keeps_partial_demos(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import appworld_teacher

    adapter = ALFWorldAdapter()
    adapter._tokenizer = object()
    monkeypatch.setenv("BFAS_DEMO_WORKERS", "1")
    monkeypatch.setenv("BFAS_TEACHER_QUOTA_DEADLINE_S", "0")
    monkeypatch.setattr(
        appworld_teacher, "load_teacher_config", lambda name: object()
    )

    def episode(policy, task_id, *args, **kwargs):
        if task_id == "quota":
            raise appworld_teacher.TeacherAPIError(
                "chat completion returned HTTP 429 after 8 attempts"
            )
        return Rollout(
            task_id,
            True,
            (Turn("prompt", "command"),),
            {"checker_verified": True},
        )

    monkeypatch.setattr(adapter, "_episode", episode)

    demos = adapter.teacher_demo(["good", "quota", "later"], attempts=1)

    assert set(demos) == {"good"}


def test_alfworld_completion_uses_raw_prompt_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, Any] = {}

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        @staticmethod
        def read() -> bytes:
            return b'{"choices":[{"text":"look"}]}'

    def urlopen(request, timeout):
        seen["url"] = request.full_url
        seen["body"] = json.loads(request.data)
        seen["timeout"] = timeout
        return Response()

    monkeypatch.setattr("bfas.adapters.alfworld.urllib.request.urlopen", urlopen)
    reply = ALFWorldAdapter(port=9123)._server_reply("raw prompt bytes", 0.7)

    assert reply == "look"
    assert seen["url"] == "http://127.0.0.1:9123/v1/completions"
    assert seen["body"] == {
        "model": "bfas-policy",
        "prompt": "raw prompt bytes",
        "max_tokens": 256 if alfworld_module._student_react() else 32,
        "temperature": 0.7,
        "add_special_tokens": False,
    }


def test_alfworld_teacher_attempts_are_parallel_and_explicit_temperature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import appworld_teacher

    adapter = ALFWorldAdapter()
    adapter._tokenizer = object()
    barrier = threading.Barrier(2, timeout=2)
    calls: dict[str, list[float]] = {"a": [], "b": []}
    configs: dict[str, Any] = {}

    monkeypatch.setenv("BFAS_DEMO_WORKERS", "2")
    monkeypatch.setenv("TEACHER_TEMP", "do-not-touch")
    monkeypatch.setattr(
        appworld_teacher, "load_teacher_config", lambda name: object()
    )

    def episode(
        policy,
        task_id,
        split,
        temperature,
        demo=None,
        teacher_config=None,
    ):
        calls[task_id].append(temperature)
        configs.setdefault(task_id, teacher_config)
        if temperature == 0.0:
            barrier.wait()
        return Rollout(
            task_id,
            temperature > 0.0,
            (Turn("prompt", "command"),),
            {"checker_verified": temperature > 0.0},
        )

    monkeypatch.setattr(adapter, "_episode", episode)
    demos = adapter.teacher_demo(["a", "b"], attempts=3)

    assert set(demos) == {"a", "b"}
    assert calls == {"a": [0.0, 0.7], "b": [0.0, 0.7]}
    assert configs["a"] is not configs["b"]
    assert all(demo.raw["attempt"] == 2 for demo in demos.values())
    assert os.environ["TEACHER_TEMP"] == "do-not-touch"


def test_teacher_request_temperature_overrides_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import appworld_teacher

    monkeypatch.setenv("TEACHER_TEMP", "not-a-number")
    config = appworld_teacher.TeacherConfig(
        name="teacher",
        model="teacher-model",
        endpoint="https://example.test/v1/chat/completions",
        api_key="secret",
    )
    body = appworld_teacher._build_request_body(
        config,
        [{"role": "user", "content": "hello"}],
        temperature=0.7,
    )
    assert body["temperature"] == 0.7
