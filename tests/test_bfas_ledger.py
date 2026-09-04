from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bfas.adapter import (  # noqa: E402
    BenchmarkAdapter,
    CollectionArtifacts,
    Demo,
    Rollout,
    TaskRef,
    TeacherEpisode,
    Turn,
)
from bfas import ledger  # noqa: E402
from bfas import run as bfas_run  # noqa: E402


def _demo(task_id: str, target: str = "open door") -> Demo:
    return Demo(
        task_id,
        (Turn("teacher prompt", target, [{"role": "user"}]),),
        target,
        {"checker_verified": True},
    )


class EpisodeAdapter(BenchmarkAdapter):
    name = "synthetic"
    needs_server = False

    def __init__(self, outcomes: list[bool] | None = None):
        self.outcomes = list(outcomes or [])
        self.episode_calls: list[tuple[str, int, float]] = []
        self.renderer_calls = 0

    def task_pool(self) -> list[TaskRef]:
        return [TaskRef("task", "tasks")]

    def official_eval_split_disjoint(self) -> bool:
        return True

    def rollout(self, policy, task_ids, temperature, guided_demos=None):
        return [Rollout(task_id, False, ()) for task_id in task_ids]

    def teacher_demo(self, task_ids, attempts):
        raise AssertionError("the ledger gateway must use one-episode purchases")

    def teacher_episode(self, task_id, attempt_index, temperature):
        self.episode_calls.append((task_id, attempt_index, temperature))
        verified = self.outcomes.pop(0)
        demo = _demo(task_id) if verified else None
        return TeacherEpisode(
            task_id,
            verified,
            demo,
            response_texts=("abcdefgh",),
        )

    def prepare_renderer(self, policy_ref) -> None:
        self.renderer_calls += 1

    def evaluate(self, policy_ref, out_dir):
        return {}

    def serving_probe(self) -> None:
        return None

    def generation_suffix(self) -> str:
        return "<GEN>"

    def target_policy(self, category: str) -> str:
        return "prose_ok"


def test_ledger_append_and_compact(tmp_path: Path) -> None:
    path = tmp_path / "alfworld.jsonl"
    ledger.append_episode(
        path,
        task_id="task",
        teacher="teacher-v1",
        attempt_index=0,
        temperature=0.0,
        verified=False,
        tokens_spent=3,
        timestamp="2026-08-29T00:00:00Z",
    )
    ledger.append_episode(
        path,
        task_id="task",
        teacher="teacher-v1",
        attempt_index=1,
        temperature=0.7,
        verified=True,
        tokens_spent=5,
        demo=_demo("task"),
        timestamp="2026-08-29T00:00:01Z",
    )

    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 2
    assert "demo" not in rows[0]
    assert set(rows[1]["demo"]) == {"turns", "worked_example"}
    state = ledger.load_ledger(path, attempts=3)["task"]
    assert state["attempts_used"] == 2
    assert state["tokens_total"] == 8
    assert state["infeasible"] is False
    assert state["best_demo"].worked_example == "open door"
    assert isinstance(state["best_demo"].turns[0], Turn)


def test_user_sim_usage_is_separate_from_teacher_attempts(tmp_path: Path) -> None:
    path = tmp_path / "tau2.jsonl"
    ledger.append_episode(
        path,
        task_id="airline:0",
        teacher="deepseek-v4-pro",
        attempt_index=0,
        temperature=0.0,
        verified=False,
        tokens_spent=17,
        purpose="user_sim",
    )
    ledger.append_episode(
        path,
        task_id="airline:0",
        teacher="deepseek-v4-pro",
        attempt_index=0,
        temperature=0.0,
        verified=False,
        tokens_spent=23,
    )

    rows = ledger.read_records(path)
    assert [row["purpose"] for row in rows] == ["user_sim", "teacher"]
    assert sum(row["tokens_spent"] for row in rows) == 40
    state = ledger.load_ledger(path, attempts=3)["airline:0"]
    assert state["attempts_used"] == 1
    assert state["tokens_total"] == 23


def test_acquire_skips_verified_without_teacher_call(tmp_path: Path) -> None:
    path = tmp_path / "verified.jsonl"
    ledger.append_episode(
        path,
        task_id="task",
        teacher="teacher-v1",
        attempt_index=0,
        temperature=0.0,
        verified=True,
        tokens_spent=4,
        demo=_demo("task"),
    )
    adapter = EpisodeAdapter()

    acquired = ledger.acquire_demos(path, adapter, ["task"], attempts=3)

    assert set(acquired) == {"task"}
    assert adapter.episode_calls == []
    assert len(path.read_text().splitlines()) == 1


def test_acquire_skips_infeasible_at_budget(tmp_path: Path) -> None:
    path = tmp_path / "infeasible.jsonl"
    for attempt_index in range(2):
        ledger.append_episode(
            path,
            task_id="task",
            teacher="teacher-v1",
            attempt_index=attempt_index,
            temperature=0.0 if attempt_index == 0 else 0.7,
            verified=False,
            tokens_spent=2,
        )
    adapter = EpisodeAdapter()

    acquired = ledger.acquire_demos(path, adapter, ["task"], attempts=2)

    assert acquired == {}
    assert acquired.infeasible == {"task"}
    assert acquired.states["task"] == {
        "best_demo": None,
        "attempts_used": 2,
        "tokens_total": 4,
        "infeasible": True,
    }
    assert adapter.episode_calls == []


def test_acquire_records_each_attempt_and_paces_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "paced.jsonl"
    adapter = EpisodeAdapter([False, True])
    sleeps: list[float] = []
    monkeypatch.setenv("BFAS_TEACHER_MIN_INTERVAL_S", "1.5")
    monkeypatch.setattr(ledger.time, "sleep", sleeps.append)

    acquired = ledger.acquire_demos(path, adapter, ["task"], attempts=3)

    assert set(acquired) == {"task"}
    assert adapter.episode_calls == [
        ("task", 0, 0.0),
        ("task", 1, 0.7),
    ]
    assert sleeps == [1.5]
    assert acquired.states["task"]["attempts_used"] == 2
    assert acquired.states["task"]["tokens_total"] == 4
    assert len(path.read_text().splitlines()) == 2


def test_shared_demo_phase_is_reused_across_seeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ledger, "LEDGER_ROOT", tmp_path / "data/teacher_ledger")
    monkeypatch.delenv("BFAS_TEACHER_MIN_INTERVAL_S", raising=False)
    shared_dir = tmp_path / "results/bfas/synthetic/collect_shared"
    seed_zero = EpisodeAdapter([True])
    seed_one = EpisodeAdapter()

    first = bfas_run._shared_teacher_demos(
        "synthetic", seed_zero, "policy", ["task"], shared_dir
    )
    second = bfas_run._shared_teacher_demos(
        "synthetic", seed_one, "policy", ["task"], shared_dir
    )

    assert set(first) == set(second) == {"task"}
    assert seed_zero.episode_calls == [("task", 0, 0.0)]
    assert seed_one.episode_calls == []
    assert seed_zero.renderer_calls == 1
    assert seed_one.renderer_calls == 0
    assert (shared_dir / "demos.json").is_file()
    assert not (shared_dir.parent / "collect_s1/demos.json").exists()


def test_shared_demo_phase_migrates_seed_zero_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ledger, "LEDGER_ROOT", tmp_path / "data/teacher_ledger")
    benchmark_dir = tmp_path / "results/bfas/synthetic"
    bfas_run._write_demos_phase(
        benchmark_dir / "collect_s0",
        {"task": _demo("task", "legacy")},
        {"task"},
        complete=True,
    )
    adapter = EpisodeAdapter()

    demos = bfas_run._shared_teacher_demos(
        "synthetic",
        adapter,
        "policy",
        ["task"],
        benchmark_dir / "collect_shared",
    )

    assert demos["task"].worked_example == "legacy"
    assert adapter.episode_calls == []
    rows = ledger.read_records("synthetic")
    assert len(rows) == 1
    assert rows[0]["teacher"] == "legacy-cache"
    assert rows[0]["tokens_spent"] == 0


def test_seed_collection_cache_does_not_embed_shared_demos(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    demo = _demo("task")
    artifacts = CollectionArtifacts(
        unguided_rollouts=(),
        p_hats={"task": 0.75},
        unguided_rounds={"task": 1},
        teacher_demos={"task": demo},
        guided_rollouts=(),
        guided_rounds={},
        checker_summary={},
    )
    monkeypatch.setattr(
        bfas_run, "_collect_seed_artifacts", lambda *args, **kwargs: artifacts
    )
    cache_dir = tmp_path / "results/bfas/synthetic/collect_s1"

    returned = bfas_run._cached_collection(
        EpisodeAdapter(),
        "policy",
        ["task"],
        "0",
        8900,
        cache_dir,
        teacher_demos={"task": demo},
    )

    assert returned.teacher_demos == {"task": demo}
    cached = json.loads((cache_dir / "collection.json").read_text())
    assert cached["teacher_demos"] == {}
    assert not (cache_dir / "demos.json").exists()
