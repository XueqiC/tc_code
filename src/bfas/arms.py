"""Benchmark-independent training-pool transforms for BFAS arms."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .adapter import BenchmarkAdapter, Demo, Rollout, Turn


ARM_NAMES = (
    "ours",
    "sft",
    "sad",
    "agentkd",
    "ddpo",
    "pbsd",
    "bbopd",
    "star",
    "base",
)

DISTILL_MODES = {
    "sad": "sad",
    "agentkd": "agentkd",
    "ddpo": "ddpo",
    "pbsd": "pbsd",
}


def _mu_for_turn(rollout: Rollout, turn_index: int) -> dict[str, Any]:
    if not isinstance(rollout.raw, Mapping):
        return {}
    fields = rollout.raw.get("mu_fields")
    if isinstance(fields, Sequence) and not isinstance(fields, (str, bytes)):
        fields = fields[turn_index] if turn_index < len(fields) else None
    if not isinstance(fields, Mapping):
        return {}
    out = {}
    if fields.get("_mu_nll_sum") is not None:
        out["_mu_nll_sum"] = float(fields["_mu_nll_sum"])
        out["_mu_ntok"] = int(fields["_mu_ntok"])
    return out


def _row(
    task_id: str,
    category: str,
    teacher: str,
    turn_index: int,
    turn: Turn,
    **extra: Any,
) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "category": category,
        "teacher": teacher,
        "turn_index": turn_index,
        "prompt": turn.prompt,
        "response": turn.target,
        "token_hint": max(len(turn.target) // 4, 1),
        "_render_context": turn.context,
        **extra,
    }


def _rollout_rows(
    rollouts: Sequence[Rollout],
    categories: Mapping[str, str],
    teacher: str,
    p_hats: Mapping[str, float] | None = None,
    guided: bool = False,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rollout_index, rollout in enumerate(rollouts):
        if not rollout.verified:
            continue
        raw = rollout.raw if isinstance(rollout.raw, Mapping) else {}
        if raw.get("training_excluded"):
            continue
        source_turns = rollout.turns
        if guided and isinstance(raw.get("deployment_turns"), Sequence):
            source_turns = raw["deployment_turns"]
        for turn_index, turn in enumerate(source_turns):
            extra: dict[str, Any] = {
                "_traj": f"{rollout.task_id}#{'g' if guided else 'u'}{rollout_index}"
            }
            if p_hats is not None:
                extra["_task_phat"] = float(p_hats[rollout.task_id])
            if guided:
                extra["_guided"] = True
                extra.update(_mu_for_turn(rollout, turn_index))
            rows.append(
                _row(
                    rollout.task_id,
                    categories[rollout.task_id],
                    teacher,
                    turn_index,
                    turn,
                    **extra,
                )
            )
    return rows


def _demo_rows(
    demos: Mapping[str, Demo], categories: Mapping[str, str], teacher: str = "teacher"
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task_id, demo in sorted(demos.items()):
        for turn_index, turn in enumerate(demo.turns):
            rows.append(
                _row(task_id, categories[task_id], teacher, turn_index, turn)
            )
    return rows


def _first_failed_targets(rollouts: Sequence[Rollout]) -> dict[str, str]:
    rejected: dict[str, str] = {}
    for rollout in rollouts:
        if rollout.verified:
            continue
        for turn in reversed(rollout.turns):
            if turn.target:
                rejected.setdefault(rollout.task_id, turn.target)
                break
    return rejected


def build_pool(
    arm: str,
    adapter: BenchmarkAdapter,
    *,
    teacher_demos: Mapping[str, Demo],
    unguided_rollouts: Sequence[Rollout],
    guided_rollouts: Sequence[Rollout],
    p_hats: Mapping[str, float],
) -> list[dict[str, Any]]:
    if arm not in ARM_NAMES:
        raise ValueError(f"unknown arm {arm!r}; expected one of {ARM_NAMES}")
    if arm == "base":
        return []
    categories = adapter.task_categories()
    if arm == "ours":
        return _rollout_rows(
            unguided_rollouts, categories, "self", p_hats
        ) + _rollout_rows(
            guided_rollouts, categories, "self", p_hats, guided=True
        )
    if arm == "star":
        return _rollout_rows(unguided_rollouts, categories, "self")

    rows = _demo_rows(teacher_demos, categories)
    if arm in {"ddpo", "pbsd"}:
        rejected = _first_failed_targets(unguided_rollouts)
        for row in rows:
            if row["task_id"] in rejected:
                row["_rejected"] = rejected[row["task_id"]]
    if arm == "agentkd":
        for row in rows:
            demo = teacher_demos[row["task_id"]]
            raw = demo.raw if isinstance(demo.raw, Mapping) else {}
            thought = raw.get("thought")
            if isinstance(thought, str) and thought.strip():
                row["_thought"] = thought
    return rows


def trainer_environment(arm: str) -> dict[str, str]:
    env = {"AW_LAMBDA_PREF": "0"}
    if arm == "ours":
        env["AW_V3"] = "1"
    if arm in DISTILL_MODES:
        env["AW_DISTILL"] = DISTILL_MODES[arm]
    return env
