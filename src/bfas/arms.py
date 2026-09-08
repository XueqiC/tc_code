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


def pair_unit_margins(logp_plus, base_plus, logp_minus=None, base_minus=None,
                      *, beta: float = 0.1):
    """Sequence-summed, base-centred margins; absent rejects contribute zero.

    Tensors have a final dimension of two (the two states). For mixed missing
    rejects, callers supply zero policy/base log-probs at the missing positions.
    """
    import math

    if not math.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be finite and positive")
    if (logp_minus is None) != (base_minus is None):
        raise ValueError("rejected policy and base log-probs must be supplied together")
    delta = logp_plus - base_plus
    if logp_minus is not None:
        delta = delta - (logp_minus - base_minus)
    return beta * delta


def pair_unit_loss(margins, *, gamma: float = 1.0, reduction: str = "worst",
                   side_weights=None):
    """Return one squared-hinge loss per pair, without reducing the batch."""
    import math

    if not math.isfinite(gamma) or gamma < 0:
        raise ValueError("gamma must be finite and non-negative")
    if margins.ndim == 0 or margins.shape[-1] != 2:
        raise ValueError("pair_unit requires exactly two side margins")
    losses = (gamma - margins).clamp_min(0).square()
    if side_weights is not None:
        losses = losses * side_weights
    if reduction == "worst":
        return losses.amax(dim=-1)
    if reduction == "sum":
        return losses.sum(dim=-1)
    raise ValueError("pair_unit reduction must be 'worst' or 'sum'")


def pair_side_rows(pairs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Flatten normalized CC pairs without changing any prompt or label."""
    rows = []
    for pair in pairs:
        for index, side in enumerate(pair["sides"]):
            row = dict(side)
            row.update(response=side["y_plus"],
                       _cc_pair_id=pair["pair_id"], _cc_side=index,
                       _cc_type=pair["type"],
                       _cc_seed_function=side["seed_function"])
            row.setdefault("task_id", f"{pair['pair_id']}:{index}")
            row.setdefault("teacher", "historical_teacher")
            row.setdefault("turn_index", 0)
            row.setdefault("token_hint", max(len(side["y_plus"]) // 4, 1))
            row.pop("_rejected", None)
            if side.get("y_minus"):
                row["_rejected"] = side["y_minus"]
            rows.append(row)
    return rows


def shuffle_pair_sides(pairs: Sequence[Mapping[str, Any]], *, seed: int = 0
                       ) -> list[dict[str, Any]]:
    """Permute entire right sides within type, preferring cross-seed matching.

    First require different seed functions and original pair IDs. If no perfect
    matching exists, retry the entire type with only different original pair
    IDs required. Neither attempt duplicates, drops, or changes any side.
    """
    import copy
    import random

    rng = random.Random(seed)
    result = copy.deepcopy(list(pairs))
    groups: dict[str, list[int]] = {}
    for i, pair in enumerate(pairs):
        groups.setdefault(str(pair["type"]), []).append(i)
    for group, members in sorted(groups.items()):
        for different_seed in (True, False):
            edges = {}
            for i in members:
                edges[i] = [j for j in members
                            if pairs[i]["pair_id"] != pairs[j]["pair_id"]
                            and (not different_seed or
                                 pairs[i]["sides"][0]["seed_function"] !=
                                 pairs[j]["sides"][1]["seed_function"])]
                rng.shuffle(edges[i])
            owner: dict[int, int] = {}

            def match(i, seen):
                for j in edges[i]:
                    if j in seen:
                        continue
                    seen.add(j)
                    if j not in owner or match(owner[j], seen):
                        owner[j] = i
                        return True
                return False

            order = list(members)
            rng.shuffle(order)
            if all(match(i, set()) for i in order):
                break
        else:
            raise ValueError(f"type {group!r}: no complete different-pair-id derangement exists")
        for j, i in owner.items():
            result[i]["sides"][1] = copy.deepcopy(pairs[j]["sides"][1])
            result[i]["source_pair_ids"] = [pairs[i]["pair_id"], pairs[j]["pair_id"]]
            result[i]["pairing"] = "shuffled"
    return result


def same_seed_pair_counts(pairs: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Count same-seed side pairings per type, including types with zero counts."""
    counts: dict[str, int] = {}
    for pair in pairs:
        group = str(pair["type"])
        counts[group] = counts.get(group, 0) + int(
            pair["sides"][0]["seed_function"] == pair["sides"][1]["seed_function"])
    return counts
