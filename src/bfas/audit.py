"""Pre-training conformance audits for BFAS pools."""

from __future__ import annotations

import json
import math
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .adapter import BenchmarkAdapter, Rollout
from .protocol import SUPPORT_SEED


class AuditError(AssertionError):
    pass


@dataclass(frozen=True)
class AuditReport:
    row_count: int
    checked_round_trips: int
    should_train: bool
    decision_trace: dict[str, Any] | None = None


def _fail(message: str) -> None:
    raise AuditError(message)


def _target(row: Mapping[str, Any], index: int) -> str:
    value = row.get("response", row.get("target"))
    if not isinstance(value, str) or not value:
        _fail(f"row {index}: target/response must be a non-empty string")
    return value


def _category(
    row: Mapping[str, Any], index: int, categories: Mapping[str, str]
) -> str:
    category = row.get("category")
    if isinstance(category, str) and category:
        return category
    task_id = str(row.get("task_id", ""))
    if task_id not in categories:
        _fail(f"row {index}: no category for task {task_id!r}")
    return categories[task_id]


def audit_verified_counts(
    rollouts: Sequence[Rollout],
    task_categories: Mapping[str, str],
    checker_summary: Mapping[str, int | Mapping[str, int]],
) -> None:
    actual_total: Counter[str] = Counter()
    actual_verified: Counter[str] = Counter()
    for rollout in rollouts:
        if rollout.task_id not in task_categories:
            _fail(f"rollout has unknown task id {rollout.task_id!r}")
        category = task_categories[rollout.task_id]
        actual_total[category] += 1
        actual_verified[category] += int(rollout.verified)

    all_categories = set(actual_total) | set(checker_summary)
    for category in sorted(all_categories):
        expected = checker_summary.get(category, 0)
        if isinstance(expected, Mapping):
            expected_verified = int(expected.get("verified", expected.get("correct", 0)))
            if "total" in expected and int(expected["total"]) != actual_total[category]:
                _fail(
                    f"verified count audit: {category} total "
                    f"{actual_total[category]} != checker {int(expected['total'])}"
                )
        else:
            expected_verified = int(expected)
        if actual_verified[category] != expected_verified:
            _fail(
                f"verified count audit: {category} verified "
                f"{actual_verified[category]} != checker {expected_verified}"
            )


def audit_pool(
    rows: Sequence[Mapping[str, Any]],
    adapter: BenchmarkAdapter,
    *,
    rollouts: Sequence[Rollout] = (),
    checker_summary: Mapping[str, int | Mapping[str, int]] | None = None,
) -> AuditReport:
    categories = adapter.task_categories()
    suffix = adapter.generation_suffix()
    if not isinstance(suffix, str) or not suffix:
        _fail("adapter generation suffix must be non-empty")

    for index, row in enumerate(rows):
        if "messages" in row:
            _fail(f"row {index}: forbidden messages key")
        prompt = row.get("prompt")
        if not isinstance(prompt, str) or not prompt.endswith(suffix):
            _fail(f"row {index}: prompt does not end with generation suffix {suffix!r}")
        category = _category(row, index, categories)
        target = _target(row, index)
        policy = adapter.target_policy(category)
        if policy not in {"call_required", "prose_ok"}:
            _fail(f"row {index}: invalid target policy {policy!r}")
        if policy == "call_required" and not adapter.is_call_target(target, category):
            _fail(f"row {index}: category {category!r} requires a call target")
        if row.get("_guided") is True:
            if row.get("_mu_nll_sum") is None or row.get("_mu_ntok") is None:
                _fail(f"row {index}: guided row lacks behavior-policy mu fields")
            if int(row["_mu_ntok"]) <= 0:
                _fail(f"row {index}: guided row has nonpositive _mu_ntok")

    sample_count = min(3, len(rows))
    sampled = random.Random(SUPPORT_SEED).sample(range(len(rows)), sample_count)
    for index in sampled:
        try:
            rendered = adapter.rerender(rows[index])
        except Exception as exc:
            _fail(f"row {index}: serving renderer failed: {type(exc).__name__}: {exc}")
        if rendered != rows[index]["prompt"]:
            _fail(f"row {index}: serving renderer round-trip mismatch")

    has_positive_advantage = False
    for index, row in enumerate(rows):
        raw_p_hat = row.get("_task_phat")
        if raw_p_hat is None:
            has_positive_advantage = True
            continue
        try:
            p_hat = float(raw_p_hat)
        except (TypeError, ValueError):
            _fail(f"row {index}: invalid p-hat {raw_p_hat!r}")
        if not math.isfinite(p_hat) or not 0.0 <= p_hat <= 1.0:
            _fail(f"row {index}: p-hat outside [0,1]: {raw_p_hat!r}")
        has_positive_advantage |= 1.0 - p_hat > 0.0

    if checker_summary is not None:
        audit_verified_counts(rollouts, categories, checker_summary)

    trace = None
    if not rows or not has_positive_advantage:
        trace = {
            "decision": "zero_update",
            "reason": "no_positive_advantage",
            "row_count": len(rows),
        }
    return AuditReport(
        row_count=len(rows),
        checked_round_trips=sample_count,
        should_train=has_positive_advantage and bool(rows),
        decision_trace=trace,
    )


def write_decision_trace(path: Path, report: AuditReport) -> None:
    if report.decision_trace is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.decision_trace, indent=2) + "\n")
