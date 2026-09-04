"""Shared BFAS protocol constants and deterministic sampling rules."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any


SUPPORT_FRACTION = 0.05
SUPPORT_MIN = 50
SUPPORT_MAX = 250
CALIB_FRACTION = 0.2
SUPPORT_SEED = 50
TEACHER_ATTEMPTS = 3
ROUNDS_CAP = 8
SAMPLING_TEMPERATURE = 0.7
POSTERIOR_LOW = 0.15
POSTERIOR_HIGH = 0.85
SEEDS = (0, 1, 2)


@dataclass(frozen=True)
class SupportSplit:
    support: tuple[str, ...]
    demand: tuple[str, ...]
    calibration: tuple[str, ...]

    def as_dict(self) -> dict[str, list[str] | int]:
        return {
            "support": list(self.support),
            "demand": list(self.demand),
            "calibration": list(self.calibration),
            "seed": SUPPORT_SEED,
        }


def support_size(pool_size: int) -> int:
    if pool_size < 0:
        raise ValueError("pool_size must be nonnegative")
    requested = math.floor(pool_size * SUPPORT_FRACTION)
    return min(pool_size, min(max(requested, SUPPORT_MIN), SUPPORT_MAX))


def calibration_size(support_count: int) -> int:
    if support_count < 0:
        raise ValueError("support_count must be nonnegative")
    return min(support_count, math.floor(support_count * CALIB_FRACTION))


def _task_fields(task: Any) -> tuple[str, str]:
    if hasattr(task, "task_id") and hasattr(task, "category"):
        return str(task.task_id), str(task.category)
    if isinstance(task, Mapping):
        return str(task["task_id"]), str(task["category"])
    task_id, category = task
    return str(task_id), str(category)


def _allocation(
    groups: Mapping[str, Sequence[str]],
    count: int,
    coverage_floor: bool = False,
) -> dict[str, int]:
    total = sum(len(ids) for ids in groups.values())
    if count > total:
        raise ValueError(f"cannot sample {count} tasks from a pool of {total}")
    if not total or not count:
        return {category: 0 for category in groups}
    if coverage_floor:
        # The support set exists to measure where the student is short, not to
        # mirror how many tasks a category happens to ship with.  Size-weighted
        # sampling hands most of the budget to the categories already covered
        # and leaves the small ones with too few tasks to estimate anything --
        # a category that lands zero verified rows then loses its whole axis
        # silently.  Give every category an equal share first, then spread the
        # remainder by size.
        floor = count // len(groups)
        reserved = {
            category: min(floor, len(ids)) for category, ids in groups.items()
        }
        remaining = count - sum(reserved.values())
        headroom = {
            category: list(ids)[reserved[category]:] for category, ids in groups.items()
        }
        extra = _allocation(headroom, remaining) if remaining else {}
        return {
            category: reserved[category] + extra.get(category, 0)
            for category in groups
        }
    quotas = {category: count * len(ids) / total for category, ids in groups.items()}
    allocated = {category: min(math.floor(quota), len(groups[category]))
                 for category, quota in quotas.items()}
    order = sorted(
        groups,
        key=lambda category: (-(quotas[category] - allocated[category]), category),
    )
    while sum(allocated.values()) < count:
        progressed = False
        for category in order:
            if allocated[category] < len(groups[category]):
                allocated[category] += 1
                progressed = True
                if sum(allocated.values()) == count:
                    break
        if not progressed:
            raise RuntimeError("stratified allocation could not satisfy sample size")
    return allocated


def stratified_sample(
    tasks: Iterable[Any],
    count: int,
    seed: int = SUPPORT_SEED,
    coverage_floor: bool = False,
) -> tuple[str, ...]:
    groups: dict[str, list[str]] = defaultdict(list)
    seen: set[str] = set()
    for task in tasks:
        task_id, category = _task_fields(task)
        if task_id in seen:
            raise ValueError(f"duplicate task id: {task_id}")
        seen.add(task_id)
        groups[category].append(task_id)
    for ids in groups.values():
        ids.sort()
    allocated = _allocation(groups, count, coverage_floor)
    rng = random.Random(seed)
    chosen: list[str] = []
    for category in sorted(groups):
        chosen.extend(rng.sample(groups[category], allocated[category]))
    rng.shuffle(chosen)
    return tuple(chosen)


def make_support_split(
    tasks: Iterable[Any],
    seed: int = SUPPORT_SEED,
    coverage_floor: bool = False,
) -> SupportSplit:
    task_list = list(tasks)
    by_id = {task_id: category for task_id, category in map(_task_fields, task_list)}
    support = stratified_sample(
        task_list, support_size(len(task_list)), seed, coverage_floor
    )
    support_refs = [(task_id, by_id[task_id]) for task_id in support]
    calibration = stratified_sample(
        support_refs, calibration_size(len(support)), seed, coverage_floor
    )
    calibration_set = set(calibration)
    demand = tuple(task_id for task_id in support if task_id not in calibration_set)
    return SupportSplit(
        support=tuple(sorted(support)),
        demand=tuple(sorted(demand)),
        calibration=tuple(sorted(calibration)),
    )


@dataclass
class BetaPosterior:
    successes: int = 0
    failures: int = 0

    @property
    def rounds(self) -> int:
        return self.successes + self.failures

    @property
    def mean(self) -> float:
        return (self.successes + 1) / (self.rounds + 2)

    def observe(self, verified: bool) -> None:
        if verified:
            self.successes += 1
        else:
            self.failures += 1

    def should_sample(self) -> bool:
        if self.rounds >= ROUNDS_CAP:
            return False
        if self.rounds < 3:
            return True
        if self.successes == self.rounds:
            return False
        if self.successes == 0:
            return True
        return POSTERIOR_LOW < self.mean < POSTERIOR_HIGH


class AdaptiveSampler:
    def __init__(self, task_ids: Iterable[str]):
        ids = tuple(dict.fromkeys(map(str, task_ids)))
        self._posteriors = {task_id: BetaPosterior() for task_id in ids}

    @property
    def posteriors(self) -> Mapping[str, BetaPosterior]:
        return self._posteriors

    def active(self) -> list[str]:
        return [
            task_id
            for task_id, posterior in self._posteriors.items()
            if posterior.should_sample()
        ]

    def observe(self, task_id: str, verified: bool) -> None:
        if task_id not in self._posteriors:
            raise KeyError(f"unknown task id: {task_id}")
        self._posteriors[task_id].observe(bool(verified))

    def p_hats(self) -> dict[str, float]:
        return {
            task_id: posterior.mean
            for task_id, posterior in self._posteriors.items()
        }

    def rounds(self) -> dict[str, int]:
        return {
            task_id: posterior.rounds
            for task_id, posterior in self._posteriors.items()
        }
