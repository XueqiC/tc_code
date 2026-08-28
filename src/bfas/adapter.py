"""Benchmark adapter contract for the unified BFAS pipeline."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, NamedTuple

from .protocol import SupportSplit, make_support_split


TargetPolicy = Literal["call_required", "prose_ok"]
PolicyRef = str | Path


class TaskRef(NamedTuple):
    task_id: str
    category: str


@dataclass(frozen=True)
class Turn:
    prompt: str
    target: str
    context: Any = field(default=None, compare=False, repr=False)


@dataclass(frozen=True)
class Rollout:
    task_id: str
    verified: bool
    turns: tuple[Turn, ...] | list[Turn]
    raw: Any = None


@dataclass(frozen=True)
class Demo:
    task_id: str
    turns: tuple[Turn, ...] | list[Turn]
    worked_example: str
    raw: Any = None

    @property
    def verified(self) -> bool:
        return True


class BenchmarkAdapter(ABC):
    name: str

    @abstractmethod
    def task_pool(self) -> list[TaskRef]:
        raise NotImplementedError

    @abstractmethod
    def official_eval_split_disjoint(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def rollout(
        self,
        policy: PolicyRef,
        task_ids: Sequence[str],
        temperature: float,
        guided_demos: Mapping[str, Demo] | None = None,
    ) -> list[Rollout]:
        raise NotImplementedError

    @abstractmethod
    def teacher_demo(
        self, task_ids: Sequence[str], attempts: int
    ) -> dict[str, Demo]:
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, policy_ref: PolicyRef, out_dir: Path) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def serving_probe(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def generation_suffix(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def target_policy(self, category: str) -> TargetPolicy:
        raise NotImplementedError

    def support_split(self) -> SupportSplit:
        return make_support_split(self.task_pool())

    def task_categories(self) -> dict[str, str]:
        return {task.task_id: task.category for task in self.task_pool()}

    def rerender(self, row: Mapping[str, Any]) -> str:
        raise NotImplementedError(f"{type(self).__name__} does not implement rerender")

    def is_call_target(self, target: str, category: str) -> bool:
        stripped = target.strip()
        return (
            "<tool_call>" in stripped
            or stripped.startswith("```python")
            or stripped.startswith("```py")
            or stripped.startswith("<code")
        )

    def checker_summary(
        self, rollouts: Sequence[Rollout]
    ) -> dict[str, dict[str, int]]:
        categories = self.task_categories()
        total: Counter[str] = Counter()
        verified: Counter[str] = Counter()
        for rollout in rollouts:
            category = categories[rollout.task_id]
            total[category] += 1
            raw = rollout.raw if isinstance(rollout.raw, Mapping) else {}
            if raw.get("checker_verified") is True:
                verified[category] += 1
        return {
            category: {"verified": verified[category], "total": count}
            for category, count in total.items()
        }
