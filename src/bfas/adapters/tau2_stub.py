"""Planned tau2 adapter.

Phase 3 must represent both the policy and the benchmark-controlled user
simulator in each rollout, preserve simulator state in turn contexts, and
charge teacher use on both sides before this adapter can satisfy BFAS.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ..adapter import BenchmarkAdapter, Demo, PolicyRef, Rollout, TaskRef


class Tau2Adapter(BenchmarkAdapter):
    name = "tau2"

    @staticmethod
    def _deferred() -> None:
        raise NotImplementedError(
            "tau2 is deferred to phase 3: the user simulator, stateful turn "
            "contexts, and two-sided teacher accounting are not implemented"
        )

    def task_pool(self) -> list[TaskRef]:
        self._deferred()

    def official_eval_split_disjoint(self) -> bool:
        self._deferred()

    def rollout(
        self,
        policy: PolicyRef,
        task_ids: Sequence[str],
        temperature: float,
        guided_demos: Mapping[str, Demo] | None = None,
    ) -> list[Rollout]:
        self._deferred()

    def teacher_demo(
        self, task_ids: Sequence[str], attempts: int
    ) -> dict[str, Demo]:
        self._deferred()

    def evaluate(self, policy_ref: PolicyRef, out_dir: Path) -> dict[str, Any]:
        self._deferred()

    def serving_probe(self) -> None:
        self._deferred()

    def generation_suffix(self) -> str:
        self._deferred()

    def target_policy(self, category: str) -> str:
        self._deferred()
