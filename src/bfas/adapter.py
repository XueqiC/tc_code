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


@dataclass(frozen=True)
class TeacherEpisode:
    """The result of one paid teacher attempt."""

    task_id: str
    verified: bool
    demo: Demo | None
    response_texts: tuple[str, ...] | list[str] = ()
    tokens_spent: int | None = None
    # Bind accounting to the actual request config, not a later env lookup.
    teacher: str | None = None
    # Sums of API counters, or conservative bounds when usage_status=estimated.
    usage: Mapping[str, int] | None = None
    usage_status: str | None = None


COLLECTION_SCHEMA_VERSION = 1
_DATACLASS_TAG = "__bfas_dataclass__"


@dataclass(frozen=True)
class CollectionArtifacts:
    """Arm-independent artifacts produced once for a benchmark seed."""

    unguided_rollouts: tuple[Rollout, ...]
    p_hats: dict[str, float]
    unguided_rounds: dict[str, int]
    teacher_demos: dict[str, Demo]
    guided_rollouts: tuple[Rollout, ...]
    guided_rounds: dict[str, int]
    checker_summary: dict[str, dict[str, int]]

    def __post_init__(self) -> None:
        for rollout in self.guided_rollouts:
            raw = rollout.raw if isinstance(rollout.raw, Mapping) else {}
            if not (
                rollout.verified
                and rollout.turns
                and raw.get("deployment_turns")
            ):
                continue
            fields = raw.get("mu_fields")
            if (
                not isinstance(fields, Sequence)
                or isinstance(fields, (str, bytes))
                or len(fields) != len(rollout.turns)
                or any(
                    not isinstance(item, Mapping)
                    or "_mu_nll_sum" not in item
                    or "_mu_ntok" not in item
                    for item in fields
                )
            ):
                raise ValueError(
                    f"verified guided rollout {rollout.task_id!r} lacks complete mu_fields"
                )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": COLLECTION_SCHEMA_VERSION,
            "unguided_rollouts": _encode_collection_value(self.unguided_rollouts),
            "p_hats": dict(self.p_hats),
            "unguided_rounds": dict(self.unguided_rounds),
            "teacher_demos": _encode_collection_value(self.teacher_demos),
            "guided_rollouts": _encode_collection_value(self.guided_rollouts),
            "guided_rounds": dict(self.guided_rounds),
            "checker_summary": {
                category: dict(summary)
                for category, summary in self.checker_summary.items()
            },
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CollectionArtifacts":
        if value.get("schema_version") != COLLECTION_SCHEMA_VERSION:
            raise ValueError(
                "unsupported BFAS collection cache schema "
                f"{value.get('schema_version')!r}"
            )
        unguided = _rollout_tuple(
            _decode_collection_value(value.get("unguided_rollouts")),
            "unguided_rollouts",
        )
        guided = _rollout_tuple(
            _decode_collection_value(value.get("guided_rollouts")),
            "guided_rollouts",
        )
        demos_value = _decode_collection_value(value.get("teacher_demos"))
        if not isinstance(demos_value, Mapping) or not all(
            isinstance(task_id, str) and isinstance(demo, Demo)
            for task_id, demo in demos_value.items()
        ):
            raise ValueError("teacher_demos must be a mapping of Demo objects")
        return cls(
            unguided_rollouts=unguided,
            p_hats=_number_mapping(value.get("p_hats"), float, "p_hats"),
            unguided_rounds=_number_mapping(
                value.get("unguided_rounds"), int, "unguided_rounds"
            ),
            teacher_demos=dict(demos_value),
            guided_rollouts=guided,
            guided_rounds=_number_mapping(
                value.get("guided_rounds"), int, "guided_rounds"
            ),
            checker_summary=_checker_summary(value.get("checker_summary")),
        )


def _encode_collection_value(value: Any) -> Any:
    if isinstance(value, Turn):
        return {
            _DATACLASS_TAG: "Turn",
            "prompt": value.prompt,
            "target": value.target,
            "context": _encode_collection_value(value.context),
        }
    if isinstance(value, Rollout):
        return {
            _DATACLASS_TAG: "Rollout",
            "task_id": value.task_id,
            "verified": value.verified,
            "turns": _encode_collection_value(tuple(value.turns)),
            "raw": _encode_collection_value(value.raw),
        }
    if isinstance(value, Demo):
        return {
            _DATACLASS_TAG: "Demo",
            "task_id": value.task_id,
            "turns": _encode_collection_value(tuple(value.turns)),
            "worked_example": value.worked_example,
            "raw": _encode_collection_value(value.raw),
        }
    if isinstance(value, tuple):
        return {
            _DATACLASS_TAG: "tuple",
            "items": [_encode_collection_value(item) for item in value],
        }
    if isinstance(value, list):
        return [_encode_collection_value(item) for item in value]
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("collection cache mappings must have string keys")
        return {key: _encode_collection_value(item) for key, item in value.items()}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"collection cache cannot encode {type(value).__name__}")


def _decode_collection_value(value: Any) -> Any:
    if isinstance(value, list):
        return [_decode_collection_value(item) for item in value]
    if not isinstance(value, Mapping):
        return value
    tag = value.get(_DATACLASS_TAG)
    if tag is None:
        return {key: _decode_collection_value(item) for key, item in value.items()}
    if tag == "tuple":
        items = value.get("items")
        if not isinstance(items, list):
            raise ValueError("cached tuple has no items list")
        return tuple(_decode_collection_value(item) for item in items)
    if tag == "Turn":
        return Turn(
            prompt=str(value["prompt"]),
            target=str(value["target"]),
            context=_decode_collection_value(value.get("context")),
        )
    if tag == "Rollout":
        turns = _turn_tuple(_decode_collection_value(value.get("turns")), "rollout")
        return Rollout(
            task_id=str(value["task_id"]),
            verified=bool(value["verified"]),
            turns=turns,
            raw=_decode_collection_value(value.get("raw")),
        )
    if tag == "Demo":
        turns = _turn_tuple(_decode_collection_value(value.get("turns")), "demo")
        return Demo(
            task_id=str(value["task_id"]),
            turns=turns,
            worked_example=str(value["worked_example"]),
            raw=_decode_collection_value(value.get("raw")),
        )
    raise ValueError(f"unknown BFAS collection dataclass tag {tag!r}")


def _turn_tuple(value: Any, field_name: str) -> tuple[Turn, ...]:
    if not isinstance(value, (list, tuple)) or not all(
        isinstance(turn, Turn) for turn in value
    ):
        raise ValueError(f"{field_name} turns must contain Turn objects")
    return tuple(value)


def _rollout_tuple(value: Any, field_name: str) -> tuple[Rollout, ...]:
    if not isinstance(value, (list, tuple)) or not all(
        isinstance(rollout, Rollout) for rollout in value
    ):
        raise ValueError(f"{field_name} must contain Rollout objects")
    return tuple(value)


def _number_mapping(
    value: Any, converter: type[float] | type[int], field_name: str
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not all(
        isinstance(key, str) for key in value
    ):
        raise ValueError(f"{field_name} must be a string-keyed mapping")
    try:
        return {key: converter(item) for key, item in value.items()}
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} contains a non-numeric value") from exc


def _checker_summary(value: Any) -> dict[str, dict[str, int]]:
    if not isinstance(value, Mapping):
        raise ValueError("checker_summary must be a mapping")
    output: dict[str, dict[str, int]] = {}
    for category, summary in value.items():
        if not isinstance(category, str) or not isinstance(summary, Mapping):
            raise ValueError("checker_summary contains an invalid category")
        try:
            output[category] = {
                "verified": int(summary["verified"]),
                "total": int(summary["total"]),
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("checker_summary contains an invalid count") from exc
    return output


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

    def teacher_episode(
        self, task_id: str, attempt_index: int, temperature: float
    ) -> TeacherEpisode:
        """Purchase one episode; adapters may override for exact accounting.

        The compatibility implementation keeps older adapters usable.  It can
        only estimate usage for verified episodes, because the legacy batch
        API discards failed response text.
        """

        del attempt_index, temperature
        demo = self.teacher_demo([task_id], 1).get(task_id)
        return TeacherEpisode(
            task_id=task_id,
            verified=demo is not None,
            demo=demo,
            response_texts=(
                tuple(turn.target for turn in demo.turns) if demo is not None else ()
            ),
        )

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

    def prepare_renderer(self, policy_ref: PolicyRef) -> None:
        """Load CPU-side rendering state needed after a collection-cache hit."""

        del policy_ref

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
