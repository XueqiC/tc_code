"""Crash-safe accounting and acquisition gateway for teacher episodes."""

from __future__ import annotations

import fcntl
import json
import math
import os
import re
import time
from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from .adapter import BenchmarkAdapter, Demo, TeacherEpisode, Turn
from .protocol import SAMPLING_TEMPERATURE


ROOT = Path(__file__).resolve().parents[2]
LEDGER_ROOT = ROOT / "data/teacher_ledger"
_BENCHMARK_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_PURCHASED_LEDGERS: set[Path] = set()


class AcquisitionResult(dict[str, Demo]):
    """Verified demos plus the compact ledger state used to acquire them."""

    def __init__(
        self,
        demos: Mapping[str, Demo],
        states: Mapping[str, Mapping[str, Any]],
    ) -> None:
        super().__init__(demos)
        self.states = {task_id: dict(state) for task_id, state in states.items()}

    @property
    def infeasible(self) -> set[str]:
        return {
            task_id
            for task_id, state in self.states.items()
            if state["infeasible"] is True
        }


def ledger_path(
    benchmark: str | Path, *, ledger_root: Path | None = None
) -> Path:
    """Resolve a benchmark name (or an explicit JSONL path) to its ledger."""

    if isinstance(benchmark, Path) or str(benchmark).endswith(".jsonl"):
        return Path(benchmark)
    if not _BENCHMARK_RE.fullmatch(str(benchmark)):
        raise ValueError(f"invalid benchmark name {benchmark!r}")
    return (ledger_root or LEDGER_ROOT) / f"{benchmark}.jsonl"


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return repr(value)


def demo_payload(demo: Demo) -> dict[str, Any]:
    payload = {
        "turns": [
            {
                "prompt": turn.prompt,
                "target": turn.target,
                "context": _json_safe(turn.context),
            }
            for turn in demo.turns
        ],
        "worked_example": demo.worked_example,
    }
    if isinstance(demo.raw, Mapping) and "prompt_version" in demo.raw:
        payload["prompt_version"] = demo.raw["prompt_version"]
    return payload


def _demo_from_payload(task_id: str, value: Any, record: Mapping[str, Any]) -> Demo:
    if not isinstance(value, Mapping):
        raise ValueError("verified teacher-ledger record has no demo payload")
    turns_value = value.get("turns")
    if not isinstance(turns_value, list):
        raise ValueError("teacher-ledger demo turns must be a list")
    turns: list[Turn] = []
    for item in turns_value:
        if not isinstance(item, Mapping):
            raise ValueError("teacher-ledger demo contains an invalid turn")
        turns.append(Turn(
            prompt=str(item["prompt"]),
            target=str(item["target"]),
            context=item.get("context"),
        ))
    return Demo(
        task_id=task_id,
        turns=tuple(turns),
        worked_example=str(value["worked_example"]),
        raw={
            "attempt": int(record["attempt_index"]) + 1,
            "checker_verified": True,
            "teacher": str(record["teacher"]),
            "ledger_timestamp": str(record["timestamp"]),
            **({"prompt_version": record["prompt_version"]}
               if "prompt_version" in record else {}),
        },
    )


def append_record(path: Path, record: Mapping[str, Any]) -> None:
    """Append exactly one encoded JSON line with one O_APPEND write."""

    path.parent.mkdir(parents=True, exist_ok=True)
    line = (json.dumps(
        dict(record), ensure_ascii=False, separators=(",", ":")
    ) + "\n").encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    descriptor = os.open(path, flags, 0o644)
    try:
        written = os.write(descriptor, line)
        if written != len(line):
            raise OSError(
                f"partial teacher-ledger append: wrote {written}/{len(line)} bytes"
            )
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def append_episode(
    benchmark: str | Path,
    *,
    task_id: str,
    teacher: str,
    attempt_index: int,
    temperature: float,
    verified: bool,
    tokens_spent: int,
    usage: Mapping[str, int] | None = None,
    usage_status: str | None = None,
    purpose: str = "teacher",
    demo: Demo | None = None,
    timestamp: str | None = None,
    ledger_root: Path | None = None,
    prompt_version: str | None = None,
) -> dict[str, Any]:
    """Append an episode with optional counters and their provenance in ``usage``.

    ``completion_tokens``, ``prompt_tokens`` and ``cached_tokens`` are sums of
    API counters unless usage_status is ``estimated``; cached input is included
    in prompt_tokens. Older rows may omit counters/status and remain readable.
    """
    if verified and demo is None:
        raise ValueError("a verified ledger episode requires a Demo")
    if not verified and demo is not None:
        raise ValueError("an unverified ledger episode cannot contain a Demo")
    if attempt_index < 0:
        raise ValueError("attempt_index must be non-negative")
    if tokens_spent < 0:
        raise ValueError("tokens_spent must be non-negative")
    if not isinstance(purpose, str) or not purpose:
        raise ValueError("purpose must be a non-empty string")
    record: dict[str, Any] = {
        "task_id": task_id,
        "teacher": teacher,
        "attempt_index": attempt_index,
        "temperature": float(temperature),
        "verified": bool(verified),
        "tokens_spent": int(tokens_spent),
        "purpose": purpose,
        "timestamp": timestamp or _timestamp(),
    }
    if demo is not None:
        record["demo"] = demo_payload(demo)
        demo_version = record["demo"].get("prompt_version")
        if prompt_version is None:
            prompt_version = demo_version
        elif demo_version is not None and demo_version != prompt_version:
            raise ValueError("teacher demo prompt_version disagrees with ledger episode")
        if prompt_version is not None:
            record["demo"]["prompt_version"] = prompt_version
    if prompt_version is not None:
        record["prompt_version"] = prompt_version
    if usage:
        record["usage"] = dict(usage)
    if usage_status is not None:
        record["usage_status"] = usage_status
    append_record(ledger_path(benchmark, ledger_root=ledger_root), record)
    return record


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _validate_record(value: Any, line_number: int) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"teacher ledger line {line_number} is not an object")
    required = {
        "task_id", "teacher", "attempt_index", "temperature", "verified",
        "tokens_spent", "timestamp",
    }
    missing = required - set(value)
    if missing:
        raise ValueError(
            f"teacher ledger line {line_number} is missing {sorted(missing)}"
        )
    record = dict(value)
    # Records written before two-sided tau2 accounting implicitly purchased
    # the teacher-agent side.
    record.setdefault("purpose", "teacher")
    if not isinstance(record["task_id"], str) or not record["task_id"]:
        raise ValueError(f"teacher ledger line {line_number} has invalid task_id")
    if not isinstance(record["teacher"], str) or not record["teacher"]:
        raise ValueError(f"teacher ledger line {line_number} has invalid teacher")
    if not isinstance(record["attempt_index"], int) or record["attempt_index"] < 0:
        raise ValueError(
            f"teacher ledger line {line_number} has invalid attempt_index"
        )
    if not isinstance(record["verified"], bool):
        raise ValueError(f"teacher ledger line {line_number} has invalid verified")
    if not isinstance(record["tokens_spent"], int) or record["tokens_spent"] < 0:
        raise ValueError(
            f"teacher ledger line {line_number} has invalid tokens_spent"
        )
    if not isinstance(record["purpose"], str) or not record["purpose"]:
        raise ValueError(
            f"teacher ledger line {line_number} has invalid purpose"
        )
    try:
        record["temperature"] = float(record["temperature"])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"teacher ledger line {line_number} has invalid temperature"
        ) from exc
    if not isinstance(record["timestamp"], str):
        raise ValueError(f"teacher ledger line {line_number} has invalid timestamp")
    if record["verified"] is True and not isinstance(record.get("demo"), Mapping):
        raise ValueError(
            f"teacher ledger line {line_number} has no verified demo payload"
        )
    return record


def read_records(
    benchmark: str | Path, *, ledger_root: Path | None = None
) -> list[dict[str, Any]]:
    path = ledger_path(benchmark, ledger_root=ledger_root)
    try:
        with path.open(encoding="utf-8") as handle:
            return [
                _validate_record(json.loads(line), line_number)
                for line_number, line in enumerate(handle, 1)
                if line.strip()
            ]
    except FileNotFoundError:
        return []
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"teacher ledger {path} contains invalid JSON on line {exc.lineno}"
        ) from exc


def compact_records(
    records: Iterable[Mapping[str, Any]], attempts: int | None = None
) -> dict[str, dict[str, Any]]:
    if attempts is not None and attempts < 0:
        raise ValueError("attempts must be non-negative")
    compact: dict[str, dict[str, Any]] = {}
    for line_number, raw_record in enumerate(records, 1):
        record = _validate_record(raw_record, line_number)
        # User-simulator calls are budgeted in the same append-only ledger but
        # must not consume the teacher-agent attempt allowance or qualify as a
        # demonstration.
        if record["purpose"] == "user_sim":
            continue
        task_id = record["task_id"]
        state = compact.setdefault(task_id, {
            "best_demo": None,
            "attempts_used": 0,
            "tokens_total": 0,
            "infeasible": False,
        })
        state["attempts_used"] += 1
        state["tokens_total"] += record["tokens_spent"]
        if record["verified"] is True:
            state["best_demo"] = _demo_from_payload(
                task_id, record["demo"], record
            )
    for state in compact.values():
        state["infeasible"] = bool(
            attempts is not None
            and state["best_demo"] is None
            and state["attempts_used"] >= attempts
        )
    return compact


def load_ledger(
    benchmark: str | Path,
    attempts: int | None = None,
    *,
    ledger_root: Path | None = None,
    prompt_version: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Load a task-keyed view, isolating versioned attempts and cached demos.

    Pre-versioning WebShop rows are v1. Other benchmarks retain their existing
    unversioned behavior. A mixed ledger requires an explicit version to read.
    """

    records = read_records(benchmark, ledger_root=ledger_root)
    if prompt_version is not None:
        records = [dict(record, prompt_version=prompt_version) for record in records
                   if record.get("prompt_version", "v1") == prompt_version]
    elif len({record.get("prompt_version", "v1") for record in records}) > 1:
        raise ValueError("mixed teacher ledger requires an explicit prompt_version")
    return compact_records(records, attempts=attempts)


@contextmanager
def _purchase_lock(
    benchmark: str | Path, *, ledger_root: Path | None = None
) -> Iterator[None]:
    path = ledger_path(benchmark, ledger_root=ledger_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with lock_path.open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def estimate_response_tokens(responses: Sequence[str]) -> int:
    """Conservative tokenizer-free estimate from teacher response lengths."""

    return sum(math.ceil(len(response) / 4) for response in responses)


def _teacher_name(benchmark: str | Path, adapter: BenchmarkAdapter) -> str:
    configured = getattr(adapter, "teacher_name", None)
    if callable(configured):
        return str(configured())
    if str(benchmark) == "bfcl":
        return os.environ.get("BFAS_BFCL_TEACHER", "gpt-5.4")
    return os.environ.get("BFAS_TEACHER", "gpt-5.4")


def _is_quota_error(exc: BaseException) -> bool:
    try:
        import appworld_teacher
    except ImportError:
        return False
    return (
        isinstance(exc, appworld_teacher.TeacherAPIError)
        and "HTTP 429" in str(exc)
    )


def _empty_state(attempts: int) -> dict[str, Any]:
    return {
        "best_demo": None,
        "attempts_used": 0,
        "tokens_total": 0,
        "infeasible": attempts == 0,
    }


def _minimum_interval() -> float:
    raw = os.environ.get("BFAS_TEACHER_MIN_INTERVAL_S", "0")
    try:
        interval = float(raw)
    except ValueError as exc:
        raise ValueError(
            "BFAS_TEACHER_MIN_INTERVAL_S must be a non-negative number"
        ) from exc
    if interval < 0:
        raise ValueError(
            "BFAS_TEACHER_MIN_INTERVAL_S must be a non-negative number"
        )
    return interval


def acquire_demos(
    benchmark: str | Path,
    adapter: BenchmarkAdapter,
    task_ids: Sequence[str],
    attempts: int,
    *,
    ledger_root: Path | None = None,
) -> AcquisitionResult:
    """Return verified demos, purchasing only ledger-unresolved attempts."""

    if attempts < 0:
        raise ValueError("attempts must be non-negative")
    requested = list(dict.fromkeys(task_ids))
    interval = _minimum_interval()
    path = ledger_path(benchmark, ledger_root=ledger_root).resolve()
    prompt_version = getattr(adapter, "prompt_version", None)
    with _purchase_lock(benchmark, ledger_root=ledger_root):
        states = load_ledger(
            benchmark, attempts=attempts, ledger_root=ledger_root, prompt_version=prompt_version,
        )
        purchased = path in _PURCHASED_LEDGERS
        teacher = _teacher_name(benchmark, adapter)
        for task_id in requested:
            state = states.setdefault(task_id, _empty_state(attempts))
            if state["best_demo"] is not None or state["infeasible"] is True:
                continue
            while state["attempts_used"] < attempts:
                if purchased and interval:
                    time.sleep(interval)
                attempt_index = int(state["attempts_used"])
                temperature = (
                    0.0 if attempt_index == 0 else SAMPLING_TEMPERATURE
                )
                try:
                    episode = adapter.teacher_episode(
                        task_id, attempt_index, temperature
                    )
                    if not isinstance(episode, TeacherEpisode):
                        raise TypeError(
                            "adapter.teacher_episode must return TeacherEpisode"
                        )
                    if episode.task_id != task_id:
                        raise ValueError(
                            "adapter.teacher_episode returned the wrong task_id"
                        )
                    if bool(episode.verified) != (episode.demo is not None):
                        raise ValueError(
                            "teacher episode verified/demo fields disagree"
                        )
                    tokens_spent = (
                        episode.tokens_spent
                        if episode.tokens_spent is not None
                        else estimate_response_tokens(episode.response_texts)
                    )
                except BaseException as exc:
                    if _is_quota_error(exc):
                        raise
                    append_episode(
                        benchmark,
                        task_id=task_id,
                        teacher=teacher,
                        attempt_index=attempt_index,
                        temperature=temperature,
                        verified=False,
                        tokens_spent=0,
                        ledger_root=ledger_root,
                        prompt_version=prompt_version,
                    )
                    purchased = True
                    _PURCHASED_LEDGERS.add(path)
                    state["attempts_used"] += 1
                    if not isinstance(exc, Exception):
                        raise
                    print(f"[bfas][demo] task={task_id} failed: {exc}", flush=True)
                    continue

                append_episode(
                    benchmark,
                    task_id=task_id,
                    teacher=episode.teacher or teacher,
                    attempt_index=attempt_index,
                    temperature=temperature,
                    verified=episode.verified,
                    tokens_spent=tokens_spent,
                    usage=episode.usage,
                    usage_status=episode.usage_status,
                    demo=episode.demo,
                    ledger_root=ledger_root,
                    prompt_version=prompt_version,
                )
                purchased = True
                _PURCHASED_LEDGERS.add(path)
                state["attempts_used"] += 1
                state["tokens_total"] += tokens_spent
                if episode.demo is not None:
                    state["best_demo"] = episode.demo
                    break
            state["infeasible"] = bool(
                state["best_demo"] is None
                and state["attempts_used"] >= attempts
            )

        final_states = load_ledger(
            benchmark, attempts=attempts, ledger_root=ledger_root, prompt_version=prompt_version,
        )
        requested_states: dict[str, dict[str, Any]] = {}
        for task_id in requested:
            requested_states[task_id] = final_states.get(
                task_id, _empty_state(attempts)
            )
        demos = {
            task_id: state["best_demo"]
            for task_id, state in requested_states.items()
            if state["best_demo"] is not None
        }
        return AcquisitionResult(demos, requested_states)


def import_demos(
    benchmark: str | Path,
    demos: Mapping[str, Demo],
    *,
    teacher: str = "legacy-cache",
    ledger_root: Path | None = None,
) -> None:
    """Idempotently seed verified legacy demos into an empty/new ledger."""

    with _purchase_lock(benchmark, ledger_root=ledger_root):
        states_by_version = {}
        for task_id, demo in demos.items():
            raw = demo.raw if isinstance(demo.raw, Mapping) else {}
            version = raw.get("prompt_version", "v1" if str(benchmark) == "webshop" else None)
            if version not in states_by_version:
                states_by_version[version] = load_ledger(
                    benchmark, ledger_root=ledger_root, prompt_version=version,
                )
            states = states_by_version[version]
            state = states.get(task_id)
            if state is not None and state["best_demo"] is not None:
                continue
            attempt_index = int(state["attempts_used"]) if state else 0
            append_episode(
                benchmark,
                task_id=task_id,
                teacher=teacher,
                attempt_index=attempt_index,
                temperature=0.0,
                verified=True,
                tokens_spent=0,
                demo=demo,
                ledger_root=ledger_root,
                prompt_version=version,
            )
            states[task_id] = {
                "best_demo": demo,
                "attempts_used": attempt_index + 1,
                "tokens_total": int(state["tokens_total"]) if state else 0,
                "infeasible": False,
            }
