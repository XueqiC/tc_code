"""Unified CRCD event record (spec 2026-09-04 §2, §3.1, §13).

An event is e_i = (s_i, y_i^S, y_i^T, O_i): the full decision state s_i, the
student continuation y^S, the teacher continuation y^T and matched branch
outcomes O_i.  Ownership superscripts are T/S -- teacher behaviour is never
assumed better, so there is no y+/y- anywhere in this schema.

The record carries exactly the §13 fields plus one extension, ``state_text``:
the serialised s_i (the deployment prompt at the decision point).  Without it
the fingerprint of §3.2 cannot be reproduced from the record alone, which §3.2
requires ("metadata sufficient to reproduce each fingerprint").

Converters map the two existing miner outputs onto the record:

  BFCL   (tools/bfcl_event_mine*.py, tools/bfcl_events_to_pools.py)
         prompt / response (= y^T, the oracle GT call block) / _rejected (= y^S)
         _event_u_plus (teacher-branch success rate) / _event_u_minus (student)
         _event_dU / _event_k / _seed_category / _traj
  ALFWorld (tools/alf_event_mine.py, tools/alf_event_reestimate.py)
         prompt / response (= y^T) / _rejected (= y^S) / _event_good (teacher
         command) / _event_bad (student command) / _event_wins ([wins_T, wins_S]
         out of _event_k) / _prefix_len / _event_turn / task_id
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

BENCHMARKS = ("appworld", "alfworld", "bfcl")
SPLITS = ("train", "dev", "calibration", "test")
BASE_STUDENT = "Qwen/Qwen3.5-4B"

# Fields of the §13 minimal record, in the order given there.
SPEC_FIELDS = (
    "benchmark", "split", "task_id", "state_id", "state_hash",
    "student_checkpoint", "student_continuation", "teacher_model",
    "teacher_continuation", "student_branch_outcomes", "teacher_branch_outcomes",
    "student_output_tokens", "teacher_output_tokens", "fingerprint_version",
    "fingerprint_path", "atom_dictionary_version", "atom_loading", "provenance",
)
EXTENSION_FIELDS = ("state_text",)
REQUIRED_FIELDS = SPEC_FIELDS + EXTENSION_FIELDS


def state_hash_of(state_text: str) -> str:
    """state_hash = sha1 of the serialised decision state (the prompt)."""
    return hashlib.sha1(state_text.encode("utf-8")).hexdigest()


def count_tokens(text: str, tokenizer: Any) -> int:
    """Number of tokens in ``text`` under ``tokenizer`` (no special tokens).

    ``tokenizer`` is a HF tokenizer or any callable returning a list of ids /
    a mapping with ``input_ids`` (tests pass a stub)."""
    try:
        out = tokenizer(text, add_special_tokens=False)
    except TypeError:
        out = tokenizer(text)
    ids = out["input_ids"] if hasattr(out, "keys") else out
    return int(len(ids))


def branch_outcome(success_rate: float | None, k: int | None = None,
                   n_success: int | None = None, metric: str = "success_rate",
                   **extra: Any) -> dict:
    """One matched-continuation outcome summary for a branch.

    The miners store only aggregates (K continuations per branch collapsed to a
    success rate), so an outcome entry is an aggregate record, not a per-rollout
    trace; ``kind`` says so explicitly."""
    rec: dict[str, Any] = {"kind": "aggregate", "metric": metric,
                           "value": None if success_rate is None else float(success_rate)}
    if k is not None:
        rec["k"] = int(k)
    if n_success is not None:
        rec["n_success"] = int(n_success)
    rec.update(extra)
    return rec


@dataclass
class UnifiedEvent:
    benchmark: str
    split: str
    task_id: str
    state_id: str
    state_hash: str
    student_checkpoint: str
    student_continuation: str          # y^S
    teacher_model: str
    teacher_continuation: str          # y^T
    student_branch_outcomes: list = field(default_factory=list)
    teacher_branch_outcomes: list = field(default_factory=list)
    student_output_tokens: int = 0
    teacher_output_tokens: int = 0
    fingerprint_version: str = ""
    fingerprint_path: str = ""
    atom_dictionary_version: str = ""
    atom_loading: list = field(default_factory=list)
    provenance: dict = field(default_factory=dict)
    state_text: str = ""               # s_i (extension, see module doc)

    # -- (de)serialisation -------------------------------------------------
    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: dict) -> "UnifiedEvent":
        known = {f.name for f in fields(cls)}
        missing = [k for k in REQUIRED_FIELDS if k not in d]
        if missing:
            raise ValueError(f"event record missing required fields: {missing}")
        unknown = sorted(set(d) - known)
        if unknown:
            raise ValueError(f"event record has unknown fields: {unknown}")
        return cls(**d)

    @classmethod
    def from_json(cls, s: str) -> "UnifiedEvent":
        return cls.from_dict(json.loads(s))

    def validate(self) -> "UnifiedEvent":
        if self.benchmark not in BENCHMARKS:
            raise ValueError(f"benchmark must be one of {BENCHMARKS}, got {self.benchmark!r}")
        if self.split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {self.split!r}")
        for name in ("task_id", "state_id", "state_hash", "student_checkpoint",
                     "teacher_model", "state_text"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be a non-empty string")
        if self.state_hash != state_hash_of(self.state_text):
            raise ValueError("state_hash does not match sha1(state_text)")
        for name in ("student_branch_outcomes", "teacher_branch_outcomes", "atom_loading"):
            if not isinstance(getattr(self, name), list):
                raise ValueError(f"{name} must be a list")
        for name in ("student_output_tokens", "teacher_output_tokens"):
            v = getattr(self, name)
            if not isinstance(v, int) or isinstance(v, bool) or v < 0:
                raise ValueError(f"{name} must be a non-negative int")
        if not isinstance(self.provenance, dict):
            raise ValueError("provenance must be a dict")
        return self


# -- converters ------------------------------------------------------------

def _outcomes_from_rates(row: dict) -> tuple[list, list]:
    k = row.get("_event_k")
    wins = row.get("_event_wins")
    n_t = n_s = None
    if isinstance(wins, (list, tuple)) and len(wins) == 2:
        n_t, n_s = int(wins[0]), int(wins[1])
    if "_event_plus_ok" in row:
        n_t = int(row["_event_plus_ok"])
    if "_event_minus_ok" in row:
        n_s = int(row["_event_minus_ok"])
    teacher = [branch_outcome(row.get("_event_u_plus"), k, n_t)]
    student = [branch_outcome(row.get("_event_u_minus"), k, n_s)]
    return student, teacher


def _state_id(row: dict, benchmark: str) -> str:
    traj = row.get("_traj")
    if traj:
        return str(traj)
    return f"{row['task_id']}#t{row.get('turn_index', 0)}"


def _int_or_none(v: Any) -> int | None:
    try:
        return None if v is None else int(v)
    except (TypeError, ValueError):
        return None


def _float_or_none(v: Any) -> float | None:
    try:
        return None if v is None else float(v)
    except (TypeError, ValueError):
        return None


def from_bfcl_row(row: dict, tokenizer: Any, *, source: str, split: str = "train",
                  student_checkpoint: str = BASE_STUDENT) -> UnifiedEvent:
    """BFCL miner row -> UnifiedEvent.  response = y^T (oracle GT), _rejected = y^S."""
    prompt = row["prompt"]
    y_t, y_s = row["response"], row["_rejected"]
    student_out, teacher_out = _outcomes_from_rates(row)
    prov = {
        "source_file": source,
        "seed_category": row.get("_seed_category"),
        "turn_index": _int_or_none(row.get("turn_index")),
        "event_turn": _int_or_none(row.get("_event_turn")),
        "event_dU": _float_or_none(row.get("_event_dU")),
        "task_phat": _float_or_none(row.get("_task_phat")),
        "teacher_field": row.get("teacher"),
        "miner_source": row.get("_source"),
        "student_calls": row.get("_student_calls"),
        "gt_calls": row.get("_gt_calls"),
        "event_reasons": row.get("_event_reasons"),
        "event_rollout": _int_or_none(row.get("_event_rollout")),
        "seed_task": row.get("_seed_task"),
        "n_student_samples": len(row["_rejected_all"]) if isinstance(row.get("_rejected_all"), list) else None,
    }
    prov = {k: v for k, v in prov.items() if v is not None}
    return UnifiedEvent(
        benchmark="bfcl", split=split, task_id=str(row["task_id"]),
        state_id=_state_id(row, "bfcl"), state_hash=state_hash_of(prompt),
        student_checkpoint=student_checkpoint, student_continuation=y_s,
        teacher_model=str(row.get("teacher", "oracle_gt")), teacher_continuation=y_t,
        student_branch_outcomes=student_out, teacher_branch_outcomes=teacher_out,
        student_output_tokens=count_tokens(y_s, tokenizer),
        teacher_output_tokens=count_tokens(y_t, tokenizer),
        provenance=prov, state_text=prompt,
    ).validate()


def from_alfworld_row(row: dict, tokenizer: Any, *, source: str, split: str = "train",
                      student_checkpoint: str = BASE_STUDENT) -> UnifiedEvent:
    """ALFWorld miner row -> UnifiedEvent.  response = y^T (demo command), _rejected = y^S."""
    prompt = row["prompt"]
    y_t, y_s = row["response"], row["_rejected"]
    student_out, teacher_out = _outcomes_from_rates(row)
    prov = {
        "source_file": source,
        "seed_category": row.get("_seed_category"),
        "turn_index": _int_or_none(row.get("turn_index")),
        "event_turn": _int_or_none(row.get("_event_turn")),
        "prefix_len": _int_or_none(row.get("_prefix_len")),
        "teacher_command": row.get("_event_good"),
        "student_command": row.get("_event_bad"),
        "event_dU": _float_or_none(row.get("_event_dU")),
        "event_dU_k3": _float_or_none(row.get("_event_dU_k3")),
        "task_phat": _float_or_none(row.get("_task_phat")),
        "teacher_field": row.get("teacher"),
    }
    prov = {k: v for k, v in prov.items() if v is not None}
    return UnifiedEvent(
        benchmark="alfworld", split=split, task_id=str(row["task_id"]),
        state_id=_state_id(row, "alfworld"), state_hash=state_hash_of(prompt),
        student_checkpoint=student_checkpoint, student_continuation=y_s,
        teacher_model=str(row.get("teacher", "demo_replay")), teacher_continuation=y_t,
        student_branch_outcomes=student_out, teacher_branch_outcomes=teacher_out,
        student_output_tokens=count_tokens(y_s, tokenizer),
        teacher_output_tokens=count_tokens(y_t, tokenizer),
        provenance=prov, state_text=prompt,
    ).validate()


CONVERTERS: dict[str, Callable[..., UnifiedEvent]] = {
    "bfcl": from_bfcl_row,
    "alfworld": from_alfworld_row,
}


# -- io -------------------------------------------------------------------

def read_events(path: str | Path) -> list[UnifiedEvent]:
    with open(path, encoding="utf-8") as f:
        return [UnifiedEvent.from_json(line) for line in f if line.strip()]


def iter_events(path: str | Path) -> Iterator[UnifiedEvent]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield UnifiedEvent.from_json(line)


def write_events(path: str | Path, events: Iterable[UnifiedEvent]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for ev in events:
            f.write(ev.to_json() + "\n")
            n += 1
    return n


def load_raw_rows(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
