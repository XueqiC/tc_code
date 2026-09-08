"""Paired behaviour observations and immutable, content-addressed JSONL bundles.

Matrices are always (probe, source). NA is ``nan`` in memory and JSON ``null``
on disk. A failed checker/rollout is not a negative outcome. By default a cell
requires every fixed, paired evaluation seed; incomplete cells remain NA.
Single deterministic observations are outcomes, not probability estimates.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

SCHEMA_VERSION = "behavior-response-v1"


def json_safe(value: Any) -> Any:
    """Convert numpy values and nonfinite floats to strict JSON (NA -> null)."""
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(json_safe(value), sort_keys=True, ensure_ascii=False,
                      allow_nan=False, separators=(",", ":"))


def content_hash(value: Any) -> str:
    """SHA256 of text/bytes, or canonical JSON for structured content."""
    if not isinstance(value, (str, bytes)):
        value = canonical_json(value)
    if isinstance(value, str):
        value = value.encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def write_immutable(path: str | Path, data: bytes, *, resume: bool = False) -> None:
    """Atomic create, never replace. Resume accepts byte-identical artifacts only.

    A temporary file plus a hard link avoids exposing partially written files
    after interruption. Concurrent writers cannot replace the winning artifact.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        try:
            os.link(temporary, path)
        except FileExistsError:
            if not resume or path.read_bytes() != data:
                raise FileExistsError(f"immutable artifact already exists or differs: {path}")
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path: str | Path, value: Any, *, resume: bool = False) -> None:
    write_immutable(path, (canonical_json(value) + "\n").encode(), resume=resume)


@dataclass(frozen=True)
class ProbeRecord:
    probe_id: str
    parent_task_id: str
    trajectory_id: str
    state_hash: str
    target_hash: str
    student_output_hash: str
    benchmark: str
    split: str
    group: str
    teacher_query_ids: tuple[str, ...] = ()
    probe_factor: str | None = None
    probe_family: str | None = None
    probe_margin_base: float | None = None

    def __post_init__(self) -> None:
        for name in ("probe_id", "parent_task_id", "trajectory_id", "state_hash",
                     "target_hash", "student_output_hash", "benchmark", "split", "group"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"probe {name} must be a nonempty string")
        object.__setattr__(self, "teacher_query_ids", tuple(self.teacher_query_ids))
        for name in ("probe_factor", "probe_family"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError(f"{name} must be a nonempty string or None")
        if self.probe_margin_base is not None and not np.isfinite(self.probe_margin_base):
            raise ValueError("probe_margin_base: use None for missing values")

    @property
    def content_key(self) -> str:
        return content_hash(asdict(self))


@dataclass(frozen=True)
class OutcomeRecord:
    source_id: str
    run_id: str
    probe_id: str
    eval_seed: int
    raw_output_hash: str | None
    checker_version: str
    base_outcome: float | None
    updated_outcome: float | None
    difference: float | None = None
    teacher_likelihood: float | None = None  # updated-model score
    student_likelihood: float | None = None
    base_teacher_likelihood: float | None = None
    base_student_likelihood: float | None = None
    likelihood_definition: str | None = None  # sequence log-prob / token mean, etc.
    status: str = "ok"
    error_type: str | None = None
    base_raw_output_hash: str | None = None
    raw_output: Any = None
    source_factor: str | None = None

    def __post_init__(self) -> None:
        for name in ("source_id", "run_id", "probe_id", "checker_version"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"outcome {name} must be a nonempty string")
        if isinstance(self.eval_seed, bool) or not isinstance(self.eval_seed, int):
            raise ValueError("eval_seed must be an integer")
        if self.status not in {"ok", "failed", "error", "missing", "pending"}:
            raise ValueError(f"unknown outcome status: {self.status}")
        if self.status in {"error", "failed"} and not self.error_type:
            raise ValueError("failed outcomes require error_type")
        for name in ("base_outcome", "updated_outcome", "difference", "teacher_likelihood",
                     "student_likelihood", "base_teacher_likelihood", "base_student_likelihood"):
            value = getattr(self, name)
            if value is not None and not np.isfinite(value):
                raise ValueError(f"{name}: use None for missing values")
        for value in (self.base_outcome, self.updated_outcome):
            if value is not None and not 0 <= value <= 1:
                raise ValueError("behaviour outcomes must be in [0, 1]")
        expected = None
        if self.status == "ok" and self.base_outcome is not None and self.updated_outcome is not None:
            if not self.raw_output_hash:
                raise ValueError("successful evaluations require raw_output_hash")
            expected = float(self.updated_outcome - self.base_outcome)
        if self.difference is not None and (expected is None or not np.isclose(
                self.difference, expected, rtol=0, atol=1e-12)):
            raise ValueError("difference must equal paired updated - base; failed/missing is NA")
        object.__setattr__(self, "difference", expected)
        if any(v is not None for v in (self.teacher_likelihood, self.student_likelihood,
                                       self.base_teacher_likelihood, self.base_student_likelihood)):
            if not self.likelihood_definition:
                raise ValueError("likelihood scores require an explicit likelihood_definition")
        if self.raw_output is not None and content_hash(self.raw_output) != self.raw_output_hash:
            raise ValueError("raw_output_hash does not match raw_output")
        if self.source_factor is not None and (not isinstance(self.source_factor, str) or not self.source_factor):
            raise ValueError("source_factor must be a nonempty string or None")


@dataclass
class ResponseMatrix:
    M: np.ndarray
    source_ids: tuple[str, ...]
    probe_ids: tuple[str, ...]
    valid_repeats: np.ndarray
    expected_repeats: int
    teacher: np.ndarray
    student: np.ndarray
    diagnostics: dict = field(default_factory=dict)


def validate_split(sources: Iterable[Mapping], probes: Iterable[ProbeRecord]) -> None:
    """Reject parent/trajectory/content overlap and protected probe splits."""
    sources, probes = list(sources), list(probes)
    for key in ("parent_task_id", "trajectory_id", "state_hash"):
        source_keys = {s[key] for s in sources if s.get(key)}
        overlap = source_keys & {getattr(p, key) for p in probes}
        if overlap:
            raise ValueError(f"source/probe overlap in {key}: {sorted(overlap)}")
    if any(p.split not in {"dev", "development", "discovery", "P_disc"} for p in probes):
        raise ValueError("P2 probes must belong to development/discovery, never calibration/test")


def assemble_matrix(outcomes: Iterable[OutcomeRecord], *, source_ids: Iterable[str],
                    probe_ids: Iterable[str], eval_seeds: Iterable[int] | None = None,
                    run_ids: Mapping[str, str] | None = None) -> ResponseMatrix:
    """Assemble paired-seed means, requiring all specified seeds in each cell.

    Never average across multiple update runs for a source. Select a run using
    ``run_ids`` if needed. Duplicate (run, source, probe, seed) records, mixed
    checkers, and inconsistent shared base outcomes are errors. Missing/failed
    repeats remain in diagnostics and invalidate the corresponding mean.
    """
    sources, probes = tuple(source_ids), tuple(probe_ids)
    if len(set(sources)) != len(sources) or len(set(probes)) != len(probes):
        raise ValueError("matrix source/probe IDs must be unique")
    si, pj = {s: i for i, s in enumerate(sources)}, {p: j for j, p in enumerate(probes)}
    records = list(outcomes)
    if run_ids is not None:
        records = [r for r in records if run_ids.get(r.source_id) == r.run_id]
    seeds = tuple(sorted({r.eval_seed for r in records}) if eval_seeds is None else eval_seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("a nonempty, unique fixed evaluation seed list is required")
    cells: dict[tuple, OutcomeRecord] = {}
    runs, checkers, bases = {}, {}, {}
    definitions = {r.likelihood_definition for r in records if r.likelihood_definition}
    if len(definitions) > 1:
        raise ValueError("cannot combine different likelihood definitions")
    for r in records:
        if r.source_id not in si or r.probe_id not in pj or r.eval_seed not in seeds:
            raise ValueError("record outside requested source/probe/seed axes")
        if r.source_id in runs and runs[r.source_id] != r.run_id:
            raise ValueError("multiple update runs for one source; select run_ids explicitly")
        runs[r.source_id] = r.run_id
        key = (r.probe_id, r.source_id, r.eval_seed)
        if key in cells:
            raise ValueError(f"duplicate evaluation: {key}")
        cells[key] = r
        if r.probe_id in checkers and checkers[r.probe_id] != r.checker_version:
            raise ValueError("mixed checker versions for a probe")
        checkers[r.probe_id] = r.checker_version
        base_key = (r.probe_id, r.eval_seed)
        if r.base_outcome is not None:
            if base_key in bases and bases[base_key] != r.base_outcome:
                raise ValueError("inconsistent paired base outcome for the same probe/seed")
            bases[base_key] = r.base_outcome
    shape = (len(probes), len(sources))
    M, teacher, student = (np.full(shape, np.nan) for _ in range(3))
    counts = np.zeros(shape, dtype=int)
    for j, p in enumerate(probes):
        for i, s in enumerate(sources):
            paired = [cells.get((p, s, seed)) for seed in seeds]
            good = [r for r in paired if r is not None and r.status == "ok" and r.difference is not None]
            counts[j, i] = len(good)
            if len(good) == len(seeds):
                M[j, i] = np.mean([r.difference for r in good])
            for matrix, updated, base in ((teacher, "teacher_likelihood", "base_teacher_likelihood"),
                                           (student, "student_likelihood", "base_student_likelihood")):
                values = [getattr(r, updated) - getattr(r, base) for r in paired
                          if r is not None and r.status == "ok" and getattr(r, updated) is not None
                          and getattr(r, base) is not None]
                if len(values) == len(seeds):
                    matrix[j, i] = np.mean(values)
    return ResponseMatrix(M, sources, probes, counts, len(seeds), teacher, student, {
        "missing_records": int(np.prod(shape) * len(seeds) - len(records)),
        "failed_records": sum(r.status in {"error", "failed"} for r in records),
        "status_counts": {s: sum(r.status == s for r in records) for s in sorted({r.status for r in records})},
        "missing_cells": int(np.isnan(M).sum()), "run_ids": runs,
        "aggregation": "mean of all fixed paired seeds; incomplete cells NA",
        "likelihood_definition": next(iter(definitions), None),
    })


def write_responses(path: str | Path, probes: Iterable[ProbeRecord],
                    outcomes: Iterable[OutcomeRecord], manifest: Mapping,
                    *, resume: bool = False) -> dict:
    """Write tagged JSONL plus ``<path>.manifest.json`` with an integrity hash."""
    path, probes, outcomes = Path(path), list(probes), list(outcomes)
    ids = [p.probe_id for p in probes]
    if len(set(ids)) != len(ids) or any(r.probe_id not in ids for r in outcomes):
        raise ValueError("duplicate probes or outcome referring to unknown probe")
    rows = [{"record_type": "probe", **asdict(p)} for p in probes]
    rows += [{"record_type": "outcome", **asdict(r)} for r in outcomes]
    data = ("\n".join(canonical_json(r) for r in rows) + "\n").encode()
    summary = {"schema_version": SCHEMA_VERSION, "sha256": content_hash(data),
               "n_probes": len(probes), "n_outcomes": len(outcomes), "provenance": dict(manifest)}
    # Claim/validate the manifest first, so a conflicting resume cannot write data.
    write_json(str(path) + ".manifest.json", summary, resume=resume)
    write_immutable(path, data, resume=resume)
    return summary


def read_responses(path: str | Path) -> tuple[list[ProbeRecord], list[OutcomeRecord], dict]:
    path = Path(path)
    manifest = json.loads(Path(str(path) + ".manifest.json").read_text())
    data = path.read_bytes()
    if manifest.get("schema_version") != SCHEMA_VERSION or content_hash(data) != manifest.get("sha256"):
        raise ValueError("response bundle version/hash mismatch")
    probes, outcomes = [], []
    for line in data.decode().splitlines():
        row = json.loads(line)
        kind = row.pop("record_type")
        if kind == "probe":
            probes.append(ProbeRecord(**row))
        elif kind == "outcome":
            outcomes.append(OutcomeRecord(**row))
        else:
            raise ValueError(f"unknown record_type: {kind}")
    if len(probes) != manifest["n_probes"] or len(outcomes) != manifest["n_outcomes"]:
        raise ValueError("response manifest record count mismatch")
    if len({p.probe_id for p in probes}) != len(probes):
        raise ValueError("duplicate probe IDs")
    if any(r.probe_id not in {p.probe_id for p in probes} for r in outcomes):
        raise ValueError("outcome refers to an unknown probe")
    return probes, outcomes, manifest
