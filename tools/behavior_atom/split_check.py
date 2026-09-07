#!/usr/bin/env python3
"""Audit P2.1 isolation without loading models or changing the manifests.

Source identities live in units' rows; noop controls may reference formal/pilot
rows without adding identities. Counts are advisory, including in strict mode;
strict mode additionally checks required
manifest fields. Generated task parents strip a prefix and its final sample index
together, so an official id such as ``simple_python_31`` stays intact.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re


def parent_id(value: str) -> str:
    """Normalize a trajectory/generated id to its official parent."""
    base = value.split("#", 1)[0]
    if re.match(r"^(genmt_|gen_|oos_)", base):
        base = re.sub(r"^(genmt_|gen_|oos_)", "", base, count=1)
        base = re.sub(r"_\d+$", "", base)
    return base


def event_key(row: dict) -> str:
    short_hash = lambda text: hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
    return f"{row['_traj']}|{short_hash(row['prompt'])}|{short_hash(row['_rejected'])}"


def _id(row: dict, fallback: str = "<unknown>") -> str:
    return str(row.get("source_id") or row.get("probe_id") or row.get("pair_id") or fallback)


def _entry(details: list[dict], *, report_only: bool = False) -> dict:
    return {
        "passed": not details,
        "report_only": report_only,
        "offending_ids": sorted({str(i) for d in details for i in d.get("ids", [])}),
        "details": details,
    }


def _finish(report: dict) -> dict:
    report["passed"] = all(c["passed"] for c in report["checks"].values()
                           if not c.get("report_only"))
    return report


def _task_values(row: dict) -> list[str]:
    return [str(v) for v in (row.get("parent_task_id"), row.get("task_id"),
                            row.get("_traj"), *(row.get("gen_ids") or [])) if v]


def _traj_bases(row: dict) -> set[str]:
    return {v.split("#", 1)[0] for v in _task_values(row)}


def _source_rows(sources: dict) -> list[dict]:
    return [{**row, "source_id": _id(unit), "role": unit.get("role"), "unit_row": index}
            for unit in sources.get("units", [])
            for index, row in enumerate(unit.get("rows", []))]


def _pair_rows(pairs: dict | None) -> list[dict]:
    return [{**pair[part], "pair_id": _id(pair), "component": part}
            for pair in (pairs or {}).get("pairs", [])
            for part in ("correction", "anchor_match", "anchor_random")
            if pair.get(part) is not None]


def _manifest_errors(sources: dict, probes: dict, folds: dict, pairs: dict | None) -> list[dict]:
    errors = []

    def require(row, fields, rid, nullable=()):
        missing = sorted(k for k in fields if k not in row or (
            k not in nullable and row[k] in (None, "")))
        if missing:
            errors.append({"ids": [rid], "missing_fields": missing})

    def choice(row, field, allowed, rid):
        if row.get(field) not in allowed:
            errors.append({"ids": [rid], "field": field, "invalid_value": row.get(field)})

    def hashes(row, rid):
        for field, length in (("prompt_sha1", 40), ("yT_sha1", 40),
                              ("yS_sha1", 40), ("prompt_hash", 16)):
            if field in row and (not isinstance(row[field], str) or
                                 not re.fullmatch(rf"[0-9a-fA-F]{{{length}}}", row[field])):
                errors.append({"ids": [rid], "field": field, "error": f"expected {length} hex characters"})

    require(sources, ("version", "created", "pool_path", "units"), "sources")
    require(probes, ("version", "created", "probes"), "probes")
    require(folds, ("n_folds", "assignment", "groups", "roles", "controls"), "folds")
    if type(folds.get("n_folds")) is not int or folds["n_folds"] < 1:
        errors.append({"ids": ["folds"], "error": "n_folds must be a positive integer"})
    for unit in sources.get("units", []):
        sid = _id(unit)
        require(unit, ("source_id", "unit_type", "rows", "parent_task_id", "group",
                       "factor", "family", "role"), sid,
                nullable=("parent_task_id", "family") if unit.get("role") == "noop" else ())
        choice(unit, "role", {"formal", "pilot", "noop", "anchor"}, sid)
        choice(unit, "unit_type", {"single", "pack", "contrastive"} |
               ({"noop"} if unit.get("role") == "noop" else set()), sid)
        if unit.get("role") == "noop":
            choice(unit, "parent_task_id", {None, "control"}, sid)
        elif not unit.get("rows"):
            errors.append({"ids": [sid], "error": "formal/pilot/anchor unit requires rows"})
        for row in unit.get("rows", []):
            require(row, ("pool_row", "event_key", "prompt_sha1", "yT_sha1", "yS_sha1",
                          "prompt_hash", "task_id", "_traj"), sid)
            if type(row.get("pool_row")) is not int:
                errors.append({"ids": [sid], "error": "pool_row must be an integer"})
            hashes(row, sid)
    for probe in probes.get("probes", []):
        pid = _id(probe)
        require(probe, ("probe_id", "set", "source", "task_id", "prompt", "function", "truth",
                        "category", "factor", "family", "expected", "base_rate", "base_margin",
                        "parent_task_id", "group", "prompt_sha1", "prompt_hash"), pid,
                nullable=("base_rate", "base_margin"))
        choice(probe, "set", {"P_disc", "P_confirm"}, pid)
        choice(probe, "source", {"official", "generated"}, pid)
        choice(probe, "expected", {"call", "abstain"}, pid)
        hashes(probe, pid)
    if pairs is not None:
        require(pairs, ("n_pairs", "pairs"), "pairs")
        for pair in pairs.get("pairs", []):
            pid = _id(pair)
            require(pair, ("pair_id", "family", "correction", "anchor_match", "anchor_random"),
                    pid, nullable=("anchor_match",))
            for part in ("correction", "anchor_match", "anchor_random"):
                row = pair.get(part)
                if row is None:
                    continue
                fields = ("pool_row", "event_key") if part == "correction" else ("anchor_file", "row", "task_id")
                require(row, (*fields, "parent_task_id", "prompt_sha1"), pid)
                hashes(row, pid)
    return errors


def check_split(sources: dict, probes: dict, folds: dict, calib_ids: set,
                pairs: dict | None = None, *, expect_formal: int = 24, expect_disc: int = 32,
                strict: bool = True) -> dict:
    """Return named checks with all offending record ids; never alter inputs."""
    src = sources.get("units", [])
    source_rows = _source_rows(sources)
    rows = [r for r in source_rows if r.get("role") != "noop"]
    noop_rows = [r for r in source_rows if r.get("role") == "noop"]
    prb = probes.get("probes", [])
    pair_rows = _pair_rows(pairs)
    corrections = [r for r in pair_rows if r["component"] == "correction"]
    assignment = folds.get("assignment", {})
    groups = folds.get("groups", {})
    checks = {}
    checks["manifest_fields"] = _entry(_manifest_errors(sources, probes, folds, pairs) if strict else [])
    existing_event_keys = {r["event_key"] for r in rows
                           if r.get("role") in {"formal", "pilot"} and r.get("event_key") is not None}
    checks["noop_rows_reference_existing"] = _entry([
        {"ids": [_id(row)], "unit_row": row["unit_row"], "event_key": row.get("event_key")}
        for row in noop_rows if row.get("event_key") not in existing_event_keys])

    for field, records in (("source_id", src), ("probe_id", prb),
                           ("event_key", rows), ("prompt_sha1", rows + prb)):
        seen = defaultdict(list)
        for index, row in enumerate(records):
            if row.get(field) is not None:
                seen[row[field]].append({"id": _id(row), "row": index})
        checks[f"unique_{field}"] = _entry([
            {"value": value, "ids": [r["id"] for r in records], "records": records}
            for value, records in seen.items() if len(records) > 1])
    identities = defaultdict(list)
    for row in src + prb:
        identities[_id(row)].append(row)
    checks["unique_record_ids"] = _entry([
        {"ids": [key], "occurrences": len(records)} for key, records in identities.items() if len(records) > 1])

    active = [s for s in src if s.get("role") in {"formal", "pilot"}]
    isolated = [s for s in src if s.get("role") in {"formal", "pilot", "anchor"}]
    def overlaps(name, left, right, values):
        seen = defaultdict(list)
        for row in left:
            for value in values(row):
                if value is not None:
                    seen[value].append(_id(row))
        details = []
        for row in right:
            for value in sorted(v for v in values(row) if v is not None):
                if value in seen:
                    detail = {"value": value, "ids": [*seen[value], _id(row)]}
                    if "component" in row:
                        detail["component"] = row["component"]
                    details.append(detail)
        checks[name] = _entry(details)

    overlaps("probe_source_parent_tasks", isolated, prb, lambda r: [r.get("parent_task_id")])
    # Retain the report check name, but compare full SHA1 identities, not short hashes.
    overlaps("probe_source_prompt_hashes", rows, prb, lambda r: [r.get("prompt_sha1")])
    overlaps("probe_source_traj_bases", rows, prb, _traj_bases)
    overlaps("confirm_disc_parent_tasks", [p for p in prb if p.get("set") == "P_disc"],
             [p for p in prb if p.get("set") == "P_confirm"], lambda r: [r.get("parent_task_id")])
    overlaps("pair_probe_parent_tasks", prb, pair_rows, lambda r: [r.get("parent_task_id")])
    overlaps("pair_source_event_keys", [r for r in rows if r.get("role") in {"formal", "pilot", "anchor"}],
             corrections, lambda r: [r.get("event_key")])
    calibration = {str(v) for v in calib_ids}
    checks["calibration_exclusion"] = _entry([
        {"ids": [_id(row)], "matched_ids": sorted(hits),
         **({"component": row["component"]} if "component" in row else {})}
        for row in src + rows + prb + pair_rows
        if (hits := {v for raw in _task_values(row) for v in (raw, parent_id(raw)) if v in calibration})])

    by_id = {_id(s): s for s in src}
    # An assignment object has one value per source id; uniqueness above also
    # prevents multiple units from sharing that same assignment entry.
    for role in ("formal", "pilot"):
        checks[f"{role}_source_fold_once"] = _entry([
            {"ids": [_id(s)], "occurrences": 0}
            for s in active if s["role"] == role and _id(s) not in assignment])
    parent_folds = defaultdict(set)
    membership_errors = []
    n_folds = folds.get("n_folds")
    for sid, fid in assignment.items():
        source = by_id.get(sid)
        if source is None or source.get("role") not in {"formal", "pilot"}:
            membership_errors.append({"ids": [sid], "error": "not a formal/pilot source"})
        if type(fid) is not int or type(n_folds) is not int or not 0 <= fid < n_folds:
            membership_errors.append({"ids": [sid], "invalid_fold": fid})
        elif source is not None and source.get("role") in {"formal", "pilot"}:
            parent_folds[source.get("parent_task_id")].add(fid)
    checks["fold_membership"] = _entry(membership_errors)
    checks["noop_not_assigned"] = _entry([
        {"ids": [_id(s)], "fold": assignment[_id(s)]}
        for s in src if s.get("role") == "noop" and _id(s) in assignment])
    checks["anchor_not_assigned"] = _entry([
        {"ids": [_id(s)],
         **({"fold": assignment[_id(s)]} if _id(s) in assignment else {}),
         **({"group": groups[_id(s)]} if _id(s) in groups else {})}
        for s in src if s.get("role") == "anchor" and (_id(s) in assignment or _id(s) in groups)])
    checks["parent_task_single_fold"] = _entry([
        {"parent_task_id": parent, "folds": sorted(indices),
         "ids": [_id(s) for s in active if s.get("parent_task_id") == parent]}
        for parent, indices in parent_folds.items() if len(indices) > 1])
    checks["fold_parent_coverage"] = _entry([
        {"ids": [_id(s)], "parent_task_id": s.get("parent_task_id"), "group": groups.get(_id(s))}
        for s in active if _id(s) not in groups or groups[_id(s)] != s.get("parent_task_id")])

    counts = {role: sum(s.get("role") == role for s in src) for role in ("formal", "pilot", "noop", "anchor")}
    counts.update({part: sum(p.get("set") == part for p in prb) for part in ("P_disc", "P_confirm")})
    counts["pairs"] = len((pairs or {}).get("pairs", []))
    counts["anchor_match_null"] = sum(p.get("anchor_match") is None for p in (pairs or {}).get("pairs", []))
    expected = {"formal": expect_formal, "pilot": 4, "noop": 2, "P_disc": expect_disc, "pairs": 12}
    checks["counts"] = _entry([
        {"ids": [], "category": key, "expected": number, "actual": counts[key]}
        for key, number in expected.items() if counts[key] != number], report_only=True)
    return _finish({"checks": checks, "counts": counts, "expected_counts": expected,
                    "strict": strict, "note": "Counts are report-only; noop rows may reference formal/pilot rows without adding identities; null anchor_match is allowed."})


def check_pool(sources: dict, path: str | Path, pairs: dict | None = None, *,
               anchor_only: bool = False) -> dict:
    """Check event keys once in the top-level pool, or anchor units in an anchor pool."""
    counts = Counter()
    errors = []
    try:
        with Path(path).open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    counts[event_key(json.loads(line))] += 1
                except (ValueError, KeyError, TypeError, AttributeError) as exc:
                    errors.append({"ids": [f"{path}:{line_number}"], "error": str(exc)})
    except OSError as exc:
        errors.append({"ids": [str(path)], "status": "missing" if isinstance(exc, FileNotFoundError) else "error", "error": str(exc)})
    if anchor_only:
        units = [s for s in sources.get("units", []) if s.get("role") == "anchor"]
    else:
        units = [s for s in sources.get("units", [])
                 if s.get("pool_path", sources.get("pool_path")) == sources.get("pool_path")]
    rows = _source_rows({"units": units})
    if not anchor_only:
        rows += [r for r in _pair_rows(pairs) if r["component"] == "correction"]
    for row in rows:
        key = row.get("event_key")
        if key is not None and counts[key] != 1:
            errors.append({"ids": [_id(row)], "event_key": key, "pool_occurrences": counts[key]})
    return _entry(errors)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", default="data/behavior_atom_v1/sources.json")
    parser.add_argument("--probes", default="data/behavior_atom_v1/probes.json")
    parser.add_argument("--folds", default="data/behavior_atom_v1/folds.json")
    parser.add_argument("--pairs", default="data/behavior_atom_v1/complementarity_pairs.json")
    parser.add_argument("--support-split", default="configs/bfcl_support_split.json")
    parser.add_argument("--calibration-v2", default="data/bfcl_sft/calibration_ids_official_v2.json")
    parser.add_argument("--pool", help="Optional JSONL pool for exact event-key membership")
    parser.add_argument("--anchor-pool", help="Optional JSONL pool for exact anchor-unit event-key membership")
    parser.add_argument("--out", help="Also write the stdout JSON report here")
    parser.add_argument("--strict", action=argparse.BooleanOptionalAction, default=True,
                        help="Check required manifest fields (default on); isolation failures always fail")
    parser.add_argument("--expect-formal", type=int, default=24, help="Advisory expected formal count")
    parser.add_argument("--expect-disc", type=int, default=32, help="Advisory expected P_disc count")
    args = parser.parse_args(argv)
    data, inputs, errors = {}, {}, []
    for key in ("sources", "probes", "folds", "pairs", "support_split", "calibration_v2"):
        path = getattr(args, key)
        try:
            data[key] = json.loads(Path(path).read_text(encoding="utf-8"))
            if not isinstance(data[key], dict):
                raise ValueError("expected a JSON object")
            inputs[key] = {"path": path, "status": "ok"}
        except (OSError, ValueError) as exc:
            data[key] = {}
            inputs[key] = {"path": path, "status": "missing" if isinstance(exc, FileNotFoundError) else "error"}
            errors.append({"ids": [path], "error": str(exc)})
    for key, field in (("support_split", "calibration"), ("calibration_v2", "ids")):
        if field not in data[key]:
            errors.append({"ids": [getattr(args, key)], "missing_fields": [field]})
    calib = set(data["support_split"].get("calibration", [])) | set(data["calibration_v2"].get("ids", []))
    report = check_split(data["sources"], data["probes"], data["folds"], calib,
                         data["pairs"],
                         expect_formal=args.expect_formal, expect_disc=args.expect_disc, strict=args.strict)
    report["inputs"] = inputs
    report["checks"]["input_files"] = _entry(errors)
    if args.pool:
        report["checks"]["pool_event_keys"] = check_pool(data["sources"], args.pool, data["pairs"])
    if args.anchor_pool:
        report["checks"]["anchor_pool_event_keys"] = check_pool(data["sources"], args.anchor_pool,
                                                             anchor_only=True)
    _finish(report)
    output = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(output, encoding="utf-8")
    print(output, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
