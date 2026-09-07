#!/usr/bin/env python3
"""P0.3 offline cost audit, using only the standard library by default.

Token figures are objects with value, basis (exact/estimate), and rule. Teacher
cost means output tokens unless explicitly named input/total; ledger tokens_spent
retains its recorded meaning. No monetary prices are inferred. Exact usage and
generation estimates are separate views, never an additive grand total.

Without --tokenizer, generation uses Unicode chars / 4 * 1.3. With --tokenizer
[Qwen/Qwen3.5-4B], try transformers and locally cached files only, then fall back
to chars/4 on any loading failure. Even tokenized generation is an estimate of
teacher output: the tokenizer is a proxy and 1.3 accounts for discarded prose.
Only question content, ground_truth and optional scenario are counted. Missing
drafts/repair calls cannot be reconstructed. No network or model calls are made.

Repeat --pool to audit multiple pools; the combined view deduplicates queries
across files. --gen-cost-json accepts {prompt_hash: number} or
{prompt_hash: {value: number, rule: string}}; mapped costs are always estimates.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import date, datetime
import glob
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable


DEMO_BASE = "envs/bfcl/gorilla/berkeley-function-call-leaderboard/result_demos_deepseek_v4_pro_FC"
DEMO_NOTE = (
    "output_token_count = usage.completion_tokens from the OpenAI-compatible endpoint (ollama). "
    "No separate reasoning-token field exists in these files; any reasoning tokens are already "
    "inside output_token_count and must NOT be added again. All attempts, including retries "
    "and failed/unverified attempts, are billed. The merged directory supplies verification only."
)


def token_figure(value: int | float | None, basis: str, rule: str) -> dict:
    return {"value": value, "basis": basis, "rule": rule}


def exact(value: int | float | None, rule: str) -> dict:
    return token_figure(value, "exact", rule)


def estimate(value: int | float | None, rule: str) -> dict:
    return token_figure(value, "estimate", rule)


def flatten_tokens(value) -> int:
    """Recursively sum nested token lists, integer strings, and null/empty lists."""
    if isinstance(value, (list, tuple)):
        return sum(flatten_tokens(item) for item in value)
    if value is None or value == "":
        return 0
    return int(value)


def read_json(path: str | Path) -> tuple[dict, dict]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("expected a JSON object")
        return data, {"path": str(path), "status": "ok"}
    except (OSError, ValueError) as exc:
        return {}, {"path": str(path), "status": "missing" if isinstance(exc, FileNotFoundError) else "error", "error": str(exc)}


def read_jsonl(path: str | Path) -> tuple[list[dict], dict]:
    rows, errors = [], []
    try:
        with Path(path).open(encoding="utf-8") as handle:
            for number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError("expected a JSON object row")
                    rows.append(row)
                except ValueError as exc:
                    errors.append({"line": number, "error": str(exc)})
    except OSError as exc:
        return rows, {"path": str(path), "status": "missing" if isinstance(exc, FileNotFoundError) else "error", "error": str(exc), "rows": len(rows)}
    return rows, {"path": str(path), "status": "partial" if errors else "ok", "rows": len(rows), "errors": errors}


def group_demo_id(task_id: str, demand_ids: set) -> str:
    """Prerequisite snapshot bootstrap takes precedence even if listed in demand."""
    if "_prereq_" in task_id:
        return "memory_prereq"
    return "demand" if task_id in demand_ids else "other_official"


def _directory_rows(directory: str | Path) -> tuple[list[tuple[str, int, dict]], dict]:
    files = sorted(Path(directory).glob("**/*_result.json"))
    records, inputs = [], []
    for path in files:
        rows, info = read_jsonl(path)
        inputs.append(info)
        records.extend((str(path), number, row) for number, row in enumerate(rows, 1))
    return records, {"path": str(directory), "status": "missing" if not Path(directory).exists() else (
        "partial" if any(i["status"] != "ok" for i in inputs) else "ok"), "files": inputs, "rows": len(records)}


def audit_demos(directories: Iterable[str | Path], verified_dir: str | Path, demand_ids: set) -> dict:
    verified_rows, verified_info = _directory_rows(verified_dir)
    verified_ids = {r.get("id") for _, _, r in verified_rows}
    per_id, inputs = {}, []
    for directory in dict.fromkeys(map(str, directories)):
        rows, info = _directory_rows(directory)
        inputs.append(info)
        for path, number, row in rows:
            task_id = str(row.get("id", f"<missing-id:{path}:{number}>"))
            record = per_id.setdefault(task_id, {
                "group": group_demo_id(task_id, demand_ids), "attempts": 0,
                "verified": task_id in verified_ids if verified_info["status"] == "ok" else None,
                "output_tokens_per_attempt": [], "attempt_details": [],
            })
            output = exact(flatten_tokens(row.get("output_token_count")), "flatten and sum output_token_count; includes reasoning")
            incoming = exact(flatten_tokens(row.get("input_token_count")), "flatten and sum input_token_count")
            record["attempts"] += 1
            record["output_tokens_per_attempt"].append(output)
            record["attempt_details"].append({"path": path, "row": number,
                "input_tokens": incoming, "output_tokens": output,
                "missing_usage_fields": [k for k in ("input_token_count", "output_token_count") if row.get(k) is None]})
    for record in per_id.values():
        for field in ("input_tokens", "output_tokens"):
            record[field] = exact(sum(a[field]["value"] for a in record["attempt_details"]), f"sum recorded {field} across all attempts for this id")
    def aggregate(records):
        return {"n_ids": len(records), "n_rows": sum(r["attempts"] for r in records),
                **{field: exact(sum(r[field]["value"] for r in records), f"sum recorded demo {field}; all attempts")
                   for field in ("input_tokens", "output_tokens")}}
    return {"status": "missing" if not any(i["status"] != "missing" for i in inputs) else (
                "partial" if any(i["status"] != "ok" for i in inputs) else "ok"),
            "inputs": inputs, "verified_input": verified_info,
            "groups": {g: aggregate([r for r in per_id.values() if r["group"] == g])
                       for g in ("demand", "memory_prereq", "other_official")},
            "per_id": per_id, "totals": aggregate(list(per_id.values())), "note": DEMO_NOTE}


def _question_text(value) -> str:
    if isinstance(value, list):
        return "\n".join(_question_text(v) for v in value)
    if isinstance(value, dict) and "content" in value:
        return _question_text(value["content"])
    return _text(value)


def _text(value) -> str:
    if value is None:
        return ""
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def generation_text(row: dict) -> str:
    return "\n".join([_question_text(row.get("question")), _text(row.get("ground_truth"))]
                     + ([_text(row["scenario"])] if "scenario" in row else []))


def load_token_counter(tokenizer_name: str | None = None):
    """Return (counter, metadata); optional tokenizer loading is strictly offline."""
    info = {"basis": "estimate", "rule": "chars/4x1.3", "label": "estimate:chars/4x1.3",
            "tokenizer": None, "note": "Unicode character count / 4, then prose factor 1.3"}
    if tokenizer_name:
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, local_files_only=True, trust_remote_code=False)
            info.update(rule=f"tokenizer:{tokenizer_name}x1.3", label=f"estimate:tokenizer:{tokenizer_name}x1.3",
                        tokenizer=tokenizer_name, note="Local proxy tokenizer, add_special_tokens=False, then prose factor 1.3")
            return lambda text: len(tokenizer.encode(text, add_special_tokens=False)), info
        except Exception as exc:
            info["fallback_reason"] = f"{type(exc).__name__}: {exc}"
    return lambda text: len(text) / 4, info


def audit_generation_usage(path: str | Path) -> dict:
    """Recognize common usage aliases once each; never add reasoning breakdowns."""
    rows, info = read_jsonl(path)
    aliases = {"input_tokens": ("prompt_tokens", "input_tokens", "input_token_count"),
               "output_tokens": ("completion_tokens", "output_tokens", "output_token_count"),
               "reported_total_tokens": ("total_tokens", "tokens_spent")}
    totals, found, unrecognized = Counter(), Counter(), []
    for number, row in enumerate(rows, 1):
        usage = row.get("usage")
        containers = [("usage.", usage), ("", row)] if isinstance(usage, dict) else [("", row)]
        recognized = False
        for metric, fields in aliases.items():
            match = next(((prefix + key, data[key]) for prefix, data in containers for key in fields
                          if key in data and data[key] is not None), None)
            if match:
                key, value = match
                totals[metric] += flatten_tokens(value)
                found[key] += 1
                recognized = True
        if not recognized:
            unrecognized.append(number)
    return {**info, "fields_found": dict(found), "unrecognized_rows": unrecognized,
            **{key: exact(totals[key], f"sum one recognized {key} field per usage row; observed fields only") for key in aliases},
            "note": "Reported total is an independent field, not added to input/output. Reasoning breakdowns are not added."}


def audit_generation(paths: Iterable[str | Path], usage_path: str | Path, *,
                     tokenizer_name: str | None = None, include_smoke: bool = False) -> dict:
    counter, method = load_token_counter(tokenizer_name)
    files, skipped = {}, []
    for path in sorted(set(map(str, paths))):
        if not include_smoke and "smoke" in Path(path).name:
            skipped.append(path)
            continue
        rows, info = read_jsonl(path)
        files[path] = {**info, "teacher_output_tokens": estimate(
            sum(counter(generation_text(row)) * 1.3 for row in rows), method["rule"])}
    usage = audit_generation_usage(usage_path)
    return {"status": "missing" if not files or all(f["status"] == "missing" for f in files.values()) else (
                "partial" if any(f["status"] != "ok" for f in files.values()) else "ok"),
            "files": files, "excluded_smoke_files": skipped, "method": method,
            "rows": sum(f["rows"] for f in files.values()), "exact_usage": usage,
            "totals": {"estimated_output_tokens": estimate(sum(f["teacher_output_tokens"]["value"] for f in files.values()), method["rule"]),
                       "exact_output_tokens": usage["output_tokens"], "exact_input_tokens": usage["input_tokens"]},
            "note": "Exact usage and stored-artifact estimates may cover the same queries; do not add them. Estimates omit unstored failed drafts/repair calls."}


def dedup_ledger_rows(rows: Iterable[dict]) -> dict:
    """Keep first (task_id, attempt_index, purpose, timestamp), including failures.

    Different purposes or timestamps are distinct calls; missing/empty purposes
    default to teacher. Missing/null/empty timestamps fall back to the three-field
    (task_id, attempt_index, purpose) key with a warning. Duplicate records are
    reported with their row numbers, purposes, timestamps, and recorded tokens.
    """
    kept, seen, duplicate_groups = [], {}, {}
    warnings = []
    for number, row in enumerate(rows, 1):
        if row.get("task_id") is None or row.get("attempt_index") is None:
            warnings.append(f"Row {number} lacks task_id/attempt_index; retained as an unkeyed record.")
            kept.append(row)
            continue
        key = (str(row["task_id"]), str(row["attempt_index"]), row.get("purpose") or "teacher")
        if row.get("timestamp") not in (None, ""):
            key += (row["timestamp"],)
        else:
            warnings.append(f"Row {number} lacks timestamp; falling back to (task_id, attempt_index, purpose) for deduplication.")
        if key not in seen:
            seen[key] = number
            kept.append(row)
        else:
            group = duplicate_groups.setdefault(key, {"task_id": key[0], "attempt_index": key[1],
                "purpose": key[2], "timestamp": key[3] if len(key) == 4 else None,
                "first_row": seen[key], "duplicates": []})
            group["duplicates"].append({"row": number, "purpose": row.get("purpose", "teacher"),
                "tokens_spent": exact(flatten_tokens(row.get("tokens_spent")), "excluded duplicate's recorded tokens_spent")})
    duplicate_count = sum(len(g["duplicates"]) for g in duplicate_groups.values())
    if duplicate_count:
        warnings.append(f"Excluded {duplicate_count} duplicate rows in {len(duplicate_groups)} groups keyed by "
                        "(task_id, attempt_index, purpose, timestamp), or (task_id, attempt_index, purpose) "
                        "when timestamp is missing; kept first occurrence.")
    return {"rows": kept, "duplicate_rows": duplicate_count, "duplicate_pairs": len(duplicate_groups),
            "duplicates": list(duplicate_groups.values()), "warnings": warnings}


def _verified(row: dict) -> bool:
    return str(row.get("verified", "")).strip().lower() == "true"


def audit_ledger(path: str | Path, name: str, today: date) -> dict:
    raw, info = read_jsonl(path)
    dedup = dedup_ledger_rows(raw)
    kept = dedup["rows"]
    purposes = Counter()
    for row in kept:
        purposes[row.get("purpose") or "teacher"] += flatten_tokens(row.get("tokens_spent"))
    task_attempts = Counter(str(r["task_id"]) for r in kept if r.get("task_id") is not None)
    future, invalid_dates = [], []
    for number, row in enumerate(raw, 1):
        stamp = row.get("timestamp")
        if not stamp:
            invalid_dates.append({"row": number, "task_id": row.get("task_id"), "timestamp": stamp})
            continue
        try:
            row_date = datetime.fromisoformat(str(stamp).replace("Z", "+00:00")).date()
        except ValueError:
            invalid_dates.append({"row": number, "task_id": row.get("task_id"), "timestamp": stamp})
            continue
        if row_date > today:
            future.append({"path": str(path), "row": number, "task_id": row.get("task_id"), "timestamp": stamp})
    return {**info, "distinct_tasks": len(task_attempts), "billed_rows": len(kept),
            "verified_rows": sum(map(_verified, raw)), "billed_verified_rows": sum(map(_verified, kept)),
            "tokens_by_purpose": {p: exact(t, "sum tokens_spent after keeping first (task_id, attempt_index, purpose, timestamp); three-field fallback when timestamp is missing") for p, t in purposes.items()},
            "tokens_spent": exact(sum(purposes.values()), "sum recorded tokens_spent after deduplication by (task_id, attempt_index, purpose, timestamp); three-field fallback when timestamp is missing; failed attempts included"),
            "attempts_per_task_histogram": dict(sorted(Counter(task_attempts.values()).items())),
            "zero_token_rows": sum(flatten_tokens(r.get("tokens_spent")) == 0 for r in raw),
            "billed_zero_token_rows": sum(flatten_tokens(r.get("tokens_spent")) == 0 for r in kept),
            "missing_token_rows": sum(r.get("tokens_spent") is None for r in raw),
            **{k: v for k, v in dedup.items() if k != "rows"},
            "rows_after_today": future, "invalid_timestamps": invalid_dates,
            "note": "Prior gpt-5.4 demos: provided expert demos, 0 new calls today." if name in {"alfworld", "appworld"}
                    else "All distinct attempts billed, including failed/unverified attempts. tokens_spent retains its recorded meaning."}


def _mapped_estimate(value) -> dict | None:
    rule = "--gen-cost-json prompt_hash mapping"
    if isinstance(value, dict):
        rule = value.get("rule", rule)
        value = value.get("value", value.get("estimated_tokens", value.get("tokens")))
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return estimate(number, rule) if math.isfinite(number) and number >= 0 else None


def pool_provenance(rows: Iterable[dict], demo_costs: dict,
                    gen_costs: dict | None = None) -> dict:
    """Attribute queries to events; bill demo task ids / generated prompts once.

    demo_costs maps task ids to audit_demos per_id records (or numeric output
    tokens). exact_tokens is completion/output usage; input and total usage are
    separately named. No prompt re-rendering or generated-id guesswork is used.
    """
    rows = list(rows)
    queries = {}
    prompts, tasks = set(), set()
    selected, missing_hints = 0, 0
    for number, row in enumerate(rows, 1):
        prompt = row.get("prompt")
        prompt_hash = hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:16] if isinstance(prompt, str) else None
        task_id, teacher = row.get("task_id"), str(row.get("teacher") or "")
        if prompt_hash:
            prompts.add(prompt_hash)
        if task_id:
            tasks.add(task_id)
        selected += flatten_tokens(row.get("token_hint"))
        missing_hints += row.get("token_hint") is None
        generated = teacher.startswith("teacher_authored")
        identity = prompt_hash if generated else task_id
        kind = "generated" if generated else "demo"
        if not teacher or identity is None:
            kind, identity = "unknown", f"row:{number}"
        key = f"{kind}:{identity}"
        query = queries.setdefault(key, {"query_id": key, "kind": kind, "teachers": set(),
            "task_ids": set(), "prompt_hashes": set(), "trajectories": set(), "events": []})
        query["teachers"].add(teacher)
        if task_id:
            query["task_ids"].add(task_id)
        if prompt_hash:
            query["prompt_hashes"].add(prompt_hash)
        trajectory = row.get("_traj")
        if trajectory:
            query["trajectories"].add(trajectory)
        rejected = row.get("_rejected")
        key_event = f"{trajectory}|{prompt_hash}|{hashlib.sha1(rejected.encode('utf-8')).hexdigest()[:16]}" if (
            trajectory and prompt_hash and isinstance(rejected, str)) else None
        query["events"].append({"row": row.get("_audit_row", number), "pool": row.get("_audit_pool"),
            "task_id": task_id, "prompt_hash": prompt_hash, "_traj": trajectory, "event_key": key_event})
    exact_output = exact_input = estimated_output = 0
    unresolved = Counter()
    for query in queries.values():
        for field in ("teachers", "task_ids", "prompt_hashes", "trajectories"):
            query[field] = sorted(query[field])
        query["rows"] = len(query["events"])
        if query["kind"] == "generated":
            mapped = _mapped_estimate((gen_costs or {}).get(query["prompt_hashes"][0]))
            query["status"] = "resolved_estimate" if mapped else "unresolved_estimate"
            query["estimated_tokens"] = mapped or estimate(None, "No usable --gen-cost-json entry for this prompt_hash")
            if mapped:
                estimated_output += mapped["value"]
        elif query["kind"] == "demo":
            record = demo_costs.get(query["task_ids"][0])
            query["status"] = "resolved_exact" if record is not None else "unresolved_exact"
            if record is not None:
                output = record["output_tokens"] if isinstance(record, dict) else exact(record, "provided demo output cost by task_id")
                incoming = record.get("input_tokens") if isinstance(record, dict) else None
                query["exact_tokens"] = output if isinstance(output, dict) else exact(output, "provided demo output cost by task_id")
                exact_output += query["exact_tokens"]["value"]
                if incoming is not None:
                    query["exact_input_tokens"] = incoming if isinstance(incoming, dict) else exact(incoming, "provided demo input cost by task_id")
                    exact_input += query["exact_input_tokens"]["value"]
                query["attempts"] = record.get("attempts") if isinstance(record, dict) else None
                query["verified"] = record.get("verified") if isinstance(record, dict) else None
            else:
                query["exact_tokens"] = exact(None, "Demo task_id absent from available attempt directories")
        else:
            query["status"] = "unresolved_identity"
        if query["status"].startswith("unresolved"):
            unresolved[query["status"] + "_rows"] += query["rows"]
            unresolved[query["status"] + "_queries"] += 1
    return {"rows": len(rows), "unique_prompts": len(prompts), "unique_task_ids": len(tasks),
            "unique_teacher_queries": len(queries), "queries": list(queries.values()),
            **{f"{status}_{unit}": unresolved[f"{status}_{unit}"] for status in (
                "unresolved_estimate", "unresolved_exact", "unresolved_identity") for unit in ("rows", "queries")},
            "exact_tokens": exact(exact_output, "demo output_token_count across all attempts, once per task_id; resolved queries only"),
            "exact_input_tokens": exact(exact_input, "demo input_token_count, once per task_id; available input usage only"),
            "exact_total_tokens": exact(exact_input + exact_output, "available exact demo input + output usage; no estimated usage"),
            "estimated_tokens": estimate(estimated_output, "sum --gen-cost-json estimates once per prompt_hash; resolved queries only"),
            "selected_supervision_text_tokens": exact(selected, "sum pool token_hint (y^T length) over all rows"),
            "missing_token_hint_rows": missing_hints,
            "note": "Query costs are attribution views of campaign costs, not additional charges. Multiple events/trajectories share one demo task query or generated prompt query; demo queries include every campaign attempt."}


def audit_training(path: str | Path, epochs: int = 1) -> dict:
    rows, info = read_jsonl(path)
    loss = sum(flatten_tokens(r.get("n_tok_T")) + flatten_tokens(r.get("n_tok_S")) for r in rows)
    context = 2 * sum(flatten_tokens(r.get("prompt_tokens")) for r in rows)
    return {**info, "epochs": epochs,
            "missing_token_rows": sum(any(r.get(k) is None for k in ("n_tok_T", "n_tok_S", "prompt_tokens")) for r in rows),
            "per_epoch": {"loss_tokens": exact(loss, "sum(n_tok_T + n_tok_S), pairwise"),
                          "context_tokens": exact(context, "2 * sum(prompt_tokens), one context for each response")},
            "all_epochs": {"loss_tokens": exact(loss * epochs, "epochs * per_epoch.loss_tokens"),
                           "context_tokens": exact(context * epochs, "epochs * per_epoch.context_tokens"),
                           "total_tokens": exact((loss + context) * epochs, "epochs * (pairwise loss tokens + doubled context tokens)")}}


def audit_appworld_events(path: str | Path) -> dict:
    rows, info = read_jsonl(path)
    totals = Counter()
    fields = ("student_calls", "student_completion_tokens", "episodes", "wall_s")
    missing = Counter()
    for row in rows:
        cost = row.get("_cost") or {}
        for field in fields:
            if cost.get(field) is None:
                missing[field] += 1
            else:
                totals[field] += float(cost[field]) if field == "wall_s" else flatten_tokens(cost[field])
        totals["teacher_tokens"] += flatten_tokens(row.get("_teacher_tokens"))
        missing["teacher_tokens"] += row.get("_teacher_tokens") is None
    if rows and all(not r.get("_cost") for r in rows):
        info["status"] = "not_logged"
    return {**info, "student_calls": totals["student_calls"], "episodes": totals["episodes"], "wall_s": totals["wall_s"],
            "student_completion_tokens": exact(totals["student_completion_tokens"], "sum _cost.student_completion_tokens over rows"),
            "teacher_tokens": exact(totals["teacher_tokens"], "sum _teacher_tokens over rows; replay values are not new queries"),
            "rows_missing_fields": dict(missing)}


def build_report(args: argparse.Namespace) -> dict:
    support, support_info = read_json(args.support_split)
    demos = audit_demos(args.demo_dirs or [DEMO_BASE + f"_a{i}" for i in (1, 2, 3)], args.verified_dir, set(support.get("demand", [])))
    generation_paths = args.generation_files if args.generation_files is not None else glob.glob(args.generation_glob)
    generation = audit_generation(generation_paths, args.generation_usage, tokenizer_name=args.tokenizer, include_smoke=args.include_smoke)
    generation["input_glob"] = args.generation_glob if args.generation_files is None else None
    ledgers = {name: audit_ledger(getattr(args, f"{name}_ledger"), name, args.today)
               for name in ("bfcl", "alfworld", "appworld", "tau2")}
    gen_costs, mapping_info = read_json(args.gen_cost_json) if args.gen_cost_json else ({}, {"status": "not_provided"})
    pools, all_rows = {}, []
    for path in dict.fromkeys(args.pool or ["data/bfcl_sft/pool_events_pref_v3t.jsonl"]):
        rows, info = read_jsonl(path)
        located = [{**r, "_audit_pool": path, "_audit_row": i} for i, r in enumerate(rows, 1)]
        pools[path] = {**info, **pool_provenance(located, demos["per_id"], gen_costs)}
        all_rows.extend(located)
    provenance = {"pools": pools, "gen_cost_mapping": mapping_info,
                  "combined": pool_provenance(all_rows, demos["per_id"], gen_costs),
                  "note": "Combined query costs are deduplicated across pools. Do not sum per-pool query costs."}
    training = audit_training(args.index, args.epochs)
    appworld = {path: audit_appworld_events(path) for path in dict.fromkeys(args.appworld_events or [
        "data/appworld_events/aw1_base_k3.jsonl", "data/appworld_events/aw1_spread_k3.jsonl"])}
    _, alf_info = read_jsonl(args.alfworld_events)
    student = {"appworld": appworld, "appworld_totals": {
        **{field: sum(v[field] for v in appworld.values()) for field in ("student_calls", "episodes", "wall_s")},
        **{field: exact(sum(v[field]["value"] for v in appworld.values()), f"sum AppWorld event-file {field}")
           for field in ("student_completion_tokens", "teacher_tokens")}},
        "alfworld": {**alf_info, "status": "not_logged" if alf_info["status"] == "ok" else alf_info["status"],
                     "note": "ALFWorld events have no cost fields; row count is not an inference-cost estimate."},
        "bfcl_mining": {"status": "not_logged", "note": "BFCL mining calls were not logged."}}
    future = [r for ledger in ledgers.values() for r in ledger["rows_after_today"]]
    invalid_dates = [{"ledger": name, **r} for name, ledger in ledgers.items() for r in ledger["invalid_timestamps"]]
    bootstrap = demos["groups"]["memory_prereq"]
    demo_expert_output = sum(demos["groups"][g]["output_tokens"]["value"] for g in ("demand", "other_official"))
    provided_components = {name: ledgers[name]["tokens_spent"] for name in ("alfworld", "appworld")}
    provided_components["bfcl_demo_campaign_excluding_bootstrap"] = exact(demo_expert_output, "BFCL demand + other_official demo output; every attempt")
    checks = {"no_ledger_rows_after_today": {"passed": not future and not invalid_dates,
        "today": args.today.isoformat(), "offending_rows": future, "invalid_timestamps": invalid_dates,
        "note": "Compare calendar dates as recorded, inclusive through --today; inspect raw rows before deduplication."}}
    summary = {
        "provided_expert_demos": {"components": provided_components,
            "exact_tokens": exact(sum(c["value"] for c in provided_components.values()), "ALFWorld/AppWorld recorded tokens_spent + non-bootstrap BFCL demo output"),
            "input_status": {"bfcl_demos": demos["status"], **{n: ledgers[n]["status"] for n in ("alfworld", "appworld")}},
            "note": "Prior ALFWorld/AppWorld gpt-5.4 demos; 0 new calls today. BFCL memory prerequisites are partitioned into fixed_bootstrap_overhead."},
        "fixed_bootstrap_overhead": {**bootstrap, "status": demos["status"],
            "note": "memory_prereq demo campaign: fixed memory-benchmark snapshot bootstrap overhead."},
        "new_paid_teacher_calls_today": {"calls": 0, "exact_tokens": exact(0, "this offline audit makes no teacher calls"),
            "timestamp_check": checks["no_ledger_rows_after_today"],
            "note": "Zero describes this audit, not an inference that historical ledgers capture every possible call."},
        "selected_supervision_text_tokens": {**provenance["combined"]["selected_supervision_text_tokens"],
            "rows": len(all_rows), "input_status": {p: v["status"] for p, v in pools.items()},
            "note": "Volume of selected y^T text; already produced supervision, not a fresh teacher charge."},
        "training_tokens": training,
        "student_inference_env_cost": student,
    }
    return {"passed": all(c["passed"] for c in checks.values()), "today": args.today.isoformat(),
            "support_split": support_info, "bfcl_demos": demos, "bfcl_generation": generation,
            "ledgers": ledgers, "pool_provenance": provenance, "training_tokens": training,
            "student_environment_cost": student, "checks": checks, "summary": summary,
            "notes": ["All token figures retain basis and rule. Missing inputs/fields mean partial observed sums, not known zero costs.",
                "Campaign usage, pool attribution, selected text, and training volume are separate accounting views; do not add them.",
                "BFCL/tau2 ledgers remain separately reported; BFCL ledger usage is not added to the demo campaign again."]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="results/behavior_atom_v1/p0/cost_ledger.json")
    parser.add_argument("--support-split", default="configs/bfcl_support_split.json")
    parser.add_argument("--demo-dirs", "--demo-dir", action="extend", nargs="+", help="Attempt directories, replacing a1/a2/a3 defaults")
    parser.add_argument("--verified-dir", default=DEMO_BASE)
    parser.add_argument("--generation-glob", default="data/bfcl_sft/gen*.jsonl")
    parser.add_argument("--generation-files", nargs="+", help="Explicit generation files, replacing --generation-glob")
    parser.add_argument("--generation-usage", default="data/teacher_ledger/bfcl_generation_usage.jsonl")
    parser.add_argument("--include-smoke", action="store_true")
    parser.add_argument("--tokenizer", nargs="?", const="Qwen/Qwen3.5-4B", help="Optional local tokenizer name/path; no downloads")
    for name in ("bfcl", "alfworld", "appworld", "tau2"):
        parser.add_argument(f"--{name}-ledger", default=f"data/teacher_ledger/{name}.jsonl")
    parser.add_argument("--pool", action="append", help="Repeatable pool JSONL; replaces default BFCL pool")
    parser.add_argument("--gen-cost-json", help="Optional prompt_hash -> estimated output tokens mapping")
    parser.add_argument("--index", default="data/fingerprints/bfcl_r3t_v1.index.jsonl")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--appworld-events", action="extend", nargs="+", help="Event files, replacing base/spread defaults")
    parser.add_argument("--alfworld-events", default="data/alf_sft/events_v1.jsonl")
    parser.add_argument("--today", type=date.fromisoformat, default=date.today(), help="Inclusive ledger date cutoff, YYYY-MM-DD")
    args = parser.parse_args(argv)
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    report = build_report(args)
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(path), "passed": report["passed"], "summary": report["summary"]}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
