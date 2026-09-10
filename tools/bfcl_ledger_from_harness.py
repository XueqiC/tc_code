#!/usr/bin/env python3
"""Recover BFAS ledger evidence from archived BFCL attempts; never call a teacher.

Missing usage is a recorded lower bound, never an invented exact zero. Batch
memory prerequisites are charged once to the first dependent demand task (in
split order). This retains the adapter's whole-prerequisite charge without
charging a shared generation twice. Attempt indices use the gateway's zero base.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from bfas.adapters.bfcl import BFCLAdapter, BFCL_ROOT, _completion_tokens, extract_verdicts
from bfas.ledger import demo_payload
from bfas.rtd.persistence import file_hash

TEACHER = "azure/gpt-5.4-FC"
STEM = "demos_azure_gpt_5_4_FC"


def source_info(path):
    path = Path(path)
    return dict(path=str(path), sha256=file_hash(path),
                timestamp=datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat())


def read_results(directory):
    """Read JSONL result files, refusing duplicate ids or an empty archive."""
    output = {}
    for path in sorted(Path(directory).rglob("*_result.json")):
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            tid = row["id"]
            if tid in output:
                raise ValueError(f"duplicate result id {tid} in {directory}")
            output[tid] = (row, path, number)
    if not output:
        raise ValueError(f"no harness results in {directory}")
    return output


def preferred_demo(adapter, result, attempt_index, cached, cache_path):
    original = demo_payload(adapter.demo_from_result(result, attempt_index))
    tid = result["id"]
    if tid not in cached:
        return original, dict(demo_source="BFCLAdapter.demo_from_result", turns_match_harness=True)
    saved = cached[tid]
    if saved.get("task_id", tid) != tid or saved.get("raw", {}).get("checker_verified") is not True:
        raise ValueError(f"invalid verified adapter demo: {tid}")
    if not isinstance(saved.get("turns"), list):
        raise ValueError(f"missing adapter turns: {tid}")
    # Keep the saved bytes, including empty stateful turns. Do not silently
    # attach a re-run target to an exact cost for a different completion.
    payload = copy.deepcopy({k: saved[k] for k in ("turns", "worked_example")})
    return payload, dict(demo_source=str(cache_path),
                         adapter_attempt=saved.get("raw", {}).get("attempt"),
                         turns_match_harness=payload["turns"] == original["turns"])


def convert_attempt(adapter, demand, result_dir, score_dir, attempt_index, temperature,
                    cached, cache_path):
    results = read_results(result_dir)
    categories = adapter.task_categories()
    task_ids = [tid for tid in demand if tid in results]
    expected = {tid: categories[tid] for tid in task_ids}
    score_paths = {p.stem.removeprefix("BFCL_v4_").removesuffix("_score"): p
                   for p in Path(score_dir).rglob("*_score.json")}
    verdicts = extract_verdicts(score_dir, expected)
    owned = {tid: [tid] for tid in task_ids}
    claimed = set(task_ids)
    dependencies = {}
    for tid in task_ids:
        dependencies[tid] = adapter._memory_prereqs.get(tid, [])
        for dep in dependencies[tid]:
            if dep in results and dep not in claimed:
                owned[tid].append(dep)
                claimed.add(dep)
    if results.keys() - claimed:
        raise ValueError(f"unattributed results: {sorted(results.keys() - claimed)}")
    rows = []
    for tid in task_ids:
        result, path, number = results[tid]
        parts, known, missing = [], 0, []
        for rid in owned[tid]:
            entry, part_path, line = results[rid]
            cost = _completion_tokens([entry])
            parts.append(dict(result_id=rid, file=source_info(part_path), line=line,
                              output_tokens=cost))
            if cost is None:
                missing.append(rid)
            else:
                known += cost
        score_path = score_paths.get(categories[tid])
        inference_error = isinstance(result.get('result'), str) and result['result'].startswith('Error during inference:')
        provenance = dict(result_file=source_info(path), result_line=number,
                          score_file=source_info(score_path) if score_path else None,
                          verification_status="scored" if score_path else "missing_score_category",
                          official_score_verified=verdicts[tid], inference_error=inference_error,
                          cost_components=parts, memory_dependencies=dependencies[tid],
                          prerequisite_attribution="once_to_first_dependent_in_split_order",
                          missing_usage_result_ids=missing,
                          tokens_spent_scope="recorded_output_tokens_lower_bound" if missing
                          else "exact_provider_completion_tokens")
        row = dict(task_id=tid, teacher=TEACHER, attempt_index=attempt_index,
                   temperature=temperature, verified=verdicts[tid] and not inference_error, tokens_spent=known,
                   purpose="teacher", timestamp=source_info(path)["timestamp"],
                   cost_confidence="estimated" if missing else "exact", provenance=provenance)
        if row["verified"]:
            if missing:
                raise ValueError(f"verified attempt missing exact usage: {tid}")
            row["demo"], detail = preferred_demo(adapter, result, attempt_index, cached, cache_path)
            provenance.update(detail)
            if not detail["turns_match_harness"]:
                row["cost_confidence"] = "estimated"
                provenance["demo_cost_basis"] = "original_harness_usage_proxy_for_unknown_rerun_cost"
        rows.append(row)
    return rows


def build_ledger(harness, split, adapter_demos, *, adapter=None, adapter_log=None):
    adapter = adapter or BFCLAdapter()
    demand = json.loads(Path(split).read_text())["demand"]
    if len(demand) != len(set(demand)):
        raise ValueError("duplicate demand task")
    cached = json.loads(Path(adapter_demos).read_text())
    if cached.keys() - set(demand):
        raise ValueError("adapter demo outside demand split")
    rows = []
    for index, temperature in enumerate((0.001, 0.7, 0.7)):
        rows.extend(convert_attempt(adapter, demand,
            Path(harness)/f"result_{STEM}_a{index + 1}",
            Path(harness)/f"score_{STEM}_a{index + 1}", index, temperature, cached, adapter_demos))
    # teacher_demo deletes its UUID result/score directories in finally. Only
    # explicitly identified surviving directories could prove extra paid rows;
    # never count the merged copy as a new call or infer API calls from demos.
    rerun = dict(extra_calls="unknown", exact_output_tokens=None, added_ledger_rows=0,
        verified_demos=len(cached),
        existing_harness_results_reused="as accounting evidence; not proof of API cache reuse",
        reason="adapter re-run usage is absent from serialized demos; deleted results cannot recover API retries or completion totals")
    if adapter_log is not None:
        log = Path(adapter_log).read_text()
        score_names = list(dict.fromkeys(re.findall(r"score_bfas_[a-f0-9]+", log)))
        rerun.update(log=source_info(adapter_log), generation_passes=log.count(
            "Generating results for ['azure/gpt-5.4-FC']"), result_directories=[])
        for name in score_names:
            directory = Path(harness)/name.replace("score_", "result_", 1)
            rerun["result_directories"].append(dict(path=str(directory), exists=directory.exists()))
            if directory.exists() and any(directory.rglob("*_result.json")):
                # Do not silently omit a recoverable re-run. Its run order and
                # temperatures need to be bound before it can be charged.
                raise ValueError(f"surviving adapter re-run needs explicit accounting: {directory}")
    unknown = [dict(task_id=r["task_id"], attempt_index=r["attempt_index"],
                    result_ids=r["provenance"]["missing_usage_result_ids"])
               for r in rows if r["provenance"]["missing_usage_result_ids"]]
    sidecar = dict(version="bfcl-harness-ledger-v1", teacher=TEACHER,
        pool_cost_status="exact-plus-unknown-rerun", split=source_info(split),
        adapter_demos=source_info(adapter_demos), adapter_rerun=rerun,
        historical_cost_status="recorded-lower-bound-plus-unknown",
        recorded_output_tokens=sum(r["tokens_spent"] for r in rows),
        missing_usage_attempts=unknown,
        unknown_usage_note="Missing usage in quota-error records is not evidence of zero spend; stateful failures may discard earlier completions. tokens_spent holds only surviving counts.",
        attempt_count=len(rows), verified_attempts=sum(r["verified"] for r in rows),
        adapter_target_substitutions=[r["task_id"] for r in rows
            if r["verified"] and not r["provenance"]["turns_match_harness"]],
        verified_without_training_turns=[r["task_id"] for r in rows
            if r["verified"] and not r["demo"]["turns"]],
        missing_score_attempts=[dict(task_id=r["task_id"], attempt_index=r["attempt_index"])
            for r in rows if r["provenance"]["score_file"] is None])
    return rows, sidecar


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--harness", type=Path, default=BFCL_ROOT)
    p.add_argument("--split", type=Path, default=ROOT/"configs/bfcl_support_split.json")
    p.add_argument("--adapter-demos", type=Path, default=ROOT/"data/bfcl_sft/demos_gpt54_adapter.json")
    p.add_argument("--adapter-log", type=Path)
    p.add_argument("--out", type=Path, default=ROOT/"data/teacher_ledger/bfcl_gpt54.jsonl")
    args = p.parse_args(argv)
    rows, sidecar = build_ledger(args.harness, args.split, args.adapter_demos, adapter_log=args.adapter_log)
    write_jsonl(args.out, rows)
    sidecar["ledger_sha256"] = file_hash(args.out)
    args.out.with_suffix(".provenance.json").write_text(json.dumps(sidecar, indent=2) + "\n")
    print(json.dumps({k: sidecar[k] for k in ("attempt_count", "verified_attempts",
        "recorded_output_tokens", "pool_cost_status", "verified_without_training_turns")}, indent=2))


if __name__ == "__main__":
    main()
