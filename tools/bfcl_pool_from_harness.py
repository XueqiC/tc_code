#!/usr/bin/env python3
"""Export adapter-compatible per-state BFCL demos from merged verified results.

Empty adapter stateful demos remain unavailable: never flatten an episode into
one invented training state. Run bfcl_pool_render_gemma4.py --verify afterwards.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from bfas.adapters.bfcl import BFCLAdapter, BFCL_ROOT
from bfas.rtd.persistence import file_hash
from tools.bfcl_ledger_from_harness import STEM, preferred_demo, read_results, source_info, write_jsonl


def build_pool(merged, ledger, adapter_demos, *, adapter=None):
    adapter = adapter or BFCLAdapter()
    results = read_results(merged)
    cached = json.loads(Path(adapter_demos).read_text())
    paid, all_ids = {}, set()
    for line in Path(ledger).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        all_ids.add(row['task_id'])
        if row["verified"]:
            if row["task_id"] in paid:
                raise ValueError("ambiguous successful ledger attempt")
            paid[row["task_id"]] = row
    if paid.keys() - results.keys() or results.keys() - all_ids:
        raise ValueError("merged verified ids differ from ledger verdicts")
    rows, excluded, sources = [], [], []
    for tid, (result, path, number) in results.items():
        if tid not in paid:
            excluded.append(dict(task_id=tid, reason="merged entry lacks a positive reconciled score"))
            continue
        purchase = paid[tid]
        # Merged is a copy of the scored attempt, never independent evidence.
        provenance = purchase["provenance"]
        original_path = Path(provenance["result_file"]["path"])
        if file_hash(original_path) != provenance["result_file"]["sha256"]:
            raise ValueError(f"harness result changed after ledger conversion: {tid}")
        original = json.loads(original_path.read_text().splitlines()[provenance["result_line"] - 1])
        if result != original:
            raise ValueError(f"merged result differs from scored attempt: {tid}")
        demo, detail = preferred_demo(adapter, result, purchase["attempt_index"], cached, adapter_demos)
        if demo != purchase["demo"]:
            raise ValueError(f"pool/ledger demo mismatch: {tid}")
        sources.append(dict(task_id=tid, file=source_info(path), line=number, **detail))
        if not demo["turns"]:
            excluded.append(dict(task_id=tid, reason="adapter has no per-state training turns"))
        for index, turn in enumerate(demo["turns"]):
            context = turn.get("context")
            if not isinstance(context, dict) or not {"messages", "functions"} <= context.keys():
                raise ValueError(f"missing actual per-state context: {tid}:{index}")
            rows.append(dict(task_id=tid, prompt=turn["prompt"], response=turn["target"],
                _render_context=copy.deepcopy(context),
                provenance=dict(demo_source=detail["demo_source"], step_index=index)))
    return rows, dict(version="bfcl-harness-pool-v1", ledger=source_info(ledger),
        merged_sources=sources, merged_tasks=len(results), verified_tasks=len(paid), pool_rows=len(rows),
        pool_tasks=len({r["task_id"] for r in rows}), excluded=excluded)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--merged", type=Path, default=BFCL_ROOT/f"result_{STEM}")
    p.add_argument("--ledger", type=Path, default=ROOT/"data/teacher_ledger/bfcl_gpt54.jsonl")
    p.add_argument("--adapter-demos", type=Path, default=ROOT/"data/bfcl_sft/demos_gpt54_adapter.json")
    p.add_argument("--out", type=Path, default=ROOT/"data/bfcl_sft/pool_gpt54_sft.jsonl")
    p.add_argument("--from-ledger", action="store_true", help="use original verified gateway demo payloads")
    args = p.parse_args(argv)
    if args.from_ledger:
        rows, provenance = pool_from_gateway_ledger(args.ledger)
    else:
        rows, provenance = build_pool(args.merged, args.ledger, args.adapter_demos)
    write_jsonl(args.out, rows)
    provenance["pool_sha256"] = file_hash(args.out)
    args.out.with_suffix(".provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({k: provenance[k] for k in ("verified_tasks", "pool_rows", "pool_tasks", "excluded")}, indent=2))


def pool_from_gateway_ledger(ledger):
    from bfas.ledger import read_records

    rows, sources, excluded, verified = [], [], [], set()
    for number, paid in enumerate(read_records(Path(ledger))):
        if not paid["verified"]:
            continue
        tid = paid["task_id"]
        if tid in verified:
            raise ValueError("ambiguous successful ledger attempt")
        verified.add(tid)
        turns = paid["demo"]["turns"]
        sources.append(dict(task_id=tid, ledger_row=number, attempt_index=paid["attempt_index"]))
        if not turns:
            excluded.append(dict(task_id=tid, reason="adapter has no per-state training turns"))
        for index, turn in enumerate(turns):
            context = turn.get("context")
            if not isinstance(context, dict) or not {"messages", "functions"} <= context.keys():
                raise ValueError(f"missing actual per-state context: {tid}:{index}")
            rows.append(dict(task_id=tid, prompt=turn["prompt"], response=turn["target"],
                _render_context=copy.deepcopy(context),
                provenance=dict(demo_source=str(ledger), ledger_row=number, step_index=index)))
    return rows, dict(version="bfcl-gateway-pool-v1", ledger=source_info(ledger),
        demo_sources=sources, verified_tasks=len(verified), pool_rows=len(rows),
        pool_tasks=len({r["task_id"] for r in rows}), excluded=excluded)


if __name__ == "__main__":
    main()
