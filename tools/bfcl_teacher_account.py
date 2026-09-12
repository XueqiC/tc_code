#!/usr/bin/env python3
"""Inventory and ledger bridge for the official batch teacher collector."""

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ / "src"))

from bfas.adapters.bfcl import BFCLAdapter, BFCL_ROOT
from bfas.bfcl_teacher import record_attempt, result_source_id, teacher_task_ids
from bfas.ledger import read_records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "complete", "record", "merge"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--split", type=Path, default=PROJ / "configs/bfcl_support_split.json")
    parser.add_argument("--attempt", type=int, choices=(1, 2, 3), default=1)
    args = parser.parse_args(argv)
    os.environ["BFAS_BFCL_TEACHER"] = args.model
    demand = teacher_task_ids(json.loads(args.split.read_text())["demand"],
                              record_inventory=args.action == "prepare")
    safe = re.sub(r"[^A-Za-z0-9]", "_", args.model).rstrip("_")
    prefix = f"demos_{safe}"
    result_dir = BFCL_ROOT / f"result_{prefix}_a{args.attempt}"
    if args.action == "prepare":
        selected = BFCLAdapter()._selective_file(demand) if demand else {}
        (BFCL_ROOT / "test_case_ids_to_generate.json").write_text(json.dumps(selected, indent=1) + "\n")
        print(f"[bfcldemos] demand={len(demand)}, generated inventory={sum(map(len, selected.values()))}")
    elif args.action == "complete":
        have = set()
        for path in result_dir.rglob("*_result.json"):
            have.update(json.loads(line)["id"] for line in path.read_text().splitlines() if line.strip())
        return 0 if set(demand) <= have else 1
    elif args.action == "record":
        record_attempt(
            BFCLAdapter(), result_dir, BFCL_ROOT / f"score_{prefix}_a{args.attempt}",
            demand, args.model, args.attempt - 1, 0.0 if args.attempt == 1 else 0.7,
        )
    else:
        # Only demos admitted by the gateway-schema import enter training.
        admitted = {row.get("source_id") for row in read_records("bfcl")
                    if row["teacher"] == args.model and row["verified"]}
        merged = BFCL_ROOT / f"result_{prefix}"
        if merged.exists():
            shutil.rmtree(merged)
        verified = set()
        for attempt in (1, 2, 3):
            directory = BFCL_ROOT / f"result_{prefix}_a{attempt}"
            for path in sorted(directory.rglob("*_result.json")):
                for line in path.read_text().splitlines():
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    task_id = row["id"]
                    source_id = result_source_id(directory, args.model, row)
                    if task_id not in demand or task_id in verified or source_id not in admitted:
                        continue
                    destination = merged / path.relative_to(directory)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with destination.open("a") as handle:
                        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    verified.add(task_id)
        output = PROJ / "data" / f"bfcl_demos_{safe}_verified.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(sorted(verified)) + "\n")
        # Retain the historical default consumed by downstream pool tools.
        (PROJ / "data/bfcl_demos_ds_verified.json").write_text(json.dumps(sorted(verified)) + "\n")
        print(f"[bfcldemos] verified={len(verified)}/{len(demand)}; ledger=data/teacher_ledger/bfcl.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
