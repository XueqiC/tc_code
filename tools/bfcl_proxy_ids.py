#!/usr/bin/env python3
"""Fix a small, stratified BFCL subset for fast iteration.

A full official run is 5217 entries and about three hours, which makes a
mechanism change cost half a day to judge. That is too slow to iterate a design.
This picks a fixed subset -- the same ids every time, so runs are comparable to
each other -- with equal weight per scored category, since the axes we are
trying to move (multi-turn, memory, irrelevance) are small ones that a
size-proportional sample would barely touch.

Support-split tasks are excluded: they are training material, and a proxy that
contains them measures memorisation.

The result is a PROXY. It is for choosing between mechanism versions, never for
a reported number -- those still come from the full official flow.

Usage: bfcl_proxy_ids.py [--per-category 20] [--out configs/bfcl_proxy_ids.json]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SEED = 71  # fixed so the proxy set never drifts between iterations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-category", type=int, default=20)
    # A stateful entry runs a dozen-plus turns and costs roughly twenty times a
    # single-turn one, and a memory task drags its whole prerequisite write
    # chain along. Budgeting the proxy by entry count therefore cuts almost only
    # the cheap half; the stateful families need their own, much smaller budget.
    parser.add_argument("--per-stateful", type=int, default=6)
    parser.add_argument("--per-memory", type=int, default=2)
    parser.add_argument("--out", default="configs/bfcl_proxy_ids.json")
    args = parser.parse_args()

    from bfas.adapters.bfcl import BFCLAdapter

    adapter = BFCLAdapter()
    split = json.loads((ROOT / "configs/bfcl_support_split.json").read_text())
    held_out = set(split["demand"]) | set(split["calibration"])

    by_category: dict[str, list[str]] = {}
    for task in adapter.task_pool():
        if task.task_id in held_out:
            continue
        by_category.setdefault(task.category, []).append(task.task_id)

    rng = random.Random(SEED)
    selection: dict[str, list[str]] = {}
    for category in sorted(by_category):
        ids = sorted(by_category[category])
        if category.startswith("memory"):
            budget = args.per_memory
        elif category.startswith(("multi_turn", "web_search")):
            budget = args.per_stateful
        else:
            budget = args.per_category
        take = min(budget, len(ids))
        selection[category] = sorted(rng.sample(ids, take))

    # memory tasks only mean anything with the write chain that precedes them
    selected = [i for ids in selection.values() for i in ids]
    full = adapter._selective_file(selected)

    out = ROOT / args.out
    out.write_text(json.dumps(full, indent=1) + "\n")
    total = sum(len(v) for v in full.values())
    print(f"[proxy] {total} entries across {len(full)} categories "
          f"-> {out}")
    for category in sorted(full):
        print(f"    {len(full[category]):4d}  {category}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
