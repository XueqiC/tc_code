#!/usr/bin/env python3
"""BFCL v4 support split for per-benchmark specialization (protocol
N=50: 40 demand + 10 calibration).

Stratified proportional allocation (largest remainder) over all test
categories, fixed seed. Support ids are excluded from every evaluation
aggregate; all methods and the base student are scored on the
complement.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "envs/bfcl/gorilla/berkeley-function-call-leaderboard/bfcl_eval/data"
SEED = 50
N_TOTAL = 50
N_CALIB = 10


def main() -> int:
    rng = random.Random(SEED)
    cats: dict[str, list[str]] = {}
    for f in sorted(DATA.glob("BFCL_v4_*.json")):
        ids: list[str] = []
        for line in f.open():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                # not JSONL (e.g. format_sensitivity is a plain JSON
                # object); such files are not scored test entries
                ids = []
                break
            if isinstance(row, dict) and "id" in row:
                ids.append(row["id"])
        cat = f.stem.replace("BFCL_v4_", "")
        # memory needs cross-conversation prerequisite snapshots; the
        # official selective generation cannot produce isolated
        # teacher demonstrations for it, so it stays eval-only
        if ids and cat != "memory":
            cats[cat] = ids
    total = sum(len(v) for v in cats.values())

    # allocation over categories: equal per capability dimension by
    # default (support v2; proportional sampling hid the deficit
    # categories, see docs/analysis-appworld-vs-bfcl.md), proportional
    # with --proportional for the archived v1 protocol
    import sys
    if "--proportional" in sys.argv:
        quotas = {c: N_TOTAL * len(v) / total for c, v in cats.items()}
    else:
        quotas = {c: N_TOTAL / len(cats) for c in cats}
    alloc = {c: int(q) for c, q in quotas.items()}
    remainder = sorted(cats, key=lambda c: quotas[c] - alloc[c], reverse=True)
    for c in remainder:
        if sum(alloc.values()) >= N_TOTAL:
            break
        alloc[c] += 1

    support: list[str] = []
    for c, k in sorted(alloc.items()):
        if k > 0:
            support.extend(rng.sample(cats[c], k))
    rng.shuffle(support)
    calib = sorted(support[:N_CALIB])
    demand = sorted(support[N_CALIB:])

    out = ROOT / "configs/bfcl_support_split.json"
    json.dump(
        {"seed": SEED, "demand": demand, "calibration": calib,
         "allocation": {c: k for c, k in sorted(alloc.items()) if k}},
        out.open("w"), indent=1, ensure_ascii=False)
    print(f"[bfclsplit] demand={len(demand)} calib={len(calib)} -> {out}")
    for c, k in sorted(alloc.items()):
        if k:
            print(f"  {c}: {k}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
