#!/usr/bin/env python3
"""BFCL v4 support split for per-benchmark specialization (protocol
N=50: 40 demand + 10 calibration).

Equal allocation per capability dimension (largest remainder over the
fractional part), fixed seed; --proportional restores the archived v1
size-weighted protocol. Categories come from the adapter so the memory
axis participates instead of being dropped as eval-only.
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
    # Categories come from the adapter, not from the data filenames:
    # `memory` is a group that only becomes runnable once it is expanded
    # into memory_kv / memory_vector / memory_rec_sum with ids rewritten
    # and the prerequisite write-chain carried along.  It used to be
    # dropped here as "eval-only" because isolated teacher demos looked
    # impossible; the adapter now carries the chain, so memory takes part
    # in support like every other axis.
    import sys as _sys
    _sys.path.insert(0, str(ROOT / "src"))
    from bfas.adapters.bfcl import BFCLAdapter

    cats: dict[str, list[str]] = {}
    for task in BFCLAdapter().task_pool():
        cats.setdefault(task.category, []).append(task.task_id)
    for ids in cats.values():
        ids.sort()
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
