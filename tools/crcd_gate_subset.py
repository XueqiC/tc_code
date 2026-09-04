#!/usr/bin/env python3
"""Validity gate, step 1: split the round-2 event pool by fingerprint cluster.

Reads the fingerprint report (with `gate.events[*].cluster`) and the preference
pool, writes one training pool per cluster plus the measurement task list
(all single-turn tasks in the pool) and the cluster centroids.

    python tools/crcd_gate_subset.py --fp results/analysis/crcd_fingerprints_v4.json \
        --pool data/bfcl_sft/pool_events_pref_v2.jsonl --out data/bfcl_sft/gate
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fp", default="results/analysis/crcd_fingerprints_v4.json")
    parser.add_argument("--pool", default="data/bfcl_sft/pool_events_pref_v2.jsonl")
    parser.add_argument("--out", default="data/bfcl_sft/gate")
    args = parser.parse_args()

    fp = json.loads((ROOT / args.fp).read_text())
    events = fp["gate"]["events"]
    rows = [json.loads(l) for l in (ROOT / args.pool).read_text().splitlines() if l.strip()]
    by_traj = {r.get("_traj", r["task_id"]): r for r in rows}
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    clusters: dict[int, list[dict]] = defaultdict(list)
    feats: dict[int, list[list[float]]] = defaultdict(list)
    for e in events:
        row = by_traj.get(e["traj"])
        if row is None:
            continue
        clusters[e["cluster"]].append(row)
        feats[e["cluster"]].append(e["a3"])
    single_ids = sorted({r["task_id"] for r in rows if int(r.get("turn_index", 0)) == 0 and "ev1" in str(r.get("_traj", ""))})
    (out / "measure_ids.json").write_text(json.dumps(single_ids))
    summary = {}
    for c, crows in sorted(clusters.items()):
        (out / f"pool_c{c}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in crows))
        cent = np.mean(np.asarray(feats[c]), axis=0)
        summary[c] = {
            "n": len(crows),
            "categories": sorted({r.get("_seed_category", "") for r in crows}),
            "single_task_ids": sorted({r["task_id"] for r in crows if r["task_id"] in single_ids}),
            "centroid": [round(float(v), 5) for v in cent],
        }
        print(f"[gate] cluster {c}: {len(crows)} events, cats={summary[c]['categories'][:6]}")
    (out / "clusters.json").write_text(json.dumps(summary, indent=1))
    print(f"[gate] measurement set: {len(single_ids)} single-turn tasks -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
