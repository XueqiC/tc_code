#!/usr/bin/env python3
"""Post-hoc analyses for the ALFWorld event files (user 2026-09-03 21:00 requests).

1. K=8 re-estimation (events_v1_k8.jsonl): how many K=3 "zero" events flip positive / negative at K=8,
   the P(dU>0) posterior distribution, and agreement between the K=3 sign and the K=8 sign.
2. Event-value model (event_value_v1.jsonl): reachability by probe index and depth, editability,
   value = reach x edit x max(dU,0); share of consequential events that are deployment-reachable;
   top events by value.

    python tools/alf_event_analyze.py --k8 data/alf_sft/events_v1_k8.jsonl --value data/alf_sft/event_value_v1.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def p_pos(wg, k1, wb, k2, n=3000, seed=0):
    rng = random.Random(seed)
    return sum(rng.betavariate(wg + 1, k1 - wg + 1) > rng.betavariate(wb + 1, k2 - wb + 1) for _ in range(n)) / n


def load(p):
    return [json.loads(l) for l in (ROOT / p).read_text().splitlines() if l.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k8", default="data/alf_sft/events_v1_k8.jsonl")
    ap.add_argument("--value", default="data/alf_sft/event_value_v1.jsonl")
    ap.add_argument("--out", default="results/analysis/alf_event_analysis.json")
    a = ap.parse_args()
    rep = {}
    if (ROOT / a.k8).is_file():
        rows = load(a.k8)
        pos = sum(1 for r in rows if r["_event_dU"] > 0)
        neg = sum(1 for r in rows if r["_event_dU"] < 0)
        zero = len(rows) - pos - neg
        pp = []
        for r in rows:
            wg, wb = r["_event_wins"]
            k = r["_event_k"]
            pp.append(p_pos(wg, k, wb, k))
        hist = Counter(round(v, 1) for v in pp)
        rep["k8"] = {"n": len(rows), "k3_zero_now_positive": pos, "now_negative": neg, "still_zero": zero,
                     "P_pos_mean": sum(pp) / len(pp), "P_pos_gt_0.8": sum(1 for v in pp if v > 0.8),
                     "P_pos_lt_0.2": sum(1 for v in pp if v < 0.2), "P_pos_hist": {str(k): v for k, v in sorted(hist.items())}}
        print("[k8]", json.dumps(rep["k8"]))
    if (ROOT / a.value).is_file():
        ev = load(a.value)
        by_probe, by_depth = defaultdict(list), defaultdict(list)
        for e in ev:
            by_probe[str(e.get("_traj", "")).split("#")[-1]].append(e["_reach"])
            by_depth[min(int(e["_prefix_len"]), 5)].append(e["_reach"])
        cons = [e for e in ev if e["_event_dU"] > 0]
        reach_cons = sum(1 for e in cons if e["_reach"] > 0)
        top = sorted(ev, key=lambda e: -e["_value"])[:10]
        rep["value"] = {
            "n": len(ev), "reach_by_probe": {k: round(sum(v) / len(v), 3) for k, v in sorted(by_probe.items())},
            "reach_by_depth": {str(k): round(sum(v) / len(v), 3) for k, v in sorted(by_depth.items())},
            "reach_soft_mean": round(sum(e["_reach_soft"] for e in ev) / len(ev), 3),
            "consequential": len(cons), "consequential_reachable": reach_cons,
            "consequential_mean_reach": round(sum(e["_reach"] for e in cons) / max(len(cons), 1), 3),
            "value_gt0": sum(1 for e in ev if e["_value"] > 0),
            "top_by_value": [{"task": e["task_id"][:45], "probe": str(e.get("_traj", "")).split("#")[-1], "t": e["_event_turn"],
                              "good": e["_event_good"], "bad": e["_event_bad"], "dU": round(e["_event_dU"], 2),
                              "reach": e["_reach"], "value": round(e["_value"], 3)} for e in top],
        }
        print("[value]", json.dumps({k: v for k, v in rep["value"].items() if k != "top_by_value"}))
        for t in rep["value"]["top_by_value"][:5]:
            print("   ", t)
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=1))
    print(f"[analysis] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
