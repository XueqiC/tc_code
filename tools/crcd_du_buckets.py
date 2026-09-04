#!/usr/bin/env python3
"""dU calibration study, step 1 (user prompt §5): bucket BFCL intervention events by estimated
intervention utility and write equal-size preference pools per bucket.

Buckets (on _event_dU): neg (<0) | zero (==0) | small (0,0.4] | medium (0.4,0.7] | large (>0.7).
Single-turn events have dU = 1 - u_minus (always > 0); negative/zero come from stateful events.
Each pool gets N rows (random, seed 0) and a suggested epoch count so that optimizer steps are
matched (~STEPS with batch 8).

    python tools/crcd_du_buckets.py --n 60 --steps 40 --out-dir data/bfcl_sft/du_calib
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVENTS = [
    "data/bfcl_sft/events_single_irrel_v2.jsonl",
    "data/bfcl_sft/events_single_oos_v2.jsonl",
    "data/bfcl_sft/events_v1.jsonl",
    "data/bfcl_sft/events_single_r3.jsonl",
    "data/bfcl_sft/events_single_oos_r3.jsonl",
    "data/bfcl_sft/events_r3.jsonl",
]


def bucket(d_u: float) -> str:
    if d_u < 0:
        return "neg"
    if d_u == 0:
        return "zero"
    if d_u <= 0.4:
        return "small"
    if d_u <= 0.7:
        return "medium"
    return "large"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", nargs="*", default=DEFAULT_EVENTS)
    parser.add_argument("--n", type=int, default=60)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--out-dir", default="data/bfcl_sft/du_calib")
    parser.add_argument("--category", default=None, help="restrict to one _seed_category (category-matched calibration)")
    args = parser.parse_args()
    rows, seen = [], set()
    for f in args.events:
        p = ROOT / f
        if not p.is_file():
            continue
        for l in p.read_text().splitlines():
            if not l.strip():
                continue
            r = json.loads(l)
            if not r.get("_rejected") or "_event_dU" not in r:
                continue
            k = (r.get("prompt", "")[:300], r["_rejected"][:200])
            if k in seen:
                continue
            seen.add(k)
            if "response" in r and "<think>" in str(r["response"]):
                r["response"] = r["response"].split("</think>")[-1].lstrip()
            if args.category and r.get("_seed_category") != args.category:
                continue
            rows.append(r)
    by = {}
    for r in rows:
        by.setdefault(bucket(float(r["_event_dU"])), []).append(r)
    out = ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(0)
    plan = {}
    for name in ("neg", "zero", "small", "medium", "large"):
        sel = list(by.get(name, []))
        rng.shuffle(sel)
        sel = sel[: args.n]
        if not sel:
            print(f"[du-calib] {name}: 0 rows (skipped)")
            continue
        epochs = max(1, round(args.steps * args.batch / len(sel)))
        (out / f"pool_{name}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in sel))
        d_u = [float(r["_event_dU"]) for r in sel]
        plan[name] = {"rows": len(sel), "epochs": epochs, "du_mean": round(sum(d_u) / len(d_u), 3),
                      "available": len(by[name])}
        print(f"[du-calib] {name}: {len(sel)}/{len(by[name])} rows, dU mean {plan[name]['du_mean']:+.3f}, epochs {epochs}")
    (out / "plan.json").write_text(json.dumps(plan, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
