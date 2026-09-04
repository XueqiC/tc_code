#!/usr/bin/env python3
"""ALFWorld consequentiality-ablation pools from the demo-replay event file.

Same candidate disagreement pool, four selections (user's next-stage prompt §4):
  A  all divergences           every probed student/demo disagreement (dU may be <= 0)
  B  first divergence only     the first probe of each demo (#p1)  -- nearest-prior baseline
  C  consequential only        dU > eps
  D  utility-weighted          same rows as C; the trainer weights by f(dU) (AW_DDPO_WEIGHT)
  E  negative teacher          C plus reversed pairs (student > demo) where dU < -0.2 (prompt 16.1)
Also writes size summaries so arms can be step-matched.

    python tools/alf_events_to_pools.py --events data/alf_sft/events_v1.jsonl --out-dir data/alf_sft
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", default="data/alf_sft/events_v1.jsonl")
    parser.add_argument("--eps", type=float, default=0.0)
    parser.add_argument("--out-dir", default="data/alf_sft")
    args = parser.parse_args()
    rows = [json.loads(l) for l in (ROOT / args.events).read_text().splitlines() if l.strip()]
    seen, uniq = set(), []
    for r in rows:
        k = (r["task_id"], int(r["turn_index"]), r["_rejected"][:200])
        if k in seen:
            continue
        seen.add(k)
        uniq.append(r)
    pools = {
        "A_all": uniq,
        "B_first": [r for r in uniq if str(r.get("_traj", "")).endswith("#p1")],
        "C_conseq": [r for r in uniq if float(r["_event_dU"]) > args.eps],
    }
    pools["D_weighted"] = pools["C_conseq"]
    # E: negative teacher evidence (user prompt 16.1) -- where the demo's action is demonstrably
    # WORSE for this student (dU < -eps_neg), prefer the student's own action over the demo's.
    eps_neg = 0.2
    reversed_rows = []
    for r in uniq:
        if float(r["_event_dU"]) < -eps_neg:
            q = dict(r)
            q["response"], q["_rejected"] = r["_rejected"], r["response"]
            q["_traj"] = str(r.get("_traj", "")) + "#neg"
            q["_negative_teacher"] = True
            q["_event_dU"] = -float(r["_event_dU"])  # magnitude of evidence for the reversal
            reversed_rows.append(q)
    pools["E_negteacher"] = pools["C_conseq"] + reversed_rows
    out = ROOT / args.out_dir
    for name, sel in pools.items():
        path = out / f"pool_{name}.jsonl"
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in sel))
        d_u = [float(r["_event_dU"]) for r in sel]
        print(f"[alf-pools] {name}: {len(sel)} rows, dU mean {sum(d_u)/max(len(d_u),1):+.3f}, "
              f"dU>0 {sum(1 for v in d_u if v > 0)}, dU<0 {sum(1 for v in d_u if v < 0)} -> {path}")
    neg = [r for r in uniq if float(r["_event_dU"]) < -args.eps]
    (out / "events_negative.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in neg))
    print(f"[alf-pools] negative-teacher evidence (dU<0): {len(neg)} rows -> events_negative.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
