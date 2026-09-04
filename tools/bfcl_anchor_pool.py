#!/usr/bin/env python3
"""Self-anchored advantage pool (mechanism doc §3).

Rows with p̂ < 1 keep their advantage weight (1−p̂). Rows of tasks the
student already solves (p̂ = 1, advantage 0, previously inert) become
behavior anchors: their weight is set to λ chosen by training-mass
conservation, λ = Σ(1−p̂)⁺ over advantage rows / #anchor rows, encoded
through the existing trainer interface as _task_phat = 1 − λ.
"""
import json, sys
from pathlib import Path

src = Path("data/bfcl_sft/pool_bfcl_ds_v3mt_v3.jsonl")
out = Path("data/bfcl_sft/pool_bfcl_anchor_v1.jsonl")
rows = [json.loads(l) for l in src.open()]
adv_rows = [r for r in rows if float(r.get("_task_phat", 0)) < 1.0]
anchor_rows = [r for r in rows if float(r.get("_task_phat", 0)) >= 1.0]
mass = sum(1.0 - float(r["_task_phat"]) for r in adv_rows)
lam = mass / max(len(anchor_rows), 1)
for r in anchor_rows:
    r["_task_phat"] = round(1.0 - lam, 6)
    r["_anchor"] = True
with out.open("w") as fh:
    for r in rows:
        fh.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"[anchor] adv_rows={len(adv_rows)} (mass {mass:.2f}) "
      f"anchor_rows={len(anchor_rows)} lambda={lam:.4f} -> {out}")
