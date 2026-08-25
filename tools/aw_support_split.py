#!/usr/bin/env python3
"""Frozen-protocol support split: k=50 tasks, 40 demand / 10 calibration.

Seeded, deterministic, written once to configs/support_split.json.
S_c is excluded from dictionary fitting, acquisition, training, and
checkpoint selection; it is reserved for boundary calibration and the
regression probe (method v3.0 §10).
"""
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "configs/support_split.json"

tasks = sorted({json.loads(l)["task_id"].rsplit("_", 1)[0] + "_" + json.loads(l)["task_id"].rsplit("_", 1)[1]
                for l in open(ROOT / "results/appworld/awb3d_samp_t07_s10/records.jsonl")})
rng = random.Random(50)
support = sorted(rng.sample(tasks, min(50, len(tasks))))
rng.shuffle(support)
split = {"support": sorted(support),
         "demand": sorted(support[:40]),
         "calibration": sorted(support[40:50]),
         "seed": 50, "source_tasks": len(tasks)}
OUT.write_text(json.dumps(split, indent=1))
print(f"support split -> {OUT}: {len(split['demand'])} demand, "
      f"{len(split['calibration'])} calibration (from {len(tasks)} train tasks)")
