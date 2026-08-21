#!/usr/bin/env python3
"""Summarize the 28-cell harvest of 2026-08-21 into one verdict table."""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

GROUPS = {
    "v12sc4b": ["v12sc4b_s0", "v12sc4b_s1", "v12sc4b_s2"],
    "v12sc9b": ["v12sc9b_s0", "v12sc9b_s1", "v12sc9b_s2"],
    "v12sql": ["v12sql_s0", "v12sql_s1", "v12sql_s2"],
    "att": ["att_s0", "att_s1", "att_s2"],
    "sv": ["sv_s0", "sv_s1", "sv_s2"],
    "r1": ["r1_s0", "r1_s1", "r1_s2"],
    "k10v12": ["k10v12_s0", "k10v12_s1", "k10v12_s2"],
    "b2kv12": ["b2kv12_s0", "b2kv12_s1"],
    "b40kv12": ["b40kv12_s0", "b40kv12_s1"],
    "toksel": ["toksel_s0", "toksel_s1", "toksel_s2"],
}


def read(tag):
    p = ROOT / "results" / f"e3_tier1_{tag}" / "metrics.json"
    if not p.exists():
        return None
    try:
        return json.load(p.open())
    except Exception:
        return None


for gname, tags in GROUPS.items():
    per_cond = {}
    for tag in tags:
        m = read(tag)
        if not m:
            print(f"{gname}: {tag} MISSING/PARTIAL")
            continue
        for c, d in m.get("conditions", {}).items():
            if not isinstance(d, dict) or not d.get("eval") or c == "base":
                continue
            ev = d["eval"]
            ext = (ev.get("gsm8k_ext") or {}).get("exec_acc")
            key = c
            if ext is not None:
                per_cond.setdefault(key, []).append(ext)
            else:
                per_cond.setdefault(key + "(int)", []).append(
                    ev["gsm8k_code"]["exec_acc"]
                )
    for c, v in sorted(per_cond.items()):
        arr = np.array(v)
        print(
            f"{gname:9} {c:22} n={len(v)} "
            f"mean={arr.mean():.3f} vals={[round(x,3) for x in v]}"
        )
    print()
