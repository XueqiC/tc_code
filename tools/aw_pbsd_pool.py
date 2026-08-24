#!/usr/bin/env python3
"""PBSD pool: teacher positives paired with on-policy student negatives.

For the first teacher-demo turn of each task where the student's own
sampled attempt failed verification, attach that failed first turn as
_rejected. All other rows train as plain positives. No teacher calls.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

fail_first: dict[str, str] = {}
for tag in [f"awb3d_samp_t07_s{s}" for s in (10, 11, 12, 13)]:
    rdir = ROOT / "results/appworld" / tag
    verdict = {}
    for line in (rdir / "records.jsonl").open():
        r = json.loads(line)
        verdict[r["task_id"]] = r["passed_tests"] > r["failed_tests"]
    for line in (rdir / "transcripts.jsonl").open():
        t = json.loads(line)
        if verdict.get(t["task_id"]) or t["task_id"] in fail_first:
            continue
        for m in t["messages"]:
            if m.get("role") == "assistant" and m["content"].strip():
                fail_first[t["task_id"]] = m["content"]
                break

rows = [json.loads(l) for l in (ROOT / "data/appworld_sft/pool.jsonl").open()]
first_turn: dict[str, int] = {}
for i, r in enumerate(rows):
    ti = r.get("turn_index", 999)
    if r["task_id"] not in first_turn or ti < rows[first_turn[r["task_id"]]].get("turn_index", 999):
        first_turn[r["task_id"]] = i
n = 0
for task, idx in first_turn.items():
    rej = fail_first.get(task)
    if rej and rej != rows[idx]["response"]:
        rows[idx]["_rejected"] = rej
        n += 1
out = ROOT / "data/appworld_sft/pool_pbsd.jsonl"
with out.open("w") as fh:
    for r in rows:
        fh.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"pbsd pairs={n} rows={len(rows)} -> {out}")
