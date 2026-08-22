#!/usr/bin/env python3
"""Attach turn-0 preference negatives to the selfmix corpus (aw11).

For every task the base model FAILED, the teacher demo's first turn is
the chosen response and the base model's own first turn is the rejected
one; both share the identical context, so the pair is exactly aligned.
Writes pool_selfmix_pref.jsonl with a _rejected field on those rows.
"""
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

rej = {}
rdir = ROOT / "results/appworld/aw7_base_rollout"
passed = set()
for line in (rdir / "records.jsonl").open():
    r = json.loads(line)
    if r["passed_tests"] > r["failed_tests"]:
        passed.add(r["task_id"])
for line in (rdir / "transcripts.jsonl").open():
    t = json.loads(line)
    if t["task_id"] in passed:
        continue
    for m in t["messages"]:
        if m.get("role") == "assistant" and m["content"].strip():
            rej[t["task_id"]] = m["content"]
            break

rows = [json.loads(l) for l in
        (ROOT / "data/appworld_sft/pool_selfmix.jsonl").open()]
first_turn = {}
for row in rows:
    if row.get("teacher") == "self":
        continue
    ti = row.get("turn_index", 999)
    if row["task_id"] not in first_turn or ti < first_turn[row["task_id"]]:
        first_turn[row["task_id"]] = ti
n = 0
out = []
for row in rows:
    if (row.get("teacher") != "self"
            and row.get("turn_index") == first_turn.get(row["task_id"])
            and row["task_id"] in rej
            and rej[row["task_id"]] != row["response"]):
        row["_rejected"] = rej[row["task_id"]]
        n += 1
    out.append(row)
p = ROOT / "data/appworld_sft/pool_selfmix_pref.jsonl"
with p.open("w") as fh:
    for r in out:
        fh.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"[prefpairs] negatives attached={n} rows={len(out)} -> {p}")
