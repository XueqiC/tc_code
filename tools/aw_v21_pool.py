#!/usr/bin/env python3
"""v2.1 round-1 pool: hinted self-authored successes as anchors.

Every assistant turn of a verified-success hinted trajectory becomes a
self-provenance training row. The demo block that was injected at
collection time is stripped from the system message, so training
reflects the hint-free deployment context. Same-task failed first
turns attach as success-axis _rejected pairs.
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TAGS = [f"awb4_hint_t07_s{s}" for s in (20, 21, 22, 23)]
DEMO_RE = re.compile(r"\n\nWorked example from an expert.*$", re.S)

rows = []
fail_first: dict[str, str] = {}
succ_seen: dict[str, int] = {}
for tag in TAGS:
    rdir = ROOT / "results/appworld" / tag
    verdict = {}
    for line in (rdir / "records.jsonl").open():
        r = json.loads(line)
        verdict[r["task_id"]] = r["passed_tests"] > r["failed_tests"]
    for line in (rdir / "transcripts.jsonl").open():
        t = json.loads(line)
        task = t["task_id"]
        msgs = [dict(m) for m in t["messages"]]
        for m in msgs:
            if m["role"] == "system":
                m["content"] = DEMO_RE.sub("", m["content"])
        if not verdict.get(task):
            for m in msgs:
                if m["role"] == "assistant" and m["content"].strip():
                    fail_first.setdefault(task, m["content"])
                    break
            continue
        if succ_seen.get(task, 0) >= 1:
            continue
        succ_seen[task] = 1
        for i, m in enumerate(msgs):
            if m["role"] != "assistant" or not m["content"].strip():
                continue
            context = msgs[:i]
            prompt_txt = "\n".join(
                f"<|{x['role']}|>\n{x['content']}" for x in context
            ) + "\n<|assistant|>\n"
            rows.append({
                "task_id": task, "teacher": "self", "turn_index": i,
                "messages": context, "prompt": prompt_txt,
                "response": m["content"],
                "token_hint": max(len(m["content"]) // 4, 1),
            })

first_self: dict[str, int] = {}
for i, r in enumerate(rows):
    if r["task_id"] not in first_self or r["turn_index"] < rows[first_self[r["task_id"]]]["turn_index"]:
        first_self[r["task_id"]] = i
n_pairs = 0
for task, idx in first_self.items():
    rej = fail_first.get(task)
    if rej and rej != rows[idx]["response"]:
        rows[idx]["_rejected"] = rej
        n_pairs += 1

out = ROOT / "data/appworld_sft/pool_v21_r1.jsonl"
with out.open("w") as fh:
    for r in rows:
        fh.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"v21 r1 pool: {len(rows)} self rows over {len(succ_seen)} tasks, "
      f"{n_pairs} success-axis pairs -> {out}")
