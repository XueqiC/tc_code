#!/usr/bin/env python3
"""Success-axis preference pairs, v2.0 admissibility.

Both sides of a pair come from the student's own policy on the same
task context. Chosen is the first assistant turn of a verified-success
trajectory (greedy rollout or temperature sample); rejected is the
first assistant turn of a verified-failure trajectory of the same task.
Style is held fixed by construction, so the pair differs on the
success axis alone. Sampled successes are additionally appended as
anchor rows (teacher == "self").
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_TAGS = [f"awb3d_samp_t07_s{s}" for s in (10, 11, 12, 13)]
GREEDY = "aw7_base_rollout"


def first_turn(messages):
    for m in messages:
        if m.get("role") == "assistant" and m["content"].strip():
            return m["content"]
    return None


def load_runs(tag):
    rdir = ROOT / "results/appworld" / tag
    verdict = {}
    for line in (rdir / "records.jsonl").open():
        r = json.loads(line)
        verdict[r["task_id"]] = r["passed_tests"] > r["failed_tests"]
    out = []
    for line in (rdir / "transcripts.jsonl").open():
        t = json.loads(line)
        ft = first_turn(t["messages"])
        if ft:
            out.append((t["task_id"], verdict.get(t["task_id"], False), ft, t["messages"]))
    return out


succ, fail = {}, {}
for tag in [GREEDY] + SAMPLE_TAGS:
    for task, ok, ft, msgs in load_runs(tag):
        (succ if ok else fail).setdefault(task, []).append((ft, msgs, tag))

rows = [json.loads(l) for l in (ROOT / "data/appworld_sft/pool_selfmix.jsonl").open()]
self_tasks = {r["task_id"] for r in rows if r.get("teacher") == "self"}

# attach rejected to the earliest self row per task
first_self = {}
for i, r in enumerate(rows):
    if r.get("teacher") != "self":
        continue
    ti = r.get("turn_index", 999)
    if r["task_id"] not in first_self or ti < rows[first_self[r["task_id"]]].get("turn_index", 999):
        first_self[r["task_id"]] = i
n_pairs = 0
for task, idx in first_self.items():
    if task in fail and fail[task]:
        rej = fail[task][0][0]
        if rej != rows[idx]["response"]:
            rows[idx]["_rejected"] = rej
            n_pairs += 1

# append sampled successes as new anchor rows, with pairs where possible
n_new = 0
seen = set(self_tasks)
for task, entries in succ.items():
    if task in seen:
        continue
    ft, msgs, tag = entries[0]
    context = []
    for m in msgs:
        if m.get("role") == "assistant" and m["content"].strip():
            break
        context.append(m)
    prompt_txt = "\n".join(f"<|{x['role']}|>\n{x['content']}" for x in context) + "\n<|assistant|>\n"
    row = {"task_id": task, "teacher": "self", "turn_index": 0,
           "messages": context, "prompt": prompt_txt, "response": ft,
           "token_hint": max(len(ft) // 4, 1), "source": tag}
    if task in fail and fail[task] and fail[task][0][0] != ft:
        row["_rejected"] = fail[task][0][0]
        n_pairs += 1
    rows.append(row)
    seen.add(task)
    n_new += 1

out = ROOT / "data/appworld_sft/pool_selfmix_v2pref.jsonl"
with out.open("w") as fh:
    for r in rows:
        fh.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"rows={len(rows)} new_self={n_new} pairs={n_pairs} -> {out}")
