#!/usr/bin/env python3
"""Build the policy-state-stratified AppWorld corpus (aw7-A).

Reads a graded base-model rollout (records + transcripts from
appworld_eval with APPWORLD_DEBUG=1) and the teacher demo pool, and
writes a mixed corpus: tasks the base model already passes contribute
the model's OWN turns (behavior preservation, zero teacher cost); tasks
it fails contribute the teacher's demo turns (correction). Mirrors the
v1.2 Stage III stratification on the agent benchmark.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rollout-tag", default="aw7_base_rollout")
    ap.add_argument("--pool", default="data/appworld_sft/pool.jsonl")
    ap.add_argument("--out", default="data/appworld_sft/pool_selfmix.jsonl")
    ap.add_argument(
        "--pass-metric", choices=("success", "majority"), default="majority",
        help="majority = passed_tests > failed_tests counts as self-solved",
    )
    args = ap.parse_args()

    rdir = ROOT / "results" / "appworld" / args.rollout_tag
    passed, failed = set(), set()
    for line in (rdir / "records.jsonl").open():
        r = json.loads(line)
        ok = (
            r.get("success")
            if args.pass_metric == "success"
            else r["passed_tests"] > r["failed_tests"]
        )
        (passed if ok else failed).add(r["task_id"])
    print(f"[selfmix] rollout tasks: pass={len(passed)} fail={len(failed)}")

    self_rows = []
    for line in (rdir / "transcripts.jsonl").open():
        t = json.loads(line)
        if t["task_id"] not in passed:
            continue
        msgs = t["messages"]
        for i, m in enumerate(msgs):
            if m.get("role") != "assistant":
                continue
            prompt_txt = "\n".join(
                f"<|{x['role']}|>\n{x['content']}" for x in msgs[:i]
            ) + "\n<|assistant|>\n"
            self_rows.append({
                "task_id": t["task_id"],
                "teacher": "self",
                "turn_index": i,
                "messages": msgs[:i],
                "prompt": prompt_txt,
                "response": m["content"],
                "token_hint": max(len(m["content"]) // 4, 1),
            })
    print(f"[selfmix] self rows from passed tasks: {len(self_rows)}")

    teacher_rows = []
    rollout_tasks = passed | failed
    for line in (ROOT / args.pool).open():
        row = json.loads(line)
        tid = row["task_id"]
        if tid in passed:
            continue  # base already solves it: keep its own behavior
        if tid not in rollout_tasks:
            continue  # outside the rollout scope: excluded for parity
        teacher_rows.append(row)
    print(f"[selfmix] teacher rows from failed tasks: {len(teacher_rows)}")

    out = ROOT / args.out
    with out.open("w", encoding="utf-8") as fh:
        for row in teacher_rows + self_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[selfmix] wrote {len(teacher_rows) + len(self_rows)} rows -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
