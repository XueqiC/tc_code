#!/usr/bin/env python3
"""Agent-distillation pool (Kang 2025 style first-thought prefix).

For the first turn of every episode in the demo pool, ask the teacher
for one short retrospective thought that motivates the demonstrated
action. The thought is stored in _thought and consumed by
AW_DISTILL=agentkd; later turns train unchanged. Thought tokens are
teacher tokens and are recorded in token_hint_thought.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import appworld_teacher as at  # noqa: E402

at.MAX_COMPLETION_TOKENS = 160

PROMPT = (
    "You are annotating an agent demonstration. Read the task context and "
    "the assistant's action below, then write one short first-person "
    "thought (2-3 sentences, no code) that would naturally precede this "
    "action, stating what the task needs and why this action is the right "
    "first step. Output only the thought.\n\nCONTEXT:\n{context}\n\n"
    "ACTION:\n{action}"
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--teacher", default="deepseek-v4-pro")
    ap.add_argument("--src", default="data/appworld_sft/pool.jsonl")
    ap.add_argument("--out", default="data/appworld_sft/pool_agentkd.jsonl")
    args = ap.parse_args()

    config = at.load_teacher_config(args.teacher)
    rows = [json.loads(l) for l in (ROOT / args.src).open()]

    first_turn: dict[str, int] = {}
    for i, r in enumerate(rows):
        ti = r.get("turn_index", 999)
        key = r["task_id"]
        if key not in first_turn or ti < rows[first_turn[key]].get("turn_index", 999):
            first_turn[key] = i

    out_path = ROOT / args.out
    done = set()
    if out_path.exists():
        for line in out_path.open():
            r = json.loads(line)
            if r.get("_thought"):
                done.add((r["task_id"], r.get("turn_index")))
        rows = [json.loads(l) for l in out_path.open()]

    spent = 0
    n = 0
    for key, idx in sorted(first_turn.items()):
        row = rows[idx]
        if (row["task_id"], row.get("turn_index")) in done:
            continue
        context = "\n".join(
            f"[{m['role']}] {m['content'][:600]}" for m in (row.get("messages") or [])[-3:]
        ) or row.get("prompt", "")[-1800:]
        try:
            reply = at.generate_reply(config, [{
                "role": "user",
                "content": PROMPT.format(context=context, action=row["response"][:1200]),
            }])
        except Exception as exc:
            print(f"[agentkd] {key}: {exc}", flush=True)
            continue
        thought = reply.strip()
        if not thought:
            continue
        if not thought.endswith("\n"):
            thought += "\n"
        row["_thought"] = thought
        spent += max(len(thought) // 4, 1)
        n += 1
        if n % 20 == 0:
            print(f"[agentkd] thoughts={n} spent={spent}", flush=True)
            with out_path.open("w") as fh:
                for r in rows:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    for row in rows:
        row.setdefault("token_hint_thought", 0)
    with out_path.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[agentkd] complete thoughts={n} spent={spent} -> {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
