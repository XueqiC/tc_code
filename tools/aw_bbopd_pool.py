#!/usr/bin/env python3
"""Build the Black-Box On-Policy Distillation pool (Table 1 baseline).

BB-OPD trains on (student-visited context, teacher response) pairs: the
student's own rollout fixes the state distribution, and the teacher is
queried once per visited context for the action it would take there.
Every returned token is charged; generation stops at the same response
token budget the other arms receive.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import appworld_teacher as at  # noqa: E402

at.MAX_COMPLETION_TOKENS = 1024


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rollout-tag", default="aw7_base_rollout")
    ap.add_argument("--teacher", default="gpt-5.4")
    ap.add_argument("--budget", type=int, default=20000)
    ap.add_argument("--out", default="data/appworld_sft/pool_bbopd.jsonl")
    args = ap.parse_args()

    config = at.load_teacher_config(args.teacher)
    rdir = ROOT / "results" / "appworld" / args.rollout_tag
    out = ROOT / args.out
    done = set()
    if out.exists():
        for line in out.open():
            r = json.loads(line)
            done.add((r["task_id"], r["turn_index"]))
    spent = sum(
        json.loads(l)["token_hint"] for l in out.open()
    ) if out.exists() else 0

    n = 0
    with out.open("a", encoding="utf-8") as fh:
        for line in (rdir / "transcripts.jsonl").open():
            t = json.loads(line)
            msgs = t["messages"]
            for i, m in enumerate(msgs):
                if m.get("role") != "assistant":
                    continue
                key = (t["task_id"], i)
                if key in done:
                    continue
                if spent >= args.budget:
                    print(f"[bbopd] budget reached: {spent}", flush=True)
                    return 0
                context = msgs[:i]
                try:
                    reply = at.generate_reply(config, context)
                except Exception as exc:
                    print(f"[bbopd] call failed {key}: {exc}", flush=True)
                    continue
                if not reply.strip():
                    continue
                tok = max(len(reply) // 4, 1)
                spent += tok
                prompt_txt = "\n".join(
                    f"<|{x['role']}|>\n{x['content']}" for x in context
                ) + "\n<|assistant|>\n"
                fh.write(json.dumps({
                    "task_id": t["task_id"],
                    "teacher": args.teacher,
                    "turn_index": i,
                    "messages": context,
                    "prompt": prompt_txt,
                    "response": reply,
                    "token_hint": tok,
                }, ensure_ascii=False) + "\n")
                fh.flush()
                n += 1
                if n % 20 == 0:
                    print(f"[bbopd] rows={n} spent={spent}", flush=True)
    print(f"[bbopd] complete rows={n} spent={spent}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
