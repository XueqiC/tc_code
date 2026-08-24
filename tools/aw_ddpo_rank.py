#!/usr/bin/env python3
"""dDPO pool (Zephyr recipe): teacher ranks student samples per task.

For every train task with at least two sampled first turns, the teacher
picks the best and worst sample. Output pool = teacher demo pool plus
one preference row per ranked task whose chosen/rejected are the
student's own samples. Ranking output tokens are teacher tokens.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import appworld_teacher as at  # noqa: E402

at.MAX_COMPLETION_TOKENS = 512

RANK_PROMPT = (
    "You are ranking candidate assistant turns for an agent task. Task "
    "context:\n{context}\n\nCandidates:\n{cands}\n\nReply with exactly "
    "two numbers separated by a space: the index of the BEST candidate "
    "and the index of the WORST candidate (1-based)."
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--teacher", default="deepseek-v4-pro")
    ap.add_argument("--out", default="data/appworld_sft/pool_ddpo.jsonl")
    args = ap.parse_args()
    config = at.load_teacher_config(args.teacher)

    samples: dict[str, list[tuple[str, list]]] = {}
    for tag in [f"awb3d_samp_t07_s{s}" for s in (10, 11, 12, 13)]:
        rdir = ROOT / "results/appworld" / tag
        for line in (rdir / "transcripts.jsonl").open():
            t = json.loads(line)
            ft, ctx = None, []
            for m in t["messages"]:
                if m.get("role") == "assistant" and m["content"].strip():
                    ft = m["content"]
                    break
                ctx.append(m)
            if ft:
                samples.setdefault(t["task_id"], []).append((ft, ctx))

    out_p = ROOT / args.out
    if out_p.exists():
        rows = [json.loads(l) for l in out_p.open()]
        done_tasks = {r["task_id"] for r in rows if r.get("teacher") == "student_sample"}
    else:
        rows = [json.loads(l) for l in (ROOT / "data/appworld_sft/pool.jsonl").open()]
        done_tasks = set()
    spent = 0
    n_pairs = 0
    for task, cands in sorted(samples.items()):
        if task in done_tasks:
            continue
        uniq = []
        for ft, ctx in cands:
            if all(ft != u[0] for u in uniq):
                uniq.append((ft, ctx))
        if len(uniq) < 2:
            continue
        context = "\n".join(
            f"[{m['role']}] {m['content'][:500]}" for m in uniq[0][1][-3:]
        )
        cand_txt = "\n\n".join(
            f"[{i+1}]\n{ft[:800]}" for i, (ft, _) in enumerate(uniq)
        )
        try:
            reply = at.generate_reply(config, [{
                "role": "user",
                "content": RANK_PROMPT.format(context=context, cands=cand_txt),
            }])
        except Exception as exc:
            print(f"[ddpo] {task}: {exc}", flush=True)
            continue
        spent += max(len(reply) // 4, 1)
        tail = re.sub(r"<think>.*?</think>", "", reply, flags=re.S)
        nums = re.findall(r"\d+", tail)
        if len(nums) < 2:
            print(f"[ddpo] unparsed {task}: {reply[-120:]!r}", flush=True)
            continue
        best, worst = int(nums[-2]) - 1, int(nums[-1]) - 1
        if not (0 <= best < len(uniq) and 0 <= worst < len(uniq)) or best == worst:
            continue
        ctx = uniq[best][1]
        prompt_txt = "\n".join(
            f"<|{x['role']}|>\n{x['content']}" for x in ctx
        ) + "\n<|assistant|>\n"
        rows.append({
            "task_id": task, "teacher": "student_sample", "turn_index": 0,
            "messages": ctx, "prompt": prompt_txt,
            "response": uniq[best][0], "_rejected": uniq[worst][0],
            "token_hint": max(len(uniq[best][0]) // 4, 1),
        })
        n_pairs += 1
        if n_pairs % 10 == 0:
            print(f"[ddpo] pairs={n_pairs} spent={spent}", flush=True)
    out = ROOT / args.out
    with out.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[ddpo] complete pairs={n_pairs} rank_tokens={spent} rows={len(rows)} -> {out}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
