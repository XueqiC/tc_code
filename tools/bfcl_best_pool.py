#!/usr/bin/env python3
"""Best-of-K self-distillation pool for BFCL.

For every demand task with verified unguided samples, the teacher
judges which sample is highest quality; only that sample becomes a
training row. Judging calls are metered teacher tokens. Tasks with a
single unique verified sample skip the judge.
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

at.MAX_COMPLETION_TOKENS = 256

JUDGE_PROMPT = (
    "You are judging candidate answers for a function-calling task. All "
    "candidates passed the automatic checker; pick the one that is "
    "cleanest and most precise (correct calls, no extraneous text, no "
    "redundant calls).\nTask:\n{question}\n\nCandidates:\n{cands}\n\n"
    "Reply with exactly one number: the index of the BEST candidate "
    "(1-based)."
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="ds")
    args = ap.parse_args()

    sys.path.insert(0, str(ROOT / "tools"))
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "pp", ROOT / "tools/bfcl_pair_pools.py")
    pp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pp)

    out_dir = ROOT / "data/bfcl_sft"
    v3_rows = [json.loads(l) for l in
               (out_dir / f"pool_bfcl_{args.tag}_v3.jsonl").open()]
    by_task = {r["task_id"]: r for r in v3_rows}
    samples = pp.student_samples(4)

    cfg = at.load_teacher_config("deepseek-v4-pro")
    rows, judged = [], 0
    for tid, base_row in sorted(by_task.items()):
        verified = []
        for s, ok in samples.get(tid, []):
            if ok and s not in verified:
                verified.append(s)
        if len(verified) <= 1:
            rows.append(base_row)
            continue
        users = [m for m in base_row["messages"] if m["role"] == "user"]
        question = users[0]["content"][:2000] if users else ""
        cands = "\n".join(
            f"[{i+1}] {s[:800]}" for i, s in enumerate(verified[:4]))
        best = None
        try:
            reply = at.generate_reply(cfg, [{
                "role": "user",
                "content": JUDGE_PROMPT.format(question=question, cands=cands),
            }])
            nums = re.findall(r"\d+", at.strip_think(reply))
            if nums and 1 <= int(nums[-1]) <= len(verified):
                best = verified[int(nums[-1]) - 1]
        except at.TeacherAPIError as exc:
            print(f"[bestpool] judge {tid} failed: {exc}")
        if best is None:
            best = base_row["response"]
        else:
            judged += 1
        row = dict(base_row)
        row["response"] = best
        row["token_hint"] = max(len(best) // 4, 1)
        rows.append(row)
        print(f"[bestpool] {tid} judged ({judged})", flush=True)

    out = out_dir / f"pool_bfcl_{args.tag}_best.jsonl"
    with out.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[bestpool] rows={len(rows)} teacher_judged={judged} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
