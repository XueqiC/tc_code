#!/usr/bin/env python3
"""Build the BFCL dDPO, PBSD, and BB-OPD pools.

- dDPO: SFT rows plus one preference row per demand task where the
  teacher ranked the student's sampled answers (best = chosen, worst =
  rejected). Ranking calls are teacher tokens.
- PBSD: SFT rows with the student's failed sample attached as
  _rejected on tasks where at least one unguided sample failed.
- BB-OPD: on a single-turn benchmark the student-visited state is the
  question itself, so on-policy relabeling coincides with the teacher
  demonstration; the pool is the SFT pool under a separate name and
  the equivalence is documented in the appendix.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BFCL = ROOT / "envs/bfcl/gorilla/berkeley-function-call-leaderboard"
sys.path.insert(0, str(ROOT / "src"))

import appworld_teacher as at  # noqa: E402

at.MAX_COMPLETION_TOKENS = 512

RANK_PROMPT = (
    "You are ranking candidate answers for a function-calling task.\n"
    "Task:\n{question}\n\nCandidates:\n{cands}\n\nReply with exactly "
    "two numbers separated by a space: the index of the BEST candidate "
    "and the index of the WORST candidate (1-based)."
)


def student_samples(repeats: int) -> dict[str, list[tuple[str, bool]]]:
    """id -> [(response, verified)] across unguided rollouts."""
    out: dict[str, list[tuple[str, bool]]] = {}
    for r in range(repeats):
        rdir = BFCL / f"result_roll_base_r{r}"
        sdir = BFCL / f"score_roll_base_r{r}"
        failed: set[str] = set()
        for f in sdir.rglob("*_score.json"):
            lines = [json.loads(l) for l in f.open() if l.strip()]
            for e in lines[1:]:
                if isinstance(e, dict) and "id" in e:
                    failed.add(e["id"])
        for f in rdir.rglob("*_result.json"):
            for line in f.open():
                if not line.strip():
                    continue
                e = json.loads(line)
                resp = e["result"]
                if not isinstance(resp, str):
                    resp = json.dumps(resp, ensure_ascii=False)
                out.setdefault(e["id"], []).append(
                    (resp, e["id"] not in failed))
    return out


def last_two_numbers(text: str) -> tuple[int, int] | None:
    nums = re.findall(r"\d+", at.strip_think(text))
    if len(nums) >= 2:
        return int(nums[-2]), int(nums[-1])
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repeats", type=int, default=4)
    ap.add_argument("--tag", default="ds")
    ap.add_argument("--skip-rank", action="store_true",
                    help="build pbsd/bbopd only (no teacher calls)")
    args = ap.parse_args()

    out_dir = ROOT / "data/bfcl_sft"
    sft_rows = [json.loads(l) for l in
                (out_dir / f"pool_bfcl_{args.tag}_sft.jsonl").open()]
    samples = student_samples(args.repeats)

    # BB-OPD: documented single-turn equivalence to the SFT pool
    with (out_dir / f"pool_bfcl_{args.tag}_bbopd.jsonl").open("w") as fh:
        for r in sft_rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    # PBSD: attach a failed student sample where one exists
    n_pairs = 0
    with (out_dir / f"pool_bfcl_{args.tag}_pbsd.jsonl").open("w") as fh:
        for r in sft_rows:
            row = dict(r)
            fails = [s for s, ok in samples.get(r["task_id"], []) if not ok]
            if fails:
                row["_rejected"] = fails[0]
                n_pairs += 1
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[bfclpairs] pbsd pairs={n_pairs}/{len(sft_rows)}")

    if args.skip_rank:
        return 0

    # dDPO: teacher ranks the student's samples per demand task
    cfg = at.load_teacher_config("gpt-5.4")
    ranked = 0
    ddpo_path = out_dir / f"pool_bfcl_{args.tag}_ddpo.jsonl"
    existing_pref: dict[str, dict] = {}
    if ddpo_path.exists():
        for l in ddpo_path.open():
            row = json.loads(l)
            if "_rejected" in row and row.get("teacher") == "rank":
                existing_pref[row["task_id"]] = row
    pref_rows = []
    for tid, cand_list in sorted(samples.items()):
        uniq: list[str] = []
        for s, _ in cand_list:
            if s not in uniq:
                uniq.append(s)
        if len(uniq) < 2:
            continue
        if tid in existing_pref:
            pref_rows.append(existing_pref[tid])
            ranked += 1
            continue
        base = next((r for r in sft_rows if r["task_id"] == tid), None)
        question = ""
        if base is not None:
            users = [m for m in base["messages"] if m["role"] == "user"]
            question = users[0]["content"][:2000] if users else ""
        cands = "\n".join(
            f"[{i+1}] {s[:800]}" for i, s in enumerate(uniq[:4]))
        try:
            reply = at.generate_reply(cfg, [{
                "role": "user",
                "content": RANK_PROMPT.format(question=question, cands=cands),
            }])
        except at.TeacherAPIError as exc:
            print(f"[bfclpairs] rank {tid} failed: {exc}")
            continue
        pair = last_two_numbers(reply)
        if pair is None:
            continue
        best, worst = pair
        if not (1 <= best <= len(uniq) and 1 <= worst <= len(uniq)) or best == worst:
            continue
        if base is None:
            continue
        pref_rows.append({
            "task_id": tid, "teacher": "rank", "turn_index": 0,
            "messages": base["messages"], "prompt": base["prompt"],
            "response": uniq[best - 1], "_rejected": uniq[worst - 1],
            "token_hint": max(len(uniq[best - 1]) // 4, 1),
        })
        ranked += 1
        print(f"[bfclpairs] ranked {tid} ({ranked})", flush=True)
    with ddpo_path.open("w") as fh:
        for r in sft_rows + pref_rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[bfclpairs] ddpo pref rows={len(pref_rows)} -> {ddpo_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
