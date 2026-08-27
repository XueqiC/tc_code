#!/usr/bin/env python3
"""Build the BFCL v3 (ours) training pool and the STaR pool from
verified student rollouts.

- Unguided verified rollouts: behavior policy equals the training
  policy and context, so the importance ratio is exactly one; rows
  carry no _mu fields.
- Guided verified rollouts: collected with the worked example in
  context but trained without it; rows carry _mu_nll_sum/_mu_ntok
  computed under the guided context with the trainer serialization
  (appworld_train.encode), so the trainer forms the clipped ratio.
- STaR pool: unguided verified rollouts only, plain rows.

Needs one GPU pass for guided mu. Run with the project venv.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BFCL = ROOT / "envs/bfcl/gorilla/berkeley-function-call-leaderboard"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(BFCL))

GUIDE_HEADER = (
    "Worked example from an expert on this exact task. Study the "
    "approach, then solve the task yourself:\n"
)


def serialize(messages: list[dict]) -> str:
    return "\n".join(
        f"<|{m['role']}|>\n{m['content']}" for m in messages
    ) + "\n<|assistant|>\n"


def load_verified(kind: str, repeats: int) -> dict[str, list[str]]:
    """id -> list of verified response strings across rollouts."""
    out: dict[str, list[str]] = {}
    for r in range(repeats):
        rdir = BFCL / f"result_roll_{kind}_r{r}"
        sdir = BFCL / f"score_roll_{kind}_r{r}"
        if not rdir.exists():
            continue
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
                if e["id"] in failed:
                    continue
                resp = e["result"]
                if not isinstance(resp, str):
                    # multi-turn results are call structures; keep the
                    # compact JSON form as the training target
                    resp = json.dumps(resp, ensure_ascii=False)
                out.setdefault(e["id"], []).append(resp)
    return out


def entry_messages(entry: dict, guided_demo: dict | None) -> list[dict]:
    from bfcl_eval.model_handler.utils import (
        system_prompt_pre_processing_chat_model,
    )
    e2 = copy.deepcopy(entry)
    if guided_demo is not None:
        calls = json.dumps(guided_demo["teacher_result"], ensure_ascii=False)
        if len(calls) > 4000:
            calls = calls[:4000] + " ..."
        for m in e2["question"][0]:
            if m.get("role") == "user":
                m["content"] = (
                    f"{GUIDE_HEADER}Expert solution: {calls}\n\n" + m["content"]
                )
                break
    e2["question"][0] = system_prompt_pre_processing_chat_model(
        e2["question"][0], e2["function"], e2["id"]
    )
    return list(e2["question"][0])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--student", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--repeats", type=int, default=4)
    ap.add_argument("--tag", default="ds")
    args = ap.parse_args()

    split = json.load((ROOT / "configs/bfcl_support_split.json").open())
    demand = set(split["demand"])
    demos = json.load((ROOT / f"data/bfcl_sft/demos_{args.tag}.json").open())

    entries: dict[str, dict] = {}
    for f in sorted((BFCL / "bfcl_eval/data").glob("BFCL_v4_*.json")):
        for line in f.open():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                break
            if isinstance(row, dict) and row.get("id") in demand:
                entries[row["id"]] = row

    unguided = load_verified("base", args.repeats)
    guided = load_verified("guided", args.repeats)
    # dedicated multi-turn rollout pass (message dumps enabled)
    # overrides the earlier base statistics for its tasks
    mt = load_verified("mt", args.repeats)
    for tid, resps in mt.items():
        unguided[tid] = resps

    # empirical per-task success rate over the K unguided rollouts;
    # rows carry it as _task_phat and the trainer's advantage weight
    # (1 - p-hat) makes converged tasks contribute nothing
    n_attempts: dict[str, int] = {}
    mt_tasks: set[str] = set()
    for r in range(args.repeats):
        for kind in ("base", "mt"):
            rdir = BFCL / f"result_roll_{kind}_r{r}"
            if not rdir.exists():
                continue
            for f in rdir.rglob("*_result.json"):
                for line in f.open():
                    if line.strip():
                        tid = json.loads(line)["id"]
                        if kind == "mt":
                            if tid not in mt_tasks:
                                mt_tasks.add(tid)
                                n_attempts[tid] = 0
                            n_attempts[tid] += 1
                        elif tid not in mt_tasks:
                            n_attempts[tid] = n_attempts.get(tid, 0) + 1

    rows: list[dict] = []
    star_rows: list[dict] = []
    for tid, resps in sorted(unguided.items()):
        # multi-turn and stateful entries carry no top-level function
        # docs; they stay out of the training rows for every method,
        # matching the baseline pools
        if "function" not in entries[tid]:
            continue
        msgs = entry_messages(entries[tid], None)
        phat = len(resps) / max(n_attempts.get(tid, args.repeats), 1)
        for k, resp in enumerate(resps):
            row = {
                "task_id": tid, "teacher": "self", "turn_index": 0,
                "messages": msgs, "prompt": serialize(msgs),
                "response": resp,
                "token_hint": max(len(resp) // 4, 1),
                "_task_phat": round(phat, 4),
                "_traj": f"{tid}#u{k}",
            }
            rows.append(row)
            star_rows.append(dict(row))

    # multi-turn rows come from message dumps written by the patched
    # handler (BFCL_DUMP_MESSAGES): the dumped stream is the exact
    # evaluation-time conversation, one training row per assistant turn
    dump_dir = ROOT / "data/bfcl_dumps"
    mt_rows = 0
    if dump_dir.exists():
        verified_ids = set(unguided) | set(guided)
        seen_dump: set[str] = set()
        for df in sorted(dump_dir.glob("*.jsonl")):
            for line in df.open():
                if not line.strip():
                    continue
                d = json.loads(line)
                tid = d["id"]
                if tid not in demand or tid in seen_dump:
                    continue
                if tid not in verified_ids:
                    continue
                seen_dump.add(tid)
                phat = len(unguided.get(tid, [])) / 4
                msgs = d["messages"]
                for i, m in enumerate(msgs):
                    if m.get("role") != "assistant" or not str(m.get("content", "")).strip():
                        continue
                    ctx = msgs[:i]
                    rows.append({
                        "task_id": tid, "teacher": "self",
                        "turn_index": i,
                        "messages": ctx, "prompt": serialize(ctx),
                        "response": str(m["content"]),
                        "token_hint": max(len(str(m["content"])) // 4, 1),
                        "_task_phat": round(phat, 4),
                        "_traj": f"{tid}#d",
                    })
                    mt_rows += 1
    if mt_rows:
        print(f"[bfclv3pool] dump-based multi-turn rows: {mt_rows}")

    guided_pending = [
        (tid, resps) for tid, resps in sorted(guided.items())
        if tid not in unguided and "function" in entries.get(tid, {})
    ]
    if guided_pending:
        import torch
        import appworld_train as tr
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.student)
        model = AutoModelForCausalLM.from_pretrained(
            args.student, dtype=torch.bfloat16, device_map="cuda")
        model.eval()
        for tid, resps in guided_pending:
            g_msgs = entry_messages(entries[tid], demos.get(tid))
            clean_msgs = entry_messages(entries[tid], None)
            for k, resp in enumerate(resps):
                mu_row = {
                    "messages": g_msgs,
                    "prompt": serialize(g_msgs),
                    "response": resp,
                }
                ids, labels = tr.encode(tok, mu_row)
                ids = ids.to(model.device)
                labels = labels.to(model.device)
                with torch.inference_mode():
                    out = model(input_ids=ids, labels=labels)
                ntok = int((labels != -100).sum())
                rows.append({
                    "task_id": tid, "teacher": "self", "turn_index": 0,
                    "messages": clean_msgs, "prompt": serialize(clean_msgs),
                    "response": resp,
                    "token_hint": max(len(resp) // 4, 1),
                    "_task_phat": 0.0,
                    "_mu_nll_sum": round(float(out.loss.item()) * ntok, 4),
                    "_mu_ntok": ntok,
                    "_traj": f"{tid}#g{k}",
                })

    out_dir = ROOT / "data/bfcl_sft"
    pool = out_dir / f"pool_bfcl_{args.tag}_v3.jsonl"
    star = out_dir / f"pool_bfcl_star.jsonl"
    with pool.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    with star.open("w") as fh:
        for r in star_rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_g = sum(1 for r in rows if "_mu_nll_sum" in r)
    print(f"[bfclv3pool] rows={len(rows)} (guided {n_g}) "
          f"tasks={len({r['task_id'] for r in rows})} star_rows={len(star_rows)} "
          f"-> {pool}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
