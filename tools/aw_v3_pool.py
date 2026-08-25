#!/usr/bin/env python3
"""v3.0 round-1 pool with behavior-policy likelihoods (method v3.0 §7-8).

From the guided debug rollouts:
- keep, per demand-split task, the shortest verified trajectory;
- strip the guidance block from every training context;
- for each assistant turn, compute and store the behavior-policy
  sum-NLL `_mu_nll_sum` and target token count `_mu_ntok`, evaluated
  under the collection-time policy (base student) WITH the guidance
  block present, so the trainer can form the clipped importance ratio
  pi_theta(y|ctx) / mu(y|ctx,h) per turn;
- form preference pairs at the FIRST assistant turn where the success
  and failure trajectories differ (shared context by environment
  determinism);
- exclude calibration-split tasks entirely.

Needs one GPU pass; run with the project venv.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
DEMO_RE = re.compile(r"\n\nWorked example from an expert.*$", re.S)
TAGS = [f"awb4_hint_t07_s{s}" for s in (20, 21, 22, 23)]


def load_trajectories():
    succ: dict[str, list[list[dict]]] = {}
    fail: dict[str, list[list[dict]]] = {}
    for tag in TAGS:
        rdir = ROOT / "results/appworld" / tag
        verdict = {}
        for line in (rdir / "records.jsonl").open():
            r = json.loads(line)
            verdict[r["task_id"]] = r["passed_tests"] > r["failed_tests"]
        for line in (rdir / "transcripts.jsonl").open():
            t = json.loads(line)
            (succ if verdict.get(t["task_id"]) else fail).setdefault(
                t["task_id"], []).append(t["messages"])
    return succ, fail


def assistant_turns(msgs):
    return [i for i, m in enumerate(msgs)
            if m.get("role") == "assistant" and m["content"].strip()]


def strip_demo(msgs):
    out = [dict(m) for m in msgs]
    for m in out:
        if m["role"] == "system":
            m["content"] = DEMO_RE.sub("", m["content"])
    return out


def sum_nll(model, tok, context_msgs, response):
    ids = tok.apply_chat_template(
        context_msgs, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    resp_ids = tok(response, add_special_tokens=False,
                   return_tensors="pt").input_ids.to(model.device)
    full = torch.cat([ids, resp_ids], dim=1)
    labels = full.clone()
    labels[0, : ids.shape[1]] = -100
    with torch.inference_mode():
        out = model(input_ids=full, labels=labels)
    ntok = int((labels != -100).sum())
    return float(out.loss.item()) * ntok, ntok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--student", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--out", default="data/appworld_sft/pool_v3_r1.jsonl")
    args = ap.parse_args()

    split = json.load((ROOT / "configs/support_split.json").open())
    demand = set(split["demand"])

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.student)
    model = AutoModelForCausalLM.from_pretrained(
        args.student, dtype=torch.bfloat16, device_map="cuda")
    model.eval()

    succ, fail = load_trajectories()
    rows, n_pairs, skipped_cal = [], 0, 0
    for task, trajs in sorted(succ.items()):
        if task not in demand:
            skipped_cal += 1
            continue
        best = min(trajs, key=lambda m: len(assistant_turns(m)))
        clean = strip_demo(best)
        turn_ids = assistant_turns(best)
        for i in turn_ids:
            mu, ntok = sum_nll(model, tok, best[:i], best[i]["content"])
            ctx = clean[:i]
            rows.append({
                "task_id": task, "teacher": "self", "turn_index": i,
                "messages": ctx,
                "prompt": "\n".join(f"<|{m['role']}|>\n{m['content']}"
                                    for m in ctx) + "\n<|assistant|>\n",
                "response": best[i]["content"],
                "token_hint": max(len(best[i]["content"]) // 4, 1),
                "_mu_nll_sum": round(mu, 4), "_mu_ntok": ntok,
                "_traj": task,
            })
        # first-divergence preference pair against a failed trajectory
        if task in fail and fail[task]:
            ft = fail[task][0]
            f_turns = assistant_turns(ft)
            b_turns = turn_ids
            for k in range(min(len(b_turns), len(f_turns))):
                yb = best[b_turns[k]]["content"]
                yf = ft[f_turns[k]]["content"]
                if yb != yf:
                    for r in rows:
                        if r["task_id"] == task and r["turn_index"] == b_turns[k]:
                            r["_rejected"] = yf
                            n_pairs += 1
                    break
    out = ROOT / args.out
    with out.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[v3pool] rows={len(rows)} tasks={len({r['task_id'] for r in rows})} "
          f"pairs={n_pairs} calib_excluded={skipped_cal} -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
