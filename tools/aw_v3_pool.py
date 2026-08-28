#!/usr/bin/env python3
"""v3.0 round-1 pool with behavior-policy likelihoods (method v3.0 §7-8).

From the guided debug rollouts:
- keep, per demand-split task, ALL verified trajectories (the positive
  objective uses the actual sampled rollout distribution, A.9);
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
import sys as _sys
_sys.path.insert(0, str(ROOT / "src"))
DEMO_RE = re.compile(r"\n\nWorked example from an expert.*$", re.S)
TAGS = [f"awb4_hint_t07_s{s}" for s in (20, 21, 22, 23)]


def load_trajectories(best_only: bool = False):
    succ: dict[str, list[list[dict]]] = {}
    fail: dict[str, list[list[dict]]] = {}
    quality: dict[str, list[int]] = {}
    for tag in TAGS:
        rdir = ROOT / "results/appworld" / tag
        verdict = {}
        passed = {}
        for line in (rdir / "records.jsonl").open():
            r = json.loads(line)
            verdict[r["task_id"]] = r["passed_tests"] > r["failed_tests"]
            passed[r["task_id"]] = r["passed_tests"]
        for line in (rdir / "transcripts.jsonl").open():
            t = json.loads(line)
            tid = t["task_id"]
            if verdict.get(tid):
                succ.setdefault(tid, []).append(t["messages"])
                quality.setdefault(tid, []).append(passed.get(tid, 0))
            else:
                fail.setdefault(tid, []).append(t["messages"])
    if best_only:
        for tid in list(succ):
            best = max(range(len(succ[tid])), key=lambda i: quality[tid][i])
            succ[tid] = [succ[tid][best]]
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
    """Behavior-policy NLL under the SAME rendering the trainer uses.

    The trainer serializes context with appworld_train.encode, so mu
    must be computed under that serialization too; otherwise the
    importance ratio absorbs a rendering mismatch instead of the
    guidance effect.
    """
    import appworld_train as tr
    row = {
        "messages": context_msgs,
        "prompt": "\n".join(f"<|{m['role']}|>\n{m['content']}"
                            for m in context_msgs) + "\n<|assistant|>\n",
        "response": response,
    }
    ids, labels = tr.encode(tok, row)
    ids = ids.to(model.device)
    labels = labels.to(model.device)
    with torch.inference_mode():
        out = model(input_ids=ids, labels=labels)
    ntok = int((labels != -100).sum())
    return float(out.loss.item()) * ntok, ntok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--student", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--out", default="data/appworld_sft/pool_v3_r1.jsonl")
    ap.add_argument("--best-only", action="store_true",
                    help="keep only the highest-quality verified "
                         "trajectory per task (max passed_tests)")
    ap.add_argument("--unguided-tags", default="",
                    help="comma-separated eval tags of unguided K-sample "
                         "rounds; enables advantage rows: per-task p-hat "
                         "from these rounds, plus their verified "
                         "trajectories as extra w=1 rows")
    args = ap.parse_args()

    split = json.load((ROOT / "configs/support_split.json").open())
    demand = set(split["demand"])

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.student)
    model = AutoModelForCausalLM.from_pretrained(
        args.student, dtype=torch.bfloat16, device_map="cuda")
    model.eval()

    succ, fail = load_trajectories(best_only=args.best_only)
    rows, n_pairs, skipped_cal = [], 0, 0
    for task, trajs in sorted(succ.items()):
        if task not in demand:
            skipped_cal += 1
            continue
        # all verified rollouts enter the positive objective (A.9)
        for t_idx, traj in enumerate(trajs):
            clean = strip_demo(traj)
            turn_ids = assistant_turns(traj)
            for i in turn_ids:
                mu, ntok = sum_nll(model, tok, traj[:i], traj[i]["content"])
                ctx = clean[:i]
                rows.append({
                    "task_id": task, "teacher": "self", "turn_index": i,
                    "messages": ctx,
                    "prompt": "\n".join(f"<|{m['role']}|>\n{m['content']}"
                                        for m in ctx) + "\n<|assistant|>\n",
                    "response": traj[i]["content"],
                    "token_hint": max(len(traj[i]["content"]) // 4, 1),
                    "_mu_nll_sum": round(mu, 4), "_mu_ntok": ntok,
                    "_traj": f"{task}#{t_idx}",
                })
        # first-divergence preference pair against a failed trajectory,
        # taken on the first verified rollout of the task
        if task in fail and fail[task]:
            base_traj = trajs[0]
            b_turns = assistant_turns(base_traj)
            ft = fail[task][0]
            f_turns = assistant_turns(ft)
            for k in range(min(len(b_turns), len(f_turns))):
                yb = base_traj[b_turns[k]]["content"]
                yf = ft[f_turns[k]]["content"]
                if yb != yf:
                    for r in rows:
                        if (r["_traj"] == f"{task}#0"
                                and r["turn_index"] == b_turns[k]):
                            r["_rejected"] = yf
                            n_pairs += 1
                    break
    if args.unguided_tags:
        # advantage extension: per-task unguided success rate stamps
        # every row, and verified unguided trajectories join the pool
        # as plain rows (behavior policy equals training policy there)
        utags = [t.strip() for t in args.unguided_tags.split(",") if t.strip()]
        succ_rounds: dict[str, int] = {}
        total_rounds: dict[str, int] = {}
        u_succ: dict[str, list[list[dict]]] = {}
        for tag in utags:
            rdir = ROOT / "results/appworld" / tag
            verdict = {}
            for line in (rdir / "records.jsonl").open():
                rec = json.loads(line)
                tid = rec["task_id"]
                verdict[tid] = rec["passed_tests"] > rec["failed_tests"]
                total_rounds[tid] = total_rounds.get(tid, 0) + 1
                if verdict[tid]:
                    succ_rounds[tid] = succ_rounds.get(tid, 0) + 1
            tpath = rdir / "transcripts.jsonl"
            if tpath.exists():
                for line in tpath.open():
                    t = json.loads(line)
                    if verdict.get(t["task_id"]):
                        u_succ.setdefault(t["task_id"], []).append(t["messages"])
        phat = {t: succ_rounds.get(t, 0) / max(total_rounds.get(t, 1), 1)
                for t in total_rounds}
        for r in rows:
            r["_task_phat"] = round(phat.get(r["task_id"], 0.0), 4)
        n_u = 0
        for tid, trajs in sorted(u_succ.items()):
            if tid not in demand:
                continue
            for t_idx, traj in enumerate(trajs):
                for i in assistant_turns(traj):
                    ctx = traj[:i]
                    rows.append({
                        "task_id": tid, "teacher": "self",
                        "turn_index": i,
                        "prompt": "\n".join(
                            f"<|{m['role']}|>\n{m['content']}" for m in ctx
                        ) + "\n<|assistant|>\n",
                        "response": traj[i]["content"],
                        "token_hint": max(len(traj[i]["content"]) // 4, 1),
                        "_task_phat": round(phat.get(tid, 0.0), 4),
                        "_traj": f"{tid}#ung{t_idx}",
                    })
                    n_u += 1
        print(f"[v3pool] advantage: unguided rows={n_u} "
              f"tasks_with_phat={len(phat)}")

    out = ROOT / args.out
    with out.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[v3pool] rows={len(rows)} tasks={len({r['task_id'] for r in rows})} "
          f"pairs={n_pairs} calib_excluded={skipped_cal} -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
