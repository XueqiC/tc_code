#!/usr/bin/env python3
"""CRCD P0-1: mine intervention events from student rollouts on BFCL stateful episodes.

An event is e = (h_t, a_t^-, a_t^+, dU_t): the student's own context at turn t, the
action it actually took (a^-), the minimal correction (a^+; here the verified
ground-truth call block of that turn -- the "oracle teacher", a real teacher
correction plugs into the same slot), and the causal utility of the correction

    dU_t = P(success | do(a^+), h_t) - P(success | do(a^-), h_t)

estimated with K student continuations from each branch. "Success" for a
stateful episode = final backend state equals the reference state after all
turns (the same state comparison the on-policy repair collector uses).

Why: CRCD distills the fail->correct behavioural difference at the earliest
*consequential* divergence, not whole teacher trajectories. The repair collector
(tools/onpolicy_repair_collect.py) already finds the first state divergence;
this adds the causal continuation test and records both branches so the same
event can feed corrected-span CE (B2), local preference (B3), both (B4) and,
with fingerprints, residual gating (B5).

Output rows are training-ready: prompt (h_t rendered), response (a^+),
_rejected (a^-), turn_index, plus _event_* fields.

Usage (student served on PORT as Qwen/Qwen3.5-4B, e.g. by vllm):
    PYTHONPATH=src:tools python tools/bfcl_event_mine.py --port 8975 \
        --episodes data/bfcl_sft/gen_stateful_v3.jsonl --rollouts 2 --k 4 \
        --out data/bfcl_sft/events_v1.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from onpolicy_repair_collect import (  # noqa: E402
    as_call_strings,
    backend_config,
    state_of,
    student_reply,
    target_block,
)


def run_turns(
    *,
    adapter,
    functions,
    item,
    execute,
    messages,
    done_calls,
    start_turn: int,
    lane: str,
    index: int,
    workdir: Path,
    port: int,
    temperature: float,
) -> tuple[bool, str]:
    """Let the student play turns start_turn..end on its own from (messages, done_calls).

    Returns (success, reason). Success = final state equals the reference
    state after the full ground-truth sequence.
    """
    stu_config, stu_id = backend_config(item, lane, index, workdir)
    ref_config, ref_id = backend_config(item, lane + "ref", index, workdir)
    messages = [dict(m) for m in messages]
    done = list(done_calls)
    turns = list(zip(item["question"], item["ground_truth"]))
    for turn_index in range(start_turn, len(turns)):
        turn, _ = turns[turn_index]
        user = turn[0]["content"] if turn else ""
        if user:
            messages.append({"role": "user", "content": user})
        prompt = adapter._render(messages, functions)
        try:
            reply = student_reply(prompt, port, temperature)
        except Exception as exc:  # serving hiccup: count as failure, say why
            return False, f"serve:{type(exc).__name__}"
        calls = as_call_strings(reply)
        done.extend(calls)
        try:
            results, _ = execute(list(done), stu_config, item["involved_classes"],
                                 "Qwen/Qwen3.5-4B-FC", stu_id, long_context=False)
        except Exception as exc:
            return False, f"exec:{type(exc).__name__}"
        messages.append({"role": "assistant", "content": reply.strip()})
        if calls:
            for result in results[-len(calls):]:
                messages.append({"role": "tool", "content": str(result)})
    all_gt = [call for _, gt in turns for call in gt]
    try:
        _, ref_instances = execute(list(all_gt), ref_config, item["involved_classes"],
                                   "Qwen/Qwen3.5-4B-FC", ref_id, long_context=False)
        _, stu_instances = execute(list(done), stu_config, item["involved_classes"],
                                   "Qwen/Qwen3.5-4B-FC", stu_id, long_context=False)
    except Exception as exc:
        return False, f"final:{type(exc).__name__}"
    return state_of(ref_instances) == state_of(stu_instances), "ok"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", default="data/bfcl_sft/gen_stateful_v3.jsonl")
    parser.add_argument("--port", type=int, default=8975)
    parser.add_argument("--rollouts", type=int, default=2, help="student rollouts per episode")
    parser.add_argument("--k", type=int, default=4, help="continuations per branch")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0,
                        help="skip the first N episode lines (resume; round 1 stopped at 30/63)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="data/bfcl_sft/events_v1.jsonl")
    args = parser.parse_args()

    from bfas.adapters.bfcl import BFCLAdapter
    from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import (
        execute_multi_turn_func_call,
    )
    from bfcl_eval.utils import populate_test_cases_with_predefined_functions

    random.seed(args.seed)
    adapter = BFCLAdapter()
    entries, _ = adapter._load_entries()
    workdir = ROOT / "results/analysis/_event_memory"
    workdir.mkdir(parents=True, exist_ok=True)
    schema_cache: dict[str, list] = {}
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    handle = out_path.open("a")
    stats = {"episodes": 0, "rollouts": 0, "clean": 0, "diverged": 0, "errored": 0,
             "events": 0, "consequential": 0}
    t0 = time.time()
    lines = [l for l in (ROOT / args.episodes).read_text().splitlines() if l.strip()]
    for index, line in enumerate(lines):
        if index < args.start_index:
            continue
        item = json.loads(line)
        if not item.get("executes"):
            continue
        if args.limit and stats["episodes"] >= args.limit:
            break
        stats["episodes"] += 1
        seed = item["seed_task"]
        if seed not in schema_cache:
            populated = populate_test_cases_with_predefined_functions(
                [json.loads(json.dumps(entries[seed]))])
            schema_cache[seed] = populated[0].get("function") or []
        functions = schema_cache[seed]
        turns = list(zip(item["question"], item["ground_truth"]))

        for r in range(args.rollouts):
            stats["rollouts"] += 1
            lane = f"r{r}"
            ref_config, ref_id = backend_config(item, f"{lane}ref0", index, workdir)
            stu_config, stu_id = backend_config(item, f"{lane}stu0", index, workdir)
            messages: list[dict] = []
            ref_done: list[str] = []
            stu_done: list[str] = []
            outcome = "clean"
            for turn_index, (turn, gt_calls) in enumerate(turns):
                user = turn[0]["content"] if turn else ""
                if user:
                    messages.append({"role": "user", "content": user})
                prompt = adapter._render(messages, functions)
                try:
                    reply = student_reply(prompt, args.port, args.temperature)
                except Exception as exc:
                    print(f"    serve error {item['id']}: {exc}", flush=True)
                    outcome = "error"
                    break
                student_calls = as_call_strings(reply)
                ref_done.extend(gt_calls)
                stu_done_next = stu_done + student_calls
                try:
                    _, ref_instances = execute_multi_turn_func_call(
                        list(ref_done), ref_config, item["involved_classes"],
                        "Qwen/Qwen3.5-4B-FC", ref_id, long_context=False)
                    stu_results, stu_instances = execute_multi_turn_func_call(
                        list(stu_done_next), stu_config, item["involved_classes"],
                        "Qwen/Qwen3.5-4B-FC", stu_id, long_context=False)
                except Exception as exc:
                    print(f"    exec error {item['id']} t{turn_index}: {type(exc).__name__}", flush=True)
                    outcome = "error"
                    break

                if state_of(ref_instances) != state_of(stu_instances):
                    outcome = "diverged"
                    target = target_block(gt_calls)
                    if not target:
                        break
                    # --- causal continuation test ----------------------------
                    # branch +: history with the correction applied at turn t
                    plus_messages = messages + [{"role": "assistant", "content": target}]
                    plus_done = stu_done + list(gt_calls)
                    try:
                        plus_results, _ = execute_multi_turn_func_call(
                            list(plus_done), stu_config, item["involved_classes"],
                            "Qwen/Qwen3.5-4B-FC", stu_id + "p", long_context=False)
                    except Exception:
                        plus_results = []
                    if gt_calls:
                        for result in plus_results[-len(gt_calls):]:
                            plus_messages.append({"role": "tool", "content": str(result)})
                    # branch -: history with the student's own action
                    minus_messages = messages + [{"role": "assistant", "content": reply.strip()}]
                    minus_done = list(stu_done_next)
                    if student_calls:
                        for result in stu_results[-len(student_calls):]:
                            minus_messages.append({"role": "tool", "content": str(result)})
                    plus_ok = minus_ok = 0
                    reasons = []
                    for k in range(args.k):
                        ok, why = run_turns(
                            adapter=adapter, functions=functions, item=item,
                            execute=execute_multi_turn_func_call,
                            messages=plus_messages, done_calls=plus_done,
                            start_turn=turn_index + 1, lane=f"{lane}p{k}",
                            index=index, workdir=workdir, port=args.port,
                            temperature=args.temperature)
                        plus_ok += int(ok); reasons.append(why)
                        ok, why = run_turns(
                            adapter=adapter, functions=functions, item=item,
                            execute=execute_multi_turn_func_call,
                            messages=minus_messages, done_calls=minus_done,
                            start_turn=turn_index + 1, lane=f"{lane}m{k}",
                            index=index, workdir=workdir, port=args.port,
                            temperature=args.temperature)
                        minus_ok += int(ok); reasons.append(why)
                    # Laplace-smoothed branch utilities
                    u_plus = (plus_ok + 1) / (args.k + 2)
                    u_minus = (minus_ok + 1) / (args.k + 2)
                    d_u = u_plus - u_minus
                    stats["events"] += 1
                    if d_u > 0:
                        stats["consequential"] += 1
                    handle.write(json.dumps({
                        "task_id": item["id"],
                        "teacher": "teacher_authored_gt",  # generated task: the teacher wrote the answer (relabelled 2026-09-04)
                        "turn_index": turn_index,
                        "prompt": prompt,
                        "response": target,
                        "_rejected": reply.strip(),
                        "token_hint": max(len(target) // 4, 1),
                        "_task_phat": 0.0,
                        "_traj": f"{item['id']}#ev{r}t{turn_index}",
                        "_seed_task": seed,
                        "_seed_category": item.get("seed_category", ""),
                        "_event_turn": turn_index,
                        "_event_rollout": r,
                        "_event_u_plus": u_plus,
                        "_event_u_minus": u_minus,
                        "_event_dU": d_u,
                        "_event_k": args.k,
                        "_event_plus_ok": plus_ok,
                        "_event_minus_ok": minus_ok,
                        "_event_reasons": reasons,
                        "_student_calls": student_calls,
                        "_gt_calls": list(gt_calls),
                    }, ensure_ascii=False) + "\n")
                    handle.flush()
                    print(f"    {item['id']} r{r}: diverged t{turn_index} "
                          f"u+={u_plus:.2f} u-={u_minus:.2f} dU={d_u:+.2f}", flush=True)
                    break

                # states match: walk on
                stu_done = stu_done_next
                messages.append({"role": "assistant", "content": reply.strip()})
                if student_calls:
                    for result in stu_results[-len(student_calls):]:
                        messages.append({"role": "tool", "content": str(result)})
            stats[outcome if outcome in stats else "errored"] += 1
        if stats["episodes"] % 10 == 0:
            print(f"[events] {stats} elapsed={time.time() - t0:.0f}s", flush=True)
    handle.close()
    print(f"[events] FINAL {stats} -> {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
