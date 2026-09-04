#!/usr/bin/env python3
"""Turn generated stateful episodes into trainer rows.

A generated episode is a blueprint: user turns plus the ground-truth calls that
satisfy them. Training wants one row per assistant turn, each showing the model
the conversation as it will actually meet it -- earlier turns, its own earlier
calls, and the tool output those calls produced. So the episode is REPLAYED
against the real backend, and the observations it returns become the tool
messages in the context, exactly as at evaluation time.

Rows carry _task_phat = 0: these come from queries the student fails, so under
the trainer's (1 - phat) weighting they enter at full weight.

Usage: bfcl_gen_rows.py [--in data/bfcl_sft/gen_stateful.jsonl]
                        [--out data/bfcl_sft/pool_gen_stateful.jsonl]
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BFCL = ROOT / "envs/bfcl/gorilla/berkeley-function-call-leaderboard"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(BFCL))


def parse_call(call: str) -> dict | None:
    """`fn(a='b', c=1)` -> {"name": "fn", "arguments": {"a": "b", "c": 1}}."""
    try:
        tree = ast.parse(call.strip(), mode="eval")
    except SyntaxError:
        return None
    node = tree.body
    if not isinstance(node, ast.Call):
        return None
    if isinstance(node.func, ast.Name):
        name = node.func.id
    elif isinstance(node.func, ast.Attribute):
        name = node.func.attr
    else:
        return None
    arguments: dict = {}
    for keyword in node.keywords:
        if keyword.arg is None:
            return None
        try:
            arguments[keyword.arg] = ast.literal_eval(keyword.value)
        except ValueError:
            return None
    for index, positional in enumerate(node.args):
        try:
            arguments[f"_arg{index}"] = ast.literal_eval(positional)
        except ValueError:
            return None
    return {"name": name, "arguments": arguments}


def target_block(calls: list[str]) -> str | None:
    parsed = [parse_call(c) for c in calls]
    if not parsed or any(p is None for p in parsed):
        return None
    return "\n".join("<tool_call>\n" + json.dumps(p, ensure_ascii=False)
                     + "\n</tool_call>" for p in parsed)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="src",
                        default="data/bfcl_sft/gen_stateful.jsonl")
    parser.add_argument("--out",
                        default="data/bfcl_sft/pool_gen_stateful.jsonl")
    args = parser.parse_args()

    from bfas.adapters.bfcl import BFCLAdapter
    from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import (
        execute_multi_turn_func_call,
    )
    from bfcl_eval.utils import populate_test_cases_with_predefined_functions

    adapter = BFCLAdapter()
    entries, _ = adapter._load_entries()
    schema_cache: dict[str, list] = {}
    # the episode file stores initial_config: {} for memory, because a memory
    # backend takes its state from a snapshot folder rather than a config blob.
    # Replay has to rebuild that slice or every memory episode dies on KeyError
    # -- and it dies AFTER passing generation, so the loss is silent.
    workdir = ROOT / "results/analysis/_genrows_memory"
    workdir.mkdir(parents=True, exist_ok=True)

    def backend_config(item: dict, index: int) -> tuple[dict, str]:
        involved = item["involved_classes"]
        memory_classes = [c for c in involved if c.startswith("MemoryAPI_")]
        if memory_classes:
            category = memory_classes[0].replace("MemoryAPI_", "memory_")
            run_id = f"{category}_prereq_{index}-rows-0"
            slice_ = {"model_result_dir": workdir, "test_id": run_id,
                      "scenario": item.get("scenario") or "gen"}
            return {name: dict(slice_) for name in memory_classes}, run_id
        if "WebSearchAPI" in involved and not item.get("initial_config"):
            return {"WebSearchAPI": {"show_snippet": True}}, item["id"]
        return item.get("initial_config") or {}, item["id"]

    rows: list[dict] = []
    episodes = skipped = 0
    for line in (ROOT / args.src).read_text().splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if not item.get("executes"):
            continue
        episodes += 1
        seed = item["seed_task"]
        if seed not in schema_cache:
            populated = populate_test_cases_with_predefined_functions(
                [json.loads(json.dumps(entries[seed]))])
            schema_cache[seed] = populated[0].get("function") or []
        functions = schema_cache[seed]

        messages: list[dict] = []
        done: list[str] = []
        ok = True
        config, run_id = backend_config(item, episodes)
        replies = item.get("replies") or []
        for index, (turn, calls) in enumerate(
                zip(item["question"], item["ground_truth"])):
            user = turn[0]["content"] if turn else ""
            if user:
                messages.append({"role": "user", "content": user})
            if not calls:
                # a silent turn: the model must answer without touching a tool,
                # which is 12.4% of the benchmark's turns and was 0% of ours
                if pending := (replies[index] if index < len(replies) else ""):
                    rows.append({
                        "task_id": item["id"], "teacher": "gen",
                        "turn_index": index,
                        "prompt": adapter._render(messages, functions),
                        "response": pending,
                        "token_hint": max(len(pending) // 4, 1),
                        "_task_phat": 0.0,
                        "_traj": f"{item['id']}#t{index}s",
                        "_seed_task": seed,
                        "_seed_category": item.get("seed_category", ""),
                        "_silent_turn": True,
                    })
                    messages.append({"role": "assistant", "content": pending})
                continue
            target = target_block(calls)
            if target is None:
                ok = False
                break
            rows.append({
                "task_id": item["id"],
                "teacher": "gen",
                "turn_index": index,
                "prompt": adapter._render(messages, functions),
                "response": target,
                "token_hint": max(len(target) // 4, 1),
                "_task_phat": 0.0,
                "_traj": f"{item['id']}#t{index}",
                "_seed_task": seed,
                "_seed_category": item.get("seed_category", ""),
            })
            messages.append({"role": "assistant", "content": target})
            # replay everything so far so this turn's tool output is the output
            # the calls really produce, not a guess about it
            done.extend(calls)
            pending_reply = replies[index] if index < len(replies) else ""
            try:
                results, _ = execute_multi_turn_func_call(
                    list(done), config, item["involved_classes"],
                    "Qwen/Qwen3.5-4B-FC", run_id,
                    long_context=False, is_evaL_run=False)
            except Exception as exc:
                print(f"    replay stopped on {item['id']} turn {index}: "
                      f"{type(exc).__name__}: {exc}"[:150])
                ok = False
                break
            for result in results[-len(calls):]:
                messages.append({"role": "tool", "content": str(result)})
            # the turn's closing reply, trained as its own row: this is the only
            # place the model sees the act of finishing. A pool of pure call
            # turns teaches it never to conclude, which is what zeroed Multi
            # Turn and pinned AppWorld at its step cap.
            if pending_reply:
                rows.append({
                    "task_id": item["id"],
                    "teacher": "gen",
                    "turn_index": index,
                    "prompt": adapter._render(messages, functions),
                    "response": pending_reply,
                    "token_hint": max(len(pending_reply) // 4, 1),
                    "_task_phat": 0.0,
                    "_traj": f"{item['id']}#t{index}r",
                    "_seed_task": seed,
                    "_seed_category": item.get("seed_category", ""),
                })
                messages.append({"role": "assistant",
                                 "content": pending_reply})
        if not ok:
            skipped += 1

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    import collections
    by_category = collections.Counter(r["_seed_category"] for r in rows)
    call_rows = sum(1 for r in rows if "<tool_call>" in r["response"])
    print(f"[genrows] call turns {call_rows}, closing turns "
          f"{len(rows) - call_rows}")
    if len(rows) == call_rows:
        print("[genrows] WARNING: no closing turns at all -- this pool teaches "
              "the model never to finish")
    print(f"[genrows] {episodes} episodes -> {len(rows)} rows "
          f"({skipped} episodes dropped mid-replay)")
    for category, count in sorted(by_category.items()):
        print(f"    {count:4d}  {category}")
    print(f"[genrows] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
