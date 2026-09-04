#!/usr/bin/env python3
"""P2: on-policy repair collection (DAgger-style, for distillation).

The student walks each generated episode itself. After every turn its backend
state is compared with the reference state reached by the verified ground-truth
calls. At the FIRST divergent turn the two states were still equal beforehand,
so the ground-truth calls for that turn are, by construction, a correct action
from the state the student is actually in -- no extra teacher call needed. The
training row pairs the STUDENT'S OWN context (its wording, its tool outputs)
with that correct action: expert correction on student-visited states, which
plain behavior cloning of teacher trajectories never provides.

Rows carry _task_phat = 0 (they exist only where the student demonstrably goes
wrong) and _repair_turn for per-cluster accounting later.

Needs a vLLM server holding the student.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BFCL = ROOT / "envs/bfcl/gorilla/berkeley-function-call-leaderboard"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(BFCL))

CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)

_spec = importlib.util.spec_from_file_location(
    "genrows", str(ROOT / "tools/bfcl_gen_rows.py"))
_genrows = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_genrows)
target_block = _genrows.target_block  # executable call strings -> tool_call text


def student_reply(prompt: str, port: int, temperature: float) -> str:
    request = urllib.request.Request(
        f"http://localhost:{port}/v1/completions",
        data=json.dumps({
            "model": "Qwen/Qwen3.5-4B",
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": 512,
        }).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.loads(response.read())["choices"][0]["text"]


def as_call_strings(text: str) -> list[str]:
    """The student's <tool_call> JSON blocks, as executable call strings."""
    calls = []
    for blob in CALL_RE.findall(text):
        try:
            call = json.loads(blob)
        except json.JSONDecodeError:
            continue
        name = call.get("name")
        if not name:
            continue
        arguments = call.get("arguments") or {}
        rendered = ", ".join(f"{key}={value!r}"
                             for key, value in arguments.items())
        calls.append(f"{name}({rendered})")
    return calls


# attributes that identify the LANE rather than the state: the ref and stu
# instances are constructed with different run ids and snapshot paths, so
# comparing these made every episode "diverge" at turn 0 with identical calls
LANE_ATTRS = {"test_id", "scenario", "model_result_dir",
              "snapshot_folder", "latest_snapshot_file"}


def state_of(instances: dict) -> str:
    """A comparable snapshot of every backend instance's public state."""
    from pathlib import Path as _Path

    snapshot = {}
    for name, instance in sorted(instances.items()):
        snapshot[name] = {
            key: value for key, value in sorted(vars(instance).items())
            if not key.startswith("_")
            and key not in LANE_ATTRS
            and not isinstance(value, _Path)
        }
    return json.dumps(snapshot, sort_keys=True, default=str)


def backend_config(item: dict, lane: str, index: int, workdir: Path):
    involved = item["involved_classes"]
    memory_classes = [c for c in involved if c.startswith("MemoryAPI_")]
    if memory_classes:
        category = memory_classes[0].replace("MemoryAPI_", "memory_")
        run_id = f"{category}_prereq_{index}-{lane}-0"
        slice_ = {"model_result_dir": workdir, "test_id": run_id,
                  "scenario": item.get("scenario") or "gen"}
        return {name: dict(slice_) for name in memory_classes}, run_id
    if "WebSearchAPI" in involved and not item.get("initial_config"):
        return {"WebSearchAPI": {"show_snippet": True}}, f"{item['id']}-{lane}"
    return item.get("initial_config") or {}, f"{item['id']}-{lane}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes",
                        default="data/bfcl_sft/gen_stateful_v3.jsonl")
    parser.add_argument("--port", type=int, default=8975)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out",
                        default="data/bfcl_sft/pool_onpolicy_repair.jsonl")
    args = parser.parse_args()

    from bfas.adapters.bfcl import BFCLAdapter
    from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import (
        execute_multi_turn_func_call,
    )
    from bfcl_eval.utils import populate_test_cases_with_predefined_functions

    adapter = BFCLAdapter()
    entries, _ = adapter._load_entries()
    workdir = ROOT / "results/analysis/_repair_memory"
    workdir.mkdir(parents=True, exist_ok=True)
    schema_cache: dict[str, list] = {}

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    handle = out_path.open("w")
    episodes = clean = diverged = errored = 0
    for index, line in enumerate(
            (ROOT / args.episodes).read_text().splitlines()):
        if not line.strip():
            continue
        item = json.loads(line)
        if not item.get("executes"):
            continue
        if args.limit and episodes >= args.limit:
            break
        episodes += 1
        seed = item["seed_task"]
        if seed not in schema_cache:
            populated = populate_test_cases_with_predefined_functions(
                [json.loads(json.dumps(entries[seed]))])
            schema_cache[seed] = populated[0].get("function") or []
        functions = schema_cache[seed]

        ref_config, ref_id = backend_config(item, "ref", index, workdir)
        stu_config, stu_id = backend_config(item, "stu", index, workdir)

        messages: list[dict] = []
        ref_done: list[str] = []
        stu_done: list[str] = []
        outcome = "clean"
        for turn_index, (turn, gt_calls) in enumerate(
                zip(item["question"], item["ground_truth"])):
            user = turn[0]["content"] if turn else ""
            if user:
                messages.append({"role": "user", "content": user})

            prompt = adapter._render(messages, functions)
            try:
                reply = student_reply(prompt, args.port, args.temperature)
            except Exception as exc:
                print(f"    serve error on {item['id']}: {exc}")
                outcome = "error"
                break
            student_calls = as_call_strings(reply)

            ref_done.extend(gt_calls)
            stu_done.extend(student_calls)
            try:
                _, ref_instances = execute_multi_turn_func_call(
                    list(ref_done), ref_config, item["involved_classes"],
                    "Qwen/Qwen3.5-4B-FC", ref_id, long_context=False)
                stu_results, stu_instances = execute_multi_turn_func_call(
                    list(stu_done), stu_config, item["involved_classes"],
                    "Qwen/Qwen3.5-4B-FC", stu_id, long_context=False)
            except Exception as exc:
                print(f"    exec error on {item['id']} turn {turn_index}: "
                      f"{type(exc).__name__}")
                outcome = "error"
                break

            if state_of(ref_instances) != state_of(stu_instances):
                target = target_block(gt_calls)
                if target:
                    handle.write(json.dumps({
                        "task_id": item["id"],
                        "teacher": "repair",
                        "turn_index": turn_index,
                        "prompt": prompt,
                        "response": target,
                        "token_hint": max(len(target) // 4, 1),
                        "_task_phat": 0.0,
                        "_traj": f"{item['id']}#rep{turn_index}",
                        "_seed_task": seed,
                        "_seed_category": item.get("seed_category", ""),
                        "_repair_turn": turn_index,
                        "_student_did": student_calls,
                    }, ensure_ascii=False) + "\n")
                outcome = "diverged"
                print(f"    {item['id']}: diverged at turn {turn_index} "
                      f"(student did {student_calls or 'nothing'}, "
                      f"gt {gt_calls})")
                break

            # states match: keep walking on the student's own trajectory
            messages.append({"role": "assistant", "content": reply.strip()})
            if student_calls:
                for result in stu_results[-len(student_calls):]:
                    messages.append({"role": "tool", "content": str(result)})

        if outcome == "clean":
            clean += 1
        elif outcome == "diverged":
            diverged += 1
        else:
            errored += 1
        handle.flush()

    handle.close()
    print(f"[repair] episodes={episodes} clean={clean} diverged={diverged} "
          f"errored={errored} -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
