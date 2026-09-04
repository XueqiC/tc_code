#!/usr/bin/env python3
"""Guided rollout pass for the BFCL demand split.

Per the method protocol, guidance activates only for demand tasks whose
unguided attempts all failed and whose teacher demonstration is
verified. The guidance context is the teacher's worked example for the
same task, injected into the first user turn. Injection is done by
patching the official data files in place (backed up and restored
afterwards) so that generation and verification run through the
untouched official pipeline.

Usage: bfcl_guided_pass.py --port 8907 [--repeats 4]
Assumes a student vllm server is already running on --port.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BFCL = ROOT / "envs/bfcl/gorilla/berkeley-function-call-leaderboard"
DATA = BFCL / "bfcl_eval/data"

GUIDE_HEADER = (
    "Worked example from an expert on this exact task. Study the "
    "approach, then solve the task yourself:\n"
)


def rollout_verdicts(repeats: int) -> dict[str, bool]:
    """id -> True if verified in ANY unguided rollout."""
    split = json.load((ROOT / "configs/bfcl_support_split.json").open())
    demand = set(split["demand"])
    ok: dict[str, bool] = {i: False for i in demand}
    for r in range(repeats):
        gen: set[str] = set()
        for f in (BFCL / f"result_roll_base_r{r}").rglob("*_result.json"):
            for line in f.open():
                if line.strip():
                    gen.add(json.loads(line)["id"])
        failed: set[str] = set()
        for f in (BFCL / f"score_roll_base_r{r}").rglob("*_score.json"):
            lines = [json.loads(l) for l in f.open() if l.strip()]
            for e in lines[1:]:
                if isinstance(e, dict) and "id" in e:
                    failed.add(e["id"])
        for i in (gen - failed) & demand:
            ok[i] = True
    return ok


def demo_block(demo: dict) -> str:
    calls = json.dumps(demo["teacher_result"], ensure_ascii=False)
    if len(calls) > 4000:
        calls = calls[:4000] + " ..."
    return f"{GUIDE_HEADER}Expert solution: {calls}\n\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8907)
    ap.add_argument("--repeats", type=int, default=4)
    ap.add_argument("--tag", default="ds")
    args = ap.parse_args()

    verdicts = rollout_verdicts(args.repeats)
    demos = json.load((ROOT / f"data/bfcl_sft/demos_{args.tag}.json").open())
    guided = sorted(
        i for i, solved in verdicts.items() if not solved and i in demos
    )
    unsolved_no_demo = sorted(
        i for i, solved in verdicts.items() if not solved and i not in demos
    )
    print(f"[guided] unguided-solved={sum(verdicts.values())}/"
          f"{len(verdicts)} guided-eligible={len(guided)} "
          f"no-demo={len(unsolved_no_demo)}")
    if not guided:
        print("[guided] nothing to do")
        return 0

    # patch data files, keeping pristine backups.
    # A memory task is stored under its group id (memory_3-x) but runs under a
    # per-backend id (memory_kv_3-x), so the row to patch is found by mapping
    # back; two backends selecting the same underlying row would need two
    # different worked examples in one place, so the later one is skipped.
    def base_id(task_id: str) -> str:
        for backend in ("kv", "vector", "rec_sum"):
            prefix = f"memory_{backend}_"
            if task_id.startswith(prefix):
                return "memory_" + task_id[len(prefix):]
        return task_id

    guided_by_row: dict[str, str] = {}
    for task_id in guided:
        row_id = base_id(task_id)
        if row_id in guided_by_row:
            print(f"[guided] skipping {task_id}: row {row_id} already patched "
                  f"for {guided_by_row[row_id]}")
            continue
        guided_by_row[row_id] = task_id
    guided_set = set(guided_by_row)
    backups: list[tuple[Path, Path]] = []
    by_cat: dict[str, list[str]] = {}
    try:
        for f in sorted(DATA.glob("BFCL_v4_*.json")):
            lines = f.read_text().splitlines()
            changed = False
            out_lines = []
            for line in lines:
                s = line.strip()
                row = None
                if s:
                    try:
                        row = json.loads(s)
                    except json.JSONDecodeError:
                        row = None
                if isinstance(row, dict) and row.get("id") in guided_set:
                    task_id = guided_by_row[row["id"]]
                    first_turn = row["question"][0]
                    for m in first_turn:
                        if m.get("role") == "user":
                            m["content"] = demo_block(demos[task_id]) + m["content"]
                            break
                    out_lines.append(json.dumps(row, ensure_ascii=False))
                    changed = True
                else:
                    out_lines.append(line)
            if changed:
                bak = f.with_suffix(".json.bak_guided")
                shutil.copy2(f, bak)
                backups.append((f, bak))
                f.write_text("\n".join(out_lines) + "\n")
        # the adapter resolves each id to its runnable category and carries the
        # memory prerequisite write-chain; filename membership would emit the
        # bare `memory` group name and crash generation
        sys.path.insert(0, str(ROOT / "src"))
        from bfas.adapters.bfcl import BFCLAdapter

        by_cat = BFCLAdapter()._selective_file(sorted(guided_by_row.values()))
        json.dump(by_cat, (BFCL / "test_case_ids_to_generate.json").open("w"),
                  indent=1)

        # Stateful tasks carry their signal in the turn stream, not in a single
        # response, so the guided run records it the same way the mt pass does.
        # Without this a stateful task that only succeeds under guidance can
        # never become training rows, and its whole category can end up empty.
        dump_dir = ROOT / "data/bfcl_dumps"
        dump_dir.mkdir(parents=True, exist_ok=True)
        env_prefix = (
            f"cd {BFCL.parent.parent} && . .venv/bin/activate && "
            f"export LOCAL_SERVER_ENDPOINT=localhost LOCAL_SERVER_PORT={args.port} && "
        )
        for r in range(args.repeats):
            dump_path = dump_dir / f"guided_dump_r{r}.jsonl"
            dump_path.write_text("")
            env_prefix_r = (
                env_prefix + f"export BFCL_DUMP_MESSAGES={dump_path} && "
            )
            for cmd in (
                f"bfcl generate --model Qwen/Qwen3.5-4B-FC --run-ids "
                f"--skip-server-setup --temperature 0.7 --num-threads 4 "
                f"--result-dir result_roll_guided_r{r}",
                f"bfcl evaluate --model Qwen/Qwen3.5-4B-FC "
                f"--result-dir result_roll_guided_r{r} "
                f"--score-dir score_roll_guided_r{r} --partial-eval",
            ):
                res = subprocess.run(
                    ["bash", "-c", env_prefix_r + cmd],
                    capture_output=True, text=True)
                if res.returncode != 0:
                    print(f"[guided] CMD FAILED r{r}: {res.stderr[-400:]}",
                          file=sys.stderr)
            print(f"[guided] rollout {r} done", flush=True)
    finally:
        for f, bak in backups:
            shutil.move(str(bak), str(f))
        print(f"[guided] data files restored ({len(backups)})")
    print("[guided] GUIDED PASS DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
