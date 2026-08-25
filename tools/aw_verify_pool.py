#!/usr/bin/env python3
"""Replay-verify teacher trajectories in the demo pool.

For every task in the demo pool, execute its recorded assistant code
blocks in a fresh AppWorld session in order, then run the official
task evaluation. Writes per-task verdicts to
data/appworld_sft/pool_verdicts.json.

Run with the appworld venv python and APPWORLD_ROOT set.
"""
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CODE_RE = re.compile(r"```(?:python)?\n(.*?)```", re.S)


def code_blocks(response: str) -> list[str]:
    blocks = CODE_RE.findall(response)
    return [b for b in blocks if b.strip()] or (
        [response] if response.strip() and "apis." in response else []
    )


def main() -> int:
    from appworld import AppWorld

    rows = [json.loads(l) for l in
            (ROOT / "data/appworld_sft/pool.jsonl").open()]
    by_task: dict[str, list] = {}
    for r in rows:
        by_task.setdefault(r["task_id"], []).append(
            (r.get("turn_index", 0), r["response"])
        )

    out_path = ROOT / "data/appworld_sft/pool_verdicts.json"
    verdicts = json.load(out_path.open()) if out_path.exists() else {}
    for n, (task_id, turns) in enumerate(sorted(by_task.items()), 1):
        if task_id in verdicts:
            continue
        try:
            with AppWorld(task_id=task_id, experiment_name="verify_pool") as world:
                for _, resp in sorted(turns):
                    for block in code_blocks(resp):
                        try:
                            world.execute(block)
                        except Exception:
                            pass
                ev = world.evaluate()
                report = ev.to_dict() if hasattr(ev, "to_dict") else dict(ev)
                passes = report.get("passes", report.get("passed_tests", []))
                fails = report.get("failures", report.get("failed_tests", []))
                np_, nf = (len(passes) if isinstance(passes, list) else int(passes),
                           len(fails) if isinstance(fails, list) else int(fails))
                verdicts[task_id] = {"passed": np_, "failed": nf,
                                     "success": np_ > nf}
        except Exception as exc:
            verdicts[task_id] = {"error": str(exc)[:200], "success": False}
        json.dump(verdicts, out_path.open("w"), indent=1)
        if n % 10 == 0:
            ok = sum(1 for v in verdicts.values() if v.get("success"))
            print(f"[verify] {n}/{len(by_task)} verified, success={ok}",
                  flush=True)
    ok = sum(1 for v in verdicts.values() if v.get("success"))
    print(f"[verify] COMPLETE {len(verdicts)} tasks, success={ok}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
