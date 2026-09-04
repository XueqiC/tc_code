#!/usr/bin/env python3
"""What would it take to generate new tasks for each benchmark?

The generation mechanism -- rank by suspicion, observe or generate, saturate on
information, repair on verification failure -- carries across benchmarks
untouched. What does not carry is the pair it needs underneath: something that
can WRITE a new task, and something that can DECIDE whether the written task has
a correct answer. This probe reports, per benchmark, what a task is actually made
of and which verifier the benchmark already ships, so the missing pieces are
named rather than discovered one crash at a time.

It touches no GPU, spends no teacher quota, and runs no rollouts.
"""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# what a generated task must carry for each benchmark, and what can judge it
EXPECTED = {
    "bfcl": {
        "task_fields": ["question", "function", "ground_truth",
                        "involved_classes", "initial_config"],
        "verifier": "bfcl_eval.eval_checker.ast_eval.ast_checker.ast_checker "
                    "(single turn) / execute_multi_turn_func_call (stateful)",
        "status": "BUILT: tools/bfcl_generate.py + tools/bfcl_generate_mt.py",
    },
    "appworld": {
        "task_fields": ["task_id", "instruction", "apps", "database state"],
        "verifier": "AppWorld's own per-task unit tests (state assertions)",
        "status": "NOT BUILT",
    },
    "alfworld": {
        "task_fields": ["game file", "goal", "initial world state"],
        "verifier": "the ALFWorld simulator's own goal condition",
        "status": "NOT BUILT",
    },
    "tau2": {
        "task_fields": ["domain", "user scenario", "policy", "expected actions"],
        "verifier": "tau2's action/state checks under its user simulator",
        "status": "NOT BUILT",
    },
}


def probe(name: str) -> dict:
    out: dict = {"benchmark": name}
    out.update(EXPECTED.get(name, {}))
    try:
        if name == "bfcl":
            from bfas.adapters.bfcl import BFCLAdapter as A
            adapter = A()
        elif name == "appworld":
            from bfas.adapters.appworld import AppWorldAdapter as A
            adapter = A(0)
        elif name == "alfworld":
            from bfas.adapters.alfworld import ALFWorldAdapter as A
            adapter = A(0)
        else:
            from bfas.adapters.tau2 import Tau2Adapter as A
            adapter = A(0)
    except Exception as exc:
        out["adapter"] = f"import failed: {type(exc).__name__}: {exc}"[:200]
        return out

    try:
        pool = adapter.task_pool()
        out["pool_size"] = len(pool)
        out["sample_task"] = pool[0].task_id if pool else None
        categories = sorted({t.category for t in pool})
        out["categories"] = categories[:8] + (
            ["..."] if len(categories) > 8 else [])
    except Exception as exc:
        out["pool"] = f"failed: {type(exc).__name__}: {exc}"[:200]
    return out


def main() -> int:
    report = [probe(name) for name in ("bfcl", "appworld", "alfworld", "tau2")]
    for row in report:
        print(f"\n=== {row['benchmark']} ===")
        print(f"  status      {row.get('status')}")
        print(f"  pool        {row.get('pool_size', row.get('pool', 'n/a'))}"
              f"  e.g. {row.get('sample_task')}")
        if row.get("categories"):
            print(f"  categories  {', '.join(row['categories'])}")
        print(f"  a task is   {', '.join(row.get('task_fields', []))}")
        print(f"  verifier    {row.get('verifier')}")
        if row.get("adapter"):
            print(f"  ADAPTER     {row['adapter']}")
    out = ROOT / "results/analysis/gen_readiness.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
