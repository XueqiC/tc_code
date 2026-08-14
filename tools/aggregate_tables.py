#!/usr/bin/env python3
"""Aggregate bootstrap-protocol results into the cross-matrix (Table 1)
and best-config (Table 2) numbers.

Scans results/e3_tier1_* metrics for matrix conditions (M_{acq}_{cur}),
standalone arms (F_*), and rescue cells; groups by (student, budget, task,
condition) across seeds; prints mean±std tables plus purity/waste columns
where available."""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

ROWS = ("selfinst", "evol", "llm2llm", "ourscorr", "ours")
COLS = ("plain", "alpagasus", "less", "ours")

def cell_key(tag):
    """Extract (student, budget, task) from a run tag."""
    student = "2B" if "2b" in tag else ("0.8B" if "08b" in tag else "?")
    m = re.search(r"b(\d+)k", tag)
    budget = m.group(1) + "k" if m else ("20k" if "2b" in tag else "?")
    task = "sqljoin" if "sqljoin" in tag else "gsm"
    return student, budget, task

def main():
    acc = defaultdict(list)
    for metrics_path in sorted(RESULTS.glob("e3_tier1_*/metrics.json")):
        tag = metrics_path.parent.name.replace("e3_tier1_", "")
        if not any(s in tag for s in ("mx_", "fam_", "e9t_", "nb_")):
            continue
        try:
            metrics = json.loads(metrics_path.read_text())
        except Exception:
            continue
        student, budget, task = cell_key(tag)
        for cond, data in metrics.get("conditions", {}).items():
            if cond == "base" or "eval" not in data:
                continue
            ev = data["eval"]
            if task == "sqljoin":
                acc[(student, budget, task, cond, "sql")].append(
                    ev["sql"]["single_statement_valid_rate"])
                acc[(student, budget, task, cond, "residue")].append(
                    ev["gsm8k_code"]["exec_acc"])
            else:
                acc[(student, budget, task, cond, "exec")].append(
                    ev["gsm8k_code"]["exec_acc"])

    want = sys.argv[1] if len(sys.argv) > 1 else "2B"
    print(f"=== Cross matrix (student={want}, gsm, exec mean±std) ===")
    for budget in ("2k", "10k", "20k"):
        header_done = False
        for row in ROWS:
            line = []
            for col in COLS:
                vals = acc.get((want, budget, "gsm", f"M_{row}_{col}", "exec"))
                line.append(
                    f"{np.mean(vals):.3f}±{np.std(vals):.3f}(n={len(vals)})"
                    if vals else "--")
            if any(v != "--" for v in line):
                if not header_done:
                    print(f"-- budget {budget} --")
                    header_done = True
                print(f"{row:9s} " + "  ".join(
                    f"{c}:{v}" for c, v in zip(COLS, line)))
    print()
    print(f"=== Standalone arms (student={want}) ===")
    for key in sorted(acc):
        student, budget, task, cond, metric = key
        if student != want or cond.startswith("M_"):
            continue
        vals = acc[key]
        print(f"{task:7s} b{budget:4s} {cond:16s} {metric:8s} "
              f"{np.mean(vals):.3f}±{np.std(vals):.3f} (n={len(vals)})")

if __name__ == "__main__":
    main()
