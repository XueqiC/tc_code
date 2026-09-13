#!/usr/bin/env python3
"""Aggregate the 27 paper-baseline runs using only the Python standard library.

Example (output files default to the current directory, never the input root)::

    python3 tools/table1_aggregate.py --results-root results/paper_baselines \
        --json-out table1_summary.json --latex-out table1_body.tex \
        --initial-alfworld 56.4 --initial-hotpotqa 38.2 --initial-bfcl 45.6

The root directly contains table1_<benchmark>_<method>_s<seed>/metrics.json;
seed zero also accepts table1_<benchmark>_<method>/metrics.json. The baseline
runner appends seed suffixes itself, so corrected seed-zero submissions can be
unsuffixed while earlier ones use _s0. Prefer the spelling with metrics.json;
if both have it, stop with a conflict. Doubled suffixes (e.g. _s1_s1) from
already-suffixed submissions are listed as ignored and never enter the table.
Only complete runs with valid primary scores and spend enter score statistics.
Sample standard deviation uses ddof=1 and is null/-- for fewer than two seeds.
Spend includes all valid receipts, even from incomplete runs; it is never capped.
HotpotQA F1 is a JSON-only secondary metric in its native fraction (0--1) units.
Average improvement equally weights the three benchmark means, when all exist;
it does not require the same available seeds across benchmarks.

The LaTeX body follows tab:main in tc-paper/paper/sections/exp_results.tex:
one decimal, ALFWorld/HotpotQA/BFCL order, and the existing row labels. GAD and
RTD are pending placeholders outside this 27-run campaign. It uses the paper's
rowhead/rowours colors and midrule macro, and does not edit the paper itself.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import statistics


BENCHMARKS = ("alfworld", "hotpotqa", "bfcl")  # Paper column order.
METHODS = ("smartad", "sad", "kang")
SEEDS = (0, 1, 2)
LABELS = {"smartad": "SmartAD", "sad": "SAD", "kang": "Agent Distillation"}
INITIAL = {"alfworld": 56.4, "hotpotqa": 38.2, "bfcl": 45.6}
PRIMARY_METRICS = {
    "alfworld": "valid_seen success rate",
    "hotpotqa": "exact match",
    "bfcl": "official overall accuracy",
}
TEACHER_CAP = 30_000


def _number(value, *, maximum=None, integer=False):
    """Reject booleans, non-finite values, and malformed metric values."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(value) or value < 0:
        return None
    if maximum is not None and value > maximum:
        return None
    if integer and value != int(value):
        return None
    return int(value) if integer else float(value)


def _stats(per_seed):
    values = list(per_seed.values())
    return {
        "n": len(values),
        "seeds": [int(seed) for seed in per_seed],
        "per_seed": per_seed,
        "mean": statistics.mean(values) if values else None,
        "std": statistics.stdev(values) if len(values) >= 2 else None,
    }


def _read_run(root, benchmark, method, seed):
    base = f"table1_{benchmark}_{method}"
    name = f"{base}_s{seed}"
    path = root / name / "metrics.json"
    if seed == 0:
        alternate = root / base / "metrics.json"
        if path.is_file() and alternate.is_file():
            raise ValueError(
                f"conflicting directories for {name} (seed 0): "
                f"both {path} and {alternate} exist"
            )
        if alternate.is_file() or (not path.parent.is_dir() and alternate.parent.is_dir()):
            path = alternate
    run = {
        "cell": name, "seed": seed, "metrics_path": str(path),
        "status": "missing", "complete": None,
        "overall_accuracy_percent": None, "teacher_tokens_charged": None,
        "issues": [],
    }
    try:
        metrics = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        run["issues"].append("missing metrics.json" if path.parent.is_dir() else "missing directory")
        return run
    except (OSError, ValueError) as exc:
        run.update(status="invalid", issues=[f"cannot read metrics.json: {exc}"])
        return run
    if not isinstance(metrics, dict):
        run.update(status="invalid", issues=["metrics.json must contain an object"])
        return run

    complete = metrics.get("complete")
    run["complete"] = complete if isinstance(complete, bool) else None
    run["overall_accuracy_percent"] = _number(metrics.get("overall_accuracy_percent"), maximum=100)
    run["teacher_tokens_charged"] = _number(metrics.get("teacher_tokens_charged"), integer=True)
    if complete is False:
        run.update(status="incomplete", issues=["complete=false; excluded from score statistics"])
    elif complete is not True:
        run.update(status="invalid", issues=["complete must be a boolean"])
    else:
        run["status"] = "complete"
    for field in ("overall_accuracy_percent", "teacher_tokens_charged"):
        if run[field] is None:
            run["issues"].append(f"missing or invalid {field}")
            if run["status"] == "complete":
                run["status"] = "invalid"
    if benchmark == "hotpotqa":
        run["em"] = _number(metrics.get("em"), maximum=1)
        run["f1"] = _number(metrics.get("f1"), maximum=1)
        if run["status"] == "complete" and run["f1"] is None:
            run["issues"].append("missing or invalid f1; excluded from secondary statistics")
    if benchmark == "alfworld" and "per_category" in metrics:
        run["per_category"] = metrics["per_category"]
    return run


def aggregate(results_root, initial=None):
    """Read expected run paths without modifying inputs; absent roots are allowed."""
    root = Path(results_root)
    initial = INITIAL | (initial or {})
    for benchmark in BENCHMARKS:
        if _number(initial[benchmark], maximum=100) is None:
            raise ValueError(f"initial {benchmark} must be a finite percentage in [0, 100]")
    summary = {
        "results_root": str(root), "expected_seeds": list(SEEDS),
        "benchmark_order": list(BENCHMARKS), "method_order": list(METHODS),
        "initial_student_percent": initial, "teacher_token_cap": TEACHER_CAP,
        "std_convention": "sample (ddof=1); null when n < 2",
        "score_policy": "complete runs with valid primary score and teacher spend only",
        "spend_policy": "all valid reported receipts, including incomplete/invalid runs",
        "average_improvement_policy": "equal mean of three benchmark improvements; null if any is absent",
        "missing_cells": [], "incomplete_cells": [], "invalid_cells": [],
        "ignored_directories": sorted(
            path.name for path in root.glob("table1_*")
            if path.is_dir() and re.search(r"_s[0-9]+(?:_s[0-9]+)+$", path.name)
        ),
        "issues": [], "benchmarks": {}, "methods": {},
    }
    all_runs = []
    for benchmark in BENCHMARKS:
        groups = summary["benchmarks"][benchmark] = {}
        for method in METHODS:
            runs = [_read_run(root, benchmark, method, seed) for seed in SEEDS]
            all_runs.extend(runs)
            for run in runs:
                if run["status"] != "complete":
                    summary[f"{run['status']}_cells"].append(run["cell"])
                if run["issues"]:
                    summary["issues"].append({"cell": run["cell"], "messages": run["issues"]})
            scores = {str(r["seed"]): r["overall_accuracy_percent"]
                      for r in runs if r["status"] == "complete"}
            spend = _stats({str(r["seed"]): r["teacher_tokens_charged"]
                            for r in runs if r["teacher_tokens_charged"] is not None})
            spend["max"] = max(spend["per_seed"].values(), default=None)
            spend["seeds_over_cap"] = [int(s) for s, v in spend["per_seed"].items() if v > TEACHER_CAP]
            group = groups[method] = {
                "primary_metric": PRIMARY_METRICS[benchmark],
                "overall_accuracy_percent": _stats(scores),
                "improvement_pp": _stats({s: v - initial[benchmark] for s, v in scores.items()}),
                "teacher_tokens_charged": spend,
                "runs": runs,
            }
            if benchmark == "hotpotqa":
                group["f1"] = dict(_stats({str(r["seed"]): r["f1"] for r in runs
                                           if r["status"] == "complete" and r["f1"] is not None}),
                                   unit="fraction (0-1)")

    for method in METHODS:
        groups = {b: summary["benchmarks"][b][method] for b in BENCHMARKS}
        counts = {b: g["overall_accuracy_percent"]["n"] for b, g in groups.items()}
        summary["methods"][method] = {
            "label": LABELS[method], "seed_counts": counts,
            "benchmark_count": sum(n > 0 for n in counts.values()),
            "average_improvement_pp": statistics.mean(g["improvement_pp"]["mean"] for g in groups.values())
            if all(counts.values()) else None,
        }
    summary["counts"] = {"expected": len(all_runs)} | {
        status: sum(r["status"] == status for r in all_runs)
        for status in ("complete", "missing", "incomplete", "invalid")
    }
    summary["teacher_tokens_charged_max"] = max(
        (r["teacher_tokens_charged"] for r in all_runs if r["teacher_tokens_charged"] is not None),
        default=None,
    )
    summary["teacher_spend_receipt_count"] = sum(r["teacher_tokens_charged"] is not None for r in all_runs)
    summary["over_cap_cells"] = [r["cell"] for r in all_runs
                                 if r["teacher_tokens_charged"] is not None
                                 and r["teacher_tokens_charged"] > TEACHER_CAP]
    return summary


def _latex_cell(stats):
    n = stats["n"]
    if n == 0:
        value = r"\cdot"
    else:
        std = f"{stats['std']:.1f}" if stats["std"] is not None else r"\text{--}"
        value = f"{stats['mean']:.1f}" + r"\pm " + std
    if n < len(SEEDS):
        value = "{" + value + r"}^{\dagger}"
    return "$" + value + "$" + rf" {{\scriptsize ($n={n}$)}}"


def latex_body(summary):
    """Return the five-column Table 1 body, including pending GAD/RTD rows."""
    lines = [
        "% ALFWorld | HotpotQA | BFCL v4 | average improvement (pp).",
        "% n counts completed training seeds; -- std is undefined for n < 2.",
        "% Average improvement uses the seed counts printed in the three metric cells.",
        f"% Maximum reported teacher spend: {summary['teacher_tokens_charged_max']} / {TEACHER_CAP} tokens; "
        f"{summary['teacher_spend_receipt_count']}/27 receipts, including incomplete runs.",
        r"\rowcolor{rowhead}Initial student (Gemma-4-12B) & "
        + " & ".join(f"${summary['initial_student_percent'][b]:.1f}$" for b in BENCHMARKS)
        + r" & --\\ % Single reference evaluation per benchmark (n=1).",
        r"\midrule",
    ]
    for method in METHODS:
        row = summary["methods"][method]
        cells = [_latex_cell(summary["benchmarks"][b][method]["overall_accuracy_percent"])
                 for b in BENCHMARKS]
        delta = row["average_improvement_pp"]
        delta_text = r"\cdot" if delta is None else f"{delta:+.1f}"
        if delta is not None and any(n < len(SEEDS) for n in row["seed_counts"].values()):
            delta_text += r"^{\dagger}"
        lines.append(LABELS[method] + " & " + " & ".join(cells) + " & $" + delta_text + r"$\\")
    pending = " & ".join([_latex_cell(_stats({}))] * 3 + [r"$\cdot$"])
    lines.extend([
        "% GAD and RTD are outside this campaign; pending placeholders only.",
        "GAD & " + pending + r"\\",
        r"\midrule",
        r"\rowcolor{rowours}\textbf{RTD (ours)} & " + pending + r"\\",
    ])
    return "\n".join(lines) + "\n"


def human_table(summary):
    lines = [
        "Table 1: primary scores in percent; sample std; complete seeds only.",
        "Initial student: " + ", ".join(f"{b}={summary['initial_student_percent'][b]:.1f}" for b in BENCHMARKS),
        "Benchmark Method              Mean +/- std (n/3)  Delta pp  Scores by seed; teacher tokens by seed (all receipts), max/cap",
    ]
    for benchmark in BENCHMARKS:
        for method in METHODS:
            group = summary["benchmarks"][benchmark][method]
            score, spend = group["overall_accuracy_percent"], group["teacher_tokens_charged"]
            mean = "--" if score["mean"] is None else f"{score['mean']:.1f}"
            std = "--" if score["std"] is None else f"{score['std']:.1f}"
            delta = group["improvement_pp"]["mean"]
            delta_text = "--" if delta is None else f"{delta:+.1f}"
            values = ",".join(f"s{s}={v:.4f}" for s, v in score["per_seed"].items()) or "--"
            tokens = ",".join(f"s{s}={v}" for s, v in spend["per_seed"].items()) or "--"
            lines.append(f"{benchmark:9} {LABELS[method]:18} {mean:>5} +/- {std:<4} ({score['n']}/3) "
                         f"{delta_text:>8}  [{values}]; [{tokens}] (n={spend['n']}, "
                         f"max={spend['max'] if spend['max'] is not None else '--'}/{TEACHER_CAP})")
    lines.append("Average improvement (pp; equal benchmark weights):")
    for method, row in summary["methods"].items():
        delta = row["average_improvement_pp"]
        value = "pending" if delta is None else f"{delta:+.1f}"
        counts = ", ".join(f"{b}:{n}" for b, n in row["seed_counts"].items())
        lines.append(f"  {LABELS[method]}: {value}; benchmarks={row['benchmark_count']}/3; seeds=[{counts}]")
    counts = summary["counts"]
    lines.append("Runs: " + ", ".join(f"{k}={v}" for k, v in counts.items()))
    lines.append(f"Maximum reported teacher spend: {summary['teacher_tokens_charged_max']} / {TEACHER_CAP} tokens "
                 f"({summary['teacher_spend_receipt_count']}/27 receipts, including incomplete runs).")
    for issue in summary["issues"]:
        lines.append(f"  {issue['cell']}: " + "; ".join(issue["messages"]))
    if summary["over_cap_cells"]:
        lines.append("Over teacher cap: " + ", ".join(summary["over_cap_cells"]))
    if summary["ignored_directories"]:
        lines.append("Ignored directories (doubled seed suffix): " + ", ".join(summary["ignored_directories"]))
    return "\n".join(lines)


def _percentage(text):
    try:
        value = float(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected a percentage") from exc
    if _number(value, maximum=100) is None:
        raise argparse.ArgumentTypeError("expected a finite percentage in [0, 100]")
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-root", type=Path, default=Path("results/paper_baselines"))
    parser.add_argument("--json-out", type=Path, default=Path("table1_summary.json"))
    parser.add_argument("--latex-out", type=Path, default=Path("table1_body.tex"))
    for benchmark in BENCHMARKS:
        parser.add_argument(f"--initial-{benchmark}", type=_percentage, default=INITIAL[benchmark])
    args = parser.parse_args(argv)
    outputs = [args.json_out.resolve(), args.latex_out.resolve()]
    if outputs[0] == outputs[1]:
        parser.error("JSON and LaTeX outputs must have different paths")
    if any(path.is_relative_to(args.results_root.resolve()) for path in outputs):
        parser.error("output files must be outside the results root")
    initial = {b: getattr(args, f"initial_{b}") for b in BENCHMARKS}
    try:
        summary = aggregate(args.results_root, initial)
    except ValueError as exc:
        parser.error(str(exc))
    json_text = json.dumps(summary, indent=2, allow_nan=False) + "\n"
    latex_text = latex_body(summary)
    for path, contents in ((args.json_out, json_text), (args.latex_out, latex_text)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents, encoding="utf-8")
    print(human_table(summary))
    print(f"JSON: {args.json_out}\nLaTeX: {args.latex_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
