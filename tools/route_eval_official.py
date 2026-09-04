#!/usr/bin/env python3
"""Selective-specialization routing with OFFICIAL aggregation.

Builds a synthetic model scoredir by choosing, per category, the score
file from either the base or the specialized arm according to the S_c
certificate, then computes the routed Overall with the official
leaderboard aggregation code (identical formula and weights).

Certificate: a category routes to the specialized student only if its
success count on the calibration tasks S_c strictly exceeds the base
student's; ties or no calibration evidence default to base.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BFCL_ROOT = ROOT / "envs/bfcl/gorilla/berkeley-function-call-leaderboard"
sys.path.insert(0, str(BFCL_ROOT))


def category_of(path: Path) -> str:
    return path.name.replace("BFCL_v4_", "").replace("_score.json", "")


def calib_success(scoredir: Path, calib: set[str]) -> dict[str, int]:
    """category -> number of calibration tasks NOT listed as failures."""
    out: dict[str, int] = {}
    for path in scoredir.rglob("BFCL_v4_*_score.json"):
        cat = category_of(path)
        failures = set()
        lines = [l for l in path.read_text().splitlines() if l.strip()]
        for line in lines[1:]:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("id") is not None:
                failures.add(str(row["id"]))
        cal_in_cat = {t for t in calib if t.rsplit("_", 1)[0] in cat or cat in t}
        # calibration membership by exact id prefix match against data ids
        # is handled by the caller passing per-category calib ids; here we
        # count calib ids whose category prefix matches this category name.
        cal_ids = {t for t in calib if t.startswith(cat + "_")}
        out[cat] = sum(1 for t in cal_ids if t not in failures)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default="base")
    ap.add_argument("--spec", required=True)
    ap.add_argument("--split", default="configs/bfcl_support_split.json")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    split = json.load((ROOT / args.split).open())
    calib = set(split.get("calibration") or [])
    base_dir = ROOT / "results/bfcl_std" / args.base / "scoredir"
    spec_dir = ROOT / "results/bfcl_std" / args.spec / "scoredir"

    sb = calib_success(base_dir, calib)
    ss = calib_success(spec_dir, calib)
    routing = {cat: ("spec" if ss.get(cat, 0) > sb.get(cat, 0) else "base")
               for cat in set(sb) | set(ss)}

    out_root = ROOT / "results/bfcl_std" / f"routed_{args.spec}"
    synth = out_root / "synthetic_scores" / "Qwen_Qwen3.5-4B-FC"
    if synth.parent.exists():
        shutil.rmtree(synth.parent)
    for cat, dst in routing.items():
        src_dir = spec_dir if dst == "spec" else base_dir
        matches = list(src_dir.rglob(f"BFCL_v4_{cat}_score.json"))
        if not matches:
            matches = list(base_dir.rglob(f"BFCL_v4_{cat}_score.json"))
        if not matches:
            continue
        src = matches[0]
        rel = src.relative_to(spec_dir if dst == "spec" else base_dir)
        target = synth / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, target)

    from bfcl_eval.eval_checker.eval_runner_helper import (
        generate_leaderboard_csv,
        update_leaderboard_table_with_local_score_file,
    )
    table: dict = {}
    update_leaderboard_table_with_local_score_file(table, synth.parent)
    generate_leaderboard_csv(table, synth.parent)

    import csv
    row = next(csv.DictReader(open(synth.parent / "data_overall.csv")))
    n_spec = sum(1 for v in routing.values() if v == "spec")
    print(f"[route-official] {args.spec}: {n_spec}/{len(routing)} categories -> spec")
    for cat, dst in sorted(routing.items()):
        if dst == "spec":
            print(f"  spec <- {cat} (S_c {ss.get(cat,0)} vs base {sb.get(cat,0)})")
    print(f"[route-official] routed Overall Acc = {row['Overall Acc']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
