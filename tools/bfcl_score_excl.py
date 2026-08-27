#!/usr/bin/env python3
"""Re-aggregate a BFCL score directory excluding the support split.

Reads per-category score files (first line = summary, rest = failed
entries), recomputes accuracy over non-support entries only, and
prints the per-group and overall numbers using the official column
weights (equal-weighted mean over category accuracies within each
group, then the official overall composition).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BFCL = ROOT / "envs/bfcl/gorilla/berkeley-function-call-leaderboard"


def category_stats(score_dir: Path, support: set[str]) -> dict[str, tuple[int, int]]:
    stats: dict[str, tuple[int, int]] = {}
    for f in score_dir.rglob("BFCL_v4_*_score.json"):
        lines = [json.loads(l) for l in f.open() if l.strip()]
        if not lines:
            continue
        summary, failures = lines[0], lines[1:]
        total = int(summary.get("total_count", 0))
        failed_ids = {e["id"] for e in failures if isinstance(e, dict) and "id" in e}
        cat = f.stem.replace("BFCL_v4_", "").replace("_score", "")
        # ids of this category in the support set
        sup = {i for i in support if _cat_of(i) == cat}
        sup_failed = sup & failed_ids
        adj_total = total - len(sup)
        adj_correct = (total - len(failed_ids)) - (len(sup) - len(sup_failed))
        if adj_total > 0:
            stats[cat] = (adj_correct, adj_total)
    return stats


_CAT_CACHE: dict[str, str] = {}


def _cat_of(tid: str) -> str:
    if not _CAT_CACHE:
        for f in (BFCL / "bfcl_eval/data").glob("BFCL_v4_*.json"):
            cat = f.stem.replace("BFCL_v4_", "")
            for line in f.open():
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    break
                if isinstance(row, dict) and "id" in row:
                    _CAT_CACHE[row["id"]] = cat
    return _CAT_CACHE.get(tid, "?")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("score_dir")
    args = ap.parse_args()
    split = json.load((ROOT / "configs/bfcl_support_split.json").open())
    support = set(split["demand"]) | set(split["calibration"])
    sd = Path(args.score_dir)
    if not sd.is_absolute():
        sd = BFCL / args.score_dir
    stats = category_stats(sd, support)
    accs = {c: k / n for c, (k, n) in sorted(stats.items())}
    for c, a in accs.items():
        k, n = stats[c]
        print(f"  {c}: {a:.4f} ({k}/{n})")
    overall = sum(accs.values()) / len(accs)
    print(f"[bfclscore] categories={len(accs)} "
          f"macro_overall_excl_support={overall:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
