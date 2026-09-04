#!/usr/bin/env python3
"""Selective-specialization routing evaluation (mechanism layer 2).

Per-category certificate on the calibration split S_c decides, for each
BFCL category, whether deployment serves the specialized student or the
base student. The routed system's official score is then computed from
the two arms' already-measured per-task outcomes (both measured under
the identical official harness), using the official category weighting.

Decision rule (conservative, no magic numbers): a category routes to
the specialized student only if its S_c success count is strictly
greater than the base student's on the same S_c tasks; ties and
insufficient evidence default to base. This makes "never below base on
certified categories" hold by construction up to S_c sampling error.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# official BFCL v4 top-level weighting: non-live, live, irrelevance,
# multi-turn, agentic (web_search + memory)
GROUP_WEIGHTS = {"non_live": 10, "live": 10, "irrelevance": 10,
                 "multi_turn": 30, "agentic": 40}


def category_group(cat: str) -> str:
    if cat.startswith("multi_turn"):
        return "multi_turn"
    if cat.startswith(("web_search", "memory")):
        return "agentic"
    if "irrelevance" in cat:
        return "irrelevance"
    if cat.startswith("live"):
        return "live"
    return "non_live"


def load_verdicts(tag: str) -> dict[str, dict[str, bool]]:
    """category -> {task_id: passed} from a std-campaign scoredir.

    Score files list FAILED entries; passed = in result but not in score
    fail list. We reconstruct from score files' accuracy blocks instead:
    each *_score.json line 1 is a summary; subsequent lines are failures.
    """
    scoredir = ROOT / "results/bfcl_std" / tag / "scoredir"
    out: dict[str, dict[str, bool]] = defaultdict(dict)
    for path in scoredir.rglob("BFCL_v4_*_score.json"):
        cat = path.name.replace("BFCL_v4_", "").replace("_score.json", "")
        lines = [l for l in path.read_text().splitlines() if l.strip()]
        if not lines:
            continue
        failures = set()
        for line in lines[1:]:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            tid = row.get("id")
            if tid is not None:
                failures.add(str(tid))
        # total ids come from the data file
        data = ROOT / "envs/bfcl/gorilla/berkeley-function-call-leaderboard/bfcl_eval/data" / f"BFCL_v4_{cat}.json"
        if not data.exists():
            continue
        for line in data.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                break
            tid = str(row.get("id"))
            out[cat][tid] = tid not in failures
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default="base")
    ap.add_argument("--spec", required=True)
    ap.add_argument("--split", default="configs/bfcl_support_split.json")
    args = ap.parse_args()

    split = json.load((ROOT / args.split).open())
    calib = set(split.get("calib") or split.get("calibration") or [])
    if not calib:
        raise SystemExit("support split has no calibration ids")

    vb = load_verdicts(args.base)
    vs = load_verdicts(args.spec)

    routing: dict[str, str] = {}
    for cat in sorted(set(vb) | set(vs)):
        cal_ids = [t for t in vb.get(cat, {}) if t in calib]
        b = sum(vb[cat].get(t, False) for t in cal_ids)
        s = sum(vs.get(cat, {}).get(t, False) for t in cal_ids)
        routing[cat] = "spec" if s > b else "base"

    group_scores: dict[str, list[float]] = defaultdict(list)
    routed_from_spec = 0
    for cat in sorted(set(vb) | set(vs)):
        src = vs if routing.get(cat) == "spec" else vb
        verdicts = src.get(cat) or vb.get(cat, {})
        if not verdicts:
            continue
        acc = sum(verdicts.values()) / len(verdicts)
        group_scores[category_group(cat)].append(acc)
        if routing.get(cat) == "spec":
            routed_from_spec += 1

    total_w = sum(GROUP_WEIGHTS.values())
    overall = sum(
        GROUP_WEIGHTS[g] * (sum(v) / len(v))
        for g, v in group_scores.items() if v
    ) / total_w
    print(f"[route] spec={args.spec} categories routed to spec: "
          f"{routed_from_spec}/{len(routing)}")
    for cat, dst in sorted(routing.items()):
        if dst == "spec":
            print(f"  spec <- {cat}")
    print(f"[route] routed overall = {overall:.4f} "
          f"(groups: " + ", ".join(
              f"{g} {sum(v)/len(v):.3f}" for g, v in sorted(group_scores.items())) + ")")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
