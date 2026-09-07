#!/usr/bin/env python3
"""AW-2 exposure/coverage controls (merged next plan, 2026-09-05, section 3).

Build with .venv/bin/python tools/aw2_expand_pool.py --mined
data/appworld_events/aw2_spread_more_k3.jsonl. The matched file is an exact byte
prefix of the output. New rows retain the miner's teacher response / student
_rejected orientation, agreements, and every dU sign, as aw2_matched_pools.py
does. Only overflow, other tasks, and repeated task/turn keys are excluded.
Per-task quotas are capped at the original counts; shortages are reported,
never filled by oversampling other tasks. No mining or model execution here.

--plan-probes prints task<TAB>max_probes<TAB>fractions for aw2_mine_more.sh,
using per-demo gap midpoints, including the gap before the first old probe.
"""

import argparse
from collections import Counter, defaultdict
from fractions import Fraction
import json
import math
from pathlib import Path
import random
import sys

from aw2_matched_pools import (
    DEFAULT_TOKENIZER, EVENT_DIR, ROWS_PER_STEP, load_tokenizer, probe_position,
    read_events, summarize_pool,
)

MATCHED = EVENT_DIR / "pool_aw2_spread_matched.jsonl"


def event_key(row):
    task = row.get("task_id")
    turn = row.get("turn_index")
    if turn is None:
        turn = row.get("_event_turn")
    if not isinstance(task, str) or not task or type(turn) is not int or turn < 0:
        raise ValueError("Each event needs a task_id and nonnegative turn_index/_event_turn")
    if row.get("_event_turn") is not None and row["_event_turn"] != turn:
        raise ValueError(f"Conflicting turn aliases for {task}: {turn}, {row['_event_turn']}")
    return task, turn


def validate_matched(rows):
    if not rows:
        raise ValueError("The matched pool is empty")
    seen = set()
    for row in rows:
        key = event_key(row)
        if row.get("_event_overflow"):
            raise ValueError(f"Matched pool contains overflow: {key}")
        if key in seen:
            raise ValueError(f"Matched pool contains duplicate task/turn: {key}")
        seen.add(key)
    return seen


def expand_pool(matched, mined, seed=0):
    old_keys = validate_matched(matched)
    targets = Counter(row["task_id"] for row in matched)
    groups, seen = defaultdict(list), set()
    excluded = dict.fromkeys(("overflow", "other_task", "existing_key", "duplicate_new_key"), 0)
    for row in mined:
        if row.get("_event_overflow"):
            excluded["overflow"] += 1
            continue
        if row.get("task_id") not in targets:
            excluded["other_task"] += 1
            continue
        key = event_key(row)
        if key in old_keys:
            excluded["existing_key"] += 1
        elif key in seen:
            excluded["duplicate_new_key"] += 1
        else:
            seen.add(key)
            groups[key[0]].append(row)
    rng, added, counts = random.Random(seed), [], {}
    for task, target in sorted(targets.items()):
        eligible = groups[task]
        n = min(target, len(eligible))
        added.extend(rng.sample(eligible, n))
        counts[task] = {
            "existing": target, "target_new": target, "eligible_new": len(eligible),
            "selected_new": n, "expanded": target + n, "shortfall": target - n,
            "residual_new_minus_target": n - target,
        }
    rng.shuffle(added)
    return list(matched) + added, counts, {
        "total_rows": len(mined), "excluded": excluded, "eligible_rows": len(seen),
        "selected_rows": len(added), "unused_eligible_rows": len(seen) - len(added),
    }


def midpoint_plan(rows):
    """Use actual old positions; verify rounding with the miner's ceil rule.

    Start with one midpoint in every old gap. If a short/uneven demo causes
    collisions, bisect the widest gaps that still contain unused states.
    Fractions remain midpoints; no old state is returned. Short demos may have
    fewer remaining states than the requested quota.
    """
    validate_matched(rows)
    groups = defaultdict(list)
    for row in rows:
        groups[row["task_id"]].append(row)
    plan = {}
    for task, events in sorted(groups.items()):
        lengths = {row.get("_demo_len") for row in events}
        if len(lengths) != 1:
            raise ValueError(f"Inconsistent demo lengths for {task}")
        length = lengths.pop()
        if type(length) is not int or length <= 0:
            raise ValueError(f"Missing positive _demo_len for {task}")
        old = {event_key(row)[1] for row in events}
        for row in events:
            turn = event_key(row)[1]
            pos, _ = probe_position(row)
            if turn >= length or pos is None or not math.isclose(pos, (turn + 1) / length):
                raise ValueError(f"Position/turn/demo length disagree for {task}")
        anchors = sorted({Fraction(0), Fraction(1),
                          *(Fraction(turn + 1, length) for turn in old)})
        gaps = list(zip(anchors, anchors[1:]))
        available = set(range(length)) - old
        selected = {}

        def bisect(left, right):
            mid = (left + right) / 2
            turn = math.ceil(mid * length) - 1
            if turn in available:
                pos = float(mid)
                # Preserve the exact rational midpoint's turn at floating-point boundaries.
                if math.ceil(pos * length) - 1 != turn:
                    pos = math.nextafter(pos, 0.0)
                if math.ceil(pos * length) - 1 != turn:
                    raise ValueError(f"Cannot represent probe midpoint for {task}")
                selected[turn] = pos
                available.remove(turn)
            return [(left, mid), (mid, right)]

        gaps = [child for left, right in gaps for child in bisect(left, right)]
        target = len(events)
        while len(selected) < target and available:
            # A state's fraction cell is (turn/L, (turn+1)/L]. Discard gaps
            # without an available cell before choosing the widest one.
            gaps = [(a, b) for a, b in gaps if any(
                a < Fraction(t + 1, length) and b > Fraction(t, length) for t in available)]
            if not gaps:
                break
            left, right = max(gaps, key=lambda gap: gap[1] - gap[0])
            gaps.remove((left, right))
            gaps.extend(bisect(left, right))
        turns = sorted(selected)
        if len(turns) > target:
            # Even coverage when a missing final old probe creates an extra gap.
            indices = ([len(turns) // 2] if target == 1 else
                       [round(i * (len(turns) - 1) / (target - 1)) for i in range(target)])
            turns = [turns[i] for i in indices]
        plan[task] = {
            "demo_length": length, "existing_turns": sorted(old),
            "existing_positions": [(t + 1) / length for t in sorted(old)],
            "probe_turns": turns, "probe_positions": [selected[t] for t in turns],
            "target_new": target, "planned_new": len(turns), "shortfall": target - len(turns),
        }
    return plan


def distribution(values, bins, missing=0):
    values = sorted(values)
    n = len(values)

    def quantile(q):
        if not n:
            return None
        index = (n - 1) * q
        lo, hi = math.floor(index), math.ceil(index)
        return values[lo] + (values[hi] - values[lo]) * (index - lo)

    histogram = []
    for i, (left, right) in enumerate(zip(bins, bins[1:])):
        count = sum((v >= left if i == 0 else v > left) and
                    (right is None or v <= right) for v in values)
        histogram.append({"lower": left, "upper": right, "count": count,
                          "fraction": count / n if n else None})
    return {
        "count": n, "missing": missing, "min": values[0] if n else None,
        "max": values[-1] if n else None, "mean": sum(values) / n if n else None,
        "quantiles": {name: quantile(q) for name, q in
                      (("p10", .1), ("p25", .25), ("p50", .5), ("p75", .75), ("p90", .9))},
        "histogram": histogram,
    }


def arm_summary(rows, tokenizer, epochs):
    stats = summarize_pool(rows, tokenizer)
    positions = [probe_position(row)[0] for row in rows]
    known_positions = [p for p in positions if p is not None]
    if any(not 0 <= p <= 1 for p in known_positions):
        raise ValueError("Probe positions must be fractions in [0, 1]")
    token_bins = [0, 64, 128, 256, 512, 1024, 2048, 4096, 8192, None]
    lengths = {}
    for name, field in (("response", "response"), ("rejected", "_rejected")):
        values = [len(tokenizer.encode(row[field], add_special_tokens=False))
                  for row in rows if isinstance(row.get(field), str)]
        lengths[name] = {**distribution(values, token_bins, len(rows) - len(values)),
                         "total_tokens": sum(values), "exposure_tokens": epochs * sum(values)}
    preference_rows = sum(isinstance(row.get("_rejected"), str) and bool(row["_rejected"])
                          for row in rows)
    return {
        **stats, "epochs": epochs, "row_exposures": epochs * len(rows),
        "steps_per_epoch": stats["steps"], "steps": epochs * stats["steps"],
        "preference_rows": preference_rows,
        "estimated_ddpo_steps": epochs * math.ceil(preference_rows / ROWS_PER_STEP),
        "stage_position_distribution": distribution(known_positions, [0, .25, .5, .75, 1],
                                                    len(rows) - len(known_positions)),
        "response_token_length_distribution": lengths,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matched", type=Path, default=MATCHED)
    parser.add_argument("--mined", type=Path, help="New miner JSONL (required except --plan-probes)")
    parser.add_argument("--out-dir", type=Path, default=EVENT_DIR)
    parser.add_argument("--report", type=Path, help="Default: <out-dir>/aw2_expand_report.json")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER, help="Locally cached tokenizer only")
    parser.add_argument("--plan-probes", action="store_true", help="Print midpoint mining TSV; no writes")
    args = parser.parse_args(argv)
    if not args.plan_probes and args.mined is None:
        parser.error("--mined is required to build an expanded pool")
    try:
        matched_bytes = args.matched.read_bytes()
        matched = [json.loads(line) for line in matched_bytes.splitlines() if line.strip()]
        if args.plan_probes:
            plan = midpoint_plan(matched)
            for task, item in plan.items():
                if item["planned_new"]:
                    positions = ",".join(repr(p) for p in item["probe_positions"])
                    print(f"{task}\t{item['planned_new']}\t{positions}")
            print(f"[aw2-plan] {len(plan)} tasks, "
                  f"{sum(p['planned_new'] for p in plan.values())} planned probes, "
                  f"{sum(p['shortfall'] for p in plan.values())} unavailable states", file=sys.stderr)
            return
        output = args.out_dir / "pool_aw2_spread_x2.jsonl"
        report_path = args.report or args.out_dir / "aw2_expand_report.json"
        paths = [args.matched, args.mined, output, report_path]
        if len({p.resolve() for p in paths}) != len(paths):
            raise ValueError("Input, output and report paths must be distinct")
        expanded, counts, input_counts = expand_pool(matched, read_events(args.mined), args.seed)
        if len(expanded) == len(matched):
            raise ValueError("No new independent non-overflow events for the matched tasks")
        tokenizer = load_tokenizer(args.tokenizer)
        added = expanded[len(matched):]
        shortfall = sum(c["shortfall"] for c in counts.values())
        a_stats = arm_summary(matched, tokenizer, 1)
        report = {
            "seed": args.seed, "sources": {"matched": str(args.matched), "mined": str(args.mined)},
            "tasks": sorted(counts), "n_tasks": len(counts), "per_task_counts": counts,
            "input_counts": input_counts,
            "residual_imbalance": {
                "target_new_rows": len(matched), "selected_new_rows": len(added),
                "shortfall_rows": shortfall, "l1_count_difference": shortfall,
                "tasks_with_shortfall": [t for t, c in counts.items() if c["shortfall"]],
                "rule": "min(existing per-task count, eligible new count); no cross-task top-up",
            },
            "legality_and_preferences": "As aw2_matched_pools: exclude overflow; retain whole rows, "
                "agreements and all dU signs; response=yT, _rejected=yS; no utility filtering or swapping",
            "independence_key": ["task_id", "turn_index (fallback _event_turn)"],
            "rows_per_step": ROWS_PER_STEP,
            "step_rule": "epochs * ceil(rows / 8); partial batch flushed each epoch (B: 40, not 39)",
            "token_lengths": {
                "tokenizer": args.tokenizer, "add_special_tokens": False,
                "scope": "Full responses, no chat template or truncation; not measured loss tokens",
                "histogram_intervals": "First bin includes both endpoints; later bins (lower, upper]",
                "quantiles": "Linear interpolation at (n - 1) * q",
            },
            "loss_normalization": {
                "response_log_probability": "sum over response tokens, not length-normalized",
                "pair_loss": "-logsigmoid(0.1 * (policy margin - frozen reference margin))",
                "batch_reduction": "mean over rows in each actual chunk (including partial chunk)",
                "event_weight": "1; AW_DDPO_WEIGHT=none", "length_matching": "diagnostic only; no length filtering",
            },
            "new_rows": arm_summary(added, tokenizer, 1),
            "arms": {
                "A": {"tag": "aw2_pref_spread_s0", "path": str(args.matched), **a_stats},
                "B": {"tag": "aw2_pref_spread_e2_s0", "path": str(args.matched),
                      **arm_summary(matched, tokenizer, 2)},
                "C": {"tag": "aw2_pref_spread_x2_s0", "path": str(output),
                      **arm_summary(expanded, tokenizer, 1)},
            },
        }
        # Finish validation/tokenization/serialization before creating output artifacts.
        report_text = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        suffix = "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in added)
        separator = b"" if matched_bytes.endswith(b"\n") else b"\n"
        output.parent.mkdir(parents=True, exist_ok=True)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(matched_bytes + separator + suffix.encode("utf-8"))
        report_path.write_text(report_text, encoding="utf-8")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"expanded: {len(matched)} + {len(added)} = {len(expanded)} rows; "
          f"{len(counts)} tasks; residual shortfall={shortfall} -> {output}")
    for arm, stats in report["arms"].items():
        print(f"{arm}: {stats['total_rows']} rows x {stats['epochs']} epoch(s), "
              f"{stats['steps']} estimated steps")
    print(f"report -> {report_path}")


if __name__ == "__main__":
    main()
