#!/usr/bin/env python3
"""AW-1 pools from the AppWorld event file (unified-CRCD brief §9, AW-1).
Rows already carry prompt / response (y^T) / _rejected (y^S) / _event_dU.
Pools: all (every retained event), conseq (ΔU != 0), pos (ΔU > 0, ablation only).
Overflow rows are excluded by default. No weighting.
"""
import argparse
import collections
import json
from pathlib import Path


def is_agreement(row):
    if row.get("_event_overflow") or row.get("_event_dU") != 0:
        return False
    # The miner records action equality as an agree rate, including API-effect
    # equality when the code differs. `teacher` is a model label, not an action.
    return bool(
        row.get("_event_agree")
        or row.get("_student_agree_rate") == 1
        or any(
            row.get(teacher) and row[teacher] == row.get(student)
            for teacher, student in (
                ("_event_good", "_event_bad"), ("response", "_rejected")
            )
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("src", nargs="?", type=Path,
                        default=Path("data/appworld_events/aw1_base_k3.jsonl"))
    parser.add_argument("--prefix", default="aw1", help="Output pool prefix (default: aw1)")
    parser.add_argument("--exclude-overflow", action=argparse.BooleanOptionalAction,
                        default=True, help="Drop overflow rows (default: enabled)")
    args = parser.parse_args()

    out_dir = Path("data/appworld_events")
    with args.src.open() as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    if args.exclude_overflow:
        rows = [r for r in rows if not r.get("_event_overflow")]
    pools = {
        "all": rows,
        "conseq": [r for r in rows if r.get("_event_dU")],
        "pos": [r for r in rows if (r.get("_event_dU") or 0) > 0],
    }
    for name, rs in pools.items():
        p = out_dir / f"pool_{args.prefix}_{name}.jsonl"
        with p.open("w") as fh:
            for r in rs:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(name, len(rs), "tasks", len({r["task_id"] for r in rs}), "->", p)
    print("dU hist", sorted(collections.Counter(round(r.get("_event_dU") or 0, 2) for r in rows).items()))
    agreements = sum(is_agreement(r) for r in rows)
    overflows = sum(bool(r.get("_event_overflow")) for r in rows)
    print("agreements", agreements, "divergences", len(rows) - agreements - overflows,
          "overflows", overflows)


if __name__ == "__main__":
    main()
