#!/usr/bin/env python3
"""Print official BFCL axes, reusing the printer from bfcl_fast_eval.sh."""

import argparse
import csv
from pathlib import Path


AXES = ["Overall Acc", "Non-Live AST Acc", "Live Acc", "Multi Turn Acc",
        "Memory Acc", "Web Search Acc", "Relevance Detection",
        "Irrelevance Detection"]
TABLE_AXES = [
    ("Non-Live", "Non-Live AST Acc"),
    ("Live", "Live Acc"),
    ("Multi-turn", "Multi Turn Acc"),
    ("Memory", "Memory Acc"),
    ("Irrelevance", "Irrelevance Detection"),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("tag")
    parser.add_argument("--prefix", default="[bfclstd]")
    parser.add_argument("--proxy", action="store_true")
    args = parser.parse_args()
    if not args.csv_path.exists():
        print(f"{args.prefix} {args.tag} NO SCORE")
        return 0 if args.proxy else 1
    with args.csv_path.open(newline="") as stream:
        row = next(csv.DictReader(stream))
    if args.proxy:
        # Preserve the existing proxy output, including its PROXY label.
        parts = []
        for axis in AXES:
            value = row.get(axis, "N/A")
            parts.append(f"{axis.replace(' Acc', '').replace(' Detection', '')}={value}")
        print(f"{args.prefix} {args.tag} PROXY " + "  ".join(parts))
    else:
        print(f"{args.prefix} {args.tag} OVERALL={row['Overall Acc']}")
        print("| Tag | " + " | ".join(label for label, _ in TABLE_AXES) + " |")
        print("| --- | " + " | ".join("---" for _ in TABLE_AXES) + " |")
        print(f"| {args.tag} | " + " | ".join(row.get(axis, "N/A")
                                               for _, axis in TABLE_AXES) + " |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
