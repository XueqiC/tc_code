#!/usr/bin/env python3
"""Check official FC generation coverage; stdout lists categories to resume.

Run with the BFCL venv's Python. Counts go to stderr (the campaign's generation
log); incomplete categories go to stdout as category<TAB>count details.
"""

import argparse
from collections import defaultdict
from pathlib import Path
import sys


def check_generation(leaderboard: Path, result_dir: Path, tag: str) -> None:
    sys.path.insert(0, str(leaderboard))
    from bfcl_eval.utils import (
        extract_test_category_from_id,
        get_directory_structure_by_category,
        get_file_name_by_category,
        is_format_sensitivity,
        load_dataset_entry,
        load_file,
        parse_test_category_argument,
    )

    # Match generate --test-category all for Qwen/Qwen3.5-4B-FC: the harness
    # deliberately skips format sensitivity for FC models. Use its data loader
    # for shared memory/web-search datasets and memory prerequisite expansion.
    for category in parse_test_category_argument(["all"]):
        if is_format_sensitivity(category):
            continue
        expected = defaultdict(list)
        for entry in load_dataset_entry(category):
            expected[extract_test_category_from_id(entry["id"])].append(entry["id"])
        failures = []
        for file_category, expected_ids in sorted(expected.items()):
            path = (result_dir / get_directory_structure_by_category(category)
                    / get_file_name_by_category(file_category, is_result_file=True))
            rows = load_file(path, use_lock=False) if path.is_file() else []
            actual_ids = [row["id"] for row in rows]
            missing = len(set(expected_ids) - set(actual_ids))
            duplicates = len(actual_ids) - len(set(actual_ids))
            unexpected = len(set(actual_ids) - set(expected_ids))
            counts = (f"{file_category}={len(rows)}/{len(expected_ids)} "
                      f"missing={missing} duplicates={duplicates} unexpected={unexpected}")
            print(f"[bfclstd] {tag} GEN COUNT {counts}", file=sys.stderr)
            if len(rows) != len(expected_ids) or missing or duplicates or unexpected:
                failures.append(counts)
        if failures:
            # A memory prereq file resumes via its parent CLI category, once.
            print(category + "\t" + "; ".join(failures))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("leaderboard", type=Path)
    parser.add_argument("result_dir", type=Path, help="Result directory for one model")
    parser.add_argument("tag")
    args = parser.parse_args()
    check_generation(args.leaderboard, args.result_dir, args.tag)


if __name__ == "__main__":
    main()
