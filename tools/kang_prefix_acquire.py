#!/usr/bin/env python3
"""Read-only FTP budget dry run. This CLI has no execution/provider path."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]


def main(argv=None):
    from bfas.rtd.baselines.paper_acquisition import KangAcquisitionConfig, prefix_dry_run
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--questions", type=Path, help="JSON list of training question strings")
    source.add_argument("--attempt-manifest", type=Path,
                        help="count-only audit from recorded charges; never substitutes task IDs for questions online")
    parser.add_argument("--ledger", required=True, type=Path, help="real JSONL ledger or purchase manifest")
    parser.add_argument("--initial-budget", type=int, help="original authorization for a JSONL ledger")
    parser.add_argument("--cot-max-output-tokens", required=True, type=int)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--kang-mode", default="kang_first_thought_prefix",
                        choices=("kang_first_thought_prefix", "kang_action_list_summary"))
    parser.add_argument("--dry-run", action="store_true", help="always enabled; no teacher is constructed")
    args = parser.parse_args(argv)
    if args.questions:
        questions = json.loads(args.questions.read_text())
        if not isinstance(questions, list):
            parser.error("questions must be a JSON list")
    else:
        charges = json.loads(args.attempt_manifest.read_text())["charges"]
        questions = [c["task_id"] for c in charges]
    config = KangAcquisitionConfig(args.run_name, args.cot_max_output_tokens, 1, args.kang_mode)
    report = prefix_dry_run(questions, config, ledger_path=args.ledger, budget=args.initial_budget)
    report["count_only_from_attempt_records"] = args.attempt_manifest is not None
    if args.attempt_manifest:
        report["unique_task_ids"] = report.pop("unique_questions")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
