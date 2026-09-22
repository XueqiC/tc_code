#!/usr/bin/env python3
"""Recompute a frozen selection from per-turn receipts, entirely offline."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT)]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scores', type=Path, required=True, help='existing selection scores/ directory')
    parser.add_argument('--output', type=Path, required=True, help='new selection directory')
    parser.add_argument('--statistic', choices=('token_mean', 'turn_mean'), default='turn_mean')
    args = parser.parse_args(argv)
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    from bfas.rtd.baselines.alfworld_cli import safe_output
    from bfas.rtd.baselines.alfworld_selection import recompute_selection
    safe_output(args.output, [args.scores.resolve().parent])
    try:
        result = recompute_selection(args.scores, args.output, statistic=args.statistic)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(dict(output=str(args.output/'selection.json'), statistic=result['statistic'],
                          model_calls=0, selected_tasks=len({r['task_id'] for r in result['training_rows']}))))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
