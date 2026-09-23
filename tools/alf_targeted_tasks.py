#!/usr/bin/env python3
"""Write a purchase task list by exact ALFWorld task-type prefixes, without calls."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]

from bfas.rtd.benchmarks.alfworld_support import validate_support

DEFAULT_SUPPORT = ROOT.parent / 'tc-alignment/data/rtd/v1_alfworld_k32_d0/public/support.json'
DEFAULT_TYPES = ('pick_cool_then_place_in_recep', 'pick_heat_then_place_in_recep',
                 'pick_two_obj_and_place')


def targeted_tasks(support, task_types=DEFAULT_TYPES):
    """Select from historical support, including tasks with no usable D0 demo."""
    support = validate_support(support)
    types = list(task_types)
    if (not types or any(not isinstance(t, str) or not t for t in types)
            or len(set(types)) != len(types)):
        raise ValueError('task types must be nonempty, unique strings')
    available = {tid.split('-', 1)[0] for tid in support['historical_task_ids']}
    unknown = set(types) - available
    if unknown:
        raise ValueError(f'task types outside frozen support: {sorted(unknown)}')
    selected = sorted(tid for tid in support['historical_task_ids'] if tid.split('-', 1)[0] in types)
    counts = Counter(tid.split('-', 1)[0] for tid in selected)
    return selected, {task_type: counts[task_type] for task_type in types}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--support', type=Path, default=DEFAULT_SUPPORT)
    parser.add_argument('--task-types', nargs='+', default=DEFAULT_TYPES,
                        help='exact type prefixes before the first hyphen (default: %(default)s)')
    parser.add_argument('--output', type=Path, required=True, help='new JSON list for --only-tasks')
    args = parser.parse_args(argv)
    try:
        selected, counts = targeted_tasks(json.loads(args.support.read_text()), args.task_types)
    except ValueError as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(selected, indent=2) + '\n')
    for task_type, count in counts.items():
        print(f'{task_type}: {count}')
    print(f'total: {len(selected)}; wrote {args.output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
