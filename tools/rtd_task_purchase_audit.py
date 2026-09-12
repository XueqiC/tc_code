#!/usr/bin/env python3
"""Offline frozen-order purchase audit, optionally rebuilding public accounting."""
import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

# Mask accelerators and remote tokenizer resolution before project imports.
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'


def main(argv=None):
    from bfas.rtd.preflight import acquisition_preflight
    from bfas.rtd.task_packages import rebuild_task_accounting
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bank', type=Path, action='append', required=True)
    parser.add_argument('--budgets', type=int, nargs='+', default=[7500, 15000, 30000, 60000])
    parser.add_argument('--rebuild-accounting', action='store_true',
        help='rebuild from sealed ledger rows only after support/folds pass a byte-identity gate')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args(argv)
    if any(b < 0 for b in args.budgets) or sorted(set(args.budgets)) != args.budgets:
        parser.error('budgets must be increasing nonnegative integers')
    reports = []
    for bank in args.bank:
        rebuild = rebuild_task_accounting(bank) if args.rebuild_accounting else None
        rows = json.loads((bank/'public/requests.json').read_text())
        result = acquisition_preflight(dict(bank_path=str(bank), budget_ceilings=args.budgets),
            SimpleNamespace(parents={r['parent_hash'] for r in rows}))
        reports.append(dict(bank=str(bank), rebuild=rebuild, acquisition=result))
        for row in result['checkpoints']:
            print(f"{bank.name} B={row['budget_tokens']}: tasks={row['tasks_purchased']}, "
                  f"usable={row['usable_packages']}, attempts={row['attempts_purchased']}, "
                  f"tokens={row['spent_tokens']}")
    if args.out:
        args.out.write_text(json.dumps(reports, indent=2) + '\n')
    return reports


if __name__ == '__main__':
    main()
