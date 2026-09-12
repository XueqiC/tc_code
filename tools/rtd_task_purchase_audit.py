#!/usr/bin/env python3
"""Offline attempt-purchase audit against the read-only paper baseline reader."""
import argparse
import json
import os
import subprocess
import sys
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
    parser.add_argument('--baseline-reader', type=Path,
        help='read-only sibling src/bfas/rtd/baselines/paper_data.py to execute for parity')
    args = parser.parse_args(argv)
    if any(b < 0 for b in args.budgets) or sorted(set(args.budgets)) != args.budgets:
        parser.error('budgets must be increasing nonnegative integers')
    reports = []
    for bank in args.bank:
        rebuild = rebuild_task_accounting(bank) if args.rebuild_accounting else None
        rows = json.loads((bank/'public/requests.json').read_text())
        result = acquisition_preflight(dict(bank_path=str(bank), budget_ceilings=args.budgets),
            SimpleNamespace(parents={r['parent_hash'] for r in rows}))
        report = dict(bank=str(bank.resolve()), rebuild=rebuild, acquisition=result)
        if args.baseline_reader:
            reader = args.baseline_reader.resolve()
            from bfas.rtd.persistence import file_hash
            benchmark = json.loads((bank/'public/cap_certificate.json').read_text())['core']['benchmark']
            # Isolate imports so this really runs the sibling code path. -B and
            # cwd=/tmp keep the baseline worktree entirely read-only.
            code = '''import json, sys
from bfas.rtd.baselines.paper_data import load_purchased
receipts = []
for budget in json.loads(sys.argv[3]):
    purchase, rows = load_purchased(sys.argv[1], sys.argv[2], budget_tokens=budget)
    receipts.append(purchase)
print(json.dumps(receipts))
'''
            run = subprocess.run([sys.executable, '-B', '-c', code, str(bank.resolve()),
                benchmark, json.dumps(args.budgets)], cwd='/tmp', check=True,
                env=dict(os.environ, PYTHONPATH=str(reader.parents[3])), capture_output=True, text=True)
            baselines = json.loads(run.stdout)
            for actual, expected in zip(result['checkpoints'], baselines, strict=True):
                assert actual['purchased_query_ids'] == expected['purchased_package_ids']
                assert actual['usable_query_ids'] == [c['package_id'] for c in expected['charges'] if c['usable']]
                assert actual['usable_packages'] == expected['purchased_usable_packages']
                assert actual['spent_tokens'] == expected['teacher_tokens_charged']
                assert actual['next_query_id'] == (expected['blocked_next'] or {}).get('package_id')
                assert result['order_sha256'] == expected['purchase_order_hash']
            report.update(baseline_reader=str(reader), baseline_reader_sha256=file_hash(reader),
                baseline=baselines, exact_attempt_ids_usable_counts_costs_and_order_equal=True)
        reports.append(report)
        for row in result['checkpoints']:
            print(f"{bank.name} B={row['budget_tokens']}: tasks={row['tasks_purchased']}, "
                  f"usable={row['usable_packages']}, attempts={row['attempts_purchased']}, "
                  f"tokens={row['spent_tokens']}")
    if args.out:
        args.out.write_text(json.dumps(reports, indent=2) + '\n')
    return reports


if __name__ == '__main__':
    main()
