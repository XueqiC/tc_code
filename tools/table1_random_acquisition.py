#!/usr/bin/env python3
"""Freeze a strict random prefix using only public package IDs and costs.

PYTHONPATH=src:. .venv/bin/python tools/table1_random_acquisition.py
Seeds 0/1/2 are TRAINING seeds; all use acquisition order seed 0 by default.
"""
from __future__ import annotations

import argparse
import random

from tools.table1_common import BENCHMARKS, DEFAULT_OUT, digest, read_json, write_json


def acquire(public, cap, training_seed=0, order_seed=0):
    if type(cap) is not int or cap < 0:
        raise ValueError('cap must be a nonnegative integer')
    if set(public) != {'schema', 'benchmark', 'packages'}:
        raise ValueError('acquisition accepts only the public projection')
    packages = public['packages']
    for p in packages:
        if set(p) != {'package_id', 'recorded_cost'}:
            raise ValueError('private fields in acquisition input')
        if type(p['recorded_cost']) is not int or p['recorded_cost'] < 0:
            raise ValueError('invalid public recorded cost')
    by_id = {p['package_id']: p['recorded_cost'] for p in packages}
    if len(by_id) != len(packages):
        raise ValueError('duplicate package ID')
    order = sorted(by_id)
    random.Random(order_seed).shuffle(order)
    purchases, spent, next_id = [], 0, None
    for q in order:
        cost = by_id[q]
        if spent + cost > cap:
            next_id = q
            break  # Do not skip expensive packages or seek a cheaper successor.
        spent += cost
        purchases.append(dict(package_id=q, recorded_cost=cost, cumulative_cost=spent))
    return dict(schema='table1-acquisition-v1', benchmark=public['benchmark'], cap=cap,
                training_seed=training_seed, order_seed=order_seed,
                order_policy='shared; training seeds differ only in training',
                public_sha256=digest(public), frozen_order=order, purchases=purchases,
                purchased_ids=[p['package_id'] for p in purchases],
                packages_purchased=len(purchases), C_m=spent, remaining_budget=cap-spent,
                next_package_id=next_id, next_package_cost=by_id.get(next_id),
                stop_reason='next_package_exceeds_cap' if next_id else 'inventory_exhausted',
                tasks_covered=None, positives=None,
                journal=[dict(stage='acquisition', allowed_fields=['package_id', 'recorded_cost'],
                              sealed_reader_calls=0, unpurchased_content_read=False,
                              verification_or_response_features_read=False,
                              note='Privileged inventory ran separately; this claim covers acquisition only.')])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--audit-root', type=str, default=str(DEFAULT_OUT))
    ap.add_argument('--benchmark', choices=BENCHMARKS)
    ap.add_argument('--cap', type=int, action='append')
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--order-seed', type=int, default=0)
    args = ap.parse_args()
    from pathlib import Path
    for benchmark in [args.benchmark] if args.benchmark else BENCHMARKS:
        directory = Path(args.audit_root) / benchmark
        public = read_json(directory / 'public.json')
        caps = args.cap or read_json(directory / 'protocol.json')['caps']
        for cap in caps:
            for seed in args.seeds:
                result = acquire(public, cap, seed, args.order_seed)
                path = directory / f'acquired_B{cap}_seed{seed}.json'
                if path.exists():
                    previous = read_json(path)
                    for key in ('public_sha256', 'order_seed', 'frozen_order', 'purchases'):
                        if previous[key] != result[key]:
                            raise ValueError('refusing to change a frozen acquisition; use a fresh audit directory')
                write_json(path, result)
                print(f'{benchmark} B={cap} seed={seed}: {result["packages_purchased"]} packages, '
                      f'C_m={result["C_m"]}, remaining={result["remaining_budget"]}')


if __name__ == '__main__':
    main()
