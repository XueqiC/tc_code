#!/usr/bin/env python3
"""Build an offline v1.1 bank from paid teacher pools and their ledger."""
import argparse
import json
from pathlib import Path
from bfas.rtd.cli import load_config

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--benchmark', choices=['bfcl', 'alfworld', 'webshop'], required=True)
    p.add_argument('--pool', type=Path, required=True)
    p.add_argument('--ledger', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--config', type=Path)
    args = p.parse_args(argv)
    name = 'bfcl_gemma4' if args.benchmark == 'bfcl' else args.benchmark
    config = load_config(args.config or ROOT/f'configs/rtd/v1_1_{name}.yaml')
    if config['benchmark'] != args.benchmark:
        p.error('config benchmark differs from --benchmark')
    if args.benchmark == 'alfworld':
        from bfas.rtd.benchmarks.alfworld_bank_v11 import convert_alfworld_bank
        result = convert_alfworld_bank(args.pool, args.out, config=config, ledger=args.ledger)
    elif args.benchmark == 'bfcl':
        from bfas.rtd.benchmarks.bfcl_bank_v11 import build_bfcl_pool_bank
        result = build_bfcl_pool_bank(ROOT, args.out, pool=args.pool, ledger=args.ledger, config=config)
    else:
        from bfas.rtd.benchmarks.webshop_bank import build_webshop_bank
        result = build_webshop_bank(ROOT, args.out, pool=args.pool, ledger=args.ledger, config=config)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
