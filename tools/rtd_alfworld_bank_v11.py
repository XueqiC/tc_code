#!/usr/bin/env python3
"""Offline conversion of the verified C26-B ALFWorld bank; no APIs or weights."""
import argparse
import json
from pathlib import Path
from bfas.rtd.cli import load_config
from bfas.rtd.benchmarks.alfworld_bank_v11 import convert_alfworld_bank


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=Path('data/rtd/v1_alfworld_c26'))
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--ledger', type=Path)
    p.add_argument('--config', type=Path, default=Path('configs/rtd/v1_1_alfworld.yaml'))
    args = p.parse_args(argv)
    print(json.dumps(convert_alfworld_bank(args.source, args.out, config=load_config(args.config), ledger=args.ledger), indent=2))


if __name__ == '__main__':
    main()
