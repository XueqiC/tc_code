#!/usr/bin/env python3
"""Build a NEW certified v1.1 bank from the immutable C25 sealed archive (CPU)."""
import argparse
import json
from pathlib import Path

from bfas.rtd.bank_v11 import build_v11_bank


def main(argv=None):
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=root/'data/rtd/v1_bfcl_c25')
    parser.add_argument('--out', type=Path, default=root/'data/rtd/v1_1_bfcl')
    args = parser.parse_args(argv)
    print(json.dumps(build_v11_bank(args.source, args.out), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
