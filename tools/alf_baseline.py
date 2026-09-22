#!/usr/bin/env python3
"""Select SmartAD candidates, train K=32 baselines, or report their ledger costs."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT)]

from bfas.rtd.baselines.alfworld_cli import main

if __name__ == '__main__':
    raise SystemExit(main())
