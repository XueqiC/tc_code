#!/usr/bin/env python3
"""Isolated C27 entry; existing RTD/evaluation code is imported read-only."""
import sys

# Keep imports read-only even when invoked from a live checkout.
sys.dont_write_bytecode = True

from bfas.rtd.baselines.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
