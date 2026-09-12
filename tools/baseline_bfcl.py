#!/usr/bin/env python3
"""Isolated official BFCL CLI worker with an optional local Kang student hook."""
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/"src"), str(ROOT/"envs/bfcl/gorilla/berkeley-function-call-leaderboard")]


def main():
    if os.environ.get("BASELINE_KANG_AUDIT") and "generate" in sys.argv:
        from bfas.rtd.baselines.paper_kang import install_bfcl_kang
        install_bfcl_kang(os.environ["BASELINE_KANG_AUDIT"])
    from bfcl_eval.__main__ import cli
    cli()


if __name__ == "__main__":
    main()
