#!/usr/bin/env python3
"""Official BFCL CLI with this repository's additional model registrations."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "envs/bfcl/gorilla/berkeley-function-call-leaderboard"))


def main():
    from bfas.bfcl_azure import register_azure_model

    register_azure_model()
    from bfcl_eval.__main__ import cli

    cli()


if __name__ == "__main__":
    main()
