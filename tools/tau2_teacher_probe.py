#!/usr/bin/env python3
"""Probe Luna as the tau2 agent, with Luna also simulating the user."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bfas.tau2_eval import main

if __name__ == "__main__":
    main(teacher=True)
