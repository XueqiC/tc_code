#!/usr/bin/env python3
"""Evaluate Gemma on tau2 test tasks using an existing student endpoint."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bfas.tau2_eval import main

if __name__ == "__main__":
    main()
