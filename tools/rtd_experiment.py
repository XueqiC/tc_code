#!/usr/bin/env python3
"""RTD v1.0.1 command entry point; CPU audit is safe without a visible GPU."""
from pathlib import Path
import os
import sys

# CUDA's default FASTEST_FIRST ordinal can differ from nvidia-smi on rai's
# mixed cards. Set the default before importing torch, honoring an explicit
# parent ordering and preserving CUDA_VISIBLE_DEVICES (including UUIDs).
os.environ.setdefault('CUDA_DEVICE_ORDER', 'PCI_BUS_ID')

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]

from bfas.rtd.cli import main

if __name__ == '__main__':
    raise SystemExit(main())
