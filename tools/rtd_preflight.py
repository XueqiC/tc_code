#!/usr/bin/env python3
"""Run RTD startup checks on CPU, using only files already on disk."""
import os
from pathlib import Path
import sys

# Set before importing torch/transformers. Never download or probe a GPU.
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT)]

if __name__ == '__main__':
    from bfas.rtd.preflight import main
    raise SystemExit(main())
