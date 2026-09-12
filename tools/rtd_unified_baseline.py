#!/usr/bin/env python3
"""D0 Table-1 trainer entry; AW_DISTILL=rtd_sft_kl, shared RTD run/smoke/resume."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT)]
os.environ.setdefault('CUDA_DEVICE_ORDER', 'PCI_BUS_ID')


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if os.environ.get('AW_DISTILL', 'rtd_sft_kl') != 'rtd_sft_kl':
        raise ValueError('D0 requires AW_DISTILL=rtd_sft_kl')
    if '--arm' in argv:
        raise ValueError('D0 entry fixes --arm D0; use rtd_experiment.py for other arms')
    from tools.rtd_experiment import pin_local_resume
    pin_local_resume(argv)
    from bfas.rtd.cli import main as run
    return run(argv+['--arm', 'D0'])


if __name__ == '__main__':
    raise SystemExit(main())
