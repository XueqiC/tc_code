#!/usr/bin/env python3
"""RTD v1.0.7 command entry point; identity audits need no visible GPU."""
import argparse
import json
from pathlib import Path
import os
import sys

# CUDA's default FASTEST_FIRST ordinal can differ from nvidia-smi on rai's
# mixed cards. Set the default before importing torch, honoring an explicit
# parent ordering and preserving CUDA_VISIBLE_DEVICES (including UUIDs).
os.environ.setdefault('CUDA_DEVICE_ORDER', 'PCI_BUS_ID')

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]

def pin_local_resume(argv):
    """Default local resumes/evaluations to the run's UUID before importing torch.

    Explicit visibility remains an operator choice; SLURM and training workers
    inherit their allocation/coordinator binding without pinning an old device.
    """
    if (not argv or argv[0] not in {'resume', 'evaluate'} or '--help' in argv
            or '--training-worker' in argv or os.environ.get('SLURM_JOB_ID')
            or os.environ.get('SLURM_CLUSTER_NAME') or 'CUDA_VISIBLE_DEVICES' in os.environ):
        return
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--run-dir', type=Path)
    args, _ = parser.parse_known_args(argv[1:])
    if args.run_dir is not None:
        hardware = json.loads((args.run_dir/'manifest.json').read_text())['hardware']
        uuid = hardware.get('metadata', hardware)['uuid'].removeprefix('GPU-')
        if not uuid or uuid == 'unknown':
            raise ValueError('local launch requires a recorded GPU UUID or explicit CUDA_VISIBLE_DEVICES')
        os.environ['CUDA_VISIBLE_DEVICES'] = 'GPU-' + uuid

if __name__ == '__main__':
    pin_local_resume(sys.argv[1:])
    from bfas.rtd.cli import main
    raise SystemExit(main())
