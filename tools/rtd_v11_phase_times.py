#!/usr/bin/env python3
"""Read-only, CPU-only inclusive phase timings from a run's compute.jsonl."""
import argparse
from collections import defaultdict
import json
from pathlib import Path


def phase_times(run_dir):
    phases = defaultdict(lambda: dict(n=0, wall_seconds=0., gpu_seconds=0.,
                                     peak_allocated_bytes=0, peak_reserved_bytes=0))
    # Do not instantiate ComputeJournal: its recovery path can repair torn files.
    with (Path(run_dir) / 'compute.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line)
            if row['kind'] != 'compute_end':
                continue
            phase = phases[row['operation']]
            phase['n'] += 1
            for key in ('wall_seconds', 'gpu_seconds'):
                phase[key] += row[key]
            for key in ('peak_allocated_bytes', 'peak_reserved_bytes'):
                phase[key] = max(phase[key], row[key])
    return dict(phases)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    args = parser.parse_args(argv)
    phases = phase_times(args.run_dir)
    width = max([len('operation'), *(len(name) for name in phases)])
    print(f"{'operation':<{width}} {'n':>5} {'wall_s':>12} {'gpu_s':>12} "
          f"{'peak_alloc_GiB':>15} {'peak_reserved_GiB':>18}")
    for operation, row in sorted(phases.items()):
        print(f"{operation:<{width}} {row['n']:5d} {row['wall_seconds']:12.3f} "
              f"{row['gpu_seconds']:12.3f} {row['peak_allocated_bytes']/2**30:15.3f} "
              f"{row['peak_reserved_bytes']/2**30:18.3f}")
    print('Inclusive times; nested operations overlap. All compute_end rows, including failed attempts.')
    print('GPU seconds are journaled wall time on CUDA, not kernel-active time. Peaks are maxima, in GiB.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
