#!/usr/bin/env python3
"""Read-only CPU check of stored evaluation hashes against each run's audit."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

from bfas.rtd.identity import audited_harness_hashes, saved_identities
from bfas.rtd.persistence import digest


def check(root, directory, round_number=1):
    root, directory = Path(root).resolve(), Path(directory).resolve()
    manifest = json.loads((directory/'manifest.json').read_text())
    result = json.loads((directory/f'evaluation-{round_number}.json').read_text())
    identities = saved_identities(root, directory, manifest)
    chain = audited_harness_hashes(manifest, identities)
    stored = result['identity']['evaluation_harness_hash']
    supplement = root/'configs/rtd/legacy_identities'/f'{digest(manifest)}.json'
    return dict(run=str(directory), round=round_number, stored_harness_hash=stored,
                current_harness_hash=identities['harness_hash'], audited_harness_hashes=chain,
                in_audited_chain=stored in chain,
                supplement_path=str(supplement) if supplement.is_file() else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dirs', type=Path, nargs='+')
    parser.add_argument('--round', type=int, default=1)
    args = parser.parse_args()
    rows = [check(ROOT, directory, args.round) for directory in args.run_dirs]
    print(json.dumps(rows, indent=2))
    return 0 if all(row['in_audited_chain'] for row in rows) else 1


if __name__ == '__main__':
    raise SystemExit(main())
