#!/usr/bin/env python3
"""C26-D standalone campaign; training-run integration is a later stage.

Prepare is CPU-only and freezes 140 valid_seen IDs, file hashes and a supplied
RTD hardware-class JSON. Run uses a new campaign root and loads a GPU model only
when tasks remain. Validate/reuse do no GPU work. All model files must be local.

Example (in an isolated checkout):
  PYTHONPATH=src:. .venv/bin/python tools/rtd_alfworld_evaluate.py prepare \
    --model /local/base --checkpoint /isolated/run/round-1 \
    --data-root envs/alfworld/data/json_2.1.1 --hardware-json /tmp/hardware.json \
    --out /tmp/alfworld-binding.json
  PYTHONPATH=src:. .venv/bin/python tools/rtd_alfworld_evaluate.py run \
    --binding /tmp/alfworld-binding.json --output-root /new/alfworld-campaigns --tag r1

Use --run-directory and --round with the full training --config to additionally
verify RTD's checkpoint.json/round_state.pt and original training manifest.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from bfas.rtd.benchmarks.alfworld_evaluation import comparison_anchors, evaluate, tag_lock_path, validate_evaluation
from bfas.rtd.benchmarks.alfworld_identity import (
    OFFICIAL_CONFIG, campaign_identity, guard_manifest, make_manifest,
)
from bfas.rtd.persistence import manifest_digest, atomic_json, digest


def read(path):
    return json.loads(Path(path).read_text())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="CPU-only freeze of evaluation identity")
    prepare.add_argument("--model", required=True, type=Path)
    prepare.add_argument("--tokenizer", type=Path)
    prepare.add_argument("--data-root", required=True, type=Path)
    prepare.add_argument("--environment-root", type=Path)
    prepare.add_argument("--hardware-json", required=True, type=Path)
    prepare.add_argument("--config", type=Path, help="full ALFWorld training config JSON")
    prepare.add_argument("--checkpoint", type=Path, help="PEFT adapter or round-N/lora parent; omit for merged/base")
    prepare.add_argument("--run-directory", type=Path)
    prepare.add_argument("--round", type=int, choices=(1, 2, 3))
    prepare.add_argument("--out", required=True, type=Path)
    for name in ("run", "validate"):
        command = commands.add_parser(name)
        command.add_argument("--binding", required=True, type=Path)
        command.add_argument("--output-root", required=True, type=Path)
        command.add_argument("--tag", required=True)
        command.add_argument("--supplement", type=Path, help="explicit manifest-bound ALFWorld continuity audit")
        if name == "run":
            command.add_argument("--base-evaluation", type=Path)
            command.add_argument("--lock-timeout", type=float)
    commands.add_parser("anchors", help="read-only historical CE/base registry and local path availability")
    args = parser.parse_args(argv)
    if args.command == "anchors":
        result = comparison_anchors(args.root)
    elif args.command == "prepare":
        if args.out.exists():
            parser.error("binding output already exists; choose a new path")
        if (args.run_directory is None) != (args.round is None):
            parser.error("--run-directory and --round must be provided together")
        if args.run_directory and args.checkpoint:
            parser.error("--run-directory resolves its checkpoint; omit --checkpoint")
        result = make_manifest(args.root, read(args.config) if args.config else OFFICIAL_CONFIG,
            data_root=args.data_root, model_path=args.model, tokenizer_path=args.tokenizer,
            checkpoint=args.checkpoint, hardware=read(args.hardware_json),
            run_directory=args.run_directory, round_number=args.round, environment_root=args.environment_root)
        atomic_json(args.out, result)
    else:
        tag_lock_path(args.root, args.tag)  # validate tag syntax without touching a lock
        manifest = read(args.binding)
        supplement = read(args.supplement) if args.supplement else None
        if args.command == "run":
            result = evaluate(args.root, manifest, output_root=args.output_root, tag=args.tag,
                hardware=manifest["hardware"], supplement=supplement,
                base_evaluation=args.base_evaluation, lock_timeout=args.lock_timeout)
        else:
            current, hashes = guard_manifest(args.root, manifest, hardware=manifest["hardware"], supplement=supplement)
            result = validate_evaluation(args.output_root / args.tag, campaign_identity(current),
                current["evaluation_harness"]["expected"], audited_hashes=hashes, manifest_hash=manifest_digest(manifest))
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
