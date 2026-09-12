#!/usr/bin/env python3
"""First-round BFCL mechanism pipeline. No model/API work happens on import."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from bfas.mech_bfcl.common import STUDENT, bind_run, lock, read_json, setup_harness


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--run-dir", type=Path, default=ROOT / "results/mech_bfcl")
    common.add_argument("--splits", type=Path, default=ROOT / "configs/mech_bfcl_splits.json")
    common.add_argument("--seed", type=int, default=0)
    common.add_argument("--tokenizer", default=STUDENT)
    common.add_argument("--base-url", help="Already running vLLM /v1 endpoint")
    common.add_argument("--served-model", help="vLLM served ID; default base or mech-C/mech-D")
    common.add_argument("--bfcl-python", default=str(ROOT / "envs/bfcl/.venv/bin/python"))
    commands = p.add_subparsers(dest="command", required=True)
    split = commands.add_parser("splits", parents=[common])
    split.add_argument("--source-split", type=Path, default=ROOT / "configs/bfcl_support_split.json")
    commands.add_parser("rollout", parents=[common])
    commands.add_parser("diagnose", parents=[common])
    gen = commands.add_parser("generate", parents=[common])
    gen.add_argument("--arm", choices=["C", "D"], required=True)
    train = commands.add_parser("train", parents=[common])
    train.add_argument("--arm", choices=["C", "D"], required=True)
    train.add_argument("--model-path", default=STUDENT)
    train.add_argument("--tokens-per-step", type=int, default=512)
    train.add_argument("--position-chunk", type=int, default=32)
    evaluate = commands.add_parser("evaluate", parents=[common])
    evaluate.add_argument("--arm", choices=["base", "C", "D"], required=True)
    evaluate.add_argument("--repeat", choices=["main", "repeat"], default="main")
    commands.add_parser("report", parents=[common])
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    args.run_dir = args.run_dir.resolve()
    setup_harness(args.run_dir / "runtime")
    if args.command == "splits":
        from bfas.mech_bfcl.splits import run
        run(args)
        return 0
    splits = read_json(args.splits)
    if args.seed != splits["seed"]:
        raise ValueError("All stages must use the frozen split seed")
    args.served_model = args.served_model or (
        "mech-" + args.arm if args.command == "evaluate" and args.arm != "base" else STUDENT)
    with lock(args.run_dir / "pipeline.lock"):
        bind_run(args.run_dir, splits)
        if args.command == "train":
            from bfas.mech_bfcl.training import train
            train(args, splits)
        elif args.command == "report":
            from bfas.mech_bfcl.stats import report
            report(args, splits)
        else:
            from bfas.mech_bfcl import pipeline
            getattr(pipeline, args.command)(args, splits)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
