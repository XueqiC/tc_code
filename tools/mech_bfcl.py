#!/usr/bin/env python3
"""BFCL mechanism validation, including R2 coverage expansion."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from bfas.mech_bfcl.common import (STUDENT, bind_run, lock, read_json, resolved_train_seed,
                                  setup_harness, variant_name)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--run-dir", type=Path, default=ROOT / "results/mech_bfcl")
    common.add_argument("--splits", type=Path, default=ROOT / "configs/mech_bfcl_splits.json")
    common.add_argument("--seed", type=int, default=0)
    common.add_argument("--base-url", help="Already running vLLM /v1 endpoint")
    common.add_argument("--served-model", help="vLLM served ID; defaults to base or mech-<arm>[-s<train-seed>]")
    common.add_argument("--bfcl-python", default=str(ROOT / "envs/bfcl/.venv/bin/python"))
    commands = p.add_subparsers(dest="command", required=True)
    split = commands.add_parser("splits", parents=[common])
    split.add_argument("--source-split", type=Path, default=ROOT / "configs/bfcl_support_split.json")
    commands.add_parser("rollout", parents=[common])
    commands.add_parser("diagnose", parents=[common])
    prepare = commands.add_parser("prepare-r2", parents=[common])
    prepare.add_argument("--round1-run-dir", type=Path, required=True)
    confirm = commands.add_parser("confirm", parents=[common])
    confirm.add_argument("--target", type=positive_int, default=24)
    gen = commands.add_parser("generate", parents=[common])
    gen.add_argument("--arm", choices=["C", "D"], required=True)
    gen.add_argument("--target-exercises", "--target", type=positive_int, default=64,
                     help="Target number of distinct validated exercises per arm (default: 64)")
    gen.add_argument("--max-output-tokens", type=positive_int, default=24000,
                     help="Total teacher output-token cap per arm (default: 24000)")
    gen.add_argument("--extend-from", type=Path, help="Frozen round-1 exercises.json for this arm")
    train = commands.add_parser("train", parents=[common])
    train.add_argument("--arm", choices=["C", "D"], required=True)
    train.add_argument("--train-seed", type=int,
                       help="Training randomness only; defaults to the frozen split seed")
    train.add_argument("--model-path", default=STUDENT)
    train.add_argument("--tokens-per-step", type=int, default=512)
    train.add_argument("--passes", type=positive_int, default=1,
                       help="Passes over each arm's exercises with the common token cap per pass (default: 1)")
    train.add_argument("--supervised-budget", type=positive_int,
                       help="Total token exposure; derives passes from the common per-pass cap")
    train.add_argument("--learning-rate", type=float, default=1e-5,
                       help="AdamW learning rate, shared by both arms (default: 1e-5)")
    train.add_argument("--position-chunk", type=int, default=32)
    audit = commands.add_parser("audit-loss", parents=[common],
                                help="Read-only before/after decomposition of the training objective")
    audit.add_argument("--arm", choices=["C", "D"], required=True)
    audit.add_argument("--train-seed", type=int,
                       help="Adapter training seed; defaults to the frozen split seed")
    audit.add_argument("--checkpoint", type=Path,
                       help="Adapter directory; defaults to the selected arm/seed's training adapter")
    audit.add_argument("--model-path", help="Cached base model; defaults to the saved training config")
    audit.add_argument("--position-chunk", type=positive_int,
                       help="Projection chunk size; defaults to the saved training config")
    evaluate = commands.add_parser("evaluate", parents=[common])
    evaluate.add_argument("--arm", choices=["base", "C", "D"], required=True)
    evaluate.add_argument("--train-seed", type=int,
                          help="Adapter training seed; defaults to the frozen split seed, ignored for base")
    evaluate.add_argument("--repeat", choices=["main", "repeat"], default="main")
    evaluate.add_argument("--checkpoint", choices=["mid", "end"], default="end")
    report = commands.add_parser("report", parents=[common])
    report.add_argument("--round1-run-dir", type=Path, help="Read-only R1 evaluations, including seed repeats")
    for command in commands.choices.values():
        command.add_argument("--tokenizer", default=None if command is audit else STUDENT,
                             help="Cached tokenizer; audit-loss defaults to the frozen training plan")
    return p


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def main(argv=None):
    args = parser().parse_args(argv)
    args.run_dir = args.run_dir.resolve()
    # An audit must not create pipeline.lock or bind/rewrite protocol.json.
    if args.command == "audit-loss":
        from bfas.mech_bfcl.training import audit_loss
        splits = read_json(args.splits)
        if args.seed != splits["seed"]:
            raise ValueError("All stages must use the frozen split seed")
        args.train_seed = resolved_train_seed(args, splits["seed"])
        audit_loss(args, splits)
        return 0
    setup_harness(args.run_dir / "runtime")
    if args.command == "splits":
        from bfas.mech_bfcl.splits import run
        run(args)
        return 0
    splits = read_json(args.splits)
    if args.seed != splits["seed"]:
        raise ValueError("All stages must use the frozen split seed")
    if args.command in ("train", "evaluate"):
        args.train_seed = resolved_train_seed(args, splits["seed"])
    args.served_model = args.served_model or (
        "mech-" + variant_name(args.arm, args.train_seed, splits["seed"], args.checkpoint)
        if args.command == "evaluate" and args.arm != "base" else STUDENT)
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
            getattr(pipeline, args.command.replace("-", "_"))(args, splits)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
