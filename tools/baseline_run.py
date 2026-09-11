#!/usr/bin/env python3
"""Budgeted SmartAD/SAD/Kang/GAD, sealed Luna bank and Gemma 4 student.

Default runs training and official evaluation in separate processes. Use
--prepare-only for a CPU-only purchase/manifest audit without loading a model.
"""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/"src"), str(ROOT)]


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=("smartad", "sad", "kang", "gad"))
    parser.add_argument("--benchmark", required=True, choices=("alfworld", "bfcl"))
    parser.add_argument("--bank", required=True, type=Path)
    parser.add_argument("--budget-fraction", default="0.25")
    parser.add_argument("--seed", type=int, default=0, choices=(0,))
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--port", type=int, default=8930)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--_phase", choices=("train", "evaluate"), help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def prepare(args):
    import yaml
    from bfas.rtd.baselines.paper_data import STUDENT, load_purchased
    from bfas.rtd.baselines.paper_evaluation import protocol
    from bfas.rtd.baselines.paper_train import hyperparameters
    from bfas.rtd.persistence import atomic_json, file_hash, digest
    if not args.bank.is_absolute():
        raise ValueError("--bank must be an absolute path")
    if not 1 <= args.port <= 65535:
        raise ValueError("--port must be between 1 and 65535")
    if args.run_dir.resolve().is_relative_to(args.bank.resolve()):
        raise ValueError("run directory must not be inside the read-only bank")
    if args.run_dir.exists():
        raise FileExistsError("choose a fresh --run-dir; existing runs are never overwritten")
    cfg_path = ROOT/"configs/rtd"/f"v1_1_{args.benchmark}_luna.yaml"
    config = yaml.safe_load(cfg_path.read_text())
    shared_path = ROOT/"configs/rtd/v1_1_alfworld_luna.yaml"
    shared = yaml.safe_load(shared_path.read_text())
    for key in ("lora_rank", "lora_alpha", "lora_target_modules"):
        config[key] = shared[key]
    config.update(student=STUDENT, training_seed=0, replay_bank_path=str(args.bank),
        model_local_files_only=True, new_teacher_calls=False, new_teacher_tokens=0,
        initial_eta=1e-5, optimizer="fixed_preconditioned_single_step")
    if args.benchmark == "bfcl":
        config["student_call_format"] = "gemma4"
    purchase, rows = load_purchased(args.bank, args.benchmark, args.budget_fraction)
    source_files = [Path(__file__), *sorted((ROOT/"src/bfas/rtd/baselines").glob("paper_*.py")),
                    ROOT/"tools/baseline_bfcl.py", ROOT/"src/bfas/adapters/alfworld.py",
                    *[ROOT/"src/bfas/rtd"/name for name in ("runtime.py", "functional_step.py",
                        "return_gradient.py", "student.py", "transport.py", "evaluation.py")]]
    manifest = dict(version="budgeted-paper-baselines-v1", method=args.method, benchmark=args.benchmark,
        seed=0, student=STUDENT, teacher="gpt-5.6-luna", **purchase, config=config,
        config_sources={str(p.relative_to(ROOT)): file_hash(p) for p in (cfg_path, shared_path)},
        hyperparameters=hyperparameters(config, args.method), evaluation_protocol=protocol(args.benchmark),
        source_hashes={str(p.relative_to(ROOT)): file_hash(p) for p in source_files},
        port=args.port, status="prepared", gpu_hours=0., gpu_accounting="single GPU reserved wall time; export/scoring CPU excluded",
        deviations_document="docs/PAPER_BASELINES.md", rows_hash=digest([asdict(r) for r in rows]))
    args.run_dir.mkdir(parents=True, exist_ok=False)
    atomic_json(args.run_dir/"purchased_rows.json", [asdict(r) for r in rows])
    atomic_json(args.run_dir/"manifest.json", manifest)
    return manifest


def train_worker(directory, manifest):
    from bfas.rtd.baselines.paper_data import TeacherRow
    from bfas.rtd.baselines.paper_train import PaperTrainer
    from bfas.rtd.persistence import ComputeJournal, atomic_json, digest, file_hash, tree_hash
    from bfas.rtd.runtime import load_backend
    from tools.bfcl_hub_merge_export import _snapshot_for_model
    from bfas.rtd.hardware import hardware_identity
    rows = json.loads((directory/"purchased_rows.json").read_text())
    if digest(rows) != manifest["rows_hash"]:
        raise ValueError("purchased rows changed")
    if not rows:
        raise ValueError("budget purchased no usable trajectory; inspect the charged-attempt manifest")
    model = _snapshot_for_model(manifest["student"])
    manifest.update(model_path=str(model), base_checkpoint_hash=tree_hash(model),
        tokenizer_hash=digest([(p.name, file_hash(p)) for p in sorted(model.glob("*"))
                              if p.is_file() and any(s in p.name for s in ("token", "vocab", "merges", "chat_template"))]),
        harness_hash=digest(manifest["source_hashes"]), hardware=hardware_identity())
    atomic_json(directory/"manifest.json", manifest)
    journal = ComputeJournal(directory/"compute.jsonl", cuda=True)
    start = time.monotonic()
    try:
        backend = load_backend(manifest["config"], manifest, journal)
        trainer = PaperTrainer(backend, [TeacherRow(**r) for r in rows], manifest["config"],
                               manifest["method"], directory, journal)
        result = trainer.train()
        manifest.update(training=result, checkpoint_sha256=tree_hash(directory/"checkpoint"), status="trained")
    finally:
        manifest["training_gpu_hours"] = (time.monotonic()-start)/3600
        atomic_json(directory/"manifest.json", manifest)


def main(argv=None):
    args = arguments(argv)
    args.run_dir = args.run_dir.resolve()
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONHASHSEED="0")
    from bfas.rtd.persistence import atomic_json, file_hash
    path = args.run_dir/"manifest.json"
    if args._phase:
        manifest = json.loads(path.read_text())
        for name, sha in manifest["source_hashes"].items():
            if file_hash(ROOT/name) != sha:
                raise ValueError("baseline code changed after preparation")
        if args._phase == "train":
            train_worker(args.run_dir, manifest)
        else:
            from bfas.rtd.baselines.paper_evaluation import evaluate_run
            evaluate_run(ROOT, args.run_dir, manifest)
        return 0
    manifest = prepare(args)
    if args.prepare_only:
        print(json.dumps(dict(status="prepared", B=manifest["B"],
            teacher_tokens_charged=manifest["teacher_tokens_charged"],
            usable_packages=manifest["purchased_usable_packages"], new_teacher_calls=0, gpu_hours=0.)))
        return 0
    worker = [sys.executable, str(Path(__file__).resolve()), "--method", args.method,
        "--benchmark", args.benchmark, "--bank", str(args.bank), "--budget-fraction", args.budget_fraction,
        "--seed", "0", "--run-dir", str(args.run_dir)]
    try:
        for phase in ("train", "evaluate"):
            with (args.run_dir/f"{phase}.log").open("w") as log:
                subprocess.run([*worker, "--_phase", phase], cwd=ROOT, check=True,
                               stdout=log, stderr=subprocess.STDOUT)
        manifest = json.loads(path.read_text())
        manifest["status"] = "complete"
    except BaseException as error:
        manifest = json.loads(path.read_text())
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        usages = [json.loads(p.read_text()) for p in args.run_dir.glob("*/gpu_usage.json")]
        manifest["evaluation_gpu_hours"] = sum(r["gpu_seconds"] for r in usages)/3600
        manifest["gpu_hours"] = manifest.get("training_gpu_hours", 0.)+manifest["evaluation_gpu_hours"]
        atomic_json(path, manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
