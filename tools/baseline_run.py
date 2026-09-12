#!/usr/bin/env python3
"""Budgeted SmartAD/SAD/Kang/GAD, sealed Luna bank and Gemma 4 student.

Default runs training and official evaluation in separate processes. Use
--prepare-only for a CPU-only purchase/manifest audit without loading a model.
Comma-separated --budget-tokens levels use --run-dir as a common directory prefix.
"""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/"src"), str(ROOT)]


def budget_token_levels(value):
    try:
        levels = [int(part) for part in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("budget tokens must be comma-separated nonnegative integers") from error
    if any(level < 0 for level in levels):
        raise argparse.ArgumentTypeError("budget tokens must be nonnegative integers")
    if len(set(levels)) != len(levels):
        raise argparse.ArgumentTypeError("budget token levels must be distinct")
    return levels


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=("smartad", "sad", "kang", "gad"))
    parser.add_argument("--benchmark", required=True, choices=("alfworld", "bfcl", "hotpotqa"))
    parser.add_argument("--bank", required=True, type=Path)
    budget = parser.add_mutually_exclusive_group(required=True)
    budget.add_argument("--budget-tokens", type=budget_token_levels, metavar="B[,B,...]",
                        help="absolute teacher-output-token cap(s); multiple levels use RUN_DIR_B<cap>")
    budget.add_argument("--budget-fraction", help="legacy fraction of the certified usable cost basis")
    parser.add_argument("--seed", type=int, default=0, choices=(0,))
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--port", type=int, default=8930)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--smoke", action="store_true",
                        help="2 student steps (2 rows each), at most 2 preconditioner prompts, 3 evaluation tasks")
    parser.add_argument("--_phase", choices=("train", "evaluate"), help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args._phase and args.budget_tokens is not None and len(args.budget_tokens) != 1:
        parser.error("an internal worker phase requires a single budget level")
    return args


def validate_run_args(args):
    if args.benchmark == "hotpotqa" and args.smoke:
        raise ValueError("HotpotQA evaluation requires the frozen 500-question dev split; omit --smoke")
    if not args.bank.is_absolute():
        raise ValueError("--bank must be an absolute path")
    if not 1 <= args.port <= 65535:
        raise ValueError("--port must be between 1 and 65535")
    if args.run_dir.resolve().is_relative_to(args.bank.resolve()):
        raise ValueError("run directory must not be inside the read-only bank")
    if args.run_dir.exists():
        raise FileExistsError("choose a fresh --run-dir; existing runs are never overwritten")


def prepare(args):
    import yaml
    from bfas.rtd.baselines.paper_data import STUDENT, load_purchased
    from bfas.rtd.baselines.paper_evaluation import protocol
    from bfas.rtd.baselines.paper_train import hyperparameters
    from bfas.rtd.persistence import atomic_json, file_hash, digest
    validate_run_args(args)
    cfg_path = ROOT/"configs/rtd"/f"v1_1_{args.benchmark}_luna.yaml"
    config = yaml.safe_load(cfg_path.read_text())
    shared_path = ROOT/"configs/rtd/v1_1_alfworld_luna.yaml"
    shared = yaml.safe_load(shared_path.read_text())
    for key in ("lora_rank", "lora_alpha", "lora_target_modules"):
        config[key] = shared[key]
    config.update(student=STUDENT, training_seed=0, replay_bank_path=str(args.bank),
        model_local_files_only=True, new_teacher_calls=False, new_teacher_tokens=0,
        initial_eta=1e-5, optimizer="fixed_preconditioned_single_step", smoke=args.smoke,
        training_device="cuda:0", cpu_threads=4, score_position_chunk_size=32)
    if args.benchmark == "bfcl":
        config["student_call_format"] = "gemma4"
    purchase, rows = load_purchased(args.bank, args.benchmark, args.budget_fraction,
                                   budget_tokens=args.budget_tokens)
    source_files = [Path(__file__), *sorted((ROOT/"src/bfas/rtd/baselines").glob("paper_*.py")),
                    ROOT/"tools/baseline_bfcl.py", ROOT/"src/bfas/adapters/alfworld.py",
                    ROOT/"src/bfas/run.py", ROOT/"src/bfas/processes.py",
                    *[ROOT/"src/bfas/rtd"/name for name in ("runtime.py", "functional_step.py",
                        "return_gradient.py", "student.py", "transport.py", "evaluation.py",
                        "source_scoring.py", "persistence.py")]]
    if args.benchmark == "hotpotqa":
        source_files.extend([ROOT/"src/bfas/adapters/hotpotqa.py", ROOT/"src/bfas/hotpotqa.py",
            ROOT/"src/bfas/hotpotqa_budget.py", ROOT/"tools/hotpotqa_eval.py",
            ROOT/"prompts/hotpotqa_react_6shot.txt",
            *sorted((ROOT/"configs").glob("hotpotqa_*_split.json")),
            *sorted((ROOT/"src/bfas/rtd/benchmarks").glob("hotpotqa_*.py")),
            *[ROOT/"src/bfas/rtd/benchmarks"/name for name in
              ("config.py", "registry.py", "webshop_evaluation.py")],
            ROOT/"src/bfas/rtd/caps.py", ROOT/"src/bfas/rtd/feedback_rng.py"])
    manifest = dict(version="budgeted-paper-baselines-v1", method=args.method, benchmark=args.benchmark,
        seed=0, student=STUDENT, teacher="gpt-5.6-luna", **purchase, config=config,
        config_sources={str(p.relative_to(ROOT)): file_hash(p) for p in (cfg_path, shared_path)},
        hyperparameters=hyperparameters(config, args.method), evaluation_protocol=protocol(args.benchmark, smoke=args.smoke),
        smoke=args.smoke,
        source_hashes={str(p.relative_to(ROOT)): file_hash(p) for p in source_files},
        port=args.port, status="prepared", gpu_hours=0., gpu_accounting="single GPU reserved wall time; export/scoring CPU excluded",
        deviations_document="docs/PAPER_BASELINES.md", rows_hash=digest([asdict(r) for r in rows]))
    args.run_dir.mkdir(parents=True, exist_ok=False)
    atomic_json(args.run_dir/"purchased_rows.json", [asdict(r) for r in rows])
    atomic_json(args.run_dir/"manifest.json", manifest)
    return manifest


def train_worker(directory, manifest):
    import torch
    from bfas.rtd.baselines.paper_progress import progress, stage
    threads = max(1, min(4, int(manifest["config"].get("cpu_threads", 4))))
    torch.set_num_threads(threads)
    progress("runtime", "configured", cpu_threads=torch.get_num_threads(),
             cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"), logical_device="cuda:0",
             position_chunk_size=manifest["config"]["score_position_chunk_size"])
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
    manifest["status"] = "training"
    atomic_json(directory/"manifest.json", manifest)
    with stage("model_identity"):
        model = _snapshot_for_model(manifest["student"])
        manifest.update(model_path=str(model), base_checkpoint_hash=tree_hash(model),
            tokenizer_hash=digest([(p.name, file_hash(p)) for p in sorted(model.glob("*"))
                                  if p.is_file() and any(s in p.name for s in ("token", "vocab", "merges", "chat_template"))]),
            harness_hash=digest(manifest["source_hashes"]), hardware=hardware_identity())
    atomic_json(directory/"manifest.json", manifest)
    # Baselines keep one action graph live. Reclaiming the whole Python heap
    # twice per forward is CPU-bound and defeats CUDA's reusable allocator.
    journal = ComputeJournal(directory/"compute.jsonl", cuda=True, release_phase_cache=False)
    start = time.monotonic()
    try:
        with stage("model_load"):
            backend = load_backend(manifest["config"], manifest, journal)
        trainer = PaperTrainer(backend, [TeacherRow(**r) for r in rows], manifest["config"],
                               manifest["method"], directory, journal)
        result = trainer.train()
        manifest.update(training=result, checkpoint_sha256=tree_hash(directory/"checkpoint"), status="trained")
    except BaseException as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        manifest["training_gpu_hours"] = (time.monotonic()-start)/3600
        atomic_json(directory/"manifest.json", manifest)


def run_budget(args):
    from bfas.rtd.persistence import atomic_json
    path = args.run_dir/"manifest.json"
    manifest = prepare(args)
    if args.prepare_only:
        print(json.dumps(dict(status="prepared", run_dir=str(args.run_dir), B=manifest["B"],
            teacher_tokens_charged=manifest["teacher_tokens_charged"],
            usable_packages=manifest["purchased_usable_packages"], new_teacher_calls=0, gpu_hours=0.)))
        return 0
    budget = (["--budget-tokens", str(args.budget_tokens)] if args.budget_tokens is not None
              else ["--budget-fraction", args.budget_fraction])
    worker = [sys.executable, str(Path(__file__).resolve()), "--method", args.method,
        "--benchmark", args.benchmark, "--bank", str(args.bank), *budget,
        "--seed", "0", "--run-dir", str(args.run_dir)]
    if args.smoke:
        worker.append("--smoke")
    try:
        for phase in ("train", "evaluate"):
            with (args.run_dir/f"{phase}.log").open("w") as log:
                run_worker([*worker, "--_phase", phase], log)
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


def run_worker(command, log):
    """Let a cancelled evaluation worker unwind its vLLM cleanup before exit."""
    from bfas.processes import stop_process_group
    process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                               start_new_session=True)
    try:
        code = process.wait()
        if code:
            raise subprocess.CalledProcessError(code, command)
    finally:
        # Workers handle TERM through Python finally blocks. subprocess.run's
        # immediate kill on KeyboardInterrupt would strand their setsid server.
        stop_process_group(process, timeout=45)


def terminate_worker(signum, frame):
    raise SystemExit(128+signum)


def main(argv=None):
    args = arguments(argv)
    args.run_dir = args.run_dir.resolve()
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONHASHSEED="0",
                      PYTHONUNBUFFERED="1", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
                      RAYON_NUM_THREADS="4", TOKENIZERS_PARALLELISM="false")
    if args._phase:
        from bfas.rtd.persistence import file_hash
        manifest = json.loads((args.run_dir/"manifest.json").read_text())
        for name, sha in manifest["source_hashes"].items():
            if file_hash(ROOT/name) != sha:
                raise ValueError("baseline code changed after preparation")
        if args._phase == "train":
            train_worker(args.run_dir, manifest)
        else:
            from bfas.rtd.baselines.paper_evaluation import evaluate_run
            evaluate_run(ROOT, args.run_dir, manifest)
        return 0
    levels = args.budget_tokens
    runs = []
    for cap in levels if levels is not None else [None]:
        directory = (args.run_dir.with_name(f"{args.run_dir.name}_B{cap}")
                     if levels is not None and len(levels) > 1 else args.run_dir)
        runs.append(argparse.Namespace(**{**vars(args), "budget_tokens": cap, "run_dir": directory}))
    # Refuse collisions across the entire curve before creating or running any level.
    for run in runs:
        validate_run_args(run)
    for run in runs:
        run_budget(run)
    return 0


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, terminate_worker)
    raise SystemExit(main())
