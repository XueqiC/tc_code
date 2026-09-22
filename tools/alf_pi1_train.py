#!/usr/bin/env python3
"""Train one offline K=32 pi1 seed, saving both pre-registered exposure endpoints."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/"src"), str(ROOT)]


def select_device(uuid):
    """Require a full caller-supplied UUID; inspect it without initializing CUDA."""
    if not re.fullmatch(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", uuid or ""):
        raise ValueError("--gpu-uuid requires a full nvidia-smi GPU UUID")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = uuid
    result = subprocess.run(["nvidia-smi", "--id="+uuid,
        "--query-gpu=uuid,pci.bus_id", "--format=csv,noheader,nounits"],
        check=True, text=True, capture_output=True, timeout=10)
    rows = [line.split(",") for line in result.stdout.strip().splitlines()]
    if len(rows) != 1 or len(rows[0]) != 2 or rows[0][0].strip().lower() != uuid.lower():
        raise ValueError("nvidia-smi did not verify the selected GPU UUID")
    return dict(uuid=rows[0][0].strip(), pci_bus_id=rows[0][1].strip(),
                cuda_device_order="PCI_BUS_ID", logical_device="cuda:0")


def verify_cuda_device(selected, config):
    """Check logical device UUID before model/tensor allocation; cap our allocator."""
    import torch
    if torch.cuda.is_initialized():
        raise ValueError("CUDA was initialized before pi1 device verification")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("pi1 requires exactly the selected visible GPU")
    props = torch.cuda.get_device_properties(0)
    uuid = str(getattr(props, "uuid", "")).removeprefix("GPU-")
    if uuid.lower() != selected["uuid"].removeprefix("GPU-").lower():
        raise ValueError("logical cuda:0 UUID differs from the caller's selected GPU")
    free, total = torch.cuda.mem_get_info(0)
    limit = min(int(config["memory_peak_budget_gb"]*1e9),
                free-int(config["memory_reserve_gb"]*1e9))
    if limit <= 0:
        raise ValueError("selected GPU has no usable memory after the configured reserve")
    torch.cuda.set_per_process_memory_fraction(limit/total, 0)
    return dict(selected, name=props.name, total_bytes=total, free_bytes_before_model=free,
                allocator_limit_bytes=limit)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT/"configs/rtd/pi1_alfworld_k32.yaml")
    parser.add_argument("--seed", type=int, choices=(0, 1))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bank", type=Path, help="read-only bank location override")
    parser.add_argument("--gpu-uuid", help="full GPU UUID from nvidia-smi; required for training")
    parser.add_argument("--model-path", type=Path, help="local base snapshot; defaults to the cached student")
    parser.add_argument("--resume", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-only", action="store_true", help="validate/write the run on CPU, without training")
    mode.add_argument("--preflight", action="store_true", help="print one real row through rendering and training tensors on CPU")
    mode.add_argument("--check-step", action="store_true", help="one checked optimizer step on four real rows on --gpu-uuid")
    args = parser.parse_args(argv)
    if not args.preflight and (args.seed is None or args.output is None):
        parser.error("--seed and --output are required except for --preflight")
    # Offline even if the caller's environment normally allows Hub downloads.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    # Do this before importing torch or any project module that imports it.
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "" if args.prepare_only or args.preflight else (args.gpu_uuid or "")
    from bfas.rtd.baselines.pi1 import (VERSION, check_output, encode_teacher_turn,
        boundary_contract, check_optimizer_step, exposure_plan, load_bank, load_config,
        preflight_row, prepare_run)
    from bfas.rtd.baselines.paper_progress import stage
    from bfas.rtd.baselines.paper_train import PaperTrainer, hyperparameters
    from bfas.rtd.baselines.paper_seeds import seed_training
    from bfas.rtd.benchmarks.alfworld_support import FrozenRenderer
    from bfas.rtd.benchmarks.alfworld_identity import SCOPES, tokenizer_identity
    from bfas.rtd.evaluation_lock import evaluation_lock
    from bfas.rtd.persistence import ComputeJournal, atomic_json, digest, file_hash, tree_hash
    from tools.bfcl_hub_merge_export import _snapshot_for_model
    if args.output is not None:
        check_output(args.output, resume=args.resume)
    config = load_config(args.config)
    config.update(training_seed=args.seed, training_device="cuda:0")
    bank = (args.bank or ROOT/config["bank"]).resolve()
    model = (args.model_path.resolve() if args.model_path else _snapshot_for_model(config["student"]).resolve())
    if args.preflight:
        renderer = FrozenRenderer(model)
        rows, _ = load_bank(bank, config, renderer)
        tokenizer = renderer.adapter._tokenizer
        sample = preflight_row(tokenizer, rows[0], config["max_context_tokens"])
        costs = [len(encode_teacher_turn(tokenizer, r, config["max_context_tokens"])["target_ids"]) for r in rows]
        assert sum(costs) == config["bank_supervised_tokens"]
        plan = exposure_plan(costs, config, args.seed or 0)
        sample["exposure"] = dict(turns=len(rows), authored_tokens=sum(costs)-len(rows),
            boundary_tokens=len(rows), bank_supervised_tokens=sum(costs),
            passes=config["exposure_passes"], target_tokens=plan["target_tokens"])
        print(json.dumps(sample, indent=2))
        return 0
    output = args.output.resolve()
    if any(output.is_relative_to(p) for p in (ROOT/"data", bank, model)):
        raise ValueError("output must be outside the bank, data, and base-model directories")
    selected = None if args.prepare_only else select_device(args.gpu_uuid)
    with stage("pi1_prepare", gpu_held=False):
        renderer = FrozenRenderer(model)
        rows, bank_identity = load_bank(bank, config, renderer)
        costs = [len(encode_teacher_turn(renderer.adapter._tokenizer, r,
                     config["max_context_tokens"])["target_ids"]) for r in rows]
        if sum(costs) != config["bank_supervised_tokens"]:
            raise ValueError("tokenizer differs from the registered bank target-token count")
        if args.check_step:
            if args.resume:
                raise ValueError("execution check requires a fresh output")
            rows, costs = rows[:4], costs[:4]
            config.update(exposure_passes=[1], supervised_tokens_per_update=sum(costs))
        plan = exposure_plan(costs, config, args.seed)
        source_files = [Path(__file__), ROOT/"tools/bfcl_hub_merge_export.py",
            ROOT/"src/bfas/behavior/deltas.py", *[ROOT/name for name in SCOPES],
            *[ROOT/"src/bfas/rtd"/name for name in
            ("baselines/pi1.py", "baselines/paper_train.py", "baselines/paper_seeds.py",
             "baselines/paper_losses.py", "baselines/exposure.py", "runtime.py",
             "checkpointing.py", "source_scoring.py", "persistence.py", "functional_step.py",
             "return_gradient.py", "transport.py", "student.py", "scoring.py")]]
        sources = {str(p.relative_to(ROOT)): file_hash(p) for p in source_files}
        identity = dict(version=VERSION, config=config, config_sha256=file_hash(args.config),
            seed=args.seed, hyperparameters=hyperparameters(config, "pi1_ce"), bank=bank_identity,
            model_path=str(model), base_checkpoint_hash=tree_hash(model),
            tokenizer_hash=tokenizer_identity(model)["hash"], source_hashes=sources,
            versions={n: importlib.metadata.version(n) for n in ("torch", "transformers", "peft", "numpy")},
            endpoint_strategy="one trajectory per seed; checkpoint after 3 and 10 complete passes",
            target_boundary=boundary_contract(renderer.adapter._tokenizer),
            execution_check=bool(args.check_step),
            new_teacher_calls=0, new_teacher_tokens=0)
        if args.check_step:
            identity["endpoint_strategy"] = "execution check: one optimizer step on four rows"
    # The lease serializes explicit resumes. New runs still fail on mkdir races.
    manifest = prepare_run(output, identity, rows, plan, resume=args.resume) if not args.resume else None
    with evaluation_lock(output/".train.lock", tag="pi1-training", timeout=0):
        if args.resume:
            manifest = prepare_run(output, identity, rows, plan, resume=True)
        if args.prepare_only:
            print(json.dumps(dict(status="prepared", output=str(output), seed=args.seed,
                                  target_tokens=plan["target_tokens"])))
            return 0
        import torch
        from bfas.rtd.runtime import load_backend
        device = verify_cuda_device(selected, config)
        atomic_json(output/"device.json", device)
        torch.set_num_threads(config["cpu_threads"])
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        seed_training(args.seed)
        journal = ComputeJournal(output/"compute.jsonl", cuda=True, release_phase_cache=False)
        journal.append("pi1_device_verified", **device, seed=args.seed, resume=args.resume)
        manifest["device"] = device
        atomic_json(output/"manifest.json", manifest)
        with stage("model_load"):
            backend = load_backend(config, dict(identity, harness_hash=digest(identity["source_hashes"])), journal)
        trainer = PaperTrainer(backend, rows, config, "pi1_ce", output, journal, manifest=manifest)
        result = check_optimizer_step(trainer) if args.check_step else trainer.train(resume=args.resume)
        print(json.dumps(dict(output=str(output), seed=args.seed, **result)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
