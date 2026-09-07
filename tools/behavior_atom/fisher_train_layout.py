#!/usr/bin/env python3
"""Estimate a diagonal empirical Fisher in the micro-update's LoRA coordinates.

F = mean_rows[(grad_theta ell(prompt, continuation)) ** 2]. ``full_sequence``
sums response log probabilities, including EOS after the trainer's response
cap; ``length_normalized`` (also spelled ``length_normalised`` on the CLI)
divides ell by the shifted, unmasked response-token count BEFORE differentiation.
This is a sequence empirical Fisher on recorded continuations, not a token
Fisher or the square of a minibatch-mean gradient. Student ``_rejected`` is the
default continuation; ``--owner teacher`` selects ``response``, still scored by
the student. No sampling, teacher calls, optimizer steps, or probe data are used.

The real model comes directly from appworld_train.build_lora_model, just like
microupdate and crcd_mirror_train: bf16 base, native PEFT adapter dtypes, BOTH
LoRA factors, unchanged parameter order/strides. Casting the entire model to
bf16 would change the layout hash when PEFT promotes adapters to fp32.
Per-sequence gradients are flattened and promoted to fp32 on the model device,
then copied once into a CPU fp32 sum. ``--batch-rows`` groups input rows, but each row gets its own
forward/backward; it does not pad sequences or average gradients before squaring.
Gradient clipping is DIAGNOSTIC ONLY: the threshold/count/norms are recorded,
but no clipping or subsequent Fisher normalization changes the estimator.

Use only an authorized training/reference pool. ``--pool-role`` records the
caller's provenance declaration; explicit incompatible row split labels are
rejected. This cannot establish split disjointness for unlabeled legacy pools.
Prompt encoding and ell reuse the trainer/micro-update functions. The default
prompt cap is 4096 with head retention (MicroUpdateConfig's defaults), overridable
with AW_MAX_PROMPT_TOKENS/AW_TRUNCATE_SIDE; every truncation is recorded.

``--init-state`` restores a save_initial_state artifact using the runner's exact
checkpoint/settings/layout checks. Alternatively ``--save-initial-state`` saves
the common initialization for later micro-updates. Match their model ID/seed.
Artifacts are exclusive: <output>.npz and <output>.npz.manifest.json. The NPZ is
accepted directly by bfas.behavior.whiten.from_fisher (returning DiagonalMetric).
It stores raw F, with eps recorded separately for H = diag(F) + eps I.

Dry-run uses a locally cached model config and meta tensors with the trainer's
LoRA config to count parameters, without loading weights, a tokenizer, or CUDA.
Its memory estimate excludes activations/logits, CUDA workspace and allocator
overhead. No files are written. CUDA is the default; CPU execution requires
``--allow-cpu``. ``--progress-every`` defaults to 10 rows. ``--benchmark-rows K``
scores at most K rows, reports row timings/peak GPU memory, and writes no files.
Examples:

  .venv/bin/python tools/behavior_atom/fisher_train_layout.py --dry-run
  .venv/bin/python tools/behavior_atom/fisher_train_layout.py \
      --output results/behavior_atom_v1/fisher.npz --max-rows 32
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
from itertools import islice
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Callable, Iterable

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from bfas.behavior.deltas import (canonical_hash, file_hash,
                                  layout_hash, parameter_layout, write_json_exclusive)
from bfas.behavior.whiten import from_fisher

VERSION = "behavior-fisher-train-layout-v1"
DEFAULT_POOL = ROOT / "data/bfcl_sft/pool_events_pref_v3t.jsonl"


def resolve_device(device: str | None = None, *, allow_cpu: bool = False) -> torch.device:
    """Never inherit the CPU/float32 micro-update defaults or silently fall back."""
    available = torch.cuda.is_available()
    device = device or ("cuda" if available else "cpu")
    if device not in {"cuda", "cpu"}:
        raise ValueError("device must be cuda or cpu")
    if device == "cuda" and not available:
        if not allow_cpu:
            raise RuntimeError("CUDA unavailable; CPU execution requires --allow-cpu")
        device = "cpu"
    if device == "cpu":
        if not allow_cpu:
            raise RuntimeError("CPU execution requires --allow-cpu; use --dry-run for a storage estimate")
        print("[fisher] device=cpu (--allow-cpu enabled)", file=sys.stderr, flush=True)
    return torch.device("cuda", torch.cuda.current_device()) if device == "cuda" else torch.device("cpu")


def ell_mode(value: str) -> str:
    value = {"length_normalised": "length_normalized"}.get(value, value)
    if value not in {"full_sequence", "length_normalized"}:
        raise ValueError("ell must be full_sequence or length_normalized")
    return value


@dataclass
class FisherEstimate:
    fisher: np.ndarray
    layout: list[dict]
    n_rows: int
    clipping_stats: dict


def accumulate_fisher(model: torch.nn.Module, rows: Iterable,
                      score: Callable, *, batch_rows: int = 1,
                      max_grad_norm: float = 1.0, device: torch.device | str | None = None,
                      progress_every: int = 10, total_rows: int | None = None,
                      token_count: Callable[[], int] | None = None,
                      benchmark_rows: int | None = None) -> FisherEstimate:
    """Average squared per-row scalar-score gradients, without mutating .grad.

    ``score(model, row)`` must return the differentiable scalar ell for ONE
    sequence. The caller chooses the ell convention and model train/eval mode.
    Unused trainables contribute zero. Frozen parameters never get coordinates.
    ``max_grad_norm`` only measures how often clipping WOULD occur.
    """
    if not isinstance(batch_rows, int) or isinstance(batch_rows, bool) or batch_rows <= 0:
        raise ValueError("batch_rows must be a positive integer")
    if not math.isfinite(max_grad_norm) or max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be finite and positive")
    if type(progress_every) is not int or progress_every <= 0:
        raise ValueError("progress_every must be a positive integer")
    if benchmark_rows is not None and (type(benchmark_rows) is not int or benchmark_rows <= 0):
        raise ValueError("benchmark_rows must be a positive integer")
    layout = parameter_layout(model)
    named = dict(model.named_parameters())
    parameters = [named[entry["name"]] for entry in layout]
    device = torch.device(device) if device is not None else parameters[0].device
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    if any(p.device != device for p in (*named.values(), *model.buffers())):
        raise ValueError(f"all model parameters and buffers must be on {device}")
    total = torch.zeros(sum(entry["numel"] for entry in layout), dtype=torch.float32, device="cpu")
    if total_rows is None and hasattr(rows, "__len__"):
        total_rows = len(rows)
    if benchmark_rows is not None:
        total_rows = min(total_rows, benchmark_rows) if total_rows is not None else benchmark_rows
        rows = islice(rows, benchmark_rows)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
    iterator, norms = iter(rows), []
    started = time.perf_counter()

    def progress():
        elapsed = time.perf_counter() - started
        rate = len(norms) / max(elapsed, 1e-12)
        eta = f"{max(0, total_rows - len(norms)) / rate:.1f}s" if total_rows is not None else "unknown"
        tokens = token_count() if token_count is not None else "unknown"
        print(f"[fisher] rows={len(norms)}/{total_rows if total_rows is not None else '?'} "
              f"rows/s={rate:.3f} tokens={tokens} elapsed={elapsed:.1f}s ETA={eta}",
              file=sys.stderr, flush=True)

    with torch.enable_grad():
        while batch := list(islice(iterator, batch_rows)):
            for row in batch:
                row_started = time.perf_counter()
                value = score(model, row)
                if value.device != device:
                    raise ValueError(f"per-sequence ell must be on {device}, got {value.device}")
                if value.ndim != 0 or not bool(torch.isfinite(value)):
                    raise ValueError("per-sequence ell must be a finite scalar")
                gradients = torch.autograd.grad(value, parameters, allow_unused=True)
                # Only build views (and zeros for unused factors) in Python.
                # Concatenate/promote/check/reduce on the model device, then
                # make ONE host transfer and ONE flat fp32 accumulation per row.
                flat = torch.cat([g.detach().reshape(-1) if g is not None else p.new_zeros(p.numel())
                                  for p, g in zip(parameters, gradients)]).to(dtype=torch.float32)
                if flat.numel() != total.numel():
                    raise ValueError("trainable parameter layout changed during accumulation")
                if not bool(torch.isfinite(flat).all()):
                    raise ValueError("nonfinite gradient")
                norms.append(float(torch.linalg.vector_norm(flat, dtype=torch.float64)))
                host = flat.to(device="cpu")
                total.addcmul_(host, host)
                del gradients, flat, host, value
                if benchmark_rows is not None:
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    seconds = time.perf_counter() - row_started
                    peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
                    print(f"[fisher benchmark] row={len(norms)} seconds={seconds:.6f} "
                          f"peak_gpu_memory_mib={peak / 2**20:.2f}", file=sys.stderr, flush=True)
                if len(norms) % progress_every == 0:
                    progress()
    if not norms:
        raise ValueError("cannot estimate Fisher from zero rows")
    # Metadata only: no rehashing, tensor clones, or checkpoint scans per row.
    current = [p for p in model.parameters() if p.requires_grad]
    if len(current) != len(layout) or any(list(p.shape) != entry["shape"] for p, entry in zip(current, layout)):
        raise ValueError("trainable parameter layout changed during accumulation")
    if len(norms) % progress_every:
        progress()
    elapsed = time.perf_counter() - started
    if not bool(torch.isfinite(total).all()):
        raise ValueError("fp32 Fisher accumulation overflow")
    total.div_(len(norms))
    clipped = sum(norm > max_grad_norm for norm in norms)
    stats = dict(applied=False, diagnostic_only=True, max_grad_norm=max_grad_norm,
                 n_rows=len(norms), would_clip_count=clipped,
                 would_clip_fraction=clipped / len(norms),
                 pre_clip_norm_mean=sum(norms) / len(norms),
                 pre_clip_norm_max=max(norms), per_row_pre_clip_norm=norms)
    if benchmark_rows is not None:
        peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        print(json.dumps(dict(benchmark=True, n_rows=len(norms), device=str(device),
                              seconds_per_row=elapsed / len(norms),
                              peak_gpu_memory_bytes=peak)), flush=True)
    return FisherEstimate(total.numpy(), layout, len(norms), stats)


def read_pool(path: Path, *, owner: str, max_rows: int | None = None):
    """Hash exact file bytes, select the first N nonblank rows, never fallback."""
    if owner not in {"student", "teacher"}:
        raise ValueError("owner must be student or teacher")
    if max_rows is not None and max_rows <= 0:
        raise ValueError("max_rows must be positive")
    field = "_rejected" if owner == "student" else "response"
    rows, digest, n_pool_rows = [], hashlib.sha256(), 0
    with open(path, "rb") as handle:
        for line_number, raw in enumerate(handle, 1):
            digest.update(raw)
            if not raw.strip():
                continue
            n_pool_rows += 1
            if max_rows is not None and len(rows) >= max_rows:
                continue
            row = json.loads(raw)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: row must be an object")
            for key in ("prompt", field):
                if not isinstance(row.get(key), str) or not row[key]:
                    raise ValueError(f"{path}:{line_number}: missing/nonempty string required: {key}")
            for key in ("split", "_split"):
                if row.get(key) is not None and row[key] not in {
                    "train", "training", "reference", "reference_rollout", "reference_rollouts"
                }:
                    raise ValueError(f"{path}:{line_number}: non-training/reference {key}: {row[key]!r}")
            rows.append(row)
    if not rows:
        raise ValueError("pool contains no selected rows")
    return rows, dict(pool_path=str(path.resolve()), pool_content_hash=digest.hexdigest(),
                      n_pool_rows=n_pool_rows, selection="first_nonblank_rows",
                      max_rows=max_rows, selected_rows_hash=canonical_hash(rows))


def memory_estimate(model) -> dict:
    """Storage estimate only; meta tensors support these same queries."""
    params = list(model.parameters())
    trainables = [p for p in params if p.requires_grad]
    n = sum(p.numel() for p in trainables)
    model_bytes = sum(p.numel() * p.element_size() for p in params)
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    gradient_bytes = sum(p.numel() * p.element_size() for p in trainables)
    return dict(parameter_count=sum(p.numel() for p in params), trainable_parameter_count=n,
                model_parameter_bytes=model_bytes, model_buffer_bytes=buffer_bytes,
                gpu_gradient_bytes=gradient_bytes,
                gpu_storage_lower_bound_bytes=model_bytes + buffer_bytes + gradient_bytes,
                host_fisher_accumulator_bytes=4 * n,
                host_largest_gradient_copy_bytes=4 * n,
                active_sequences=1,
                note="Excludes activations/logits, flat GPU gradient workspace, CUDA workspace, allocator overhead, serialization and init-state copies.")


def build_meta_model(model_id: str):
    """Count the trainer's exact architecture/adapters using local config only."""
    from accelerate import init_empty_weights
    from peft import get_peft_model
    from transformers import AutoConfig, AutoModelForCausalLM
    from appworld_train import DTYPE, lora_config

    config = AutoConfig.from_pretrained(model_id, local_files_only=True, trust_remote_code=False)
    with init_empty_weights(include_buffers=True):
        base = AutoModelForCausalLM.from_config(config, torch_dtype=DTYPE, trust_remote_code=False)
        base.tie_weights()
        model = get_peft_model(base, lora_config())
    return model


def save_fisher(estimate: FisherEstimate, path: Path, *, model_id: str,
                pool_content_hash: str, ell: str = "full_sequence",
                owner: str = "student", eps: float = 1e-6,
                provenance: dict | None = None) -> dict:
    """Write a pickle-free NPZ accepted by whiten.from_fisher and its manifest."""
    path = Path(path)
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    if path.suffix != ".npz":
        raise ValueError("output must end in .npz")
    if path.exists() or manifest_path.exists():
        raise FileExistsError(path)
    ell = ell_mode(ell)
    if estimate.n_rows <= 0 or owner not in {"student", "teacher"}:
        raise ValueError("positive n_rows and student/teacher owner required")
    saved_hash = canonical_hash(estimate.layout)
    # Validate shape, layout, nonnegativity, finiteness and positive damping
    # with the same consumer used by downstream behavior geometry.
    fisher = np.asarray(estimate.fisher, dtype=np.float32)
    metric = from_fisher(fisher, estimate.layout, eps=eps,
                         layout_hash=saved_hash, version=VERSION)
    manifest = dict(provenance or {})
    manifest.update(version=VERSION, estimator="mean_of_squared_per_sequence_gradients",
                    layout=estimate.layout, layout_hash=saved_hash,
                    parameter_names=[e["name"] for e in estimate.layout],
                    parameter_shapes=[e["shape"] for e in estimate.layout],
                    parameter_count=metric.size, n_rows=estimate.n_rows, ell=ell,
                    ell_definition=("sum_response_log_probs_including_eos" if ell == "full_sequence"
                                    else "sum_response_log_probs_including_eos / shifted_unmasked_tokens"),
                    owner=owner, continuation_field="_rejected" if owner == "student" else "response",
                    model_id=model_id, pool_content_hash=pool_content_hash,
                    fisher_dtype="float32", accumulator_dtype="float32",
                    fisher_normalization="mean_over_sequences_only", eps=eps,
                    damping_applied_to_saved_fisher=False, fisher_hash=metric.fisher_hash,
                    clipping_stats=estimate.clipping_stats)
    del metric
    # Shapes can have different ranks; JSON strings avoid object arrays/pickle.
    arrays = dict(fisher=fisher, layout_hash=saved_hash,
                  parameter_names=np.asarray(manifest["parameter_names"]),
                  parameter_shapes=np.asarray([json.dumps(s) for s in manifest["parameter_shapes"]]),
                  layout=json.dumps(estimate.layout, sort_keys=True), n_rows=estimate.n_rows,
                  ell=ell, model_id=model_id, pool_content_hash=pool_content_hash,
                  version=VERSION, owner=owner, eps=eps,
                  manifest=json.dumps(manifest, sort_keys=True, allow_nan=False))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "xb") as handle:
        np.savez(handle, **arrays)
    manifest["payload_hash"] = file_hash(path)
    write_json_exclusive(manifest_path, manifest)
    return manifest


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--model-id", "--student", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--output", type=Path, default=ROOT / "results/behavior_atom_v1/fisher_train_layout.npz")
    parser.add_argument("--owner", choices=("student", "teacher"), default="student")
    parser.add_argument("--pool-role", choices=("training", "reference"), default="training")
    parser.add_argument("--ell", choices=("full_sequence", "length_normalized", "length_normalised"), default="full_sequence")
    parser.add_argument("--max-rows", type=positive_int)
    parser.add_argument("--batch-rows", type=positive_int, default=1, help="row grouping; backwards remain per sequence")
    parser.add_argument("--progress-every", type=positive_int, default=10, help="log progress every N rows (and at completion)")
    parser.add_argument("--benchmark-rows", type=positive_int, help="time at most K rows, report peak GPU memory, write no artifacts")
    parser.add_argument("--device", choices=("cuda", "cpu"), help="defaults to cuda when available; CPU requires --allow-cpu")
    parser.add_argument("--allow-cpu", action="store_true", help="explicitly allow CPU execution, including when CUDA is unavailable")
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="diagnostic clipping threshold; gradients remain unclipped")
    parser.add_argument("--eps", type=float, default=1e-6, help="recorded downstream damping; saved Fisher remains raw")
    parser.add_argument("--seed", type=int, default=0, help="same initialization seed as the micro-update builder")
    initial = parser.add_mutually_exclusive_group()
    initial.add_argument("--init-state", type=Path)
    initial.add_argument("--save-initial-state", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if not 0 <= args.seed < 2**32:
        parser.error("seed must fit uint32")
    if any(not math.isfinite(x) or x <= 0 for x in (args.eps, args.max_grad_norm)):
        parser.error("eps and max-grad-norm must be finite and positive")
    args.ell = ell_mode(args.ell)
    device = None if args.dry_run else resolve_device(args.device, allow_cpu=args.allow_cpu)
    limit = (min(args.max_rows, args.benchmark_rows) if args.max_rows and args.benchmark_rows
             else args.max_rows or args.benchmark_rows)
    rows, provenance = read_pool(args.pool, owner=args.owner, max_rows=limit)
    # Match MicroUpdateConfig defaults, respecting campaign environment overrides.
    os.environ.setdefault("AW_MAX_PROMPT_TOKENS", "4096")
    os.environ.setdefault("AW_TRUNCATE_SIDE", "head")
    import appworld_train as trainer
    from appworld_train import (MAX_RESPONSE_TOKENS, build_lora_model,
                                encode, load_tokenizer, prompt_token_ids,
                                prompt_truncation_config, rejected_row)

    prompt_cap, prompt_side = prompt_truncation_config()
    if args.dry_run:
        model = build_meta_model(args.model_id)
        print(json.dumps(dict(dry_run=True, n_rows=len(rows), model_id=args.model_id,
                              layout_hash=layout_hash(model), memory_estimate=memory_estimate(model),
                              **provenance), indent=2))
        return
    output_paths = () if args.benchmark_rows else (
        args.output, args.output.with_suffix(args.output.suffix + ".manifest.json"), args.save_initial_state)
    for path in output_paths:
        if path is not None and path.exists():
            raise FileExistsError(path)
    if not args.benchmark_rows and args.output.suffix != ".npz":
        parser.error("output must end in .npz")
    print(f"[fisher] loading model={args.model_id} device={device}", file=sys.stderr, flush=True)
    # build_base_model reads this module global. Override it for CPU opt-in too,
    # so the builder never allocates on an unintended device. Do not cast dtype.
    previous_device = trainer.DEVICE
    try:
        trainer.DEVICE = str(device)
        model = build_lora_model(args.model_id, args.seed).to(device=device)
    finally:
        trainer.DEVICE = previous_device
    print(f"[fisher] model loaded device={device} dtypes={sorted({str(p.dtype) for p in model.parameters()})}",
          file=sys.stderr, flush=True)
    from bfas.behavior.microupdate import (evaluation_score, save_initial_state,
                                         _frozen_hash, _model_settings, _restore)

    initial_hash = None
    if args.init_state:
        print("[fisher] restoring initial state; verifying frozen checkpoint hash", file=sys.stderr, flush=True)
        state = torch.load(args.init_state, map_location="cpu", weights_only=True)
        if _frozen_hash(model) != state["frozen_hash"] or _model_settings(model) != state["model_settings"]:
            raise ValueError("initial-state frozen checkpoint/model settings mismatch")
        _restore(model, state)
        initial_hash = file_hash(args.init_state)
        del state
    elif args.save_initial_state and not args.benchmark_rows:
        print("[fisher] saving initial state; hashing frozen checkpoint (one-time CPU work)", file=sys.stderr, flush=True)
        initial_hash = save_initial_state(model, args.save_initial_state)
    model.config.use_cache = False
    model.eval()
    print("[fisher] loading tokenizer", file=sys.stderr, flush=True)
    tokenizer = load_tokenizer(args.model_id)
    preprocessing = []
    tokens = 0

    def score(student, row):
        nonlocal tokens
        target = rejected_row(row) if args.owner == "student" else row
        prompt_tokens = len(prompt_token_ids(tokenizer, target))
        if prompt_tokens == 0:
            raise ValueError("an initial prompt token is required for causal response scoring")
        response_tokens = len(tokenizer(target["response"], add_special_tokens=False)["input_ids"])
        ids, labels = encode(tokenizer, target, device=device)
        if ids.device != device or labels.device != device:
            raise ValueError(f"encoded inputs and labels must be on {device}")
        tokens += ids.numel()
        preprocessing.append(dict(pool_index=len(preprocessing), prompt_tokens=prompt_tokens,
                                  prompt_tokens_dropped=max(0, prompt_tokens - prompt_cap),
                                  response_tokens_dropped=max(0, response_tokens - MAX_RESPONSE_TOKENS),
                                  effective_loss_tokens=int((labels[:, 1:] != -100).sum())))
        return evaluation_score(student, ids, labels, ell=args.ell)

    print(f"[fisher] starting forward/backward rows={len(rows)} device={device}", file=sys.stderr, flush=True)
    estimate = accumulate_fisher(model, rows, score, batch_rows=args.batch_rows,
                                 max_grad_norm=args.max_grad_norm, device=device,
                                 progress_every=args.progress_every, total_rows=len(rows),
                                 token_count=lambda: tokens, benchmark_rows=args.benchmark_rows)
    if args.benchmark_rows:
        return
    provenance.update(seed=args.seed, init_state_path=str(args.init_state or args.save_initial_state)
                      if (args.init_state or args.save_initial_state) else None,
                      init_state_hash=initial_hash, model_builder="appworld_train.build_lora_model",
                      model_mode="eval", model_revision=getattr(model.config, "_commit_hash", None),
                      model_parameter_dtypes=sorted({str(p.dtype) for p in model.parameters()}),
                      pool_role=args.pool_role, split_verification="explicit row labels checked; pool role caller-declared",
                      batch_rows=args.batch_rows, active_sequences=1,
                      prompt_cap=prompt_cap, prompt_side=prompt_side, response_cap=MAX_RESPONSE_TOKENS,
                      eos_included=True, preprocessing=preprocessing,
                      effective_loss_tokens=sum(r["effective_loss_tokens"] for r in preprocessing),
                      gradient_checkpointing_env=os.environ.get("AW_GRAD_CKPT", "0"),
                      memory_estimate=memory_estimate(model))
    print(f"[fisher] saving Fisher to {args.output}", file=sys.stderr, flush=True)
    manifest = save_fisher(estimate, args.output, model_id=args.model_id,
                           pool_content_hash=provenance["pool_content_hash"], ell=args.ell,
                           owner=args.owner, eps=args.eps, provenance=provenance)
    print(json.dumps(dict(output=str(args.output), manifest=str(args.output) + ".manifest.json",
                          n_rows=estimate.n_rows, parameter_count=manifest["parameter_count"],
                          layout_hash=manifest["layout_hash"]), indent=2))


if __name__ == "__main__":
    main()
