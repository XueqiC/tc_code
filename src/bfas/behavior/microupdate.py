"""Independent one-source updates using the existing trainer's pairwise loss.

Loss = -logsigmoid(beta * ((ell_T - ell_S) - (ell_T0 - ell_S0))).
``ell`` is a REQUIRED shared convention for gradients and evaluation:
``full_sequence`` (default) sums scored response log probabilities including
EOS; ``length_normalized`` divides by the actual shifted, unmasked token count.
Both use appworld_train.completion_log_prob, without changing its logit precision.
evaluation_score is the same callable used by the training loss. No environment
DDPO weighting, reference-free mode or span-only ablation is inherited.

save_initial_state must be called ONCE before sources are run. It records all
trainables (both LoRA factors), all buffers, and the frozen checkpoint digest.
Each run calls a zero-argument base_model_factory and restores the saved state;
the GPU driver supplies a verified ResidentState to avoid per-source disk reads
and full checkpoint hashes. Standalone factories verify frozen weights each run; AdamW and a scheduler are newly constructed.
Factories must load the same architecture and fixed model configuration. Passing
None uses imported build_lora_model(model_id, init_seed). No previous optimizer
or scheduler state is accepted. Checkpoint hashes are computed, not inferred
from a model ID. Full frozen-weight verification trades startup time for safety.

source_rows is one event or an explicitly identified fixed-size package. Rows
cycle in supplied order, averaging batch_size * accumulation equally weighted
pairs per optimizer step. Each pair is forwarded separately to match the trainer
and avoid padding differences. There is no other training data or teacher call.
Prompt head/tail and response caps use the imported encode; prompt truncation
requires explicit allow_prompt_truncation=True, and all truncation is recorded.

Precision is native parameter storage (no hidden autocast/master-weight change).
Clipping is optional and happens once per accumulated optimizer step. The
constant scheduler matches the existing trainer; linear decay is explicit.
Probe KL is KL(updated || initial), over the FULL vocabulary, averaged across
scored response positions on fixed teacher-forced probe contexts, not rollout KL.
Probe reference probabilities are kept on the update device. This is not an environment
behavior evaluation or an assertion that off-probe behavior is preserved.

The runner serializes in-process access to trainer environment variables and
saves/restores Python, NumPy and torch RNG states and deterministic flags.
Other threads must not mutate these process globals concurrently. GPU use is
explicit via cfg.device; deterministic-kernel errors propagate, never silently
relaxing precision/batch/determinism. dry_run never constructs a model. Existing
run directories are immutable; completed runs require resume=True and matching
identities. Partial/interrupted directories are refused rather than overwritten.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
import faulthandler
import json
import math
import os
from pathlib import Path
import random
import signal
import sys
import threading
import time
from typing import Any
import uuid
import warnings

import numpy as np
import torch
import torch.nn.functional as F

from appworld_train import (DEVICE as TRAINER_DEVICE, MAX_RESPONSE_TOKENS, build_lora_model, completion_log_prob,
                            encode, load_tokenizer, prompt_token_ids, rejected_row)
from .deltas import (Delta, canonical_hash, capture_delta, file_hash, layout_hash,
                     load_delta, parameter_layout, save_delta, tensor_state_hash,
                     write_json_exclusive)
from .provenance import repo_commit


_RUN_LOCK = threading.RLock()


def install_stack_dump():
    """Works without ptrace: kill -USR1 PID prints every Python thread."""
    if hasattr(signal, "SIGUSR1"):
        stream = sys.stderr
        try:
            stream.fileno()
        except (AttributeError, OSError, ValueError):
            stream = sys.__stderr__
        faulthandler.register(signal.SIGUSR1, file=stream, all_threads=True)


def cap_cpu_threads(count=8):
    if type(count) is not int or count < 1:
        raise ValueError("cpu_threads must be a positive integer")
    # Never expand an already smaller pool. Set once, before model/checker work.
    if torch.get_num_threads() > count:
        torch.set_num_threads(count)
    return torch.get_num_threads()


class Progress:
    """Flushed stderr events; synchronized phase timings include CUDA work."""
    def __init__(self, device, profile=False):
        self.device, self.profile, self.elapsed = torch.device(device), profile, {}
        self.on_phase = None

    def emit(self, phase, event, **fields):
        stamp = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        details = " ".join(f"{key}={value}" for key, value in fields.items())
        print(f"[{stamp}] {phase} {event} {details}".rstrip(), file=sys.stderr, flush=True)

    @contextmanager
    def phase(self, name, **fields):
        self.emit(name, "start", **fields)
        start = time.perf_counter()
        try:
            yield fields
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
        except BaseException as exc:
            fields["error"] = type(exc).__name__
            raise
        finally:
            elapsed = time.perf_counter() - start
            if self.profile:
                self.elapsed[name] = self.elapsed.get(name, 0.0) + elapsed
            self.emit(name, "failed" if "error" in fields else "done", seconds=f"{elapsed:.6f}", **fields)
            if self.profile and self.on_phase is not None:
                self.on_phase()


def assert_update_device(model, batches, device):
    """Check placement without moving a model or silently repairing a batch."""
    expected = torch.device(device)
    if expected.type == "cuda" and expected.index is None:
        expected = torch.device("cuda", torch.cuda.current_device())
    def check(tensor, name):
        if tensor.device != expected:
            raise ValueError(f"micro_update device assertion: {name} is on {tensor.device}, expected {expected}")
    for name, parameter in _trainables(model).items():
        check(parameter, f"trainable {name}")
    def visit(value, name):
        if isinstance(value, torch.Tensor):
            check(value, name)
        elif isinstance(value, dict):
            for key, child in value.items():
                visit(child, f"{name}.{key}")
        elif isinstance(value, (tuple, list)):
            for index, child in enumerate(value):
                visit(child, f"{name}[{index}]")
    visit(batches, "batch")


class ResidentState:
    """Run-local cache, constructed after the driver verifies initialization.

    Owns device snapshots of adapters/buffers, never a copy of the frozen model.
    Frozen tensor versions/identities guard cache reuse without reading 4B weights.
    """
    def __init__(self, model, state, path, commit):
        self.model, self.commit = model, commit
        self.init_hash = file_hash(path)
        self.base_hash = tensor_state_hash(state["trainables"])
        self.state = dict(state)
        for key, destination in (("trainables", dict(model.named_parameters())), ("buffers", dict(model.named_buffers()))):
            self.state[key] = {n: t.to(destination[n].device, copy=True) for n, t in state[key].items()}
        _restore(model, self.state, verify_hash=False)
        self.checkpoint_hash = tensor_state_hash(model.state_dict())
        self.frozen_versions = self._versions()

    def _versions(self):
        return [(n, id(p), p.data_ptr(), p._version, p.device, p.dtype, tuple(p.shape))
                for n, p in self.model.named_parameters() if not p.requires_grad]

    def restore(self):
        if self._versions() != self.frozen_versions:
            raise ValueError("resident frozen checkpoint changed")
        _restore(self.model, self.state, verify_hash=False)


@dataclass
class MicroUpdateConfig:
    model_id: str
    init_state_path: str | Path
    source_id: str
    steps: int = 3
    seed: int = 0
    init_seed: int = 0
    lr: float = 5e-6
    beta: float = 0.1
    batch_size: int = 1
    accumulation: int = 1
    weight_decay: float = 0.01
    adam_betas: tuple[float, float] = (0.9, 0.999)
    adam_eps: float = 1e-8
    scheduler: str = "constant"
    max_grad_norm: float | None = None
    device: str = "cpu"
    precision: str = "float32"
    ell: str = "full_sequence"
    prompt_cap: int = 4096
    prompt_side: str = "head"
    response_cap: int = MAX_RESPONSE_TOKENS
    allow_prompt_truncation: bool = False
    tokenizer: Any = field(default=None, repr=False)
    tokenizer_id: str | None = None
    fingerprint_version: str = "behavior-gradient-v1"
    fisher_version: str | None = None
    damping: float | None = None
    basis_version: str = "unfitted"
    eval_config: dict = field(default_factory=dict)
    probe_rows: list[dict] = field(default_factory=list)
    probe_block_size: int = 128
    checkpoint_hash: str | None = None
    data_manifest_hash: str | None = None
    run_id: str | None = None
    output_dir: str | Path | None = None
    dry_run: bool = False
    resume: bool = False
    strict_init_state: bool = True
    cpu_threads: int = 8
    profile: bool = False
    progress: Any = field(default=None, repr=False)
    resident_state: Any = field(default=None, repr=False)


@dataclass
class MicroUpdateResult:
    delta: Delta | None
    manifest: dict


def _trainables(model):
    return {n: p for n, p in model.named_parameters() if p.requires_grad}


def _frozen_hash(model):
    return tensor_state_hash({n: p for n, p in model.named_parameters() if not p.requires_grad})


def _model_settings(model):
    """Fingerprint architecture/LoRA settings, excluding runtime placement/modes."""
    runtime_keys = {"gradient_checkpointing", "gradient_checkpointing_kwargs", "use_cache",
                    "is_training", "training", "device", "device_map", "hf_device_map",
                    "dtype", "torch_dtype"}
    def json_value(value):
        if isinstance(value, dict):
            return {str(k): json_value(v) for k, v in value.items() if k not in runtime_keys}
        if isinstance(value, (set, frozenset)):
            return sorted(json_value(v) for v in value)
        if isinstance(value, (tuple, list)):
            return [json_value(v) for v in value]
        if isinstance(value, (torch.dtype, Path)):
            return str(value)
        return value

    config = getattr(model, "config", None)
    record = dict(modules=[dict(name=name, type=f"{type(module).__module__}.{type(module).__qualname__}",
                               settings=module.extra_repr()) for name, module in model.named_modules()],
                  model_config=config.to_dict() if hasattr(config, "to_dict") else None,
                  peft_config={name: cfg.to_dict() for name, cfg in getattr(model, "peft_config", {}).items()})
    # PEFT enums subclass str; a JSON round trip removes those pickle globals
    # so the initialization can always be read with torch.load(weights_only=True).
    return json.loads(json.dumps(json_value(record), sort_keys=True, allow_nan=False))


def _settings_diff(saved, current, prefix=""):
    """List differing leaf keys, including keys present only in legacy states."""
    if isinstance(saved, dict) and isinstance(current, dict):
        result = []
        for key in sorted(saved.keys() | current.keys()):
            path = f"{prefix}.{key}" if prefix else key
            if key not in saved or key not in current:
                result.append(path)
            else:
                result.extend(_settings_diff(saved[key], current[key], path))
        return result
    if isinstance(saved, list) and isinstance(current, list) and len(saved) == len(current):
        return [key for i, (a, b) in enumerate(zip(saved, current))
                for key in _settings_diff(a, b, f"{prefix}[{i}]")]
    return [prefix] if saved != current else []


def _check_settings_diff(diff, strict):
    if diff:
        message = "initial-state model settings differ: " + ", ".join(diff)
        if strict:
            raise ValueError(message + " (--strict-init-state)")
        warnings.warn(message, UserWarning, stacklevel=2)


def save_initial_state(model, path):
    """Save immutable LoRA init + buffers and verified frozen-checkpoint identity."""
    path = Path(path)
    state = dict(format="behavior-init-v1", layout=parameter_layout(model),
                 layout_hash=layout_hash(model), frozen_hash=_frozen_hash(model),
                 model_settings=_model_settings(model),
                 trainables={n: p.detach().cpu().clone() for n, p in _trainables(model).items()},
                 buffers={n: p.detach().cpu().clone() for n, p in model.named_buffers()})
    state["trainables_hash"] = tensor_state_hash(state["trainables"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "xb") as handle:
        torch.save(state, handle)
    return file_hash(path)


def _restore(model, state, *, verify_hash=True):
    if state["format"] != "behavior-init-v1" or layout_hash(model) != state["layout_hash"]:
        raise ValueError("initial-state layout mismatch")
    if verify_hash and "trainables_hash" in state and tensor_state_hash(state["trainables"]) != state["trainables_hash"]:
        raise ValueError("initial-state trainables hash mismatch")
    for destination, saved in ((_trainables(model), state["trainables"]),
                                (dict(model.named_buffers()), state["buffers"])):
        if destination.keys() != saved.keys():
            raise ValueError("initial-state tensor names mismatch")
        with torch.no_grad():
            for name, tensor in destination.items():
                source = saved[name]
                if tensor.shape != source.shape or tensor.dtype != source.dtype:
                    raise ValueError(f"initial-state shape/dtype mismatch: {name}")
                tensor.copy_(source)
    model.zero_grad(set_to_none=True)


def evaluation_score(model, input_ids, labels, *, ell="full_sequence"):
    """The shared differentiable ell for training, probe gradients and scoring."""
    if ell not in {"full_sequence", "length_normalized"}:
        raise ValueError("ell must be full_sequence or length_normalized")
    count = int((labels[:, 1:] != -100).sum())
    if count == 0:
        raise ValueError("sequence has no effective loss tokens")
    value = completion_log_prob(model, input_ids, labels)
    return value if ell == "full_sequence" else value / count


def _seed(seed, cuda):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.default_generator.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)


@contextmanager
def _training_context(cfg):
    cuda = torch.device(cfg.device).type == "cuda"
    if cuda and not torch.cuda.is_available():
        raise ValueError("CUDA explicitly requested but unavailable")
    py_state, np_state = random.getstate(), np.random.get_state()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    cudnn = (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)
    overrides = {"AW_MAX_PROMPT_TOKENS": str(cfg.prompt_cap), "AW_TRUNCATE_SIDE": cfg.prompt_side}
    if cuda:
        overrides["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    previous = {k: os.environ.get(k) for k in overrides}
    devices = list(range(torch.cuda.device_count())) if cuda else []
    try:
        with torch.random.fork_rng(devices=devices):
            os.environ.update(overrides)
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            _seed(cfg.seed, cuda)
            yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = cudnn
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _config_record(cfg):
    # Do not deepcopy or serialize a tokenizer or runtime output destination.
    record = {k: v for k, v in vars(cfg).items() if k not in
              {"tokenizer", "output_dir", "run_id", "dry_run", "resume", "init_state_path",
               "resident_state", "progress", "profile"}}
    record["tokenizer_id"] = cfg.tokenizer_id or cfg.model_id
    record["eval_config"] = {**cfg.eval_config, "ell": cfg.ell, "precision": cfg.precision}
    return record


def _validate(cfg, rows):
    for name in ("steps", "batch_size", "accumulation", "prompt_cap", "probe_block_size", "cpu_threads"):
        value = getattr(cfg, name)
        if not isinstance(value, int) or value < (0 if name == "steps" else 1):
            raise ValueError(f"invalid {name}")
    if not rows or not cfg.source_id or not cfg.model_id:
        raise ValueError("one nonempty source unit and explicit source_id/model_id are required")
    if any(r.get("source_id", cfg.source_id) != cfg.source_id for r in rows):
        raise ValueError("source_rows mixes source units")
    if any(not r.get("response") or not r.get("_rejected") for r in rows):
        raise ValueError("each source row requires response and _rejected")
    if cfg.ell not in {"full_sequence", "length_normalized"}:
        raise ValueError("invalid ell")
    if cfg.eval_config.get("ell", cfg.ell) != cfg.ell:
        raise ValueError("gradient and evaluation ell must match")
    if cfg.eval_config.get("precision", cfg.precision) != cfg.precision:
        raise ValueError("training and evaluation precision must match")
    if cfg.precision not in {"float32", "bfloat16", "float16", "native"}:
        raise ValueError("unsupported precision")
    if cfg.prompt_side not in {"head", "tail"} or cfg.response_cap != MAX_RESPONSE_TOKENS:
        raise ValueError("prompt side invalid or response cap differs from imported trainer")
    if cfg.scheduler not in {"constant", "linear"}:
        raise ValueError("unsupported scheduler")
    if not all(math.isfinite(v) and v > 0 for v in (cfg.lr, cfg.beta, cfg.adam_eps)):
        raise ValueError("lr, beta and adam_eps must be finite and positive")
    if len(cfg.adam_betas) != 2 or not all(math.isfinite(v) and 0 <= v < 1 for v in cfg.adam_betas):
        raise ValueError("invalid Adam betas")
    if not math.isfinite(cfg.weight_decay) or cfg.weight_decay < 0:
        raise ValueError("invalid weight_decay")
    if cfg.max_grad_norm is not None and (not math.isfinite(cfg.max_grad_norm) or cfg.max_grad_norm <= 0):
        raise ValueError("invalid max_grad_norm")
    if not 0 <= cfg.seed < 2**32:
        raise ValueError("seed must fit NumPy's uint32 seed range")
    if cfg.run_id and (Path(cfg.run_id).name != cfg.run_id or cfg.run_id in {".", ".."}):
        raise ValueError("run_id must be a single directory name")


def _encode_row(tokenizer, row, cfg):
    n_prompt = len(prompt_token_ids(tokenizer, row))
    n_response = len(tokenizer(row["response"], add_special_tokens=False)["input_ids"])
    if n_prompt == 0:
        raise ValueError("an initial prompt token is required for causal response scoring")
    if n_prompt > cfg.prompt_cap and not cfg.allow_prompt_truncation:
        raise ValueError("prompt exceeds cap; explicitly allow_prompt_truncation to proceed")
    ids, labels = encode(tokenizer, row, device=cfg.device)
    return (ids, labels), dict(prompt_tokens=n_prompt,
                               prompt_tokens_dropped=max(0, n_prompt - cfg.prompt_cap),
                               response_tokens_dropped=max(0, n_response - cfg.response_cap),
                               effective_loss_tokens=int((labels[:, 1:] != -100).sum()))


def _probe_logps(model, encoded, block_size):
    ids, labels = encoded
    logits = model(input_ids=ids).logits[:, :-1, :]
    positions = (labels[0, 1:] != -100).nonzero().flatten()
    for start in range(0, len(positions), block_size):
        yield F.log_softmax(logits[0, positions[start:start + block_size]].float(), dim=-1)


def run_micro_update(base_model_factory, source_rows, cfg) -> MicroUpdateResult:
    """Execute exactly cfg.steps optimizer steps from the saved common init.

    cfg may be MicroUpdateConfig or a mapping. output_dir is a root under which
    run_id is created. Without it, the delta and manifest are returned in memory.
    resume validates artifacts and identities without repeating training.
    """
    cfg = MicroUpdateConfig(**cfg) if isinstance(cfg, dict) else cfg
    rows = list(source_rows)
    _validate(cfg, rows)
    if not cfg.dry_run:
        cap_cpu_threads(cfg.cpu_threads)
        if cfg.progress is None:
            install_stack_dump()
    progress = cfg.progress or Progress(cfg.device, cfg.profile)
    prefix = f"source/{cfg.source_id}"
    resident = cfg.resident_state
    config_record = _config_record(cfg)
    config_hash, rows_hash = canonical_hash(config_record), canonical_hash(rows)
    init_hash = resident.init_hash if resident else file_hash(cfg.init_state_path)
    commit = resident.commit if resident else repo_commit(Path(__file__).resolve().parents[3])
    identity = dict(config_hash=config_hash, source_content_hash=rows_hash,
                    init_state_hash=init_hash, commit=commit["commit"])
    metadata = {**identity, **commit}
    run_id = cfg.run_id or f"micro-{uuid.uuid4().hex}"
    if cfg.dry_run:
        manifest = dict(status="dry_run", run_id=run_id, **metadata, models=1, source_units=1,
                        source_rows=len(rows), probe_count=len(cfg.probe_rows), steps=cfg.steps,
                        max_rollouts=0, new_teacher_tokens=0,
                        dependencies=[str(cfg.init_state_path), cfg.model_id],
                        resources={"device": cfg.device, "optimizer_steps": cfg.steps,
                                   "training_pair_forwards": 2 * cfg.steps * cfg.batch_size * cfg.accumulation,
                                   "gpu_hours": None, "environment_jobs": 0,
                                   "estimate_note": "GPU time requires measured model throughput"})
        print(json.dumps(manifest, sort_keys=True))
        return MicroUpdateResult(None, manifest)
    if base_model_factory is None and torch.device(cfg.device).type != torch.device(TRAINER_DEVICE).type:
        raise ValueError(f"imported build_lora_model uses {TRAINER_DEVICE}; supply a factory for {cfg.device}")
    run_dir = Path(cfg.output_dir) / run_id if cfg.output_dir is not None else None
    if run_dir is not None and run_dir.exists():
        state_path = run_dir / "state.manifest.json"
        if not cfg.resume or not state_path.exists():
            raise FileExistsError("existing or interrupted run; use a new run_id")
        saved = json.loads(state_path.read_text())
        if saved.get("status") != "complete" or any(saved.get(k) != v for k, v in identity.items()):
            raise ValueError("resume identity/status mismatch")
        delta = load_delta(run_dir / "delta.safetensors")
        if delta.manifest["metadata"] != saved:
            raise ValueError("run/delta manifest mismatch")
        return MicroUpdateResult(delta, saved)
    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=False)
        write_json_exclusive(run_dir / "started.json", dict(run_id=run_id, status="started", **metadata))
    with _RUN_LOCK, _training_context(cfg):
        with progress.phase(f"{prefix}/state_restore"):
            state = resident.state if resident else torch.load(cfg.init_state_path, map_location="cpu", weights_only=True)
            model = (base_model_factory() if base_model_factory is not None else
                     build_lora_model(cfg.model_id, cfg.init_seed))
            if resident:
                if model is not resident.model:
                    raise ValueError("resident cache belongs to a different model")
                resident.restore()
            else:
                if _frozen_hash(model) != state["frozen_hash"]:
                    raise ValueError("factory frozen checkpoint differs from saved initialization")
                _restore(model, state)
            settings = _model_settings(model)
            settings_diff = _settings_diff(state["model_settings"], settings)
            _check_settings_diff(settings_diff, cfg.strict_init_state)
        if hasattr(model, "config"):
            model.config.use_cache = False
        parameters = list(_trainables(model).values())
        actual_device = torch.device(cfg.device)
        if any(p.device.type != actual_device.type or
               (actual_device.index is not None and p.device.index != actual_device.index)
               for p in model.parameters()):
            raise ValueError("factory device differs from cfg.device")
        dtypes = sorted({str(p.dtype).removeprefix("torch.") for p in model.parameters() if p.is_floating_point()})
        trainable_dtypes = sorted({str(p.dtype).removeprefix("torch.") for p in parameters})
        if cfg.precision != "native" and dtypes != [cfg.precision]:
            raise ValueError("factory storage precision differs from cfg.precision; use native for mixed storage")
        checkpoint_hash = resident.checkpoint_hash if resident else tensor_state_hash(model.state_dict())
        if cfg.checkpoint_hash is not None and cfg.checkpoint_hash != checkpoint_hash:
            raise ValueError("checkpoint hash mismatch")
        with progress.phase(f"{prefix}/encode"):
            tokenizer = cfg.tokenizer if cfg.tokenizer is not None else load_tokenizer(cfg.model_id)
            pairs, preprocessing = [], []
            for row in rows:
                chosen, ct = _encode_row(tokenizer, row, cfg)
                rejected, rt = _encode_row(tokenizer, rejected_row(row), cfg)
                pairs.append((chosen, rejected))
                preprocessing.append(dict(chosen=ct, rejected=rt))
            probes, probe_preprocessing = [], []
            for row in cfg.probe_rows:
                encoded, counts = _encode_row(tokenizer, row, cfg)
                probes.append(encoded)
                probe_preprocessing.append(counts)
        assert_update_device(model, (pairs, probes), cfg.device)
        model.eval()
        _seed(cfg.seed, actual_device.type == "cuda")
        with progress.phase(f"{prefix}/reference_forward"):
            with torch.no_grad():
                refs = [float(evaluation_score(model, *t, ell=cfg.ell) -
                              evaluation_score(model, *s, ell=cfg.ell)) for t, s in pairs]
                probe_refs = [[block for block in _probe_logps(model, probe, cfg.probe_block_size)]
                              for probe in probes]
        # Reference passes must not consume the training RNG or alter buffers.
        with progress.phase(f"{prefix}/reference_restore"):
            if resident:
                resident.restore()
            else:
                _restore(model, state)
        _seed(cfg.seed, actual_device.type == "cuda")
        model.train()
        with progress.phase(f"{prefix}/optimizer_setup"):
            optimizer = torch.optim.AdamW(parameters, lr=cfg.lr, betas=cfg.adam_betas,
                                         eps=cfg.adam_eps, weight_decay=cfg.weight_decay)
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lambda step: 1.0 if cfg.scheduler == "constant" else max(0.0, 1 - step / max(1, cfg.steps)))
        losses, gradient_norms, clip_scales, learning_rates = [], [], [], []
        effective_tokens, cursor = 0, 0
        pairs_per_step = cfg.batch_size * cfg.accumulation
        # Validate labels as well as input IDs, including later accumulation rows.
        assert_update_device(model, (pairs, probes), cfg.device)
        for step in range(cfg.steps):
            with progress.phase(f"{prefix}/optimizer_step/{step + 1}") as timing:
                optimizer.zero_grad(set_to_none=True)
                step_loss = 0.0
                learning_rates.append(float(optimizer.param_groups[0]["lr"]))
                for _ in range(pairs_per_step):
                    index = cursor % len(pairs)
                    cursor += 1
                    teacher, student = pairs[index]
                    margin = (evaluation_score(model, *teacher, ell=cfg.ell) -
                              evaluation_score(model, *student, ell=cfg.ell))
                    loss = -F.logsigmoid(cfg.beta * (margin - refs[index])) / pairs_per_step
                    if not torch.isfinite(loss):
                        raise ValueError("nonfinite pairwise loss")
                    loss.backward()
                    step_loss += float(loss.detach())
                    effective_tokens += sum(preprocessing[index][side]["effective_loss_tokens"]
                                            for side in ("chosen", "rejected"))
                norm = float(torch.nn.utils.clip_grad_norm_(parameters, cfg.max_grad_norm or float("inf"),
                                                            error_if_nonfinite=True))
                gradient_norms.append(norm)
                clip_scales.append(min(1.0, cfg.max_grad_norm / (norm + 1e-6)) if cfg.max_grad_norm else 1.0)
                optimizer.step()
                scheduler.step()
                losses.append(step_loss)
                timing["loss"] = step_loss
        optimizer.zero_grad(set_to_none=True)
        # LoRA-only deltas cannot replay mutated running-stat buffers.
        if any(not torch.equal(buffer, state["buffers"][name].to(buffer.device))
               for name, buffer in model.named_buffers()):
            raise ValueError("training mutated buffers; a parameter-only delta cannot replay this model")
        model.eval()
        policy_kl, probe_tokens = None, 0
        with progress.phase(f"{prefix}/kl", probes=len(probes)):
            with torch.no_grad():
                if probes:
                    total = 0.0
                    for probe, reference in zip(probes, probe_refs):
                        for current, initial in zip(_probe_logps(model, probe, cfg.probe_block_size), reference):
                            initial = initial.to(current.device)
                            total += float((current.exp() * (current - initial)).double().sum())
                            probe_tokens += len(current)
                    policy_kl = total / probe_tokens
        clipping = dict(status="measured", max_grad_norm=cfg.max_grad_norm,
                        pre_clip_norms=gradient_norms, scales=clip_scales,
                        count=sum(s < 1.0 for s in clip_scales), steps=cfg.steps)
        manifest = dict(status="complete", run_id=run_id, **metadata, config=config_record,
                        data_manifest_hash=cfg.data_manifest_hash or rows_hash,
                        checkpoint_hash=checkpoint_hash, layout_hash=layout_hash(model),
                        model_settings=settings, model_settings_hash=canonical_hash(settings),
                        init_state_settings_diff=settings_diff,
                        parameter_layout=parameter_layout(model), source_ids=[cfg.source_id],
                        event_ids=[r.get("event_id") for r in rows], seed=cfg.seed,
                        optimizer_config=dict(name="AdamW", lr=cfg.lr, betas=list(cfg.adam_betas),
                                              eps=cfg.adam_eps, weight_decay=cfg.weight_decay,
                                              amsgrad=False, initial_state="empty"),
                        scheduler_config=dict(name=cfg.scheduler, initial_state="fresh", steps=cfg.steps),
                        precision=cfg.precision, parameter_dtypes=dtypes, trainable_dtypes=trainable_dtypes,
                        score_precision="native logits; fp32 completion reduction", ell=cfg.ell,
                        batch_size=cfg.batch_size, accumulation=cfg.accumulation,
                        effective_loss_tokens=effective_tokens, steps=cfg.steps,
                        loss_trace=losses, learning_rates=learning_rates, clipping_stats=clipping,
                        preprocessing=dict(prompt_cap=cfg.prompt_cap, prompt_side=cfg.prompt_side,
                                           response_cap=cfg.response_cap, source_counts=preprocessing,
                                           probe_counts=probe_preprocessing),
                        evaluation_config=config_record["eval_config"],
                        policy_kl=policy_kl, probe_loss_tokens=probe_tokens,
                        policy_kl_definition="KL(updated||initial), full vocabulary, mean response position",
                        missing_values=["policy_kl"] if policy_kl is None else [],
                        new_teacher_tokens=0, max_rollouts=0)
        with progress.phase(f"{prefix}/delta_capture") as timing:
            delta = capture_delta(model, state["trainables"], clipping_stats=clipping, metadata=manifest,
                                  base_hash=resident.base_hash if resident else None)
            timing["bytes"] = sum(t.numel() * t.element_size() for t in
                                  [*delta.tensors.values(), *delta.roundoff.values()])
        if cfg.profile:
            manifest["phase_elapsed_seconds"] = {k: v for k, v in progress.elapsed.items() if k.startswith(prefix + "/")}
        if run_dir is not None:
            with progress.phase(f"{prefix}/delta_save") as timing:
                save_delta(None, None, run_dir / "delta.safetensors", delta=delta)
                timing["bytes"] = (run_dir / "delta.safetensors").stat().st_size
            write_json_exclusive(run_dir / "state.manifest.json", manifest)
        return MicroUpdateResult(delta, manifest)
