"""E3 tier-1: boundary-guided versus standard trajectory distillation.

The experiment derives gradient and embedding boundaries from the stable
GSM8K-code fit/calibration split, selects fixed-token training sets under the
requested conditions, trains a fresh LoRA student for each condition, and
evaluates in-boundary capability and out-of-boundary behavior.
"""

import argparse
import ast
import gc
import hashlib
import json
import math
import os
import random
import re
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

import boundary
import features
import verifier
from distill_pilot import MODEL, encode, last_number, split


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "configs" / "e3_tier1.yaml"
CONDITIONS = (
    "base", "A_all", "B_random", "C_emb", "D_grad", "E_less", "F_refusal",
    "H_nll", "E_less_std", "H_smartad_std", "D_grad_iter",
    "D_grad_pre", "D_grad_cov", "D_less_bnd", "D_atom",
    "F_uni_boot", "F_emb_boot", "F_atom_boot", "F_atom_boot2",
    "F_atom_boot3", "F_atom_boot4", "F_atom_boot5", "F_atom_boot6",
    "F_atom_boot7", "F_selfinst",
) + tuple(
    f"M_{a}_{c}"
    for a in ("selfinst", "evol", "llm2llm", "ourscorr", "ours")
    for c in ("plain", "alpagasus", "less", "ours")
)
COLORS = {
    "base": "#6b6a63",
    "A_all": "#eb6834",
    "B_random": "#eda100",
    "C_emb": "#1baf7a",
    "D_grad": "#2a78d6",
    "E_less": "#9085e9",
    "E_less_std": "#6b5bd6",
    "F_refusal": "#e87ba4",
    "H_nll": "#c98500",
    "H_smartad_std": "#a86e00",
    "D_grad_iter": "#134a8e",
    "D_grad_pre": "#0e6f6a",
    "D_grad_cov": "#7a4fb3",
    "D_less_bnd": "#0b8457",
    "D_atom": "#b8860b",
    "F_uni_boot": "#8a8a8a",
    "F_emb_boot": "#2aa198",
    "F_atom_boot": "#c0392b",
    "F_atom_boot2": "#7d1f14",
    "F_atom_boot3": "#4a0e08",
    "F_selfinst": "#6b6a63",
}
SHORT_LABELS = {
    "base": "base",
    "A_all": "A",
    "B_random": "B",
    "C_emb": "C",
    "D_grad": "D",
    "E_less": "E",
    "E_less_std": "E*",
    "F_refusal": "F",
    "H_nll": "H",
    "H_smartad_std": "H*",
    "D_grad_iter": "D-iter",
    "D_grad_pre": "D-pre",
    "D_grad_cov": "D-cov",
    "D_less_bnd": "D-bnd",
    "D_atom": "D-atom",
    "F_uni_boot": "F-uni",
    "F_emb_boot": "F-emb",
    "F_atom_boot": "F-atom",
    "F_atom_boot2": "F-atom2",
    "F_atom_boot3": "F-atom3",
    "F_selfinst": "SelfInst",
}
BOOT_CONDS = (
    "F_uni_boot", "F_emb_boot", "F_atom_boot", "F_atom_boot2",
    "F_atom_boot3", "F_atom_boot4", "F_atom_boot5", "F_atom_boot6",
    "F_atom_boot7", "F_selfinst",
)
for _a in ("selfinst", "evol", "llm2llm", "ourscorr", "ours"):
    for _c in ("plain", "alpagasus", "less", "ours"):
        COLORS[f"M_{_a}_{_c}"] = "#555555"
        SHORT_LABELS[f"M_{_a}_{_c}"] = f"{_a[:4]}x{_c[:4]}"
COLORS["F_atom_boot4"] = "#2d0a06"
SHORT_LABELS["F_atom_boot4"] = "F-atom4"
COLORS["F_atom_boot5"] = "#1a0503"
SHORT_LABELS["F_atom_boot5"] = "F-atom5"
COLORS["F_atom_boot6"] = "#33110a"
SHORT_LABELS["F_atom_boot6"] = "F-atom6"
COLORS["F_atom_boot7"] = "#000000"
SHORT_LABELS["F_atom_boot7"] = "F-atom7"
BOOT_TEACHER = "deepseek-v4-pro"
_BOOT_STATE = {}
MUTED = "#6b6a63"
LESS_WARMUP_EPOCHS = 4
LESS_WARMUP_FRACTION = 0.05
LESS_WARMUP_LR = 2e-4
LESS_ADAM_EPS = 1e-8


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG,
        help="experiment YAML (default: configs/e3_tier1.yaml)",
    )
    return parser.parse_args()


def load_config(path):
    with path.open() as handle:
        config = yaml.safe_load(handle)
    validate_config(config)
    apply_environment_overrides(config)
    return config


def _environment_int(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer, got {value!r}") from error


def _environment_float(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        result = float(value)
    except ValueError as error:
        raise ValueError(f"{name} must be a number, got {value!r}") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return result


def apply_environment_overrides(config):
    """Resolve sweep parameters while retaining the tier-1 bare-run defaults."""
    conditions_text = os.environ.get("E3_CONDITIONS", ",".join(CONDITIONS))
    conditions = [value.strip() for value in conditions_text.split(",")]
    if not conditions or any(not value for value in conditions):
        raise ValueError("E3_CONDITIONS must be a non-empty comma-separated list")
    unknown = [value for value in conditions if value not in CONDITIONS]
    if unknown:
        raise ValueError(
            f"unknown E3_CONDITIONS {unknown}; expected values from {CONDITIONS}"
        )
    if len(set(conditions)) != len(conditions):
        raise ValueError("E3_CONDITIONS must not contain duplicates")

    student = os.environ.get("E3_STUDENT", MODEL)
    budget = _environment_int("E3_BUDGET", 40000)
    seed = _environment_int("E3_SEED", 0)
    task = os.environ.get("E3_TASK", "gsm8k-code")
    task_filter = os.environ.get("E3_TASK_FILTER") or None
    tag = os.environ.get("E3_TAG", "")
    early_stop = _environment_int("E3_EARLYSTOP", 0)
    selection_lambda = _environment_float("E3_LAMBDA", 0.0)
    if not student:
        raise ValueError("E3_STUDENT must not be empty")
    if budget <= 0:
        raise ValueError(f"E3_BUDGET must be positive, got {budget}")
    if not task:
        raise ValueError("E3_TASK must not be empty")
    if "/" in tag or "\\" in tag:
        raise ValueError("E3_TAG must be a filename suffix, not a path")
    if early_stop not in (0, 1):
        raise ValueError(f"E3_EARLYSTOP must be 0 or 1, got {early_stop}")
    if selection_lambda < 0:
        raise ValueError(
            f"E3_LAMBDA must be non-negative, got {selection_lambda}"
        )

    config["model"] = student
    config["seed"] = seed
    config["conditions"] = conditions
    config["tag"] = tag
    config["selection"]["response_token_budget"] = budget
    config["selection"]["lambda"] = selection_lambda
    config["training"]["learning_rate"] = float(os.environ.get("E3_LR", config["training"]["learning_rate"]))
    iter_rounds = _environment_int("E3_ITER", 2)
    if iter_rounds < 2:
        raise ValueError(f"E3_ITER must be >= 2, got {iter_rounds}")
    config["selection"]["iter_rounds"] = iter_rounds
    iter_mode = os.environ.get("E3_ITER_MODE", "fresh")
    if iter_mode not in ("fresh", "cum"):
        raise ValueError(f"E3_ITER_MODE must be fresh or cum, got {iter_mode}")
    config["selection"]["iter_mode"] = iter_mode
    config["training"]["early_stop"] = early_stop
    config["task_boundary"]["domain"] = task
    config["task_boundary"]["filter"] = task_filter
    spec_rows = _environment_int("E3_SPEC_ROWS", 0)
    if spec_rows:
        if spec_rows < 5:
            raise ValueError(
                f"E3_SPEC_ROWS must be >= 5, got {spec_rows}"
            )
        fit_n = max(1, min(spec_rows - 1, int(round(spec_rows * 0.7))))
        config["task_boundary"]["spec_rows"] = spec_rows
        config["task_boundary"]["fit_rows"] = fit_n
        config["task_boundary"]["calibration_rows"] = spec_rows - fit_n

    output_directory = f"results/e3_tier1{tag}"
    config["outputs"].update({
        "directory": output_directory,
        "metrics": f"{output_directory}/metrics.json",
        "summary": f"{output_directory}/summary.md",
        "figure": f"results/figs/e3_tier1{tag}.png",
    })


def resolve_task_rows(config):
    """Select T from the stable pilot split, optionally by response substring."""
    task = config["task_boundary"]
    non_test, _ = split(task["domain"])
    task_filter = task["filter"]
    eligible = non_test
    if task_filter is not None:
        needle = task_filter.casefold()
        eligible = [
            row for row in non_test if needle in row["response"].casefold()
        ]

    original_spec = task["spec_rows"]
    original_fit = task["fit_rows"]
    spec_n = min(original_spec, len(eligible))
    if spec_n < 2:
        raise ValueError(
            f"task boundary needs at least two spec rows; found {spec_n} "
            f"for domain={task['domain']!r} filter={task_filter!r}"
        )
    if spec_n != original_spec:
        fit_fraction = original_fit / original_spec
        fit_n = max(1, min(spec_n - 1, int(round(spec_n * fit_fraction))))
        task["spec_rows"] = spec_n
        task["fit_rows"] = fit_n
        task["calibration_rows"] = spec_n - fit_n
    return eligible[:spec_n], len(non_test), len(eligible)


def print_resolved_config(config):
    task = config["task_boundary"]
    resolved = {
        "student": config["model"],
        "budget": config["selection"]["response_token_budget"],
        "seed": config["seed"],
        "conditions": config["conditions"],
        "task": task["domain"],
        "task_filter": task["filter"],
        "spec_rows": task["spec_rows"],
        "fit_rows": task["fit_rows"],
        "calibration_rows": task["calibration_rows"],
        "lambda": config["selection"]["lambda"],
        "early_stop": config["training"]["early_stop"],
        "tag": config["tag"],
        "output_directory": config["outputs"]["directory"],
        "figure": config["outputs"]["figure"],
    }
    print(f"[config] {json.dumps(resolved, sort_keys=True)}", flush=True)


def validate_config(config):
    """Fail early if the YAML no longer describes the requested E3 protocol."""
    expected = {
        ("model",): MODEL,
        ("device",): "cuda",
        ("dtype",): "bfloat16",
        ("seed",): 0,
        ("task_boundary", "spec_rows"): config["task_boundary"]["spec_rows"],
        ("task_boundary", "fit_rows"): config["task_boundary"]["fit_rows"],
        ("task_boundary", "calibration_rows"):
            config["task_boundary"]["calibration_rows"],
        ("task_boundary", "energy"): 0.90,
        ("task_boundary", "alpha"): 0.10,
        ("selection", "response_token_budget"): 40000,
        ("selection", "refusal_fraction"): 0.15,
        ("training", "epochs"): 3,
        ("training", "batch_size"): 1,
        ("training", "gradient_accumulation"): 8,
        ("training", "learning_rate"): 2e-4,
        ("training", "lora", "r"): 16,
        ("training", "lora", "alpha"): 32,
        ("generation", "max_new_tokens"): 320,
    }
    for keys, value in expected.items():
        actual = config
        for key in keys:
            actual = actual[key]
        if actual != value:
            joined = ".".join(keys)
            raise ValueError(f"{joined} must be {value!r}, got {actual!r}")
    task = config["task_boundary"]
    if task["fit_rows"] + task["calibration_rows"] != task["spec_rows"]:
        raise ValueError("task fit and calibration rows must sum to spec_rows")
    if not math.isclose(task["energy"], boundary.ENERGY):
        raise ValueError(
            f"boundary.fit_subspace uses energy={boundary.ENERGY}; "
            f"config requested {task['energy']}"
        )


def resolve_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def progress(condition, phase, message):
    """Emit exactly one concise completion line per condition phase."""
    print(f"[{condition}][{phase}] {message}", flush=True)


def _pool_item(domain, prompt, response, teacher):
    return {
        "domain": str(domain),
        "prompt": str(prompt),
        "response": str(response),
        "teacher": str(teacher),
    }


def load_candidate_pool(config):
    """Load eligible teacher trajectories and stable non-test distractors."""
    pattern = config["data"]["pool_glob"]
    paths = sorted(ROOT.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"no pool files matched {pattern!r}")

    pool = []
    for path in paths:
        with path.open() as handle:
            for line_number, line in enumerate(handle, start=1):
                row = json.loads(line)
                code = row.get("code")
                if not isinstance(code, str) or not code.strip():
                    continue
                domain = row.get("domain", path.stem)
                if domain == "gsm8k-code" and row.get("verified") is not True:
                    continue
                prompt = row.get("prompt")
                if not isinstance(prompt, str) or not prompt.strip():
                    raise ValueError(f"missing prompt at {path}:{line_number}")
                teacher = row.get("teacher") or path.parent.name
                pool.append(_pool_item(domain, prompt, code.strip(), teacher))

    for domain in config["data"]["distractor_domains"]:
        non_test, _ = split(domain)
        if len(non_test) != 90:
            raise ValueError(f"expected 90 non-test {domain} rows, got {len(non_test)}")
        pool.extend(
            _pool_item(domain, row["prompt"], row["response"], "gold")
            for row in non_test
        )
    teachers = {item["teacher"] for item in pool}
    if teachers <= {"gold"}:
        raise RuntimeError(
            "candidate pool contains only gold seed items — teacher traces "
            "(data/pool_v0) are missing; refusing to run on a degenerate diet"
        )
    return pool


def _maybe_extend_pool_with_evol(pool):
    """Merge Evol-Instruct generations (data/evol_pool_v1.jsonl) when
    E3_EVOL_POOL=1. Changes the pool fingerprint, hence feature caches."""
    if os.environ.get("E3_EVOL_POOL") != "1":
        return pool
    path = ROOT / os.environ.get(
        "E3_EVOL_POOL_FILE", "data/evol_pool_v1.jsonl")
    if not path.is_file():
        raise RuntimeError(f"E3_EVOL_POOL=1 but {path} missing")
    added = 0
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            item = _pool_item(row["domain"], row["prompt"], row["response"],
                              row.get("teacher", "deepseek-v4-pro"))
            item["evol"] = 1
            pool.append(item)
            added += 1
    print(f"[setup][pool] evol extension rows={added}", flush=True)
    return pool


def composition_rows(rows):
    counts = Counter((row["teacher"], row["domain"]) for row in rows)
    return [
        {"teacher": teacher, "domain": domain, "n_items": int(count)}
        for (teacher, domain), count in sorted(counts.items())
    ]


def conformal_threshold(scores, alpha):
    """Lower split-conformal cutoff matching the boundary pilot."""
    scores = np.asarray(scores, dtype=np.float64)
    if scores.size == 0:
        raise ValueError("cannot calibrate an empty score set")
    k = int(np.floor(alpha * (len(scores) + 1)))
    return float(np.sort(scores)[max(k - 1, 0)])


def _less_rows_fingerprint(rows):
    """Hash the ordered, selection-relevant contents of a row collection."""
    digest = hashlib.sha256()
    for row in rows:
        identity = {
            key: row.get(key) for key in ("domain", "prompt", "response", "teacher")
        }
        digest.update(
            json.dumps(
                identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def _less_cache(config, task_rows, pool):
    """Return a seed-independent cache directory and its protocol manifest."""
    feat_cfg = config["gradient_features"]
    lora_cfg = config["training"]["lora"]
    manifest = {
        "version": 1,
        "model": config["model"],
        "pool_fingerprint": _less_rows_fingerprint(pool),
        "task_rows_fingerprint": _less_rows_fingerprint(task_rows),
        "pool_rows": len(pool),
        "task_rows": len(task_rows),
        "warmup_fraction": LESS_WARMUP_FRACTION,
        "warmup_epochs": LESS_WARMUP_EPOCHS,
        "warmup_learning_rate": LESS_WARMUP_LR,
        "lora": {
            "r": 16,
            "alpha": 32,
            "dropout": lora_cfg["dropout"],
            "bias": lora_cfg["bias"],
            "target_modules": lora_cfg["target_modules"],
        },
        "projection_dim": feat_cfg["projection_dim"],
        "max_prompt_tokens": feat_cfg["max_prompt_tokens"],
        "max_response_tokens": feat_cfg["max_response_tokens"],
        "adam_eps": LESS_ADAM_EPS,
    }
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    cache_key = hashlib.sha256(encoded).hexdigest()[:16]
    if os.environ.get("TC_USE_TMP") == "1" and os.environ.get("TMPDIR"):
        cache_root = Path(os.environ["TMPDIR"]) / f"less_cache{config['tag']}"
    else:
        # shared across tags/seeds: the manifest hash already keys the
        # model, pool fingerprint, and feature params. Per-tag isolation
        # recomputed ~50min of identical features for every run.
        cache_root = ROOT / "results" / "less_cache_shared"
    return cache_root / cache_key, manifest


def _build_less_model(config, warmup_seed):
    """Build the fixed r16/alpha32 LoRA model used by faithful LESS."""
    seed_everything(warmup_seed)
    model = AutoModelForCausalLM.from_pretrained(
        config["model"], torch_dtype=torch.bfloat16
    ).to(config["device"])
    lora_cfg = config["training"]["lora"]
    adapter = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=lora_cfg["dropout"],
        bias=lora_cfg["bias"],
        task_type="CAUSAL_LM",
        target_modules=lora_cfg["target_modules"],
    )
    return get_peft_model(model, adapter)


def _less_adapter_state(model):
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def _less_second_moments(model, optimizer):
    moments = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        state = optimizer.state.get(parameter, {})
        moment = state.get("exp_avg_sq")
        moments[name] = (
            torch.zeros_like(parameter, device="cpu", dtype=torch.float32)
            if moment is None
            else moment.detach().cpu().float().clone()
        )
    return moments


def _run_less_warmup(config, tokenizer, pool, checkpoint_paths, warmup_seed):
    """Train the deterministic 5% warmup and save four adapter/Adam states."""
    warmup_n = max(1, math.ceil(len(pool) * LESS_WARMUP_FRACTION))
    rng = np.random.default_rng(warmup_seed)
    warmup_indices = rng.choice(len(pool), size=warmup_n, replace=False)
    accumulation = config["training"]["gradient_accumulation"]
    model = _build_less_model(config, warmup_seed)
    model.config.use_cache = False
    model.train()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=LESS_WARMUP_LR,
    )
    optimizer_steps = 0
    try:
        print(
            f"[setup][E_less_std][warmup] rows={warmup_n}/{len(pool)} "
            f"epochs={LESS_WARMUP_EPOCHS}",
            flush=True,
        )
        for epoch, checkpoint_path in enumerate(checkpoint_paths, start=1):
            order = rng.permutation(warmup_indices)
            optimizer.zero_grad(set_to_none=True)
            for start in range(0, len(order), accumulation):
                chunk = order[start:start + accumulation]
                for pool_index in chunk:
                    input_ids, labels = encode(
                        tokenizer, pool[int(pool_index)], config["device"]
                    )
                    loss = model(input_ids=input_ids, labels=labels).loss / len(chunk)
                    loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
            torch.save(
                {
                    "epoch": epoch,
                    "optimizer_steps": optimizer_steps,
                    "warmup_indices": torch.as_tensor(warmup_indices.copy()),
                    "adapter_state": _less_adapter_state(model),
                    "second_moments": _less_second_moments(model, optimizer),
                },
                checkpoint_path,
            )
            print(
                f"[setup][E_less_std][warmup] epoch={epoch}/"
                f"{LESS_WARMUP_EPOCHS} optimizer_steps={optimizer_steps} saved",
                flush=True,
            )
    finally:
        del optimizer, model
        gc.collect()
        torch.cuda.empty_cache()


def _load_less_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    named_parameters = dict(model.named_parameters())
    adapter_state = checkpoint["adapter_state"]
    missing = sorted(set(adapter_state) - set(named_parameters))
    if missing:
        raise RuntimeError(
            f"LESS checkpoint has {len(missing)} unknown adapter parameters"
        )
    with torch.no_grad():
        for name, value in adapter_state.items():
            named_parameters[name].copy_(
                value.to(
                    device=named_parameters[name].device,
                    dtype=named_parameters[name].dtype,
                )
            )
    return checkpoint["second_moments"]


def _less_projections(lora_b, projection_dim, device):
    """Construct the same stable per-LoRA-B-module JL maps as features.py."""
    projections = {}
    for name, parameter in lora_b:
        generator = torch.Generator(device=device).manual_seed(
            features.stable_seed(name)
        )
        projections[name] = torch.randn(
            parameter.numel(),
            projection_dim,
            generator=generator,
            device=device,
            dtype=torch.float32,
        ) / math.sqrt(projection_dim)
    return projections


def _less_preconditioned_feature(lora_b, projections, second_moments):
    """Adam-precondition current LoRA-B gradients, then JL-project and unitize."""
    projected = []
    for name, parameter in lora_b:
        if parameter.grad is None:
            raise RuntimeError(f"LESS feature gradient is missing for {name}")
        moment = second_moments[name].to(
            device=parameter.device, dtype=torch.float32
        )
        preconditioned = parameter.grad.detach().flatten().float() / (
            moment.flatten().sqrt() + LESS_ADAM_EPS
        )
        projected.append(projections[name].T @ preconditioned)
    feature = torch.cat(projected)
    return feature / (feature.norm() + 1e-8)


def _extract_less_features(
    config, tokenizer, rows, checkpoint_paths, feature_paths, warmup_seed
):
    """Extract and cache Adam-preconditioned features at all four checkpoints."""
    feat_cfg = config["gradient_features"]
    model = _build_less_model(config, warmup_seed)
    lora_b = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if "lora_B" in name
    ]
    if not lora_b:
        raise RuntimeError("faithful LESS found no LoRA-B parameters")
    for name, parameter in model.named_parameters():
        parameter.requires_grad = "lora_B" in name
    projections = _less_projections(
        lora_b, feat_cfg["projection_dim"], config["device"]
    )
    model.config.use_cache = False
    model.eval()
    try:
        for epoch, (checkpoint_path, feature_path) in enumerate(
            zip(checkpoint_paths, feature_paths), start=1
        ):
            saved_moments = _load_less_checkpoint(model, checkpoint_path)
            second_moments = {
                name: saved_moments[name].to(
                    device=parameter.device, dtype=torch.float32
                )
                for name, parameter in lora_b
            }
            del saved_moments
            checkpoint_features = []
            print(
                f"[setup][E_less_std][features] checkpoint={epoch}/"
                f"{LESS_WARMUP_EPOCHS} rows={len(rows)}",
                flush=True,
            )
            for row_index, row in enumerate(rows, start=1):
                input_ids, labels = encode(tokenizer, row, config["device"])
                model.zero_grad(set_to_none=True)
                model(input_ids=input_ids, labels=labels).loss.backward()
                checkpoint_features.append(
                    _less_preconditioned_feature(
                        lora_b, projections, second_moments
                    ).cpu().numpy()
                )
                if row_index % 100 == 0 or row_index == len(rows):
                    print(
                        f"[setup][E_less_std][features] checkpoint={epoch}/"
                        f"{LESS_WARMUP_EPOCHS} rows={row_index}/{len(rows)}",
                        flush=True,
                    )
            np.save(
                feature_path,
                np.stack(checkpoint_features).astype(np.float32, copy=False),
            )
    finally:
        del projections, model
        gc.collect()
        torch.cuda.empty_cache()


def _load_valid_less_features(feature_paths, expected_rows):
    matrices = []
    expected_columns = None
    try:
        for path in feature_paths:
            matrix = np.load(path, allow_pickle=False)
            if matrix.ndim != 2 or matrix.shape[0] != expected_rows:
                return None
            if expected_columns is None:
                expected_columns = matrix.shape[1]
            if matrix.shape[1] != expected_columns:
                return None
            matrices.append(matrix.astype(np.float64, copy=False))
    except (OSError, ValueError):
        return None
    return matrices


def attach_less_std_scores(config, tokenizer, task_rows, pool):
    """Run or reuse faithful LESS warmup/features and attach InfAdam scores."""
    cache_dir, manifest = _less_cache(config, task_rows, pool)
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / "manifest.json"
    with manifest_path.open("w") as handle:
        json.dump(manifest, handle, indent=2)

    checkpoint_paths = [
        cache_dir / f"adapter_epoch_{epoch}.pt"
        for epoch in range(1, LESS_WARMUP_EPOCHS + 1)
    ]
    feature_paths = [
        cache_dir / f"features_epoch_{epoch}.npy"
        for epoch in range(1, LESS_WARMUP_EPOCHS + 1)
    ]
    combined = task_rows + pool
    matrices = _load_valid_less_features(feature_paths, len(combined))
    if matrices is not None:
        print(
            f"[setup][E_less_std][cache] hit={cache_dir.relative_to(ROOT)}",
            flush=True,
        )
    else:
        pool_fingerprint = manifest["pool_fingerprint"]
        warmup_seed = features.stable_seed(
            f"less-warmup-{config['model']}-{pool_fingerprint}"
        )
        if not all(path.is_file() for path in checkpoint_paths):
            _run_less_warmup(
                config, tokenizer, pool, checkpoint_paths, warmup_seed
            )
        else:
            print(
                f"[setup][E_less_std][cache] adapter-hit="
                f"{cache_dir.relative_to(ROOT)}",
                flush=True,
            )
        _extract_less_features(
            config,
            tokenizer,
            combined,
            checkpoint_paths,
            feature_paths,
            warmup_seed,
        )
        matrices = _load_valid_less_features(feature_paths, len(combined))
        if matrices is None:
            raise RuntimeError("failed to create valid faithful LESS feature cache")

    spec_n = len(task_rows)
    checkpoint_scores = []
    for matrix in matrices:
        matrix = boundary.unit(matrix)
        mean_spec = boundary.unit(
            np.mean(matrix[:spec_n], axis=0, keepdims=True)
        )[0]
        checkpoint_scores.append(matrix[spec_n:] @ mean_spec)
    scores = np.max(np.stack(checkpoint_scores), axis=0)
    if len(scores) != len(pool):
        raise RuntimeError("faithful LESS scores do not align with the pool")
    for row, score in zip(pool, scores):
        row["_less_std_score"] = float(score)

    if "D_atom" in config["conditions"]:
        # capability atoms over preconditioned features: one dictionary
        # serves boundary, utility, and monitoring. Demand = spec atom
        # mass; support = atoms carrying 90% of spec mass (reuses the
        # energy convention, no new constant).
        from sklearn.decomposition import MiniBatchDictionaryLearning
        mean_feat = np.mean(
            [boundary.unit(matrix) for matrix in matrices], axis=0
        )
        mean_feat = boundary.unit(mean_feat)
        dictionary = MiniBatchDictionaryLearning(
            n_components=64, alpha=0.05,
            transform_algorithm="lasso_lars", transform_alpha=0.05,
            random_state=config["seed"], max_iter=200, batch_size=32,
        )
        # fit on pool rows only so the vocabulary is not shaped by the
        # spec queries; spec rows are merely encoded against it
        dictionary.fit(mean_feat[spec_n:])
        codes = np.abs(dictionary.transform(mean_feat))
        spec_codes = codes[:spec_n]
        pool_codes = codes[spec_n:]
        demand = spec_codes.mean(axis=0)
        order_atoms = np.argsort(-demand)
        cum = np.cumsum(demand[order_atoms]) / max(demand.sum(), 1e-12)
        support_atoms = order_atoms[: int(np.searchsorted(cum, 0.90)) + 1]
        support_mask = np.zeros(codes.shape[1], dtype=bool)
        support_mask[support_atoms] = True
        for row, code in zip(pool, pool_codes):
            row["_atom_supply"] = code[support_mask].astype(np.float32)
        config["task_boundary"]["_atom_demand"] = [
            float(x) for x in demand[support_mask]
        ]
        print(
            f"[setup][D_atom] atoms=64 support={support_mask.sum()} "
            f"spec_mass_covered=0.90",
            flush=True,
        )

    if set(BOOT_CONDS) & set(config["conditions"]):
        # metered-teacher bootstrap arms. Planning may only use what a
        # deployer would hold BEFORE paying for a response: the spec
        # queries and the candidate prompts. Response-view features are
        # consulted only for rows already bought.
        boot_feat = np.mean(
            [boundary.unit(matrix) for matrix in matrices], axis=0
        )
        boot_feat = boundary.unit(boot_feat)
        prompt_rows = [
            {"prompt": "", "response": row["prompt"]} for row in pool
        ]
        prompt_feature_paths = [
            cache_dir / f"prompt_features_epoch_{epoch}.npy"
            for epoch in range(1, LESS_WARMUP_EPOCHS + 1)
        ]
        prompt_matrices = _load_valid_less_features(
            prompt_feature_paths, len(prompt_rows)
        )
        if prompt_matrices is None:
            warmup_seed = features.stable_seed(
                f"less-warmup-{config['model']}-"
                f"{manifest['pool_fingerprint']}"
            )
            _extract_less_features(
                config, tokenizer, prompt_rows,
                checkpoint_paths, prompt_feature_paths, warmup_seed,
            )
            prompt_matrices = _load_valid_less_features(
                prompt_feature_paths, len(prompt_rows)
            )
            if prompt_matrices is None:
                raise RuntimeError("failed to cache prompt-view features")
        prompt_feat = boundary.unit(np.mean(
            [boundary.unit(matrix) for matrix in prompt_matrices], axis=0
        ))
        spec_prompt_rows = [
            {"prompt": "", "response": row["prompt"]} for row in task_rows
        ]
        spec_prompt_paths = [
            cache_dir / f"spec_prompt_features_epoch_{epoch}.npy"
            for epoch in range(1, LESS_WARMUP_EPOCHS + 1)
        ]
        spec_prompt_matrices = _load_valid_less_features(
            spec_prompt_paths, len(spec_prompt_rows)
        )
        if spec_prompt_matrices is None:
            warmup_seed = features.stable_seed(
                f"less-warmup-{config['model']}-"
                f"{manifest['pool_fingerprint']}"
            )
            _extract_less_features(
                config, tokenizer, spec_prompt_rows,
                checkpoint_paths, spec_prompt_paths, warmup_seed,
            )
            spec_prompt_matrices = _load_valid_less_features(
                spec_prompt_paths, len(spec_prompt_rows)
            )
            if spec_prompt_matrices is None:
                raise RuntimeError("failed to cache spec prompt features")
        spec_prompt_feat = boundary.unit(np.mean(
            [boundary.unit(matrix) for matrix in spec_prompt_matrices],
            axis=0,
        ))
        emb_cfg = config["embedding_features"]
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer(
            emb_cfg["model"], device=emb_cfg["device"]
        )
        spec_prompt_emb = embedder.encode(
            [row["prompt"] for row in task_rows],
            batch_size=emb_cfg["batch_size"],
            normalize_embeddings=True, show_progress_bar=False,
        ).astype(np.float64, copy=False)
        pool_prompt_emb = embedder.encode(
            [row["prompt"] for row in pool],
            batch_size=emb_cfg["batch_size"],
            normalize_embeddings=True, show_progress_bar=False,
        ).astype(np.float64, copy=False)
        del embedder
        gc.collect()
        # admission gate lives in PROMPT-view space too: response-view
        # spec features form a tight self-cluster that rejects every
        # pool row (measured passrate 0.00 even on-task), while
        # prompt-view separates on-task (0.89) from off-task (0.35-0.47)
        # at the min-calibration threshold. Response quality is judged
        # by the execution check, topic membership by this gate.
        boot_fit_n = config["task_boundary"]["fit_rows"]
        gate_subspace = boundary.fit_subspace(spec_prompt_feat[:boot_fit_n])
        gate_cal = boundary.score(
            gate_subspace, spec_prompt_feat[boot_fit_n:]
        )
        gate_pool_scores = boundary.score(gate_subspace, prompt_feat)
        _BOOT_STATE.clear()
        _BOOT_STATE.update({
            "gate_thr": float(gate_cal.min()),
            "gate_pool_scores": gate_pool_scores,
            "spec_feat": boot_feat[:spec_n],
            "pool_feat": boot_feat[spec_n:],
            "pool_prompt_feat": prompt_feat,
            "spec_prompt_feat": spec_prompt_feat,
            "spec_prompt_emb": spec_prompt_emb,
            "pool_prompt_emb": pool_prompt_emb,
        })
        print(
            f"[setup][F_boot] prompt-view features ready pool={len(pool)}",
            flush=True,
        )

    if "D_grad_pre" in config["conditions"]:
        # our subspace + ranking machinery on Adam-preconditioned features
        fit_n = config["task_boundary"]["fit_rows"]
        pre_scores = []
        for matrix in matrices:
            matrix = boundary.unit(matrix)
            subspace = boundary.fit_subspace(matrix[:fit_n])
            pre_scores.append(boundary.score(subspace, matrix[spec_n:]))
        pre = np.max(np.stack(pre_scores), axis=0)
        for row, score in zip(pool, pre):
            row["_d_grad_pre_score"] = float(score)
        print(
            f"[setup][D_grad_pre][score] checkpoints={len(matrices)} "
            f"subspace_on_preconditioned_features",
            flush=True,
        )
    print(
        f"[setup][E_less_std][score] checkpoints={LESS_WARMUP_EPOCHS} "
        f"pool_items={len(pool)}",
        flush=True,
    )


def score_pool(config, tokenizer, task_rows, pool):
    """Compute gradient and BGE trajectory scores against the same task split."""
    feat_cfg = config["gradient_features"]
    combined = task_rows + pool
    selection_lambda = config["selection"]["lambda"]
    out_rows = []
    gradient_rows = combined
    if selection_lambda != 0:
        alpaca_non_test, _ = split("alpaca")
        out_rows = alpaca_non_test[:35]
        if len(out_rows) != 35:
            raise ValueError(
                f"D_grad OUT subspace needs 35 alpaca rows, got {len(out_rows)}"
            )
        gradient_rows = combined + out_rows
    print(
        f"[setup][gradient-features] extracting {len(gradient_rows)} rows on cuda",
        flush=True,
    )
    grad = features.extract_features(
        config["model"],
        gradient_rows,
        proj_dim=feat_cfg["projection_dim"],
        max_prompt_tok=feat_cfg["max_prompt_tokens"],
        max_resp_tok=feat_cfg["max_response_tokens"],
        device=config["device"],
    ).astype(np.float64, copy=False)
    grad = boundary.unit(grad)

    fit_n = config["task_boundary"]["fit_rows"]
    spec_n = config["task_boundary"]["spec_rows"]
    alpha = config["task_boundary"]["alpha"]
    mean_spec_grad = boundary.unit(
        np.mean(grad[:spec_n], axis=0, keepdims=True)
    )[0]
    pool_end = spec_n + len(pool)
    pool_grad = grad[spec_n:pool_end]
    less_scores = pool_grad @ mean_spec_grad
    if "D_grad_cov" in config["conditions"]:
        spec_sims = pool_grad @ grad[:spec_n].T
        for row_index, row_sims in enumerate(spec_sims):
            pool[row_index]["_cov_sims"] = row_sims.astype(np.float32)
    grad_subspace = boundary.fit_subspace(grad[:fit_n])
    grad_cal = boundary.score(grad_subspace, grad[fit_n:spec_n])
    grad_scores = boundary.score(grad_subspace, pool_grad)
    grad_threshold = conformal_threshold(grad_cal, alpha)
    config["task_boundary"]["_grad_threshold"] = float(grad_threshold)
    config["task_boundary"]["_grad_cal"] = [float(x) for x in grad_cal]
    d_grad_scores = grad_scores
    out_subspace = None
    if selection_lambda != 0:
        out_features = grad[pool_end:]
        out_subspace = boundary.fit_subspace(out_features)
        out_scores = boundary.score(out_subspace, pool_grad)
        d_grad_scores = grad_scores - selection_lambda * out_scores

    # extract_features owns the temporary base model. Force its unreachable CUDA
    # allocations out before constructing the CPU embedding model.
    del grad
    gc.collect()
    torch.cuda.empty_cache()

    emb_cfg = config["embedding_features"]
    print(
        f"[setup][embedding-features] encoding {len(combined)} rows on cpu",
        flush=True,
    )
    from sentence_transformers import SentenceTransformer

    embedder = SentenceTransformer(emb_cfg["model"], device=emb_cfg["device"])
    texts = [row["prompt"] + "\n" + row["response"] for row in combined]
    embeddings = embedder.encode(
        texts,
        batch_size=emb_cfg["batch_size"],
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype(np.float64, copy=False)
    embeddings = boundary.unit(embeddings)
    emb_subspace = boundary.fit_subspace(embeddings[:fit_n])
    emb_cal = boundary.score(emb_subspace, embeddings[fit_n:spec_n])
    emb_scores = boundary.score(emb_subspace, embeddings[spec_n:])
    emb_threshold = conformal_threshold(emb_cal, alpha)
    del embedder, embeddings
    gc.collect()

    if not (
        len(grad_scores) == len(less_scores) == len(emb_scores) == len(pool)
    ):
        raise RuntimeError("feature scores do not align with the candidate pool")
    for row, grad_score, d_grad_score, less_score, emb_score in zip(
        pool, grad_scores, d_grad_scores, less_scores, emb_scores
    ):
        row["_grad_score"] = float(grad_score)
        if selection_lambda != 0:
            row["_d_grad_score"] = float(d_grad_score)
        row["_less_score"] = float(less_score)
        row["_emb_score"] = float(emb_score)

    if {"E_less_std", "D_grad_pre", "D_less_bnd", "D_atom",
        *BOOT_CONDS} & set(config["conditions"]):
        attach_less_std_scores(config, tokenizer, task_rows, pool)

    if {"H_nll", "H_smartad_std"} & set(config["conditions"]):
        attach_student_nll_scores(config, tokenizer, pool)

    boundary_metrics = {
        "domain": config["task_boundary"]["domain"],
        "filter": config["task_boundary"]["filter"],
        "n_spec": spec_n,
        "n_fit": fit_n,
        "n_calibration": spec_n - fit_n,
        "energy": config["task_boundary"]["energy"],
        "alpha": alpha,
        "gradient": {
            "rank": int(grad_subspace.shape[1]),
            "threshold": grad_threshold,
            "calibration_coverage": float(np.mean(grad_cal >= grad_threshold)),
            "pool_above_threshold": int(np.sum(grad_scores >= grad_threshold)),
            "lambda": selection_lambda,
        },
        "embedding": {
            "model": emb_cfg["model"],
            "rank": int(emb_subspace.shape[1]),
            "threshold": emb_threshold,
            "calibration_coverage": float(np.mean(emb_cal >= emb_threshold)),
            "pool_above_threshold": int(np.sum(emb_scores >= emb_threshold)),
        },
    }
    if out_subspace is not None:
        boundary_metrics["gradient"]["out_subspace"] = {
            "domain": "alpaca",
            "n_fit": len(out_rows),
            "rank": int(out_subspace.shape[1]),
        }
    return boundary_metrics


def response_token_count(tokenizer, response):
    return len(tokenizer(response, add_special_tokens=False)["input_ids"])


def attach_token_counts(tokenizer, pool):
    for row in pool:
        row["_token_count"] = response_token_count(tokenizer, row["response"])


def take_prefix(
    pool, order, budget, response=None, fixed_token_count=None, excluded=None
):
    """Take the longest ranked prefix whose response tokens fit the budget."""
    selected = []
    used = 0
    excluded = excluded or set()
    for index in order:
        index = int(index)
        if index in excluded:
            continue
        token_count = (
            fixed_token_count
            if fixed_token_count is not None
            else pool[index]["_token_count"]
        )
        if used + token_count > budget:
            break
        item = dict(pool[index])
        if response is not None:
            item["response"] = response
            item["_is_refusal"] = True
        else:
            item["_is_refusal"] = False
        item["_token_count"] = int(token_count)
        item["_pool_index"] = index
        selected.append(item)
        used += token_count
    return selected


def build_selections(config, tokenizer, pool):
    attach_token_counts(tokenizer, pool)
    n = len(pool)
    budget = config["selection"]["response_token_budget"]
    seed = config["seed"]

    natural = np.arange(n)
    random_order = np.random.default_rng(seed).permutation(n)
    emb_order = np.argsort(
        -np.asarray([row["_emb_score"] for row in pool]), kind="stable"
    )
    grad_order = np.argsort(
        -np.asarray([row["_grad_score"] for row in pool]), kind="stable"
    )
    d_grad_order = grad_order
    if config["selection"]["lambda"] != 0:
        d_grad_order = np.argsort(
            -np.asarray([row["_d_grad_score"] for row in pool]), kind="stable"
        )
    less_order = np.argsort(
        -np.asarray([row["_less_score"] for row in pool]), kind="stable"
    )
    low_grad_order = grad_order[::-1]

    selections = {
        "A_all": take_prefix(pool, natural, sum(r["_token_count"] for r in pool)),
        "B_random": take_prefix(pool, random_order, budget),
        "C_emb": take_prefix(pool, emb_order, budget),
        "D_grad": take_prefix(pool, d_grad_order, budget),
        "E_less": take_prefix(pool, less_order, budget),
    }

    if "D_grad_pre" in config["conditions"]:
        pre_order = np.argsort(
            -np.asarray([row["_d_grad_pre_score"] for row in pool]),
            kind="stable",
        )
        selections["D_grad_pre"] = take_prefix(pool, pre_order, budget)

    if "D_grad_cov" in config["conditions"]:
        # cost-scaled greedy facility location over task-query similarities
        sims = np.stack([row["_cov_sims"] for row in pool])
        costs = np.asarray([max(row["_token_count"], 1) for row in pool])
        covered = np.zeros(sims.shape[1])
        chosen, used = [], 0
        available = set(range(len(pool)))
        while available:
            gains = np.maximum(sims - covered, 0.0).sum(axis=1) / costs
            best = max(available, key=lambda i: gains[i])
            if gains[best] <= 0:
                break
            if used + pool[best]["_token_count"] > budget:
                available.discard(best)
                continue
            chosen.append(best)
            used += pool[best]["_token_count"]
            covered = np.maximum(covered, sims[best])
            available.discard(best)
        # coverage saturates well before the budget: top up by score order
        if used < budget:
            topup_order = np.argsort(
                -np.asarray([row["_grad_score"] for row in pool]),
                kind="stable",
            )
            for i in topup_order:
                i = int(i)
                if i in set(chosen):
                    continue
                if used + pool[i]["_token_count"] > budget:
                    continue
                chosen.append(i)
                used += pool[i]["_token_count"]
        selections["D_grad_cov"] = take_prefix(
            pool, chosen, budget
        )
        print(
            f"[setup][D_grad_cov][selection] items={len(chosen)} "
            f"tokens={used} coverage={covered.mean():.3f}",
            flush=True,
        )

    if "D_less_bnd" in config["conditions"]:
        threshold = config["task_boundary"]["_grad_threshold"]
        inside = [
            i for i, row in enumerate(pool)
            if row["_grad_score"] >= threshold
        ]
        pad_mode_env = os.environ.get("E3_BND_PAD", "auto")
        if pad_mode_env == "auto":
            # single learned rule, no per-task switches: admit any row whose
            # conformal p-value (from the calibration score distribution)
            # is at least alpha/2, rank admitted rows by utility, fill the
            # budget. Rows the calibration distribution rejects at alpha/2
            # never enter, so single-domain pools pad naturally and
            # off-domain rows stay out.
            cal_scores = sorted(config["task_boundary"]["_grad_cal"])
            n_cal = len(cal_scores)
            alpha_admit = config["task_boundary"]["alpha"] / 2

            def p_value(score):
                import bisect
                below = bisect.bisect_right(cal_scores, score)
                return (1 + below) / (n_cal + 1)

            admitted = [
                i for i, row in enumerate(pool)
                if p_value(row["_grad_score"]) >= alpha_admit
            ]
            auto_order = sorted(
                admitted, key=lambda i: -pool[i]["_less_std_score"]
            )
            selections["D_less_bnd"] = take_prefix(pool, auto_order, budget)
            print(
                f"[setup][D_less_bnd][selection] admitted={len(admitted)}"
                f"/{len(pool)} rule=p>=alpha/2 (auto)",
                flush=True,
            )
        inside_set = set(inside)
        if os.environ.get("E3_BND_DENSITY") == "1":
            bnd_order = sorted(
                inside,
                key=lambda i: -pool[i]["_less_std_score"]
                / max(pool[i]["_token_count"], 1),
            )
        else:
            bnd_order = sorted(
                inside, key=lambda i: -pool[i]["_less_std_score"]
            )
        # the gold-calibrated threshold is strict for teacher traces: after
        # the inside prefix, pad toward the budget with the nearest-boundary
        # rows (highest grad score outside), still ranked before selection
        outside_pad = []
        pad_mode = os.environ.get("E3_BND_PAD", "1")
        if pad_mode != "0":
            pad_key = (
                (lambda i: -pool[i]["_less_std_score"])
                if pad_mode == "less"
                else (lambda i: -pool[i]["_grad_score"])
            )
            outside_pad = sorted(
                (i for i in range(len(pool)) if i not in inside_set),
                key=pad_key,
            )
        if pad_mode_env != "auto":
            selections["D_less_bnd"] = take_prefix(
                pool, bnd_order + outside_pad, budget
            )
        print(
            f"[setup][D_less_bnd][selection] inside_boundary={len(inside)}"
            f"/{len(pool)} threshold={threshold:.3f} pad=nearest-boundary",
            flush=True,
        )

    if "D_atom" in config["conditions"]:
        # market clearing: tokens are the currency, atoms the goods.
        # Demand is the spec's atom mass scaled to the budget; each trace
        # supplies its per-token atom mass; greedy cost-scaled clearing
        # with linear depletion. All scales endogenous.
        # admission: same calibrated boundary gate D_less_bnd uses.
        # The earlier p-value form was vacuous at n_cal=15 (min p =
        # 1/16 > alpha/2), which let off-task traces sell shared-atom
        # supply into legitimate accounts.
        gate_thr_a = config["task_boundary"]["_grad_threshold"]
        admitted_a = [
            i for i, row in enumerate(pool)
            if row["_grad_score"] >= gate_thr_a
        ]
        print(
            f"[setup][D_atom] admitted={len(admitted_a)}/{len(pool)} "
            f"gate_threshold={gate_thr_a:.3f}",
            flush=True,
        )
        demand_vec = np.asarray(
            config["task_boundary"]["_atom_demand"], dtype=np.float64
        )
        W = demand_vec / max(demand_vec.sum(), 1e-12) * budget
        supplies = {}
        for i in admitted_a:
            code = np.asarray(pool[i]["_atom_supply"], dtype=np.float64)
            total = code.sum()
            tokens_i = max(pool[i]["_token_count"], 1)
            supplies[i] = (code / max(total, 1e-12)) * tokens_i
        chosen_a, used_a = [], 0
        remaining = set(admitted_a)
        while remaining and used_a < budget:
            best_i, best_gain = None, 0.0
            for i in remaining:
                cost = max(pool[i]["_token_count"], 1)
                gain = np.minimum(W, supplies[i]).sum() / cost
                if gain > best_gain:
                    best_gain, best_i = gain, i
            if best_i is None:
                break
            if used_a + pool[best_i]["_token_count"] > budget:
                remaining.discard(best_i)
                continue
            chosen_a.append(best_i)
            used_a += pool[best_i]["_token_count"]
            W = np.maximum(W - supplies[best_i], 0.0)
            remaining.discard(best_i)
        if used_a < budget:
            # demand cleared: surplus reinforces support atoms by density
            leftover = sorted(
                remaining,
                key=lambda i: -supplies[i].sum()
                / max(pool[i]["_token_count"], 1),
            )
            for i in leftover:
                if used_a + pool[i]["_token_count"] > budget:
                    continue
                chosen_a.append(i)
                used_a += pool[i]["_token_count"]
        selections["D_atom"] = take_prefix(pool, chosen_a, budget)
        print(
            f"[setup][D_atom][selection] items={len(chosen_a)} "
            f"tokens={used_a} demand_left={W.sum():.1f}",
            flush=True,
        )

    if set(BOOT_CONDS) & set(config["conditions"]):
        # Clean few-shot protocol simulation: a single metered teacher
        # (deepseek cache rows). Prompts are visible before purchase;
        # responses, their token counts, and response-view features
        # become visible only after paying for the row.
        cache_indices = [
            i for i, row in enumerate(pool)
            if row.get("teacher") == BOOT_TEACHER
        ]
        if not cache_indices:
            raise RuntimeError("bootstrap arms found no metered-teacher rows")

        def purchase_in_order(order):
            bought_local, spent_local = [], 0
            for i in order:
                i = int(i)
                cost = max(pool[i]["_token_count"], 1)
                if spent_local + cost > budget:
                    continue
                bought_local.append(i)
                spent_local += cost
                if spent_local >= budget:
                    break
            return bought_local, spent_local

        if "F_uni_boot" in config["conditions"]:
            rng_boot = np.random.default_rng(seed + 1013)
            bought_u, spent_u = purchase_in_order(
                rng_boot.permutation(cache_indices)
            )
            selections["F_uni_boot"] = [
                {**pool[i], "_is_refusal": False} for i in bought_u
            ]
            print(
                f"[setup][F_uni_boot] items={len(bought_u)} tokens={spent_u}",
                flush=True,
            )

        if "F_selfinst" in config["conditions"]:
            # Self-Instruct (Wang et al., ACL 2023) adapted to the metered
            # protocol: uniform generation order, official ROUGE-L
            # similarity filter (drop a return whose ROUGE-L with any kept
            # instruction exceeds 0.7); filtered returns are still paid
            # for, as in the original pipeline where they are discarded.
            def _lcs(a, b):
                m, n = len(a), len(b)
                dp = [0] * (n + 1)
                for i in range(1, m + 1):
                    prev = 0
                    for j in range(1, n + 1):
                        cur = dp[j]
                        dp[j] = prev + 1 if a[i-1] == b[j-1] else max(dp[j], dp[j-1])
                        prev = cur
                return dp[n]
            def _rougeL(a, b):
                ta, tb = a.split(), b.split()
                if not ta or not tb:
                    return 0.0
                l = _lcs(ta, tb)
                p, r = l / len(ta), l / len(tb)
                return 0.0 if p + r == 0 else 2 * p * r / (p + r)
            rng_si = np.random.default_rng(seed + 2027)
            kept_si, kept_prompts, spent_si = [], [], 0
            for i in rng_si.permutation(cache_indices):
                i = int(i)
                cost = max(pool[i]["_token_count"], 1)
                if spent_si + cost > budget:
                    continue
                spent_si += cost  # paid regardless of the filter
                prompt_i = pool[i]["prompt"]
                if any(_rougeL(prompt_i, p) > 0.7 for p in kept_prompts):
                    continue
                kept_si.append(i)
                kept_prompts.append(prompt_i)
                if spent_si >= budget:
                    break
            selections["F_selfinst"] = [
                {**pool[i], "_is_refusal": False} for i in kept_si
            ]
            _BOOT_STATE["selfinst_rows"] = list(kept_si)
            print(
                f"[setup][F_selfinst] kept={len(kept_si)} spent={spent_si}",
                flush=True,
            )

        if "F_emb_boot" in config["conditions"]:
            spec_emb = _BOOT_STATE["spec_prompt_emb"]
            pool_emb = _BOOT_STATE["pool_prompt_emb"]
            centroid = spec_emb.mean(axis=0)
            centroid /= (np.linalg.norm(centroid) + 1e-12)
            sims = pool_emb @ centroid
            emb_boot_order = sorted(cache_indices, key=lambda i: -sims[i])
            bought_e, spent_e = purchase_in_order(emb_boot_order)
            selections["F_emb_boot"] = [
                {**pool[i], "_is_refusal": False} for i in bought_e
            ]
            print(
                f"[setup][F_emb_boot] items={len(bought_e)} tokens={spent_e}",
                flush=True,
            )

        for boot_name in ("F_atom_boot", "F_atom_boot2", "F_atom_boot3",
                          "F_atom_boot4", "F_atom_boot5", "F_atom_boot6",
                          "F_atom_boot7"):
            if boot_name not in config["conditions"]:
                continue
            # boot2 adds (a) implicit-demand completion: bought teacher
            # responses reveal skills the queries never mentioned, and
            # (b) per-row demand-aligned loss weights for training.
            # boot5 turns the probe phase into hypothesis testing: each
            # demanded atom gets probes whose PREDICTED skill activation
            # is checked against the teacher's actual response; the
            # agreement (per-atom reliability) rescales demand, so the
            # budget flows toward skills the teacher confirmed.
            use_implicit = boot_name.endswith(("2", "3", "4", "5", "6", "7"))
            use_bridge = boot_name.endswith(("3", "4", "5", "6", "7"))
            use_probe = boot_name.endswith(("4", "5", "7"))
            use_validate = boot_name.endswith(("5", "7"))
            # boot6 = boot3 + a marginal-diversity factor on the
            # acquisition objective: homogenized corpora came out of
            # pure quota-chasing (bought rows drift toward the support
            # centroid), so each candidate's gain is discounted by its
            # similarity to what is already bought.
            use_diversity = boot_name.endswith(("6", "7"))
            from sklearn.decomposition import MiniBatchDictionaryLearning
            if use_bridge:
                # skills live in RESPONSE-view gradient space (all atom
                # identifiability evidence is response-side); prompt-view
                # demand degrades to a topic distribution. boot3 keeps
                # the ledger in response-skill space and predicts a
                # candidate's response feature from its prompt feature
                # via a ridge bridge fit on already-paid (prompt,
                # response) pairs — legal, and sharper every round.
                spec_feat = _BOOT_STATE["spec_feat"]
                pool_feat = _BOOT_STATE["pool_feat"]
            else:
                # boot/boot2: everything in prompt view (legacy variant)
                spec_feat = _BOOT_STATE["spec_prompt_feat"]
                pool_feat = _BOOT_STATE["pool_prompt_feat"]
            spec_pfeat = _BOOT_STATE["spec_prompt_feat"]
            pool_pfeat = _BOOT_STATE["pool_prompt_feat"]
            # admission for PURCHASED goods is an outlier test in
            # prompt-view space (see gate construction above): trainable
            # if the prompt scores at least as spec-like as the least
            # spec-like genuine calibration query.
            gate_thr_boot = _BOOT_STATE["gate_thr"]
            gate_scores_boot = _BOOT_STATE["gate_pool_scores"]
            bought_a, spent_a = [], 0
            remaining_a = set(cache_indices)
            expected_tokens = 200.0
            rounds = 0
            ledger = None
            dict_boot = None
            if use_probe:
                # probe-refined dictionary: the k queries are a small,
                # possibly unlucky sample, so atoms they support weakly
                # are refined by BUYING targeted probes before the main
                # loop. Uncertainty is measured on the queries
                # themselves (leave-one-out demand variance +
                # single-anchor mass); probes join the corpus so every
                # later dictionary refit sees them.
                from sklearn.decomposition import (
                    MiniBatchDictionaryLearning as _MBDL,
                )
                dict_p = _MBDL(
                    n_components=int(min(64, max(8, len(spec_feat) // 2))),
                    alpha=0.05, transform_algorithm="lasso_lars",
                    transform_alpha=0.05, random_state=seed,
                    max_iter=100, batch_size=16,
                )
                dict_p.fit(spec_feat)
                spec_codes_p = np.abs(dict_p.transform(spec_feat))
                k_n = spec_codes_p.shape[0]
                loo = np.stack([
                    np.delete(spec_codes_p, i, axis=0).mean(axis=0)
                    for i in range(k_n)
                ])
                var_a = loo.std(axis=0)
                mass = spec_codes_p.sum(axis=0)
                with np.errstate(divide="ignore", invalid="ignore"):
                    anchor = np.where(
                        mass > 1e-12, spec_codes_p.max(axis=0) / mass, 0.0
                    )
                demand_mask = spec_codes_p.mean(axis=0) > 1e-9
                uncertain = [
                    int(a) for a in np.argsort(-var_a)
                    if demand_mask[a]
                    and (var_a[a] > np.median(var_a[demand_mask])
                         or anchor[a] > 0.5)
                ][:5]
                if use_validate:
                    # boot5 probes every top-demand atom, not only the
                    # uncertain tail: probes double as validation data.
                    top_demand = [
                        int(a) for a in
                        np.argsort(-spec_codes_p.mean(axis=0))
                        if demand_mask[a]
                    ][:8]
                    uncertain = list(dict.fromkeys(top_demand + uncertain))[:8]
                probe_map = {}
                if uncertain:
                    pair_p0 = spec_pfeat
                    pair_r0 = spec_feat
                    gram0 = pair_p0 @ pair_p0.T
                    lam0 = 0.1 * np.trace(gram0) / max(gram0.shape[0], 1)
                    dual0 = np.linalg.solve(
                        gram0 + lam0 * np.eye(gram0.shape[0]), pair_r0
                    )
                    cand0 = np.vstack(
                        [pool_pfeat[i][None] for i in sorted(remaining_a)]
                    )
                    pred0 = np.abs(dict_p.transform(
                        (cand0 @ pair_p0.T) @ dual0
                    ))
                    rem0 = sorted(remaining_a)
                    probe_div = _environment_int(
                        "E3_PROBE_DIV", 10 if use_validate else 20
                    )
                    probe_budget = max(200, budget // probe_div)
                    probe_spent = 0
                    for atom_id in uncertain:
                        order0 = np.argsort(-pred0[:, atom_id])
                        for pos in order0[:2]:
                            i = rem0[int(pos)]
                            if i not in remaining_a:
                                continue
                            cost = max(pool[i]["_token_count"], 1)
                            if probe_spent + cost > probe_budget:
                                continue
                            bought_a.append(i)
                            spent_a += cost
                            probe_spent += cost
                            remaining_a.discard(i)
                            probe_map.setdefault(atom_id, []).append(
                                (int(pos), i)
                            )
                    print(
                        f"[setup][{boot_name}][probe] "
                        f"uncertain_atoms={len(uncertain)} "
                        f"probes={len(bought_a)} tokens={probe_spent}",
                        flush=True,
                    )
                rel_p = None
                if use_validate and probe_map:
                    # per-atom reliability: predicted activation (bridge
                    # forecast at purchase time) vs the activation the
                    # teacher's actual response produced. min/max ratio
                    # in [0,1]; atoms without probes keep 1.0 (no
                    # evidence against them).
                    # compare L1-normalized activation SHARES, not raw
                    # magnitudes: the ridge bridge shrinks feature
                    # magnitude wholesale (regression to the mean), so
                    # raw predicted codes are systematically ~100x
                    # smaller than actual ones and a raw min/max ratio
                    # collapses to ~0 for every atom.
                    rel_p = np.ones(spec_codes_p.shape[1])
                    for atom_id, pairs in probe_map.items():
                        ratios = []
                        for pos, i in pairs:
                            pred_vec = pred0[pos]
                            act_vec = np.abs(dict_p.transform(
                                pool_feat[i][None]
                            ))[0]
                            p_share = float(pred_vec[atom_id]) / max(
                                float(pred_vec.sum()), 1e-12
                            )
                            a_share = float(act_vec[atom_id]) / max(
                                float(act_vec.sum()), 1e-12
                            )
                            hi = max(p_share, a_share, 1e-12)
                            ratios.append(min(p_share, a_share) / hi)
                        rel_p[atom_id] = float(np.mean(ratios))
                    low_rel = [
                        a for a in probe_map if rel_p[a] < 0.5
                    ]
                    print(
                        f"[setup][{boot_name}][validate] "
                        f"probed_atoms={len(probe_map)} "
                        f"mean_rel={rel_p[list(probe_map)].mean():.3f} "
                        f"low_rel={low_rel}",
                        flush=True,
                    )
            while remaining_a and spent_a < budget:
                rounds += 1
                corpus = np.vstack(
                    [spec_feat] + [pool_feat[i][None] for i in bought_a]
                )
                n_atoms = int(min(64, max(8, corpus.shape[0] // 2)))
                dict_boot = MiniBatchDictionaryLearning(
                    n_components=n_atoms, alpha=0.05,
                    transform_algorithm="lasso_lars", transform_alpha=0.05,
                    random_state=seed, max_iter=100, batch_size=16,
                )
                dict_boot.fit(corpus)
                spec_codes_b = np.abs(dict_boot.transform(spec_feat))
                demand_b = spec_codes_b.mean(axis=0)
                if use_validate and rel_p is not None:
                    # carry per-atom reliability from the probe-time
                    # dictionary onto this round's refit atoms via
                    # component similarity (atoms are not index-aligned
                    # across refits).
                    comp_b = dict_boot.components_
                    comp_b = comp_b / np.maximum(
                        np.linalg.norm(comp_b, axis=1, keepdims=True), 1e-12
                    )
                    comp_p = dict_p.components_
                    comp_p = comp_p / np.maximum(
                        np.linalg.norm(comp_p, axis=1, keepdims=True), 1e-12
                    )
                    sim_bp = np.abs(comp_b @ comp_p.T)
                    sim_bp = sim_bp / np.maximum(
                        sim_bp.sum(axis=1, keepdims=True), 1e-12
                    )
                    demand_b = demand_b * (sim_bp @ rel_p)
                # implicit-demand completion needs a response-to-prompt
                # bridge to stay modality-consistent; deferred. boot2
                # currently differs from boot only in the per-row loss
                # weights attached below.
                ledger = demand_b / max(demand_b.sum(), 1e-12) * budget
                # supply uses RAW code magnitudes, never normalized:
                # lasso gives off-task prompts small codes (the atoms
                # cannot reduce their error), and normalizing that away
                # fabricated full-strength supply from weak alignments
                # (95% overhead). A raw-code token supplies little when
                # its gradient does not lie in demanded directions.
                round_kappa = 1.0
                if use_bridge:
                    known_resp = np.vstack(
                        [spec_feat] + [pool_feat[i][None] for i in bought_a]
                    )
                    known_mass = np.abs(
                        dict_boot.transform(known_resp)
                    ).sum(axis=1)
                    round_kappa = max(float(known_mass.mean()), 1e-9)
                if bought_a:
                    bought_codes = np.abs(dict_boot.transform(
                        np.vstack([pool_feat[i][None] for i in bought_a])
                    )) / round_kappa
                    for code_row, i in zip(bought_codes, bought_a):
                        supply = code_row * max(pool[i]["_token_count"], 1)
                        ledger = np.maximum(ledger - supply, 0.0)
                rem_list = sorted(remaining_a)
                if use_bridge:
                    # ridge bridge in dual form: pairs are the spec
                    # examples plus everything bought so far
                    pair_p = np.vstack(
                        [spec_pfeat] + [pool_pfeat[i][None] for i in bought_a]
                    )
                    pair_r = np.vstack(
                        [spec_feat] + [pool_feat[i][None] for i in bought_a]
                    )
                    gram = pair_p @ pair_p.T
                    lam = 0.1 * np.trace(gram) / max(gram.shape[0], 1)
                    dual = np.linalg.solve(
                        gram + lam * np.eye(gram.shape[0]), pair_r
                    )
                    cand_p = np.vstack([pool_pfeat[i][None] for i in rem_list])
                    pred_resp = (cand_p @ pair_p.T) @ dual
                    # exchange-rate calibration: an average on-task
                    # example's code mass counts as one token of supply
                    rem_codes = np.abs(
                        dict_boot.transform(pred_resp)
                    ) / round_kappa
                else:
                    rem_codes = np.abs(dict_boot.transform(
                        np.vstack([pool_pfeat[i][None] for i in rem_list])
                    ))
                gains = np.array([
                    np.minimum(
                        ledger, code_row * expected_tokens
                    ).sum() / expected_tokens
                    for code_row in rem_codes
                ])
                if use_diversity and bought_a:
                    # marginal-diversity discount: a candidate whose
                    # skill code duplicates an already-bought row
                    # supplies redundant tokens even when quotas remain.
                    bought_n = np.abs(dict_boot.transform(
                        np.vstack([pool_feat[i][None] for i in bought_a])
                    ))
                    bought_n = bought_n / np.maximum(
                        np.linalg.norm(bought_n, axis=1, keepdims=True),
                        1e-12,
                    )
                    rem_n = rem_codes / np.maximum(
                        np.linalg.norm(rem_codes, axis=1, keepdims=True),
                        1e-12,
                    )
                    max_sim = (rem_n @ bought_n.T).max(axis=1)
                    div_w = float(os.environ.get("E3_DIV_WEIGHT", "0.5"))
                    gains = gains * (1.0 - div_w * np.clip(max_sim, 0.0, 1.0))
                picked = 0
                ledger_work = ledger.copy()
                for order_pos in np.argsort(-gains):
                    i = rem_list[int(order_pos)]
                    if gains[int(order_pos)] <= 1e-9:
                        break
                    code_row = rem_codes[int(order_pos)]
                    # within-batch sequential accounting: a batch of
                    # near-duplicates stops paying after the first
                    if np.minimum(
                        ledger_work, code_row * expected_tokens
                    ).sum() / expected_tokens <= 1e-9:
                        continue
                    cost = max(pool[i]["_token_count"], 1)
                    if spent_a + cost > budget:
                        remaining_a.discard(i)
                        continue
                    ledger_work = np.maximum(
                        ledger_work - code_row * cost, 0.0
                    )
                    bought_a.append(i)
                    spent_a += cost
                    remaining_a.discard(i)
                    picked += 1
                    if picked >= 8 or spent_a >= budget:
                        break
                if bought_a:
                    expected_tokens = float(np.mean(
                        [max(pool[i]["_token_count"], 1) for i in bought_a]
                    ))
                if picked == 0:
                    break
            if spent_a < budget and remaining_a and dict_boot is not None:
                # surplus phase: quotas cleared but budget remains. Spend
                # it on gate-passing candidates (prompts are visible
                # pre-purchase, so the prompt-view gate is a legal
                # planning signal) with the highest on-vocabulary supply
                # density, instead of returning the money.
                surplus_list = sorted(remaining_a)
                surplus_codes = np.abs(dict_boot.transform(
                    np.vstack([pool_pfeat[i][None] for i in surplus_list])
                ))
                density = surplus_codes.sum(axis=1)
                for order_pos in np.argsort(-density):
                    i = surplus_list[int(order_pos)]
                    if _BOOT_STATE["gate_pool_scores"][i] < \
                            _BOOT_STATE["gate_thr"]:
                        continue
                    cost = max(pool[i]["_token_count"], 1)
                    if spent_a + cost > budget:
                        continue
                    bought_a.append(i)
                    spent_a += cost
                    remaining_a.discard(i)
                    if spent_a >= budget:
                        break
            admitted_boot = [
                i for i in bought_a
                if gate_scores_boot[i] >= gate_thr_boot
            ]
            overhead_boot = sum(
                max(pool[i]["_token_count"], 1)
                for i in bought_a if i not in set(admitted_boot)
            )
            selected_boot = [
                {**pool[i], "_is_refusal": False} for i in admitted_boot
            ]
            if use_implicit and selected_boot and dict_boot is not None:
                demand_dir = ledger / max(ledger.sum(), 1e-12)
                sel_codes = np.abs(dict_boot.transform(
                    np.vstack([pool_feat[i][None] for i in admitted_boot])
                ))
                sel_codes = sel_codes / np.maximum(
                    sel_codes.sum(axis=1, keepdims=True), 1e-12
                )
                spec_dir = np.abs(dict_boot.transform(spec_feat)).mean(axis=0)
                spec_dir = spec_dir / max(spec_dir.sum(), 1e-12)
                raw_w = sel_codes @ spec_dir
                raw_w = raw_w / max(raw_w.mean(), 1e-12)
                # weight range shrinks with corpus size: on a dozen rows
                # a 4x weight is pure gradient variance, on 64+ it is
                # signal (observed seed swings .60/.40 at small n)
                clip_c = 1.0 + 3.0 * min(1.0, len(selected_boot) / 64.0)
                raw_w = np.clip(raw_w, 1.0 / clip_c, clip_c)
                for row_sel, weight in zip(selected_boot, raw_w):
                    row_sel["_v3_weight"] = float(weight)
            selections[boot_name] = selected_boot
            if use_bridge:
                _BOOT_STATE[boot_name + "_bought"] = list(bought_a)
                _BOOT_STATE[boot_name + "_spent"] = spent_a
            print(
                f"[setup][{boot_name}] bought={len(bought_a)} "
                f"spent={spent_a} admitted={len(admitted_boot)} "
                f"overhead={overhead_boot} rounds={rounds} "
                f"unspent={budget - spent_a} "
                f"demand_left={ledger.sum():.1f}",
                flush=True,
            )

    if "D_grad_iter" in config["conditions"]:
        # closed-loop selection: round 1 takes budget/R by the theta_0 score;
        # later rounds are chosen inside run_iterative_condition against the
        # residual demand read from the partially trained student.
        rounds = config["selection"]["iter_rounds"]
        selections["D_grad_iter"] = take_prefix(
            pool, d_grad_order, budget // rounds
        )

    if "H_nll" in config["conditions"]:
        nll_order = np.argsort(
            np.asarray([row["_student_nll"] for row in pool]), kind="stable"
        )
        selections["H_nll"] = take_prefix(pool, nll_order, budget)

    if "E_less_std" in config["conditions"]:
        less_std_order = np.argsort(
            -np.asarray([row["_less_std_score"] for row in pool]), kind="stable"
        )
        selections["E_less_std"] = take_prefix(pool, less_std_order, budget)

    if "H_smartad_std" in config["conditions"]:
        best_by_prompt = {}
        for pool_index, row in enumerate(pool):
            if row["teacher"] == "gold":
                continue
            current = best_by_prompt.get(row["prompt"])
            candidate = (row["_student_nll"], pool_index)
            if current is None or candidate < current:
                best_by_prompt[row["prompt"]] = candidate
        if not best_by_prompt:
            raise ValueError("H_smartad_std found no generated-teacher candidates")
        smartad_order = [
            pool_index
            for _nll, pool_index in sorted(best_by_prompt.values())
        ]
        selections["H_smartad_std"] = take_prefix(
            pool, smartad_order, budget
        )
        print(
            f"[setup][H_smartad_std][selection] prompts="
            f"{len(best_by_prompt)} generated_candidates="
            f"{sum(row['teacher'] != 'gold' for row in pool)}",
            flush=True,
        )

    refusal_fraction = config["selection"]["refusal_fraction"]
    refusal_budget = int(round(budget * refusal_fraction))
    original_budget = budget - refusal_budget
    original = take_prefix(pool, grad_order, original_budget)
    used_indices = {item["_pool_index"] for item in original}
    refusals = take_prefix(
        pool,
        low_grad_order,
        refusal_budget,
        response=config["selection"]["refusal_response"],
        fixed_token_count=response_token_count(
            tokenizer, config["selection"]["refusal_response"]
        ),
        excluded=used_indices,
    )
    matrix_conditions = [
        c for c in config["conditions"] if c.startswith("M_")
    ]
    if matrix_conditions:
        import hashlib as _hl
        cache_idx_m = [
            i for i, row in enumerate(pool)
            if row.get("teacher") == BOOT_TEACHER
        ]
        scores_path = ROOT / "data" / "alpagasus_scores.json"
        alpa_scores = {}
        if scores_path.is_file():
            alpa_scores = json.loads(scores_path.read_text())

        def _row_key(row):
            return _hl.sha256(
                (row["prompt"] + row["response"]).encode()
            ).hexdigest()[:16]

        def _acquire(name):
            if name == "ours":
                if "F_atom_boot3_bought" not in _BOOT_STATE:
                    raise RuntimeError(
                        "M_ours_* requires F_atom_boot3 in conditions"
                    )
                return list(_BOOT_STATE["F_atom_boot3_bought"])
            if name == "selfinst":
                if "selfinst_rows" not in _BOOT_STATE:
                    raise RuntimeError(
                        "M_selfinst_* requires F_selfinst in conditions"
                    )
                return list(_BOOT_STATE["selfinst_rows"])
            if name == "evol":
                evol_idx = [
                    i for i in cache_idx_m if pool[i].get("evol") == 1
                ]
                if not evol_idx:
                    raise RuntimeError(
                        "M_evol_* requires E3_EVOL_POOL=1 with evol rows"
                    )
                rng_e = np.random.default_rng(seed + 4099)
                bought_e, spent_e = [], 0
                for i in rng_e.permutation(evol_idx):
                    i = int(i)
                    cost = max(pool[i]["_token_count"], 1)
                    if spent_e + cost > budget:
                        continue
                    bought_e.append(i)
                    spent_e += cost
                    if spent_e >= budget:
                        break
                return bought_e
            if name == "ourscorr":
                # Skill-resolved corrective acquisition (our distillation
                # innovation): identical protocol to LLM2LLM except the
                # augmentation rule — failed rollouts are encoded against
                # the atom dictionary to NAME the deficient skills, and
                # the remaining budget buys predicted supply for that
                # deficit (atom-deficit targeting vs. LLM2LLM's
                # instance-similarity targeting).
                rng_c = np.random.default_rng(seed + 5077)  # same seed
                seed_budget_c = budget // 2
                seed_idx_c, spent_c = [], 0
                for i in rng_c.permutation(cache_idx_m):
                    i = int(i)
                    cost = max(pool[i]["_token_count"], 1)
                    if spent_c + cost > seed_budget_c:
                        continue
                    seed_idx_c.append(i)
                    spent_c += cost
                    if spent_c >= seed_budget_c:
                        break
                seed_rows_c = [
                    {**pool[i], "_is_refusal": False} for i in seed_idx_c
                ]
                print(
                    f"[setup][ourscorr] seed={len(seed_idx_c)} "
                    f"tokens={spent_c}",
                    flush=True,
                )
                temp_model_c = build_model(config, with_lora=True)
                train_condition(
                    temp_model_c, tokenizer, seed_rows_c, config,
                    condition="ourscorr_seed",
                )
                gen_c = generate_texts(
                    temp_model_c, tokenizer, seed_rows_c, config
                )
                del temp_model_c
                gc.collect()
                torch.cuda.empty_cache()
                fail_c = []
                for row_i, text in zip(seed_idx_c, gen_c):
                    gold = None
                    resp = pool[row_i]["response"]
                    g0 = resp.find("def solution")
                    if g0 >= 0:
                        gold = verifier.run_solution(resp[g0:])
                    pred = None
                    p0 = text.find("def solution")
                    if p0 >= 0:
                        pred = verifier.run_solution(text[p0:])
                    if not close_enough(pred, gold):
                        fail_c.append(row_i)
                print(
                    f"[setup][ourscorr] failures={len(fail_c)}"
                    f"/{len(seed_idx_c)}",
                    flush=True,
                )
                from sklearn.decomposition import (
                    MiniBatchDictionaryLearning as _MBDLc,
                )
                spec_r_c = _BOOT_STATE["spec_feat"]
                pool_r_c = _BOOT_STATE["pool_feat"]
                pool_p_c = _BOOT_STATE["pool_prompt_feat"]
                spec_p_c = _BOOT_STATE["spec_prompt_feat"]
                dict_c = _MBDLc(
                    n_components=int(min(64, max(8, len(spec_r_c) // 2))),
                    alpha=0.05, transform_algorithm="lasso_lars",
                    transform_alpha=0.05, random_state=seed,
                    max_iter=100, batch_size=16,
                )
                dict_c.fit(np.vstack(
                    [spec_r_c] + [pool_r_c[i][None] for i in seed_idx_c]
                ))
                if fail_c:
                    deficit = np.abs(dict_c.transform(
                        np.vstack([pool_r_c[i][None] for i in fail_c])
                    )).mean(axis=0)
                else:
                    deficit = np.abs(
                        dict_c.transform(spec_r_c)
                    ).mean(axis=0)
                deficit = deficit / max(deficit.sum(), 1e-12)
                pair_p_c = np.vstack(
                    [spec_p_c] + [pool_p_c[i][None] for i in seed_idx_c]
                )
                pair_r_c = np.vstack(
                    [spec_r_c] + [pool_r_c[i][None] for i in seed_idx_c]
                )
                gram_c = pair_p_c @ pair_p_c.T
                lam_c = 0.1 * np.trace(gram_c) / max(gram_c.shape[0], 1)
                dual_c = np.linalg.solve(
                    gram_c + lam_c * np.eye(gram_c.shape[0]), pair_r_c
                )
                seed_set_c = set(seed_idx_c)
                rem_c = [i for i in cache_idx_m if i not in seed_set_c]
                cand_c = np.vstack([pool_p_c[i][None] for i in rem_c])
                pred_codes_c = np.abs(dict_c.transform(
                    (cand_c @ pair_p_c.T) @ dual_c
                ))
                score_c = pred_codes_c @ deficit
                for pos in np.argsort(-score_c):
                    i = rem_c[int(pos)]
                    cost = max(pool[i]["_token_count"], 1)
                    if spent_c + cost > budget:
                        continue
                    seed_idx_c.append(i)
                    spent_c += cost
                    if spent_c >= budget:
                        break
                print(
                    f"[setup][ourscorr] total={len(seed_idx_c)} "
                    f"tokens={spent_c}",
                    flush=True,
                )
                return seed_idx_c
            if name == "llm2llm":
                # LLM2LLM (Lee et al., Findings of ACL 2024): train on a
                # seed set, find training examples the student still
                # fails, acquire similar examples, train on the union.
                rng_l = np.random.default_rng(seed + 5077)
                seed_budget = budget // 2
                seed_idx, spent_l = [], 0
                for i in rng_l.permutation(cache_idx_m):
                    i = int(i)
                    cost = max(pool[i]["_token_count"], 1)
                    if spent_l + cost > seed_budget:
                        continue
                    seed_idx.append(i)
                    spent_l += cost
                    if spent_l >= seed_budget:
                        break
                seed_rows = [
                    {**pool[i], "_is_refusal": False} for i in seed_idx
                ]
                print(
                    f"[setup][llm2llm] seed={len(seed_idx)} "
                    f"tokens={spent_l}",
                    flush=True,
                )
                temp_model = build_model(config, with_lora=True)
                train_condition(
                    temp_model, tokenizer, seed_rows, config,
                    condition="llm2llm_seed",
                )
                gen_texts = generate_texts(
                    temp_model, tokenizer, seed_rows, config
                )
                del temp_model
                gc.collect()
                torch.cuda.empty_cache()
                failures = []
                for row_i, text in zip(seed_idx, gen_texts):
                    gold = None
                    resp = pool[row_i]["response"]
                    g0 = resp.find("def solution")
                    if g0 >= 0:
                        gold = verifier.run_solution(resp[g0:])
                    pred = None
                    p0 = text.find("def solution")
                    if p0 >= 0:
                        pred = verifier.run_solution(text[p0:])
                    if not close_enough(pred, gold):
                        failures.append(row_i)
                print(
                    f"[setup][llm2llm] failures={len(failures)}"
                    f"/{len(seed_idx)}",
                    flush=True,
                )
                pe_l = _BOOT_STATE["pool_prompt_emb"]
                seed_set = set(seed_idx)
                if failures:
                    fail_emb = pe_l[failures]
                    sims_l = pe_l @ fail_emb.T
                    sim_of = {
                        i: float(sims_l[i].max())
                        for i in cache_idx_m if i not in seed_set
                    }
                else:
                    sim_of = {
                        i: 0.0 for i in cache_idx_m if i not in seed_set
                    }
                for i in sorted(sim_of, key=lambda j: -sim_of[j]):
                    cost = max(pool[i]["_token_count"], 1)
                    if spent_l + cost > budget:
                        continue
                    seed_idx.append(i)
                    spent_l += cost
                    if spent_l >= budget:
                        break
                print(
                    f"[setup][llm2llm] total={len(seed_idx)} "
                    f"tokens={spent_l}",
                    flush=True,
                )
                return seed_idx
            raise NotImplementedError(f"acquisition {name} pending")

        def _curate(name, bought):
            rows_out = []
            if name == "plain":
                rows_out = [
                    {**pool[i], "_is_refusal": False} for i in bought
                ]
            elif name == "alpagasus":
                for i in bought:
                    score = alpa_scores.get(_row_key(pool[i]))
                    if score is not None and float(score) >= 4.5:
                        rows_out.append(
                            {**pool[i], "_is_refusal": False}
                        )
                if not rows_out:
                    # official fallback: if the filter empties the set,
                    # keep the top-scoring half instead
                    ranked = sorted(
                        bought,
                        key=lambda i: -float(
                            alpa_scores.get(_row_key(pool[i]), 0.0)
                        ),
                    )
                    rows_out = [
                        {**pool[i], "_is_refusal": False}
                        for i in ranked[: max(1, len(ranked) // 2)]
                    ]
            elif name == "less":
                raw = np.array(
                    [pool[i]["_less_std_score"] for i in bought]
                )
                shifted = raw - raw.min() + 1e-6
                wts = shifted / shifted.mean()
                wts = np.clip(wts, 0.25, 4.0)
                for i, w in zip(bought, wts):
                    rows_out.append(
                        {**pool[i], "_is_refusal": False,
                         "_v3_weight": float(w)}
                    )
            elif name == "ours":
                # E3_CURATE_GATE=0: weights-only curation — the outlier
                # gate stays in the acquisition loop (where surplus
                # rebuy exists) instead of discarding sunk purchases.
                if os.environ.get("E3_CURATE_GATE", "1") == "0":
                    admitted = list(bought)
                else:
                    thr = _BOOT_STATE["gate_thr"]
                    gsc = _BOOT_STATE["gate_pool_scores"]
                    admitted = [i for i in bought if gsc[i] >= thr]
                    if not admitted:
                        admitted = bought
                from sklearn.decomposition import MiniBatchDictionaryLearning
                spec_r = _BOOT_STATE["spec_feat"]
                pool_r = _BOOT_STATE["pool_feat"]
                dict_m = MiniBatchDictionaryLearning(
                    n_components=int(min(64, max(8, len(spec_r) // 2))),
                    alpha=0.05, transform_algorithm="lasso_lars",
                    transform_alpha=0.05, random_state=seed,
                    max_iter=100, batch_size=16,
                )
                dict_m.fit(spec_r)
                spec_dir = np.abs(dict_m.transform(spec_r)).mean(axis=0)
                spec_dir = spec_dir / max(spec_dir.sum(), 1e-12)
                codes_m = np.abs(dict_m.transform(
                    np.vstack([pool_r[i][None] for i in admitted])
                ))
                codes_m = codes_m / np.maximum(
                    codes_m.sum(axis=1, keepdims=True), 1e-12
                )
                raw_w = codes_m @ spec_dir
                raw_w = raw_w / max(raw_w.mean(), 1e-12)
                clip_c = 1.0 + 3.0 * min(1.0, len(admitted) / 64.0)
                raw_w = np.clip(raw_w, 1.0 / clip_c, clip_c)
                for i, w in zip(admitted, raw_w):
                    rows_out.append(
                        {**pool[i], "_is_refusal": False,
                         "_v3_weight": float(w)}
                    )
            else:
                raise NotImplementedError(name)
            return rows_out

        acquired_cache = {}
        for cond in matrix_conditions:
            _, acq, cur = cond.split("_", 2)
            if acq not in acquired_cache:
                acquired_cache[acq] = _acquire(acq)
            bought_m = acquired_cache[acq]
            rows_m = _curate(cur, bought_m)
            selections[cond] = rows_m
            print(
                f"[setup][{cond}] bought={len(bought_m)} "
                f"kept={len(rows_m)}",
                flush=True,
            )

    selections["F_refusal"] = original + refusals
    return selections


def selection_stats(rows, budget, refusal_fraction=None):
    domain_mix = Counter(row["domain"] for row in rows)
    refusal_rows = [row for row in rows if row["_is_refusal"]]
    stats = {
        "n_items": len(rows),
        "token_count": int(sum(row["_token_count"] for row in rows)),
        "budget": budget,
        "domain_mix": {key: int(value) for key, value in sorted(domain_mix.items())},
        "teacher_domain_mix": composition_rows(rows),
        "mean_s": float(np.mean([row["_grad_score"] for row in rows])),
        "mean_embedding_score": float(
            np.mean([row["_emb_score"] for row in rows])
        ),
        "n_refusal": len(refusal_rows),
        "refusal_token_count": int(
            sum(row["_token_count"] for row in refusal_rows)
        ),
        "refusal_domain_mix": {
            key: int(value)
            for key, value in sorted(
                Counter(row["domain"] for row in refusal_rows).items()
            )
        },
    }
    if refusal_fraction is not None:
        stats["reserved_refusal_fraction"] = refusal_fraction
    return stats


def build_model(config, with_lora):
    """Construct a freshly seeded bf16 base model, optionally with fresh LoRA."""
    seed_everything(config["seed"])
    model = AutoModelForCausalLM.from_pretrained(
        config["model"], torch_dtype=torch.bfloat16
    ).to(config["device"])
    if not with_lora:
        return model

    lora_cfg = config["training"]["lora"]
    adapter = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        bias=lora_cfg["bias"],
        task_type="CAUSAL_LM",
        target_modules=lora_cfg["target_modules"],
    )
    return get_peft_model(model, adapter)


def lora_modules(peft_model):
    """Return PEFT LoRA layers keyed identically across train/reference models."""
    return {
        name: module
        for name, module in peft_model.named_modules()
        if hasattr(module, "base_layer") and hasattr(module, "lora_B")
    }


def sync_ref_weights(train_modules, ref_modules):
    """Copy each training layer's effective LoRA weight into the reference."""
    with torch.no_grad():
        for name, train_module in train_modules.items():
            lora_a = train_module.lora_A["default"].weight
            lora_b = train_module.lora_B["default"].weight
            scale = train_module.scaling["default"]
            effective = train_module.base_layer.weight + scale * (
                lora_b @ lora_a
            ).to(train_module.base_layer.weight.dtype)
            ref_modules[name].base_layer.weight.copy_(effective)


def probe_features(ref_model, tokenizer, rows, device, ref_lora_b, projections):
    """Extract unit-normalized reference-model probe features as in M17."""
    extracted = []
    ref_model.train()
    for row in rows:
        input_ids, labels = encode(tokenizer, row, device)
        ref_model.zero_grad(set_to_none=True)
        ref_model(input_ids=input_ids, labels=labels).loss.backward()
        extracted.append(
            features._project_gradients(ref_lora_b, projections).cpu().numpy()
        )
    ref_model.zero_grad(set_to_none=True)
    return np.stack(extracted).astype(np.float64)


def prepare_absorption_monitor(config, task_rows):
    """Build the shared M17 reference extractor and fixed task subspace."""
    fit_n = config["task_boundary"]["fit_rows"]
    calibration_rows = task_rows[fit_n:fit_n + 10]
    if len(calibration_rows) != 10:
        raise ValueError(
            "E3_EARLYSTOP needs 10 task-spec calibration rows; "
            f"found {len(calibration_rows)}"
        )

    tokenizer, ref_model, ref_lora_b, projections = features._prepare_extractor(
        config["model"],
        config["gradient_features"]["projection_dim"],
        config["device"],
    )
    spec_features = probe_features(
        ref_model,
        tokenizer,
        task_rows[:fit_n],
        config["device"],
        ref_lora_b,
        projections,
    )
    subspace = boundary.fit_subspace(boundary.unit(spec_features))
    print(
        f"[setup][early-stop] spec_rank={subspace.shape[1]} "
        f"fit_rows={fit_n} calibration_rows={len(calibration_rows)}",
        flush=True,
    )
    return {
        "tokenizer": tokenizer,
        "ref_model": ref_model,
        "ref_lora_b": ref_lora_b,
        "projections": projections,
        "ref_modules": lora_modules(ref_model),
        "subspace": subspace,
        "calibration_rows": calibration_rows,
        "device": config["device"],
    }


def absorption_demand(train_modules, monitor):
    """Measure mean unit-feature energy inside the fixed task subspace."""
    sync_ref_weights(train_modules, monitor["ref_modules"])
    probe = probe_features(
        monitor["ref_model"],
        monitor["tokenizer"],
        monitor["calibration_rows"],
        monitor["device"],
        monitor["ref_lora_b"],
        monitor["projections"],
    )
    projected = np.linalg.norm(probe @ monitor["subspace"], axis=1)
    return float(np.mean(projected ** 2))


def release_absorption_monitor(monitor):
    """Drop every shared-extractor reference after all conditions finish."""
    monitor.clear()
    release_cuda()


@torch.inference_mode()
def attach_student_nll_scores(config, tokenizer, pool):
    """Attach base-student teacher-forced NLL and its negative selection score."""
    requested = sorted(
        {"H_nll", "H_smartad_std"} & set(config["conditions"])
    )
    score_indices = (
        range(len(pool))
        if "H_nll" in requested
        else [
            pool_index
            for pool_index, row in enumerate(pool)
            if row["teacher"] != "gold"
        ]
    )
    print(
        f"[setup][student-nll] conditions={','.join(requested)} "
        f"scoring={len(score_indices)} device={config['device']}",
        flush=True,
    )
    model = build_model(config, with_lora=False)
    model.eval()
    try:
        for index, pool_index in enumerate(score_indices, start=1):
            row = pool[int(pool_index)]
            input_ids, labels = encode(tokenizer, row, config["device"])
            nll = model(input_ids=input_ids, labels=labels).loss.item()
            row["_student_nll"] = float(nll)
            row["_h_nll_score"] = float(-nll)
            if index % 100 == 0 or index == len(score_indices):
                print(
                    f"[setup][student-nll] scored={index}/{len(score_indices)}",
                    flush=True,
                )
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()


_ACTION_LINE = re.compile(
    r"^\s*(?:action|tool(?:\s+call)?|function\s+call)\s*:", re.IGNORECASE
)
_ANSWER_LINE = re.compile(
    r"^\s*(?:(?:final\s+)?answer\s*(?::|=)|####\s*)|\\boxed\s*\{",
    re.IGNORECASE,
)
_ASSIGNMENT_LINE = re.compile(
    r"^\s*(?:[A-Za-z_]\w*(?:\s*,\s*[A-Za-z_]\w*)*|"
    r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*|\[[^\n]+\])+|"
    r"\([^\n]+\))\s*(?::[^=\n]+)?\s*"
    r"(?:\+=|-=|\*=|/=|//=|%=|\*\*=|&=|\|=|\^=|>>=|<<=|=(?!=))"
)


def _line_spans(text):
    """Return (start, end) character spans, excluding newline characters."""
    spans = []
    offset = 0
    for line in text.splitlines(keepends=True):
        content = line.rstrip("\r\n")
        spans.append((offset, offset + len(content)))
        offset += len(line)
    if not spans and text == "":
        return []
    if text and not text.splitlines(keepends=True):
        spans.append((0, len(text)))
    return spans


def _segment_line_weights(row):
    """Assign SmartAD's 1/1.5/2 weights to response lines."""
    response = row["response"]
    lines = response.splitlines()
    weights = [1.0] * len(lines)
    code_domain = row.get("domain") in {"gsm8k-code", "pandas"}
    looks_like_code = response.lstrip().startswith(
        ("def ", "async def ", "import ", "from ")
    )
    return_lines = []
    answer_lines = []

    if code_domain or looks_like_code:
        action_lines = set()
        try:
            tree = ast.parse(response)
        except SyntaxError:
            tree = None
        if tree is not None:
            for node in ast.walk(tree):
                if isinstance(
                    node, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.NamedExpr)
                ):
                    action_lines.add(node.lineno - 1)
                    targets = (
                        node.targets
                        if isinstance(node, ast.Assign)
                        else [node.target]
                    )
                    target_names = {
                        descendant.id.casefold()
                        for target in targets
                        for descendant in ast.walk(target)
                        if isinstance(descendant, ast.Name)
                    }
                    if target_names & {"answer", "final_answer", "result"}:
                        answer_lines.append(node.lineno - 1)
                elif isinstance(node, ast.Return):
                    line_index = node.lineno - 1
                    action_lines.add(line_index)
                    return_lines.append(line_index)
        else:
            for line_index, line in enumerate(lines):
                if _ASSIGNMENT_LINE.match(line) or re.match(
                    r"^\s*return\b", line
                ):
                    action_lines.add(line_index)
                if re.match(r"^\s*return\b", line):
                    return_lines.append(line_index)
                if re.match(
                    r"^\s*(?:answer|final_answer|result)\s*(?::[^=\n]+)?=",
                    line,
                    re.IGNORECASE,
                ):
                    answer_lines.append(line_index)
        for line_index in action_lines:
            if 0 <= line_index < len(weights):
                weights[line_index] = 1.5
    else:
        for line_index, line in enumerate(lines):
            if _ACTION_LINE.match(line):
                weights[line_index] = 1.5

    final_statement_lines = return_lines + answer_lines + [
        line_index
        for line_index, line in enumerate(lines)
        if _ANSWER_LINE.search(line)
    ]
    if final_statement_lines:
        weights[max(final_statement_lines)] = 2.0
    return weights


def smartad_token_weights(tokenizer, row, labels):
    """Build a per-token SmartAD vector exactly aligned with ``labels``."""
    response = row["response"]
    tokenized = tokenizer(
        response, add_special_tokens=False, return_offsets_mapping=True
    )
    offsets = tokenized["offset_mapping"]
    label_positions = torch.nonzero(labels[0] != -100, as_tuple=False).flatten()
    response_token_count = len(label_positions) - 1  # final labeled token is EOS
    if response_token_count < 0 or len(offsets) < response_token_count:
        raise RuntimeError("could not align SmartAD weights with response labels")

    line_spans = _line_spans(response)
    line_weights = _segment_line_weights(row)
    weights = torch.ones(labels.shape, device=labels.device, dtype=torch.float32)
    for token_index, (token_start, token_end) in enumerate(
        offsets[:response_token_count]
    ):
        if token_end <= token_start:
            continue
        token_weight = 1.0
        for (line_start, line_end), line_weight in zip(
            line_spans, line_weights
        ):
            if token_start < line_end and token_end > line_start:
                token_weight = max(token_weight, line_weight)
        weights[0, label_positions[token_index]] = token_weight
    return weights


def smartad_weighted_loss(model, input_ids, labels, token_weights):
    """Compute shifted, ignore-aware token-weighted causal cross-entropy."""
    logits = model(input_ids=input_ids).logits
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    shift_weights = token_weights[:, 1:].contiguous()
    token_losses = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.shape[-1]),
        shift_labels.reshape(-1),
        reduction="none",
        ignore_index=-100,
    ).reshape_as(shift_labels)
    valid = shift_labels != -100
    denominator = shift_weights.masked_select(valid).sum()
    if denominator.item() == 0:
        raise RuntimeError("SmartAD row has no supervised response tokens")
    return (
        token_losses.float() * shift_weights * valid.to(shift_weights.dtype)
    ).sum() / denominator


def _behavioral_probe(model, tokenizer, rows, config):
    """Fraction of calibration prompts yielding runnable solution code."""
    texts = generate_texts(model, tokenizer, rows, config)
    runnable = 0
    for text in texts:
        start = text.find("def solution")
        if start >= 0:
            prediction = verifier.run_solution(text[start:])
            runnable += int(prediction is not None)
    model.train()
    model.config.use_cache = False
    return runnable / max(len(texts), 1)


def train_condition(
    model,
    tokenizer,
    rows,
    config,
    condition=None,
    absorption_monitor=None,
    training_metrics=None,
):
    """Train on every selected row for three epochs with true accumulation."""
    train_cfg = config["training"]
    if not rows:
        raise ValueError("a training condition selected no rows")
    model.config.use_cache = False
    model.train()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=train_cfg["learning_rate"],
    )
    rng = np.random.default_rng(config["seed"])
    accumulation = train_cfg["gradient_accumulation"]
    optimizer_steps = 0
    stop_step = None
    stop_training = False
    consecutive_low_demand = 0
    train_modules = None
    initial_demand = None
    if absorption_monitor is not None:
        train_modules = lora_modules(model)
        ref_modules = absorption_monitor["ref_modules"]
        if set(train_modules) != set(ref_modules):
            raise RuntimeError(
                "LoRA module mismatch between training and reference models"
            )
        initial_demand = absorption_demand(train_modules, absorption_monitor)
        print(
            f"[{condition}][early-stop] step=0 demand={initial_demand:.6f}",
            flush=True,
        )

    behav_rows = config.get("_behav_rows")
    best_behav = -1.0
    best_state = None
    optimizer.zero_grad(set_to_none=True)
    for _epoch in range(train_cfg["epochs"]):
        order = rng.permutation(len(rows))
        for start in range(0, len(order), accumulation):
            chunk = order[start:start + accumulation]
            # Divide by the actual final-chunk size so all selected examples
            # receive equal weight and no end-of-epoch examples are dropped.
            for row_index in chunk:
                input_ids, labels = encode(
                    tokenizer, rows[int(row_index)], config["device"]
                )
                if condition == "H_smartad_std":
                    token_weights = smartad_token_weights(
                        tokenizer, rows[int(row_index)], labels
                    )
                    loss = smartad_weighted_loss(
                        model, input_ids, labels, token_weights
                    ) / len(chunk)
                else:
                    loss = model(input_ids=input_ids, labels=labels).loss / len(chunk)
                row_weight = rows[int(row_index)].get("_v3_weight")
                if row_weight is not None:
                    # first epoch trains uniformly for coverage; weights
                    # focus later epochs (small-corpus variance control)
                    if not (
                        train_cfg.get("weight_warm_epoch") and _epoch == 0
                    ):
                        loss = loss * float(row_weight)
                loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1
            if behav_rows and optimizer_steps % 8 == 0:
                behav = _behavioral_probe(model, tokenizer, behav_rows, config)
                print(
                    f"[{condition}][behav-stop] step={optimizer_steps} "
                    f"runnable={behav:.2f} best={max(best_behav, 0):.2f}",
                    flush=True,
                )
                if behav >= best_behav:
                    best_behav = behav
                    best_state = {
                        name: parameter.detach().clone()
                        for name, parameter in model.named_parameters()
                        if parameter.requires_grad
                    }
            should_measure = optimizer_steps <= 12 or optimizer_steps % 4 == 0
            if absorption_monitor is not None and should_measure:
                demand = absorption_demand(train_modules, absorption_monitor)
                stop_frac = float(os.environ.get("E3_STOP_FRAC", "0.05"))
                threshold = stop_frac * initial_demand
                consecutive_low_demand = (
                    consecutive_low_demand + 1 if demand < threshold else 0
                )
                print(
                    f"[{condition}][early-stop] step={optimizer_steps} "
                    f"demand={demand:.6f} threshold={threshold:.6f} "
                    f"consecutive={consecutive_low_demand}",
                    flush=True,
                )
                if consecutive_low_demand >= 2:
                    stop_step = optimizer_steps
                    stop_training = True
                    break
        if stop_training:
            break
    if behav_rows and best_state is not None:
        final = _behavioral_probe(model, tokenizer, behav_rows, config)
        if final < best_behav:
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    if parameter.requires_grad and name in best_state:
                        parameter.copy_(best_state[name])
            print(
                f"[{condition}][behav-stop] restored best checkpoint "
                f"(runnable {best_behav:.2f} > final {final:.2f})",
                flush=True,
            )
    model.config.use_cache = True
    if training_metrics is not None:
        training_metrics["stop_step"] = stop_step
    return optimizer_steps


def rescore_pool_residual(model, monitor, config, task_rows, pool, excluded):
    """Score remaining pool rows against the residual demand subspace.

    The residual subspace is fit on the task-spec queries' gradients read at
    the CURRENT student weights (via the shared reference extractor), so it
    spans only what the student still has to learn.
    """
    fit_n = config["task_boundary"]["fit_rows"]
    sync_ref_weights(lora_modules(model), monitor["ref_modules"])
    task_features = probe_features(
        monitor["ref_model"], monitor["tokenizer"], task_rows[:fit_n],
        monitor["device"], monitor["ref_lora_b"], monitor["projections"],
    )
    residual_subspace = boundary.fit_subspace(boundary.unit(task_features))
    remaining = [i for i in range(len(pool)) if i not in excluded]
    scores = np.empty(len(remaining), dtype=np.float64)
    for position, pool_index in enumerate(remaining):
        feature = probe_features(
            monitor["ref_model"], monitor["tokenizer"], [pool[pool_index]],
            monitor["device"], monitor["ref_lora_b"], monitor["projections"],
        )
        scores[position] = boundary.score(
            residual_subspace, boundary.unit(feature)
        )[0]
        if (position + 1) % 200 == 0:
            print(
                f"[D_grad_iter][rescore] {position + 1}/{len(remaining)}",
                flush=True,
            )
    order = [remaining[i] for i in np.argsort(-scores, kind="stable")]
    print(
        f"[D_grad_iter][rescore] residual_rank={residual_subspace.shape[1]} "
        f"remaining={len(remaining)}",
        flush=True,
    )
    return order


def run_iterative_condition(
    model, tokenizer, config, task_rows, pool, first_tranche, monitor,
):
    """Closed-loop distillation: select, absorb, re-read demand, select again."""
    rounds = config["selection"]["iter_rounds"]
    budget = config["selection"]["response_token_budget"]
    tranche_budget = budget // rounds
    all_selected = list(first_tranche)
    excluded = {item["_pool_index"] for item in first_tranche}
    total_steps = 0
    round_stats = []
    for round_index in range(rounds):
        if round_index == 0:
            tranche = first_tranche
        else:
            order = rescore_pool_residual(
                model, monitor, config, task_rows, pool, excluded
            )
            tranche = take_prefix(
                pool, order, tranche_budget, excluded=excluded
            )
            if not tranche:
                print(
                    f"[D_grad_iter][round {round_index + 1}] pool exhausted",
                    flush=True,
                )
                break
            excluded |= {item["_pool_index"] for item in tranche}
            all_selected.extend(tranche)
        train_rows = (
            all_selected
            if config["selection"]["iter_mode"] == "cum"
            else tranche
        )
        steps = train_condition(
            model, tokenizer, train_rows, config, condition="D_grad_iter"
        )
        total_steps += steps
        round_stats.append({
            "round": round_index + 1,
            "n_items": len(tranche),
            "token_count": int(sum(r["_token_count"] for r in tranche)),
            "optimizer_steps": steps,
        })
        print(
            f"[D_grad_iter][round {round_index + 1}/{rounds}] "
            f"items={len(tranche)} steps={steps}",
            flush=True,
        )
    return all_selected, total_steps, round_stats


@torch.inference_mode()
def generate_texts(model, tokenizer, rows, config):
    model.eval()
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": row["prompt"]}],
            add_generation_prompt=True,
            tokenize=False,
        )
        for row in rows
    ]
    batch_size = config["generation"]["eval_batch_size"]
    texts = []
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        for start in range(0, len(prompts), batch_size):
            batch = tokenizer(
                prompts[start:start + batch_size],
                add_special_tokens=False,
                padding=True,
                return_tensors="pt",
            ).to(config["device"])
            generated = model.generate(
                **batch,
                max_new_tokens=config["generation"]["max_new_tokens"],
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
            continuation = generated[:, batch["input_ids"].shape[1]:]
            texts.extend(
                tokenizer.batch_decode(continuation, skip_special_tokens=True)
            )
    finally:
        tokenizer.padding_side = original_padding_side
    return texts


def close_enough(prediction, gold):
    return (
        prediction is not None
        and gold is not None
        and math.isfinite(prediction)
        and math.isfinite(gold)
        and abs(prediction - gold) < 1e-4
    )


def code_metrics(texts, golds, refusal_response):
    executable = 0
    formatted = 0
    refused = 0
    for text, gold in zip(texts, golds):
        start = text.find("def solution")
        if start >= 0:
            formatted += 1
            prediction = verifier.run_solution(text[start:])
            executable += int(close_enough(prediction, gold))
        refused += int(refusal_response in text)
    n = len(texts)
    return {
        "exec_acc": executable / n,
        "format_rate": formatted / n,
        "refusal_rate": refused / n,
    }


def cot_metrics(texts, golds):
    n = len(texts)
    return {
        "any_acc": sum(
            close_enough(last_number(text), gold)
            for text, gold in zip(texts, golds)
        ) / n,
        "writes_code_rate": sum("def solution" in text for text in texts) / n,
    }


def alpaca_metrics(texts, refusal_response):
    n = len(texts)
    return {
        "format_bleed_rate": sum("def solution" in text for text in texts) / n,
        "refusal_rate": sum(refusal_response in text for text in texts) / n,
    }


def is_single_sql_statement(code):
    """Check statement count while ignoring semicolons in quotes and comments."""
    count = 0
    has_content = False
    quote = None
    i = 0
    while i < len(code):
        char = code[i]
        following = code[i + 1] if i + 1 < len(code) else ""

        if quote is not None:
            has_content = True
            if char == quote:
                if following == quote:
                    i += 2
                    continue
                quote = None
            i += 1
            continue

        if char == "-" and following == "-":
            newline = code.find("\n", i + 2)
            i = len(code) if newline < 0 else newline + 1
            continue
        if char == "/" and following == "*":
            comment_end = code.find("*/", i + 2)
            if comment_end < 0:
                return False
            i = comment_end + 2
            continue
        if char in {"'", '"', "`"}:
            quote = char
            has_content = True
        elif char == "[":
            quote = "]"
            has_content = True
        elif char == ";":
            if has_content:
                count += 1
                has_content = False
        elif not char.isspace():
            has_content = True
        i += 1

    if quote is not None:
        return False
    if has_content:
        count += 1
    return count == 1


def sql_metrics(texts):
    return {
        "single_statement_valid_rate": (
            sum(is_single_sql_statement(text) for text in texts) / len(texts)
        ),
    }


def pandas_metrics(texts):
    compilable = 0
    for text in texts:
        try:
            compile(text, "<e3-pandas-generation>", "exec")
            compilable += 1
        except (SyntaxError, ValueError, TypeError):
            pass
    return {"compile_rate": compilable / len(texts)}


@torch.inference_mode()
def mean_eval_losses(model, tokenizer, tests, device):
    """Teacher-forced response loss, matching distill_pilot.eval_loss."""
    model.eval()
    means = {}
    for domain, rows in tests.items():
        losses = []
        for row in rows:
            input_ids, labels = encode(tokenizer, row, device)
            losses.append(model(input_ids=input_ids, labels=labels).loss.item())
        means[domain] = float(np.mean(losses))
    return means


def build_eval_data(config):
    tests = {}
    for domain in config["data"]["eval_domains"]:
        _, test = split(domain)
        if len(test) != config["data"]["test_rows_per_domain"]:
            raise ValueError(f"expected 30 test rows for {domain}, got {len(test)}")
        tests[domain] = test

    code_golds = [verifier.run_solution(row["response"]) for row in tests["gsm8k-code"]]
    if any(gold is None for gold in code_golds):
        raise ValueError("a GSM8K-code test reference did not execute")
    cot_golds = [last_number(row["response"]) for row in tests["gsm8k-cot"]]
    if any(gold is None for gold in cot_golds):
        raise ValueError("a GSM8K-CoT test reference has no numeric answer")
    return tests, code_golds, cot_golds


def evaluate_model(model, tokenizer, tests, code_golds, cot_golds, config):
    refusal = config["selection"]["refusal_response"]
    code_texts = generate_texts(model, tokenizer, tests["gsm8k-code"], config)
    cot_texts = generate_texts(model, tokenizer, tests["gsm8k-cot"], config)
    alpaca_texts = generate_texts(model, tokenizer, tests["alpaca"], config)
    sql_texts = generate_texts(model, tokenizer, tests["sql"], config)
    pandas_texts = generate_texts(model, tokenizer, tests["pandas"], config)
    return {
        "gsm8k_code": code_metrics(code_texts, code_golds, refusal),
        "gsm8k_cot": cot_metrics(cot_texts, cot_golds),
        "alpaca": alpaca_metrics(alpaca_texts, refusal),
        "sql": sql_metrics(sql_texts),
        "pandas": pandas_metrics(pandas_texts),
        "mean_loss": mean_eval_losses(
            model, tokenizer, tests, config["device"]
        ),
    }


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2)


def _fmt(value, digits=3):
    if value is None:
        return "—"
    return f"{value:.{digits}f}"


def make_summary(metrics):
    conditions = metrics["config"]["conditions"]
    lines = [
        "# E3 tier-1: boundary-guided vs standard distillation",
        "",
        "## Pool composition",
        "",
        "| teacher | domain | eligible items |",
        "|---|---|---:|",
    ]
    for row in metrics["pool_composition"]["by_teacher_domain"]:
        lines.append(
            f"| {row['teacher']} | {row['domain']} | {row['n_items']} |"
        )

    boundary_metrics = metrics["task_boundary"]
    if boundary_metrics is not None:
        lines.extend([
            "",
            "## Boundary",
            "",
            "| space | rank | conformal threshold | calibration coverage | pool above threshold |",
            "|---|---:|---:|---:|---:|",
        ])
        for name in ("gradient", "embedding"):
            values = boundary_metrics[name]
            lines.append(
                f"| {name} | {values['rank']} | {values['threshold']:.3f} | "
                f"{values['calibration_coverage']:.3f} | "
                f"{values['pool_above_threshold']} |"
            )

    lines.extend([
        "",
        "## Selection and behavior",
        "",
        "| condition | items | response tokens | refusals | mean s(x) | code exec | code format | code refusal | CoT any-answer | CoT writes code | Alpaca bleed | Alpaca refusal | SQL single statement | pandas compile |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for condition in conditions:
        result = metrics["conditions"][condition]
        selection = result.get("selection") or {}
        evaluation = result["eval"]
        lines.append(
            f"| {condition} | {selection.get('n_items', '—')} | "
            f"{selection.get('token_count', '—')} | "
            f"{selection.get('n_refusal', '—')} | "
            f"{_fmt(selection.get('mean_s'))} | "
            f"{evaluation['gsm8k_code']['exec_acc']:.3f} | "
            f"{evaluation['gsm8k_code']['format_rate']:.3f} | "
            f"{evaluation['gsm8k_code']['refusal_rate']:.3f} | "
            f"{evaluation['gsm8k_cot']['any_acc']:.3f} | "
            f"{evaluation['gsm8k_cot']['writes_code_rate']:.3f} | "
            f"{evaluation['alpaca']['format_bleed_rate']:.3f} | "
            f"{evaluation['alpaca']['refusal_rate']:.3f} | "
            f"{evaluation['sql']['single_statement_valid_rate']:.3f} | "
            f"{evaluation['pandas']['compile_rate']:.3f} |"
        )

    domains = metrics["config"]["data"]["eval_domains"]
    lines.extend([
        "",
        "## Teacher-forced mean test loss",
        "",
        "| condition | " + " | ".join(domains) + " |",
        "|---|" + "---:|" * len(domains),
    ])
    for condition in conditions:
        losses = metrics["conditions"][condition]["eval"]["mean_loss"]
        lines.append(
            f"| {condition} | "
            + " | ".join(f"{losses[domain]:.3f}" for domain in domains)
            + " |"
        )
    return "\n".join(lines) + "\n"


def style_axis(axis):
    axis.set_facecolor("white")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color(MUTED)
    axis.spines["bottom"].set_color(MUTED)
    axis.tick_params(colors=MUTED, labelsize=9)
    axis.set_ylim(0, 1)
    axis.grid(axis="y", color=MUTED, alpha=0.18, linewidth=0.6)
    axis.set_axisbelow(True)


def make_figure(metrics, path):
    conditions = metrics["config"]["conditions"]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6), facecolor="white")
    x_left = np.arange(len(conditions))
    exec_values = [
        metrics["conditions"][condition]["eval"]["gsm8k_code"]["exec_acc"]
        for condition in conditions
    ]
    axes[0].bar(
        x_left,
        exec_values,
        color=[COLORS[condition] for condition in conditions],
        width=0.72,
    )
    axes[0].set_xticks(x_left, [SHORT_LABELS[c] for c in conditions])
    axes[0].set_ylabel("Rate")
    if metrics["config"]["task_boundary"]["domain"] == "gsm8k-code":
        axes[0].set_title("In-T execution accuracy")
    else:
        axes[0].set_title("GSM8K-code execution accuracy")

    metric_names = ("CoT any-answer", "Alpaca format bleed")
    x_right = np.arange(len(metric_names))
    width = 0.12
    offsets = (np.arange(len(conditions)) - (len(conditions) - 1) / 2) * width
    for offset, condition in zip(offsets, conditions):
        evaluation = metrics["conditions"][condition]["eval"]
        axes[1].bar(
            x_right + offset,
            [
                evaluation["gsm8k_cot"]["any_acc"],
                evaluation["alpaca"]["format_bleed_rate"],
            ],
            width=width,
            color=COLORS[condition],
            label=SHORT_LABELS[condition],
        )
    axes[1].set_xticks(x_right, metric_names)
    axes[1].set_ylabel("Rate")
    axes[1].set_title("Leakage and behavior")
    axes[1].legend(frameon=False, ncol=3, fontsize=8, title="condition")

    for axis in axes:
        style_axis(axis)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def release_cuda():
    gc.collect()
    torch.cuda.empty_cache()


def main():
    args = parse_args()
    config = load_config(args.config)
    task_rows, task_non_test_count, task_eligible_count = resolve_task_rows(config)
    print_resolved_config(config)
    if config["task_boundary"]["filter"] is not None:
        print(
            f"[setup][task-filter] domain={config['task_boundary']['domain']} "
            f"substring={config['task_boundary']['filter']!r} "
            f"matched={task_eligible_count}/{task_non_test_count}",
            flush=True,
        )
    if not torch.cuda.is_available():
        raise RuntimeError("E3 tier-1 requires a CUDA device")
    seed_everything(config["seed"])

    tokenizer = AutoTokenizer.from_pretrained(config["model"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    conditions = config["conditions"]
    trained_conditions = [condition for condition in conditions if condition != "base"]
    pool = []
    pool_composition = []
    boundary_metrics = None
    selections = {}
    if trained_conditions:
        pool = load_candidate_pool(config)
        pool = _maybe_extend_pool_with_evol(pool)
        pool_composition = composition_rows(pool)
        print(f"[setup][pool] eligible items={len(pool)}", flush=True)
        for row in pool_composition:
            print(
                f"[setup][pool] teacher={row['teacher']} domain={row['domain']} "
                f"n={row['n_items']}",
                flush=True,
            )

        boundary_metrics = score_pool(config, tokenizer, task_rows, pool)
        selections = build_selections(config, tokenizer, pool)
    tests, code_golds, cot_golds = build_eval_data(config)

    metrics_path = resolve_path(config["outputs"]["metrics"])
    summary_path = resolve_path(config["outputs"]["summary"])
    figure_path = resolve_path(config["outputs"]["figure"])
    metrics = {
        "config": config,
        "pool_composition": {
            "n_items": len(pool),
            "by_teacher_domain": pool_composition,
        },
        "task_boundary": boundary_metrics,
        "conditions": {},
    }

    if "base" in conditions:
        progress("base", "selection", "not applicable")
        progress("base", "train", "skipped (unadapted base model)")
        base_model = build_model(config, with_lora=False)
        base_eval = evaluate_model(
            base_model, tokenizer, tests, code_golds, cot_golds, config
        )
        metrics["conditions"]["base"] = {"selection": None, "eval": base_eval}
        progress(
            "base", "eval",
            f"exec_acc={base_eval['gsm8k_code']['exec_acc']:.3f}",
        )
        write_json(metrics_path, metrics)
        del base_model
        release_cuda()

    if os.environ.get("E3_BEHAV_STOP") == "1":
        fit_n = config["task_boundary"]["fit_rows"]
        spec_n = config["task_boundary"]["spec_rows"]
        config["_behav_rows"] = task_rows[fit_n:spec_n]
    absorption_monitor = None
    bnd_epochs = int(os.environ.get("E3_BND_EPOCHS", "0"))
    needs_monitor = (
        config["training"]["early_stop"] == 1
        or "D_grad_iter" in trained_conditions
    )
    if trained_conditions and needs_monitor:
        absorption_monitor = prepare_absorption_monitor(config, task_rows)

    budget = config["selection"]["response_token_budget"]
    for condition in trained_conditions:
        selected = selections[condition]
        cap = None if condition == "A_all" else budget
        stats = selection_stats(
            selected,
            cap,
            config["selection"]["refusal_fraction"]
            if condition == "F_refusal" else None,
        )
        progress(
            condition,
            "selection",
            f"n={stats['n_items']} tokens={stats['token_count']} "
            f"mean_s={stats['mean_s']:.3f}",
        )

        model = build_model(config, with_lora=True)
        training_metrics = None
        if condition == "D_grad_iter":
            selected, optimizer_steps, round_stats = run_iterative_condition(
                model, tokenizer, config, task_rows, pool,
                selected, absorption_monitor,
            )
            stats = selection_stats(selected, budget)
            training_metrics = {"rounds": round_stats}
        elif condition == "D_less_bnd" and bnd_epochs > 0:
            # geometric absorption bottoms out orders of magnitude before
            # behavioral convergence (E2 separation principle) — train the
            # boundary-filtered diet for the full raised epoch count
            bnd_config = json.loads(json.dumps(config))
            bnd_config["training"]["epochs"] = bnd_epochs
            optimizer_steps = train_condition(
                model, tokenizer, selected, bnd_config, condition=condition
            )
        elif (condition in ("F_atom_boot", "F_atom_boot2", "F_atom_boot3",
                            "F_atom_boot4")
              or condition.startswith("M_")):
            # compute-matched training: the budget caps teacher tokens,
            # not local compute. A targeted corpus is smaller than an
            # unfiltered one, so epochs scale up until optimizer steps
            # match what the full-budget corpus would receive (cap 8);
            # the behavioral probe guards against over-training.
            sel_tokens = max(
                sum(row["_token_count"] for row in selected), 1
            )
            boot_epochs = int(min(8, max(
                config["training"]["epochs"],
                round(config["training"]["epochs"] * budget / sel_tokens),
            )))
            boot_config = json.loads(json.dumps(config))
            boot_config["training"]["epochs"] = boot_epochs
            if condition in ("F_atom_boot3", "F_atom_boot4") \
                    or condition.endswith("_ours") \
                    or condition.endswith("_less"):
                boot_config["training"]["weight_warm_epoch"] = 1
            print(
                f"[{condition}][compute-match] tokens={sel_tokens} "
                f"epochs={boot_epochs}",
                flush=True,
            )
            optimizer_steps = train_condition(
                model, tokenizer, selected, boot_config, condition=condition
            )
        elif (
            absorption_monitor is None
            or config["training"]["early_stop"] != 1
        ):
            optimizer_steps = train_condition(
                model, tokenizer, selected, config, condition=condition
            )
        else:
            training_metrics = {}
            optimizer_steps = train_condition(
                model,
                tokenizer,
                selected,
                config,
                condition=condition,
                absorption_monitor=absorption_monitor,
                training_metrics=training_metrics,
            )
        stats["optimizer_steps"] = optimizer_steps
        progress(
            condition,
            "train",
            f"epochs={config['training']['epochs']} optimizer_steps={optimizer_steps}",
        )

        evaluation = evaluate_model(
            model, tokenizer, tests, code_golds, cot_golds, config
        )
        condition_metrics = {
            "selection": stats,
            "eval": evaluation,
        }
        if training_metrics is not None:
            condition_metrics["training"] = training_metrics
        metrics["conditions"][condition] = condition_metrics
        progress(
            condition,
            "eval",
            f"exec_acc={evaluation['gsm8k_code']['exec_acc']:.3f}",
        )
        write_json(metrics_path, metrics)
        del model
        release_cuda()

    if absorption_monitor is not None:
        release_absorption_monitor(absorption_monitor)

    summary = make_summary(metrics)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(summary)
    make_figure(metrics, figure_path)
    print(f"saved {metrics_path}", flush=True)
    print(f"saved {summary_path}", flush=True)
    print(f"saved {figure_path}", flush=True)


if __name__ == "__main__":
    main()
