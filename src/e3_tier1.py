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
    "H_nll", "E_less_std", "H_smartad_std",
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
}
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
    if not student:
        raise ValueError("E3_STUDENT must not be empty")
    if budget <= 0:
        raise ValueError(f"E3_BUDGET must be positive, got {budget}")
    if not task:
        raise ValueError("E3_TASK must not be empty")
    if "/" in tag or "\\" in tag:
        raise ValueError("E3_TAG must be a filename suffix, not a path")

    config["model"] = student
    config["seed"] = seed
    config["conditions"] = conditions
    config["tag"] = tag
    config["selection"]["response_token_budget"] = budget
    config["training"]["learning_rate"] = float(os.environ.get("E3_LR", config["training"]["learning_rate"]))
    config["task_boundary"]["domain"] = task
    config["task_boundary"]["filter"] = task_filter

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
        ("task_boundary", "spec_rows"): 50,
        ("task_boundary", "fit_rows"): 35,
        ("task_boundary", "calibration_rows"): 15,
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
    cache_root = ROOT / "results" / f"less_cache{config['tag']}"
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
    print(
        f"[setup][E_less_std][score] checkpoints={LESS_WARMUP_EPOCHS} "
        f"pool_items={len(pool)}",
        flush=True,
    )


def score_pool(config, tokenizer, task_rows, pool):
    """Compute gradient and BGE trajectory scores against the same task split."""
    feat_cfg = config["gradient_features"]
    combined = task_rows + pool
    print(
        f"[setup][gradient-features] extracting {len(combined)} rows on cuda",
        flush=True,
    )
    grad = features.extract_features(
        config["model"],
        combined,
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
    less_scores = grad[spec_n:] @ mean_spec_grad
    grad_subspace = boundary.fit_subspace(grad[:fit_n])
    grad_cal = boundary.score(grad_subspace, grad[fit_n:spec_n])
    grad_scores = boundary.score(grad_subspace, grad[spec_n:])
    grad_threshold = conformal_threshold(grad_cal, alpha)

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
    for row, grad_score, less_score, emb_score in zip(
        pool, grad_scores, less_scores, emb_scores
    ):
        row["_grad_score"] = float(grad_score)
        row["_less_score"] = float(less_score)
        row["_emb_score"] = float(emb_score)

    if "E_less_std" in config["conditions"]:
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
        },
        "embedding": {
            "model": emb_cfg["model"],
            "rank": int(emb_subspace.shape[1]),
            "threshold": emb_threshold,
            "calibration_coverage": float(np.mean(emb_cal >= emb_threshold)),
            "pool_above_threshold": int(np.sum(emb_scores >= emb_threshold)),
        },
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
    less_order = np.argsort(
        -np.asarray([row["_less_score"] for row in pool]), kind="stable"
    )
    low_grad_order = grad_order[::-1]

    selections = {
        "A_all": take_prefix(pool, natural, sum(r["_token_count"] for r in pool)),
        "B_random": take_prefix(pool, random_order, budget),
        "C_emb": take_prefix(pool, emb_order, budget),
        "D_grad": take_prefix(pool, grad_order, budget),
        "E_less": take_prefix(pool, less_order, budget),
    }

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


def train_condition(model, tokenizer, rows, config, condition=None):
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
                loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1
    model.config.use_cache = True
    return optimizer_steps


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
        optimizer_steps = train_condition(
            model, tokenizer, selected, config, condition=condition
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
        metrics["conditions"][condition] = {
            "selection": stats,
            "eval": evaluation,
        }
        progress(
            condition,
            "eval",
            f"exec_acc={evaluation['gsm8k_code']['exec_acc']:.3f}",
        )
        write_json(metrics_path, metrics)
        del model
        release_cuda()

    summary = make_summary(metrics)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(summary)
    make_figure(metrics, figure_path)
    print(f"saved {metrics_path}", flush=True)
    print(f"saved {summary_path}", flush=True)
    print(f"saved {figure_path}", flush=True)


if __name__ == "__main__":
    main()
