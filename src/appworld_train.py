#!/usr/bin/env python3
"""Select AppWorld SFT turns and train a baseline LoRA student.

The selection budget is measured in response tokens from the student
tokenizer.  LESS and SmartAD follow the standard baselines used by
``e3_tier1.py``, adapted to the AppWorld episode-turn pool.
"""

from __future__ import annotations

import argparse
import os
import gc
import hashlib
import json
import math
import os
import random
import re
import tempfile
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
POOL_PATH = Path(
    os.environ.get(
        "AW_POOL_PATH", str(ROOT / "data" / "appworld_sft" / "pool.jsonl")
    )
)
OUTPUT_ROOT = ROOT / "results" / "appworld_students"

DEVICE = "cuda"
DTYPE = torch.bfloat16
EPOCHS = 3
GRADIENT_ACCUMULATION = 8
LEARNING_RATE = 1e-4
MAX_PROMPT_TOKENS = 640
MAX_RESPONSE_TOKENS = 512

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.0
LORA_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_BATCH_SIZE = 32

LESS_WARMUP_EPOCHS = 4
LESS_WARMUP_FRACTION = 0.05
LESS_WARMUP_LR = 2e-4
LESS_ADAM_EPS = 1e-8
LESS_PROJECTION_DIM = 64
LESS_TARGET_ROWS = 10


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def tag_name(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value):
        raise argparse.ArgumentTypeError(
            "must start alphanumerically and contain only letters, digits, '.', '_', or '-'"
        )
    return value


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selection",
        required=True,
        choices=("full", "random", "embedding", "less_std", "smartad_std", "ours",
                 "atoms"),
    )
    parser.add_argument(
        "--budget",
        type=positive_int,
        default=20_000,
        help="response-token budget; ignored by --selection full (default: 20000)",
    )
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--student", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--tag", required=True, type=tag_name)
    return parser.parse_args(argv)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def release_cuda() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def load_pool(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"AppWorld SFT pool does not exist: {path}")

    rows: list[dict[str, Any]] = []
    required = ("task_id", "teacher", "turn_index", "prompt", "response", "token_hint")
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: row must be an object")
            missing = [field for field in required if field not in row]
            if missing:
                raise ValueError(f"{path}:{line_number}: missing fields {missing}")
            if row["task_id"] is None:
                raise ValueError(f"{path}:{line_number}: task_id must not be null")
            if not isinstance(row["teacher"], str) or not row["teacher"]:
                raise ValueError(f"{path}:{line_number}: teacher must be a non-empty string")
            if not isinstance(row["turn_index"], int) or isinstance(row["turn_index"], bool):
                raise ValueError(f"{path}:{line_number}: turn_index must be an integer")
            if not isinstance(row["prompt"], str) or not row["prompt"]:
                raise ValueError(f"{path}:{line_number}: prompt must be a non-empty string")
            if not isinstance(row["response"], str) or not row["response"]:
                raise ValueError(f"{path}:{line_number}: response must be a non-empty string")
            copied = dict(row)
            copied["_pool_index"] = len(rows)
            rows.append(copied)

    assert rows, f"AppWorld SFT pool is empty: {path}"
    return rows


def load_tokenizer(student: str) -> Any:
    tokenizer = AutoTokenizer.from_pretrained(student, trust_remote_code=False)
    if tokenizer.eos_token_id is None:
        raise ValueError(f"student tokenizer has no EOS token: {student}")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def attach_response_token_counts(tokenizer: Any, rows: list[dict[str, Any]]) -> None:
    for row in rows:
        # token_hint is deliberately ignored: budgets use the actual student tokenizer.
        row["_response_tokens"] = len(
            tokenizer(row["response"], add_special_tokens=False)["input_ids"]
        )


def encode(
    tokenizer: Any, row: dict[str, Any], device: str = DEVICE
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode one response-only-loss example using the E3 training caps.

    AppWorld SFT prompts are already serialized multi-turn transcripts ending
    at the assistant marker, so they must not be wrapped in another chat turn.
    """
    if row.get("messages"):
        prompt_text = tokenizer.apply_chat_template(
            row["messages"], add_generation_prompt=True, tokenize=False
        )
    else:
        prompt_text = row["prompt"]
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    prompt_ids = prompt_ids[:MAX_PROMPT_TOKENS]
    response_ids = tokenizer(row["response"], add_special_tokens=False)["input_ids"]
    response_ids = response_ids[:MAX_RESPONSE_TOKENS] + [tokenizer.eos_token_id]
    input_ids = torch.tensor([prompt_ids + response_ids], device=device)
    labels = torch.tensor(
        [[-100] * len(prompt_ids) + response_ids], device=device
    )
    return input_ids, labels


def lora_config() -> LoraConfig:
    return LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(LORA_TARGET_MODULES),
    )


def build_base_model(student: str, seed: int) -> Any:
    seed_everything(seed)
    return AutoModelForCausalLM.from_pretrained(
        student,
        torch_dtype=DTYPE,
        trust_remote_code=False,
    ).to(DEVICE)


def build_lora_model(student: str, seed: int) -> Any:
    return get_peft_model(build_base_model(student, seed), lora_config())


def take_ranked_prefix(
    rows: list[dict[str, Any]], order: Sequence[int], budget: int
) -> list[dict[str, Any]]:
    """Take the longest ranked prefix whose real response tokens fit."""
    selected: list[dict[str, Any]] = []
    used = 0
    for pool_index in order:
        row = rows[int(pool_index)]
        token_count = int(row["_response_tokens"])
        if used + token_count > budget:
            break
        selected.append(row)
        used += token_count
    return selected


def embedding_scores(rows: list[dict[str, Any]]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    print(f"[selection][embedding] encoding rows={len(rows)}", flush=True)
    embedder = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
    try:
        embeddings = embedder.encode(
            [row["prompt"] + "\n" + row["response"] for row in rows],
            batch_size=EMBEDDING_BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float64, copy=False)
    finally:
        del embedder
        gc.collect()

    # Embedding-retrieval baseline: pool centrality, i.e. similarity to the
    # normalized mean of the whole pool (there are no held-out probe episodes).
    center = embeddings.mean(axis=0)
    center /= np.linalg.norm(center) + 1e-12
    return embeddings @ center


def stable_seed(name: str) -> int:
    return int.from_bytes(hashlib.sha256(name.encode("utf-8")).digest()[:4], "little")


def less_adapter_state(model: Any) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def less_second_moments(
    model: Any, optimizer: torch.optim.Optimizer
) -> dict[str, torch.Tensor]:
    moments: dict[str, torch.Tensor] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        moment = optimizer.state.get(parameter, {}).get("exp_avg_sq")
        moments[name] = (
            torch.zeros_like(parameter, device="cpu", dtype=torch.float32)
            if moment is None
            else moment.detach().cpu().float().clone()
        )
    return moments


def run_less_warmup(
    student: str,
    seed: int,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    warmup_indices: np.ndarray,
    checkpoint_paths: list[Path],
) -> int:
    """Train the seeded five-percent warmup and retain all four Adam states."""
    model = build_lora_model(student, seed)
    model.config.use_cache = False
    model.train()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=LESS_WARMUP_LR,
    )
    rng = np.random.default_rng(seed)
    optimizer_steps = 0
    try:
        optimizer.zero_grad(set_to_none=True)
        for epoch, checkpoint_path in enumerate(checkpoint_paths, start=1):
            order = rng.permutation(warmup_indices)
            for start in range(0, len(order), GRADIENT_ACCUMULATION):
                chunk = order[start : start + GRADIENT_ACCUMULATION]
                for pool_index in chunk:
                    input_ids, labels = encode(tokenizer, rows[int(pool_index)])
                    loss = model(input_ids=input_ids, labels=labels).loss / len(chunk)
                    loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
            torch.save(
                {
                    "epoch": epoch,
                    "adapter_state": less_adapter_state(model),
                    "second_moments": less_second_moments(model, optimizer),
                },
                checkpoint_path,
            )
            print(
                f"[selection][less_std] warmup_epoch={epoch}/{LESS_WARMUP_EPOCHS} "
                f"optimizer_steps={optimizer_steps}",
                flush=True,
            )
    finally:
        del optimizer, model
        release_cuda()
    return optimizer_steps


def load_less_checkpoint(model: Any, path: Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    named_parameters = dict(model.named_parameters())
    adapter_state = checkpoint["adapter_state"]
    unknown = sorted(set(adapter_state) - set(named_parameters))
    if unknown:
        raise RuntimeError(f"LESS checkpoint has unknown parameters: {unknown[:3]}")
    with torch.no_grad():
        for name, value in adapter_state.items():
            parameter = named_parameters[name]
            parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))
    return checkpoint["second_moments"]


def less_projections(
    lora_b: list[tuple[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    projections: dict[str, torch.Tensor] = {}
    for name, parameter in lora_b:
        generator = torch.Generator(device=parameter.device).manual_seed(stable_seed(name))
        projections[name] = torch.randn(
            parameter.numel(),
            LESS_PROJECTION_DIM,
            generator=generator,
            device=parameter.device,
            dtype=torch.float32,
        ) / math.sqrt(LESS_PROJECTION_DIM)
    return projections


def less_preconditioned_feature(
    lora_b: list[tuple[str, torch.Tensor]],
    projections: dict[str, torch.Tensor],
    second_moments: dict[str, torch.Tensor],
) -> torch.Tensor:
    projected: list[torch.Tensor] = []
    for name, parameter in lora_b:
        if parameter.grad is None:
            raise RuntimeError(f"LESS feature gradient is missing for {name}")
        moment = second_moments[name].to(parameter.device, dtype=torch.float32)
        preconditioned = parameter.grad.detach().flatten().float() / (
            moment.flatten().sqrt() + LESS_ADAM_EPS
        )
        projected.append(projections[name].T @ preconditioned)
    feature = torch.cat(projected)
    return feature / (feature.norm() + 1e-8)


def less_scores(
    student: str,
    seed: int,
    tokenizer: Any,
    rows: list[dict[str, Any]],
) -> tuple[np.ndarray, dict[str, Any]]:
    if len(rows) < LESS_TARGET_ROWS:
        raise ValueError(
            f"less_std requires at least {LESS_TARGET_ROWS} pool rows; found {len(rows)}"
        )

    seed_sequence = np.random.SeedSequence(seed)
    warmup_seed_state, target_seed_state = seed_sequence.spawn(2)
    warmup_seed = int(warmup_seed_state.generate_state(1, dtype=np.uint32)[0])
    target_seed = int(target_seed_state.generate_state(1, dtype=np.uint32)[0])
    warmup_n = max(1, math.ceil(len(rows) * LESS_WARMUP_FRACTION))
    warmup_indices = np.random.default_rng(warmup_seed).choice(
        len(rows), size=warmup_n, replace=False
    )
    target_indices = np.random.default_rng(target_seed).choice(
        len(rows), size=LESS_TARGET_ROWS, replace=False
    )

    with tempfile.TemporaryDirectory(prefix="appworld_less_") as temporary:
        temporary_path = Path(temporary)
        checkpoint_paths = [
            temporary_path / f"epoch_{epoch}.pt"
            for epoch in range(1, LESS_WARMUP_EPOCHS + 1)
        ]
        warmup_steps = run_less_warmup(
            student,
            warmup_seed,
            tokenizer,
            rows,
            warmup_indices,
            checkpoint_paths,
        )

        model = build_lora_model(student, warmup_seed)
        lora_b = [
            (name, parameter)
            for name, parameter in model.named_parameters()
            if "lora_B" in name
        ]
        if not lora_b:
            raise RuntimeError("less_std found no LoRA-B parameters")
        for name, parameter in model.named_parameters():
            parameter.requires_grad = "lora_B" in name
        projections = less_projections(lora_b)
        model.config.use_cache = False
        model.eval()
        checkpoint_scores: list[np.ndarray] = []
        try:
            for epoch, checkpoint_path in enumerate(checkpoint_paths, start=1):
                saved_moments = load_less_checkpoint(model, checkpoint_path)
                second_moments = {
                    name: saved_moments[name].to(parameter.device, dtype=torch.float32)
                    for name, parameter in lora_b
                }
                features: list[np.ndarray] = []
                for row_number, row in enumerate(rows, start=1):
                    input_ids, labels = encode(tokenizer, row)
                    model.zero_grad(set_to_none=True)
                    model(input_ids=input_ids, labels=labels).loss.backward()
                    features.append(
                        less_preconditioned_feature(
                            lora_b, projections, second_moments
                        ).cpu().numpy()
                    )
                    if row_number % 100 == 0 or row_number == len(rows):
                        print(
                            f"[selection][less_std] checkpoint={epoch}/"
                            f"{LESS_WARMUP_EPOCHS} rows={row_number}/{len(rows)}",
                            flush=True,
                        )
                matrix = np.stack(features).astype(np.float64, copy=False)
                target_mean = matrix[target_indices].mean(axis=0)
                target_mean /= np.linalg.norm(target_mean) + 1e-12
                checkpoint_scores.append(matrix @ target_mean)
                del saved_moments, second_moments, features, matrix
        finally:
            del projections, model
            release_cuda()

    # Standard LESS/InfAdam aggregates influence by its best warmup checkpoint.
    scores = np.max(np.stack(checkpoint_scores), axis=0)
    details = {
        "warmup_fraction": LESS_WARMUP_FRACTION,
        "warmup_rows": warmup_n,
        "warmup_indices": [int(index) for index in warmup_indices],
        "warmup_epochs": LESS_WARMUP_EPOCHS,
        "warmup_learning_rate": LESS_WARMUP_LR,
        "warmup_optimizer_steps": warmup_steps,
        "target_rows": LESS_TARGET_ROWS,
        "target_indices": [int(index) for index in target_indices],
        "projection_dim_per_lora_b": LESS_PROJECTION_DIM,
        "checkpoint_aggregation": "max",
    }
    return scores, details


@torch.inference_mode()
def attach_student_nll(
    student: str, seed: int, tokenizer: Any, rows: list[dict[str, Any]]
) -> None:
    """Attach base-student teacher-forced NLL to every teacher candidate."""
    model = build_base_model(student, seed)
    model.eval()
    try:
        for row_number, row in enumerate(rows, start=1):
            input_ids, labels = encode(tokenizer, row)
            row["_student_nll"] = float(
                model(input_ids=input_ids, labels=labels).loss.item()
            )
            if row_number % 100 == 0 or row_number == len(rows):
                print(
                    f"[selection][smartad_std] nll_rows={row_number}/{len(rows)}",
                    flush=True,
                )
    finally:
        del model
        release_cuda()


def task_turn_key(row: dict[str, Any]) -> tuple[str, int]:
    task_key = json.dumps(
        row["task_id"], ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return task_key, int(row["turn_index"])


def select_rows(
    args: argparse.Namespace, tokenizer: Any, rows: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    n_rows = len(rows)
    details: dict[str, Any] = {}

    if args.selection == "full":
        selected = list(rows)
    elif args.selection == "random":
        order = np.random.default_rng(args.seed).permutation(n_rows)
        selected = take_ranked_prefix(rows, order, args.budget)
    elif args.selection == "embedding":
        scores = embedding_scores(rows)
        for row, score in zip(rows, scores):
            row["_selection_score"] = float(score)
        order = np.argsort(-scores, kind="stable")
        selected = take_ranked_prefix(rows, order, args.budget)
        details["embedding_model"] = EMBEDDING_MODEL
        details["retrieval_query"] = "normalized_mean_of_whole_pool"
    elif args.selection == "less_std":
        scores, less_details = less_scores(args.student, args.seed, tokenizer, rows)
        for row, score in zip(rows, scores):
            row["_selection_score"] = float(score)
        order = np.argsort(-scores, kind="stable")
        selected = take_ranked_prefix(rows, order, args.budget)
        details["less"] = less_details
    elif args.selection == "atoms":
        # v3 capability-atom acquisition, smoke-ported from e3_tier1:
        # sparse dictionary on the support episodes' gradient features
        # -> per-atom token quota (demand * budget) -> greedy purchase
        # by quota-limited supply per token. Cached-pool smoke: response
        # features are visible, so no prompt->response bridge needed.
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).resolve().parent))
        import features as _features
        from sklearn.decomposition import MiniBatchDictionaryLearning
        first_turn: dict[str, dict[str, Any]] = {}
        for row in rows:
            key = str(row["task_id"])
            if key not in first_turn or row["turn_index"] < first_turn[key]["turn_index"]:
                first_turn[key] = row
        spec_pool = sorted(first_turn.values(), key=lambda r: str(r["task_id"]))
        rng = np.random.default_rng(args.seed)
        rng.shuffle(spec_pool)
        spec_rows = spec_pool[:45]
        if len(spec_rows) < 20:
            raise ValueError("atoms needs at least 20 distinct episodes")
        combined = spec_rows + rows
        grad = _features.extract_features(
            args.student, combined, proj_dim=64,
            max_prompt_tok=1024, max_resp_tok=512, device="cuda",
        ).astype(np.float64)
        spec_feat = grad[: len(spec_rows)]
        pool_feat = grad[len(spec_rows):]
        dict_a = MiniBatchDictionaryLearning(
            n_components=int(min(64, max(8, len(spec_feat) // 2))),
            alpha=0.05, transform_algorithm="lasso_lars",
            transform_alpha=0.05, random_state=args.seed,
            max_iter=100, batch_size=16,
        )
        dict_a.fit(spec_feat)
        demand = np.abs(dict_a.transform(spec_feat)).mean(axis=0)
        ledger = demand / max(demand.sum(), 1e-12) * args.budget
        codes = np.abs(dict_a.transform(pool_feat))
        tok = np.array([max(int(r["_response_tokens"]), 1) for r in rows])
        expected = float(tok.mean())
        gains = np.array([
            np.minimum(ledger, c * expected).sum() / expected for c in codes
        ])
        selected = []
        spent = 0
        ledger_work = ledger.copy()
        for i in np.argsort(-gains):
            i = int(i)
            gain_now = np.minimum(
                ledger_work, codes[i] * expected
            ).sum() / expected
            if gain_now <= 1e-9:
                continue
            if spent + tok[i] > args.budget:
                continue
            ledger_work = np.maximum(ledger_work - codes[i] * tok[i], 0.0)
            rows[i]["_selection_score"] = float(gain_now)
            selected.append(rows[i])
            spent += int(tok[i])
        if spent < args.budget:
            density = codes.sum(axis=1)
            chosen = {id(r) for r in selected}
            for i in np.argsort(-density):
                i = int(i)
                if id(rows[i]) in chosen or spent + tok[i] > args.budget:
                    continue
                rows[i]["_selection_score"] = float(density[i])
                selected.append(rows[i])
                spent += int(tok[i])
                if spent >= args.budget:
                    break
        details["atoms"] = {
            "n_atoms": int(dict_a.n_components),
            "spec_episodes": len(spec_rows),
            "spent_tokens": int(spent),
            "unspent": int(args.budget - spent),
        }
    elif args.selection == "ours":
        # boundary admission (conformal p-value gate learned from the task's
        # own calibration split) composed with the utility ranking; on a
        # single-domain pool the gate admits ~everything naturally.
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).resolve().parent))
        import features as _features
        import boundary as _boundary
        first_turn: dict[str, dict[str, Any]] = {}
        for row in rows:
            key = str(row["task_id"])
            if key not in first_turn or row["turn_index"] < first_turn[key]["turn_index"]:
                first_turn[key] = row
        spec_pool = sorted(first_turn.values(), key=lambda r: str(r["task_id"]))
        rng = np.random.default_rng(args.seed)
        rng.shuffle(spec_pool)
        spec_rows = spec_pool[:45]
        if len(spec_rows) < 20:
            raise ValueError("ours needs at least 20 distinct episodes")
        fit_n = max(int(len(spec_rows) * 0.75), 10)
        combined = spec_rows + rows
        grad = _features.extract_features(
            args.student, combined, proj_dim=64,
            max_prompt_tok=1024, max_resp_tok=512, device="cuda",
        ).astype(np.float64)
        grad = _boundary.unit(grad)
        subspace = _boundary.fit_subspace(grad[:fit_n])
        cal_scores = sorted(
            _boundary.score(subspace, grad[fit_n:len(spec_rows)])
        )
        pool_scores = _boundary.score(subspace, grad[len(spec_rows):])
        import bisect
        alpha_admit = 0.10 / 2
        n_cal = len(cal_scores)
        admitted = [
            i for i, s in enumerate(pool_scores)
            if (1 + bisect.bisect_right(cal_scores, float(s))) / (n_cal + 1)
            >= alpha_admit
        ]
        util, less_details = less_scores(args.student, args.seed, tokenizer, rows)
        for row, score in zip(rows, util):
            row["_selection_score"] = float(score)
        order = sorted(admitted, key=lambda i: -util[i])
        selected = take_ranked_prefix(rows, order, args.budget)
        details["less"] = less_details
        details["boundary"] = {
            "spec_rows": len(spec_rows), "fit_rows": fit_n,
            "admitted": len(admitted), "pool": len(rows),
            "rule": "p>=alpha/2", "alpha": 0.10,
        }
    else:
        attach_student_nll(args.student, args.seed, tokenizer, rows)
        best_by_task_turn: dict[tuple[str, int], tuple[float, int]] = {}
        for pool_index, row in enumerate(rows):
            key = task_turn_key(row)
            candidate = (float(row["_student_nll"]), pool_index)
            current = best_by_task_turn.get(key)
            if current is None or candidate < current:
                best_by_task_turn[key] = candidate
        order = [
            pool_index
            for _nll, pool_index in sorted(best_by_task_turn.values())
        ]
        selected = take_ranked_prefix(rows, order, args.budget)
        details["candidate_groups"] = len(best_by_task_turn)
        details["discarded_teacher_candidates"] = n_rows - len(best_by_task_turn)

    if not selected:
        raise ValueError(
            f"{args.selection} selected no rows under response-token budget {args.budget}"
        )
    return selected, details


def line_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    offset = 0
    for line in text.splitlines(keepends=True):
        content = line.rstrip("\r\n")
        spans.append((offset, offset + len(content)))
        offset += len(line)
    return spans


def smartad_line_weights(response: str) -> list[float]:
    weights: list[float] = []
    for line in response.splitlines():
        if line.lstrip().startswith("apis.supervisor.complete_task"):
            weights.append(2.0)
        elif "apis." in line:
            weights.append(1.5)
        else:
            weights.append(1.0)
    return weights


def smartad_token_weights(
    tokenizer: Any, row: dict[str, Any], labels: torch.Tensor
) -> torch.Tensor:
    """Align AppWorld API-line weights with the supervised response labels."""
    tokenized = tokenizer(
        row["response"], add_special_tokens=False, return_offsets_mapping=True
    )
    offsets = tokenized["offset_mapping"]
    label_positions = torch.nonzero(labels[0] != -100, as_tuple=False).flatten()
    response_token_count = len(label_positions) - 1  # the last label is EOS
    if response_token_count < 0 or len(offsets) < response_token_count:
        raise RuntimeError("could not align SmartAD weights with response labels")

    spans = line_spans(row["response"])
    line_weights = smartad_line_weights(row["response"])
    weights = torch.ones(labels.shape, device=labels.device, dtype=torch.float32)
    for token_index, (token_start, token_end) in enumerate(
        offsets[:response_token_count]
    ):
        if token_end <= token_start:
            continue
        token_weight = 1.0
        for (line_start, line_end), line_weight in zip(spans, line_weights):
            if token_start < line_end and token_end > line_start:
                token_weight = max(token_weight, line_weight)
        weights[0, label_positions[token_index]] = token_weight
    return weights


def smartad_weighted_loss(
    model: Any,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    token_weights: torch.Tensor,
) -> torch.Tensor:
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


def distillation_config() -> dict[str, Any] | None:
    raw_mode = os.environ.get("AW_DISTILL", "").strip().lower()
    if not raw_mode:
        return None
    if raw_mode not in {"sad", "ddpo", "pbsd", "agentkd"}:
        raise ValueError(
            "AW_DISTILL must be 'sad', 'ddpo', 'pbsd', or 'agentkd'"
        )

    def env_float(name: str, default: float, *, strictly_positive: bool) -> float:
        raw_value = os.environ.get(name, str(default))
        try:
            value = float(raw_value)
        except ValueError as exc:
            raise ValueError(f"{name} must be a number; got {raw_value!r}") from exc
        lower_bound_violated = value <= 0 if strictly_positive else value < 0
        if not math.isfinite(value) or lower_bound_violated:
            qualifier = "positive" if strictly_positive else "non-negative"
            raise ValueError(f"{name} must be a finite {qualifier} number")
        return value

    if raw_mode == "sad":
        return {
            "mode": "sad",
            "reasoning_weight": env_float(
                "AW_SAD_WR", 1.0, strictly_positive=False
            ),
            "action_weight": env_float(
                "AW_SAD_WA", 1.0, strictly_positive=False
            ),
        }

    if raw_mode == "pbsd":
        return {
            "mode": "pbsd",
            "beta": env_float("AW_PBSD_BETA", 0.1, strictly_positive=True),
        }

    if raw_mode == "agentkd":
        return {"mode": "agentkd"}

    raw_epochs = os.environ.get("AW_DDPO_EPOCHS", "1")
    try:
        epochs = int(raw_epochs)
    except ValueError as exc:
        raise ValueError(
            f"AW_DDPO_EPOCHS must be a positive integer; got {raw_epochs!r}"
        ) from exc
    if epochs <= 0:
        raise ValueError("AW_DDPO_EPOCHS must be a positive integer")
    return {
        "mode": "ddpo",
        "epochs": epochs,
        "learning_rate": env_float(
            "AW_DDPO_LR", 5e-6, strictly_positive=True
        ),
        "beta": env_float("AW_DDPO_BETA", 0.1, strictly_positive=True),
    }


def action_spans(response: str) -> list[tuple[int, int]]:
    """Return character spans for code inside paired triple-backtick fences."""
    spans: list[tuple[int, int]] = []
    cursor = 0
    while True:
        opening = response.find("```", cursor)
        if opening < 0:
            break
        after_opening = opening + 3
        newline = response.find("\n", after_opening)
        closing = response.find("```", after_opening)
        if closing < 0:
            break
        if newline < 0 or closing < newline:
            action_start = after_opening
        else:
            # The opening fence's optional language tag is not executable code.
            action_start = newline + 1
            closing = response.find("```", action_start)
            if closing < 0:
                break
        if action_start < closing:
            spans.append((action_start, closing))
        cursor = closing + 3
    return spans


def sad_token_masks(
    tokenizer: Any, row: dict[str, Any], labels: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Align reasoning/action character spans with supervised label tokens."""
    tokenized = tokenizer(
        row["response"], add_special_tokens=False, return_offsets_mapping=True
    )
    offsets = tokenized["offset_mapping"]
    label_positions = torch.nonzero(labels[0] != -100, as_tuple=False).flatten()
    response_token_count = len(label_positions) - 1  # the last label is EOS
    if response_token_count < 0 or len(offsets) < response_token_count:
        raise RuntimeError("could not align SAD spans with response labels")

    action_mask = torch.zeros_like(labels, dtype=torch.bool)
    spans = action_spans(row["response"])
    for token_index, (token_start, token_end) in enumerate(
        offsets[:response_token_count]
    ):
        if token_end <= token_start:
            continue
        if any(
            token_start < span_end and token_end > span_start
            for span_start, span_end in spans
        ):
            action_mask[0, label_positions[token_index]] = True
    reasoning_mask = (labels != -100) & ~action_mask
    return reasoning_mask, action_mask


def sad_loss(
    model: Any,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    reasoning_mask: torch.Tensor,
    action_mask: torch.Tensor,
    reasoning_weight: float,
    action_weight: float,
) -> torch.Tensor:
    logits = model(input_ids=input_ids).logits
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    token_losses = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.shape[-1]),
        shift_labels.reshape(-1),
        reduction="none",
        ignore_index=-100,
    ).reshape_as(shift_labels).float()
    shift_reasoning = reasoning_mask[:, 1:].contiguous()
    shift_action = action_mask[:, 1:].contiguous()

    # A response need not contain both span types.  The absent component is
    # exactly zero; each present component is normalized only by its own count.
    zero = token_losses.sum() * 0.0
    reasoning_loss = (
        token_losses.masked_select(shift_reasoning).mean()
        if shift_reasoning.any()
        else zero
    )
    action_loss = (
        token_losses.masked_select(shift_action).mean()
        if shift_action.any()
        else zero
    )
    return reasoning_weight * reasoning_loss + action_weight * action_loss


def rejected_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "prompt": row["prompt"],
        "messages": row.get("messages"),
        "response": row["_rejected"],
    }


_REASONING_START_RE = re.compile(
    r"^\s*(?:<think(?:ing)?>|<reasoning>|<analysis>|"
    r"(?:#{1,6}\s*)?(?:thought|reasoning)\s*:)",
    flags=re.IGNORECASE,
)


def starts_with_reasoning_segment(response: str) -> bool:
    """Whether an assistant response already opens with an explicit thought."""
    if _REASONING_START_RE.match(response):
        return True
    # Qwen thinking responses can omit the opening tag while retaining the
    # closing tag before the first action block.
    stripped = response.lstrip()
    thought_end = stripped.find("</think>")
    action_start = stripped.find("```")
    return thought_end >= 0 and (action_start < 0 or thought_end < action_start)


def agentkd_row(row: dict[str, Any]) -> dict[str, Any]:
    """Return a row whose target has the optional first-thought prefix."""
    thought = row.get("_thought")
    if (
        not isinstance(thought, str)
        or not thought.strip()
        or starts_with_reasoning_segment(row["response"])
    ):
        return row
    training_row = dict(row)
    training_row["response"] = thought + row["response"]
    return training_row


def completion_log_prob(
    model: Any, input_ids: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    """Return the summed response log-probability used by sequence-level DPO."""
    logits = model(input_ids=input_ids).logits
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    token_nll = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.shape[-1]),
        shift_labels.reshape(-1),
        reduction="none",
        ignore_index=-100,
    ).reshape_as(shift_labels)
    return -token_nll.float().sum()


def ddpo_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if isinstance(row.get("_rejected"), str) and bool(row["_rejected"])
    ]


@torch.inference_mode()
def cache_ddpo_reference_log_probs(
    model: Any, tokenizer: Any, rows: list[dict[str, Any]]
) -> int:
    """Cache initial-policy log-probs before SFT changes the LoRA adapter."""
    preference_rows = ddpo_rows(rows)
    if not preference_rows:
        print("[train][ddpo] reference_rows=0", flush=True)
        return 0

    was_training = model.training
    was_use_cache = model.config.use_cache
    model.config.use_cache = False
    model.eval()
    try:
        with model.disable_adapter():
            for row_number, row in enumerate(preference_rows, start=1):
                chosen_ids, chosen_labels = encode(tokenizer, row)
                rejected_ids, rejected_labels = encode(tokenizer, rejected_row(row))
                row["_ddpo_ref_chosen_logp"] = float(
                    completion_log_prob(model, chosen_ids, chosen_labels).item()
                )
                row["_ddpo_ref_rejected_logp"] = float(
                    completion_log_prob(model, rejected_ids, rejected_labels).item()
                )
                if row_number % 100 == 0 or row_number == len(preference_rows):
                    print(
                        f"[train][ddpo] reference_rows={row_number}/"
                        f"{len(preference_rows)}",
                        flush=True,
                    )
    finally:
        model.train(was_training)
        model.config.use_cache = was_use_cache
    return len(preference_rows)


def train_ddpo(
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    seed: int,
    epochs: int,
    learning_rate: float,
    beta: float,
) -> int:
    """Run distilled DPO after SFT using the cached initial-policy reference."""
    preference_rows = ddpo_rows(rows)
    if not preference_rows:
        print("[train][ddpo] skipped: no rows carry a non-empty _rejected", flush=True)
        return 0
    for row in preference_rows:
        if (
            "_ddpo_ref_chosen_logp" not in row
            or "_ddpo_ref_rejected_logp" not in row
        ):
            raise RuntimeError("DDPO reference log-probabilities were not cached")

    model.config.use_cache = False
    model.train()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=learning_rate,
    )
    rng = np.random.default_rng(seed)
    optimizer_steps = 0
    try:
        optimizer.zero_grad(set_to_none=True)
        for epoch in range(epochs):
            order = rng.permutation(len(preference_rows))
            for start in range(0, len(order), GRADIENT_ACCUMULATION):
                chunk = order[start : start + GRADIENT_ACCUMULATION]
                for row_index in chunk:
                    row = preference_rows[int(row_index)]
                    chosen_ids, chosen_labels = encode(tokenizer, row)
                    rejected_ids, rejected_labels = encode(
                        tokenizer, rejected_row(row)
                    )
                    policy_margin = completion_log_prob(
                        model, chosen_ids, chosen_labels
                    ) - completion_log_prob(model, rejected_ids, rejected_labels)
                    reference_margin = (
                        float(row["_ddpo_ref_chosen_logp"])
                        - float(row["_ddpo_ref_rejected_logp"])
                    )
                    loss = -F.logsigmoid(
                        beta * (policy_margin - reference_margin)
                    ) / len(chunk)
                    loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
            print(
                f"[train][ddpo] epoch={epoch + 1}/{epochs} "
                f"optimizer_steps={optimizer_steps}",
                flush=True,
            )
    finally:
        del optimizer
    model.config.use_cache = True
    return optimizer_steps


_AW_NLL_EMA = None


def train_student(
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    seed: int,
    smartad: bool,
    distillation: dict[str, Any] | None = None,
) -> int:
    """Train for three epochs with true final-chunk gradient accumulation."""
    model.config.use_cache = False
    model.train()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=LEARNING_RATE,
    )
    rng = np.random.default_rng(seed)
    optimizer_steps = 0
    distill_mode = None if distillation is None else distillation["mode"]
    pbsd_nll_ema: float | None = None
    try:
        optimizer.zero_grad(set_to_none=True)
        for epoch in range(EPOCHS):
            order = rng.permutation(len(rows))
            for start in range(0, len(order), GRADIENT_ACCUMULATION):
                chunk = order[start : start + GRADIENT_ACCUMULATION]
                for row_index in chunk:
                    row = rows[int(row_index)]
                    training_row = (
                        agentkd_row(row) if distill_mode == "agentkd" else row
                    )
                    input_ids, labels = encode(tokenizer, training_row)
                    if distill_mode == "sad":
                        reasoning_mask, action_mask = sad_token_masks(
                            tokenizer, row, labels
                        )
                        loss = sad_loss(
                            model,
                            input_ids,
                            labels,
                            reasoning_mask,
                            action_mask,
                            float(distillation["reasoning_weight"]),
                            float(distillation["action_weight"]),
                        ) / len(chunk)
                    elif distill_mode == "ddpo":
                        # DDPO begins with the same vanilla SFT stage.
                        loss = model(
                            input_ids=input_ids, labels=labels
                        ).loss / len(chunk)
                    elif distill_mode == "pbsd":
                        # PBSD jointly learns every teacher positive and the
                        # currently learnable teacher/student preference pairs.
                        chosen_nll = model(
                            input_ids=input_ids, labels=labels
                        ).loss
                        raw_nll = float(chosen_nll.detach().item())
                        pbsd_nll_ema = (
                            raw_nll
                            if pbsd_nll_ema is None
                            else 0.99 * pbsd_nll_ema + 0.01 * raw_nll
                        )
                        loss = chosen_nll
                        if (
                            isinstance(row.get("_rejected"), str)
                            and bool(row["_rejected"])
                            and raw_nll <= pbsd_nll_ema
                        ):
                            rejected_ids, rejected_labels = encode(
                                tokenizer, rejected_row(row)
                            )
                            policy_margin = completion_log_prob(
                                model, input_ids, labels
                            ) - completion_log_prob(
                                model, rejected_ids, rejected_labels
                            )
                            preference_loss = -F.logsigmoid(policy_margin)
                            loss = (
                                loss
                                + float(distillation["beta"]) * preference_loss
                            )
                        loss = loss / len(chunk)
                    elif distill_mode == "agentkd":
                        loss = model(
                            input_ids=input_ids, labels=labels
                        ).loss / len(chunk)
                    elif os.environ.get("AW_UNIORPO") == "1":
                        loss = model(input_ids=input_ids, labels=labels).loss / len(chunk)
                        raw_nll = float(loss.item()) * len(chunk)
                        global _AW_NLL_EMA
                        _AW_NLL_EMA = (raw_nll if _AW_NLL_EMA is None
                                       else 0.99 * _AW_NLL_EMA + 0.01 * raw_nll)
                        if (row.get("_rejected")
                                and raw_nll <= _AW_NLL_EMA):
                            rej_row = {"prompt": row["prompt"],
                                       "messages": row.get("messages"),
                                       "response": row["_rejected"]}
                            ids_r, lab_r = encode(tokenizer, rej_row)
                            resp_c = labels[0][labels[0] != -100]
                            resp_r = lab_r[0][lab_r[0] != -100]
                            shared = 0
                            for a_t, b_t in zip(resp_c.tolist(), resp_r.tolist()):
                                if a_t != b_t:
                                    break
                                shared += 1
                            if shared < min(len(resp_c), len(resp_r)):
                                lab_c2 = labels.clone()
                                lab_r2 = lab_r.clone()
                                for lab in (lab_c2, lab_r2):
                                    pos = (lab[0] != -100).nonzero().squeeze(-1)
                                    lab[0][pos[:shared]] = -100
                                lc = model(input_ids=input_ids, labels=lab_c2).loss
                                lr2 = model(input_ids=ids_r, labels=lab_r2).loss
                                lpc, lpr = -lc, -lr2
                                l1c = torch.log1p(-torch.exp(torch.clamp(lpc, max=-1e-4)))
                                l1r = torch.log1p(-torch.exp(torch.clamp(lpr, max=-1e-4)))
                                ratio = (lpc - l1c) - (lpr - l1r)
                                import torch.nn.functional as _F
                                loss = loss + 0.25 * (-_F.logsigmoid(ratio)) / len(chunk)
                    elif os.environ.get("AW_TOKSEL") == "1":
                        # Token-selective loss against the base model
                        # (Rho-1 flavored, zero extra models: disabling
                        # the adapter recovers the initialization).
                        # Style and format tokens are the ones the base
                        # already predicts, so they get no training
                        # signal and the student keeps its own agentic
                        # style; supervision lands only on the tokens
                        # that carry new content. Threshold is the row
                        # median base loss, so no tuned constant.
                        with torch.no_grad(), model.disable_adapter():
                            base_logits = model(input_ids=input_ids).logits
                        shift_logits = base_logits[0, :-1]
                        shift_labels = labels[0, 1:]
                        keep = shift_labels != -100
                        tok_loss = torch.nn.functional.cross_entropy(
                            shift_logits[keep], shift_labels[keep],
                            reduction="none",
                        )
                        thresh = tok_loss.median()
                        sel = tok_loss >= thresh
                        idx = keep.nonzero().squeeze(-1)[~sel]
                        new_labels = labels.clone()
                        new_labels[0, 1:][idx] = -100
                        if (new_labels != -100).sum() == 0:
                            new_labels = labels
                        loss = model(
                            input_ids=input_ids, labels=new_labels
                        ).loss / len(chunk)
                    elif smartad:
                        token_weights = smartad_token_weights(tokenizer, row, labels)
                        loss = smartad_weighted_loss(
                            model, input_ids, labels, token_weights
                        ) / len(chunk)
                    else:
                        loss = model(input_ids=input_ids, labels=labels).loss / len(chunk)
                    loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
            print(
                f"[train] epoch={epoch + 1}/{EPOCHS} optimizer_steps={optimizer_steps}",
                flush=True,
            )
    finally:
        del optimizer
    model.config.use_cache = True
    return optimizer_steps


def number_summary(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(array.min()),
        "mean": float(array.mean()),
        "max": float(array.max()),
    }


def selection_stats(
    args: argparse.Namespace,
    pool: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    details: dict[str, Any],
) -> dict[str, Any]:
    token_count = int(sum(int(row["_response_tokens"]) for row in selected))
    stats: dict[str, Any] = {
        "pool_rows": len(pool),
        "pool_response_tokens": int(
            sum(int(row["_response_tokens"]) for row in pool)
        ),
        "selected_rows": len(selected),
        "selected_response_tokens": token_count,
        "requested_budget": args.budget,
        "budget_ignored": args.selection == "full",
        "by_teacher": {
            teacher: int(count)
            for teacher, count in sorted(
                Counter(str(row["teacher"]) for row in selected).items()
            )
        },
        **details,
    }
    if args.selection != "full":
        stats["budget_utilization"] = token_count / args.budget
    selected_scores = [
        float(row["_selection_score"])
        for row in selected
        if "_selection_score" in row
    ]
    if selected_scores:
        stats["selected_score"] = number_summary(selected_scores)
    selected_nlls = [
        float(row["_student_nll"])
        for row in selected
        if "_student_nll" in row
    ]
    if selected_nlls:
        stats["selected_nll"] = number_summary(selected_nlls)
    return stats


def manifest_row(row: dict[str, Any]) -> dict[str, Any]:
    output = {key: value for key, value in row.items() if not key.startswith("_")}
    output["pool_index"] = int(row["_pool_index"])
    output["response_tokens"] = int(row["_response_tokens"])
    if "_selection_score" in row:
        output["selection_score"] = float(row["_selection_score"])
    if "_student_nll" in row:
        output["student_nll"] = float(row["_student_nll"])
    return output


def write_manifest(
    path: Path,
    args: argparse.Namespace,
    pool: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    details: dict[str, Any],
    optimizer_steps: int | None = None,
    distillation: dict[str, Any] | None = None,
    ddpo_optimizer_steps: int | None = None,
) -> dict[str, Any]:
    stats = selection_stats(args, pool, selected, details)
    manifest: dict[str, Any] = {
        "version": 1,
        "selection": args.selection,
        "budget": args.budget,
        "seed": args.seed,
        "student": args.student,
        "training": {
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "gradient_accumulation": GRADIENT_ACCUMULATION,
            "dtype": "bfloat16",
            "lora": {
                "r": LORA_R,
                "alpha": LORA_ALPHA,
                "dropout": LORA_DROPOUT,
                "bias": "none",
                "target_modules": list(LORA_TARGET_MODULES),
            },
            "smartad_segment_weighting": (
                args.selection == "smartad_std" and distillation is None
            ),
        },
        "selection_stats": stats,
        "rows": [manifest_row(row) for row in selected],
        "artifact": {
            "path": "adapter",
            "format": "merged_huggingface_model",
        },
    }
    if distillation is not None:
        manifest["training"]["distillation"] = dict(distillation)
        if ddpo_optimizer_steps is not None:
            manifest["training"]["distillation"]["optimizer_steps"] = (
                ddpo_optimizer_steps
            )
    if optimizer_steps is not None:
        manifest["training"]["optimizer_steps"] = optimizer_steps
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.student.strip():
        raise ValueError("--student must not be empty")
    if not torch.cuda.is_available():
        raise RuntimeError("AppWorld student training requires a visible CUDA device")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("the visible CUDA device does not support bfloat16")

    distillation = distillation_config()
    seed_everything(args.seed)
    rows = load_pool(POOL_PATH)
    teacher_filter = os.environ.get("AW_TEACHER", "").strip()
    if teacher_filter:
        rows = [r for r in rows if r["teacher"] == teacher_filter]
        if not rows:
            raise ValueError(f"AW_TEACHER={teacher_filter!r} matches no rows")
        print(f"[pool] teacher filter {teacher_filter}: {len(rows)} rows")
    tokenizer = load_tokenizer(args.student)
    attach_response_token_counts(tokenizer, rows)
    selected, details = select_rows(args, tokenizer, rows)

    output_dir = OUTPUT_ROOT / args.tag
    adapter_dir = output_dir / "adapter"
    manifest_path = output_dir / "selection_manifest.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    if distillation is not None and distillation["mode"] == "ddpo":
        distillation["preference_rows"] = len(ddpo_rows(selected))
    write_manifest(
        manifest_path,
        args,
        rows,
        selected,
        details,
        distillation=distillation,
    )

    model = build_lora_model(args.student, args.seed)
    if distillation is not None and distillation["mode"] == "ddpo":
        cache_ddpo_reference_log_probs(model, tokenizer, selected)
    sft_optimizer_steps = train_student(
        model,
        tokenizer,
        selected,
        args.seed,
        smartad=args.selection == "smartad_std",
        distillation=distillation,
    )
    ddpo_optimizer_steps = 0
    if distillation is not None and distillation["mode"] == "ddpo":
        ddpo_optimizer_steps = train_ddpo(
            model,
            tokenizer,
            selected,
            args.seed,
            epochs=int(distillation["epochs"]),
            learning_rate=float(distillation["learning_rate"]),
            beta=float(distillation["beta"]),
        )
    optimizer_steps = sft_optimizer_steps + ddpo_optimizer_steps

    # The requested adapter directory is self-contained: fold LoRA into the
    # student weights before saving so evaluation does not need a base model plus PEFT.
    merged_model = model.merge_and_unload()
    adapter_dir.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(adapter_dir, safe_serialization=True)
    tokenizer.save_pretrained(adapter_dir)
    manifest = write_manifest(
        manifest_path,
        args,
        rows,
        selected,
        details,
        optimizer_steps=optimizer_steps,
        distillation=distillation,
        ddpo_optimizer_steps=(
            ddpo_optimizer_steps
            if distillation is not None and distillation["mode"] == "ddpo"
            else None
        ),
    )

    selected_tokens = manifest["selection_stats"]["selected_response_tokens"]
    print(
        f"SUMMARY selection={args.selection} rows={len(selected)} "
        f"response_tokens={selected_tokens} optimizer_steps={optimizer_steps} "
        f"saved={output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
