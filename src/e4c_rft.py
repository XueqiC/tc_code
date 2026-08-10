"""E4c: two-stage SFT-then-RFT distillation with demand instrumentation.

The training model is a rank-16 LoRA student.  Probe demand is measured with
the M2/M3 v3 two-model protocol: effective training weights are copied into a
rank-8 reference extractor whose fixed gradient basis matches the atom
dictionary fit from the pilot step features.
"""

import gc
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from sklearn.decomposition import MiniBatchDictionaryLearning, SparseCoder
from transformers import AutoModelForCausalLM

import verifier
from distill_pilot import MODEL, encode, last_number, split
from features import _prepare_extractor, _project_gradients


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "e4c"
FIGURE = ROOT / "results" / "figs" / "e4c.png"

DEVICE = "cuda"
SEED = 0
ACCUM = 8
STAGE1_EPOCHS = 2
STAGE1_LR = 2e-4
STAGE1_STEPS = 22
RFT_ROUNDS = 3
RFT_LR = 1e-4
SAMPLES_PER_QUERY = 4
MAX_KEPT_PER_QUERY = 2
SAMPLE_BATCH_SIZE = 4
EVAL_BATCH_SIZE = 4
MAX_NEW_TOKENS = 320
TEMPERATURE = 0.8
TOP_P = 0.95
PROJ_DIM = 64
N_ATOMS = 128
ALPHA = 0.05

RFT_COLOR = "#2a78d6"
CONTROL_COLOR = "#eb6834"


# Prefer the literal M2/M3 v3 helpers.  The fallback preserves that protocol
# when this file is copied into an environment where m2_v3 is unavailable.
try:
    from m2_v3 import (  # type: ignore
        fit_dictionary,
        lora_modules,
        probe_features,
        sync_ref_weights,
    )
except ImportError:  # pragma: no cover - used only outside this repository
    def fit_dictionary():
        data = np.load(
            ROOT / "results" / "pilot" / "features_gradsteps.npz",
            allow_pickle=True,
        )
        features = data["X_steps"].astype(np.float64)
        features /= np.linalg.norm(features, axis=1, keepdims=True) + 1e-12
        dictionary = MiniBatchDictionaryLearning(
            n_components=N_ATOMS,
            alpha=ALPHA,
            batch_size=256,
            max_iter=150,
            fit_algorithm="lars",
            transform_algorithm="lasso_lars",
            transform_alpha=ALPHA,
            random_state=0,
            n_jobs=4,
        ).fit(features)
        return dictionary.components_

    def lora_modules(peft_model):
        return {
            name: module
            for name, module in peft_model.named_modules()
            if hasattr(module, "base_layer") and hasattr(module, "lora_B")
        }

    def sync_ref_weights(train_mods, ref_mods):
        with torch.no_grad():
            for name, train_module in train_mods.items():
                a_weight = train_module.lora_A["default"].weight
                b_weight = train_module.lora_B["default"].weight
                scale = train_module.scaling["default"]
                effective = train_module.base_layer.weight + scale * (
                    b_weight @ a_weight
                ).to(train_module.base_layer.weight.dtype)
                ref_mods[name].base_layer.weight.copy_(effective)

    def probe_features(ref_model, tokenizer, rows, device, ref_lora_b, projs):
        features = []
        ref_model.train()
        for row in rows:
            input_ids, labels = encode(tokenizer, row, device)
            ref_model.zero_grad(set_to_none=True)
            ref_model(input_ids=input_ids, labels=labels).loss.backward()
            features.append(
                _project_gradients(ref_lora_b, projs).cpu().numpy()
            )
        ref_model.zero_grad(set_to_none=True)
        return np.stack(features).astype(np.float64)


def close_enough(prediction, gold):
    return (
        prediction is not None
        and gold is not None
        and math.isfinite(prediction)
        and math.isfinite(gold)
        and abs(prediction - gold) < 1e-4
    )


def prompt_text(tokenizer, row):
    messages = [{"role": "user", "content": row["prompt"]}]
    return tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )


@torch.inference_mode()
def generate_greedy(model, tokenizer, rows, device):
    """Generate one greedy continuation per row in small padded batches."""
    model.eval()
    texts = []
    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        for start in range(0, len(rows), EVAL_BATCH_SIZE):
            prompts = [
                prompt_text(tokenizer, row)
                for row in rows[start:start + EVAL_BATCH_SIZE]
            ]
            batch = tokenizer(
                prompts,
                add_special_tokens=False,
                padding=True,
                return_tensors="pt",
            ).to(device)
            generated = model.generate(
                **batch,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
            continuations = generated[:, batch["input_ids"].shape[1]:]
            texts.extend(
                tokenizer.batch_decode(continuations, skip_special_tokens=True)
            )
    finally:
        tokenizer.padding_side = old_padding_side
        model.train()
    return texts


@torch.inference_mode()
def sample_completions(model, tokenizer, rows, device):
    """Sample exactly four continuations for every query."""
    model.eval()
    samples = []
    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        for start in range(0, len(rows), SAMPLE_BATCH_SIZE):
            batch_rows = rows[start:start + SAMPLE_BATCH_SIZE]
            prompts = [prompt_text(tokenizer, row) for row in batch_rows]
            batch = tokenizer(
                prompts,
                add_special_tokens=False,
                padding=True,
                return_tensors="pt",
            ).to(device)
            generated = model.generate(
                **batch,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                num_return_sequences=SAMPLES_PER_QUERY,
                pad_token_id=tokenizer.eos_token_id,
            )
            continuations = generated[:, batch["input_ids"].shape[1]:]
            decoded = tokenizer.batch_decode(
                continuations, skip_special_tokens=True
            )
            for index in range(len(batch_rows)):
                left = index * SAMPLES_PER_QUERY
                samples.append(decoded[left:left + SAMPLES_PER_QUERY])
    finally:
        tokenizer.padding_side = old_padding_side
        model.train()
    assert len(samples) == len(rows)
    assert all(len(group) == SAMPLES_PER_QUERY for group in samples)
    return samples


def extract_solution_code(text):
    """Extract the generated program beginning at its solution definition."""
    start = text.find("def solution")
    if start < 0:
        return None
    code = text[start:]
    fence = code.find("```")
    if fence >= 0:
        code = code[:fence]
    code = code.strip()
    return code or None


def select_verified(rows, sampled, golds):
    """Verify, exact-dedupe per query, and retain at most two programs."""
    kept = []
    verified_before_dedupe = 0
    unique_verified = 0
    queries_with_kept = 0
    for row, completions, gold in zip(rows, sampled, golds):
        seen = set()
        query_codes = []
        for completion in completions:
            code = extract_solution_code(completion)
            if code is None:
                continue
            prediction = verifier.run_solution(code)
            if not close_enough(prediction, gold):
                continue
            verified_before_dedupe += 1
            if code in seen:
                continue
            seen.add(code)
            unique_verified += 1
            if len(query_codes) < MAX_KEPT_PER_QUERY:
                query_codes.append(code)
        if query_codes:
            queries_with_kept += 1
        kept.extend(
            {"prompt": row["prompt"], "response": code}
            for code in query_codes
        )
    stats = {
        "sampled": len(rows) * SAMPLES_PER_QUERY,
        "verified_before_dedupe": verified_before_dedupe,
        "unique_verified": unique_verified,
        "queries_with_kept": queries_with_kept,
        "kept": len(kept),
    }
    return kept, stats


def optimizer_for(model, learning_rate):
    parameters = [parameter for parameter in model.parameters()
                  if parameter.requires_grad]
    return torch.optim.AdamW(parameters, lr=learning_rate)


def train_stage1(model, tokenizer, rows, device):
    """Two drop-last epochs: floor(90 / 8) * 2 = 22 optimizer steps."""
    optimizer = optimizer_for(model, STAGE1_LR)
    rng = np.random.default_rng(SEED)
    steps = 0
    model.train()
    optimizer.zero_grad(set_to_none=True)
    for _epoch in range(STAGE1_EPOCHS):
        order = rng.permutation(len(rows))
        full = order[:(len(order) // ACCUM) * ACCUM]
        for offset in range(0, len(full), ACCUM):
            batch_indices = full[offset:offset + ACCUM]
            for index in batch_indices:
                input_ids, labels = encode(tokenizer, rows[index], device)
                (model(input_ids=input_ids, labels=labels).loss / ACCUM).backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            steps += 1
    optimizer.zero_grad(set_to_none=True)
    del optimizer
    if steps != STAGE1_STEPS:
        raise RuntimeError(f"stage 1 made {steps} steps, expected {STAGE1_STEPS}")
    return steps


def train_rft_round(model, tokenizer, rows, device, rng):
    """Train one shuffled epoch, retaining a final partial accumulation."""
    if not rows:
        return 0
    optimizer = optimizer_for(model, RFT_LR)
    order = rng.permutation(len(rows))
    steps = 0
    model.train()
    optimizer.zero_grad(set_to_none=True)
    for offset in range(0, len(order), ACCUM):
        batch_indices = order[offset:offset + ACCUM]
        divisor = len(batch_indices)
        for index in batch_indices:
            input_ids, labels = encode(tokenizer, rows[index], device)
            (model(input_ids=input_ids, labels=labels).loss / divisor).backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        steps += 1
    optimizer.zero_grad(set_to_none=True)
    del optimizer
    return steps


def train_control_steps(model, tokenizer, gold_rows, device, target_steps, rng):
    """Continue gold SFT for an exact number of full eight-example updates."""
    if target_steps == 0:
        return 0
    optimizer = optimizer_for(model, RFT_LR)
    completed = 0
    model.train()
    optimizer.zero_grad(set_to_none=True)
    while completed < target_steps:
        order = rng.permutation(len(gold_rows))
        full = order[:(len(order) // ACCUM) * ACCUM]
        for offset in range(0, len(full), ACCUM):
            for index in full[offset:offset + ACCUM]:
                input_ids, labels = encode(tokenizer, gold_rows[index], device)
                (
                    model(input_ids=input_ids, labels=labels).loss / ACCUM
                ).backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            completed += 1
            if completed == target_steps:
                break
    optimizer.zero_grad(set_to_none=True)
    del optimizer
    return completed


def snapshot_lora(model):
    return {
        name: value.detach().cpu().clone()
        for name, value in get_peft_model_state_dict(model).items()
    }


def restore_lora(model, state):
    set_peft_model_state_dict(model, state)
    restored = get_peft_model_state_dict(model)
    for name, expected in state.items():
        actual = restored[name].detach().cpu()
        if not torch.equal(actual, expected):
            raise RuntimeError(f"failed to restore stage-1 LoRA tensor {name}")


def behavior_metrics(model, tokenizer, code_rows, code_golds,
                     cot_rows, cot_golds, device):
    code_texts = generate_greedy(model, tokenizer, code_rows, device)
    code_predictions = []
    for text in code_texts:
        code = extract_solution_code(text)
        code_predictions.append(
            verifier.run_solution(code) if code is not None else None
        )
    exec_accuracy = float(np.mean([
        close_enough(prediction, gold)
        for prediction, gold in zip(code_predictions, code_golds)
    ]))

    cot_texts = generate_greedy(model, tokenizer, cot_rows, device)
    cot_accuracy = float(np.mean([
        close_enough(last_number(text), gold)
        for text, gold in zip(cot_texts, cot_golds)
    ]))
    return exec_accuracy, cot_accuracy


def measure(model, tokenizer, ref_model, train_mods, ref_mods,
            ref_lora_b, projections, coder, code_rows, code_golds,
            cot_rows, cot_golds, device, arm, round_index,
            optimizer_steps, stage2_steps, kept_count=None,
            require_nondegenerate=False):
    sync_ref_weights(train_mods, ref_mods)
    projected = probe_features(
        ref_model, tokenizer, code_rows, device, ref_lora_b, projections
    )
    codes = np.abs(coder.transform(projected))
    demand_mass = float(codes.sum(axis=1).mean())
    active_atoms = float((codes > 1e-8).sum(axis=1).mean())
    if require_nondegenerate and active_atoms <= 1.0:
        raise RuntimeError(
            f"probe coding is degenerate (active atoms/row={active_atoms})"
        )
    exec_accuracy, cot_accuracy = behavior_metrics(
        model,
        tokenizer,
        code_rows,
        code_golds,
        cot_rows,
        cot_golds,
        device,
    )
    point = {
        "arm": arm,
        "round": round_index,
        "optimizer_steps": optimizer_steps,
        "stage2_optimizer_steps": stage2_steps,
        "exec_accuracy": exec_accuracy,
        "probe_demand_mass": demand_mass,
        "probe_active_atoms_per_row": active_atoms,
        "cot_any_answer_accuracy": cot_accuracy,
    }
    if kept_count is not None:
        point["kept_samples"] = kept_count
    kept_text = "n/a" if kept_count is None else str(kept_count)
    print(
        f"measure arm={arm:7s} round={round_index} step={optimizer_steps:3d} "
        f"exec={exec_accuracy:.3f} cot_any={cot_accuracy:.3f} "
        f"demand={demand_mass:.3f} kept={kept_text}",
        flush=True,
    )
    return point


def save_metrics(metrics):
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "metrics.json").open("w") as handle:
        json.dump(metrics, handle, indent=2)


def make_figure(metrics):
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), facecolor="white")
    for axis in axes:
        axis.set_facecolor("white")
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    curves = (
        ("rft", "RFT", RFT_COLOR, "-"),
        ("control", "Control SFT", CONTROL_COLOR, "--"),
    )
    for key, label, color, linestyle in curves:
        points = metrics["curves"][key]
        steps = [point["optimizer_steps"] for point in points]
        axes[0].plot(
            steps,
            [point["exec_accuracy"] for point in points],
            color=color,
            linestyle=linestyle,
            marker="o",
            linewidth=2,
            label=label,
        )
        axes[1].plot(
            steps,
            [point["probe_demand_mass"] for point in points],
            color=color,
            linestyle=linestyle,
            marker="o",
            linewidth=2,
            label=label,
        )

    axes[0].set_xlabel("optimizer steps")
    axes[0].set_ylabel("exec accuracy")
    axes[0].set_ylim(-0.03, 1.03)
    axes[0].set_title("GSM8K-code execution")
    axes[1].set_xlabel("optimizer steps")
    axes[1].set_ylabel("mean atom-space mass")
    axes[1].set_ylim(bottom=0)
    axes[1].set_title("Probe demand")
    for axis in axes:
        axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURE, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("E4c requires a CUDA GPU")
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    dictionary = fit_dictionary()
    coder = SparseCoder(
        dictionary,
        transform_algorithm="lasso_lars",
        transform_alpha=ALPHA,
    )

    tokenizer, ref_model, ref_lora_b, projections = _prepare_extractor(
        MODEL, PROJ_DIM, DEVICE
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch.manual_seed(SEED)
    train_model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16
    ).to(DEVICE)
    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    train_model = get_peft_model(train_model, lora)
    train_mods = lora_modules(train_model)
    ref_mods = lora_modules(ref_model)
    if set(train_mods) != set(ref_mods):
        raise RuntimeError("module name mismatch between training/reference models")

    train_rows, code_test_rows = split("gsm8k-code")
    _, cot_test_rows = split("gsm8k-cot")
    if len(train_rows) != 90 or len(code_test_rows) != 30:
        raise RuntimeError(
            f"expected stable GSM8K-code split 90/30, got "
            f"{len(train_rows)}/{len(code_test_rows)}"
        )
    if len(cot_test_rows) != 30:
        raise RuntimeError(
            f"expected 30 GSM8K-CoT test rows, got {len(cot_test_rows)}"
        )

    # Cache every reference-derived gold exactly once.
    train_golds = [verifier.run_solution(row["response"])
                   for row in train_rows]
    code_test_golds = [verifier.run_solution(row["response"])
                       for row in code_test_rows]
    cot_test_golds = [last_number(row["response"]) for row in cot_test_rows]
    if any(gold is None for gold in train_golds + code_test_golds):
        raise RuntimeError("a GSM8K-code reference solution did not execute")
    if any(gold is None for gold in cot_test_golds):
        raise RuntimeError("a GSM8K-CoT reference response has no numeric answer")

    metrics = {
        "config": {
            "model": MODEL,
            "seed": SEED,
            "train_rows": len(train_rows),
            "test_rows_per_channel": 30,
            "stage1": {
                "epochs": STAGE1_EPOCHS,
                "learning_rate": STAGE1_LR,
                "gradient_accumulation": ACCUM,
                "optimizer_steps": STAGE1_STEPS,
            },
            "rft": {
                "rounds": RFT_ROUNDS,
                "samples_per_query": SAMPLES_PER_QUERY,
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
                "max_new_tokens": MAX_NEW_TOKENS,
                "max_kept_per_query_per_round": MAX_KEPT_PER_QUERY,
                "epochs_per_round": 1,
                "learning_rate": RFT_LR,
                "gradient_accumulation": ACCUM,
            },
            "probe": {
                "projection_dim_per_module": PROJ_DIM,
                "dictionary_atoms": N_ATOMS,
                "sparse_coder_alpha": ALPHA,
                "dictionary_source": "results/pilot/features_gradsteps.npz",
            },
        },
        "kept_samples_per_round": [],
        "curves": {"rft": [], "control": []},
    }

    train_stage1(train_model, tokenizer, train_rows, DEVICE)
    stage1_state = snapshot_lora(train_model)
    shared = measure(
        train_model,
        tokenizer,
        ref_model,
        train_mods,
        ref_mods,
        ref_lora_b,
        projections,
        coder,
        code_test_rows,
        code_test_golds,
        cot_test_rows,
        cot_test_golds,
        DEVICE,
        arm="shared",
        round_index=0,
        optimizer_steps=STAGE1_STEPS,
        stage2_steps=0,
        require_nondegenerate=True,
    )
    metrics["stage1"] = shared
    metrics["curves"]["rft"].append({**shared, "arm": "rft"})
    metrics["curves"]["control"].append({**shared, "arm": "control"})
    save_metrics(metrics)

    rft_rng = np.random.default_rng(SEED)
    rft_stage2_steps = 0
    for round_index in range(1, RFT_ROUNDS + 1):
        print(f"sampling RFT round {round_index}/{RFT_ROUNDS}", flush=True)
        sampled = sample_completions(
            train_model, tokenizer, train_rows, DEVICE
        )
        kept_rows, kept_stats = select_verified(
            train_rows, sampled, train_golds
        )
        round_steps = train_rft_round(
            train_model, tokenizer, kept_rows, DEVICE, rft_rng
        )
        expected_steps = math.ceil(len(kept_rows) / ACCUM)
        if round_steps != expected_steps:
            raise RuntimeError(
                f"RFT round {round_index} made {round_steps} steps; "
                f"expected {expected_steps}"
            )
        rft_stage2_steps += round_steps
        kept_stats.update({
            "round": round_index,
            "optimizer_steps": round_steps,
            "cumulative_stage2_optimizer_steps": rft_stage2_steps,
        })
        metrics["kept_samples_per_round"].append(kept_stats)
        point = measure(
            train_model,
            tokenizer,
            ref_model,
            train_mods,
            ref_mods,
            ref_lora_b,
            projections,
            coder,
            code_test_rows,
            code_test_golds,
            cot_test_rows,
            cot_test_golds,
            DEVICE,
            arm="rft",
            round_index=round_index,
            optimizer_steps=STAGE1_STEPS + rft_stage2_steps,
            stage2_steps=rft_stage2_steps,
            kept_count=len(kept_rows),
        )
        metrics["curves"]["rft"].append(point)
        save_metrics(metrics)

    # Fork the control from the exact post-stage-1 adapter, not the RFT arm.
    restore_lora(train_model, stage1_state)
    gc.collect()
    torch.cuda.empty_cache()
    control_rng = np.random.default_rng(SEED)
    control_stage2_steps = 0
    for kept_stats in metrics["kept_samples_per_round"]:
        round_index = kept_stats["round"]
        target_steps = kept_stats["optimizer_steps"]
        completed = train_control_steps(
            train_model,
            tokenizer,
            train_rows,
            DEVICE,
            target_steps,
            control_rng,
        )
        if completed != target_steps:
            raise RuntimeError(
                f"control round {round_index} made {completed} steps; "
                f"expected {target_steps}"
            )
        control_stage2_steps += completed
        point = measure(
            train_model,
            tokenizer,
            ref_model,
            train_mods,
            ref_mods,
            ref_lora_b,
            projections,
            coder,
            code_test_rows,
            code_test_golds,
            cot_test_rows,
            cot_test_golds,
            DEVICE,
            arm="control",
            round_index=round_index,
            optimizer_steps=STAGE1_STEPS + control_stage2_steps,
            stage2_steps=control_stage2_steps,
            kept_count=kept_stats["kept"],
        )
        point["matched_rft_round_optimizer_steps"] = target_steps
        metrics["curves"]["control"].append(point)
        save_metrics(metrics)

    if control_stage2_steps != rft_stage2_steps:
        raise RuntimeError(
            f"arm step mismatch: RFT={rft_stage2_steps}, "
            f"control={control_stage2_steps}"
        )
    make_figure(metrics)
    save_metrics(metrics)
    print(f"saved {OUT / 'metrics.json'} and {FIGURE}", flush=True)


if __name__ == "__main__":
    main()
