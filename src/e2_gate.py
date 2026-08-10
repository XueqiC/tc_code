"""E2 gate: compare LoRA residual gradients with executable capability.

The experiment trains the pilot student on gsm8k-code and measures, throughout
training, whether smaller per-example residual gradients correspond to higher
greedy-generation execution success.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from scipy.stats import spearmanr
from transformers import AutoModelForCausalLM, AutoTokenizer

import verifier
from distill_pilot import MODEL, encode, last_number, split


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "e2_gate"
FIG = ROOT / "results" / "figs" / "e2_gate.png"

SEED = 0
EPOCHS = 6
LR = 2e-4
ACCUM = 8
N_TEST = 30
MAX_NEW_TOKENS = 320
CHECKPOINTS = (0, 2, 5, 11, 22, 33, 44, 66)
DOMAINS = ("gsm8k-code", "gsm8k-cot")

PRIMARY = "#2a78d6"
SECONDARY = "#eb6834"
TICK_GRAY = "#6b6a63"


def _correct(pred, gold):
    return pred is not None and gold is not None and abs(pred - gold) < 1e-4


@torch.no_grad()
def _behavior(model, tok, row, gold, dev):
    """Greedily generate and score executable and fallback answers."""
    model.eval()
    messages = [{"role": "user", "content": row["prompt"]}]
    prompt = tok.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    input_ids = tok(
        prompt, add_special_tokens=False, return_tensors="pt"
    ).input_ids.to(dev)
    generated = model.generate(
        input_ids,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        pad_token_id=tok.eos_token_id,
    )
    text = tok.decode(
        generated[0, input_ids.shape[1] :], skip_special_tokens=True
    )

    start = text.find("def solution")
    format_ok = start >= 0
    pred = verifier.run_solution(text[start:]) if format_ok else None
    fallback_pred = pred if pred is not None else last_number(text)
    return bool(_correct(pred, gold)), bool(format_ok), bool(
        _correct(fallback_pred, gold)
    )


def _gradnorm(model, tok, row, dev):
    """Return the reference-response gradient norm over LoRA parameters."""
    model.train()
    model.zero_grad(set_to_none=True)
    input_ids, labels = encode(tok, row, dev)
    model(input_ids=input_ids, labels=labels).loss.backward()

    # With bias="none", PEFT leaves only LoRA parameters trainable. Filtering by
    # name states the geometric quantity explicitly and guards against that
    # trainability convention changing later.
    component_norms = [
        torch.linalg.vector_norm(param.grad.detach().float())
        for name, param in model.named_parameters()
        if "lora_" in name and param.grad is not None
    ]
    if not component_norms:
        raise RuntimeError("No LoRA gradients were produced")
    norm = torch.linalg.vector_norm(torch.stack(component_norms)).item()
    model.zero_grad(set_to_none=True)
    return float(norm)


def measure(model, tok, tests, golds, dev, step, records):
    """Append one gradient/behavior record for every held-out example."""
    # Although this model has no active dropout, preserve RNG state so adding a
    # stochastic layer later cannot make checkpoint measurement alter training.
    cpu_rng_state = torch.get_rng_state()
    cuda_rng_states = torch.cuda.get_rng_state_all()
    try:
        for domain, rows in tests.items():
            for idx, row in enumerate(rows):
                gradnorm = _gradnorm(model, tok, row, dev)
                success, format_ok, any_ok = _behavior(
                    model, tok, row, golds[domain][idx], dev
                )
                records.append(
                    {
                        "step": int(step),
                        "domain": domain,
                        "idx": int(idx),
                        "gradnorm": gradnorm,
                        "success": success,
                        "format": format_ok,
                        "any": any_ok,
                    }
                )
    finally:
        model.zero_grad(set_to_none=True)
        model.train()
        torch.set_rng_state(cpu_rng_state)
        torch.cuda.set_rng_state_all(cuda_rng_states)

    current = [
        record
        for record in records
        if record["step"] == step and record["domain"] == "gsm8k-code"
    ]
    mean_grad = np.mean([record["gradnorm"] for record in current])
    mean_success = np.mean([record["success"] for record in current])
    print(
        f"checkpoint step={step:2d} measured={sum(map(len, tests.values()))} "
        f"code_gradnorm={mean_grad:.4f} code_exec={mean_success:.3f}",
        flush=True,
    )


def _spearman(records):
    gradnorm = np.asarray([record["gradnorm"] for record in records], dtype=float)
    success = np.asarray([record["success"] for record in records], dtype=float)
    result = spearmanr(gradnorm, success)
    return float(result.statistic), float(result.pvalue)


def analyze(records):
    """Compute pooled/per-checkpoint E2 correlations for gsm8k-code."""
    code_records = [r for r in records if r["domain"] == "gsm8k-code"]
    pooled_rho, pooled_p = _spearman(code_records)

    per_checkpoint = {}
    for step in CHECKPOINTS:
        checkpoint_records = [r for r in code_records if r["step"] == step]
        success = {r["success"] for r in checkpoint_records}
        if len(success) > 1:
            rho, pvalue = _spearman(checkpoint_records)
            per_checkpoint[str(step)] = {"rho": rho, "pvalue": pvalue}

    return {
        "pooled": {"rho": pooled_rho, "pvalue": pooled_p},
        "per_checkpoint": per_checkpoint,
        "pass": bool(np.isfinite(pooled_rho) and pooled_rho <= -0.5),
    }


def _style_axis(ax):
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(TICK_GRAY)
    ax.spines["bottom"].set_color(TICK_GRAY)
    ax.tick_params(colors=TICK_GRAY)
    ax.xaxis.label.set_color(TICK_GRAY)
    ax.yaxis.label.set_color(TICK_GRAY)
    ax.title.set_color(TICK_GRAY)


def make_figure(records):
    """Plot separate training curves and pooled gradient-norm deciles."""
    code_records = [r for r in records if r["domain"] == "gsm8k-code"]
    steps = np.asarray(CHECKPOINTS)
    mean_gradnorm = np.asarray(
        [
            np.mean([r["gradnorm"] for r in code_records if r["step"] == step])
            for step in steps
        ]
    )
    mean_success = np.asarray(
        [
            np.mean([r["success"] for r in code_records if r["step"] == step])
            for step in steps
        ]
    )

    # Equal-count rank bins are deciles even if several norms happen to tie.
    gradnorm = np.asarray([r["gradnorm"] for r in code_records])
    success = np.asarray([r["success"] for r in code_records], dtype=float)
    order = np.argsort(gradnorm, kind="stable")
    decile = np.empty(len(order), dtype=int)
    decile[order] = np.minimum(np.arange(len(order)) * 10 // len(order), 9)
    decile_success = np.asarray(
        [np.mean(success[decile == bin_idx]) for bin_idx in range(10)]
    )

    fig = plt.figure(figsize=(10.5, 4.8), facecolor="white")
    grid = fig.add_gridspec(2, 2, width_ratios=(1.15, 1.0), hspace=0.18, wspace=0.34)
    ax_grad = fig.add_subplot(grid[0, 0])
    ax_success = fig.add_subplot(grid[1, 0], sharex=ax_grad)
    ax_bins = fig.add_subplot(grid[:, 1])

    ax_grad.plot(steps, mean_gradnorm, color=PRIMARY, marker="o", linewidth=2)
    ax_grad.set_ylabel("Mean grad norm")
    ax_grad.set_title("Capability across training")
    ax_grad.tick_params(axis="x", labelbottom=False)

    ax_success.plot(steps, mean_success, color=SECONDARY, marker="o", linewidth=2)
    ax_success.set_xlabel("Optimizer step")
    ax_success.set_ylabel("Mean exec success")
    ax_success.set_ylim(-0.03, 1.03)
    ax_success.set_xticks(steps)

    ax_bins.bar(np.arange(1, 11), decile_success, color=SECONDARY, width=0.78)
    ax_bins.set_xlabel("Gradient-norm decile (low to high)")
    ax_bins.set_ylabel("Mean exec success")
    ax_bins.set_title("Pooled gsm8k-code")
    ax_bins.set_xticks(np.arange(1, 11))
    ax_bins.set_ylim(0, 1)

    for ax in (ax_grad, ax_success, ax_bins):
        _style_axis(ax)
    fig.savefig(FIG, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _json_number(value):
    return value if np.isfinite(value) else None


def main():
    # Keep this before model construction so initialization is reproducible.
    torch.manual_seed(SEED)
    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16
    ).to(dev)
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
    model = get_peft_model(model, lora)

    train, code_test = split("gsm8k-code")
    _, cot_test = split("gsm8k-cot")
    tests = {"gsm8k-code": code_test, "gsm8k-cot": cot_test}
    if len(train) != 90 or any(len(rows) != N_TEST for rows in tests.values()):
        raise ValueError("E2 expects 90 training rows and 30 test rows per domain")

    # Reference execution is model-independent, so cache it once for all eight
    # measurement passes.
    golds = {
        domain: [verifier.run_solution(row["response"]) for row in rows]
        for domain, rows in tests.items()
    }
    records = []
    measure(model, tok, tests, golds, dev, 0, records)

    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad], lr=LR
    )
    rng = np.random.default_rng(SEED)
    step = 0
    for _epoch in range(EPOCHS):
        order = rng.permutation(len(train))
        # Match the pilot's 11 updates per 90-row epoch. The final two rows do
        # not form a complete accumulation batch and are intentionally dropped.
        full_epoch = order[: (len(order) // ACCUM) * ACCUM]
        for microstep, row_idx in enumerate(full_epoch, start=1):
            input_ids, labels = encode(tok, train[row_idx], dev)
            loss = model(input_ids=input_ids, labels=labels).loss / ACCUM
            loss.backward()
            if microstep % ACCUM == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                if step in CHECKPOINTS:
                    measure(model, tok, tests, golds, dev, step, records)

    if step != CHECKPOINTS[-1]:
        raise RuntimeError(f"Expected 66 optimizer steps, got {step}")
    optimizer.zero_grad(set_to_none=True)

    analysis = analyze(records)
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.parent.mkdir(parents=True, exist_ok=True)
    make_figure(records)

    serializable_analysis = {
        "pooled": {
            "rho": _json_number(analysis["pooled"]["rho"]),
            "pvalue": _json_number(analysis["pooled"]["pvalue"]),
        },
        "per_checkpoint": {
            checkpoint: {
                "rho": _json_number(values["rho"]),
                "pvalue": _json_number(values["pvalue"]),
            }
            for checkpoint, values in analysis["per_checkpoint"].items()
        },
        "pass": analysis["pass"],
    }
    config = {
        "model": MODEL,
        "seed": SEED,
        "train_domain": "gsm8k-code",
        "train_rows": len(train),
        "test_domains": list(DOMAINS),
        "test_rows_per_domain": N_TEST,
        "epochs": EPOCHS,
        "optimizer": "AdamW",
        "learning_rate": LR,
        "batch_size": 1,
        "gradient_accumulation": ACCUM,
        "optimizer_steps": step,
        "checkpoints": list(CHECKPOINTS),
        "lora": {
            "r": 16,
            "alpha": 32,
            "dropout": 0.0,
            "targets": list(lora.target_modules),
        },
        "generation": {"strategy": "greedy", "max_new_tokens": MAX_NEW_TOKENS},
    }
    with open(OUT / "records.json", "w") as handle:
        json.dump(
            {"config": config, "records": records, "analysis": serializable_analysis},
            handle,
            indent=2,
        )

    rho = analysis["pooled"]["rho"]
    if analysis["pass"]:
        print("E2 GATE: PASS")
    else:
        print(f"E2 GATE: FAIL (rho={rho:.3f})")


if __name__ == "__main__":
    main()
