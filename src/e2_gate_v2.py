"""E2-v2: drift-corrected re-test of the residual-gradient gate.

The primary analysis is deliberately model-free: it reuses the E2-v1 records
and removes checkpoint-level scale drift before pooling the observations.  The
optional secondary analysis recreates the final E2-v1 student and measures
residual gradients on its own greedy trajectories as well as on the teacher
references.
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

# The managed experiment environment may not expose the user's matplotlib
# config directory.  Keep primary runs warning-free while respecting overrides.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/tc-alignment-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parent.parent
V1_RECORDS = ROOT / "results" / "e2_gate" / "records.json"
OUT = ROOT / "results" / "e2_gate" / "v2_analysis.json"
FIG = ROOT / "results" / "figs" / "e2_gate_v2.png"

GATE_THRESHOLD = -0.5
PRIMARY = "#2a78d6"
SECONDARY = "#eb6834"
TICK_GRAY = "#6b6a63"


def _spearman(x, y):
    """Return a JSON-safe Spearman result, including constant-input cases."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return {"rho": None, "pvalue": None}
    result = spearmanr(x, y)
    rho = float(result.statistic)
    pvalue = float(result.pvalue)
    return {
        "rho": rho if np.isfinite(rho) else None,
        "pvalue": pvalue if np.isfinite(pvalue) else None,
    }


def _load_v1_records(path):
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing E2-v1 records: {path}. Run src/e2_gate.py first."
        )
    with path.open() as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or not isinstance(payload.get("records"), list):
        raise ValueError(f"Expected an object with a records list in {path}")

    required = {"step", "domain", "idx", "gradnorm", "success"}
    for record_idx, record in enumerate(payload["records"]):
        missing = required.difference(record)
        if missing:
            raise ValueError(
                f"Record {record_idx} in {path} is missing {sorted(missing)}"
            )
    return payload


def analyze_primary(v1_payload):
    """Z-normalize within checkpoint and compute the E2-v2 gate statistic."""
    code_records = [
        record
        for record in v1_payload["records"]
        if record["domain"] == "gsm8k-code"
    ]
    if not code_records:
        raise ValueError("E2-v1 records contain no gsm8k-code rows")

    grouped = defaultdict(list)
    for record in code_records:
        gradnorm = float(record["gradnorm"])
        if not np.isfinite(gradnorm):
            raise ValueError(
                f"Non-finite gradnorm at step={record['step']} idx={record['idx']}"
            )
        grouped[int(record["step"])].append(record)

    normalized_records = []
    per_checkpoint = {}
    for step in sorted(grouped):
        rows = grouped[step]
        gradnorm = np.asarray([row["gradnorm"] for row in rows], dtype=float)
        success = np.asarray([row["success"] for row in rows], dtype=float)
        mean = float(gradnorm.mean())
        # A checkpoint is the population being normalized, so use ddof=0.
        std = float(gradnorm.std(ddof=0))
        if not np.isfinite(std) or std <= 0.0:
            raise ValueError(f"Cannot z-normalize checkpoint {step}: std={std}")
        z_gradnorm = (gradnorm - mean) / std

        checkpoint_correlation = _spearman(gradnorm, success)
        per_checkpoint[str(step)] = {
            "n": len(rows),
            "successes": int(success.sum()),
            "gradnorm_mean": mean,
            "gradnorm_std": std,
            **checkpoint_correlation,
        }
        for row, z_value in zip(rows, z_gradnorm):
            normalized_records.append(
                {
                    "step": int(step),
                    "idx": int(row["idx"]),
                    "gradnorm": float(row["gradnorm"]),
                    "z_gradnorm": float(z_value),
                    "success": bool(row["success"]),
                }
            )

    pooled_z = _spearman(
        [row["z_gradnorm"] for row in normalized_records],
        [row["success"] for row in normalized_records],
    )
    pooled_raw = _spearman(
        [row["gradnorm"] for row in normalized_records],
        [row["success"] for row in normalized_records],
    )
    rho = pooled_z["rho"]
    passed = bool(rho is not None and rho <= GATE_THRESHOLD)

    v1_stored_rho = (
        v1_payload.get("analysis", {}).get("pooled", {}).get("rho")
    )
    return {
        "metric": "within-checkpoint z-normalized residual gradnorm",
        "normalization": {
            "group": "checkpoint",
            "mean": "checkpoint mean",
            "std": "checkpoint population std (ddof=0)",
        },
        "n_records": len(normalized_records),
        "n_checkpoints": len(grouped),
        "pooled_z_normalized": pooled_z,
        "pooled_raw_reference": {
            **pooled_raw,
            "v1_reported_rho": v1_stored_rho,
        },
        "per_checkpoint": per_checkpoint,
        "gate": {
            "operator": "<=",
            "threshold": GATE_THRESHOLD,
            "pass": passed,
        },
        "records": normalized_records,
    }


def _style_axis(axis):
    axis.set_facecolor("white")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color(TICK_GRAY)
    axis.spines["bottom"].set_color(TICK_GRAY)
    axis.tick_params(colors=TICK_GRAY)
    axis.xaxis.label.set_color(TICK_GRAY)
    axis.yaxis.label.set_color(TICK_GRAY)
    axis.title.set_color(TICK_GRAY)


def make_figure(primary_analysis):
    """Plot z-scored observations/deciles and checkpoint correlations."""
    rows = primary_analysis["records"]
    z_gradnorm = np.asarray([row["z_gradnorm"] for row in rows], dtype=float)
    success = np.asarray([row["success"] for row in rows], dtype=float)

    # Equal-count rank bins avoid ambiguity if gradient norms tie.
    order = np.argsort(z_gradnorm, kind="stable")
    decile = np.empty(len(order), dtype=int)
    decile[order] = np.minimum(np.arange(len(order)) * 10 // len(order), 9)
    decile_x = np.asarray(
        [z_gradnorm[decile == bin_idx].mean() for bin_idx in range(10)]
    )
    decile_success = np.asarray(
        [success[decile == bin_idx].mean() for bin_idx in range(10)]
    )

    # Fixed jitter makes reruns byte-level stable apart from image metadata.
    jitter = np.random.default_rng(0).uniform(-0.045, 0.045, size=len(success))

    checkpoints = list(primary_analysis["per_checkpoint"])
    checkpoint_rho = [
        primary_analysis["per_checkpoint"][step]["rho"] for step in checkpoints
    ]
    bar_values = [np.nan if rho is None else rho for rho in checkpoint_rho]

    fig, (ax_scatter, ax_bars) = plt.subplots(
        1, 2, figsize=(10.5, 4.4), facecolor="white"
    )
    ax_scatter.scatter(
        z_gradnorm,
        success + jitter,
        color=PRIMARY,
        alpha=0.50,
        edgecolors="none",
        s=24,
        label="checkpoint/sample",
    )
    ax_scatter.plot(
        decile_x,
        decile_success,
        color=SECONDARY,
        marker="o",
        linewidth=2.2,
        markersize=5,
        label="per-decile success",
    )
    ax_scatter.set_xlabel("Within-checkpoint z-scored gradnorm")
    ax_scatter.set_ylabel("Execution success (jittered)")
    ax_scatter.set_title("Pooled gsm8k-code")
    ax_scatter.set_yticks((0, 1), labels=("failure", "success"))
    ax_scatter.set_ylim(-0.12, 1.12)
    legend = ax_scatter.legend(frameon=False, loc="best")
    for text in legend.get_texts():
        text.set_color(TICK_GRAY)

    positions = np.arange(len(checkpoints))
    ax_bars.bar(positions, bar_values, color=PRIMARY, width=0.72)
    ax_bars.axhline(
        GATE_THRESHOLD, color=TICK_GRAY, linestyle="--", linewidth=1.5
    )
    for position, rho in zip(positions, checkpoint_rho):
        if rho is None:
            ax_bars.text(
                position,
                0.025,
                "n/a",
                ha="center",
                va="bottom",
                rotation=90,
                color=TICK_GRAY,
                fontsize=8,
            )
    ax_bars.set_xticks(positions, labels=checkpoints)
    ax_bars.set_xlabel("Optimizer step")
    ax_bars.set_ylabel("Spearman rho")
    ax_bars.set_title("Per-checkpoint correlation")
    ax_bars.set_ylim(-1.0, 0.15)

    for axis in (ax_scatter, ax_bars):
        _style_axis(axis)
    fig.tight_layout(w_pad=2.5)
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _train_final_student():
    """Recreate the E2-v1 student after its 66th optimizer step."""
    # Heavy/GPU imports stay behind --with-self-residual so primary is CPU-only.
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import e2_gate as v1

    if not torch.cuda.is_available():
        raise RuntimeError("--with-self-residual requires a CUDA GPU")

    torch.manual_seed(v1.SEED)
    device = "cuda"
    tokenizer = AutoTokenizer.from_pretrained(v1.MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        v1.MODEL, dtype=torch.bfloat16
    ).to(device)
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

    train, code_test = v1.split("gsm8k-code")
    if len(train) != 90 or len(code_test) != v1.N_TEST:
        raise ValueError("E2 expects 90 training rows and 30 gsm8k-code test rows")

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=v1.LR,
    )
    model.train()
    rng = np.random.default_rng(v1.SEED)
    step = 0
    for epoch in range(v1.EPOCHS):
        order = rng.permutation(len(train))
        full_epoch = order[: (len(order) // v1.ACCUM) * v1.ACCUM]
        for microstep, row_idx in enumerate(full_epoch, start=1):
            input_ids, labels = v1.encode(tokenizer, train[row_idx], device)
            loss = model(input_ids=input_ids, labels=labels).loss / v1.ACCUM
            loss.backward()
            if microstep % v1.ACCUM == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
        print(
            f"secondary training epoch {epoch + 1}/{v1.EPOCHS} "
            f"complete ({step} steps)",
            flush=True,
        )

    if step != v1.CHECKPOINTS[-1]:
        raise RuntimeError(f"Expected 66 optimizer steps, got {step}")
    optimizer.zero_grad(set_to_none=True)
    return model, tokenizer, code_test, device, lora, step


def _generate_and_verify(model, tokenizer, row, gold, device):
    """Recompute E2-v1 greedy behavior while retaining the trajectory."""
    import torch

    import e2_gate as v1
    import verifier

    model.eval()
    messages = [{"role": "user", "content": row["prompt"]}]
    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    input_ids = tokenizer(
        prompt, add_special_tokens=False, return_tensors="pt"
    ).input_ids.to(device)
    with torch.no_grad():
        generated = model.generate(
            input_ids,
            max_new_tokens=v1.MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(
        generated[0, input_ids.shape[1] :], skip_special_tokens=True
    )

    start = text.find("def solution")
    format_ok = start >= 0
    prediction = verifier.run_solution(text[start:]) if format_ok else None
    success = bool(v1._correct(prediction, gold))
    return text, success, format_ok, prediction


def analyze_secondary(primary_analysis):
    """Measure final-state residuals on generated and reference responses."""
    import e2_gate as v1
    import verifier

    model, tokenizer, test_rows, device, lora, step = _train_final_student()
    v1_final = {
        int(row["idx"]): float(row["gradnorm"])
        for row in primary_analysis["records"]
        if int(row["step"]) == step
    }

    records = []
    for idx, row in enumerate(test_rows):
        gold = verifier.run_solution(row["response"])
        generated_text, success, format_ok, prediction = _generate_and_verify(
            model, tokenizer, row, gold, device
        )

        # The own-response norm is defined for all rows so it can be correlated
        # with the binary success outcome across the full 30-row test set.
        generated_row = {"prompt": row["prompt"], "response": generated_text}
        own_gradnorm = v1._gradnorm(model, tokenizer, generated_row, device)
        teacher_gradnorm = v1._gradnorm(model, tokenizer, row, device)
        records.append(
            {
                "idx": int(idx),
                "success": success,
                "format": bool(format_ok),
                "gold": gold,
                "prediction": prediction,
                "generated_response": generated_text,
                "own_trajectory_gradnorm": own_gradnorm,
                "teacher_reference_gradnorm": teacher_gradnorm,
                "v1_final_teacher_reference_gradnorm": v1_final.get(idx),
            }
        )
        print(
            f"secondary row={idx:02d} success={int(success)} "
            f"own_gradnorm={own_gradnorm:.4f} "
            f"teacher_gradnorm={teacher_gradnorm:.4f}",
            flush=True,
        )

    correlation = _spearman(
        [row["own_trajectory_gradnorm"] for row in records],
        [row["success"] for row in records],
    )
    successful_records = [row for row in records if row["success"]]
    return {
        "requested": True,
        "metric": "final-state residual gradnorm on greedy student trajectory",
        "correlation_population": "all 30 gsm8k-code test rows",
        "own_trajectory_vs_success": correlation,
        "n_records": len(records),
        "n_successes": len(successful_records),
        "successful_row_indices": [row["idx"] for row in successful_records],
        "training": {
            "model": v1.MODEL,
            "seed": v1.SEED,
            "epochs": v1.EPOCHS,
            "optimizer": "AdamW",
            "learning_rate": v1.LR,
            "gradient_accumulation": v1.ACCUM,
            "optimizer_steps": step,
            "lora": {
                "r": lora.r,
                "alpha": lora.lora_alpha,
                "dropout": lora.lora_dropout,
                "targets": sorted(lora.target_modules),
            },
            "generation": {
                "strategy": "greedy",
                "max_new_tokens": v1.MAX_NEW_TOKENS,
            },
        },
        "records": records,
    }


def _print_primary(primary_analysis):
    raw = primary_analysis["pooled_raw_reference"]
    print(
        f"E2-v1 pooled raw reference: rho={raw['rho']:.3f} "
        f"(p={raw['pvalue']:.3g})"
    )
    for step, values in primary_analysis["per_checkpoint"].items():
        rho = "n/a" if values["rho"] is None else f"{values['rho']:.3f}"
        print(
            f"checkpoint step={int(step):2d}: rho={rho}, "
            f"success={values['successes']}/{values['n']}"
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Re-test E2 using within-checkpoint z-normalized gradnorms."
    )
    parser.add_argument(
        "--with-self-residual",
        action="store_true",
        help="retrain the final student on GPU and add own-trajectory residuals",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    v1_payload = _load_v1_records(V1_RECORDS)
    primary_analysis = analyze_primary(v1_payload)
    make_figure(primary_analysis)
    _print_primary(primary_analysis)

    if args.with_self_residual:
        secondary_analysis = analyze_secondary(primary_analysis)
        secondary_rho = secondary_analysis["own_trajectory_vs_success"]["rho"]
        rho_text = "n/a" if secondary_rho is None else f"{secondary_rho:.3f}"
        print(f"E2-v2 own-trajectory residual: rho={rho_text}")
    else:
        secondary_analysis = {
            "requested": False,
            "status": "not run; pass --with-self-residual to enable GPU analysis",
        }

    output = {
        "method": "E2-v2 drift-corrected gate (amendment A1)",
        "source_records": str(V1_RECORDS.relative_to(ROOT)),
        "primary": primary_analysis,
        "secondary": secondary_analysis,
        "artifacts": {
            "analysis": str(OUT.relative_to(ROOT)),
            "figure": str(FIG.relative_to(ROOT)),
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as handle:
        json.dump(output, handle, indent=2, allow_nan=False)

    rho = primary_analysis["pooled_z_normalized"]["rho"]
    verdict = "PASS" if primary_analysis["gate"]["pass"] else "FAIL"
    rho_text = "nan" if rho is None else f"{rho:.3f}"
    print(f"E2-V2 GATE (z-norm): {verdict} (rho={rho_text})")


if __name__ == "__main__":
    main()
