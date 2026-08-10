"""M2/M3: atom absorption dynamics and the carrier of behavior change.

This experiment replays the E2 gsm8k-code training run while measuring two
quantities in the fixed atom basis learned by ``atoms_pilot.py``:

* M2: absolute sparse codes of held-out reference-gradient features.
* M3: absolute sparse codes of consecutive mean probe-feature deltas.

Outputs:
  results/m2_m3/records.npz
  results/m2_m3/summary.md
  results/figs/m2_m3.png
"""

import json
import math
import os
from pathlib import Path

# Avoid matplotlib trying to create a config directory outside managed runs.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/tc-alignment-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from sklearn.decomposition import MiniBatchDictionaryLearning
from transformers import AutoModelForCausalLM, AutoTokenizer

import verifier
from distill_pilot import MODEL, encode, split
from features import stable_seed


ROOT = Path(__file__).resolve().parent.parent
BLOCK_FEATURES = ROOT / "results" / "pilot" / "features_gradsteps.npz"
OUT = ROOT / "results" / "m2_m3"
FIG = ROOT / "results" / "figs" / "m2_m3.png"

SEED = 0
EPOCHS = 6
LR = 2e-4
ACCUM = 8
N_TEST = 30
MAX_NEW_TOKENS = 320
CHECKPOINTS = (0, 2, 5, 11, 22, 33, 44, 66)

N_ATOMS = 128
SPARSITY_ALPHA = 0.05
EXPECTED_PROJ_DIM = 64
LORA_TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

SUCCESS_COLOR = "#eb6834"
DELTA_COLOR = "#2a78d6"
TICK_GRAY = "#6b6a63"


def _unit_rows(array):
    """L2-normalize a float64 matrix in the same way as atoms_pilot."""
    array = np.asarray(array, dtype=np.float64)
    return array / (np.linalg.norm(array, axis=1, keepdims=True) + 1e-12)


def _fit_dictionary():
    """Refit the fixed 128-atom dictionary from cached block features."""
    if not BLOCK_FEATURES.is_file():
        raise FileNotFoundError(
            f"Missing block features: {BLOCK_FEATURES}. "
            "Run src/grad_features_steps.py first."
        )
    with np.load(BLOCK_FEATURES, allow_pickle=True) as payload:
        if "X_steps" not in payload:
            raise ValueError(f"{BLOCK_FEATURES} has no X_steps array")
        block_features = _unit_rows(payload["X_steps"])

    dictionary = MiniBatchDictionaryLearning(
        n_components=N_ATOMS,
        alpha=SPARSITY_ALPHA,
        batch_size=256,
        max_iter=150,
        fit_algorithm="lars",
        transform_algorithm="lasso_lars",
        transform_alpha=SPARSITY_ALPHA,
        random_state=SEED,
        n_jobs=4,
    )
    dictionary.fit(block_features)
    if dictionary.components_.shape != (N_ATOMS, block_features.shape[1]):
        raise RuntimeError(
            "Unexpected dictionary shape: "
            f"{dictionary.components_.shape}, expected "
            f"({N_ATOMS}, {block_features.shape[1]})"
        )
    return dictionary


def _prepare_model(device):
    """Construct the exact rank-16 E2 student and its tokenizer."""
    torch.manual_seed(SEED)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16
    ).to(device)
    adapter = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(LORA_TARGETS),
    )
    return get_peft_model(model, adapter), tokenizer


def _prepare_probe_projections(model, dictionary_dim, device):
    """Create the original rank-8 probe's fixed Gaussian JL matrices."""
    lora_b = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if "lora_B.probe" in name
    ]
    if not lora_b:
        raise RuntimeError("The E2 model has no probe LoRA-B parameters")
    if dictionary_dim % len(lora_b):
        raise ValueError(
            f"Dictionary dimension {dictionary_dim} is not divisible by "
            f"the {len(lora_b)} LoRA-B modules"
        )
    projection_dim = dictionary_dim // len(lora_b)
    if projection_dim != EXPECTED_PROJ_DIM:
        raise ValueError(
            f"Expected {EXPECTED_PROJ_DIM} JL dimensions per module, "
            f"found {projection_dim}"
        )

    projections = {}
    for name, parameter in lora_b:
        default_name = name.replace(".probe.", ".default.")
        generator = torch.Generator(device=device).manual_seed(
            stable_seed(default_name)
        )
        projections[name] = torch.randn(
            parameter.numel(),
            projection_dim,
            generator=generator,
            device=device,
            dtype=torch.float32,
        ) / math.sqrt(projection_dim)
        assert projections[name].shape[0] == parameter.numel(), (
            f"Projection for {name} has {projections[name].shape[0]} rows, "
            f"expected {parameter.numel()}"
        )
    return lora_b, projections


def _normalized_projection(parts):
    vector = torch.cat(parts)
    return vector / (vector.norm() + 1e-8)


def _project_current_gradients(lora_b, projections):
    """Project and concatenate current LoRA-B gradients."""
    parts = []
    for name, parameter in lora_b:
        if parameter.grad is None:
            raise RuntimeError(f"No gradient was produced for {name}")
        parts.append(
            projections[name].T @ parameter.grad.detach().flatten().float()
        )
    return _normalized_projection(parts)


def _extract_probe_features(model, tokenizer, rows, device, lora_b, projections):
    """Extract one rank-8 probe-gradient feature per row."""
    features = []
    model.train()
    try:
        for row in rows:
            model.zero_grad(set_to_none=True)
            input_ids, labels = encode(tokenizer, row, device)
            model(input_ids=input_ids, labels=labels).loss.backward()
            feature = _project_current_gradients(lora_b, projections)
            features.append(feature.detach().cpu().numpy())
    finally:
        # Checkpoint instrumentation must never leak gradients into training.
        model.zero_grad(set_to_none=True)
        model.train()
    return np.stack(features).astype(np.float64, copy=False)


def _suspend_merged_adapter_markers(model):
    """Let PEFT apply the probe on top of the already-merged default weights.

    PEFT's merged-layer fast path otherwise skips every active adapter. The
    default weights stay folded into each base layer; only the bookkeeping is
    suspended until cleanup so the zero-init probe participates in forward.
    """
    states = []
    for module in model.modules():
        merged_adapters = getattr(module, "merged_adapters", None)
        if not merged_adapters or not hasattr(module, "lora_B"):
            continue
        names = tuple(merged_adapters)
        if names != ("default",):
            raise RuntimeError(f"Unexpected merged adapters: {names}")
        states.append((module, names))
    if not states:
        raise RuntimeError("PEFT did not mark any default adapter layers merged")
    for module, _names in states:
        module.merged_adapters.clear()
    return states


def _restore_merged_adapter_markers(states):
    """Restore PEFT's merge bookkeeping before unmerging the default adapter."""
    for module, names in states:
        module.merged_adapters.extend(names)


def _correct(prediction, gold):
    return (
        prediction is not None
        and gold is not None
        and abs(prediction - gold) < 1e-4
    )


@torch.no_grad()
def _exec_success(model, tokenizer, rows, golds, device):
    """Run the E2 greedy 320-token executable-solution evaluation."""
    model.eval()
    successes = []
    for row, gold in zip(rows, golds):
        messages = [{"role": "user", "content": row["prompt"]}]
        prompt = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        input_ids = tokenizer(
            prompt, add_special_tokens=False, return_tensors="pt"
        ).input_ids.to(device)
        generated = model.generate(
            input_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        text = tokenizer.decode(
            generated[0, input_ids.shape[1] :], skip_special_tokens=True
        )
        start = text.find("def solution")
        prediction = verifier.run_solution(text[start:]) if start >= 0 else None
        successes.append(_correct(prediction, gold))
    return np.asarray(successes, dtype=bool)


def _measure_checkpoint(
    model,
    tokenizer,
    rows,
    golds,
    device,
    dictionary,
):
    """Measure M2/behavior through a fresh rank-8 probe at current weights."""
    cpu_rng_state = torch.get_rng_state()
    cuda_rng_states = torch.cuda.get_rng_state_all()
    merged = False
    probe_added = False
    merged_marker_states = None
    try:
        model.merge_adapter()
        merged = True

        # Recreate the same fresh rank-8 LoRA initialization used by
        # features.py; checkpoint instrumentation restores RNG state below.
        torch.manual_seed(SEED)
        probe_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=list(LORA_TARGETS),
        )
        model.add_adapter("probe", probe_config)
        probe_added = True
        model.set_adapter("probe")
        for name, parameter in model.named_parameters():
            parameter.requires_grad = "lora_B.probe" in name

        lora_b, projections = _prepare_probe_projections(
            model, dictionary.components_.shape[1], device
        )
        merged_marker_states = _suspend_merged_adapter_markers(model)
        features = _extract_probe_features(
            model, tokenizer, rows, device, lora_b, projections
        )
        if features.shape[1] != dictionary.components_.shape[1]:
            raise RuntimeError(
                f"Feature dimension {features.shape[1]} does not match "
                f"dictionary dimension {dictionary.components_.shape[1]}"
            )
        absolute_codes = np.abs(dictionary.transform(features))
        successes = _exec_success(model, tokenizer, rows, golds, device)
        return absolute_codes, successes, features.mean(axis=0)
    finally:
        model.zero_grad(set_to_none=True)
        try:
            if merged_marker_states is not None:
                _restore_merged_adapter_markers(merged_marker_states)
            if probe_added:
                model.delete_adapter("probe")
        finally:
            try:
                model.set_adapter("default")
                if merged:
                    model.unmerge_adapter()
            finally:
                for name, parameter in model.named_parameters():
                    parameter.requires_grad = ".default." in name
                model.train()
                torch.set_rng_state(cpu_rng_state)
                torch.cuda.set_rng_state_all(cuda_rng_states)


def _top_k_mass_fraction(absolute_code, k=5):
    total = float(np.sum(absolute_code))
    if total <= 0.0:
        return 0.0
    top = np.partition(np.asarray(absolute_code), -k)[-k:]
    return float(np.sum(top) / total)


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


def _make_figure(mean_codes, mean_success, delta_fractions):
    """Render the requested M2/M3 three-panel figure."""
    checkpoints = np.asarray(CHECKPOINTS)
    top_atoms = np.argsort(-mean_codes[0], kind="stable")[:30]
    heatmap = mean_codes[:, top_atoms].T
    interval_labels = [
        f"{start}\N{RIGHTWARDS ARROW}{end}"
        for start, end in zip(CHECKPOINTS[:-1], CHECKPOINTS[1:])
    ]

    fig, (ax_heat, ax_success, ax_delta) = plt.subplots(
        1,
        3,
        figsize=(15.5, 4.8),
        gridspec_kw={"width_ratios": (1.45, 1.0, 1.15)},
        facecolor="white",
    )

    image = ax_heat.imshow(heatmap, aspect="auto", cmap="viridis")
    ax_heat.set_xticks(np.arange(len(checkpoints)), labels=checkpoints)
    ax_heat.set_yticks(np.arange(len(top_atoms)), labels=top_atoms)
    ax_heat.set_xlabel("Optimizer step")
    ax_heat.set_ylabel("Atom (ranked by step-0 mass)")
    ax_heat.set_title("M2 residual demand")
    colorbar = fig.colorbar(image, ax=ax_heat, fraction=0.046, pad=0.04)
    colorbar.set_label("Mean |coefficient|", color=TICK_GRAY)
    colorbar.ax.tick_params(colors=TICK_GRAY)
    colorbar.outline.set_visible(False)

    ax_success.plot(
        checkpoints,
        mean_success,
        color=SUCCESS_COLOR,
        marker="o",
        linewidth=2.2,
    )
    ax_success.set_xticks(checkpoints)
    ax_success.set_ylim(-0.03, 1.03)
    ax_success.set_xlabel("Optimizer step")
    ax_success.set_ylabel("Mean exec success")
    ax_success.set_title("Held-out behavior")

    positions = np.arange(len(delta_fractions))
    ax_delta.bar(positions, delta_fractions, color=DELTA_COLOR, width=0.72)
    ax_delta.set_xticks(positions, labels=interval_labels, rotation=35, ha="right")
    ax_delta.set_ylim(0.0, 1.0)
    ax_delta.set_xlabel("Checkpoint interval")
    ax_delta.set_ylabel("Top-5 atom mass fraction")
    ax_delta.set_title("M3 delta concentration")

    for axis in (ax_heat, ax_success, ax_delta):
        _style_axis(axis)
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(w_pad=2.0)
    fig.savefig(FIG, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _half_life(mean_codes, atom):
    initial = mean_codes[0, atom]
    below_half = np.flatnonzero(mean_codes[:, atom] < 0.5 * initial)
    return str(CHECKPOINTS[int(below_half[0])]) if len(below_half) else "not reached"


def _write_summary(mean_codes, mean_success, delta_fractions):
    """Write the requested absorption and delta-concentration tables."""
    # Initial demand that is absent at the final checkpoint is net absorbed mass.
    absorbed = np.maximum(mean_codes[0] - mean_codes[-1], 0.0)
    top_absorbed = np.argsort(-absorbed, kind="stable")[:10]

    lines = [
        "# M2/M3 atom dynamics",
        "",
        (
            "Absorbed mass is defined as `max(mean |code| at step 0 - "
            "mean |code| at step 66, 0)` across the 30 held-out rows. "
            "Half-life is the first measured checkpoint strictly below 50% "
            "of the atom's step-0 coefficient."
        ),
        "",
        "## M2: atoms with the most absorbed demand",
        "",
        "| Rank | Atom | Step 0 | Step 66 | Absorbed mass | Half-life |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, atom in enumerate(top_absorbed, start=1):
        lines.append(
            f"| {rank} | {atom} | {mean_codes[0, atom]:.6f} | "
            f"{mean_codes[-1, atom]:.6f} | {absorbed[atom]:.6f} | "
            f"{_half_life(mean_codes, atom)} |"
        )

    lines.extend(
        [
            "",
            "## M3: carrier concentration",
            "",
            (
                "Each interval codes the L2-normalized difference between "
                "consecutive checkpoint mean rank-8 probe features."
            ),
            "",
            "| Interval | Top-5 atom mass fraction |",
            "|:---:|---:|",
        ]
    )
    for start, end, fraction in zip(
        CHECKPOINTS[:-1], CHECKPOINTS[1:], delta_fractions
    ):
        lines.append(f"| {start}→{end} | {fraction:.3f} |")

    first = float(delta_fractions[0])
    later = float(np.mean(delta_fractions[1:]))
    if first > later + 1e-12:
        relation = "more concentrated than"
    elif first < later - 1e-12:
        relation = "more diffuse than"
    else:
        relation = "equally concentrated as"
    lines.extend(
        [
            "",
            (
                f"The step 0→2 delta has top-5 mass fraction {first:.3f}; "
                f"it is {relation} the later-interval mean ({later:.3f})."
            ),
            "",
            "## Behavior alignment",
            "",
            "| Checkpoint | Mean exec success |",
            "|---:|---:|",
        ]
    )
    for checkpoint, success in zip(CHECKPOINTS, mean_success):
        lines.append(f"| {checkpoint} | {success:.3f} |")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")


def _save_records(
    dictionary,
    atom_codes,
    mean_codes,
    mean_probe_features,
    delta_codes,
    delta_fractions,
    successes,
    mean_success,
):
    config = {
        "model": MODEL,
        "seed": SEED,
        "train_domain": "gsm8k-code",
        "train_rows": 90,
        "test_rows": N_TEST,
        "epochs": EPOCHS,
        "optimizer": "AdamW",
        "learning_rate": LR,
        "gradient_accumulation": ACCUM,
        "optimizer_steps": CHECKPOINTS[-1],
        "checkpoints": list(CHECKPOINTS),
        "lora": {"r": 16, "alpha": 32, "targets": list(LORA_TARGETS)},
        "measurement_probe": {
            "adapter_name": "probe",
            "r": 8,
            "alpha": 16,
            "dropout": 0.0,
            "targets": list(LORA_TARGETS),
        },
        "m3": "normalized consecutive checkpoint mean-probe-feature delta",
        "dictionary": {
            "n_components": N_ATOMS,
            "alpha": SPARSITY_ALPHA,
            "batch_size": 256,
            "max_iter": 150,
            "fit_algorithm": "lars",
            "transform_algorithm": "lasso_lars",
            "transform_alpha": SPARSITY_ALPHA,
            "random_state": SEED,
        },
        "generation": {"strategy": "greedy", "max_new_tokens": MAX_NEW_TOKENS},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT / "records.npz",
        checkpoints=np.asarray(CHECKPOINTS, dtype=np.int64),
        atom_abs_codes=np.asarray(atom_codes, dtype=np.float64),
        atom_mean_abs_codes=np.asarray(mean_codes, dtype=np.float64),
        checkpoint_mean_probe_features=np.asarray(
            mean_probe_features, dtype=np.float64
        ),
        delta_start_steps=np.asarray(CHECKPOINTS[:-1], dtype=np.int64),
        delta_end_steps=np.asarray(CHECKPOINTS[1:], dtype=np.int64),
        delta_abs_codes=np.asarray(delta_codes, dtype=np.float64),
        delta_top5_mass_fraction=np.asarray(delta_fractions, dtype=np.float64),
        exec_success=np.asarray(successes, dtype=bool),
        mean_exec_success=np.asarray(mean_success, dtype=np.float64),
        dictionary_components=np.asarray(dictionary.components_, dtype=np.float64),
        config_json=np.asarray(json.dumps(config, sort_keys=True)),
    )


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("M2/M3 instrumented training requires a CUDA GPU")
    device = "cuda"

    dictionary = _fit_dictionary()
    model, tokenizer = _prepare_model(device)

    train_rows, test_rows = split("gsm8k-code")
    if len(train_rows) != 90 or len(test_rows) != N_TEST:
        raise ValueError("E2 expects 90 train and 30 test gsm8k-code rows")
    golds = [verifier.run_solution(row["response"]) for row in test_rows]

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=LR,
    )
    rng = np.random.default_rng(SEED)
    atom_codes = []
    successes = []
    mean_probe_features = []
    delta_codes = []
    delta_fractions = []
    previous_mean_feature = None

    def record(checkpoint):
        nonlocal previous_mean_feature
        absolute_codes, checkpoint_success, mean_feature = _measure_checkpoint(
            model,
            tokenizer,
            test_rows,
            golds,
            device,
            dictionary,
        )
        if checkpoint == CHECKPOINTS[0]:
            mean_active_atoms = np.count_nonzero(absolute_codes, axis=1).mean()
            assert mean_active_atoms > 1.0, (
                "Rank-8 probe basis mismatch: checkpoint-0 rows have only "
                f"{mean_active_atoms:.3f} active atoms on average"
            )

        if previous_mean_feature is not None:
            delta_feature = mean_feature - previous_mean_feature
            delta_feature = delta_feature / (
                np.linalg.norm(delta_feature) + 1e-8
            )
            absolute_delta_code = np.abs(
                dictionary.transform(delta_feature[None, :])[0]
            )
            delta_codes.append(absolute_delta_code)
            delta_fractions.append(_top_k_mass_fraction(absolute_delta_code, 5))

        atom_codes.append(absolute_codes)
        successes.append(checkpoint_success)
        mean_probe_features.append(mean_feature)
        previous_mean_feature = mean_feature

        delta_text = (
            "n/a"
            if not delta_fractions
            else f"{delta_fractions[-1]:.3f}"
        )
        print(
            f"checkpoint step={checkpoint:2d} "
            f"exec={checkpoint_success.mean():.3f} "
            f"mean_atom_mass={absolute_codes.mean(axis=0).sum():.4f} "
            f"delta_top5={delta_text}",
            flush=True,
        )

    record(0)
    step = 0
    model.train()
    for _epoch in range(EPOCHS):
        order = rng.permutation(len(train_rows))
        full_epoch = order[: (len(order) // ACCUM) * ACCUM]
        for microstep, row_index in enumerate(full_epoch, start=1):
            input_ids, labels = encode(
                tokenizer, train_rows[int(row_index)], device
            )
            loss = model(input_ids=input_ids, labels=labels).loss / ACCUM
            loss.backward()
            if microstep % ACCUM == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                if step in CHECKPOINTS:
                    record(step)

    if step != CHECKPOINTS[-1]:
        raise RuntimeError(f"Expected 66 optimizer steps, got {step}")
    optimizer.zero_grad(set_to_none=True)

    atom_codes_array = np.stack(atom_codes)
    mean_codes = atom_codes_array.mean(axis=1)
    mean_probe_features_array = np.stack(mean_probe_features)
    delta_codes_array = np.stack(delta_codes)
    delta_fractions_array = np.asarray(delta_fractions, dtype=np.float64)
    successes_array = np.stack(successes)
    mean_success = successes_array.mean(axis=1)

    if atom_codes_array.shape != (len(CHECKPOINTS), N_TEST, N_ATOMS):
        raise RuntimeError(f"Unexpected atom-code shape {atom_codes_array.shape}")
    if delta_codes_array.shape != (len(CHECKPOINTS) - 1, N_ATOMS):
        raise RuntimeError(f"Unexpected delta-code shape {delta_codes_array.shape}")
    if mean_probe_features_array.shape != (
        len(CHECKPOINTS),
        dictionary.components_.shape[1],
    ):
        raise RuntimeError(
            "Unexpected mean probe-feature shape "
            f"{mean_probe_features_array.shape}"
        )

    _save_records(
        dictionary,
        atom_codes_array,
        mean_codes,
        mean_probe_features_array,
        delta_codes_array,
        delta_fractions_array,
        successes_array,
        mean_success,
    )
    _write_summary(mean_codes, mean_success, delta_fractions_array)
    _make_figure(mean_codes, mean_success, delta_fractions_array)


if __name__ == "__main__":
    main()
