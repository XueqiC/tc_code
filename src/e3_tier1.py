"""E3 tier-1: boundary-guided versus standard trajectory distillation.

The experiment derives gradient and embedding boundaries from the stable
GSM8K-code fit/calibration split, selects a fixed-token training set under five
conditions, trains a fresh LoRA student for each condition, and evaluates
in-boundary capability and out-of-boundary behavior.
"""

import argparse
import gc
import json
import math
import os
import random
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
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
    "base", "A_all", "B_random", "C_emb", "D_grad", "F_refusal"
)
COLORS = {
    "base": "#6b6a63",
    "A_all": "#eb6834",
    "B_random": "#eda100",
    "C_emb": "#1baf7a",
    "D_grad": "#2a78d6",
    "F_refusal": "#e87ba4",
}
SHORT_LABELS = {
    "base": "base",
    "A_all": "A",
    "B_random": "B",
    "C_emb": "C",
    "D_grad": "D",
    "F_refusal": "F",
}
MUTED = "#6b6a63"


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
        ("training", "learning_rate"): float(os.environ.get("E3_LR", 2e-4)),
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


def score_pool(config, task_rows, pool):
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

    if len(grad_scores) != len(pool) or len(emb_scores) != len(pool):
        raise RuntimeError("feature scores do not align with the candidate pool")
    for row, grad_score, emb_score in zip(pool, grad_scores, emb_scores):
        row["_grad_score"] = float(grad_score)
        row["_emb_score"] = float(emb_score)

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
    low_grad_order = grad_order[::-1]

    selections = {
        "A_all": take_prefix(pool, natural, sum(r["_token_count"] for r in pool)),
        "B_random": take_prefix(pool, random_order, budget),
        "C_emb": take_prefix(pool, emb_order, budget),
        "D_grad": take_prefix(pool, grad_order, budget),
    }

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


def train_condition(model, tokenizer, rows, config):
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

        boundary_metrics = score_pool(config, task_rows, pool)
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
        optimizer_steps = train_condition(model, tokenizer, selected, config)
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
