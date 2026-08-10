"""Issue one-sided Clopper--Pearson certificates from E3 sweep outputs.

This script is deliberately post-hoc and CPU-only: it reads the behavioral
rates already stored in ``results/sweep/e3_tier1_*/metrics.json`` and writes
machine-readable and Markdown certificate summaries.  It does not import or
run any training code.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from scipy.stats import beta


ROOT = Path(__file__).resolve().parent.parent
INPUT_GLOB = "results/sweep/e3_tier1_*/metrics.json"
OUTPUT_DIR = ROOT / "results" / "certificates"
JSON_OUTPUT = OUTPUT_DIR / "certificates.json"
MARKDOWN_OUTPUT = OUTPUT_DIR / "summary.md"

N = 30
DELTA = 0.1
CONDITIONS = ("B_random", "C_emb", "D_grad")

RUN_NAME_RE = re.compile(r"^e3_tier1_(?P<family>.+)_seed(?P<seed>\d+)$")

# Positive behavioral-retention metrics used only if gsm8k-cot accuracy is
# absent.  Failure-oriented rates (refusal, format bleed, or writing code on a
# non-code task) are not retention statistics and therefore are not averaged.
RETENTION_METRICS = (
    ("gsm8k_code", "exec_acc"),
    ("gsm8k_cot", "any_acc"),
    ("sql", "single_statement_valid_rate"),
    ("pandas", "compile_rate"),
)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _rate(value: Any, *, label: str, source: Path) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} in {source} is not numeric: {value!r}")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{label} in {source} is outside [0, 1]: {result}")
    return result


def _nested_rate(
    evaluation: dict[str, Any], domain: str, metric: str, *, source: Path
) -> float | None:
    domain_metrics = evaluation.get(domain)
    if not isinstance(domain_metrics, dict) or metric not in domain_metrics:
        return None
    return _rate(
        domain_metrics[metric], label=f"eval.{domain}.{metric}", source=source
    )


def clopper_pearson_lower(k: int, n: int = N, delta: float = DELTA) -> float:
    """Return a one-sided (1-delta) Clopper--Pearson lower bound."""
    if not 0 <= k <= n:
        raise ValueError(f"Expected 0 <= k <= n, got k={k}, n={n}")
    if not 0.0 < delta < 1.0:
        raise ValueError(f"Expected 0 < delta < 1, got {delta}")
    if k == 0:
        return 0.0
    # This expression is also well-defined at k == n (Beta(n, 1)).
    return float(beta.ppf(delta, k, n - k + 1))


def clopper_pearson_upper(k: int, n: int = N, delta: float = DELTA) -> float:
    """Return a one-sided (1-delta) Clopper--Pearson upper bound."""
    if not 0 <= k <= n:
        raise ValueError(f"Expected 0 <= k <= n, got k={k}, n={n}")
    if not 0.0 < delta < 1.0:
        raise ValueError(f"Expected 0 < delta < 1, got {delta}")
    if k == n:
        return 1.0
    # This expression is also well-defined at k == 0 (Beta(1, n)).
    return float(beta.ppf(1.0 - delta, k + 1, n - k))


def _binomial_count(rate: float) -> int:
    """Recover the integer success count from a rate measured on N rows."""
    return min(N, max(0, int(round(rate * N))))


def _coverage_rate(
    evaluation: dict[str, Any], task: str, *, source: Path
) -> tuple[str, float]:
    if task.startswith("gsm8k"):
        domain, metric = "gsm8k_code", "exec_acc"
    elif task == "sql":
        domain, metric = "sql", "single_statement_valid_rate"
    else:
        raise ValueError(f"Unsupported task boundary {task!r} in {source}")

    value = _nested_rate(evaluation, domain, metric, source=source)
    if value is None:
        raise ValueError(f"Missing eval.{domain}.{metric} in {source}")
    return f"{domain}.{metric}", value


def _leakage_rate(
    evaluation: dict[str, Any], task: str, *, source: Path
) -> tuple[str, float, dict[str, float]]:
    if task == "sql":
        domain, metric = "gsm8k_code", "exec_acc"
        value = _nested_rate(evaluation, domain, metric, source=source)
        if value is None:
            raise ValueError(f"Missing eval.{domain}.{metric} in {source}")
        label = f"{domain}.{metric}"
        return label, value, {label: value}

    if not task.startswith("gsm8k"):
        raise ValueError(f"Unsupported task boundary {task!r} in {source}")

    preferred = _nested_rate(evaluation, "gsm8k_cot", "any_acc", source=source)
    if preferred is not None:
        label = "gsm8k_cot.any_acc"
        return label, preferred, {label: preferred}

    components: dict[str, float] = {}
    for domain, metric in RETENTION_METRICS:
        # gsm8k-code is the in-task statistic for a gsm8k-code boundary.
        if domain == "gsm8k_code":
            continue
        value = _nested_rate(evaluation, domain, metric, source=source)
        if value is not None:
            components[f"{domain}.{metric}"] = value
    if not components:
        raise ValueError(
            f"No off-domain behavioral retention rates are available in {source}"
        )
    mean_rate = sum(components.values()) / len(components)
    label = "mean_off_domain_behavioral_rate"
    return label, mean_rate, components


def _certificate(rate: float, *, side: str) -> dict[str, Any]:
    k = _binomial_count(rate)
    if side == "lower":
        bound_name = "lower_bound"
        bound = clopper_pearson_lower(k)
    elif side == "upper":
        bound_name = "upper_bound"
        bound = clopper_pearson_upper(k)
    else:
        raise ValueError(f"Unknown certificate side: {side}")
    return {"rate": rate, "k": k, "n": N, bound_name: bound}


def _family_and_seed(path: Path) -> tuple[str, int]:
    match = RUN_NAME_RE.fullmatch(path.parent.name)
    if match is None:
        raise ValueError(f"Unexpected sweep run directory name: {path.parent.name}")
    return match.group("family"), int(match.group("seed"))


def _model_size(model: str) -> str:
    prefix = "Qwen/Qwen3.5-"
    return model[len(prefix) :] if model.startswith(prefix) else model


def build_certificates(paths: list[Path]) -> dict[str, Any]:
    """Build the complete JSON-serializable certificate payload."""
    if not paths:
        raise FileNotFoundError(f"No inputs matched {ROOT / INPUT_GLOB}")

    families: dict[str, dict[str, Any]] = {}
    for path in sorted(paths):
        payload = _read_json(path)
        family, path_seed = _family_and_seed(path)
        config = payload.get("config")
        conditions = payload.get("conditions")
        if not isinstance(config, dict) or not isinstance(conditions, dict):
            raise ValueError(f"Missing config or conditions object in {path}")

        seed = int(config.get("seed", path_seed))
        if seed != path_seed:
            raise ValueError(
                f"Seed mismatch in {path}: directory={path_seed}, config={seed}"
            )
        rows = config.get("data", {}).get("test_rows_per_domain")
        if rows != N:
            raise ValueError(
                f"Expected {N} held-out rows in {path}, found {rows!r}"
            )

        task_config = config.get("task_boundary")
        selection_config = config.get("selection")
        if not isinstance(task_config, dict) or not isinstance(selection_config, dict):
            raise ValueError(f"Missing task_boundary or selection config in {path}")
        task = str(task_config.get("domain"))
        task_filter = task_config.get("filter")
        model = str(config.get("model"))
        budget = int(selection_config.get("response_token_budget"))

        metadata = {
            "model": model,
            "size": _model_size(model),
            "budget": budget,
            "task": task,
            "task_filter": task_filter,
        }
        family_payload = families.setdefault(
            family,
            {
                **metadata,
                "conditions": {name: {} for name in CONDITIONS},
                "missing_conditions_by_seed": {},
            },
        )
        for key, expected in metadata.items():
            if family_payload[key] != expected:
                raise ValueError(
                    f"Inconsistent {key} within family {family}: "
                    f"{family_payload[key]!r} != {expected!r}"
                )

        for condition in CONDITIONS:
            condition_payload = conditions.get(condition)
            if not isinstance(condition_payload, dict):
                family_payload["missing_conditions_by_seed"].setdefault(
                    str(seed), []
                ).append(condition)
                continue
            evaluation = condition_payload.get("eval")
            if not isinstance(evaluation, dict):
                raise ValueError(f"Missing eval for condition {condition} in {path}")

            coverage_name, coverage_rate = _coverage_rate(
                evaluation, task, source=path
            )
            leakage_name, leakage_rate, leakage_components = _leakage_rate(
                evaluation, task, source=path
            )
            seed_key = str(seed)
            condition_seeds = family_payload["conditions"][condition]
            if seed_key in condition_seeds:
                raise ValueError(
                    f"Duplicate seed {seed} for family {family}, condition {condition}"
                )
            condition_seeds[seed_key] = {
                "source": str(path.relative_to(ROOT)),
                "coverage": {
                    "statistic": coverage_name,
                    **_certificate(coverage_rate, side="lower"),
                },
                "leakage": {
                    "statistic": leakage_name,
                    "components": leakage_components,
                    **_certificate(leakage_rate, side="upper"),
                },
            }

    for family_payload in families.values():
        for condition in CONDITIONS:
            seeds = family_payload["conditions"][condition]
            ordered_seeds = dict(sorted(seeds.items(), key=lambda item: int(item[0])))
            if not ordered_seeds:
                raise ValueError(f"No seeds found for condition {condition}")
            coverage_worst = min(
                row["coverage"]["lower_bound"] for row in ordered_seeds.values()
            )
            leakage_worst = max(
                row["leakage"]["upper_bound"] for row in ordered_seeds.values()
            )
            family_payload["conditions"][condition] = {
                "seeds": ordered_seeds,
                "worst_case": {
                    "coverage_lower_bound": coverage_worst,
                    "leakage_upper_bound": leakage_worst,
                },
            }

    return {
        "schema_version": 1,
        "method": "one-sided Clopper-Pearson",
        "confidence": 1.0 - DELTA,
        "delta": DELTA,
        "n": N,
        "input_glob": INPUT_GLOB,
        "n_runs": len(paths),
        "n_certificates": sum(
            len(condition_payload["seeds"])
            for family_payload in families.values()
            for condition_payload in family_payload["conditions"].values()
        ),
        "families": dict(sorted(families.items())),
    }


def _per_seed_cell(condition_payload: dict[str, Any], certificate: str) -> str:
    bound_key = "lower_bound" if certificate == "coverage" else "upper_bound"
    worst_key = (
        "coverage_lower_bound"
        if certificate == "coverage"
        else "leakage_upper_bound"
    )
    seed_values = [
        f"seed{seed}: {row[certificate][bound_key]:.4f}"
        for seed, row in condition_payload["seeds"].items()
    ]
    worst = condition_payload["worst_case"][worst_key]
    return "; ".join([*seed_values, f"**worst: {worst:.4f}**"])


def _comparison(family: str, family_payload: dict[str, Any]) -> str:
    pairs = {
        condition: family_payload["conditions"][condition]["worst_case"]
        for condition in CONDITIONS
    }
    values = "; ".join(
        f"`{condition}` {bounds['coverage_lower_bound']:.4f} / "
        f"{bounds['leakage_upper_bound']:.4f}"
        for condition, bounds in pairs.items()
    )
    best_coverage = max(
        bounds["coverage_lower_bound"] for bounds in pairs.values()
    )
    best_leakage = min(bounds["leakage_upper_bound"] for bounds in pairs.values())
    coverage_winners = [
        condition
        for condition, bounds in pairs.items()
        if math.isclose(bounds["coverage_lower_bound"], best_coverage)
    ]
    leakage_winners = [
        condition
        for condition, bounds in pairs.items()
        if math.isclose(bounds["leakage_upper_bound"], best_leakage)
    ]

    def winners_text(winners: list[str]) -> str:
        return " = ".join(f"`{winner}`" for winner in winners)

    return (
        f"**{family} comparison.** Worst-case coverage LB / leakage UB: {values}. "
        f"Highest certified coverage: {winners_text(coverage_winners)}; "
        f"lowest certified leakage: {winners_text(leakage_winners)}."
    )


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Conformal certificate summary",
        "",
        (
            f"One-sided Clopper–Pearson certificates at "
            f"{result['confidence']:.0%} confidence (delta={result['delta']}, "
            f"n={result['n']} held-out rows per statistic). Worst-case coverage "
            "is the minimum lower bound across seeds; worst-case leakage is the "
            "maximum upper bound across seeds."
        ),
        "",
    ]
    for family, family_payload in result["families"].items():
        task_label = family_payload["task"]
        if family_payload["task_filter"] is not None:
            task_label += f" (filter={family_payload['task_filter']})"
        lines.extend(
            [
                f"## {family}",
                "",
                (
                    f"Model: `{family_payload['model']}` · "
                    f"budget: {family_payload['budget']:,} response tokens · "
                    f"task: `{task_label}`"
                ),
                "",
            ]
        )
        missing = family_payload["missing_conditions_by_seed"]
        if missing:
            missing_text = "; ".join(
                f"seed{seed}: {', '.join(conditions)}"
                for seed, conditions in missing.items()
            )
            lines.extend(
                [
                    f"Source outputs missing (not certified): {missing_text}.",
                    "",
                ]
            )
        lines.extend(
            [
                "| condition | coverage LB (per seed) | leakage UB (per seed) |",
                "|---|---|---|",
            ]
        )
        for condition in CONDITIONS:
            condition_payload = family_payload["conditions"][condition]
            lines.append(
                f"| {condition} | "
                f"{_per_seed_cell(condition_payload, 'coverage')} | "
                f"{_per_seed_cell(condition_payload, 'leakage')} |"
            )
        lines.append("")
        if family in {"s2b_sqljoin", "s08b_b10k"}:
            lines.extend([f"> {_comparison(family, family_payload)}", ""])
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    paths = list(ROOT.glob(INPUT_GLOB))
    result = build_certificates(paths)
    summary = render_markdown(result)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with JSON_OUTPUT.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=False, allow_nan=False)
        handle.write("\n")
    with MARKDOWN_OUTPUT.open("w", encoding="utf-8") as handle:
        handle.write(summary)

    sqljoin = result["families"].get("s2b_sqljoin")
    if sqljoin is None:
        raise ValueError("The required s2b_sqljoin family was not found")
    print(_comparison("s2b_sqljoin", sqljoin))


if __name__ == "__main__":
    main()
