"""Shared credentials, inventory, and accounting for BFCL teacher collection."""

from __future__ import annotations

import json
import os
from pathlib import Path

from . import ledger


# Shared by registration, the handler, and adapter routing. New compatible
# providers need only an entry here (plus their model registrations).
PROVIDERS = {
    "ollama": ("OLLAMA_BASE_URL", "OLLAMA_API_KEY", "https://ollama.com/v1"),
    "openrouter": ("OPENROUTER_BASE_URL", "OPENROUTER_API_KEY", "https://openrouter.ai/api/v1"),
    "openai": ("OPENAI_BASE_URL", "OPENAI_API_KEY", "https://api.openai.com/v1"),
}


def provider_credentials(prefix: str) -> tuple[str, str]:
    base_env, key_env, default_url = PROVIDERS[prefix]
    base = os.environ.get(base_env, default_url).rstrip("/")
    if prefix == "ollama" and not base.endswith("/v1"):
        base += "/v1"
    key = os.environ.get(key_env, "").strip()
    # Only the historical Ollama provider supports credentials from files.
    if not key and prefix == "ollama":
        for name in (".ollama_api_key2", ".ollama_api_key"):
            path = Path.home() / name
            if path.is_file():
                key = path.read_text().strip()
                if key:
                    break
    if not key:
        raise RuntimeError(f"{prefix.capitalize()} credentials not found: set {key_env}")
    return base, key


def ollama_credentials() -> tuple[str, str]:
    return provider_credentials("ollama")


def teacher_task_ids(task_ids, *, record_inventory=True):
    """Exclude memory demand before expanding any prerequisite write chains."""
    task_ids = list(dict.fromkeys(task_ids))
    if os.environ.get("BFAS_BFCL_SKIP_MEMORY_PREREQ") != "1":
        return task_ids
    skipped = {task_id for task_id in task_ids if task_id.startswith("memory_")}
    if skipped and record_inventory:
        teacher = os.environ.get("BFAS_BFCL_TEACHER", "deepseek-v4-pro-FC")
        path = ledger.ledger_path("bfcl").with_name("bfcl_inventory.jsonl")
        with ledger._purchase_lock(path):
            existing = {
                (row["teacher"], row["task_id"])
                for row in (json.loads(line) for line in path.read_text().splitlines())
            } if path.exists() else set()
            for task_id in sorted(skipped):
                if (teacher, task_id) not in existing:
                    ledger.append_record(path, {
                        "task_id": task_id, "teacher": teacher,
                        "status": "unavailable inventory",
                        "reason": "BFAS_BFCL_SKIP_MEMORY_PREREQ=1",
                        "timestamp": ledger._timestamp(),
                    })
    return [task_id for task_id in task_ids if task_id not in skipped]


def read_results(result_dir: Path) -> list[dict]:
    """Recover usage even when a process failed before BFCL wrote its result."""
    from .adapters.bfcl import _completion_tokens

    results = {}
    for path in sorted(result_dir.rglob("*_result.json")):
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                if row["id"] in results:
                    raise ValueError(f"duplicate BFCL result: {row['id']}")
                results[row["id"]] = row
    usage_path = result_dir / "bfas_usage.jsonl"
    usage = {}
    errors = {}
    if usage_path.exists():
        for line in usage_path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                key = (row["id"], row.get("bfas_attempt_id"))
                counts = usage.setdefault(key, [0, 0])
                counts[0] += row["input_token_count"]
                counts[1] += _completion_tokens([row])
                if row.get("error"):
                    errors[key] = row["error"]
    recovered = []
    for (task_id, attempt_id), (input_tokens, output_tokens) in usage.items():
        row = results.get(task_id)
        if row is not None and row.get("bfas_attempt_id") == attempt_id:
            results.pop(task_id)
        else:
            row = {
                "id": task_id, "result": "Error during inference: no completed BFCL result",
                "error": "incomplete_generation", "bfas_attempt_id": attempt_id,
            }
        # Preserve BFCL's nested turn/step structure when it reconciles.
        if _completion_tokens([row]) != output_tokens:
            row["output_token_count"] = output_tokens
        row["input_token_count"] = input_tokens
        if error := errors.get((task_id, attempt_id)):
            row["error"] = error
            # Only request failures before any usage may be retried for free.
            if input_tokens == output_tokens == 0:
                row["failure_kind"] = "provider_error"
        recovered.append(row)
    return [*results.values(), *recovered]


def provider_failure_kind(results):
    """Classify an episode, including usage from any prerequisite requests."""
    from .adapters.bfcl import _completion_tokens

    if (
        any(row.get("failure_kind") == "provider_error" for row in results)
        and _completion_tokens(results) == 0
        and _completion_tokens([
            {"output_token_count": row.get("input_token_count")} for row in results
        ]) == 0
    ):
        return "provider_error"
    return None


def result_source_id(result_dir, model, result):
    source = f"{Path(result_dir).resolve()}:{model}:{result['id']}"
    if result.get("bfas_attempt_id"):
        source += ":" + result["bfas_attempt_id"]
    return source


def record_attempt(adapter, result_dir, score_dir, task_ids, model, attempt_index, temperature):
    """Import a batch once into the gateway schema, including failed/prereq rows."""
    from .adapters.bfcl import _completion_tokens, extract_verdicts

    result_dir, score_dir = Path(result_dir), Path(score_dir)
    results = read_results(result_dir)
    by_id = {row["id"]: row for row in results}
    categories = adapter.task_categories()
    verdicts = {}
    # The checker excludes prerequisite writes. The adapter categorizes those
    # with their memory backend for generation, so remove them from scoring
    # inventory while still charging them below.
    scored = {tid for tid in by_id if tid in categories and "prereq" not in tid}
    # Include old demand left in a resumed directory when reconciling summaries.
    for category in {categories[tid] for tid in scored}:
        expected = {tid: category for tid in scored if categories[tid] == category}
        try:
            verdicts.update(extract_verdicts(score_dir, expected))
        except Exception as exc:
            print(f"[bfcldemos] unverified {category}: {exc}", flush=True)
    demand = set(task_ids)
    with ledger._purchase_lock("bfcl"):
        recorded = {row.get("source_id") for row in ledger.read_records("bfcl")}
        for result in results:
            task_id = result["id"]
            source_id = result_source_id(result_dir, model, result)
            if source_id in recorded:
                continue
            tokens = _completion_tokens([result])
            if tokens is None:
                raise ValueError(f"{task_id}: missing BFCL provider usage; keep raw artifacts for reconciliation")
            demo = None
            if task_id in demand and verdicts.get(task_id) and not result.get("error"):
                try:
                    demo = adapter._teacher_demo_from_result(result, attempt_index)
                except Exception as exc:
                    print(f"[bfcldemos] cannot render {task_id}: {exc}", flush=True)
            ledger.append_episode(
                "bfcl", task_id=task_id, teacher=model, attempt_index=attempt_index,
                temperature=temperature, verified=demo is not None,
                tokens_spent=tokens, demo=demo, source_id=source_id,
                failure_kind=provider_failure_kind([result]),
            )
    return verdicts
