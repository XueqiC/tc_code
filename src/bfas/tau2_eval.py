"""Single-trial tau2 test evaluation against an already-running endpoint."""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import urllib.error
import urllib.request
from itertools import zip_longest
from pathlib import Path
from typing import Any

from .adapters.tau2 import LUNA_MODEL, Tau2Adapter, encode_task_id, extract_verdict
from .tau2_budget import FIELDS, price_rates, service_tier, write_json

DOMAINS = ("retail", "airline", "telecom")


def probe_student_tool_choice(base_url: str, model: str) -> None:
    """Check tools + auto acceptance once, before buying any simulator calls.

    One output token suffices for server-side validation; this is not a test of
    the model's ability to produce a complete tool call. Never execute a tool.
    """
    payload = {
        "model": model.removeprefix("openai/"),
        "messages": [{"role": "user", "content": "Call bfas_tool_probe."}],
        "tools": [{"type": "function", "function": {
            "name": "bfas_tool_probe", "description": "Check tool support.",
            "parameters": {"type": "object", "properties": {}},
        }}],
        "tool_choice": "auto", "max_tokens": 1, "temperature": 0, "stream": False,
    }
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + os.environ.get("BFAS_STUDENT_API_KEY", "EMPTY")},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            body = json.load(response)
        if not isinstance(body, dict) or body.get("error") or not body.get("choices"):
            raise ValueError("endpoint did not return a Chat Completions response")
    except (OSError, ValueError) as exc:
        detail = str(exc)
        if isinstance(exc, urllib.error.HTTPError):
            detail += ": " + exc.read(4096).decode("utf-8", errors="replace")
            exc.close()
        raise RuntimeError(
            f"Student tool-choice preflight failed at {base_url}: {detail}. "
            "The endpoint must accept tools with tool_choice='auto'. "
            "For Gemma 4 on vLLM 0.27.1, start the student server with "
            "--enable-auto-tool-choice --tool-call-parser gemma4 "
            "(see docs/tau2_setup.md). No tau2 tasks or paid calls were started."
        ) from exc


def nonnegative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return number


def nonnegative_usd(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("must be finite and nonnegative")
    return number


def parser(teacher: bool = False) -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--out-dir", "--output", type=Path, required=True,
                        help="New output directory; existing runs are never overwritten")
    result.add_argument("--max-usd", type=nonnegative_usd, default=3.0)
    result.add_argument("--max-tasks", type=nonnegative_int, default=None,
                        help="Hard total task cap across all domains (0 makes no calls)")
    result.add_argument("--seed", type=int, default=0)
    result.add_argument("--max-completion-tokens", type=int, default=2048,
                        help="Per-request output ceiling for both paid participants")
    if teacher:
        result.add_argument("--tasks-per-domain", "--n", type=nonnegative_int, default=5)
    else:
        result.add_argument("--base-url", default="http://localhost:8950/v1")
        result.add_argument("--model", default="gemma4-12b-base",
                            help="Served model name for google/gemma-4-12B-it")
    return result


def _usage(events: list[dict], *purposes: str) -> dict[str, int]:
    return {key: sum(event.get("usage", {}).get(key, 0) for event in events
                     if event["purpose"] in purposes) for key in FIELDS}


def _score(rows: list[dict]) -> dict[str, Any]:
    return {"passed": sum(row["passed"] for row in rows), "tasks": len(rows),
            "pass^1": sum(row["passed"] for row in rows) / len(rows) if rows else None,
            "infrastructure_errors": sum(row["termination_reason"] == "infrastructure_error" for row in rows)}


def evaluate(args: argparse.Namespace, *, teacher: bool = False) -> dict[str, Any]:
    if not 1 <= args.max_completion_tokens <= 128_000:
        raise ValueError("--max-completion-tokens must be in [1, 128000]")
    tier = service_tier()
    # These tools define one fixed pair. Ambient Azure/Ollama settings cannot
    # silently change either participant or the cost model.
    os.environ["BFAS_TAU2_USER_MODEL"] = LUNA_MODEL
    os.environ["BFAS_TAU2_TEACHER_MODEL"] = LUNA_MODEL
    adapter = Tau2Adapter(seed=args.seed,
                         served_model_name=LUNA_MODEL if teacher else args.model,
                         base_url=None if teacher else args.base_url)
    selected: dict[str, list[str]] = {}
    split_sizes = {}
    for domain in DOMAINS:
        splits = adapter._splits(domain)
        if not splits.get("test"):
            raise ValueError(f"{domain} has no official test split; refusing a base/train fallback")
        ids = splits["test"]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate test task ids for {domain}")
        split_sizes[domain] = len(ids)
        selected[domain] = ids[:args.tasks_per_domain] if teacher else ids
    # Interleave domains so small total caps retain coverage where possible.
    plan = [(domain, task_id) for group in zip_longest(*(selected[d] for d in DOMAINS))
            for domain, task_id in zip(DOMAINS, group) if task_id is not None]
    available = len(plan)
    if args.max_tasks is not None:
        plan = plan[:args.max_tasks]
    if plan and args.max_usd > 0 and not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required")
    out = args.out_dir.resolve()
    if out.exists():
        raise FileExistsError(f"output directory already exists: {out}")
    if not teacher and plan and args.max_usd > 0:
        probe_student_tool_choice(adapter.base_url, args.model)
    out.mkdir(parents=True, exist_ok=False)
    (out / "tasks").mkdir()
    (out / "tasks.jsonl").touch()
    state_path = out / "budget.json"
    state = {"max_usd": args.max_usd, "estimated_usd": 0.0,
             "service_tier": tier, "rates_usd_per_mtok": price_rates(tier),
             "charged_upper_bound_usd": 0.0, "events": [],
             "stop_reason": "max_usd" if args.max_usd == 0 else None}
    write_json(state_path, state)
    rows: list[dict] = []

    def save_metrics(*, final: bool = False) -> dict[str, Any]:
        metrics = {"benchmark": "tau2", "split": "test", "num_trials": 1,
                   "max_steps": 30, "seed": args.seed,
                   "agent": LUNA_MODEL if teacher else args.model,
                   "student_checkpoint": None if teacher else "google/gemma-4-12B-it",
                   "base_url": None if teacher else args.base_url,
                   "user_simulator": LUNA_MODEL, "service_tier": tier,
                   "nl_assertion_judge": LUNA_MODEL,
                   "max_usd": args.max_usd, "max_tasks": args.max_tasks,
                   "max_completion_tokens": args.max_completion_tokens,
                   "split_sizes": split_sizes, "available_tasks": available,
                   "scheduled_tasks": len(plan), "complete": final and len(rows) == available and not state["stop_reason"],
                   "stop_reason": state["stop_reason"] or ("running" if not final else "max_tasks" if len(plan) < available else "completed"),
                   **_score(rows),
                   "per_domain": {d: _score([r for r in rows if r["domain"] == d]) for d in DOMAINS},
                   "simulator_usage": _usage(state["events"], "user_sim"),
                   "teacher_usage": _usage(state["events"], "teacher_probe", "teacher_judge"),
                   "judge_usage": _usage(state["events"], "teacher_judge"),
                   "estimated_usd": state["estimated_usd"],
                   "charged_upper_bound_usd": state["charged_upper_bound_usd"],
                   "rates_usd_per_mtok": price_rates(tier),
                   "estimated_requests": sum(e["status"] == "estimated" for e in state["events"]),
                   "unknown_charge_requests": sum(e["status"] not in {"settled", "estimated"}
                                                   for e in state["events"])}
        write_json(out / "metrics.json", metrics)
        return metrics

    save_metrics()
    for index, (domain, native_id) in enumerate(plan):
        if state["stop_reason"]:
            break
        task_id = encode_task_id(domain, native_id)
        config_path = out / "request-config.json"
        write_json(config_path, {"state_path": str(state_path), "task_id": task_id,
                                "service_tier": tier,
                                "ledger_path": str(out / "ledger/tau2.jsonl"),
                                "max_completion_tokens": args.max_completion_tokens})
        model, model_args = adapter._teacher_args(0) if teacher else adapter._student_args(0)
        model_args["metadata"] = {"bfas_purpose": "teacher_probe" if teacher else "student"}
        if not teacher:
            model_args.update(max_tokens=2048, num_retries=0)
        simulation = None
        error = None
        native_run = None
        try:
            native_run = adapter._run_cli(domain=domain, task_ids=[native_id], split="test",
                                         agent_model=model, agent_args=model_args, num_trials=1,
                                         max_steps=30, budget_config=config_path)
            simulations = adapter._simulations(native_run.results)
            if len(simulations) != 1 or str(simulations[0].get("task_id")) != native_id:
                raise ValueError("native CLI did not return exactly the requested single trial")
            simulation = simulations[0]
            write_json(out / "tasks" / f"{index:04d}-native.json", native_run.results)
        except (subprocess.CalledProcessError, OSError, ValueError) as exc:
            # Save a counted failure and retain durable usage after CLI/parser errors.
            error = type(exc).__name__
        finally:
            if native_run is not None:
                native_run.cleanup()
        state = json.loads(state_path.read_text())
        if any(event["status"] == "reserved" for event in state["events"]):
            state["stop_reason"] = "unknown_charge"
            write_json(state_path, state)
        events = [event for event in state["events"] if event["task_id"] == task_id]
        row = {"task_id": task_id, "domain": domain, "native_id": native_id,
               "trial": 0, "passed": bool(simulation and extract_verdict(simulation)),
               "termination_reason": (simulation or {}).get("termination_reason", "infrastructure_error"),
               "error": error, "stop_reason": state["stop_reason"],
               "simulator_usage": _usage(events, "user_sim"),
               "teacher_usage": _usage(events, "teacher_probe", "teacher_judge"),
               "judge_usage": _usage(events, "teacher_judge"),
               "estimated_requests": sum(e["status"] == "estimated" for e in events),
               "estimated_usd": sum(e.get("estimated_usd", 0) for e in events)}
        rows.append(row)
        with (out / "tasks.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        save_metrics()
    return save_metrics(final=True)


def main(argv: list[str] | None = None, *, teacher: bool = False) -> None:
    cli = parser(teacher)
    args = cli.parse_args(argv)
    try:
        metrics = evaluate(args, teacher=teacher)
    except (ValueError, RuntimeError, OSError) as exc:
        cli.error(str(exc))
    print(json.dumps(metrics, indent=2))
