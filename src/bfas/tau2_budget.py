"""Durable, sequential request reservations for the tau2 Luna experiments."""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

from .ledger import append_episode

FIELDS = ("prompt_tokens", "cached_tokens", "completion_tokens")
RATES = {"prompt_tokens": 0.10, "cached_tokens": 0.01, "completion_tokens": 0.60}


def cost_usd(usage: dict[str, int]) -> float:
    """Cached input is already included in prompt_tokens."""
    prompt = usage.get("prompt_tokens", 0)
    cached = min(prompt, usage.get("cached_tokens", 0))
    return ((prompt - cached) * 0.10 + cached * 0.01
            + usage.get("completion_tokens", 0) * 0.60) / 1_000_000


def response_usage(response: Any) -> dict[str, int]:
    raw = response.get("usage")
    if hasattr(raw, "model_dump"):
        raw = raw.model_dump()
    if not isinstance(raw, dict):
        raise ValueError("provider response has no usage; reservation retained")
    details = raw.get("prompt_tokens_details") or {}
    usage = {key: raw.get(key) for key in ("prompt_tokens", "completion_tokens")}
    usage["cached_tokens"] = details.get("cached_tokens", 0)
    if any(not isinstance(n, int) or isinstance(n, bool) or n < 0 for n in usage.values()):
        raise ValueError("provider response has invalid usage; reservation retained")
    if usage["cached_tokens"] > usage["prompt_tokens"]:
        raise ValueError("cached input exceeds total input")
    return usage


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


class BudgetStopped(RuntimeError):
    """No further model request may be issued in this run."""


class RequestBudget:
    """Reserve a conservative text-input bound plus enforced output maximum.

    Serialized UTF-8 bytes bound byte-tokenized content. Extra framing allowance
    covers chat/tool protocol tokens. Only text requests below 272K input are
    accepted. Calls are sequential, retries disabled, and unknown charges retain
    their full reservation and stop the run. The cap uses the requested rates.
    """

    def __init__(self, config_path: Path):
        self.config = json.loads(config_path.read_text())
        self.path = Path(self.config["state_path"])
        self.lock = threading.Lock()

    def _stop(self, state: dict, reason: str) -> None:
        state["stop_reason"] = reason
        write_json(self.path, state)
        raise BudgetStopped(reason)

    def call(self, completion, **kwargs):
        with self.lock:
            state = json.loads(self.path.read_text())
            if state.get("stop_reason"):
                raise BudgetStopped(state["stop_reason"])
            purpose = (kwargs.get("metadata") or {}).get("bfas_purpose")
            if purpose == "student":
                kwargs["api_key"] = os.environ.get("BFAS_STUDENT_API_KEY", "EMPTY")
                return completion(**kwargs)
            if purpose not in {"user_sim", "teacher_probe", "teacher_judge"}:
                self._stop(state, "unrecognized_paid_request")
            if kwargs.get("model") != "openai/gpt-5.6-luna":
                self._stop(state, "unpriced_model")
            kwargs.pop("temperature", None)
            kwargs.pop("max_tokens", None)
            kwargs.update(service_tier="flex", base_url="https://api.openai.com/v1",
                          max_completion_tokens=self.config["max_completion_tokens"],
                          num_retries=0, max_retries=0, caching=False)
            payload = {key: kwargs.get(key) for key in ("messages", "tools", "tool_choice")}
            # Reject non-text content: images/audio have different token accounting.
            if any(not isinstance(m.get("content"), (str, type(None)))
                   for m in kwargs.get("messages", [])):
                self._stop(state, "non_text_request")
            framing = 4096 + 1024 * (len(kwargs.get("messages") or []) + len(kwargs.get("tools") or []))
            input_bound = len(json.dumps(payload, ensure_ascii=True).encode("utf-8")) + framing
            if input_bound > 272_000:
                self._stop(state, "input_bound_exceeded")
            reserve = cost_usd({"prompt_tokens": input_bound,
                                "completion_tokens": self.config["max_completion_tokens"]})
            if state["charged_upper_bound_usd"] + reserve > state["max_usd"]:
                self._stop(state, "max_usd")
            event = {"task_id": self.config["task_id"], "purpose": purpose,
                     "status": "reserved", "reserved_usd": reserve}
            state["events"].append(event)
            state["charged_upper_bound_usd"] += reserve
            write_json(self.path, state)  # persisted before the network request
            try:
                response = completion(**kwargs)
                usage = response_usage(response)
                if usage["prompt_tokens"] > input_bound or usage["completion_tokens"] > self.config["max_completion_tokens"]:
                    raise ValueError("provider exceeded the request reservation")
            except Exception:
                # A timeout can hide a billable completion. Do not release or retry.
                event["status"] = "unknown_charge"
                self._stop(state, "unknown_charge")
            actual = cost_usd(usage)
            event.update(status="settled", usage=usage, estimated_usd=actual)
            state["charged_upper_bound_usd"] += actual - reserve
            state["estimated_usd"] += actual
            write_json(self.path, state)
            append_episode(
                Path(self.config["ledger_path"]), task_id=self.config["task_id"],
                teacher="openai/gpt-5.6-luna", attempt_index=len(state["events"]) - 1,
                temperature=0.0, verified=False,
                tokens_spent=usage["prompt_tokens"] + usage["completion_tokens"],
                usage=usage, purpose=purpose,
            )
            return response
