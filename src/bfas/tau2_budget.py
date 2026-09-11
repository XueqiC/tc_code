"""Durable, sequential request reservations for the tau2 Luna experiments."""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

from .ledger import append_episode

FIELDS = ("prompt_tokens", "cached_tokens", "completion_tokens")
LUNA_PRICE_TABLE = {
    "default": {"prompt_tokens": 0.20, "cached_tokens": 0.02, "completion_tokens": 1.20},
    "flex": {"prompt_tokens": 0.10, "cached_tokens": 0.01, "completion_tokens": 0.60},
}


def service_tier(value: str | None = None) -> str:
    """Resolve the experiment's priced tiers to OpenAI's wire names."""
    if value is None:
        value = os.environ.get("BFAS_OPENAI_SERVICE_TIER", "")
    value = value.strip().lower()
    if value in {"", "standard", "default"}:
        return "default"
    if value == "flex":
        return value
    raise ValueError("BFAS_OPENAI_SERVICE_TIER must be flex, default/standard, or unset; "
                     "other tiers have no budget price")


def price_rates(tier: str | None = None) -> dict[str, float]:
    return dict(LUNA_PRICE_TABLE[service_tier(tier)])


def register_luna_price(litellm: Any, tier: str | None = None) -> None:
    """Seed LiteLLM before tau2 starts; our ledger never uses its cost lookup.

    Register both the request and returned model names. Base rates use the
    selected tier even when a response omits service_tier; explicit flex keys
    also cover LiteLLM versions that select tier-specific cost fields.
    """
    rates = price_rates(tier)
    flex = price_rates("flex")
    entry = {
        "litellm_provider": "openai", "mode": "chat",
        "input_cost_per_token": rates["prompt_tokens"] / 1_000_000,
        "cache_read_input_token_cost": rates["cached_tokens"] / 1_000_000,
        "output_cost_per_token": rates["completion_tokens"] / 1_000_000,
        "input_cost_per_token_flex": flex["prompt_tokens"] / 1_000_000,
        "cache_read_input_token_cost_flex": flex["cached_tokens"] / 1_000_000,
        "output_cost_per_token_flex": flex["completion_tokens"] / 1_000_000,
    }
    litellm.register_model({name: dict(entry) for name in
                            ("gpt-5.6-luna", "openai/gpt-5.6-luna")})


def cost_usd(usage: dict[str, int], tier: str | None = None) -> float:
    """Charge returned usage at our rates; cached input is part of prompt_tokens."""
    rates = price_rates(tier)
    prompt = usage.get("prompt_tokens", 0)
    cached = min(prompt, usage.get("cached_tokens", 0))
    return ((prompt - cached) * rates["prompt_tokens"] + cached * rates["cached_tokens"]
            + usage.get("completion_tokens", 0) * rates["completion_tokens"]) / 1_000_000


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
        # Freeze the selected tier for every reservation and settlement in this run.
        self.service_tier = service_tier(self.config.get("service_tier"))
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
            kwargs.update(service_tier=self.service_tier, base_url="https://api.openai.com/v1",
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
                                "completion_tokens": self.config["max_completion_tokens"]},
                               self.service_tier)
            if state["charged_upper_bound_usd"] + reserve > state["max_usd"]:
                self._stop(state, "max_usd")
            event = {"task_id": self.config["task_id"], "purpose": purpose,
                     "status": "reserved", "reserved_usd": reserve,
                     "service_tier": self.service_tier}
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
            actual = cost_usd(usage, self.service_tier)
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
