"""Durable, sequential request reservations for the tau2 Luna experiments."""

from __future__ import annotations

import json
import os
import re
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .ledger import append_episode

FIELDS = ("prompt_tokens", "cached_tokens", "completion_tokens")
LUNA_PRICE_TABLE = {
    "default": {"prompt_tokens": 0.20, "cached_tokens": 0.02, "completion_tokens": 1.20},
    "flex": {"prompt_tokens": 0.10, "cached_tokens": 0.01, "completion_tokens": 0.60},
}


def normalize_model_name(model: str) -> str:
    """Use one price key for provider-qualified and dated Luna aliases."""
    name = model.strip().rsplit("/", 1)[-1]
    if re.fullmatch(r"gpt-5\.6-luna(?:-\d{4}-\d{2}-\d{2})?", name):
        return "gpt-5.6-luna"
    return name


def is_luna_model(model: str | None) -> bool:
    return isinstance(model, str) and normalize_model_name(model) == "gpt-5.6-luna"


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


def register_luna_price(litellm: Any, tier: str | None = None, *models: str) -> None:
    """Seed LiteLLM before tau2 starts; our ledger never uses its cost lookup.

    Register both the request and returned model names. Base rates use the
    selected tier even when a response omits service_tier; explicit flex keys
    also cover LiteLLM versions that select tier-specific cost fields.
    """
    rates = price_rates(tier)
    flex = price_rates("flex")
    entry = {
        "litellm_provider": "openai", "mode": "chat",
        # GPT-5 parameter validation consults this map before sending tools.
        "supports_function_calling": True, "supports_tool_choice": True,
        "supports_parallel_function_calling": True,
        "input_cost_per_token": rates["prompt_tokens"] / 1_000_000,
        "cache_read_input_token_cost": rates["cached_tokens"] / 1_000_000,
        "output_cost_per_token": rates["completion_tokens"] / 1_000_000,
        "input_cost_per_token_flex": flex["prompt_tokens"] / 1_000_000,
        "cache_read_input_token_cost_flex": flex["cached_tokens"] / 1_000_000,
        "output_cost_per_token_flex": flex["completion_tokens"] / 1_000_000,
    }
    names = {"gpt-5.6-luna", "openai/gpt-5.6-luna"}
    for model in models:
        if is_luna_model(model):
            bare = model.strip().rsplit("/", 1)[-1]
            names.update((model, bare, f"openai/{bare}"))
    litellm.register_model({name: dict(entry) for name in names})


def cost_usd(usage: dict[str, int], tier: str | None = None) -> float:
    """Charge returned usage at our rates; cached input is part of prompt_tokens."""
    rates = price_rates(tier)
    prompt = usage.get("prompt_tokens", 0)
    cached = min(prompt, usage.get("cached_tokens", 0))
    return ((prompt - cached) * rates["prompt_tokens"] + cached * rates["cached_tokens"]
            + usage.get("completion_tokens", 0) * rates["completion_tokens"]) / 1_000_000


def response_field(value: Any, key: str, default: Any = None) -> Any:
    """Read JSON mappings, LiteLLM/Pydantic responses and attribute objects."""
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


class MissingUsage(ValueError):
    """A completed response lacks the counters required for exact charging."""


def response_usage(response: Any, *, fallback: Any = None) -> dict[str, int]:
    """Normalize provider counters, optionally filling gaps from a native summary."""
    raw = response_field(response, "usage")
    usage = {
        "prompt_tokens": response_field(raw, "prompt_tokens", response_field(raw, "input_tokens")),
        "completion_tokens": response_field(raw, "completion_tokens", response_field(raw, "output_tokens")),
    }
    for key, alias in (("prompt_tokens", "input_tokens"), ("completion_tokens", "output_tokens")):
        if usage[key] is None:
            usage[key] = response_field(fallback, key, response_field(fallback, alias))
    details = (response_field(raw, "prompt_tokens_details")
               or response_field(raw, "input_tokens_details"))
    cached = response_field(details, "cached_tokens", response_field(
        raw, "cached_tokens", response_field(fallback, "cached_tokens")))
    usage["cached_tokens"] = 0 if cached is None else cached
    if any(n is not None and (not isinstance(n, int) or isinstance(n, bool) or n < 0)
           for n in usage.values()):
        raise ValueError("provider response has invalid usage; reservation retained")
    if any(n is None for n in usage.values()):
        raise MissingUsage("provider response has missing usage counters")
    if usage["cached_tokens"] > usage["prompt_tokens"]:
        raise ValueError("cached input exceeds total input")
    return usage


def request_usage_bound(request: Mapping[str, Any]) -> dict[str, int]:
    """Bound text/tool input by serialized bytes plus generous chat framing."""
    messages = request.get("messages") or []
    tools = request.get("tools") or []
    if any(not isinstance(m.get("content"), (str, type(None))) for m in messages):
        raise ValueError("non_text_request")
    # Include structured-output schemas as well as tool schemas and arguments.
    payload = {key: request.get(key) for key in
               ("messages", "tools", "tool_choice", "response_format")}
    framing = 4096 + 1024 * (len(messages) + len(tools))
    prompt = len(json.dumps(payload, ensure_ascii=True).encode("utf-8")) + framing
    if prompt > 272_000:
        raise ValueError("input_bound_exceeded")
    maximum = request.get("max_completion_tokens", request.get("max_tokens"))
    if not isinstance(maximum, int) or isinstance(maximum, bool) or maximum <= 0:
        raise ValueError("missing_completion_bound")
    if request.get("n", 1) != 1 or request.get("stream", False):
        raise ValueError("unsupported_request_shape")
    return {"prompt_tokens": prompt, "cached_tokens": 0, "completion_tokens": maximum}


def charge_response(response: Any, request: Mapping[str, Any], tier: str) -> dict:
    """Price every Luna response locally; missing usage consumes the full bound.

    Keep the provenance on raw_data so native tau2 cost/usage reporting and
    adapter ledgers use exactly the same counters without inventing API usage.
    """
    if not is_luna_model(request.get("model")):
        raise ValueError("unpriced_model")
    returned_model = response_field(response, "model")
    if returned_model and not is_luna_model(returned_model):
        raise ValueError("unpriced_response_model")
    bound = request_usage_bound(request)
    status = "settled"
    try:
        usage = response_usage(response)
        # LiteLLM synthesizes Usage(0, 0, 0) when HTTP omitted usage entirely.
        # A completed chat request has input framing, so this is not a free call.
        if usage["prompt_tokens"] == usage["completion_tokens"] == 0:
            raise MissingUsage("provider response has placeholder usage")
    except MissingUsage:
        usage, status = bound, "estimated"
    if any(usage[key] > bound[key] for key in ("prompt_tokens", "completion_tokens")):
        raise ValueError("provider exceeded the request reservation")
    charge = {"model": normalize_model_name(request["model"]), "service_tier": tier,
              "status": status, "usage": usage, "estimated_usd": cost_usd(usage, tier)}
    if isinstance(response, dict):
        response["bfas_charge"] = charge
    else:
        setattr(response, "bfas_charge", charge)
    return charge


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
    accepted. Missing response usage consumes the full reservation as an
    estimate. Request failures retain their reservation and stop the run.
    Calls are sequential, retries disabled; the cap uses the requested rates.
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
            if not is_luna_model(kwargs.get("model")):
                self._stop(state, "unpriced_model")
            kwargs["model"] = "openai/" + kwargs["model"].strip().rsplit("/", 1)[-1]
            kwargs.pop("temperature", None)
            kwargs.pop("max_tokens", None)
            kwargs.update(service_tier=self.service_tier, base_url="https://api.openai.com/v1",
                          max_completion_tokens=self.config["max_completion_tokens"],
                          num_retries=0, max_retries=0, caching=False)
            try:
                bound = request_usage_bound(kwargs)
            except ValueError as exc:
                self._stop(state, str(exc))
            reserve = cost_usd(bound, self.service_tier)
            if state["charged_upper_bound_usd"] + reserve > state["max_usd"]:
                self._stop(state, "max_usd")
            event = {"task_id": self.config["task_id"], "purpose": purpose,
                     "status": "reserved", "reserved_usd": reserve,
                     "service_tier": self.service_tier, "model": kwargs["model"],
                     "reserved_usage": bound}
            state["events"].append(event)
            state["charged_upper_bound_usd"] += reserve
            write_json(self.path, state)  # persisted before the network request
            try:
                response = completion(**kwargs)
                charge = charge_response(response, kwargs, self.service_tier)
            except Exception as exc:
                # A timeout can hide a billable completion. Do not release or retry.
                event["status"] = "unknown_charge"
                event["error_type"] = type(exc).__name__
                event["error"] = str(exc)
                self._stop(state, "unknown_charge")
            usage, actual = charge["usage"], charge["estimated_usd"]
            event.update(charge)
            state["charged_upper_bound_usd"] += actual - reserve
            state["estimated_usd"] += actual
            write_json(self.path, state)
            append_episode(
                Path(self.config["ledger_path"]), task_id=self.config["task_id"],
                teacher="openai/gpt-5.6-luna", attempt_index=len(state["events"]) - 1,
                temperature=0.0, verified=False,
                tokens_spent=usage["prompt_tokens"] + usage["completion_tokens"],
                usage=usage, purpose=purpose, usage_status=charge["status"],
            )
            return response
