"""CPU regressions from live run 4; all provider responses are synthetic.

The saved failures have empty message lists and no provider response. Fixtures
retain their task/configuration and final reservation, not invented API usage.
"""
from __future__ import annotations

import json
import socket
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bfas.adapters.tau2 import Tau2Adapter, side_usage
from bfas.tau2_budget import (
    BudgetStopped, RequestBudget, charge_response, cost_usd, normalize_model_name,
    register_luna_price, request_usage_bound, response_usage, write_json,
)

FIXTURE = Path(__file__).parent / "fixtures/tau2_charging/live_run4.json"
CASES = json.loads(FIXTURE.read_text())


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr(socket.socket, "connect", lambda *a: pytest.fail("network forbidden"))


def recorded_request(case):
    # Rebuild the tool-bearing shape from the recorded rubric. Native tau2
    # supplies tool_choice=auto whenever tools are present.
    action = case["task"]["evaluation_criteria"]["actions"][0]
    properties = {key: {"type": "string"} for key in action["arguments"]}
    return {"model": case["actor"]["llm"], **case["actor"]["llm_args"],
            "messages": [{"role": "system", "content": case["policy"]},
                         {"role": "user", "content": json.dumps(case["task"]["user_scenario"])}],
            "tools": [{"type": "function", "function": {
                "name": action["name"], "description": "Recorded task action",
                "parameters": {"type": "object", "properties": properties}}}],
            "tool_choice": "auto", "max_completion_tokens": 2048}


def make_budget(tmp_path, case, tier, max_usd=3):
    state = tmp_path / "budget.json"
    write_json(state, {"max_usd": max_usd, "estimated_usd": 0.0,
                      "charged_upper_bound_usd": 0.0, "events": [], "stop_reason": None})
    config = tmp_path / "config.json"
    write_json(config, {"state_path": str(state), "task_id": case["budget_event"]["task_id"],
                        "ledger_path": str(tmp_path / "ledger.jsonl"), "service_tier": tier,
                        "max_completion_tokens": 2048})
    return RequestBudget(config), state


@pytest.mark.parametrize("case", CASES, ids=["telecom_hard", "luna_agent_task5"])
@pytest.mark.parametrize("tier", ["default", "flex"])
@pytest.mark.parametrize("shape", ["no_details", "null_cache", "attributes", "missing_usage", "zero_usage"])
def test_recorded_paid_calls_settle_or_estimate(tmp_path, case, tier, shape):
    assert case["error"]["error"] == "unknown_charge"
    assert case["simulation"]["messages"] == []
    request = recorded_request(case)
    assert request["metadata"]["bfas_purpose"] == case["budget_event"]["purpose"]
    guard, path = make_budget(tmp_path, case, tier)
    usage = {"prompt_tokens": 500, "completion_tokens": 25}
    if shape == "null_cache":
        usage["prompt_tokens_details"] = {"cached_tokens": None}
    response = {"model": request["model"], "usage": usage}
    if shape == "attributes":
        response = SimpleNamespace(model="gpt-5.6-luna-2026-09-10",
                                   usage=SimpleNamespace(**usage))
    if shape == "missing_usage":
        response = SimpleNamespace(model=request["model"])
    if shape == "zero_usage":
        response["usage"] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    estimated = shape in {"missing_usage", "zero_usage"}
    sent = []
    def completion(**kwargs):
        sent.append(kwargs)
        return response
    assert guard.call(completion, **request) is response
    state = json.loads(path.read_text())
    event = state["events"][0]
    assert len(sent) == 1 and sent[0]["tool_choice"] == "auto"
    assert state["stop_reason"] is None
    assert event["status"] == ("estimated" if estimated else "settled")
    expected_usage = request_usage_bound(sent[0]) if estimated else {
        "prompt_tokens": 500, "completion_tokens": 25, "cached_tokens": 0}
    assert event["usage"] == expected_usage
    assert state["estimated_usd"] == pytest.approx(cost_usd(expected_usage, tier))
    assert state["charged_upper_bound_usd"] == pytest.approx(state["estimated_usd"])
    assert state["charged_upper_bound_usd"] <= state["max_usd"]
    ledger = json.loads((tmp_path / "ledger.jsonl").read_text())
    assert ledger["usage_status"] == event["status"]
    role = "user" if case["role"] == "user" else "assistant"
    assert side_usage({"messages": [{"role": role, "raw_data": response}]}, role) == expected_usage


@pytest.mark.parametrize("model", ["gpt-5.6-luna", "openai/gpt-5.6-luna",
                                    "gpt-5.6-luna-2026-09-10",
                                    "openai/gpt-5.6-luna-2026-09-10"])
def test_luna_aliases_use_official_channel_and_tool_capabilities(tmp_path, monkeypatch, model):
    monkeypatch.setenv("BFAS_TAU2_TEACHER_MODEL", model)
    monkeypatch.setenv("BFAS_TAU2_USER_MODEL", model)
    monkeypatch.setenv("BFAS_OPENAI_SERVICE_TIER", "flex")
    assert normalize_model_name(model) == "gpt-5.6-luna"
    adapter = Tau2Adapter()
    assert adapter._teacher_args(0.7) == adapter._user_args(0.7)
    wire_model, args = adapter._teacher_args(0.7)
    assert wire_model == "openai/" + model.rsplit("/", 1)[-1]
    assert args == {"base_url": "https://api.openai.com/v1", "service_tier": "flex"}
    registered = {}
    register_luna_price(SimpleNamespace(register_model=registered.update), "flex", model)
    for alias in (model, wire_model, model.rsplit("/", 1)[-1]):
        assert registered[alias]["supports_tool_choice"] is True
        assert registered[alias]["supports_function_calling"] is True
    guard, path = make_budget(tmp_path, CASES[1], "flex")
    guard.call(lambda **kw: {"model": model, "usage": {"prompt_tokens": 10, "completion_tokens": 1}},
               **{**recorded_request(CASES[1]), "model": model})
    assert json.loads(path.read_text())["events"][0]["status"] == "settled"


@pytest.mark.parametrize("purpose", ["user_sim", "teacher_probe", "teacher_judge"])
def test_estimates_consume_cap_and_block_next_call(tmp_path, purpose):
    request = recorded_request(CASES[1])
    request["metadata"] = {"bfas_purpose": purpose}
    reserve = cost_usd(request_usage_bound(request), "flex")
    guard, path = make_budget(tmp_path, CASES[1], "flex", max_usd=reserve * 1.5)
    guard.call(lambda **kw: {"usage": None}, **request)
    with pytest.raises(BudgetStopped, match="max_usd"):
        guard.call(lambda **kw: pytest.fail("cap must block before HTTP"), **request)
    state = json.loads(path.read_text())
    assert state["events"][0]["status"] == "estimated"
    assert state["charged_upper_bound_usd"] == pytest.approx(reserve)
    assert state["estimated_usd"] == pytest.approx(reserve)


@pytest.mark.parametrize("raw", [
    {"prompt_tokens": 50, "completion_tokens": 2, "prompt_tokens_details": None},
    {"input_tokens": 50, "output_tokens": 2, "input_tokens_details": {"cached_tokens": None}},
    SimpleNamespace(prompt_tokens=50, completion_tokens=2,
                    prompt_tokens_details=SimpleNamespace(cached_tokens=None)),
])
def test_usage_shapes_default_missing_cache_to_zero(raw):
    assert response_usage(SimpleNamespace(usage=raw)) == {
        "prompt_tokens": 50, "completion_tokens": 2, "cached_tokens": 0}


def test_flat_cached_usage_and_tool_schema_estimate():
    assert response_usage({"usage": {"prompt_tokens": 50, "completion_tokens": 2,
                                     "cached_tokens": 40}})["cached_tokens"] == 40
    request = recorded_request(CASES[0])
    assert request_usage_bound(request)["prompt_tokens"] > request_usage_bound({**request, "tools": []})["prompt_tokens"]
    row = charge_response({}, request, "flex")
    assert row["status"] == "estimated" and row["usage"]["cached_tokens"] == 0


def test_adapter_combines_summary_with_partial_raw_cache_details():
    simulation = {"messages": [{"role": "assistant", "usage": {
        "prompt_tokens": 50, "completion_tokens": 2}, "raw_data": {
            "usage": {"prompt_tokens_details": {"cached_tokens": 40}}}}]}
    assert side_usage(simulation, "assistant") == {
        "prompt_tokens": 50, "completion_tokens": 2, "cached_tokens": 40}


@pytest.mark.parametrize("change, reason", [
    ({"model": "openai/gpt-5.6-luna-typo"}, "unpriced_model"),
    ({"n": 2}, "unsupported_request_shape"),
    ({"stream": True}, "unsupported_request_shape"),
    ({"messages": [{"role": "user", "content": [{"type": "image_url"}]}]}, "non_text_request"),
])
def test_unbounded_or_unpriced_calls_stop_before_http(tmp_path, change, reason):
    guard, path = make_budget(tmp_path, CASES[0], "flex")
    with pytest.raises(BudgetStopped, match=reason):
        guard.call(lambda **kw: pytest.fail("must stop before HTTP"),
                   **{**recorded_request(CASES[0]), **change})
    assert json.loads(path.read_text())["events"] == []
