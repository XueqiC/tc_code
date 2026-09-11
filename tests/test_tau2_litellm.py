"""Offline wire-format check against the installed tau2 LiteLLM dependency."""
from __future__ import annotations

import json
import os
import socket
import sys
from pathlib import Path

import pytest

os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.mark.parametrize("tier, expected_usd", [("flex", 0.0000094), ("default", 0.0000188)])
def test_luna_wire_parameters_and_cached_usage(monkeypatch, tmp_path, tier, expected_usd):
    litellm = pytest.importorskip("litellm")
    httpx = pytest.importorskip("httpx")
    openai = pytest.importorskip("openai")
    from bfas.adapters.tau2 import LUNA_MODEL, Tau2Adapter
    from bfas.tau2_budget import RequestBudget, register_luna_price, write_json

    monkeypatch.setattr(socket.socket, "connect", lambda *args: pytest.fail("network forbidden"))
    monkeypatch.setenv("BFAS_TAU2_TEACHER_MODEL", LUNA_MODEL)
    monkeypatch.setenv("BFAS_OPENAI_SERVICE_TIER", tier)
    # Reproduce the installed dependency's missing map entry, then run startup
    # registration before a real LiteLLM completion over mocked HTTP.
    for name in (LUNA_MODEL, "gpt-5.6-luna"):
        monkeypatch.delitem(litellm.model_cost, name, raising=False)
    register_luna_price(litellm)
    requests = []
    def handler(request):
        requests.append(request)
        return httpx.Response(200, json={
            "id": "chatcmpl-stub", "object": "chat.completion", "created": 0,
            "model": "gpt-5.6-luna",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": "OK"}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 5, "total_tokens": 105,
                      "prompt_tokens_details": {"cached_tokens": 40}},
        })
    state_path = tmp_path / "budget.json"
    write_json(state_path, {"max_usd": 3, "estimated_usd": 0.0,
                           "charged_upper_bound_usd": 0.0, "events": [], "stop_reason": None})
    config_path = tmp_path / "config.json"
    write_json(config_path, {"state_path": str(state_path), "task_id": "retail:1",
                            "ledger_path": str(tmp_path / "ledger/tau2.jsonl"),
                            "max_completion_tokens": 100})
    model, args = Tau2Adapter()._teacher_args(0.7)
    with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
        client = openai.OpenAI(api_key="test-key", http_client=http_client, max_retries=0)
        response = RequestBudget(config_path).call(
            litellm.completion, model=model, **args, client=client,
            messages=[{"role": "user", "content": "Hello"}],
            metadata={"bfas_purpose": "teacher_probe"},
        )
    # tau2 also asks LiteLLM for native result costs after the guarded call.
    assert litellm.completion_cost(completion_response=response) == pytest.approx(expected_usd)
    assert len(requests) == 1
    assert str(requests[0].url) == "https://api.openai.com/v1/chat/completions"
    body = json.loads(requests[0].content)
    assert body["model"] == "gpt-5.6-luna"
    assert body["service_tier"] == tier
    assert body["max_completion_tokens"] == 100
    assert "temperature" not in body
    state = json.loads(state_path.read_text())
    assert state["events"][0]["usage"]["cached_tokens"] == 40
    assert state["events"][0]["status"] == "settled"
    assert state["stop_reason"] is None
    assert state["estimated_usd"] == pytest.approx(expected_usd)
    assert state["charged_upper_bound_usd"] == pytest.approx(expected_usd)


def test_cli_runtime_routes_and_records_native_judge(monkeypatch, tmp_path):
    import importlib.util
    pytest.importorskip("tau2")
    from tau2.evaluator import evaluator_nl_assertions as judge
    from tau2.utils import llm_utils
    import litellm
    from bfas.tau2_budget import register_luna_price

    path = Path(__file__).resolve().parents[1] / "tools/tau2_guarded_cli.py"
    spec = importlib.util.spec_from_file_location("tau2_runtime_test", path)
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)
    usage_path = tmp_path / "judge.jsonl"
    monkeypatch.delenv("BFAS_TAU2_BUDGET_CONFIG", raising=False)
    monkeypatch.setenv("BFAS_TAU2_JUDGE_USAGE_PATH", str(usage_path))
    monkeypatch.setenv("BFAS_OPENAI_SERVICE_TIER", "default")
    registered = []
    def register(module, tier):
        registered.append(tier)
        register_luna_price(module, tier)
    monkeypatch.setattr("bfas.tau2_budget.register_luna_price", register)
    monkeypatch.setattr(sys, "argv", [str(path), "stub-tau2", "run"])
    # Register these globals with monkeypatch before the runtime changes them.
    monkeypatch.setattr(judge, "DEFAULT_LLM_NL_ASSERTIONS", judge.DEFAULT_LLM_NL_ASSERTIONS)
    monkeypatch.setattr(judge, "DEFAULT_LLM_NL_ASSERTIONS_ARGS", judge.DEFAULT_LLM_NL_ASSERTIONS_ARGS)
    monkeypatch.setattr(llm_utils, "completion", lambda **kwargs: {
        "usage": {"prompt_tokens": 100, "completion_tokens": 5,
                  "prompt_tokens_details": {"cached_tokens": 40}}})
    context = llm_utils.llm_log_dir.set(tmp_path / "sim_abc/llm_debug")
    def cli_stub(filename, run_name):
        assert registered == ["default"]
        assert filename == "stub-tau2" and run_name == "__main__"
        assert sys.argv == ["stub-tau2", "run"]
        assert judge.DEFAULT_LLM_NL_ASSERTIONS == "openai/gpt-5.6-luna"
        assert "temperature" not in judge.DEFAULT_LLM_NL_ASSERTIONS_ARGS
        assert judge.DEFAULT_LLM_NL_ASSERTIONS_ARGS["service_tier"] == "default"
        assert litellm.get_model_info("gpt-5.6-luna")["input_cost_per_token"] == 0.20 / 1_000_000
        llm_utils.completion(model=judge.DEFAULT_LLM_NL_ASSERTIONS,
                             **judge.DEFAULT_LLM_NL_ASSERTIONS_ARGS)
    monkeypatch.setattr(runtime.runpy, "run_path", cli_stub)
    try:
        runtime.main()
    finally:
        llm_utils.llm_log_dir.reset(context)
    row = json.loads(usage_path.read_text())
    assert row["simulation_id"] == "abc"
    assert row["usage"] == {"prompt_tokens": 100, "completion_tokens": 5, "cached_tokens": 40}
