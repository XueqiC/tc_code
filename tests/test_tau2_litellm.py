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


def test_luna_wire_parameters_and_cached_usage(monkeypatch, tmp_path):
    litellm = pytest.importorskip("litellm")
    httpx = pytest.importorskip("httpx")
    openai = pytest.importorskip("openai")
    from bfas.adapters.tau2 import LUNA_MODEL, Tau2Adapter
    from bfas.tau2_budget import RequestBudget, write_json

    monkeypatch.setattr(socket.socket, "connect", lambda *args: pytest.fail("network forbidden"))
    monkeypatch.setenv("BFAS_TAU2_TEACHER_MODEL", LUNA_MODEL)
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
        RequestBudget(config_path).call(
            litellm.completion, model=model, **args, client=client,
            messages=[{"role": "user", "content": "Hello"}],
            metadata={"bfas_purpose": "teacher_probe"},
        )
    assert len(requests) == 1
    assert str(requests[0].url) == "https://api.openai.com/v1/chat/completions"
    body = json.loads(requests[0].content)
    assert body["model"] == "gpt-5.6-luna"
    assert body["service_tier"] == "flex"
    assert body["max_completion_tokens"] == 100
    assert "temperature" not in body
    state = json.loads(state_path.read_text())
    assert state["events"][0]["usage"]["cached_tokens"] == 40


def test_cli_runtime_routes_and_records_native_judge(monkeypatch, tmp_path):
    import importlib.util
    pytest.importorskip("tau2")
    from tau2.evaluator import evaluator_nl_assertions as judge
    from tau2.utils import llm_utils

    path = Path(__file__).resolve().parents[1] / "tools/tau2_guarded_cli.py"
    spec = importlib.util.spec_from_file_location("tau2_runtime_test", path)
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)
    usage_path = tmp_path / "judge.jsonl"
    monkeypatch.delenv("BFAS_TAU2_BUDGET_CONFIG", raising=False)
    monkeypatch.setenv("BFAS_TAU2_JUDGE_USAGE_PATH", str(usage_path))
    monkeypatch.setattr(sys, "argv", [str(path), "stub-tau2", "run"])
    # Register these globals with monkeypatch before the runtime changes them.
    monkeypatch.setattr(judge, "DEFAULT_LLM_NL_ASSERTIONS", judge.DEFAULT_LLM_NL_ASSERTIONS)
    monkeypatch.setattr(judge, "DEFAULT_LLM_NL_ASSERTIONS_ARGS", judge.DEFAULT_LLM_NL_ASSERTIONS_ARGS)
    monkeypatch.setattr(llm_utils, "completion", lambda **kwargs: {
        "usage": {"prompt_tokens": 100, "completion_tokens": 5,
                  "prompt_tokens_details": {"cached_tokens": 40}}})
    context = llm_utils.llm_log_dir.set(tmp_path / "sim_abc/llm_debug")
    def cli_stub(filename, run_name):
        assert filename == "stub-tau2" and run_name == "__main__"
        assert sys.argv == ["stub-tau2", "run"]
        assert judge.DEFAULT_LLM_NL_ASSERTIONS == "openai/gpt-5.6-luna"
        assert "temperature" not in judge.DEFAULT_LLM_NL_ASSERTIONS_ARGS
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
