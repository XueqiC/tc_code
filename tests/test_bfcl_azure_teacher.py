"""CPU-only Azure mapping and BFCL teacher ledger integration; no API traffic."""

import copy
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from openai.types.chat import ChatCompletion

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "envs/bfcl/gorilla/berkeley-function-call-leaderboard"))

from bfas import bfcl_azure
from bfas.adapters import bfcl
from bfas.ledger import acquire_demos, ledger_path


FUNCTIONS = [{
    "name": "weather.lookup", "description": "Look up the weather",
    "parameters": {"type": "dict", "properties": {
        "city": {"type": "string", "description": "City name"},
    }, "required": ["city"]},
}]


def completion(calls=True, *, empty=False):
    message = {"role": "assistant", "content": None if calls else "No suitable tool."}
    if calls:
        message["tool_calls"] = [{
            "id": f"call_{i}", "type": "function",
            "function": {"name": "weather_lookup", "arguments": json.dumps({"city": city})},
        } for i, city in enumerate(("Paris", "東京"))]
    elif empty:
        message["content"] = ""
        message["tool_calls"] = []
    return ChatCompletion.model_validate({
        "id": "stub", "object": "chat.completion", "created": 0, "model": "gpt-5.4",
        "choices": [{"index": 0, "finish_reason": "tool_calls" if calls else "stop", "message": message}],
        "usage": {"prompt_tokens": 19, "completion_tokens": 80, "total_tokens": 99,
                  "completion_tokens_details": {"reasoning_tokens": 60}},
    })


@pytest.fixture
def client(monkeypatch, tmp_path):
    for key in ("AZURE_LLM_ENDPOINT", "AZURE_LLM_KEY", "AZURE_OPENAI_API_VERSION", "BFAS_BFCL_AZURE_REASONING_EFFORT"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("AZURE_LLM_ENDPOINT", "https://stub.openai.azure.com/")
    monkeypatch.setenv("AZURE_LLM_KEY", "fake-azure-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://wrong.invalid/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "wrong-key")
    stub = SimpleNamespace(requests=[], constructor=None, response=completion())

    def create(**kwargs):
        stub.requests.append(copy.deepcopy(kwargs))
        return stub.response

    def construct(**kwargs):
        stub.constructor = kwargs
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    monkeypatch.setattr(bfcl_azure, "AzureOpenAI", construct)
    return stub


def handler():
    return bfcl_azure.AzureOpenAIHandler("gpt-5.4", 0.7, "azure/gpt-5.4-FC", True)


def test_credentials(client, monkeypatch, tmp_path):
    monkeypatch.delenv("AZURE_LLM_KEY")
    (tmp_path / ".azure_llm_api").write_text(
        "AZURE_LLM_ENDPOINT=https://file.invalid/\nAZURE_LLM_KEY=file-key\n"
    )
    monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "test-version")
    handler()
    assert client.constructor == {"azure_endpoint": "https://stub.openai.azure.com",
                                  "api_key": "file-key", "api_version": "test-version"}
    monkeypatch.delenv("AZURE_LLM_ENDPOINT")
    handler()
    assert client.constructor["azure_endpoint"] == "https://file.invalid"
    (tmp_path / ".azure_llm_api").unlink()
    with pytest.raises(RuntimeError, match="Azure credentials not found"):
        handler()
    assert not client.requests


def test_registry_in_harness_environment():
    # The main training venv does not install unrelated providers (e.g. Google
    # GenAI). Registration is used by the CLI in BFCL's own dependency env.
    code = """
from bfas import bfcl_azure
bfcl_azure.register_azure_model()
bfcl_azure.register_azure_model()
from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING, api_inference_model_map
from bfcl_eval.constants.supported_models import SUPPORTED_MODELS
config = MODEL_CONFIG_MAPPING['azure/gpt-5.4-FC']
assert config is api_inference_model_map['azure/gpt-5.4-FC']
assert config.model_name == 'gpt-5.4'
assert config.model_handler is bfcl_azure.AzureOpenAIHandler
assert config.is_fc_model and config.underscore_to_dot
assert SUPPORTED_MODELS.count('azure/gpt-5.4-FC') == 1
"""
    outcome = subprocess.run(
        [str(bfcl.BFCL_BIN.with_name("python")), "-c", code],
        env={**os.environ, "PYTHONPATH": os.pathsep.join((str(ROOT / "src"), str(bfcl.BFCL_ROOT)))},
        capture_output=True, text=True,
    )
    assert outcome.returncode == 0, outcome.stderr


def test_native_fc_request_response_and_history(client):
    h = handler()
    original = copy.deepcopy(FUNCTIONS)
    entry = {"id": "parallel_0", "function": FUNCTIONS,
             "question": [[{"role": "user", "content": "Weather in Paris and Tokyo?"}]]}
    result, metadata = h.inference(entry, include_input_log=True, exclude_state_log=True)
    request = client.requests[0]
    assert request["model"] == "gpt-5.4"
    assert request["messages"] == entry["question"][0]
    assert request["temperature"] == 0.7 and request["reasoning_effort"] == "none"
    assert request["tools"] == [{"type": "function", "function": {
        **FUNCTIONS[0], "name": "weather_lookup",
        "parameters": {**FUNCTIONS[0]["parameters"], "type": "object"},
    }}]
    assert FUNCTIONS == original
    assert h.decode_ast(result, "Python", False) == [
        {"weather_lookup": {"city": "Paris"}}, {"weather_lookup": {"city": "東京"}},
    ]
    assert h.decode_execute(result, False) == ["weather_lookup(city='Paris')", "weather_lookup(city='東京')"]
    assert metadata["output_token_count"] == 80  # includes 60 reasoning, not 140
    assert metadata["input_token_count"] == 19
    parsed = h._parse_query_response_FC(client.response)
    history = h._add_assistant_message_FC({"message": []}, parsed)
    h._add_execution_results_FC(history, ["sunny", "rainy"], parsed)
    assert [m["tool_call_id"] for m in history["message"][1:]] == ["call_0", "call_1"]


@pytest.mark.parametrize("empty", [False, True])
def test_no_call_and_no_tools(client, empty):
    client.response = completion(False, empty=empty)
    h = handler()
    response, _ = h._query_FC({"message": [{"role": "user", "content": "Hello"}], "tools": []})
    assert "tools" not in client.requests[0]
    parsed = h._parse_query_response_FC(response)
    assert parsed["model_responses"] == ("" if empty else "No suitable tool.")
    assert h.decode_ast(parsed["model_responses"], "Python", False) == []
    assert h.decode_execute(parsed["model_responses"], False) == []


def test_reasoning_omits_sampling_parameters(client, monkeypatch):
    monkeypatch.setenv("BFAS_BFCL_AZURE_REASONING_EFFORT", "high")
    handler()._query_FC({"message": [], "tools": []})
    assert client.requests[0]["reasoning_effort"] == "high"
    assert "temperature" not in client.requests[0]


def test_adapter_azure_command_avoids_ollama_and_vllm(monkeypatch):
    calls = []
    monkeypatch.setattr(bfcl.subprocess, "run", lambda command, **kwargs: calls.append((command, kwargs)))
    monkeypatch.setenv("OPENAI_BASE_URL", "https://untouched.invalid")
    bfcl.BFCLAdapter()._run_generate(["--model", "azure/gpt-5.4-FC"])
    command, options = calls[0]
    assert command[:3] == [str(bfcl.BFCL_BIN.with_name("python")), str(bfcl.BFCL_CLI), "generate"]
    assert "--backend" not in command
    assert options["env"]["OPENAI_BASE_URL"] == "https://untouched.invalid"
    assert str(bfcl.BFCL_CLI) in bfcl.BFCLAdapter._bfcl_command("evaluate", "--model", "azure/gpt-5.4-FC")


def test_ledger_charges_failed_attempts_and_nested_prerequisites(monkeypatch, tmp_path):
    adapter = bfcl.BFCLAdapter()
    monkeypatch.setenv("BFAS_BFCL_TEACHER", "azure/gpt-5.4-FC")
    monkeypatch.setattr(adapter, "task_categories", lambda: {"memory_kv_0": "memory_kv"})
    monkeypatch.setattr(adapter, "_messages", lambda task_id: [])
    monkeypatch.setattr(adapter, "_functions", lambda task_id: [])
    attempts = []

    def official(model, ids, temperature):
        attempts.append(temperature)
        directory = tmp_path / f"attempt_{len(attempts)}"
        directory.mkdir()
        return [
            {"id": "memory_kv_prereq_0", "output_token_count": [[5, 6], [7]]},
            {"id": "memory_kv_0", "result": "answer", "output_token_count": [[80, 2], [3]]},
        ], directory, directory

    monkeypatch.setattr(adapter, "_official_pass", official)
    monkeypatch.setattr(bfcl, "extract_verdicts", lambda *args: {"memory_kv_0": len(attempts) == 2})
    acquire_demos("bfcl", adapter, ["memory_kv_0"], 2, ledger_root=tmp_path / "ledger")
    records = [json.loads(line) for line in ledger_path("bfcl", ledger_root=tmp_path / "ledger").read_text().splitlines()]
    assert [r["tokens_spent"] for r in records] == [103, 103]
    assert [r["verified"] for r in records] == [False, True]
    assert attempts == [0.0, 0.7]
    assert bfcl._completion_tokens([{"output_token_count": 0}]) == 0
    assert bfcl._completion_tokens([{}]) is None
