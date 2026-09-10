"""API FC mapping and both collection paths, entirely on CPU with fake APIs."""

import copy
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest
from openai.types.chat import ChatCompletion

ROOT = Path(__file__).resolve().parents[1]
HARNESS = ROOT / "envs/bfcl/gorilla/berkeley-function-call-leaderboard"
sys.path[:0] = [str(ROOT / "src"), str(HARNESS), str(ROOT / "tools")]

from bfas import bfcl_ollama, bfcl_teacher, ledger
from bfas.adapters import bfcl

MODELS = ("gpt-oss:120b", "mistral-large-3:675b")
OPENROUTER_MODELS = (
    "openai/gpt-5.4", "anthropic/claude-sonnet-5",
    "google/gemini-3.1-pro-preview", "openai/gpt-5.6-luna",
)
PROVIDER_MODELS = (
    *(("ollama", model) for model in MODELS),
    *(("openrouter", model) for model in OPENROUTER_MODELS),
)
TEACHERS = tuple(f"{prefix}/{model}-FC" for prefix, model in PROVIDER_MODELS)
FUNCTIONS = [{"name": "weather.lookup", "description": "Weather",
              "parameters": {"type": "dict", "properties": {"city": {"type": "string"}},
                             "required": ["city"]}}]


def completion(content=None, *, tools=True):
    message = {"role": "assistant", "content": content}
    if tools:
        message["tool_calls"] = [
            {"id": f"call_{i}", "type": "function", "function": {
                "name": "weather_lookup", "arguments": json.dumps({"city": city})}}
            for i, city in enumerate(("Paris", "東京"))
        ]
    else:
        message["tool_calls"] = []
    return ChatCompletion.model_validate({
        "id": "stub", "object": "chat.completion", "created": 0, "model": "stub",
        "choices": [{"index": 0, "finish_reason": "tool_calls" if tools else "stop", "message": message}],
        "usage": {"prompt_tokens": 19, "completion_tokens": 80, "total_tokens": 99,
                  "completion_tokens_details": {"reasoning_tokens": 60}},
    })


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("OLLAMA_API_KEY", "stub-key")
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "stub-openrouter-key")
    monkeypatch.delenv("OPENROUTER_BASE_URL", raising=False)
    monkeypatch.delenv("BFAS_BFCL_USAGE_LOG", raising=False)
    monkeypatch.delenv("BFAS_BFCL_SKIP_MEMORY_PREREQ", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "wrong-provider-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://wrong.invalid/v1")
    stub = SimpleNamespace(requests=[], constructor=None, response=completion(), error=None)

    def create(**kwargs):
        stub.requests.append(copy.deepcopy(kwargs))
        if stub.error:
            raise stub.error
        return stub.response

    def construct(**kwargs):
        stub.constructor = kwargs
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    monkeypatch.setattr(bfcl_ollama, "OpenAI", construct)
    return stub


def handler(model=MODELS[0], prefix="ollama"):
    return bfcl_ollama.OpenAICompatibleHandler(model, 0.7, f"{prefix}/{model}-FC", True)


def entry(task_id="parallel_0"):
    return {"id": task_id, "function": copy.deepcopy(FUNCTIONS),
            "question": [[{"role": "user", "content": "Weather in Paris and Tokyo?"}]]}


@pytest.mark.parametrize(("prefix", "model"), PROVIDER_MODELS)
def test_native_fc_mapping_and_tool_history(client, prefix, model):
    h = handler(model, prefix)
    task = entry()
    original = copy.deepcopy(task)
    result, metadata = h.inference(task, True, True)
    request = client.requests[0]
    base_url, key = {
        "ollama": ("https://ollama.com/v1", "stub-key"),
        "openrouter": ("https://openrouter.ai/api/v1", "stub-openrouter-key"),
    }[prefix]
    assert client.constructor == {"base_url": base_url, "api_key": key, "max_retries": 0}
    assert request == {
        "model": model, "messages": task["question"][0], "temperature": 0.7,
        "tools": [{"type": "function", "function": {
            **FUNCTIONS[0], "name": "weather_lookup",
            "parameters": {**FUNCTIONS[0]["parameters"], "type": "object"},
        }}],
    }
    assert task == original
    assert h.decode_ast(result, "Python", False) == [
        {"weather_lookup": {"city": "Paris"}}, {"weather_lookup": {"city": "東京"}},
    ]
    assert h.decode_execute(result, False) == ["weather_lookup(city='Paris')", "weather_lookup(city='東京')"]
    assert metadata["output_token_count"] == 80  # includes reasoning, not 80+60
    assert metadata["input_token_count"] == 19
    parsed = h._parse_query_response_FC(client.response)
    history = h._add_assistant_message_FC({"message": []}, parsed)
    h._add_execution_results_FC(history, ["sunny", "rainy"], parsed)
    assert [m["tool_call_id"] for m in history["message"][1:]] == ["call_0", "call_1"]


@pytest.mark.parametrize("content", [None, "", "No matching tool."])
@pytest.mark.parametrize(("prefix", "model"), PROVIDER_MODELS)
def test_no_call_and_no_tools(client, content, prefix, model):
    client.response = completion(content, tools=False)
    task = entry("irrelevance_0")
    task["function"] = []
    h = handler(model, prefix)
    result, metadata = h.inference(task, False, True)
    assert result == (content or "")
    assert "tools" not in client.requests[0]
    assert h.decode_ast(result, "Python", False) == []
    assert h.decode_execute(result, False) == []
    assert metadata["output_token_count"] == 80


@pytest.mark.parametrize("base", ["https://stub.invalid", "https://stub.invalid/", "https://stub.invalid/v1/"])
def test_credentials_base_normalization_and_file_fallback(client, monkeypatch, tmp_path, base):
    monkeypatch.setenv("OLLAMA_BASE_URL", base)
    monkeypatch.delenv("OLLAMA_API_KEY")
    (tmp_path / ".ollama_api_key2").write_text("file-key\n")
    handler()
    assert client.constructor["base_url"] == "https://stub.invalid/v1"
    assert client.constructor["api_key"] == "file-key"
    (tmp_path / ".ollama_api_key2").unlink()
    with pytest.raises(RuntimeError, match="set OLLAMA_API_KEY"):
        handler()
    assert not client.requests


@pytest.mark.parametrize("key", [None, "", " \n\t"])
def test_openrouter_requires_own_env_key_without_file_fallback(client, monkeypatch, tmp_path, key):
    if key is None:
        monkeypatch.delenv("OPENROUTER_API_KEY")
    else:
        monkeypatch.setenv("OPENROUTER_API_KEY", key)
    for name in (".ollama_api_key2", ".ollama_api_key", ".openrouter_api_key"):
        (tmp_path / name).write_text("file-key\n")
    # Neither other providers' env keys nor any key files may satisfy this.
    with pytest.raises(RuntimeError, match="set OPENROUTER_API_KEY"):
        handler(OPENROUTER_MODELS[0], "openrouter")
    monkeypatch.setattr(bfcl.subprocess, "run", lambda *_args, **_kwargs: pytest.fail("missing key must fail before launch"))
    with pytest.raises(RuntimeError, match="set OPENROUTER_API_KEY"):
        bfcl.BFCLAdapter()._run_generate(["--model", f"openrouter/{OPENROUTER_MODELS[0]}-FC"])
    assert client.constructor is None and not client.requests


@pytest.mark.parametrize("base", ["https://stub.invalid/api/v1", "https://stub.invalid/api/v1/", "https://stub.invalid/gateway"])
def test_openrouter_custom_endpoint_and_env_key(client, monkeypatch, base):
    monkeypatch.setenv("OPENROUTER_BASE_URL", base)
    monkeypatch.setenv("OPENROUTER_API_KEY", "  env-key\n")
    handler(OPENROUTER_MODELS[0], "openrouter")
    assert client.constructor == {"base_url": base.rstrip("/"), "api_key": "env-key", "max_retries": 0}
    assert not client.requests


def test_registration_and_cli_without_credentials():
    code = """
from bfas.bfcl_ollama import register_ollama_models, OllamaOpenAIHandler
register_ollama_models()
register_ollama_models()
from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING, api_inference_model_map
from bfcl_eval.constants.supported_models import SUPPORTED_MODELS
for name in TEACHERS:
    model = name.split('/', 1)[1].removesuffix('-FC')
    config = MODEL_CONFIG_MAPPING[name]
    assert config is api_inference_model_map[name]
    assert config.model_name == model and config.model_handler is OllamaOpenAIHandler
    assert config.is_fc_model and config.underscore_to_dot
    assert SUPPORTED_MODELS.count(name) == 1
"""
    env = {**os.environ, "PYTHONPATH": os.pathsep.join((str(ROOT / "src"), str(HARNESS)))}
    for key in ("OPENROUTER_API_KEY", "OLLAMA_API_KEY", "OPENAI_API_KEY"):
        env.pop(key, None)
    code = f"TEACHERS = {TEACHERS!r}\n" + code
    for args in (["-c", code], [str(ROOT / "tools/bfcl_cli.py"), "models"]):
        outcome = subprocess.run([str(bfcl.BFCL_BIN.with_name("python")), *args],
                                 env=env, capture_output=True, text=True)
        assert outcome.returncode == 0, outcome.stderr
        if args[-1] == "models":
            assert all(name in outcome.stdout for name in TEACHERS)


@pytest.mark.parametrize("model", MODELS)
def test_adapter_routes_ollama_through_registered_cli(client, monkeypatch, model):
    calls = []
    monkeypatch.setattr(bfcl.subprocess, "run", lambda command, **kwargs: calls.append((command, kwargs)))
    monkeypatch.setenv("OLLAMA_BASE_URL", "https://stub.invalid/v1/")
    name = f"ollama/{model}-FC"
    bfcl.BFCLAdapter()._run_generate(["--model", name])
    command, kwargs = calls[0]
    assert command[:3] == [str(bfcl.BFCL_BIN.with_name("python")), str(bfcl.BFCL_CLI), "generate"]
    assert "--backend" not in command and "--num-gpus" not in command
    assert kwargs["env"]["OPENAI_BASE_URL"] == "https://stub.invalid/v1"
    assert kwargs["env"]["OPENAI_API_KEY"] == "stub-key"
    assert str(bfcl.BFCL_CLI) in bfcl.BFCLAdapter._bfcl_command("evaluate", "--model", name)
    assert not client.requests


@pytest.mark.parametrize("model", OPENROUTER_MODELS)
def test_adapter_routes_openrouter_through_registered_cli(client, monkeypatch, tmp_path, model):
    calls = []
    monkeypatch.setattr(bfcl.subprocess, "run", lambda command, **kwargs: calls.append((command, kwargs)))
    monkeypatch.delenv("OLLAMA_API_KEY")
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://stub.invalid/api/v1/")
    name = f"openrouter/{model}-FC"
    usage_log = tmp_path / "bfas_usage.jsonl"
    bfcl.BFCLAdapter()._run_generate(["--model", name], usage_log=usage_log)
    command, kwargs = calls[0]
    assert command[:3] == [str(bfcl.BFCL_BIN.with_name("python")), str(bfcl.BFCL_CLI), "generate"]
    assert "--backend" not in command and "--num-gpus" not in command
    assert kwargs["env"]["OPENAI_BASE_URL"] == "https://stub.invalid/api/v1"
    assert kwargs["env"]["OPENAI_API_KEY"] == "stub-openrouter-key"
    assert kwargs["env"]["BFAS_BFCL_USAGE_LOG"] == str(usage_log)
    assert str(bfcl.BFCL_CLI) in bfcl.BFCLAdapter._bfcl_command("evaluate", "--model", name)
    assert not client.requests


def test_provider_registry_drives_handler_and_adapter(client, monkeypatch):
    monkeypatch.setitem(bfcl_teacher.PROVIDERS, "stub", ("STUB_BASE_URL", "STUB_API_KEY", "https://stub.invalid/v1"))
    monkeypatch.setenv("STUB_API_KEY", "stub-provider-key")
    h = handler("org/model", "stub")
    h.inference(entry(), False, True)
    assert client.constructor == {"base_url": "https://stub.invalid/v1", "api_key": "stub-provider-key", "max_retries": 0}
    assert client.requests[0]["model"] == "org/model"
    calls = []
    monkeypatch.setattr(bfcl.subprocess, "run", lambda command, **kwargs: calls.append((command, kwargs)))
    bfcl.BFCLAdapter()._run_generate(["--model", "stub/org/model-FC"])
    command, kwargs = calls[0]
    assert str(bfcl.BFCL_CLI) in command and "--backend" not in command
    assert kwargs["env"]["OPENAI_API_KEY"] == "stub-provider-key"


@pytest.mark.parametrize(("prefix", "model"), PROVIDER_MODELS)
def test_usage_survives_parser_error_and_is_thread_local(client, monkeypatch, tmp_path, prefix, model):
    usage = tmp_path / "bfas_usage.jsonl"
    monkeypatch.setenv("BFAS_BFCL_USAGE_LOG", str(usage))
    h = handler(model, prefix)
    monkeypatch.setattr(h, "_parse_query_response_FC", lambda _: (_ for _ in ()).throw(ValueError("bad response")))
    with ThreadPoolExecutor(max_workers=4) as pool:
        outputs = list(pool.map(lambda i: h.inference(entry(f"parallel_{i}"), False, True), range(8)))
    assert all(meta["output_token_count"] == 80 and meta["error"] == "ValueError" for _, meta in outputs)
    rows = [json.loads(line) for line in usage.read_text().splitlines()]
    assert {row["id"] for row in rows} == {f"parallel_{i}" for i in range(8)}
    assert sum(row["output_token_count"] for row in rows) == 640
    assert len({row["bfas_attempt_id"] for row in rows}) == 8
    for _, metadata in outputs:
        attempt_rows = [row for row in rows if row["bfas_attempt_id"] == metadata["bfas_attempt_id"]]
        assert [row["output_token_count"] for row in attempt_rows] == [0, 80]
        assert [row["input_token_count"] for row in attempt_rows] == [0, 19]


@pytest.mark.parametrize(("prefix", "model"), PROVIDER_MODELS)
def test_failed_request_has_zero_usage_and_no_hidden_retry(client, prefix, model):
    client.error = RuntimeError("stub unavailable")
    _, metadata = handler(model, prefix).inference(entry(), False, True)
    assert metadata["error"] == "RuntimeError"
    assert metadata["output_token_count"] == 0
    assert len(client.requests) == 1


@pytest.fixture
def adapter(monkeypatch, tmp_path):
    monkeypatch.setenv("BFAS_BFCL_TEACHER", f"ollama/{MODELS[0]}-FC")
    monkeypatch.delenv("BFAS_BFCL_SKIP_MEMORY_PREREQ", raising=False)
    monkeypatch.setattr(ledger, "LEDGER_ROOT", tmp_path / "data/teacher_ledger")
    h = bfcl.BFCLAdapter()
    monkeypatch.setattr(h, "task_categories", lambda: {
        "parallel_0": "parallel", "parallel_1": "parallel", "memory_kv_0": "memory_kv",
        "memory_kv_prereq_0": "memory_kv_prereq",
    })
    monkeypatch.setattr(h, "_messages", lambda _: [])
    monkeypatch.setattr(h, "_functions", lambda _: [])
    monkeypatch.setattr(h, "_render", lambda *_: "prompt")
    return h


@pytest.mark.parametrize("model", OPENROUTER_MODELS)
def test_openrouter_gateway_usage_teacher_label_and_memory_skip(client, adapter, monkeypatch, tmp_path, model):
    name = f"openrouter/{model}-FC"
    monkeypatch.setenv("BFAS_BFCL_TEACHER", name)
    monkeypatch.setenv("BFAS_BFCL_SKIP_MEMORY_PREREQ", "1")

    def official(teacher, ids, temperature):
        assert teacher == name and ids == ["parallel_0"]
        directory = tmp_path / "results"
        monkeypatch.setenv("BFAS_BFCL_USAGE_LOG", str(directory / "bfas_usage.jsonl"))
        h = handler(model, "openrouter")
        result, metadata = h.inference(entry(), False, True)
        (directory / "BFCL_v4_parallel_result.json").write_text(
            json.dumps({"id": "parallel_0", "result": result, **metadata}) + "\n")
        return bfcl_teacher.read_results(directory), directory, directory

    monkeypatch.setattr(adapter, "_official_pass", official)
    monkeypatch.setattr(bfcl, "extract_verdicts", lambda *_: {"parallel_0": True})
    for _ in range(2):
        assert set(adapter.teacher_demo(["parallel_0", "memory_kv_0"], 1)) == {"parallel_0"}
    records = ledger.read_records("bfcl")
    assert [(row["teacher"], row["tokens_spent"], row["verified"]) for row in records] == [(name, 80, True)]
    assert len(client.requests) == 1
    inventory = ledger.LEDGER_ROOT / "bfcl_inventory.jsonl"
    rows = [json.loads(line) for line in inventory.read_text().splitlines()]
    assert [(row["teacher"], row["task_id"]) for row in rows] == [(name, "memory_kv_0")]


def test_adapter_teacher_demo_records_exact_failures_and_reuses_same_teacher(adapter, monkeypatch, tmp_path):
    calls = []

    def official(model, ids, temperature):
        calls.append((model, ids, temperature))
        directory = tmp_path / f"a{len(calls)}"
        directory.mkdir()
        return [{"id": ids[0], "result": "answer", "output_token_count": [[80, 2], [3]]},
                {"id": "memory_kv_prereq_0", "output_token_count": [[5, 6], [7]]}], directory, directory

    monkeypatch.setattr(adapter, "_official_pass", official)
    monkeypatch.setattr(bfcl, "extract_verdicts", lambda *_: {"memory_kv_0": len(calls) >= 2})
    assert set(adapter.teacher_demo(["memory_kv_0"], 3)) == {"memory_kv_0"}
    records = ledger.read_records("bfcl")
    assert [row["tokens_spent"] for row in records] == [103, 103]
    assert [row["verified"] for row in records] == [False, True]
    assert [row["temperature"] for row in records] == [0.0, 0.7]
    assert all(row["purpose"] == "teacher" for row in records)
    adapter.teacher_demo(["memory_kv_0"], 3)
    assert len(calls) == 2
    monkeypatch.setenv("BFAS_BFCL_TEACHER", f"ollama/{MODELS[1]}-FC")
    adapter.teacher_demo(["memory_kv_0"], 3)
    assert len(calls) == 3 and len(ledger.read_records("bfcl")) == 3


def test_evaluator_failure_charges_usage_in_gateway(adapter, monkeypatch, tmp_path):
    directory = tmp_path / "results"
    directory.mkdir()
    monkeypatch.setattr(adapter, "_official_pass", lambda *_: (
        [{"id": "parallel_0", "result": "answer", "output_token_count": 80}], directory, directory))
    monkeypatch.setattr(bfcl, "extract_verdicts", lambda *_: (_ for _ in ()).throw(ValueError("broken checker")))
    assert adapter.teacher_demo(["parallel_0"], 1) == {}
    assert [(r["verified"], r["tokens_spent"]) for r in ledger.read_records("bfcl")] == [(False, 80)]


def test_subprocess_failure_recovers_journal_and_restores_selection(adapter, client, monkeypatch, tmp_path):
    monkeypatch.setattr(bfcl, "BFCL_ROOT", tmp_path)
    selection = tmp_path / "test_case_ids_to_generate.json"
    selection.write_text('"original"\n')
    monkeypatch.setattr(adapter, "_selective_file", lambda ids: {"parallel": list(ids)})

    def generate(args, policy=None, *, usage_log):
        monkeypatch.setenv("BFAS_BFCL_USAGE_LOG", str(usage_log))
        handler().inference(entry(), False, True)
        raise subprocess.CalledProcessError(1, "stub generator")

    monkeypatch.setattr(adapter, "_run_generate", generate)
    assert adapter.teacher_demo(["parallel_0"], 1) == {}
    assert ledger.read_records("bfcl")[0]["tokens_spent"] == 80
    assert selection.read_text() == '"original"\n'


def test_memory_skip_is_unavailable_inventory_and_can_be_reenabled(adapter, monkeypatch):
    monkeypatch.setenv("BFAS_BFCL_SKIP_MEMORY_PREREQ", "1")
    monkeypatch.setattr(adapter, "_official_pass", lambda *_: pytest.fail("skipped memory must never generate"))
    assert adapter.teacher_demo(["memory_kv_0"], 3) == {}
    assert adapter.teacher_demo(["memory_kv_0"], 3) == {}
    assert ledger.read_records("bfcl") == []
    inventory = ledger.LEDGER_ROOT / "bfcl_inventory.jsonl"
    rows = [json.loads(line) for line in inventory.read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["status"] == "unavailable inventory"
    assert rows[0]["task_id"] == "memory_kv_0"
    monkeypatch.delenv("BFAS_BFCL_SKIP_MEMORY_PREREQ")
    assert adapter.teacher_task_ids(["memory_kv_0"]) == ["memory_kv_0"]
    adapter._memory_prereqs = {"memory_kv_0": ["memory_kv_prereq_0"]}
    assert adapter._selective_file(["memory_kv_0"]) == {"memory_kv": ["memory_kv_prereq_0", "memory_kv_0"]}


def test_batch_import_charges_failed_and_prereq_rows_once(adapter, tmp_path):
    results, scores = tmp_path / "results", tmp_path / "scores"
    results.mkdir()
    scores.mkdir()
    rows = [
        {"id": "parallel_0", "result": "good", "output_token_count": 80},
        {"id": "parallel_1", "result": "bad", "output_token_count": 21},
        {"id": "memory_kv_prereq_0", "output_token_count": [[5, 6], [7]]},
    ]
    (results / "BFCL_v4_parallel_result.json").write_text("\n".join(map(json.dumps, rows)) + "\n")
    (scores / "BFCL_v4_parallel_score.json").write_text(
        json.dumps({"total_count": 2, "correct_count": 1}) + '\n{"id":"parallel_1","valid":false}\n')
    for _ in range(2):
        bfcl_teacher.record_attempt(adapter, results, scores, ["parallel_0", "parallel_1"], "stub", 0, 0.0)
    records = ledger.read_records("bfcl")
    assert [row["tokens_spent"] for row in records] == [80, 21, 18]
    assert [row["verified"] for row in records] == [True, False, False]
    assert all("source_id" in row for row in records)
    assert records[0]["demo"]["worked_example"] == "good"


def test_batch_missing_scores_never_admits_pass(adapter, tmp_path):
    result = tmp_path / "results"
    result.mkdir()
    (result / "BFCL_v4_parallel_result.json").write_text(
        json.dumps({"id": "parallel_0", "result": "text", "output_token_count": 80}) + "\n")
    bfcl_teacher.record_attempt(adapter, result, tmp_path / "no-scores", ["parallel_0"], "stub", 0, 0)
    assert ledger.read_records("bfcl")[0]["verified"] is False


def test_batch_memory_prerequisites_are_charged_but_not_scored(adapter, monkeypatch, tmp_path):
    monkeypatch.setattr(adapter, "task_categories", lambda: {
        "memory_kv_0": "memory_kv", "memory_kv_prereq_0": "memory_kv",
    })
    results, scores = tmp_path / "results", tmp_path / "scores"
    results.mkdir()
    scores.mkdir()
    (results / "BFCL_v4_memory_kv_result.json").write_text("\n".join(map(json.dumps, [
        {"id": "memory_kv_prereq_0", "output_token_count": [[5, 6], [7]]},
        {"id": "memory_kv_0", "result": "answer", "output_token_count": [[80, 2], [3]]},
    ])) + "\n")
    (scores / "BFCL_v4_memory_kv_score.json").write_text('{"total_count":1,"correct_count":1}\n')
    bfcl_teacher.record_attempt(adapter, results, scores, ["memory_kv_0"], "stub", 0, 0)
    records = ledger.read_records("bfcl")
    assert [row["tokens_spent"] for row in records] == [18, 85]
    assert [row["verified"] for row in records] == [False, True]


def test_interrupted_batch_resume_charges_each_actual_attempt(client, adapter, monkeypatch, tmp_path):
    directory = tmp_path / "results"
    usage = directory / "bfas_usage.jsonl"
    monkeypatch.setenv("BFAS_BFCL_USAGE_LOG", str(usage))
    h = handler()
    # First response was paid but the worker died before writing a result.
    h.inference(entry(), False, True)
    bfcl_teacher.record_attempt(adapter, directory, tmp_path / "no-scores", ["parallel_0"], "stub", 0, 0)
    result, metadata = h.inference(entry(), False, True)
    (directory / "BFCL_v4_parallel_result.json").write_text(
        json.dumps({"id": "parallel_0", "result": result, **metadata}) + "\n")
    for _ in range(2):
        bfcl_teacher.record_attempt(adapter, directory, tmp_path / "no-scores", ["parallel_0"], "stub", 0, 0)
    records = ledger.read_records("bfcl")
    assert [row["tokens_spent"] for row in records] == [80, 80]
    assert len({row["source_id"] for row in records}) == 2


@pytest.mark.parametrize("teacher", TEACHERS)
def test_shell_collector_stub_client_ledger_resume_and_memory_skip(tmp_path, teacher):
    project = tmp_path / "project"
    harness = project / "envs/bfcl/gorilla/berkeley-function-call-leaderboard"
    harness.mkdir(parents=True)
    (project / "tools").mkdir()
    (project / "configs").mkdir()
    (project / "configs/bfcl_support_split.json").write_text(json.dumps({
        "demand": ["parallel_0", "parallel_1", "memory_kv_0"],
    }))
    original = '{"previous": ["previous_0"]}\n'
    (harness / "test_case_ids_to_generate.json").write_text(original)
    for name in ("bfcl_teacher_demos.sh", "bfcl_teacher_account.py"):
        shutil.copy2(ROOT / "tools" / name, project / "tools" / name)
    # Both interpreters run the real tracked collector/accounting code. Only
    # harness processes and the prompt renderer are replaced with CPU stubs.
    launcher = f'''#!{sys.executable}
import sys, runpy
from pathlib import Path
sys.path[:0] = [{str(ROOT / "src")!r}, {str(HARNESS)!r}]
from bfas.adapters import bfcl
from bfas import ledger
bfcl.BFCL_ROOT = Path({str(harness)!r})
ledger.LEDGER_ROOT = Path({str(project / "data/teacher_ledger")!r})
bfcl.BFCLAdapter.task_categories = lambda self: {{'parallel_0':'parallel', 'parallel_1':'parallel', 'memory_kv_0':'memory_kv'}}
bfcl.BFCLAdapter._messages = lambda self, tid: []
bfcl.BFCLAdapter._functions = lambda self, tid: []
bfcl.BFCLAdapter._render = lambda self, *args: 'stub prompt'
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name='__main__')
'''
    for location in (".venv/bin/python", "envs/bfcl/.venv/bin/python"):
        path = project / location
        path.parent.mkdir(parents=True)
        path.write_text(launcher)
        path.chmod(0o755)
    fake_cli = '''import json, os, sys
from pathlib import Path
from types import SimpleNamespace
from openai.types.chat import ChatCompletion
from bfas import bfcl_ollama
from bfas.adapters.bfcl import BFCL_ROOT
args = sys.argv[1:]
model = args[args.index('--model') + 1]
def option(name):
    return args[args.index(name) + 1]
result_dir = BFCL_ROOT / option('--result-dir')
if args[0] == 'generate':
    selection = json.loads((BFCL_ROOT / 'test_case_ids_to_generate.json').read_text())
    assert selection == {'parallel': ['parallel_0', 'parallel_1']}
    attempt = int(result_dir.name[-1])
    def create(**kwargs):
        assert kwargs['model'] == model.split('/', 1)[1].removesuffix('-FC')
        assert 'tools' in kwargs
        with (BFCL_ROOT / 'requests.jsonl').open('a') as handle:
            handle.write(json.dumps(kwargs) + '\\n')
        return ChatCompletion.model_validate({
            'id': 'stub', 'object': 'chat.completion', 'created': 0, 'model': kwargs['model'],
            'choices': [{'index': 0, 'finish_reason': 'tool_calls', 'message': {
                'role': 'assistant', 'content': None, 'tool_calls': [{
                    'id': 'call_1', 'type': 'function', 'function': {'name': 'lookup', 'arguments': '{}'},
                }],
            }}],
            'usage': {'prompt_tokens': 10, 'completion_tokens': 80, 'total_tokens': 90,
                      'completion_tokens_details': {'reasoning_tokens': 60}},
        })
    bfcl_ollama.OpenAI = lambda **kw: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    os.environ['OLLAMA_API_KEY'] = 'stub-key'
    os.environ['OPENROUTER_API_KEY'] = 'stub-openrouter-key'
    h = bfcl_ollama.OpenAICompatibleHandler(model.split('/', 1)[1].removesuffix('-FC'), float(option('--temperature')), model, True)
    for task_id in selection['parallel']:
        result, metadata = h.inference({'id': task_id, 'function': [{'name': 'lookup', 'description': 'Lookup',
            'parameters': {'type': 'dict', 'properties': {}}}],
            'question': [[{'role': 'user', 'content': 'Look it up'}]]}, False, True)
        result_dir.mkdir(parents=True, exist_ok=True)
        with (result_dir / 'BFCL_v4_parallel_result.json').open('a') as handle:
            handle.write(json.dumps({'id': task_id, 'result': result, **metadata}) + '\\n')
else:
    score_dir = BFCL_ROOT / option('--score-dir')
    score_dir.mkdir(parents=True)
    (score_dir / 'BFCL_v4_parallel_score.json').write_text(
        json.dumps({'total_count': 2, 'correct_count': 1}) + '\\n' + json.dumps({'id':'parallel_1','valid':False}) + '\\n')
'''
    (project / "tools/bfcl_cli.py").write_text(fake_cli)
    env = {**os.environ, "BFAS_BFCL_TEACHER": teacher, "BFAS_BFCL_SKIP_MEMORY_PREREQ": "1"}
    command = ["bash", str(project / "tools/bfcl_teacher_demos.sh")]
    for _ in range(2):
        outcome = subprocess.run(command, env=env, capture_output=True, text=True)
        assert outcome.returncode == 0, outcome.stdout + outcome.stderr
    records = ledger.read_records(project / "data/teacher_ledger/bfcl.jsonl")
    assert len(records) == 6 and sum(row["tokens_spent"] for row in records) == 480
    assert sum(row["verified"] for row in records) == 3
    assert {row["attempt_index"] for row in records} == {0, 1, 2}
    assert {row["teacher"] for row in records} == {teacher}
    assert len((harness / "requests.jsonl").read_text().splitlines()) == 6
    inventory = [json.loads(line) for line in (project / "data/teacher_ledger/bfcl_inventory.jsonl").read_text().splitlines()]
    assert len(inventory) == 1 and inventory[0]["status"] == "unavailable inventory"
    assert json.loads((project / "data/bfcl_demos_ds_verified.json").read_text()) == ["parallel_0"]
    assert (harness / "test_case_ids_to_generate.json").read_text() == original
