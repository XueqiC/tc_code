"""CPU-only teacher resolution, HTTP transport and ledger accounting tests."""

from datetime import datetime, timezone
from email.utils import format_datetime
import argparse
import json
from pathlib import Path
import socket
import sys
from types import SimpleNamespace
import urllib.error

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import appworld_teacher as teacher
from bfas import ledger
from bfas.adapters.alfworld import ALFWorldAdapter
from bfas.adapters import alfworld
from bfas.adapters._teacher import TeacherSession


NAMES = [
    "openrouter/openai/gpt-5.4",
    "openrouter/anthropic/claude-sonnet-5",
    "openrouter/google/gemini-3.1-pro-preview",
    "openrouter/openai/gpt-5.6-luna",
    "openrouter/vendor/another/model:free",
]


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        pytest.fail("teacher tests must never access the network")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(teacher.urllib.request, "build_opener", forbidden)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(ledger, "LEDGER_ROOT", tmp_path / "ledger")
    for name in ("OPENROUTER_BASE_URL", "OPENROUTER_API_KEY", "BFAS_TEACHER",
                 "OLLAMA_BASE_URL", "OLLAMA_API_KEY", "AZURE_LLM_ENDPOINT",
                 "AZURE_LLM_KEY", "AZURE_USAGE_LOG", "TEACHER_TEMP", "TEACHER_THINK"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("BFAS_TEACHER_MIN_INTERVAL_S", "0")


def config(monkeypatch, name=NAMES[0]):
    monkeypatch.setenv("OPENROUTER_API_KEY", "router-test-key")
    return teacher.load_teacher_config(name)


def payload(content="ACTION: look", usage=None):
    result = {"choices": [{"message": {"content": content}}]}
    if usage is not None:
        result["usage"] = usage
    return result


def stub_client(monkeypatch, responses):
    pending = iter(responses)
    calls = []

    class Response:
        def __init__(self, value):
            self.value = value

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self):
            return json.dumps(self.value).encode()

    def open_request(request, timeout):
        calls.append((request, timeout))
        value = next(pending)
        if isinstance(value, BaseException):
            raise value
        return Response(value)

    monkeypatch.setattr(teacher.urllib.request, "build_opener",
                        lambda: SimpleNamespace(open=open_request))
    return calls


def http_error(status, retry_after=None):
    headers = {} if retry_after is None else {"Retry-After": retry_after}
    return urllib.error.HTTPError("https://stub.invalid", status, "stub", headers, None)


@pytest.mark.parametrize("name", NAMES)
def test_openrouter_name_resolution_and_cli(monkeypatch, name):
    resolved = config(monkeypatch, name)
    assert resolved.name == name
    assert resolved.model == name.removeprefix("openrouter/")
    assert resolved.backend == "openrouter"
    assert resolved.endpoint == "https://openrouter.ai/api/v1/chat/completions"
    assert resolved.api_key == "router-test-key"
    assert "router-test-key" not in repr(resolved)
    assert teacher.parse_args(["--teacher", name, "--tag", "cpu-test"]).teacher == name


@pytest.mark.parametrize("key", [None, "", "  "])
def test_openrouter_requires_own_key(monkeypatch, key):
    monkeypatch.setenv("OLLAMA_API_KEY", "wrong-ollama-key")
    monkeypatch.setenv("AZURE_LLM_KEY", "wrong-azure-key")
    if key is not None:
        monkeypatch.setenv("OPENROUTER_API_KEY", key)
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY is not set"):
        teacher.load_teacher_config(NAMES[0])


def test_openrouter_base_url_override(monkeypatch):
    monkeypatch.setenv("OPENROUTER_BASE_URL", " https://router.example/custom/v1/ ")
    assert config(monkeypatch).endpoint == "https://router.example/custom/v1/chat/completions"


@pytest.mark.parametrize("base", ["", "/v1", "ftp://example/v1", "https://", "https://example/v1?x=1"])
def test_openrouter_rejects_invalid_base(monkeypatch, base):
    monkeypatch.setenv("OPENROUTER_BASE_URL", base)
    with pytest.raises(RuntimeError, match="OPENROUTER_BASE_URL"):
        config(monkeypatch)


@pytest.mark.parametrize("name", ["openrouter/", "openrouter/openai", "openrouter//model",
                                  "openrouter/openai/", "openrouter/openai/bad model"])
def test_openrouter_rejects_malformed_names(name):
    with pytest.raises(argparse.ArgumentTypeError, match="openrouter/<vendor>/<model>"):
        teacher.load_teacher_config(name)
    with pytest.raises(SystemExit):
        teacher.parse_args(["--teacher", name, "--tag", "cpu-test"])


@pytest.mark.parametrize("name", ["gpt-5.4", "gpt-5.6-luna", "deepseek-v4-pro", "gpt-oss:120b", "mistral-large-3"])
def test_existing_provider_resolution(monkeypatch, name):
    monkeypatch.setenv("AZURE_LLM_ENDPOINT", "https://azure.example")
    monkeypatch.setenv("AZURE_LLM_KEY", "azure-test-key")
    monkeypatch.setenv("OLLAMA_BASE_URL", "https://ollama.example")
    monkeypatch.setenv("OLLAMA_API_KEY", "ollama-test-key")
    resolved = teacher.load_teacher_config(name)
    assert resolved.name == resolved.model == name
    if name in teacher.AZURE_OPENAI_MODELS:
        assert resolved.backend == "azure_openai"
        assert resolved.api_key == "azure-test-key"
    else:
        assert resolved.backend == "openai"
        assert resolved.endpoint == "https://ollama.example/v1/chat/completions"


@pytest.mark.parametrize("name", NAMES[:4])
@pytest.mark.parametrize("temperature", [0.0, 0.7])
def test_openrouter_request_and_raw_usage(monkeypatch, tmp_path, name, temperature):
    resolved = config(monkeypatch, name)
    (tmp_path / ".ollama_api_key2").write_text("wrong-alternate-key")
    monkeypatch.setattr(teacher, "_OLLAMA_KEY_IDX", 1)
    monkeypatch.setenv("TEACHER_THINK", "1")
    usage = {"completion_tokens": 83, "prompt_tokens": 201, "total_tokens": 284,
             "completion_tokens_details": {"reasoning_tokens": 70}}
    calls = stub_client(monkeypatch, [payload(usage=usage)])
    usages = []
    messages = [{"role": "user", "content": "look"}]
    assert teacher.generate_reply(resolved, messages, temperature, usage_callback=usages.append) == "ACTION: look"
    assert usages == [usage]
    request, timeout = calls[0]
    assert request.full_url == resolved.endpoint
    assert request.method == "POST"
    assert request.get_header("Authorization") == "Bearer router-test-key"
    assert timeout == teacher.CHAT_COMPLETION_TIMEOUT_SECONDS
    body = json.loads(request.data)
    assert body["model"] == name.removeprefix("openrouter/")
    assert body["messages"] == messages
    assert body["max_tokens"] == teacher.MAX_COMPLETION_TOKENS
    assert body["stream"] is False
    assert "think" not in body
    if name.startswith("openrouter/openai/gpt-5"):
        assert "temperature" not in body
    else:
        assert body["temperature"] == temperature


@pytest.mark.parametrize("status", [429, 503])
@pytest.mark.parametrize("retry_after,delay", [("12", 12), ("-1", 0), ("invalid", None), ("nan", None), ("date", 17)])
def test_retry_after_is_honoured(monkeypatch, status, retry_after, delay):
    resolved = config(monkeypatch)
    sleeps = []
    monkeypatch.setattr(teacher.time, "sleep", sleeps.append)
    monkeypatch.setattr(teacher.time, "time", lambda: 1000)
    if retry_after == "date":
        retry_after = format_datetime(datetime.fromtimestamp(1017, timezone.utc), usegmt=True)
    monkeypatch.setattr(teacher, "_OLLAMA_KEY_IDX", 7)
    calls = stub_client(monkeypatch, [http_error(status, retry_after), payload()])
    assert teacher.generate_reply(resolved, []) == "ACTION: look"
    assert len(calls) == 2
    assert sleeps == [delay if delay is not None else (5 if status == 429 else 1)]
    assert teacher._OLLAMA_KEY_IDX == 7


@pytest.mark.parametrize("failure,attempts", [
    (429, teacher.RATE_LIMIT_RETRIES + 1),
    (503, teacher.CHAT_COMPLETION_RETRIES + 1),
    (401, 1), (400, 1), ("timeout", teacher.CHAT_COMPLETION_RETRIES + 1),
])
def test_retries_are_bounded_and_usage_not_invented(monkeypatch, failure, attempts):
    resolved = config(monkeypatch)
    monkeypatch.setattr(teacher.time, "sleep", lambda delay: None)
    responses = [TimeoutError() if failure == "timeout" else http_error(failure) for _ in range(attempts)]
    calls = stub_client(monkeypatch, responses)
    usages = []
    with pytest.raises(teacher.TeacherAPIError):
        teacher.generate_reply(resolved, [], usage_callback=usages.append)
    assert len(calls) == attempts
    assert usages == []


@pytest.mark.parametrize("usage,expected,recorded", [
    ({"completion_tokens": 0, "prompt_tokens": 12, "total_tokens": 12}, 0,
     {"completion_tokens": 0, "prompt_tokens": 12}),
    ({"completion_tokens": 83, "prompt_tokens": 201, "total_tokens": 284,
      "completion_tokens_details": {"reasoning_tokens": 70}}, 83,
     {"completion_tokens": 83, "prompt_tokens": 201}),
    ({"completion_tokens": 5}, 5, {"completion_tokens": 5}),
    ({"output_tokens": 8, "input_tokens": 21}, 8, {}),
    ({"prompt_tokens": 21}, 3, {"prompt_tokens": 21}),
    (None, 3, {}),
])
def test_session_records_exact_usage_or_estimates_only_missing_output(monkeypatch, usage, expected, recorded):
    session = TeacherSession(config(monkeypatch))
    stub_client(monkeypatch, [payload(usage=usage), payload(usage=usage)])
    session.generate_reply([], 0.0)
    session.generate_reply([], 0.0)
    assert session.tokens_spent == 2 * expected
    assert session.usage == {key: 2 * value for key, value in recorded.items()}
    assert session.response_texts == ["ACTION: look"] * 2


@pytest.fixture
def alfworld_adapter(monkeypatch):
    class Bridge:
        def __init__(self, *args):
            self.steps = 0

        def _read(self):
            return {"observation": "Your task is to: find an apple", "admissible": ["look"]}

        def step(self, action):
            self.steps += 1
            return dict(self._read(), done=self.steps == 2, won=True)

        def close(self):
            pass

    monkeypatch.setattr(alfworld, "_EnvBridge", Bridge)
    adapter = ALFWorldAdapter()
    adapter._tokenizer = object()
    monkeypatch.setattr(adapter, "_render", lambda messages: json.dumps(messages))
    return adapter


@pytest.mark.parametrize("name", NAMES[:4] + ["gpt-5.6-luna", "gpt-oss:120b", "mistral-large-3", None])
def test_alfworld_ledger_records_resolved_teacher_and_usage(monkeypatch, alfworld_adapter, name):
    config(monkeypatch)
    monkeypatch.setenv("AZURE_LLM_ENDPOINT", "https://azure.example")
    monkeypatch.setenv("AZURE_LLM_KEY", "azure-test-key")
    monkeypatch.setenv("OLLAMA_BASE_URL", "https://ollama.example")
    monkeypatch.setenv("OLLAMA_API_KEY", "ollama-test-key")
    if name is not None:
        monkeypatch.setenv("BFAS_TEACHER", name)
    calls = stub_client(monkeypatch, [payload(usage={"completion_tokens": 83, "prompt_tokens": 201}),
                                      payload(usage={"completion_tokens": 7, "prompt_tokens": 11})])
    task = "pick_and_place_simple-Apple-None-DiningTable-1/trial"
    assert task in ledger.acquire_demos("alfworld", alfworld_adapter, [task], 1)
    row, = ledger.read_records("alfworld")
    assert row["teacher"] == (name or "deepseek-v4-pro")
    assert row["tokens_spent"] == 90
    assert row["usage"] == {"completion_tokens": 90, "prompt_tokens": 212}
    assert len(calls) == 2


@pytest.mark.parametrize("bad_reply", ["http", "invalid_choices"])
def test_alfworld_paid_failure_retains_resolved_label_and_usage(monkeypatch, alfworld_adapter, bad_reply):
    resolved = config(monkeypatch)
    monkeypatch.setattr(teacher, "load_teacher_config", lambda name: resolved)
    second = http_error(400) if bad_reply == "http" else {"usage": {"completion_tokens": 5, "prompt_tokens": 12}}
    stub_client(monkeypatch, [payload(usage={"completion_tokens": 83, "prompt_tokens": 201}), second])
    task = "pick_and_place_simple-Apple-None-DiningTable-1/trial"
    assert ledger.acquire_demos("alfworld", alfworld_adapter, [task], 1) == {}
    row, = ledger.read_records("alfworld")
    assert row["teacher"] == resolved.name  # differs from the default environment name
    assert row["tokens_spent"] == (83 if bad_reply == "http" else 88)
    assert row["usage"]["prompt_tokens"] == (201 if bad_reply == "http" else 213)
    assert row["verified"] is False


def test_transport_retries_do_not_multiply_ledger_attempts(monkeypatch, alfworld_adapter):
    config(monkeypatch)
    monkeypatch.setenv("BFAS_TEACHER", NAMES[0])
    monkeypatch.setattr(teacher.time, "sleep", lambda delay: None)
    calls = stub_client(monkeypatch, [http_error(429, "0"), http_error(503, "0"),
                                      payload(usage={"completion_tokens": 7}),
                                      payload(usage={"completion_tokens": 8})])
    task = "pick_and_place_simple-Apple-None-DiningTable-1/trial"
    assert task in ledger.acquire_demos("alfworld", alfworld_adapter, [task], 3)
    row, = ledger.read_records("alfworld")
    assert row["attempt_index"] == 0
    assert row["tokens_spent"] == row["usage"]["completion_tokens"] == 15
    assert len(calls) == 4


def test_initial_quota_failure_does_not_consume_ledger_attempt(monkeypatch, alfworld_adapter):
    config(monkeypatch)
    monkeypatch.setenv("BFAS_TEACHER", NAMES[0])
    monkeypatch.setattr(teacher.time, "sleep", lambda delay: None)
    stub_client(monkeypatch, [http_error(429, "0") for _ in range(teacher.RATE_LIMIT_RETRIES + 1)])
    task = "pick_and_place_simple-Apple-None-DiningTable-1/trial"
    with pytest.raises(teacher.TeacherAPIError, match="HTTP 429"):
        ledger.acquire_demos("alfworld", alfworld_adapter, [task], 3)
    assert ledger.read_records("alfworld") == []


def test_alfworld_empty_reply_retry_preserves_all_paid_usage(monkeypatch, alfworld_adapter):
    config(monkeypatch)
    monkeypatch.setenv("BFAS_TEACHER", NAMES[0])
    calls = stub_client(monkeypatch, [payload("", {"completion_tokens": 83, "prompt_tokens": 201}),
                                      payload(usage={"completion_tokens": 7, "prompt_tokens": 11}),
                                      payload(usage={"completion_tokens": 8, "prompt_tokens": 12})])
    task = "pick_and_place_simple-Apple-None-DiningTable-1/trial"
    assert task in ledger.acquire_demos("alfworld", alfworld_adapter, [task], 1)
    row, = ledger.read_records("alfworld")
    assert row["tokens_spent"] == 98
    assert row["usage"] == {"completion_tokens": 98, "prompt_tokens": 224}
    assert len(calls) == 3
