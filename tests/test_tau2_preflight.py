"""Student capability checks use fake HTTP responses and never contact a server."""
from __future__ import annotations

import io
import json
import socket
import sys
import urllib.error
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bfas.tau2_eval import probe_student_tool_choice


@pytest.fixture(autouse=True)
def forbid_network(monkeypatch):
    monkeypatch.setattr(socket.socket, "connect", lambda *a: pytest.fail("network forbidden"))


@pytest.mark.parametrize("student_key", [None, "student-secret"])
def test_probe_sends_one_tiny_auto_tools_request(monkeypatch, student_key):
    monkeypatch.setenv("OPENAI_API_KEY", "never-send-official-secret")
    if student_key is None:
        monkeypatch.delenv("BFAS_STUDENT_API_KEY", raising=False)
    else:
        monkeypatch.setenv("BFAS_STUDENT_API_KEY", student_key)
    calls = []
    def urlopen(request, timeout):
        calls.append(request)
        assert request.full_url == "http://student.test:8950/v1/chat/completions"
        assert request.get_method() == "POST" and timeout == 30
        assert request.get_header("Authorization") == "Bearer " + (student_key or "EMPTY")
        payload = json.loads(request.data)
        assert payload["model"] == "gemma4-12b-base"
        assert payload["tool_choice"] == "auto"
        assert payload["max_tokens"] == 1 and payload["stream"] is False
        assert payload["tools"][0]["function"]["name"] == "bfas_tool_probe"
        assert "never-send-official-secret" not in str(request.headers)
        # A truncated response still proves the server accepts auto + tools.
        return io.BytesIO(json.dumps({"choices": [{"message": {"content": ""},
                                                   "finish_reason": "length"}]}).encode())
    monkeypatch.setattr("bfas.tau2_eval.urllib.request.urlopen", urlopen)
    probe_student_tool_choice("http://student.test:8950/v1/", "openai/gemma4-12b-base")
    assert len(calls) == 1


def test_missing_auto_tool_choice_reports_server_flags_without_retry(monkeypatch):
    calls = []
    def urlopen(request, timeout):
        calls.append(request)
        raise urllib.error.HTTPError(request.full_url, 400, "Bad Request", {}, io.BytesIO(
            b'{"error":{"message":"auto tool choice requires --enable-auto-tool-choice '
            b'and --tool-call-parser"}}'))
    monkeypatch.setattr("bfas.tau2_eval.urllib.request.urlopen", urlopen)
    with pytest.raises(RuntimeError, match="--enable-auto-tool-choice --tool-call-parser gemma4") as error:
        probe_student_tool_choice("http://student.test/v1", "gemma4-12b-base")
    assert "HTTP Error 400" in str(error.value)
    assert "auto tool choice requires" in str(error.value)
    assert "No tau2 tasks or paid calls" in str(error.value)
    assert len(calls) == 1


@pytest.mark.parametrize("failure", ["timeout", "invalid_json", "error_body", "no_choices"])
def test_unusable_endpoint_fails_preflight(monkeypatch, failure):
    def urlopen(request, timeout):
        if failure == "timeout":
            raise TimeoutError("student timed out")
        return io.BytesIO({"invalid_json": b"not json", "error_body": b'{"error":"unsupported"}',
                           "no_choices": b'{"choices":[]}' }[failure])
    monkeypatch.setattr("bfas.tau2_eval.urllib.request.urlopen", urlopen)
    with pytest.raises(RuntimeError, match="Student tool-choice preflight failed"):
        probe_student_tool_choice("http://student.test/v1", "gemma4-12b-base")
