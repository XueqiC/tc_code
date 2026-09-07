"""CPU-only BFCL bridge protocol, recovery, and GPU-driver wiring tests."""
from __future__ import annotations

import builtins
from enum import Enum
import io
import json
from pathlib import Path
import sys
import textwrap
from types import ModuleType, SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.behavior_atom import checker_bridge as bridge
from tools.behavior_atom import gpu_driver as driver
from tools.bfcl_event_mine_single import parsed_calls
from test_gpu_driver import FakeStudent, fixture_files, cpu_threads  # reuse tiny CPU pipeline/fixture

MODEL = "Qwen/Qwen3.5-4B-FC"
CALLS = [{"lookup": {"city": "Montréal\nQC"}}]
PROBE = {"function": [{"name": "lookup", "description": "City lookup\nUnicode: é"}],
         "truth": [{"lookup": {"city": ["Montréal\nQC"]}}],
         "category": "simple_python", "expected": "call"}


@pytest.fixture
def force_subprocess(monkeypatch):
    def unavailable():
        raise ImportError("missing vendor SDK")
    monkeypatch.setattr(bridge, "_load_direct", unavailable)


@pytest.fixture
def fake_worker(tmp_path, monkeypatch):
    """Executable stand-in for BFCL_VENV_PYTHON, with real pipes/processes."""
    python = tmp_path / "fake worker python"
    starts, requests, crashed = [tmp_path / name for name in ("starts.jsonl", "requests.jsonl", "crashed")]
    python.write_text(f"#!{sys.executable}\n" + textwrap.dedent(f"""\
        import json
        import os
        from pathlib import Path
        import sys
        import time

        assert sys.argv[1:] == ["-u", {str(Path(bridge.__file__).resolve())!r}, "--worker"]
        with open({str(starts)!r}, "a") as stream:
            stream.write(json.dumps(os.getpid()) + "\\n")
        behavior = os.environ.get("FAKE_CHECKER_BEHAVIOR", "normal")
        for line in sys.stdin:
            request = json.loads(line)
            with open({str(requests)!r}, "a") as stream:
                stream.write(json.dumps({{"pid": os.getpid(), "request": request}}) + "\\n")
            if behavior == "always_crash" or (behavior == "crash_once" and not Path({str(crashed)!r}).exists()):
                Path({str(crashed)!r}).touch()
                os._exit(17)
            if behavior == "partial_line":
                sys.stdout.write("{{")
                sys.stdout.flush()
                time.sleep(30)
            valid = request["calls"] == {CALLS!r}
            response = {{"valid": valid, "error": None if valid else "incorrect_call",
                        "checker_version": {bridge._checker_version(MODEL)!r}}}
            if behavior == "checker_error":
                response.update(valid=None, error="ValueError: bad checker input")
            if behavior == "bad_version":
                response["checker_version"] = "different-checkout"
            print(json.dumps(response), flush=True)
        """))
    python.chmod(0o755)
    monkeypatch.setenv("BFCL_VENV_PYTHON", str(python))
    monkeypatch.delenv("FAKE_CHECKER_BEHAVIOR", raising=False)
    return SimpleNamespace(python=python, starts=starts, requests=requests)


def _json_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_protocol_batch_round_trip_reuses_worker_and_closes(fake_worker, force_subprocess):
    probes = [PROBE, {**PROBE, "category": "simple_java"}, {**PROBE, "category": "simple_javascript"}]
    with bridge.CheckerBridge() as checker:
        assert checker.mode == "subprocess"
        assert checker._proc is None
        verdicts = checker.check_many(zip(probes, [CALLS, [], CALLS]))
        proc = checker._proc
        assert [v["valid"] for v in verdicts] == [True, False, True]
        assert all(v["checker_version"] == checker.checker_version for v in verdicts)
        assert verdicts[1]["error"] == "incorrect_call"
        assert proc.poll() is None
    assert proc.poll() == 0
    checker.close()  # idempotent
    requests = _json_lines(fake_worker.requests)
    assert len(_json_lines(fake_worker.starts)) == 1
    assert {r["pid"] for r in requests} == {proc.pid}
    assert [r["request"]["language"] for r in requests] == ["python", "java", "javascript"]
    assert requests[0]["request"] == {k: PROBE[k] for k in ("function", "truth", "category")} | {
        "calls": CALLS, "language": "python", "checker_model": MODEL}
    with pytest.raises(bridge.CheckerBridgeError, match="closed"):
        checker.check(PROBE, CALLS)


def test_restart_once_replays_request_then_keeps_worker(fake_worker, force_subprocess, monkeypatch):
    monkeypatch.setenv("FAKE_CHECKER_BEHAVIOR", "crash_once")
    with bridge.CheckerBridge() as checker:
        assert checker.check(PROBE, CALLS)["valid"] is True
        proc = checker._proc
        assert checker.check(PROBE, CALLS)["valid"] is True
        assert checker._proc is proc
    starts, requests = _json_lines(fake_worker.starts), _json_lines(fake_worker.requests)
    assert len(starts) == 2
    assert [r["pid"] for r in requests] == [starts[0], starts[1], starts[1]]
    assert requests[0]["request"] == requests[1]["request"]
    assert proc.poll() == 0


@pytest.mark.parametrize("behavior", ["always_crash", "partial_line"])
def test_recovery_is_bounded(fake_worker, force_subprocess, monkeypatch, behavior):
    monkeypatch.setenv("FAKE_CHECKER_BEHAVIOR", behavior)
    with bridge.CheckerBridge(timeout=0.25) as checker:
        with pytest.raises(bridge.CheckerBridgeError, match="after one restart"):
            checker.check(PROBE, CALLS)
        assert checker._proc is None
    assert len(_json_lines(fake_worker.starts)) == 2
    assert len(_json_lines(fake_worker.requests)) == 2


@pytest.mark.parametrize("behavior,match", [
    ("checker_error", "ValueError: bad checker input"), ("bad_version", "checker_version differs"),
])
def test_checker_errors_are_not_negative_verdicts_or_retried(fake_worker, force_subprocess, monkeypatch, behavior, match):
    monkeypatch.setenv("FAKE_CHECKER_BEHAVIOR", behavior)
    with bridge.CheckerBridge() as checker:
        with pytest.raises(bridge.CheckerBridgeError, match=match):
            checker.check(PROBE, CALLS)
    assert len(_json_lines(fake_worker.starts)) == 1


@pytest.mark.parametrize("text,valid", [
    ("No suitable tool.", True),
    ('<tool_call>{broken json}</tool_call>', True),
    ('<tool_call>{"name":"lookup","arguments":{}}</tool_call>', False),
])
def test_abstention_needs_no_worker_or_probe_truth(fake_worker, force_subprocess, monkeypatch, text, valid):
    monkeypatch.setattr(bridge.subprocess, "Popen", lambda *a, **kw: pytest.fail("abstention started worker"))
    with bridge.CheckerBridge() as checker:
        verdict = checker.check({"expected": "abstain"}, parsed_calls(text))
        assert verdict["valid"] is valid
        assert verdict["error_type"] == (None if valid else "unexpected_call")
    assert not fake_worker.starts.exists()


def test_direct_path_preserves_arguments_verdict_and_version(monkeypatch):
    received = []
    verdict = {"valid": False, "error": ["bad arguments"], "error_type": "ast:type_error"}
    language = object()
    def ast_checker(*args):
        received.append(args)
        return verdict
    monkeypatch.setattr(bridge, "_load_direct", lambda: (ast_checker, lambda category: language))
    monkeypatch.setattr(bridge.subprocess, "Popen", lambda *a, **kw: pytest.fail("direct path started worker"))
    with bridge.CheckerBridge() as checker:
        assert checker.mode == "direct"
        assert checker.check(PROBE, CALLS) is verdict
        assert received == [(PROBE["function"], CALLS, PROBE["truth"], language, PROBE["category"], MODEL)]
        hashes = {str(p.relative_to(bridge.BFCL)): driver.content_hash(p.read_bytes())
                  for p in sorted((bridge.BFCL / "bfcl_eval/eval_checker").rglob("*.py"))}
        assert checker.checker_version == driver.content_hash({
            "checker": hashes, "entry": Path(__file__).name,
            "parser": driver.content_hash((ROOT / "tools/bfcl_event_mine_single.py").read_bytes()),
            "checker_model": MODEL})


def test_worker_protocol_redirects_logs_and_survives_checker_errors(monkeypatch, capsys):
    class Language(Enum):
        PYTHON = "python"
        JAVA = "java"
        JAVASCRIPT = "javascript"
    enums = ModuleType("bfcl_eval.constants.enums")
    enums.Language = Language
    monkeypatch.setitem(sys.modules, enums.__name__, enums)
    def ast_checker(function, calls, truth, language, category, model):
        print("checker log")
        assert language is Language.JAVA
        assert (function, truth, category, model) == (PROBE["function"], PROBE["truth"], "simple_java", MODEL)
        if not calls:
            raise ValueError("injected checker failure")
        return {"valid": True}
    def load():
        print("import log")
        return ast_checker, None
    monkeypatch.setattr(bridge, "_load_direct", load)
    monkeypatch.setattr(bridge, "_checker_version", lambda *args: "worker-version")
    request = {"function": PROBE["function"], "truth": PROBE["truth"], "calls": [],
               "language": "java", "category": "simple_java", "checker_model": MODEL}
    monkeypatch.setattr(sys, "stdin", io.StringIO("not json\n" + json.dumps(request) + "\n" +
                                                json.dumps({**request, "calls": CALLS}) + "\n"))
    output = io.StringIO()
    monkeypatch.setattr(sys, "stdout", output)
    assert bridge.worker_main() == 0
    responses = [json.loads(line) for line in output.getvalue().splitlines()]
    assert len(responses) == 3
    assert responses[0]["valid"] is None
    assert responses[1] == {"valid": None, "error": "ValueError: injected checker failure",
                            "checker_version": "worker-version"}
    assert responses[2] == {"valid": True, "error": None, "checker_version": "worker-version"}
    logs = capsys.readouterr().err
    assert "import log" in logs and logs.count("checker log") == 2


def test_load_student_falls_back_on_direct_import_error(fake_worker, monkeypatch):
    import appworld_train as trainer
    original_import = builtins.__import__
    attempts = []
    def failing_import(name, *args, **kwargs):
        if name == "bfcl_eval.eval_checker.ast_eval.ast_checker":
            attempts.append(name)
            raise ImportError("No module named 'anthropic'")
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", failing_import)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    model = SimpleNamespace(to=lambda device: None, config=SimpleNamespace(use_cache=True),
                            gradient_checkpointing_enable=lambda **kwargs: None,
                            enable_input_require_grads=lambda: None)
    tokenizer = object()
    monkeypatch.setattr(trainer, "build_lora_model", lambda *args: model)
    monkeypatch.setattr(trainer, "load_tokenizer", lambda *args: tokenizer)
    student = driver.load_student(driver.frozen_settings({}))
    try:
        assert len(attempts) == 1
        assert student.model is model and student.tokenizer is tokenizer
        assert student.checker_mode == "subprocess"
        assert student.parse('<tool_call>{"name":"f"}</tool_call>') == [{"f": {}}]
        verdict = student.check(PROBE, CALLS)
        assert verdict["valid"] is True
        assert verdict["checker_version"] == student.checker_version
        proc = student.check.__self__._proc
    finally:
        student.close()
    assert proc.poll() == 0


@pytest.mark.parametrize("mode", ["direct", "subprocess"])
def test_driver_records_mode_and_closes_checker_on_run_and_resume(tmp_path, fake_worker, monkeypatch, mode):
    def ast_checker(*args):
        return {"valid": True}
    def load():
        if mode == "subprocess":
            raise ImportError("missing SDK")
        return ast_checker, lambda category: "python"
    monkeypatch.setattr(bridge, "_load_direct", load)
    config, protocol, data, _ = fixture_files(tmp_path)
    created = []
    def factory(_):
        student = FakeStudent()
        checker = bridge.CheckerBridge()
        created.append(checker)
        student.check, student.checker_version = checker.check, checker.checker_version
        student.checker_mode, student._close_checker = checker.mode, checker.close
        return student
    out = tmp_path / "pilot"
    result = driver.run("pilot", config, protocol, data, out, student_factory=factory)
    assert result["status"] == "complete"
    assert json.loads((out / "manifest.json").read_text())["checker_mode"] == mode
    assert created[0]._closed and created[0]._proc is None
    if mode == "subprocess":
        assert len(_json_lines(fake_worker.starts)) == 1
    else:
        assert not fake_worker.starts.exists()
    resumed = driver.run("pilot", config, protocol, data, out, resume=True,
                         student_factory=lambda _: pytest.fail("completed resume loaded student"))
    assert resumed == result


def test_driver_closes_worker_when_run_fails(tmp_path, fake_worker, force_subprocess, monkeypatch):
    checker = bridge.CheckerBridge()
    assert checker.check(PROBE, CALLS)["valid"]
    proc = checker._proc
    student = FakeStudent()
    student._close_checker = checker.close
    def fail(*args, student_factory, **kwargs):
        student_factory({})
        raise RuntimeError("injected driver failure")
    monkeypatch.setattr(driver, "_run", fail)
    with pytest.raises(RuntimeError, match="injected driver failure"):
        driver.run("pilot", {}, {}, {}, tmp_path, student_factory=lambda _: student)
    assert checker._closed and proc.poll() == 0
