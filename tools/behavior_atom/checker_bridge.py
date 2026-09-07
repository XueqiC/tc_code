"""BFCL AST checks in this interpreter or a persistent, isolated BFCL venv.

Set BFCL_VENV_PYTHON to override envs/bfcl/.venv/bin/python. The worker uses
one JSON request/response per line; stdout is reserved for the protocol.
Checker exceptions are errors (valid=null), never negative benchmark verdicts.
Only the standard library is needed to import this module or start the worker.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout, suppress
import hashlib
import inspect
import json
import os
from pathlib import Path
import selectors
import subprocess
import sys
import threading
import time
import weakref
import uuid
import re

ROOT = Path(__file__).resolve().parents[2]
BFCL = ROOT / "envs/bfcl/gorilla/berkeley-function-call-leaderboard"


def _load_direct():
    for path in (ROOT, ROOT / "tools", BFCL):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker
    from tools.bfcl_event_mine_single import language_for
    return ast_checker, language_for


def _content_hash(value):
    # Identical to response.content_hash for these bytes/string/dict inputs,
    # without importing the GPU environment's numpy-dependent response module.
    if not isinstance(value, (str, bytes)):
        value = json.dumps(value, sort_keys=True, ensure_ascii=False,
                           allow_nan=False, separators=(",", ":"))
    return hashlib.sha256(value.encode("utf-8") if isinstance(value, str) else value).hexdigest()


def _checker_version(checker_model, entry="ast_checker.py"):
    hashes = {str(p.relative_to(BFCL)): _content_hash(p.read_bytes())
              for p in sorted((BFCL / "bfcl_eval/eval_checker").rglob("*.py"))}
    return _content_hash({"checker": hashes, "entry": entry,
                          "parser": _content_hash((ROOT / "tools/bfcl_event_mine_single.py").read_bytes()),
                          "checker_model": checker_model})


def _language_name(category):
    # language_for imports BFCL's enum, so mirror its mapping on the wire only.
    if "java" in category and "javascript" not in category:
        return "java"
    if "javascript" in category:
        return "javascript"
    return "python"


def _check_multi_turn(request):
    """Official evaluator in a fresh simulator namespace, including repeated tasks."""
    from copy import deepcopy
    from bfcl_eval.eval_checker.eval_runner import _evaluate_single_multi_turn_entry
    from bfcl_eval.eval_checker.multi_turn_eval import multi_turn_utils
    from bfcl_eval.model_handler.local_inference.qwen_fc import QwenFCHandler
    model = request["checker_model"]
    handler = QwenFCHandler(model, 1., model, True)
    run_model = model + "_rtd_" + uuid.uuid4().hex
    prefix = re.sub(r'[-./:]', '_', run_model) + "_"
    try:
        return _evaluate_single_multi_turn_entry(handler, request["entry"]["id"], request["result"],
            request["truth"], deepcopy(request["entry"]), run_model, request["category"])
    finally:
        for key in list(vars(multi_turn_utils)):
            if key.startswith(prefix) and key.endswith("_instance"):
                delattr(multi_turn_utils, key)


def _check_relevance(request):
    from bfcl_eval.eval_checker.eval_runner import _evaluate_single_relevance_entry
    from bfcl_eval.model_handler.local_inference.qwen_fc import QwenFCHandler
    model = request["checker_model"]
    handler = QwenFCHandler(model, 1., model, True)
    return _evaluate_single_relevance_entry(handler, request["entry"]["id"], request["result"],
                                            request["entry"], model, request["category"])


class CheckerBridgeError(RuntimeError):
    """Unavailable checker, invalid protocol, or an exception inside the checker."""


class _WorkerFailure(CheckerBridgeError):
    """Transport failure that permits one restart and replay of a pure check."""


def _stop_process(proc):
    # EOF is the normal shutdown signal. Reap even an unresponsive worker.
    with suppress(OSError):
        proc.stdin.close()
    try:
        proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    finally:
        proc.stdout.close()


class CheckerBridge:
    """Callable probe checker; check_many also accepts batches of (probe, calls).

    The fallback starts lazily so abstention checks require no BFCL worker.
    Its version is the same checkout hash used by the direct checker; each
    worker response must confirm that hash before its verdict can be used.
    Use close() or a context manager for prompt cleanup; a finalizer also
    closes the worker when the bridge is discarded or the interpreter exits.
    """

    def __init__(self, checker_model="Qwen/Qwen3.5-4B-FC", *, python=None, timeout=120.0):
        self.checker_model = checker_model
        self.python = str(Path(python or os.environ.get(
            "BFCL_VENV_PYTHON", ROOT / "envs/bfcl/.venv/bin/python")).expanduser())
        self.timeout = timeout
        self._proc = self._finalizer = None
        self._buffer = b""
        self._closed = False
        self._lock = threading.RLock()
        try:
            self._checker, self._language_for = _load_direct()
        except ImportError:
            self._checker = self._language_for = None
            self.mode = "subprocess"
            self.checker_version = _checker_version(checker_model)
        else:
            self.mode = "direct"
            self.checker_version = _checker_version(checker_model, Path(inspect.getfile(self._checker)).name)

    def _start(self):
        self._proc = subprocess.Popen(
            [self.python, "-u", str(Path(__file__).resolve()), "--worker"],
            cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            # Inherit stderr: vendor logs cannot fill an unread stderr pipe.
            text=True, encoding="utf-8", bufsize=1,
        )
        self._buffer = b""
        self._finalizer = weakref.finalize(self, _stop_process, self._proc)

    def _stop(self):
        if self._finalizer is not None:
            self._finalizer()
        self._proc = self._finalizer = None
        self._buffer = b""

    def _exchange(self, line):
        if self._proc is None:
            self._start()
        proc = self._proc
        if proc.poll() is not None:
            raise _WorkerFailure(f"worker exited with status {proc.returncode}")
        proc.stdin.write(line)
        proc.stdin.flush()
        deadline = time.monotonic() + self.timeout
        with selectors.DefaultSelector() as selector:
            selector.register(proc.stdout, selectors.EVENT_READ)
            while b"\n" not in self._buffer:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not selector.select(remaining):
                    raise _WorkerFailure(f"worker response timed out after {self.timeout}s")
                chunk = os.read(proc.stdout.fileno(), 65536)
                if not chunk:
                    raise _WorkerFailure(f"worker closed stdout (status={proc.poll()})")
                self._buffer += chunk
        response, self._buffer = self._buffer.split(b"\n", 1)
        try:
            result = json.loads(response)
        except (ValueError, UnicodeError) as exc:
            raise _WorkerFailure("worker returned invalid JSON") from exc
        if not isinstance(result, dict) or not {"valid", "error", "checker_version"} <= result.keys():
            raise _WorkerFailure("worker returned an invalid response envelope")
        return result

    def _request(self, request):
        line = json.dumps(request, ensure_ascii=False, allow_nan=False) + "\n"
        for attempt in range(2):
            try:
                result = self._exchange(line)
                break
            except (OSError, _WorkerFailure) as exc:
                self._stop()
                if attempt:
                    raise CheckerBridgeError(
                        f"BFCL worker failed after one restart ({self.python}); "
                        f"check BFCL_VENV_PYTHON: {exc}"
                    ) from exc
        if result["checker_version"] != self.checker_version:
            raise CheckerBridgeError("BFCL worker checker_version differs from the local checkout")
        if type(result["valid"]) is not bool:
            raise CheckerBridgeError(f"BFCL worker checker error: {result['error']}")
        return result

    def check(self, probe, calls):
        with self._lock:
            if self._closed:
                raise CheckerBridgeError("checker bridge is closed")
            if probe.get("expected") == "abstain":
                error = "unexpected_call" if calls else None
                return {"valid": not calls, "error": error, "error_type": error,
                        "checker_version": self.checker_version}
            if self._checker is not None:
                # Preserve the original AST result, including error_type.
                return self._checker(probe["function"], calls, probe["truth"],
                                     self._language_for(probe["category"]),
                                     probe["category"], self.checker_model)
            return self._request({"function": probe["function"], "calls": calls,
                                  "truth": probe["truth"], "language": _language_name(probe["category"]),
                                  "category": probe["category"], "checker_model": self.checker_model})

    __call__ = check

    def check_multi_turn(self, entry, result, truth, category):
        with self._lock:
            if self._closed:
                raise CheckerBridgeError("checker bridge is closed")
            request = dict(kind="multi_turn", entry=entry, result=result, truth=truth,
                           category=category, checker_model=self.checker_model)
            if self._checker is not None:
                return _check_multi_turn(request)
            return self._request(request)

    def check_relevance(self, entry, result, category):
        with self._lock:
            if self._closed:
                raise CheckerBridgeError("checker bridge is closed")
            request = dict(kind="relevance", entry=entry, result=result, category=category,
                           checker_model=self.checker_model)
            if self._checker is not None:
                return _check_relevance(request)
            return self._request(request)

    def check_many(self, probes_and_calls):
        return [self.check(probe, calls) for probe, calls in probes_and_calls]

    def close(self):
        with self._lock:
            self._stop()
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _diagnostic_repr(value):
    """JSON fallback for live BFCL state; never coerce JSON-native verdict values.

    json.dumps applies this recursively inside dicts/lists. Simulator objects
    such as Directory already have useful reprs; remove process-specific object
    addresses and sort unordered sets so their diagnostic strings are stable.
    """
    try:
        if isinstance(value, (set, frozenset)):
            return type(value).__name__ + "(" + ", ".join(sorted(
                _diagnostic_repr(item) for item in value)) + ")"
        text = repr(value)
    except Exception:
        text = f"<{type(value).__module__}.{type(value).__qualname__}: unrepresentable>"
    return re.sub(r" at 0x[0-9a-fA-F]+(?=>)", "", text)


def worker_main():
    protocol_out = sys.stdout
    with redirect_stdout(sys.stderr):
        checker, _ = _load_direct()
        from bfcl_eval.constants.enums import Language
    versions = {}
    for line in sys.stdin:
        version = None
        try:
            request = json.loads(line)
            model = request["checker_model"]
            with redirect_stdout(sys.stderr):
                if model not in versions:
                    versions[model] = _checker_version(model, Path(inspect.getfile(checker)).name)
                version = versions[model]
                if request.get("kind") == "multi_turn":
                    verdict = _check_multi_turn(request)
                elif request.get("kind") == "relevance":
                    verdict = _check_relevance(request)
                else:
                    verdict = checker(request["function"], request["calls"], request["truth"],
                                      Language(request["language"]), request["category"], model)
                if not isinstance(verdict, dict) or type(verdict.get("valid")) is not bool:
                    raise ValueError("checker must return boolean valid")
            response = {**verdict, "error": verdict.get("error"), "checker_version": version}
        except Exception as exc:
            response = {"valid": None, "error": f"{type(exc).__name__}: {exc}", "checker_version": version}
        with redirect_stdout(sys.stderr):
            encoded = json.dumps(response, ensure_ascii=False, default=_diagnostic_repr)
        protocol_out.write(encoded + "\n")
        protocol_out.flush()
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", required=True)
    parser.parse_args()
    raise SystemExit(worker_main())
