#!/usr/bin/env python3
"""CRCD intervention-event mining for AppWorld (no teacher API, zero new teacher tokens).

AppWorld analogue of tools/alf_event_mine.py.  For every archived, verified teacher demo
(results/bfas/appworld/collect_shared/demos.json, gpt-5.4 under the OFFICIAL
simplified_react_code_agent scaffold) the demo's code blocks are replayed in the official
AppWorld environment as the agent's own actions.  At a probed teacher decision state s_t
the base STUDENT (served by vllm, chat-completions, the same request the official agent
makes) is sampled; when its executed code differs from the teacher's (``code_ast`` or
``api_effects`` equality, unchanged by selection) the state is an intervention event

    e = (s_t, y^S, y^T, O)      y^S = student reply, y^T = teacher demo reply

and the matched branch outcomes O are estimated with K student continuations after y^S and
K after y^T (student continues in both branches, continuation j of both branches shares the
sampler seed).  Utility of one continuation = official evaluator ``success`` (task-goal
completion, 0/1); the per-continuation test pass fraction is recorded alongside.

    Q^S = (sum_j U_j^S + 1) / (K + 2),   Q^T likewise,   dU = Q^T - Q^S   (derived, never a filter)

Probe selection: ``--probe-select first`` (default) scans for the first ``--max-probes``
divergences, preserving the original behaviour.  ``spread`` selects evenly spaced demo
decision steps with a slot reserved for the final state; ``late`` selects the last
``--max-probes`` states.  ``--probe-positions '0.25,0.5,0.75,1.0'`` overrides either policy:
fractions in [0, 1] map to ceil(fraction * demo_length) - 1, clamped to valid indices,
deduplicated, sorted and capped at ``--max-probes``.  The final state (position 1.0) is
immediately before executing the teacher's final block, normally ``complete_task``.
Selection uses fixed quantiles (no randomness); sampling remains deterministic given
``--seed``.  Each selected state gets a fresh teacher-prefix replay independently of
earlier probe outcomes.  If all samples agree, an agreement row is emitted without
branches: K=0, empty continuation records, Q^T=Q^S=0.5 (prior only), dU=0.  Rows add
``_probe_select`` (effective policy, ``explicit`` for positions) and ``_probe_pos``
((t + 1) / demo_length); ``_agree_trace`` covers sampled states only.

State handling: AppWorld tasks are stateful and the REPL keeps variables across steps, so a
branch cannot be forked from a DB snapshot (AppWorld.save_state/load_state restore only the
app databases, not the REPL namespace).  Every continuation therefore opens a FRESH AppWorld
instance (own experiment directory, own bridge process), re-executes the prefix code blocks
c_0..c_{t-1}, executes the branch action, and only then lets the student continue.  Replay
is cheap (no LM calls); LM calls dominate the cost.

Ownership superscripts T/S only; teacher behaviour is not assumed better (docs/METHOD.md §2).

    PYTHONPATH=src .venv/bin/python tools/appworld_event_mine.py --split demand --port 8988 \
        --k 3 --max-probes 4 --out data/appworld_events/events_v1.jsonl

The environment side runs in the official AppWorld venv; this file doubles as that worker:
``envs/appworld-official/.venv/bin/python tools/appworld_event_mine.py --bridge`` (internal).
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "envs/appworld-repo"                      # official agent code + data symlink
OFFICIAL_PYTHON = ROOT / "envs/appworld-official/.venv/bin/python"
OUTPUTS = REPO / "experiments/outputs"
AGENT = "simplified_react_code_agent"
MINE_EXPERIMENT_ROOT = f"{AGENT}/bfas_mine"          # every env of this tool lives under here
DEFAULT_DEMOS = "results/bfas/appworld/collect_shared/demos.json"
DEFAULT_SPLIT_FILE = "results/bfas/appworld/ours_s0/support_split.json"
DEFAULT_TEACHER = "gpt-5.4"
BASE_STUDENT = "Qwen/Qwen3.5-4B"
CONTEXT_SAFETY = 64
MIN_STUDENT_TOKENS = 256

# exact regexes of the official agent (ignore_multiple_calls=True)
FULL_CODE_RE = re.compile(r"```python\n(.*?)```", re.DOTALL)
PARTIAL_CODE_RE = re.compile(r".*```python\n(.*)", re.DOTALL)


# --------------------------------------------------------------------------- scaffold port
def extract_code_and_fix_content(text: str) -> tuple[str, str]:
    """Port of SimplifiedReActCodeAgent.extract_code_and_fix_content (first block wins)."""
    match = FULL_CODE_RE.search(text)
    if match:
        return match.group(1).strip(), text[: match.end()]
    partial = PARTIAL_CODE_RE.match(text)
    if partial:
        code = partial.group(1).strip()
        fixed = text if text.endswith("\n") else text + "\n"
        return code, fixed + "```"
    return "", text


def assistant_message(reply: str) -> str:
    """What the official agent appends to the conversation after a reply."""
    _, fixed = extract_code_and_fix_content(reply or "")
    return fixed + "\n\n"


def observation_message(output: str) -> str:
    """What the official agent appends after executing code."""
    maybe_new_line = "\n" if not output.endswith("\n") else ""
    return "Output:\n```\n" + output + maybe_new_line + "```\n\n"


# JWT access tokens carry an `exp` claim derived from the wall clock, so two otherwise
# identical replays print different tokens; they are masked before outputs are compared.
_JWT_RE = re.compile(r"eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+")


def normalise_output(text: str) -> str:
    """Execution output with volatile, semantics-free spans masked (currently JWT tokens)."""
    return _JWT_RE.sub("<JWT>", text)


def same_output(a: str, b: str) -> bool:
    return normalise_output(a) == normalise_output(b)


_VOLATILE_KEYS = {"access_token", "token", "authorization", "api_key", "session_id"}
_TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(:\d{2})?")


def _normalise_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _normalise_value(v) for k, v in sorted(value.items()) if k not in _VOLATILE_KEYS}
    if isinstance(value, list):
        return [_normalise_value(v) for v in value]
    if isinstance(value, str):
        if _JWT_RE.fullmatch(value):
            return "<JWT>"
        if _TIMESTAMP_RE.match(value):
            return "<TIMESTAMP>"
    return value


def normalise_api_calls(calls: list[dict[str, Any]]) -> list[tuple[str, str, str]]:
    """(method, endpoint, canonical args) per call; auth tokens / timestamps masked, admin
    calls are never tracked by AppWorld, api_docs look-ups ARE calls (they change what the
    agent knows)."""
    out = []
    for call in calls:
        data = _normalise_value(call.get("data") or {})
        out.append((str(call.get("method", "")).lower(), str(call.get("url", "")),
                    json.dumps(data, sort_keys=True, ensure_ascii=False)))
    return out


def same_effects(code_a: str, calls_a: list[dict[str, Any]], output_a: str,
                 code_b: str, calls_b: list[dict[str, Any]], output_b: str) -> bool:
    """Effect-level action equality.  Blocks that made API calls agree iff their normalised
    call sequences agree.  Blocks that made no API call (pure computation / printing) agree
    iff they are the same code (AST) or printed the same (masked) output."""
    norm_a, norm_b = normalise_api_calls(calls_a), normalise_api_calls(calls_b)
    if norm_a or norm_b:
        return norm_a == norm_b
    return same_action(code_a, code_b) or same_output(output_a, output_b)


def canonical_code(code: str) -> str:
    """Formatting/comment-insensitive key of a code block.

    AST dump when the block parses (so `x=1  # c` == `x = 1`); otherwise the lines with
    comments removed and whitespace collapsed.  Different variable names, argument order or
    print wrappers are NOT identified: those are different actions for the environment."""
    try:
        return "ast:" + ast.dump(ast.parse(code))
    except (SyntaxError, ValueError):
        lines = []
        for line in code.splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                lines.append(" ".join(line.split()))
        return "txt:" + "\n".join(lines)


def same_action(code_a: str, code_b: str) -> bool:
    return canonical_code(code_a) == canonical_code(code_b)


def _counter(obj: Any, name: str) -> int:
    value = getattr(obj, name, 0)
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def laplace_mean(values: list[float], k: int) -> float:
    """Beta(1,1)-posterior-mean form used by every CRCD miner: (sum + 1) / (K + 2)."""
    if len(values) != k:
        raise ValueError(f"expected {k} outcomes, got {len(values)}")
    return (float(sum(values)) + 1.0) / (k + 2.0)


# --------------------------------------------------------------------------- demos / split
def load_demos(path: Path) -> dict[str, dict[str, Any]]:
    """task_id -> {turns: [(target, code)], messages0: [...], demo_outputs: [...], raw}."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    demos = payload.get("demos", payload)
    out: dict[str, dict[str, Any]] = {}
    for task_id, demo in demos.items():
        turns = demo["turns"]
        items = turns["items"] if isinstance(turns, dict) else turns
        messages0 = items[0]["context"]["messages"]
        targets = [turn["target"] for turn in items]
        # the demo's own recorded execution outputs (user messages after the instruction block)
        last_ctx = items[-1]["context"]["messages"]
        recorded = [m["content"] for m in last_ctx[len(messages0):] if m["role"] == "user"]
        out[task_id] = {
            "task_id": task_id,
            "targets": targets,
            "codes": [extract_code_and_fix_content(t)[0] for t in targets],
            "messages0": messages0,
            "recorded_outputs": recorded,
            "raw": demo.get("raw", {}),
        }
    return out


def select_task_ids(demos: dict[str, Any], split_file: Path | None, split: str) -> list[str]:
    ids = sorted(demos)
    if split_file is None or split == "all":
        return ids
    payload = json.loads(split_file.read_text(encoding="utf-8"))
    if split not in payload:
        raise KeyError(f"split {split!r} not in {split_file}")
    allowed = set(map(str, payload[split]))
    if split == "support" and "calibration" in payload:
        # calibration ids are never mined (METHOD.md §9)
        allowed -= set(map(str, payload["calibration"]))
    return [t for t in ids if t in allowed]


# --------------------------------------------------------------------------- interfaces
class Env(Protocol):
    def start(self, task_id: str, experiment_name: str, seed: int) -> dict[str, Any]: ...
    def execute(self, code: str) -> str: ...
    def completed(self) -> bool: ...
    def evaluate(self) -> dict[str, Any]: ...
    def stop(self) -> None: ...
    def close(self) -> None: ...


class Student(Protocol):
    def sample(self, messages: list[dict[str, str]], seed: int) -> dict[str, Any]: ...


class Renderer(Protocol):
    def render(self, messages: list[dict[str, str]]) -> str: ...


class PlainRenderer:
    """Dependency-free renderer (tests / dry-run); the real one uses the student's template."""

    def render(self, messages: list[dict[str, str]]) -> str:
        return "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages) \
            + "<|im_start|>assistant\n<think>\n"


class HFRenderer:
    """Student chat template.  ``thinking="off"`` renders with enable_thinking=False, i.e. the
    same prompt the server builds when the request carries chat_template_kwargs."""

    def __init__(self, policy: str, thinking: str = "off"):
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(policy, trust_remote_code=False)
        self.kwargs = {"enable_thinking": False} if thinking == "off" else {}

    def render(self, messages: list[dict[str, str]]) -> str:
        return self.tokenizer.apply_chat_template(
            list(messages), tokenize=False, add_generation_prompt=True, **self.kwargs
        )

    def count_tokens(self, messages: list[dict[str, str]]) -> int:
        """Reuse the loaded student tokenizer and the server's chat-template settings."""
        return len(self.tokenizer.apply_chat_template(
            list(messages), tokenize=True, add_generation_prompt=True, **self.kwargs
        ))


# --------------------------------------------------------------------------- vllm student
class StudentContextOverflow(RuntimeError):
    """The state still cannot be sampled with the minimum output allowance."""

    max_tokens = MIN_STUDENT_TOKENS


class VLLMStudent:
    """The official agent's request: chat completions with model/messages/temperature/seed."""

    THINKING_MODES = ("parser", "off")

    def __init__(self, port: int, served_model: str, temperature: float, max_tokens: int | None,
                 thinking: str = "off", timeout: float = 1800.0, retries: int = 5,
                 context_len: int = 32768,
                 prompt_token_counter: Callable[[list[dict[str, str]]], int] | None = None):
        if thinking not in self.THINKING_MODES:
            raise ValueError(f"thinking must be one of {self.THINKING_MODES}")
        self.url = f"http://127.0.0.1:{port}/v1/chat/completions"
        self.served_model = served_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.context_len = context_len
        self.prompt_token_counter = prompt_token_counter
        self.thinking = thinking
        self.timeout = timeout
        self.retries = retries
        self.calls = 0
        self.completion_tokens = 0
        self.reasoning_replies = 0
        self._lock = threading.Lock()
        # Exact message prefixes isolate tasks and concurrent continuation branches. Bound
        # the history so a long mining run does not retain every conversation ever sampled.
        self._prompt_usage: OrderedDict[tuple[tuple[str, str], ...], int] = OrderedDict()

    def estimate_prompt_tokens(self, messages: list[dict[str, str]]) -> int:
        if self.prompt_token_counter is not None:
            try:
                return int(self.prompt_token_counter(messages))
            except Exception:  # tokenizer unavailable/incompatible: use server usage below
                pass
        key = tuple((m["role"], m["content"]) for m in messages)
        with self._lock:
            for n in range(len(key), 0, -1):
                tokens = self._prompt_usage.get(key[:n])
                if tokens is not None:
                    return tokens + math.ceil(sum(len(content) for _, content in key[n:]) / 4)
        return math.ceil(sum(len(content) for _, content in key) / 4)

    def request_body(self, messages: list[dict[str, str]], seed: int) -> dict[str, Any]:
        """Qwen3.5 thinks by default.  ``parser``: the server runs --reasoning-parser qwen3 and
        the thought arrives as reasoning_content (dropped, as the official agent config does);
        ``off``: chat_template_kwargs.enable_thinking=false suppresses the thought."""
        body: dict[str, Any] = {
            "model": self.served_model, "messages": messages,
            "temperature": self.temperature, "seed": int(seed),
        }
        if self.max_tokens:
            body["max_tokens"] = int(self.max_tokens)
        if self.thinking == "off":
            body["chat_template_kwargs"] = {"enable_thinking": False}
        return body

    def check_server(self) -> dict[str, Any]:
        """One short request; refuse to mine if thinking text would leak into content."""
        out = self.sample([{"role": "user", "content": "Reply with the single word OK."}], seed=0)
        content = out["content"]
        if "</think>" in content or content.lstrip().startswith("<think>"):
            raise RuntimeError(
                "student server returns thinking inside content: start vllm with "
                "--reasoning-parser qwen3 (thinking=parser) or use --thinking off")
        return out

    def sample(self, messages: list[dict[str, str]], seed: int) -> dict[str, Any]:
        body = self.request_body(messages, seed)
        if self.max_tokens:
            body["max_tokens"] = max(MIN_STUDENT_TOKENS, min(
                self.max_tokens, self.context_len - self.estimate_prompt_tokens(messages) - CONTEXT_SAFETY))
        last: Exception | None = None
        attempt = 0
        overflow_retried = False
        while attempt < self.retries:
            request = urllib.request.Request(
                self.url, data=json.dumps(body).encode("utf-8"), method="POST",
                headers={"Content-Type": "application/json",
                         "Authorization": "Bearer " + os.environ.get("BFAS_STUDENT_API_KEY", "EMPTY")},
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    value = json.loads(response.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as exc:
                if 400 <= exc.code < 500:
                    detail = exc.read().decode("utf-8", errors="replace")
                    error = f"student server HTTP {exc.code}: {detail[:500]}"
                    if exc.code == 400 and any(marker in detail.lower() for marker in (
                        "context length", "context_length", "context window", "max_model_len",
                    )):
                        if overflow_retried:
                            raise StudentContextOverflow(error) from exc
                        body["max_tokens"] = MIN_STUDENT_TOKENS
                        overflow_retried = True
                        continue  # one context retry, independent of the transient retry budget
                    raise RuntimeError(error) from exc
                last = exc
                time.sleep(min(30.0, 2.0 * (attempt + 1)))
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
                last = exc
                time.sleep(min(30.0, 2.0 * (attempt + 1)))
            attempt += 1
        else:
            raise RuntimeError(f"student server unreachable after {self.retries} tries: {last}")
        choice = value["choices"][0]
        message = choice["message"]
        content = message.get("content") or ""
        reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
        usage = value.get("usage") or {}
        tokens = int(usage.get("completion_tokens") or 0)
        with self._lock:
            if usage.get("prompt_tokens") is not None:
                key = tuple((m["role"], m["content"]) for m in messages)
                self._prompt_usage[key] = int(usage["prompt_tokens"])
                self._prompt_usage.move_to_end(key)
                if len(self._prompt_usage) > 128:
                    self._prompt_usage.popitem(last=False)
            self.calls += 1
            self.completion_tokens += tokens
            if reasoning:
                self.reasoning_replies += 1
        return {"content": content, "finish_reason": choice.get("finish_reason"),
                "completion_tokens": tokens, "reasoning_chars": len(reasoning),
                "prompt_tokens": usage.get("prompt_tokens"), "max_tokens": body.get("max_tokens")}


# --------------------------------------------------------------------------- env bridge
def bridge_main() -> int:
    """Worker side (runs under envs/appworld-official/.venv): JSON lines on stdin/stdout."""
    protocol_out = sys.stdout
    with contextlib.redirect_stdout(sys.stderr):
        from appworld import AppWorld
        from appworld.common.random import set_random_seed

    world: Any = None

    def close_world() -> None:
        nonlocal world
        if world is not None:
            try:
                world.close()
            finally:
                world = None

    def handle(req: dict[str, Any]) -> dict[str, Any]:
        nonlocal world
        op = req["op"]
        if op == "start":
            close_world()
            seed = int(req.get("seed", 1))
            world = AppWorld(task_id=str(req["task_id"]), experiment_name=str(req["experiment_name"]),
                             random_seed=seed, raise_on_extra_parameters=True)
            set_random_seed(seed)   # as Agent.initialize does
            task = world.task
            return {"ok": True, "instruction": str(getattr(task, "instruction", "")),
                    "output_directory": str(world.output_directory)}
        if world is None:
            raise RuntimeError("no active AppWorld task")
        if op == "execute":
            tracker = world.requester.request_tracker
            n0 = len(tracker.requests)
            output = world.batch_execute([str(req.get("code", ""))])[0]
            calls = [dict(c) for c in tracker.requests[n0:]]
            return {"ok": True, "output": output, "api_calls": calls}
        if op == "completed":
            return {"ok": True, "completed": bool(world.task_completed())}
        if op == "evaluate":
            tracker = world.evaluate()
            return {"ok": True, "success": bool(tracker.success), "passed": int(tracker.pass_count),
                    "failed": int(tracker.fail_count), "num_tests": int(tracker.num_tests)}
        if op == "stop":
            close_world()
            return {"ok": True}
        raise ValueError(f"unknown op {op!r}")

    try:
        for line in sys.stdin:
            if not line.strip():
                continue
            req = json.loads(line)
            try:
                with contextlib.redirect_stdout(sys.stderr):
                    resp = handle(req)
            except Exception as exc:  # noqa: BLE001
                resp = {"ok": False, "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(limit=6)}
            resp["request_id"] = req.get("request_id")
            protocol_out.write(json.dumps(resp, ensure_ascii=False, default=str) + "\n")
            protocol_out.flush()
    finally:
        with contextlib.redirect_stdout(sys.stderr):
            close_world()
    return 0


class BridgeError(RuntimeError):
    pass


class SubprocessEnv:
    """One official-venv AppWorld process; `start` may be called repeatedly (fresh world each time)."""

    def __init__(self, timeout: float = 600.0):
        if not OFFICIAL_PYTHON.is_file():
            raise FileNotFoundError(f"official AppWorld venv missing: {OFFICIAL_PYTHON}")
        env = os.environ.copy()
        env["APPWORLD_ROOT"] = str(REPO)
        env.setdefault("PYTHONUNBUFFERED", "1")
        self._proc = subprocess.Popen(
            [str(OFFICIAL_PYTHON), str(Path(__file__).resolve()), "--bridge"],
            cwd=str(REPO), env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, encoding="utf-8", bufsize=1,
        )
        self._next = 1
        self.timeout = timeout
        self.last_api_calls: list[dict[str, Any]] = []

    def _request(self, op: str, **payload: Any) -> dict[str, Any]:
        if self._proc.poll() is not None:
            raise BridgeError(f"bridge exited (rc={self._proc.returncode})")
        rid = self._next
        self._next += 1
        assert self._proc.stdin is not None and self._proc.stdout is not None
        self._proc.stdin.write(json.dumps({"op": op, "request_id": rid, **payload}, ensure_ascii=False) + "\n")
        self._proc.stdin.flush()
        while True:
            line = self._proc.stdout.readline()
            if line == "":
                raise BridgeError(f"bridge died during {op!r} (rc={self._proc.poll()})")
            try:
                resp = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(resp, dict) or resp.get("request_id") != rid:
                continue
            if not resp.get("ok"):
                raise BridgeError(f"{op}: {resp.get('error')}")
            return resp

    def start(self, task_id: str, experiment_name: str, seed: int) -> dict[str, Any]:
        return self._request("start", task_id=task_id, experiment_name=experiment_name, seed=seed)

    def execute(self, code: str) -> str:
        resp = self._request("execute", code=code)
        self.last_api_calls = list(resp.get("api_calls") or [])
        return str(resp["output"])

    def completed(self) -> bool:
        return bool(self._request("completed")["completed"])

    def evaluate(self) -> dict[str, Any]:
        return self._request("evaluate")

    def stop(self) -> None:
        with contextlib.suppress(BridgeError):
            self._request("stop")

    def close(self) -> None:
        if self._proc.poll() is None:
            self.stop()
            with contextlib.suppress(OSError):
                self._proc.stdin.close()  # type: ignore[union-attr]
            try:
                self._proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self._proc.kill()


# --------------------------------------------------------------------------- miner
def parse_probe_positions(value: str) -> tuple[float, ...]:
    """CLI fractions; reject malformed/nonfinite positions instead of silently clipping."""
    try:
        positions = tuple(float(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("probe positions must be comma-separated fractions in [0, 1]") from exc
    if any(not 0.0 <= pos <= 1.0 for pos in positions):
        raise argparse.ArgumentTypeError("probe positions must be finite fractions in [0, 1]")
    return positions


def select_probe_steps(n_steps: int, max_probes: int, probe_select: str = "first",
                       probe_positions: tuple[float, ...] | None = None) -> list[int]:
    """Sorted candidate steps, independent of student outcomes and sampler randomness.

    ``first`` visits every step until the miner finds enough divergences.  Fixed policies
    visit at most max_probes steps; spread reserves one slot for the last decision state.
    """
    if probe_select not in ("first", "spread", "late"):
        raise ValueError(f"unknown probe selection policy: {probe_select!r}")
    if probe_positions is not None and any(not 0.0 <= pos <= 1.0 for pos in probe_positions):
        raise ValueError("probe positions must be finite fractions in [0, 1]")
    if n_steps <= 0 or max_probes <= 0:
        return []
    count = min(n_steps, max_probes)
    if probe_positions is not None:
        return sorted({max(0, min(n_steps - 1, math.ceil(pos * n_steps) - 1))
                       for pos in probe_positions})[:count]
    if probe_select == "first":
        return list(range(n_steps))
    if probe_select == "late":
        return list(range(n_steps - count, n_steps))
    # Integer nearest-rank quantiles at 1/count, ..., 1.0 avoid floating-point ties.
    return [(i * n_steps + count - 1) // count - 1 for i in range(1, count + 1)]


@dataclass
class MinerConfig:
    k: int = 3
    max_probes: int = 4
    probe_samples: int | None = None       # student samples per decision state (default K)
    max_steps: int = 50                    # official episode cap (LM calls / interactions)
    max_cont_steps: int | None = None      # extra cap on continuation length (decision variable)
    env_seed: int = 1                      # AppWorld random_seed (teacher demos: seed 0 + serial 1)
    sample_seed: int = 0                   # base sampler seed; continuation j uses base + 1000*j
    teacher: str = DEFAULT_TEACHER
    student_checkpoint: str = BASE_STUDENT
    student_max_tokens: int | None = 4096
    thinking: str = "off"                  # deployment config: tools/awoff_base_chain.sh serves with
                                           # --default-chat-template-kwargs '{"enable_thinking": false}'
    run_tag: str = "mine"
    keep_outputs: bool = False
    equality: str = "code_ast"             # or "api_effects" (student block executed in a scratch env)
    probe_select: str = "first"
    probe_positions: tuple[float, ...] | None = None
    context_len: int = 32768


@dataclass
class ContinuationResult:
    success: bool
    pass_frac: float
    passed: int
    num_tests: int
    steps: int                              # student LM calls in the continuation
    completed: bool
    replay_mismatch: int                    # prefix outputs differing from the trunk's
    error: str | None = None


@dataclass
class Miner:
    env_factory: Callable[[], Env]
    student: Student
    renderer: Renderer
    cfg: MinerConfig
    log: Callable[[str], None] = print
    _serial: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _free: list[Env] = field(default_factory=list)
    _all: list[Env] = field(default_factory=list)

    # ---- env pool: persistent bridge processes, one per concurrent episode ---
    def _acquire(self) -> Env:
        with self._lock:
            if self._free:
                return self._free.pop()
            env = self.env_factory()
            self._all.append(env)
            return env

    def _release(self, env: Env) -> None:
        with self._lock:
            self._free.append(env)

    def _experiment_name(self, label: str) -> str:
        with self._lock:
            self._serial += 1
            return f"{MINE_EXPERIMENT_ROOT}/{self.cfg.run_tag}/{label}_{self._serial}_{os.getpid()}"

    def close(self) -> None:
        for env in self._all:
            with contextlib.suppress(Exception):
                env.close()
        self._all.clear()
        self._free.clear()
        if not self.cfg.keep_outputs:
            shutil.rmtree(OUTPUTS / MINE_EXPERIMENT_ROOT / self.cfg.run_tag, ignore_errors=True)

    def _cleanup_run(self, experiment_name: str) -> None:
        if not self.cfg.keep_outputs:
            shutil.rmtree(OUTPUTS / experiment_name, ignore_errors=True)

    # ---- effect-level agreement ---------------------------------------------
    def block_effects(self, task_id: str, prefix_codes: list[str], code: str, label: str
                      ) -> tuple[list[dict[str, Any]], str]:
        """API calls + output of `code` executed right after the prefix, in a scratch env."""
        env = self._acquire()
        experiment_name = self._experiment_name(label)
        try:
            env.start(task_id, experiment_name, self.cfg.env_seed)
            for c in prefix_codes:
                env.execute(c)
            output = env.execute(code)
            return list(getattr(env, "last_api_calls", []) or []), output
        finally:
            with contextlib.suppress(Exception):
                env.stop()
            self._release(env)
            self._cleanup_run(experiment_name)

    def agreement(self, task_id: str, prefix_codes: list[str], teacher_code: str,
                  teacher_calls: list[dict[str, Any]] | None, teacher_output: str | None,
                  sample_codes: list[str], t: int) -> list[bool]:
        if self.cfg.equality != "api_effects":
            return [same_action(c, teacher_code) for c in sample_codes]
        assert teacher_calls is not None and teacher_output is not None
        agree: list[bool | None] = [True if same_action(c, teacher_code) else None for c in sample_codes]
        todo = [i for i, a in enumerate(agree) if a is None]
        if todo:
            with ThreadPoolExecutor(max_workers=len(todo)) as pool:
                effects = list(pool.map(
                    lambda i: self.block_effects(task_id, prefix_codes, sample_codes[i], f"{task_id}_t{t}_eff{i}"),
                    todo))
            for i, (calls, output) in zip(todo, effects):
                agree[i] = same_effects(sample_codes[i], calls, output, teacher_code, teacher_calls, teacher_output)
        return [bool(a) for a in agree]

    # ---- one branch continuation --------------------------------------------
    def continuation(self, task_id: str, prefix_codes: list[str], prefix_outputs: list[str],
                     messages_at_state: list[dict[str, str]], forced_reply: str, forced_code: str,
                     seed: int, label: str) -> ContinuationResult:
        """Fresh env -> replay prefix -> forced action -> student continues -> evaluate."""
        env = self._acquire()
        experiment_name = self._experiment_name(label)
        mismatch = 0
        steps = 0
        try:
            env.start(task_id, experiment_name, self.cfg.env_seed)
            for code, expected in zip(prefix_codes, prefix_outputs):
                output = env.execute(code)
                if not same_output(output, expected):
                    mismatch += 1
            messages = [dict(m) for m in messages_at_state]
            messages.append({"role": "assistant", "content": assistant_message(forced_reply)})
            output = env.execute(forced_code)
            messages.append({"role": "user", "content": observation_message(output)})
            completed = env.completed()
            used = len(prefix_codes) + 1
            budget = self.cfg.max_steps - used
            if self.cfg.max_cont_steps is not None:
                budget = min(budget, self.cfg.max_cont_steps)
            while not completed and steps < budget:
                reply = self.student.sample(messages, seed=seed + steps)["content"]
                steps += 1
                code, _ = extract_code_and_fix_content(reply)
                messages.append({"role": "assistant", "content": assistant_message(reply)})
                output = env.execute(code)
                messages.append({"role": "user", "content": observation_message(output)})
                completed = env.completed()
            verdict = env.evaluate()
            num_tests = int(verdict.get("num_tests") or 0)
            passed = int(verdict.get("passed") or 0)
            return ContinuationResult(
                success=bool(verdict.get("success")),
                pass_frac=(passed / num_tests) if num_tests else float(bool(verdict.get("success"))),
                passed=passed, num_tests=num_tests, steps=steps, completed=completed,
                replay_mismatch=mismatch,
            )
        except Exception as exc:  # noqa: BLE001
            return ContinuationResult(False, 0.0, 0, 0, steps, False, mismatch,
                                      error=f"{type(exc).__name__}: {str(exc)[:200]}")
        finally:
            with contextlib.suppress(Exception):
                env.stop()
            self._release(env)
            self._cleanup_run(experiment_name)

    def branch_utilities(self, task_id: str, prefix_codes: list[str], prefix_outputs: list[str],
                         messages_at_state: list[dict[str, str]], teacher_reply: str, teacher_code: str,
                         student_reply: str, student_code: str, probe_index: int) -> dict[str, Any]:
        """K matched continuations per branch, all 2K in parallel; branch j shares seed j."""
        k = self.cfg.k
        base = self.cfg.sample_seed + 100_000 * probe_index
        jobs = []
        for j in range(k):
            seed = base + 1000 * (j + 1)
            jobs.append(("T", seed, teacher_reply, teacher_code))
        for j in range(k):
            seed = base + 1000 * (j + 1)
            jobs.append(("S", seed, student_reply, student_code))

        def run(job: tuple[str, int, str, str]) -> ContinuationResult:
            branch, seed, reply, code = job
            return self.continuation(task_id, prefix_codes, prefix_outputs, messages_at_state,
                                     reply, code, seed, label=f"{task_id}_p{probe_index}_{branch}")

        with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
            results = list(pool.map(run, jobs))
        res_t, res_s = results[:k], results[k:]
        wins_t = [float(r.success) for r in res_t]
        wins_s = [float(r.success) for r in res_s]
        u_plus = laplace_mean(wins_t, k)
        u_minus = laplace_mean(wins_s, k)
        return {
            "u_plus": u_plus, "u_minus": u_minus, "dU": u_plus - u_minus,
            "wins": [int(sum(wins_t)), int(sum(wins_s))],
            "pass_frac": [sum(r.pass_frac for r in res_t) / k, sum(r.pass_frac for r in res_s) / k],
            "pass_frac_q": [laplace_mean([r.pass_frac for r in res_t], k),
                            laplace_mean([r.pass_frac for r in res_s], k)],
            "seeds": [base + 1000 * (j + 1) for j in range(k)],
            "cont_steps": [[r.steps for r in res_t], [r.steps for r in res_s]],
            "completed": [[r.completed for r in res_t], [r.completed for r in res_s]],
            "replay_mismatch": sum(r.replay_mismatch for r in results),
            "errors": [r.error for r in results if r.error],
        }

    # ---- one demo task ------------------------------------------------------
    def mine_task(self, demo: dict[str, Any], emit: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
        task_id = demo["task_id"]
        targets, codes = demo["targets"], demo["codes"]
        recorded = demo["recorded_outputs"]
        n_samples = self.cfg.probe_samples or self.cfg.k
        fixed_states = self.cfg.probe_positions is not None or self.cfg.probe_select != "first"
        probe_steps = select_probe_steps(min(len(targets), len(codes)), self.cfg.max_probes,
                                         self.cfg.probe_select, self.cfg.probe_positions)
        stats = {"task_id": task_id, "steps": 0, "probes": 0, "events": 0, "consequential": 0,
                 "agreements": 0, "overflows": 0, "agree_trace": [],
                 "prefix_matches_demo": True, "error": None}
        trunk = self._acquire()
        experiment_name = self._experiment_name(f"{task_id}_trunk")
        started = False
        t0 = time.time()
        try:
            if probe_steps and not fixed_states:
                started = True
                trunk.start(task_id, experiment_name, self.cfg.env_seed)
            messages = [dict(m) for m in demo["messages0"]]
            prefix_codes: list[str] = []
            prefix_outputs: list[str] = []
            for t in probe_steps:
                if fixed_states:
                    if started:
                        trunk.stop()
                        started = False
                        self._cleanup_run(experiment_name)
                    experiment_name = self._experiment_name(f"{task_id}_t{t}_trunk")
                    started = True
                    trunk.start(task_id, experiment_name, self.cfg.env_seed)
                    messages = [dict(m) for m in demo["messages0"]]
                    prefix_codes, prefix_outputs = [], []
                    # Rebuild this state only from teacher actions and live outputs.
                    stats["prefix_matches_demo"] = True
                    for s in range(t):
                        output = trunk.execute(codes[s])
                        if s < len(recorded) and not same_output(observation_message(output), recorded[s]):
                            stats["prefix_matches_demo"] = False
                        messages.append({"role": "assistant", "content": assistant_message(targets[s])})
                        messages.append({"role": "user", "content": observation_message(output)})
                        prefix_codes.append(codes[s])
                        prefix_outputs.append(output)
                        if trunk.completed():
                            break
                    if trunk.completed():
                        break  # no decision state exists after task completion
                teacher_reply, teacher_code = targets[t], codes[t]
                stats["steps"] += 1
                state_messages = [dict(m) for m in messages]
                # K student samples at s_t (matched seeds with the continuation seeds of probe 0)
                seeds = [self.cfg.sample_seed + 10 * t + j for j in range(n_samples)]
                overflow = None
                try:
                    with ThreadPoolExecutor(max_workers=n_samples) as pool:
                        samples = list(pool.map(lambda s: self.student.sample(state_messages, seed=s), seeds))
                except StudentContextOverflow as exc:
                    overflow = str(exc)
                    samples = []  # an incomplete sample set cannot establish agreement/divergence
                sample_codes = [extract_code_and_fix_content(s["content"])[0] for s in samples]
                teacher_calls = teacher_output = None
                if overflow is None and self.cfg.equality == "api_effects":
                    # the teacher's effects at s_t come from a scratch env so the trunk stays at s_t
                    teacher_calls, teacher_output = self.block_effects(
                        task_id, prefix_codes, teacher_code, f"{task_id}_t{t}_effT")
                agree = [] if overflow is not None else self.agreement(
                    task_id, prefix_codes, teacher_code, teacher_calls, teacher_output, sample_codes, t)
                agree_rate = sum(agree) / len(agree) if agree else None
                if overflow is None:
                    stats["agree_trace"].append(agree_rate)
                diverging = [i for i, a in enumerate(agree) if not a]
                if (diverging or fixed_states or overflow is not None) and stats["probes"] < self.cfg.max_probes:
                    i = diverging[0] if diverging else 0
                    sample = samples[i] if samples else {"content": "", "finish_reason": "overflow",
                                                         "max_tokens": MIN_STUDENT_TOKENS}
                    student_reply = sample["content"]
                    student_code = sample_codes[i] if sample_codes else ""
                    stats["probes"] += 1
                    probe = stats["probes"]
                    tb = time.time()
                    calls0 = _counter(self.student, "calls")
                    tok0 = _counter(self.student, "completion_tokens")
                    out = None
                    if diverging:
                        out = self.branch_utilities(task_id, prefix_codes, prefix_outputs, state_messages,
                                                    teacher_reply, teacher_code, student_reply, student_code,
                                                    probe_index=probe)
                        stats["events"] += 1
                        if out["dU"] > 0:
                            stats["consequential"] += 1
                    elif overflow is not None:
                        stats["overflows"] += 1
                    else:
                        stats["agreements"] += 1
                    wall = time.time() - tb
                    prompt = self.renderer.render(state_messages)
                    emit(self.event_row(
                        demo, t, prompt, state_messages, teacher_reply, student_reply, teacher_code,
                        student_code, prefix_codes, out, probe, agree_rate, len(samples),
                        sample.get("finish_reason"), stats, wall,
                        _counter(self.student, "calls") - calls0,
                        _counter(self.student, "completion_tokens") - tok0,
                        int(sample.get("reasoning_chars") or 0),
                        student_max_tokens=sample.get("max_tokens", self.cfg.student_max_tokens),
                        overflow=overflow,
                    ))
                    if overflow is not None:
                        self.log(f"  {task_id} t={t} probe={probe} overflow (no branches): {overflow}")
                    elif out is None:
                        self.log(f"  {task_id} t={t} probe={probe} agree=1.00 (no branches)")
                    else:
                        self.log(f"  {task_id} t={t} probe={probe} agree={agree_rate:.2f} "
                                 f"u+={out['u_plus']:.2f} u-={out['u_minus']:.2f} dU={out['dU']:+.2f} "
                                 f"wins={out['wins']} cont_steps={out['cont_steps']} "
                                 f"mismatch={out['replay_mismatch']} errs={len(out['errors'])} {wall:.0f}s")
                # follow the teacher on the trunk
                output = trunk.execute(teacher_code)
                if t < len(recorded) and not same_output(observation_message(output), recorded[t]):
                    stats["prefix_matches_demo"] = False
                messages.append({"role": "assistant", "content": assistant_message(teacher_reply)})
                messages.append({"role": "user", "content": observation_message(output)})
                prefix_codes.append(teacher_code)
                prefix_outputs.append(output)
                if stats["probes"] >= self.cfg.max_probes:
                    break
                if trunk.completed():
                    break
        except Exception as exc:  # noqa: BLE001
            stats["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
        finally:
            if started:
                with contextlib.suppress(Exception):
                    trunk.stop()
            self._release(trunk)
            self._cleanup_run(experiment_name)
        stats["wall"] = time.time() - t0
        return stats

    def event_row(self, demo: dict[str, Any], t: int, prompt: str, state_messages: list[dict[str, str]],
                  teacher_reply: str, student_reply: str, teacher_code: str, student_code: str,
                  prefix_codes: list[str], out: dict[str, Any] | None, probe: int, agree_rate: float | None,
                  n_samples: int, finish_reason: Any, stats: dict[str, Any], wall: float,
                  student_calls: int, student_tokens: int, reasoning_chars: int = 0,
                  student_max_tokens: int | None = None, overflow: str | None = None) -> dict[str, Any]:
        task_id = demo["task_id"]
        k = self.cfg.k if out is not None else 0
        if out is None:
            # Agreement/overflow has no measured outcomes: neutral prior, zero episodes.
            out = {"u_plus": 0.5, "u_minus": 0.5, "dU": 0.0, "wins": [0, 0],
                   "pass_frac": [0.0, 0.0], "pass_frac_q": [0.5, 0.5], "seeds": [],
                   "cont_steps": [[], []], "completed": [[], []], "errors": [], "replay_mismatch": 0}
        return {
            "task_id": task_id,
            "teacher": f"demo_replay:{self.cfg.teacher}",
            "turn_index": t,
            "prompt": prompt,
            "response": teacher_reply.strip(),          # y^T
            "_rejected": student_reply.strip(),         # y^S
            "_render_context": {"messages": state_messages},
            "token_hint": max(len(teacher_reply) // 4, 8),
            "_task_phat": out["u_minus"],
            "_traj": f"{task_id}#p{probe}",
            "_seed_category": task_id.rsplit("_", 1)[0],
            "_event_turn": t,
            "_event_good": teacher_code,
            "_event_bad": student_code,
            "_event_u_plus": out["u_plus"],
            "_event_u_minus": out["u_minus"],
            "_event_dU": out["dU"],
            "_event_k": k,
            "_event_wins": out["wins"],
            "_event_pass_frac": out["pass_frac"],
            "_event_pass_frac_q": out["pass_frac_q"],
            "_event_seeds": out["seeds"],
            "_event_cont_steps": out["cont_steps"],
            "_event_completed": out["completed"],
            "_event_errors": out["errors"] + ([overflow] if overflow is not None else []),
            "_event_overflow": overflow is not None,
            "_prefix_len": len(prefix_codes),
            "_prefix_codes": list(prefix_codes),
            "_demo_len": len(demo["codes"]),
            "_probe_select": "explicit" if self.cfg.probe_positions is not None else self.cfg.probe_select,
            "_probe_pos": (t + 1) / len(demo["codes"]),
            "_prefix_matches_demo": stats["prefix_matches_demo"],
            "_replay_output_mismatch": out["replay_mismatch"],
            "_student_agree_rate": agree_rate,
            "_student_samples": n_samples,
            "_student_finish_reason": finish_reason,
            "_agree_trace": list(stats["agree_trace"]),
            "_student_checkpoint": self.cfg.student_checkpoint,
            "_student_max_tokens": student_max_tokens,
            "_student_thinking": self.cfg.thinking,
            "_student_reasoning_chars": reasoning_chars,
            "_env_seed": self.cfg.env_seed,
            "_max_steps": self.cfg.max_steps,
            "_max_cont_steps": self.cfg.max_cont_steps,
            "_teacher_tokens": 0,
            "_cost": {"student_calls": student_calls, "student_completion_tokens": student_tokens,
                      "episodes": 2 * k, "wall_s": round(wall, 1)},
            "_utility": "official_success",
            "_equality": self.cfg.equality,
            "_scaffold": "official_simplified_react_code_agent",
        }


# --------------------------------------------------------------------------- cli
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--demos", default=DEFAULT_DEMOS)
    p.add_argument("--split-file", default=DEFAULT_SPLIT_FILE)
    p.add_argument("--split", choices=("demand", "support", "all"), default="demand")
    p.add_argument("--port", type=int, default=8988)
    p.add_argument("--served-model", default="bfas-policy")
    p.add_argument("--student", default=BASE_STUDENT, help="tokenizer for prompt rendering")
    p.add_argument("--k", type=int, default=3, help="continuations per branch")
    p.add_argument("--probe-samples", type=int, default=None, help="student samples per state (default K)")
    p.add_argument("--max-probes", type=int, default=4, help="max divergences (first) or selected states per demo")
    p.add_argument("--probe-select", choices=("first", "spread", "late"), default="first",
                   help="first divergences (default), uniform decision states including the last, or last states")
    p.add_argument("--probe-positions", type=parse_probe_positions, default=None,
                   help="explicit fractions in [0, 1], e.g. '0.25,0.5,0.75,1.0'; overrides --probe-select, "
                        "deduplicated and capped by --max-probes")
    p.add_argument("--max-steps", type=int, default=50, help="episode cap (official agent max_steps)")
    p.add_argument("--max-cont-steps", type=int, default=None, help="cap on continuation LM calls")
    p.add_argument("--max-tokens", type=int, default=4096, help="student max_tokens per call (0 = uncapped)")
    p.add_argument("--context-len", type=int, default=32768, help="student server context window in tokens")
    p.add_argument("--thinking", choices=VLLMStudent.THINKING_MODES, default="off",
                   help="off (deployment config, awoff_base_chain.sh): enable_thinking=false, sent per request "
                        "and expected as the server's --default-chat-template-kwargs; "
                        "parser: server runs --reasoning-parser qwen3 and the thought is dropped")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=0, help="sampler base seed")
    p.add_argument("--env-seed", type=int, default=1, help="AppWorld random_seed")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--start-index", type=int, default=0)
    p.add_argument("--tasks", default=None, help="comma-separated task ids (overrides split)")
    p.add_argument("--max-episodes", type=int, default=None, help="global budget of continuations")
    p.add_argument("--out", default="data/appworld_events/events_v1.jsonl")
    p.add_argument("--run-tag", default=None)
    p.add_argument("--keep-outputs", action="store_true")
    p.add_argument("--equality", choices=("code_ast", "api_effects"), default="code_ast",
                   help="code_ast: AST-canonical code equality; api_effects: same normalised API-call "
                        "sequence (student block executed in a scratch env; AST/output fallback for "
                        "blocks without API calls)")
    p.add_argument("--resume", action="store_true", help="skip tasks listed in <out>.done")
    p.add_argument("--shard", default=None, help="i/n: mine only tasks with index %% n == i")
    p.add_argument("--dry-run", action="store_true", help="plan only: no server, no environment")
    p.add_argument("--bridge", action="store_true", help=argparse.SUPPRESS)
    return p


def plan(demos: dict[str, Any], task_ids: list[str], args: argparse.Namespace) -> dict[str, Any]:
    per_task = []
    for task_id in task_ids:
        d = demos[task_id]
        steps = select_probe_steps(len(d["codes"]), args.max_probes, args.probe_select, args.probe_positions)
        max_probes = min(len(steps), max(0, args.max_probes))
        per_task.append({"task_id": task_id, "demo_turns": len(d["codes"]),
                         "probe_steps": steps, "max_probes": max_probes,
                         "max_episodes": 2 * args.k * max_probes})
    return {
        "tasks": len(task_ids), "k": args.k, "max_probes": args.max_probes,
        "max_episodes_total": sum(x["max_episodes"] for x in per_task),
        "max_probe_student_calls": sum(len(d["probe_steps"]) for d in per_task) * (args.probe_samples or args.k),
        "teacher_tokens": 0, "per_task": per_task,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.bridge:
        return bridge_main()
    demos = load_demos(ROOT / args.demos)
    if args.tasks:
        task_ids = [t for t in args.tasks.split(",") if t]
        missing = [t for t in task_ids if t not in demos]
        if missing:
            raise SystemExit(f"no demo for tasks: {missing}")
    else:
        split_file = (ROOT / args.split_file) if args.split_file else None
        task_ids = select_task_ids(demos, split_file, args.split)
    task_ids = task_ids[args.start_index:]
    if args.limit:
        task_ids = task_ids[: args.limit]
    if args.shard:
        i, n = (int(x) for x in args.shard.split("/"))
        task_ids = [t for j, t in enumerate(task_ids) if j % n == i]
    done_path = Path(str(ROOT / args.out) + ".done")
    if args.resume and done_path.is_file():
        done = {l.strip() for l in done_path.read_text().splitlines() if l.strip()}
        skipped = [t for t in task_ids if t in done]
        task_ids = [t for t in task_ids if t not in done]
        print(f"[aw-events] resume: skipping {len(skipped)} finished tasks", flush=True)
    summary = plan(demos, task_ids, args)
    probe_select = "explicit" if args.probe_positions is not None else args.probe_select
    print(f"[aw-events] {len(task_ids)} demos ({args.split}), K={args.k}, max_probes={args.max_probes}, "
          f"probe_select={probe_select}, equality={args.equality}, "
          f"max episodes={summary['max_episodes_total']}, teacher_tokens=0", flush=True)
    if args.dry_run:
        for row in summary["per_task"]:
            print(f"  {row['task_id']} demo_turns={row['demo_turns']} max_episodes={row['max_episodes']} "
                  f"probe_steps={row['probe_steps']}")
        print("[aw-events] dry-run: nothing written", flush=True)
        return 0

    cfg = MinerConfig(
        k=args.k, max_probes=args.max_probes, probe_samples=args.probe_samples,
        probe_select=args.probe_select, probe_positions=args.probe_positions,
        max_steps=args.max_steps, max_cont_steps=args.max_cont_steps, env_seed=args.env_seed,
        sample_seed=args.seed, student_checkpoint=args.student,
        student_max_tokens=(args.max_tokens or None), context_len=args.context_len, thinking=args.thinking,
        run_tag=args.run_tag or f"{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}",
        keep_outputs=args.keep_outputs, equality=args.equality,
    )
    student = VLLMStudent(args.port, args.served_model, args.temperature, cfg.student_max_tokens,
                          thinking=args.thinking, context_len=cfg.context_len)
    probe = student.check_server()
    print(f"[aw-events] student server ok (thinking={args.thinking}, max_tokens={cfg.student_max_tokens}, "
          f"probe finish={probe['finish_reason']} reasoning_chars={probe['reasoning_chars']})", flush=True)
    student.calls = 0
    student.completion_tokens = 0
    student.reasoning_replies = 0
    renderer = HFRenderer(args.student, thinking=args.thinking)
    student.prompt_token_counter = getattr(renderer, "count_tokens", None)
    miner = Miner(env_factory=SubprocessEnv, student=student, renderer=renderer, cfg=cfg)
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    handle = out.open("a", encoding="utf-8")
    totals = {"demos": 0, "events": 0, "agreements": 0, "overflows": 0, "consequential": 0,
              "agree_all": 0, "errored": 0, "episodes": 0}
    t0 = time.time()

    def emit(row: dict[str, Any]) -> None:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()

    try:
        for task_id in task_ids:
            if args.max_episodes is not None and totals["episodes"] >= args.max_episodes:
                print("[aw-events] episode budget exhausted", flush=True)
                break
            totals["demos"] += 1
            stats = miner.mine_task(demos[task_id], emit)
            totals["events"] += stats["events"]
            totals["agreements"] += stats["agreements"]
            totals["overflows"] += stats["overflows"]
            totals["consequential"] += stats["consequential"]
            totals["episodes"] += 2 * cfg.k * stats["events"]
            if stats["error"]:
                totals["errored"] += 1
                print(f"  {task_id} ERROR {stats['error']}", flush=True)
            else:
                with done_path.open("a", encoding="utf-8") as done_handle:
                    done_handle.write(task_id + "\n")
                if stats["events"] == 0 and stats["overflows"] == 0:
                    totals["agree_all"] += 1
            print(f"[aw-events] {task_id} steps={stats['steps']} probes={stats['probes']} "
                  f"events={stats['events']} consequential={stats['consequential']} "
                  f"prefix_matches_demo={stats['prefix_matches_demo']} wall={stats['wall']:.0f}s | "
                  f"{totals} calls={student.calls} tokens={student.completion_tokens} "
                  f"elapsed={time.time() - t0:.0f}s", flush=True)
    finally:
        handle.close()
        miner.close()
    print(f"[aw-events] FINAL {totals} calls={student.calls} tokens={student.completion_tokens} "
          f"reasoning_replies={student.reasoning_replies} thinking={args.thinking} "
          f"elapsed={time.time() - t0:.0f}s -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
