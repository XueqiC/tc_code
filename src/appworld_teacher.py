# Usage: .venv/bin/python src/appworld_teacher.py --teacher TEACHER --tag RUN_NAME
# Outputs: data/appworld_traces/TEACHER/train.jsonl and metrics.json.
"""Collect successful AppWorld trajectories from a remote teacher model.

OpenRouter teachers use ``openrouter/<vendor>/<model>`` and require
OPENROUTER_API_KEY. OPENROUTER_BASE_URL defaults to https://openrouter.ai/api/v1.
Official OpenAI teachers use ``openai/<model>`` and require OPENAI_API_KEY.
OPENAI_BASE_URL defaults to https://api.openai.com/v1. BFAS_OPENAI_SERVICE_TIER
optionally selects ``flex`` or ``priority`` processing.
BFAS adapters select their teacher with BFAS_TEACHER (WebShop and ALFWorld
both default to gpt-5.4).
"""

from __future__ import annotations

import argparse
import ast
import html
import json
import math
import os
import re
import select
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any


CODE_BLOCK_RE = re.compile(
    r"```[ \t]*(?:(?:python|py)[ \t]*)?\r?\n(.*?)```", re.IGNORECASE | re.DOTALL
)
CODE_TAG_RE = re.compile(r"<code(?:\s[^>]*)?>(.*?)</code>", re.IGNORECASE | re.DOTALL)
BRIDGE_RESPONSE_TIMEOUT_SECONDS = 300.0
CHAT_COMPLETION_TIMEOUT_SECONDS = 180.0
CHAT_COMPLETION_RETRIES = 3
RATE_LIMIT_RETRIES = 7
MAX_COMPLETION_TOKENS = 2048
DEFAULT_SEED = 42

# Registry values are public model identifiers only. Credentials and the endpoint
# are resolved from the environment when the selected teacher is loaded.
# OpenRouter and official OpenAI names resolve dynamically without registry entries.
TEACHER_MODELS = {
    "deepseek-v4-pro": "deepseek-v4-pro",
    "kimi-k2.7-code": "kimi-k2.7-code",
    "qwen3.5:397b": "qwen3.5:397b",
    "gpt-oss:120b": "gpt-oss:120b",
    "mistral-large-3": "mistral-large-3",
    "gpt-5.6-luna": "gpt-5.6-luna",
    "gpt-5.4": "gpt-5.4",
    "claude-sonnet-4-6": "claude-sonnet-4-6",
    "claude-haiku-4-5": "claude-haiku-4-5",
}
AZURE_OPENAI_MODELS = {"gpt-5.6-luna", "gpt-5.4"}
AZURE_ANTHROPIC_MODELS = {"claude-sonnet-4-6", "claude-haiku-4-5"}
AZURE_OPENAI_API_VERSION = "2024-10-21"
OPENROUTER_PREFIX = "openrouter/"
OPENROUTER_DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
OPENAI_PREFIX = "openai/"
OPENAI_DEFAULT_BASE_URL = "https://api.openai.com/v1"


class BridgeError(RuntimeError):
    """The bridge process exited, timed out, or violated its protocol."""


class BridgeRemoteError(RuntimeError):
    """AppWorld reported an error while handling a valid bridge request."""

    def __init__(self, response: dict[str, Any]):
        super().__init__(str(response.get("error", "Unknown AppWorld bridge error")))
        self.response = response


class TeacherAPIError(RuntimeError):
    """The remote chat-completions request failed or returned invalid data."""


@dataclass(frozen=True)
class TeacherConfig:
    name: str
    model: str
    endpoint: str
    api_key: str = field(repr=False)
    backend: str = "openai"  # Legacy Ollama-compatible transport; official is openai_api.
    service_tier: str | None = None


class AppWorldBridge:
    """Client for the JSON-lines protocol implemented by appworld_bridge.py."""

    def __init__(self, project_root: Path):
        bridge_python = project_root / "envs" / "appworld-venv" / "bin" / "python"
        bridge_script = project_root / "src" / "appworld_bridge.py"
        bridge_cwd = project_root / "envs"
        bridge_env = os.environ.copy()
        bridge_env["APPWORLD_ROOT"] = str(project_root / "envs" / "appworld-data")

        self._process = subprocess.Popen(
            [str(bridge_python), "-u", str(bridge_script)],
            cwd=str(bridge_cwd),
            env=bridge_env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        self._next_request_id = 1

    @property
    def alive(self) -> bool:
        return self._process.poll() is None

    def request(self, operation: str, **payload: Any) -> dict[str, Any]:
        if not self.alive or self._process.stdin is None or self._process.stdout is None:
            raise BridgeError(f"AppWorld bridge is not running (exit={self._process.poll()}).")

        request_id = self._next_request_id
        self._next_request_id += 1
        request = {"op": operation, "request_id": request_id, **payload}
        try:
            self._process.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
            self._process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise BridgeError(f"Could not write to AppWorld bridge: {exc}") from exc

        deadline = time.monotonic() + BRIDGE_RESPONSE_TIMEOUT_SECONDS
        ignored_lines = 0
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise BridgeError(f"AppWorld bridge timed out during {operation!r}.")
            ready, _, _ = select.select([self._process.stdout], [], [], remaining)
            if not ready:
                raise BridgeError(f"AppWorld bridge timed out during {operation!r}.")
            line = self._process.stdout.readline()
            if line == "":
                raise BridgeError(
                    f"AppWorld bridge exited during {operation!r} "
                    f"(exit={self._process.poll()})."
                )
            try:
                response = json.loads(line)
            except json.JSONDecodeError:
                ignored_lines += 1
                if ignored_lines > 100:
                    raise BridgeError("Too much non-JSON output from AppWorld bridge.")
                continue
            if not isinstance(response, dict):
                continue
            if response.get("request_id") != request_id:
                continue
            if not response.get("ok", False):
                raise BridgeRemoteError(response)
            return response

    def close(self) -> None:
        if self.alive:
            try:
                self.request("stop")
            except (BridgeError, BridgeRemoteError):
                pass
        if self._process.stdin is not None:
            try:
                self._process.stdin.close()
            except OSError:
                pass
        if self.alive:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
        else:
            self._process.wait()


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _tag(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value) or value in {".", ".."}:
        raise argparse.ArgumentTypeError(
            "must contain only letters, digits, '.', '_', or '-' and start alphanumerically"
        )
    return value


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--teacher", required=True, type=_teacher_name,
        help="Registered teacher name, openrouter/<vendor>/<model>, or openai/<model>",
    )
    parser.add_argument("--split", choices=("train",), default="train")
    parser.add_argument(
        "--max-tasks",
        type=_positive_int,
        default=90,
        help="Maximum tasks from the split to consider (default: all 90 train tasks)",
    )
    parser.add_argument("--attempts-per-task", type=_positive_int, default=3)
    parser.add_argument("--max-steps", type=_positive_int, default=12)
    parser.add_argument("--tag", required=True, type=_tag, help="AppWorld experiment tag")
    return parser.parse_args(argv)


def _azure_credentials() -> tuple[str, str]:
    endpoint = os.environ.get("AZURE_LLM_ENDPOINT", "").strip()
    api_key = os.environ.get("AZURE_LLM_KEY", "")
    if not endpoint or not api_key:
        cred_path = Path.home() / ".azure_llm_api"
        if cred_path.exists():
            for line in cred_path.read_text().splitlines():
                key, _, value = line.partition("=")
                if key == "AZURE_LLM_ENDPOINT" and not endpoint:
                    endpoint = value.strip()
                elif key == "AZURE_LLM_KEY" and not api_key:
                    api_key = value.strip()
    if not endpoint or not api_key:
        raise RuntimeError(
            "Azure credentials not found in environment or ~/.azure_llm_api"
        )
    return endpoint.rstrip("/"), api_key


def _teacher_name(name: str) -> str:
    if name.startswith(OPENAI_PREFIX):
        model = name[len(OPENAI_PREFIX):]
        if (
            model and "/" not in model and model not in {".", ".."}
            and not any(c.isspace() for c in model)
        ):
            return name
        raise argparse.ArgumentTypeError("expected openai/<model>")
    if name.startswith(OPENROUTER_PREFIX):
        model = name[len(OPENROUTER_PREFIX):]
        if (
            "/" in model and all(model.split("/"))
            and not any(c.isspace() for c in model)
        ):
            return name
        raise argparse.ArgumentTypeError("expected openrouter/<vendor>/<model>")
    if name not in TEACHER_MODELS:
        raise argparse.ArgumentTypeError(f"unknown teacher: {name!r}")
    return name


def load_teacher_config(name: str) -> TeacherConfig:
    if name.startswith((OPENROUTER_PREFIX, OPENAI_PREFIX)):
        _teacher_name(name)
        official_openai = name.startswith(OPENAI_PREFIX)
        env_prefix = "OPENAI" if official_openai else "OPENROUTER"
        default_url = OPENAI_DEFAULT_BASE_URL if official_openai else OPENROUTER_DEFAULT_BASE_URL
        base_url = os.environ.get(f"{env_prefix}_BASE_URL", default_url).strip()
        api_key = os.environ.get(f"{env_prefix}_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError(f"{env_prefix}_API_KEY is not set")
        parsed = urllib.parse.urlsplit(base_url)
        if (
            parsed.scheme not in {"http", "https"} or not parsed.netloc
            or parsed.query or parsed.fragment
        ):
            raise RuntimeError(f"{env_prefix}_BASE_URL must be an absolute HTTP(S) base URL")
        service_tier = None
        if official_openai:
            service_tier = os.environ.get("BFAS_OPENAI_SERVICE_TIER", "").strip() or None
            if service_tier not in {None, "flex", "priority"}:
                raise RuntimeError("BFAS_OPENAI_SERVICE_TIER must be flex, priority, or unset")
        return TeacherConfig(
            name=name,
            model=name.split("/", 1)[1],
            endpoint=base_url.rstrip("/") + "/chat/completions",
            api_key=api_key,
            backend="openai_api" if official_openai else "openrouter",
            service_tier=service_tier,
        )
    model = TEACHER_MODELS[name]
    if name in AZURE_OPENAI_MODELS:
        endpoint, api_key = _azure_credentials()
        return TeacherConfig(
            name=name,
            model=model,
            endpoint=(
                f"{endpoint}/openai/deployments/{model}/chat/completions"
                f"?api-version={AZURE_OPENAI_API_VERSION}"
            ),
            api_key=api_key,
            backend="azure_openai",
        )
    if name in AZURE_ANTHROPIC_MODELS:
        endpoint, api_key = _azure_credentials()
        return TeacherConfig(
            name=name,
            model=model,
            endpoint=f"{endpoint}/anthropic/v1/messages",
            api_key=api_key,
            backend="anthropic",
        )
    base_url = os.environ.get("OLLAMA_BASE_URL", "").strip()
    api_key = os.environ.get("OLLAMA_API_KEY", "")
    if not base_url:
        raise RuntimeError("OLLAMA_BASE_URL is not set")
    if not api_key:
        raise RuntimeError("OLLAMA_API_KEY is not set")

    parsed = urllib.parse.urlsplit(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise RuntimeError("OLLAMA_BASE_URL must be an absolute HTTP(S) URL")
    endpoint = base_url.rstrip("/") + "/v1/chat/completions"
    return TeacherConfig(name=name, model=model, endpoint=endpoint, api_key=api_key)


def _retry_delay(retry_number: int) -> float:
    return float(2 ** (retry_number - 1))


def _retry_after_delay(value: str | None, fallback: float) -> float:
    try:
        delay = float(value)
    except (TypeError, ValueError):
        try:
            delay = parsedate_to_datetime(value).timestamp() - time.time()
        except (TypeError, ValueError, OverflowError):
            return fallback
    return max(0.0, delay) if math.isfinite(delay) else fallback


def _is_timeout_reason(reason: Any) -> bool:
    if isinstance(reason, (socket.timeout, TimeoutError)):
        return True
    return "timed out" in str(reason).lower()


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, Sequence) and not isinstance(content, (str, bytes)):
        parts: list[str] = []
        for part in content:
            if isinstance(part, Mapping) and isinstance(part.get("text"), str):
                parts.append(part["text"])
        return "".join(parts).strip()
    raise TeacherAPIError("chat completion message content is not text")


def _build_request_body(
    config: TeacherConfig,
    messages: list[dict[str, str]],
    temperature: float | None = None,
) -> dict[str, Any]:
    if temperature is None:
        temperature = float(os.environ.get("TEACHER_TEMP", "0.0"))
    if config.backend == "azure_openai":
        # The deployment is fixed by the URL; gpt-5.x deployments accept
        # only max_completion_tokens and reject non-default temperature.
        body: dict[str, Any] = {
            "messages": messages,
            "max_completion_tokens": MAX_COMPLETION_TOKENS,
        }
        if temperature > 0 and config.model != "gpt-5.6-luna":
            body["temperature"] = temperature
        return body
    if config.backend == "anthropic":
        system_parts = [
            m["content"] for m in messages if m.get("role") == "system"
        ]
        chat = [m for m in messages if m.get("role") != "system"]
        body = {
            "model": config.model,
            "messages": chat,
            "max_tokens": MAX_COMPLETION_TOKENS,
            "temperature": temperature,
        }
        if system_parts:
            body["system"] = "\n\n".join(system_parts)
        return body
    if config.backend == "openai_api":
        body = {
            "model": config.model,
            "messages": messages,
            "max_completion_tokens": MAX_COMPLETION_TOKENS,
            "stream": False,
        }
        # Includes Luna: omit temperature even on sampled BFAS retries.
        if not config.model.startswith(("gpt-5", "o1", "o3", "o4")):
            body["temperature"] = temperature
        if config.service_tier is not None:
            body["service_tier"] = config.service_tier
        return body
    if config.backend == "openrouter":
        body = {
            "model": config.model,
            "messages": messages,
            "max_tokens": MAX_COMPLETION_TOKENS,
            "stream": False,
        }
        # OpenAI reasoning models reject non-default sampling temperatures.
        # Other routes retain greedy decoding on the first teacher attempt.
        if not config.model.startswith(
            ("openai/gpt-5", "openai/o1", "openai/o3", "openai/o4")
        ):
            body["temperature"] = temperature
        return body
    return {
        "model": config.model,
        "messages": messages,
        "temperature": temperature,
        # deepseek's hidden reasoning consumes the whole completion budget
        # and returns empty content in tight agent loops; thinking stays
        # off unless TEACHER_THINK=1 explicitly requests it.
        "think": os.environ.get("TEACHER_THINK") == "1",
        "max_tokens": MAX_COMPLETION_TOKENS,
        "stream": False,
    }


_OLLAMA_KEY_IDX = 0
_OLLAMA_KEY_LOCK = threading.Lock()


def _ollama_keys(primary: str) -> list[str]:
    keys = [primary]
    alt = Path.home() / ".ollama_api_key2"
    if alt.exists():
        k = alt.read_text().strip()
        if k and k != primary:
            keys.append(k)
    return keys


def _request_headers(config: TeacherConfig) -> dict[str, str]:
    if config.backend in {"openrouter", "openai_api"}:
        return {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
    if config.backend == "azure_openai":
        return {
            "api-key": config.api_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
    if config.backend == "anthropic":
        return {
            "x-api-key": config.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
    keys = _ollama_keys(config.api_key)
    with _OLLAMA_KEY_LOCK:
        key = keys[_OLLAMA_KEY_IDX % len(keys)]
    return {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def generate_reply(
    config: TeacherConfig,
    messages: list[dict[str, str]],
    temperature: float | None = None,
    *,
    usage_callback: Callable[[Mapping[str, Any]], None] | None = None,
    max_completion_tokens: int | None = None,
    retries: int | None = None,
) -> str:
    global _OLLAMA_KEY_IDX
    body = _build_request_body(config, messages, temperature)
    if max_completion_tokens is not None:
        if type(max_completion_tokens) is not int or max_completion_tokens <= 0:
            raise ValueError("max_completion_tokens must be a positive integer")
        key = "max_completion_tokens" if "max_completion_tokens" in body else "max_tokens"
        body[key] = max_completion_tokens
    if retries is not None and (type(retries) is not int or retries < 0):
        raise ValueError("retries must be a non-negative integer")
    payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
    # Direct HTTP has no SDK auto-retries; all attempts are bounded below.
    opener = urllib.request.build_opener()

    total_attempts = (CHAT_COMPLETION_RETRIES if retries is None else retries) + 1
    rate_limit_attempts = (RATE_LIMIT_RETRIES if retries is None else retries) + 1
    for request_attempt in range(1, max(total_attempts, rate_limit_attempts) + 1):
        request = urllib.request.Request(
            config.endpoint,
            data=payload,
            headers=_request_headers(config),
            method="POST",
        )
        try:
            with opener.open(
                request, timeout=CHAT_COMPLETION_TIMEOUT_SECONDS
            ) as response:
                response_data = response.read()
            break
        except urllib.error.HTTPError as exc:
            status = exc.code
            retry_after = exc.headers.get("Retry-After") if exc.headers else None
            exc.close()
            if status == 429 and request_attempt < rate_limit_attempts:
                if config.backend == "openai":
                    # rotate between available Ollama keys before waiting
                    with _OLLAMA_KEY_LOCK:
                        _OLLAMA_KEY_IDX += 1
                delay = _retry_after_delay(
                    retry_after, min(5.0 * 2 ** (request_attempt - 1), 60.0)
                )
                time.sleep(delay)
                continue
            if status == 429:
                raise TeacherAPIError(
                    f"chat completion returned HTTP 429 after "
                    f"{rate_limit_attempts} attempts"
                ) from None
            if 500 <= status < 600 and request_attempt < total_attempts:
                time.sleep(_retry_after_delay(retry_after, _retry_delay(request_attempt)))
                continue
            if 500 <= status < 600:
                raise TeacherAPIError(
                    f"chat completion returned HTTP {status} after {request_attempt} attempts"
                ) from None
            raise TeacherAPIError(f"chat completion returned HTTP {status}") from None
        except (socket.timeout, TimeoutError):
            if request_attempt < total_attempts:
                time.sleep(_retry_delay(request_attempt))
                continue
            raise TeacherAPIError(
                f"chat completion timed out after {request_attempt} attempts"
            ) from None
        except urllib.error.URLError as exc:
            if _is_timeout_reason(exc.reason):
                if request_attempt < total_attempts:
                    time.sleep(_retry_delay(request_attempt))
                    continue
                raise TeacherAPIError(
                    f"chat completion timed out after {request_attempt} attempts"
                ) from None
            raise TeacherAPIError(
                f"chat completion request failed ({type(exc.reason).__name__})"
            ) from None
    else:  # pragma: no cover - all loop exits above are explicit
        raise TeacherAPIError("chat completion request failed")

    try:
        data = json.loads(response_data)
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise TeacherAPIError("chat completion returned invalid JSON") from None
    if not isinstance(data, Mapping):
        raise TeacherAPIError("chat completion response is not an object")
    if usage_callback is not None:
        usage_callback(data.get("usage") or {})
    usage_log = os.environ.get("AZURE_USAGE_LOG")
    if usage_log and config.backend in ("azure_openai", "anthropic"):
        usage = data.get("usage") or {}
        total = usage.get("total_tokens") or (
            (usage.get("input_tokens") or 0) + (usage.get("output_tokens") or 0)
        )
        if total:
            try:
                with open(usage_log, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps({
                        "t": time.time(), "model": config.model,
                        "total_tokens": int(total),
                    }) + "\n")
            except OSError:
                pass
    if config.backend == "anthropic":
        content = data.get("content")
        if not isinstance(content, Sequence) or isinstance(content, (str, bytes)):
            raise TeacherAPIError("anthropic response has no content list")
        parts = [
            part["text"]
            for part in content
            if isinstance(part, Mapping) and isinstance(part.get("text"), str)
        ]
        if not parts:
            raise TeacherAPIError("anthropic response contains no text parts")
        return "".join(parts).strip()
    choices = data.get("choices")
    if not isinstance(choices, Sequence) or isinstance(choices, (str, bytes)) or not choices:
        raise TeacherAPIError("chat completion response has no choices")
    first_choice = choices[0]
    if not isinstance(first_choice, Mapping) or not isinstance(first_choice.get("message"), Mapping):
        raise TeacherAPIError("chat completion choice has no message")
    return _content_text(first_choice["message"].get("content"))


def strip_think(reply: str) -> str:
    if "</think>" in reply:
        return reply.split("</think>", 1)[1].lstrip()
    return reply


def extract_python_code(reply: str) -> str | None:
    reply = strip_think(reply)
    match = CODE_BLOCK_RE.search(reply)
    if match:
        code = match.group(1).strip()
        return code or None
    match = CODE_TAG_RE.search(reply)
    if match:
        code = html.unescape(match.group(1)).strip()
        return code or None
    return None


def calls_complete_task(code: str) -> bool:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "complete_task":
            continue
        supervisor = node.func.value
        if (
            isinstance(supervisor, ast.Attribute)
            and supervisor.attr == "supervisor"
            and isinstance(supervisor.value, ast.Name)
            and supervisor.value.id == "apis"
        ):
            return True
    return False


def truncate_output(text: str, limit: int = 1500) -> str:
    if len(text) <= limit:
        return text
    marker = "\n...[output truncated]...\n"
    left = (limit - len(marker)) // 2
    right = limit - len(marker) - left
    return text[:left] + marker + text[-right:]


def make_system_prompt(task: dict[str, Any]) -> str:
    supervisor = task.get("supervisor") or {}
    details = ", ".join(
        f"{key}={value}"
        for key, value in (
            ("name", supervisor.get("name")),
            ("email", supervisor.get("email")),
            ("phone", supervisor.get("phone")),
        )
        if value
    ) or "not available"
    instruction = task.get("instruction", "")
    return (
        "You are an agent operating inside AppWorld. Write Python code that "
        "uses the preloaded `apis` object to complete the user's task.\n\n"
        "How the `apis` object works (important):\n"
        "- Apps are namespaces, APIs are methods: call "
        "`apis.<app_name>.<api_name>(...)`. Never call an app itself "
        "(`apis.api_docs()` is a TypeError).\n"
        "- Discover what exists, in this order:\n"
        "  1. `print(apis.api_docs.show_app_descriptions())`\n"
        "  2. `print(apis.api_docs.show_api_descriptions(app_name='<app>'))`\n"
        "  3. `print(apis.api_docs.show_api_doc(app_name='<app>', "
        "api_name='<api>'))`\n"
        "- Most apps require login. Get the supervisor's stored passwords "
        "with `print(apis.supervisor.show_account_passwords())`, then call "
        "the app's `login` API with the supervisor's email/username and that "
        "password; pass the returned access token to later calls of that app "
        "as its api_doc specifies.\n"
        "- Print API results so you can inspect them before deciding the "
        "next step. Do not invent APIs or argument names.\n"
        "- If the task asks a question, submit the answer explicitly: "
        "`apis.supervisor.complete_task(answer=<value>)`. If the task only "
        "requires actions, call `apis.supervisor.complete_task()` with no "
        "arguments.\n"
        "- List APIs are paginated (`page_index` starts at 0): loop over "
        "pages until an empty page when you need a complete list.\n"
        "- Stay with the app(s) the task points to; do not wander into "
        "unrelated apps.\n"
        "\n"
        "Return exactly one executable Python code block per turn, with no "
        "additional code blocks. You will receive the execution output and "
        "may then write the next block. State persists across turns. When "
        "the task is fully complete, call `apis.supervisor.complete_task()` "
        "in the final code block.\n\n"
        f"Task instruction:\n{instruction}\n\n"
        f"Supervisor details: {details}"
    )

def _safe_error(stage: str, exc: BaseException, api_key: str) -> str:
    message = f"{stage}: {type(exc).__name__}: {exc}"
    return message.replace(api_key, "[REDACTED]") if api_key else message


def _load_successful_task_ids(path: Path) -> set[str]:
    successful: set[str] = set()
    if not path.exists():
        return successful
    with path.open("r", encoding="utf-8", errors="replace") as trace_file:
        for line in trace_file:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, Mapping) and record.get("success") is True:
                task_id = record.get("task_id")
                if isinstance(task_id, str):
                    successful.add(task_id)
    return successful


def _ensure_trailing_newline(path: Path) -> None:
    if not path.exists() or path.stat().st_size == 0:
        return
    with path.open("rb") as trace_file:
        trace_file.seek(-1, os.SEEK_END)
        has_newline = trace_file.read(1) == b"\n"
    if not has_newline:
        with path.open("ab") as trace_file:
            trace_file.write(b"\n")


def _write_metrics(path: Path, metrics: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2, ensure_ascii=False)
        metrics_file.write("\n")
    os.replace(temporary, path)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_teacher_config(args.teacher)
    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / "data" / "appworld_traces" / config.name
    output_dir.mkdir(parents=True, exist_ok=True)
    traces_path = output_dir / f"{args.split}.jsonl"
    metrics_path = output_dir / "metrics.json"

    successful_task_ids = _load_successful_task_ids(traces_path)
    _ensure_trailing_newline(traces_path)
    tasks_tried = 0
    solved = 0
    attempts_used = 0
    skipped_solved = 0
    errors: list[dict[str, Any]] = []
    bridge: AppWorldBridge | None = None
    reached_end_of_split = False

    try:
        with traces_path.open("a", encoding="utf-8") as traces_file:
            for task_index in range(args.max_tasks):
                task_id = f"{args.split}[{task_index}]"
                task_counted = False

                for attempt in range(1, args.attempts_per_task + 1):
                    if bridge is None or not bridge.alive:
                        if bridge is not None:
                            bridge.close()
                        bridge = AppWorldBridge(project_root)

                    start_payload: dict[str, Any] = {
                        "split": args.split,
                        "experiment_name": f"{args.tag}-attempt-{attempt}",
                        "seed": DEFAULT_SEED,
                    }
                    if task_id.startswith(f"{args.split}["):
                        start_payload["index"] = task_index
                    else:
                        start_payload["task_id"] = task_id

                    try:
                        task = bridge.request("start", **start_payload)
                    except BridgeRemoteError as exc:
                        if exc.response.get("error_code") == "end_of_split":
                            reached_end_of_split = True
                            break
                        if not task_counted:
                            tasks_tried += 1
                            task_counted = True
                        attempts_used += 1
                        errors.append(
                            {
                                "task_id": task_id,
                                "attempt": attempt,
                                "error": _safe_error("start", exc, config.api_key),
                            }
                        )
                        continue
                    except Exception as exc:
                        if not task_counted:
                            tasks_tried += 1
                            task_counted = True
                        attempts_used += 1
                        errors.append(
                            {
                                "task_id": task_id,
                                "attempt": attempt,
                                "error": _safe_error("start", exc, config.api_key),
                            }
                        )
                        continue

                    task_id = str(task["task_id"])
                    if not task_counted and task_id in successful_task_ids:
                        skipped_solved += 1
                        try:
                            bridge.request("stop")
                        except Exception as exc:
                            errors.append(
                                {
                                    "task_id": task_id,
                                    "attempt": attempt,
                                    "error": _safe_error("stop skipped task", exc, config.api_key),
                                }
                            )
                        break

                    if not task_counted:
                        tasks_tried += 1
                        task_counted = True
                    attempts_used += 1
                    turns = [
                        {"role": "system", "content": make_system_prompt(task)},
                        {"role": "user", "content": "Begin by consulting the API documentation."},
                    ]
                    code_blocks: list[str] = []
                    hard_fail_task = False

                    try:
                        for _ in range(args.max_steps):
                            reply = generate_reply(config, turns)
                            reply = strip_think(reply)
                            turns.append({"role": "assistant", "content": reply})
                            code = extract_python_code(reply)
                            if code is None:
                                break
                            code_blocks.append(code)
                            execution = bridge.request("execute", code=code)
                            output = truncate_output(str(execution.get("output", "")))
                            turns.append(
                                {
                                    "role": "user",
                                    "content": f"Execution output:\n{output}\n\nContinue the task.",
                                }
                            )
                            if calls_complete_task(code):
                                break
                    except TeacherAPIError as exc:
                        hard_fail_task = True
                        errors.append(
                            {
                                "task_id": task_id,
                                "attempt": attempt,
                                "error": _safe_error("teacher API", exc, config.api_key),
                            }
                        )
                    except Exception as exc:
                        errors.append(
                            {
                                "task_id": task_id,
                                "attempt": attempt,
                                "error": _safe_error("agent loop", exc, config.api_key),
                            }
                        )

                    success = False
                    try:
                        evaluation = bridge.request("evaluate")
                        success = bool(evaluation.get("success", False))
                    except Exception as exc:
                        errors.append(
                            {
                                "task_id": task_id,
                                "attempt": attempt,
                                "error": _safe_error("evaluate", exc, config.api_key),
                            }
                        )
                    try:
                        bridge.request("stop")
                    except Exception as exc:
                        errors.append(
                            {
                                "task_id": task_id,
                                "attempt": attempt,
                                "error": _safe_error("stop", exc, config.api_key),
                            }
                        )

                    if os.environ.get("APPWORLD_DEBUG") == "1":
                        debug_episode = {
                            "task_id": task_id,
                            "teacher": config.name,
                            "attempt": attempt,
                            "success": success,
                            "turns": turns,
                            "code_blocks": code_blocks,
                            "instruction": str(task.get("instruction", "")),
                        }
                        debug_path = traces_path.with_name(
                            f"{args.split}_debug.jsonl"
                        )
                        with debug_path.open("a", encoding="utf-8") as debug_file:
                            debug_file.write(
                                json.dumps(debug_episode, ensure_ascii=False) + "\n"
                            )

                    if success:
                        episode = {
                            "task_id": task_id,
                            "teacher": config.name,
                            "attempt": attempt,
                            "success": True,
                            "turns": turns,
                            "code_blocks": code_blocks,
                            "instruction": str(task.get("instruction", "")),
                        }
                        traces_file.write(json.dumps(episode, ensure_ascii=False) + "\n")
                        traces_file.flush()
                        successful_task_ids.add(task_id)
                        solved += 1
                        break
                    if hard_fail_task:
                        break

                if reached_end_of_split:
                    break
    finally:
        if bridge is not None:
            bridge.close()

    metrics = {
        "teacher": config.name,
        "tag": args.tag,
        "split": args.split,
        "tasks_tried": tasks_tried,
        "solved": solved,
        "attempts_used": attempts_used,
        "skipped_solved": skipped_solved,
        "errors": errors,
        "config": {
            "max_tasks": args.max_tasks,
            "attempts_per_task": args.attempts_per_task,
            "max_steps": args.max_steps,
        },
    }
    _write_metrics(metrics_path, metrics)

    print(
        f"AppWorld teacher {config.name}: {solved}/{tasks_tried} solved; "
        f"attempts={attempts_used}; traces={traces_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
