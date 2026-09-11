"""Native tau2-bench adapter for the BFAS pipeline.

Episodes are always executed by the vendored ``tau2 run`` command.  The
student and teacher differ only in the OpenAI-compatible endpoint supplied to
LiteLLM; task setup, user simulation, tools, orchestration, and grading remain
owned by tau2.
"""

from __future__ import annotations

import copy
import json
import math
import os
import shutil
import subprocess
import tempfile
import urllib.parse
import urllib.request
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..adapter import (
    BenchmarkAdapter,
    Demo,
    PolicyRef,
    Rollout,
    TaskRef,
    TeacherEpisode,
    Turn,
)


ROOT = Path(__file__).resolve().parents[3]
TAU2_ROOT = ROOT / "envs/tau2/repo"
TAU2_BIN = ROOT / "envs/tau2/.venv/bin/tau2"
TAU2_DATA = TAU2_ROOT / "data/tau2"
CORE_DOMAINS = ("airline", "retail", "telecom")
SERVED_MODEL_NAME = "bfas-policy"
DEFAULT_STUDENT = "Qwen/Qwen3.5-2B"
OPENAI_BASE = "https://api.openai.com/v1"
LUNA_MODEL = "openai/gpt-5.6-luna"
GUIDE_HEADER = (
    "Worked example from a verified expert episode on this exact task. "
    "Use it as a strategy reference, then solve the current interaction "
    "yourself."
)


@dataclass(frozen=True)
class NativeRun:
    """Artifacts produced by one invocation of the official CLI."""

    results: dict[str, Any]
    run_dir: Path
    cleanup_root: Path | None = None
    guidance_block: str | None = None

    def cleanup(self) -> None:
        if self.cleanup_root is not None:
            shutil.rmtree(self.cleanup_root, ignore_errors=True)


def encode_task_id(domain: str, task_id: str) -> str:
    """Make tau2's domain-local task ids globally unique to BFAS."""

    if domain not in CORE_DOMAINS:
        raise ValueError(f"unsupported tau2 domain {domain!r}")
    return f"{domain}:{task_id}"


def decode_task_id(task_id: str) -> tuple[str, str]:
    """Decode a BFAS tau2 task id into the official domain and task id."""

    domain, separator, native_id = str(task_id).partition(":")
    if not separator or domain not in CORE_DOMAINS or not native_id:
        raise ValueError(
            f"invalid tau2 task id {task_id!r}; expected '<domain>:<task-id>'"
        )
    return domain, native_id


def extract_verdict(simulation: Mapping[str, Any]) -> bool:
    """Return tau2's positive pass signal for one exact simulation.

    A missing reward, an infrastructure error, or any non-unit reward is not a
    pass.  In particular, this never infers success from the absence of a
    failure record.
    """

    if simulation.get("termination_reason") == "infrastructure_error":
        return False
    reward_info = simulation.get("reward_info")
    if not isinstance(reward_info, Mapping):
        return False
    reward = reward_info.get("reward")
    if isinstance(reward, bool) or not isinstance(reward, (int, float)):
        return False
    return math.isclose(float(reward), 1.0, rel_tol=0.0, abs_tol=1e-6)


def extract_verdicts(
    results: Mapping[str, Any], expected_task_ids: Sequence[str]
) -> dict[str, bool]:
    """Extract one-trial verdicts, leaving missing tasks explicitly false."""

    verdicts = {str(task_id): False for task_id in expected_task_ids}
    simulations = results.get("simulations")
    if not isinstance(simulations, list):
        return verdicts
    seen: set[str] = set()
    for simulation in simulations:
        if not isinstance(simulation, Mapping):
            continue
        task_id = str(simulation.get("task_id", ""))
        if task_id not in verdicts:
            continue
        if task_id in seen:
            raise ValueError(f"tau2 returned multiple simulations for task {task_id!r}")
        seen.add(task_id)
        verdicts[task_id] = extract_verdict(simulation)
    return verdicts


def official_pass_one(results: Mapping[str, Any]) -> float:
    """Compute tau2's pass^1 over non-infrastructure simulations."""

    by_task: dict[str, list[bool]] = defaultdict(list)
    simulations = results.get("simulations")
    if not isinstance(simulations, list):
        return 0.0
    for simulation in simulations:
        if not isinstance(simulation, Mapping):
            continue
        if simulation.get("termination_reason") == "infrastructure_error":
            continue
        task_id = simulation.get("task_id")
        if task_id is None:
            continue
        by_task[str(task_id)].append(extract_verdict(simulation))
    per_task = [sum(values) / len(values) for values in by_task.values() if values]
    return sum(per_task) / len(per_task) if per_task else 0.0


def _restore_logged_content(value: Any) -> Any:
    # tau2's verbose logger splits multiline strings into a list of lines for
    # readability.  Text-mode requests have no multimodal list content, so
    # joining lists of strings exactly reverses that logging transform.
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return "\n".join(value)
    return copy.deepcopy(value)


def _renderer_messages(value: Any) -> list[dict[str, Any]]:
    """Apply the same tool-argument normalization as vLLM's chat server."""

    if not isinstance(value, list):
        raise ValueError("tau2 LLM log request has no messages list")
    output: list[dict[str, Any]] = []
    for raw_message in value:
        if not isinstance(raw_message, Mapping):
            raise ValueError("tau2 LLM log contains a non-object message")
        message = copy.deepcopy(dict(raw_message))
        if "content" in message:
            message["content"] = _restore_logged_content(message["content"])
        tool_calls = message.get("tool_calls")
        if message.get("role") == "assistant" and isinstance(tool_calls, list):
            for tool_call in tool_calls:
                if not isinstance(tool_call, dict):
                    continue
                function = tool_call.get("function")
                if not isinstance(function, dict):
                    continue
                arguments = function.get("arguments")
                if isinstance(arguments, str):
                    function["arguments"] = json.loads(arguments) if arguments else {}
        output.append(message)
    return output


def _assistant_from_log(response: Mapping[str, Any]) -> dict[str, Any]:
    content = response.get("content")
    raw_calls = response.get("tool_calls")
    calls: list[dict[str, Any]] | None = None
    if isinstance(raw_calls, list) and raw_calls:
        calls = []
        for raw_call in raw_calls:
            if not isinstance(raw_call, Mapping):
                raise ValueError("tau2 LLM log contains an invalid tool call")
            arguments = raw_call.get("arguments", {})
            if isinstance(arguments, str):
                arguments = json.loads(arguments) if arguments else {}
            calls.append({
                "id": str(raw_call.get("id", "")),
                "type": "function",
                "function": {
                    "name": str(raw_call["name"]),
                    "arguments": arguments,
                },
            })
    # deepseek (and other OpenAI-compatible teachers) log tool-only turns
    # with content "" rather than null; keep the prose/tool distinction
    # structural so the renderer does not mistake "" for a prose reply.
    if calls and isinstance(content, str) and not content.strip():
        content = None
    return {"role": "assistant", "content": content, "tool_calls": calls}


def _apply_template(
    tokenizer: Any,
    messages: Sequence[Mapping[str, Any]],
    tools: Sequence[Mapping[str, Any]],
    **kwargs: Any,
) -> str:
    arguments: dict[str, Any] = {
        "tokenize": False,
        **kwargs,
    }
    if tools:
        arguments["tools"] = list(tools)
    rendered = tokenizer.apply_chat_template(list(messages), **arguments)
    if not isinstance(rendered, str):
        raise TypeError("student chat renderer did not return text")
    return rendered


def render_logged_turn(tokenizer: Any, call_log: Mapping[str, Any]) -> Turn:
    """Render one native tau2 agent call with the student serving tokenizer."""

    request = call_log.get("request")
    response = call_log.get("response")
    if not isinstance(request, Mapping) or not isinstance(response, Mapping):
        raise ValueError("tau2 LLM log must contain request and response objects")
    messages = _renderer_messages(request.get("messages"))
    raw_tools = request.get("tools")
    tools = copy.deepcopy(raw_tools) if isinstance(raw_tools, list) else []
    prompt = _apply_template(
        tokenizer, messages, tools, add_generation_prompt=True
    )
    assistant = _assistant_from_log(response)
    full = _apply_template(
        tokenizer,
        [*messages, assistant],
        tools,
        add_generation_prompt=False,
    )
    if not full.startswith(prompt):
        raise ValueError(
            "student chat template did not preserve the native serving prompt prefix"
        )
    target = full[len(prompt) :]

    # Prose is retained byte-for-byte by tau2's response log.  Tool-only
    # messages are stored structurally, so serialize them with the serving
    # template and derive/remove only that template's turn terminator.
    content = assistant.get("content")
    if isinstance(content, str) and content.strip():
        target = content
    else:
        sentinel = "__BFAS_TAU2_SENTINEL__"
        dummy = {"role": "assistant", "content": sentinel}
        dummy_full = _apply_template(
            tokenizer,
            [*messages, dummy],
            tools,
            add_generation_prompt=False,
        )
        dummy_continued = _apply_template(
            tokenizer,
            [*messages, dummy],
            tools,
            continue_final_message=True,
        )
        if dummy_full.startswith(dummy_continued):
            closing = dummy_full[len(dummy_continued) :]
            if closing and target.endswith(closing):
                target = target[: -len(closing)]
    if not target:
        raise ValueError("tau2 agent call rendered an empty target")
    return Turn(prompt, target, {"messages": messages, "tools": tools})


def _message_token_count(message: Mapping[str, Any]) -> int:
    usage = message.get("usage")
    if isinstance(usage, Mapping):
        total = usage.get("total_tokens")
        if isinstance(total, int) and not isinstance(total, bool):
            return max(total, 0)
        count = 0
        found = False
        for field in ("prompt_tokens", "completion_tokens"):
            value = usage.get(field)
            if isinstance(value, int) and not isinstance(value, bool):
                count += max(value, 0)
                found = True
        if found:
            return count
    content = message.get("content")
    return math.ceil(len(content) / 4) if isinstance(content, str) else 0


def _side_token_count(simulation: Mapping[str, Any], role: str) -> int:
    messages = simulation.get("messages")
    if not isinstance(messages, list):
        return 0
    total = 0
    for message in messages:
        if not isinstance(message, Mapping) or message.get("role") != role:
            continue
        # Harness-provided initial/message-history entries are not LLM calls.
        # Both native LLM participants retain the provider response in
        # raw_data, even when usage itself is unavailable.
        if message.get("raw_data") is None:
            continue
        total += _message_token_count(message)
    return total


def side_usage(simulation: Mapping[str, Any], role: str) -> dict[str, int]:
    """Sum provider counters; cached input is a subset of prompt tokens.

    Native tau2's summary drops cached counts. The complete provider response
    in raw_data retains them. Never count both representations of one call.
    """
    totals: dict[str, int] = {}
    for message in simulation.get("messages", []) or []:
        if not isinstance(message, Mapping) or message.get("role") != role:
            continue
        raw = message.get("raw_data")
        if raw is None:
            continue
        usage = dict(message.get("usage") or {})
        if isinstance(raw, Mapping) and isinstance(raw.get("usage"), Mapping):
            usage.update(raw["usage"])
        details = usage.get("prompt_tokens_details") or {}
        if isinstance(details, Mapping) and "cached_tokens" in details:
            usage["cached_tokens"] = details["cached_tokens"]
        for key in ("prompt_tokens", "completion_tokens", "cached_tokens"):
            count = usage.get(key)
            if isinstance(count, int) and not isinstance(count, bool) and count >= 0:
                totals[key] = totals.get(key, 0) + count
    return totals


class Tau2Adapter(BenchmarkAdapter):
    """BFAS adapter backed by the official tau2 text harness."""

    name = "tau2"
    needs_server = True

    def __init__(
        self,
        seed: int = 0,
        port: int = 8900,
        served_model_name: str = SERVED_MODEL_NAME,
        base_url: str | None = None,
    ) -> None:
        self.seed = seed
        self.port = port
        self.served_model_name = served_model_name
        self.base_url = (base_url or f"http://localhost:{port}/v1").rstrip("/")
        self._tokenizer: Any = None
        self._loaded_policy: str | None = None
        self._suffix = os.environ.get(
            "BFAS_TAU2_SUFFIX", "<|im_start|>assistant\n"
        )
        self._run_serial = 0

    @staticmethod
    def _domain_dir(domain: str) -> Path:
        return TAU2_DATA / "domains" / domain

    @classmethod
    def _splits(cls, domain: str) -> dict[str, list[str]]:
        path = cls._domain_dir(domain) / "split_tasks.json"
        if not path.is_file():
            return {}
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, Mapping):
            raise ValueError(f"invalid tau2 split file {path}")
        return {
            str(name): [str(task_id) for task_id in task_ids]
            for name, task_ids in value.items()
            if isinstance(task_ids, list)
        }

    @classmethod
    def _all_task_ids(cls, domain: str) -> list[str]:
        path = cls._domain_dir(domain) / "tasks.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, list):
            raise ValueError(f"invalid tau2 task file {path}")
        return [str(task["id"]) for task in value if isinstance(task, Mapping)]

    def task_pool(self) -> list[TaskRef]:
        tasks: list[TaskRef] = []
        for domain in CORE_DOMAINS:
            splits = self._splits(domain)
            native_ids = splits.get("train")
            if native_ids is None:
                native_ids = splits.get("base", self._all_task_ids(domain))
            tasks.extend(
                TaskRef(encode_task_id(domain, task_id), domain)
                for task_id in native_ids
            )
        if not tasks:
            raise FileNotFoundError("the vendored tau2 task sets are unavailable")
        return tasks

    def official_eval_split_disjoint(self) -> bool:
        for domain in CORE_DOMAINS:
            splits = self._splits(domain)
            train = set(splits.get("train", ()))
            test = set(splits.get("test", ()))
            if not train or not test or not train.isdisjoint(test):
                return False
        return True

    def _query_ids(self, domain: str) -> list[str]:
        splits = self._splits(domain)
        train = set(splits.get("train", ()))
        test = splits.get("test")
        if train and test is not None and train.isdisjoint(test):
            return list(test)
        base = splits.get("base", self._all_task_ids(domain))
        support = set(self.support_split().support)
        return [
            task_id
            for task_id in base
            if encode_task_id(domain, task_id) not in support
        ]

    def teacher_name(self) -> str:
        return os.environ.get(
            "BFAS_TAU2_TEACHER_MODEL",
            os.environ.get("BFAS_TAU2_TEACHER", os.environ.get("BFAS_TEACHER", "gpt-5.4")),
        )

    @staticmethod
    def _teacher_endpoint() -> tuple[str, str]:
        base_url = os.environ.get("OLLAMA_BASE_URL", "").strip()
        api_key = os.environ.get("OLLAMA_API_KEY", "")
        if not base_url:
            raise RuntimeError("OLLAMA_BASE_URL is not set")
        if not api_key:
            raise RuntimeError("OLLAMA_API_KEY is not set")
        parsed = urllib.parse.urlsplit(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise RuntimeError("OLLAMA_BASE_URL must be an absolute HTTP(S) URL")
        normalized = base_url.rstrip("/")
        if not normalized.endswith("/v1"):
            normalized += "/v1"
        return normalized, api_key

    @staticmethod
    def _openai_model(model: str) -> str:
        return model if "/" in model else f"openai/{model}"

    def _student_args(self, temperature: float) -> tuple[str, dict[str, Any]]:
        return self._openai_model(self.served_model_name), {
            "temperature": float(temperature),
            "base_url": self.base_url,
            "api_key": "EMPTY",
        }

    @staticmethod
    def _official_openai_args(model: str, temperature: float) -> dict[str, Any]:
        args: dict[str, Any] = {"base_url": OPENAI_BASE, "service_tier": "flex"}
        if model != LUNA_MODEL:
            args["temperature"] = float(temperature)
        return args

    def _teacher_args(self, temperature: float) -> tuple[str, dict[str, Any]]:
        teacher = self.teacher_name()
        if teacher.startswith("azure/"):
            return teacher, {"temperature": float(temperature)}
        if teacher.startswith("openai/"):
            return teacher, self._official_openai_args(teacher, temperature)
        base_url, _ = self._teacher_endpoint()
        return self._openai_model(teacher), {
            "temperature": float(temperature),
            "base_url": base_url,
        }

    @staticmethod
    def _azure_env() -> dict[str, str]:
        """litellm's azure provider credentials, env-only (never in argv/logs).

        Used when BFAS_TAU2_USER_MODEL / BFAS_TAU2_TEACHER is an ``azure/<deployment>``
        name (2026-09-02: both ollama keys rate-limited the user simulator).
        """
        import appworld_teacher

        endpoint, api_key = appworld_teacher._azure_credentials()
        return {
            "AZURE_API_KEY": api_key,
            "AZURE_API_BASE": endpoint.rstrip("/"),
            "AZURE_API_VERSION": appworld_teacher.AZURE_OPENAI_API_VERSION,
        }

    def _user_args(self, temperature: float) -> tuple[str, dict[str, Any]]:
        user_model = os.environ.get("BFAS_TAU2_USER_MODEL", "gpt-5.4")
        if user_model.startswith("azure/"):
            return user_model, {"temperature": float(temperature)}
        if user_model.startswith("openai/"):
            return user_model, self._official_openai_args(user_model, temperature)
        base_url, _ = self._teacher_endpoint()
        return self._openai_model(user_model), {
            "temperature": float(temperature),
            "base_url": base_url,
        }

    @staticmethod
    def _guidance_block(demo: Demo | None) -> str | None:
        if demo is None:
            return None
        return (
            "\n\n<bfas_worked_example>\n"
            + GUIDE_HEADER
            + "\n\n"
            + demo.worked_example
            + "\n</bfas_worked_example>"
        )

    @staticmethod
    def _policy_filename(domain: str) -> str:
        return "main_policy.md" if domain == "telecom" else "policy.md"

    def _make_data_root(
        self, domain: str, guidance_block: str | None
    ) -> tuple[Path, Path]:
        temporary = Path(tempfile.mkdtemp(prefix="bfas-tau2-"))
        data_root = temporary / "data"
        tau_root = data_root / "tau2"
        domain_target = tau_root / "domains" / domain
        domain_target.mkdir(parents=True)
        source_domain = self._domain_dir(domain)
        policy_filename = self._policy_filename(domain)
        for source in source_domain.iterdir():
            target = domain_target / source.name
            if source.name == policy_filename and guidance_block is not None:
                target.write_text(
                    source.read_text(encoding="utf-8") + guidance_block,
                    encoding="utf-8",
                )
            else:
                target.symlink_to(source.resolve(), target_is_directory=source.is_dir())
        (tau_root / "user_simulator").symlink_to(
            (TAU2_DATA / "user_simulator").resolve(), target_is_directory=True
        )
        return temporary, data_root

    def _run_cli(
        self,
        *,
        domain: str,
        task_ids: Sequence[str],
        split: str,
        agent_model: str,
        agent_args: Mapping[str, Any],
        num_trials: int = 1,
        guidance: Demo | None = None,
        max_steps: int | None = None,
        budget_config: Path | None = None,
    ) -> NativeRun:
        if not TAU2_BIN.is_file():
            raise FileNotFoundError(f"vendored tau2 CLI is unavailable: {TAU2_BIN}")
        guidance_block = self._guidance_block(guidance)
        user_model, user_args = self._user_args(
            float(os.environ.get("BFAS_TAU2_USER_TEMPERATURE", "0"))
        )
        temporary, data_root = self._make_data_root(domain, guidance_block)
        if budget_config is not None:
            user_args["metadata"] = {"bfas_purpose": "user_sim"}
        self._run_serial += 1
        run_seed = self.seed + self._run_serial * 10_000
        command = [
            str(TAU2_BIN),
            "run",
            "--domain",
            domain,
            "--agent",
            "llm_agent",
            "--agent-llm",
            agent_model,
            "--agent-llm-args",
            json.dumps(dict(agent_args), separators=(",", ":")),
            "--user",
            "user_simulator",
            "--user-llm",
            user_model,
            "--user-llm-args",
            json.dumps(user_args, separators=(",", ":")),
            "--task-split-name",
            split,
            "--task-ids",
            *map(str, task_ids),
            "--num-trials",
            str(num_trials),
            "--max-concurrency",
            "1" if budget_config else os.environ.get("BFAS_TAU2_MAX_CONCURRENCY", "1"),
            "--max-steps",
            str(max_steps) if max_steps is not None else os.environ.get("BFAS_TAU2_MAX_STEPS", "200"),
            "--seed",
            str(run_seed),
            "--save-to",
            "bfas_native",
            "--verbose-logs",
            "--llm-log-mode",
            "all",
            "--auto-resume",
            "--log-level",
            "ERROR",
        ]
        env = os.environ.copy()
        env["TAU2_DATA_DIR"] = str(data_root)
        official = any(args.get("base_url") == OPENAI_BASE for args in (agent_args, user_args))
        if official and not env.get("OPENAI_API_KEY"):
            shutil.rmtree(temporary, ignore_errors=True)
            raise RuntimeError("OPENAI_API_KEY is required for the official OpenAI channel")
        if official and any(
            name.startswith("openai/") and args.get("base_url") != OPENAI_BASE
            and "api_key" not in args
            for name, args in ((agent_model, agent_args), (user_model, user_args))
        ):
            shutil.rmtree(temporary, ignore_errors=True)
            raise RuntimeError("mixed OpenAI/Ollama channels require separate per-request credentials")
        if not official:
            env["OPENAI_API_KEY"] = os.environ.get("OLLAMA_API_KEY", "EMPTY")
        env["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        judge_usage_path = data_root / "bfas_judge_usage.jsonl"
        if budget_config is not None or self.teacher_name() == LUNA_MODEL:
            # Execute the unchanged vendored CLI under a per-request budget guard.
            command = [str(TAU2_BIN.with_name("python")), str(ROOT / "tools/tau2_guarded_cli.py"), *command]
            env.pop("BFAS_TAU2_BUDGET_CONFIG", None)
            env["BFAS_TAU2_JUDGE_USAGE_PATH"] = str(judge_usage_path)
            if budget_config is not None:
                env["BFAS_TAU2_BUDGET_CONFIG"] = str(budget_config)
        if any(
            name.startswith("azure/")
            for name in (agent_model, user_model, self.teacher_name())
        ):
            env.update(self._azure_env())
        try:
            subprocess.run(command, cwd=TAU2_ROOT, env=env, check=True)
            run_dir = data_root / "simulations/bfas_native"
            results_path = run_dir / "results.json"
            results = json.loads(results_path.read_text(encoding="utf-8"))
            if not isinstance(results, dict):
                raise ValueError(f"tau2 produced invalid results at {results_path}")
            if judge_usage_path.exists():
                judges = [json.loads(line) for line in judge_usage_path.read_text().splitlines()]
                for simulation in self._simulations(results):
                    simulation["bfas_judge_usage"] = [
                        row["usage"] for row in judges if row["simulation_id"] == simulation.get("id")
                    ]
            return NativeRun(
                results=results,
                run_dir=run_dir,
                cleanup_root=temporary,
                guidance_block=guidance_block,
            )
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise

    @staticmethod
    def _simulations(results: Mapping[str, Any]) -> list[Mapping[str, Any]]:
        simulations = results.get("simulations")
        if not isinstance(simulations, list):
            return []
        return [item for item in simulations if isinstance(item, Mapping)]

    @staticmethod
    def _agent_logs(run_dir: Path, simulation_id: str) -> list[dict[str, Any]]:
        files = sorted(
            run_dir.glob(
                f"artifacts/**/sim_{simulation_id}/llm_debug/*_agent_response_*.json"
            )
        )
        logs: list[dict[str, Any]] = []
        for path in files:
            value = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(value, dict):
                logs.append(value)
        logs.sort(key=lambda item: str(item.get("timestamp", "")))
        return logs

    def _render_logs(
        self,
        run_dir: Path,
        simulation: Mapping[str, Any],
        guidance_block: str | None,
    ) -> tuple[list[Turn], list[Turn]]:
        if self._tokenizer is None:
            raise RuntimeError("student tokenizer is not prepared")
        simulation_id = str(simulation.get("id", ""))
        logs = self._agent_logs(run_dir, simulation_id)
        messages = simulation.get("messages")
        generated_messages = (
            [
                message
                for message in messages
                if isinstance(message, Mapping)
                and message.get("role") == "assistant"
                and message.get("raw_data") is not None
            ]
            if isinstance(messages, list)
            else []
        )
        if generated_messages and not logs:
            raise RuntimeError(
                f"tau2 simulation {simulation_id!r} has agent messages but no "
                "native verbose call logs"
            )
        turns: list[Turn] = []
        for index, log in enumerate(logs):
            try:
                turns.append(render_logged_turn(self._tokenizer, log))
            except ValueError as exc:
                # One unrenderable call must not sink a multi-hour collection
                # (2026-09-02: a guided tau2 batch died on a prefix mismatch and
                # the temporary run dir was gone before anyone could look).
                # Keep the raw call for diagnosis and skip the turn.
                dump_dir = ROOT / "logs/bfas/tau2_render_failures"
                dump_dir.mkdir(parents=True, exist_ok=True)
                dump = dump_dir / f"{simulation_id or 'sim'}_{index}.json"
                dump.write_text(
                    json.dumps({"error": str(exc), "call_log": log}, ensure_ascii=False, indent=1),
                    encoding="utf-8",
                )
                print(
                    f"[tau2][render] skipped call {index} of {simulation_id!r}: {exc} "
                    f"(dumped to {dump})",
                    flush=True,
                )
        if guidance_block is None:
            return turns, list(turns)

        deployment_turns: list[Turn] = []
        for turn in turns:
            context = copy.deepcopy(turn.context)
            found = False
            for message in context["messages"]:
                if message.get("role") != "system":
                    continue
                content = message.get("content")
                if isinstance(content, str) and guidance_block in content:
                    message["content"] = content.replace(guidance_block, "", 1)
                    found = True
                    break
            if not found:
                raise RuntimeError(
                    "guided tau2 request log does not contain the injected example"
                )
            deployment_turns.append(
                Turn(self._render(context), turn.target, context)
            )
        return turns, deployment_turns

    def _render(self, context: Mapping[str, Any]) -> str:
        if self._tokenizer is None:
            raise RuntimeError("student tokenizer is not prepared")
        messages = context.get("messages")
        tools = context.get("tools", [])
        if not isinstance(messages, list) or not isinstance(tools, list):
            raise ValueError("tau2 render context must contain messages and tools")
        rendered = _apply_template(
            self._tokenizer, messages, tools, add_generation_prompt=True
        )
        plain = _apply_template(
            self._tokenizer, messages, tools, add_generation_prompt=False
        )
        if rendered.startswith(plain) and len(rendered) > len(plain):
            self._suffix = rendered[len(plain) :]
        elif not rendered.endswith(self._suffix):
            raise ValueError("could not identify tau2's generation marker")
        return rendered

    def prepare_renderer(self, policy_ref: PolicyRef) -> None:
        policy = str(policy_ref)
        if self._loaded_policy != policy or self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                policy, trust_remote_code=False
            )
            self._loaded_policy = policy
        self._render({"messages": [{"role": "user", "content": ""}], "tools": []})

    def _record_user_sim_usage(
        self, domain: str, simulations: Sequence[Mapping[str, Any]]
    ) -> None:
        # append_episode uses one O_APPEND write, so it is safe to use while
        # acquire_demos holds the purchase lock for the teacher-agent record.
        from ..ledger import append_episode

        temperature = float(os.environ.get("BFAS_TAU2_USER_TEMPERATURE", "0"))
        for simulation in simulations:
            native_id = simulation.get("task_id")
            if native_id is None:
                continue
            append_episode(
                self.name,
                task_id=encode_task_id(domain, str(native_id)),
                teacher=os.environ.get("BFAS_TAU2_USER_MODEL", "gpt-5.4"),
                attempt_index=self._run_serial,
                temperature=temperature,
                verified=False,
                tokens_spent=_side_token_count(simulation, "user"),
                usage=side_usage(simulation, "user"),
                purpose="user_sim",
            )
            for usage in simulation.get("bfas_judge_usage", []):
                append_episode(
                    self.name, task_id=encode_task_id(domain, str(native_id)),
                    teacher=self.teacher_name(), attempt_index=self._run_serial,
                    temperature=0.0, verified=False, purpose="teacher_judge",
                    tokens_spent=usage["prompt_tokens"] + usage["completion_tokens"],
                    usage=usage,
                )

    def _rollouts_from_run(
        self,
        domain: str,
        requested: Sequence[str],
        native_run: NativeRun,
        guided: bool,
    ) -> list[Rollout]:
        simulations = self._simulations(native_run.results)
        self._record_user_sim_usage(domain, simulations)
        by_id = {str(item.get("task_id")): item for item in simulations}
        verdicts = extract_verdicts(native_run.results, requested)
        output: list[Rollout] = []
        for native_id in requested:
            task_id = encode_task_id(domain, native_id)
            simulation = by_id.get(native_id)
            turns: list[Turn] = []
            deployment_turns: list[Turn] = []
            if simulation is not None:
                turns, deployment_turns = self._render_logs(
                    native_run.run_dir,
                    simulation,
                    native_run.guidance_block,
                )
            verified = verdicts.get(native_id, False)
            output.append(
                Rollout(
                    task_id,
                    verified,
                    turns,
                    {
                        "checker_verified": verified,
                        "category": domain,
                        "simulation": copy.deepcopy(simulation),
                        "guided": guided,
                        "deployment_turns": deployment_turns,
                        "user_sim_tokens": (
                            _side_token_count(simulation, "user")
                            if simulation is not None
                            else 0
                        ),
                    },
                )
            )
        return output

    def rollout(
        self,
        policy: PolicyRef,
        task_ids: Sequence[str],
        temperature: float,
        guided_demos: Mapping[str, Demo] | None = None,
    ) -> list[Rollout]:
        self.prepare_renderer(policy)
        grouped: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for task_id in task_ids:
            domain, native_id = decode_task_id(task_id)
            grouped[domain].append((task_id, native_id))
        student_model, student_args = self._student_args(temperature)
        output: dict[str, Rollout] = {}
        demos = guided_demos or {}
        for domain, pairs in grouped.items():
            unguided = [
                native_id
                for task_id, native_id in pairs
                if task_id not in demos
            ]
            if unguided:
                native_run = self._run_cli(
                    domain=domain,
                    task_ids=unguided,
                    split="train" if "train" in self._splits(domain) else "base",
                    agent_model=student_model,
                    agent_args=student_args,
                )
                try:
                    for rollout in self._rollouts_from_run(
                        domain, unguided, native_run, guided=False
                    ):
                        output[rollout.task_id] = rollout
                finally:
                    native_run.cleanup()
            for task_id, native_id in pairs:
                demo = demos.get(task_id)
                if demo is None:
                    continue
                native_run = self._run_cli(
                    domain=domain,
                    task_ids=[native_id],
                    split="train" if "train" in self._splits(domain) else "base",
                    agent_model=student_model,
                    agent_args=student_args,
                    guidance=demo,
                )
                try:
                    rollout = self._rollouts_from_run(
                        domain, [native_id], native_run, guided=True
                    )[0]
                    output[task_id] = rollout
                finally:
                    native_run.cleanup()
        return [output[task_id] for task_id in task_ids]

    def teacher_demo(
        self, task_ids: Sequence[str], attempts: int
    ) -> dict[str, Demo]:
        """Acquire demos only through BFAS's authoritative purchase ledger."""

        if self._tokenizer is None:
            self.prepare_renderer(os.environ.get("BFAS_TAU2_MODEL", DEFAULT_STUDENT))
        from ..ledger import acquire_demos

        return dict(acquire_demos(self.name, self, task_ids, attempts))

    @staticmethod
    def _worked_example(simulation: Mapping[str, Any], turns: Sequence[Turn]) -> str:
        messages = simulation.get("messages")
        lines: list[str] = []
        if isinstance(messages, list):
            for message in messages:
                if not isinstance(message, Mapping):
                    continue
                role = str(message.get("role", "message"))
                content = message.get("content")
                if isinstance(content, str) and content.strip():
                    lines.append(f"{role}: {content.strip()}")
                calls = message.get("tool_calls")
                if isinstance(calls, list) and calls:
                    lines.append(
                        f"{role} tool calls: "
                        + json.dumps(calls, ensure_ascii=False, separators=(",", ":"))
                    )
        if not lines:
            lines = [f"assistant: {turn.target}" for turn in turns]
        limit = int(os.environ.get("BFAS_TAU2_DEMO_CHARS", "6000"))
        return "\n".join(lines)[-limit:]

    def teacher_episode(
        self, task_id: str, attempt_index: int, temperature: float
    ) -> TeacherEpisode:
        if self._tokenizer is None:
            raise RuntimeError(
                "prepare the student renderer before purchasing tau2 demonstrations"
            )
        domain, native_id = decode_task_id(task_id)
        teacher_model, teacher_args = self._teacher_args(temperature)
        native_run = self._run_cli(
            domain=domain,
            task_ids=[native_id],
            split="train" if "train" in self._splits(domain) else "base",
            agent_model=teacher_model,
            agent_args=teacher_args,
        )
        try:
            simulations = self._simulations(native_run.results)
            self._record_user_sim_usage(domain, simulations)
            simulation = next(
                (
                    item
                    for item in simulations
                    if str(item.get("task_id")) == native_id
                ),
                None,
            )
            verified = simulation is not None and extract_verdict(simulation)
            turns: list[Turn] = []
            # Failed teacher episodes still return exact provider usage so the
            # purchase gateway charges them. Rendering is only needed for a
            # verified trajectory that can become a demo.
            if simulation is not None and verified:
                turns, _ = self._render_logs(
                    native_run.run_dir, simulation, guidance_block=None
                )
            demo = None
            if verified and simulation is not None:
                demo = Demo(
                    task_id,
                    turns,
                    self._worked_example(simulation, turns),
                    {
                        "attempt": attempt_index + 1,
                        "checker_verified": True,
                        "simulation": copy.deepcopy(simulation),
                    },
                )
            return TeacherEpisode(
                task_id=task_id,
                verified=verified,
                demo=demo,
                response_texts=tuple(turn.target for turn in turns),
                tokens_spent=(
                    _side_token_count(simulation, "assistant")
                    if simulation is not None
                    else 0
                ),
                usage=side_usage(simulation, "assistant") if simulation is not None else {},
            )
        finally:
            native_run.cleanup()

    def evaluate(self, policy_ref: PolicyRef, out_dir: Path) -> dict[str, Any]:
        trials = int(os.environ.get("BFAS_TAU2_EVAL_TRIALS", "1"))
        if trials <= 0:
            raise ValueError("BFAS_TAU2_EVAL_TRIALS must be positive")
        student_model, student_args = self._student_args(0.0)
        per_category: dict[str, float] = {}
        total_passes = 0
        total_simulations = 0
        out_dir.mkdir(parents=True, exist_ok=True)
        for domain in CORE_DOMAINS:
            query_ids = self._query_ids(domain)
            split = "test" if query_ids == self._splits(domain).get("test") else "base"
            native_run = self._run_cli(
                domain=domain,
                task_ids=query_ids,
                split=split,
                agent_model=student_model,
                agent_args=student_args,
                num_trials=trials,
            )
            try:
                simulations = self._simulations(native_run.results)
                self._record_user_sim_usage(domain, simulations)
                per_category[domain] = official_pass_one(native_run.results)
                evaluated = [
                    simulation
                    for simulation in simulations
                    if simulation.get("termination_reason") != "infrastructure_error"
                ]
                total_passes += sum(extract_verdict(item) for item in evaluated)
                total_simulations += len(evaluated)
                (out_dir / f"tau2_{domain}_results.json").write_text(
                    json.dumps(native_run.results, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
            finally:
                native_run.cleanup()
        # Every domain uses the same trial count, but task counts differ.  The
        # official aggregate is therefore the query-task-weighted pass^1.
        headline = (
            total_passes / total_simulations if total_simulations else 0.0
        )
        metrics = {
            "headline": headline,
            "pass^1": headline,
            "per_category": per_category,
            "passed": total_passes,
            "total": total_simulations,
        }
        (out_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
        )
        return metrics

    def serving_probe(self) -> None:
        body = json.dumps({
            "model": self.served_model_name,
            "messages": [{"role": "user", "content": "Reply with OK."}],
            "temperature": 0,
            "max_tokens": 1,
        }).encode("utf-8")
        request = urllib.request.Request(
            f"http://localhost:{self.port}/v1/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer "
                + os.environ.get("BFAS_STUDENT_API_KEY", "EMPTY"),
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            if response.status != 200:
                raise RuntimeError(
                    f"tau2 student serving probe returned HTTP {response.status}"
                )
            value = json.loads(response.read().decode("utf-8"))
        if not isinstance(value, Mapping) or not value.get("choices"):
            raise RuntimeError("tau2 student serving probe returned no completion")

    def generation_suffix(self) -> str:
        return self._suffix

    def target_policy(self, category: str) -> str:
        if category not in CORE_DOMAINS:
            raise ValueError(f"unknown tau2 category {category!r}")
        return "prose_ok"

    def rerender(self, row: Mapping[str, Any]) -> str:
        context = row.get("_render_context")
        if not isinstance(context, Mapping):
            raise ValueError("row lacks tau2 render context")
        return self._render(context)


__all__ = [
    "CORE_DOMAINS",
    "NativeRun",
    "Tau2Adapter",
    "decode_task_id",
    "encode_task_id",
    "extract_verdict",
    "extract_verdicts",
    "official_pass_one",
    "render_logged_turn",
]
