"""AppWorld adapter backed by the OFFICIAL scaffold (stonybrooknlp/appworld 0.2,
``simplified_react_code_agent``), replacing the in-house harness in
``adapters/appworld.py``.

Why (2026-09-01): under the in-house harness the 4B base scored 0/40 and even the
teacher only 7/40, so every AppWorld number was harness-limited. Under the official
ReAct scaffold the same base scores 14.0% TGC on dev. This adapter therefore runs
EVERYTHING -- student rollouts, teacher demonstrations and the final evaluation --
through the official agent and evaluator, exactly like ``adapters/tau2.py`` runs
tau2 through its official harness:

* the student is served by the shared vLLM lane and reached through the agent's
  OpenAI client (``OPENAI_BASE_URL``/``OPENAI_API_KEY`` in the subprocess env);
* the teacher is the same agent pointed at the teacher endpoint;
* each agent call is logged natively (``tasks/<id>/logs/lm_calls.jsonl``) and is
  rendered with the student's chat template into training ``Turn``s;
* ``verified`` is the official evaluator's per-task ``success``.

Runs live in ``envs/appworld-official`` (data 0.2.0) and ``envs/appworld-repo``
(agent code); nothing here touches the legacy ``envs/appworld-venv``.
"""
from __future__ import annotations

import copy
import json
import os
import re
import shutil
import subprocess
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
    SupportSplit,
    TaskRef,
    TeacherEpisode,
    Turn,
)

ROOT = Path(__file__).resolve().parents[3]
REPO = ROOT / "envs/appworld-repo"
OFFICIAL_BIN = ROOT / "envs/appworld-official/.venv/bin/appworld"
DATA = ROOT / "envs/appworld-official/data"
AGENT = "simplified_react_code_agent"
CONFIG_DIR = REPO / "experiments/configs" / AGENT / "bfas"
PROMPT_DIR = REPO / "experiments/prompts/react_code_agent"
OFFICIAL_PROMPT = PROMPT_DIR / "instructions.txt"
OUTPUTS = REPO / "experiments/outputs"
SAMPLING_TEMPERATURE = 0.7
GUIDANCE_HEADER = (
    "\n\nWorked example from an expert on this task. Study the approach, then "
    "solve the task yourself step by step:\n"
)


@dataclass
class OfficialRun:
    """One ``appworld run`` invocation: where its outputs and verdicts live."""

    experiment_name: str
    dataset_name: str
    outputs_dir: Path
    evaluation: dict[str, Any]
    guidance_block: str | None = None
    prompt_file: Path | None = None
    config_file: Path | None = None
    dataset_file: Path | None = None

    def success(self, task_id: str) -> bool:
        item = self.evaluation.get("individual", {}).get(task_id)
        return bool(isinstance(item, Mapping) and item.get("success") is True)

    def cleanup(self, keep_outputs: bool) -> None:
        for path in (self.prompt_file, self.config_file, self.dataset_file):
            if path is not None:
                path.unlink(missing_ok=True)
        if not keep_outputs:
            shutil.rmtree(self.outputs_dir, ignore_errors=True)


def read_lm_calls(outputs_dir: Path, task_id: str) -> list[dict[str, Any]]:
    path = outputs_dir / "tasks" / task_id / "logs/lm_calls.jsonl"
    if not path.is_file():
        return []
    calls: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                calls.append(json.loads(line))
    return calls


def call_output_text(call: Mapping[str, Any]) -> str:
    output = call.get("output")
    if isinstance(output, Mapping):
        choices = output.get("choices")
        if isinstance(choices, list) and choices:
            message = choices[0].get("message") if isinstance(choices[0], Mapping) else None
            if isinstance(message, Mapping):
                content = message.get("content")
                return content if isinstance(content, str) else ""
        content = output.get("content")
        if isinstance(content, str):
            return content
    if isinstance(output, str):
        return output
    return ""


def call_token_usage(call: Mapping[str, Any]) -> int:
    output = call.get("output")
    if not isinstance(output, Mapping):
        return 0
    usage = output.get("usage")
    if not isinstance(usage, Mapping):
        return 0
    total = usage.get("total_tokens")
    if isinstance(total, int) and not isinstance(total, bool):
        return max(total, 0)
    count = 0
    for key in ("prompt_tokens", "completion_tokens"):
        value = usage.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            count += max(value, 0)
    return count


def call_messages(call: Mapping[str, Any]) -> list[dict[str, Any]]:
    request = call.get("input")
    messages = request.get("messages") if isinstance(request, Mapping) else None
    if not isinstance(messages, list):
        raise ValueError("official AppWorld call log lacks input.messages")
    return [
        {"role": str(m.get("role")), "content": str(m.get("content") or "")}
        for m in messages
        if isinstance(m, Mapping)
    ]


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)[:80]


class AppWorldOfficialAdapter(BenchmarkAdapter):
    name = "appworld"
    server_backed = True

    def __init__(self, seed: int = 0, port: int = 8930):
        self.seed = seed
        self.port = port
        self.served_model_name = os.environ.get("BFAS_APPWORLD_SERVED_NAME", "bfas-policy")
        self._loaded_policy: str | None = None
        self._tokenizer: Any = None
        self._tasks: list[TaskRef] | None = None
        self._suffix = os.environ.get("BFAS_APPWORLD_SUFFIX", "<|im_start|>assistant\n")
        self._run_serial = 0
        self.keep_outputs = os.environ.get("BFAS_APPWORLD_KEEP_RUNS") == "1"

    # ----------------------------------------------------------------- pool
    @staticmethod
    def _dataset_ids(name: str) -> list[str]:
        path = DATA / "datasets" / f"{name}.txt"
        if not path.is_file():
            raise FileNotFoundError(f"official AppWorld dataset missing: {path}")
        return [line.strip() for line in path.read_text().splitlines() if line.strip()]

    def task_pool(self) -> list[TaskRef]:
        if self._tasks is None:
            self._tasks = [
                TaskRef(task_id, task_id.rsplit("_", 1)[0])
                for task_id in sorted(self._dataset_ids("train"))
            ]
        return list(self._tasks)

    def support_split(self) -> SupportSplit:
        path = ROOT / "configs/support_split.json"
        if not path.is_file():
            return super().support_split()
        split = json.loads(path.read_text())
        support = tuple(sorted(map(str, split["support"])))
        demand = tuple(sorted(map(str, split["demand"])))
        calibration = tuple(sorted(map(str, split["calibration"])))
        if set(demand) | set(calibration) != set(support):
            raise ValueError("configs/support_split.json does not partition support")
        unknown = set(support) - {task.task_id for task in self.task_pool()}
        if unknown:
            raise ValueError(f"configs/support_split.json has unknown tasks: {sorted(unknown)}")
        return SupportSplit(support, demand, calibration)

    def official_eval_split_disjoint(self) -> bool:
        return True

    # ------------------------------------------------------------ rendering
    def prepare_renderer(self, policy_ref: PolicyRef) -> None:
        policy = str(policy_ref)
        if self._loaded_policy != policy or self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(policy, trust_remote_code=False)
            self._loaded_policy = policy
        self._render({"messages": [{"role": "user", "content": ""}]})

    def _render(self, context: Mapping[str, Any]) -> str:
        if self._tokenizer is None:
            raise RuntimeError("student tokenizer is not prepared")
        messages = context.get("messages")
        if not isinstance(messages, list):
            raise ValueError("AppWorld render context must contain messages")
        rendered = self._tokenizer.apply_chat_template(
            list(messages), tokenize=False, add_generation_prompt=True
        )
        plain = self._tokenizer.apply_chat_template(
            list(messages), tokenize=False, add_generation_prompt=False
        )
        if rendered.startswith(plain) and len(rendered) > len(plain):
            self._suffix = rendered[len(plain):]
        elif not rendered.endswith(self._suffix):
            raise ValueError("could not identify the student's generation marker")
        return rendered

    def _turns_from_calls(
        self, calls: Sequence[Mapping[str, Any]], guidance_block: str | None
    ) -> tuple[list[Turn], list[Turn]]:
        turns: list[Turn] = []
        deployment: list[Turn] = []
        for call in calls:
            messages = call_messages(call)
            target = call_output_text(call)
            if not target:
                continue
            context = {"messages": messages}
            turns.append(Turn(self._render(context), target, context))
            if guidance_block is None:
                deployment.append(turns[-1])
                continue
            plain = copy.deepcopy(context)
            found = False
            for message in plain["messages"]:
                if message["role"] == "user" and guidance_block in message["content"]:
                    message["content"] = message["content"].replace(guidance_block, "", 1)
                    found = True
                    break
            if not found:
                raise RuntimeError("guided AppWorld call log lacks the injected example")
            deployment.append(Turn(self._render(plain), target, plain))
        return turns, deployment

    # ------------------------------------------------------------ official run
    @staticmethod
    def _guidance_block(demo: Demo | None) -> str | None:
        if demo is None:
            return None
        return GUIDANCE_HEADER + demo.worked_example

    def _write_prompt_file(self, run_name: str, guidance_block: str) -> Path:
        base = OFFICIAL_PROMPT.read_text(encoding="utf-8")
        path = PROMPT_DIR / f"bfas_{run_name}.txt"
        # The worked example goes at the end of the final USER message, after
        # the task instruction, so the official instructions stay byte-identical.
        path.write_text(base.rstrip("\n") + guidance_block + "\n", encoding="utf-8")
        return path

    @staticmethod
    def _config_text(
        *, model_name: str, temperature: float, seed: int, prompt_file: Path,
        dataset_name: str, max_steps: int, client_name: str = "openai",
    ) -> str:
        config = {
            "type": "simplified",
            "config": {
                "agent": {
                    "type": AGENT,
                    "model_config": {
                        "client_name": client_name,
                        "api_type": "chat_completions",
                        "name": model_name,
                        "temperature": temperature,
                        "seed": seed,
                        "drop_reasoning_content": True,
                        "cost_per_token": {
                            "input_cache_hit": 0.0, "input_cache_miss": 0.0,
                            "input_cache_write": 0.0, "output": 0.0,
                        },
                        # Azure deployments are shared with the ALFWorld teacher
                        # lane and rate-limit per minute; wait, don't abort.
                        "retry_after_n_seconds": 15,
                        "use_cache": False,
                        "max_retries": 60,
                    },
                    "appworld_config": {"random_seed": seed, "raise_on_extra_parameters": True},
                    "logger_config": {"color": False, "verbose": False},
                    "usage_tracker_config": {
                        "max_cost_overall": 1000, "max_cost_per_task": 10,
                        "max_output_tokens_per_task": 100000,
                    },
                    "prompt_file_path": str(prompt_file),
                    "ignore_multiple_calls": True,
                    "max_prompt_length": None,
                    "max_output_length": None,
                    "max_steps": max_steps,
                    "log_lm_calls": True,
                    "skip_if_finished": False,
                },
                "dataset": dataset_name,
            },
            "metadata": {
                "model": {"file_name": model_name, "humanized_name": model_name,
                          "precise_name": model_name, "creator": "bfas", "provider": "bfas"},
                "agent": {"file_name": AGENT, "humanized_name": "ReAct Code Agent"},
            },
        }
        return json.dumps(config, indent=1)

    def _run_official(
        self,
        *,
        task_ids: Sequence[str],
        model_name: str,
        base_url: str,
        api_key: str,
        temperature: float,
        label: str,
        guidance: Demo | None = None,
        num_processes: int | None = None,
        client_name: str = "openai",
        extra_env: Mapping[str, str] | None = None,
    ) -> OfficialRun:
        if not OFFICIAL_BIN.is_file():
            raise FileNotFoundError(f"official AppWorld CLI is unavailable: {OFFICIAL_BIN}")
        self._run_serial += 1
        run_name = _safe_name(f"s{self.seed}_{label}_{self._run_serial}_{os.getpid()}")
        dataset_name = f"bfas_{run_name}"
        dataset_file = DATA / "datasets" / f"{dataset_name}.txt"
        dataset_file.write_text("\n".join(task_ids) + "\n", encoding="utf-8")
        guidance_block = self._guidance_block(guidance)
        prompt_file = (
            self._write_prompt_file(run_name, guidance_block)
            if guidance_block is not None
            else None
        )
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        config_file = CONFIG_DIR / f"{run_name}.jsonnet"
        config_file.write_text(
            self._config_text(
                model_name=model_name,
                temperature=temperature,
                seed=self.seed + self._run_serial,
                prompt_file=prompt_file or OFFICIAL_PROMPT,
                dataset_name=dataset_name,
                max_steps=int(os.environ.get("BFAS_APPWORLD_MAX_STEPS", "50")),
                client_name=client_name,
            ),
            encoding="utf-8",
        )
        experiment_name = f"{AGENT}/bfas/{run_name}"
        outputs_dir = OUTPUTS / experiment_name
        processes = num_processes or int(os.environ.get("BFAS_APPWORLD_PROCESSES", "4"))
        command = [
            str(OFFICIAL_BIN), "run", experiment_name,
            "--root", str(REPO), "--without-setup", "--clear-first",
            "--num-processes", str(max(1, min(processes, len(task_ids)))),
        ]
        env = os.environ.copy()
        # The official agent's OpenAI client only reads these two variables;
        # the config's own base_url/api_key are ignored on that path.
        env["OPENAI_BASE_URL"] = base_url
        env["OPENAI_API_KEY"] = api_key
        if extra_env:
            env.update(extra_env)
        log_path = ROOT / "logs/bfas/appworld_official" / f"{run_name}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with log_path.open("w") as log:
                subprocess.run(command, cwd=REPO, env=env, check=True, stdout=log, stderr=subprocess.STDOUT)
            evaluation_path = outputs_dir / "evaluations" / f"{dataset_name}.json"
            evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        except BaseException:
            for path in (prompt_file, config_file, dataset_file):
                if path is not None:
                    path.unlink(missing_ok=True)
            raise
        return OfficialRun(
            experiment_name=experiment_name,
            dataset_name=dataset_name,
            outputs_dir=outputs_dir,
            evaluation=evaluation,
            guidance_block=guidance_block,
            prompt_file=prompt_file,
            config_file=config_file,
            dataset_file=dataset_file,
        )

    def _student_endpoint(self) -> tuple[str, str, str]:
        return (
            self.served_model_name,
            f"http://localhost:{self.port}/v1",
            os.environ.get("BFAS_STUDENT_API_KEY", "EMPTY"),
        )

    @staticmethod
    def _teacher_endpoint() -> tuple[str, str, str, str, dict[str, str]]:
        """(model name for the agent config, base_url, api_key, client_name, extra env).

        ollama-hosted teachers go through the agent's OpenAI client; Azure
        deployments (gpt-5.6-luna / gpt-5.4, same set as appworld_teacher)
        go through litellm's azure provider, which is what the official
        experiments use for non-OpenAI hosts. Credentials only ever enter the
        subprocess environment, never the config file or the call logs.
        """
        import appworld_teacher

        model = os.environ.get("BFAS_TEACHER", "gpt-5.4")
        if model in appworld_teacher.AZURE_OPENAI_MODELS:
            endpoint, api_key = appworld_teacher._azure_credentials()
            extra = {
                "AZURE_API_KEY": api_key,
                "AZURE_API_BASE": endpoint.rstrip("/"),
                "AZURE_API_VERSION": appworld_teacher.AZURE_OPENAI_API_VERSION,
            }
            return f"azure/{model}", "", "", "litellm", extra
        base_url = os.environ.get("OLLAMA_BASE_URL", "").strip()
        api_key = os.environ.get("OLLAMA_API_KEY", "")
        if not base_url or not api_key:
            raise RuntimeError("OLLAMA_BASE_URL and OLLAMA_API_KEY are required for the teacher")
        return model, base_url.rstrip("/") + "/v1", api_key, "openai", {}

    def _rollouts_from_run(
        self, task_ids: Sequence[str], run: OfficialRun, guided: bool
    ) -> list[Rollout]:
        output: list[Rollout] = []
        for task_id in task_ids:
            calls = read_lm_calls(run.outputs_dir, task_id)
            turns, deployment = self._turns_from_calls(calls, run.guidance_block)
            verified = run.success(task_id)
            output.append(
                Rollout(
                    task_id,
                    verified,
                    turns,
                    {
                        "checker_verified": verified,
                        "category": task_id.rsplit("_", 1)[0],
                        "guided": guided,
                        "deployment_turns": deployment,
                        "steps": len(calls),
                        "tokens": sum(call_token_usage(call) for call in calls),
                    },
                )
            )
        return output

    # --------------------------------------------------------------- protocol
    def rollout(
        self,
        policy: PolicyRef,
        task_ids: Sequence[str],
        temperature: float,
        guided_demos: Mapping[str, Demo] | None = None,
    ) -> list[Rollout]:
        self.prepare_renderer(policy)
        demos = guided_demos or {}
        model, base_url, api_key = self._student_endpoint()
        output: dict[str, Rollout] = {}
        unguided = [task_id for task_id in task_ids if task_id not in demos]
        if unguided:
            run = self._run_official(
                task_ids=unguided, model_name=model, base_url=base_url,
                api_key=api_key, temperature=temperature, label="roll",
            )
            try:
                for rollout in self._rollouts_from_run(unguided, run, guided=False):
                    output[rollout.task_id] = rollout
            finally:
                run.cleanup(self.keep_outputs)
        for task_id in task_ids:
            demo = demos.get(task_id)
            if demo is None:
                continue
            run = self._run_official(
                task_ids=[task_id], model_name=model, base_url=base_url,
                api_key=api_key, temperature=temperature, label="guided",
                guidance=demo, num_processes=1,
            )
            try:
                output[task_id] = self._rollouts_from_run([task_id], run, guided=True)[0]
            finally:
                run.cleanup(self.keep_outputs)
        return [output[task_id] for task_id in task_ids]

    def teacher_demo(self, task_ids: Sequence[str], attempts: int) -> dict[str, Demo]:
        demos: dict[str, Demo] = {}
        for task_id in task_ids:
            for attempt_index in range(attempts):
                temperature = 0.0 if attempt_index == 0 else SAMPLING_TEMPERATURE
                episode = self.teacher_episode(task_id, attempt_index, temperature)
                if episode.demo is not None:
                    demos[task_id] = episode.demo
                    break
        return demos

    def teacher_episode(
        self, task_id: str, attempt_index: int, temperature: float
    ) -> TeacherEpisode:
        if self._tokenizer is None:
            raise RuntimeError("prepare the student renderer before purchasing demonstrations")
        model, base_url, api_key, client_name, extra_env = self._teacher_endpoint()
        run = self._run_official(
            task_ids=[task_id], model_name=model, base_url=base_url, api_key=api_key,
            temperature=temperature, label=f"teacher{attempt_index + 1}", num_processes=1,
            client_name=client_name, extra_env=extra_env,
        )
        try:
            calls = read_lm_calls(run.outputs_dir, task_id)
            verified = run.success(task_id)
            tokens = sum(call_token_usage(call) for call in calls)
            turns: list[Turn] = []
            demo = None
            if verified:
                turns, _ = self._turns_from_calls(calls, None)
                worked = "\n".join(turn.target for turn in turns)[
                    -int(os.environ.get("BFAS_APPWORLD_DEMO_CHARS", "6000")):
                ]
                demo = Demo(
                    task_id, turns, worked,
                    {"attempt": attempt_index + 1, "checker_verified": True, "steps": len(calls)},
                )
            return TeacherEpisode(
                task_id=task_id, verified=verified, demo=demo,
                response_texts=tuple(turn.target for turn in turns),
                tokens_spent=tokens,
            )
        finally:
            run.cleanup(self.keep_outputs)

    def evaluate(self, policy_ref: PolicyRef, out_dir: Path) -> dict[str, Any]:
        split = os.environ.get("BFAS_APPWORLD_EVAL_SPLIT", "dev")
        task_ids = self._dataset_ids(split)
        limit = os.environ.get("BFAS_APPWORLD_EVAL_TASKS")
        if limit:
            task_ids = task_ids[: int(limit)]
        model, base_url, api_key = self._student_endpoint()
        run = self._run_official(
            task_ids=task_ids, model_name=model, base_url=base_url, api_key=api_key,
            temperature=0.0, label=f"eval_{split}",
        )
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(run.outputs_dir / "evaluations" / f"{run.dataset_name}.json",
                         out_dir / "evaluation.json")
            records = []
            grouped: dict[str, list[bool]] = defaultdict(list)
            for task_id in task_ids:
                success = run.success(task_id)
                calls = read_lm_calls(run.outputs_dir, task_id)
                records.append({"task_id": task_id, "success": success, "steps_used": len(calls)})
                grouped[task_id.rsplit("_", 1)[0]].append(success)
            with (out_dir / "records.jsonl").open("w", encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(record) + "\n")
            aggregate = run.evaluation.get("aggregate", {})
            tgc = float(aggregate.get("task_goal_completion", 0.0)) / 100.0
            sgc = float(aggregate.get("scenario_goal_completion", 0.0)) / 100.0
            metrics = {
                "n_tasks": len(task_ids), "success_rate": tgc, "tgc": tgc, "sgc": sgc,
                "split": split, "scaffold": "official_simplified_react_code_agent",
                "mean_steps": (sum(r["steps_used"] for r in records) / len(records)) if records else 0.0,
            }
            (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=1))
            return {
                "headline": tgc, "tgc": tgc, "sgc": sgc,
                "per_category": {k: sum(v) / len(v) for k, v in grouped.items()},
            }
        finally:
            run.cleanup(keep_outputs=True)

    def serving_probe(self) -> None:
        body = json.dumps({
            "model": self.served_model_name,
            "messages": [{"role": "user", "content": "Reply with OK."}],
            "temperature": 0, "max_tokens": 1,
        }).encode("utf-8")
        request = urllib.request.Request(
            f"http://localhost:{self.port}/v1/chat/completions", data=body,
            headers={"Content-Type": "application/json",
                     "Authorization": "Bearer " + os.environ.get("BFAS_STUDENT_API_KEY", "EMPTY")},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            if response.status != 200:
                raise RuntimeError(f"AppWorld student serving probe returned HTTP {response.status}")
            value = json.loads(response.read().decode("utf-8"))
        if not isinstance(value, Mapping) or not value.get("choices"):
            raise RuntimeError("AppWorld student serving probe returned no completion")

    def release_policy(self) -> None:
        return None

    def generation_suffix(self) -> str:
        return self._suffix

    def target_policy(self, category: str) -> str:
        return "prose_ok"

    def is_call_target(self, target: str, category: str) -> bool:
        return "```python" in target

    def rerender(self, row: Mapping[str, Any]) -> str:
        context = row.get("_render_context")
        if not isinstance(context, Mapping):
            raise ValueError("row lacks AppWorld render context")
        return self._render(context)


__all__ = [
    "AppWorldOfficialAdapter",
    "OfficialRun",
    "call_messages",
    "call_output_text",
    "call_token_usage",
    "read_lm_calls",
]
