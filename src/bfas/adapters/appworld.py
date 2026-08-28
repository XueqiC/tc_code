"""AppWorld adapter backed by the repository's evaluation and teacher paths."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ..adapter import BenchmarkAdapter, Demo, PolicyRef, Rollout, TaskRef, Turn
from ..protocol import SAMPLING_TEMPERATURE, SupportSplit


ROOT = Path(__file__).resolve().parents[3]


class AppWorldAdapter(BenchmarkAdapter):
    name = "appworld"

    def __init__(self, seed: int = 0):
        self.seed = seed
        self._loaded_policy: str | None = None
        self._model: Any = None
        self._tokenizer: Any = None
        self._tasks: list[TaskRef] | None = None
        self._suffix = os.environ.get("BFAS_APPWORLD_SUFFIX", "<|assistant|>\n")

    def task_pool(self) -> list[TaskRef]:
        if self._tasks is not None:
            return list(self._tasks)
        bridge_python = ROOT / "envs/appworld-venv/bin/python"
        bridge_env = os.environ.copy()
        bridge_env["APPWORLD_ROOT"] = str(ROOT / "envs/appworld-data")
        result = subprocess.run(
            [
                str(bridge_python),
                "-c",
                "import json; from appworld import load_task_ids; "
                "print(json.dumps(list(load_task_ids('train'))))",
            ],
            cwd=ROOT / "envs",
            env=bridge_env,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            for line in reversed(result.stdout.splitlines()):
                try:
                    values = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(values, list) and all(isinstance(value, str) for value in values):
                    self._tasks = [
                        TaskRef(task_id, task_id.rsplit("_", 1)[0])
                        for task_id in sorted(values)
                    ]
                    return list(self._tasks)
        paths = (
            ROOT / "data/appworld_sft/episodes.jsonl",
            ROOT / "data/appworld_sft/pool.jsonl",
        )
        ids: set[str] = set()
        for path in paths:
            if not path.is_file():
                continue
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if isinstance(row, dict) and isinstance(row.get("task_id"), str):
                        ids.add(row["task_id"])
            if ids:
                break
        if not ids:
            raise FileNotFoundError("AppWorld train task manifest is unavailable")
        self._tasks = [
            TaskRef(task_id, task_id.rsplit("_", 1)[0]) for task_id in sorted(ids)
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

    def _load_policy(self, policy: PolicyRef) -> tuple[Any, Any]:
        policy_text = str(policy)
        if self._loaded_policy == policy_text:
            return self._model, self._tokenizer
        import appworld_eval as appworld_eval

        args = argparse.Namespace(model=policy_text, adapter=None)
        self._model, self._tokenizer = appworld_eval.load_model(args)
        self._loaded_policy = policy_text
        return self._model, self._tokenizer

    def _render(self, messages: Sequence[Mapping[str, str]]) -> str:
        if self._tokenizer is None:
            raise RuntimeError("the student policy must be loaded before rendering")
        rendered = self._tokenizer.apply_chat_template(
            list(messages), tokenize=False, add_generation_prompt=True
        )
        if not isinstance(rendered, str):
            raise TypeError("AppWorld serving renderer did not return text")
        for marker in (
            "<|im_start|>assistant\n",
            "<|assistant|>\n",
            "ASSISTANT:\n",
        ):
            if rendered.endswith(marker):
                self._suffix = marker
                break
        return rendered

    def _episode(
        self,
        policy: PolicyRef,
        task_id: str,
        temperature: float,
        demo: Demo | None,
    ) -> Rollout:
        import appworld_eval as appworld_eval

        model, tokenizer = self._load_policy(policy)
        bridge = appworld_eval.AppWorldBridge(ROOT)
        messages: list[dict[str, str]] = []
        turns: list[Turn] = []
        deployment_turns: list[Turn] = []
        evaluation: dict[str, Any] = {}
        try:
            task = bridge.request(
                "start",
                split="train",
                task_id=task_id,
                experiment_name=f"bfas-{self.seed}",
                seed=self.seed,
            )
            deployment_system = appworld_eval.make_system_prompt(task)
            system = deployment_system
            if demo is not None:
                system += (
                    "\n\nWorked example from an expert on this task. Study the "
                    "approach, then solve the task yourself step by step:\n"
                    + demo.worked_example
                )
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": "Begin by consulting the API documentation."},
            ]
            for _ in range(int(os.environ.get("BFAS_APPWORLD_MAX_STEPS", "12"))):
                context = [dict(message) for message in messages]
                prompt = self._render(context)
                reply = appworld_eval.generate_reply(
                    model,
                    tokenizer,
                    messages,
                    int(os.environ.get("BFAS_MAX_NEW_TOKENS", "512")),
                    temperature,
                )
                turns.append(Turn(prompt, reply, context))
                deployment_context = [dict(message) for message in context]
                deployment_context[0]["content"] = deployment_system
                deployment_turns.append(Turn(
                    self._render(deployment_context), reply, deployment_context
                ))
                messages.append({"role": "assistant", "content": reply})
                code = appworld_eval.extract_python_code(reply)
                if code is None:
                    break
                execution = bridge.request("execute", code=code)
                output = appworld_eval.truncate_output(str(execution.get("output", "")))
                messages.append({
                    "role": "user",
                    "content": f"Execution output:\n{output}\n\nContinue the task.",
                })
                if appworld_eval.calls_complete_task(code):
                    break
            evaluation = bridge.request("evaluate")
        finally:
            try:
                bridge.request("stop")
            except Exception:
                pass
            bridge.close()
        verified = evaluation.get("success") is True
        return Rollout(
            task_id,
            verified,
            turns,
            {
                "checker_verified": verified,
                "passed_tests": int(evaluation.get("passed_tests", 0)),
                "failed_tests": int(evaluation.get("failed_tests", 0)),
                "deployment_turns": deployment_turns,
            },
        )

    def rollout(
        self,
        policy: PolicyRef,
        task_ids: Sequence[str],
        temperature: float,
        guided_demos: Mapping[str, Demo] | None = None,
    ) -> list[Rollout]:
        demos = guided_demos or {}
        return [
            self._episode(policy, task_id, temperature, demos.get(task_id))
            for task_id in task_ids
        ]

    def teacher_demo(
        self, task_ids: Sequence[str], attempts: int
    ) -> dict[str, Demo]:
        import appworld_teacher as teacher

        if self._tokenizer is None:
            raise RuntimeError("collect a student rollout before teacher demonstrations")
        teacher_name = os.environ.get("BFAS_TEACHER", "deepseek-v4-pro")
        config = teacher.load_teacher_config(teacher_name)
        demos: dict[str, Demo] = {}
        old_temperature = os.environ.get("TEACHER_TEMP")
        try:
            for task_id in task_ids:
                for attempt in range(attempts):
                    os.environ["TEACHER_TEMP"] = (
                        "0" if attempt == 0 else str(SAMPLING_TEMPERATURE)
                    )
                    bridge = teacher.AppWorldBridge(ROOT)
                    messages: list[dict[str, str]] = []
                    rendered_turns: list[Turn] = []
                    evaluation: dict[str, Any] = {}
                    try:
                        task = bridge.request(
                            "start",
                            split="train",
                            task_id=task_id,
                            experiment_name=f"bfas-teacher-{attempt + 1}",
                            seed=teacher.DEFAULT_SEED,
                        )
                        messages = [
                            {"role": "system", "content": teacher.make_system_prompt(task)},
                            {"role": "user", "content": "Begin by consulting the API documentation."},
                        ]
                        for _ in range(int(os.environ.get("BFAS_APPWORLD_MAX_STEPS", "12"))):
                            context = [dict(message) for message in messages]
                            reply = teacher.strip_think(teacher.generate_reply(config, messages))
                            rendered_turns.append(Turn(self._render(context), reply, context))
                            messages.append({"role": "assistant", "content": reply})
                            code = teacher.extract_python_code(reply)
                            if code is None:
                                break
                            execution = bridge.request("execute", code=code)
                            output = teacher.truncate_output(str(execution.get("output", "")))
                            messages.append({
                                "role": "user",
                                "content": f"Execution output:\n{output}\n\nContinue the task.",
                            })
                            if teacher.calls_complete_task(code):
                                break
                        evaluation = bridge.request("evaluate")
                    finally:
                        try:
                            bridge.request("stop")
                        except Exception:
                            pass
                        bridge.close()
                    if evaluation.get("success") is True:
                        worked = "\n".join(turn.target for turn in rendered_turns)[-6000:]
                        demos[task_id] = Demo(
                            task_id,
                            rendered_turns,
                            worked,
                            {"attempt": attempt + 1, "checker_verified": True},
                        )
                        break
        finally:
            if old_temperature is None:
                os.environ.pop("TEACHER_TEMP", None)
            else:
                os.environ["TEACHER_TEMP"] = old_temperature
        return demos

    def evaluate(self, policy_ref: PolicyRef, out_dir: Path) -> dict[str, Any]:
        split = os.environ.get("BFAS_APPWORLD_EVAL_SPLIT", "test_normal")
        max_tasks = os.environ.get("BFAS_APPWORLD_EVAL_TASKS", "40")
        tag = "bfas_" + "_".join(out_dir.parts[-3:]).replace("-", "_")
        command = [
            str(ROOT / ".venv/bin/python"),
            str(ROOT / "src/appworld_eval.py"),
            "--model", str(policy_ref),
            "--split", split,
            "--max-tasks", max_tasks,
            "--tag", tag,
            "--seed", str(self.seed),
        ]
        subprocess.run(command, cwd=ROOT, check=True)
        source = ROOT / "results/appworld" / tag
        out_dir.mkdir(parents=True, exist_ok=True)
        for name in ("records.jsonl", "metrics.json"):
            shutil.copy2(source / name, out_dir / name)
        metrics = json.loads((out_dir / "metrics.json").read_text())
        records = [json.loads(line) for line in (out_dir / "records.jsonl").read_text().splitlines() if line]
        grouped: dict[str, list[bool]] = defaultdict(list)
        for record in records:
            grouped[str(record["task_id"]).rsplit("_", 1)[0]].append(bool(record["success"]))
        return {
            "headline": float(metrics["tgc"]),
            "tgc": float(metrics["tgc"]),
            "sgc": float(metrics["sgc"]),
            "per_category": {
                category: sum(values) / len(values) for category, values in grouped.items()
            },
        }

    def serving_probe(self) -> None:
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("no AppWorld policy is loaded")
        prompt = self._render([{"role": "user", "content": "Reply with OK."}])
        if not prompt.endswith(self._suffix):
            raise RuntimeError("AppWorld renderer round-trip failed")

    def release_policy(self) -> None:
        import gc
        import torch

        self._model = None
        self._tokenizer = None
        self._loaded_policy = None
        gc.collect()
        torch.cuda.empty_cache()

    def generation_suffix(self) -> str:
        return self._suffix

    def target_policy(self, category: str) -> str:
        return "call_required"

    def is_call_target(self, target: str, category: str) -> bool:
        import appworld_eval as appworld_eval

        return appworld_eval.extract_python_code(target) is not None

    def rerender(self, row: Mapping[str, Any]) -> str:
        context = row.get("_render_context")
        if not isinstance(context, list):
            raise ValueError("row lacks AppWorld render context")
        return self._render(context)
