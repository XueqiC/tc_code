"""ALFWorld adapter using the local TextWorld sandbox and unseen evaluation."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.request
from collections import defaultdict
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

from ..adapter import (
    BenchmarkAdapter,
    Demo,
    PolicyRef,
    Rollout,
    TaskRef,
    TeacherEpisode,
    Turn,
)
from ..protocol import SAMPLING_TEMPERATURE


ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "envs/alfworld/data/json_2.1.1"
ENV_PYTHON = ROOT / "envs/alfworld/.venv/bin/python"
SERVER_MODEL_NAME = "bfas-policy"
PROMPT = (
    "You are an agent in a household. Complete the task by issuing one "
    "command at a time from the admissible commands.\n\nTask context:\n"
    "{obs}\n\nAdmissible commands:\n{cmds}\n\nHistory:\n{hist}\n\n"
    "Reply with exactly one admissible command and nothing else."
)
TEACHER_REACT_INSTRUCTION = (
    "You are an expert ALFWorld household agent. Think step by step about "
    "the goal, observation, history, and admissible commands before choosing "
    "one command. End each reply with `ACTION: <one admissible command>`."
)
TEACHER_REACT_PROMPT = (
    "Canonical worked example for task type {task_type}:\n"
    "{example}\n\n"
    "Now solve the current episode one command at a time.\n\n"
    "Task context:\n{obs}\n\nAdmissible commands:\n{cmds}\n\n"
    "History:\n{hist}\n\nThink step by step, then give the next ACTION."
)
TEACHER_REACT_EXAMPLES = {
    "pick_and_place_simple": (
        "Task: Put apple 1 in/on diningtable 1.\n"
        "THOUGHT: I need to find and pick up the apple first.\n"
        "ACTION: go to countertop 1\n"
        "THOUGHT: The apple is here, so I should take it.\n"
        "ACTION: take apple 1 from countertop 1\n"
        "THOUGHT: I am holding it and can go to the destination.\n"
        "ACTION: go to diningtable 1\n"
        "THOUGHT: Placing the apple completes the task.\n"
        "ACTION: put apple 1 in/on diningtable 1"
    ),
    "look_at_obj_in_light": (
        "Task: Look at creditcard 1 under the light of desklamp 1.\n"
        "THOUGHT: I need to pick up the credit card.\n"
        "ACTION: go to sidetable 1\n"
        "THOUGHT: The credit card is here.\n"
        "ACTION: take creditcard 1 from sidetable 1\n"
        "THOUGHT: I should carry it to the desk lamp.\n"
        "ACTION: go to desk 1\n"
        "THOUGHT: Using the lamp while holding the card completes the task.\n"
        "ACTION: use desklamp 1"
    ),
    "pick_clean_then_place_in_recep": (
        "Task: Clean plate 1 and put it in/on cabinet 1.\n"
        "THOUGHT: First I need the plate.\n"
        "ACTION: go to countertop 1\n"
        "THOUGHT: I can take the plate now.\n"
        "ACTION: take plate 1 from countertop 1\n"
        "THOUGHT: The sink basin can clean what I am holding.\n"
        "ACTION: go to sinkbasin 1\n"
        "THOUGHT: I should clean the plate before placing it.\n"
        "ACTION: clean plate 1 with sinkbasin 1\n"
        "THOUGHT: Now I need to reach and open the cabinet.\n"
        "ACTION: go to cabinet 1\n"
        "THOUGHT: The closed cabinet must be opened.\n"
        "ACTION: open cabinet 1\n"
        "THOUGHT: The clean plate can now be placed.\n"
        "ACTION: put plate 1 in/on cabinet 1"
    ),
    "pick_heat_then_place_in_recep": (
        "Task: Heat apple 1 and put it in/on diningtable 1.\n"
        "THOUGHT: First I need the apple.\n"
        "ACTION: go to countertop 1\n"
        "THOUGHT: I can take the apple now.\n"
        "ACTION: take apple 1 from countertop 1\n"
        "THOUGHT: The microwave can heat what I am holding.\n"
        "ACTION: go to microwave 1\n"
        "THOUGHT: I should heat the apple before placing it.\n"
        "ACTION: heat apple 1 with microwave 1\n"
        "THOUGHT: Now I can carry it to the destination.\n"
        "ACTION: go to diningtable 1\n"
        "THOUGHT: Placing the heated apple completes the task.\n"
        "ACTION: put apple 1 in/on diningtable 1"
    ),
    "pick_cool_then_place_in_recep": (
        "Task: Cool lettuce 1 and put it in/on countertop 1.\n"
        "THOUGHT: First I need the lettuce.\n"
        "ACTION: go to diningtable 1\n"
        "THOUGHT: I can take the lettuce now.\n"
        "ACTION: take lettuce 1 from diningtable 1\n"
        "THOUGHT: The fridge can cool what I am holding.\n"
        "ACTION: go to fridge 1\n"
        "THOUGHT: I should cool the lettuce before placing it.\n"
        "ACTION: cool lettuce 1 with fridge 1\n"
        "THOUGHT: Now I can carry it to the destination.\n"
        "ACTION: go to countertop 1\n"
        "THOUGHT: Placing the cooled lettuce completes the task.\n"
        "ACTION: put lettuce 1 in/on countertop 1"
    ),
    "pick_two_obj_and_place": (
        "Task: Put two apples in/on diningtable 1.\n"
        "THOUGHT: I need to move the first apple.\n"
        "ACTION: go to countertop 1\n"
        "THOUGHT: I can take the first apple.\n"
        "ACTION: take apple 1 from countertop 1\n"
        "THOUGHT: I should place it before collecting the second one.\n"
        "ACTION: go to diningtable 1\n"
        "THOUGHT: This places the first apple.\n"
        "ACTION: put apple 1 in/on diningtable 1\n"
        "THOUGHT: I need to return for the second apple.\n"
        "ACTION: go to countertop 1\n"
        "THOUGHT: I can take the second apple now.\n"
        "ACTION: take apple 2 from countertop 1\n"
        "THOUGHT: I should bring it to the same destination.\n"
        "ACTION: go to diningtable 1\n"
        "THOUGHT: Placing it completes the two-object task.\n"
        "ACTION: put apple 2 in/on diningtable 1"
    ),
}
_ACTION_MARKER_RE = re.compile(r"\bACTION\s*:\s*", re.IGNORECASE)


class _TeacherQuotaDeadline(RuntimeError):
    """The demo phase exhausted its total teacher-quota wait budget."""


def _quota_seconds(name: str, default: str) -> float:
    try:
        value = float(os.environ.get(name, default))
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


class _TeacherQuotaGate:
    """Coordinate one quota recovery cycle across all demo workers."""

    def __init__(self, teacher: Any):
        self.teacher = teacher
        self.wait_s = _quota_seconds("BFAS_TEACHER_QUOTA_WAIT_S", "1800")
        self.deadline_s = _quota_seconds(
            "BFAS_TEACHER_QUOTA_DEADLINE_S", "86400"
        )
        self.waited_s = 0.0
        self.generation = 0
        self.recovering = False
        self.exhausted = False
        self.condition = threading.Condition()

    def _is_quota_error(self, exc: BaseException) -> bool:
        return (
            isinstance(exc, self.teacher.TeacherAPIError)
            and "HTTP 429" in str(exc)
        )

    def _raise_if_exhausted(self) -> None:
        if self.exhausted:
            raise _TeacherQuotaDeadline(
                "teacher quota wait deadline exhausted"
            )

    def _finish_recovery(self, *, recovered: bool) -> None:
        with self.condition:
            if recovered:
                self.generation += 1
            self.recovering = False
            self.condition.notify_all()

    def recover(self, config: Any, observed_generation: int) -> None:
        with self.condition:
            self._raise_if_exhausted()
            if observed_generation != self.generation:
                return
            if self.recovering:
                while (
                    self.recovering
                    and observed_generation == self.generation
                    and not self.exhausted
                ):
                    self.condition.wait()
                self._raise_if_exhausted()
                return
            self.recovering = True

        recovered = False
        try:
            while True:
                with self.condition:
                    remaining_s = self.deadline_s - self.waited_s
                    if remaining_s <= 0:
                        self.exhausted = True
                        self.condition.notify_all()
                        raise _TeacherQuotaDeadline(
                            "teacher quota wait deadline exhausted"
                        )
                    sleep_s = min(self.wait_s, remaining_s)
                print(
                    "[bfas][demo] teacher quota exhausted, "
                    f"sleeping {sleep_s:g}s",
                    flush=True,
                )
                time.sleep(sleep_s)
                with self.condition:
                    self.waited_s += sleep_s
                try:
                    self.teacher.generate_reply(
                        config,
                        [{"role": "user", "content": "Reply with OK."}],
                        temperature=0.0,
                    )
                except self.teacher.TeacherAPIError as exc:
                    if self._is_quota_error(exc):
                        if sleep_s == 0:
                            with self.condition:
                                self.waited_s = self.deadline_s
                        continue
                    raise
                recovered = True
                return
        finally:
            self._finish_recovery(recovered=recovered)

    def generate_reply(
        self,
        config: Any,
        messages: list[dict[str, str]],
        temperature: float,
    ) -> str:
        while True:
            with self.condition:
                while self.recovering and not self.exhausted:
                    self.condition.wait()
                self._raise_if_exhausted()
                observed_generation = self.generation
            try:
                reply = self.teacher.generate_reply(
                    config, messages, temperature=temperature
                )
            except self.teacher.TeacherAPIError as exc:
                if not self._is_quota_error(exc):
                    raise
                self.recover(config, observed_generation)
                continue
            with self.condition:
                while (
                    self.recovering
                    and observed_generation == self.generation
                    and not self.exhausted
                ):
                    self.condition.wait()
                self._raise_if_exhausted()
            return reply


class _TeacherSession:
    def __init__(self, config: Any, quota_gate: _TeacherQuotaGate):
        self.config = config
        self.quota_gate = quota_gate

    def generate_reply(
        self, messages: list[dict[str, str]], temperature: float
    ) -> str:
        return self.quota_gate.generate_reply(
            self.config, messages, temperature
        )


_GOAL_RE = re.compile(r"Your task is to:\s*(.+?)(?:\s*$|\n)", re.IGNORECASE)


def _goal_line(observation: str) -> str:
    match = _GOAL_RE.search(observation)
    return match.group(1).strip() if match else ""


def _obs_with_goal(goal: str, observation: str) -> str:
    current = observation[-2000:]
    if not goal or goal in current:
        return current
    return f"Goal: {goal}\n\nCurrent observation:\n{current}"


def _student_react() -> bool:
    """BFAS_ALFWORLD_STUDENT_REACT=1 gives the student the teacher's ReAct scaffold."""
    return os.environ.get("BFAS_ALFWORLD_STUDENT_REACT", "1") == "1"


def _category(task_id: str) -> str:
    return task_id.split("/", 1)[0].split("-", 1)[0]


def _game_ids(split_dir: Path) -> list[str]:
    ids: list[str] = []
    for path in split_dir.rglob("game.tw-pddl"):
        if "movable" in str(path) or "Sliced" in str(path):
            continue
        try:
            game = json.loads(path.read_text())
            trajectory = json.loads((path.parent / "traj_data.json").read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            continue
        if game.get("solvable") is not True:
            continue
        if trajectory.get("task_type") not in {
            "pick_and_place_simple",
            "look_at_obj_in_light",
            "pick_clean_then_place_in_recep",
            "pick_heat_then_place_in_recep",
            "pick_cool_then_place_in_recep",
            "pick_two_obj_and_place",
        }:
            continue
        ids.append(str(path.parent.relative_to(split_dir)))
    return sorted(ids)


class _EnvBridge:
    def __init__(self, split: str, task_id: str):
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT / "src")
        self.process = subprocess.Popen(
            [str(ENV_PYTHON), "-u", "-m", "bfas.adapters.alfworld", "--worker",
             "--split", split, "--task-id", task_id],
            cwd=ROOT,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        self._request_id = 0
        ready = self._read()
        if ready.get("op") != "ready":
            raise RuntimeError(f"ALFWorld worker did not become ready: {ready}")

    def _read(self) -> dict[str, Any]:
        if self.process.stdout is None:
            raise RuntimeError("ALFWorld worker has no stdout")
        while True:
            line = self.process.stdout.readline()
            if not line:
                raise RuntimeError(f"ALFWorld worker exited ({self.process.poll()})")
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict) and value.get("bfas_worker") is True:
                return value

    def step(self, command: str) -> dict[str, Any]:
        if self.process.stdin is None:
            raise RuntimeError("ALFWorld worker has no stdin")
        self._request_id += 1
        self.process.stdin.write(json.dumps({"id": self._request_id, "command": command}) + "\n")
        self.process.stdin.flush()
        response = self._read()
        if response.get("id") != self._request_id:
            raise RuntimeError("ALFWorld worker response id mismatch")
        return response

    def close(self) -> None:
        if self.process.stdin is not None:
            self.process.stdin.close()
        if self.process.poll() is None:
            self.process.terminate()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()


class ALFWorldAdapter(BenchmarkAdapter):
    name = "alfworld"

    def __init__(self, seed: int = 0, port: int = 8900):
        self.seed = seed
        self.port = port
        self._loaded_policy: str | None = None
        self._model: Any = None
        self._tokenizer: Any = None
        self._suffix = os.environ.get("BFAS_ALFWORLD_SUFFIX", "<|assistant|>\n")

    def task_pool(self) -> list[TaskRef]:
        ids = _game_ids(DATA / "train")
        if not ids:
            raise FileNotFoundError("ALFWorld train games are unavailable")
        return [TaskRef(task_id, _category(task_id)) for task_id in ids]

    def official_eval_split_disjoint(self) -> bool:
        return True

    def _load_tokenizer(self, policy: PolicyRef) -> Any:
        policy_text = str(policy)
        if self._loaded_policy == policy_text and self._tokenizer is not None:
            return self._tokenizer
        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(
            policy_text, trust_remote_code=False
        )
        self._loaded_policy = policy_text
        return self._tokenizer

    def _load_policy(self, policy: PolicyRef) -> tuple[Any, Any]:
        policy_text = str(policy)
        if (
            self._loaded_policy == policy_text
            and self._model is not None
            and self._tokenizer is not None
        ):
            return self._model, self._tokenizer
        import appworld_eval

        self._model, self._tokenizer = appworld_eval.load_model(
            argparse.Namespace(model=policy_text, adapter=None)
        )
        self._loaded_policy = policy_text
        return self._model, self._tokenizer

    def prepare_renderer(self, policy_ref: PolicyRef) -> None:
        self._load_tokenizer(policy_ref)
        self._render([{"role": "user", "content": ""}])

    def _render(self, messages: Sequence[Mapping[str, str]]) -> str:
        if self._tokenizer is None:
            raise RuntimeError("the student policy must be loaded before rendering")
        rendered = self._tokenizer.apply_chat_template(
            list(messages), tokenize=False, add_generation_prompt=True
        )
        # Learn the generation marker from the template itself (Qwen3.5 ends
        # its generation prompt with "<|im_start|>assistant\n<think>\n", which
        # no fixed marker list anticipated; the audit then rejected every row).
        plain = self._tokenizer.apply_chat_template(
            list(messages), tokenize=False, add_generation_prompt=False
        )
        if rendered.startswith(plain) and len(rendered) > len(plain):
            self._suffix = rendered[len(plain):]
        else:
            for marker in (
                "<|im_start|>assistant\n", "<|assistant|>\n", "ASSISTANT:\n"
            ):
                if rendered.endswith(marker):
                    self._suffix = marker
                    break
        return rendered

    @staticmethod
    def _pick_command(text: str, admissible: Sequence[str]) -> str:
        import alfworld_eval

        return alfworld_eval.pick_command(text, list(admissible))

    @staticmethod
    def _teacher_command_text(reply: str) -> str:
        """Put the teacher's most likely final action first for pick_command."""

        action_markers = list(_ACTION_MARKER_RE.finditer(reply))
        if action_markers:
            return reply[action_markers[-1].end():].lstrip()
        lines = [line.strip() for line in reply.splitlines() if line.strip()]
        return "\n".join(reversed(lines))

    @classmethod
    def _teacher_command(
        cls, reply: str, admissible: Sequence[str]
    ) -> str:
        return cls._pick_command(cls._teacher_command_text(reply), admissible)

    @classmethod
    def _teacher_commands(cls, rollout: Rollout) -> list[str]:
        raw = rollout.raw if isinstance(rollout.raw, Mapping) else {}
        commands = raw.get("teacher_commands")
        if (
            isinstance(commands, Sequence)
            and not isinstance(commands, (str, bytes))
            and all(isinstance(command, str) for command in commands)
        ):
            return list(commands)
        extracted: list[str] = []
        for turn in rollout.turns:
            candidate = cls._teacher_command_text(turn.target).splitlines()
            if candidate:
                extracted.append(candidate[0].strip())
        return extracted

    @classmethod
    def _teacher_worked_example(cls, rollout: Rollout) -> str:
        return "\n".join(cls._teacher_commands(rollout))[-4000:]

    @classmethod
    def _demo_turns(cls, rollout: Rollout) -> Sequence[Turn]:
        raw = rollout.raw if isinstance(rollout.raw, Mapping) else {}
        deployment_turns = raw.get("deployment_turns")
        if (
            isinstance(deployment_turns, Sequence)
            and not isinstance(deployment_turns, (str, bytes))
            and all(isinstance(turn, Turn) for turn in deployment_turns)
        ):
            source_turns = deployment_turns
        else:
            source_turns = rollout.turns
        commands = cls._teacher_commands(rollout)
        if len(commands) != len(source_turns):
            return source_turns
        return tuple(
            Turn(turn.prompt, command, turn.context)
            for turn, command in zip(source_turns, commands)
        )

    @staticmethod
    def _worker_count(name: str, default: int) -> int:
        value = int(os.environ.get(name, str(default)))
        if value <= 0:
            raise ValueError(f"{name} must be greater than zero")
        return value

    def _server_reply(self, prompt: str, temperature: float) -> str:
        payload = json.dumps({
            "model": SERVER_MODEL_NAME,
            "prompt": prompt,
            "max_tokens": 256 if _student_react() else 32,
            "temperature": temperature,
            "add_special_tokens": False,
        }).encode("utf-8")
        request = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/v1/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        timeout = float(os.environ.get("BFAS_SERVER_TIMEOUT", "180"))
        with urllib.request.urlopen(request, timeout=timeout) as response:
            if response.status != 200:
                raise RuntimeError(
                    f"ALFWorld completion returned HTTP {response.status}"
                )
            value = json.loads(response.read())
        choices = value.get("choices") if isinstance(value, Mapping) else None
        if (
            not isinstance(choices, Sequence)
            or isinstance(choices, (str, bytes))
            or not choices
            or not isinstance(choices[0], Mapping)
            or not isinstance(choices[0].get("text"), str)
        ):
            raise RuntimeError("ALFWorld completion response has no text choice")
        return choices[0]["text"]

    def _episode(
        self,
        policy: PolicyRef | None,
        task_id: str,
        split: str,
        temperature: float,
        demo: Demo | None = None,
        teacher_config: Any = None,
    ) -> Rollout:
        model = tokenizer = None
        if teacher_config is None:
            if os.environ.get("BFAS_NO_SERVER") == "1":
                model, tokenizer = self._load_policy(policy or "")
            else:
                tokenizer = self._load_tokenizer(policy or "")
        bridge = _EnvBridge(split, task_id)
        turns: list[Turn] = []
        deployment_turns: list[Turn] = []
        teacher_commands: list[str] = []
        history: list[str] = []
        state = bridge._read()
        # The goal sentence only appears in the very first observation; with an
        # 8-line history window both the student and the teacher lost it after a
        # few steps and wandered (2026-09-02: gpt-5.4 teacher 0/20). Carry it on
        # every turn, in front of the current observation.
        goal = _goal_line(str(state["observation"]))
        won = False
        try:
            for _ in range(int(os.environ.get("BFAS_ALFWORLD_MAX_STEPS", "40"))):
                obs_text = _obs_with_goal(goal, str(state["observation"]))
                deployment_prompt_text = PROMPT.format(
                    obs=obs_text,
                    cmds="\n".join(state["admissible"]),
                    hist="\n".join(history[-8:]) or "(start)",
                )
                student_react = teacher_config is None and _student_react()
                if student_react:
                    # Deployment scaffold = the teacher's ReAct prompt (worked
                    # example per task type + THOUGHT/ACTION). The bare
                    # "reply with one command" prompt left the 2B base at
                    # 0/134 on the unseen split (2026-09-02).
                    task_type = _category(task_id)
                    react_text = TEACHER_REACT_PROMPT.format(
                        task_type=task_type,
                        example=TEACHER_REACT_EXAMPLES[task_type],
                        obs=obs_text,
                        cmds="\n".join(state["admissible"]),
                        hist="\n".join(history[-8:]) or "(start)",
                    )
                    deployment_prompt_text = react_text
                    prompt_text = react_text
                    if demo is not None:
                        prompt_text = (
                            "Worked example from an expert on this task:\n"
                            + demo.worked_example
                            + "\n\n"
                            + react_text
                        )
                    messages = [
                        {"role": "system", "content": TEACHER_REACT_INSTRUCTION},
                        {"role": "user", "content": prompt_text},
                    ]
                elif teacher_config is None:
                    prompt_text = deployment_prompt_text
                    if demo is not None:
                        prompt_text = (
                            "Worked example from an expert on this task:\n"
                            + demo.worked_example
                            + "\n\n"
                            + prompt_text
                        )
                    messages = [{"role": "user", "content": prompt_text}]
                else:
                    task_type = _category(task_id)
                    messages = [
                        {"role": "system", "content": TEACHER_REACT_INSTRUCTION},
                        {
                            "role": "user",
                            "content": TEACHER_REACT_PROMPT.format(
                                task_type=task_type,
                                example=TEACHER_REACT_EXAMPLES[task_type],
                                obs=obs_text,
                                cmds="\n".join(state["admissible"]),
                                hist="\n".join(history[-8:]) or "(start)",
                            ),
                        },
                    ]
                prompt = self._render(messages)
                if teacher_config is None:
                    if os.environ.get("BFAS_NO_SERVER") == "1":
                        import appworld_eval

                        reply = appworld_eval.generate_reply(
                            model, tokenizer, messages, 32, temperature
                        )
                    else:
                        reply = self._server_reply(prompt, temperature)
                else:
                    import appworld_teacher

                    if isinstance(teacher_config, _TeacherSession):
                        teacher_reply = teacher_config.generate_reply(
                            messages, temperature
                        )
                    else:
                        teacher_reply = appworld_teacher.generate_reply(
                            teacher_config, messages, temperature=temperature
                        )
                    reply = appworld_teacher.strip_think(teacher_reply)
                    if not reply.strip():
                        # deepseek intermittently returns empty content at
                        # temperature 0; one mildly warmed retry recovers it
                        if isinstance(teacher_config, _TeacherSession):
                            teacher_reply = teacher_config.generate_reply(
                                messages, 0.3
                            )
                        else:
                            teacher_reply = appworld_teacher.generate_reply(
                                teacher_config, messages, temperature=0.3
                            )
                        reply = appworld_teacher.strip_think(teacher_reply)
                if teacher_config is None and not student_react:
                    command = self._pick_command(reply, state["admissible"])
                else:
                    command = self._teacher_command(
                        reply, state["admissible"]
                    )
                    if teacher_config is not None:
                        teacher_commands.append(command)
                turns.append(Turn(prompt, reply, [dict(message) for message in messages]))
                if student_react:
                    deployment_messages = [
                        {"role": "system", "content": TEACHER_REACT_INSTRUCTION},
                        {"role": "user", "content": deployment_prompt_text},
                    ]
                else:
                    deployment_messages = [{"role": "user", "content": deployment_prompt_text}]
                deployment_turns.append(Turn(
                    self._render(deployment_messages), reply, deployment_messages
                ))
                state = bridge.step(command)
                history.append(f"> {command}\n{str(state['observation'])[:300]}")
                if state["done"]:
                    won = state["won"] is True
                    break
        finally:
            bridge.close()
        raw: dict[str, Any] = {
            "checker_verified": won,
            "category": _category(task_id),
            "deployment_turns": deployment_turns,
        }
        if teacher_config is not None:
            raw["teacher_commands"] = teacher_commands
        return Rollout(
            task_id,
            won,
            turns,
            raw,
        )

    def rollout(
        self,
        policy: PolicyRef,
        task_ids: Sequence[str],
        temperature: float,
        guided_demos: Mapping[str, Demo] | None = None,
    ) -> list[Rollout]:
        self.prepare_renderer(policy)
        demos = guided_demos or {}
        def run_episode(task_id: str) -> Rollout:
            return self._episode(
                policy, task_id, "train", temperature, demos.get(task_id)
            )

        if os.environ.get("BFAS_NO_SERVER") == "1" or len(task_ids) < 2:
            return [run_episode(task_id) for task_id in task_ids]
        workers = min(
            self._worker_count("BFAS_ROLLOUT_WORKERS", 16), len(task_ids)
        )
        with ThreadPoolExecutor(max_workers=workers) as executor:
            return list(executor.map(run_episode, task_ids))

    def teacher_demo(
        self, task_ids: Sequence[str], attempts: int
    ) -> dict[str, Demo]:
        return self._collect_teacher_demos(task_ids, attempts)

    def teacher_episode(
        self, task_id: str, attempt_index: int, temperature: float
    ) -> TeacherEpisode:
        """Run exactly one ledger-accountable ALFWorld teacher episode."""

        import appworld_teacher

        if self._tokenizer is None:
            raise RuntimeError("collect a student rollout before teacher demonstrations")
        teacher_name = os.environ.get("BFAS_TEACHER", "deepseek-v4-pro")
        config = appworld_teacher.load_teacher_config(teacher_name)
        # Let HTTP 429 escape to the run-level pause/resume wrapper around the
        # ledger gateway.  A quota rejection is not a paid attempt and must not
        # consume the task's ledger budget.
        rollout = self._episode(
            None,
            task_id,
            "train",
            temperature,
            teacher_config=config,
        )
        demo = None
        if rollout.verified:
            worked = self._teacher_worked_example(rollout)
            demo = Demo(
                task_id,
                self._demo_turns(rollout),
                worked,
                {
                    "attempt": attempt_index + 1,
                    "checker_verified": True,
                },
            )
        return TeacherEpisode(
            task_id=task_id,
            verified=rollout.verified,
            demo=demo,
            response_texts=tuple(turn.target for turn in rollout.turns),
        )

    def teacher_demo_incremental(
        self,
        task_ids: Sequence[str],
        attempts: int,
        on_result: Callable[[str, Demo | None], None],
    ) -> dict[str, Demo]:
        return self._collect_teacher_demos(task_ids, attempts, on_result)

    def _collect_teacher_demos(
        self,
        task_ids: Sequence[str],
        attempts: int,
        on_result: Callable[[str, Demo | None], None] | None = None,
    ) -> dict[str, Demo]:
        import appworld_teacher

        if self._tokenizer is None:
            raise RuntimeError("collect a student rollout before teacher demonstrations")
        teacher_name = os.environ.get("BFAS_TEACHER", "deepseek-v4-pro")
        quota_gate = _TeacherQuotaGate(appworld_teacher)

        def collect_task(
            task_id: str,
        ) -> tuple[str, Demo | None, BaseException | None]:
            # TeacherConfig is immutable and urllib creates a request-local client;
            # loading per task avoids sharing any client/config state across workers.
            try:
                config = appworld_teacher.load_teacher_config(teacher_name)
                session = _TeacherSession(config, quota_gate)
                for attempt in range(attempts):
                    temperature = (
                        0.0 if attempt == 0 else SAMPLING_TEMPERATURE
                    )
                    while True:
                        with quota_gate.condition:
                            quota_gate._raise_if_exhausted()
                            observed_generation = quota_gate.generation
                        try:
                            rollout = self._episode(
                                None,
                                task_id,
                                "train",
                                temperature,
                                teacher_config=session,
                            )
                        except appworld_teacher.TeacherAPIError as exc:
                            if not quota_gate._is_quota_error(exc):
                                raise
                            quota_gate.recover(config, observed_generation)
                            continue
                        break
                    if rollout.verified:
                        worked = self._teacher_worked_example(rollout)
                        return task_id, Demo(
                            task_id,
                            self._demo_turns(rollout),
                            worked,
                            {"attempt": attempt + 1, "checker_verified": True},
                        ), None
            except _TeacherQuotaDeadline:
                return task_id, None, None
            except Exception as exc:
                return task_id, None, exc
            return task_id, None, None

        if not task_ids:
            return {}
        workers = min(self._worker_count("BFAS_DEMO_WORKERS", 8), len(task_ids))
        demos: dict[str, Demo] = {}
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(collect_task, task_id): task_id
                for task_id in task_ids
            }
            for future in as_completed(futures):
                task_id, demo, error = future.result()
                if error is not None:
                    print(
                        f"[bfas][demo] task={task_id} failed: {error}",
                        flush=True,
                    )
                if demo is not None:
                    demos[task_id] = demo
                if on_result is not None:
                    on_result(task_id, demo)
        return demos

    def evaluate(self, policy_ref: PolicyRef, out_dir: Path) -> dict[str, Any]:
        eval_split = os.environ.get("BFAS_ALFWORLD_EVAL_SPLIT", "valid_unseen")  # dev = valid_seen; valid_unseen = frozen confirmation only
        eval_ids = _game_ids(DATA / eval_split)
        limit = int(os.environ.get("BFAS_ALFWORLD_EVAL_GAMES", str(len(eval_ids))))
        selected = eval_ids[:limit]
        self.prepare_renderer(policy_ref)

        def run_episode(task_id: str) -> Rollout:
            return self._episode(
                policy_ref, task_id, eval_split, 0.0
            )

        if os.environ.get("BFAS_NO_SERVER") == "1" or len(selected) < 2:
            rollouts = [run_episode(task_id) for task_id in selected]
        else:
            workers = min(
                self._worker_count("BFAS_ROLLOUT_WORKERS", 16), len(selected)
            )
            with ThreadPoolExecutor(max_workers=workers) as executor:
                rollouts = list(executor.map(run_episode, selected))
        grouped: dict[str, list[bool]] = defaultdict(list)
        for rollout in rollouts:
            grouped[_category(rollout.task_id)].append(rollout.verified)
        headline = sum(rollout.verified for rollout in rollouts) / len(rollouts) if rollouts else 0.0
        metrics = {
            "headline": headline,
            "success_rate": headline,
            "per_category": {
                category: sum(values) / len(values) for category, values in grouped.items()
            },
        }
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / "records.jsonl").open("w") as handle:
            for rollout in rollouts:
                handle.write(json.dumps({
                    "task_id": rollout.task_id,
                    "won": rollout.verified,
                    "steps": len(rollout.turns),
                }) + "\n")
        (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        return metrics

    def serving_probe(self) -> None:
        if os.environ.get("BFAS_NO_SERVER") == "1":
            if self._model is None or self._tokenizer is None:
                raise RuntimeError("no in-process ALFWorld policy is loaded")
            if not self._render(
                [{"role": "user", "content": "look"}]
            ).endswith(self._suffix):
                raise RuntimeError("ALFWorld renderer round-trip failed")
            return
        self._server_reply("Reply with OK.", 0.0)

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
        return "prose_ok"

    def rerender(self, row: Mapping[str, Any]) -> str:
        context = row.get("_render_context")
        if not isinstance(context, list):
            raise ValueError("row lacks ALFWorld render context")
        return self._render(context)


def _worker_config(game_dir: Path) -> dict[str, Any]:
    return {
        "dataset": {
            "data_path": str(game_dir),
            "eval_id_data_path": str(game_dir),
            "eval_ood_data_path": str(game_dir),
            "num_train_games": -1,
            "num_eval_games": -1,
        },
        "env": {
            "type": "AlfredTWEnv",
            "goal_desc_human_anns_prob": 0,
            "task_types": [1, 2, 3, 4, 5, 6],
            "domain_randomization": False,
            "expert_type": "handcoded",
        },
        "general": {"training_method": "dagger"},
        "dagger": {"training": {"max_nb_steps_per_episode": 50}},
    }


def _worker(split: str, task_id: str) -> int:
    import alfworld.agents.environment as environment

    env_class = environment.get_environment("AlfredTWEnv")
    split_dir = DATA / {"train": "train", "valid_seen": "valid_seen"}.get(split, "valid_unseen")
    game_path = split_dir / task_id / "game.tw-pddl"
    if not game_path.is_file():
        raise FileNotFoundError(game_path)
    train_eval = {"train": "train", "valid_seen": "eval_in_distribution"}.get(split, "eval_out_of_distribution")
    alf = env_class(_worker_config(game_path.parent), train_eval=train_eval)
    env = alf.init_env(batch_size=1)
    observation, info = env.reset()
    print(json.dumps({"bfas_worker": True, "op": "ready"}), flush=True)
    state = {
        "bfas_worker": True,
        "op": "state",
        "observation": observation[0],
        "admissible": list(info["admissible_commands"][0]),
        "done": False,
        "won": False,
    }
    print(json.dumps(state), flush=True)
    for line in sys.stdin:
        request = json.loads(line)
        observation, _, dones, info = env.step([str(request["command"])])
        response = {
            "bfas_worker": True,
            "op": "state",
            "id": request["id"],
            "observation": observation[0],
            "admissible": list(info["admissible_commands"][0]),
            "done": bool(dones[0]),
            "won": bool(info["won"][0]),
        }
        print(json.dumps(response), flush=True)
        if dones[0]:
            break
    env.close()
    return 0


def _parse_worker_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--split", choices=("train", "valid_seen", "valid_unseen", "eval_out_of_distribution"))
    parser.add_argument("--task-id")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_worker_args()
    if not args.worker or not args.split or not args.task_id:
        raise SystemExit("alfworld adapter module is only executable as --worker")
    raise SystemExit(_worker(args.split, args.task_id))
