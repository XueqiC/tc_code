"""WebShop support acquisition and deployment-exact served-student evaluation.

Only the subprocess imports WebShop's Python 3.8 dependencies. The existing
tools/webshop_eval.py owns the prompt, parser, decoding and episode semantics.
"""

from __future__ import annotations

import atexit
from collections.abc import Mapping, Sequence
from copy import deepcopy
import json
import os
from pathlib import Path
import select
import subprocess
import sys
import time
from types import SimpleNamespace
from typing import Any

from ..adapter import BenchmarkAdapter, Demo, PolicyRef, Rollout, TaskRef, TeacherEpisode, Turn
from ..ledger import acquire_demos
from ..protocol import TEACHER_ATTEMPTS
from ._teacher import TeacherSession


ROOT = Path(__file__).resolve().parents[3]
ENV_PYTHON = ROOT / "envs/webshop/venv/bin/python"
# tools is a repository-local namespace package, also when launched with only
# PYTHONPATH=src (as bfas.run and bfas_eval_ckpt.py are).
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import webshop_eval  # noqa: E402


EVAL_SESSIONS = range(500)
SUPPORT_SESSIONS = range(500, 6910)
SERVER_MODEL_NAME = "bfas-policy"

# Keep the worker standalone: importing bfas would execute Python 3.12-only
# type aliases in adapter.py. make_env itself is lazy and Python 3.8 compatible.
WORKER_SOURCE = r'''
import contextlib
import json
import sys

sys.path.insert(0, sys.argv[1])
from tools.webshop_eval import make_env

def send(value):
    value["bfas_worker"] = True
    print(json.dumps(value, ensure_ascii=False), flush=True)

env = None
try:
    with contextlib.redirect_stdout(sys.stderr):
        env = make_env()
        goals = getattr(getattr(env.unwrapped, "server", None), "goals", None)
        categories = [
            goal.get("category") or goal.get("product_category")
            for goal in goals
        ] if goals is not None else []
    send({"op": "ready", "goal_count": len(goals) if goals is not None else None,
          "categories": categories})
    for line in sys.stdin:
        request = json.loads(line)
        try:
            op = request["op"]
            with contextlib.redirect_stdout(sys.stderr):
                if op == "reset":
                    initial = env.reset(session=int(request["session"]))
                    observation = initial[0] if isinstance(initial, tuple) else initial
                    value = {"observation": str(observation)}
                elif op == "step":
                    observation, reward, done, _ = env.step(request["action"])
                    value = {"observation": str(observation), "reward": float(reward),
                             "done": bool(done)}
                elif op == "close":
                    break
                else:
                    raise ValueError("unknown operation: " + op)
            send(dict(value, id=request["id"], op=op))
        except Exception as exc:
            send({"id": request["id"], "op": "error",
                  "error": type(exc).__name__ + ": " + str(exc)})
finally:
    if env is not None:
        with contextlib.redirect_stdout(sys.stderr):
            env.close()
'''


class _EnvBridge:
    """Persistent JSON-lines worker with a gym-compatible reset/step facade."""

    def __init__(self):
        self.process = subprocess.Popen(
            [str(ENV_PYTHON), "-u", "-c", WORKER_SOURCE, str(ROOT)],
            cwd=ROOT / "envs/webshop/repo",
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            bufsize=0,
        )
        self._request_id = 0
        self._read_buffer = b""
        self.done = False
        try:
            ready = self._read()
            if ready.get("op") != "ready":
                raise RuntimeError(f"WebShop worker did not become ready: {ready}")
            count = ready.get("goal_count")
            if count is not None and count < SUPPORT_SESSIONS.stop:
                raise RuntimeError(f"WebShop requires all 6910 goals; worker has {count}")
            self.categories = ready.get("categories", [])
        except BaseException:
            self.close()
            raise

    def _read(self) -> dict[str, Any]:
        assert self.process.stdout is not None
        deadline = time.monotonic() + float(os.environ.get("BFAS_WEBSHOP_BRIDGE_TIMEOUT", "600"))
        while True:
            # Own the buffer: TextIOWrapper read-ahead plus select can hide a
            # ready frame behind native-library logs already read into Python.
            while b"\n" not in self._read_buffer:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not select.select([self.process.stdout], [], [], remaining)[0]:
                    raise RuntimeError("WebShop worker response timed out")
                chunk = os.read(self.process.stdout.fileno(), 65536)
                if not chunk:
                    raise RuntimeError(f"WebShop worker exited ({self.process.poll()})")
                self._read_buffer += chunk
            line, self._read_buffer = self._read_buffer.split(b"\n", 1)
            try:
                value = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if isinstance(value, dict) and value.get("bfas_worker") is True:
                return value

    def _request(self, op: str, **payload: Any) -> dict[str, Any]:
        assert self.process.stdin is not None
        self._request_id += 1
        request = json.dumps(dict(payload, op=op, id=self._request_id)) + "\n"
        self.process.stdin.write(request.encode("utf-8"))
        self.process.stdin.flush()
        response = self._read()
        if response.get("id") != self._request_id:
            raise RuntimeError("WebShop worker response id mismatch")
        if response.get("op") == "error":
            raise RuntimeError(f"WebShop environment error: {response['error']}")
        return response

    def reset(self, session: int) -> str:
        self.done = False
        return self._request("reset", session=session)["observation"]

    def step(self, action: str) -> tuple[str, float, bool, dict[str, Any]]:
        state = self._request("step", action=action)
        self.done = state["done"]
        return state["observation"], state["reward"], self.done, {}

    def close(self) -> None:
        if self.process.stdin is not None and not self.process.stdin.closed:
            try:
                self.process.stdin.close()  # EOF lets the worker close its env.
            except BrokenPipeError:
                pass
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        if self.process.stdout is not None:
            self.process.stdout.close()


class _TeacherSession(TeacherSession):
    def __init__(self):
        import appworld_teacher

        super().__init__(appworld_teacher.load_teacher_config(WebShopAdapter.teacher_name()))


class _EpisodeClient:
    """Capture BFAS turns while the validated evaluator drives the episode."""

    def __init__(self, adapter, temperature, client=None, teacher=None, demo=None):
        self.adapter = adapter
        self.temperature = temperature
        self.client = client
        self.teacher = teacher
        self.demo = demo
        self.turns: list[Turn] = []
        self.deployment_turns: list[Turn] = []
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs):
        deployment_messages = deepcopy(kwargs["messages"])
        messages = deepcopy(deployment_messages)
        if self.demo is not None:
            # Task-specific guidance is collection-only. Preserve the exact
            # deployment context separately for pool rendering and mu scoring.
            guidance = "Expert demonstration for this task:\n" + self.demo.worked_example + "\n\n"
            room = webshop_eval.MAX_PROMPT_CHARS - len(messages[1]["content"])
            messages[1]["content"] = guidance[:max(0, room)] + messages[1]["content"]
        prompt = self.adapter._render(messages)
        deployment_prompt = self.adapter._render(deployment_messages)
        if self.teacher is not None:
            reply = self.teacher.generate_reply(messages, self.temperature)
            completion = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=reply))])
        else:
            kwargs.update(messages=messages, temperature=self.temperature)
            completion = self.client.chat.completions.create(**kwargs)
            reply = completion.choices[0].message.content or ""
        self.turns.append(Turn(prompt, reply, messages))
        self.deployment_turns.append(Turn(deployment_prompt, reply, deployment_messages))
        return completion


class WebShopAdapter(BenchmarkAdapter):
    name = "webshop"
    server_backed = True
    served_model_name = SERVER_MODEL_NAME

    def __init__(self, seed: int = 0, port: int = 8900):
        self.seed = seed
        self.port = port
        self._bridge: _EnvBridge | None = None
        self._tasks: list[TaskRef] | None = None
        self._tokenizer: Any = None
        self._loaded_policy: str | None = None
        self._suffix = "<|assistant|>\n"
        # Reuse the large product/search index across acquisition rounds.
        atexit.register(self.close)

    def _environment(self) -> _EnvBridge:
        if self._bridge is None:
            self._bridge = _EnvBridge()
        return self._bridge

    def _category(self, session: int) -> str:
        categories = self._environment().categories
        category = categories[session] if session < len(categories) else None
        if isinstance(category, str) and category.strip():
            return category.strip()
        return f"index_bucket_{session // 500:02d}"

    def task_pool(self) -> list[TaskRef]:
        if self._tasks is None:
            self._tasks = [TaskRef(str(i), self._category(i)) for i in SUPPORT_SESSIONS]
        return list(self._tasks)

    def official_eval_split_disjoint(self) -> bool:
        return True

    @staticmethod
    def teacher_name() -> str:
        return os.environ.get("BFAS_TEACHER", "gpt-5.4")

    def prepare_renderer(self, policy_ref: PolicyRef) -> None:
        if self._tokenizer is None or self._loaded_policy != str(policy_ref):
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(str(policy_ref), trust_remote_code=False)
            self._loaded_policy = str(policy_ref)
        self._render(webshop_eval.build_messages([], "", webshop_eval.OBS_CHARS))

    def _render(self, messages: Sequence[Mapping[str, str]]) -> str:
        if self._tokenizer is None:
            raise RuntimeError("prepare the student renderer before WebShop acquisition")
        rendered = self._tokenizer.apply_chat_template(
            list(messages), tokenize=False, add_generation_prompt=True
        )
        plain = self._tokenizer.apply_chat_template(
            list(messages), tokenize=False, add_generation_prompt=False
        )
        if not rendered.startswith(plain) or len(rendered) == len(plain):
            raise RuntimeError("WebShop tokenizer has no identifiable generation suffix")
        self._suffix = rendered[len(plain):]
        return rendered

    def _client(self):
        return webshop_eval.make_client(f"http://127.0.0.1:{self.port}/v1")

    def _episode(self, task_id: str, temperature: float, *, client=None,
                 teacher: _TeacherSession | None = None, demo: Demo | None = None) -> Rollout:
        session = int(task_id)
        if str(session) != task_id or session not in range(SUPPORT_SESSIONS.stop):
            raise ValueError(f"invalid WebShop goal index: {task_id!r}")
        bridge = self._environment()
        category = self._category(session)
        capture = _EpisodeClient(self, temperature, client, teacher, demo)
        try:
            record = webshop_eval.run_episode(
                bridge, capture, session, SERVER_MODEL_NAME,
                webshop_eval.MAX_STEPS, webshop_eval.OBS_CHARS,
                webshop_eval.HISTORY_OBS_CHARS, webshop_eval.MAX_PROMPT_CHARS,
            )
        except Exception as exc:
            if teacher is None or not (teacher.response_texts or teacher.tokens_spent or teacher.usage):
                raise
            # A later API/env failure must not discard earlier paid output.
            record = {"session": session, "reward": 0.0, "success": False,
                      "steps": len(capture.turns), "error": f"{type(exc).__name__}: {exc}"}
            self.close()
        verified = record["success"]
        if teacher is not None:
            verified = verified and bridge.done and record["steps"] <= webshop_eval.MAX_STEPS
        raw = dict(record, checker_verified=bool(verified), category=category)
        if demo is not None:
            raw["deployment_turns"] = capture.deployment_turns
        return Rollout(task_id, bool(verified), capture.turns, raw)

    def rollout(self, policy: PolicyRef, task_ids: Sequence[str], temperature: float,
                guided_demos: Mapping[str, Demo] | None = None) -> list[Rollout]:
        self.prepare_renderer(policy)
        with self._client() as client:
            return [self._episode(task_id, temperature, client=client,
                                  demo=(guided_demos or {}).get(task_id)) for task_id in task_ids]

    def teacher_episode(self, task_id: str, attempt_index: int, temperature: float) -> TeacherEpisode:
        if not 0 <= attempt_index < TEACHER_ATTEMPTS:
            raise ValueError("WebShop allows at most three teacher attempts per task")
        if int(task_id) not in SUPPORT_SESSIONS:
            raise ValueError("WebShop teacher acquisition is restricted to train goals 500..6909")
        if self._tokenizer is None:
            raise RuntimeError("prepare the student renderer before WebShop teacher acquisition")
        teacher = _TeacherSession()
        rollout = self._episode(task_id, temperature, teacher=teacher)
        demo = None
        if rollout.verified:
            turns = []
            worked = []
            for turn in rollout.turns:
                action = webshop_eval.parse_action(turn.target)
                if action is None:
                    continue  # Format failures are paid, but have no action target.
                turns.append(Turn(turn.prompt, action, turn.context))
                current = turn.context[1]["content"].rsplit("\nObservation: ", 1)[-1]
                current = current.removesuffix("\nAction:")
                worked.append(f"Observation: {current}\nAction: {action}")
            demo = Demo(task_id, turns, "\n\n".join(worked),
                        dict(rollout.raw, attempt=attempt_index + 1))
        return TeacherEpisode(task_id, rollout.verified, demo,
                              tuple(teacher.response_texts), teacher.tokens_spent,
                              teacher=teacher.config.name, usage=teacher.usage)

    def teacher_demo(self, task_ids: Sequence[str], attempts: int) -> dict[str, Demo]:
        # The same gateway is used by run.py. Do not append here as well as in
        # teacher_episode: each paid attempt must have exactly one ledger row.
        return dict(acquire_demos(self.name, self, task_ids, min(attempts, TEACHER_ATTEMPTS)))

    def evaluate(self, policy_ref: PolicyRef, out_dir: Path) -> dict[str, Any]:
        self.prepare_renderer(policy_ref)
        out_dir.mkdir(parents=True, exist_ok=True)
        records = []
        try:
            with self._client() as client, (out_dir / "records.jsonl").open("w", encoding="utf-8") as stream:
                for session in EVAL_SESSIONS:
                    rollout = self._episode(str(session), 0.0, client=client)
                    record = dict(rollout.raw, task_id=rollout.task_id)
                    records.append(record)
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                    stream.flush()
        finally:
            self.close()
        metrics = webshop_eval.compute_metrics(records, {
            "split": "test", "start": 0, "n": 500, "policy": str(policy_ref),
            "max_steps": webshop_eval.MAX_STEPS, "obs_chars": webshop_eval.OBS_CHARS,
            "history_obs_chars": webshop_eval.HISTORY_OBS_CHARS,
            "max_prompt_chars": webshop_eval.MAX_PROMPT_CHARS, "seed": self.seed,
        })
        metrics["headline"] = metrics["success_rate"]
        metrics["mean_score"] = metrics["score"]
        categories = sorted({record["category"] for record in records})
        metrics["per_category"] = {
            category: webshop_eval.compute_metrics(
                [r for r in records if r["category"] == category], {}
            )["success_rate"] for category in categories
        }
        (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        return metrics

    def serving_probe(self) -> None:
        with self._client() as client:
            client.chat.completions.create(
                model=SERVER_MODEL_NAME,
                messages=webshop_eval.build_messages([], "WebShop [SEP] Search", webshop_eval.OBS_CHARS),
                temperature=0, max_tokens=128, stop=["\nObservation", "Observation:"],
            )

    def generation_suffix(self) -> str:
        return self._suffix

    def target_policy(self, category: str) -> str:
        return "prose_ok"

    def is_call_target(self, target: str, category: str) -> bool:
        return webshop_eval.parse_action(target) is not None

    def rerender(self, row: Mapping[str, Any]) -> str:
        context = row.get("_render_context")
        if not isinstance(context, list):
            raise ValueError("row lacks WebShop render context")
        return self._render(context)

    def close(self) -> None:
        if self._bridge is not None:
            self._bridge.close()
            self._bridge = None

    def release_policy(self) -> None:
        self.close()
        self._tokenizer = None
        self._loaded_policy = None
