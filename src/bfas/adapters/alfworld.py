"""ALFWorld adapter using the local TextWorld sandbox and unseen evaluation."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ..adapter import BenchmarkAdapter, Demo, PolicyRef, Rollout, TaskRef, Turn
from ..protocol import SAMPLING_TEMPERATURE


ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "envs/alfworld/data/json_2.1.1"
ENV_PYTHON = ROOT / "envs/alfworld/.venv/bin/python"
PROMPT = (
    "You are an agent in a household. Complete the task by issuing one "
    "command at a time from the admissible commands.\n\nTask context:\n"
    "{obs}\n\nAdmissible commands:\n{cmds}\n\nHistory:\n{hist}\n\n"
    "Reply with exactly one admissible command and nothing else."
)


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

    def __init__(self, seed: int = 0):
        self.seed = seed
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

    def _load_policy(self, policy: PolicyRef) -> tuple[Any, Any]:
        policy_text = str(policy)
        if self._loaded_policy == policy_text:
            return self._model, self._tokenizer
        import appworld_eval

        self._model, self._tokenizer = appworld_eval.load_model(
            argparse.Namespace(model=policy_text, adapter=None)
        )
        self._loaded_policy = policy_text
        return self._model, self._tokenizer

    def _render(self, messages: Sequence[Mapping[str, str]]) -> str:
        if self._tokenizer is None:
            raise RuntimeError("the student policy must be loaded before rendering")
        rendered = self._tokenizer.apply_chat_template(
            list(messages), tokenize=False, add_generation_prompt=True
        )
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

    def _episode(
        self,
        policy: PolicyRef | None,
        task_id: str,
        split: str,
        temperature: float,
        demo: Demo | None = None,
        teacher_config: Any = None,
    ) -> Rollout:
        import appworld_eval

        model = tokenizer = None
        if teacher_config is None:
            model, tokenizer = self._load_policy(policy or "")
        bridge = _EnvBridge(split, task_id)
        turns: list[Turn] = []
        deployment_turns: list[Turn] = []
        history: list[str] = []
        state = bridge._read()
        won = False
        try:
            for _ in range(int(os.environ.get("BFAS_ALFWORLD_MAX_STEPS", "40"))):
                deployment_prompt_text = PROMPT.format(
                    obs=str(state["observation"])[-2000:],
                    cmds="\n".join(state["admissible"]),
                    hist="\n".join(history[-8:]) or "(start)",
                )
                prompt_text = deployment_prompt_text
                if demo is not None:
                    prompt_text = (
                        "Worked example from an expert on this task:\n"
                        + demo.worked_example
                        + "\n\n"
                        + prompt_text
                    )
                messages = [{"role": "user", "content": prompt_text}]
                prompt = self._render(messages)
                if teacher_config is None:
                    reply = appworld_eval.generate_reply(
                        model, tokenizer, messages, 32, temperature
                    )
                else:
                    import appworld_teacher

                    reply = appworld_teacher.strip_think(
                        appworld_teacher.generate_reply(teacher_config, messages)
                    )
                command = self._pick_command(reply, state["admissible"])
                turns.append(Turn(prompt, reply, [dict(message) for message in messages]))
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
        return Rollout(
            task_id,
            won,
            turns,
            {
                "checker_verified": won,
                "category": _category(task_id),
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
            self._episode(policy, task_id, "train", temperature, demos.get(task_id))
            for task_id in task_ids
        ]

    def teacher_demo(
        self, task_ids: Sequence[str], attempts: int
    ) -> dict[str, Demo]:
        import appworld_teacher

        if self._tokenizer is None:
            raise RuntimeError("collect a student rollout before teacher demonstrations")
        config = appworld_teacher.load_teacher_config(
            os.environ.get("BFAS_TEACHER", "deepseek-v4-pro")
        )
        demos: dict[str, Demo] = {}
        old_temperature = os.environ.get("TEACHER_TEMP")
        try:
            for task_id in task_ids:
                for attempt in range(attempts):
                    os.environ["TEACHER_TEMP"] = (
                        "0" if attempt == 0 else str(SAMPLING_TEMPERATURE)
                    )
                    rollout = self._episode(
                        None,
                        task_id,
                        "train",
                        0.0 if attempt == 0 else SAMPLING_TEMPERATURE,
                        teacher_config=config,
                    )
                    if rollout.verified:
                        worked = "\n".join(turn.target for turn in rollout.turns)[-4000:]
                        demos[task_id] = Demo(
                            task_id,
                            rollout.turns,
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
        eval_ids = _game_ids(DATA / "valid_unseen")
        limit = int(os.environ.get("BFAS_ALFWORLD_EVAL_GAMES", str(len(eval_ids))))
        rollouts = [
            self._episode(policy_ref, task_id, "eval_out_of_distribution", 0.0)
            for task_id in eval_ids[:limit]
        ]
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
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("no ALFWorld policy is loaded")
        if not self._render([{"role": "user", "content": "look"}]).endswith(self._suffix):
            raise RuntimeError("ALFWorld renderer round-trip failed")

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
    split_dir = DATA / ("train" if split == "train" else "valid_unseen")
    game_path = split_dir / task_id / "game.tw-pddl"
    if not game_path.is_file():
        raise FileNotFoundError(game_path)
    alf = env_class(_worker_config(game_path.parent), train_eval=split)
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
    parser.add_argument("--split", choices=("train", "eval_out_of_distribution"))
    parser.add_argument("--task-id")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_worker_args()
    if not args.worker or not args.split or not args.task_id:
        raise SystemExit("alfworld adapter module is only executable as --worker")
    raise SystemExit(_worker(args.split, args.task_id))
