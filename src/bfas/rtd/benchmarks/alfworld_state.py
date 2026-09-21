"""C26-B replay contract, with no ALFWorld/TextWorld imports or I/O.

The stepper is functional: reset(request) -> (cursor, observation), and
step(cursor, command) -> (new_cursor, observation). C26-B must supply a bounded,
cleaned-up environment implementation and a frozen renderer. No archived bare
prompt is promoted to a reconstructed FullState here.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import re
from typing import Callable, Protocol

from ..transport import Behavior, FullState


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def parent_hash(task_id):
    if not isinstance(task_id, str) or len(task_id.split("/")) != 2 or any(
            p in {"", ".", ".."} for p in task_id.split("/")):
        raise ValueError("task_id must name a game and trial")
    return canonical_hash(dict(benchmark="alfworld", split="train", parent_game=task_id.split("/")[0]))


def parent_fold(value):
    return int(value, 16) % 2


def validate_parent_folds(task_folds):
    """Validate the §5 proposal without creating/freezing a support manifest."""
    for task_id, fold in task_folds.items():
        if type(fold) is not int or fold != parent_fold(parent_hash(task_id)):
            raise ValueError("trial fold differs from deterministic parent fold")
    return dict(task_folds)


def validate_request(request):
    required = {"benchmark", "split", "task_id", "goal", "world_hash", "world_files",
                "environment_hash", "max_episode_steps", "state_kind"}
    if set(request) != required or request["benchmark"] != "alfworld" or request["split"] != "train":
        raise ValueError("complete train reset request required")
    parent_hash(request["task_id"])
    if request["state_kind"] != "train_reset_request":
        raise ValueError("reset request required")
    if not isinstance(request["goal"], str) or not request["goal"].strip():
        raise ValueError("missing goal")
    files = request["world_files"]
    if (not isinstance(files, dict) or not files or
            any(not isinstance(k, str) or not re.fullmatch(r"[a-f0-9]{64}", str(v))
                for k, v in files.items()) or canonical_hash(files) != request["world_hash"]):
        raise ValueError("wrong world hash")
    if not re.fullmatch(r"[a-f0-9]{64}", str(request["environment_hash"])):
        raise ValueError("environment hash required")
    if type(request["max_episode_steps"]) is not int or request["max_episode_steps"] < 1:
        raise ValueError("configured episode horizon required")
    return request


@dataclass(frozen=True)
class Observation:
    index: int
    observation: str
    admissible: tuple[str, ...]
    world_hash: str
    done: bool = False
    won: bool = False  # privileged outcome: never included in FullState/public data

    def __post_init__(self):
        object.__setattr__(self, "admissible", tuple(self.admissible))
        if type(self.index) is not int or self.index < 0 or not isinstance(self.observation, str):
            raise ValueError("invalid observation record")
        if any(not isinstance(c, str) or not c.strip() for c in self.admissible):
            raise ValueError("invalid admissible commands")
        if type(self.done) is not bool or type(self.won) is not bool:
            raise ValueError("environment done/won must be bool")


class Stepper(Protocol):
    def reset(self, request: dict) -> tuple[object, Observation]: ...
    def step(self, cursor: object, command: str) -> tuple[object, Observation]: ...


def _observed(o):
    return dict(role="user" if o.index == 0 else "tool", index=o.index,
                content=o.observation, admissible=list(o.admissible), done=o.done)


def validate_full_state(state, *, expected_world_hash):
    state.validate()
    request = validate_request(json.loads(state.task_json))
    if request["world_hash"] != expected_world_hash:
        raise ValueError("wrong world hash")
    if state.parent_hash != parent_hash(request["task_id"]):
        raise ValueError("wrong parent hash")
    history = json.loads(state.history_json)
    if len(history) % 2 != 1:
        raise ValueError("out-of-order history: observation must end the prefix")
    for i, row in enumerate(history):
        index = (i + 1) // 2
        role = "user" if i == 0 else "assistant" if i % 2 else "tool"
        fields = {"role", "index", "content"} | ({"admissible", "done"} if i % 2 == 0 else set())
        if (set(row) != fields or row["role"] != role or type(row["index"]) is not int
                or row["index"] != index or not isinstance(row["content"], str)):
            raise ValueError("out-of-order history")
        if i % 2 == 0:
            if (type(row["done"]) is not bool or not isinstance(row["admissible"], list)
                    or any(not isinstance(c, str) or not c.strip() for c in row["admissible"])):
                raise ValueError("invalid observation/admissible record")
            if row["done"] and i != len(history) - 1:
                raise ValueError("history continues after done")
        elif not row["content"].strip() or row["content"] not in history[i - 1]["admissible"]:
            raise ValueError("history action was not admissible")
    if (len(history) - 1) // 2 > request["max_episode_steps"]:
        raise ValueError("history exceeds configured horizon")
    return state


@dataclass(frozen=True)
class ReplayResult:
    behaviors: tuple[Behavior, ...]
    states: tuple[FullState, ...]  # reset, then every observation including terminal
    transcript_hash: str
    done: bool
    won: bool


def replay_commands(request: dict, commands, stepper: Stepper,
                    renderer: Callable[[dict, list[dict]], str]) -> ReplayResult:
    """Pure orchestration for an injected functional stepper; never starts an env.

    Store untruncated observations and all ordered admissible/action records.
    The injected renderer alone applies the frozen policy prompt window. Hashes
    are deterministic for the same world/configuration, commands and observations.
    This is an offline auditor primitive, not a pre-purchase source sampler.
    """
    request = json.loads(json.dumps(validate_request(request)))
    commands = tuple(commands)
    if (not commands or len(commands) > request["max_episode_steps"] or
            any(not isinstance(c, str) or not c.strip() for c in commands)):
        raise ValueError("complete command sequence within configured horizon required")
    cursor, observation = stepper.reset(json.loads(json.dumps(request)))
    history, states, behaviors = [], [], []
    for i in range(len(commands) + 1):
        if observation.index != i:
            raise ValueError("out-of-order history from stepper")
        if observation.world_hash != request["world_hash"]:
            raise ValueError("wrong world hash from stepper")
        history.append(_observed(observation))
        # Copy at the boundary so even a renderer cannot mutate the archive.
        prompt = renderer(json.loads(json.dumps(request)), json.loads(json.dumps(history)))
        state = FullState.create(request, history, prompt, parent_hash(request["task_id"]))
        validate_full_state(state, expected_world_hash=request["world_hash"])
        states.append(state)
        if i == len(commands):
            break
        command = commands[i]
        if observation.done:
            raise ValueError("command sequence continues after terminal state")
        if command not in observation.admissible:
            raise ValueError("archived command is not admissible in reconstructed state")
        behaviors.append(Behavior(state, command))
        history.append(dict(role="assistant", index=i + 1, content=command))
        cursor, observation = stepper.step(cursor, command)
    return ReplayResult(tuple(behaviors), tuple(states),
                        canonical_hash(dict(states=[asdict(s) for s in states], commands=commands,
                                            done=observation.done, won=observation.won)),
                        observation.done, observation.won)


def reconstruct_package(payload, request, stepper, renderer, *, owned_ids, inner_parent_hashes):
    """Owned teacher prefixes only; crossing a round's fold is forbidden."""
    identity = payload["provenance"]
    query_id = payload["query_id"]
    if query_id not in owned_ids or not set(payload.get("dependencies", ())) <= set(owned_ids):
        raise ValueError("teacher prefix package and dependencies must be owned")
    if parent_hash(request["task_id"]) not in inner_parent_hashes:
        raise ValueError("teacher prefix outside inner fold")
    if payload.get("exclusion_reasons"):
        raise ValueError("protected teacher prefix")
    if identity["task_id"] != request["task_id"] or payload["request_state_hash"] != canonical_hash(request):
        raise ValueError("package/reset request mismatch")
    if payload["status"] == "unavailable" or payload["payload_kind"] not in (
            "extracted_teacher_commands", "teacher_react_turns"):
        raise ValueError("payload unavailable")
    return replay_commands(request, payload["commands"], stepper, renderer)
