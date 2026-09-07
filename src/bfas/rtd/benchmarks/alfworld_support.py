"""C26-B support and privileged CPU replay; C26-A files remain unchanged.

The environment bridge is an additive subclass of the existing adapter bridge.
Only the adapter worker imports ALFWorld/TextWorld. Training integration is a
later stage: callers must use the round/ownership guards before consuming states.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import queue
import re
import signal
import subprocess
import tempfile
import threading
import time

from ...adapters import alfworld as adapter
from ...cc_pairs import digest
from ..broker import seal_bank
from ..transport import FullState
from . import alfworld_bank as bank
from .alfworld_caps import CapConfiguration, cap_audit
from .alfworld_state import (
    Observation, canonical_hash, parent_fold, parent_hash, replay_commands,
    validate_full_state, validate_parent_folds, validate_request,
)

VERSION = "rtd-v1-alfworld-c26-b"
WORLD_NAMES = ("game.tw-pddl", "traj_data.json", "initial_state.pddl")
ROUND_FOLDS = {1: (0, 1), 2: (1, 0), 3: (0, 1)}
INNER_USES = frozenset({"inner", "P", "normalizer", "projection", "source", "candidate", "pending"})


def _signed(value):
    value = deepcopy(value)
    value.pop("manifest_hash", None)
    return {**value, "manifest_hash": canonical_hash(value)}


def _check_signature(value):
    if value.get("manifest_hash") != _signed(value)["manifest_hash"]:
        raise ValueError("support manifest integrity mismatch")


def environment_identity(root, tokenizer_directory):
    """Read installed code/data bytes without importing third-party environments."""
    bank._privileged()
    root, tok = Path(root).resolve(), Path(tokenizer_directory).resolve()
    env_root = root / "envs/alfworld"
    python = env_root / ".venv/bin/python"
    if not python.is_file():
        raise FileNotFoundError(f"ALFWorld environment Python unavailable: {python}")
    paths = [root / name for name in (
        "src/bfas/adapters/alfworld.py", "src/alfworld_eval.py", "src/bfas/cc_pairs.py",
        "tools/behavior_atom/gpu_driver.py", "src/bfas/rtd/transport.py",
        "src/bfas/rtd/benchmarks/alfworld_state.py",
    )]
    versions = {}
    for package in ("alfworld", "textworld"):
        dirs = list((env_root / ".venv/lib").glob(f"python*/site-packages/{package}"))
        if len(dirs) != 1:
            raise FileNotFoundError(f"installed {package} environment unavailable")
        paths.extend(p for p in dirs[0].rglob("*") if p.is_file() and
                     "__pycache__" not in p.parts and p.suffix != ".pyc")
        metadata = sorted(dirs[0].parent.glob(f"{package}-*.dist-info/METADATA"))
        if len(metadata) != 1:
            raise ValueError(f"ambiguous {package} version")
        paths.extend(metadata)
        versions[package] = next(line.removeprefix("Version: ") for line in
                                 metadata[0].read_text().splitlines() if line.startswith("Version: "))
    # Bind dependency installation metadata too, including Gym/numpy/PDDL engines.
    paths.extend((env_root / ".venv/lib").glob("python*/site-packages/*.dist-info/RECORD"))
    source_files = {str(p.relative_to(root)): bank.file_hash(p) for p in sorted(set(paths))}
    tokenizer_files = {name: bank.file_hash(tok / name) for name in
                       ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")}
    scaffold = dict(instruction=adapter.TEACHER_REACT_INSTRUCTION,
                    prompt=adapter.TEACHER_REACT_PROMPT, examples=adapter.TEACHER_REACT_EXAMPLES)
    identity = dict(
        version=VERSION, versions=versions, source_files=source_files,
        python_sha256=bank.file_hash(python), tokenizer_files=tokenizer_files,
        scaffold=scaffold, scaffold_hash=canonical_hash(scaffold),
        environment_data_version="json_2.1.1", worker_config=adapter._worker_config(Path("<task-directory>")),
        renderer=dict(react=True, observation_tail=2000, history_entries=8, feedback_head=300,
                      chat_template="local Qwen tokenizer through ALFWorldAdapter._render",
                      thinking="existing cc_pairs.thinking_off", prior_examples="six fixed harness examples"),
        max_episode_steps=40, worker_horizon=50, domain_randomization=False,
        cpu_only=True, python_hash_seed=0, environment_seed="existing adapter default",
    )
    return {**identity, "environment_hash": canonical_hash(identity)}


def freeze_support(root, environment):
    """Freeze all historical demand parents, never select support by teacher success."""
    bank._privileged()
    sources = bank._Sources(root)
    split = sources.json(bank.SUPPORT)
    demand, calibration = sorted(split["demand"]), sorted(split["calibration"])
    probe = sorted(r["task_id"] for _, r, _ in sources.rows(bank.PROBE))
    ledger_tasks = {r["task_id"] for _, r, _ in sources.rows(bank.LEDGER)}
    if len(set(demand)) != len(demand) or set(demand) != ledger_tasks:
        raise ValueError("historical demand/ledger mismatch or duplicate task")
    if set(demand) & set(calibration):
        raise ValueError("calibration task in historical demand")
    protected = {parent_hash(t) for t in calibration + probe}
    groups = defaultdict(list)
    tasks = {}
    for tid in demand:
        h = parent_hash(tid)
        groups[h].append(tid)
        request = bank._reset_request(sources, tid, CapConfiguration(), environment["environment_hash"])
        tasks[tid] = dict(parent_hash=h, fold=parent_fold(h), request=request,
                          request_hash=canonical_hash(request), excluded=h in protected)
    parents = {h: dict(parent_game=tids[0].split("/")[0], task_ids=sorted(tids),
                       selected_task_id=min(tids), fold=parent_fold(h))
               for h, tids in sorted(groups.items()) if h not in protected}
    # Exact world duplication across different game names requires a new grouping
    # decision; fail rather than silently allowing related trials across folds.
    worlds = defaultdict(set)
    for tid, item in tasks.items():
        worlds[item["request"]["world_hash"]].add(item["parent_hash"])
    if any(len(hashes) > 1 for hashes in worlds.values()):
        raise ValueError("identical world bytes across parent names; grouping must be revised")
    overlaps = {}
    train_names = {t.split("/")[0] for t in demand}
    calibration_names = {t.split("/")[0] for t in calibration}
    for name in ("valid_seen", "valid_unseen"):
        directory = sources.path(f"envs/alfworld/data/json_2.1.1/{name}")
        names = {p.name for p in directory.iterdir() if p.is_dir()} if directory.exists() else set()
        same_names = sorted(train_names & names)
        matches = []
        for game in same_names:
            for gamefile in sorted((directory / game).glob("*/game.tw-pddl")):
                hashes = {n: bank.file_hash(gamefile.parent / n) for n in WORLD_NAMES}
                for tid in demand:
                    if tid.split("/")[0] == game:
                        expected = tasks[tid]["request"]["world_files"]
                        matches.append(dict(train_task=tid, evaluation_task=str(gamefile.parent.relative_to(directory)),
                                            world_files=hashes, world_hash=canonical_hash(hashes),
                                            identical_files=[n for n in WORLD_NAMES if hashes[n] == expected[n]],
                                            identical_world=hashes == expected))
        overlaps[name] = dict(parent_names=same_names, calibration_parent_names=sorted(calibration_names & names),
                              comparisons=matches, independent_certificate=False)
    value = dict(
        version=VERSION, benchmark="alfworld", split="train", scope="exploratory", frozen=True,
        historical_task_ids=demand, historical_parent_groups=len(groups), tasks=tasks, parents=parents,
        m=len(parents), training_task_ids=sorted(t for t in demand if not tasks[t]["excluded"]),
        calibration_task_ids=calibration, probe_task_ids=probe, protected_parent_hashes=sorted(protected),
        fold_parent_counts={str(f): sum(p["fold"] == f for p in parents.values()) for f in (0, 1)},
        fold_task_counts={str(f): sum(not t["excluded"] and t["fold"] == f for t in tasks.values()) for f in (0, 1)},
        fold_rule="SHA256(canonical JSON {benchmark:alfworld,split:train,parent_game:first task_id segment}) mod 2",
        rounds={str(r): dict(inner=i, feedback=f) for r, (i, f) in ROUND_FOLDS.items()},
        feedback_task_rule="uniform parents; lexicographically first trial; same trial for both rollouts",
        source_files=sources.files, environment=environment, evaluation_overlap=overlaps,
    )
    return validate_support(_signed(value))


def validate_support(manifest):
    _check_signature(manifest)
    if (manifest["version"] != VERSION or manifest["benchmark"] != "alfworld" or
            manifest["split"] != "train" or manifest["frozen"] is not True):
        raise ValueError("frozen C26-B train support required")
    protected = {parent_hash(t) for t in manifest["probe_task_ids"] + manifest["calibration_task_ids"]}
    if protected != set(manifest["protected_parent_hashes"]):
        raise ValueError("protected parent exclusion mismatch")
    tasks = manifest["tasks"]
    if sorted(tasks) != manifest["historical_task_ids"]:
        raise ValueError("support task inventory mismatch")
    if manifest["historical_parent_groups"] != len({parent_hash(t) for t in tasks}):
        raise ValueError("historical parent count mismatch")
    validate_parent_folds({tid: item["fold"] for tid, item in tasks.items()})
    expected = defaultdict(list)
    for tid, item in tasks.items():
        h = parent_hash(tid)
        request = validate_request(item["request"])
        if (item["parent_hash"] != h or item["excluded"] is not (h in protected)
                or request["task_id"] != tid or item["request_hash"] != canonical_hash(request)
                or request["environment_hash"] != manifest["environment"]["environment_hash"]):
            raise ValueError("task/parent/world/environment binding mismatch")
        if h not in protected:
            expected[h].append(tid)
    if set(expected) != set(manifest["parents"]) or manifest["m"] != len(expected):
        raise ValueError("protected or missing training parent")
    for h, tids in expected.items():
        p = manifest["parents"][h]
        if p != dict(parent_game=tids[0].split("/")[0], task_ids=sorted(tids),
                     selected_task_id=min(tids), fold=parent_fold(h)):
            raise ValueError("trials must share their deterministic parent fold")
    training = sorted(t for tids in expected.values() for t in tids)
    if training != manifest["training_task_ids"]:
        raise ValueError("protected task in training support")
    for f in (0, 1):
        if (manifest["fold_parent_counts"][str(f)] != sum(parent_fold(h) == f for h in expected)
                or manifest["fold_task_counts"][str(f)] != sum(parent_fold(parent_hash(t)) == f for t in training)):
            raise ValueError("fold count mismatch")
    if manifest["rounds"] != {str(r): dict(inner=i, feedback=f) for r, (i, f) in ROUND_FOLDS.items()}:
        raise ValueError("round fold schedule mismatch")
    env = deepcopy(manifest["environment"])
    env_hash = env.pop("environment_hash")
    if canonical_hash(env) != env_hash:
        raise ValueError("environment identity mismatch")
    return deepcopy(manifest)


class ALFWorldSupport:
    def __init__(self, manifest):
        self._manifest = validate_support(manifest)

    @property
    def manifest(self):
        return deepcopy(self._manifest)

    def parent_hashes(self, round_number, *, use="inner"):
        if type(round_number) is not int or round_number not in ROUND_FOLDS:
            raise ValueError("round must be 1, 2 or 3")
        if use not in INNER_USES | {"feedback"}:
            raise ValueError("unknown support use")
        fold = ROUND_FOLDS[round_number][use == "feedback"]
        return frozenset(h for h in self._manifest["parents"] if parent_fold(h) == fold)

    def guard_tasks(self, task_ids, round_number, *, use="inner"):
        tids = tuple(task_ids)
        allowed = self.parent_hashes(round_number, use=use)
        if any(t not in self._manifest["training_task_ids"] or parent_hash(t) not in allowed for t in tids):
            raise ValueError(f"probe/calibration/outside {use} fold task rejected")
        return tids

    def guard_states(self, states, round_number, *, use="inner", owned_packages=None):
        """No teacher prefix (even same-fold) without a matching owned behavior."""
        states = tuple(states)
        for state in states:
            tid = json.loads(state.task_json)["task_id"]
            self.guard_tasks([tid], round_number, use=use)
            expected = self._manifest["tasks"][tid]["request"]
            validate_full_state(state, expected_world_hash=expected["world_hash"])
            if json.loads(state.task_json) != expected:
                raise ValueError("full state differs from frozen support request")
            if len(json.loads(state.history_json)) > 1:
                # This API is for archived teacher states, not fresh student traces.
                if use == "feedback":
                    raise ValueError("feedback requires full task reset, never a teacher prefix")
                matches = []
                checked, visiting = set(), set()
                def check_package(q):
                    if q in visiting or q not in owned_packages:
                        raise ValueError("teacher prefix dependencies not owned or cyclic")
                    if q in checked:
                        return
                    visiting.add(q)
                    package = owned_packages[q]
                    if package.query_id != q:
                        raise ValueError("owned package identity mismatch")
                    for behavior in package.behaviors:
                        task = json.loads(behavior.state.task_json)
                        self.guard_tasks([task["task_id"]], round_number, use=use)
                        expected = self._manifest["tasks"][task["task_id"]]["request"]
                        validate_full_state(behavior.state, expected_world_hash=expected["world_hash"])
                        if task != expected:
                            raise ValueError("owned state differs from frozen support")
                    for dep in package.dependencies:
                        check_package(dep)
                    visiting.remove(q)
                    checked.add(q)
                for q, package in (owned_packages or {}).items():
                    if any(b.state == state for b in package.behaviors):
                        check_package(q)
                        matches.append(state)
                if not matches:
                    raise ValueError("teacher prefix not owned")
        return states

    def feedback_tasks(self, round_number, rng, count=4):
        parents = sorted(self.parent_hashes(round_number, use="feedback"))
        return tuple(self._manifest["parents"][h]["selected_task_id"] for h in rng.sample(parents, count))


class EnvironmentUnavailable(RuntimeError):
    """Infrastructure failed; never interpreted as reward zero."""


class BoundedEnvBridge(adapter._EnvBridge):
    """Preserve adapter JSON/step protocol, replacing spawn/read/cleanup only."""
    def __init__(self, task_id, *, timeout=30.0, episode_timeout=120.0):
        bank._privileged()
        parent_hash(task_id)
        if timeout <= 0 or episode_timeout <= 0:
            raise ValueError("positive worker deadlines required")
        self.timeout = timeout
        self.deadline = time.monotonic() + episode_timeout
        self.process = None
        self._tmp = tempfile.TemporaryDirectory(prefix="rtd-alfworld-c26b-")
        self._stderr = open(Path(self._tmp.name) / "worker.stderr", "w+")
        self._queue = queue.Queue()
        self._request_id = 0
        self._reader = None
        env = os.environ.copy()
        env.update(PYTHONPATH=str(adapter.ROOT / "src"), PYTHONDONTWRITEBYTECODE="1",
                   ALFWORLD_DATA=str(adapter.DATA.parent), ALFRED_DATA=str(adapter.DATA),
                   ALFWORLD_DATA_ROOT=str(adapter.DATA.parent), CUDA_VISIBLE_DEVICES="",
                   OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                   TOKENIZERS_PARALLELISM="false", PYTHONHASHSEED="0", TMPDIR=self._tmp.name,
                   HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", BFAS_ALFWORLD_MAX_STEPS="40")
        try:
            self.process = subprocess.Popen(
                [str(adapter.ENV_PYTHON), "-u", "-m", "bfas.adapters.alfworld", "--worker",
                 "--split", "train", "--task-id", task_id],
                cwd=self._tmp.name, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=self._stderr, text=True, encoding="utf-8", bufsize=1, start_new_session=True)
            self._reader = threading.Thread(target=self._read_lines, daemon=True)
            self._reader.start()
            if self._read().get("op") != "ready":
                raise EnvironmentUnavailable("ALFWorld worker did not become ready")
        except BaseException:
            self.close()
            raise

    def _read_lines(self):
        try:
            for line in self.process.stdout:
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(value, dict) and value.get("bfas_worker") is True:
                    self._queue.put(value)
        finally:
            self._queue.put(None)

    def _read(self):
        wait = min(self.timeout, self.deadline - time.monotonic())
        try:
            if wait <= 0:
                raise queue.Empty
            value = self._queue.get(timeout=wait)
        except queue.Empty as exc:
            raise EnvironmentUnavailable(f"ALFWorld worker timeout (read={self.timeout}s, episode deadline)") from exc
        if value is None:
            self._stderr.flush()
            self._stderr.seek(0)
            detail = self._stderr.read()[-3000:]
            raise EnvironmentUnavailable(f"ALFWorld worker exited ({self.process.poll()}): {detail}")
        return value

    def step(self, command):
        bank._privileged()
        return super().step(command)

    def close(self):
        process = self.process
        if process is not None:
            # Terminate the entire new session, including any TextWorld child.
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=3)
            for stream in (process.stdin, process.stdout):
                if stream is not None:
                    try:
                        stream.close()
                    except BrokenPipeError:
                        pass
            if self._reader is not None:
                self._reader.join(timeout=1)
            self.process = None
        self._stderr.close()
        self._tmp.cleanup()


class RealStepper:
    def __init__(self, *, timeout=30.0, episode_timeout=120.0, environment_hash):
        self.timeout, self.episode_timeout = timeout, episode_timeout
        self.environment_hash = environment_hash
        self.bridge = None
        self.request = None
        self.index = 0
        self.observations = []

    def _world(self, request):
        files = {n: bank.file_hash(adapter.DATA / "train" / request["task_id"] / n) for n in WORLD_NAMES}
        if files != request["world_files"] or canonical_hash(files) != request["world_hash"]:
            raise ValueError("wrong world hash: environment bytes differ from frozen request")

    def _observation(self, raw):
        if raw.get("op") != "state":
            raise ValueError("expected environment state")
        obs = Observation(self.index, raw["observation"], raw["admissible"],
                          self.request["world_hash"], raw["done"], raw["won"])
        self.observations.append(asdict(obs))
        return obs

    def reset(self, request):
        bank._privileged()
        validate_request(request)
        if self.bridge is not None:
            raise ValueError("fresh worker required for each reset")
        if request["environment_hash"] != self.environment_hash or request["max_episode_steps"] != 40:
            raise ValueError("environment/horizon differs from frozen request")
        self._world(request)
        self.request = deepcopy(request)
        self.bridge = BoundedEnvBridge(request["task_id"], timeout=self.timeout, episode_timeout=self.episode_timeout)
        raw = self.bridge._read()
        goal = adapter._goal_line(raw["observation"])
        if not goal:
            raise ValueError("missing goal in real reset observation")
        if goal != request["goal"]:
            raise ValueError("real reset goal differs from frozen request")
        return 0, self._observation(raw)

    def step(self, cursor, command):
        bank._privileged()
        if type(cursor) is not int or cursor != self.index:
            raise ValueError("out-of-order history cursor")
        if self.index >= 40:
            raise ValueError("configured episode horizon exceeded")
        raw = self.bridge.step(command)
        self.index += 1
        self._world(self.request)
        return self.index, self._observation(raw)

    def close(self):
        if self.bridge is not None:
            self.bridge.close()
            self.bridge = None


def prompt_messages(request, history, *, react=True):
    """Exact adapter context window, with the complete archive left intact."""
    lines = [f"> {history[i]['content']}\n{history[i + 1]['content'][:300]}"
             for i in range(1, len(history), 2)]
    fields = dict(obs=adapter._obs_with_goal(request["goal"], history[-1]["content"]),
                  cmds="\n".join(history[-1]["admissible"]), hist="\n".join(lines[-8:]) or "(start)")
    if not react:
        return [dict(role="user", content=adapter.PROMPT.format(**fields))]
    category = adapter._category(request["task_id"])
    return [dict(role="system", content=adapter.TEACHER_REACT_INSTRUCTION),
            dict(role="user", content=adapter.TEACHER_REACT_PROMPT.format(
                **fields, task_type=category, example=adapter.TEACHER_REACT_EXAMPLES[category]))]


class FrozenRenderer:
    def __init__(self, tokenizer_directory):
        bank._privileged()
        from transformers import AutoTokenizer
        self.adapter = adapter.ALFWorldAdapter()
        self.adapter._tokenizer = AutoTokenizer.from_pretrained(
            str(Path(tokenizer_directory).resolve()), local_files_only=True, trust_remote_code=False)

    def __call__(self, request, history):
        from ...cc_pairs import thinking_off
        return thinking_off(self.adapter._render(prompt_messages(request, history)))


def verify_package(payload, request, stepper, renderer, *, support=None):
    """Offline privileged audit; failures retain partial states and explicit reasons."""
    bank._privileged()
    output = deepcopy(payload)
    states, checks = [], []
    result = None
    reason = None
    error_type = None
    def capture(req, history):
        prompt = renderer(req, history)
        state = FullState.create(req, history, prompt, parent_hash(req["task_id"]))
        validate_full_state(state, expected_world_hash=req["world_hash"])
        states.append(asdict(state))
        index = (len(history) - 1) // 2
        turns = payload["historical_response"]["demo"]["turns"]
        if index < len(turns):
            expected = turns[index].get("context")
            actual = prompt_messages(req, history, react=False)
            if expected != actual:
                raise ValueError(f"replay divergence: archived deployment context at command {index}")
            checks.append(index)
        return prompt
    try:
        validate_request(request)
        if payload["status"] != "candidate" or payload["payload_kind"] != "extracted_teacher_commands":
            raise ValueError("candidate command payload required")
        if payload["provenance"]["task_id"] != request["task_id"]:
            raise ValueError("package/reset task mismatch")
        if payload.get("exclusion_reasons"):
            raise ValueError("protected probe/calibration parent")
        if support is not None:
            tid = request["task_id"]
            if tid not in support._manifest["training_task_ids"]:
                raise ValueError("outside frozen training support")
            if request != support._manifest["tasks"][tid]["request"]:
                raise ValueError("package/reset differs from frozen support")
        result = replay_commands(request, payload["commands"], stepper, capture)
        if result.won is not payload["success"] or not result.done or not result.won:
            raise ValueError("recorded success not reproduced")
    except (ValueError, KeyError, OSError, RuntimeError) as exc:
        reason, error_type = str(exc), type(exc).__name__
    finally:
        stepper.close()
    output.update(status="usable" if reason is None else "unavailable", unavailable_reason=reason,
                  request_state_hash=canonical_hash(request),
                  behaviors=[asdict(b) for b in result.behaviors] if result is not None and reason is None else [])
    output["provenance"].update(state_validation="C26-B real state replay" if isinstance(stepper, RealStepper)
                                else "C26-B injected stepper", original_request_state_hash=payload["request_state_hash"])
    output["verification"] = dict(
        replayed=True, success_reproduced=reason is None, unavailable_reason=reason, error_type=error_type,
        done=result.done if result else None, won=result.won if result else None,
        transcript_hash=result.transcript_hash if result else None, states=states,
        raw_observations=deepcopy(getattr(stepper, "observations", [])),
        archived_context_checks=checks, completed_commands=len(states) - 1 if states else 0,
    )
    return output


def verification_summary(records, payloads, support, *, historical_summary=None):
    usable = [p for p in payloads.values() if p["status"] == "usable"]
    replayed = [p for p in payloads.values() if p.get("verification", {}).get("replayed")]
    unavailable = [p for p in payloads.values() if p["status"] == "unavailable"]
    return dict(version=VERSION, scope="exploratory", attempts=len(payloads),
                candidate_packages=len(replayed), usable_packages=len(usable),
                unavailable_candidates=sum(p["status"] == "unavailable" for p in replayed),
                unavailable_packages=len(unavailable),
                unavailable_reasons=dict(Counter(p["unavailable_reason"] for p in unavailable)),
                command_turns=sum(len(p["behaviors"]) for p in usable),
                legacy_token_estimate=sum(p["cost"] or 0 for p in payloads.values()),
                usable_legacy_token_estimate=sum(p["cost"] for p in usable),
                m=support["m"], task_ids=len(support["training_task_ids"]),
                fold_parent_counts=support["fold_parent_counts"], fold_task_counts=support["fold_task_counts"],
                per_fold_usable={str(f): sum(parent_fold(parent_hash(p["provenance"]["task_id"])) == f for p in usable)
                                 for f in (0, 1)},
                per_fold_usable_parents={str(f): len({parent_hash(p["provenance"]["task_id"]) for p in usable
                                                     if parent_fold(parent_hash(p["provenance"]["task_id"])) == f})
                                         for f in (0, 1)},
                cap_audit=cap_audit(records, payloads), support_manifest_hash=support["manifest_hash"],
                new_teacher_calls=0, new_teacher_tokens=0, gpu_used=False,
                historical_inventory=historical_summary,
                limitations=[s for s in bank.LIMITATIONS if not s.startswith(("C26-A:", "107 is", "Proposed parent"))] + [
                    "Full ordered states are reconstructed now, not recovered historical raw API states.",
                    "Every saved deployment context is compared; historical context was windowed, so hidden historical world evolution cannot be proven beyond retained evidence.",
                    "State hashes bind per-trial world bytes, installed environment, tokenizer and frozen ReAct scaffold.",
                    "Aliases remain historical origin mappings, not validated off-policy event targets; only replayed command behaviors are usable.",
                    "valid_seen name/world overlap is disclosed in support; no independent certification claim.",
                    "No existing interfaces changed. BoundedEnvBridge subclasses the adapter; the C26-B auditor is additive because the C26-A auditor rejects usable evidence.",
                    "Training, P/normalizer integration and production selector import dispatch remain later C26 stages.",
                ])


def seal_verified_bank(directory, archive, support, payloads, reset_states):
    bank._privileged()
    directory = Path(directory).resolve()
    if directory.exists():
        raise FileExistsError(directory)
    records, requests = [], {}
    for old in archive.records:
        q = old.spec.query_id
        p = payloads[q]
        request = support["tasks"][p["provenance"]["task_id"]]["request"]
        requests[q] = request
        # Public metadata reveals availability only, never its hidden reason/outcome.
        state = reset_states.get(request["task_id"])
        state_hash = FullState(**state).state_hash if state else canonical_hash(request)
        records.append(replace(old, spec=replace(old.spec, state_hash=state_hash),
                               unavailable_reason=None if p["status"] == "usable" else "unavailable"))
    summary = verification_summary(records, payloads, support, historical_summary=archive.summary)
    seal_bank(directory, records, payloads)
    bank._write_new(directory / "public/reset_requests.json", requests)
    bank._write_new(directory / "public/reset_states.json", reset_states)
    bank._write_new(directory / "public/support.json", support)
    bank._write_new(directory / "sealed/audit.json", summary)
    bank._write_new(directory / "sealed/event_aliases.json", archive.aliases)
    artifacts = {str(p.relative_to(directory)): bank.file_hash(p) for p in sorted(directory.rglob("*.json"))}
    bank._write_new(directory / "sealed/manifest.json", dict(
        version=VERSION, artifacts=artifacts, support_manifest_hash=support["manifest_hash"],
        environment_hash=support["environment"]["environment_hash"], source_files=archive.summary["source_files"]))
    return summary


def audit_verified_bank(directory, *, expected_manifest_sha256=None):
    """Validate sealed bytes plus state/order/command/support semantics, without replay."""
    bank._privileged()
    directory = Path(directory).resolve()
    path = directory / "sealed/manifest.json"
    if expected_manifest_sha256 and bank.file_hash(path) != expected_manifest_sha256:
        raise ValueError("manifest integrity mismatch")
    manifest = json.loads(path.read_text())
    if manifest["version"] != VERSION:
        raise ValueError("C26-B bank required")
    files = {str(p.relative_to(directory)) for p in directory.rglob("*") if p.is_file()}
    if files != set(manifest["artifacts"]) | {"sealed/manifest.json"}:
        raise ValueError("artifact inventory mismatch")
    for name, expected in manifest["artifacts"].items():
        p = (directory / name).resolve()
        if not p.is_relative_to(directory) or bank.file_hash(p) != expected:
            raise ValueError(f"artifact integrity mismatch: {name}")
    support = validate_support(json.loads((directory / "public/support.json").read_text()))
    if manifest["support_manifest_hash"] != support["manifest_hash"]:
        raise ValueError("support manifest binding mismatch")
    resets = json.loads((directory / "public/reset_states.json").read_text())
    requests = json.loads((directory / "public/reset_requests.json").read_text())
    raws = json.loads((directory / "public/requests.json").read_text())
    integrity = json.loads((directory / "sealed/integrity.json").read_text())
    records, payloads = [], {}
    for raw in raws:
        q = raw["spec"]["query_id"]
        if not re.fullmatch(r"[a-f0-9]{64}", q):
            raise ValueError("invalid opaque query ID")
        p = json.loads((directory / f"sealed/{q}.json").read_text())
        request = requests[q]
        tid = request["task_id"]
        if request != support["tasks"][tid]["request"] or digest(p) != integrity[q] or q in payloads:
            raise ValueError("payload/reset/integrity mismatch")
        prov = p["provenance"]
        if bank.query_id(prov["ledger_sha256"], prov, prov["line"]) != q:
            raise ValueError("opaque query identity mismatch")
        if (p["query_id"] != q or p["request_state_hash"] != canonical_hash(request)
                or json.loads(p["raw_ledger_line"]) != p["historical_response"]):
            raise ValueError("historical payload binding mismatch")
        row = p["historical_response"]
        if (bank.query_id(prov["ledger_sha256"], row, prov["line"]) != q or prov["task_id"] != tid
                or p["cost"] != row.get("tokens_spent") or p["success"] is not row["verified"]):
            raise ValueError("historical cost/outcome mismatch")
        expected_record = bank.public_record(q, request)
        state_hash = canonical_hash(request)
        if tid in resets:
            reset = FullState(**resets[tid])
            validate_full_state(reset, expected_world_hash=request["world_hash"])
            if json.loads(reset.task_json) != request or len(json.loads(reset.history_json)) != 1:
                raise ValueError("public state must be the fixed reset")
            state_hash = reset.state_hash
        expected_record = replace(expected_record, spec=replace(expected_record.spec, state_hash=state_hash),
                                  unavailable_reason=None if p["status"] == "usable" else "unavailable")
        if json.loads(json.dumps(asdict(expected_record))) != raw:
            raise ValueError("public metadata contains hidden or inconsistent fields")
        if p["status"] == "usable":
            if tid not in support["training_task_ids"] or p["exclusion_reasons"] or p["dependencies"]:
                raise ValueError("protected/dependent usable package")
            verification = p["verification"]
            states = [FullState(**s) for s in verification["states"]]
            commands = [t["target"] for t in row["demo"]["turns"]]
            if (p["commands"] != commands or len(states) != len(commands) + 1 or
                    len(p["behaviors"]) != len(commands) or verification["won"] is not True or
                    verification["done"] is not True or verification["success_reproduced"] is not True):
                raise ValueError("successful complete command replay required")
            if asdict(states[0]) != resets[tid]:
                raise ValueError("public/sealed reset mismatch")
            for i, state in enumerate(states):
                validate_full_state(state, expected_world_hash=request["world_hash"])
                history = json.loads(state.history_json)
                if json.loads(state.task_json) != request or len(history) != 2 * i + 1:
                    raise ValueError("full state/reset/history mismatch")
                if i and history[:-2] != json.loads(states[i - 1].history_json):
                    raise ValueError("out-of-order history between archived states")
                if i < len(commands):
                    if (p["behaviors"][i] != dict(state=asdict(state), text=commands[i]) or
                            json.loads(states[i + 1].history_json)[-2]["content"] != commands[i]):
                        raise ValueError("teacher command/full state mismatch")
                    if prompt_messages(request, history, react=False) != row["demo"]["turns"][i]["context"]:
                        raise ValueError("archived context divergence")
            if canonical_hash(dict(states=[asdict(s) for s in states], commands=commands, done=True, won=True)) != verification["transcript_hash"]:
                raise ValueError("replay transcript integrity mismatch")
        elif p["status"] != "unavailable" or not p["unavailable_reason"] or p["behaviors"]:
            raise ValueError("invalid unavailable package")
        records.append(expected_record)
        payloads[q] = p
    if set(payloads) != set(integrity) or set(payloads) != set(requests):
        raise ValueError("request inventory mismatch")
    summary = json.loads((directory / "sealed/audit.json").read_text())
    if verification_summary(records, payloads, support, historical_summary=summary["historical_inventory"]) != summary:
        raise ValueError("verification summary mismatch")
    return dict(passed=True, attempts=len(records), usable_packages=summary["usable_packages"],
                manifest_sha256=bank.file_hash(path), support_manifest_hash=support["manifest_hash"])


def markdown_report(summary):
    cap = summary["cap_audit"]
    rows = [("Historical attempts", summary["attempts"]), ("Candidates replayed", summary["candidate_packages"]),
            ("Usable packages", summary["usable_packages"]), ("Unavailable candidates", summary["unavailable_candidates"]),
            ("All unavailable attempts", summary["unavailable_packages"]),
            ("Verified command turns", summary["command_turns"]), ("Training parents / task IDs", f"{summary['m']} / {summary['task_ids']}"),
            ("Fold 0 / 1 usable packages", f"{summary['per_fold_usable']['0']} / {summary['per_fold_usable']['1']}"),
            ("B_bank (public cap sum)", cap["B_bank"])]
    rows.extend((f"{r['percent']}% ceiling", r["budget"]) for r in cap["budget_affordability"])
    lines = ["# C26-B ALFWorld real replay verification", "", "CPU only; no teacher calls or model inference.", "",
             "| Metric | Result |", "|---|---:|"] + [f"| {k} | {v} |" for k, v in rows]
    lines += ["", "Unavailable reasons:", ""] + [f"- {n}: {reason}" for reason, n in summary["unavailable_reasons"].items()]
    lines += ["", f"Historical estimated spend: {summary['legacy_token_estimate']}; usable-package estimated spend: {summary['usable_legacy_token_estimate']}.",
              "Budget ceilings use uniform public caps, not these estimated costs.", "", "Limitations:", ""]
    lines += [f"- {s}" for s in summary["limitations"]]
    return "\n".join(lines) + "\n"
