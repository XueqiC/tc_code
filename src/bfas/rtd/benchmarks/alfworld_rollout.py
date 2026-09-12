"""C26-C full-task stochastic ALFWorld episodes; no model or worker on import.

Inject an HFGenerateBackend-compatible backend configured with
max_action_tokens=256, a C26-B renderer, a fresh zero-argument Stepper factory,
and a journal with append(kind, **record). RealStepper from C26-B is the real
environment implementation: it uses only the adapter's bounded CPU subprocess,
explicit data roots, offline settings and a temporary working directory.

Task-start feedback uses collect_feedback/feedback and the existing RTD LOO
kernel unchanged. Owned teacher-prefix continuations are separately available
for diagnostics; they cannot masquerade as task-start return gradients. No
teacher calls, shaped rewards, command retokenization or downloads. Feedback
retries a failed RPC once with a fresh worker and the same sampled trajectory.
"""
from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
import json
import logging
import math
import time
from typing import Protocol

import torch

from ...adapters.alfworld import ALFWorldAdapter, ALFWorldRPCError
from ..return_gradient import ActionTrace, TaskRollout, reinforce_gradient
from ..transport import FullState
from .alfworld_state import (
    Observation, Stepper, _observed, canonical_hash, parent_hash,
    validate_full_state, validate_request,
)


MAX_ACTION_TOKENS = 256
MAX_EPISODE_STEPS = 40


class SamplingBackend(Protocol):
    """Same sampling and differentiable scoring boundary as HFGenerateBackend."""
    backend_id: str
    max_action_tokens: int
    max_context_tokens: int
    tokenizer: object

    def identity(self, parameters): ...
    def sample_action(self, prompt, parameters, generator, *, temperature=1., top_p=1.) -> ActionTrace: ...
    def checked_score_action(self, action, parameters, *, expected_prompt_ids=None,
                             record=None, score_atol=None, score_rtol=None): ...


class EnvironmentFactory(Protocol):
    """A new owned worker per call; reset fixes the world, not the policy seed."""
    def __call__(self) -> Stepper: ...  # implementation must also supply close()


class IncompleteFeedbackError(RuntimeError):
    """An incomplete K-rollout measurement must not become a zero reward."""

    def __init__(self, message, episodes=()):
        super().__init__(message)
        self.episodes = tuple(episodes)


class _EnvironmentFailure(Exception):
    def __init__(self, stage, cause, elapsed):
        self.reason = dict(stage=stage, exception_type=type(cause).__name__, message=str(cause),
            elapsed_seconds=elapsed, rpc_failure=isinstance(cause, (ALFWorldRPCError, OSError)),
            worker_exception=f"{type(cause).__name__}: {cause}", exit_code=None, stderr_tail="")
        self.reason.update(getattr(cause, "diagnostics", {}))
        super().__init__(str(cause))


def _env_call(stage, fn, *args):
    started = time.monotonic()
    try:
        return fn(*args)
    except Exception as exc:
        raise _EnvironmentFailure(stage, exc, time.monotonic() - started) from exc


def _env_step(executor, stage, env, cursor, command):
    if executor is None:
        return _env_call(stage, env.step, cursor, command)
    # Only the RPC runs on a thread. The caller resumes validation, rendering,
    # retry/RNG bookkeeping and journaling after receiving this future.
    return (yield executor.submit(_env_call, stage, env.step, cursor, command))


def episode_seed(task_id, rollout_index, *, base_seed=0):
    """Stable across processes, task ordering and resume; no Python hash/RNG use.

    A new feedback measurement supplies a new base_seed (e.g. round/step/role).
    The fixed environment reset retains C26-B's environment seed convention.
    """
    parent_hash(task_id)
    if any(type(x) is not int or x < 0 for x in (base_seed, rollout_index)):
        raise ValueError("nonnegative integer seed and rollout index required")
    return int(canonical_hash(dict(version="c26-c", task_id=task_id,
                                   rollout_index=rollout_index, base_seed=base_seed))[:16], 16) % (2**63)


class _MatchedCommand(str):
    """Observe exact membership without changing string matching semantics."""

    def __new__(cls, value, exact_matches):
        obj = super().__new__(cls, value)
        obj.exact_matches = exact_matches
        return obj

    def __eq__(self, other):
        equal = super().__eq__(other)
        if equal is True:
            self.exact_matches.append(True)
        return equal

    __hash__ = str.__hash__


def parse_action(text, admissible):
    # Exact membership returns the normalized input line; case/substring
    # matches return an admissible object. Observe both paths to distinguish
    # matched 'look' from literal fallback without maintaining another parser.
    exact_matches = []
    command = ALFWorldAdapter._teacher_command(text, [_MatchedCommand(c, exact_matches) for c in admissible])
    return str(command), not exact_matches and not isinstance(command, _MatchedCommand)


@dataclass(frozen=True)
class EpisodeStep:
    state: FullState
    action: ActionTrace
    command: str
    parser_fallback: bool
    observation: Observation | None = None  # None when the step RPC failed


@dataclass(frozen=True)
class ALFWorldEpisode:
    task_id: str
    rollout_index: int
    seed: int
    policy_id: str
    prefix_package_id: str | None
    prefix_steps: int
    start_state: FullState | None
    steps: tuple[EpisodeStep, ...]
    final_observation: Observation | None
    failure: dict | None = None
    horizon_reached: bool = False
    attempt: int = 0

    @property
    def actions(self):
        return tuple(s.action for s in self.steps)

    @property
    def from_task_start(self):
        return self.prefix_steps == 0 and self.prefix_package_id is None

    @property
    def success(self):
        if self.failure is not None or self.final_observation is None:
            return None
        return self.final_observation.won

    @property
    def reward(self):
        return None if self.success is None else float(self.success)

    @property
    def truncated(self):
        return self.horizon_reached or any(a.truncated for a in self.actions)

    def as_task_rollout(self):
        if self.failure is not None or self.reward is None:
            raise IncompleteFeedbackError("failed environment episode excluded from feedback", [self])
        if not self.from_task_start:
            raise ValueError("feedback requires full task reset, never a teacher prefix")
        return TaskRollout(self.task_id, self.actions, self.reward, self.policy_id)

    def record(self):
        return dict(**asdict(self), status="failed_with_reason" if self.failure else "completed",
                    excluded=self.failure is not None, success=self.success, reward=self.reward,
                    from_task_start=self.from_task_start, truncated=self.truncated,
                    sampled_steps=len(self.steps),
                    total_steps=self.final_observation.index if self.final_observation else 0,
                    parser_fallbacks=sum(s.parser_fallback for s in self.steps),
                    token_counts=dict(action=sum(len(a.action_ids) for a in self.actions),
                                      prompt=sum(len(a.prompt_ids) for a in self.actions),
                                      sampled_eos=sum(not a.truncated for a in self.actions),
                                      observation_loss=0))


def _observation(value, request, index):
    if not isinstance(value, Observation) or value.index != index:
        raise ValueError("out-of-order environment observation")
    if value.world_hash != request["world_hash"]:
        raise ValueError("environment world hash mismatch")
    if type(value.done) is not bool or type(value.won) is not bool:
        raise ValueError("environment done/won must be bool")
    return value


def _state(request, history, renderer):
    # C26-B's teacher validator requires admissible archived commands. Student
    # fallback 'look' can be absent from that list, so retain it in a FullState
    # without imposing the teacher-only admissibility constraint.
    return FullState.create(request, history, renderer(deepcopy(request), deepcopy(history)),
                            parent_hash(request["task_id"]))


def _validate_action(action, state, backend, parameters):
    if not isinstance(action, ActionTrace) or action.malformed:
        raise ValueError("ALFWorld requires ActionTrace without BFCL malformed flags")
    if action.backend_id != backend.backend_id or action.policy_id != backend.identity(parameters):
        raise ValueError("generation backend/policy mismatch")
    if action.prompt_ids != tuple(backend.tokenizer.encode(state.prompt, add_special_tokens=False)):
        raise ValueError("prompt token boundary/template mismatch")
    from ..student import termination_ids
    if action.eos_token_id not in termination_ids(backend):
        raise ValueError("configured EOS differs from sampled EOS")
    metadata = action.generation_metadata
    if any(metadata.get(k) != v for k, v in dict(temperature=1., top_p=1., top_k=0,
                                               max_action_tokens=MAX_ACTION_TOKENS).items()):
        raise ValueError("generation requires temperature=1/top_p=1/top_k=0 and agent_action=256")
    limit = min(MAX_ACTION_TOKENS, backend.max_context_tokens - len(action.prompt_ids))
    if (metadata.get("effective_action_limit") != limit or not 0 < len(action.action_ids) <= limit
            or (action.truncated and len(action.action_ids) != limit)):
        raise ValueError("generation must end at sampled EOS or the effective action cap")
    values = action.generation_token_logprobs
    if (len(values) != len(action.action_ids) or any(not math.isfinite(v) or v > 0 for v in values)
            or not math.isclose(sum(values), action.generation_logprob, rel_tol=2e-7, abs_tol=1e-6)):
        raise ValueError("complete finite generation token logprobs required")
    decoded = backend.tokenizer.decode(action.action_ids if action.truncated else action.action_ids[:-1],
                                       skip_special_tokens=False)
    if decoded != action.text:
        raise ValueError("action text differs from the sampled token IDs")


def _alfworld_task_rollout_stream(task_ref, backend: SamplingBackend, parameters, *, env_factory: EnvironmentFactory,
                         renderer, journal, rollout_index=0, base_seed=0,
                         prefix_package_id=None, owned_packages=None, support=None, round_number=None,
                         attempt=0, step_executor=None):
    """Run to terminal success/failure or 40 total environment actions.

    task_ref is a fixed train request dict, a reset FullState, or an owned
    teacher-prefix FullState. A supplied FullState is verified by deterministic
    reset/replay before sampling. Prefixes additionally require C26-B support,
    inner-fold and dependency ownership guards. Prefix actions consume the same
    40-step task horizon but contribute no student loss.

    Environment exceptions return an explicitly failed, journaled episode with
    reward=None. Policy/renderer/contract exceptions are journaled and re-raised.
    The backend is caller-owned and must already have agent_action=256; no
    mutation of its BFCL action_limit or any process-global backend is needed.
    """
    expected = task_ref if isinstance(task_ref, FullState) else None
    request = deepcopy(json.loads(expected.task_json) if expected else task_ref)
    validate_request(request)
    if request["max_episode_steps"] != MAX_EPISODE_STEPS:
        raise ValueError("C26-C requires alfworld_max_episode_steps=40")
    if (backend.max_action_tokens != MAX_ACTION_TOKENS or
            getattr(backend, "action_caps", {}).get("agent_action", MAX_ACTION_TOKENS) != MAX_ACTION_TOKENS):
        raise ValueError("configure backend max_action_tokens and agent_action=256 before rollout")
    target_history = json.loads(expected.history_json) if expected else None
    prefix_steps = (len(target_history) - 1) // 2 if expected else 0
    if expected:
        validate_full_state(expected, expected_world_hash=request["world_hash"])
        if target_history[-1]["done"] or prefix_steps >= MAX_EPISODE_STEPS:
            raise ValueError("sampling requires a nonterminal state with remaining steps")
    if prefix_steps:
        if support is None or round_number is None or prefix_package_id not in (owned_packages or {}):
            raise ValueError("teacher prefix package must be owned with a support/fold guard")
        package = owned_packages[prefix_package_id]
        if package.query_id != prefix_package_id or not any(b.state == expected for b in package.behaviors):
            raise ValueError("prefix package does not own this exact FullState")
        support.guard_states([expected], round_number, use="inner", owned_packages=owned_packages)
    elif prefix_package_id is not None:
        raise ValueError("prefix package ID requires a teacher-prefix FullState")
    seed = episode_seed(request["task_id"], rollout_index, base_seed=base_seed)
    # CPU tests create only a CPU generator. No device detection or model load.
    generator = torch.Generator(device=next(iter(parameters.values())).device).manual_seed(seed)
    policy_id = backend.identity(parameters)
    env, observation, start, failure = None, None, None, None
    steps, history = [], []
    pending = None
    try:
        env = _env_call("factory", env_factory)
        cursor, observation = _env_call("reset", env.reset, deepcopy(request))
        observation = _observation(observation, request, 0)
        history.append(_observed(observation))
        for i in range(prefix_steps + 1):
            if expected and history != target_history[:len(history)]:
                raise ValueError("reset/prefix replay differs from owned FullState history")
            if i == prefix_steps:
                break
            command = target_history[2 * i + 1]["content"]
            cursor, observation = yield from _env_step(step_executor, "prefix_step", env, cursor, command)
            observation = _observation(observation, request, i + 1)
            history.extend([dict(role="assistant", index=i + 1, content=command), _observed(observation)])
        start = _state(request, history, renderer)
        if expected:
            expected.assert_matches(start)
        if observation.done:
            raise ValueError("task reset/prefix unexpectedly terminal")
        for index in range(prefix_steps, MAX_EPISODE_STEPS):
            state = start if index == prefix_steps else _state(request, history, renderer)
            action = yield (state.prompt, generator)
            _validate_action(action, state, backend, parameters)
            command, fallback = parse_action(action.text, observation.admissible)
            steps.append(EpisodeStep(state, action, command, fallback))
            cursor, observation = yield from _env_step(step_executor, "step", env, cursor, command)
            observation = _observation(observation, request, index + 1)
            steps[-1] = replace(steps[-1], observation=observation)
            history.extend([dict(role="assistant", index=index + 1, content=command), _observed(observation)])
            if observation.done:
                break
    except _EnvironmentFailure as exc:
        failure = exc.reason
    except BaseException as exc:
        # Interrupts still propagate, but their partial measurement must not
        # leave a misleading completed/reward-zero record in the journal.
        failure = dict(stage="policy_or_state_contract", exception_type=type(exc).__name__, message=str(exc))
        pending = exc
    finally:
        if env is not None:
            try:
                _env_call("close", env.close)
            except _EnvironmentFailure as exc:
                failure = dict(failure, cleanup_failure=exc.reason) if failure else exc.reason
        episode = ALFWorldEpisode(request["task_id"], rollout_index, seed, policy_id,
            prefix_package_id, prefix_steps, start, tuple(steps), observation, failure,
            horizon_reached=bool(observation and not observation.done and
                                 observation.index == MAX_EPISODE_STEPS), attempt=attempt)
        if failure:
            logging.getLogger(__name__).error(
                "ALFWorld episode failed task_id=%s rollout_index=%s seed=%s attempt=%s steps=%s %s",
                episode.task_id, rollout_index, seed, attempt, len(steps), json.dumps(failure))
        journal.append("alfworld_episode", **episode.record())
    if pending is not None:
        raise pending
    return episode


def _run_serial(stream, backend, parameters):
    """Drive the same episode state machine one sampling request at a time."""
    try:
        request = next(stream)
        while True:
            prompt, generator = request
            try:
                action = backend.sample_action(prompt, parameters, generator, temperature=1., top_p=1.)
            except BaseException as exc:
                stream.throw(exc)  # retain the failed episode before propagating
                raise
            request = stream.send(action)
    except StopIteration as completed:
        return completed.value
    finally:
        stream.close()


def alfworld_task_rollout(task_ref, backend, parameters, **kwargs):
    """Serial driver of the full-task/prefix episode state machine."""
    return _run_serial(_alfworld_task_rollout_stream(task_ref, backend, parameters, **kwargs),
                       backend, parameters)


def _alfworld_retry_stream(task_ref, backend, parameters, *, journal, **kwargs):
    """Cache actions and RNG transitions across one fresh-worker RPC retry.

    Cached requests never reach the scheduler or consume a second RNG ticket.
    A batched first action leaves the episode generator untouched, as before.
    """
    cached = []
    for attempt in range(2):
        stream = _alfworld_task_rollout_stream(task_ref, backend, parameters,
            journal=journal, attempt=attempt, **kwargs)
        try:
            request = next(stream)
            index = 0
            while True:
                if isinstance(request, Future):
                    try:
                        response = yield request
                    except BaseException as exc:
                        request = stream.throw(exc)
                    else:
                        request = stream.send(response)
                    continue
                prompt, generator = request
                before = generator.get_state().clone()
                if index < len(cached):
                    previous_prompt, action, previous_rng, after = cached[index]
                    if prompt != previous_prompt or not torch.equal(before, previous_rng):
                        raise ValueError("RPC retry differs from original prompt/sampling stream")
                    generator.set_state(after)
                else:
                    try:
                        action = yield (prompt, generator)
                    except BaseException as exc:
                        stream.throw(exc)
                        raise
                    cached.append((prompt, action, before, generator.get_state().clone()))
                index += 1
                request = stream.send(action)
        except StopIteration as completed:
            episode = completed.value
        except BaseException as exc:
            try:
                stream.throw(exc)
            except BaseException:
                pass
            raise
        finally:
            stream.close()
        if (attempt or not episode.failure or not episode.failure.get("rpc_failure") or
                episode.failure["stage"] not in {"factory", "reset", "prefix_step", "step"}):
            return episode
        journal.append("alfworld_episode_retry", task_id=episode.task_id,
            rollout_index=episode.rollout_index, seed=episode.seed, attempt=1,
            prefix_package_id=episode.prefix_package_id, prefix_steps=episode.prefix_steps,
            failure=episode.failure)


def alfworld_task_rollout_with_retry(task_ref, backend, parameters, *, journal, **kwargs):
    """Serial execution, with identical reset, traces and RNG on an RPC retry."""
    return _run_serial(_alfworld_retry_stream(task_ref, backend, parameters,
        journal=journal, **kwargs), backend, parameters)


def alfworld_task_rollouts_lockstep(task_refs, backend, parameters, *, env_factory, renderer,
                                   journal, rollout_indices, first_actions, base_seed=0):
    """Drive one owned environment subprocess per episode at sampling barriers.

    Threads overlap step RPCs to the existing bounded workers. Only the caller
    touches the policy, RNG streams, renderer and journal. Physical generation
    sub-batches dispatch their steps immediately, overlapping later generation.
    Retry replay runs to the next unsampled prompt before joining the barrier.
    Results retain input order, even when shorter episodes finish first.
    """
    refs, indices, starts = tuple(task_refs), tuple(rollout_indices), tuple(first_actions)
    if not refs or len(refs) != len(indices) or len(refs) != len(starts):
        raise ValueError('aligned nonempty lockstep episode inputs required')
    executor = ThreadPoolExecutor(max_workers=len(refs), thread_name_prefix='alfworld-feedback')
    streams = [_alfworld_retry_stream(ref, backend, parameters, env_factory=env_factory,
        renderer=renderer, journal=journal, rollout_index=index, base_seed=base_seed,
        step_executor=executor)
        for ref, index in zip(refs, indices)]
    from ..generation_batch import run_episode_streams_lockstep
    return run_episode_streams_lockstep(streams, backend, parameters, executor, first_actions=starts)


def collect_feedback(task_refs, backend, parameters, *, env_factory, renderer, journal,
                     support, round_number, base_seed=0, rollouts_per_task=2):
    """Collect K independent resets of each selected legal actual trial.

    The caller chooses four parents through C26-B support.feedback_tasks (which
    includes parents without teacher payloads), then supplies their reset states
    or requests here. RPC failures get one fresh-worker retry before invalidating
    this measurement. Retain every attempt but never compute a partial-task LOO
    or change task weights by dropping an unlucky worker/task.
    """
    refs = tuple(task_refs)
    if type(rollouts_per_task) is not int or rollouts_per_task < 2 or not refs:
        raise ValueError("nonempty feedback and at least two rollouts per task required")
    requests = [json.loads(r.task_json) if isinstance(r, FullState) else r for r in refs]
    tids = [r["task_id"] for r in requests]
    if len(set(map(parent_hash, tids))) != len(tids):
        raise ValueError("duplicate feedback parent/trial")
    support.guard_tasks(tids, round_number, use="feedback")
    manifest = support.manifest
    for ref, request in zip(refs, requests):
        if (request != manifest["tasks"][request["task_id"]]["request"] or
                request["task_id"] != manifest["parents"][parent_hash(request["task_id"])]["selected_task_id"]):
            raise ValueError("feedback differs from frozen request/selected trial")
        if isinstance(ref, FullState):
            support.guard_states([ref], round_number, use="feedback")
    episodes = tuple(alfworld_task_rollout_with_retry(ref, backend, parameters, env_factory=env_factory,
        renderer=renderer, journal=journal, rollout_index=i, base_seed=base_seed)
        for ref in refs for i in range(rollouts_per_task))
    if any(e.failure for e in episodes):
        journal.append("alfworld_feedback_excluded", reason="incomplete K-rollout measurement",
                       task_ids=tids, failed=[dict(task_id=e.task_id, rollout_index=e.rollout_index,
                                                  failure=e.failure) for e in episodes if e.failure])
        raise IncompleteFeedbackError("environment failure: feedback block excluded; no gradient", episodes)
    return episodes


def alfworld_reinforce_gradient(episodes, backend, parameters, *, journal, score_atol=None, score_rtol=None):
    """Stream every sampled action through unchanged checked_score_action/LOO.

    scoring.py's state_hash remains its prompt-token hash; full_state_hash is
    additionally journaled to distinguish full histories behind windowed prompts.
    """
    episodes = tuple(episodes)
    rollouts = tuple(e.as_task_rollout() for e in episodes)
    starts, identities = {}, set()
    for e in episodes:
        identity = (e.task_id, e.rollout_index)
        if identity in identities or e.start_state is None:
            raise ValueError("distinct same-task rollout indices and complete starts required")
        identities.add(identity)
        prior = starts.setdefault(e.task_id, e.start_state)
        prior.assert_matches(e.start_state)
        for step in e.steps:
            _validate_action(step.action, step.state, backend, parameters)
    by_rollout = {id(r): e for r, e in zip(rollouts, episodes)}
    def record(rollout, index, diagnostic):
        e = by_rollout[id(rollout)]
        journal.append("score_consistency", **diagnostic, task_id=e.task_id,
                       rollout_index=e.rollout_index, action_index=index,
                       full_state_hash=e.steps[index].state.state_hash, prefix_package_id=None)
    return reinforce_gradient(rollouts, backend, parameters, diagnostic_record=record,
                              score_atol=score_atol, score_rtol=score_rtol)


def feedback(task_refs, backend, parameters, *, env_factory, renderer, journal,
             support, round_number, base_seed=0, rollouts_per_task=2):
    """Additive feedback boundary for later experiment integration (not wired in)."""
    episodes = collect_feedback(task_refs, backend, parameters, env_factory=env_factory,
        renderer=renderer, journal=journal, support=support, round_number=round_number,
        base_seed=base_seed, rollouts_per_task=rollouts_per_task)
    return alfworld_reinforce_gradient(episodes, backend, parameters, journal=journal)
