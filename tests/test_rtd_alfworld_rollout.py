"""CPU trajectory enumeration, stochastic score checks and one real train task."""
from copy import deepcopy
from dataclasses import replace
from itertools import product
import json
from pathlib import Path
import warnings

import pytest
import torch
import torch.nn.functional as F

from bfas.behavior.deltas import tensor_state_hash
from bfas.rtd.benchmarks.alfworld_rollout import (
    IncompleteFeedbackError, alfworld_reinforce_gradient, alfworld_task_rollout,
    collect_feedback, episode_seed, feedback, parse_action,
)
from bfas.rtd.benchmarks.alfworld_state import Observation, canonical_hash, replay_commands
from bfas.rtd.benchmarks.alfworld_support import (
    ALFWorldSupport, EnvironmentUnavailable, FrozenRenderer, RealStepper,
    adapter, freeze_support, prompt_messages,
)
from bfas.rtd.broker import PurchasedEvidencePackage
from bfas.rtd.persistence import ComputeJournal, digest
from bfas.rtd.return_gradient import ActionTrace, TorchPolicyBackend, reinforce_gradient
from bfas.rtd.scoring import ScoreTolerance
from bfas.rtd.transport import Behavior, FullState


ADVANCE = "go to table 1"
PIECES = {0: "", 10: "THOUGHT: route A.\n", 11: "THOUGHT: route B.\n",
          20: "ACTION: look", 21: f"ACTION: {ADVANCE}", 30: "?"}


class MemoryJournal:
    def __init__(self):
        self.events = []

    def append(self, kind, **values):
        # Enforce the same JSON finiteness/serialization boundary as the journal.
        self.events.append(json.loads(json.dumps(dict(kind=kind, **values), allow_nan=False)))


class TinyTokenizer:
    eos_token_id = 0

    def encode(self, text, *, add_special_tokens=False):
        assert add_special_tokens is False
        # The runner may encode the full prompt, never the executed command.
        assert text not in (ADVANCE, "look"), "executed command retokenized"
        return [1000 + ord(c) for c in text]

    def decode(self, ids, *, skip_special_tokens=False):
        assert skip_special_tokens is False
        return "".join(chr(i - 1000) if i >= 1000 else PIECES[i] for i in ids)


def render(request, history):
    return json.dumps(dict(index=history[-1]["index"], observation=history[-1]["content"],
                           commands=[h["content"] for h in history if h["role"] == "assistant"]))


class EnumerableEnv:
    """Two/three turns: succeed iff every command advances; no intermediate reward.

    A recovery mode succeeds after two advances, so a fallback/look can be
    followed by success. Terminal observations carry an irrelevant reward-like
    string to expose accidental observation scoring.
    """
    def __init__(self, *, horizon=2, recover=False, fault=None, long_observation=False):
        self.horizon, self.recover, self.fault = horizon, recover, fault
        self.long_observation = long_observation
        self.commands, self.closed = [], False

    def observation(self):
        i = len(self.commands)
        won = (self.commands.count(ADVANCE) >= 2 if self.recover else
               i == self.horizon and all(c == ADVANCE for c in self.commands))
        text = f"Observation {i}; intermediate reward=999; your task is unchanged."
        if self.long_observation:
            text += " unscored-observation" * 120
        return Observation(i, text, (ADVANCE, "look"), self.request["world_hash"],
                           done=won or i == self.horizon, won=won)

    def reset(self, request):
        self.request = deepcopy(request)
        if self.fault == "reset":
            raise EnvironmentUnavailable("worker startup unavailable")
        return 0, self.observation()

    def step(self, cursor, command):
        assert cursor == len(self.commands)
        if self.fault == "step":
            raise BrokenPipeError("worker died after sampling")
        self.commands.append(command)
        return len(self.commands), self.observation()

    def close(self):
        self.closed = True
        if self.fault == "close":
            raise TimeoutError("worker cleanup timed out")


class TinyBackend(TorchPolicyBackend):
    """Explicit CPU categorical policy, reusing checked_score_action unchanged.

    Each action samples a thought z and a command a, followed by deterministic
    EOS. P(z=1)=sigmoid(theta[0]); P(a=1|z,s)=sigmoid(theta[s+1]+.8*(2z-1)).
    Thus omitting thoughts or any later action changes the exact task gradient.
    """
    def __init__(self, script=None):
        self.tokenizer = TinyTokenizer()
        self.max_action_tokens, self.max_context_tokens = 256, 32768
        self.action_caps = {"agent_action": 256}
        self.backend_id = "c26-c-enumerable-cpu"
        self.score_tolerance = ScoreTolerance(mean_abs=1e-10, max_abs=1e-10,
                                             max_abs_outlier_tokens=0, max_abs_hard=1e-9)
        self.script = iter(script) if script is not None else None
        self.scored = []
        self.score_offset = None

    def identity(self, parameters):
        return tensor_state_hash(parameters)

    def values(self, prompt_ids, action_ids, parameters):
        theta = parameters["theta"]
        s = min(json.loads(self.tokenizer.decode(prompt_ids))["index"], len(theta) - 2)
        z = int(action_ids[0] == 11)
        values = []
        for token in action_ids:
            if token in (10, 11):
                values.append(F.logsigmoid(theta[0] * (1 if token == 11 else -1)))
            elif token in (20, 21):
                logit = theta[s + 1] + .8 * (2 * z - 1)
                values.append(F.logsigmoid(logit * (1 if token == 21 else -1)))
            else:
                values.append(theta.sum() * 0)  # deterministic EOS/filler
        return torch.stack(values)

    def sample_action(self, prompt, parameters, generator, *, temperature=1., top_p=1.):
        assert temperature == top_p == 1 and generator.device.type == "cpu"
        pids = tuple(self.tokenizer.encode(prompt, add_special_tokens=False))
        limit = min(self.max_action_tokens, self.max_context_tokens - len(pids))
        if self.script is None:
            theta = parameters["theta"].detach()
            s = json.loads(prompt)["index"]
            z = int(torch.rand((), generator=generator) < theta[0].sigmoid())
            a = int(torch.rand((), generator=generator) < (theta[s + 1] + .8 * (2 * z - 1)).sigmoid())
            ids = (10 + z, 20 + a, 0)
        else:
            ids = tuple(next(self.script))
        values = tuple(self.values(pids, ids, parameters).detach().tolist())
        truncated = ids[-1] != 0
        return ActionTrace(pids, ids, 0, self.tokenizer.decode(ids if truncated else ids[:-1]),
            sum(values), self.backend_id, self.identity(parameters), values,
            dict(implementation="enumerable-categorical", temperature=temperature, top_p=top_p, top_k=0,
                 max_action_tokens=self.max_action_tokens, effective_action_limit=limit), truncated=truncated)

    def score_action(self, action, parameters, *, verify_policy=True, return_details=False):
        assert action.backend_id == self.backend_id
        if verify_policy:
            assert action.policy_id == self.identity(parameters)
        self.scored.append(action.action_ids)
        values = self.values(action.prompt_ids, action.action_ids, parameters)
        if self.score_offset is not None:
            values = values + values.new_tensor(self.score_offset)
        score = values.sum()
        if return_details:
            return score, values, dict(implementation="enumerable-teacher-forced", device="cpu")
        return score


@pytest.fixture
def parameters():
    return {"theta": torch.tensor([.35, -.2, .4, .1], dtype=torch.float64, requires_grad=True)}


@pytest.fixture
def request_state():
    files = {n: canonical_hash(n) for n in ("game.tw-pddl", "traj_data.json", "initial_state.pddl")}
    return dict(benchmark="alfworld", split="train", task_id="pick_and_place_simple-Apple-None-Table-1/trial-1",
                goal="put apple on table.", world_files=files, world_hash=canonical_hash(files),
                environment_hash="a" * 64, max_episode_steps=40, state_kind="train_reset_request")


def run(request, parameters, *, backend=None, env=None, journal=None, index=0, **kwargs):
    backend = backend if backend is not None else TinyBackend()
    env = env if env is not None else EnumerableEnv()
    journal = journal if journal is not None else MemoryJournal()
    return alfworld_task_rollout(request, backend, parameters, env_factory=lambda: env,
                                 renderer=render, journal=journal, rollout_index=index, **kwargs)


def test_exact_enumerated_task_gradient_and_exact_k2_loo(request_state, parameters):
    # Independent analytic task-start J, summing the latent thought each turn.
    theta = parameters["theta"]
    pz = theta[0].sigmoid()
    pa = (1 - pz) * (theta[1:3] - .8).sigmoid() + pz * (theta[1:3] + .8).sigmoid()
    expected = torch.autograd.grad(pa.prod(), theta)[0]
    assert expected[:3].abs().min() > .01  # both thoughts and later commands matter
    episodes, probabilities, contributions = [], [], []
    for trajectory in product(product((0, 1), repeat=2), repeat=2):
        backend = TinyBackend([(10 + z, 20 + a, 0) for z, a in trajectory])
        episode = run(request_state, parameters, backend=backend)
        probability = sum(a.generation_logprob for a in episode.actions)
        episodes.append(episode)
        probabilities.append(torch.tensor(probability, dtype=torch.float64).exp())
        # Direct unchanged RTD streaming kernel, including all actions and EOS.
        contributions.append(reinforce_gradient([episode.as_task_rollout()], backend, parameters,
                                                baseline="smoke_zero").gradient["theta"])
    probs = torch.stack(probabilities)
    assert probs.sum().item() == pytest.approx(1, abs=1e-14)
    exact = (probs[:, None] * torch.stack(contributions)).sum(0)
    torch.testing.assert_close(exact, expected, atol=1e-13, rtol=1e-13)
    # Every ordered pair of trajectories: E[(R1-R2)(score1-score2)/2].
    # This checks the actual K=2 LOO implementation, not a hand-coded estimator.
    exact_loo = torch.zeros_like(theta)
    backend = TinyBackend()
    for i, first in enumerate(episodes):
        for j, second in enumerate(episodes):
            result = reinforce_gradient([first.as_task_rollout(), second.as_task_rollout()], backend, parameters)
            exact_loo += probs[i] * probs[j] * result.gradient["theta"]
    torch.testing.assert_close(exact_loo, expected, atol=1e-13, rtol=1e-13)


def test_finite_sample_three_step_reinforce_estimate(request_state, parameters):
    theta = parameters["theta"]
    pz = theta[0].sigmoid()
    pa = (1 - pz) * (theta[1:] - .8).sigmoid() + pz * (theta[1:] + .8).sigmoid()
    expected = torch.autograd.grad(pa.prod(), theta)[0]
    backend = TinyBackend()
    episodes = [run(request_state, parameters, backend=backend, env=EnumerableEnv(horizon=3), index=i)
                for i in range(1400)]
    result = alfworld_reinforce_gradient(episodes, backend, parameters, journal=MemoryJournal())
    torch.testing.assert_close(result.gradient["theta"], expected, atol=.025, rtol=0)
    assert result.metadata["identifiable"] and result.metadata["action_tokens"] == 1400 * 3 * 3
    assert result.metadata["baseline"] == "leave_one_out_same_task"


def test_eos_sampled_ids_full_history_observation_masks_and_durable_journal(request_state, parameters, tmp_path):
    backend = TinyBackend([(11, 21, 0)] * 2 + [(10, 20, 0)] * 2)
    journal = ComputeJournal(tmp_path / "rollout.jsonl", cuda=False)
    first = run(request_state, parameters, backend=backend, env=EnumerableEnv(long_observation=True), journal=journal)
    second = run(request_state, parameters, backend=backend, env=EnumerableEnv(long_observation=True), journal=journal, index=1)
    result = alfworld_reinforce_gradient([first, second], backend, parameters, journal=journal)
    assert first.reward == 1 and second.reward == 0 and result.metadata["identifiable"]
    history = json.loads(first.steps[1].state.history_json)
    assert len(history[0]["content"]) > 2000 and len(history[-1]["content"]) > 300
    assert history[1]["content"] == ADVANCE and "won" not in first.steps[1].state.history_json
    records = [e for e in journal.events if e["kind"] == "score_consistency"]
    assert len(records) == 4 and backend.scored == [a.action_ids for e in (first, second) for a in e.actions]
    for d in records:
        assert d["passed"] and d["state_hash"] == digest(dict(prompt_ids=d["prompt_ids"]))
        assert len(d["full_state_hash"]) == 64
        assert d["generation_token_logprobs"] == d["teacher_forced_token_logprobs"]
        assert d["eos"]["included_in_both_scores"] and not d["eos"]["appended"]
        assert d["eos"]["action_positions"] == [2]
        mask = d["masks"]["teacher_forced_action"]
        assert not any(mask[:-3]) and mask[-3:] == [True] * 3
    assert journal.events[0]["token_counts"] == dict(action=6, prompt=sum(len(a.prompt_ids) for a in first.actions),
                                                    sampled_eos=2, observation_loss=0)
    assert len(ComputeJournal(journal.path, cuda=False).events) == 6


@pytest.mark.parametrize("ids", [(10, 21) + (30,) * 254, (0,)])
def test_no_eos_cap_and_eos_only_action(request_state, parameters, ids):
    backend = TinyBackend([ids] * 2)
    episode = run(request_state, parameters, backend=backend)
    diagnostics = []
    reinforce_gradient([episode.as_task_rollout()], backend, parameters, baseline="smoke_zero",
                       diagnostic_record=lambda r, i, d: diagnostics.append(d))
    assert episode.truncated is (ids[-1] != 0)
    assert all(a.action_ids == ids for a in episode.actions)
    for d in diagnostics:
        assert d["eos"]["appended"] is False
        assert sum(d["masks"]["teacher_forced_action"]) == len(ids)
        assert d["eos"]["action_positions"] == ([] if episode.truncated else [0])


@pytest.mark.parametrize("text,commands,expected,fallback", [
    ("THOUGHT: ???", (ADVANCE, "look"), "look", True),
    ("THOUGHT: ???", (ADVANCE,), "look", True),
    ("ACTION: look", (ADVANCE, "look"), "look", False),
    ("ACTION: LOOK", (ADVANCE, "look"), "look", False),
    (f"ACTION: look\nTHOUGHT: next\nACTION: {ADVANCE.upper()}", (ADVANCE, "look"), ADVANCE, False),
    (f"ACTION: `{ADVANCE}`", (ADVANCE, "look"), ADVANCE, False),
    ("ACTION: go to table", (ADVANCE, "look"), ADVANCE, False),
])
def test_official_parser_and_fallback_provenance(text, commands, expected, fallback):
    assert parse_action(text, commands) == (expected, fallback)
    assert adapter.ALFWorldAdapter._teacher_command(text, commands) == expected


def test_parser_fallback_continues_and_later_success_counts(request_state, parameters):
    # The first sampled thought has no command; its sampled IDs still get scored.
    backend = TinyBackend([(10, 0), (11, 21, 0), (10, 21, 0)])
    env, journal = EnumerableEnv(horizon=3, recover=True), MemoryJournal()
    episode = run(request_state, parameters, backend=backend, env=env, journal=journal)
    assert env.commands == ["look", ADVANCE, ADVANCE] and env.closed
    assert episode.reward == 1 and episode.steps[0].parser_fallback
    assert not episode.as_task_rollout().malformed and not any(a.malformed for a in episode.actions)
    result = reinforce_gradient([episode.as_task_rollout()], backend, parameters, baseline="smoke_zero")
    assert result.metadata["action_tokens"] == 8 and backend.scored[0] == (10, 0)
    assert journal.events[0]["parser_fallbacks"] == 1


def test_fallback_absent_from_admissible_still_records_next_full_state(request_state, parameters):
    class NoLookEnv(EnumerableEnv):
        def observation(self):
            return replace(super().observation(), admissible=(ADVANCE,))
    backend = TinyBackend([(10, 0), (11, 21, 0), (10, 21, 0)])
    episode = run(request_state, parameters, backend=backend, env=NoLookEnv(horizon=3, recover=True))
    assert episode.reward == 1 and episode.steps[0].parser_fallback
    history = json.loads(episode.steps[1].state.history_json)
    assert history[1]["content"] == "look" and "look" not in history[0]["admissible"]


def test_effective_context_cap_preserves_ids_without_eos(request_state, parameters):
    backend = TinyBackend([(11, 21)])
    env = EnumerableEnv(horizon=1)
    _, obs = env.reset(request_state)
    history = [dict(role="user", index=0, content=obs.observation,
                    admissible=list(obs.admissible), done=False)]
    backend.max_context_tokens = len(backend.tokenizer.encode(render(request_state, history))) + 2
    episode = run(request_state, parameters, backend=backend, env=env)
    assert episode.actions[0].action_ids == (11, 21) and episode.truncated
    assert episode.actions[0].generation_metadata["effective_action_limit"] == 2
    assert episode.actions[0].generation_metadata["max_action_tokens"] == 256


def test_backend_failure_is_not_an_environment_zero_or_silent_exclusion(request_state, parameters):
    class FailedBackend(TinyBackend):
        def sample_action(self, *args, **kwargs):
            raise RuntimeError("generation failed")
    env, journal = EnumerableEnv(), MemoryJournal()
    with pytest.raises(RuntimeError, match="generation failed"):
        run(request_state, parameters, backend=FailedBackend(), env=env, journal=journal)
    assert env.closed and journal.events[-1]["reward"] is None
    assert journal.events[-1]["failure"]["stage"] == "policy_or_state_contract"


def test_interrupted_episode_is_journaled_and_interrupt_propagates(request_state, parameters):
    class InterruptedBackend(TinyBackend):
        def sample_action(self, *args, **kwargs):
            raise KeyboardInterrupt("cancelled sampling")
    env, journal = EnumerableEnv(), MemoryJournal()
    with pytest.raises(KeyboardInterrupt):
        run(request_state, parameters, backend=InterruptedBackend(), env=env, journal=journal)
    assert env.closed and journal.events[-1]["reward"] is None
    assert journal.events[-1]["status"] == "failed_with_reason"


def test_fixed_40_step_horizon_is_not_a_shaped_reward(request_state, parameters):
    backend = TinyBackend([(10, 20, 0)] * 40)
    env = EnumerableEnv(horizon=100)
    episode = run(request_state, parameters, backend=backend, env=env)
    assert len(env.commands) == 40 and env.closed
    assert episode.reward == 0 and episode.horizon_reached and episode.truncated
    assert not any(a.truncated for a in episode.actions)  # EOS/action cap and task horizon are distinct


def test_prompt_window_retains_frozen_react_scaffold_and_goal(request_state, parameters):
    # The injected C26-B renderer remains authoritative; the episode keeps all
    # observations even when the policy's frozen history window hides them.
    backend = TinyBackend([(10, 20, 0)] * 10)
    episode = run(request_state, parameters, backend=backend, env=EnumerableEnv(horizon=10, long_observation=True))
    state = episode.steps[-1].state
    history = json.loads(state.history_json)
    messages = prompt_messages(request_state, history)
    assert len(history) == 19 and all(len(h["content"]) > 2000 for h in history[::2])
    assert messages[0]["content"] == adapter.TEACHER_REACT_INSTRUCTION
    assert request_state["goal"] in messages[1]["content"]
    actual_history = messages[1]["content"].split("History:\n", 1)[1]
    assert "Observation 1;" not in actual_history and "Observation 2;" in actual_history


@pytest.mark.parametrize("fault", ["reset", "step", "close"])
def test_worker_exceptions_are_journaled_excluded_never_zero(request_state, parameters, fault):
    backend = TinyBackend([(11, 21, 0)] * 2)
    env, journal = EnumerableEnv(fault=fault), MemoryJournal()
    episode = run(request_state, parameters, backend=backend, env=env, journal=journal)
    assert env.closed and episode.reward is None and episode.success is None
    assert episode.failure["stage"] == fault and not backend.scored
    assert journal.events[0]["status"] == "failed_with_reason" and journal.events[0]["excluded"]
    if fault == "step":
        assert len(episode.actions) == 1 and episode.steps[0].observation is None
        assert journal.events[0]["token_counts"]["action"] == 3
        assert journal.events[0]["total_steps"] == 0
    with pytest.raises(IncompleteFeedbackError, match="excluded"):
        alfworld_reinforce_gradient([episode], backend, parameters, journal=journal)
    assert not backend.scored


def test_factory_failure_is_journaled(request_state, parameters):
    def factory():
        raise OSError("cannot spawn worker")
    journal = MemoryJournal()
    episode = alfworld_task_rollout(request_state, TinyBackend(), parameters, env_factory=factory,
                                    renderer=render, journal=journal)
    assert episode.reward is None and journal.events[0]["failure"]["stage"] == "factory"


@pytest.mark.parametrize("success", [False, True])
def test_all_equal_returns_loo_zero(request_state, parameters, success):
    backend = TinyBackend([(11, 21 if success else 20, 0)] * 6)
    episodes = [run(request_state, parameters, backend=backend, index=i) for i in range(3)]
    result = alfworld_reinforce_gradient(episodes, backend, parameters, journal=MemoryJournal())
    assert result.metadata["baselines"] == [float(success)] * 3
    assert torch.count_nonzero(result.gradient["theta"]) == 0 and not result.metadata["identifiable"]
    assert len(backend.scored) == 6  # even zero-advantage actions must pass score consistency


def test_per_token_score_disagreement_recorded_before_rejection(request_state, parameters):
    backend = TinyBackend([(11, 21, 0)] * 4)
    episodes = [run(request_state, parameters, backend=backend, index=i) for i in range(2)]
    backend.score_offset = (.1, -.2, .1)  # sequence cancellation, including EOS disagreement
    journal = MemoryJournal()
    with pytest.raises(ValueError, match="likelihood differs"):
        alfworld_reinforce_gradient(episodes, backend, parameters, journal=journal)
    d = journal.events[-1]
    assert not d["passed"] and d["sequence_abs_difference"] < 1e-14
    assert d["outlier_positions"] == [0, 1, 2]


def test_seed_resume_and_order_independence(request_state, parameters):
    first = run(request_state, parameters, index=9, base_seed=81)
    run(request_state, parameters, index=100, base_seed=22)
    assert first == run(request_state, parameters, index=9, base_seed=81)
    assert first.seed != run(request_state, parameters, index=10, base_seed=81).seed
    other = request_state["task_id"].replace("trial-1", "trial-2")
    assert episode_seed(other, 9, base_seed=81) != first.seed
    with pytest.raises(ValueError, match="distinct"):
        alfworld_reinforce_gradient([first, first], TinyBackend(), parameters, journal=MemoryJournal())


@pytest.mark.parametrize("fault", ["cap", "top_k", "coverage", "prompt", "text", "early_no_eos", "policy"])
def test_generation_contract_rejected_and_worker_closed(request_state, parameters, fault):
    class BadBackend(TinyBackend):
        def sample_action(self, *args, **kwargs):
            a = super().sample_action(*args, **kwargs)
            if fault == "top_k":
                return replace(a, generation_metadata={**a.generation_metadata, "top_k": 5})
            if fault == "coverage":
                return replace(a, generation_token_logprobs=())
            if fault == "prompt":
                return replace(a, prompt_ids=(1000,))
            if fault == "text":
                return replace(a, text="ACTION: look")
            if fault == "policy":
                return replace(a, policy_id="wrong")
            if fault == "early_no_eos":
                return replace(a, action_ids=a.action_ids[:-1], truncated=True)
            return a
    backend = BadBackend([(11, 21, 0)])
    env, journal = EnumerableEnv(), MemoryJournal()
    if fault == "cap":
        backend.max_action_tokens = 512
    with pytest.raises(ValueError):
        run(request_state, parameters, backend=backend, env=env, journal=journal)
    if fault == "cap":
        assert not journal.events and not env.commands and backend.max_action_tokens == 512
    else:
        assert env.closed and journal.events[-1]["excluded"]


@pytest.fixture(scope="module")
def support(tmp_path_factory):
    from bfas.rtd.benchmarks import alfworld_bank as bank
    root = tmp_path_factory.mktemp("c26c-support")
    tids = [f"pick_and_place_simple-Apple-None-Table-{i}/trial-1" for i in range(16)]
    tids.append(tids[0].replace("trial-1", "trial-2"))
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
    write(root / bank.SUPPORT, dict(demand=tids, calibration=[], seed=50))
    write(root / bank.PROBE, [])
    # Probe/ledger readers use JSONL; an empty probe and a single ledger row suffice.
    (root / bank.PROBE).write_text("")
    (root / bank.LEDGER).parent.mkdir(parents=True, exist_ok=True)
    (root / bank.LEDGER).write_text("\n".join(json.dumps(dict(task_id=t)) for t in tids) + "\n")
    for tid in tids:
        world = root / "envs/alfworld/data/json_2.1.1/train" / tid
        write(world / "game.tw-pddl", dict(grammar={"task": [{"rhs": "Your task is to: put apple on table."}]}))
        write(world / "traj_data.json", dict(task_id=tid))
        write(world / "initial_state.pddl", dict(room=tid))
    identity = dict(fixture="C26-C CPU", max_episode_steps=40)
    return ALFWorldSupport(freeze_support(root, {**identity, "environment_hash": canonical_hash(identity)}))


def feedback_requests(support, round_number=1):
    import random
    return [support.manifest["tasks"][t]["request"] for t in support.feedback_tasks(round_number, random.Random(9))]


def test_feedback_four_parents_two_resets_each_and_same_kernel(support, parameters):
    requests, envs, journal = feedback_requests(support), [], MemoryJournal()
    def factory():
        envs.append(EnumerableEnv())
        return envs[-1]
    result = feedback(requests, TinyBackend(), parameters, env_factory=factory, renderer=render,
                      journal=journal, support=support, round_number=1, base_seed=34)
    records = [e for e in journal.events if e["kind"] == "alfworld_episode"]
    assert len(envs) == 8 and all(e.closed for e in envs)
    assert result.metadata["rollouts"] == 8 and result.metadata["tasks"] == 4
    assert len({r["seed"] for r in records}) == 8
    assert all(records[i]["start_state"] == records[i + 1]["start_state"] for i in range(0, 8, 2))


def test_incomplete_k_block_is_excluded_before_scoring(support, parameters):
    count, journal, backend = 0, MemoryJournal(), TinyBackend()
    def factory():
        nonlocal count
        count += 1
        return EnumerableEnv(fault="step" if count == 2 else None)
    with pytest.raises(IncompleteFeedbackError, match="no gradient") as exc:
        feedback(feedback_requests(support), backend, parameters, env_factory=factory, renderer=render,
                 journal=journal, support=support, round_number=1)
    assert len(exc.value.episodes) == 8 and count == 8 and not backend.scored
    assert sum(e.reward is None for e in exc.value.episodes) == 1
    assert journal.events[-1]["kind"] == "alfworld_feedback_excluded"


@pytest.mark.parametrize("fault", ["inner", "duplicate", "changed_request", "one_rollout", "alternate_trial"])
def test_feedback_guards_before_environment(support, parameters, fault):
    requests = feedback_requests(support)
    if fault == "inner":
        requests = feedback_requests(support, 2)
    elif fault == "duplicate":
        requests = [requests[0], requests[0]]
    elif fault == "changed_request":
        requests[0]["goal"] = "another goal"
    elif fault == "alternate_trial":
        alternate = next(t for t in support.manifest["training_task_ids"] if t.endswith("trial-2"))
        requests = [support.manifest["tasks"][alternate]["request"]]
    with pytest.raises(ValueError):
        collect_feedback(requests, TinyBackend(), parameters,
            env_factory=lambda: pytest.fail("guard ran after environment launch"), renderer=render,
            journal=MemoryJournal(), support=support, round_number=1,
            rollouts_per_task=1 if fault == "one_rollout" else 2)


def owned_prefix(support):
    request = feedback_requests(support, 2)[0]  # inner fold in round 1
    replay = replay_commands(request, [ADVANCE, ADVANCE], EnumerableEnv(), render)
    prefix = replay.states[1]
    package = PurchasedEvidencePackage("owned-query", (), (Behavior(prefix, ADVANCE),),
                                        12, "estimated", {}, {}, None)
    return prefix, package


def test_owned_prefix_replayed_exactly_and_not_task_start_feedback(support, parameters):
    prefix, package = owned_prefix(support)
    backend, env = TinyBackend([(11, 21, 0)]), EnumerableEnv()
    episode = run(prefix, parameters, backend=backend, env=env, prefix_package_id=package.query_id,
                  owned_packages={package.query_id: package}, support=support, round_number=1)
    assert episode.reward == 1 and episode.prefix_steps == 1 and len(episode.actions) == 1
    assert env.commands == [ADVANCE, ADVANCE] and episode.start_state == prefix
    assert episode.record()["token_counts"]["action"] == 3 and not episode.from_task_start
    assert episode.record()["prefix_package_id"] == package.query_id
    with pytest.raises(ValueError, match="never a teacher prefix"):
        episode.as_task_rollout()


def test_prefix_worker_failure_counts_only_completed_steps(support, parameters):
    prefix, package = owned_prefix(support)
    journal = MemoryJournal()
    episode = run(prefix, parameters, env=EnumerableEnv(fault="step"), journal=journal,
                  prefix_package_id=package.query_id, owned_packages={package.query_id: package},
                  support=support, round_number=1)
    assert episode.failure["stage"] == "prefix_step" and episode.reward is None
    assert episode.prefix_steps == 1 and not episode.actions
    assert journal.events[0]["total_steps"] == 0 and journal.events[0]["sampled_steps"] == 0


@pytest.mark.parametrize("fault", ["unowned", "wrong_package", "dependency", "fold", "history", "prompt"])
def test_prefix_ownership_dependencies_fold_and_full_replay(support, parameters, fault):
    prefix, package = owned_prefix(support)
    owned, round_number = {package.query_id: package}, 1
    if fault == "unowned":
        owned = {}
    elif fault == "wrong_package":
        owned = {package.query_id: replace(package, behaviors=())}
    elif fault == "dependency":
        owned = {package.query_id: replace(package, dependencies=("not-owned",))}
    elif fault == "fold":
        round_number = 2
    elif fault in {"history", "prompt"}:
        history = json.loads(prefix.history_json)
        if fault == "history":
            history[0]["content"] += " hidden changed observation"
        prefix = FullState.create(json.loads(prefix.task_json), history,
                                  prefix.prompt + ("wrong" if fault == "prompt" else ""), prefix.parent_hash)
        owned = {package.query_id: replace(package, behaviors=(Behavior(prefix, ADVANCE),))}
    with pytest.raises(ValueError):
        run(prefix, parameters, prefix_package_id=package.query_id, owned_packages=owned,
            support=support, round_number=round_number)


with warnings.catch_warnings():
    warnings.simplefilter("ignore", pytest.PytestUnknownMarkWarning)
    integration = pytest.mark.integration


@integration
def test_real_alfworld_train_task_with_sealed_expert_backend(tmp_path, monkeypatch):
    """One four-command sealed train task, CPU worker only, no model/teacher.

    A missing installation or failure to start the worker can skip. A failure
    after reset, command mismatch or unsuccessful expert replay must fail.
    """
    from tools.rtd_alfworld_verify import TOKENIZER
    root = adapter.ROOT
    q = "00400e2dfbce6c70f1ec8393c8b1d36bc58050fab0852364c026436e94762a93"
    payload_path = root / f"data/rtd/v1_alfworld_c26/sealed/{q}.json"
    manifest_path = root / "configs/rtd/v1_alfworld_support_c26.json"
    required = [adapter.ENV_PYTHON, TOKENIZER / "tokenizer.json", payload_path, manifest_path]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        pytest.skip(f"real ALFWorld prerequisites unavailable (no download): {missing}")
    payload = json.loads(payload_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    request = manifest["tasks"][payload["provenance"]["task_id"]]["request"]
    if not (adapter.DATA / "train" / request["task_id"] / "game.tw-pddl").is_file():
        pytest.skip("sealed real ALFWorld train world unavailable; no downloader")
    assert payload["status"] == "usable" and payload["request_state_hash"] == canonical_hash(request)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    renderer = FrozenRenderer(TOKENIZER)

    class ExpertBackend(TorchPolicyBackend):
        def __init__(self):
            self.tokenizer = renderer.adapter._tokenizer
            self.backend_id = "scripted-sealed-expert-cpu"
            self.max_action_tokens, self.max_context_tokens = 256, 32768
            self.score_tolerance = ScoreTolerance()
            self.commands = iter(payload["commands"])

        def identity(self, parameters):
            return "scripted-expert"

        def sample_action(self, prompt, parameters, generator, *, temperature=1., top_p=1.):
            assert temperature == top_p == 1 and generator.device.type == "cpu"
            text = "THOUGHT: Follow the route.\nACTION: " + next(self.commands)
            pids = tuple(self.tokenizer.encode(prompt, add_special_tokens=False))
            ids = tuple(self.tokenizer.encode(text, add_special_tokens=False)) + (self.tokenizer.eos_token_id,)
            return ActionTrace(pids, ids, self.tokenizer.eos_token_id, text, 0., self.backend_id,
                self.identity(parameters), (0.,) * len(ids), dict(implementation="scripted-deterministic",
                    temperature=1., top_p=1., top_k=0, max_action_tokens=256,
                    effective_action_limit=min(256, self.max_context_tokens - len(pids))))

        def score_action(self, action, parameters, *, verify_policy=True, return_details=False):
            values = parameters["theta"].sum() * torch.zeros(len(action.action_ids), dtype=torch.float64)
            return (values.sum(), values, dict(implementation="scripted-teacher-forced")) if return_details else values.sum()

    backend = ExpertBackend()
    journal = ComputeJournal(tmp_path / "real-rollout.jsonl", cuda=False)
    parameters = {"theta": torch.tensor(0., dtype=torch.float64, requires_grad=True)}
    # Public sealed reset binds full environment observations and frozen ReAct prompt.
    reset = FullState(**payload["behaviors"][0]["state"])
    episode = alfworld_task_rollout(reset, backend, parameters, renderer=renderer, journal=journal,
        env_factory=lambda: RealStepper(environment_hash=request["environment_hash"], timeout=30., episode_timeout=120.))
    if episode.failure and episode.failure["stage"] == "reset" and episode.failure["exception_type"] in {
            "EnvironmentUnavailable", "FileNotFoundError"}:
        pytest.skip(f"real ALFWorld worker unavailable: {episode.failure['message']}")
    assert episode.failure is None, episode.failure
    assert episode.success is True and episode.reward == 1 and not episode.truncated
    assert [s.command for s in episode.steps] == payload["commands"]
    assert len(episode.steps) == 4 and all(not s.parser_fallback for s in episode.steps)
    # Rescore the fake backend's sampled text, including thoughts/EOS, unchanged.
    result = reinforce_gradient([episode.as_task_rollout()], backend, parameters, baseline="smoke_zero",
        diagnostic_record=lambda r, i, d: journal.append("score_consistency", **d))
    assert result.metadata["action_tokens"] > sum(len(c.split()) for c in payload["commands"])
