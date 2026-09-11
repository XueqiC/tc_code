"""CPU-only serial/lockstep trajectory, score, accounting and retry contracts."""
from collections import Counter
from contextlib import nullcontext
from dataclasses import replace
import json
from types import MethodType, SimpleNamespace

import pytest
import torch

from bfas.rtd.benchmarks.alfworld_rollout import IncompleteFeedbackError
from bfas.rtd.benchmarks.alfworld_state import _observed, parent_hash
from bfas.rtd.benchmarks.registry import (
    ALFWorldExperimentSupport, ALFWorldFeedbackContext, alfworld_action_limit,
)
from bfas.rtd.experiment import RTDExperiment
from bfas.rtd.generation_batch import GenerationBatch, HFGenerationBatchMixin
from bfas.rtd.transport import FullState
from test_rtd_alfworld_rollout import (
    EnumerableEnv, MemoryJournal, TinyBackend, parameters, request_state, render,
)


class FakePolicy(HFGenerationBatchMixin, TinyBackend):
    """Use real ticket planning with the enumerable differentiable CPU policy."""
    def __init__(self, policy):
        super().__init__()
        self.generation_batch = policy
        self.model = SimpleNamespace(training=False)
        self.action_limit = MethodType(alfworld_action_limit, self)
        self.calls = []
        self.journal = MemoryJournal()

    def sample_action(self, prompt, parameters, generator, **settings):
        return self.sample_actions(prompt, 1, parameters, generator, **settings)[0]

    def _generate_group(self, rows, parameters):
        self.calls.append((len(rows), max(len(r['ids']) for r in rows)+rows[0]['limit']))
        result = []
        for row in rows:
            generator = torch.Generator().manual_seed(row['ticket']['seed'])
            prompt = self.tokenizer.decode(row['ids'])
            action = TinyBackend.sample_action(self, prompt, parameters, generator)
            action = replace(action, generation_metadata=action.generation_metadata | dict(
                rng_ticket=row['ticket'], batch_size=len(rows)))
            result.append(action)
            self.journal.append('generated_tokens', **row['ticket'], action_ids=action.action_ids)
        return tuple(result)


def campaign(request, parameters, monkeypatch, *, lockstep, fault=None, policy=GenerationBatch(),
             bad_score=False, horizons=(1, 3, 2, 3), fail_generation=False):
    monkeypatch.setenv('BFAS_FEEDBACK_LOCKSTEP', str(int(lockstep)))
    support = object.__new__(ALFWorldExperimentSupport)
    support.config = dict(meta_tasks_per_feedback=4, rollouts_per_meta_task=2)
    requests = [request | dict(task_id=f'game-{i}/trial-1') for i in range(4)]
    support.parents = {parent_hash(r['task_id']): r['task_id'] for r in requests}
    support.categories = {r['task_id']: 'agent_action' for r in requests}
    support.states = {}
    for r in requests:
        env = EnumerableEnv()
        _, observation = env.reset(r)
        history = [_observed(observation)]
        support.states[parent_hash(r['task_id'])] = FullState.create(
            r, history, render(r, history), parent_hash(r['task_id']))
    support.protocol = SimpleNamespace(manifest=dict(
        tasks={r['task_id']: dict(request=r) for r in requests},
        parents={parent_hash(r['task_id']): dict(selected_task_id=r['task_id']) for r in requests}),
        guard_tasks=lambda *args, **kwargs: None)
    # Faults are keyed by task/first trajectory command, never worker order.
    # This gives the same RPC failure in serial and concurrent execution.
    failed = set()
    workers = []
    active = set()
    peaks = []
    class Worker(EnumerableEnv):
        def reset(self, request):
            self.horizon = horizons[int(request['task_id'][5])]
            active.add(id(self))
            peaks.append(len(active))
            result = super().reset(request)
            if fault == 'reset_once' and request['task_id'] == 'game-1/trial-1' and 'reset' not in failed:
                failed.add('reset')
                raise BrokenPipeError('injected reset RPC failure')
            return result

        def step(self, cursor, command):
            key = (self.request['task_id'], command)
            if (fault in {'once', 'always'} and self.request['task_id'] == 'game-1/trial-1' and cursor == 1 and
                    (fault == 'always' or key not in failed)):
                failed.add(key)
                raise BrokenPipeError('injected RPC failure after sampling')
            return super().step(cursor, command)

        def close(self):
            active.discard(id(self))
            super().close()

    def factory():
        worker = Worker()
        workers.append(worker)
        return worker

    backend = FakePolicy(policy)
    backend.max_action_tokens = 512  # scopes must restore the ordinary backend cap
    backend.score_offset = 1. if bad_score else None
    if fail_generation:
        generate = backend._generate_group
        def fail(rows, parameters):
            if backend.calls:
                raise RuntimeError('injected generation failure')
            return generate(rows, parameters)
        backend._generate_group = fail
    journal = MemoryJournal()
    context = ALFWorldFeedbackContext(support.protocol, 1, render, factory, journal)
    support.feedback_context = lambda *args: context
    # Exercise the production experiment -> cross-parent dispatch -> registry
    # path, including per-rollout records and unchanged checked-score/LOO calls.
    engine = SimpleNamespace(support=support, backend=backend, checker=context,
        state=dict(round=1, step=1, feedback_tasks=[(h, 2) for h in support.parents], smoke=False),
        journal=journal, sampling_rng=torch.Generator().manual_seed(19), alpha_d=True, v11=True,
        scope=lambda role: nullcontext())
    try:
        result = RTDExperiment.feedback(engine, parameters, 'acquisition_reference_feedback')
    except Exception as exc:
        result = exc
    assert not active and all(worker.closed for worker in workers)
    assert backend.max_action_tokens == 512
    return result, engine, backend, journal, max(peaks)


def normalized(record):
    # Physical batch layout and RPC wall time are expected to change. Retain
    # every trajectory, RNG ticket, reward, score and score-tolerance field.
    if isinstance(record, dict):
        return {k: normalized(v) for k, v in record.items() if k not in {'batch_size', 'elapsed_seconds'}}
    if isinstance(record, (tuple, list)):
        return [normalized(v) for v in record]
    return record


@pytest.mark.parametrize('fault', [None, 'once', 'reset_once'])
@pytest.mark.parametrize('policy', [GenerationBatch(), GenerationBatch(prompts_per_batch=4),
                                   GenerationBatch(max_batch_tokens=1000)])
def test_feedback_lockstep_trajectories_scores_rng_and_retry(request_state, parameters, monkeypatch, fault, policy):
    serial = campaign(request_state, parameters, monkeypatch, lockstep=False, fault=fault, policy=policy)
    batched = campaign(request_state, parameters, monkeypatch, lockstep=True, fault=fault, policy=policy)
    sg, se, sb, sj, sp = serial
    bg, be, bb, bj, bp = batched
    assert not isinstance(sg, Exception) and not isinstance(bg, Exception), (sg, bg)
    assert torch.equal(sg.gradient['theta'], bg.gradient['theta'])
    assert normalized(sg.metadata) == normalized(bg.metadata)
    assert torch.equal(se.sampling_rng.get_state(), be.sampling_rng.get_state())
    sr = [normalized(r) for r in sj.events if r['kind'] == 'alfworld_episode']
    br = [normalized(r) for r in bj.events if r['kind'] == 'alfworld_episode']
    key = lambda r: (r['task_id'], r['rollout_index'], r['attempt'])
    assert sorted(sr, key=key) == sorted(br, key=key)
    # Equality includes the sampled thoughts, commands, observations, seeds,
    # all per-token generation scores and both attempts on the retry path.
    assert Counter(r['sampled_steps'] for r in br)[1] >= 2
    for kind in ('feedback_rollout', 'score_consistency'):
        assert [normalized(r) for r in sj.events if r['kind'] == kind] == [
            normalized(r) for r in bj.events if r['kind'] == kind]
    assert Counter(json.dumps(r, sort_keys=True) for r in sb.journal.events) == Counter(
        json.dumps(r, sort_keys=True) for r in bb.journal.events)
    assert sp == 1 and 1 < bp <= policy.prompts_per_batch
    assert all(size <= policy.prompts_per_batch and size*width <= policy.max_batch_tokens
               for size, width in bb.calls)
    if fault is None and policy == GenerationBatch():
        assert [size for size, _ in bb.calls] == [8, 6, 4]
        assert len(sb.calls) == 14
    if fault:
        assert any(r['kind'] == 'alfworld_episode_retry' for r in bj.events)


@pytest.mark.parametrize('lockstep', [False, True])
def test_feedback_lockstep_exhausted_retry_excludes_block(request_state, parameters, monkeypatch, lockstep):
    result, _, _, journal, _ = campaign(request_state, parameters, monkeypatch,
                                       lockstep=lockstep, fault='always')
    assert isinstance(result, IncompleteFeedbackError)
    assert not any(r['kind'] in {'return_gradient', 'score_consistency'} for r in journal.events)
    assert any(r['kind'] == 'alfworld_episode' and r['attempt'] == 1 and r['excluded'] for r in journal.events)


def test_feedback_lockstep_score_guard_unchanged(request_state, parameters, monkeypatch):
    result, _, _, journal, _ = campaign(request_state, parameters, monkeypatch,
                                       lockstep=True, bad_score=True)
    assert isinstance(result, Exception)
    assert any(r['kind'] == 'score_consistency' and not r['passed'] for r in journal.events)
    assert not any(r['kind'] == 'return_gradient' for r in journal.events)


def test_feedback_lockstep_horizon_and_generation_failure_cleanup(request_state, monkeypatch):
    parameters = dict(theta=torch.zeros(42, dtype=torch.float64, requires_grad=True))
    serial = campaign(request_state, parameters, monkeypatch, lockstep=False, horizons=(1, 41, 2, 41))
    batched = campaign(request_state, parameters, monkeypatch, lockstep=True, horizons=(1, 41, 2, 41))
    assert torch.equal(serial[0].gradient['theta'], batched[0].gradient['theta'])
    assert normalized(serial[0].metadata) == normalized(batched[0].metadata)
    assert len(batched[2].calls) == 40
    assert sum(r.get('horizon_reached', False) for r in batched[3].events) == 4
    result, _, _, journal, _ = campaign(request_state, parameters, monkeypatch,
                                       lockstep=True, fail_generation=True)
    assert isinstance(result, RuntimeError) and str(result) == 'injected generation failure'
    failures = [r for r in journal.events if r['kind'] == 'alfworld_episode' and r['failure']]
    assert len(failures) == 6
    assert all(r['failure']['exception_type'] == 'RuntimeError' for r in failures)
