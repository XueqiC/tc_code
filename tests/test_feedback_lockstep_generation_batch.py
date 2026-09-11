"""CPU-only serial/lockstep trajectory, score, accounting and retry contracts."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import replace
import json
from pathlib import Path
import sys
import threading
from types import MethodType, SimpleNamespace

import pytest
import torch

from bfas.rtd.benchmarks.alfworld_rollout import IncompleteFeedbackError
from bfas.rtd.benchmarks.alfworld_state import _observed, parent_hash
from bfas.rtd.benchmarks.alfworld_support import BoundedEnvBridge, adapter
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
        self.owner = threading.get_ident()

    def identity(self, parameters):
        assert threading.get_ident() == self.owner, 'policy accessed from an RPC thread'
        return super().identity(parameters)

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
             bad_score=False, horizons=(1, 3, 2, 3), fail_generation=False,
             lockstep_episodes=None, task_count=4, rollouts=2, step_probe=None, generation_probe=None):
    monkeypatch.setenv('BFAS_FEEDBACK_LOCKSTEP', str(int(lockstep)))
    if lockstep_episodes is None:
        monkeypatch.delenv('BFAS_FEEDBACK_LOCKSTEP_EPISODES', raising=False)
    else:
        monkeypatch.setenv('BFAS_FEEDBACK_LOCKSTEP_EPISODES', str(lockstep_episodes))
    support = object.__new__(ALFWorldExperimentSupport)
    support.config = dict(meta_tasks_per_feedback=4, rollouts_per_meta_task=2)
    requests = [request | dict(task_id=f'game-{i}/trial-1') for i in range(task_count)]
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
    # Each game-1 episode fails once, with its fresh retry worker exempted.
    # Assignment happens on caller-thread close/factory, never RPC finish order.
    failed = set()
    workers = []
    active = set()
    peaks = []
    retry_next = False
    caller = threading.get_ident()
    class Worker(EnumerableEnv):
        def reset(self, request):
            assert threading.get_ident() == caller
            self.horizon = horizons[int(request['task_id'].split('/')[0][5:]) % len(horizons)]
            active.add(id(self))
            peaks.append(len(active))
            result = super().reset(request)
            if fault == 'reset_once' and request['task_id'] == 'game-1/trial-1' and 'reset' not in failed:
                failed.add('reset')
                raise BrokenPipeError('injected reset RPC failure')
            return result

        def step(self, cursor, command):
            assert (threading.get_ident() != caller) == lockstep
            self.stepping = True
            try:
                if step_probe:
                    step_probe(self, cursor, command)
                if (fault in {'once', 'always'} and self.request['task_id'] == 'game-1/trial-1' and cursor == 1 and
                        (fault == 'always' or not self.retry)):
                    self.rpc_failed = True
                    raise BrokenPipeError('injected RPC failure after sampling')
                return super().step(cursor, command)
            finally:
                self.stepping = False

        def close(self):
            nonlocal retry_next
            assert threading.get_ident() == caller and not self.stepping
            retry_next = self.rpc_failed
            active.discard(id(self))
            super().close()

    def factory():
        nonlocal retry_next
        assert threading.get_ident() == caller
        worker = Worker()
        worker.retry, retry_next = retry_next, False
        worker.stepping = worker.rpc_failed = False
        workers.append(worker)
        return worker

    backend = FakePolicy(policy)
    backend.max_action_tokens = 512  # scopes must restore the ordinary backend cap
    backend.score_offset = 1. if bad_score else None
    if fail_generation or generation_probe:
        generate = backend._generate_group
        def fail(rows, parameters):
            if generation_probe:
                generation_probe(rows)
            if fail_generation and backend.calls:
                raise RuntimeError('injected generation failure')
            return generate(rows, parameters)
        backend._generate_group = fail
    class CallerJournal(MemoryJournal):
        def append(self, kind, **values):
            assert threading.get_ident() == caller
            return super().append(kind, **values)

    def renderer(request, history):
        assert threading.get_ident() == caller
        return render(request, history)

    journal = CallerJournal()
    backend.journal = CallerJournal()
    context = ALFWorldFeedbackContext(support.protocol, 1, renderer, factory, journal)
    support.feedback_context = lambda *args: context
    # Exercise the production experiment -> cross-parent dispatch -> registry
    # path, including per-rollout records and unchanged checked-score/LOO calls.
    engine = SimpleNamespace(support=support, backend=backend, checker=context,
        state=dict(round=1, step=1, feedback_tasks=[(h, rollouts) for h in support.parents], smoke=False),
        journal=journal, sampling_rng=torch.Generator().manual_seed(19), alpha_d=True, v11=True,
        scope=lambda role: nullcontext())
    try:
        result = RTDExperiment.feedback(engine, parameters, 'acquisition_reference_feedback')
    except BaseException as exc:
        result = exc
    assert not active and all(worker.closed for worker in workers)
    assert backend.max_action_tokens == 512
    assert not any(t.name.startswith('alfworld-feedback') for t in threading.enumerate())
    return result, engine, backend, journal, max(peaks, default=0)


def normalized(record):
    # Physical batch layout and RPC wall time are expected to change. Retain
    # every trajectory, RNG ticket, reward, score and score-tolerance field.
    if isinstance(record, dict):
        return {k: normalized(v) for k, v in record.items() if k not in {'batch_size', 'elapsed_seconds'}}
    if isinstance(record, (tuple, list)):
        return [normalized(v) for v in record]
    return record


def assert_equivalent(serial, batched):
    sg, se, sb, sj, sp = serial
    bg, be, bb, bj, bp = batched
    assert not isinstance(sg, BaseException) and not isinstance(bg, BaseException), (sg, bg)
    assert torch.equal(sg.gradient['theta'], bg.gradient['theta'])
    assert normalized(sg.metadata) == normalized(bg.metadata)
    assert torch.equal(se.sampling_rng.get_state(), be.sampling_rng.get_state())
    sr = [normalized(r) for r in sj.events if r['kind'] == 'alfworld_episode']
    br = [normalized(r) for r in bj.events if r['kind'] == 'alfworld_episode']
    key = lambda r: (r['task_id'], r['rollout_index'], r['attempt'])
    assert sorted(sr, key=key) == sorted(br, key=key)
    # Equality includes the sampled thoughts, commands, observations, seeds,
    # all per-token generation scores and both attempts on the retry path.
    for kind in ('feedback_rollout', 'score_consistency'):
        assert [normalized(r) for r in sj.events if r['kind'] == kind] == [
            normalized(r) for r in bj.events if r['kind'] == kind]
    assert Counter(json.dumps(r, sort_keys=True) for r in sb.journal.events) == Counter(
        json.dumps(r, sort_keys=True) for r in bb.journal.events)
    # Journal ordering may differ, but every logical entry (including retries,
    # feedback plans, D15 diagnostics and final gradients) remains identical.
    assert Counter(json.dumps(normalized(r), sort_keys=True) for r in sj.events) == Counter(
        json.dumps(normalized(r), sort_keys=True) for r in bj.events)


@pytest.mark.parametrize('fault', [None, 'once', 'reset_once'])
@pytest.mark.parametrize('policy', [GenerationBatch(), GenerationBatch(prompts_per_batch=4),
                                   GenerationBatch(max_batch_tokens=1000)])
def test_feedback_lockstep_trajectories_scores_rng_and_retry(request_state, parameters, monkeypatch, fault, policy):
    serial = campaign(request_state, parameters, monkeypatch, lockstep=False, fault=fault, policy=policy)
    batched = campaign(request_state, parameters, monkeypatch, lockstep=True, fault=fault, policy=policy)
    assert_equivalent(serial, batched)
    _, _, sb, _, sp = serial
    _, _, bb, bj, bp = batched
    assert sp == 1 and bp == 8
    assert all(size <= 8 and size*width <= policy.max_batch_tokens for size, width in bb.calls)
    if fault is None and policy == GenerationBatch():
        assert [size for size, _ in bb.calls] == [8, 6, 4]
        assert len(sb.calls) == 14
    if fault:
        assert any(r['kind'] == 'alfworld_episode_retry' for r in bj.events)


@pytest.mark.parametrize('fault', [None, 'once', 'reset_once'])
def test_feedback_lockstep_full_block_serial_k8_k32(request_state, parameters, monkeypatch, fault):
    settings = dict(task_count=8, rollouts=4, fault=fault)
    serial = campaign(request_state, parameters, monkeypatch, lockstep=False, **settings)
    for limit in (None, 8, 32):
        batched = campaign(request_state, parameters, monkeypatch, lockstep=True,
                           lockstep_episodes=limit, **settings)
        assert_equivalent(serial, batched)
        assert batched[4] == (limit or 8)
        assert len([r for r in batched[3].events if r['kind'] == 'feedback_rollout']) == 32
        if fault is None:
            assert [n for n, _ in batched[2].calls] == ([32, 24, 16] if limit == 32 else
                                                       [8, 4, 4, 8, 8, 4] * 2)


@pytest.mark.parametrize('limit', [1, 3, 7, 64])
def test_feedback_lockstep_arbitrary_worker_bound_keeps_start_groups(request_state, parameters, monkeypatch, limit):
    settings = dict(task_count=3, rollouts=4)
    serial = campaign(request_state, parameters, monkeypatch, lockstep=False, **settings)
    batched = campaign(request_state, parameters, monkeypatch, lockstep=True,
                       lockstep_episodes=limit, **settings)
    assert_equivalent(serial, batched)
    assert batched[4] == min(limit, 12)
    # A legacy four-row task-start RNG group stays intact even at K=1/3.
    assert batched[2].calls[0][0] >= 4


def test_feedback_lockstep_token_budget_splits_live_round(request_state, parameters, monkeypatch):
    policy = GenerationBatch(max_batch_tokens=1000)
    settings = dict(task_count=8, rollouts=4, policy=policy)
    serial = campaign(request_state, parameters, monkeypatch, lockstep=False, **settings)
    for limit in (8, 32):
        batched = campaign(request_state, parameters, monkeypatch, lockstep=True,
                           lockstep_episodes=limit, **settings)
        assert_equivalent(serial, batched)
        assert batched[4] == limit  # the token budget never shrinks the live set
        assert all(size * width <= policy.max_batch_tokens for size, width in batched[2].calls)
        if limit == 32:
            # 32 starts, then 24 and 16 live continuations, two rows per call.
            assert [size for size, _ in batched[2].calls] == [2] * 36


@pytest.mark.parametrize('limit', [8, 32])
def test_feedback_lockstep_cpu_worker_directory_isolation(tmp_path, monkeypatch, limit):
    # Exercise the production subprocess/pipe/cleanup layer with a local stub;
    # no ALFWorld installation, model, GPU, network or timing benchmark.
    script = tmp_path / 'stub-worker'
    script.write_text(f'#!{sys.executable}\n' + '''import json, os, sys
print(json.dumps(dict(bfas_worker=True, op='ready')), flush=True)
for line in sys.stdin:
    request = json.loads(line)
    print(json.dumps(dict(bfas_worker=True, op='state', id=request['id'],
        cwd=os.getcwd(), tmpdir=os.environ['TMPDIR'], cuda=os.environ['CUDA_VISIBLE_DEVICES'],
        threads=[os.environ[k] for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')]
    )), flush=True)
''')
    script.chmod(0o700)
    monkeypatch.setattr(adapter, 'ENV_PYTHON', script)
    workers, processes, directories = [], [], []
    try:
        for i in range(limit):
            worker = BoundedEnvBridge(f'game-{i}/trial-1', timeout=10)
            workers.append(worker)
            processes.append(worker.process)
            directories.append(Path(worker._tmp.name))
        with ThreadPoolExecutor(max_workers=limit) as executor:
            replies = tuple(executor.map(lambda worker: worker.step('look'), workers))
        assert len({p.pid for p in processes}) == len(set(directories)) == limit
        assert all(p.poll() is None for p in processes)
        for directory, reply in zip(directories, replies):
            assert reply['cwd'] == reply['tmpdir'] == str(directory)
            assert reply['cuda'] == '' and reply['threads'] == ['1'] * 3
    finally:
        for worker in workers:
            worker.close()
    assert all(p.poll() is not None for p in processes)
    assert all(not directory.exists() for directory in directories)


def test_feedback_lockstep_steps_all_envs_in_parallel(request_state, parameters, monkeypatch):
    barrier = threading.Barrier(8, timeout=10)
    completed = Counter()
    lock = threading.Lock()

    def step(worker, cursor, command):
        barrier.wait()  # a sequential RPC driver cannot pass this barrier
        with lock:
            completed[cursor] += 1

    def generation(rows):
        index = json.loads(''.join(chr(i - 1000) for i in rows[0]['ids']))['index']
        if index:
            assert completed[index - 1] == 8

    result, _, _, _, peak = campaign(request_state, parameters, monkeypatch, lockstep=True,
        horizons=(3,), step_probe=step, generation_probe=generation)
    assert not isinstance(result, BaseException), result
    assert peak == 8 and completed == {0: 8, 1: 8, 2: 8}


@pytest.mark.parametrize('failure', [None, RuntimeError, KeyboardInterrupt])
def test_feedback_lockstep_pipelines_subbatches_and_joins_on_failure(request_state, parameters, monkeypatch, failure):
    entered, release = threading.Event(), threading.Event()
    continuation_calls = 0

    def step(worker, cursor, command):
        if cursor == 1:
            entered.set()
            assert release.wait(10), 'next physical generate did not overlap this RPC'

    def generation(rows):
        nonlocal continuation_calls
        index = json.loads(''.join(chr(i - 1000) for i in rows[0]['ids']))['index']
        if index == 1:
            continuation_calls += 1
            if continuation_calls == 2:
                try:
                    assert entered.wait(10), 'earlier sub-batch steps were not dispatched'
                finally:
                    release.set()  # also unblock cleanup if the assertion fails
                if failure:
                    raise failure('injected pipelined generation failure')

    result, _, _, journal, peak = campaign(request_state, parameters, monkeypatch, lockstep=True,
        horizons=(3,), policy=GenerationBatch(max_batch_tokens=1000),
        step_probe=step, generation_probe=generation)
    assert entered.is_set() and release.is_set() and peak == 8
    if failure:
        assert isinstance(result, failure), result
        failures = [r for r in journal.events if r['kind'] == 'alfworld_episode' and r['failure']]
        assert len(failures) == 8
        assert all(r['failure']['exception_type'] == failure.__name__ for r in failures)
        assert not any(r['kind'] in {'return_gradient', 'score_consistency'} for r in journal.events)
    else:
        assert not isinstance(result, BaseException), result


@pytest.mark.parametrize('limit', ['', '0', '-1', '1.5', 'many'])
def test_feedback_lockstep_rejects_invalid_limit_before_sampling(request_state, parameters, monkeypatch, limit):
    result, engine, backend, journal, peak = campaign(request_state, parameters, monkeypatch,
        lockstep=True, lockstep_episodes=limit)
    assert isinstance(result, ValueError) and 'BFAS_FEEDBACK_LOCKSTEP_EPISODES' in str(result)
    assert peak == 0 and not backend.calls
    assert torch.equal(engine.sampling_rng.get_state(), torch.Generator().manual_seed(19).get_state())


@pytest.mark.parametrize('lockstep,limit', [(False, None), (True, 8), (True, 32)])
def test_feedback_lockstep_exhausted_retry_excludes_block(request_state, parameters, monkeypatch, lockstep, limit):
    result, _, _, journal, _ = campaign(request_state, parameters, monkeypatch,
        lockstep=lockstep, lockstep_episodes=limit, task_count=8, rollouts=4, fault='always')
    assert isinstance(result, IncompleteFeedbackError)
    assert not any(r['kind'] in {'return_gradient', 'score_consistency'} for r in journal.events)
    assert any(r['kind'] == 'alfworld_episode' and r['attempt'] == 1 and r['excluded'] for r in journal.events)


@pytest.mark.parametrize('limit', [8, 32])
def test_feedback_lockstep_score_guard_unchanged(request_state, parameters, monkeypatch, limit):
    result, _, _, journal, _ = campaign(request_state, parameters, monkeypatch,
        lockstep=True, lockstep_episodes=limit, task_count=8, rollouts=4, bad_score=True)
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
