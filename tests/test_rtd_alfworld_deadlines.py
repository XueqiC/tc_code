"""C26-G CPU-only deadline, accounting, retry and factory regression tests."""
from collections import deque
import io
import json
from pathlib import Path
import queue
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import pytest

from bfas.rtd.benchmarks import alfworld_config as config
from bfas.rtd.benchmarks import alfworld_evaluation as evaluation
from bfas.rtd.benchmarks import alfworld_identity as identity
from bfas.rtd.benchmarks import alfworld_support as support_module
from bfas.rtd.benchmarks.alfworld_rollout import alfworld_task_rollout, IncompleteFeedbackError
from bfas.rtd.benchmarks.registry import ALFWorldExperimentSupport
from rtd_alfworld_evaluation_fixtures import campaign, FakeBackend, FakeEnv, run as run_campaign
from test_rtd_alfworld_rollout import (
    EnumerableEnv, MemoryJournal, TinyBackend, parameters, request_state, support,
    owned_prefix, render,
)


class Clock:
    def __init__(self):
        self.now = 0.

    def advance(self, seconds):
        self.now += seconds


@pytest.fixture
def worker(monkeypatch):
    """Exercise the real bridge send/read/cleanup with a clocked worker queue."""
    clock = Clock()
    replies = deque([(1., dict(op='ready')), (1., dict(op='state'))])
    waits, processes = [], []
    class Replies:
        def get(self, *, timeout):
            waits.append(timeout)
            delay, value = replies.popleft()
            clock.advance(min(delay, timeout))
            if delay >= timeout:
                raise queue.Empty
            return value
    class Process:
        def __init__(self, *args, **kwargs):
            self.stdin, self.stdout = tempfile.TemporaryFile(mode='w+'), io.StringIO()
            self.pid = 123456789
            processes.append(self)

        def wait(self, *, timeout):
            return 0

        def poll(self):
            return 0
    monkeypatch.setattr(support_module.time, 'monotonic', lambda: clock.now)
    monkeypatch.setattr(support_module.queue, 'Queue', Replies)
    monkeypatch.setattr(support_module.subprocess, 'Popen', Process)
    monkeypatch.setattr(support_module.threading, 'Thread',
        lambda **kwargs: SimpleNamespace(start=lambda: None, join=lambda **kwargs: None))
    monkeypatch.setattr(support_module.os, 'killpg', lambda *args: None)
    return SimpleNamespace(clock=clock, replies=replies, waits=waits, processes=processes)


def bridge(worker, **kwargs):
    return support_module.BoundedEnvBridge('pick_and_place_simple-fixture/trial', **kwargs)


def test_slow_generation_does_not_spend_environment_budget(worker):
    env = bridge(worker, timeout=30, env_time_budget=10, episode_wall_guard=3600)
    try:
        env._read()
        for i in range(2):
            worker.clock.advance(200)  # Already exceeds the former 120-second deadline.
            worker.replies.append((1., dict(op='state', id=i + 1)))
            env.step('look')
        assert env.timing == dict(env_seconds=4., generation_seconds=400., wall_seconds=404., n_env_calls=4)
    finally:
        env.close()
    assert all(p.stdin.closed and p.stdout.closed for p in worker.processes)


@pytest.mark.parametrize('budget,delay,reason,elapsed', [
    (600, 31, 'command read timeout', 32.),
    (5, 4, 'environment time budget', 5.),
])
def test_slow_environment_exhausts_command_or_cumulative_budget(worker, budget, delay, reason, elapsed):
    env = bridge(worker, env_time_budget=budget)
    try:
        env._read()
        worker.replies.append((delay, dict(op='state', id=1)))
        with pytest.raises(support_module.EnvironmentUnavailable, match=reason):
            env.step('look')
        assert env.timing['env_seconds'] == elapsed
        assert env.timing['n_env_calls'] == 3
    finally:
        env.close()


def test_wall_guard_remains_independent_of_environment_budget(worker):
    env = bridge(worker, episode_wall_guard=100)
    try:
        env._read()
        worker.clock.advance(101)
        with pytest.raises(support_module.EnvironmentUnavailable, match='episode wall guard'):
            env.step('look')
        assert env.timing['env_seconds'] == 2
        assert len(worker.waits) == 2  # Expired before sending/reading another command.
    finally:
        env.close()


def test_stalled_worker_send_is_bounded_and_process_is_reaped(monkeypatch):
    original = subprocess.Popen
    code = "import json,time; print(json.dumps({'bfas_worker':True,'op':'ready'}),flush=True); time.sleep(60)"
    def spawn(*args, **kwargs):
        return original([sys.executable, '-u', '-c', code], **kwargs)
    monkeypatch.setattr(support_module.subprocess, 'Popen', spawn)
    env = support_module.BoundedEnvBridge('pick_and_place_simple-fixture/trial', timeout=2)
    process = env.process
    env.timeout = .05
    try:
        with pytest.raises(support_module.EnvironmentUnavailable, match='command read timeout'):
            env.step('x' * 1_000_000)  # Fills the pipe while the worker never reads stdin.
    finally:
        env.close()
    assert process.poll() is not None


@pytest.mark.parametrize('key', list(support_module.WORKER_DEFAULTS))
@pytest.mark.parametrize('value', [0, -1, True, False, None, '30', float('inf'), float('-inf'), float('nan')])
def test_invalid_deadline_config(key, value):
    with pytest.raises(ValueError, match=key):
        config.validate_config(dict(benchmark='alfworld', **{key: value}))
    with pytest.raises(ValueError, match=key):
        identity.checked_config(dict(identity.OFFICIAL_CONFIG, **{key: value}))


def test_configurable_limits_are_identity_bound():
    values = dict(alfworld_worker_read_timeout_s=.5, alfworld_env_time_budget_s=4,
                  alfworld_episode_wall_guard_s=20)
    actual = config.validate_config(dict(benchmark='alfworld', **values))
    assert all(actual[k] == v for k, v in values.items())
    assert support_module.worker_options(actual) == dict(timeout=.5, env_time_budget=4, episode_wall_guard=20)


def test_registry_threads_limits_to_real_bridge(worker, monkeypatch, request_state):
    holder = ALFWorldExperimentSupport.__new__(ALFWorldExperimentSupport)
    holder.config = dict(alfworld_worker_read_timeout_s=7, alfworld_env_time_budget_s=23,
                         alfworld_episode_wall_guard_s=47)
    holder.protocol = SimpleNamespace(manifest=dict(environment=dict(environment_hash=request_state['environment_hash'])))
    context = holder.feedback_context(1, TinyBackend(), MemoryJournal())
    stepper = context.env_factory()
    monkeypatch.setattr(stepper, '_world', lambda request: None)
    worker.replies[1] = (1., dict(op='state', observation='Your task is to: ' + request_state['goal'],
                                 admissible=['look'], done=False, won=False))
    # Goal extraction is checked independently by existing real replay tests.
    monkeypatch.setattr(support_module.adapter, '_goal_line', lambda observation: request_state['goal'])
    try:
        stepper.reset(request_state)
        assert stepper.bridge.timeout == 7
        assert stepper.bridge.env_time_budget == 23
        assert stepper.bridge.wall_deadline == 47
    finally:
        stepper.close()


class UnavailableEnv(EnumerableEnv):
    def step(self, cursor, command):
        raise support_module.EnvironmentUnavailable('temporary worker timeout')


@pytest.mark.parametrize('prefix', [False, True])
def test_retry_same_seed_and_prefix_once_with_journal(request_state, parameters, support, prefix):
    task, kwargs = request_state, {}
    if prefix:
        task, package = owned_prefix(support)
        kwargs = dict(prefix_package_id=package.query_id, owned_packages={package.query_id: package},
                      support=support, round_number=1)
    journal, envs = MemoryJournal(), []
    def factory():
        env = UnavailableEnv() if not envs else EnumerableEnv()
        envs.append(env)
        return env
    result = alfworld_task_rollout(task, TinyBackend(), parameters, env_factory=factory,
        renderer=render, journal=journal, rollout_index=9, base_seed=81, **kwargs)
    assert result.failure is None and result.attempt == 2
    assert len(envs) == 2 and all(e.closed for e in envs)
    first, retry, second = journal.events
    assert [r['kind'] for r in journal.events] == ['alfworld_episode', 'alfworld_episode_retry', 'alfworld_episode']
    assert first['seed'] == retry['seed'] == second['seed'] == result.seed
    assert first['prefix_package_id'] == retry['prefix_package_id'] == second['prefix_package_id']
    assert retry['retry_attempt'] == retry['max_attempts'] == 2
    assert first['reward'] is None and first['excluded']
    if prefix:
        assert retry['prefix_state_hash'] == task.state_hash and result.start_state == task
    else:
        assert first['steps'][0]['action'] == second['steps'][0]['action']
        assert first['start_state'] == second['start_state']


def test_retry_exhaustion_still_excludes_feedback(request_state, parameters):
    envs, journal = [], MemoryJournal()
    def factory():
        envs.append(UnavailableEnv())
        return envs[-1]
    result = alfworld_task_rollout(request_state, TinyBackend(), parameters, env_factory=factory,
        renderer=render, journal=journal)
    with pytest.raises(IncompleteFeedbackError):
        result.as_task_rollout()
    assert len(envs) == 2 and all(e.closed for e in envs)
    assert sum(r['kind'] == 'alfworld_episode_retry' for r in journal.events) == 1
    assert result.reward is None and result.attempt == 2


@pytest.mark.parametrize('source', ['model', 'parser', 'environment_contract'])
def test_non_infrastructure_errors_never_retry(request_state, parameters, monkeypatch, source):
    backend, journal, envs = TinyBackend(), MemoryJournal(), []
    def fail(*args, **kwargs):
        raise ValueError('invalid ' + source)
    if source == 'model':
        backend.sample_action = fail
    elif source == 'parser':
        monkeypatch.setattr('bfas.rtd.benchmarks.alfworld_rollout.parse_action', fail)
    def factory():
        env = EnumerableEnv()
        if source == 'environment_contract':
            env.step = fail
        envs.append(env)
        return env
    def run():
        return alfworld_task_rollout(request_state, backend, parameters, env_factory=factory,
                                    renderer=render, journal=journal)
    if source == 'environment_contract':
        assert run().reward is None
    else:
        with pytest.raises(ValueError, match=source):
            run()
    assert len(envs) == len(journal.events) == 1 and envs[0].closed


def test_episode_journals_generation_and_environment_time(request_state, parameters, monkeypatch):
    clock, journal = Clock(), MemoryJournal()
    monkeypatch.setattr(support_module.time, 'monotonic', lambda: clock.now)
    class TimedEnv(EnumerableEnv):
        def reset(self, request):
            clock.advance(1)
            return super().reset(request)

        def step(self, cursor, command):
            clock.advance(2)
            return super().step(cursor, command)
    class TimedBackend(TinyBackend):
        def sample_action(self, *args, **kwargs):
            clock.advance(200)
            return super().sample_action(*args, **kwargs)
    result = alfworld_task_rollout(request_state, TimedBackend(), parameters, env_factory=TimedEnv,
                                   renderer=render, journal=journal)
    assert result.env_seconds == 5 and result.generation_seconds == 400
    assert result.wall_seconds == 405 and result.n_env_calls == 3
    assert {k: journal.events[-1][k] for k in ('env_seconds', 'generation_seconds', 'wall_seconds', 'n_env_calls')} == dict(
        env_seconds=5, generation_seconds=400, wall_seconds=405, n_env_calls=3)


def test_evaluation_campaign_uses_configured_limits(campaign, monkeypatch):
    c = campaign
    c.config.update(alfworld_worker_read_timeout_s=11, alfworld_env_time_budget_s=22,
                    alfworld_episode_wall_guard_s=333)
    c.manifest = identity.make_manifest(c.root, c.config, data_root=c.data, model_path=c.model, hardware=c.hardware)
    calls = []
    def factory(tid, **kwargs):
        calls.append(kwargs)
        return FakeEnv(tid)
    monkeypatch.setattr(evaluation, 'EvaluationEnvBridge', factory)
    run_campaign(c, env_factory=None)
    assert len(calls) == 140
    assert all({k: r[k] for k in ('timeout', 'env_time_budget', 'episode_wall_guard')} == dict(
        timeout=11, env_time_budget=22, episode_wall_guard=333) for r in calls)
    harness_config = c.manifest['evaluation_harness']['config']
    assert all(harness_config[k] == c.config[k] for k in support_module.WORKER_DEFAULTS)


@pytest.mark.parametrize('persistent', [False, True])
def test_evaluation_retry_is_bounded_and_journaled(persistent):
    journal, envs = MemoryJournal(), []
    class BrokenEnv(FakeEnv):
        def step(self, command):
            raise support_module.EnvironmentUnavailable('evaluation worker timeout')
    def factory(tid):
        env = BrokenEnv(tid) if persistent or not envs else FakeEnv(tid)
        envs.append(env)
        return env
    kwargs = dict(env_factory=factory, journal=journal, identity=dict(split='valid_seen',
        evaluation_temperature=0, max_steps=40, max_action_tokens=256))
    if persistent:
        with pytest.raises(support_module.EnvironmentUnavailable):
            evaluation.official_episode('pick_and_place_simple-fixture/trial', FakeBackend(), **kwargs)
    else:
        assert evaluation.official_episode('pick_and_place_simple-fixture/trial', FakeBackend(), **kwargs)['success']
    assert len(envs) == 2 and all(e.closed for e in envs)
    assert [r['kind'] for r in journal.events] == ['alfworld_episode', 'alfworld_episode_retry', 'alfworld_episode']
    assert all(k in journal.events[0] for k in ('env_seconds', 'generation_seconds', 'wall_seconds', 'n_env_calls'))
