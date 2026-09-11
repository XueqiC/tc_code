"""CPU-only collection, hard reservation, crash recovery, and bank conversion."""
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from decimal import Decimal
import fcntl
import json
import multiprocessing
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import alfworld_teacher_pool as pool
from bfas import ledger as gateway
from bfas.rtd.benchmarks.alfworld_support import adapter as env_adapter, freeze_support, seal_verified_bank, verify_package
from test_rtd_alfworld_state import FakeStepper, make_payload, render
from test_rtd_alfworld_support import write_json
from test_webshop_adapter import StubTokenizer


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    import socket
    def forbidden(*args, **kwargs):
        pytest.fail('CPU collector test attempted network access')
    monkeypatch.setattr(socket.socket, 'connect', forbidden)
    monkeypatch.setenv('BFAS_TEACHER_MIN_INTERVAL_S', '0')
    monkeypatch.delenv('BFAS_TEACHER', raising=False)


@pytest.fixture
def source(tmp_path, request):
    root = tmp_path / 'archive'
    tids = [f'pick_and_place_simple-Apple-None-Table-{i}/trial-1' for i in range(getattr(request, 'param', 3))]
    write_json(root / pool.bank.SUPPORT, dict(demand=tids, calibration=[]))
    write_json(root / pool.bank.PROBE, [], lines=True)
    write_json(root / pool.bank.LEDGER, [dict(task_id=t) for t in tids], lines=True)
    for tid in tids:
        world = root / 'envs/alfworld/data/json_2.1.1/train' / tid
        write_json(world / 'game.tw-pddl', dict(grammar={'task': [{'rhs': 'Your task is to: put apple on table.'}]}))
        write_json(world / 'traj_data.json', dict(task_id=tid))
        write_json(world / 'initial_state.pddl', dict(room=tid))
    environment = dict(fixture='CPU only')
    environment['environment_hash'] = pool.canonical_hash(environment)
    support = freeze_support(root, environment)
    records, payloads, resets = [], {}, {}
    for tid in tids:
        request = support['tasks'][tid]['request']
        p = make_payload(request)
        p = verify_package(p, request, FakeStepper(request), render)
        q = p['query_id']
        records.append(pool.bank.public_record(q, request))
        payloads[q] = p
        resets[tid] = p['verification']['states'][0]
    archive = pool.bank.Archive(records, payloads, {}, [], dict(source_files=[]))
    result = tmp_path / 'source'
    seal_verified_bank(result, archive, support, payloads, resets)
    assert pool.audit_verified_bank(result)['passed']
    return result


class Teacher:
    def __init__(self, *, error=None, usage=True):
        self.calls, self.error, self.report_usage = [], error, usage

    def __call__(self, config, messages, **kwargs):
        self.calls.append((config, messages, kwargs))
        assert kwargs['retries'] == 0
        if self.error:
            raise self.error
        if self.report_usage:
            kwargs['usage_callback'](dict(prompt_tokens=100, completion_tokens=20,
                                         prompt_tokens_details=dict(cached_tokens=80)))
        command = messages[-1]['content'].split('Admissible commands:\n')[1].split('\n')[0]
        return 'THOUGHT: solve the task.\nACTION: ' + command


def factory(teacher):
    def adapter(support, budget, name, **kwargs):
        return pool.PoolAdapter(support, budget, name, generate_reply=teacher,
                                config_loader=lambda name: SimpleNamespace(name=name), **kwargs)
    return adapter


def collect(source, out, teacher, **kwargs):
    return pool.collect(source, out, adapter_factory=factory(teacher), stepper_factory=FakeStepper, **kwargs)


@pytest.mark.parametrize('workers', [1, 3])
def test_teacher_pool_layout_resume_and_real_converter(source, tmp_path, monkeypatch, workers):
    teacher, out = Teacher(), tmp_path / 'luna'
    summary = collect(source, out, teacher, workers=workers)
    assert summary['tasks'] == summary['tasks_attempted'] == summary['verified'] == 3
    assert summary['teacher'] == 'openai/gpt-5.6-luna'
    assert summary['service_tier'] == 'flex'
    assert len(teacher.calls) == 6
    assert summary['tokens'] == 720
    assert summary['prompt_tokens'] == 600 and summary['cached_tokens'] == 480
    assert summary['completion_tokens'] == 120
    assert summary['estimated_usd'] == pytest.approx((120*.1 + 480*.01 + 120*.6)/1e6)
    assert {p.name for p in out.iterdir()} == {'public', 'sealed'}
    assert {p.name for p in (out/'public').iterdir()} == {'requests.json', 'support.json', 'reset_states.json', 'reset_requests.json'}
    assert pool.audit_verified_bank(out)['usable_packages'] == 3
    support = pool.load_source(out)
    assert support['historical_task_ids'] == pool.load_source(source)['historical_task_ids']
    ledger = Path(summary['ledger'])
    rows = gateway.read_records(ledger)
    assert len(rows) == 3 and {r['attempt_index'] for r in rows} == {0}
    assert {r['teacher'] for r in rows} == {'openai/gpt-5.6-luna'}
    assert {r['tokens_spent'] for r in rows} == {40}
    before = ledger.read_bytes()
    hashes = {str(p.relative_to(out)): pool.bank.file_hash(p) for p in out.rglob('*.json')}
    resumed = collect(source, out, teacher)
    assert resumed == summary and ledger.read_bytes() == before and len(teacher.calls) == 6
    assert hashes == {str(p.relative_to(out)): pool.bank.file_hash(p) for p in out.rglob('*.json')}
    from bfas.rtd.benchmarks.alfworld_bank_v11 import convert_alfworld_bank
    from bfas.rtd.bank_build import validate_state_certificate
    from bfas.rtd.cli import load_config
    config = load_config(pool.ROOT / 'configs/rtd/v1_1_alfworld.yaml')
    config['support_manifest'] = str(source/'public/support.json')
    built = tmp_path/'v1_1_alfworld_luna'
    converted = convert_alfworld_bank(out, built, config=config, tokenizer=StubTokenizer(), ledger=ledger)
    assert converted['available_packages'] == 3
    assert converted['recorded_bank_usage'] == 120
    assert validate_state_certificate(built)['core']['benchmark'] == 'alfworld'
    converted_support = json.loads((built/'public/support.json').read_text())
    assert converted_support == pool.load_source(source)
    assert converted_support['parents'] == pool.load_source(source)['parents']
    assert all(p['fold'] == int(h, 16) % 2 for h, p in converted_support['parents'].items())
    monkeypatch.setenv('BFAS_TEACHER', 'openai/different')
    with pytest.raises(ValueError, match='resume'):
        collect(source, out, teacher)


@pytest.mark.parametrize('limits', [pool.Limits(max_tokens=0), pool.Limits(max_usd=0), pool.Limits(max_tokens=10)])
def test_teacher_pool_stops_before_purchase_without_consuming_attempt(source, tmp_path, limits):
    teacher, out = Teacher(), tmp_path/'luna'
    summary = collect(source, out, teacher, limits=limits)
    assert summary['tokens'] == summary['verified'] == summary['tasks_attempted'] == 0
    assert teacher.calls == []
    assert gateway.read_records(Path(summary['ledger'])) == []
    assert pool.audit_verified_bank(out)['attempts'] == 0
    # A raised cap resumes the same attempt 0 for every task.
    resumed = collect(source, out, teacher)
    assert resumed['verified'] == 3
    assert {r['attempt_index'] for r in gateway.read_records(Path(resumed['ledger']))} == {0}


def test_teacher_pool_partial_episode_cap_is_charged_and_deduped(source, tmp_path):
    request = pool.load_source(source)['tasks'][pool.load_source(source)['historical_task_ids'][0]]['request']
    stepper = FakeStepper(request)
    _, obs = stepper.reset(request)
    bound = pool.prompt_bound(pool.prompt_messages(request, [pool._observed(obs)]))
    teacher = Teacher(usage=False)  # all tokens remain conservatively reserved
    summary = collect(source, tmp_path/'luna', teacher, limits=pool.Limits(max_tokens=bound+7))
    assert len(teacher.calls) == 1
    assert teacher.calls[0][2]['max_completion_tokens'] == 7
    assert summary['tokens'] == bound + 7 and summary['verified'] == 0
    rows = gateway.read_records(Path(summary['ledger']))
    assert len(rows) == 1 and rows[0]['tokens_spent'] == 7 and rows[0]['attempt_index'] == 0
    collect(source, tmp_path/'luna', teacher, limits=pool.Limits(max_tokens=bound+7))
    assert len(teacher.calls) == 1
    resumed = collect(source, tmp_path/'luna', Teacher())
    rows = gateway.read_records(Path(resumed['ledger']))
    assert rows[1]['task_id'] == rows[0]['task_id'] and rows[1]['attempt_index'] == 1
    assert resumed['verified'] == 3


def test_teacher_pool_usd_reserves_uncached_prompt_and_limits_output(tmp_path):
    messages = [dict(role='user', content='hi')]
    prompt = pool.prompt_bound(messages)
    limits = pool.Limits(max_usd=Decimal(prompt)*Decimal('.1')/1000000 + Decimal(3)*Decimal('.6')/1000000)
    budget = pool.Budget(tmp_path/'usage.jsonl', limits)
    call = budget.reserve('task/trial', 0, messages)
    assert call['usage']['completion_tokens'] == 3
    assert limits.cost(budget.usage()) == limits.max_usd
    with pytest.raises(gateway.AcquisitionStopped):
        budget.reserve('task/trial', 0, messages)
    # Cached input is charged once and hidden completion is not added twice.
    budget.settle(call, dict(prompt_tokens=100, completion_tokens=2,
                           prompt_tokens_details=dict(cached_tokens=80),
                           completion_tokens_details=dict(reasoning_tokens=1)))
    assert budget.usage() == dict(prompt_tokens=100, cached_tokens=80, completion_tokens=2)
    assert limits.cost(budget.usage()) == Decimal('0.000004')


def test_teacher_pool_crash_reservations_recover_without_teacher(tmp_path):
    path = tmp_path/'usage.jsonl'
    budget = pool.Budget(path, pool.Limits())
    call = budget.reserve('task/trial', 0, [dict(role='user', content='hi')])
    restarted = pool.Budget(path, pool.Limits())
    assert restarted.usage() == budget.usage()
    teacher = Teacher()
    adapter = pool.PoolAdapter({}, restarted, 'openai/stub', generate_reply=teacher)
    ledger = tmp_path/'ledger.jsonl'
    result = gateway.acquire_demos(ledger, adapter, ['task/trial'], attempts=1)
    assert not result and not teacher.calls
    assert gateway.read_records(ledger)[0]['tokens_spent'] == call['usage']['completion_tokens']
    gateway.acquire_demos(ledger, adapter, ['task/trial'], attempts=1)
    assert len(gateway.read_records(ledger)) == 1


@pytest.mark.parametrize('error', [TimeoutError('uncertain bill'), KeyboardInterrupt()])
def test_teacher_pool_api_failure_and_interrupt_keep_paid_reservations(source, tmp_path, error):
    teacher = Teacher(error=error)
    summary = collect(source, tmp_path/'luna', teacher)
    rows = gateway.read_records(Path(summary['ledger']))
    expected = 1 if isinstance(error, KeyboardInterrupt) else 9
    assert len(rows) == len(teacher.calls) == expected
    assert all(not r['verified'] and r['tokens_spent'] == 2048 for r in rows)
    assert summary['completion_tokens'] == 2048 * expected
    assert summary['uncertain_calls'] == expected
    assert len({(r['task_id'], r['attempt_index']) for r in rows}) == len(rows)


@pytest.mark.parametrize('kwargs', [dict(max_tokens=-1), dict(max_usd='NaN'), dict(usd_out=-1), dict(usd_in='Infinity')])
def test_teacher_pool_invalid_caps(kwargs):
    with pytest.raises(ValueError):
        pool.Limits(**kwargs)


def test_teacher_pool_source_tampering_rejected_before_call(source, tmp_path):
    path = source/'public/support.json'
    value = json.loads(path.read_text())
    value['historical_task_ids'].append('outside/trial')
    path.write_text(json.dumps(value))
    teacher = Teacher()
    with pytest.raises(ValueError, match='integrity'):
        collect(source, tmp_path/'luna', teacher)
    assert teacher.calls == []


def test_teacher_pool_missing_usage_journal_fails_closed(source, tmp_path):
    teacher, out = Teacher(), tmp_path/'luna'
    collect(source, out, teacher)
    (tmp_path/'luna.collection/usage.jsonl').unlink()
    with pytest.raises(ValueError, match='accounting disagree'):
        collect(source, out, teacher)
    assert len(teacher.calls) == 6


def test_teacher_pool_missing_configuration_does_not_consume_attempt(tmp_path):
    def missing(_):
        raise RuntimeError('OPENAI_API_KEY is not set')
    adapter = pool.PoolAdapter({}, pool.Budget(tmp_path/'usage.jsonl', pool.Limits()),
                               'openai/stub', config_loader=missing)
    ledger = tmp_path/'ledger.jsonl'
    with pytest.raises(gateway.AcquisitionStopped, match='configuration'):
        gateway.acquire_demos(ledger, adapter, ['task/trial'], attempts=3)
    assert gateway.read_records(ledger) == []


def assert_accounting(summary, limits=pool.Limits()):
    ledger = Path(summary['ledger'])
    rows = gateway.read_records(ledger)
    journal = [json.loads(line) for line in (ledger.parent/'usage.jsonl').read_text().splitlines()]
    calls, reservations = {}, []
    # Audit every durable prefix, including simultaneously outstanding envelopes.
    for record in journal:
        if record['status'] == 'reserved':
            reservations.append(record['id'])
            assert record['id'] not in calls
        else:
            assert record['id'] in calls
            assert calls[record['id']]['status'] == 'reserved'
        calls[record['id']] = record
        usage = {key: sum(c['usage'][key] for c in calls.values())
                 for key in ('prompt_tokens', 'cached_tokens', 'completion_tokens')}
        assert usage['prompt_tokens'] + usage['completion_tokens'] <= limits.max_tokens
        assert limits.cost(usage) <= limits.max_usd
    assert len(reservations) == len(set(reservations))
    assert len(rows) == len({(r['task_id'], r['attempt_index']) for r in rows})
    assert {(c['task_id'], c['attempt_index']) for c in calls.values()} == {
        (r['task_id'], r['attempt_index']) for r in rows}
    for row in rows:
        purchases = [c for c in calls.values()
                     if (c['task_id'], c['attempt_index']) == (row['task_id'], row['attempt_index'])]
        assert row['usage'] == {key: sum(c['usage'][key] for c in purchases) for key in row['usage']}
        assert row['tokens_spent'] == row['usage']['completion_tokens']
    for key in ('prompt_tokens', 'cached_tokens', 'completion_tokens'):
        assert summary[key] == sum(r['usage'][key] for r in rows)
    assert summary['estimated_usd'] == pytest.approx(float(limits.cost(summary)))
    return rows


class ConcurrentTeacher(Teacher):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.barrier = threading.Barrier(3, timeout=10)
        self.threads = set()
        self.lock = threading.Lock()

    def __call__(self, config, messages, **kwargs):
        with self.lock:
            first = threading.get_ident() not in self.threads
            self.threads.add(threading.get_ident())
        if first:
            self.barrier.wait()  # Fails if episodes or API calls are serialized.
        return super().__call__(config, messages, **kwargs)


def test_alfworld_teacher_pool_workers_three_cli_and_locks(source, tmp_path, monkeypatch):
    teacher, adapters, steppers = ConcurrentTeacher(), [], []

    def adapter_factory(*args, **kwargs):
        adapter = factory(teacher)(*args, **kwargs)
        adapters.append(adapter)
        return adapter

    def stepper_factory(request):
        stepper = FakeStepper(request)
        steppers.append(stepper)
        return stepper

    def assert_locked(path):
        with Path(path).open('a+') as handle:
            with pytest.raises(BlockingIOError):
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)

    append = gateway.append_record
    def locked_append(path, record):
        assert_locked(path.with_suffix(path.suffix + '.lock'))
        append(path, record)

    seal = pool.seal_verified_bank
    def locked_seal(*args, **kwargs):
        directory = tmp_path/'parallel.collection'
        assert_locked(directory/'collection.lock')
        assert_locked(directory/'teacher_ledger.jsonl.lock')
        return seal(*args, **kwargs)

    monkeypatch.setattr(pool, 'append_record', locked_append)
    monkeypatch.setattr(gateway, 'append_record', locked_append)
    monkeypatch.setattr(pool, 'seal_verified_bank', locked_seal)
    real_collect = pool.collect
    summaries = []
    def stub_collect(*args, **kwargs):
        summaries.append(real_collect(*args, **kwargs, adapter_factory=adapter_factory,
                                      stepper_factory=stepper_factory))
    monkeypatch.setattr(pool, 'collect', stub_collect)
    out = tmp_path/'parallel'
    pool.main(['--source', str(source), '--out', str(out), '--workers', '3'])
    summary = summaries[0]
    assert summary['verified'] == 3 and summary['tokens'] == 720
    assert len(teacher.calls) == 6 and len(teacher.threads) == len(adapters) == 3
    assert len({id(a) for a in adapters}) == 3
    assert len(steppers) == 6 and all(s.closed for s in steppers)
    assert len(assert_accounting(summary)) == 3
    assert {p.name for p in out.iterdir()} == {'public', 'sealed'}


def initial_prompt_bound(source):
    support = pool.load_source(source)
    request = support['tasks'][support['historical_task_ids'][0]]['request']
    _, observation = FakeStepper(request).reset(request)
    return pool.prompt_bound(pool.prompt_messages(request, [pool._observed(observation)]))


@pytest.mark.parametrize('cap', ['tokens', 'usd'])
def test_alfworld_teacher_pool_workers_three_global_caps(source, tmp_path, cap):
    prompt = initial_prompt_bound(source)
    usage = dict(prompt_tokens=3*prompt, cached_tokens=0,
                 completion_tokens=3*pool.appworld_teacher.MAX_COMPLETION_TOKENS)
    limits = (pool.Limits(max_tokens=usage['prompt_tokens'] + usage['completion_tokens']) if cap == 'tokens'
              else pool.Limits(max_usd=pool.Limits().cost(usage)))
    teacher, out = ConcurrentTeacher(usage=False), tmp_path/'capped'
    summary = collect(source, out, teacher, workers=3, limits=limits)
    assert len(teacher.calls) == summary['uncertain_calls'] == summary['tasks_attempted'] == 3
    assert summary['verified'] == 0
    assert {r['attempt_index'] for r in assert_accounting(summary, limits)} == {0}
    if cap == 'tokens':
        assert summary['tokens'] == limits.max_tokens
    else:
        assert pool.Limits().cost(summary) == limits.max_usd
    before = Path(summary['ledger']).read_bytes()
    collect(source, out, teacher, workers=3, limits=limits)
    assert len(teacher.calls) == 3 and Path(summary['ledger']).read_bytes() == before
    resumed = collect(source, out, Teacher(), workers=3)
    assert resumed['verified'] == 3
    rows = assert_accounting(resumed)
    assert len(rows) == 6
    assert all([r['attempt_index'] for r in rows if r['task_id'] == tid] == [0, 1]
               for tid in {r['task_id'] for r in rows})


def test_alfworld_teacher_pool_workers_wait_for_reported_headroom(source, tmp_path, monkeypatch):
    waiting, lock, all_waiting = set(), threading.Lock(), threading.Event()
    envelope = pool.Budget._envelope
    def observed_envelope(self, messages):
        result = envelope(self, messages)
        if result[1]:
            with lock:
                waiting.add(threading.get_ident())
                if len(waiting) == 2:
                    all_waiting.set()
        return result
    monkeypatch.setattr(pool.Budget, '_envelope', observed_envelope)
    teacher = Teacher()
    def reply(*args, **kwargs):
        assert all_waiting.wait(10), 'other workers never waited for the active reservation'
        return teacher(*args, **kwargs)
    limits = pool.Limits(max_tokens=initial_prompt_bound(source) + pool.appworld_teacher.MAX_COMPLETION_TOKENS)
    summary = collect(source, tmp_path/'refunds', reply, workers=3, limits=limits)
    assert summary['verified'] == 3 and summary['stop_reason'] == 'complete'
    assert len(teacher.calls) == 6
    assert_accounting(summary, limits)


def test_alfworld_teacher_pool_workers_interrupt_and_resume(source, tmp_path, monkeypatch):
    barrier, stopped = threading.Barrier(3, timeout=10), threading.Event()
    cancel = pool.Budget.cancel
    def observed_cancel(self, reason):
        cancel(self, reason)
        stopped.set()
    monkeypatch.setattr(pool.Budget, 'cancel', observed_cancel)
    teacher = Teacher()
    def interrupt(config, messages, **kwargs):
        index = barrier.wait()
        if index == 0:
            raise KeyboardInterrupt()
        assert stopped.wait(10)
        return teacher(config, messages, **kwargs)
    out = tmp_path/'interrupted'
    summary = collect(source, out, interrupt, workers=3)
    assert summary['stop_reason'].startswith('interrupted')
    assert summary['uncertain_calls'] == 1 and summary['verified'] == 0
    rows = assert_accounting(summary)
    assert len(rows) == 3 and {r['attempt_index'] for r in rows} == {0}
    resumed = collect(source, out, Teacher(), workers=3)
    assert resumed['verified'] == 3 and len(assert_accounting(resumed)) == 6
    before = Path(resumed['ledger']).read_bytes()
    no_calls = Teacher()
    assert collect(source, out, no_calls, workers=1) == resumed
    assert not no_calls.calls and Path(resumed['ledger']).read_bytes() == before


def test_alfworld_teacher_pool_budget_instances_share_file_lock(tmp_path):
    messages = [dict(role='user', content='hi')]
    tokens = pool.prompt_bound(messages) + pool.appworld_teacher.MAX_COMPLETION_TOKENS
    limits = pool.Limits(max_tokens=3*tokens)
    path = tmp_path/'usage.jsonl'
    budgets = [pool.Budget(path, limits) for _ in range(3)]
    barrier = threading.Barrier(3, timeout=10)
    def reserve(index):
        barrier.wait()
        return budgets[index].reserve(f'task-{index}/trial', 0, messages)
    with ThreadPoolExecutor(max_workers=3) as executor:
        calls = list(executor.map(reserve, range(3)))
    assert {c['id'] for c in calls} == {0, 1, 2}
    resumed = pool.Budget(path, limits)
    with pytest.raises(gateway.AcquisitionStopped):
        resumed.reserve('task-4/trial', 0, messages)
    assert resumed.usage()['prompt_tokens'] + resumed.usage()['completion_tokens'] == limits.max_tokens


@pytest.mark.parametrize('workers', [0, -1])
def test_alfworld_teacher_pool_invalid_workers(workers, monkeypatch):
    monkeypatch.setattr(pool, 'collect', lambda *a, **k: pytest.fail('invalid workers started collection'))
    with pytest.raises(SystemExit) as exc:
        pool.main(['--workers', str(workers)])
    assert exc.value.code == 2


def interrupted_fixture_run(source, out, limits, ready, release, report_usage):
    # Spawn does not run pytest's autouse fixture, so deny network explicitly.
    import socket
    def forbidden(*args, **kwargs):
        raise AssertionError('CPU fixture attempted network access')
    socket.socket.connect = forbidden
    barrier = threading.Barrier(3, timeout=10)
    def reply(config, messages, **kwargs):
        if report_usage:
            kwargs['usage_callback'](dict(prompt_tokens=100, completion_tokens=20))
        if barrier.wait() == 0:
            ready.set()
        release.wait(30)
        raise AssertionError('fixture should have been killed before returning')
    collect(source, out, reply, workers=3, limits=limits)


@pytest.mark.parametrize('report_usage', [False, True])
def test_alfworld_teacher_pool_workers_killed_before_ledger_resume(source, tmp_path, report_usage):
    # Kill only this fixture process, with stubbed teacher/environment and tmp
    # output. All three paid calls precede any episode append or export.
    context = multiprocessing.get_context('spawn')
    ready, release = context.Event(), context.Event()
    out = tmp_path/'killed'
    bound = initial_prompt_bound(source)
    limits = pool.Limits(max_tokens=3*(bound + pool.appworld_teacher.MAX_COMPLETION_TOKENS))

    process = context.Process(target=interrupted_fixture_run,
                              args=(source, out, limits, ready, release, report_usage))
    process.start()
    try:
        assert ready.wait(15), 'fixture did not reserve three concurrent calls'
        process.kill()
        process.join(10)
        assert not process.is_alive() and process.exitcode < 0
    finally:
        if process.is_alive():
            process.kill()
            process.join(10)
    directory = tmp_path/'killed.collection'
    assert gateway.read_records(directory/'teacher_ledger.jsonl') == []
    journal = (directory/'usage.jsonl').read_bytes()
    charged = pool.Budget(directory/'usage.jsonl', limits).usage()
    # Even a fully exhausted/lowered cap must reconcile *all* orphan attempts.
    no_calls = Teacher()
    capped = pool.Limits(max_tokens=charged['prompt_tokens'] + charged['completion_tokens'])
    recovered = collect(source, out, no_calls, workers=3, limits=capped)
    assert no_calls.calls == [] and recovered['verified'] == 0
    assert len(assert_accounting(recovered, limits)) == 3
    assert (directory/'usage.jsonl').read_bytes() == journal
    teacher = Teacher()
    resumed = collect(source, out, teacher, workers=3)
    assert resumed['verified'] == 3 and len(teacher.calls) == 6
    assert len(assert_accounting(resumed)) == 6
    for row in gateway.read_records(directory/'teacher_ledger.jsonl'):
        assert row['attempt_index'] == int(row['verified'])


def test_alfworld_teacher_pool_workers_own_environment_processes(source, tmp_path, monkeypatch):
    # Exercise the real bridge's subprocess/session/tempdir lifecycle using a
    # tiny protocol server. No ALFWorld installation or teacher API is involved.
    script = tmp_path/'stub-environment'
    script.write_text(f'''#!{sys.executable}
import json, os, sys
from pathlib import Path
Path('environment.json').write_text(json.dumps(dict(pid=os.getpid(), cwd=os.getcwd(), tmp=os.environ['TMPDIR'])))
def emit(value):
    print(json.dumps(dict(bfas_worker=True, **value)), flush=True)
def state(index, request_id=None):
    text = ['Your task is to: put apple on table.\\n' + 'x'*2100, 'Full observation ' + 'y'*500, 'Task complete.'][index]
    commands = [['go to table 1'], ['put apple 1 on table 1'], []][index]
    emit(dict(op='state', id=request_id, observation=text, admissible=commands, done=index==2, won=index==2))
emit(dict(op='ready'))
state(0)
for index, line in enumerate(sys.stdin, 1):
    state(index, json.loads(line)['id'])
''')
    script.chmod(0o700)
    monkeypatch.setattr(env_adapter, 'ENV_PYTHON', script)
    monkeypatch.setattr(env_adapter, 'DATA', tmp_path/'archive/envs/alfworld/data/json_2.1.1')
    sessions = []
    class Stepper(pool.RealStepper):
        def reset(self, request):
            result = super().reset(request)
            directory = Path(self.bridge._tmp.name)
            metadata = json.loads((directory/'environment.json').read_text())
            assert self.bridge.process.poll() is None
            sessions.append((self.bridge.process, directory, metadata))
            return result
    def stepper_factory(request):
        return Stepper(environment_hash=request['environment_hash'])
    teacher = ConcurrentTeacher()
    summary = pool.collect(source, tmp_path/'processes', workers=3,
                           adapter_factory=factory(teacher), stepper_factory=stepper_factory)
    assert summary['verified'] == 3 and len(teacher.threads) == 3
    assert len(sessions) == len({p.pid for p, _, _ in sessions}) == len({d for _, d, _ in sessions}) == 6
    for process, directory, metadata in sessions:
        assert metadata == dict(pid=process.pid, cwd=str(directory), tmp=str(directory))
        assert process.poll() is not None and not directory.exists()
    assert_accounting(summary)


def test_alfworld_teacher_pool_worker_dies_after_paid_episode(source, tmp_path):
    barrier, out = threading.Barrier(3, timeout=10), tmp_path/'worker-crash'
    teacher = Teacher()
    class CrashingAdapter(pool.PoolAdapter):
        def teacher_episode(self, *args):
            episode = super().teacher_episode(*args)
            if barrier.wait() == 0:
                raise SystemExit('stub worker died before ledger append')
            return episode
    def adapter_factory(*args, **kwargs):
        return CrashingAdapter(*args, **kwargs, generate_reply=teacher,
                               config_loader=lambda name: SimpleNamespace(name=name))
    with pytest.raises(SystemExit, match='stub worker died'):
        pool.collect(source, out, workers=3, adapter_factory=adapter_factory,
                     stepper_factory=FakeStepper)
    assert len(teacher.calls) == 6
    assert pool.audit_verified_bank(out)['usable_packages'] == 2
    no_calls = Teacher()
    summary = collect(source, out, no_calls, workers=3, limits=pool.Limits(max_tokens=720))
    assert summary['verified'] == 2 and not no_calls.calls
    rows = assert_accounting(summary)
    assert len(rows) == 3 and {r['tokens_spent'] for r in rows} == {40}
    retry = Teacher()
    resumed = collect(source, out, retry, workers=3)
    assert resumed['verified'] == 3 and len(retry.calls) == 2
    assert len(assert_accounting(resumed)) == 4


@pytest.mark.parametrize('source', [7], indirect=True)
def test_alfworld_teacher_pool_workers_drain_queue_and_exhaust_attempts(source, tmp_path):
    teacher = ConcurrentTeacher(error=TimeoutError('uncertain bill'))
    summary = collect(source, tmp_path/'queue', teacher, workers=3)
    assert summary['tasks_attempted'] == 7 and summary['verified'] == 0
    assert len(teacher.calls) == summary['uncertain_calls'] == 21
    assert len(teacher.threads) == 3
    rows = assert_accounting(summary)
    assert len(rows) == 21
    for task_id in {r['task_id'] for r in rows}:
        attempts = [r for r in rows if r['task_id'] == task_id]
        assert [r['attempt_index'] for r in attempts] == [0, 1, 2]
        assert [r['temperature'] for r in attempts] == [0.0, pool.SAMPLING_TEMPERATURE, pool.SAMPLING_TEMPERATURE]
    no_calls = Teacher()
    assert collect(source, tmp_path/'queue', no_calls, workers=3) == summary
    assert not no_calls.calls
