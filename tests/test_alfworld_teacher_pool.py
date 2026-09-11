"""CPU-only collection, hard reservation, crash recovery, and bank conversion."""
from copy import deepcopy
from dataclasses import asdict
from decimal import Decimal
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import alfworld_teacher_pool as pool
from bfas import ledger as gateway
from bfas.rtd.benchmarks.alfworld_support import freeze_support, seal_verified_bank, verify_package
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
def source(tmp_path):
    root = tmp_path / 'archive'
    tids = [f'pick_and_place_simple-Apple-None-Table-{i}/trial-1' for i in range(3)]
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


def test_teacher_pool_layout_resume_and_real_converter(source, tmp_path, monkeypatch):
    teacher, out = Teacher(), tmp_path / 'luna'
    summary = collect(source, out, teacher)
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
    built = tmp_path/'v1_1_alfworld_luna'
    converted = convert_alfworld_bank(out, built, config=config, tokenizer=StubTokenizer(), ledger=ledger)
    assert converted['available_packages'] == 3
    assert converted['recorded_bank_usage'] == 120
    assert validate_state_certificate(built)['core']['benchmark'] == 'alfworld'
    converted_support = json.loads((built/'public/support.json').read_text())
    assert converted_support == support
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
