"""Attempt accounting: failed attempts, prefix blocking, sealed isolation and resume."""
from dataclasses import asdict
import json
from types import SimpleNamespace

import pytest
from test_rtd_checks import toy_bank

from bfas.rtd.bank_build import seal_v11, state_record, validate_state_certificate
from bfas.rtd.broker import SealedReplayBroker, UnavailableError
from bfas.rtd.ledger import Ledger, BudgetError, LedgerError
from bfas.rtd.persistence import digest
from bfas.rtd.preflight import acquisition_preflight
from bfas.rtd.selector import StudentSnapshot
from bfas.rtd.task_packages import ATTEMPT_LEDGER, rebuild_task_accounting
from bfas.rtd.transport import FullState


@pytest.fixture(params=['alfworld', 'bfcl', 'hotpotqa'])
def task_bank(tmp_path, request):
    parent = digest('shared parent')
    state = FullState.create({'task': 'public'}, [{'role': 'user', 'content': 'public'}], 'public prompt', parent)
    records, payloads = [], {}
    # Seed zero shuffles IDs [0, 1, 2, 3] into [2, 0, 1, 3].
    # Buy failed a, verified c, failed c, then cheap b; never bypass a blocker.
    for tid, index, tokens, verified, confidence in [
            ('a', 0, 6, False, 'exact'), ('b', 0, 1, True, 'exact'),
            ('c', 0, 7, False, 'exact'), ('c', 2, 2, True, 'estimated')]:
        q = f"{ {('a', 0): 2, ('b', 0): 3, ('c', 0): 1, ('c', 2): 0}[tid, index]:064x}"
        row = dict(task_id=tid, attempt_index=index, tokens_spent=tokens, verified=verified,
            cost_confidence=confidence, usage={'completion_tokens': tokens, 'reasoning_tokens': tokens-1})
        records.append(state_record(q, state if verified else None, 'demo', tokens, confidence,
            parent=parent, unavailable=None if verified else 'failed attempt'))
        payloads[q] = dict(cost=tokens, cost_confidence=confidence, usage=row['usage'],
            provenance={'task_id': tid}, historical_response=row,
            behaviors=[dict(state=asdict(state), text='SECRET long teacher text')] if verified else [])
    bank = tmp_path/'bank'
    seal_v11(bank, records, payloads, benchmark=request.param, student='local',
        public={'support.json': {'parents': [{'parent_hash': parent, 'fold': 0}]}}, audit={}, inputs={})
    return bank, parent


def view(broker):
    return StudentSnapshot('cpu', broker.inner_parent_hashes)


def test_task_broker_prices_without_sealed_reads_and_charges_failed_prefix(task_bank, monkeypatch):
    bank, parent = task_bank
    with monkeypatch.context() as patch:
        patch.setattr(SealedReplayBroker, '_read_payload', lambda *a: pytest.fail('pre-purchase sealed read'))
        ledger = Ledger(8)
        broker = SealedReplayBroker(bank, ledger, inner_parent_hashes={parent})
        offered = broker.list_candidates(view(broker), (), 8)
    assert len(broker._records) == 4
    assert [broker.attempt_packages[q]['task_id'] for q in broker.purchase_order] == ['a', 'c', 'c', 'b']
    assert [c.query_id for c in offered] == list(broker.purchase_order[:2])
    failed = broker.acquire(offered[0].query_id)
    assert failed.behaviors == () and failed.cost == ledger.spent == 6
    candidates = broker.list_candidates(view(broker), ledger.owned_ids, ledger.remaining)
    package = broker.acquire(candidates[0].query_id)
    assert package.cost == 2 and ledger.spent == 8
    assert package.cost_confidence == 'estimated' and len(package.behaviors) == 1
    assert package.usage['completion_tokens'] == 2
    assert len(ledger.owned_ids) == 2  # Its failed sibling is still unpurchased.
    assert not broker.list_candidates(view(broker), ledger.owned_ids, ledger.remaining)
    ledger.authorize(15)
    specs = broker.list_candidates(view(broker), ledger.owned_ids, ledger.remaining)
    assert broker.acquire(specs[0].query_id).behaviors == ()
    assert ledger.spent == 15
    assert not ledger.reservations


def test_task_bank_preflight_exact_prefix_and_original_denominator(task_bank):
    bank, parent = task_bank
    assert validate_state_certificate(bank)['core']['budget_denominator'] == 3
    result = acquisition_preflight(dict(bank_path=str(bank), budget_ceilings=[8, 15, 16]),
        SimpleNamespace(parents={parent}))
    assert [(c['tasks_purchased'], c['usable_packages'], c['attempts_purchased'], c['spent_tokens'])
            for c in result['checkpoints']] == [(2, 1, 2, 8), (2, 1, 3, 15), (3, 2, 4, 16)]


def test_task_broker_resume_and_out_of_order_stale_offers(task_bank, tmp_path):
    bank, parent = task_bank
    path = tmp_path/'spend.jsonl'
    ledger = Ledger(16, path)
    broker = SealedReplayBroker(bank, ledger, inner_parent_hashes={parent})
    candidates = broker.list_candidates(view(broker), (), 16)
    with pytest.raises(UnavailableError, match='frozen attempt order'):
        broker.acquire(candidates[-1].query_id)
    ledger.open_window('window', 7, 3)
    failed = broker.acquire(candidates[0].query_id)
    with pytest.raises(BudgetError):
        broker.acquire(candidates[1].query_id)
    assert ledger.spent == 6 and not ledger.reservations
    ledger.close_window('window')
    resumed = Ledger.resume(16, path)
    restored = SealedReplayBroker(bank, resumed, inner_parent_hashes={parent})
    assert restored.acquire(failed.query_id) == failed
    assert resumed.spent == 6 and len(resumed.owned_ids) == 1
    offered = restored.list_candidates(view(restored), resumed.owned_ids, resumed.remaining)
    package = restored.acquire(offered[0].query_id)
    assert package.cost == 2 and resumed.spent == 8
    restored.set_inner_parents(set())
    assert len(restored.list_candidates(view(restored), resumed.owned_ids, resumed.remaining)) == 2
    assert not restored.training_eligible(package.query_id)
    assert restored.acquire(package.query_id) == package


def test_task_bank_public_summary_corruption_rejected(task_bank):
    bank, parent = task_bank
    path = bank/'public'/ATTEMPT_LEDGER
    value = json.loads(path.read_text())
    value['tasks'][0]['attempts'][0]['tokens'] = 0
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='artifact mismatch'):
        SealedReplayBroker(bank, Ledger(16), inner_parent_hashes={parent})


def test_task_broker_rejects_old_task_accounting_resume(task_bank):
    bank, parent = task_bank
    ledger = Ledger(16)
    summary = json.loads((bank/'public'/ATTEMPT_LEDGER).read_text())
    task = next(t for t in summary['tasks'] if t['task_id'] == 'c')
    q = task['query_id']
    ledger.reserve(q, 9)
    ledger.settle(q, 9, confidence='estimated', usage={'output_tokens': 9})
    with pytest.raises(LedgerError, match='different attempt purchase accounting'):
        SealedReplayBroker(bank, ledger, inner_parent_hashes={parent})


def test_task_bank_reveal_rechecks_failed_attempt_integrity(task_bank):
    bank, parent = task_bank
    broker = SealedReplayBroker(bank, Ledger(16), inner_parent_hashes={parent})
    candidates = broker.list_candidates(view(broker), (), 16)
    q = candidates[0].query_id
    path = bank/'sealed'/f'{q}.json'
    payload = json.loads(path.read_text()); payload['cost'] = 0
    path.write_text(json.dumps(payload))
    with pytest.raises(LedgerError, match='integrity'):
        broker.acquire(q)
    assert broker.ledger.spent == 0 and not broker.ledger.reservations


def test_task_bank_rebuild_preserves_support_folds_sealed_bytes(task_bank):
    bank, _ = task_bank
    before = {p.relative_to(bank): p.read_bytes() for p in bank.rglob('*.json')}
    report = rebuild_task_accounting(bank)
    assert report['support_and_folds_byte_identical']
    assert report['budget_denominator'] == 3
    assert {name: (bank/name).read_bytes() for name in before} == before


def test_task_bank_rebuild_refuses_changed_support(task_bank, monkeypatch):
    from bfas.rtd import bank_build
    bank, _ = task_bank
    before = {p.relative_to(bank): p.read_bytes() for p in bank.rglob('*.json')}
    original = bank_build.validate_state_certificate
    def changing_candidate(path, **kwargs):
        result = original(path, **kwargs)
        if path != bank:
            (path/'public/support.json').write_text('{}')
        return result
    monkeypatch.setattr(bank_build, 'validate_state_certificate', changing_candidate)
    with pytest.raises(ValueError, match='refusing rebuild'):
        rebuild_task_accounting(bank)
    assert {name: (bank/name).read_bytes() for name in before} == before


@pytest.mark.parametrize('alpha_d', [False, True])
def test_task_acquisition_executor_keeps_failed_purchases_for_replay(task_bank, alpha_d):
    from bfas.rtd.experiment_v11 import BatchExperimentMixin
    bank, parent = task_bank
    ledger = Ledger(16)
    broker = SealedReplayBroker(bank, ledger, inner_parent_hashes={parent})
    specs = broker.list_candidates(view(broker), (), 16)
    ids = [s.query_id for s in specs]
    state = dict(decision=True, window_id='w', window_budget=16, selected=ids,
        transaction_index=0, transaction_query=ids[0], hard_stops=[], pending_ids=[], round=1, step=1,
        cost_model=SimpleNamespace(observe_revealed=lambda *a, **kw: None),
        batch_specs={s.query_id: s for s in specs}, batch_rows=dict.fromkeys(ids))
    prepared = []
    executor = SimpleNamespace(state=state, fixed=False, alpha_d=alpha_d, ledger=ledger, broker=broker,
        max_new_packages=20, journal=SimpleNamespace(append=lambda *a, **kw: None),
        batch_remaining_candidates=lambda: [], save=lambda: None,
        batch_prepare_package=lambda p: prepared.append(p.query_id))
    for _ in ids:
        BatchExperimentMixin.batch_selected(executor)
    assert state['pending_ids'] == ids and ledger.spent == 16
    assert broker.acquire(ids[0]).behaviors == ()
    assert prepared == ([] if alpha_d else [q for q in ids if broker.training_eligible(q)])


@pytest.mark.parametrize('purchase_kind', ['failed', 'other_fold', 'usable'])
def test_task_acquisition_v0_d3_purchase_smoke_and_resume(toy_bank, tmp_path, purchase_kind):
    from copy import deepcopy
    from dataclasses import replace
    from test_rtd_v11_alpha_d import engine
    from test_unified_p1 import p1_engine
    _, records, payloads, states = toy_bank
    payloads = deepcopy(payloads)
    records = list(records)
    if purchase_kind == 'other_fold':
        # Give the first global query ID to parent 1, outside round 1's inner
        # fold. Keep its state/trajectory intact and all four IDs unchanged.
        q0, q1 = [r.spec.query_id for r in records[:2]]
        payloads[q0], payloads[q1] = payloads[q1], payloads[q0]
        records[0] = replace(records[0], spec=replace(records[0].spec, query_id=q1))
        records[1] = replace(records[1], spec=replace(records[1].spec, query_id=q0))
    # Purchase the global prefix even when its parent is outside round 1.
    from bfas.rtd.task_packages import frozen_attempt_order
    first_id = frozen_attempt_order(payloads)[0]
    for i, record in enumerate(records):
        q = record.spec.query_id
        p = payloads[q]
        p['provenance']['task_id'] = f'task-{i}'
        p['historical_response'] = dict(task_id=f'task-{i}', attempt_index=0,
            tokens_spent=p['cost'], verified=purchase_kind != 'failed' or q != first_id)
        if q == first_id and purchase_kind == 'failed':
            p['behaviors'] = []
            records[i] = replace(record, unavailable_reason='failed attempt')
    bank = tmp_path/'task-smoke-bank'
    seal_v11(bank, records, payloads, benchmark='bfcl', student='local',
        public={'support.json': {}}, audit={}, inputs={})
    fixture = bank, records, payloads, states
    source = engine(tmp_path/'V0', fixture, arm='V0', slots_per_step=2, max_new_packages_per_window=1)
    source.run()
    ids = [first_id]
    charges = {q: payloads[q]['cost'] for q in ids}
    assert source.ledger.charges == charges
    assert source.state['owned'] == ids
    assert bool(source.packages()) == (purchase_kind == 'usable')
    assert not source.state['posterior'].observations
    if purchase_kind == 'other_fold':
        # A usable attempt on the other fold is paid now, and can only become
        # supervision when that task's unchanged fold rotates into training.
        assert source.broker.acquire(first_id).behaviors
        assert not source.broker.training_eligible(first_id)
    schedule = source.state['steps'][0]['exposure_schedule']
    assert schedule['selected'] == ids
    assert any(row['has_teacher'] for row in schedule['records']) == (purchase_kind == 'usable')
    target = p1_engine(tmp_path/'D3', fixture, replay_schedule=source.directory)
    target.run()
    assert target.state['steps'][0]['exposure_schedule'] == schedule
    assert target.ledger.charges == source.ledger.charges
    restored = p1_engine(tmp_path/'D3', fixture, resume=True)
    restored.run()
    assert restored.ledger.charges == charges
