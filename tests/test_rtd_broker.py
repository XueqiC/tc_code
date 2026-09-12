"""Sealing, request dependencies, provenance, and hard-cap CPU checks."""
from dataclasses import asdict
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    sys.path.insert(0, str(path))

from bfas.cc_pairs import digest
from bfas.rtd.broker import RequestRecord, SealedReplayBroker, UnavailableError, seal_bank
from bfas.rtd.ledger import BudgetError, Ledger, LedgerError
from bfas.rtd.selector import PublicFeatures, PublicQuerySpec, StudentSnapshot, public_only, select_public
from bfas.rtd.transport import FullState


@pytest.fixture
def bank(tmp_path):
    parent = digest("parent")
    q1, q2, q3 = [digest(f"request {i}") for i in range(3)]
    state = FullState.create({"question": "public"}, [{"role": "user", "content": "public"}], "prompt", parent)
    records, payloads = [], {}
    for i, (q, deps) in enumerate([(q1, ()), (q2, (q1,)), (q3, ())]):
        records.append(RequestRecord(PublicQuerySpec(q, state.state_hash, PublicFeatures(), 1, 20,
                        "exact" if i != 2 else "estimated", "public uniform test cap"), parent, deps))
        payloads[q] = dict(cost=7, cost_confidence=records[-1].spec.cost_confidence,
            usage={"input_tokens": 11, "output_tokens": 7, "reasoning_tokens": None, "monetary_cost": None},
            provenance={"request_id": q}, historical_response="SEALED RESPONSE " + q,
            behaviors=[{"state": asdict(state), "text": "SECRET TEACHER"}] * 3)
    path = seal_bank(tmp_path / "bank", records, payloads)
    ledger = Ledger(60, tmp_path / "ledger.jsonl")
    broker = SealedReplayBroker(path, ledger, inner_parent_hashes={parent})
    snapshot = StudentSnapshot("frozen", frozenset({parent}))
    return broker, ledger, snapshot, (q1, q2, q3), path


def candidates(broker, ledger, snapshot):
    return broker.list_candidates(snapshot, ledger.owned_ids, ledger.remaining)


def test_dependencies_locked_until_bought_and_package_single_charge(bank):
    broker, ledger, snapshot, (first, dependent, _), path = bank
    ids = {q.query_id for q in candidates(broker, ledger, snapshot)}
    assert first in ids and dependent not in ids
    with pytest.raises(UnavailableError, match="dependencies"):
        broker.acquire(dependent)
    package = broker.acquire(first)
    assert len(package.behaviors) == 3 and ledger.spent == 7
    assert broker.acquire(first) is package and ledger.spent == 7
    assert dependent in {q.query_id for q in candidates(broker, ledger, snapshot)}
    acquired = broker.acquire(dependent)
    assert acquired.dependencies == (first,) and ledger.spent == 14
    events = [json.loads(l) for l in ledger.path.read_text().splitlines()]
    assert [e["query_id"] for e in events if e["kind"] == "reveal"] == [first, dependent]
    assert all("timestamp" in e for e in events)
    assert broker.assert_no_hidden_access([]).reveals == (first, dependent)


def test_public_specs_have_no_teacher_cost_usage_or_checker_fields(bank):
    broker, ledger, snapshot, ids, path = bank
    specs = candidates(broker, ledger, snapshot)
    for spec in specs:
        fields = asdict(spec)
        assert set(fields) == {"query_id", "state_hash", "features", "L", "cost_upper_bound", "cost_confidence", "cap_provenance"}
        assert fields["features"] == {"projection": None, "source_logprob": None, "source_length": None}
    raw = (path / "public/requests.json").read_text()
    assert "SECRET" not in raw and '"usage"' not in raw and '"cost":' not in raw
    assert not broker.assert_no_hidden_access([{"kind": "read", "resource": "sealed"}]).passed
    assert not broker.assert_no_hidden_access([{"kind": "public", "query_id": ids[0], "fields": ["teacher_embedding"]}]).passed
    assert not broker.assert_no_hidden_access([{"kind": "purchased", "query_id": ids[0], "sequence": 100}]).passed


def test_selector_cannot_import_payload_or_read_bank_or_raw_cache(bank):
    import bfas.rtd.selector as selector
    broker, ledger, snapshot, ids, path = bank
    assert not hasattr(selector, "SealedReplayBroker") and not hasattr(selector, "seal_bank")
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("bfas.rtd.selector.sealed")
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("bfas.rtd.sealed." + ids[0])
    trace = []
    public = candidates(broker, ledger, snapshot)
    with public_only(trace):
        assert public[0].cost_upper_bound == 20
        with pytest.raises(PermissionError, match="cannot read"):
            (path / "sealed" / (ids[0] + ".json")).read_text()
        with pytest.raises(PermissionError):
            (ROOT / "data/bfcl_sft/demos_ds.json").read_text()
        with pytest.raises(PermissionError, match="privileged"):
            importlib.import_module("bfas.rtd.broker")  # already cached
        with pytest.raises(PermissionError, match="privileged"):
            from bfas.rtd import bank
    assert len(trace) == 4 and not broker.assert_no_hidden_access(trace).passed
    # A fresh interpreter also rejects importing the privileged module.
    import subprocess
    script = "from bfas.rtd.selector import public_only\nwith public_only([]):\n import bfas.rtd.bank\n"
    env = {**__import__('os').environ, "PYTHONPATH": str(ROOT / "src") + ':' + str(ROOT)}
    run = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True)
    assert run.returncode != 0 and "cannot import privileged" in run.stderr


def test_public_selection_boundary_accepts_replay_and_audits_reveal_order(bank):
    broker, ledger, snapshot, (first, dependent, _), path = bank
    public = candidates(broker, ledger, snapshot)
    selected, trace = select_public(public, lambda specs: first)
    assert selected == first and broker.assert_no_hidden_access(trace).passed
    assert select_public(public, lambda specs: None)[0] is None
    with pytest.raises(ValueError, match="unavailable"):
        select_public(public, lambda specs: dependent)
    with pytest.raises(TypeError):
        select_public([{"teacher_text": "leak"}], lambda specs: None)
    broker.acquire(first)
    sequence = next(e['sequence'] for e in ledger.events if e['kind'] == 'reveal')
    assert not broker.assert_no_hidden_access([dict(kind='purchased', query_id=first, sequence=sequence-1)]).passed
    assert broker.assert_no_hidden_access([dict(kind='purchased', query_id=first, sequence=sequence+1)]).passed


def test_hard_budget_rechecks_after_listing_before_any_payload_read(bank, monkeypatch):
    broker, ledger, snapshot, (first, _, third), path = bank
    candidates(broker, ledger, snapshot)
    ledger.reserve(digest("other outstanding request"), 45)
    reads = []
    original = Path.read_text
    def spy(self, *args, **kwargs):
        reads.append(self)
        return original(self, *args, **kwargs)
    monkeypatch.setattr(Path, "read_text", spy)
    with pytest.raises(BudgetError):
        broker.acquire(first)
    assert not reads and not ledger.charges
    assert candidates(broker, ledger, snapshot) == []


def test_unpaid_dependency_chain_reservation_refuses_total_over_budget():
    ledger = Ledger(29)
    with pytest.raises(BudgetError, match="dependencies"):
        ledger.reserve_chain({"prefix": 10, "request": 20})
    assert ledger.remaining == 29 and not ledger.events
    ledger = Ledger(30)
    ledger.reserve_chain({"prefix": 10, "request": 20})
    with pytest.raises(LedgerError, match="dependencies"):
        ledger.settle("request", 12, confidence="exact", usage={}, dependencies=("prefix",))
    ledger.settle("prefix", 8, confidence="exact", usage={})
    ledger.settle("request", 12, confidence="exact", usage={}, dependencies=("prefix",))
    assert ledger.spent == 20 and ledger.remaining == 10
    assert not ledger.settle("request", 12, confidence="exact", usage={})
    ledger.reserve("overflow", 10)
    with pytest.raises(LedgerError, match="upper bound"):
        ledger.settle("overflow", 11, confidence="exact", usage={})
    assert ledger.spent == 20


def test_inner_fold_is_enforced_for_candidates_and_acquisition(bank):
    broker, ledger, snapshot, ids, path = bank
    candidates(broker, ledger, snapshot)
    with pytest.raises(LedgerError, match="ownership"):
        broker.list_candidates(snapshot, {ids[0]}, 60)
    broker.set_inner_parents({"feedback"})
    with pytest.raises(ValueError, match="different inner"):
        candidates(broker, ledger, snapshot)
    with pytest.raises(UnavailableError):
        broker.acquire(ids[0])
    assert broker.list_candidates(StudentSnapshot("s", frozenset({"feedback"})), (), 60) == []


def test_payload_integrity_and_overwrite_refusal(bank):
    broker, ledger, snapshot, ids, path = bank
    candidates(broker, ledger, snapshot)
    payload = path / "sealed" / (ids[0] + ".json")
    data = json.loads(payload.read_text()); data["cost"] = 61
    payload.write_text(json.dumps(data))
    with pytest.raises(LedgerError, match="integrity"):
        broker.acquire(ids[0])
    assert not ledger.charges and not ledger.reservations
    with pytest.raises(LedgerError, match="existing ledger"):
        Ledger(60, ledger.path)


@pytest.fixture
def recorded_bank(tmp_path, request):
    from bfas.rtd.bank_build import seal_v11, state_record
    benchmark = getattr(request, 'param', 'hotpotqa')
    parent = digest('paid benchmark parent')
    state = FullState.create({'task': 'public'}, [{'role': 'user', 'content': 'public'}],
                             'fixed prompt', parent)
    records, payloads = [], {}
    for i, cost in enumerate((7000, 8889, 1)):
        q = f'{i + 1:064x}'
        confidence = 'estimated' if i == 2 else 'exact'
        records.append(state_record(q, state, benchmark + '_demo_episode', cost, confidence))
        payloads[q] = dict(cost=cost, cost_confidence=confidence,
            usage={'output_tokens': cost, 'input_tokens': 9000}, provenance={'attempts': [0, 1]},
            historical_response='SECRET', behaviors=[dict(state=asdict(state), text='SECRET ACTION')])
    path = tmp_path / 'recorded-bank'
    seal_v11(path, records, payloads, benchmark=benchmark, student='local', public={}, audit={}, inputs={})
    return path, parent, [r.spec.query_id for r in records]


@pytest.mark.parametrize('recorded_bank', ['hotpotqa', 'alfworld', 'bfcl'], indirect=True)
def test_broker_recorded_reservation_exact_fit_and_resume(recorded_bank, tmp_path):
    path, parent, ids = recorded_bank
    original = {p: p.read_bytes() for p in path.rglob('*.json')}
    ledger = Ledger(7000, tmp_path / 'paid-ledger.jsonl')
    broker = SealedReplayBroker(path, ledger, inner_parent_hashes={parent})
    view = StudentSnapshot('paid', frozenset({parent}))
    public = {s.query_id: s for s in candidates(broker, ledger, view)}
    assert broker.reservation_basis == 'recorded_package_cost'
    assert set(public) == {ids[0], ids[2]}  # 16,384 class cap no longer locks out 7k.
    spec = public[ids[0]]
    assert spec.cost_upper_bound == 7000
    assert json.loads(spec.cap_provenance)['archived_class_cap'] == 16384
    package = broker.acquire(ids[0])
    assert package.cost == ledger.spent == 7000 and ledger.remaining == 0
    assert [e['cap'] for e in ledger.events if e['kind'] == 'reserve'] == [7000]
    assert not ledger.reservations
    resumed = Ledger.resume(7000, ledger.path)
    restored = SealedReplayBroker(path, resumed, inner_parent_hashes={parent})
    assert restored.acquire(ids[0]) == package
    assert len([e for e in resumed.events if e['kind'] == 'reveal']) == 1
    resumed.authorize(15890)
    candidates(restored, resumed, view)
    restored.acquire(ids[1])
    estimated = restored.acquire(ids[2])
    assert estimated.cost == 1 and estimated.cost_confidence == 'estimated'
    assert resumed.spent == 15890 and not resumed.reservations
    assert all(p.read_bytes() == content for p, content in original.items())


@pytest.mark.parametrize('recorded_bank', ['webshop'], indirect=True)
def test_broker_other_benchmark_keeps_class_reservations(recorded_bank):
    path, parent, _ = recorded_bank
    ledger = Ledger(15000)
    broker = SealedReplayBroker(path, ledger, inner_parent_hashes={parent})
    assert broker.reservation_basis == 'public_class_cap'
    assert candidates(broker, ledger, StudentSnapshot('s', frozenset({parent}))) == []


@pytest.mark.parametrize('constraint', ['global', 'window', 'package_count'])
def test_broker_recorded_cost_rechecks_stale_offer_before_reveal(recorded_bank, monkeypatch, constraint):
    path, parent, ids = recorded_bank
    ledger = Ledger(15000)
    broker = SealedReplayBroker(path, ledger, inner_parent_hashes={parent})
    candidates(broker, ledger, StudentSnapshot('s', frozenset({parent})))
    if constraint == 'global':
        ledger.reserve('other', 8001)
    elif constraint == 'window':
        ledger.open_window('w', 6999, 20)
    else:
        ledger.open_window('w', 15000, 0)
    monkeypatch.setattr(broker, '_read_payload', lambda q: pytest.fail('overflow must not reveal'))
    with pytest.raises(BudgetError):
        broker.acquire(ids[0])
    assert not ledger.charges and ids[0] not in ledger.reservations


@pytest.mark.parametrize('when', ['startup', 'reveal'])
def test_broker_recorded_cost_validates_payload_integrity(recorded_bank, when):
    path, parent, ids = recorded_bank
    ledger = Ledger(15000)
    if when == 'reveal':
        broker = SealedReplayBroker(path, ledger, inner_parent_hashes={parent})
        candidates(broker, ledger, StudentSnapshot('s', frozenset({parent})))
    payload = path / 'sealed' / (ids[0] + '.json')
    value = json.loads(payload.read_text())
    value['cost'] = 1
    payload.write_text(json.dumps(value))
    with pytest.raises(LedgerError, match='integrity'):
        if when == 'startup':
            SealedReplayBroker(path, ledger, inner_parent_hashes={parent})
        else:
            broker.acquire(ids[0])
    assert not ledger.charges and not ledger.reservations


@pytest.mark.parametrize('recorded_bank', ['hotpotqa', 'alfworld', 'bfcl'], indirect=True)
def test_preflight_recorded_cost_stops_before_first_overflow(recorded_bank):
    from bfas.rtd.preflight import acquisition_preflight
    path, parent, ids = recorded_bank
    manifest = dict(bank_path=str(path), budget_ceilings=[15000, 30000])
    result = acquisition_preflight(manifest, SimpleNamespace(parents={parent: 'task'}))
    first, second = result['checkpoints']
    assert first['purchase_count'] == 1 and first['spent_tokens'] == 7000
    assert first['next_query_id'] == ids[1] and first['next_reservation_tokens'] == 8889
    assert first['stop_reason'] == 'first_overflow'  # The later one-token package fits, but is not skipped to.
    assert second['purchase_count'] == 3 and second['spent_tokens'] == 15890
    assert second['stop_reason'] == 'inventory_exhausted'


def test_acquisition_executor_recorded_cost_stops_at_overflow(recorded_bank):
    from bfas.rtd.experiment_v11 import BatchExperimentMixin
    path, parent, ids = recorded_bank
    ledger = Ledger(15000)
    broker = SealedReplayBroker(path, ledger, inner_parent_hashes={parent})
    specs = {s.query_id: s for s in candidates(broker, ledger, StudentSnapshot('s', frozenset({parent})))}
    state = dict(decision=True, window_id='w', window_budget=15000, selected=ids,
        transaction_index=0, transaction_query=ids[0], hard_stops=[], pending_ids=[], round=1, step=1,
        cost_model=SimpleNamespace(observe_revealed=lambda *a, **kw: None),
        batch_specs=specs, batch_rows=dict.fromkeys(ids))
    executor = SimpleNamespace(state=state, fixed=False, alpha_d=True, ledger=ledger, broker=broker,
        max_new_packages=20, journal=SimpleNamespace(append=lambda *a, **kw: None),
        batch_remaining_candidates=lambda: [], save=lambda: None)
    BatchExperimentMixin.batch_selected(executor)
    BatchExperimentMixin.batch_selected(executor)
    assert state['pending_ids'] == [ids[0]] and ledger.spent == 7000
    assert state['transaction_index'] == len(ids) and state['transaction_query'] is None
    assert state['hard_stops'][0]['query_id'] == ids[1]
    assert not ledger.reservations and ids[2] not in ledger.owned_ids


def test_acquisition_policy_uses_recorded_bounds_below_class_cap(recorded_bank):
    import numpy as np
    from bfas.rtd.acquisition import BatchAcquisitionPolicy, CostRegressor, PrePurchaseFeatures, ValuePosterior
    path, parent, ids = recorded_bank
    ledger = Ledger(15000)
    broker = SealedReplayBroker(path, ledger, inner_parent_hashes={parent})
    specs = candidates(broker, ledger, StudentSnapshot('s', frozenset({parent})))
    rows = {q: PrePurchaseFeatures(q, (1.,), 'r1') for q in ids}
    policy = BatchAcquisitionPolicy(ValuePosterior(1, round_id='r1'), CostRegressor(1),
                                    rng=np.random.default_rng(0))
    selected = policy.choose_batch(specs, rows, remaining_budget=15000, random_control=True)
    assert selected and policy.last_decision['query_ids'] == ids
    assert policy.last_decision['expected_costs'] == [7000, 8889, 1]
    for q in sorted(selected):
        broker.acquire(q)
    assert ledger.spent <= 15000 and not ledger.reservations
