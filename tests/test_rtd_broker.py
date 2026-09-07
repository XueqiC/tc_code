"""Sealing, request dependencies, provenance, and hard-cap CPU checks."""
from dataclasses import asdict
import importlib
import json
from pathlib import Path
import sys

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
