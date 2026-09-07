import json

import numpy as np
import pytest
import yaml

from bfas.rtd.baselines.acquisition import AcquisitionConfig, historical_cost_allocation, random_policy, sealed_random_window
from bfas.rtd.baselines.journal import ResourceJournal
from bfas.rtd.broker import SealedReplayBroker
from bfas.rtd.ledger import Ledger
from bfas.rtd.selector import StudentSnapshot
from test_rtd_baselines_helpers import make_fixture


def test_sealed_small_actual_budget_cannot_override_public_cap(tmp_path):
    _, bank, support, _ = make_fixture(tmp_path)
    ledger = Ledger(31)
    broker = SealedReplayBroker(bank, ledger, inner_parent_hashes=set(support.states))
    journal = ResourceJournal(tmp_path/"procurement.jsonl")
    snapshot = StudentSnapshot("frozen-student", frozenset(support.states))
    policy = random_policy(round_number=1, rng=np.random.default_rng(0))
    assert sealed_random_window(broker, snapshot, policy, journal, round_number=1, step=1) is None
    assert ledger.spent == 0 and not ledger.events  # all public caps are 100, despite historical spend=31
    record = journal.events[-1]
    assert record["distribution"]["query_ids"] == [None]
    assert record["distribution"]["probabilities"] == [1.]
    assert not record["learned_value"]


def test_historical_cost_uniform_feasible_distribution_charges_whole_dependencies():
    catalog = [dict(query_id="a", dependencies=[], cost=3, confidence="exact"),
               dict(query_id="b", dependencies=["a"], cost=5, confidence="estimated"),
               dict(query_id="c", dependencies=[], cost=100, confidence="exact")]
    result = historical_cost_allocation(catalog, 9)
    assert result["owned"] == ["a", "b"] and result["spent"] == 8 and result["unspent"] == 1
    assert result["draws"][0]["feasible"] == ["a"]
    assert result["draws"][1]["feasible"] == ["b"]
    assert all(row["probability"] == 1/len(row["feasible"]) for row in result["draws"])
    with pytest.raises(ValueError, match="teacher text"):
        historical_cost_allocation([dict(catalog[0], text="forbidden answer")], 9)


def test_acquisition_schema_keeps_protocols_distinct():
    assert AcquisitionConfig().protocol == "sealed_random_L1"
    assert AcquisitionConfig(protocol="historical_cost_public").protocol == "historical_cost_public"
    with pytest.raises(ValueError):
        AcquisitionConfig(replay_length=4)
    with pytest.raises(ValueError):
        AcquisitionConfig(new_teacher_calls=True)
