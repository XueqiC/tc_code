from dataclasses import replace
import json

import numpy as np
import pytest
import torch

from bfas.rtd.baselines.config import BaselineConfig, GRIDS, load_config
from bfas.rtd.baselines.exposure import ExposureDistribution, VT_FORMULA, budget_match
from bfas.rtd.baselines.pool import build_slots
from bfas.rtd.functional_step import lora_parameters
from bfas.rtd.transport import Behavior, SourceSample
from test_rtd_baselines_helpers import TinyTokenizer, h, prepared_fixture, tiny_backend


def test_inverse_cost_exact_distribution_and_unbiased_complete_action_objective():
    dist = ExposureDistribution((.25, .75), (2, 6), "tokens")
    assert dist.Z == .25
    assert dist.q == (.5, .5)
    assert dist.rho == (.5, 1.5)
    assert dist.expected_cost == 4
    losses = [3., 19.]  # complete action sums, with no length normalization
    assert sum(q*rho*loss for q, rho, loss in zip(dist.q, dist.rho, losses)) == 15.
    assert sum(p*loss for p, loss in zip(dist.p, losses)) == 15.
    assert dist.journal()["formula"] == VT_FORMULA
    assert VT_FORMULA in ExposureDistribution.__doc__
    batches = dist.schedule(commits=12, seed=0, target_tokens=48000, pg_reserved_tokens=4000)
    draws = sum(batches, [])
    assert len(draws) == 11000  # budget assigned in advance, not token-based stopping
    assert sum(i == 0 for i in draws)/len(draws) == pytest.approx(.5, abs=.02)
    assert batches == dist.schedule(commits=12, seed=0, target_tokens=48000, pg_reserved_tokens=4000)
    with pytest.raises(ValueError, match="feedback"):
        dist.schedule(commits=12, seed=0, target_tokens=40, pg_reserved_tokens=40)


def test_slots_and_72_commit_grid_preserve_exact_fold_boundaries_and_windows():
    for commits, batch_size in ((12, 8), (24, 4)):
        batches = ExposureDistribution((1.,), (7,), "slots").schedule(commits=commits, seed=0)
        assert len(batches) == commits
        assert [len(b) for b in batches] == [batch_size]*commits
        assert sum(map(len, batches))*3 == 288
        assert [sum(map(len, batches[:i])) for i in range(0, commits, commits//4)] == [0, 24, 48, 72]


def test_matched_budget_requires_both_round_microbatch_and_final_one_percent():
    assert budget_match([1000]*3, [1005, 995, 1001], [8]*3)["matched"]
    assert not budget_match([1000]*3, [1010, 990, 1000], [8]*3)["matched"]
    assert not budget_match([1000]*3, [1020]*3, [100]*3)["matched"]


def test_same_pool_p0_conditioning_preserves_paid_multiplicity_and_parent_mass(tmp_path):
    evidence, _, support = prepared_fixture(tmp_path)
    backend = tiny_backend()
    params = lora_parameters(backend.model)
    sources = {s.state_hash: tuple(SourceSample(Behavior(s, ""), backend.identity(params), (3,), 3, -1.) for _ in range(2))
               for s in support.states.values()}
    slots, audit = build_slots(evidence, sources, TinyTokenizer(), round_number=1, recipe="b1")
    assert len(slots) == 6  # two samples x THREE paid teacher items, no alias inflation
    assert [s.p for s in slots] == pytest.approx([1/6]*6)
    assert audit["teacher_available_mass"] == .5
    assert audit["parent_mass"] == pytest.approx({f"{0:064x}": 1.})
    assert audit["query_mass"] == pytest.approx({h("paid_a"): 2/3, h("paid_a_independent"): 1/3})
    mix, audit = build_slots(evidence, sources, TinyTokenizer(), round_number=1, recipe="b3_mix")
    assert len(mix) == 8
    assert sum(s.p for s in mix if s.teacher is None) == .5
    assert sum(s.p for s in mix) == pytest.approx(1)
    assert all(s.source.behavior.state.parent_hash[-1] in "02" for s in mix)
    assert sum(c["cost"] for c in evidence["charges"]) == 31  # independent of slots/views/repeated reads


@pytest.mark.parametrize("patch", [dict(features_off=False), dict(optimizer="adamw", recipe="b4"),
    dict(commits=72, recipe="b3_sft"), dict(seed=1), dict(beta=.2), dict(new_teacher_calls=True),
    dict(token_cost="action_only"), dict(token_relative_tolerance=.1), dict(lambda_pg=0),
    dict(lr=float("nan")), dict(evaluate_after_round=True)])
def test_schema_rejects_silent_protocol_changes(patch):
    with pytest.raises(ValueError):
        BaselineConfig(**(dict(recipe="b1") | patch))


def test_schema_unknown_yaml_and_declared_grids(tmp_path):
    path = tmp_path/"bad.yaml"
    path.write_text("recipe: b1\nsource_quality_transport: true\n")
    with pytest.raises(ValueError, match="unknown"):
        load_config(path)
    assert GRIDS["b1"] == {"lr": [3e-6, 1e-5, 3e-5], "commits": [36, 72]}
    assert GRIDS["b2"] == {"lr": [1e-6, 5e-6, 1e-5], "commits": [36, 72]}
