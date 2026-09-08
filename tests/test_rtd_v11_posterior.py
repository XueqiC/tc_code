"""Persistent, heteroscedastic paid-only value/cost inference (CPU)."""
import copy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from bfas.rtd.acquisition import ValuePosterior, LegacyValuePosterior, CostRegressor, BatchAcquisitionPolicy
from bfas.rtd.insertion import BatchInsertionLabel, LabelType
from bfas.rtd.value_feedback import insertion_statistics, reliability
from test_rtd_acquisition import features, spec
from test_rtd_checks import toy_bank, experiment, engine_config, tiny_backend


def label(q='q', value=2., variance=.1, round_id='r1'):
    return BatchInsertionLabel(q, LabelType.PENDING_NEW, value, round_id, 'start', 'ref', variance=variance)


def test_rounds_preserve_models_observations_and_round_identity():
    p, c = ValuePosterior(2, round_id='r1'), CostRegressor(2)
    p.observe(features(), label())
    c.observe_revealed(spec(), features(), cost=5, confidence='exact', revealed_ids={'q'})
    pm, cm = p.model, c.model
    precision = pm.precision.copy()
    p.begin_round('r2'); c.begin_round('r2')
    assert p.model is pm and c.model is cm and p.history[0]['round_id'] == 'r1'
    np.testing.assert_array_equal(pm.precision, precision)
    assert c.observations['q']['round_id'] == 'r1'
    with pytest.raises(ValueError, match='stale'):
        p.observe(features('other'), label('other'))
    p.observe(features('other', round_id='r2'), label('other', round_id='r2'))
    assert len(p.history) == 2 and len(c.observations) == 1
    legacy = LegacyValuePosterior(2, round_id='r1')
    legacy.observe(features(), label()); legacy.begin_round('r2')
    np.testing.assert_array_equal(legacy.model.precision, np.eye(2))
    assert not legacy.observations


def test_heteroscedastic_and_correlated_labels_do_not_fake_independent_feedback():
    precise, noisy = (ValuePosterior(2, round_id='r1') for _ in range(2))
    precise.observe(features(), label(variance=.01))
    noisy.observe(features(), label(variance=10.))
    assert np.trace(precise.model.precision) > np.trace(noisy.model.precision)
    p = ValuePosterior(2, round_id='r1')
    rows = {q: features(q, values=(1., 0.)) for q in ('a', 'b')}
    labels = {q: label(q, variance=1.) for q in rows}
    covariance = np.array([[1., .9], [.9, 1.]])
    p.observe_batch(rows, labels, covariance)
    assert p.model.precision[0, 0] == pytest.approx(1+2/1.9)
    assert all(r['noise_variance'] == 1. for r in p.history)


def test_drift_inflates_covariance_preserving_mean_and_all_history():
    p = ValuePosterior(2, round_id='r1'); p.observe(features(), label())
    mean = p.model.mean.copy(); covariance = np.linalg.inv(p.model.precision)
    p.begin_round('r2')
    event = p.inflate_for_drift(query_id='q', old_value=2., new_value=-3., variance=.1, previous_variance=.1)
    assert event['uncertainty_inflation'] > 1 and len(p.observations) == 1
    np.testing.assert_allclose(p.model.mean, mean)
    assert np.linalg.eigvalsh(np.linalg.inv(p.model.precision)-covariance).min() > 0
    reweight = replace(label(value=-3., round_id='r2'), label_type=LabelType.REWEIGHT_EXISTING)
    p.observe(features(round_id='r2'), reweight)
    assert len(p.reweight_observations) == 1 and p.history[-1]['round_id'] == 'r2'
    np.testing.assert_allclose(p.model.mean, mean)
    with pytest.raises(ValueError, match='purchased'):
        p.inflate_for_drift(query_id='hidden', old_value=0., new_value=1., variance=1.)


def test_significant_differences_change_comparable_decision():
    p = ValuePosterior(2, round_id='r1'); previous = copy.deepcopy(p.model)
    rows = dict(a=features('a', values=(1., 0.)), b=features('b', values=(0., 1.)))
    # First establish an opposite, precise policy; compare on the SAME current
    # features/costs/candidates with the SAME standard-normal posterior draw.
    previous.observe((1., 0.), -10., noise_variance=1e-6)
    previous.observe((0., 1.), 10., noise_variance=1e-6)
    p.observe(rows['a'], label('a', value=10., variance=1e-6))
    p.observe(rows['b'], label('b', value=-10., variance=1e-6))
    policy = BatchAcquisitionPolicy(p, CostRegressor(2), rng=np.random.default_rng(3))
    assert policy.choose_batch([spec('a', 1), spec('b', 1)], rows, remaining_budget=1,
                               max_new_packages=1, previous_model=previous) == ['a']
    significance = policy.last_decision['value_difference_significance']
    assert significance['significant'] and significance['decision_changed']
    assert significance['previous_comparable_selected'] == ['b']


def batch_config(**extra):
    return engine_config() | dict(protocol_version='1.1.0', rounds=3, exposure_slots_per_window=4,
                                  max_new_packages_per_window=2, acquisition='random') | extra


def test_unpurchased_diagnostics_rejected_before_backend_or_pool_access():
    class Forbidden:
        def __getattribute__(self, key):
            pytest.fail('diagnostic touched backend/reference before proving purchase')
    with pytest.raises(ValueError, match='purchased'):
        insertion_statistics([], Forbidden(), Forbidden(), {'hidden': {}}, revealed_ids=set())
    with pytest.raises(ValueError, match='purchased'):
        reliability(dict(query_ids=['hidden']), {}, revealed_ids=set())


def test_runner_reliability_and_drift_never_read_unpaid_sealed_pool(toy_bank, tmp_path, monkeypatch):
    e = experiment(tmp_path/'run', toy_bank, config=batch_config())
    original = Path.read_text
    reads = []
    def guarded(path, *args, **kwargs):
        if path.parent.name == 'sealed' and len(path.stem) == 64:
            # The acquisition broker may read after reservation. Diagnostics
            # additionally require settled ownership; verify both scopes.
            if 'feedback' in getattr(e.backend, 'context', '') or 'drift' in getattr(e.backend, 'context', ''):
                assert path.stem in e.ledger.owned_ids
            assert path.stem in e.ledger.owned_ids or path.stem in e.ledger.reservations
            reads.append(path.stem)
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'read_text', guarded)
    e.run(stop_after_round=1)
    first = copy.deepcopy(e.state['posterior'].model.precision)
    e.run(stop_after_round=2)
    assert e.state['round_drift']['status'] == 'no_legal_purchased_reference'
    assert {r['round_id'] for r in e.state['posterior'].history} == {'r1', 'r2'}
    assert not np.array_equal(first, e.state['posterior'].model.precision)
    e.run()
    assert e.state['round_drift']['status'] == 'remeasured'
    assert e.state['round_drift']['reliability']['independent_batches']
    assert set(reads) <= e.ledger.owned_ids
    saved = torch.load(e.directory/'round-3/round_state.pt', weights_only=False)
    assert saved['batch_state']['drift_measurements'] == e.state['drift_measurements']
    assert len(saved['posterior'].observations) == 4
