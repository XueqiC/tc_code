"""Request-level Bayesian updates, entropy cost dual and independent reservation."""
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / 'src'):
    sys.path.insert(0, str(p))
from bfas.cc_pairs import digest
from bfas.rtd.acquisition import (AcquisitionPolicy, BayesianLinearRegression, CostRegressor,
    PrePurchaseFeatures, ValuePosterior, budget_distribution, select_and_acquire)
from bfas.rtd.broker import RequestRecord, SealedReplayBroker, seal_bank
from bfas.rtd.insertion import InsertionLabel, LabelType
from bfas.rtd.ledger import Ledger
from bfas.rtd.selector import PublicFeatures, PublicQuerySpec, StudentSnapshot, select_public
from bfas.rtd.transport import FullState


def spec(q='q', cap=100, confidence='exact'):
    return PublicQuerySpec(q, 'state', PublicFeatures(tuple([0.] * 32), -3., 2), 1, cap, confidence, 'public config')


def features(q='q', round_id='r1', values=(1., .3)):
    return PrePurchaseFeatures(q, values, round_id)


def label(q='q', kind=LabelType.PENDING_NEW, value=2., round_id='r1'):
    return InsertionLabel(q, kind, value, round_id, 'start', 'reference')


def test_bayesian_posterior_mean_and_sampling_covariance():
    model = BayesianLinearRegression(2)
    np.testing.assert_array_equal(model.precision, np.eye(2))
    model.observe([1., 1.], 2.)
    np.testing.assert_allclose(model.mean, [2/3, 2/3])
    rng = np.random.default_rng(10)
    samples = np.array([model.sample(rng) for _ in range(12000)])
    np.testing.assert_allclose(samples.mean(0), [2/3, 2/3], atol=.025)
    np.testing.assert_allclose(np.cov(samples.T), [[2/3, -1/3], [-1/3, 2/3]], atol=.035)


def test_request_label_types_duplicates_and_persistent_round_history():
    posterior = ValuePosterior(2, round_id='r1')
    posterior.observe(features(), label(kind=LabelType.REWEIGHT_EXISTING))
    assert len(posterior.reweight_observations) == 1 and not posterior.observations
    np.testing.assert_array_equal(posterior.model.mean, [0., 0.])
    posterior.observe(features('new'), label('new'))
    assert np.linalg.norm(posterior.model.mean) > 0
    with pytest.raises(ValueError, match='duplicate'):
        posterior.observe(features('new'), label('new'))
    precision = posterior.model.precision.copy()
    posterior.begin_round('r2')
    np.testing.assert_array_equal(posterior.model.precision, precision)
    assert set(posterior.observations) == {'new'}
    with pytest.raises(ValueError, match='stale'):
        posterior.observe(features('new'), label('new'))


def test_cost_regressor_only_revealed_usage_with_confidence_flags():
    model = CostRegressor(2)
    assert model.predict(spec(), features()) == 100
    with pytest.raises(ValueError, match='revealed'):
        model.observe_revealed(spec(), features(), cost=10, confidence='exact', revealed_ids=set())
    model.observe_revealed(spec(), features(), cost=10, confidence='exact', revealed_ids={'q'})
    assert 0 < model.predict(spec(), features()) < 100
    assert model.observations['q']['confidence'] == 'exact'
    model.observe_revealed(spec('estimated', confidence='estimated'), features('estimated'),
                          cost=50, confidence='estimated', revealed_ids={'estimated'})
    assert model.observations['estimated']['confidence'] == 'estimated'
    with pytest.raises(ValueError, match='reservation'):
        model.observe_revealed(spec('overflow'), features('overflow'), cost=101, confidence='exact', revealed_ids={'overflow'})


def test_entropy_dual_kkt_empty_prior_and_window_budget():
    p, meta = budget_distribution([0, 0, 0], [0, 10, 30], remaining_budget=100, remaining_windows=1)
    np.testing.assert_allclose(p, [.5, .25, .25])
    assert meta['lambda_c'] == 0
    values, costs = np.array([0., 3., -1., 2.]), np.array([0., 10., 30., 100.])
    p, meta = budget_distribution(values, costs, remaining_budget=40, remaining_windows=4)
    assert meta['lambda_c'] > 0 and meta['per_window_budget'] == 10
    assert p @ costs == pytest.approx(10., abs=1e-10)
    # Stationarity: log(p/p0) = epsilon*v/scale - lambda*c/scale + common constant.
    residual = np.log(p / meta['prior']) - .25 * values + meta['lambda_c'] * costs / meta['cost_scale']
    np.testing.assert_allclose(residual, np.full(len(p), residual[0]), atol=1e-12)
    assert p[2] > 0  # negative values are not threshold-filtered
    p0, zero = budget_distribution(values, costs, remaining_budget=0, remaining_windows=4)
    np.testing.assert_array_equal(p0, [1., 0., 0., 0.])
    assert zero['zero_budget_limit']
    p, _ = budget_distribution([0.], [0.], remaining_budget=2, remaining_windows=1)
    np.testing.assert_array_equal(p, [1.])


def test_opportunity_cost_changes_purchase_vs_empty_but_not_conditional_ranking():
    costs = [0., 1., 1.]
    p, _ = budget_distribution([0., 2., 1.], costs, remaining_budget=100, remaining_windows=1)
    shifted, _ = budget_distribution([0., -2., -3.], costs, remaining_budget=100, remaining_windows=1)
    assert shifted[0] > p[0]
    assert shifted[1] / shifted[2] == pytest.approx(p[1] / p[2])


def test_guarded_public_selection_dedup_no_redraw_and_hard_cap_filter():
    policy = AcquisitionPolicy(ValuePosterior(2, round_id='r1'), CostRegressor(2), rng=np.random.default_rng(0))
    ss = [spec('a', 10), spec('a', 10), spec('b', 11)]
    rows = {q: features(q) for q in ['a', 'b']}
    selected, trace = select_public(ss, lambda public: policy.choose(public, rows,
        remaining_budget=10, remaining_windows=1, random_control=True))
    assert policy.last_decision['query_ids'] == [None, 'a']
    assert policy.last_decision['decision_windows'] == 1 and selected in {None, 'a'}
    assert all(t['kind'] == 'public' for t in trace)
    # No hidden fields accepted, and missing model features aren't fabricated.
    with pytest.raises(ValueError, match='missing source'):
        PrePurchaseFeatures.from_public(PublicQuerySpec('q', 's', PublicFeatures(), 1, 20, 'exact', 'cap'),
            round_id='r1', coverage=0, progress=0, support_return=0)
    policy.choose([spec('a', 10)], rows, remaining_budget=9, remaining_windows=1)
    assert policy.last_decision['selected'] is None and policy.last_decision['probabilities'] == [1.]


def test_real_broker_reservation_precedes_reveal_and_expensive_package_never_offered(tmp_path):
    q, expensive = digest('q'), digest('expensive')
    parent = digest('parent')
    state = FullState.create({'question': 'public'}, [{'role': 'user', 'content': 'public'}], 'prompt', parent)
    records = [RequestRecord(spec(q, 20), parent), RequestRecord(spec(expensive, 100), parent)]
    payloads = {r.spec.query_id: dict(cost=7, cost_confidence='exact', usage={'output_tokens': 7},
        provenance={}, historical_response='hidden', behaviors=[dict(state=asdict(state), text='teacher')]) for r in records}
    bank = seal_bank(tmp_path / 'bank', records, payloads)
    ledger = Ledger(20)
    broker = SealedReplayBroker(bank, ledger, inner_parent_hashes={parent})
    policy = AcquisitionPolicy(ValuePosterior(2, round_id='r1'), CostRegressor(2), rng=np.random.default_rng(0))
    package, decision, trace = select_and_acquire(broker, StudentSnapshot('snapshot', frozenset({parent})), policy,
        {q: features(q)}, remaining_windows=1, random_control=True)
    assert package.query_id == q and expensive not in decision['query_ids']
    assert [e['kind'] for e in ledger.events] == ['reserve', 'reveal']
    assert ledger.events[0]['cap'] == 20 and ledger.spent == 7 and not ledger.reservations
    assert broker.assert_no_hidden_access(trace).passed
    # Once spent, even the cheap hidden actual cost cannot afford the 100 cap.
    package, decision, _ = select_and_acquire(broker, StudentSnapshot('snapshot', frozenset({parent})), policy,
        {}, remaining_windows=1, random_control=True)
    assert package is None and decision['selected'] is None and ledger.spent == 7
