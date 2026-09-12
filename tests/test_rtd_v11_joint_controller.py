"""Rev 3.1 reference restoration and D9 paid-window controller contracts."""
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest
import torch

from bfas.behavior.deltas import tensor_state_hash
from bfas.rtd.alpha_d import BatchReference, FeedbackStatistic, dot, gram
from bfas.rtd.functional_step import FrozenStep
from bfas.rtd.joint_surrogate import control, execute_update, marginal_values, replacement_batch, set_solution
from bfas.rtd.return_gradient import ReturnGradient
from test_rtd_checks import toy_bank
from test_rtd_v11_alpha_d import engine, finish_step, pair_problem


def reveal(e):
    e.round_start(); e.step_start(); e.reference()
    while e.state['phase'] == 'selected':
        e.selected()
    e.revealed()


def test_rev31_three_feedback_roles_and_same_batch_z(toy_bank, tmp_path):
    e = engine(tmp_path/'roles', toy_bank, smoke=False, d_warmup_windows=0, d_lambda=0.)
    choose = e.choose_feedback_tasks
    e.choose_feedback_tasks = lambda: [(parent, 32) for parent, _ in choose()]
    reveal(e)
    s, ref = e.state, e.state['d_reference']
    assert s['d_feedback'].parameter_hash == tensor_state_hash(ref.theta0)
    assert s['d_feedback'].parameter_hash != s['reference_feedback'].parameter_hash
    np.testing.assert_allclose(s['alpha_d_control']['z_hat'],
        [dot(s['d_reference_feedback'].gradient, v) for v in ref.directions])
    for n in ref.theta0:
        expected = sum((float(d)*v[n] for d, v in zip(s['d_solution'], ref.directions)),
                       torch.zeros_like(ref.theta0[n]))
        torch.testing.assert_close(s['actual'][n]-ref.theta0[n], expected, atol=1e-15, rtol=1e-12)
    e.actual()
    assert s['actual_feedback'].parameter_hash == tensor_state_hash(s['actual'])
    assert np.any(s['d_solution'])
    assert len({s['reference_feedback'].parameter_hash, s['d_feedback'].parameter_hash,
                s['actual_feedback'].parameter_hash}) == 3
    for role in ('acquisition_reference_feedback', 'same_batch_reference_feedback', 'post_commit_feedback'):
        assert any(r['kind'] == 'return_gradient' and r['role'] == role for r in e.journal.events)


@pytest.mark.parametrize('empty', [False, True])
def test_old_acquisition_feedback_cannot_substitute_for_same_batch_even_same_hash(toy_bank, tmp_path, empty):
    e = engine(tmp_path/'substitution', toy_bank, max_new_packages_per_window=0 if empty else 2)
    feedback = e.feedback
    saved = {}
    def substituted(parameters, role, **kw):
        if role == 'same_batch_reference_feedback':
            return saved['old']
        result = feedback(parameters, role, **kw)
        if role == 'acquisition_reference_feedback':
            saved['old'] = result
        return result
    e.feedback = substituted
    with pytest.raises(ValueError, match='same_batch_reference_feedback.*substituted'):
        reveal(e)


def test_v0_all_24_committed_parameter_hashes_match_pre_restoration(toy_bank, tmp_path):
    e = engine(tmp_path/'V0', toy_bank, arm='V0', smoke=False)
    e.run()
    baseline = json.loads((Path(__file__).parent/'fixtures/rtd_v11/v0_rev2_parameter_hashes.json').read_text())
    assert [s['actual_hash'] for s in e.state['steps']] == baseline['hashes']


def numeric_problem(baseline=((-1., 0.), (0., -1.)), directions=((1., 0.), (0., 1.)),
                    gradient=(.2, .2), weights=(.5, .5)):
    start = {'x': torch.zeros(2, dtype=torch.float64)}
    step = FrozenStep({'x': torch.ones(2, dtype=torch.float64)}, 1., 'r1', {})
    tensor = lambda x: {'x': torch.tensor(x, dtype=torch.float64)}
    gs, vs = tuple(map(tensor, baseline)), tuple(map(tensor, directions))
    w = torch.tensor(weights, dtype=torch.float64)
    theta0 = step.update(start, {'x': sum(g['x']*wi for g, wi in zip(gs, w))})
    batch = BatchReference(tensor_state_hash(start), theta0, vs, w, torch.full((2,), .5), gs)
    zero = tensor((0., 0.))
    old = replace(batch, theta0=start, baseline_gradients=(zero, zero), directions=(zero, zero), alpha=torch.zeros(2))
    fb = ReturnGradient(tensor(gradient), tensor_state_hash(start), {})
    stat = FeedbackStatistic(fb.gradient, (fb.gradient, fb.gradient), (1., 1.), ('task', 'task'),
                             fb.parameter_hash, 1, 'smoke_zero')
    _, _, _, sources, _ = pair_problem()
    pairs = tuple(replace(p, record=replace(p.record, query_id=q, is_new=True))
                  for p, q in zip(sources, ('q1', 'q2')))
    return pairs, batch, old, start, step, fb, stat


@pytest.mark.parametrize('collinear', [False, True])
def test_joint_gain_equals_orthogonal_sum_and_discounts_collinear_redundancy(collinear):
    directions = ((1., 0.), (1., 0.)) if collinear else ((1., 0.), (0., 1.))
    _, batch, _, _, step, fb, stat = numeric_problem(directions=directions)
    z, error = stat.project(batch.directions, 'loo')
    K = gram(batch.directions, step.diagonal)
    d, result = control(z, error, K, batch.alpha)
    assert result['predicted_joint_gain'] >= result['predicted_independent_joint_gain']-1e-12
    if collinear:
        assert result['predicted_joint_gain'] < result['sum_independent_gains']
        assert result['K_statistics']['cosine_above_threshold_fraction'] == 1.
        assert result['predicted_joint_gain'] == pytest.approx(.02)
    else:
        assert result['predicted_joint_gain'] == pytest.approx(result['sum_independent_gains'])
        np.testing.assert_allclose(d, [.2, .2])
    assert result['optimized_variables'] == 'd_only' and not result['raw_z_prefilter']


def test_joint_controller_does_not_prefilter_a_raw_zero_gain_direction():
    d, _ = control([1., 0.], [.1, .1], [[1., .9], [.9, 1.]], [.5, .5])
    assert d[0] > 0 and d[1] < 0


def test_binary_exact_same_batch_affine_update_and_stop_gradient():
    _, batch, _, start, step, _, _ = numeric_problem()
    d = torch.tensor([.25, -.5], dtype=torch.float64, requires_grad=True)
    selected = execute_update(batch, d, start, step)
    expected = sum(float(di.detach())*v['x'] for di, v in zip(d, batch.directions))
    assert torch.equal(selected['x'].detach()-batch.theta0['x'].detach(), expected)
    assert selected['x'].grad_fn is None
    assert torch.equal(execute_update(batch, [0., 0.], start, step)['x'], batch.theta0['x'])


@pytest.mark.parametrize('mode', ['joint', 'independent'])
def test_controller_legality_frozen_alpha_and_source_shares(mode):
    a = np.array([0., .01, .3, .5, 1.])
    K = np.eye(5)+.5*np.ones((5, 5))
    d, result = control([4., -3., 2., -.7, 6.], [.1]*5, K, a, mode=mode)
    assert np.all(np.abs(d) <= np.minimum(a, 1-a))
    shares = np.array(result['source_teacher_shares'])
    np.testing.assert_allclose(shares.sum(1), 1.)
    np.testing.assert_array_equal(shares[:, 2], a)
    assert np.all(shares >= 0)


def test_full_set_at_same_batch_reference_is_exactly_d8_control():
    pairs, batch, old, start, step, fb, stat = numeric_problem()
    fb = replace(fb, parameter_hash=tensor_state_hash(batch.theta0))
    stat = replace(stat, parameter_hash=fb.parameter_hash)
    value, d, result = set_solution(batch.baseline_gradients, batch.directions, batch.weights,
        batch.alpha, start, batch.theta0, step, fb, stat)
    expected, diagnostic = control(*stat.project(batch.directions, 'loo'), gram(batch.directions, step.diagonal), batch.alpha)
    np.testing.assert_array_equal(d, expected)
    assert value == diagnostic['predicted_joint_gain']
    assert result['baseline_linear_gain'] == 0.


def test_full_update_surrogate_keeps_good_and_harmful_teacher_effect_with_d_forced_zero():
    p, b, old, start, step, fb, stat = numeric_problem(baseline=((-1., 0.), (1., 0.)), gradient=(1., 0.))
    values, result = marginal_values(p, b, old, start, start, step, fb, stat,
                                    revealed_ids={'q1', 'q2'}, zero=True, d_lambda=0.)
    assert values == {'q1': .5, 'q2': -.5}
    assert result['label_kind'] == 'acquisition_surrogate'
    assert result['full_set']['d_star'] == [0., 0.]
    assert not result['paired_task_evaluation']


def test_leave_one_out_reproduces_additive_case_and_preserves_other_exposures():
    p, b, old, start, step, fb, stat = numeric_problem(gradient=(2., 2.), weights=(.25, .75))
    values, result = marginal_values(p, b, old, start, start, step, fb, stat, revealed_ids={'q1', 'q2'})
    assert sum(values.values()) == pytest.approx(result['full_set_value'])
    reduced = replacement_batch(p, b, old, 'q1', start, step)
    torch.testing.assert_close(reduced.weights, b.weights, rtol=0, atol=0)
    assert reduced.baseline_gradients[1] is b.baseline_gradients[1]
    assert reduced.directions[1] is b.directions[1]
    assert reduced.alpha[1] == b.alpha[1]
    np.testing.assert_array_equal(reduced.theta0['x'].detach(), [0., .75])
    update = execute_update(reduced, result['leave_one_out']['q1']['d_star'], start, step)
    assert tensor_state_hash(start) == b.start_hash
    assert update['x'] is not reduced.theta0['x']


def test_surrogate_requires_same_reference_gradient_and_purchased_set():
    p, b, old, start, step, fb, stat = numeric_problem()
    with pytest.raises(ValueError, match='purchased evidence'):
        marginal_values(p, b, old, start, start, step, fb, stat, revealed_ids={'q1'})
    for wrong in (replace(stat, parameter_hash='wrong'), replace(stat, gradient={'x': torch.zeros(2)})):
        with pytest.raises(ValueError, match='reference'):
            marginal_values(p, b, old, start, start, step, fb, wrong, revealed_ids={'q1', 'q2'})


def test_deleting_package_backfills_all_its_units_with_predetermined_weights():
    p, b, old, start, step, fb, stat = numeric_problem(weights=(.25, .75))
    p = tuple(replace(pair, record=replace(pair.record, query_id='q1')) for pair in p)
    values, diagnostic = marginal_values(p, b, old, start, start, step, fb, stat, revealed_ids={'q1'})
    assert values['q1'] == diagnostic['full_set_value']
    reduced = replacement_batch(p, b, old, 'q1', start, step)
    assert torch.equal(reduced.theta0['x'], start['x'])
    assert torch.equal(reduced.weights, b.weights)


def test_control_switch_config_is_validated_and_v2_defaults_to_joint(toy_bank, tmp_path):
    from bfas.rtd.alpha_d import validate_config
    joint = engine(tmp_path/'joint-config', toy_bank)
    independent = engine(tmp_path/'independent-config', toy_bank, acquisition_value_mode='independent')
    assert joint.config['acquisition_value_mode'] == 'joint'
    assert independent.config['acquisition_value_mode'] == 'independent'
    assert joint.config | {'acquisition_value_mode': 'independent'} == independent.config
    with pytest.raises(ValueError, match='acquisition_value_mode'):
        validate_config(joint.config | {'acquisition_value_mode': 'joint_surrogate'})


def test_joint_vs_independent_pair_switches_only_controller(toy_bank, tmp_path):
    e = engine(tmp_path/'paired', toy_bank, smoke=False, d_warmup_windows=0)
    reveal(e)
    s, ref = e.state, e.state['d_reference']
    z, error = s['d_feedback'].project(ref.directions, 'loo')
    K = gram(ref.directions, s['step_rule'].diagonal)
    frozen = (ref.weights.clone(), ref.alpha.clone(), s['selected'][:], s['feedback_tasks'][:],
              e.ledger.spent, e.sampling_rng.get_state().clone(), tensor_state_hash(s['parameters']))
    outputs = []
    for mode in ('joint', 'independent'):
        d, diagnostic = control(z, error, K, ref.alpha, mode=mode)
        outputs.append(diagnostic)
        execute_update(ref, d, s['alpha_start'], s['step_rule'])
    for key in outputs[0]:
        if key not in {'controller', 'd_star', 'solver', 'source_teacher_shares'}:
            assert outputs[0][key] == outputs[1][key]
    torch.testing.assert_close(ref.weights, frozen[0], rtol=0, atol=0)
    torch.testing.assert_close(ref.alpha, frozen[1], rtol=0, atol=0)
    assert (s['selected'], s['feedback_tasks'], e.ledger.spent) == frozen[2:5]
    assert torch.equal(e.sampling_rng.get_state(), frozen[5])
    assert tensor_state_hash(s['parameters']) == frozen[6]


def test_sealed_candidate_gradients_never_enter_runtime_surrogate_or_posterior(toy_bank, tmp_path, monkeypatch):
    from bfas.rtd import alpha_d
    e = engine(tmp_path/'leak', toy_bank, max_new_packages_per_window=1)
    seen = set()
    original = alpha_d.components
    def guarded(pair, *a, **kw):
        if pair.record.teacher is not None:
            assert pair.record.query_id in e.ledger.owned_ids
            seen.add(pair.record.query_id)
        return original(pair, *a, **kw)
    monkeypatch.setattr(alpha_d, 'components', guarded)
    e.run()
    assert seen == e.ledger.owned_ids and len(seen) == 1
    assert set(e.state['posterior'].observations) == seen
    row = e.state['steps'][0]
    assert set(row['selection']['query_ids'])-seen
    assert set(row['acquisition_surrogate']['marginal_values']) == seen
    for _, label in e.state['posterior'].observations.values():
        assert label.approximation.startswith('acquisition_surrogate')
    # A fabricated unpurchased teacher fails BEFORE any source or gradient call.
    _, _, _, sources, _ = pair_problem()
    record = replace(sources[0].record, query_id='sealed-not-purchased')
    with pytest.raises(ValueError, match='purchased evidence'):
        e.alpha_draw_pairs([record], role='leak_probe')


def test_pre_purchase_context_and_strong_shrinkage_persist_across_windows(toy_bank, tmp_path):
    from bfas.rtd.acquisition import CONTEXTUAL_DIMENSION, TRAINING_CONTEXT_FEATURES
    e = engine(tmp_path/'context', toy_bank, smoke=False)
    e.round_start()
    for _ in range(4):
        finish_step(e)
    contexts = [r for r in e.journal.events if r['kind'] == 'acquisition_pre_purchase_context']
    assert list(contexts[0]['context']) == list(TRAINING_CONTEXT_FEATURES)
    assert contexts[0]['purchased_set'] == [] and contexts[-1]['purchased_set']
    assert contexts[0]['context']['step_fraction'] != contexts[-1]['context']['step_fraction']
    assert contexts[0]['context']['purchased_count'] < contexts[-1]['context']['purchased_count']
    assert contexts[-1]['context']['feedback_age'] == .25
    posterior = e.state['posterior']
    assert posterior.dimension == CONTEXTUAL_DIMENSION and posterior.shrinkage_factor == .05
    for q, (features, label) in posterior.observations.items():
        np.testing.assert_array_equal(features.values, contexts[0]['feature_rows'][q])
    restored = engine(e.directory, toy_bank, resume=True, smoke=False)
    assert restored.state['posterior'].predictive_history == posterior.predictive_history
    np.testing.assert_array_equal(restored.state['posterior'].model.information, posterior.model.information)


def test_low_predictive_power_keeps_strong_shrinkage_after_four_windows_and_drift():
    from bfas.rtd.acquisition import PrePurchaseFeatures, ValuePosterior
    from bfas.rtd.insertion import BatchInsertionLabel, LabelType
    posterior = ValuePosterior(2, round_id='r1', contextual_shrinkage=True)
    for window in range(4):
        ids = [f'q{window}a', f'q{window}b']
        rows = {q: PrePurchaseFeatures(q, (1., 1.), 'r1') for q in ids}
        labels = {q: BatchInsertionLabel(q, LabelType.PENDING_NEW, value, 'r1', 'start', 'reference',
                  approximation='acquisition_surrogate_full_update_with_teacher_injection')
                  for q, value in zip(ids, (-1., 2.))}
        posterior.observe_batch(rows, labels, np.eye(2))
    assert len(posterior.predictive_history) == 4 and len(posterior.observations) == 8
    assert posterior.shrinkage_factor == .05
    mean = posterior.model.mean.copy()
    posterior.inflate_for_drift(query_id='q0a', old_value=-1., new_value=20., variance=1.)
    np.testing.assert_allclose(posterior.model.mean, mean)
    np.testing.assert_allclose(posterior.model.information, .05*posterior.unshrunk_information)


def test_training_context_changes_value_features_without_changing_cost_or_v0_budget(toy_bank):
    from bfas.rtd.acquisition import CostRegressor, PrePurchaseFeatures
    spec = toy_bank[1][0].spec
    values = tuple(np.linspace(.1, 1., 39))
    old = PrePurchaseFeatures(spec.query_id, values, 'r1')
    context = replace(old, values=(*values[:38], *range(10), values[-1]))
    legacy, contextual = CostRegressor(39), CostRegressor(39)
    for model, row in ((legacy, old), (contextual, context)):
        model.begin_round('r1')
        model.observe_revealed(spec, row, cost=17, confidence=spec.cost_confidence, revealed_ids={spec.query_id})
    np.testing.assert_array_equal(contextual.model.precision, legacy.model.precision)
    np.testing.assert_array_equal(contextual.model.information, legacy.model.information)
    assert contextual.predict(spec, context) == legacy.predict(spec, old)


def test_realised_gains_use_new_batches_and_never_fit_the_posterior(toy_bank, tmp_path):
    e = engine(tmp_path/'validation', toy_bank)
    reveal(e)
    params = tensor_state_hash(e.state['parameters'])
    e.actual()
    s = e.state
    labels = {q: asdict(label) for q, label in s['labels'].items()}
    assert len(s['paired_validations']) == 2
    assert len(s['posterior'].observations) == 0
    for row in s['paired_validations']:
        assert row['independent_update_executions'] == 2 and row['updates_per_arm'] == 1
        assert row['full']['batch_id'] != row['control']['batch_id']
        assert row['validation_only'] and not row['used_for_posterior'] and not row['selection_feedback_reused']
        assert row['realised_paired_gain'] == row['full']['mean_return']-row['control']['mean_return']
        row['realised_paired_gain'] = 123456.  # validation journal cannot affect labels
    e.feedback_commit()
    assert {q: asdict(label) for q, (_, label) in s['posterior'].observations.items()} == labels
    assert tensor_state_hash(s['parameters']) == params


@pytest.mark.parametrize('phase', ['revealed', 'actual', 'feedback'])
def test_resume_mid_window_preserves_controller_labels_validation_and_posterior(toy_bank, tmp_path, phase):
    clean = engine(tmp_path/'clean', toy_bank, smoke=False, d_warmup_windows=0)
    clean.round_start(); finish_step(clean)
    broken = engine(tmp_path/'broken', toy_bank, smoke=False, d_warmup_windows=0)
    def stop(current):
        if current == phase:
            raise RuntimeError('D9 crash')
    broken.after_save = stop
    with pytest.raises(RuntimeError, match='D9 crash'):
        broken.run()
    restored = engine(broken.directory, toy_bank, resume=True, smoke=False, d_warmup_windows=0)
    while not restored.state['steps']:
        getattr(restored, 'feedback_commit' if restored.state['phase'] == 'feedback' else restored.state['phase'])()
    if restored.state['phase'] == 'committed':
        restored.committed()
    assert restored.state['steps'] == clean.state['steps']
    assert restored.ledger.charges == clean.ledger.charges
    np.testing.assert_array_equal(restored.state['posterior'].model.information, clean.state['posterior'].model.information)
