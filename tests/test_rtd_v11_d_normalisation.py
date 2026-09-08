"""D11 scale-free QP, full-update acquisition and frozen-window calibration."""
from dataclasses import replace
import json

import numpy as np
import pytest
import torch

from bfas.behavior.deltas import tensor_state_hash
from bfas.rtd.alpha_d import FeedbackStatistic, build_reference, calibrate_d, gram, solve_d
from bfas.rtd.joint_surrogate import control, marginal_values, set_solution
from test_rtd_checks import toy_bank
from test_rtd_v11_alpha_d import engine, finish_step, pair_problem
from test_rtd_v11_joint_controller import numeric_problem


@pytest.mark.parametrize('controller', ['joint', 'independent'])
def test_direction_rescaling_preserves_solution_and_normalised_diagnostics(controller):
    directions = np.array([[2., 0.], [.6, .8]])
    gradient, error = np.array([.22, .17]), np.array([.02, .01])
    results = []
    for factor in (1., 1e-6):
        v = factor*directions
        z, K = v @ gradient, v @ v.T
        d, diagnostic = control(z, factor*error, K, [.5, .5], mode=controller, tolerance=1e-12)
        direct, meta = solve_d(z, factor*error, K if controller == 'joint' else np.diag(np.diag(K)),
                               [.5, .5], tolerance=1e-12)
        np.testing.assert_array_equal(d, direct)
        assert meta['lambda_effective'] == diagnostic['lambda_effective']
        results.append((d, diagnostic))
    d, before = results[0]
    scaled, after = results[1]
    assert np.all((d > 0) & (d < .5))  # An interior optimum, not box saturation.
    np.testing.assert_allclose(scaled, d, atol=1e-12, rtol=0)
    assert after['lambda_effective']/before['lambda_effective'] == pytest.approx(1e12)
    assert after['lambda_original_units']/before['lambda_original_units'] == pytest.approx(1e6)
    for key in ('trace', 'frobenius_norm', 'diagonal'):
        assert after['K_statistics'][key] == pytest.approx(before['K_statistics'][key])
    for key in ('z_hat_normalised', 'epsilon_hat_normalised', 'predicted_joint_gain',
                'joint_minus_independent_control'):
        assert after[key] == pytest.approx(before[key])
    assert after['K_statistics']['raw_trace'] == pytest.approx(5e-12, rel=1e-12, abs=0)
    assert float(after['K_statistics']['raw_trace_scientific']) == after['K_statistics']['raw_trace']


def test_original_unit_equivalence_and_joint_zero_condition():
    K = 9*np.array([[1., .9, 0.], [.9, 1., 0.], [0., 0., 1.]])
    z, error = 3*np.array([1., 0., .05]), 3*np.array([.1, .1, .2])
    d, result = solve_d(z, error, K, [.5]*3, tolerance=1e-12)
    raw, _ = solve_d(z, error, K, [.5]*3, d_lambda=result['lambda_original_units'],
                     d_lambda_normalisation='none', tolerance=1e-12)
    np.testing.assert_allclose(raw, d, atol=1e-12, rtol=0)
    assert d[1] < 0 and d[2] == 0  # No per-coordinate prefilter.
    i = 2
    cross = K[i] @ d-K[i, i]*d[i]
    assert abs(z[i]/3-result['lambda_effective']*cross) <= error[i]/3
    assert abs(z[i]-result['lambda_original_units']*cross) <= error[i]
    # lambda/s with RAW linear terms is a different, non-equivalent problem.
    wrong, _ = solve_d(z, error, K, [.5]*3, d_lambda=result['lambda_effective'],
                       d_lambda_normalisation='none', tolerance=1e-12)
    assert not np.allclose(wrong, d)


@pytest.mark.parametrize('reason', [None, 'warmup', 'configured_zero', 'insufficient_trajectories'])
def test_zero_direction_fallback_is_finite_and_journaled_even_when_forced_zero(reason):
    d, diagnostic = control([0., 0.], [np.inf]*2 if reason == 'insufficient_trajectories' else [0., 0.],
                            np.zeros((2, 2)), [.5, .5], zero_reason=reason)
    np.testing.assert_array_equal(d, [0., 0.])
    assert diagnostic['K_scale'] == diagnostic['lambda_effective'] == 1.
    assert diagnostic['K_scale_fallback'] is True
    assert diagnostic['solver']['K_scale_fallback'] is True
    assert diagnostic['K_statistics']['trace'] == 0.
    json.dumps(diagnostic, allow_nan=False)


def test_scale_excludes_only_exactly_zero_directions_even_at_alpha_endpoint():
    K = np.diag([4e-24, 0., 9e-24])
    _, diagnostic = control([0.]*3, [0.]*3, K, [0., .5, .5])
    assert diagnostic['K_scale'] == pytest.approx(6.5e-24, rel=1e-12, abs=0)
    assert diagnostic['K_scale_nonzero_directions'] == 2
    assert not diagnostic['K_scale_fallback']
    assert diagnostic['K_statistics']['trace'] == pytest.approx(2.)
    assert diagnostic['K_statistics']['diagonal'] == pytest.approx([4/6.5, 0., 9/6.5])


def test_tiny_model_collinear_directions_have_joint_effect_under_default():
    backend, start, source, pairs, step = pair_problem()
    pair = pairs[0]
    duplicate = replace(pair, sources=tuple(replace(src) for src in pair.sources), draw_ids=('5', '6'))
    step = replace(step, eta=1e-6)
    batch = build_reference((pair, duplicate), [.5, .5], [.5, .5], backend, start, source, step)
    K = gram(batch.directions, step.diagonal)
    scale = np.sqrt(K[0, 0])
    gradient = {n: .2*v.cpu().double()/step.diagonal[n].cpu().double()/scale
                for n, v in batch.directions[0].items()}
    stat = FeedbackStatistic(gradient, (gradient, gradient), (1., 1.), ('task', 'task'),
                             tensor_state_hash(batch.theta0), 1, 'smoke_zero')
    z, error = stat.project(batch.directions, 'loo')
    d, result = control(z, error, K, batch.alpha)
    assert K.max() < 1e-10
    np.testing.assert_allclose(d, [.2, 0.], atol=1e-10)
    np.testing.assert_allclose(result['independent_d'], [.2, .2], atol=1e-10)
    assert result['joint_minus_independent_control'] == pytest.approx(.02)
    assert any(not torch.equal(batch.selected(d)[n], batch.selected(result['independent_d'])[n]) for n in start)
    _, legacy = control(z, error, K, batch.alpha, d_lambda_normalisation='none')
    assert legacy['joint_d'] == legacy['independent_d'] == [.5, .5]
    assert legacy['K_scale'] == legacy['lambda_effective'] == 1.


@pytest.mark.parametrize('factor', [1., 1e-6])
def test_forced_zero_teacher_value_survives_with_default_lambda_and_shared_set_scale(factor):
    p, b, old, start, step, fb, stat = numeric_problem(
        baseline=((-factor, 0.), (factor, 0.)), directions=((factor, 0.), (0., 2*factor)), gradient=(1., 0.))
    values, diagnostic = marginal_values(p, b, old, start, start, step, fb, stat,
                                         revealed_ids={'q1', 'q2'}, zero=True)
    s = 2.5*factor**2
    assert values['q1'] == pytest.approx(.5/np.sqrt(2.5)+.05)
    assert values['q2'] == pytest.approx(-.5/np.sqrt(2.5)+.05)
    for row in (diagnostic, diagnostic['full_set'], *diagnostic['leave_one_out'].values()):
        assert row['K_scale'] == pytest.approx(s, rel=1e-12, abs=0)
    assert diagnostic['full_set']['d_star'] == [0., 0.]
    assert diagnostic['leave_one_out']['q1']['full_update_quadratic_cost'] == pytest.approx(.05)


@pytest.mark.parametrize('normalisation', ['mean_diagonal', 'none'])
def test_full_update_matches_expanded_qp_and_same_batch_control(normalisation):
    p, batch, old, start, step, fb, stat = numeric_problem(
        baseline=((-1e-6, 0.), (0., -1e-6)), directions=((2e-6, 0.), (.6e-6, .8e-6)), gradient=(.6, .5))
    options = dict(d_lambda_normalisation=normalisation, tolerance=1e-12)
    value, d, diagnostic = set_solution(batch.baseline_gradients, batch.directions, batch.weights,
        batch.alpha, start, start, step, fb, stat, **options)
    s = diagnostic['K_scale']
    # Independently expand the full displacement into the normalised QP.
    v = np.stack([direction['x'].numpy() for direction in batch.directions])
    delta0 = batch.theta0['x'].detach().numpy()
    expanded_z = v @ fb.gradient['x'].numpy()/np.sqrt(s)-(v @ delta0)/s
    expanded, _ = solve_d(expanded_z, [0., 0.], v @ v.T/s, batch.alpha,
                          d_lambda_normalisation='none', tolerance=1e-12)
    np.testing.assert_allclose(d, expanded, atol=1e-12, rtol=0)
    delta = batch.selected(d)['x'].detach().numpy()
    expected = np.dot(fb.gradient['x'].numpy(), delta)/np.sqrt(s)-np.dot(delta, delta)/(2*s)
    assert value == pytest.approx(expected)
    assert diagnostic['full_update_quadratic_cost'] == pytest.approx(np.dot(delta, delta)/(2*s))
    if normalisation == 'none':
        assert s == 1.
    fb = replace(fb, parameter_hash=tensor_state_hash(batch.theta0))
    stat = replace(stat, parameter_hash=fb.parameter_hash)
    value, d, _ = set_solution(batch.baseline_gradients, batch.directions, batch.weights,
        batch.alpha, start, batch.theta0, step, fb, stat, **options)
    expected, result = control(*stat.project(batch.directions, 'loo'), gram(batch.directions, step.diagonal),
                               batch.alpha, **options)
    np.testing.assert_array_equal(d, expected)
    assert value == pytest.approx(result['predicted_joint_gain'])


def test_window_calibration_survives_resume_and_is_shared_with_acquisition(toy_bank, tmp_path):
    root = tmp_path/'window'
    e = engine(root, toy_bank, smoke=False, d_warmup_windows=0)
    e.round_start()
    finish_step(e)
    first = e.state['steps'][0]
    calibration = e.state['d_calibration']
    e = engine(root, toy_bank, smoke=False, d_warmup_windows=0, resume=True)
    assert e.state['d_calibration'] == calibration
    for _ in range(3):
        finish_step(e)
    rows = e.state['steps']
    assert [r['alpha_d']['K_scale'] for r in rows[:3]] == [calibration.scale]*3
    assert rows[1]['alpha_d']['K'] != first['alpha_d']['K']
    assert rows[1]['alpha_d']['epsilon_reprojected']
    assert rows[0]['acquisition_surrogate'] is not None
    for row in (rows[0], rows[3]):
        ctrl = row['alpha_d']
        assert ctrl['K_scale'] == calibrate_d(ctrl['K']).scale
        surrogate = row['acquisition_surrogate']
        if surrogate is None:  # No new purchases remain in the tiny bank.
            assert not row['selected']
            continue
        assert surrogate['K_scale'] == ctrl['K_scale']
        for candidate in (surrogate['full_set'], *surrogate['leave_one_out'].values()):
            assert candidate['K_scale'] == ctrl['K_scale']
            assert candidate['lambda_effective'] == ctrl['lambda_effective']
        for q, label in row['labels'].items():
            assert label['variance'] == pytest.approx(row['independent_control_labels'][q]['variance']/ctrl['K_scale'])
        for validation in row['realised_paired_gain_validation']:
            assert validation['prediction'] == pytest.approx(validation['surrogate_prediction']*np.sqrt(ctrl['K_scale']))


def test_normalisation_config_default_none_and_invalid_values():
    from bfas.rtd.alpha_d import ALPHA_D_DEFAULTS, validate_config
    assert ALPHA_D_DEFAULTS['d_lambda'] == 1.
    assert ALPHA_D_DEFAULTS['d_lambda_normalisation'] == 'mean_diagonal'
    config = ALPHA_D_DEFAULTS | dict(protocol_version='1.1.0')
    validate_config(config | dict(d_lambda_normalisation='none'))
    with pytest.raises(ValueError, match='d_lambda_normalisation'):
        validate_config(config | dict(d_lambda_normalisation='per_coordinate'))
