"""D8 rev 3.1 CPU math, real D1 components, ordering and recovery oracles."""
from dataclasses import replace
import itertools
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from bfas.behavior.deltas import tensor_state_hash
from bfas.rtd.alpha_d import (ALPHA_D_DEFAULTS, BatchReference, ExposureRecord, FeedbackStatistic,
    SourcePair, blocked_alpha_vjp, build_reference, components, estimate, gram, gram_statistics,
    injection, solve_d, state_features, target_distribution, target_weights, validate_config, validate_arm)
from bfas.rtd.cli import load_config
from bfas.rtd.functional_step import FrozenStep, gradients, lora_parameters, snapshot
from bfas.rtd.return_gradient import ReturnGradient, reinforce_gradient
from bfas.rtd.selector import StudentSnapshot
from test_rtd_checks import experiment, toy_bank
from test_rtd_v11_source_estimator import DT, enumerable, problem
from test_rtd_v11_posterior import batch_config
from test_rtd_manifest_tolerance import manifest_inputs


@pytest.mark.parametrize('a', [0., .1, .5, .9, 1.])
def test_probability_distribution_and_teacher_mass_independent_of_legal_d(a):
    nu = torch.tensor([.2, .3, .5], dtype=DT)
    for d in np.linspace(-min(a, 1-a), min(a, 1-a), 13):
        w = target_weights(torch.tensor(a, dtype=DT), torch.tensor(d, dtype=DT))
        assert float(w[-1]) == a
        torch.testing.assert_close(w.sum(), nu.new_tensor(1.))
        for y1, y2 in itertools.product(range(3), repeat=2):
            q = target_distribution(y1, y2, nu, a, d)
            assert (q >= 0).all()
            torch.testing.assert_close(q.sum(), nu.new_tensor(1.))
    with pytest.raises(ValueError, match='illegal'):
        target_weights(a, min(a, 1-a)+.01)


def test_state_gate_has_no_source_input_and_scalar_variant():
    _, _, _, targets, _, _ = problem()
    state = targets[0][0].behavior.state
    chi = state_features(state, [1., 2.], dtype=DT).reshape(1, -1)
    phi = torch.tensor([.1, .2, .3], dtype=DT, requires_grad=True)
    torch.testing.assert_close(injection(chi, phi, mode='learned_alpha'), (chi@phi).sigmoid())
    scalar = state_features(state, [999.], scalar=True, dtype=DT).reshape(1, -1)
    torch.testing.assert_close(injection(scalar, phi[:1], mode='learned_alpha'), phi[:1].sigmoid())
    with pytest.raises(TypeError, match='never a source'):
        state_features(targets[0][0], [1.])


def test_enumerable_unbiased_estimator_fixed_alpha_weights_sample_dependent_d():
    b, p, frozen, targets, _, _ = problem()
    rows = enumerable(b, p, frozen, targets[0][0])
    teacher = gradients(-b.score_behavior(targets[0][1], p), p)['lora_transition'].flatten()
    expected = torch.zeros_like(teacher)
    actual = torch.zeros_like(teacher)
    # Different state slots have different PRE-DRAW alpha/w; d uses both actions.
    for alpha, weight in [(.2, .3), (.6, .7)]:
        for i, j in itertools.product(range(len(rows)), repeat=2):
            _, pi, h1, s1 = rows[i]
            _, pj, h2, s2 = rows[j]
            d = min(alpha, 1-alpha)*np.tanh((i-j)/3)
            expected += weight*pi*pj*((1-alpha-d)*h1/2+(1-alpha+d)*h2/2+alpha*teacher)
            actual += weight*pi*pj*estimate({'x': (s1+s2)/2}, {'x': teacher}, {'x': h1}, {'x': h2}, alpha, d)['x']
    torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)


def pair_problem():
    b, p, frozen, targets, _, _ = problem()
    record = ExposureRecord('paid', 0, targets[0][0].behavior.state, targets[0][1])
    pairs = (SourcePair(record, (targets[0][0], targets[1][0]), ('1', '2')),
             SourcePair(record, (replace(targets[1][0]), targets[2][0]), ('3', '4')))
    step = FrozenStep({n: torch.linspace(.5, 1.5, t.numel(), dtype=DT).reshape_as(t) for n, t in p.items()}, .125, 'r1', {})
    return b, p, frozen, pairs, step


def test_soft_zero_per_prefix_does_not_imply_whole_update_zero():
    b, _, frozen, pairs, _ = pair_problem()
    soft, teacher, h1, h2 = components(pairs[0], b, frozen, frozen)
    assert all(torch.count_nonzero(g) == 0 for g in soft.values())
    zero = estimate(soft, teacher, h1, h2, 0., 0.)
    assert all(torch.count_nonzero(g) == 0 for g in zero.values())
    for d in (0., .2, -.2):
        full = estimate(soft, teacher, h1, h2, .5, d)
        assert any(torch.count_nonzero(g) > 0 for g in full.values())


def test_same_batch_reference_equals_weighted_estimator_and_affine_direction():
    b, p, frozen, pairs, step = pair_problem()
    w = torch.tensor([.25, .75], dtype=DT)
    a = torch.tensor([.25, .5], dtype=DT, requires_grad=True)
    ref = build_reference(pairs, w, a, b, p, frozen, step)
    d = torch.tensor([.125, -.25], dtype=DT, requires_grad=True)
    expected = {n: torch.zeros_like(t) for n, t in p.items()}
    for pair, weight, alpha, di in zip(pairs, w, a, d):
        gi = estimate(*components(pair, b, p, frozen), alpha, di)
        assert all(not g.requires_grad for g in gi.values())
        for n in expected:
            expected[n] += weight*gi[n]
    actual = ref.selected(d)
    oracle = step.update(p, expected)
    for n in p:
        torch.testing.assert_close(actual[n], oracle[n], atol=2e-15, rtol=2e-15)
        delta = sum(di.detach()*v[n] for di, v in zip(d, ref.directions))
        torch.testing.assert_close(actual[n]-ref.theta0[n], delta, atol=2e-16, rtol=2e-15)
    assert torch.autograd.grad(sum(v.sum() for v in actual.values()), (a, d), allow_unused=True) == (None, None)
    with pytest.raises(ValueError, match='sum to one'):
        build_reference(pairs, [1., 1.], a, b, p, frozen, step)


def test_affine_identity_bit_exact_on_dyadic_tiny_model():
    # Exactly representable values make this an equality, not a tolerance test.
    ref = BatchReference('start', {'x': torch.tensor([1., 2.], dtype=DT)},
        ({'x': torch.tensor([.5, .25], dtype=DT)}, {'x': torch.tensor([.25, -.5], dtype=DT)}),
        torch.tensor([.5, .5], dtype=DT), torch.tensor([.5, .5], dtype=DT), ())
    d = torch.tensor([.25, -.5], dtype=DT)
    assert torch.equal(ref.selected(d)['x']-ref.theta0['x'], sum(di*v['x'] for di, v in zip(d, ref.directions)))


def test_joint_solver_uses_compensating_coordinate_below_raw_error_threshold():
    K = np.array([[1., .9], [.9, 1.]])
    z, error = np.array([1., 0.]), np.array([.1, .1])
    d, meta = solve_d(z, error, K, [.5, .5], tolerance=1e-12)
    assert abs(z[1]) <= error[1] and d[1] < -.3
    assert meta['converged'] and meta['proximal_residual'] <= 1e-12
    assert d[0] == .5
    # Interior nonzero coordinate's conditional score equals its L1 threshold.
    assert z[1]-(K@d)[1] == pytest.approx(-error[1])
    assert meta['objective'] > .325  # raw-z screening would keep d2=0.


def test_solver_interior_zero_condition_and_convergence_failure():
    d, meta = solve_d([.05, 1.], [.2, .1], np.eye(2), [.4, .4])
    assert d[0] == 0 and abs(.05-0*d[1]) <= .2 and meta['converged']
    with pytest.raises(RuntimeError, match='did not converge'):
        solve_d([.1, .11], [0., 0.], [[1., .9], [.9, 1.]], [.5, .5], max_iterations=1, tolerance=1e-15)
    with pytest.raises(ValueError, match='positive semidefinite'):
        solve_d([1., 1.], [0., 0.], [[1., 2.], [2., 1.]], [.5, .5])
    d, _ = solve_d([1., -1., 0.], [0., 0., 0.], np.zeros((3, 3)), [0., .2, 1.], d_lambda=0.)
    np.testing.assert_array_equal(d, [0., -.2, 0.])


def test_symmetry_swap_one_source_pair_flips_d_preserves_commit():
    b, p, frozen, pairs, step = pair_problem()
    ref = build_reference(pairs, [.4, .6], [.5, .5], b, p, frozen, step)
    swapped = (replace(pairs[0], sources=tuple(reversed(pairs[0].sources)), draw_ids=tuple(reversed(pairs[0].draw_ids))), pairs[1])
    other = build_reference(swapped, [.4, .6], [.5, .5], b, p, frozen, step)
    K = gram(ref.directions, step.diagonal)
    sign = np.diag([-1., 1.])
    np.testing.assert_allclose(gram(other.directions, step.diagonal), sign@K@sign, atol=1e-15)
    z, error = np.array([.006, -.004]), np.array([.0001, .0002])
    d, _ = solve_d(z, error, K, [.5, .5], tolerance=1e-12)
    ds, _ = solve_d(sign@z, error, sign@K@sign, [.5, .5], tolerance=1e-12)
    np.testing.assert_allclose(ds, sign@d, atol=1e-12)
    for n in p:
        torch.testing.assert_close(ref.selected(d)[n], other.selected(ds)[n], rtol=0, atol=0)


@pytest.mark.parametrize('mode', ['loo', 'split_half'])
def test_uncertainty_reprojects_new_directions_not_old_scalar_error(mode):
    scores = tuple({'x': torch.tensor(x, dtype=DT)} for x in ([1., 0.], [2., 4.], [-2., 1.], [1., -2.]))
    rewards = (1., 0., 1., 0.)
    advantages = np.array(rewards)-(sum(rewards)-np.array(rewards))/3
    gradient = {'x': sum(g['x']*a/4 for g, a in zip(scores, advantages))}
    fb = FeedbackStatistic(gradient, scores, rewards, ('task',)*4, 'theta0', 1, 'leave_one_out_same_task')
    v = ({'x': torch.tensor([1., 0.], dtype=DT)},)
    z, e = fb.project(v, mode)
    z3, e3 = fb.project(({'x': 3*v[0]['x']},), mode)
    np.testing.assert_allclose(z3, 3*z)
    np.testing.assert_allclose(e3, 3*e)
    _, e2 = fb.project(({'x': torch.tensor([0., 1.], dtype=DT)},), mode)
    assert e[0] > 0 and e2[0] != e[0]
    assert 'staleness bias' in FeedbackStatistic.__doc__


@pytest.mark.parametrize('scalar', [False, True])
def test_blocked_alpha_vjp_matches_nonlinear_finite_difference_with_fixed_d(scalar):
    b, p, frozen, pairs, step = pair_problem()
    chi = torch.ones(2, 1, dtype=DT) if scalar else torch.tensor([[.2, 1.], [-.4, 1.]], dtype=DT)
    phi = torch.full((chi.shape[1],), .2, dtype=DT, requires_grad=True)
    d = np.array([.05, -.1])
    def build(control):
        return build_reference(pairs, [.3, .7], injection(chi, control, mode='learned_alpha'), b, p, frozen, step)
    ref = build(phi)
    updated = ref.selected(d)
    def reward(params):
        return sum(t.sin().sum() for t in params.values())
    fb = ReturnGradient(gradients(reward(updated), updated), tensor_state_hash(updated), {})
    vjp = blocked_alpha_vjp(pairs, chi, phi, ref, b, p, frozen, step, fb, mode='learned_alpha')
    finite = []
    for i in range(phi.numel()):
        delta = torch.zeros_like(phi); delta[i] = 1e-5
        finite.append((reward(build(phi+delta).selected(d))-reward(build(phi-delta).selected(d)))/2e-5)
    torch.testing.assert_close(vjp, torch.stack(finite), atol=1e-9, rtol=1e-6)


def engine(path, bank, *, resume=False, smoke=True, **options):
    arm = options.pop('arm', None)
    c = batch_config(rounds=2, exposure_slots_per_window=40, slots_per_step=4, source_estimator='alpha_d',
        gate='linear_sigmoid', **ALPHA_D_DEFAULTS)
    c.update(options)
    from bfas.rtd.conventions import arm_config
    c, arm = arm_config(c, arm or ('V0' if c['gate_mode'] == 'fixed_alpha' else 'V2'))
    e = experiment(path, bank, config=c, smoke=smoke, resume=resume, arm=arm)
    model = e.backend.model
    model.head = torch.nn.Identity()
    model.get_output_embeddings = lambda: model.head
    original = model.forward
    def forward(*a, **kw):
        out = original(*a, **kw); out.logits = model.head(out.logits); return out
    model.forward = forward
    return e


def finish_step(e):
    e.step_start(); e.reference()
    while e.state['phase'] == 'selected':
        e.selected()
    e.revealed(); e.actual(); e.feedback_commit(); e.committed()


def test_runner_acquire_freeze_sample_same_batch_commit_then_alpha_feedback(toy_bank, tmp_path):
    e = engine(tmp_path/'run', toy_bank)
    e.run()
    events = e.journal.events
    ends = [ev for ev in events if ev['kind'] == 'compute_end']
    for ev in ends:
        assert ev['operation'] == events[ev['begin_sequence']]['operation']
    operations = {ev['operation'] for ev in ends}
    assert {'alpha_d_pilot', 'commit', 'acquisition_reference_feedback',
            'same_batch_reference_feedback', 'post_commit_feedback'} <= operations
    validation_roles = {ev['role'] for ev in events if ev['kind'] == 'validation_rollout'}
    assert len(validation_roles) == 4 and validation_roles <= operations
    def index(kind, **fields):
        return next(i for i, ev in enumerate(events) if ev['kind'] == kind and all(ev.get(k) == v for k, v in fields.items()))
    assert index('decision') < index('request_reveal_link') < index('alpha_d_exposure_frozen', role='commit')
    assert index('alpha_d_exposure_frozen', role='commit') < index('alpha_d_source_pair', role='commit')
    assert index('alpha_d_reference') < index('return_gradient', role='same_batch_reference_feedback')
    assert index('alpha_d_solver') < index('alpha_d_student_commit') < index('feedback_rollout', role='post_commit_feedback')
    assert index('alpha_d_gate_update') < index('alpha_d_acquisition_reference')
    row = e.state['steps'][0]
    assert row['slots'] == 4 and row['source_actions'] == 8 and row['exposure_mode'] == 'random exposure'
    assert row['exposure_units'] == 4 and row['teacher_evidence_units'] == 2 and row['reference_pool_units'] == 2
    assert row['alpha_d']['warmup'] and row['alpha_d']['d_star'] == [0.]*4
    assert not {'alpha_pairs', 'd_reference', 'd_solution'} & e.state.keys()
    assert len(e.state['d_feedback'].scores) == 2
    # Persisted targets are plain source/teacher records, without signed CE.
    assert all('negative_weight' not in r and 'loss' not in r for r in row['exposure_records'])


def test_full_40_units_80_actions_ten_microbatches_and_fresh_nondecision_feedback(toy_bank, tmp_path):
    e = engine(tmp_path/'run', toy_bank, smoke=False, slots_per_step=40, d_warmup_windows=0)
    e.round_start()
    s = e.state
    view = StudentSnapshot(s['source_id'], frozenset(s['inner']))
    offered = e.broker.list_candidates(view, e.ledger.owned_ids, e.ledger.remaining)
    package = e.broker.acquire(offered[0].query_id)
    s['owned'].append(package.query_id)
    e.calibrate([package])
    finish_step(e)
    first = s['steps'][0]
    assert first['exposure_units'] == 40 and first['source_actions'] == 80
    assert first['microbatches'] == 10
    assert first['teacher_evidence_units']+first['reference_pool_units'] == 40
    assert first['raw_new_slots'] <= 20 and first['raw_old_slots']+first['raw_new_slots'] == 40
    feedback = s['d_feedback']
    old_draws = set(s['last_committed_draw_ids'])
    finish_step(e)
    second = s['steps'][1]
    assert second['alpha_d']['feedback_age'] == 1 and second['alpha_d']['epsilon_reprojected']
    assert feedback is s['d_feedback'] and not old_draws.intersection(s['last_committed_draw_ids'])
    assert second['alpha_d']['K'] != first['alpha_d']['K']
    assert not any(ev['kind'] == 'return_gradient' and ev['step'] == 2 for ev in e.journal.events)
    # Ten complete state microbatches precede each of the two backbone commits.
    assert len([ev for ev in e.journal.events if ev['kind'] == 'alpha_d_microbatch']) == 20


@pytest.mark.parametrize('phase,paid', [('reference', 0), ('selected', 1), ('revealed', 2),
                                      ('actual', 2), ('feedback', 2), ('committed', 2)])
def test_resume_preserves_same_draws_single_commit_and_posterior(toy_bank, tmp_path, phase, paid):
    clean = engine(tmp_path/'clean', toy_bank); clean.run()
    broken = engine(tmp_path/'broken', toy_bank)
    def stop(current):
        if current == phase and len(broken.ledger.owned_ids) == paid:
            raise RuntimeError('D8 crash probe')
    broken.after_save = stop
    with pytest.raises(RuntimeError, match='D8 crash probe'):
        broken.run()
    restored = engine(tmp_path/'broken', toy_bank, resume=True); restored.run()
    assert restored.state['steps'] == clean.state['steps']
    assert tensor_state_hash(restored.state['parameters']) == tensor_state_hash(clean.state['parameters'])
    assert restored.ledger.charges == clean.ledger.charges
    np.testing.assert_array_equal(restored.state['posterior'].model.precision, clean.state['posterior'].model.precision)


def test_configuration_defaults_v0_and_rejection(tmp_path):
    path = Path(__file__).resolve().parents[1]/'configs/rtd/v1_1_bfcl.yaml'
    c = load_config(path)
    assert c['gate_mode'] == 'learned_alpha' and c['d_mode'] == 'learned'
    assert c['exposure_slots_per_window'] == c['slots_per_step'] == 40
    c.update(gate_mode='fixed_alpha', fixed_alpha=.5, d_mode='zero')
    validate_config(c)
    for key, bad in [('source_samples_per_state', 3), ('microbatch_states', 2), ('d_lambda', -1.),
                     ('d_warmup_windows', -1), ('z_error_mode', 'old_scalars'), ('exposure_slots_per_window', 20)]:
        with pytest.raises(ValueError):
            validate_config(c | {key: bad})
    c['slots_per_step'] = 4
    file = tmp_path/'random.yaml'; file.write_text(yaml.safe_dump(c))
    assert load_config(file)['slots_per_step'] == 4


def test_redundancy_statistics_and_source_cache_rejection():
    stats = gram_statistics(np.array([[1., .95, 0.], [.95, 1., 0.], [0., 0., 0.]]), .9)
    assert stats['pair_count'] == 3 and stats['cosine_above_threshold_fraction'] == pytest.approx(1/3)
    _, _, _, pairs, _ = pair_problem()
    with pytest.raises(ValueError, match='cache reuse'):
        replace(pairs[0], sources=(pairs[0].sources[0],)*2)


def test_full_exposure_empty_paid_pool_uses_reference_states_without_teacher(toy_bank, tmp_path):
    e = engine(tmp_path/'run', toy_bank, slots_per_step=40)
    e.round_start(); e.step_start(); e.reference()
    while e.state['phase'] == 'selected':
        e.selected()
    e.revealed(); e.actual(); e.feedback_commit()
    row = e.state['steps'][0]
    assert row['exposure_units'] == 40 and row['source_actions'] == 80
    assert row['reference_pool_units'] == 40-len(row['selected'])
    assert all(r['alpha'] == 0 for r in row['exposure_records'] if r['teacher'] is None)


def test_random_exposure_records_untrained_purchased_packages(toy_bank, tmp_path):
    e = engine(tmp_path/'run', toy_bank, slots_per_step=1)
    e.run()
    row = e.state['steps'][0]
    assert row['exposure_mode'] == 'random exposure' and row['exposure_units'] == 1
    assert len(row['selected']) == 2 and len(row['unexposed_purchased_packages']) == 1
    assert set(row['selected']) == e.ledger.owned_ids
    assert set(row['trained_new_packages']) == set(row['labels'])


def test_manifest_labels_full_and_random_exposure_and_arm_contract(manifest_inputs):
    from bfas.rtd import cli
    fixture = manifest_inputs
    config = fixture.config | dict(protocol_version='1.1.0', rounds=2, exposure_slots_per_window=40,
        max_new_packages_per_window=20, slots_per_step=4, source_estimator='alpha_d') | ALPHA_D_DEFAULTS
    audit = fixture.audit | dict(budget_denominator=100, cost_scope='cached content',
                                cap_certificate_sha256='certificate', public_cost_assumption='class cap')
    manifest = cli.make_manifest(config, 'V2', audit)
    assert manifest['exposure_mode'] == 'random exposure'
    assert not manifest['all_purchased_packages_trained_each_step']
    full = cli.make_manifest(config | {'slots_per_step': 40}, 'V2', audit)
    assert full['exposure_mode'] == 'full exposure' and full['source_action_capacity_per_commit'] == 80
    assert full['microbatches_per_commit'] == 10
    validate_arm(config, 'V2')
    with pytest.raises(ValueError, match='replay schedule'):
        validate_arm(config | {'acquisition': 'random'}, 'V1')
    with pytest.raises(ValueError, match='V0 requires'):
        validate_arm(config, 'V0')
    v0 = config | dict(gate_mode='fixed_alpha', fixed_alpha=.5, d_mode='zero', acquisition_value_mode='independent')
    validate_arm(v0 | {'acquisition': 'random'}, 'V0')
    with pytest.raises(ValueError, match='V1/V2 require'):
        validate_arm(v0, 'V1')


def test_v0_alpha_stays_point_five_and_d_zero_with_feedback(toy_bank, tmp_path):
    e = engine(tmp_path/'v0', toy_bank, gate_mode='fixed_alpha', d_mode='zero', d_warmup_windows=0)
    e.manifest['arm'] = 'V0'
    e.run()
    row = e.state['steps'][0]
    assert row['alpha_d']['alpha'] == [.5 if r['teacher'] else 0. for r in row['exposure_records']]
    assert row['alpha_d']['d_star'] == [0.]*row['slots']
    assert not any(ev['kind'] == 'alpha_d_gate_update' for ev in e.journal.events)


def test_feedback_contributions_reconstruct_existing_return_gradient(toy_bank, tmp_path):
    e = engine(tmp_path/'feedback', toy_bank, smoke=False)
    e.round_start()
    s = e.state
    s['feedback_tasks'] = e.choose_feedback_tasks()
    scores = []
    feedback = e.feedback(s['parameters'], 'test_trajectory_contributions', trajectory_scores=scores)
    rewards = np.array(feedback.metadata['rewards'])
    advantages = rewards-np.array(feedback.metadata['baselines'])
    for n in s['parameters']:
        expected = sum(g[n]*a/len(scores) for g, a in zip(scores, advantages))
        torch.testing.assert_close(expected, feedback.gradient[n], atol=2e-15, rtol=2e-15)
    assert all(not g.requires_grad and g.device.type == 'cpu' for score in scores for g in score.values())


def test_adapter_episode_mapping_and_pre_sampling_freeze(toy_bank, tmp_path):
    e = engine(tmp_path/'mapping', toy_bank)
    e.round_start()
    specs = e.broker.list_candidates(StudentSnapshot(e.state['source_id'], frozenset(e.state['inner'])),
                                    e.ledger.owned_ids, e.ledger.remaining)
    package = e.broker.acquire(specs[0].query_id)
    # The adapter, rather than an implicit episode-as-one-action convention,
    # determines the ordered supervision records.
    e.support.supervision_records = lambda p: tuple(reversed(p.behaviors))
    records = tuple(e.supervision_records(package))
    assert [r.teacher for r in records] == list(reversed(package.behaviors))
    chi, a, w = e.alpha_freeze(records, role='test')
    frozen = (chi.clone(), a.clone(), w.clone())
    e.alpha_draw_pairs(records, role='test')
    for actual, expected in zip((chi, a, w), frozen):
        assert torch.equal(actual, expected)


def test_warmup_ends_at_next_decision_window_and_feedback_age_resets(toy_bank, tmp_path):
    e = engine(tmp_path/'warmup', toy_bank, smoke=False, d_warmup_windows=1, d_lambda=0.)
    e.round_start()
    for _ in range(4):
        finish_step(e)
    rows = e.state['steps']
    assert [r['alpha_d']['warmup'] for r in rows] == [True, True, True, False]
    assert [r['alpha_d']['feedback_age'] for r in rows] == [0, 1, 2, 0]
    assert rows[-1]['alpha_d']['solver']['converged']
    assert len([ev for ev in e.journal.events if ev['kind'] == 'return_gradient' and
                ev['role'] == 'same_batch_reference_feedback']) == 2
