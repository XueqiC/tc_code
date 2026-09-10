from dataclasses import FrozenInstanceError, replace
from itertools import product

import numpy as np
import pytest

from bfas.rtd.unified import Array, TeachingObjective, checked_psd, solve_exact, build_teaching_problem
from bfas.rtd.unified.solver import auxiliary_constraints
from unified_fixtures import from_directions, with_oracle_feedback


def test_immutable_content_versions_and_frozen_P_identity():
    original = np.array([[1., -.2], [.3, .8]])
    problem = from_directions(original, h=[.3, .4], P=[2., .5])
    identity = problem.identity
    original[:] = 9
    assert problem.identity == identity
    for array in (problem.U, problem.K, problem.a_ref, problem.H, problem.context.theta.values):
        with pytest.raises(ValueError):
            array.numpy().flat[0] = 7
        with pytest.raises(ValueError):
            array.numpy().setflags(write=True)
    with pytest.raises(FrozenInstanceError):
        problem.feedback_version = 'changed'
    with pytest.raises(FrozenInstanceError):
        problem.feedback.rewards = (0., 0.)
    theta = problem.context.theta.tensors()
    theta['lora_weight'].data.add_(1.)
    assert problem.identity == identity
    assert replace(problem, feedback_version='new-version').identity != identity
    assert problem.P_hash and problem.H_hash and problem.source_snapshot_hash
    np.testing.assert_allclose(problem.H.numpy(), [.5, 2.])
    with pytest.raises(ValueError, match='theta/loss'):
        build_teaching_problem(problem.context, problem.evidence,
            [replace(problem.exposure[0], theta_hash='stale')])


def test_psd_full_cross_terms_box_kkt_and_enumerated_optimality():
    problem = from_directions([[1., .8], [.0, .6]], h=[1.5, -.3], epsilon=[.07, .03], a_ref=.35)
    solution = solve_exact(problem)
    assert np.linalg.eigvalsh(problem.K.numpy()).min() >= -1e-14
    assert problem.K.numpy()[0, 1] == pytest.approx(.8)
    problem.check_coefficients(solution.coefficients)
    assert max(solution.stationarity, solution.feasibility, solution.complementarity) <= problem.context.qp_tolerance
    A, b = auxiliary_constraints(problem)
    y = np.r_[solution.x.numpy(), solution.auxiliary.numpy()]
    gradient = TeachingObjective(problem).auxiliary_value_gradient(y)[1]
    np.testing.assert_allclose(gradient+A.T @ solution.multipliers.numpy(), 0., atol=1e-8)
    assert (solution.multipliers.numpy() >= 0).all()
    assert np.max(A @ y-b) <= 1e-12
    objective = TeachingObjective(problem)
    grid_best = max(objective.value(a) for a in product(np.linspace(0, 1, 81), repeat=2))
    assert solution.value >= grid_best-1e-10
    assert objective.value(problem.a_ref) == 0.
    assert objective.regret(problem.a_ref, solution.coefficients) == solution.value


def test_numerical_psd_repair_is_recorded_and_material_negativity_rejected():
    K, repair = checked_psd([[1., 0.], [0., -1e-12]])
    assert np.linalg.eigvalsh(K.numpy()).min() == 0
    assert repair.minimum_eigenvalue == -1e-12 and repair.correction_norm == pytest.approx(1e-12)
    with pytest.raises(ValueError, match='repair refused'):
        checked_psd([[1., 0.], [0., -.01]])
    with pytest.raises(ValueError, match='repair refused'):
        checked_psd([[1., 0.], [.1, 1.]])


def test_cross_coordinate_cancellation_defeats_raw_z_freeze_rule():
    # K=[[1,1],[1,2]], z=[.8,0], eps=[0,.1]. Coordinate 2 is worth moving
    # despite |z2|<=eps2: it cancels some of coordinate 1's geometric cost.
    problem = from_directions([[1., 1.], [0., 1.]], h=[.8, -.8], epsilon=[0., .1])
    result = solve_exact(problem)
    z = problem.U.numpy().T @ problem.h.numpy()
    assert abs(z[1]) <= problem.epsilon.numpy()[1]
    assert result.x.numpy()[1] < -.1
    frozen = result.coefficients.numpy().copy()
    frozen[1] = .5
    assert result.value > TeachingObjective(problem).value(frozen)+.01


def test_source_swap_coefficients_equivariant_and_update_invariant():
    problem = from_directions([[1., .2], [.1, 1.]], h=[.3, -.2], epsilon=[.01, .02])
    e = problem.exposure[0]
    swapped = replace(e, hard=Array.of(e.hard.numpy()[::-1]), source_hashes=e.source_hashes[::-1],
                      sample_hashes=e.sample_hashes[::-1], draw_ids=e.draw_ids[::-1])
    other = with_oracle_feedback(build_teaching_problem(problem.context, problem.evidence, [swapped]), [.3, -.2])
    other = replace(other, epsilon=Array.of(problem.epsilon.numpy()[::-1]))
    first, second = solve_exact(problem), solve_exact(other)
    np.testing.assert_allclose(first.coefficients.numpy()[::-1], second.coefficients.numpy(), atol=1e-8)
    np.testing.assert_allclose(TeachingObjective(problem).increment(first.coefficients),
                               TeachingObjective(other).increment(second.coefficients), atol=1e-8)


@pytest.mark.parametrize('U', [[[1., 1.], [2., 2.]], [[0., 1.], [0., 2.]], [[0., 0.], [0., 0.]]])
def test_rank_deficient_equal_zero_and_teacher_equal_directions(U):
    problem = from_directions(U, h=[1., 1.])
    result = solve_exact(problem)
    assert result.value >= 0
    assert result.stationarity <= 1e-8
    if np.count_nonzero(U) == 0:
        np.testing.assert_array_equal(result.coefficients.numpy(), [.5, .5])
    elif np.array_equal(np.asarray(U)[:, 0], np.asarray(U)[:, 1]):
        # Minimum-displacement initialization/SLSQP respects equal columns.
        assert result.coefficients.numpy()[0] == pytest.approx(result.coefficients.numpy()[1], abs=1e-8)


def test_zero_information_preserves_reference_but_keeps_teacher_teaching():
    problem = from_directions([[1., 1.]], h=[0.])
    result = solve_exact(problem)
    np.testing.assert_array_equal(result.coefficients.numpy(), [.5, .5])
    np.testing.assert_array_equal(TeachingObjective(problem).increment(result.coefficients), [1.])


def test_restricted_scalar_control_cannot_reach_selective_retention_direction():
    problem = from_directions([[1., 1.], [1., -1.]], h=[0., 1.])
    objective = TeachingObjective(problem)
    # Scalar x1=x2 cannot improve coordinate 2; source selection can preserve
    # coordinate 1 while improving coordinate 2. Claim is only for this family.
    scalar = max(objective.value([a, a]) for a in np.linspace(0, 1, 101))
    selected = solve_exact(problem)
    assert scalar == 0 and selected.value > .4
    assert objective.displacement(selected.coefficients)[0] == pytest.approx(0., abs=1e-8)


def test_problem_rejects_mutable_feedback_or_non_gram_penalty():
    problem = from_directions([[1., .8], [0., 1.]])
    with pytest.raises(ValueError, match='immutable feedback'):
        replace(problem, feedback={})
    with pytest.raises(ValueError, match='full teacher-conditioned Gram'):
        replace(problem, K=Array.of(np.eye(2)))


@pytest.mark.parametrize('seed', range(5))
def test_joint_qp_four_sources_kkt_with_general_reference(seed):
    rng = np.random.default_rng(seed)
    problem = from_directions(rng.normal(size=(3, 4)), h=rng.normal(size=3),
        epsilon=rng.uniform(0, .1, size=4), a_ref=.2)
    result = solve_exact(problem)
    assert result.stationarity <= 1e-8 and result.value >= 0


@pytest.mark.parametrize('a_ref', [0., 1.])
def test_zero_curvature_qp_with_reference_at_box_endpoint(a_ref):
    problem = from_directions([[1., -.2]], h=[1.], a_ref=a_ref, regularization=0.)
    result = solve_exact(problem)
    np.testing.assert_allclose(result.coefficients.numpy(), [1., 0.], atol=1e-8)
    assert result.stationarity <= 1e-8


def test_forty_direction_rank_deficient_joint_problem():
    rng = np.random.default_rng(2026)
    problem = from_directions(rng.normal(size=(8, 40)), h=rng.normal(size=8),
                              epsilon=rng.uniform(0, .01, size=40))
    result = solve_exact(problem)
    assert result.stationarity <= 1e-8 and result.feasibility <= 1e-8


def test_zero_student_gradients_still_open_nonzero_teacher_columns():
    base = from_directions([[0., 0.], [0., 0.]])
    teacher = replace(base.evidence[0], gradient=Array.of([1., 2.]))
    problem = build_teaching_problem(base.context, [teacher], base.exposure)
    np.testing.assert_array_equal(problem.U.numpy(), [[-.5, -.5], [-1., -1.]])
    problem = with_oracle_feedback(problem, [-1., -2.])
    chosen = solve_exact(problem)
    assert chosen.coefficients.numpy().sum() > 1.


def test_same_state_hash_different_parent_never_aliases_teacher_permission():
    base = from_directions([[1., .2]])
    first = replace(base.exposure[0], weight=.5)
    other = replace(first, slot_id='other', parent_hash='another-inner', draw_ids=('other0', 'other1'))
    context = replace(base.context, inner_parent_hashes=('inner', 'another-inner'))
    problem = build_teaching_problem(context, base.evidence, (first, other))
    assert len(problem.coordinates) == 2
    assert all(c.slot_id == first.slot_id for c in problem.coordinates)


def test_overflow_cannot_masquerade_as_a_successful_qp_certificate():
    problem = from_directions([[10., 10.]])
    problem = replace(problem, h=Array.of([1e308]))
    with np.errstate(over='ignore'), pytest.raises(ValueError, match='overflowed'):
        solve_exact(problem)
