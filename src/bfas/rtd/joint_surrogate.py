"""Paid-set acquisition proxy around the shared alpha/d update.

Deleting a package replaces its unit with a pre-drawn old-pool record, retaining
all exposure weights. No unpurchased teacher gradient or candidate rollout is
needed. This is a local surrogate, not a measured paired task improvement.
"""
from dataclasses import replace

import numpy as np
import torch

from ..behavior.deltas import tensor_state_hash
from .alpha_d import dot, gram, gram_statistics, solve_d, target_weights
from .functional_step import snapshot

ACQUISITION_SURROGATE = 'acquisition_surrogate_full_update_with_teacher_injection'


def objective(z, error, K, d, d_lambda):
    return float(z @ d-d_lambda/2*(d @ K @ d)-error @ np.abs(d))


def control(z, error, K, alpha, *, mode='joint', d_lambda=1., zero_reason=None,
            tolerance=1e-8, max_iterations=10000, redundancy_threshold=.9):
    """Only d is optimized. Independent control drops off-diagonal curvature.

    Both arms see all coordinates; no screening by raw z is performed. Always
    evaluate both solutions under the TRUE joint objective for a fair control.
    """
    if mode not in {'joint', 'independent'}:
        raise ValueError('controller mode must be joint or independent')
    z, error, K, alpha = (np.asarray(x, dtype=np.float64) for x in (z, error, K, alpha))
    if zero_reason is None and not np.isfinite(error).all():
        zero_reason = 'insufficient_trajectories'
    if zero_reason:
        joint = independent = np.zeros(len(z))
        solver = dict(converged=None, iterations=0, reason=zero_reason)
        independent_solver = dict(solver)
        # Missing SE is not zero uncertainty; only evaluate the forced no-op.
        joint_gain = independent_gain = additive = 0.
    else:
        options = dict(d_lambda=d_lambda, tolerance=tolerance, max_iterations=max_iterations)
        joint, solver = solve_d(z, error, K, alpha, **options)
        independent, independent_solver = solve_d(z, error, np.diag(np.diag(K)), alpha, **options)
        joint_gain = objective(z, error, K, joint, d_lambda)
        independent_gain = objective(z, error, K, independent, d_lambda)
        additive = objective(z, error, np.diag(np.diag(K)), independent, d_lambda)
    d = joint if mode == 'joint' else independent
    shares = target_weights(torch.from_numpy(alpha), torch.from_numpy(d)).tolist()
    return d, dict(controller=mode, d_star=d.tolist(), joint_d=joint.tolist(), independent_d=independent.tolist(),
        solver=solver if mode == 'joint' else independent_solver,
        joint_solver=solver, independent_solver=independent_solver,
        source_teacher_shares=shares, K=K.tolist(), K_statistics=gram_statistics(K, redundancy_threshold),
        predicted_joint_gain=joint_gain, predicted_independent_joint_gain=independent_gain,
        sum_independent_gains=additive, joint_minus_sum_independent=joint_gain-additive,
        joint_minus_independent_control=joint_gain-independent_gain,
        optimized_variables='d_only', weights_and_alpha_frozen=True, raw_z_prefilter=False)


def set_solution(baseline, directions, weights, alpha, start, reference, step, feedback,
                 statistic, *, d_lambda=1., zero=False, error_mode='loo', mode='joint',
                 tolerance=1e-8, max_iterations=10000):
    """Full displacement, including baseline teacher learning even when d=0.

    Expanding the quadratic gives the identical d QP with effective score
    g^T v - lambda * delta0^T H v and a constant baseline value. When reference
    is this batch's theta0, the constant/cross term vanish: exactly D8 control.
    Acquisition comparisons instead hold their old-evidence reference fixed.
    """
    if (feedback.parameter_hash != tensor_state_hash(reference) or
            statistic.parameter_hash != feedback.parameter_hash or
            not all(torch.equal(feedback.gradient[n].detach().cpu(), statistic.gradient[n].detach().cpu())
                    for n in feedback.gradient)):
        raise ValueError('surrogate requires one common reference point and return gradient')
    if (len(baseline) != len(directions) or len(weights) != len(directions) or len(alpha) != len(directions)
            or not np.isfinite(np.asarray(weights)).all() or np.any(np.asarray(weights) < 0)
            or abs(float(sum(weights))-1) > 1e-7):
        raise ValueError('aligned frozen exposure weights and gradients required')
    # Match the actual one-step accumulation order/precision before subtracting
    # the reference. In particular, theta0-reference is exactly zero when equal.
    total = {n: torch.zeros_like(p) for n, p in start.items()}
    for g, w in zip(baseline, weights):
        for n, p in start.items():
            total[n].add_(g[n].to(p)*float(w))
    theta0 = snapshot(step.update(start, total))
    delta = {n: theta0[n].detach().cpu().double()-reference[n].detach().cpu().double() for n in start}
    z, error = statistic.project(directions, error_mode)
    K = gram(directions, step.diagonal)
    cross = np.array([sum(float((delta[n]*v[n].cpu().double()/step.diagonal[n].detach().cpu().double()).sum())
                          for n in delta) for v in directions])
    effective_z = z-d_lambda*cross
    d, diagnostic = control(effective_z, error, K, alpha, mode=mode, d_lambda=d_lambda,
        zero_reason='configured_zero' if zero else None, tolerance=tolerance, max_iterations=max_iterations)
    update = {n: delta[n]+sum((float(di)*v[n].cpu().double() for di, v in zip(d, directions)),
                             torch.zeros_like(delta[n])) for n in delta}
    cost = sum(float((u.square()/step.diagonal[n].detach().cpu().double()).sum()) for n, u in update.items())
    penalty = float(error @ np.abs(d)) if np.isfinite(error).all() else 0.
    value = dot(feedback.gradient, update)-d_lambda*cost/2-penalty
    diagnostic.update(value=value, baseline_linear_gain=dot(feedback.gradient, delta),
        full_update_quadratic_cost=d_lambda*cost/2, uncertainty_penalty=penalty,
        effective_z=effective_z.tolist(), reference_hash=feedback.parameter_hash,
        theta0_hash=tensor_state_hash(theta0), label_kind='acquisition_surrogate')
    return value, d, diagnostic


def set_value(baseline, directions, weights, alpha, start, reference, step, feedback,
              statistic, **options):
    return set_solution(baseline, directions, weights, alpha, start, reference, step, feedback,
                        statistic, **options)[0]


def replacement_batch(pairs, batch, old, query_id, start, step):
    """Predetermined backfill; never change other units, w, or source draws."""
    if len(old.baseline_gradients) != len(pairs):
        raise ValueError('one predetermined old-pool replacement per unit required')
    baseline, directions, alpha = list(batch.baseline_gradients), list(batch.directions), batch.alpha.clone()
    for i, pair in enumerate(pairs):
        if pair.record.is_new and pair.record.query_id == query_id:
            baseline[i] = old.baseline_gradients[i]
            if old.weights[i] == 0 and batch.weights[i] != 0:
                raise ValueError('replacement direction requires a positive predetermined weight')
            directions[i] = {n: v*(float(batch.weights[i])/float(old.weights[i]))
                             for n, v in old.directions[i].items()} if old.weights[i] else old.directions[i]
            alpha[i] = old.alpha[i]
    total = {n: torch.zeros_like(p) for n, p in start.items()}
    for g, w in zip(baseline, batch.weights):
        for n, p in start.items():
            total[n].add_(g[n].to(p)*float(w))
    return replace(batch, theta0=snapshot(step.update(start, total)), baseline_gradients=tuple(baseline),
                   directions=tuple(directions), alpha=alpha)


def execute_update(batch, d, start, step):
    """Independently execute one frozen finite update for paired validation."""
    if tensor_state_hash(start) != batch.start_hash:
        raise ValueError('paired update must execute from the same frozen start')
    total = {n: torch.zeros_like(p) for n, p in start.items()}
    for g, w in zip(batch.baseline_gradients, batch.weights):
        for n, p in start.items():
            total[n].add_(g[n].to(p)*float(w))
    return replace(batch, theta0=snapshot(step.update(start, total))).selected(d)


def validate_pair(kind, query_id, first, d1, second, d2, prediction, start, step, evaluate, *, check_updates=None):
    """D9/D6 shared execution and validation journal schema. Never fit gains."""
    left = execute_update(first, d1, start, step)
    right = execute_update(second, d2, start, step)
    if check_updates is not None:
        check_updates(left, right)
    a = evaluate(left, f'validation_{kind}_full')
    b = evaluate(right, f'validation_{kind}_control')
    if a['batch_id'] == b['batch_id']:
        raise ValueError('paired validation cannot reuse a feedback batch')
    realised = a['mean_return']-b['mean_return']
    return dict(comparison=kind, query_id=query_id, prediction=prediction, realised_paired_gain=realised,
        realised_minus_predicted=realised-prediction, full=a, control=b,
        independent_update_executions=2, updates_per_arm=1, start_hash=first.start_hash,
        validation_only=True, used_for_posterior=False, selection_feedback_reused=False,
        feedback_split='new_independent_batches_on_feedback_tasks',
        package_selection='first_purchased_id_before_inspecting_values' if query_id else None)


def marginal_values(pairs, batch, old, start, reference, step, feedback, statistic, *, revealed_ids, **options):
    evidence = {p.record.query_id for p in pairs if p.record.teacher is not None}
    if not evidence <= set(revealed_ids):
        raise ValueError('sealed pool: surrogate requires purchased evidence only')
    if feedback.parameter_hash != statistic.parameter_hash:
        raise ValueError('joint surrogate requires the common virtual-reference feedback')
    weights = batch.weights
    if len(old.baseline_gradients) != len(pairs):
        raise ValueError('one predetermined old-pool replacement per unit required')
    def value(candidate):
        return set_solution(candidate.baseline_gradients, candidate.directions, weights, candidate.alpha,
                            start, reference, step, feedback, statistic, **options)
    full, _, full_diagnostic = value(batch)
    values, deleted = {}, {}
    for q in sorted({p.record.query_id for p in pairs if p.record.is_new}):
        reduced = replacement_batch(pairs, batch, old, q, start, step)
        without, _, deleted[q] = value(reduced)
        values[q] = full-without
    return values, dict(full_set_value=full, marginal_values=values,
        full_set=full_diagnostic, leave_one_out=deleted,
        approximation=ACQUISITION_SURROGATE, label_kind='acquisition_surrogate',
        replacement='predetermined_old_pool_record; same_exposure_weight',
        reference_hash=feedback.parameter_hash, paired_task_evaluation=False)
