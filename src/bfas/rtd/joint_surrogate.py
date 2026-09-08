"""Paid-set acquisition proxy around the shared alpha/d update.

Deleting a package replaces its unit with a pre-drawn old-pool record, retaining
all exposure weights. No unpurchased teacher gradient or candidate rollout is
needed. This is a local surrogate, not a measured paired task improvement.
"""
import numpy as np
import torch

from .alpha_d import dot, gram, solve_d


def set_value(baseline, directions, weights, alpha, start, reference, step, feedback,
              statistic, *, d_lambda=1., zero=False, error_mode='loo'):
    delta = {n: (p.detach()-reference[n].detach()).cpu().double() - sum(
        (step.eta*step.diagonal[n].detach().cpu().double()*g[n].cpu().double()*float(w)
         for g, w in zip(baseline, weights)), torch.zeros_like(p, device='cpu', dtype=torch.float64))
        for n, p in start.items()}
    z, error = statistic.project(directions, error_mode)
    K = gram(directions, step.diagonal)
    cross = np.array([sum(float((delta[n]*v[n].cpu().double()/step.diagonal[n].detach().cpu().double()).sum())
                          for n in delta) for v in directions])
    if zero or not np.isfinite(error).all():
        d = np.zeros(len(directions))
    else:
        d, _ = solve_d(z-d_lambda*cross, error, K, alpha, d_lambda=d_lambda)
    update = {n: delta[n]+sum((float(di)*v[n].cpu().double() for di, v in zip(d, directions)),
                             torch.zeros_like(delta[n])) for n in delta}
    cost = sum(float((u.square()/step.diagonal[n].detach().cpu().double()).sum()) for n, u in update.items())
    penalty = float(error @ np.abs(d)) if np.isfinite(error).all() else 0.
    return dot(feedback.gradient, update)-d_lambda*cost/2-penalty


def marginal_values(pairs, batch, old, start, reference, step, feedback, statistic, **options):
    if feedback.parameter_hash != statistic.parameter_hash:
        raise ValueError('joint surrogate requires the common virtual-reference feedback')
    weights = batch.weights
    if len(old.baseline_gradients) != len(pairs):
        raise ValueError('one predetermined old-pool replacement per unit required')
    def value(baseline, directions, alpha):
        return set_value(baseline, directions, weights, alpha, start, reference, step,
                         feedback, statistic, **options)
    full = value(batch.baseline_gradients, batch.directions, batch.alpha)
    values = {}
    for q in sorted({p.record.query_id for p in pairs if p.record.is_new}):
        baseline, directions, alpha = list(batch.baseline_gradients), list(batch.directions), batch.alpha.clone()
        for i, pair in enumerate(pairs):
            if pair.record.is_new and pair.record.query_id == q:
                baseline[i] = old.baseline_gradients[i]
                # Replacements retain this unit's weight, including V1 replay.
                directions[i] = {n: v*(float(weights[i])/float(old.weights[i]))
                                 for n, v in old.directions[i].items()} if old.weights[i] else {
                                     n: torch.zeros_like(v) for n, v in old.directions[i].items()}
                alpha[i] = old.alpha[i]
        values[q] = full-value(baseline, directions, alpha)
    return values, dict(full_set_value=full, marginal_values=values,
        approximation='joint_full_update_surrogate_with_teacher_injection',
        replacement='predetermined_old_pool_record; same_exposure_weight',
        reference_hash=feedback.parameter_hash, paired_task_evaluation=False)
