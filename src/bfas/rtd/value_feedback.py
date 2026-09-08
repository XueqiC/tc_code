"""Paid-only insertion uncertainty from shared full-task rollout batches.

Directional scores are streamed (one action gradient at a time); only a small
rollout-by-package matrix is retained. The bootstrap recomputes the same-task
LOO baseline, retaining dependence between labels from the common feedback.
"""
import numpy as np

from .functional_step import gradients
from ..behavior.deltas import tensor_state_hash


def require_paid(ids, revealed_ids):
    if not set(ids) <= set(revealed_ids):
        raise ValueError('value diagnostics require purchased or charged calibration packages')


def insertion_statistics(rollouts, backend, reference, package_gradients, *, revealed_ids,
                         baseline='leave_one_out_same_task', noise_floor=1e-8, prior_noise=1., seed=0):
    require_paid(package_gradients, revealed_ids)
    if not package_gradients:
        return dict(query_ids=[], values=[], covariance=[], variance_method='no_paid_packages')
    if not rollouts or noise_floor <= 0 or prior_noise <= 0:
        raise ValueError('feedback batch and positive noise required')
    ids = list(package_gradients)
    parameters = reference.updated
    if tensor_state_hash(parameters) != reference.reference_hash:
        raise ValueError('feedback uncertainty must use the shared reference')
    rewards = np.array([r.reward for r in rollouts], dtype=np.float64)
    scores = np.zeros((len(rollouts), len(ids)))
    for i, rollout in enumerate(rollouts):
        for action in rollout.actions:
            # Already checked during reinforce_gradient; verify policy again.
            gradient = gradients(backend.score_action(action, parameters), parameters)
            for j, q in enumerate(ids):
                scores[i, j] += float(-reference.step.eta * sum(
                    (gradient[n].detach() * reference.step.diagonal[n] *
                     (package_gradients[q][n] - reference.g_D[n])).sum() for n in gradient))
            del gradient
    groups = [np.array([i for i, r in enumerate(rollouts) if r.task_id == task])
              for task in dict.fromkeys(r.task_id for r in rollouts)]
    if baseline not in {'smoke_zero', 'leave_one_out_same_task'} or (baseline != 'smoke_zero' and any(len(g) < 2 for g in groups)):
        raise ValueError('same-task LOO needs two rollouts')
    def estimate(indices):
        result = np.zeros(len(ids))
        for selected in indices:
            y, x = rewards[selected], scores[selected]
            advantage = y if baseline == 'smoke_zero' else y-(y.sum()-y)/(len(y)-1)
            result += (advantage[:, None]*x).sum(0)/len(rollouts)
        return result
    value = estimate(groups)
    rng = np.random.default_rng(seed)
    estimates = np.array([estimate([rng.choice(g, len(g), replace=True) for g in groups]) for _ in range(256)])
    covariance = np.atleast_2d(np.cov(estimates, rowvar=False, ddof=1))
    # Uniform rewards and two-rollout groups cannot establish small noise.
    # Keep a conservative prior floor instead of declaring a precise zero.
    identifiable = any(np.ptp(rewards[g]) > 0 for g in groups)
    floor = noise_floor if identifiable and all(len(g) >= 3 for g in groups) else max(noise_floor, prior_noise)
    covariance += np.diag(np.maximum(0., floor-np.diag(covariance)))
    # Identical request gradients can make the empirical covariance singular.
    # A common eigenvalue floor preserves correlation and positive definiteness.
    covariance += np.eye(len(ids))*max(0., noise_floor-float(np.linalg.eigvalsh(covariance).min()))
    return dict(query_ids=ids, values=value.tolist(), covariance=covariance.tolist(),
        variance_method='stratified_rollout_bootstrap_recomputed_loo', bootstrap_replicates=256,
        identifiable=identifiable, conservative_noise_floor=floor, rollout_count=len(rollouts),
        note='conditional on fixed tasks and sampled package gradients; correlated labels, approximate noise')


def reliability(first, second, *, revealed_ids):
    require_paid(first['query_ids'], revealed_ids)
    require_paid(second['query_ids'], revealed_ids)
    if first['query_ids'] != second['query_ids']:
        raise ValueError('reliability estimates must cover identical paid requests')
    a, b = np.asarray(first['values']), np.asarray(second['values'])
    correlation = None
    if len(a) >= 2 and np.std(a) > 0 and np.std(b) > 0:
        correlation = float(np.corrcoef(a, b)[0, 1])
    covariance = np.asarray(first['covariance']) + np.asarray(second['covariance'])
    return dict(query_ids=first['query_ids'], first=a.tolist(), second=b.tolist(),
        independent_batches=True, correlation=correlation,
        difference_z_scores=((a-b)/np.sqrt(np.maximum(np.diag(covariance), 1e-12))).tolist(),
        scope='purchased_or_explicitly_charged_calibration_only')
