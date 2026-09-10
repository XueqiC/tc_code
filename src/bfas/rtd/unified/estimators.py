"""Frozen-coefficient losses and unweighted control variates; no gate derivative.

See RTD_UNIFIED_METHOD_ZH.md for the random-length correction. In particular,
per-sequence-mean retention is NOT generally stationary at the source policy.
"""
from dataclasses import dataclass
import math

import numpy as np
import torch

from ..persistence import digest


@dataclass(frozen=True)
class LossDefinition:
    normalization: str = 'per_sequence_mean'
    # Predictable, frozen BEFORE the action: typically inverse generation cap.
    retention_scale: float = 1.0
    soft_mode: str = 'length_corrected_prefix'
    version: str = 'unified-loss-1'
    sampling: str = 'temperature_1_top_p_1_categorical_eos_or_cap'

    def __post_init__(self):
        scale = self.retention_scale
        object.__setattr__(self, 'retention_scale', float(scale.detach()) if torch.is_tensor(scale) else float(scale))
        if (self.normalization not in {'per_sequence_mean', 'total_token_nll'}
                or self.soft_mode not in {'length_corrected_prefix', 'conditional_length'}
                or not math.isfinite(self.retention_scale) or self.retention_scale <= 0
                or self.version != 'unified-loss-1'
                or self.sampling != 'temperature_1_top_p_1_categorical_eos_or_cap'):
            raise ValueError('invalid loss/sampling definition')

    @property
    def hash(self):
        return digest(vars(self))


def _logprobs(logits, labels):
    if logits.ndim != 2 or logits.shape[0] < 1 or labels.shape != (logits.shape[0],):
        raise ValueError('one action-only logit row per sampled token required')
    if not torch.isfinite(logits).all():
        raise ValueError('finite logits required')
    dtype = torch.float64 if logits.dtype == torch.float64 else torch.float32
    return logits.to(dtype).log_softmax(-1)


def hard_loss(logits, labels, definition=LossDefinition()):
    logq = _logprobs(logits, labels)
    total = -logq.gather(-1, labels[:, None]).sum()
    return total/len(labels) if definition.normalization == 'per_sequence_mean' else total


def soft_retention_loss(logits, snapshot_logits, labels, definition=LossDefinition(), *, inverse_lengths=None):
    """A gradient surrogate on SOURCE prefixes (snapshot and labels detached).

    total: sum_t CE(p_t(.|prefix), pi_theta(.|prefix)).
    mean: c*soft_total + (1/L-c)*hard_total. Its expectation matches hard/L
          because c is predictable, while dividing soft_total by L does not.
    conditional_length: exact Rao-Blackwell alternative, given
          inverse_lengths[t,v] = E[1/L | prefix_t, next_token=v].
    The analytic CE cotangent gives exact zero for identical total-NLL policies.
    Its scalar VALUE is a surrogate, not a KL or the empirical-target loss.
    """
    logq = _logprobs(logits, labels)
    if snapshot_logits.shape != logits.shape:
        raise ValueError('source/current prefix-vocabulary layouts differ')
    logp = _logprobs(snapshot_logits.detach(), labels).detach()
    p = logp.exp()
    if definition.normalization == 'per_sequence_mean' and definition.soft_mode == 'conditional_length':
        if inverse_lengths is None:
            raise ValueError('conditional E[1/L | prefix, token] required; realised 1/L is invalid')
        rho = torch.as_tensor(inverse_lengths, device=logq.device, dtype=logq.dtype).detach()
        if rho.shape != logq.shape or not torch.isfinite(rho).all() or (rho <= 0).any() or (rho > 1).any():
            raise ValueError('invalid conditional inverse length weights')
        return -(p*rho*logq).sum()
    residual = (logq.exp()-p).detach()
    soft_total = (logits.to(logq.dtype)*residual).sum()
    if definition.normalization == 'total_token_nll':
        return soft_total
    hard_total = -logq.gather(-1, labels[:, None]).sum()
    c = definition.retention_scale
    return c*soft_total + (1/len(labels)-c)*hard_total


def coefficients(a, m, versions=1):
    if torch.is_tensor(a):
        a = a.detach().cpu().double().numpy()
    a = np.asarray(a, dtype=float)
    if a.shape == (m,) and versions == 1:
        a = a[:, None]
    if (a.shape != (m, versions) or not np.isfinite(a).all() or (a < 0).any()
            or (a > 1).any() or (a.sum(1) > 1+1e-12).any()):
        raise ValueError('each source has nonnegative teacher shares with sum <= 1')
    return a


def empirical_target(sources, teachers, a):
    """Hashable complete behaviours; equal behaviours merge their mass."""
    if not sources or not teachers:
        raise ValueError('nonempty source and teacher behaviour lists required')
    a = coefficients(a, len(sources), len(teachers))
    q = {}
    for j, source in enumerate(sources):
        q[source] = q.get(source, 0.) + max(0., 1-a[j].sum())/len(sources)
        for v, teacher in enumerate(teachers):
            q[teacher] = q.get(teacher, 0.) + a[j, v]/len(sources)
    return {y: mass for y, mass in q.items() if mass != 0.}


def estimate(hard, teacher, soft, a, *, estimator='cv'):
    """Dense per-state oracle; hard[m,p], teacher[v,p], soft[p]."""
    hard, teacher, soft = (np.asarray(t, dtype=float) for t in (hard, teacher, soft))
    if teacher.ndim == 1:
        teacher = teacher[None, :]
    if (hard.ndim != 2 or not len(hard) or teacher.ndim != 2 or soft.shape != hard.shape[1:]
            or teacher.shape[1:] != soft.shape or estimator not in {'raw', 'cv'}
            or not all(np.isfinite(t).all() for t in (hard, teacher, soft))):
        raise ValueError('aligned finite gradients and raw/cv estimator required')
    a = coefficients(a, len(hard), len(teacher))
    correction = np.einsum('jv,jvp->p', a, teacher[None, :, :]-hard[:, None, :])/len(hard)
    return (soft if estimator == 'cv' else hard.mean(0)) + correction


def alpha_d(a):
    a = coefficients(a, 2)[:, 0]
    return float(a.mean()), float((a[0]-a[1])/2)
