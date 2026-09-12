"""Rev 3.1 selective distillation of temperature-1 stochastic-policy return J.

Student gradients are detached, single-step estimators, never signed CE replay
losses. State features, evidence and exposure weights precede source sampling.
The outer alpha derivative is blocked: d is held fixed, including at a bound.
"""
from dataclasses import dataclass
import math

import numpy as np
import torch

from ..behavior.deltas import tensor_state_hash
from .functional_step import gradients, snapshot
from .source_scoring import source_gradient_pair
from .transport import Behavior, FullState

RETURN_OBJECTIVE = 'temperature_1_stochastic_policy_expected_return'
ALPHA_D_DEFAULTS = dict(gate_mode='learned_alpha', fixed_alpha=.5, d_mode='learned',
    d_lambda=1., d_lambda_normalisation='mean_diagonal', d_warmup_windows=1,
    z_error_mode='loo', d_solver_tolerance=1e-8,
    d_solver_max_iterations=10000, d_redundancy_cosine_threshold=.9,
    microbatch_states=4, acquisition_value_mode='joint')


def enabled(config):
    return config.get('protocol_version') == '1.1.0' and 'gate_mode' in config


def validate_config(config):
    """Explicit opt-in keeps historical v1.0 and D1 diagnostic runs intact."""
    if not enabled(config):
        if any(k in config for k in ALPHA_D_DEFAULTS):
            raise ValueError('alpha/d settings require v1.1 and gate_mode')
        return
    c = ALPHA_D_DEFAULTS | config
    if c['acquisition_value_mode'] not in {'independent', 'joint'}:
        raise ValueError('acquisition_value_mode must be independent or joint')
    if c['gate_mode'] not in {'fixed_alpha', 'learned_alpha'} or c['d_mode'] not in {'zero', 'learned'}:
        raise ValueError('invalid gate_mode/d_mode')
    if c['z_error_mode'] not in {'loo', 'split_half'}:
        raise ValueError('z_error_mode must be loo or split_half')
    if c['d_lambda_normalisation'] not in {'mean_diagonal', 'none'}:
        raise ValueError('d_lambda_normalisation must be mean_diagonal or none')
    if not math.isfinite(c['fixed_alpha']) or not 0 <= c['fixed_alpha'] <= 1:
        raise ValueError('fixed_alpha must be in [0,1]')
    for k in ('d_lambda', 'd_solver_tolerance'):
        if not math.isfinite(c[k]) or c[k] < 0 or (k == 'd_solver_tolerance' and c[k] == 0):
            raise ValueError('finite nonnegative d_lambda and positive solver tolerance required')
    for k, minimum in [('d_warmup_windows', 0), ('d_solver_max_iterations', 1)]:
        if type(c[k]) is not int or c[k] < minimum:
            raise ValueError('invalid ' + k)
    if not -1 <= c['d_redundancy_cosine_threshold'] <= 1:
        raise ValueError('invalid redundancy cosine threshold')
    if c['microbatch_states'] != 4 or c.get('source_samples_per_state', 2) != 2:
        raise ValueError('rev 3 requires microbatch=4 states and two source actions per state')
    if c.get('source_temperature', 1.) != 1. or c.get('source_top_p', 1.) != 1.:
        raise ValueError('all arms require source temperature=1 and top_p=1')
    if c.get('source_estimator', 'alpha_d') != 'alpha_d' or 'cv_cs_mode' in c:
        raise ValueError('alpha/d requires its paired estimator, without CV coefficients')
    if c.get('gate_override') is not None:
        raise ValueError('alpha/d endpoints use gate_mode/fixed_alpha, not legacy gate_override')
    E = c.get('exposure_slots_per_window', 40)
    count = c.get('slots_per_step', E)
    if E != 40 or type(count) is not int or not 1 <= count <= E:
        raise ValueError('rev 3 exposure capacity E=40; actual states must be in [1,40]')
    if not 0 <= c.get('max_new_packages_per_window', 20) <= 20:
        raise ValueError('rev 3 K must be <=20')


def validate_arm(config, arm):
    if not enabled(config):
        if arm in {'V0', 'V1', 'V2'}:
            raise ValueError('V0/V1/V2 require the alpha/d protocol')
        return
    c = ALPHA_D_DEFAULTS | config
    if arm == 'V0' and (c['gate_mode'], c['fixed_alpha'], c['d_mode']) != ('fixed_alpha', .5, 'zero'):
        raise ValueError('V0 requires fixed_alpha=.5 and d_mode=zero')
    if arm in {'V1', 'V2'} and (c['gate_mode'], c['d_mode']) != ('learned_alpha', 'learned'):
        raise ValueError('V1/V2 require learned_alpha and learned d')
    from .conventions import ARMS
    if arm in ARMS:
        for key, value in ARMS[arm].items():
            if key == 'acquisition_value_mode' and arm != 'V0':
                continue  # frozen-pool independent-controller intervention
            if key in config and config[key] != value:
                raise ValueError(f'{arm} requires {key}={value}')
    if arm == 'V1' and not config.get('replay_schedule'):
        raise ValueError('V1 requires an exposure replay schedule')


def state_features(state, projection, *, scalar=False, device=None, dtype=torch.float32):
    """Only an initial-model STATE hidden projection and an intercept.

    No SourceSample argument, action likelihood, action length, action gradient,
    category or error type can enter chi. Projection is computed before draws.
    """
    if not isinstance(state, FullState):
        raise TypeError('alpha features require FullState, never a source action')
    values = (1.,) if scalar else (*projection, 1.)
    chi = torch.tensor(values, device=device, dtype=dtype)
    if not torch.isfinite(chi).all():
        raise ValueError('finite state projection required')
    return chi


def injection(chi, phi, *, mode, fixed_alpha=.5):
    if chi.ndim != 2 or chi.shape[1] != phi.numel() or not torch.isfinite(chi).all():
        raise ValueError('aligned state-only features required')
    if not torch.isfinite(phi).all():
        raise ValueError('finite alpha parameters required')
    if mode == 'fixed_alpha':
        if not 0 <= fixed_alpha <= 1:
            raise ValueError('invalid fixed alpha')
        return torch.full((len(chi),), fixed_alpha, dtype=phi.dtype, device=phi.device)
    if mode != 'learned_alpha':
        raise ValueError('invalid gate mode')
    return (chi.detach() @ phi).sigmoid()


def target_weights(alpha, d):
    a, d = torch.broadcast_tensors(torch.as_tensor(alpha), torch.as_tensor(d))
    if (not torch.isfinite(a).all() or not torch.isfinite(d).all() or
            (a < 0).any() or (a > 1).any() or (d.abs() > torch.minimum(a, 1-a)).any()):
        raise ValueError('illegal alpha/d probability transfer')
    return torch.stack(((1-a-d)/2, (1-a+d)/2, a), dim=-1)


def target_distribution(y1, y2, teacher_distribution, alpha, d):
    nu = torch.as_tensor(teacher_distribution)
    if nu.ndim != 1 or (nu < 0).any() or not torch.isfinite(nu).all() or not torch.isclose(nu.sum(), nu.new_tensor(1.)):
        raise ValueError('teacher must be a distribution')
    weights = target_weights(nu.new_tensor(alpha), nu.new_tensor(d))
    q = weights[2]*nu
    q = q.clone()
    q[y1] += weights[0]
    q[y2] += weights[1]
    return q


def estimate(soft, teacher, hard1, hard2, alpha, d):
    """Per-state ghat_d; alpha/d stop here, independent of their producers."""
    target_weights(alpha, d)
    a = float(alpha.detach()) if torch.is_tensor(alpha) else float(alpha)
    d = float(d.detach()) if torch.is_tensor(d) else float(d)
    return {n: ((1-a)*soft[n] + a*teacher[n] - (d/2)*(hard1[n]-hard2[n])).detach() for n in soft}


@dataclass(frozen=True)
class ExposureRecord:
    query_id: str | None
    record_index: int
    state: FullState
    teacher: Behavior | None
    is_new: bool = False


@dataclass(frozen=True)
class SourcePair:
    record: ExposureRecord
    sources: tuple
    draw_ids: tuple[str, str]

    def __post_init__(self):
        if len(self.sources) != 2 or self.sources[0] is self.sources[1] or len(set(self.draw_ids)) != 2:
            raise ValueError('two independently drawn sources required; cache reuse prohibited')
        for src in self.sources:
            self.record.state.assert_matches(src.behavior.state)
        if self.sources[0].frozen_snapshot_id != self.sources[1].frozen_snapshot_id:
            raise ValueError('pair source snapshots differ')
        if self.record.teacher is not None:
            self.record.state.assert_matches(self.record.teacher.state)


def components(pair, backend, start, source):
    h1, s1, _ = source_gradient_pair(backend, pair.sources[0], start, source)
    h2, s2, _ = source_gradient_pair(backend, pair.sources[1], start, source)
    soft = {n: (s1[n]+s2[n])/2 for n in start}
    teacher = ({n: torch.zeros_like(p) for n, p in start.items()} if pair.record.teacher is None else
               gradients(-backend.score_behavior(pair.record.teacher, start), start))
    return soft, teacher, h1, h2


def cpu_detached(values):
    return {n: p.detach().cpu().clone() for n, p in values.items()}


def dot(left, right):
    return sum(float((left[n].detach().cpu().double()*right[n].detach().cpu().double()).sum()) for n in left)


@dataclass
class BatchReference:
    """The same evidence/w/alpha/sources define both theta_S(0) and theta_S(d)."""
    start_hash: str
    theta0: dict
    directions: tuple
    weights: torch.Tensor
    alpha: torch.Tensor
    # Detached per-slot baseline gradients for full-update acquisition Fhat
    # and the independently journaled insertion control.
    baseline_gradients: tuple

    def selected(self, d):
        d = torch.as_tensor(d).detach().cpu().double()
        if d.shape != self.alpha.shape:
            raise ValueError('one d per exposure required')
        target_weights(self.alpha.double(), d)
        result = {}
        for n, p in self.theta0.items():
            delta = torch.zeros_like(p)
            for di, vi in zip(d, self.directions):
                delta.add_(vi[n].to(p)*float(di))
            result[n] = p.detach()+delta
        return snapshot(result)


def build_reference(pairs, weights, alpha, backend, start, source, step, *, record=None):
    w = torch.as_tensor(weights, dtype=torch.float64).detach().cpu().clone()
    a = torch.as_tensor(alpha, dtype=torch.float64).detach().cpu().clone()
    if (not pairs or w.shape != (len(pairs),) or a.shape != w.shape or
            not torch.isfinite(w).all() or (w < 0).any() or abs(float(w.sum())-1) > 1e-7):
        raise ValueError('frozen nonnegative exposure weights must sum to one')
    target_weights(a, torch.zeros_like(a))
    total = {n: torch.zeros_like(p) for n, p in start.items()}
    directions, baseline = [], []
    for offset in range(0, len(pairs), 4):
        for i in range(offset, min(offset+4, len(pairs))):
            if pairs[i].record.teacher is None and a[i] != 0:
                raise ValueError('source-only cold-start placeholder requires alpha=0')
            soft, teacher, h1, h2 = components(pairs[i], backend, start, source)
            gi = estimate(soft, teacher, h1, h2, a[i], 0.)
            baseline.append(cpu_detached(gi))
            directions.append(cpu_detached({n: (step.eta*step.diagonal[n])*(h1[n]-h2[n])*(float(w[i])/2) for n in start}))
            for n in total:
                total[n].add_(gi[n]*float(w[i]))
        if record:
            record(dict(microbatch_index=offset//4, states=min(4, len(pairs)-offset),
                        accumulated_states=min(offset+4, len(pairs)), committed=False))
    return BatchReference(tensor_state_hash(start), snapshot(step.update(start, total)), tuple(directions), w, a, tuple(baseline))


def gram(directions, diagonal):
    """K_ij=v_i^T H v_j with H=P^{-1}; accumulate in CPU float64."""
    K = np.zeros((len(directions), len(directions)))
    for n, p in diagonal.items():
        rows = torch.stack([v[n].cpu().double().flatten() for v in directions])
        K += ((rows/p.detach().cpu().double().flatten()) @ rows.T).numpy()
    return (K+K.T)/2


@dataclass(frozen=True)
class DCalibration:
    """One full-evidence scale, frozen across a window and its set comparisons."""
    mode: str
    scale: float
    nonzero_directions: int

    @property
    def linear_scale(self):
        return math.sqrt(self.scale)

    def journal(self, d_lambda):
        return dict(d_lambda=d_lambda, d_lambda_normalisation=self.mode, K_scale=self.scale,
            K_scale_fallback=self.mode == 'mean_diagonal' and self.nonzero_directions == 0,
            K_scale_nonzero_directions=self.nonzero_directions,
            lambda_effective=d_lambda/self.scale,
            lambda_original_units=d_lambda/self.linear_scale,
            objective_units='normalised' if self.mode == 'mean_diagonal' else 'original')


def calibrate_d(K, d_lambda_normalisation='mean_diagonal'):
    """Exclude exactly zero directions, never small directions or small gains."""
    if d_lambda_normalisation not in {'mean_diagonal', 'none'}:
        raise ValueError('d_lambda_normalisation must be mean_diagonal or none')
    K = np.asarray(K, dtype=np.float64)
    if K.ndim != 2 or K.shape[0] != K.shape[1] or not K.size or not np.isfinite(K).all():
        raise ValueError('finite nonempty square K required')
    nonzero = np.diag(K)[np.diag(K) > 0]
    scale = float(nonzero.mean()) if len(nonzero) and d_lambda_normalisation == 'mean_diagonal' else 1.
    return DCalibration(d_lambda_normalisation, scale, len(nonzero))


def solve_d(z, error, K, alpha, *, d_lambda=1., d_lambda_normalisation='mean_diagonal',
            calibration=None, tolerance=1e-8, max_iterations=10000):
    """Joint box/L1 quadratic coordinate descent, with a proximal KKT test.

    NO screening by raw z. At a zero interior coordinate the relevant score
    is z_tilde_i-lambda_effective*sum_{j!=i} K_ij*d_j, not z_i alone.
    lambda_effective=lambda/s acts on raw K with NORMALISED linear terms;
    the equivalent ORIGINAL-unit coefficient is lambda/sqrt(s). A failed
    solve raises. With normalisation='none', the historical QP is unchanged.
    """
    z, error, K, alpha = (np.asarray(x, dtype=np.float64) for x in (z, error, K, alpha))
    if (z.ndim != 1 or error.shape != z.shape or alpha.shape != z.shape or K.shape != (len(z), len(z)) or
            not all(np.isfinite(x).all() for x in (z, error, K, alpha)) or (error < 0).any() or
            (alpha < 0).any() or (alpha > 1).any() or not math.isfinite(d_lambda) or d_lambda < 0 or
            not math.isfinite(tolerance) or tolerance <= 0 or type(max_iterations) is not int or max_iterations < 1):
        raise ValueError('invalid joint solver inputs')
    calibration = calibration or calibrate_d(K, d_lambda_normalisation)
    if calibration.mode != d_lambda_normalisation:
        raise ValueError('shared calibration normalisation mismatch')
    z, error, K = z/calibration.linear_scale, error/calibration.linear_scale, K/calibration.scale
    if not np.allclose(K, K.T, atol=1e-12, rtol=1e-10) or np.linalg.eigvalsh(K).min() < -1e-9*max(1., np.linalg.norm(K)):
        raise ValueError('K must be symmetric positive semidefinite')
    bound, d = np.minimum(alpha, 1-alpha), np.zeros_like(z)
    L = max(float(d_lambda*np.linalg.norm(K, ord=np.inf)), 1.)
    def soft(x, t):
        return np.sign(x)*np.maximum(np.abs(x)-t, 0.)
    for iteration in range(1, max_iterations+1):
        for i in range(len(d)):
            residual = z[i]-d_lambda*(K[i] @ d-K[i, i]*d[i])
            curvature = d_lambda*K[i, i]
            if curvature > 0:
                d[i] = np.clip(soft(residual, error[i])/curvature, -bound[i], bound[i])
            else:
                d[i] = np.sign(residual)*bound[i] if abs(residual) > error[i] else 0.
        prox = np.clip(soft(d+(z-d_lambda*K @ d)/L, error/L), -bound, bound)
        residual = float(np.max(np.abs(d-prox)))
        if residual <= tolerance:
            break
    else:
        raise RuntimeError(f'd solver did not converge: proximal residual={residual}')
    return d, dict(converged=True, iterations=iteration, proximal_residual=residual,
        objective=float(z @ d-d_lambda/2*(d @ K @ d)-error @ np.abs(d)),
        active_lower=np.flatnonzero(np.isclose(d, -bound, atol=tolerance, rtol=0)).tolist(),
        active_upper=np.flatnonzero(np.isclose(d, bound, atol=tolerance, rtol=0)).tolist(),
        zero_coordinates=np.flatnonzero(d == 0).tolist(), **calibration.journal(d_lambda))


def gram_statistics(K, threshold=.9):
    scales = np.sqrt(np.maximum(np.diag(K), 0))
    denom = np.outer(scales, scales)
    cosine = np.divide(K, denom, out=np.zeros_like(K), where=denom > 0)
    mask = np.triu(np.ones_like(K, dtype=bool), 1)
    return dict(trace=float(np.trace(K)), frobenius_norm=float(np.linalg.norm(K)),
        diagonal=np.diag(K).tolist(), cosine_threshold=threshold, pair_count=int(mask.sum()),
        cosine_above_threshold_fraction=float(np.mean(cosine[mask] > threshold)) if mask.any() else 0.)


@dataclass
class FeedbackStatistic:
    """Re-projectable trajectory score gradients and LOO-weighted contributions.

    The uncertainty is a conditional Monte Carlo estimate. It does not cover
    gradient staleness bias, source sampling noise, or unobserved reward modes.
    CPU storage trades host memory for exact re-projection on NEW directions.
    """
    gradient: dict
    scores: tuple
    rewards: tuple
    task_ids: tuple
    parameter_hash: str
    refreshed_step: int
    baseline: str
    feedback_role: str = ''

    def project(self, directions, mode):
        if mode not in {'loo', 'split_half'}:
            raise ValueError('unknown z_error_mode')
        z = np.array([dot(self.gradient, v) for v in directions])
        raw = np.array([[dot(g, v) for v in directions] for g in self.scores])
        N = len(self.scores)
        if N != len(self.rewards) or N != len(self.task_ids) or not N:
            raise ValueError('aligned per-trajectory feedback required')
        error2 = np.zeros(len(directions))
        for task in sorted(set(self.task_ids)):
            ids = [i for i, t in enumerate(self.task_ids) if t == task]
            r = np.asarray(self.rewards)[ids]
            b = np.zeros(len(ids)) if self.baseline == 'smoke_zero' else (r.sum()-r)/max(1, len(r)-1)
            contributions = (r-b)[:, None]*raw[ids]
            if len(ids) < 2:
                # A smoke-only one-trajectory batch cannot estimate variance.
                # Explicitly suppress control instead of claiming zero error.
                error2[:] = np.inf
            elif mode == 'loo':
                if self.baseline != 'smoke_zero' and len(ids) >= 3:
                    # Delete each trajectory, THEN recompute the remaining
                    # same-task LOO baselines. Keep baseline dependence in SE.
                    means = []
                    for deleted in range(len(ids)):
                        keep = np.arange(len(ids)) != deleted
                        y, x = r[keep], raw[ids][keep]
                        advantage = y-(y.sum()-y)/(len(y)-1)
                        means.append((advantage[:, None]*x).mean(0))
                    means = np.array(means)
                else:
                    # n=2 cannot re-fit LOO after deletion. This is explicitly
                    # a paired-contribution proxy, not an error bound.
                    means = (contributions.sum(0)-contributions)/(len(ids)-1)
                error2 += (len(ids)/N)**2*(len(ids)-1)/len(ids)*np.square(means-means.mean(0)).sum(0)
            elif mode == 'split_half':
                first, second = np.arange(len(ids))[::2], np.arange(len(ids))[1::2]
                def half_estimate(keep, other):
                    # Opposite-half reward mean is independent of every action
                    # scored in this half (no action-dependent self baseline).
                    base = 0. if self.baseline == 'smoke_zero' else r[other].mean()
                    return ((r[keep]-base)[:, None]*raw[ids][keep]).mean(0)
                difference = half_estimate(first, second)-half_estimate(second, first)
                error2 += len(first)*len(second)/N**2*np.square(difference)
        return z, np.sqrt(error2)


def blocked_alpha_vjp(pairs, chi, phi, reference, backend, start, source, step, feedback, *, mode):
    """dJ(theta_S(d))/dphi with d fixed; no derivative through d*(alpha)."""
    if mode == 'fixed_alpha':
        return torch.zeros_like(phi)
    a = injection(chi, phi, mode=mode)
    cotangent = torch.zeros_like(a)
    for i, pair in enumerate(pairs):
        if pair.record.teacher is None:
            continue
        soft, teacher, _, _ = components(pair, backend, start, source)
        direction = {n: -(step.eta*step.diagonal[n])*(teacher[n]-soft[n])*float(reference.weights[i]) for n in start}
        cotangent[i] = dot(feedback.gradient, direction)
    return torch.autograd.grad(a, phi, grad_outputs=cotangent.detach())[0].detach()
