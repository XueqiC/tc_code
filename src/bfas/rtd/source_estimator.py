"""RTD v1.1 source estimators; samples/features are frozen, gates are not.

The CV coefficient estimates E[1-a(s,Y)|s] using other independent draws.
It is a valid control coefficient, not a claim of variance optimality.
CV is a refreshed-sample update estimator, not a signed replay objective.
"""
from dataclasses import dataclass

import torch

from .transport import SourceSample


SOURCE_ESTIMATORS = {'hard2', 'soft', 'cv'}
CS_MODES = {'loo', 'independent', 'fixed_one_minus_a'}


def validate_source_config(config):
    """Validate without inserting defaults into historical manifest configs."""
    estimator = config.get('source_estimator', 'hard2')
    mode = config.get('cv_cs_mode', 'loo')
    count = config.get('source_samples_per_state', 2)
    if estimator not in SOURCE_ESTIMATORS:
        raise ValueError('source_estimator must be hard2/soft/cv')
    if mode not in CS_MODES:
        raise ValueError('cv_cs_mode must be loo/independent/fixed_one_minus_a')
    if type(count) is not int or count < 1:
        raise ValueError('source_samples_per_state must be a positive integer')
    if estimator == 'cv' and mode == 'loo' and count < 2:
        raise ValueError('loo requires at least two independent source draws per state')
    return estimator, mode, count


@dataclass(frozen=True)
class SourceControl:
    """Durable provenance for one slot's coefficient, including unselected draws.

    LOO excludes a draw by index, not by token equality: independent draws may
    have identical tokens. Independent mode requires a separately sampled pool.
    Never construct a LOO pool from exposure slots sampled with replacement.
    """
    samples: tuple[SourceSample, ...]
    chi: torch.Tensor
    excluded_index: int | None = None

    def features(self, source, mode):
        if not self.samples or self.chi.ndim != 2 or len(self.samples) != len(self.chi):
            raise ValueError('aligned nonempty coefficient draws/features required')
        for sample in self.samples:
            source.behavior.state.assert_matches(sample.behavior.state)
            if sample.frozen_snapshot_id != source.frozen_snapshot_id:
                raise ValueError('coefficient source snapshot mismatch')
        if len({id(sample) for sample in self.samples}) != len(self.samples):
            raise ValueError('coefficient pool must contain distinct independent draws')
        if mode == 'loo':
            i = self.excluded_index
            if type(i) is not int or not 0 <= i < len(self.samples) or self.samples[i] is not source:
                raise ValueError('loo must identify the current draw by index')
            if len(self.samples) < 2:
                raise ValueError('loo requires another independent draw')
            return torch.cat((self.chi[:i], self.chi[i+1:])).detach()
        if mode != 'independent' or self.excluded_index is not None or any(s is source for s in self.samples):
            raise ValueError('independent coefficient pool must exclude the current draw')
        return self.chi.detach()


def gate_value(chi, phi, gate):
    if gate not in {'linear_sigmoid', 'scalar_sigmoid', 'fixed_half', 'teacher_only'}:
        raise ValueError('unsupported source estimator gate')
    if phi.ndim != 1 or chi.shape[-1] != phi.numel() or not torch.isfinite(chi).all() or not torch.isfinite(phi).all():
        raise ValueError('aligned finite gate parameters/features required')
    if gate == 'scalar_sigmoid' and (phi.numel() != 1 or not torch.all(chi == 1)):
        raise ValueError('scalar gate requires constant intercept-only features')
    a = (chi.detach() @ phi).sigmoid()
    if gate == 'fixed_half':
        a = a * 0 + .5
    elif gate == 'teacher_only':
        a = a * 0 + 1.
    return a


def estimator_coefficients(targets, chi, phi, *, gate, source_estimator, cs_mode='loo', controls=None):
    """Return differentiable (hard, soft, teacher) weights and c_s per slot.

    The same operator supplies the actual gradient and its gate VJP. In
    particular c_s is differentiated through the gates on the auxiliary draws.
    """
    if source_estimator not in SOURCE_ESTIMATORS or cs_mode not in CS_MODES:
        raise ValueError('invalid source estimator or cs_mode')
    if not targets or len(targets) != len(chi):
        raise ValueError('aligned nonempty slots required')
    dependent = gate == 'linear_sigmoid' and any(t is not None for _, t in targets)
    if source_estimator == 'soft' and dependent:
        raise ValueError('soft source requires a fixed/scalar gate; use cv for a source-dependent gate')
    if source_estimator == 'cv' and cs_mode == 'fixed_one_minus_a' and dependent:
        raise ValueError('fixed_one_minus_a requires a fixed/scalar gate')
    if controls is not None and len(controls) != len(targets):
        raise ValueError('one source control per slot required')
    weights, coefficients = [], []
    for i, ((source, teacher), features) in enumerate(zip(targets, chi)):
        if teacher is not None:
            source.behavior.state.assert_matches(teacher.state)
        a = gate_value(features, phi, gate)
        if teacher is None:
            a = a * 0
        if source_estimator == 'hard2':
            c = a * 0
        elif source_estimator == 'soft' or cs_mode == 'fixed_one_minus_a' or teacher is None:
            c = 1 - a
        else:
            if controls is None or controls[i] is None:
                raise ValueError('cv requires independent or strict loo source controls')
            auxiliary = controls[i].features(source, cs_mode)
            c = (1 - gate_value(auxiliary, phi, gate)).mean()
        weights.append(torch.stack((1-a-c, c, a)))
        coefficients.append(c)
    return torch.stack(weights), torch.stack(coefficients)


def gradient_norm(gradient):
    return float(sum(g.detach().double().square().sum() for g in gradient.values()).sqrt())


def source_variance_diagnostic(state, backend, parameters, source_parameters, phi, feature_fn, *,
                               generator, K, teacher=None, gate='fixed_half',
                               source_samples_per_state=2, cs_mode='loo'):
    """Resample K independent batches at one caller-authorized state (D6 API).

    Uses identical source draws/teacher/phi/theta for all three estimators.
    ``feature_fn`` must use already frozen statistics, not fit on these draws.
    For a source-dependent gate, soft is a *biased comparison*, labelled below.
    Variance is the population variance of batch gradient L2 norms, not trace
    covariance, not variance of squared norms, and not a promised reduction.
    """
    from .runtime import streamed_gradient
    from .transport import SamplingRequest, sample_sources
    if type(K) is not int or K < 2:
        raise ValueError('K must be an integer >= 2')
    validate_source_config(dict(source_estimator='cv', cv_cs_mode=cs_mode,
                                source_samples_per_state=source_samples_per_state))
    sampler = backend.source_sampler(source_parameters)
    request = SamplingRequest(state, backend.identity(source_parameters), samples=source_samples_per_state)
    norms = {name: [] for name in ('hard2', 'soft', 'cv')}
    for _ in range(K):
        sources = sample_sources(request, sampler, generator)
        chi = torch.stack([feature_fn(s).detach() for s in sources])
        aux = sample_sources(request, sampler, generator) if cs_mode == 'independent' else sources
        aux_chi = torch.stack([feature_fn(s).detach() for s in aux]) if cs_mode == 'independent' else chi
        controls = [SourceControl(aux, aux_chi, i if cs_mode == 'loo' else None) for i in range(len(sources))]
        targets = [(s, teacher) for s in sources]
        # Computing all component norms in one streamed pass avoids three
        # redundant source forwards, and allows the labelled soft comparison.
        rows = []
        streamed_gradient(targets, chi, phi, backend, parameters, gate=gate, source_estimator='cv',
                          source_parameters=source_parameters, cv_cs_mode=cs_mode, source_controls=controls,
                          diagnostic_gradients=rows)
        for name in norms:
            norms[name].append(gradient_norm(rows[0][name]))
    return dict(state_hash=state.state_hash, source_id=request.frozen_snapshot_id, K=K,
        source_samples_per_state=source_samples_per_state, cs_mode=cs_mode,
        soft_preserves_target=gate != 'linear_sigmoid' or teacher is None,
        variance_definition='population variance of per-resample mean-gradient L2 norms',
        estimators={name: dict(gradient_norms=values, mean_gradient_norm=float(torch.tensor(values, dtype=torch.float64).mean()),
            gradient_norm_variance=float(torch.tensor(values, dtype=torch.float64).var(correction=0)))
            for name, values in norms.items()})
