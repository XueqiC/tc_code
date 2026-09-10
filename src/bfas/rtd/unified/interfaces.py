"""Semantic P0 boundaries. Prediction is explicitly untrained/PLANNED."""
from dataclasses import dataclass, replace

import numpy as np

from ..selector import PublicQuerySpec
from ..persistence import digest
from .immutable import Array
from .objective import TeachingObjective
from .problem import build_teaching_problem, attach_feedback
from .solver import solve_exact


@dataclass(frozen=True)
class PrequeryContext:
    student_version: str
    optimizer_version: str
    progress: float
    remaining_budget: int
    teaching_summary: tuple[float, ...] = ()
    owned_summary: tuple[float, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, 'teaching_summary', tuple(self.teaching_summary))
        object.__setattr__(self, 'owned_summary', tuple(self.owned_summary))
        if (not self.student_version or not self.optimizer_version or not 0 <= self.progress <= 1
                or type(self.remaining_budget) is not int or self.remaining_budget < 0
                or not np.isfinite((*self.teaching_summary, *self.owned_summary)).all()):
            raise ValueError('finite pre-purchase context required')


@dataclass(frozen=True)
class PrequeryFeatures:
    query_id: str
    state_hash: str
    source_projection: tuple[float, ...]
    source_logprob: float | None
    source_length: int | None
    continuation_limit: int
    public_cap: int
    context: PrequeryContext

    @property
    def identity(self):
        return digest((self.query_id, self.state_hash, self.source_projection, self.source_logprob,
            self.source_length, self.continuation_limit, self.public_cap, vars(self.context)))


def build_prequery_features(query, context):
    if type(query) is not PublicQuerySpec or type(context) is not PrequeryContext:
        raise TypeError('only explicit public query/context values cross the prequery boundary')
    f = query.features
    # Deliberate field allowlist: no cap provenance, teacher, verdict, or actual usage.
    return PrequeryFeatures(query.query_id, query.state_hash, tuple(f.projection or ()),
        f.source_logprob, f.source_length, query.L, query.cost_upper_bound, context)


def purchase_or_reveal(query, ledger, *, broker):
    """Use the existing hard reserve/reveal ledger; P0 has no teacher API path."""
    from ..broker import SealedReplayBroker
    if type(query) is not PublicQuerySpec or not isinstance(broker, SealedReplayBroker) or broker.ledger is not ledger:
        raise TypeError('public request and its authoritative sealed broker/ledger required')
    record = broker._records.get(query.query_id)
    if record is None or (record.spec.state_hash, record.spec.L, record.spec.cost_upper_bound) != (
            query.state_hash, query.L, query.cost_upper_bound):
        raise ValueError('public request changed')
    package = broker.acquire(query.query_id)
    if package.query_id not in ledger.owned_ids or ledger.charges[package.query_id] != package.cost or ledger.remaining < 0:
        raise ValueError('reveal/charge mismatch')
    return package


@dataclass(frozen=True)
class ControlObservables:
    problem_id: str
    a_ref: Array


def predict_control(observables):
    if type(observables) is not ControlObservables or not isinstance(observables.a_ref, Array):
        raise TypeError('immutable controller observables required')
    return observables.a_ref  # P0 stub: generic reference, not a learned decision


@dataclass(frozen=True)
class QueryBatchScores:
    feature_ids: tuple[str, ...]
    scores: tuple = ()
    status: str = 'PLANNED; no acquisition predictor fitted'


def score_query_batch(prequery_features, context):
    if type(context) is not PrequeryContext or any(type(f) is not PrequeryFeatures or f.context != context
                                                  for f in prequery_features):
        raise TypeError('matching pre-purchase features/context only')
    return QueryBatchScores(tuple(f.identity for f in prequery_features))


@dataclass(frozen=True)
class RevealedEvidence:
    evidence: tuple
    owned_query_ids: frozenset[str]
    prequery_features: tuple[PrequeryFeatures, ...]
    ledger_hash: str


@dataclass(frozen=True)
class AcquisitionLabel:
    old_problem_id: str
    new_problem_id: str
    prequery_feature_ids: tuple[str, ...]
    ledger_hash: str
    old_value: float
    new_value: float
    value: float
    zero_use_value_error: float
    zero_use_increment_error: float
    expanded_problem: object
    label_kind: str = 'revealed_evidence_local_surrogate; not observed task gain'


def build_acquisition_labels(old_problem, revealed_evidence):
    """Open new teacher columns around the SAME reference, h, exposure and P.

    New a_ref coordinates are zero. Thus the old feasible set embeds by zero,
    including its retention baseline and uncertainty penalty. No recentering.
    """
    old, reveal = old_problem, revealed_evidence
    if old.feedback is None or old.feedback_version == 'unmeasured':
        raise ValueError('acquisition labels require measured common-reference feedback')
    if (type(reveal) is not RevealedEvidence or not reveal.ledger_hash
            or not set(old.context.owned_query_ids) <= reveal.owned_query_ids
            or any(e.query_id not in reveal.owned_query_ids for e in reveal.evidence)
            or not {e.query_id for e in reveal.evidence} <= {f.query_id for f in reveal.prequery_features}
            or any(f.context.student_version != old.theta_hash or
                   f.context.optimizer_version != old.context.optimizer_version for f in reveal.prequery_features)):
        raise ValueError('saved prequery features, purchased receipt and current versions required')
    context = replace(old.context, owned_query_ids=tuple(sorted(reveal.owned_query_ids)))
    new = build_teaching_problem(context, (*old.evidence, *reveal.evidence), old.exposure)
    n = len(old.coordinates)
    if new.coordinates[:n] != old.coordinates:
        raise ValueError('old direction coordinates changed')
    new = replace(new, a_ref=Array.of(np.r_[old.a_ref.numpy(), np.zeros(len(new.coordinates)-n)]))
    new = attach_feedback(new, old.feedback)
    a, b = solve_exact(old), solve_exact(new)
    embedded = np.r_[a.coefficients.numpy(), np.zeros(len(new.coordinates)-n)]
    first, second = TeachingObjective(old), TeachingObjective(new)
    value_error = abs(first.value(a.coefficients)-second.value(embedded))
    increment_error = float(np.max(np.abs(first.increment(a.coefficients)-second.increment(embedded)), initial=0))
    if increment_error != 0 or value_error > max(old.context.qp_tolerance, 1e-10):
        raise ValueError('new evidence cannot be ignored under the common objective')
    return AcquisitionLabel(old.identity, new.identity, tuple(f.identity for f in reveal.prequery_features),
        reveal.ledger_hash, a.value, b.value, b.value-a.value, value_error, increment_error, new)


def validate_and_report(run):
    """Validate supplied P0 result records; no invented benchmark scores."""
    problem, solution, commit = run['problem'], run['solution'], run['commit']
    if solution.problem_id != problem.identity or commit.problem_id != problem.identity:
        raise ValueError('report problem/version mismatch')
    expected = TeachingObjective(problem)
    permutation = np.asarray(run.get('permutation', np.arange(len(problem.coordinates))))
    if sorted(permutation.tolist()) != list(range(len(problem.coordinates))):
        raise ValueError('invalid source coefficient permutation')
    if (not np.array_equal(solution.coefficients.numpy()[permutation], commit.coefficients.numpy())
            or expected.parameters(commit.coefficients).hash != commit.parameters.hash
            or abs(expected.value(solution.coefficients)-solution.value) > problem.context.qp_tolerance
            or commit.rl_updates != 0):
        raise ValueError('solution/commit/objective mismatch')
    return dict(status='unified_local_step_validated', problem_id=problem.identity,
        objective=expected.value(commit.coefficients), qp_stationarity=solution.stationarity,
        qp_certificate_applies_to='solver_coefficients_before_optional_permutation',
        qp_feasibility=solution.feasibility, qp_complementarity=solution.complementarity,
        gram_repair=vars(problem.psd_repair), uncertainty=problem.uncertainty_label,
        feedback_version=problem.feedback_version, increment_error=commit.increment_error,
        backbone_updates=commit.backbone_updates, rl_updates=commit.rl_updates,
        benchmark_results='not_evaluated')
