"""Versioned local teaching problem in ordered, dense CPU coordinates (P0).

Candidate exposure slots exist before purchase. Purchase only opens directions;
it never creates a new retention term or renormalizes existing weights.
"""
from dataclasses import dataclass, replace
import math

import numpy as np

from ..persistence import digest
from .estimators import LossDefinition
from .immutable import Array, Parameters


@dataclass(frozen=True)
class PSDRepair:
    minimum_eigenvalue: float
    asymmetry: float
    correction_norm: float
    tolerance: float


def checked_psd(matrix, tolerance=1e-10):
    a = np.asarray(matrix, dtype=float)
    if (a.ndim != 2 or a.shape[0] != a.shape[1] or not np.isfinite(a).all()
            or not math.isfinite(tolerance) or tolerance <= 0):
        raise ValueError('finite square PSD matrix and positive tolerance required')
    if not len(a):
        return Array.of(a), PSDRepair(0., 0., 0., tolerance)
    scale = max(1., float(np.linalg.norm(a, 2)))
    asymmetry = float(np.max(np.abs(a-a.T)))
    symmetric = (a+a.T)/2
    eig, vectors = np.linalg.eigh(symmetric)
    if asymmetry > tolerance*scale or eig[0] < -tolerance*scale:
        raise ValueError('materially non-PSD/asymmetric geometry; numerical repair refused')
    repaired = symmetric if eig[0] >= 0 else (vectors*np.maximum(eig, 0)) @ vectors.T
    return Array.of(repaired), PSDRepair(float(eig[0]), asymmetry, float(np.linalg.norm(repaired-a)), tolerance)


@dataclass(frozen=True)
class Exposure:
    slot_id: str
    state_hash: str
    parent_hash: str
    source_hashes: tuple[str, ...]
    sample_hashes: tuple[str, ...]
    draw_ids: tuple[str, ...]
    source_policy_hash: str
    theta_hash: str
    loss_hash: str
    hard: Array
    soft: Array
    weight: float

    def __post_init__(self):
        for name in ('source_hashes', 'sample_hashes', 'draw_ids'):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        object.__setattr__(self, 'hard', Array.of(self.hard))
        object.__setattr__(self, 'soft', Array.of(self.soft))
        object.__setattr__(self, 'weight', float(Array.of(self.weight).numpy()))
        m = len(self.source_hashes)
        if (m != 2 or len(self.sample_hashes) != m or len(self.draw_ids) != m
                or len(set(self.draw_ids)) != m or len(self.hard.shape) != 2 or self.hard.shape[0] != m
                or self.soft.shape != self.hard.shape[1:] or not math.isfinite(self.weight) or self.weight < 0
                or not all(type(v) is str and v for v in (self.slot_id, self.state_hash, self.parent_hash,
                    self.source_policy_hash, self.theta_hash, self.loss_hash,
                    *self.source_hashes, *self.sample_hashes, *self.draw_ids))):
            raise ValueError('two independent, unfiltered, aligned source draws and frozen weight required')


@dataclass(frozen=True)
class Evidence:
    query_id: str
    version: str
    state_hash: str
    parent_hash: str
    text: str
    behavior_hash: str
    theta_hash: str
    loss_hash: str
    gradient: Array

    def __post_init__(self):
        object.__setattr__(self, 'gradient', Array.of(self.gradient))
        if (not all(type(v) is str and v for v in (self.query_id, self.version, self.state_hash, self.parent_hash,
                     self.behavior_hash, self.theta_hash, self.loss_hash))
                or type(self.text) is not str or len(self.gradient.shape) != 1):
            raise ValueError('purchased version, complete text and gradient binding required')

    @property
    def text_hash(self):
        return digest(self.text)

    @property
    def key(self):
        return self.query_id, self.version, self.state_hash


@dataclass(frozen=True)
class TeachingContext:
    theta: Parameters
    source_snapshot: Parameters
    source_policy_hash: str
    preconditioner: Array
    loss: LossDefinition = LossDefinition()
    eta: float = 1e-5
    a_ref_default: float = .5
    regularization: float = 1.
    qp_tolerance: float = 1e-8
    qp_max_iterations: int = 10000
    psd_tolerance: float = 1e-10
    estimator: str = 'cv'
    exposure_version: str = 'exposure-1'
    sampling_version: str = 'draw-1'
    optimizer_version: str = 'frozen-P-1'
    owned_query_ids: tuple[str, ...] = ()
    inner_parent_hashes: tuple[str, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, 'theta', Parameters.of(self.theta))
        object.__setattr__(self, 'source_snapshot', Parameters.of(self.source_snapshot))
        object.__setattr__(self, 'preconditioner', Array.of(self.preconditioner))
        object.__setattr__(self, 'owned_query_ids', tuple(self.owned_query_ids))
        object.__setattr__(self, 'inner_parent_hashes', tuple(self.inner_parent_hashes))
        for name in ('eta', 'a_ref_default', 'regularization', 'qp_tolerance', 'psd_tolerance'):
            object.__setattr__(self, name, float(Array.of(getattr(self, name)).numpy()))
        p = self.preconditioner.numpy()
        if (p.shape != self.theta.values.shape or (p <= 0).any()
                or self.theta.names != self.source_snapshot.names or self.theta.shapes != self.source_snapshot.shapes
                or not all(type(v) is str and v for v in (self.source_policy_hash, self.exposure_version,
                    self.sampling_version, self.optimizer_version, *self.owned_query_ids, *self.inner_parent_hashes))
                or type(self.loss) is not LossDefinition
                or not 0 <= self.a_ref_default <= 1 or self.estimator not in {'cv', 'raw'}
                or not all(math.isfinite(v) for v in (self.eta, self.regularization, self.qp_tolerance, self.psd_tolerance))
                or self.eta <= 0 or self.regularization < 0 or self.qp_tolerance <= 0 or self.psd_tolerance <= 0
                or type(self.qp_max_iterations) is not int or self.qp_max_iterations < 1):
            raise ValueError('invalid frozen teaching context')


@dataclass(frozen=True)
class Coordinate:
    slot_id: str
    source_index: int
    evidence_key: tuple[str, str, str]

    def __post_init__(self):
        object.__setattr__(self, 'evidence_key', tuple(self.evidence_key))
        if (type(self.slot_id) is not str or not self.slot_id or type(self.source_index) is not int
                or self.source_index not in {0, 1} or len(self.evidence_key) != 3
                or not all(type(v) is str and v for v in self.evidence_key)):
            raise ValueError('immutable source/teacher coordinate required')


@dataclass(frozen=True)
class TeachingProblem:
    context: TeachingContext
    exposure: tuple[Exposure, ...]
    evidence: tuple[Evidence, ...]
    coordinates: tuple[Coordinate, ...]
    a_ref: Array
    base_increment: Array
    U: Array
    K: Array
    H: Array
    psd_repair: PSDRepair
    feedback_version: str
    h: Array
    epsilon: Array
    feedback_hash: str
    # Reprojectable feedback retained for opening new evidence directions.
    feedback: object = None
    version: str = 'unified-teaching-problem-1'
    uncertainty_label: str = 'uncertainty regulariser'

    def __post_init__(self):
        # Public construction is validated as well as the factory; no mutable
        # array/tensor references may enter the supposedly immutable problem.
        if (type(self.context) is not TeachingContext
                or type(self.exposure) is not tuple or type(self.evidence) is not tuple
                or type(self.coordinates) is not tuple or self.version != 'unified-teaching-problem-1'
                or any(type(e) is not Exposure for e in self.exposure)
                or any(type(e) is not Evidence for e in self.evidence)
                or any(type(c) is not Coordinate for c in self.coordinates)
                or type(self.psd_repair) is not PSDRepair
                or self.uncertainty_label != 'uncertainty regulariser'
                or any(not isinstance(getattr(self, n), Array) for n in
                       ('a_ref', 'base_increment', 'U', 'K', 'H', 'h', 'epsilon'))):
            raise ValueError('immutable problem representations required')
        n, p = len(self.coordinates), len(self.context.theta.values.numpy())
        if (self.U.shape != (p, n) or self.K.shape != (n, n) or self.H.shape != (p,)
                or self.h.shape != (p,) or self.epsilon.shape != (n,) or self.a_ref.shape != (n,)
                or self.base_increment.shape != (p,) or (self.epsilon.numpy() < 0).any()
                or not all(type(v) is str and v for v in (self.feedback_version, self.feedback_hash))):
            raise ValueError('problem dimensions/feedback invalid')
        from .feedback import FeedbackEstimate
        if self.feedback is not None and type(self.feedback) is not FeedbackEstimate:
            raise ValueError('immutable feedback representation required')
        if not np.array_equal(self.H.numpy(), 1/self.context.preconditioner.numpy()):
            raise ValueError('H must be the inverse frozen diagonal preconditioner')
        expected_K = self.U.numpy().T @ (self.H.numpy()[:, None]*self.U.numpy())
        if not np.allclose(self.K.numpy(), expected_K, rtol=self.context.psd_tolerance,
                           atol=self.context.psd_tolerance*max(1., np.linalg.norm(expected_K))):
            raise ValueError('K must contain the full teacher-conditioned Gram cross terms')
        checked_psd(self.K.numpy(), self.context.psd_tolerance)
        self.check_coefficients(self.a_ref.numpy())

    @property
    def theta_hash(self):
        return self.context.theta.hash

    @property
    def source_snapshot_hash(self):
        return self.context.source_snapshot.hash

    @property
    def P_hash(self):
        return self.context.preconditioner.hash

    @property
    def H_hash(self):
        return self.H.hash

    @property
    def lower(self):
        return -self.a_ref.numpy()

    @property
    def upper(self):
        return 1-self.a_ref.numpy()

    @property
    def groups(self):
        groups = {}
        for j, c in enumerate(self.coordinates):
            groups.setdefault((c.slot_id, c.source_index), []).append(j)
        return tuple(tuple(ids) for ids in groups.values())

    def check_coefficients(self, a, *, tolerance=1e-12):
        # Explicit stop-gradient even when a later controller supplies tensors.
        a = Array.of(a).numpy()
        if (a.shape != self.a_ref.shape or (a < -tolerance).any() or (a > 1+tolerance).any()
                or any(a[list(g)].sum() > 1+tolerance for g in self.groups)):
            raise ValueError('infeasible source replacement coefficients')
        return a

    @property
    def reference_parameters(self):
        # Same objective/update implementation used by trials and commits.
        from .objective import TeachingObjective
        return TeachingObjective(self).parameters(self.a_ref.numpy())

    @property
    def identity(self):
        return digest(dict(version=self.version, theta=self.theta_hash, source=self.source_snapshot_hash,
            source_policy=self.context.source_policy_hash, P=self.P_hash, H=self.H_hash,
            loss=self.context.loss.hash, exposure=[(e.slot_id, e.state_hash, e.sample_hashes,
                e.draw_ids, e.weight, e.hard.hash, e.soft.hash) for e in self.exposure],
            evidence=[(e.key, e.text_hash, e.behavior_hash, e.gradient.hash) for e in self.evidence],
            coordinates=[(c.slot_id, c.source_index, c.evidence_key) for c in self.coordinates],
            arrays={n: getattr(self, n).hash for n in ('a_ref', 'base_increment', 'U', 'K', 'epsilon', 'h')},
            feedback=(self.feedback_version, self.feedback_hash), regularization=self.context.regularization,
            estimator=self.context.estimator, eta=self.context.eta,
            numerical=(self.context.qp_tolerance, self.context.qp_max_iterations, vars(self.psd_repair)),
            versions=(self.context.exposure_version, self.context.sampling_version, self.context.optimizer_version),
            feasible='box_and_per_source_teacher_version_simplex'))


def build_teaching_problem(context, owned_evidence, frozen_exposure, *, a_ref=None, evidence_by_slot=None):
    exposure, evidence = tuple(frozen_exposure), tuple(owned_evidence)
    if (not exposure or len({e.slot_id for e in exposure}) != len(exposure)
            or len({e.key for e in evidence}) != len(evidence)
            or abs(sum(e.weight for e in exposure)-1) > 1e-10):
        raise ValueError('unique slots/evidence and fixed exposure weights summing to one required')
    theta_hash, p = context.theta.hash, context.preconditioner.numpy()
    if evidence_by_slot is not None and (set(evidence_by_slot) != {e.slot_id for e in exposure}
            or any(key is not None and key not in {t.key for t in evidence} for key in evidence_by_slot.values())):
        raise ValueError('each scheduled slot must bind its purchased teacher or explicit retention-only baseline')
    for e in (*exposure, *evidence):
        g = e.soft if isinstance(e, Exposure) else e.gradient
        if (e.theta_hash != theta_hash or e.loss_hash != context.loss.hash
                or g.shape != p.shape or e.parent_hash not in context.inner_parent_hashes):
            raise ValueError('gradient theta/loss/layout or inner parent mismatch')
    if (any(e.source_policy_hash != context.source_policy_hash for e in exposure)
            or any(e.query_id not in context.owned_query_ids for e in evidence)
            or any(not any((s.state_hash, s.parent_hash) == (e.state_hash, e.parent_hash)
                           for s in exposure) for e in evidence)):
            raise ValueError('only purchased evidence on pre-existing exposure slots and frozen source policy allowed')
    draws = [d for e in exposure for d in e.draw_ids]
    if len(set(draws)) != len(draws):
        raise ValueError('source draw IDs must be unique across exposure slots')
    baseline = sum((e.weight*(e.soft.numpy() if context.estimator == 'cv' else e.hard.numpy().mean(0))
                    for e in exposure), np.zeros_like(p))
    columns, coordinates, defaults = [], [], []
    # Evidence-major ordering preserves ALL old columns exactly upon extension.
    opened = set()
    for t in evidence:
        for e in exposure:
            if evidence_by_slot is not None and evidence_by_slot[e.slot_id] != t.key:
                continue
            if (e.state_hash, e.parent_hash) != (t.state_hash, t.parent_hash):
                continue
            for j, hard in enumerate(e.hard.numpy()):
                columns.append(-context.eta*e.weight/2*p*(t.gradient.numpy()-hard))
                coordinates.append(Coordinate(e.slot_id, j, t.key))
                slot = (e.slot_id, j)
                # Generic initialization: .5 for first teacher, zero for later
                # versions; no re-averaging the reference after purchase.
                defaults.append(context.a_ref_default if slot not in opened else 0.)
                opened.add(slot)
    U = np.column_stack(columns) if columns else np.zeros((len(p), 0))
    H = 1/p
    K, repair = checked_psd(U.T @ (H[:, None]*U), context.psd_tolerance)
    return TeachingProblem(context, exposure, evidence, tuple(coordinates),
        Array.of(defaults if a_ref is None else a_ref), Array.of(-context.eta*p*baseline),
        Array.of(U), K, Array.of(H), repair, 'unmeasured', Array.of(np.zeros_like(p)),
        Array.of(np.zeros(len(columns))), digest('unmeasured; no claimed zero standard error'))


def attach_feedback(problem, feedback):
    if feedback.parameter_hash != problem.reference_parameters.hash:
        raise ValueError('feedback must be at this exact shared reference checkpoint')
    h, epsilon = feedback.project(problem.context.theta, problem.U.numpy())
    return replace(problem, h=Array.of(h), epsilon=Array.of(epsilon), feedback=feedback,
                   feedback_version=feedback.version, feedback_hash=feedback.identity)
