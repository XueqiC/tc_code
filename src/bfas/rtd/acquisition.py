"""Public-only Bayesian acquisition and entropy-budget dual (9).

Only revealed request-level labels/usage update regressors. Selection sees
PublicQuerySpec and pre-purchase numerical features, never a broker payload.
"""
from dataclasses import dataclass
import math

import numpy as np

from .insertion import InsertionLabel, LabelType
from .selector import PublicQuerySpec, select_public


@dataclass(frozen=True)
class PrePurchaseFeatures:
    query_id: str
    values: tuple[float, ...]
    round_id: str

    def __post_init__(self):
        if not self.query_id or not self.round_id or not self.values or not np.isfinite(self.values).all():
            raise ValueError("finite pre-purchase features and request/round identity required")

    @classmethod
    def from_public(cls, spec: PublicQuerySpec, *, round_id, coverage, progress, support_return):
        f = spec.features
        if f.projection is None or f.source_logprob is None or f.source_length is None:
            raise ValueError("missing source features; run legal source sampling first")
        return cls(spec.query_id, (*f.projection, f.source_logprob, f.source_length, float(spec.L),
            float(coverage), float(progress), float(support_return), 1.), round_id)


class BayesianLinearRegression:
    def __init__(self, dimension, *, prior_precision=1., noise_variance=1.):
        if dimension < 1 or prior_precision != 1 or not math.isfinite(noise_variance) or noise_variance <= 0:
            raise ValueError("v1 precision=1 and positive fixed observation noise required")
        self.dimension, self.noise_variance = dimension, noise_variance
        self.precision = np.eye(dimension, dtype=np.float64)
        self.information = np.zeros(dimension, dtype=np.float64)

    def vector(self, values):
        x = np.asarray(values, dtype=np.float64)
        if x.shape != (self.dimension,) or not np.isfinite(x).all():
            raise ValueError("invalid regression feature vector")
        return x

    def observe(self, values, target, *, noise_variance=None):
        x = self.vector(values)
        noise = self.noise_variance if noise_variance is None else noise_variance
        if not math.isfinite(target) or not math.isfinite(noise) or noise <= 0:
            raise ValueError("finite target and positive noise required")
        self.precision += np.outer(x, x) / noise
        self.information += x * target / noise

    @property
    def mean(self):
        return np.linalg.solve(self.precision, self.information)

    def sample(self, rng):
        # precision=L L^T, so L^-T z has covariance precision^-1.
        L = np.linalg.cholesky(self.precision)
        return self.mean + np.linalg.solve(L.T, rng.standard_normal(self.dimension))

    def prior_predictive_std(self, values):
        x = self.vector(values)
        return math.sqrt(self.noise_variance + float(x @ x))


class ValuePosterior:
    def __init__(self, dimension, *, round_id, noise_variance=1.):
        self.dimension, self.noise_variance = dimension, noise_variance
        self.begin_round(round_id)

    def begin_round(self, round_id):
        if not round_id:
            raise ValueError("round identity required")
        self.round_id = round_id
        self.model = BayesianLinearRegression(self.dimension, noise_variance=self.noise_variance)
        self.observations = {}
        self.reweight_observations = {}  # separate diagnostic records, never fit as new purchases

    def observe(self, features: PrePurchaseFeatures, label: InsertionLabel):
        if (features.query_id != label.query_id or
                features.round_id != self.round_id or label.round_id != self.round_id):
            raise ValueError("pre-purchase feature/request/round mismatch; stale labels excluded")
        target = self.observations if label.label_type is LabelType.PENDING_NEW else self.reweight_observations
        if label.query_id in target:
            raise ValueError("duplicate request-level label")
        self.model.vector(features.values)
        if label.label_type is LabelType.PENDING_NEW:
            self.model.observe(features.values, label.value)
        target[label.query_id] = (features, label)


class CostRegressor:
    """Regress revealed cost/cap, retaining exact/estimated observation flags.

    No observations: use the public cap. Thereafter ridge-shrink residuals
    around cap ratio=1. Estimated usage has fixed variance 4; exact variance 1.
    Predictions clipped to [0,cap] are expectations, not ledger reservations.
    """
    def __init__(self, dimension):
        self.model = BayesianLinearRegression(dimension)
        self.observations = {}

    def observe_revealed(self, spec, features, *, cost, confidence, revealed_ids):
        if spec.query_id not in revealed_ids or features.query_id != spec.query_id:
            raise ValueError("cost fitting requires a revealed matching request")
        if spec.query_id in self.observations:
            raise ValueError("cost charged/observed once per package")
        if confidence not in {"exact", "estimated"} or confidence != spec.cost_confidence:
            raise ValueError("preserve exact/estimated cost flags")
        if not math.isfinite(cost) or cost < 0 or cost > spec.cost_upper_bound:
            raise ValueError("revealed usage outside public reservation")
        ratio = cost / spec.cost_upper_bound if spec.cost_upper_bound else 0.
        self.model.observe(features.values, ratio - 1., noise_variance=1. if confidence == "exact" else 4.)
        self.observations[spec.query_id] = dict(cost=cost, confidence=confidence, cap=spec.cost_upper_bound)

    def predict(self, spec, features):
        if features.query_id != spec.query_id:
            raise ValueError("cost feature request mismatch")
        x = self.model.vector(features.values)
        return float(spec.cost_upper_bound * np.clip(1. + x @ self.model.mean, 0., 1.))


def budget_distribution(values, costs, *, remaining_budget, remaining_windows,
                        epsilon=0.25, tau=1., value_scale=1.):
    """Index zero is empty. Solve the one-dimensional convex cost dual.

    Values/costs are original units; metadata retains every normalization and
    b=remaining_budget/remaining_windows. lambda_c is in normalized units.
    """
    values, costs = np.asarray(values, dtype=np.float64), np.asarray(costs, dtype=np.float64)
    if (values.ndim != 1 or costs.shape != values.shape or not len(values)
            or not np.isfinite(values).all() or not np.isfinite(costs).all() or (costs < 0).any()
            or values[0] != 0 or costs[0] != 0):
        raise ValueError("finite candidates with explicit zero-value, zero-cost empty required")
    if (not math.isfinite(remaining_budget) or remaining_budget < 0 or type(remaining_windows) is not int
            or remaining_windows < 1 or tau <= 0 or not math.isfinite(tau)
            or value_scale <= 0 or not math.isfinite(value_scale) or epsilon != 0.25):
        raise ValueError("invalid v1 budget/normalization settings")
    n = len(values) - 1
    prior = np.r_[0.5, np.full(n, 0.5 / n)] if n else np.ones(1)
    cost_scale = float(costs[1:].mean()) if n and costs[1:].mean() > 0 else 1.
    normalized_cost = costs / cost_scale
    budget = remaining_budget / remaining_windows
    b = budget / cost_scale
    logits = np.log(prior) + epsilon * values / (tau * value_scale)
    def probabilities(lam, allowed=None):
        z = logits - lam * normalized_cost / tau
        if allowed is not None:
            z = np.where(allowed, z, -np.inf)
        w = np.exp(z - z.max())
        return w / w.sum()
    p, lam = probabilities(0.), 0.
    if float(p @ normalized_cost) > b:
        if b == 0:
            p, lam = probabilities(0., costs == 0), math.inf
        else:
            lo, hi = 0., 1.
            while float(probabilities(hi) @ normalized_cost) > b:
                hi *= 2
                if not math.isfinite(hi):
                    raise ArithmeticError("cost dual failed to bracket")
            for _ in range(100):
                mid = (lo + hi) / 2
                if float(probabilities(mid) @ normalized_cost) > b:
                    lo = mid
                else:
                    hi = mid
            lam, p = hi, probabilities(hi)
    return p, dict(lambda_c=None if math.isinf(lam) else lam, zero_budget_limit=math.isinf(lam),
        expected_cost=float(p @ costs), per_window_budget=budget, remaining_windows=remaining_windows,
        value_scale=value_scale, cost_scale=cost_scale, tau=tau, epsilon=epsilon, prior=prior.tolist())


class AcquisitionPolicy:
    def __init__(self, posterior, cost_model, *, rng):
        self.posterior, self.cost_model, self.rng = posterior, cost_model, rng
        self.last_decision = None

    def choose(self, candidates, feature_rows, *, remaining_budget, remaining_windows, random_control=False):
        # Deduplicate real request IDs, not states (real repeated requests cost again).
        unique = {}
        for spec in candidates:
            if type(spec) is not PublicQuerySpec:
                raise TypeError("public query specs only")
            if spec.query_id in unique and unique[spec.query_id] != spec:
                raise ValueError("conflicting duplicate public request")
            if spec.cost_upper_bound <= remaining_budget:
                unique[spec.query_id] = spec
        candidates = tuple(unique.values())
        beta = self.posterior.model.sample(self.rng) if candidates and not random_control else None
        values, costs, scales = [0.], [0.], []
        for spec in candidates:
            row = feature_rows.get(spec.query_id)
            if row is None and random_control:
                values.append(0.); costs.append(float(spec.cost_upper_bound)); continue
            if row is None or row.query_id != spec.query_id or row.round_id != self.posterior.round_id:
                raise ValueError("missing or stale pre-purchase features")
            x = self.posterior.model.vector(row.values)
            values.append(0. if random_control else float(x @ beta))
            costs.append(self.cost_model.predict(spec, row))
            scales.append(self.posterior.model.prior_predictive_std(x))
        scale = float(np.sqrt(np.mean(np.square(scales)))) if scales else 1.
        p, metadata = budget_distribution(values, costs, remaining_budget=remaining_budget,
            remaining_windows=remaining_windows, value_scale=scale)
        index = int(self.rng.choice(len(p), p=p))  # exactly one draw, including empty
        ids = [None] + [s.query_id for s in candidates]
        self.last_decision = dict(metadata, query_ids=ids, probabilities=p.tolist(),
            public_cost_caps=[0] + [s.cost_upper_bound for s in candidates],
            sampled_values=values, expected_costs=costs, selected=ids[index], decision_windows=1,
            beta=None if beta is None else beta.tolist(), round_id=self.posterior.round_id,
            prior_precision=1., noise_variance=self.posterior.model.noise_variance,
            noise_estimation="fixed_prior_noise", pending_new_observations=len(self.posterior.observations),
            reweight_observations_separate=len(self.posterior.reweight_observations),
            cost_observations_by_confidence={c: sum(r['confidence'] == c for r in self.cost_model.observations.values())
                                            for c in ('exact', 'estimated')})
        return ids[index]


def select_and_acquire(broker, student_snapshot, policy, feature_rows, *, remaining_windows, random_control=False):
    """Privileged orchestration boundary. Reservation/reveal happens in the broker.

    The guarded selector closure receives only copied public candidates and
    budget/features. Even a cost estimate of zero cannot bypass the ledger cap.
    """
    remaining = broker.ledger.remaining
    candidates = broker.list_candidates(student_snapshot, broker.ledger.owned_ids, remaining)
    selected, trace = select_public(candidates, lambda public: policy.choose(public, feature_rows,
        remaining_budget=remaining, remaining_windows=remaining_windows, random_control=random_control))
    package = None if selected is None else broker.acquire(selected)
    return package, policy.last_decision, trace
