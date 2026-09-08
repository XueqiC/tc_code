"""Public-only Bayesian acquisition and entropy-budget dual (9).

Only revealed request-level labels/usage update regressors. Selection sees
PublicQuerySpec and pre-purchase numerical features, never a broker payload.
"""
from dataclasses import dataclass
import math

import numpy as np

from .insertion import InsertionLabel, LabelType
from .selector import PublicQuerySpec, select_public

TRAINING_CONTEXT_FEATURES = ('round_fraction', 'step_fraction', 'purchased_count',
    'purchased_state_count', 'purchased_projection_norm', 'alpha_mean', 'alpha_std',
    'alpha_min', 'alpha_max', 'feedback_age')
CONTEXTUAL_DIMENSION = 39 + len(TRAINING_CONTEXT_FEATURES)


@dataclass(frozen=True)
class PrePurchaseFeatures:
    query_id: str
    values: tuple[float, ...]
    round_id: str

    def __post_init__(self):
        if not self.query_id or not self.round_id or not self.values or not np.isfinite(self.values).all():
            raise ValueError("finite pre-purchase features and request/round identity required")

    @classmethod
    def from_public(cls, spec: PublicQuerySpec, *, round_id, coverage, progress, support_return, context=()):
        f = spec.features
        if f.projection is None or f.source_logprob is None or f.source_length is None:
            raise ValueError("missing source features; run legal source sampling first")
        return cls(spec.query_id, (*f.projection, f.source_logprob, f.source_length, float(spec.L),
            float(coverage), float(progress), float(support_return), *context, 1.), round_id)


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
    def __init__(self, dimension, *, round_id, noise_variance=1., contextual_shrinkage=False):
        self.dimension, self.noise_variance = dimension, noise_variance
        self.model = BayesianLinearRegression(dimension, noise_variance=noise_variance)
        self.observations = {}
        self.reweight_observations = {}
        self.history = []
        self.drift_history = []
        self.contextual_shrinkage = contextual_shrinkage
        self.predictive_history = []
        self.shrinkage_factor = .05 if contextual_shrinkage else 1.
        self.unshrunk_information = np.zeros(dimension)
        self.begin_round(round_id)

    def begin_round(self, round_id):
        if not round_id:
            raise ValueError("round identity required")
        self.round_id = round_id

    def observe(self, features: PrePurchaseFeatures, label: InsertionLabel, *, _fit=True):
        if (features.query_id != label.query_id or
                features.round_id != self.round_id or label.round_id != self.round_id):
            raise ValueError("pre-purchase feature/request/round mismatch; stale labels excluded")
        target = self.observations if label.label_type is LabelType.PENDING_NEW else self.reweight_observations
        key = label.query_id if label.label_type is LabelType.PENDING_NEW else (label.round_id, label.query_id, label.reference_hash)
        if key in target:
            raise ValueError("duplicate request-level label")
        self.model.vector(features.values)
        noise = getattr(label, 'variance', None)
        if label.label_type is LabelType.PENDING_NEW and _fit:
            self.model.observe(features.values, label.value, noise_variance=noise)
        target[key] = (features, label)
        if not hasattr(self, 'history'):  # v1.0 recovery pickles predate persistent history
            self.history = []
        self.history.append(dict(round_id=self.round_id, features=features, label=label,
                                 noise_variance=self.noise_variance if noise is None else noise))

    def observe_batch(self, rows, labels, covariance):
        """Generalized least squares retains shared-rollout label covariance."""
        ids = list(labels)
        if not ids:
            return
        for q in ids:
            row, label = rows[q], labels[q]
            if (q in self.observations or q != row.query_id or q != label.query_id or
                    row.round_id != self.round_id or label.round_id != self.round_id or
                    label.label_type is not LabelType.PENDING_NEW):
                raise ValueError('duplicate or mismatched batch request/round label')
        X = np.array([self.model.vector(rows[q].values) for q in ids])
        y = np.array([labels[q].value for q in ids])
        noise = np.asarray(covariance, dtype=np.float64)
        if (noise.shape != (len(ids), len(ids)) or not np.isfinite(noise).all()
                or not np.allclose(noise, noise.T)
                or not np.allclose(np.diag(noise), [labels[q].variance for q in ids])):
            raise ValueError('label-aligned heteroscedastic covariance required')
        np.linalg.cholesky(noise)
        if getattr(self, 'contextual_shrinkage', False):
            # Predictions are saved BEFORE fitting this window. Assess only
            # purchased surrogate labels; task-validation rewards never enter.
            self.predictive_history.append(dict(round_id=self.round_id, query_ids=ids,
                predicted=(X @ self.model.mean).tolist(), observed=y.tolist()))
        precision = X.T @ np.linalg.solve(noise, X)
        information = X.T @ np.linalg.solve(noise, y)
        for q in ids:
            self.observe(rows[q], labels[q], _fit=False)
        self.model.precision += precision
        if getattr(self, 'contextual_shrinkage', False):
            self.unshrunk_information += information
            observed = np.array([v for row in self.predictive_history for v in row['observed']])
            predicted = np.array([v for row in self.predictive_history for v in row['predicted']])
            # Positive out-of-window explained variance is needed to relax the
            # 95% mean shrinkage. Small/constant samples retain strong shrinkage.
            variance = float(np.square(observed-observed.mean()).sum())
            skill = (max(0., 1-float(np.square(observed-predicted).sum())/variance)
                     if len(self.predictive_history) >= 4 and len(observed) >= 8 and variance > 1e-12 else 0.)
            self.shrinkage_factor = max(.05, min(1., skill))
            self.model.information = self.shrinkage_factor*self.unshrunk_information
        else:
            self.model.information += information

    def inflate_for_drift(self, *, query_id, old_value, new_value, variance, previous_variance=0.):
        """Preserve the mean and all observations; discount precision on drift.

        Reweight labels are diagnostics, not additional pending-new purchases.
        Inflation is monotone in excess squared drift over measurement noise.
        """
        if query_id not in self.observations:
            raise ValueError('drift needs a previously purchased observation')
        if not np.isfinite([old_value, new_value, variance, previous_variance]).all() or min(variance, previous_variance) < 0:
            raise ValueError('finite drift estimates and nonnegative variances required')
        delta = new_value - old_value
        noise = max(variance + previous_variance, 1e-12)
        factor = 1. + min(100., max(0., delta * delta / noise - 1.))
        self.model.precision /= factor
        self.model.information /= factor
        if getattr(self, 'contextual_shrinkage', False):
            self.unshrunk_information /= factor
        row = dict(round_id=self.round_id, query_id=query_id, old_value=old_value, new_value=new_value,
                   delta=delta, variance=variance, previous_variance=previous_variance,
                   uncertainty_inflation=factor)
        self.drift_history.append(row)
        return row


class LegacyValuePosterior(ValuePosterior):
    """Frozen v1.0 round-reset behavior, used only by legacy configurations."""
    def begin_round(self, round_id):
        super().begin_round(round_id)
        self.model = BayesianLinearRegression(self.dimension, noise_variance=self.noise_variance)
        self.observations = {}
        self.reweight_observations = {}
        self.history = []


class CostRegressor:
    """Regress revealed cost/cap, retaining exact/estimated observation flags.

    No observations: use the public cap. Thereafter ridge-shrink residuals
    around cap ratio=1. Estimated usage has fixed variance 4; exact variance 1.
    Predictions clipped to [0,cap] are expectations, not ledger reservations.
    """
    def __init__(self, dimension):
        self.model = BayesianLinearRegression(dimension)
        self.observations = {}
        self.round_id = None

    def begin_round(self, round_id):
        if not round_id:
            raise ValueError('round identity required')
        self.round_id = round_id

    def _values(self, features):
        # D9 context is for marginal learning value. Keep the existing cost
        # predictor and V0 budget decisions exactly on their original features.
        values = features.values
        if self.model.dimension == 39 and len(values) == CONTEXTUAL_DIMENSION:
            return (*values[:38], values[-1])
        return values

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
        self.model.observe(self._values(features), ratio - 1., noise_variance=1. if confidence == "exact" else 4.)
        self.observations[spec.query_id] = dict(cost=cost, confidence=confidence, cap=spec.cost_upper_bound,
                                              round_id=features.round_id)

    def predict(self, spec, features):
        if features.query_id != spec.query_id:
            raise ValueError("cost feature request mismatch")
        x = self.model.vector(self._values(features))
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


def window_budget(remaining, remaining_windows, candidates):
    """Carry actual unspent authorization; public cap floor avoids class lockout."""
    if type(remaining) is not int or remaining < 0 or type(remaining_windows) is not int or remaining_windows < 1:
        raise ValueError('nonnegative authorization and positive window count required')
    cap = max((c.cost_upper_bound for c in candidates if c.cost_upper_bound <= remaining), default=0)
    base = (remaining + remaining_windows - 1) // remaining_windows
    return min(remaining, max(base, cap))


def greedy_batch(ids, values, costs, *, budget, limit):
    """Deterministic value/cost greedy with a best-singleton safeguard.

    This is a local knapsack heuristic, not an optimality certificate. Nonpositive
    sampled marginal values leave exposure with old data. Zero-cost items still
    occupy one package slot.
    """
    order = sorted(range(len(ids)), key=lambda i: (-(values[i] / max(costs[i], 1e-12)), -values[i], ids[i]))
    chosen, spent = [], 0.
    for i in order:
        if values[i] > 0 and len(chosen) < limit and spent + costs[i] <= budget + 1e-9:
            chosen.append(i)
            spent += costs[i]
    singles = [i for i in order if costs[i] <= budget and values[i] > 0]
    if limit and singles:
        best = max(singles, key=lambda i: values[i])
        if values[best] > sum(values[i] for i in chosen):
            chosen = [best]
    return chosen


class BatchAcquisitionPolicy(AcquisitionPolicy):
    def choose_batch(self, candidates, feature_rows, *, remaining_budget, exposure_slots=40,
                     max_new_packages=20, random_control=False, previous_model=None):
        if (type(exposure_slots) is not int or type(max_new_packages) is not int
                or not 0 <= max_new_packages <= exposure_slots or exposure_slots < 1
                or not math.isfinite(remaining_budget) or remaining_budget < 0):
            raise ValueError('invalid batch exposure or budget')
        unique = {}
        for spec in candidates:
            if type(spec) is not PublicQuerySpec:
                raise TypeError('public query specs only')
            if spec.query_id in unique and unique[spec.query_id] != spec:
                raise ValueError('conflicting duplicate public request')
            if spec.cost_upper_bound <= remaining_budget:
                unique[spec.query_id] = spec
        specs = list(unique.values())
        ids = [c.query_id for c in specs]
        rows = [feature_rows[q] for q in ids]
        if any(r.query_id != q or r.round_id != self.posterior.round_id for q, r in zip(ids, rows)):
            raise ValueError('missing or stale pre-purchase features')
        model = self.posterior.model
        xs = np.asarray([model.vector(r.values) for r in rows]).reshape(-1, model.dimension)
        z = self.rng.standard_normal(model.dimension) if ids and not random_control else None
        beta = None if z is None else model.mean + np.linalg.solve(np.linalg.cholesky(model.precision).T, z)
        values = np.zeros(len(ids)) if beta is None else xs @ beta
        costs = np.asarray([self.cost_model.predict(c, r) for c, r in zip(specs, rows)])
        if random_control:
            # Random ordering under the same expected-cost/K/hard-cap rules; no
            # artificial empty mass and no value-dependent labels enter R0.
            chosen, cost = [], 0.
            for i in self.rng.permutation(len(ids)):
                if len(chosen) < max_new_packages and cost + costs[i] <= remaining_budget + 1e-9:
                    chosen.append(int(i)); cost += costs[i]
        else:
            chosen = greedy_batch(ids, values, costs, budget=remaining_budget, limit=max_new_packages)
        predicted = sum(float(values[i]) for i in chosen) / exposure_slots
        chosen_ids = [ids[i] for i in chosen]
        positive = [i for i in range(len(ids)) if random_control or values[i] > 0]
        skipped = set(positive) - set(chosen)
        binding = bool(skipped and any(sum(costs[i] for i in chosen) + costs[j] > remaining_budget + 1e-9 for j in skipped))
        reason = ('no_feasible_candidates' if not ids else 'package_limit' if len(chosen) == max_new_packages
                  else 'predicted_budget' if binding else 'nonpositive_sampled_gain' if skipped or len(positive) < len(ids)
                  else 'candidate_exhaustion')
        significance = dict(pair=None, difference=None, standard_error=None, z_score=None, significant=False,
                            threshold=1.96, method='posterior_difference_covariance',
                            units='unit_exposure_insertion_value',
                            previous_comparable_selected=None, decision_changed=False if random_control else None,
                            comparison='same_current_candidates_features_costs_budget_and_standard_normal_draw')
        if len(ids) >= 2:
            means = xs @ model.mean
            order = np.argsort(-means, kind='stable')
            a, b = order[:2]
            diff = xs[a] - xs[b]
            se = math.sqrt(max(0., float(diff @ np.linalg.solve(model.precision, diff))))
            delta = float(means[a] - means[b])
            score = delta / max(se, 1e-12)
            significance.update(pair=[ids[a], ids[b]], difference=delta, standard_error=se,
                                z_score=score, significant=score >= 1.96)
        if previous_model is not None and z is not None:
            previous_beta = previous_model.mean + np.linalg.solve(np.linalg.cholesky(previous_model.precision).T, z)
            old = greedy_batch(ids, xs @ previous_beta, costs, budget=remaining_budget, limit=max_new_packages)
            previous_ids = [ids[i] for i in old]
            significance.update(previous_comparable_selected=previous_ids,
                                decision_changed=set(previous_ids) != set(chosen_ids))
        self.last_decision = dict(acquisition_protocol='batch_common_reference_v1', round_id=self.posterior.round_id,
            query_ids=ids, selected=chosen_ids, sampled_values=values.tolist(), expected_costs=costs.tolist(),
            public_cost_caps=[c.cost_upper_bound for c in specs], beta=None if beta is None else beta.tolist(),
            posterior_standard_normal=None if z is None else z.tolist(), exposure_slots=exposure_slots,
            position_share=1./exposure_slots, max_new_packages=max_new_packages,
            window_budget=remaining_budget, predicted_cost=sum(float(costs[i]) for i in chosen),
            predicted_additive_gain=predicted, budget_binding=binding, stop_reason=reason,
            optimizer='random_order' if random_control else 'value_cost_greedy_best_singleton',
            value_difference_significance=significance, replay_prior_mass=None,
            replay_semantics='unfilled exposure uses old data; empty set continues training',
            pending_new_observations=len(self.posterior.observations), noise_estimation='rollout_heteroscedastic')
        return chosen_ids
