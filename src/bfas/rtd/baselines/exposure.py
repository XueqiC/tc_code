"""Frozen complete-action exposure schedules; never stop early on realized tokens."""
from dataclasses import dataclass
import math

import numpy as np


VT_FORMULA = "Z = sum_j p_j/c_j;  q_i = (p_i/c_i)/Z;  rho_i = p_i/q_i = Z*c_i"


@dataclass(frozen=True)
class ExposureDistribution:
    """V-T: Z = sum_j p_j/c_j;  q_i = (p_i/c_i)/Z;  rho_i = p_i/q_i = Z*c_i.

    Each draw contributes rho_i * L_i, averaged by the predetermined draw count;
    B4 additionally multiplies L_i by w_i. Full-sequence log probability is a
    SUM, never divided by action length. p,c,q,rho are frozen and stop-gradient
    within a round. No clipping, minibatch self-normalization, token-based early
    stopping, truncation, or resampling to fit a realized budget is permitted.
    E_q[c]=1/Z. T_update includes supervised and B3 PG score/backprop inputs;
    outer gJ/VJP, pilot/reference and generation are separately accounted.
    """
    p: tuple
    costs: tuple
    view: str

    def __post_init__(self):
        if (self.view not in {"slots", "tokens"} or not self.p or len(self.p) != len(self.costs)
                or any(not math.isfinite(v) or v <= 0 for v in self.p)
                or any(not math.isfinite(v) or v < 0 or (v == 0 and self.view == "tokens") for v in self.costs)
                or not math.isclose(sum(self.p), 1., rel_tol=1e-10, abs_tol=1e-10)):
            raise ValueError("positive finite costs/probabilities summing to one required")

    @property
    def Z(self):
        return sum(p / c for p, c in zip(self.p, self.costs)) if all(self.costs) else None

    @property
    def q(self):
        return tuple(p / c / self.Z for p, c in zip(self.p, self.costs)) if self.view == "tokens" else self.p

    @property
    def rho(self):
        return tuple(p / q for p, q in zip(self.p, self.q))

    @property
    def expected_cost(self):
        return sum(q * c for q, c in zip(self.q, self.costs))

    def journal(self):
        return dict(formula=VT_FORMULA, view=self.view, p=list(self.p), c=list(self.costs),
                    q=list(self.q), rho=list(self.rho), Z=self.Z, expected_cost=self.expected_cost,
                    reduction="predetermined_draw_mean", round_frozen=True, stop_gradient=True)

    def schedule(self, *, commits, seed, slots=96, target_tokens=None, pg_reserved_tokens=0):
        if type(commits) is not int or commits < 4 or commits % 4:
            raise ValueError("four feedback windows require divisible commit count")
        if self.view == "slots":
            draws = slots
        else:
            if target_tokens is None or target_tokens <= pg_reserved_tokens:
                raise ValueError("infeasible: matched feedback consumes the T_update target")
            draws = int(math.floor((target_tokens - pg_reserved_tokens) / self.expected_cost + .5))
        if draws < commits:
            raise ValueError("infeasible: fewer full draws than commits; increase the common token target")
        sizes = [draws // commits + (i < draws % commits) for i in range(commits)]
        rng = np.random.default_rng(seed)
        # Freeze the entire schedule before any student update or feedback.
        indices = rng.choice(len(self.p), size=draws, p=self.q).tolist()
        batches, offset = [], 0
        for size in sizes:
            batches.append(indices[offset:offset + size])
            offset += size
        return batches


def budget_match(targets, actuals, max_microbatch_costs, *, relative_tolerance=.01):
    if not (len(targets) == len(actuals) == len(max_microbatch_costs) == 3) or any(t <= 0 for t in targets):
        raise ValueError("three positive round targets and measured costs required")
    errors = [a - t for t, a in zip(targets, actuals)]
    relative = abs(sum(actuals) - sum(targets)) / sum(targets)
    matched = relative <= relative_tolerance and all(abs(e) <= b for e, b in zip(errors, max_microbatch_costs))
    return dict(matched=matched, round_errors=errors, final_relative_error=relative,
                target_tokens=list(targets), actual_tokens=list(actuals),
                round_max_microbatch=list(max_microbatch_costs), relative_tolerance=relative_tolerance)
