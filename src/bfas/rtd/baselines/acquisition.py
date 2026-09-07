"""Ledger-independent B1 procurement primitives for later full-bank orchestration.

The sealed protocol and the cost-public retrospective protocol are deliberately
different schemas. Neither samples new student occupancy or authorizes a teacher
API. Both select whole L=1 requests without inspecting teacher text or reward.
"""
from dataclasses import dataclass

import numpy as np

from ..acquisition import AcquisitionPolicy, CostRegressor, ValuePosterior, select_and_acquire


@dataclass(frozen=True)
class AcquisitionConfig:
    version: str = "c27-acquisition-v1"
    protocol: str = "sealed_random_L1"
    seed: int = 0
    replay_length: int = 1
    windows_per_round: int = 4
    max_new_packages: int = 1
    empty_prior: float = .5
    new_teacher_calls: bool = False

    def __post_init__(self):
        if (self.version != "c27-acquisition-v1" or self.protocol not in {"sealed_random_L1", "historical_cost_public"}
                or self.seed != 0 or self.replay_length != 1 or self.windows_per_round != 4
                or self.max_new_packages != 1 or self.empty_prior != .5 or self.new_teacher_calls is not False):
            raise ValueError("unsupported B1 acquisition protocol")


def random_policy(*, round_number, rng):
    """No learned value or teacher/source feature predictors; public-cap costs."""
    return AcquisitionPolicy(ValuePosterior(1, round_id=f"r{round_number}"), CostRegressor(1), rng=rng)


def sealed_random_window(broker, snapshot, policy, journal, *, round_number, step):
    if step not in (1, 4, 7, 10):
        raise ValueError("four fixed sealed acquisition windows per round")
    candidates = broker.list_candidates(snapshot, broker.ledger.owned_ids, broker.ledger.remaining)
    if any(spec.L != 1 for spec in candidates):
        raise ValueError("B1-replay-L1 cannot buy unrecorded continuation lengths")
    package, decision, trace = select_and_acquire(broker, snapshot, policy, {},
        remaining_windows=sum(s >= step for s in (1, 4, 7, 10)), random_control=True)
    journal.append("baseline_random_purchase", round=round_number, step=step, protocol="sealed_random_L1",
        distribution=decision, selector_trace=trace, learned_value=False,
        cost_predictor="public_cap_without_source_features", actual_spend=broker.ledger.spent)
    return package


def historical_cost_allocation(catalog, budget, *, seed=0):
    """Pr(q|owned,b)=1/|F(owned,b)|; F uses publicly disclosed historical costs.

    The input contains query/dependency/cost/confidence only, never payload text.
    Draw once uniformly from dependency-closed affordable whole packages, charge
    the published cost and stop when F is empty. This is NOT sealed selection.
    """
    if type(budget) is not int or budget < 0 or seed != 0:
        raise ValueError("nonnegative historical budget and seed0 required")
    if any(set(row) != {"query_id", "dependencies", "cost", "confidence"} for row in catalog):
        raise ValueError("historical public cost directory must not contain teacher text/features")
    by_id = {r["query_id"]: r for r in catalog}
    if len(by_id) != len(catalog) or any(type(r["cost"]) is not int or r["cost"] < 0
            or r["confidence"] not in {"exact", "estimated"} or not set(r["dependencies"]) <= set(by_id) for r in catalog):
        raise ValueError("invalid historical cost/dependency directory")
    rng, owned, rows, remaining = np.random.default_rng(seed), set(), [], budget
    while True:
        feasible = sorted(q for q, r in by_id.items() if q not in owned and set(r["dependencies"]) <= owned and r["cost"] <= remaining)
        if not feasible:
            break
        q = feasible[int(rng.integers(len(feasible)))]
        rows.append(dict(query_id=q, feasible=feasible, probability=1/len(feasible),
                         remaining_before=remaining, cost=by_id[q]["cost"]))
        remaining -= by_id[q]["cost"]
        owned.add(q)
    return dict(protocol="historical_cost_public", distribution="Pr(q|owned,b)=1/|F(owned,b)|",
                draws=rows, owned=sorted(owned), spent=budget-remaining, unspent=remaining)
