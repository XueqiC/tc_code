"""Auditable complete-row exposure plans for the K=32 hard-label adaptations."""
from collections import Counter

import numpy as np


SAD_CURRICULUM = dict(
    name='short_to_long_trajectory_each_pass',
    difficulty='number of teacher turns in the verified trajectory',
    ordering='ascending trajectory length; seeded permutation breaks equal-length row ties',
    weighting='unchanged span_ce(method="sad"); token-weighted row reduction per update',
    source='docs/2026-09-15-user-direction-unified-method-and-theory-zh.txt:232',
    local_passage='对 reasoning/action 分别对齐教师分布，并结合课程机制',
    paper_passage=None,
    fidelity='unverified schedule adaptation: original curriculum passage is not vendored',
    deviation='Local notes identify a curriculum but specify no difficulty metric, pacing, or stages. '
              'Use a static short-to-long ordering within every exhaustive pass, with full coverage. '
              'This is an explicit fallback, not a verified reproduction of the paper schedule.',
    rejected_alternatives=[
        'shuffle only: no curriculum mechanism',
        'growing prefixes/repeated easy rows: changes exhaustive per-row exposure',
        'student NLL difficulty: introduces another model-scoring phase absent from local SAD notes',
        'reason-only then action-only: changes the required two-group hard-label objective'],
)


def exposure_schedule(rows, costs, config, seed, method):
    """Exactly 3 and 10 full passes; complete turns and pi1 token batching."""
    endpoints, budget = config['exposure_passes'], config['supervised_tokens_per_update']
    if (endpoints != [3, 10] or len(rows) != len(costs) or not costs
            or any(type(c) is not int or c < 1 for c in costs)
            or type(budget) is not int or budget < 1 or method not in {'smartad', 'sad', 'ce', 'kang'}):
        raise ValueError('K=32 needs positive costs/budget, known method, and endpoints [3, 10]')
    lengths = Counter(r.package_id for r in rows)
    difficulty = [lengths[r.package_id] for r in rows]
    rng = np.random.default_rng(seed)
    batches, orders, pending, tokens, total = [], [], [], 0, 0
    for epoch in range(1, endpoints[-1]+1):
        order = rng.permutation(len(rows)).tolist()
        if method == 'sad':
            order.sort(key=lambda i: difficulty[i])  # stable: seed breaks ties
        orders.append(order)
        for offset, i in enumerate(order):
            pending.append(i)
            tokens += costs[i]
            endpoint = epoch if epoch in endpoints and offset == len(order)-1 else None
            if tokens >= budget or endpoint is not None:
                total += tokens
                batches.append(dict(indices=pending, supervised_tokens=tokens,
                                    cumulative_tokens=total, endpoint=endpoint))
                pending, tokens = [], 0
    return dict(rule='exhaustive complete-turn passes; flush at exposure endpoints', seed=seed,
        costs=list(costs), bank_supervised_tokens=sum(costs),
        target_tokens=[p*sum(costs) for p in endpoints], batches=batches, pass_orders=orders,
        curriculum=dict(SAD_CURRICULUM, row_difficulty=difficulty) if method == 'sad' else None)
