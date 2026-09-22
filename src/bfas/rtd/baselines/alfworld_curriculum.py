"""Auditable complete-row exposure plans for the K=32 hard-label adaptations."""
from collections import Counter
import math

import numpy as np


SAD_CURRICULUM = dict(
    name='short_to_long_trajectory_each_pass',
    difficulty='number of teacher turns in the verified trajectory',
    ordering='ascending trajectory length; seeded permutation breaks equal-length row ties',
    weighting='chosen SAD span loss (hyperparameters.sad_variant); token-weighted row reduction per update',
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


TRAJECTORY_COST_CURRICULUM = dict(
    name='trajectory_cost',
    difficulty='alpha * reason tokens + beta * action/final tokens per trajectory',
    ordering='ascending trajectory cost; seeded trajectory ties; original turn order within trajectory',
    gamma=0., entropy='unavailable for black-box teacher; omitted, not measured as zero',
    source='SAD arXiv:2505.13820v1 Eq. 7 / v5 Eq. 13',
    deviation='Length-only curriculum; teacher entropy unavailable; exhaustive passes and token-budget '
              'updates retain the matched ALFWorld protocol, not a claimed original pacing recipe.',
    token_scope='authored tokens through existing span masks; native boundary excluded from complexity',
)


def trajectory_costs(rows, tokenizer, *, alpha=1., beta=1.):
    """Cost each complete trajectory, using the current reasoning/action partition."""
    from .paper_losses import token_kinds
    if any(not math.isfinite(v) or v < 0 for v in (alpha, beta)):
        raise ValueError('curriculum alpha and beta must be finite and nonnegative')
    totals = {}
    for row in rows:
        _, kinds = token_kinds(tokenizer, row.target, benchmark=row.benchmark,
                               final_step=row.final_step, prompt=row.prompt)
        counts = totals.setdefault(row.package_id, dict(reason_tokens=0, action_tokens=0))
        counts['reason_tokens'] += kinds.count('reason')
        counts['action_tokens'] += sum(k in {'action', 'final'} for k in kinds)
    return {qid: dict(counts, cost=alpha*counts['reason_tokens'] + beta*counts['action_tokens'])
            for qid, counts in totals.items()}


def exposure_schedule(rows, costs, config, seed, method, *, sad_curriculum='turn_count',
                      trajectory_scores=None, alpha=1., beta=1.):
    """Exactly 3 and 10 full passes; complete turns and pi1 token batching."""
    endpoints, budget = config['exposure_passes'], config['supervised_tokens_per_update']
    if (endpoints != [3, 10] or len(rows) != len(costs) or not costs
            or any(type(c) is not int or c < 1 for c in costs)
            or type(budget) is not int or budget < 1 or method not in {'smartad', 'sad', 'ce', 'kang'}):
        raise ValueError('K=32 needs positive costs/budget, known method, and endpoints [3, 10]')
    lengths = Counter(r.package_id for r in rows)
    difficulty = [lengths[r.package_id] for r in rows]
    if sad_curriculum not in {'turn_count', 'trajectory_cost'} or (
            method != 'sad' and sad_curriculum != 'turn_count'):
        raise ValueError('sad_curriculum requires SAD and a known curriculum')
    groups = {}
    if method == 'sad' and sad_curriculum == 'trajectory_cost':
        if trajectory_scores is None or set(trajectory_scores) != set(lengths):
            raise ValueError('complete trajectory scores required for trajectory_cost')
        if any(not math.isfinite(v) or v < 0 for v in (alpha, beta)):
            raise ValueError('curriculum alpha and beta must be finite and nonnegative')
        for i, row in enumerate(rows):
            groups.setdefault(row.package_id, []).append(i)
        for qid, indices in groups.items():
            indices.sort(key=lambda i: rows[i].index)
            if [rows[i].index for i in indices] != list(range(len(indices))):
                raise ValueError('curriculum requires complete ordered trajectory turns')
            score = trajectory_scores[qid]
            if (any(type(score[k]) is not int or score[k] < 0 for k in ('reason_tokens', 'action_tokens'))
                    or score['cost'] != alpha*score['reason_tokens'] + beta*score['action_tokens']):
                raise ValueError('invalid trajectory cost')
        difficulty = [trajectory_scores[r.package_id]['cost'] for r in rows]
    rng = np.random.default_rng(seed)
    batches, orders, pending, tokens, total = [], [], [], 0, 0
    for epoch in range(1, endpoints[-1]+1):
        if groups:
            packages = sorted(groups)
            packages = [packages[i] for i in rng.permutation(len(packages))]
            packages.sort(key=lambda q: trajectory_scores[q]['cost'])
            order = [i for q in packages for i in groups[q]]
        else:
            order = rng.permutation(len(rows)).tolist()
        if method == 'sad' and not groups:
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
    curriculum = (dict(TRAJECTORY_COST_CURRICULUM, alpha=alpha, beta=beta,
                       trajectories=trajectory_scores, row_difficulty=difficulty) if groups else
                  dict(SAD_CURRICULUM, row_difficulty=difficulty) if method == 'sad' else None)
    return dict(rule='exhaustive complete-turn passes; flush at exposure endpoints', seed=seed,
        costs=list(costs), bank_supervised_tokens=sum(costs),
        target_tokens=[p*sum(costs) for p in endpoints], batches=batches, pass_orders=orders,
        curriculum=curriculum)
