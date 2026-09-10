"""Rev 3.1 diagnostics. No labels from this module enter training or acquisition."""
from contextlib import nullcontext
from dataclasses import replace
import math

import numpy as np
import torch

from ..behavior.deltas import tensor_state_hash
from .alpha_d import BatchReference, SourcePair, cpu_detached, estimate, gram
from .functional_step import gradients, snapshot
from .joint_surrogate import control, execute_update
from .persistence import digest
from .return_gradient import ActionTrace, IncompleteRolloutError
from .source_scoring import source_gradient_pair
from .transport import Behavior, SourceSample

ESTIMATORS = ('hard2', 'soft', 'hard8', 'syntax_filtered_hard2')


def options(config):
    value = config.get('metrics_v11', {})
    if not isinstance(value, dict) or set(value)-{'enabled', 'variance_resamples', 'archive_windows', 'short_fold'}:
        raise ValueError('invalid metrics_v11 options')
    result = dict(enabled=False, variance_resamples=4, archive_windows=True, short_fold='repeat') | value
    if type(result['enabled']) is not bool or type(result['archive_windows']) is not bool:
        raise ValueError('metrics switches must be booleans')
    if type(result['variance_resamples']) is not int or result['variance_resamples'] < 2:
        raise ValueError('variance_resamples must be an integer R >= 2')
    if result['short_fold'] not in {'repeat', 'error'}:
        raise ValueError('short_fold must be repeat or error')
    return result


def freeze_tasks(parents, *, short_fold='repeat', states=None):
    """Twenty fixed slots per fold; shortage/duplicates are explicit, never hidden."""
    if len({p['parent_hash'] for p in parents}) != len(parents):
        raise ValueError('duplicate support parent')
    folds = {}
    for fold in (0, 1):
        pool = sorted((p for p in parents if int(p['parent_hash'], 16) % 2 == fold
                       and (states is None or p['parent_hash'] in states)),
                      key=lambda p: (digest(['rtd-v11-fixed-tasks', p['parent_hash']]), p['parent_hash']))
        if not pool or (len(pool) < 20 and short_fold == 'error'):
            raise ValueError('fixed metrics need 20 states per fold; support fold is too small')
        rows = [dict(slot=i, parent_hash=pool[i % len(pool)]['parent_hash'],
                     task_id=pool[i % len(pool)]['official_id'], state_locator='task_start') for i in range(20)]
        if states is not None:
            for row in rows:
                if row['parent_hash'] not in states:
                    raise ValueError('fixed support state unavailable at run start')
                row['state_hash'] = states[row['parent_hash']].state_hash
        folds[str(fold)] = dict(states=rows, unique_states=len({r['parent_hash'] for r in rows}),
                               repeated_slots=max(0, 20-len(pool)))
    value = dict(schema='rtd-v11-fixed-task-set-1', selection='support_parent_content_hash_order_at_run_start',
                 states_per_fold=20, samples_per_state=4, temperature=1., top_p=1.,
                 excluded_unavailable_parents=sorted(p['parent_hash'] for p in parents
                     if states is not None and p['parent_hash'] not in states),
                 short_fold=short_fold, folds=folds)
    return value | dict(hash=digest(value))


def task_rows(fixed, support):
    if digest({k: v for k, v in fixed.items() if k != 'hash'}) != fixed['hash']:
        raise ValueError('fixed task manifest changed')
    rows = []
    for fold in (0, 1):
        for row in fixed['folds'][str(fold)]['states']:
            h = row['parent_hash']
            if h not in support.states or support.parents[h] != row['task_id'] or int(h, 16) % 2 != fold:
                raise ValueError('fixed support state missing or changed; never substitute a task')
            if row.get('state_hash') != support.states[h].state_hash:
                raise ValueError('fixed full-state content changed; never substitute a task')
            rows.append(row | dict(fold=fold, key=f'{fold}:{row["slot"]}:{h}', state_hash=support.states[h].state_hash))
    return rows


def generator_for(parameters, *identity):
    seed = int(digest(['v11_diagnostic_only', *identity])[:15], 16)
    return torch.Generator(device=next(iter(parameters.values())).device).manual_seed(seed)


def action_scope(backend, support, parent):
    return (backend.action_limit(support.categories[support.parents[parent]])
            if hasattr(backend, 'action_limit') else nullcontext())


def syntax_success(checker, text):
    verdict = checker.check_syntax(text)
    if type(verdict.get('valid')) is not bool:
        raise ValueError('syntax checker infrastructure failure is not a parse failure')
    return verdict['valid']


def parse_rate(rows):
    if not rows:
        raise ValueError('parse rate needs samples')
    if any(type(r['success']) is not bool for r in rows):
        raise ValueError('parse verdicts must be booleans')
    failures = sum(not r['success'] for r in rows)
    return dict(samples=len(rows), failures=failures, failure_rate=failures/len(rows))


def parse_battery(fixed, support, backend, parameters, checker, *, identity):
    from .generation_batch import action_cap, sample_actions
    rng = generator_for(parameters, identity, 'parse')
    draws = []
    rows = task_rows(fixed, support)
    scope = nullcontext()
    if getattr(backend, 'generation_batch', None) is not None:
        requests = [(support.states[row['parent_hash']].prompt, 4,
            action_cap(backend, support.categories[support.parents[row['parent_hash']]])) for row in rows]
        scope = backend.prefetch_actions(requests, parameters, rng)
    with scope:
        for row in rows:
            with action_scope(backend, support, row['parent_hash']):
                for k, action in enumerate(sample_actions(backend, support.states[row['parent_hash']].prompt, 4,
                                                          parameters, rng, temperature=1., top_p=1.)):
                    draws.append(row | dict(draw=k, success=(support.syntax_success(action.text, support.states[row['parent_hash']])
                        if hasattr(support, 'syntax_success') else syntax_success(checker, action.text)),
                        action_hash=digest(list(action.action_ids)), truncated=action.truncated))
    return dict(**parse_rate(draws), by_fold={str(f): parse_rate([r for r in draws if r['fold'] == f]) for f in (0, 1)},
                fixed_task_set_hash=fixed['hash'], K=4, temperature=1., draws=draws,
                checker_version=getattr(checker, 'checker_version', 'injected-test-checker'))


class GreedyBackend:
    diagnostic_only = True
    """Evaluation-only proxy; same task runner/checker, deterministic decoding.

    Production HF uses cached generation. Tiny models use the functional logits
    path. The probability recorded is the model likelihood, never an RL score.
    """
    def __init__(self, backend):
        self.backend = backend

    def __getattr__(self, name):
        return getattr(self.backend, name)

    def sample_action(self, prompt, parameters, generator, **_):
        from .runtime import HFGenerateBackend, installed_parameters
        b = self.backend
        prompt_ids = tuple(b.tokenizer.encode(prompt, add_special_tokens=False))
        limit = min(b.max_action_tokens, b.max_context_tokens-len(prompt_ids))
        if not prompt_ids or limit < 1:
            raise IncompleteRolloutError('complete greedy prompt exceeds context')
        eos, device = b.tokenizer.eos_token_id, next(iter(parameters.values())).device
        from .student import termination_ids
        stops = termination_ids(b)
        tokens, scores = [], []
        if isinstance(b, HFGenerateBackend):
            from transformers import GenerationConfig
            settings = GenerationConfig(do_sample=False, num_beams=1, max_new_tokens=limit,
                eos_token_id=list(stops), pad_token_id=eos, bos_token_id=b.tokenizer.bos_token_id,
                repetition_penalty=1., use_cache=True, return_dict_in_generate=True, output_scores=True)
            with b.measured('diagnostic_greedy_generation'), installed_parameters(b.model, parameters), torch.no_grad():
                output = b.model.generate(input_ids=torch.tensor([prompt_ids], device=device),
                    attention_mask=torch.ones((1, len(prompt_ids)), device=device, dtype=torch.long),
                    generation_config=settings)
                tokens = output.sequences[0, len(prompt_ids):].tolist()
                scores = [float(logits[0].float().log_softmax(-1)[t]) for logits, t in zip(output.scores, tokens)]
        else:
            with torch.no_grad():
                for _ in range(limit):
                    logits = b._logits(parameters, torch.tensor([prompt_ids+tuple(tokens)], device=device))[0, -1]
                    logp = logits.log_softmax(-1)
                    token = int(logits.argmax())
                    tokens.append(token); scores.append(float(logp[token]))
                    if token in stops:
                        break
        truncated = tokens[-1] not in stops
        eos = eos if truncated else tokens[-1]
        return ActionTrace(prompt_ids, tuple(tokens), eos,
            b.tokenizer.decode(tokens if truncated else tokens[:-1], skip_special_tokens=False),
            sum(scores), b.backend_id, b.identity(parameters), tuple(scores),
            dict(temperature=0., do_sample=False, diagnostic_only=True), truncated=truncated)


def greedy_success(fixed, support, backend, parameters, checker, *, identity):
    greedy = GreedyBackend(backend)
    rng = generator_for(parameters, identity, 'greedy')
    result, cache = {}, {}
    for row in task_rows(fixed, support):
        parent = row['parent_hash']
        if parent not in cache:
            rollout = support.feedback(parent, greedy, parameters, rng, checker)
            if not rollout.from_task_start or rollout.policy_id != backend.identity(parameters):
                raise ValueError('greedy metric requires a full task at the evaluated checkpoint')
            if rollout.reward not in (0., 1.):
                raise ValueError('repair/damage requires binary task success')
            cache[parent] = bool(rollout.reward)
        result[row['key']] = cache[parent]
    return result


def repair_damage(before, after):
    if not before or before.keys() != after.keys():
        raise ValueError('repair/damage requires the identical fixed task set')
    if any(type(v) is not bool for v in (*before.values(), *after.values())):
        raise ValueError('repair/damage requires boolean greedy success')
    repair = sum(not before[k] and after[k] for k in before)
    damage = sum(before[k] and not after[k] for k in before)
    return dict(tasks=len(before), before_success=sum(before.values()), after_success=sum(after.values()),
                repaired=repair, damaged=damage, retained=sum(before[k] and after[k] for k in before),
                still_failed=sum(not before[k] and not after[k] for k in before), net_repair=repair-damage)


def norm_variance(norms):
    values = np.asarray(norms, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError('norm variance requires R >= 2 finite nonnegative norms')
    return dict(R=len(values), norms=values.tolist(), mean_norm=float(values.mean()),
                variance=float(values.var(ddof=0)), correction=0,
                quantity='L2 norm of one preconditioned update displacement; not covariance trace')


def correlation(predictions, gains):
    x, y = np.asarray(predictions, dtype=float), np.asarray(gains, dtype=float)
    if x.ndim != 1 or x.shape != y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('correlation requires aligned finite pairs')
    reason = 'fewer_than_two_pairs' if len(x) < 2 else 'constant_prediction_or_gain' if x.std() == 0 or y.std() == 0 else None
    return dict(n=len(x), pearson=None if reason else float(np.corrcoef(x, y)[0, 1]), reason=reason)


def prediction_errors(predictions, gains):
    c = correlation(predictions, gains)
    error = np.asarray(gains)-np.asarray(predictions)
    return dict(correlation=c, n=len(error), mean_error=float(error.mean()) if len(error) else None,
                mae=float(np.abs(error).mean()) if len(error) else None,
                rmse=float(np.sqrt(np.mean(error**2))) if len(error) else None)


def paired_correlations(rows):
    result = {}
    for kind in sorted({r['comparison'] for r in rows}):
        subset = [r for r in rows if r['comparison'] == kind]
        for r in subset:
            if (not r['validation_only'] or r['used_for_posterior'] or r['selection_feedback_reused'] or
                    r['full']['batch_id'] == r['control']['batch_id']):
                raise ValueError('prediction validation cannot reuse selection feedback')
            selection = r.get('selection_feedback_batch')
            if any(side['batch_id'] == side.get('selection_feedback_batch', selection) for side in (r['full'], r['control'])):
                raise ValueError('prediction validation reused the selection rollout batch')
        result[kind] = prediction_errors([r['prediction'] for r in subset], [r['realised_paired_gain'] for r in subset])
        if kind == 'z_direction':
            valid = [r for r in subset if r['probe_d'] != 0]
            result['z_hat_vs_realised_directional_gain'] = correlation(
                [r['z_hat'] for r in valid], [r['realised_paired_gain']/r['probe_d'] for r in valid])
    return result


def window_metrics(steps):
    seen, rows, previous = set(), [], 0
    for s in sorted(steps, key=lambda s: (s['round'], s['step'])):
        key = (s['round'], s['step'])
        if key in seen:
            raise ValueError('duplicate committed step; compute attempts are not windows')
        seen.add(key)
        if not s['decision']:
            continue
        spend, ceiling = s['actual_spend'], s['authorized_budget']
        if not previous <= spend <= ceiling:
            raise ValueError('nonmonotonic spend or exceeded authorization')
        selection = s.get('selection') or {}
        controller = s.get('alpha_d', {})
        reasons = list(selection.get('binding_reasons') or [])
        stop = selection.get('stop_reason', 'unrecorded')
        if stop not in reasons:
            reasons.append(stop)
        if selection.get('budget_binding') and 'budget' not in reasons:
            reasons.append('budget')
        limit = selection.get('max_new_packages', s.get('max_new_packages'))
        if limit is not None and len(s['selected']) >= limit and 'package_limit' not in reasons:
            reasons.append('package_limit')
        if s['remaining_budget'] == 0:
            reasons.append('authorization_exhausted')
        rows.append(dict(round=s['round'], step=s['step'], window_id=s['window_id'],
            purchases=len(s['selected']), spend=s.get('window_cost', spend-previous), cumulative_spend=spend,
            prior_unattributed_spend=s.get('window_start_spend', previous)-previous,
            authorization=ceiling, remaining_authorization=s['remaining_budget'],
            window_authorization=selection.get('window_budget'),
            planned_purchases=len(selection.get('planned_selected', s['selected'])),
            binding_reasons=reasons,
            budget_binding=selection.get('budget_binding'),
            predicted_joint_gain=controller.get('predicted_joint_gain'),
            sum_independent_gains=controller.get('sum_independent_gains'),
            joint_minus_additive=controller.get('joint_minus_sum_independent'),
            independent_insertion_control=s.get('independent_control_labels', {}),
            realised_paired_gain_validation=s.get('realised_paired_gain_validation', [])))
        if s['remaining_budget'] != ceiling-spend:
            raise ValueError('remaining authorization disagrees with spend')
        previous = spend
    return rows


def estimator_batches(records, weights, alpha, backend, start, source, step, checker, sample, *, repeat):
    """Shared 8 draws/state; diagnostics fix d=0 and teacher mass for all baselines.

    Syntax filtering retains valid hard sources; all-invalid falls back to soft.
    This is explicitly biased. It never drops teacher targets or exposure units.
    """
    baseline = {name: [] for name in ESTIMATORS}
    pairs, rejected = [], 0
    for i, (record, a) in enumerate(zip(records, alpha)):
        draws = [sample(record, repeat, i, j) for j in range(8)]
        pairs.append(SourcePair(record, tuple(draws[:2]), (f'{repeat}:{i}:0', f'{repeat}:{i}:1')))
        hs, ss = [], []
        for src in draws:
            h, soft, _ = source_gradient_pair(backend, src, start, source)
            hs.append(h); ss.append(soft)
        teacher = ({n: torch.zeros_like(p) for n, p in start.items()} if record.teacher is None else
                   gradients(-backend.score_behavior(record.teacher, start), start))
        valid = [j for j in range(2) if syntax_success(checker, draws[j].behavior.text)]
        rejected += 2-len(valid)
        soft = {n: (ss[0][n]+ss[1][n])/2 for n in start}
        sources = dict(hard2={n: (hs[0][n]+hs[1][n])/2 for n in start}, soft=soft,
            hard8={n: sum(h[n] for h in hs)/8 for n in start},
            syntax_filtered_hard2={n: sum(hs[j][n] for j in valid)/len(valid) for n in start} if valid else soft)
        for name in ESTIMATORS:
            baseline[name].append(cpu_detached(estimate(sources[name], teacher, hs[0], hs[1], a, 0.)))
    w, a = torch.as_tensor(weights).double().cpu(), torch.as_tensor(alpha).double().cpu()
    if not records or len(w) != len(records) or len(a) != len(records) or not torch.isfinite(w).all() or (w < 0).any() or abs(float(w.sum())-1) > 1e-7:
        raise ValueError('aligned frozen weights/alpha required')
    if any(r.teacher is None and ai != 0 for r, ai in zip(records, a)):
        raise ValueError('teacher-free diagnostic state requires alpha=0')
    zeros = tuple({n: torch.zeros_like(p).cpu() for n, p in start.items()} for _ in records)
    batches = {name: BatchReference(tensor_state_hash(start), snapshot(start), zeros, w.clone(), a.clone(), tuple(gs))
               for name, gs in baseline.items()}
    return batches, pairs, rejected


def gradient_variance(records, weights, alpha, backend, start, source, step, checker, sample, *, R,
                      statistic=None, controller_options=None, on_first=None):
    if type(R) is not int or R < 2:
        raise ValueError('variance requires integer R >= 2')
    from .alpha_d import build_reference
    norms = {name: [] for name in (*ESTIMATORS, 'alpha_d') if name != 'alpha_d' or statistic is not None}
    rejected = []
    for repeat in range(R):
        batches, pairs, filtered = estimator_batches(records, weights, alpha, backend, start, source,
            step, checker, sample, repeat=repeat)
        rejected.append(filtered)
        if on_first is not None and repeat == 0:
            on_first(batches)
        for name, batch in batches.items():
            after = execute_update(batch, np.zeros(len(records)), start, step)
            norms[name].append(math.sqrt(sum(float((after[n].detach().double()-start[n].detach().double()).square().sum()) for n in start)))
        if statistic is not None:
            if statistic.feedback_role != 'same_batch_reference_feedback':
                raise ValueError('variance d must reuse same-batch feedback role')
            batch = build_reference(pairs, weights, alpha, backend, start, source, step)
            opts = dict(controller_options or {})
            mode = opts.pop('error_mode', 'loo')
            d, _ = control(*statistic.project(batch.directions, mode), gram(batch.directions, step.diagonal), alpha, **opts)
            after = execute_update(batch, d, start, step)
            norms['alpha_d'].append(math.sqrt(sum(float((after[n].detach().double()-start[n].detach().double()).square().sum()) for n in start)))
    return dict(estimators={name: norm_variance(v) for name, v in norms.items()}, fixed_states=[r.state.state_hash for r in records],
        alpha=list(map(float, alpha)), weights=list(map(float, weights)), rejected_hard2_by_repeat=rejected,
        baseline_d=0., syntax_filter='biased; valid hard mean; all-invalid soft fallback; teacher mass unchanged',
        alpha_d_feedback='frozen same-batch statistic; uncertainty reprojected; staleness bias excluded',
        source_draws=R*8*len(records))
