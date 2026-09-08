"""Opt-in D5 hooks; diagnostic RNG and labels never enter training state."""
import numpy as np
import torch

from .alpha_d import ExposureRecord, injection
from .controls_v11 import sampler
from ..behavior.deltas import tensor_state_hash
from .functional_step import FrozenStep
from .joint_surrogate import validate_pair
from .metrics_v11 import (gradient_variance, greedy_success, options, parse_battery, repair_damage,
                          task_rows, paired_correlations)
from .persistence import atomic_json


def window_metrics(experiment):
    e, s = experiment, experiment.state
    fixed, ref = e.manifest['fixed_task_set_v11'], s['d_reference']
    before = greedy_success(fixed, e.support, e.backend, s['alpha_start'], e.checker,
                            identity=[s['window_id'], 'window_before'])
    def evaluate(parameters, role):
        result = e.alpha_validation_return(parameters, role)
        success = greedy_success(fixed, e.support, e.backend, parameters, e.checker, identity=[s['window_id'], role])
        result['greedy_success'] = success
        result['repair_damage'] = repair_damage(before, success)
        return result
    d = s['d_solution']
    zero = np.zeros(len(d))
    row = validate_pair('learned_d_vs_zero', None, ref, d, ref, zero,
        s['alpha_d_control']['predicted_joint_gain'] if s['alpha_d_control']['controller'] == 'joint' else
        s['alpha_d_control']['predicted_independent_joint_gain'], s['alpha_start'], s['step_rule'], evaluate)
    row.update(equal_alpha=True, alpha=ref.alpha.tolist(), weights=ref.weights.tolist(),
               fixed_task_set_hash=fixed['hash'], before_greedy=before)
    s['paired_validations'].append(row)
    e.journal.append('realised_paired_gain_validation', round=s['round'], step=s['step'], **row)
    # One coordinate per window, chosen by state identity before inspecting z.
    eligible = [i for i in range(len(d)) if 0 < ref.alpha[i] < 1]
    if eligible:
        i = min(eligible, key=lambda i: (s['alpha_pairs'][i].record.state.state_hash, i))
        probe = min(float(ref.alpha[i]), 1-float(ref.alpha[i]), .1)
        direction = zero.copy(); direction[i] = probe
        z = s['alpha_d_control']['z_hat'][i]
        row = validate_pair('z_direction', None, ref, direction, ref, zero, probe*z,
            s['alpha_start'], s['step_rule'], e.alpha_validation_return)
        row.update(coordinate=i, probe_d=probe, z_hat=z, equal_alpha=True)
        s['paired_validations'].append(row)
        e.journal.append('realised_paired_gain_validation', round=s['round'], step=s['step'], **row)


def round_metrics(experiment):
    e, s = experiment, experiment.state
    fixed = e.manifest['fixed_task_set_v11']
    identity = [e.manifest['arm'], s['round'], 'round_end_metrics']
    with e.scope('v11_round_metrics'):
        parse = parse_battery(fixed, e.support, e.backend, s['parameters'], e.checker, identity=identity)
        # Freeze states, alpha and teacher lookup BEFORE all R source resamples.
        # Held-out-fold fixed states are evaluation only, never gradient inputs.
        paid = {r.state.state_hash: r for package in sorted(e.packages(), key=lambda p: p.query_id)
                for r in e.supervision_records(package)}
        records = []
        for row in task_rows(fixed, e.support):
            if row['parent_hash'] in s['inner']:
                state = e.support.states[row['parent_hash']]
                records.append(paid.get(state.state_hash, ExposureRecord(None, 0, state, None)))
        chi = torch.stack([e.alpha_chi(r) for r in records])
        alpha = injection(chi, s['phi'], mode=e.alpha_option('gate_mode'), fixed_alpha=e.alpha_option('fixed_alpha')).detach()
        alpha = torch.tensor([float(a) if r.teacher is not None else 0. for r, a in zip(records, alpha)], dtype=torch.float64)
        weights = torch.full((len(records),), 1/len(records), dtype=torch.float64)
        step = FrozenStep(s['diagonal'], s['eta'], f'r{s["round"]}', s['preconditioner_metadata'])
        payload = dict(source=s['source'], window_id=f'r{s["round"]}-metrics')
        variance = gradient_variance(records, weights, alpha, e.backend, s['parameters'], s['source'], step,
            e.checker, sampler(payload, e.backend, e.support, identity=identity),
            R=options(e.config)['variance_resamples'], statistic=s.get('d_feedback'),
            controller_options=dict(d_lambda=e.alpha_option('d_lambda'), error_mode=e.alpha_option('z_error_mode'),
                zero_reason='configured_zero' if e.alpha_option('d_mode') == 'zero' else None,
                tolerance=e.alpha_option('d_solver_tolerance'), max_iterations=e.alpha_option('d_solver_max_iterations')))
    validations = [v for row in s['steps'] if row['round'] == s['round'] for v in row.get('realised_paired_gain_validation', [])]
    result = dict(schema='rtd-v11-round-metrics-1', arm=e.manifest['arm'], round=s['round'],
        parameter_hash=tensor_state_hash(s['parameters']),
        fixed_task_set_hash=fixed['hash'], parse=parse, gradient_variance=variance,
        correlations=paired_correlations(validations), realised_paired_gain_validation=validations,
        actual_spend=e.ledger.spent, authorization=e.ledger.budget, remaining_authorization=e.ledger.remaining,
        validation_only=True, used_for_control_or_posterior=False)
    atomic_json(e.directory/'metrics'/f'round-{s["round"]}.json', result)
    e.journal.append('v11_round_metrics', **result)
    return result
