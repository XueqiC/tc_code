"""Embedded paired interventions using D8 BatchReference and D9 validation."""
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from ..behavior.deltas import tensor_state_hash
from . import alpha_d
from .alpha_d import cpu_detached
from .experiment_alpha_d import AlphaDExperimentMixin
from .functional_step import lora_parameters, snapshot
from .joint_surrogate import control, execute_update, objective, replacement_batch, validate_pair
from .metrics_v11 import (ESTIMATORS, action_scope, generator_for, gradient_variance, greedy_success,
                          paired_correlations, parse_battery, repair_damage, task_rows)
from .persistence import atomic_json, digest, file_hash
from .transport import Behavior, SourceSample

SCHEMA = 'rtd-v11-frozen-window-rev31-1'


def selection_batch_id(statistic):
    return digest(dict(parameter_hash=statistic.parameter_hash, rewards=statistic.rewards,
        tasks=statistic.task_ids, scores=[tensor_state_hash(g) for g in statistic.scores],
        refreshed_step=statistic.refreshed_step))


def window_payload(experiment):
    e, s = experiment, experiment.state
    return dict(schema=SCHEMA, manifest=e.manifest, round=s['round'], step=s['step'], window_id=s['window_id'],
        start=snapshot(cpu_detached(s['alpha_start'])), source=cpu_detached(s['source']), step_rule=s['step_rule'],
        batch=s['d_reference'], pairs=s['alpha_pairs'], old_batch=s['old_reference'], statistic=s['d_feedback'],
        controller=s['alpha_d_control'], purchased_pool=sorted(e.ledger.owned_ids), inner=sorted(s['inner']),
        feedback_tasks=s['feedback_tasks'], selection_feedback_batch=selection_batch_id(s['d_feedback']),
        acquisition_surrogate=s.get('joint_surrogate'), independent_insertion_control=s.get('independent_control_labels', {}),
        realised_paired_gain_validation=s['paired_validations'], selection=s['selection'],
        actual_spend=e.ledger.spent, remaining_authorization=e.ledger.remaining, authorization=e.ledger.budget)


def window_identity(payload):
    return digest(dict(start=payload['batch'].start_hash, source=tensor_state_hash(payload['source']),
        pairs=[asdict(pair) for pair in payload['pairs']], alpha=payload['batch'].alpha.tolist(),
        weights=payload['batch'].weights.tolist(), eta=payload['step_rule'].eta,
        diagonal=tensor_state_hash(payload['step_rule'].diagonal),
        old_theta0=tensor_state_hash(payload['old_batch'].theta0),
        selection_feedback_batch=payload['selection_feedback_batch']))


def save_window(experiment):
    payload = window_payload(experiment)
    root = experiment.directory/'controls/windows'
    root.mkdir(parents=True, exist_ok=True)
    path = root/f'r{payload["round"]}-s{payload["step"]:02d}.pt'
    binding = dict(schema=SCHEMA, manifest_hash=digest(experiment.manifest),
                   start_hash=payload['batch'].start_hash, selection_feedback_batch=payload['selection_feedback_batch'],
                   frozen_input_hash=window_identity(payload))
    if path.exists() and path.with_suffix('.json').exists():
        saved = json.loads(path.with_suffix('.json').read_text())
        if any(saved[k] != v for k, v in binding.items()) or file_hash(path) != saved['sha256']:
            raise ValueError('frozen controls archive changed')
        return path
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        torch.save(payload, stream)
        stream.flush(); os.fsync(stream.fileno())
    # Publish descriptor last. An orphan blob without a descriptor was never
    # published and can be regenerated from the durable window state on resume.
    os.replace(temporary, path)
    atomic_json(path.with_suffix('.json'), binding | dict(file=path.name, sha256=file_hash(path)))
    return path


def load_window(path, *, device='cpu'):
    """Only load a locally produced, hash-bound archive; never repair inputs."""
    path = Path(path)
    binding = json.loads(path.with_suffix('.json').read_text())
    if binding['schema'] != SCHEMA or binding['file'] != path.name or file_hash(path) != binding['sha256']:
        raise ValueError('frozen window archive hash/schema mismatch')
    # E x parameter directions and per-trajectory scores MUST remain on CPU.
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if (digest(payload['manifest']) != binding['manifest_hash'] or payload['batch'].start_hash != binding['start_hash'] or
            window_identity(payload) != binding['frozen_input_hash']):
        raise ValueError('frozen checkpoint binding mismatch')
    validate_window(payload)
    return place_window(payload, device=device)


def place_window(payload, *, device):
    """Move only model-sized functional snapshots/diagonal to the device."""
    result = dict(payload)
    result['start'] = snapshot({n: p.to(device) for n, p in payload['start'].items()})
    result['source'] = {n: p.detach().to(device) for n, p in payload['source'].items()}
    result['step_rule'] = replace(payload['step_rule'],
        diagonal={n: p.to(device) for n, p in payload['step_rule'].diagonal.items()})
    result['batch'] = replace(payload['batch'],
        theta0=snapshot({n: p.to(device) for n, p in payload['batch'].theta0.items()}))
    return result


def validate_window(p):
    ref, stat = p['batch'], p['statistic']
    if p['schema'] != SCHEMA or tensor_state_hash(p['start']) != ref.start_hash:
        raise ValueError('controls require a frozen start checkpoint')
    if stat.feedback_role != 'same_batch_reference_feedback' or stat.parameter_hash != tensor_state_hash(ref.theta0):
        raise ValueError('d feedback must be same_batch_reference_feedback at theta_S(0)')
    if selection_batch_id(stat) != p['selection_feedback_batch']:
        raise ValueError('selection feedback batch identity changed')
    if len(p['pairs']) != len(ref.alpha):
        raise ValueError('frozen evidence is not aligned')
    inner, purchased = set(p['inner']), set(p['purchased_pool'])
    if any(r.record.state.parent_hash not in inner or
           (r.record.teacher is not None and r.record.query_id not in purchased) for r in p['pairs']):
        raise ValueError('controls require purchased inner-fold evidence')
    if not p['feedback_tasks'] or any(h in inner or count < 1 for h, count in p['feedback_tasks']):
        raise ValueError('controls require held-out feedback parents')
    expected = p['controller']
    if expected.get('feedback_role') != 'same_batch_reference_feedback':
        raise ValueError('controller feedback role cannot be substituted')
    if expected['alpha'] != ref.alpha.tolist() or expected['weights'] != ref.weights.tolist():
        raise ValueError('frozen alpha/weights changed')


def flip_pairs(pairs, flags):
    if len(pairs) != len(flags):
        raise ValueError('one orientation per exposure required')
    return tuple(replace(p, sources=p.sources[::-1], draw_ids=p.draw_ids[::-1]) if flag else p
                 for p, flag in zip(pairs, flags))


def shuffled_pairing(pairs, *, seed):
    """Permute Y1's canonical source index across unique states.

    Source TEXT never crosses states. Each state retains its two sampled actions,
    teacher and signed gate d. Only which local sample receives the Y1 gate is
    randomized. Canonical local order alternates forward/reversed by state hash
    order, making the original Y1 assignment balanced. The identity permutation
    therefore leaves every source unchanged. Repeated exposures share the same
    relative orientation. This is NOT
    the equivariant symmetry test (that also flips d).
    """
    states = sorted({p.record.state.state_hash for p in pairs})
    rng = np.random.default_rng(seed)
    assignment = np.arange(len(states)) % 2
    permutation = rng.permutation(len(states))
    indices = assignment[permutation]
    lookup = dict(zip(states, indices != assignment))
    flags = [bool(lookup[p.record.state.state_hash]) for p in pairs]
    return flip_pairs(pairs, flags), dict(seed=seed, states=states, assignment_pool=assignment.tolist(),
        state_permutation=permutation.tolist(), y1_index_by_state=indices.tolist(), swapped_exposures=flags,
        cross_state_assignment=len(states) > 1, source_text_crosses_states=False,
        gate_multiset_preserved=True, alpha_per_state_preserved=True,
        canonical_local_order='forward/reversed alternating in state-hash order',
        changed=any(flags), degenerate_reason='no_orientation_changed' if not any(flags) else None)


def controller_options(payload):
    c = payload['manifest']['config']
    saved = payload['controller']
    # Archives predating D11 retain their original, unnormalised convention.
    mode = c.get('d_lambda_normalisation', saved.get('d_lambda_normalisation', 'none'))
    calibration = (alpha_d.DCalibration(mode, saved['K_scale'], saved['K_scale_nonzero_directions'])
                   if 'K_scale' in saved else None)
    return dict(d_lambda=c.get('d_lambda', 1.), tolerance=c.get('d_solver_tolerance', 1e-8),
                d_lambda_normalisation=mode, calibration=calibration,
                max_iterations=c.get('d_solver_max_iterations', 10000),
                zero_reason=payload['controller']['solver'].get('reason'))


def solve_reference(payload, reference):
    stat = payload['statistic']
    z, error = stat.project(reference.directions, payload['manifest']['config'].get('z_error_mode', 'loo'))
    return control(z, error, alpha_d.gram(reference.directions, payload['step_rule'].diagonal),
                   reference.alpha, mode=payload['controller']['controller'], **controller_options(payload))


def sampler(payload, backend, support, *, identity):
    source = payload['source']
    rng = generator_for(source, payload['window_id'], identity)
    def draw(record, repeat, i, j):
        with action_scope(backend, support, record.state.parent_hash):
            action = backend.sample_action(record.state.prompt, source, rng, temperature=1., top_p=1.)
        return SourceSample(Behavior(record.state, action.text), backend.identity(source), action.action_ids,
            action.eos_token_id, action.generation_logprob, truncated=action.truncated)
    return draw


def run_controls(payload, backend, support, checker, journal, *, R=4, seed=0, z_probes=4):
    validate_window(payload)
    if type(z_probes) is not int or z_probes < 1:
        raise ValueError('z_probes must be positive')
    p, ref = payload, payload['batch']
    fixed = p['manifest']['fixed_task_set_v11']
    task_rows(fixed, support)  # fail before updates if any frozen state disappeared
    resident = tensor_state_hash(lora_parameters(backend.model))
    start, step = p['start'], p['step_rule']
    before = greedy_success(fixed, support, backend, start, checker, identity=[p['window_id'], 'before'])
    # Reuse D9's dedicated stream, rollout validation and journal path verbatim.
    adapter = SimpleNamespace(config=p['manifest']['config'], manifest=p['manifest'],
        state=dict(round=p['round'], step=p['step'], window_id=p['window_id'], feedback_tasks=p['feedback_tasks']),
        device=next(iter(start.values())).device, backend=backend, support=support, checker=checker,
        journal=journal, scope=lambda role: journal.measure_phase(role))
    evaluation_index = 0
    def evaluate(parameters, role):
        nonlocal evaluation_index
        evaluation_index += 1
        role = f'{role}_evaluation_{evaluation_index}'
        result = AlphaDExperimentMixin.alpha_validation_return(adapter, parameters, role)
        if result['batch_id'] == p['selection_feedback_batch']:
            raise ValueError('selection feedback reused for validation')
        result['selection_feedback_batch'] = p['selection_feedback_batch']
        success = greedy_success(fixed, support, backend, parameters, checker, identity=[p['window_id'], role])
        result['metrics'] = dict(parse=parse_battery(fixed, support, backend, parameters, checker,
            identity=[p['window_id'], role]), greedy=success, repair_damage=repair_damage(before, success),
            update_direction_norm=float(sum((parameters[n].detach().double()-start[n].detach().double()).square().sum()
                for n in start).sqrt()), teacher_mass=float(ref.weights @ ref.alpha))
        return result
    learned, controller = solve_reference(p, ref)
    if not np.allclose(learned, p['controller']['d_star'], rtol=0, atol=1e-7):
        raise ValueError('frozen d no longer matches the shared solver')
    pairs = []
    def compare(kind, first, d1, second, d2, prediction=0., q=None, equal_alpha=True, check_updates=None,
                prediction_scale=None):
        if equal_alpha and (not torch.equal(first.alpha, second.alpha) or not torch.equal(first.weights, second.weights)):
            raise ValueError('paired controls must preserve per-state alpha and frozen exposure weights')
        def evaluate_branch(parameters, role):
            result = evaluate(parameters, role)
            batch = first if role.endswith('_full') else second
            result['metrics']['teacher_mass'] = float(batch.weights @ batch.alpha)
            return result
        row = validate_pair(kind, q, first, d1, second, d2, float(prediction), start, step,
                            evaluate_branch, check_updates=check_updates,
                            prediction_scale=np.sqrt(controller['K_scale']) if prediction_scale is None else prediction_scale)
        row.update(equal_alpha=equal_alpha, weights=first.weights.tolist(), alpha=first.alpha.tolist(),
                   full_d=list(map(float, d1)), control_d=list(map(float, d2)))
        if not equal_alpha:
            row['control_alpha'] = second.alpha.tolist()
        pairs.append(row)
        journal.append('realised_paired_gain_validation', round=p['round'], step=p['step'], **row)
        return row
    zero = np.zeros(len(ref.alpha))
    compare('learned_d_vs_zero', ref, learned, ref, zero,
        controller['predicted_joint_gain'] if controller['controller'] == 'joint' else controller['predicted_independent_joint_gain'])
    swapped = alpha_d.build_reference(flip_pairs(p['pairs'], [True]*len(ref.alpha)), ref.weights, ref.alpha,
                                     backend, start, p['source'], step)
    swapped_d, _ = solve_reference(p, swapped)
    if not np.allclose(swapped_d, -learned, rtol=0, atol=1e-7):
        raise ValueError('source swap failed to flip d')
    def check_symmetry(left, right):
        for n in start:
            torch.testing.assert_close(left[n], right[n])
    compare('source_swap_symmetry', ref, learned, swapped, swapped_d, check_updates=check_symmetry)
    shuffled, shuffling = shuffled_pairing(p['pairs'], seed=seed)
    shuffled_ref = alpha_d.build_reference(shuffled, ref.weights, ref.alpha, backend, start, p['source'], step)
    predicted_shuffle_difference = 0.
    if np.any(learned):
        values = []
        for batch in (ref, shuffled_ref):
            z, error = p['statistic'].project(batch.directions, p['controller']['z_error_mode'])
            values.append(objective(z, error, alpha_d.gram(batch.directions, step.diagonal),
                                    learned, p['controller']['d_lambda'],
                                    **{k: controller_options(p)[k] for k in ('d_lambda_normalisation', 'calibration')}))
        predicted_shuffle_difference = values[0]-values[1]
    compare('learned_d_vs_shuffled_pairing', ref, learned, shuffled_ref, learned, predicted_shuffle_difference)
    compare('joint_vs_independent_control', ref, controller['joint_d'], ref, controller['independent_d'],
            controller['joint_minus_independent_control'])
    journal.append('joint_vs_independent_control', round=p['round'], step=p['step'], **controller)
    journal.append('independent_insertion_control', round=p['round'], step=p['step'],
                   labels=p['independent_insertion_control'], reused_from_frozen_window=True)
    surrogate = p.get('acquisition_surrogate')
    if surrogate and surrogate['marginal_values']:
        q = min(surrogate['marginal_values'])
        reduced = replacement_batch(p['pairs'], ref, p['old_batch'], q, start, step)
        compare('acquisition_surrogate', ref, surrogate['full_set']['d_star'], reduced,
            surrogate['leave_one_out'][q]['d_star'], surrogate['marginal_values'][q], q=q, equal_alpha=False)
    # Predetermined coordinate probes; z predicts slope, d*z predicts finite gain.
    eligible = [i for i in range(len(p['pairs'])) if 0 < ref.alpha[i] < 1]
    indices = sorted(eligible, key=lambda i: (p['pairs'][i].record.state.state_hash, i))[:z_probes]
    for i in indices:
        probe = min(float(ref.alpha[i]), 1-float(ref.alpha[i]), .1)
        d = zero.copy(); d[i] = probe
        row = compare('z_direction', ref, d, ref, zero, probe*p['controller']['z_hat'][i], prediction_scale=1.)
        row.update(coordinate=i, probe_d=probe, z_hat=p['controller']['z_hat'][i])
    # Run source-estimator diagnostic LAST, after every core intervention/probe.
    diagnostic_rows = []
    def diagnostic_updates(batches):
        for name in ESTIMATORS:
            batch = batches[name]
            result = evaluate(execute_update(batch, zero, start, step), f'validation_estimator_{name}')
            diagnostic_rows.append(dict(estimator=name, updates=1, d=zero.tolist(), alpha=batch.alpha.tolist(), **result))
    variance = gradient_variance([pair.record for pair in p['pairs']], ref.weights, ref.alpha, backend,
        start, p['source'], step, checker, sampler(p, backend, support, identity='estimator_diagnostic'), R=R,
        statistic=p['statistic'], controller_options=controller_options(p) |
            dict(error_mode=p['manifest']['config'].get('z_error_mode', 'loo')), on_first=diagnostic_updates)
    if resident != tensor_state_hash(lora_parameters(backend.model)):
        raise AssertionError('paired diagnostics mutated the resident checkpoint')
    report = dict(schema='rtd-v11-controls-rev31-1', arm=p['manifest']['arm'], round=p['round'], step=p['step'],
        window_id=p['window_id'], start_hash=ref.start_hash, manifest_hash=digest(p['manifest']),
        fixed_task_set=fixed, before_greedy=before, pairs=pairs, shuffling=shuffling, controller=controller,
        gradient_variance=variance, source_estimator_diagnostics=diagnostic_rows, correlations=paired_correlations(pairs),
        independent_insertion_control=p['independent_insertion_control'],
        existing_d9_validations=p['realised_paired_gain_validation'],
        selection_feedback_batch=p['selection_feedback_batch'], actual_spend=p['actual_spend'],
        remaining_authorization=p['remaining_authorization'], authorization=p['authorization'],
        diagnostic_order='core comparisons, z probes, source estimators last',
        variance_scope='same frozen window states/start/source/alpha/weights; R fresh source batches',
        joint_vs_additive=dict(predicted_joint_gain=controller['predicted_joint_gain'],
            sum_independent_gains=controller['sum_independent_gains'],
            surrogate_joint_minus_additive=controller['joint_minus_sum_independent'],
            realised_joint_minus_additive=(pairs[0]['realised_paired_gain']-controller['sum_independent_gains']
                if controller['controller'] == 'joint' else None)),
        validation_only=True, used_for_posterior=False)
    return report
