"""Opt-in batch phases; v1.0's finite update and artifact schema stay isolated."""
import copy
from dataclasses import asdict, replace

import numpy as np
import torch

from ..behavior.deltas import tensor_state_hash
from .acquisition import BatchAcquisitionPolicy, PrePurchaseFeatures, window_budget
from .functional_step import FrozenStep, commit_step, lora_parameters, snapshot
from .insertion import InsertionReference, LabelType
from .ledger import BudgetError
from .runtime import streamed_gate_vjp
from .selector import StudentSnapshot, select_public_batch
from .value_feedback import insertion_statistics, reliability
from .broker import UnavailableError


class BatchExperimentMixin:
    def batch_remaining_candidates(self):
        s = self.state
        features = tuple((h, self.feature(srcs[0]).features) for h, srcs in s['source_cache'].items())
        view = StudentSnapshot(s['source_id'], frozenset(s['inner']), features)
        return [c.query_id for c in self.broker.list_candidates(view, self.ledger.owned_ids, self.ledger.remaining)]

    @property
    def max_new_packages(self):
        return self.config.get('max_new_packages_per_window', 20)

    def batch_features(self, specs):
        s = self.state
        projections = [s['projection_cache'][b.state.state_hash] for p in self.packages() for b in p.behaviors
                       if b.state.state_hash in s['projection_cache']]
        def coverage(spec):
            x = np.asarray(spec.features.projection)
            return max((float(np.dot(x, p)/max(np.linalg.norm(x)*np.linalg.norm(p), 1e-12))
                        for p in projections), default=0.)
        context = self.alpha_training_context() if self.alpha_d else ()
        rows = {c.query_id: PrePurchaseFeatures.from_public(c, round_id=f"r{s['round']}", coverage=coverage(c),
            progress=((s['round']-1)*12+s['step']-1)/(12*s['rounds']), support_return=s['support_return'],
            context=context) for c in specs}
        if self.alpha_d:
            from .acquisition import TRAINING_CONTEXT_FEATURES
            self.journal.append('acquisition_pre_purchase_context', round=s['round'], step=s['step'],
                before_purchase=True, before_source_sampling=True, purchased_set=sorted(s['owned']),
                context=dict(zip(TRAINING_CONTEXT_FEATURES, context)),
                feature_rows={q: list(row.values) for q, row in rows.items()},
                label_kind='acquisition_surrogate' if self.alpha_option('acquisition_value_mode') == 'joint' else 'independent_control')
        return rows

    def paid_statistics(self, package_gradients, role):
        s = self.state
        with self.scope('insertion_uncertainty'):
            return insertion_statistics(s['feedback_rollouts'][role], self.backend, s['reference'], package_gradients,
                revealed_ids=self.ledger.owned_ids, baseline='smoke_zero' if s['smoke'] else 'leave_one_out_same_task',
                noise_floor=self.config.get('value_noise_floor', 1e-8),
                prior_noise=s['posterior'].noise_variance, seed=self.config['training_seed']+1000*s['round']+s['step'])

    def remeasure_owned(self):
        """Only legal inner-fold, already-paid references; no candidate payloads."""
        s = self.state
        if not s.get('drift_pending'):
            return
        legal = sorted(q for q in s['owned'] if self.broker._records[q].parent_hash in s['inner']
                       and q in s['posterior'].observations)
        ids = legal[:self.config.get('drift_reference_packages', 4)]
        rows, gs = [], {}
        for q in ids:
            package = self.broker.acquire(q)  # ledger-owned before this round
            targets, chi = self.draw_slots([package], package_only=True, count=1)
            controls = s.pop('draw_controls', None)
            with self.scope('drift_package_gradient'):
                gs[q] = self.gradient(targets, chi, s['reference'].start, controls=controls)
            rows.append(replace(self.broker._records[q].spec, features=self.feature(targets[0][0]).features))
        drift = dict(round_id=f"r{s['round']}", query_ids=ids, measurements=[],
                     status='remeasured' if ids else 'no_legal_purchased_reference',
                     owned_outside_inner=sum(self.broker._records[q].parent_hash not in s['inner'] for q in s['owned']))
        if ids:
            statistics = self.paid_statistics(gs, 'reference_feedback')
            variance = dict(zip(ids, np.diag(statistics['covariance'])))
            labels = s['reference'].batch_labels(gs, s['reference_feedback'], owned_before=s['owned_before'],
                pending_ids=set(), window_id=s['window_id'], variances=variance, label_type=LabelType.REWEIGHT_EXISTING)
            features = self.batch_features(rows)
            for q, label in labels.items():
                previous = s['drift_measurements'].get(q, s['posterior'].observations[q][1])
                event = s['posterior'].inflate_for_drift(query_id=q, old_value=previous.value, new_value=label.value,
                    variance=label.variance, previous_variance=previous.variance)
                s['posterior'].observe(features[q], label)
                s['drift_measurements'][q] = label
                drift['measurements'].append(event)
            self.feedback(s['reference'].updated, 'drift_independent_feedback')
            second = self.paid_statistics(gs, 'drift_independent_feedback')
            drift['reliability'] = reliability(statistics, second, revealed_ids=self.ledger.owned_ids)
            drift['label_covariance'] = statistics
        s['round_drift'] = drift
        s['drift_pending'] = False
        self.journal.append('value_drift', round=s['round'], step=s['step'], **drift)

    def batch_reference(self):
        s = self.state
        s['trace'] = []
        s['batch_rows'], s['batch_specs'] = {}, {}
        if s['decision'] and not self.alpha_d:
            self.remeasure_owned()
        if s['decision'] and not self.fixed:
            features = tuple((h, self.feature(sources[0]).features) for h, sources in s['source_cache'].items())
            view = StudentSnapshot(s['source_id'], frozenset(s['inner']), features)
            candidates = self.broker.list_candidates(view, self.ledger.owned_ids, self.ledger.remaining)
            remaining_windows = 1 if s['smoke'] else sum(step >= s['step'] for step in (1, 4, 7, 10))
            quota = window_budget(self.ledger.remaining, remaining_windows, candidates)
            rows = self.batch_features(candidates)
            policy = BatchAcquisitionPolicy(s['posterior'], s['cost_model'], rng=self.rng)
            replay = s.get('replay_exposure') if self.alpha_d else None
            if replay:
                selected = replay['selected']
                if not set(selected) <= rows.keys():
                    raise ValueError('V0 replay purchase is unavailable under V1 authorization')
                selected, trace = select_public_batch(candidates, lambda public:
                    [c.query_id for c in public if c.query_id in set(replay['selected'])])
                quota = replay['window_budget']
                policy.last_decision = dict(selected=list(selected), query_ids=list(rows),
                    sampled_values=[0.]*len(rows), predicted_additive_gain=0., predicted_cost=None,
                    budget_binding=False, stop_reason='exposure_schedule_replay',
                    replay_schedule_hash=self.config['replay_schedule_hash'])
            else:
                selected, trace = select_public_batch(candidates, lambda public: policy.choose_batch(public, rows,
                    remaining_budget=quota, exposure_slots=40 if self.alpha_d else self.slots, max_new_packages=self.max_new_packages,
                    random_control=self.manifest['arm'] in {'R0', 'V0'} or self.config.get('acquisition') == 'random',
                    previous_model=s['previous_decision_model']))
            if self.alpha_d:
                # Stable executor order is independent of random/learned solver rank.
                selected = sorted(selected)
                policy.last_decision['selected'] = list(selected)
            s['selected'], s['trace'], s['selection'] = list(selected), trace, policy.last_decision
            s['batch_rows'], s['batch_specs'] = rows, {c.query_id: c for c in candidates}
            s['selection'].update(window_id=s['window_id'], remaining_outer_windows=remaining_windows,
                                  global_remaining=self.ledger.remaining)
            s['window_budget'], s['window_start_spend'] = quota, self.ledger.spent
            s['previous_decision_model'] = copy.deepcopy(s['posterior'].model)
            s['transaction_query'] = selected[0] if selected else None
        else:
            s['selection'] = dict(selected=[], fixed_evidence=self.fixed, replay_step=not s['decision'],
                predicted_additive_gain=0., budget_binding=False, stop_reason='fixed_evidence' if self.fixed else 'replay_step')
        s['audit_passed'] = self.broker.assert_no_hidden_access(s['trace']).passed
        if not s['audit_passed']:
            raise AssertionError('selector read audit failed')
        if s['decision']:
            self.journal.append('decision', round=s['round'], step=s['step'], selection=copy.deepcopy(s['selection']),
                                selector_trace=s['trace'], audit_passed=s['audit_passed'])
        self.transition('selected')

    def batch_selected(self):
        s = self.state
        if s['decision'] and not self.fixed:
            self.ledger.open_window(s['window_id'], s['window_budget'], self.max_new_packages)
        if s['transaction_index'] < len(s['selected']):
            q = s['selected'][s['transaction_index']]
            if q != s['transaction_query']:
                raise ValueError('durable batch transaction does not match planned choice')
            self.broker._offered.add(q)
            try:
                package = self.broker.acquire(q)
            except BudgetError:
                # The predicted-cost knapsack does not override a reservation.
                # Keep the sampled set fixed; visit remaining choices once.
                s['hard_stops'].append(dict(query_id=q, reason='hard_cap_tail',
                    global_remaining=self.ledger.remaining, window_remaining=self.ledger.window_remaining))
            except UnavailableError as error:
                if not self.alpha_d:
                    raise
                s['hard_stops'].append(dict(query_id=q, reason='local_precheck_unavailable', detail=str(error),
                    global_remaining=self.ledger.remaining, window_remaining=self.ledger.window_remaining))
            else:
                self.journal.append('request_reveal_link', round=s['round'], step=s['step'], query_id=q,
                    window_id=s['window_id'], ledger_reveal_sequence=next(e['sequence'] for e in self.ledger.events
                        if e['kind'] == 'reveal' and e['query_id'] == q))
                if not self.alpha_d:
                    self.batch_prepare_package(package)
                s['cost_model'].observe_revealed(s['batch_specs'][q], s['batch_rows'][q], cost=package.cost,
                    confidence=package.cost_confidence, revealed_ids=self.ledger.owned_ids)
                s['pending_ids'].append(q)
            if self.alpha_d:
                # Recompute legal public inventory after settlement/release.
                # No refill: the solver's fixed plan remains the only purchase set.
                s['remaining_candidate_ids'] = self.batch_remaining_candidates()
                self.journal.append('acquisition_transaction', round=s['round'], step=s['step'], query_id=q,
                    acquired=q in s['pending_ids'], reservations=dict(self.ledger.reservations),
                    remaining_authorization=self.ledger.remaining, window_remaining=self.ledger.window_remaining,
                    actual_spend=self.ledger.spent, candidates=s['remaining_candidate_ids'])
            s['transaction_index'] += 1
            s['transaction_query'] = (s['selected'][s['transaction_index']]
                                      if s['transaction_index'] < len(s['selected']) else None)
            self.save()  # current request only may lead recovery in the ledger
            return
        s['selection']['planned_selected'] = list(s['selected'])
        s['selected'] = list(s['pending_ids'])
        s['selection']['selected'] = list(s['selected'])
        if s['hard_stops']:
            hard_cap = any(r['reason'] == 'hard_cap_tail' for r in s['hard_stops'])
            s['selection'].update(budget_binding=hard_cap, stop_reason='hard_cap_tail' if hard_cap else 'local_precheck_unavailable',
                                  hard_stops=s['hard_stops'])
        if self.alpha_d:
            if s.get('replay_exposure') and s['selected'] != s['replay_exposure']['selected']:
                raise ValueError('V1 hard ledger could not acquire the V0 schedule')
            s['remaining_candidate_ids'] = self.batch_remaining_candidates()
            self.journal.append('acquisition_execution', round=s['round'], step=s['step'],
                planned=s['selection']['planned_selected'], acquired=list(s['selected']),
                drops=list(s['hard_stops']), reservation_order='query_id_ascending',
                remaining_authorization=self.ledger.remaining, actual_spend=self.ledger.spent,
                remaining_candidates=s.get('remaining_candidate_ids', sorted(set(s['batch_specs'])-self.ledger.owned_ids)))
        ids, values = s['selection'].get('query_ids', []), s['selection'].get('sampled_values', [])
        s['selection']['planned_predicted_additive_gain'] = s['selection']['predicted_additive_gain']
        value_by_id = dict(zip(ids, values))
        s['selection']['predicted_additive_gain'] = sum(value_by_id[q]/(40 if self.alpha_d else self.slots) for q in s['selected'])
        if not self.alpha_d:
            s['new_targets'] = tuple(t for q in s['selected'] for t in s['batch_targets'][q])
        self.transition('revealed')

    def batch_prepare_package(self, package):
        # Historical D1/D2 path; rev 3 freezes the whole evidence batch
        # after all purchases instead of drawing here.
        s, q = self.state, package.query_id
        with self.scope('pending_source_sampling'):
            targets, chi = self.draw_slots([package], package_only=True, count=1)
        # Save this draw's controls before the pilot samples its own
        # independent group. Never reconstruct LOO from exposure slots.
        if self.config.get('source_estimator') == 'cv':
            s.setdefault('batch_controls', {})[q] = s.pop('draw_controls')
        s['batch_targets'][q], s['batch_chi'][q] = targets, chi
        if not s['calibrated']:
            if not s['old_noop']:
                raise ValueError('first-purchase pilot would change a nonidentity reference')
            self.calibrate([package])
            s['step_rule'] = FrozenStep(s['diagonal'], s['eta'], f"r{s['round']}", s['preconditioner_metadata'])
            s['reference'] = InsertionReference(s['parameters'], None, s['step_rule'], old_slots=self.slots,
                                               exposure_slots=self.slots, exact_noop=True)
            if s['reference_feedback'].parameter_hash != s['reference'].reference_hash:
                raise ValueError('pilot changed the identity reference')

    def batch_revealed(self):
        s, gs = self.state, {}
        for q in s['selected']:
            with self.scope('new_package_gradient'):
                gs[q] = self.gradient(s['batch_targets'][q], s['batch_chi'][q], s['reference'].start,
                                      controls=s.get('batch_controls', {}).get(q))
        if gs:
            stats = self.paid_statistics(gs, 'reference_feedback')
            variances = dict(zip(stats['query_ids'], np.diag(stats['covariance'])))
            self.feedback(s['reference'].updated, 'reliability_independent_feedback')
            second = self.paid_statistics(gs, 'reliability_independent_feedback')
            s['reliability_check'] = reliability(stats, second, revealed_ids=self.ledger.owned_ids)
            actual, labels, accounting = s['reference'].insert_batch(gs, s['reference_feedback'],
                owned_before=s['owned_before'], pending_ids=s['pending_ids'], window_id=s['window_id'],
                variances=variances, max_new_packages=self.max_new_packages)
            # Mathematically equal, but BF16 accumulation of gJ and a streamed
            # scalar contraction have different rounding. Keep the residual
            # visible instead of requiring float64 agreement from a BF16 model.
            stats['shared_gradient_projection_error'] = (
                np.array([labels[q].value for q in stats['query_ids']])-np.array(stats['values'])).tolist()
            s['value_statistics'] = stats
            s['labels'] = labels
        else:
            actual, accounting = s['reference'].empty()
            accounting.update(predicted_additive_gain=0., old_coefficient=1., position_share=1/self.slots)
            s['value_statistics'] = dict(query_ids=[], values=[], covariance=[])
            s['reliability_check'] = dict(query_ids=[], status='no_purchased_packages')
        s['actual'], s['accounting'] = snapshot(actual), accounting
        self.transition('actual')

    def batch_gate_vjp(self, targets, chi, feedback, *, controls=None):
        """Differentiate the committed estimator using its saved source draws."""
        s = self.state
        return streamed_gate_vjp(targets, chi, s['phi'], self.backend, s['reference'].start,
                                 s['step_rule'], feedback, gate=self.gate, **self.estimator_options(controls))

    def batch_actual(self):
        s = self.state
        s['next_phi'] = s['phi']
        if s['decision']:
            if tensor_state_hash(s['actual']) == s['reference'].reference_hash:
                s['actual_feedback'] = s['reference_feedback']
                self.journal.append('feedback_reused', round=s['round'], step=s['step'], parameter_hash=tensor_state_hash(s['actual']))
            else:
                s['actual_feedback'] = self.feedback(s['actual'], 'actual_feedback')
            feedback = s['actual_feedback']
            if feedback.parameter_hash != tensor_state_hash(s['actual']):
                raise ValueError('gate feedback is not at the actual model')
            s['support_return'] = float(np.mean(feedback.metadata['rewards']))
            if self.gate in {'linear_sigmoid', 'scalar_sigmoid'}:
                with self.scope('gate_vjp'):
                    vjp = torch.zeros_like(s['phi'])
                    if not s['old_noop']:
                        vjp = (1-len(s['selected'])/self.slots)*self.batch_gate_vjp(
                            s['old_targets'], s['old_chi'], feedback, controls=s.get('old_controls'))
                    for q in s['selected']:
                        vjp += self.batch_gate_vjp(s['batch_targets'][q], s['batch_chi'][q], feedback,
                                                 controls=s.get('batch_controls', {}).get(q))/self.slots
                    s['next_phi'], meta = s['controller'].update(s['phi'], vjp)
                self.journal.append('gate_update', round=s['round'], step=s['step'], vjp=vjp.tolist(),
                    next_phi=s['next_phi'].detach().tolist(), identifiable=feedback.metadata['identifiable'], **meta)
        self.transition('feedback')

    def batch_feedback_commit(self):
        from .experiment import assert_run_invariants
        s = self.state
        commit_step(self.backend.model, s['actual'], expected_start_hash=s['reference'].start_hash)
        s['parameters'] = snapshot(lora_parameters(self.backend.model))
        s['phi'] = s['next_phi'].detach().requires_grad_(True)
        for q in s['selected']:
            s['owned'].append(q)
        s['posterior'].observe_batch(s['batch_rows'], s['labels'], s['value_statistics']['covariance'])
        m, E = len(s['selected']), self.slots
        row = dict(round=s['round'], step=s['step'], decision=s['decision'], inner_fold=(s['round']-1)%2,
            acquisition_protocol='batch_common_reference_v1', selected=s['selected'], slots=E,
            max_new_packages=self.max_new_packages, fixed_evidence=self.fixed,
            exact_noop=s['old_noop'] and not m, start_hash=s['reference'].start_hash,
            reference_hash=s['reference'].reference_hash, actual_hash=tensor_state_hash(s['parameters']),
            audit_passed=s['audit_passed'], raw_old_slots=0 if s['old_noop'] else E, raw_new_slots=m,
            weighted_new_slots=m, weighted_old_slots=0 if s['old_noop'] else E-m,
            old_coefficient=1-m/E, position_share=1/E, unfilled_old_slots=E-m,
            authorized_budget=self.ledger.budget, actual_spend=self.ledger.spent, remaining_budget=self.ledger.remaining,
            eta=s['eta'], selection=s['selection'], labels={q: asdict(v) for q, v in s['labels'].items()},
            label=None, value_covariance=s['value_statistics'], reliability=s['reliability_check'],
            new_slots_by_query={q: [dict(source=asdict(src), teacher=asdict(t) if t else None)
                                    for src, t in s['batch_targets'][q]] for q in s['selected']})
        def exposure(targets):
            return dict(source_action_tokens=sum(src.length for src, _ in targets),
                teacher_action_tokens=sum(len(self.backend.tokenizer.encode(t.text, add_special_tokens=False)) +
                    int(not self.backend.tokenizer.encode(t.text, add_special_tokens=False) or
                        self.backend.tokenizer.encode(t.text, add_special_tokens=False)[-1] != self.backend.tokenizer.eos_token_id)
                    for _, t in targets if t is not None),
                prompt_tokens=sum(len(self.backend.tokenizer.encode(src.behavior.state.prompt, add_special_tokens=False))
                                  *(2 if t is not None else 1) for src, t in targets),
                teacher_slots=sum(t is not None for _, t in targets))
        row['old_exposure'] = exposure(()) if s['old_noop'] else exposure(s['old_targets'])
        row['new_exposure'] = exposure(s['new_targets'] or ())
        row['source_truncated_slots'] = sum(src.truncated for src, _ in (*s['old_targets'], *(s['new_targets'] or ())))
        if s['decision']:
            reference, actual = s['reference_feedback'], s['actual_feedback']
            realised = float(np.mean(actual.metadata['rewards'])-np.mean(reference.metadata['rewards']))
            additive = s['accounting']['predicted_additive_gain']
            row.update(window_id=s['window_id'], window_cost=self.ledger.spent-s.get('window_start_spend', self.ledger.spent),
                predicted_additive_gain=additive, acquisition_predicted_additive_gain=s['selection']['predicted_additive_gain'],
                realised_joint_update_gain=realised, joint_vs_additive_error=realised-additive,
                joint_vs_acquisition_prediction_error=realised-s['selection']['predicted_additive_gain'],
                gain_estimator='sampled_actual_return_minus_shared_reference_return; includes rollout noise',
                value_difference_significance=s['selection'].get('value_difference_significance'),
                round_drift=s.get('round_drift'), budget_binding=s['selection']['budget_binding'],
                stop_reason=s['selection']['stop_reason'])
            reused = actual.parameter_hash == reference.parameter_hash
            for name, keys in [('truncation', ('rollouts', 'truncated_rollouts', 'truncated_actions')),
                               ('malformed_feedback', ('rollouts', 'malformed_rollouts', 'malformed_actions'))]:
                ref = {k: reference.metadata.get(k, 0) for k in keys}
                act = {k: actual.metadata.get(k, 0) for k in keys}
                row[name] = dict(reference=ref, actual=act, actual_feedback_reused=reused,
                                 **{k: ref[k]+(0 if reused else act[k]) for k in keys})
            exceptions = dict(reference.metadata.get('malformed_exception_types', {}))
            if not reused:
                for kind, count in actual.metadata.get('malformed_exception_types', {}).items():
                    exceptions[kind] = exceptions.get(kind, 0)+count
            row['malformed_feedback']['malformed_exception_types'] = exceptions
            self.journal.append('batch_window', round=s['round'], step=s['step'], **{k: row[k] for k in (
                'window_id', 'selected', 'window_cost', 'predicted_additive_gain', 'acquisition_predicted_additive_gain',
                'realised_joint_update_gain', 'joint_vs_additive_error', 'value_difference_significance',
                'value_covariance', 'reliability', 'stop_reason', 'budget_binding', 'remaining_budget')})
        s['steps'].append(row)
        assert_run_invariants(s, self.ledger)
        self.transition('committed')
