"""D7 rev-2 conventions on the merged executor, with durable source pairs.

Acquisition consumes its historical posterior before any current-step return
rollout. Old-evidence virtual feedback supplies insertion labels and z; the
independent post-commit batch supplies alpha. Same-batch theta0 is an algebraic
baseline for the selected update, not the return-gradient evaluation point.
Only the same-batch distillation update is installed; feedback is never an RL
backbone step. All rollout objectives here are temperature-1 stochastic J.
"""
from dataclasses import asdict, replace

import numpy as np
import torch

from ..behavior.deltas import tensor_state_hash
from .alpha_d import (ALPHA_D_DEFAULTS, RETURN_OBJECTIVE, ExposureRecord, SourcePair,
    FeedbackStatistic, blocked_alpha_vjp, build_reference, cpu_detached, gram,
    gram_statistics, injection, solve_d, state_features)
from .features import FrozenProjection
from .functional_step import FrozenStep, commit_step, kl_pilot, lora_parameters, snapshot
from .insertion import InsertionReference
from .persistence import digest
from .return_gradient import ActionTrace
from .conventions import exposure_step, record_identity, repetition_counts


class AlphaDExperimentMixin:
    def alpha_old_pool(self):
        """Paid records plus reference states without teacher evidence, in fixed order."""
        paid = [r for p in self.packages() for r in self.supervision_records(p)]
        reference = [ExposureRecord(None, 0, self.support.states[h], None)
                     for h in sorted(self.state['inner']) if h in self.support.states]
        return tuple(paid + reference)

    def alpha_replay_records(self, rows, pool):
        lookup = {(r.query_id, r.record_index, r.state.state_hash, r.is_new): r for r in pool}
        records = []
        for row in rows:
            key = (row['query_id'], row['record_index'], row['state_hash'], row['is_new'])
            if key not in lookup or (lookup[key].teacher is not None) != row['has_teacher']:
                raise ValueError('replay record not in the authorized pool or teacher flag changed')
            records.append(lookup[key])
        if repetition_counts([record_identity(r, row['weight']) for r, row in zip(records, rows)]) != rows:
            raise ValueError('replay repetition counts changed')
        return records, [r['weight'] for r in rows]

    def alpha_option(self, name):
        return self.config.get(name, ALPHA_D_DEFAULTS[name])

    def supervision_records(self, package, *, is_new=False):
        """Adapter-fixed episode mapping: ordered command/state teacher records.

        An ALFWorld episode's distinct commands are distinct records. Buying an
        episode purchases all records, but one new-package unit trains ONE of
        them. The others become eligible in the old pool after commit.
        """
        mapper = getattr(self.support, 'supervision_records', lambda p: p.behaviors)
        records = tuple(mapper(package))
        if not records:
            raise ValueError('purchased package has no supervision records')
        for i, teacher in enumerate(records):
            if teacher.state.parent_hash not in self.state['inner']:
                raise ValueError('teacher supervision outside inner fold')
            yield ExposureRecord(package.query_id, i, teacher.state, teacher, is_new)

    def alpha_chi(self, record):
        s, state = self.state, record.state
        if state.parent_hash not in s['inner']:
            raise ValueError('alpha state outside inner fold')
        if state.state_hash not in s['projection_cache']:
            initial_id = self.backend.identity(s['initial'])
            hidden = self.backend.initial_hidden(state.prompt, s['initial'], initial_snapshot_id=initial_id)
            projection = FrozenProjection(hidden.numel(), initial_snapshot_id=initial_id,
                                          seed=self.config['gate_projection_seed'])
            s['projection_cache'][state.state_hash] = projection(hidden, snapshot_id=initial_id)
        return state_features(state, s['projection_cache'][state.state_hash], scalar=s['phi'].numel() == 1,
                              device=self.device, dtype=self.dtype)

    def alpha_draw_pairs(self, records, *, role):
        """Fresh independent calls per occurrence, even for repeated states.

        Equal token hashes are legal. Draw identities and RNG advancement,
        rather than token inequality, detect accidental source-cache reuse.
        """
        s, pairs = self.state, []
        drawn = []
        previous = set(s.get('last_committed_draw_ids', ()))
        for i, record in enumerate(records):
            before = digest(self.sampling_rng.get_state().tolist())
            cached = s['source_cache'].get(record.state.state_hash, ())
            sources = self.sample_state(record.state, refresh=True, cache=False)
            after = digest(self.sampling_rng.get_state().tolist())
            if (before == after or any(src is old for src in sources for old in (*cached, *drawn))
                    or any(src.frozen_snapshot_id != s['source_id'] for src in sources)):
                raise AssertionError('source sampling reused cache or did not advance RNG')
            drawn.extend(sources)
            draw_ids = tuple(f"r{s['round']}/s{s['step']}/{role}/{i}/{j}/{before}" for j in range(2))
            if previous.intersection(draw_ids):
                raise AssertionError('source draw reused across commit steps')
            pair = SourcePair(record, sources, draw_ids)
            pairs.append(pair)
            self.journal.append('alpha_d_source_pair', round=s['round'], step=s['step'], role=role,
                unit_index=i, query_id=record.query_id, record_index=record.record_index,
                state_hash=record.state.state_hash, source_id=s['source_id'], draw_ids=list(draw_ids),
                sample_hashes=[digest(asdict(src)) for src in sources],
                action_hashes=[digest(src.token_ids) for src in sources],
                rng_before=before, rng_after=after, cache_reused=False, temperature=1., top_p=1.)
        return tuple(pairs)

    def alpha_freeze(self, records, *, role, fixed=False, weights=None):
        s = self.state
        chi = torch.stack([self.alpha_chi(r) for r in records]).detach()
        alpha = injection(chi, s['phi'], mode='fixed_alpha' if fixed else self.alpha_option('gate_mode'),
                          fixed_alpha=.5 if fixed else self.alpha_option('fixed_alpha')).detach().clone()
        alpha *= alpha.new_tensor([r.teacher is not None for r in records])
        # Double weights sum to one independently of backbone precision.
        weights = (torch.full((len(records),), 1/len(records), dtype=torch.float64) if weights is None
                   else torch.as_tensor(weights, dtype=torch.float64).clone())
        if weights.shape != (len(records),) or not torch.isfinite(weights).all() or (weights < 0).any() or abs(float(weights.sum())-1) > 1e-7:
            raise ValueError('exposure weights must be nonnegative and sum to one')
        self.journal.append('alpha_d_exposure_frozen', round=s['round'], step=s['step'], role=role,
            before_source_sampling=True, weights=weights.tolist(), alpha=alpha.cpu().tolist(),
            state_features=chi.cpu().tolist(),
            records=[dict(query_id=r.query_id, record_index=r.record_index, state_hash=r.state.state_hash,
                          is_new=r.is_new, has_teacher=r.teacher is not None) for r in records])
        return chi, alpha, weights

    def alpha_calibrate(self, packages):
        """Training-only pilot, separately drawn and charged as pilot compute."""
        s = self.state
        pool = [r for p in packages for r in self.supervision_records(p)]
        records = [pool[int(self.rng.integers(len(pool)))] for _ in range(self.slots)]
        _, alpha, weights = self.alpha_freeze(records, role='pilot', fixed=True)
        with self.scope('alpha_d_pilot'):
            pairs = self.alpha_draw_pairs(records, role='pilot')
            step = FrozenStep(s['diagonal'], s['eta'], f"r{s['round']}", s['preconditioner_metadata'])
            reference = build_reference(pairs, weights, alpha, self.backend, s['parameters'], s['source'], step)
            gradient = {n: sum(g[n].to(p)*float(w) for g, w in zip(reference.baseline_gradients, weights))
                        for n, p in s['parameters'].items()}
            def loss(a):
                if a != .5:
                    raise ValueError('pilot requires alpha=.5')
                return sum((s['parameters'][n]*g.detach()).sum() for n, g in gradient.items())
            actions = [ActionTrace(tuple(self.backend.tokenizer.encode(src.behavior.state.prompt, add_special_tokens=False)),
                src.token_ids, src.eos_token_id, src.behavior.text, src.logprob, self.backend.backend_id,
                s['source_id'], truncated=src.truncated) for pair in pairs for src in pair.sources]
            eta, meta = kl_pilot(s['parameters'], s['diagonal'], loss,
                lambda updated: self.backend.source_kl(actions, s['source'], updated),
                candidates=self.config['pilot_eta_candidates'], owned_ids=self.ledger.owned_ids,
                evidence_ids={p.query_id for p in packages}, inner_parent_hashes=s['inner'],
                evidence_parents={r.state.parent_hash for r in pool}, source_parents={r.state.parent_hash for r in records})
        s['eta'], s['calibrated'] = eta, True
        self.journal.append('kl_pilot', round=s['round'], metadata=meta, source_slots=len(actions),
                            role='alpha_d_training_only_pilot')

    def alpha_step_start(self):
        s = self.state
        s.update(decision=s['step'] in (1, 4, 7, 10), selected=[], pending_ids=[], labels={},
            label=None, new_targets=(), old_targets=(), batch_targets={}, batch_chi={}, feedback_rollouts={},
            transaction_index=0, transaction_query=None, window_id=f"r{s['round']}/s{s['step']}", hard_stops=[],
            owned_before=tuple(s['owned']), old_noop=False, remaining_candidate_ids=[])
        if s['decision']:
            s['alpha_window_index'] = s.get('alpha_window_index', 0)+1
        s['replay_exposure'] = self.replay_schedule.get((s['round'], s['step'])) if self.replay_schedule else None
        if s['replay_exposure'] and s['replay_exposure']['pool_before'] != sorted(s['owned']):
            raise ValueError('V1 accumulated pool differs from V0 before the step')
        self.journal.append('alpha_d_pool', round=s['round'], step=s['step'], stage='before_acquisition',
                            paid_pool=sorted(s['owned']), reference_states=sorted(r.state.state_hash for r in self.alpha_old_pool() if r.teacher is None))
        # No current feedback, source draw or old reference before acquisition.
        self.journal.append('alpha_d_window_order', round=s['round'], step=s['step'],
            stage='historical_acquisition', posterior_observations=len(s['posterior'].observations))
        self.transition('reference')

    def alpha_revealed(self):
        s = self.state
        pending = [self.broker.acquire(q) for q in s['selected']]
        if not s['calibrated'] and pending:
            self.alpha_calibrate(pending)
        records = []
        old = self.alpha_old_pool()
        if not old:
            raise ValueError('old pool needs paid records or legal reference-pool states')
        new = list(pending)
        if len(new) > self.slots:
            indices = self.rng.choice(len(new), self.slots, replace=False)
            new = [new[i] for i in sorted(indices)]
        for package in new:
            choices = tuple(self.supervision_records(package, is_new=True))
            records.append(choices[int(self.rng.integers(len(choices)))])
        old_records = [old[int(self.rng.integers(len(old)))] for _ in range(self.slots)]
        records.extend(old_records[len(records):])
        weights = old_weights = None
        replay = s['replay_exposure']
        if replay:
            available = list(old) + [r for p in pending for r in self.supervision_records(p, is_new=True)]
            records, weights = self.alpha_replay_records(replay['records'], available)
            old_records, old_weights = self.alpha_replay_records(replay['reference_records'], old)
            if len(records) != self.slots or len(old_records) != self.slots:
                raise ValueError('V1 exposure count changed')
        s['alpha_chi'], alpha, weights = self.alpha_freeze(records, role='commit', weights=weights)
        _, old_alpha, old_weights = self.alpha_freeze(old_records, role='virtual_reference', weights=old_weights)
        with self.scope('alpha_d_commit_source_sampling'):
            s['alpha_pairs'] = self.alpha_draw_pairs(records, role='commit')
        s['step_rule'] = FrozenStep(s['diagonal'], s['eta'], f"r{s['round']}", s['preconditioner_metadata'])
        # Rev-2 D7 contract: old-evidence d=0 point supplies BOTH z and insertion
        # labels. These extra reference actions/forwards have separate compute.
        with self.scope('alpha_d_old_virtual_reference'):
            s['old_reference_pairs'] = self.alpha_draw_pairs(old_records, role='virtual_reference')
            s['old_reference'] = build_reference(s['old_reference_pairs'], old_weights, old_alpha,
                self.backend, s['parameters'], s['source'], s['step_rule'])
        old_ref = s['old_reference']
        old_gradient = {n: sum((g[n].to(p)*float(w) for g, w in zip(old_ref.baseline_gradients, old_ref.weights)),
                              torch.zeros_like(p)) for n, p in s['parameters'].items()}
        s['reference'] = InsertionReference(s['parameters'], None, s['step_rule'], old_slots=self.slots,
            exposure_slots=self.slots, old_gradient=old_gradient,
            exact_noop=all(not g.any() for g in old_gradient.values()))
        with self.scope('alpha_d_same_batch_reference'):
            s['d_reference'] = build_reference(s['alpha_pairs'], weights, alpha, self.backend,
                s['parameters'], s['source'], s['step_rule'], record=lambda row:
                    self.journal.append('alpha_d_microbatch', round=s['round'], step=s['step'], **row))
        ref = s['d_reference']
        self.journal.append('alpha_d_reference', round=s['round'], step=s['step'],
            role='old_evidence_virtual_reference', parameter_hash=s['reference'].reference_hash,
            batch_zero_hash=tensor_state_hash(ref.theta0), start_hash=ref.start_hash,
            same_batch=False, return_objective=RETURN_OBJECTIVE)
        if s['decision']:
            s['feedback_tasks'] = self.choose_feedback_tasks()
            scores = []
            feedback = self.feedback(s['reference'].updated, 'virtual_reference_feedback', trajectory_scores=scores)
            if feedback.parameter_hash != s['reference'].reference_hash:
                raise ValueError('source-selection feedback must be at the old-evidence virtual reference')
            rollouts = s['feedback_rollouts']['virtual_reference_feedback']
            s['d_feedback'] = FeedbackStatistic(cpu_detached(feedback.gradient), tuple(scores),
                tuple(r.reward for r in rollouts), tuple(r.task_id for r in rollouts), feedback.parameter_hash,
                (s['round']-1)*12+s['step'], feedback.metadata['baseline'])
            s['d_reference_feedback'] = feedback
            s['reference_feedback'] = feedback
        if 'd_feedback' not in s:
            raise ValueError('non-decision step has no historical source-selection feedback')
        statistic = s['d_feedback']
        z, error = statistic.project(ref.directions, self.alpha_option('z_error_mode'))
        K = gram(ref.directions, s['step_rule'].diagonal)
        warmup = s['alpha_window_index'] <= self.alpha_option('d_warmup_windows')
        enough_feedback = bool(np.isfinite(error).all())
        if self.alpha_option('d_mode') == 'zero' or warmup or not enough_feedback:
            d = np.zeros(len(records))
            solver = dict(converged=None, iterations=0, reason='configured_zero' if self.alpha_option('d_mode') == 'zero'
                          else 'warmup' if warmup else 'insufficient_trajectories')
        else:
            d, solver = solve_d(z, error, K, ref.alpha, d_lambda=self.alpha_option('d_lambda'),
                tolerance=self.alpha_option('d_solver_tolerance'), max_iterations=self.alpha_option('d_solver_max_iterations'))
        bound = np.minimum(ref.alpha.double().numpy(), 1-ref.alpha.double().numpy())
        solver.update(active_lower=np.flatnonzero(np.isclose(d, -bound, atol=1e-8, rtol=0)).tolist(),
                      active_upper=np.flatnonzero(np.isclose(d, bound, atol=1e-8, rtol=0)).tolist())
        s['d_solution'], s['actual'] = d, ref.selected(d)
        s['alpha_d_control'] = dict(z_hat=z.tolist(), epsilon_hat=[float(x) if np.isfinite(x) else None for x in error],
            K=K.tolist(), K_statistics=gram_statistics(K, self.alpha_option('d_redundancy_cosine_threshold')),
            d_star=d.tolist(), alpha=ref.alpha.tolist(), weights=ref.weights.tolist(), solver=solver,
            d_lambda=self.alpha_option('d_lambda'), z_error_mode=self.alpha_option('z_error_mode'),
            warmup=warmup, feedback_age=(s['round']-1)*12+s['step']-statistic.refreshed_step,
            feedback_parameter_hash=statistic.parameter_hash, epsilon_reprojected=True,
            uncertainty_covers_staleness_bias=False, return_objective=RETURN_OBJECTIVE,
            uncertainty_estimator=('delete_trajectory_recompute_loo; n2_paired_contribution_proxy'
                if self.alpha_option('z_error_mode') == 'loo' else 'cross_half_independent_baselines'),
            reference_hash=s['reference'].reference_hash, batch_zero_hash=tensor_state_hash(ref.theta0),
            reference_evidence='old_pool', actual_hash=tensor_state_hash(s['actual']))
        self.journal.append('alpha_d_solver', round=s['round'], step=s['step'], **s['alpha_d_control'])
        # Install once BEFORE post-update return estimation and alpha update.
        # On recovery from this phase __init__ restores this exact student.
        commit_step(self.backend.model, s['actual'], expected_start_hash=ref.start_hash)
        s['alpha_start'] = s['parameters']
        s['parameters'] = snapshot(lora_parameters(self.backend.model))
        self.journal.append('alpha_d_student_commit', round=s['round'], step=s['step'],
            parameter_hash=tensor_state_hash(s['parameters']), alpha_stop_gradient=True, d_stop_gradient=True)
        self.transition('actual')

    def alpha_acquisition_labels(self):
        """Paid-only labels at the virtual old reference; never actual feedback."""
        s, ref = self.state, self.state['d_reference']
        acquisition = s['reference']
        gs = {pair.record.query_id: {n: g.to(s['alpha_start'][n]) for n, g in gi.items()}
              for pair, gi in zip(s['alpha_pairs'], ref.baseline_gradients) if pair.record.is_new}
        s['value_statistics'] = dict(query_ids=[], values=[], covariance=[])
        if gs:
            feedback = s['reference_feedback']
            if feedback.parameter_hash != acquisition.reference_hash or feedback is s.get('actual_feedback'):
                raise ValueError('insertion feedback substituted with committed-student feedback')
            stats = self.paid_statistics(gs, 'virtual_reference_feedback')
            variances = dict(zip(stats['query_ids'], np.diag(stats['covariance'])))
            s['labels'] = acquisition.batch_labels(gs, feedback, owned_before=s['owned_before'],
                pending_ids=set(s['selected']), window_id=s['window_id'], variances=variances)
            shares = {p.record.query_id: float(w) for p, w in zip(s['alpha_pairs'], ref.weights) if p.record.is_new}
            s['labels'] = {q: replace(label, position_share=shares[q]) for q, label in s['labels'].items()}
            if self.config.get('acquisition_value_mode') == 'joint_surrogate':
                from .joint_surrogate import marginal_values
                values, diagnostic = marginal_values(s['alpha_pairs'], ref, s['old_reference'], s['alpha_start'],
                    acquisition.updated, s['step_rule'], feedback, s['d_feedback'],
                    d_lambda=self.alpha_option('d_lambda'), error_mode=self.alpha_option('z_error_mode'),
                    zero=self.alpha_option('d_mode') == 'zero' or s['alpha_d_control']['warmup'])
                s['labels'] = {q: replace(label, value=values[q]/shares[q], position_share=shares[q],
                    approximation=diagnostic['approximation'], variance_method='additive_feedback_noise_proxy')
                    for q, label in s['labels'].items()}
                stats.update(values=[s['labels'][q].value for q in stats['query_ids']],
                             variance_scope='additive feedback proxy; excludes joint solver uncertainty')
                self.journal.append('joint_acquisition_surrogate', round=s['round'], step=s['step'], **diagnostic)
            s['value_statistics'] = stats
        self.journal.append('alpha_d_acquisition_reference', round=s['round'], step=s['step'],
            role=self.config.get('acquisition_value_mode', 'additive_insertion_surrogate'), parameter_hash=acquisition.reference_hash,
            source_selection_reference_hash=tensor_state_hash(ref.theta0), return_objective=RETURN_OBJECTIVE,
            feedback_role='virtual_reference_feedback' if gs else None,
            labels={q: asdict(v) for q, v in s['labels'].items()})

    def alpha_actual(self):
        s = self.state
        s['next_phi'] = s['phi']
        if s['decision']:
            # Always a separate batch/role, even if d=0 and hashes coincide.
            feedback = self.feedback(s['actual'], 'alpha_post_commit_feedback')
            if feedback.parameter_hash != tensor_state_hash(s['parameters']) or feedback is s.get('reference_feedback'):
                raise ValueError('alpha feedback must be at committed theta_S(d)')
            s['actual_feedback'] = feedback
            s['support_return'] = float(np.mean(feedback.metadata['rewards']))
            if self.alpha_option('gate_mode') == 'learned_alpha':
                with self.scope('alpha_d_blocked_gate_vjp'):
                    vjp = blocked_alpha_vjp(s['alpha_pairs'], s['alpha_chi'], s['phi'], s['d_reference'],
                        self.backend, s['alpha_start'], s['source'], s['step_rule'], feedback,
                        mode=self.alpha_option('gate_mode'))
                    s['next_phi'], meta = s['controller'].update(s['phi'], vjp)
                self.journal.append('alpha_d_gate_update', round=s['round'], step=s['step'],
                    vjp=vjp.tolist(), next_phi=s['next_phi'].tolist(), blocked_d_fixed=True,
                    differentiates_through_d_solver=False, parameter_hash=feedback.parameter_hash, **meta)
            self.alpha_acquisition_labels()
        self.transition('feedback')

    def alpha_feedback_commit(self):
        from .experiment import assert_run_invariants
        s, ref, pairs = self.state, self.state['d_reference'], self.state['alpha_pairs']
        if tensor_state_hash(lora_parameters(self.backend.model)) != tensor_state_hash(s['actual']):
            raise AssertionError('student changed after distillation commit')
        s['phi'] = s['next_phi'].detach().requires_grad_(True)
        schedule = exposure_step(s)
        if s['replay_exposure'] is not None and schedule != s['replay_exposure']:
            raise ValueError('V1 exposure schedule differs from V0')
        s['owned'].extend(s['selected'])
        self.journal.append('alpha_d_pool', round=s['round'], step=s['step'], stage='after_commit',
                            paid_pool=sorted(s['owned']), added=list(s['selected']))
        if s['labels']:
            s['posterior'].observe_batch(s['batch_rows'], s['labels'], s['value_statistics']['covariance'])
        s['last_committed_draw_ids'] = tuple(d for pair in pairs for d in pair.draw_ids)
        new = [p for p in pairs if p.record.is_new]
        old = [p for p in pairs if not p.record.is_new]
        def exposure(items):
            return dict(source_action_tokens=sum(src.length for p in items for src in p.sources),
                teacher_action_tokens=sum(len(self.backend.tokenizer.encode(p.record.teacher.text, add_special_tokens=False)) +
                    int(not self.backend.tokenizer.encode(p.record.teacher.text, add_special_tokens=False) or
                        self.backend.tokenizer.encode(p.record.teacher.text, add_special_tokens=False)[-1] != self.backend.tokenizer.eos_token_id)
                    for p in items if p.record.teacher is not None),
                prompt_tokens=sum(len(self.backend.tokenizer.encode(p.record.state.prompt, add_special_tokens=False)) *
                                  (3 if p.record.teacher else 2) for p in items),
                teacher_slots=sum(p.record.teacher is not None for p in items))
        row = dict(round=s['round'], step=s['step'], decision=s['decision'], inner_fold=(s['round']-1)%2,
            acquisition_protocol='alpha_d_batch_mixture_v11', selected=list(s['selected']), slots=len(pairs),
            max_new_packages=self.max_new_packages, fixed_evidence=self.fixed, start_hash=ref.start_hash,
            reference_hash=s['reference'].reference_hash, batch_zero_hash=tensor_state_hash(ref.theta0),
            actual_hash=tensor_state_hash(s['parameters']), exposure_schedule=schedule,
            exact_noop=ref.start_hash == tensor_state_hash(s['parameters']), audit_passed=s['audit_passed'],
            raw_old_slots=len(old), raw_new_slots=len(new),
            weighted_new_slots=sum(float(w) for p, w in zip(pairs, ref.weights) if p.record.is_new)*len(pairs),
            weighted_old_slots=sum(float(w) for p, w in zip(pairs, ref.weights) if not p.record.is_new)*len(pairs),
            old_coefficient=sum(float(w) for p, w in zip(pairs, ref.weights) if not p.record.is_new), position_share=1/len(pairs),
            exposure_mode='full exposure' if self.slots == 40 else 'random exposure',
            exposure_units=len(pairs), teacher_evidence_units=sum(p.record.teacher is not None for p in pairs),
            reference_pool_units=sum(p.record.teacher is None for p in pairs),
            source_only_placeholders=sum(p.record.teacher is None for p in pairs), source_actions=2*len(pairs),
            microbatch_states=4, microbatches=(len(pairs)+3)//4,
            trained_new_packages=[p.record.query_id for p in new],
            unexposed_purchased_packages=sorted(set(s['selected'])-{p.record.query_id for p in new}),
            authorized_budget=self.ledger.budget, actual_spend=self.ledger.spent, remaining_budget=self.ledger.remaining,
            eta=s['eta'], selection=s['selection'], label=None, labels={q: asdict(v) for q, v in s['labels'].items()},
            alpha_d=s['alpha_d_control'], old_exposure=exposure(old), new_exposure=exposure(new),
            source_truncated_slots=sum(src.truncated for p in pairs for src in p.sources),
            exposure_records=[dict(query_id=p.record.query_id, record_index=p.record.record_index,
                state_hash=p.record.state.state_hash, teacher=asdict(p.record.teacher) if p.record.teacher else None,
                is_new=p.record.is_new, weight=float(w), alpha=float(a), draw_ids=list(p.draw_ids),
                sources=[asdict(src) for src in p.sources]) for p, w, a in zip(pairs, ref.weights, ref.alpha)])
        if s['decision']:
            row.update(window_id=s['window_id'], acquisition_reference_hash=s['reference'].reference_hash,
                value_covariance=s['value_statistics'],
                temperature_1_sampled_source_selection_return=float(np.mean(s['d_reference_feedback'].metadata['rewards'])),
                temperature_1_sampled_post_commit_return=float(np.mean(s['actual_feedback'].metadata['rewards'])))
        s['steps'].append(row)
        self.journal.append('alpha_d_step', **row)
        # Keep only positive source/teacher records for legacy artifact consumers.
        s['old_targets'] = tuple((src, p.record.teacher) for p in old for src in p.sources)
        s['new_targets'] = tuple((src, p.record.teacher) for p in new for src in p.sources)
        assert_run_invariants(s, self.ledger)
        self.transition('committed')
