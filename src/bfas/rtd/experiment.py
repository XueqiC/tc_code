"""RTD section 9.4 driver. Every durable phase can be resumed independently.

    step_start -> reference -> selected -> revealed -> actual -> feedback
      -> committed -> step_start / round_end -> round_start

Only the actual branch is committed. Source/initial policies, P, normalization,
slot draws, RNGs and pending labels survive restart. Importing launches nothing.
"""
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np
import torch

from ..behavior.deltas import tensor_state_hash
from ..cc_pairs import render_prompt, thinking_off
from .acquisition import AcquisitionPolicy, CostRegressor, PrePurchaseFeatures, ValuePosterior, LegacyValuePosterior
from .experiment_v11 import BatchExperimentMixin
from .experiment_alpha_d import AlphaDExperimentMixin
from .alpha_d import enabled as alpha_d_enabled, validate_config as validate_alpha_d_config, validate_arm
from .bank import parent_hash
from .broker import SealedReplayBroker
from .features import FeatureRow, FrozenProjection, FrozenStandardizer
from .functional_step import FrozenStep, commit_step, gradients, kl_pilot, lora_parameters, rms_diagonal, snapshot
from .insertion import InsertionReference, LabelType
from .ledger import Ledger
from .persistence import ComputeJournal, StateStore, atomic_json, digest, file_hash, fsync_directory, tree_hash, manifest_hash
from .return_gradient import ActionTrace, GateController, bfcl_task_rollout, reinforce_gradient
from .runtime import streamed_gate_vjp, streamed_gradient
from .source_estimator import SourceControl, validate_source_config
from .memory import MemoryPolicy, memory_batches
from .generation_batch import sample_actions, feedback_rollouts
from .forward_batch import source_score_scope
from .selector import PublicFeatures, StudentSnapshot, select_public
from .transport import Behavior, FullState, SourceSample, TransportSlot, is_exact_noop


class BFCLSupport:
    """Support-only prompts; feedback truth is fetched only for the selected task.

    Memory/web-search remain explicitly unavailable (no validated initial-state
    adapter). They cannot silently become zero-reward feedback or flat targets.
    """
    def __init__(self, root, config):
        from ..adapters.bfcl import BFCLAdapter
        adapter = BFCLAdapter()
        entries, categories = adapter._load_entries()
        parents = json.loads((Path(root) / config['support_manifest']).read_text())['parents']
        self.parents, self.states, self.entries, self.categories = {}, {}, {}, {}
        self.unavailable = {}
        from .student import bfcl_prompt
        tokenizer = None
        if config.get('student_call_format') == 'gemma4':
            from .bank_build import load_student_tokenizer
            tokenizer = load_student_tokenizer(config)
        for p in parents:
            tid, h = p['official_id'], p['parent_hash']
            entry = entries[tid]
            if parent_hash(entry) != h or int(h, 16) % 2 != p['fold']:
                raise ValueError('support content/fold changed')
            category = categories[tid]
            self.parents[h] = tid
            if category.startswith(('memory', 'web_search')):
                self.unavailable[h] = 'no validated agentic initial-state adapter'
                continue
            self.entries[tid], self.categories[tid] = entry, category
            observed = entry
            if category.startswith('multi_turn'):
                from bfcl_eval.utils import populate_test_cases_with_predefined_functions
                import copy
                observed = populate_test_cases_with_predefined_functions([copy.deepcopy(entry)])[0]
            task = {k: observed[k] for k in ('function', 'initial_config', 'involved_classes', 'scenario') if k in observed}
            task['question'] = observed['question'][0]
            prompt = bfcl_prompt(config, observed['question'][0], observed.get('function', []), tokenizer)
            self.states[h] = FullState.create(task, observed['question'][0], prompt, h)
        self._truth = {}

    def feedback(self, parent, backend, parameters, generator, checker):
        from bfcl_eval.utils import load_ground_truth_entry
        tid = self.parents[parent]
        category = self.categories[tid]
        if 'relevance' in category:
            return bfcl_task_rollout(self.entries[tid], category, [], backend, parameters, generator, checker=checker)
        if category not in self._truth:
            # Only feedback uses official labels; no checker-derived selector features.
            self._truth[category] = {r['id']: r for r in load_ground_truth_entry(category)}
        row = self._truth[category].get(tid)
        if row is None and 'relevance' not in category:
            raise ValueError(f'official feedback truth missing: {tid}')
        truth = [] if row is None else row.get('ground_truth', row.get('possible_answer'))
        if truth is None:
            raise ValueError(f'official feedback truth malformed: {tid}')
        return bfcl_task_rollout(self.entries[tid], category, truth, backend, parameters, generator, checker=checker)


def assert_run_invariants(state, ledger, *, complete=False):
    """Executable assertions over committed steps, folds, spend and exposure."""
    if ledger.remaining < 0 or ledger.spent > ledger.budget:
        raise AssertionError('hard budget violated')
    if ledger.window is not None and (ledger.window_remaining < 0 or
            ledger.window_purchases+len(ledger.reservations) > ledger.window['max_packages']):
        raise AssertionError('hard window budget violated')
    reveals = [e for e in ledger.events if e['kind'] == 'reveal']
    if len(reveals) != len(ledger.owned_ids):
        raise AssertionError('duplicate package charges')
    owned = set()
    for e in reveals:
        if not set(e['dependencies']) <= owned:
            raise AssertionError('dependency reveal order')
        owned.add(e['query_id'])
    steps = state.get('steps', [])
    if len({(e['round'], e['step']) for e in steps}) != len(steps):
        raise AssertionError('duplicate committed step')
    decisions = [e for e in steps if e['decision']]
    if any(e['step'] not in (1, 4, 7, 10) for e in decisions):
        raise AssertionError('wrong decision schedule')
    if any(e['inner_fold'] != (e['round']-1) % 2 for e in steps):
        raise AssertionError('fold rotation mismatch')
    for e in steps:
        if e.get('acquisition_protocol') in {'alpha_d_historical_additive_surrogate', 'alpha_d_batch_mixture_v11'}:
            if (e['source_actions'] != 2*e['slots'] or e['raw_new_slots'] > 20 or
                    e['exposure_units'] != e['slots'] or
                    e['teacher_evidence_units']+e['reference_pool_units'] != e['slots'] or
                    abs(sum(r['weight'] for r in e['exposure_records'])-1) > 1e-7):
                raise AssertionError('alpha/d exposure-unit accounting mismatch')
        elif e.get('acquisition_protocol') == 'batch_common_reference_v1':
            m, E = len(e['selected']), e['slots']
            if (m > e['max_new_packages'] or m > E or e['raw_new_slots'] != m
                    or e['weighted_new_slots'] != m or e['old_coefficient'] != 1-m/E):
                raise AssertionError('batch insertion exposure mismatch')
        elif e['selected'] is not None and not e['fixed_evidence']:
            if e['raw_new_slots'] != e['slots'] or e['weighted_new_slots'] != .25*e['slots']:
                raise AssertionError('insertion exposure mismatch')
        if e['exact_noop'] and e['start_hash'] != e['actual_hash']:
            raise AssertionError('no-op changed the student')
        if not e['audit_passed']:
            raise AssertionError('public selector audit failed')
    rounds = state.get('rounds', 3)
    if len([e for e in decisions if e['selected']]) > (1 if state['smoke'] else 4*rounds):
        raise AssertionError('too many new packages')
    if complete:
        expected = 1 if state['smoke'] else 12*rounds
        if len(steps) != expected or len(decisions) != (1 if state['smoke'] else 4*rounds):
            raise AssertionError('incomplete committed schedule')
        if ledger.reservations or set(state['owned']) != ledger.owned_ids:
            raise AssertionError('pending/unmerged evidence at completion')
    return dict(passed=True, steps=len(steps), decision_windows=len(decisions),
                packages=len(ledger.owned_ids), actual_spend=ledger.spent, authorized_budget=ledger.budget)


class RTDExperiment(AlphaDExperimentMixin, BatchExperimentMixin):
    def __init__(self, config, manifest, directory, backend, support, *, resume=False, smoke=False,
                 checker=None, journal=None, after_save=None):
        self.config, self.manifest = config, manifest
        validate_alpha_d_config(config)
        validate_arm(config, manifest['arm'])
        self.alpha_d = alpha_d_enabled(config)
        estimator, mode, _ = validate_source_config(config)
        if self.gate == 'linear_sigmoid' and (estimator == 'soft' or
                (estimator == 'cv' and mode == 'fixed_one_minus_a')):
            raise ValueError('soft/fixed_one_minus_a requires a fixed/scalar gate; use cv with loo/independent')
        self.v11 = config.get('protocol_version') == '1.1.0'
        self.directory, self.backend, self.support = Path(directory), backend, support
        self.device = next(iter(lora_parameters(backend.model).values())).device
        self.dtype = next(iter(lora_parameters(backend.model).values())).dtype
        self.directory.mkdir(parents=True, exist_ok=True)
        if self.alpha_d:
            from .metrics_v11 import options as metric_options, freeze_tasks, task_rows
            if metric_options(config)['enabled']:
                expected = freeze_tasks([dict(parent_hash=h, official_id=t) for h, t in support.parents.items()],
                                        short_fold=metric_options(config)['short_fold'], states=support.states)
                if manifest.get('fixed_task_set_v11', expected) != expected:
                    raise ValueError('fixed task set changed on resume')
                manifest['fixed_task_set_v11'] = expected
                task_rows(expected, support)
                atomic_json(self.directory/'manifest.json', manifest)
        self.journal = journal or ComputeJournal(self.directory / 'compute.jsonl', cuda=self.device.type == 'cuda')
        from .streaming_replay import enabled as streaming_enabled, prepare_manifest, StreamingSchedule
        streaming = streaming_enabled(config)
        if streaming:
            prepare_manifest(self)
        self.store = StateStore(self.directory / 'recovery', manifest)
        self.rng = np.random.default_rng(config['training_seed'])
        self.sampling_rng = torch.Generator(device=self.device).manual_seed(config['training_seed'])
        self.replay_schedule = None
        if self.alpha_d:
            from .conventions import load_schedule, ARMS
            if manifest['arm'] in ARMS:
                # Independently seeded streams; initialization/training seed remains 0.
                self.sampling_rng.manual_seed(int(digest([config['training_seed'], manifest['arm'], 'source_and_feedback'])[:15], 16))
            if manifest['arm'] == 'V1':
                self.replay_schedule = (StreamingSchedule(self, smoke=smoke) if streaming else
                                        load_schedule(config, manifest, support, smoke=smoke))
        self.slot_rng = torch.Generator().manual_seed(config['training_seed'])
        self.after_save = after_save
        self.checker = checker
        self.ceilings = manifest['budget_ceilings']
        self.ledger = (Ledger.resume(self.ceilings[0], self.directory / 'teacher.jsonl') if resume else
                       Ledger(self.ceilings[0], self.directory / 'teacher.jsonl'))
        self.broker = SealedReplayBroker(manifest['bank_path'], self.ledger, inner_parent_hashes=set(support.parents))
        if resume and self.store.pointer.exists():
            self.state = self.store.load(self.ledger, device=self.device)
            for n, p in lora_parameters(backend.model).items():
                with torch.no_grad():
                    p.copy_(self.state['parameters'][n])
            self.rng.bit_generator.state = self.state['numpy_rng']
            self.sampling_rng.set_state(self.state['sampling_rng'].cpu())
            self.slot_rng.set_state(self.state['slot_rng'].cpu())
            if 'inner' in self.state:
                self.broker.set_inner_parents(self.state['inner'])
        else:
            if resume and self.ledger.events:
                raise ValueError('ledger exists without an initial checkpoint; cannot reconstruct state')
            if self.store.pointer.exists():
                raise ValueError('existing run; use resume')
            parameters = snapshot(lora_parameters(backend.model))
            fixed_ids = config.get('fixed_evidence_ids', [])
            if config.get('fixed_initial_parameter_hash', tensor_state_hash(parameters)) != tensor_state_hash(parameters):
                raise ValueError('fixed ledger retraining must restore the identical initial student')
            self.state = dict(phase='initialize_fixed' if config['mode'] == 'fixed_evidence' else 'round_start', round=1, step=1,
                parameters=parameters, initial=snapshot(parameters), owned=[], steps=[], smoke=smoke,
                fixed_ids=fixed_ids, eta=config['initial_eta'], projection_cache={}, source_cache={},
                phi=torch.zeros(1 if config.get('gate') == 'scalar_sigmoid' else 35, device=self.device,
                                dtype=self.dtype, requires_grad=True), controller=GateController(),
                cost_model=CostRegressor(39), support_return=0., checkpoints=[])
            if self.v11:
                self.state.update(rounds=config.get('rounds', 3), batch_schema_version=1,
                    drift_measurements={}, previous_decision_model=None)
            if self.alpha_d:
                self.state['phi'] = torch.full((1 if config.get('gate') == 'scalar_sigmoid' else 33,),
                    config.get('gate_initial_logit', 0.), device=self.device, dtype=self.dtype, requires_grad=True)
                self.state['alpha_window_index'] = 0
            self.save()

        if streaming:
            self.replay_schedule.start()
        if resume and self.alpha_d and manifest['arm'] == 'V0':
            from .conventions import export_schedule
            export_schedule(self)

    def save(self):
        self.state.update(numpy_rng=self.rng.bit_generator.state, sampling_rng=self.sampling_rng.get_state(),
                          slot_rng=self.slot_rng.get_state())
        self.store.save(self.state, self.ledger)
        if self.after_save:
            self.after_save(self.state['phase'])

    def transition(self, phase):
        self.state['phase'] = phase
        self.save()

    @contextmanager
    def scope(self, label):
        previous = getattr(self.backend, 'context', 'unspecified')
        self.backend.context = f"r{self.state['round']}/s{self.state['step']}/{label}"
        try:
            with self.journal.measure_phase(label, round=self.state['round'], step=self.state['step']):
                yield
        finally:
            self.backend.context = previous

    def batches(self, items, operation):
        return memory_batches(items, self.device, MemoryPolicy.from_config(self.config),
            record=lambda row: self.journal.append('memory_batch', operation=operation,
                round=self.state['round'], step=self.state['step'], **row))

    def packages(self):
        return [self.broker.acquire(q) for q in self.state['owned']
                if self.broker._records[q].parent_hash in self.state['inner']]

    def sample_state(self, state, *, refresh=False, cache=True):
        s = self.state
        if state.parent_hash not in s['inner']:
            raise ValueError('source/feature state outside inner fold')
        if refresh or not cache or state.state_hash not in s['source_cache']:
            sources = []
            category = self.support.categories[self.support.parents[state.parent_hash]]
            def actions():
                count = getattr(self, 'config', {}).get('source_samples_per_state', 2)
                draws = sample_actions(self.backend, state.prompt, count, s['source'], self.sampling_rng)
                for _ in range(count):
                    with self.backend.action_limit(category) if hasattr(self.backend, 'action_limit') else nullcontext():
                        action = next(draws)
                    yield action
            with source_score_scope(self.backend, actions(), s['source']) as draws:
                for sample_index, action in enumerate(draws):
                    source = SourceSample(Behavior(state, action.text), s['source_id'], action.action_ids,
                                          action.eos_token_id, action.generation_logprob, truncated=action.truncated)
                    # Persist the full check BEFORE enforcing tolerances, including
                    # the failed action that previously disappeared from the log.
                    def record(diagnostic):
                        self.journal.append('score_consistency', **(diagnostic | dict(
                            round=s['round'], step=s['step'], role='source', sample_index=sample_index,
                            state_hash=state.state_hash, parent_hash=state.parent_hash, source_id=s['source_id'])))
                    with torch.no_grad():
                        checked, diagnostic = self.backend.checked_score_action(action, s['source'], record=record,
                            expected_prompt_ids=self.backend.tokenizer.encode(state.prompt, add_special_tokens=False))
                    score = float(checked)
                    error = diagnostic['sequence_abs_difference']
                    sources.append(source)
                    self.journal.append('source_sample', round=s['round'], state_hash=state.state_hash,
                        parent_hash=state.parent_hash, source_id=s['source_id'], action=asdict(action),
                        teacher_forced_logprob=score, score_discrepancy=error)
            sources = tuple(sources)
            if cache:
                s['source_cache'][state.state_hash] = sources
        else:
            sources = s['source_cache'][state.state_hash]
        if state.state_hash not in s['projection_cache']:
            initial_id = self.backend.identity(s['initial'])
            hidden = self.backend.initial_hidden(state.prompt, s['initial'], initial_snapshot_id=initial_id)
            projection = FrozenProjection(hidden.numel(), initial_snapshot_id=initial_id,
                                           seed=self.config['gate_projection_seed'])
            s['projection_cache'][state.state_hash] = projection(hidden, snapshot_id=initial_id)
        return sources

    def feature(self, source):
        state = source.behavior.state
        return FeatureRow(state.parent_hash, state.state_hash, self.backend.identity(self.state['initial']),
            PublicFeatures(self.state['projection_cache'][state.state_hash], source.logprob, source.length))

    def chi(self, source):
        standardizer = self.state['standardizer']
        h = source.behavior.state.state_hash
        # Newly PURCHASED states may be transformed with frozen moments; they
        # never refit the round's normalization or RMS preconditioner.
        if h not in standardizer.allowed_state_hashes:
            if h not in self.state['source_cache']:
                raise ValueError('unknown/unpurchased feature state')
            standardizer = replace(standardizer, allowed_state_hashes=standardizer.allowed_state_hashes | {h})
        chi = standardizer.transform(self.feature(source), device=self.device, dtype=self.dtype)
        return chi[-1:] if self.state['phi'].numel() == 1 else chi

    def pool(self, packages=None, *, package_only=False):
        by_state, teachers = {}, defaultdict(dict)
        if not package_only:
            for parent in self.state['inner']:
                if parent in self.support.states:
                    state = self.support.states[parent]
                    by_state[state.state_hash] = state
        for package in self.packages() if packages is None else packages:
            for behavior in package.behaviors:
                if behavior.state.parent_hash not in self.state['inner']:
                    raise ValueError('teacher from feedback fold')
                h = behavior.state.state_hash
                by_state[h] = behavior.state
                # Preserve empirical multiplicity across independently paid
                # requests. Event aliases within one package add no extra mass.
                teachers[h][package.query_id, behavior.text] = behavior
        by_parent = defaultdict(list)
        ordered = sorted(by_state.items())
        pending = [(h, state) for h, state in ordered if h not in self.state['source_cache']
                   or h not in self.state['projection_cache']]
        for batch in self.batches(pending, 'source_states'):
            for _, state in batch:
                self.sample_state(state)
        for h, state in ordered:
            by_parent[state.parent_hash].append((state, tuple(teachers[h].values())))
        return by_parent

    def draw_slots(self, packages=None, *, package_only=False, count=None):
        pool = self.pool(packages, package_only=package_only)
        parents = sorted(pool)
        if not parents:
            raise ValueError('no legal source states')
        targets = []
        controls, groups = [], {}
        cv = self.config.get('source_estimator', 'hard2') == 'cv'
        mode = self.config.get('cv_cs_mode', 'loo')
        for _ in range(self.slots if count is None else count):
            parent = parents[int(self.rng.integers(len(parents)))]
            states = pool[parent]
            state, teachers = states[int(self.rng.integers(len(states)))]
            if cv and state.state_hash not in groups:
                # Fresh independent draws for each new update. The coefficient
                # pool is sampled before exposure selection, never reconstructed
                # from replacement slots or cached previous-update outcomes.
                sources = self.sample_state(state, refresh=True)
                auxiliary = self.sample_state(state, cache=False) if mode == 'independent' else sources
                groups[state.state_hash] = (sources, auxiliary,
                    torch.stack([self.chi(sample) for sample in auxiliary]))
            sources = groups[state.state_hash][0] if cv else self.state['source_cache'][state.state_hash]
            index = int(self.rng.integers(len(sources)))
            source = sources[index]
            if cv:
                _, auxiliary, auxiliary_chi = groups[state.state_hash]
                controls.append(SourceControl(auxiliary, auxiliary_chi, index if mode == 'loo' else None))
            slot = TransportSlot(source, teachers)
            targets.append((source, slot.sample_teacher(self.slot_rng)))
        if cv:
            self.state['draw_controls'] = tuple(controls)
        return tuple(targets), torch.stack([self.chi(source) for source, _ in targets])

    @property
    def slots(self):
        if self.alpha_d:
            return self.config.get('slots_per_step', 40)
        if self.v11:
            return self.config.get('exposure_slots_per_window', 40)
        return 2 if self.state['smoke'] else 8

    @property
    def gate(self):
        return self.config.get('gate_override', 'fixed_half' if self.manifest['arm'] == 'R0' else self.config['gate'])

    @property
    def fixed(self):
        return self.config['mode'] == 'fixed_evidence'

    def estimator_options(self, controls=None):
        if self.config.get('source_estimator', 'hard2') == 'hard2':
            return {}  # Keep legacy calls and checkpoint state unchanged.
        return dict(source_estimator=self.config['source_estimator'], source_parameters=self.state['source'],
                    cv_cs_mode=self.config.get('cv_cs_mode', 'loo'), source_controls=controls)

    def gradient(self, targets, chi, parameters, *, pilot=False, controls=None):
        return streamed_gradient(targets, chi, self.state['phi'], self.backend, parameters,
                                 gate='fixed_half' if pilot else self.gate, **self.estimator_options(controls))

    def calibrate(self, packages):
        if self.alpha_d:
            return self.alpha_calibrate(packages)
        s = self.state
        targets, chi = self.draw_slots(packages, package_only=True)
        controls = s.pop('draw_controls', None)
        parameters = s['parameters']
        with self.scope('kl_pilot'):
            with self.scope('pilot_gradient'):
                g = self.gradient(targets, chi, parameters, pilot=True, controls=controls)
            # Linear proxy has exactly the streamed gradient, without retaining
            # multiple full-sequence graphs. kl_pilot differentiates it once.
            def loss(alpha):
                if alpha != .5:
                    raise ValueError('pilot alpha must be .5')
                return sum((parameters[n]*g[n].detach()).sum() for n in parameters)
            actions = [ActionTrace(tuple(self.backend.tokenizer.encode(src.behavior.state.prompt, add_special_tokens=False)),
                        src.token_ids, src.eos_token_id, src.behavior.text, src.logprob,
                        self.backend.backend_id, s['source_id'], truncated=src.truncated) for src, _ in targets]
            with self.scope('pilot_kl_trials'):
                eta, metadata = kl_pilot(parameters, s['diagonal'], loss,
                    lambda updated: self.backend.source_kl(actions, s['source'], updated),
                    candidates=self.config['pilot_eta_candidates'], owned_ids=self.ledger.owned_ids,
                    evidence_ids={p.query_id for p in packages}, inner_parent_hashes=s['inner'],
                    evidence_parents={b.state.parent_hash for p in packages for b in p.behaviors},
                    source_parents={src.behavior.state.parent_hash for src, _ in targets})
        s['eta'], s['calibrated'] = eta, True
        self.journal.append('kl_pilot', round=s['round'], metadata=metadata,
                            source_slots=len(actions), source_action_tokens=sum(len(a.action_ids) for a in actions))

    def feedback(self, parameters, label, *, trajectory_scores=None):
        s = self.state
        rollouts = []
        if hasattr(self.support, 'feedback_context'):
            self.checker = self.support.feedback_context(s['round'], self.backend, self.journal)
        with self.scope(label):
            for parent, count in s['feedback_tasks']:
                for rollout in feedback_rollouts(self.support, parent, count, self.backend,
                                                 parameters, self.sampling_rng, self.checker):
                    rollouts.append(rollout)
                    self.journal.append('feedback_rollout', round=s['round'], step=s['step'], role=label,
                                        parent_hash=parent, rollout=asdict(rollout))
            result = reinforce_gradient(rollouts, self.backend, parameters,
                diagnostic_record=lambda rollout, index, diagnostic: self.journal.append('score_consistency',
                    **(diagnostic | dict(round=s['round'], step=s['step'], role=label,
                        task_id=rollout.task_id, action_index=index))),
                baseline='smoke_zero' if s['smoke'] else 'leave_one_out_same_task',
                **(dict(trajectory_scores=trajectory_scores) if trajectory_scores is not None else {}))
        if self.alpha_d:
            result.metadata.update(return_objective='temperature_1_stochastic_policy_expected_return',
                                   temperature=1., feedback_role=label)
        self.journal.append('return_gradient', round=s['round'], step=s['step'], role=label,
                            parameter_hash=result.parameter_hash, metadata=result.metadata)
        if self.v11:
            s.setdefault('feedback_rollouts', {})[label] = tuple(rollouts)
        return result

    def choose_feedback_tasks(self):
        s = self.state
        groups = defaultdict(list)
        for parent in sorted(s['feedback']):
            tid = self.support.parents[parent]
            if parent in self.support.states:
                groups[self.support.categories[tid].startswith('multi_turn')].append(parent)
        result = []
        for multi, group in sorted(groups.items()):
            limit = 2 if s['smoke'] else (4 if multi else 8)
            count = 1 if s['smoke'] else (2 if multi else 4)
            for parent in self.rng.choice(group, size=min(limit, len(group)), replace=False):
                result.append((str(parent), count))
        if not result:
            raise ValueError('no legal feedback parents')
        return result

    def round_start(self):
        s = self.state
        fold = (s['round']-1) % 2
        parents = set(self.support.parents)
        if s['smoke']:
            # Stable content-hash slice of runnable single-turn parents; no labels.
            parents = set()
            for f in (0, 1):
                possible = sorted(h for h in self.support.states if int(h, 16) % 2 == f
                                  and not self.support.categories[self.support.parents[h]].startswith('multi_turn'))
                if len(possible) < 2:
                    raise ValueError('smoke needs two single-turn parents per fold')
                parents.update(possible[:2])
        s['inner'] = {h for h in parents if int(h, 16) % 2 == fold}
        s['feedback'] = parents - s['inner']
        self.broker.set_inner_parents(s['inner'])
        if hasattr(self.support, 'feedback_context'):
            self.checker = self.support.feedback_context(s['round'], self.backend, self.journal)
        if not self.fixed:
            self.ledger.authorize(self.ceilings[s['round']-1])
        s['source'] = snapshot(s['parameters'])
        s['source_id'] = self.backend.identity(s['source'])
        s['source_cache'] = {}
        if self.v11:
            if 'posterior' not in s:
                from .acquisition import CONTEXTUAL_DIMENSION
                s['posterior'] = ValuePosterior(CONTEXTUAL_DIMENSION if self.alpha_d else 39,
                    round_id=f"r{s['round']}", contextual_shrinkage=self.alpha_d)
            else:
                s['posterior'].begin_round(f"r{s['round']}")
            s['cost_model'].begin_round(f"r{s['round']}")
            s['drift_pending'] = True
        else:
            s['posterior'] = LegacyValuePosterior(39, round_id=f"r{s['round']}")
        s['calibrated'] = False
        with self.scope('round_sources_and_geometry'):
            with self.scope('source_sampling'):
                pool = self.pool()
            sources = [src for rows in pool.values() for state, _ in rows for src in self.sample_state(state)]
            initial_id = self.backend.identity(s['initial'])
            s['standardizer'] = FrozenStandardizer.fit([self.feature(src) for src in sources],
                inner_parent_hashes=s['inner'], allowed_state_hashes={src.behavior.state.state_hash for src in sources},
                initial_snapshot_id=initial_id)
            def source_gradients():
                for batch in self.batches(sources, 'preconditioner_sources'):
                    for src in batch:
                        yield src.behavior.state.parent_hash, gradients(-self.backend.score_source(src, s['source']), s['source'])
            with self.scope('source_preconditioner'):
                diagonal, metadata = rms_diagonal(s['parameters'], source_gradients(), inner_parent_hashes=s['inner'],
                                                   source_snapshot_id=s['source_id'])
            if self.config.get('preconditioner') == 'identity':
                diagonal = {n: torch.ones_like(p) for n, p in diagonal.items()}
                metadata['component_override'] = 'identity'
            s['diagonal'], s['preconditioner_metadata'] = diagonal, metadata
            packages = self.packages()
            if packages:
                self.calibrate(packages)
        self.journal.append('round_frozen', round=s['round'], inner=sorted(s['inner']), feedback=sorted(s['feedback']),
            source_id=s['source_id'], standardizer=asdict(s['standardizer']) | {
                'inner_parent_hashes': sorted(s['standardizer'].inner_parent_hashes),
                'allowed_state_hashes': sorted(s['standardizer'].allowed_state_hashes)},
            preconditioner=metadata, eta=s['eta'], pilot_deferred=not s['calibrated'])
        self.transition('step_start')

    def step_start(self):
        if self.alpha_d:
            return self.alpha_step_start()
        s = self.state
        s['decision'] = s['step'] in (1, 4, 7, 10)
        s['selected'], s['label'], s['new_targets'], s['new_chi'] = None, None, None, None
        if self.v11:
            s.update(selected=[], pending_ids=[], batch_targets={}, batch_chi={}, labels={},
                     feedback_rollouts={}, transaction_index=0, transaction_query=None,
                     window_id=f"r{s['round']}/s{s['step']}", hard_stops=[])
            if self.config.get('source_estimator') == 'cv':
                s['batch_controls'] = {}
        s['owned_before'] = tuple(s['owned'])
        s['old_targets'], s['old_chi'] = self.draw_slots()
        if self.config.get('source_estimator') == 'cv':
            s['old_controls'] = s.pop('draw_controls')
        # no-op tests availability in the actual sampled finite loss, not merely
        # whether some other state in the owned pool has a teacher response.
        s['old_noop'] = is_exact_noop(has_teacher_evidence=any(t is not None for _, t in s['old_targets']),
            student_snapshot_id=self.backend.identity(s['parameters']), frozen_snapshot_id=s['source_id'])
        s['feedback_tasks'] = self.choose_feedback_tasks() if s['decision'] else []
        s['step_rule'] = FrozenStep(s['diagonal'], s['eta'], f"r{s['round']}", s['preconditioner_metadata'])
        with self.scope('reference_gradient'):
            g = None if s['old_noop'] else self.gradient(s['old_targets'], s['old_chi'], s['parameters'],
                                                       controls=s.get('old_controls'))
            s['reference'] = InsertionReference(s['parameters'], None, s['step_rule'], old_slots=self.slots,
                exact_noop=s['old_noop'], smoke=s['smoke'], old_gradient=g,
                **(dict(exposure_slots=self.slots) if self.v11 else {}))
        if s['decision']:
            s['reference_feedback'] = self.feedback(s['reference'].updated, 'reference_feedback')
        self.transition('reference')

    def reference(self):
        if self.v11:
            return self.batch_reference()
        s = self.state
        s['trace'] = []
        if s['decision'] and not self.fixed:
            features = tuple((h, self.feature(sources[0]).features) for h, sources in s['source_cache'].items())
            view = StudentSnapshot(s['source_id'], frozenset(s['inner']), features)
            candidates = self.broker.list_candidates(view, self.ledger.owned_ids, self.ledger.remaining)
            # No hidden state access: all pre-purchase features come from official
            # seed states. A missing public request state is an explicit error.
            s['candidate_specs'] = {c.query_id: c for c in candidates}
            projections = [self.state['projection_cache'][b.state.state_hash] for p in self.packages() for b in p.behaviors]
            def coverage(spec):
                if not projections:
                    return 0.
                x = np.asarray(spec.features.projection)
                return float(max(np.dot(x, p) / max(np.linalg.norm(x)*np.linalg.norm(p), 1e-12) for p in projections))
            rows = {c.query_id: PrePurchaseFeatures.from_public(c, round_id=f"r{s['round']}", coverage=coverage(c),
                     progress=((s['round']-1)*12+s['step']-1)/36, support_return=s['support_return']) for c in candidates}
            policy = AcquisitionPolicy(s['posterior'], s['cost_model'], rng=self.rng)
            remaining_windows = 1 if s['smoke'] else sum(step >= s['step'] for step in (1, 4, 7, 10))
            selected, trace = select_public(candidates, lambda public: policy.choose(public, rows,
                remaining_budget=self.ledger.remaining, remaining_windows=remaining_windows,
                random_control=self.manifest['arm'] == 'R0' or self.config.get('acquisition') == 'random'))
            s['selected'], s['trace'], s['selection'] = selected, trace, policy.last_decision
            s['selected_features'] = rows.get(selected)
        else:
            s['selection'] = dict(selected=None, fixed_evidence=self.fixed, replay_step=not s['decision'])
        s['audit_passed'] = self.broker.assert_no_hidden_access(s['trace']).passed
        if not s['audit_passed']:
            raise AssertionError('selector read audit failed')
        if s['decision']:
            self.journal.append('decision', round=s['round'], step=s['step'], selection=s['selection'],
                                selector_trace=s['trace'], audit_passed=s['audit_passed'])
        # Choice + RNG are durable BEFORE the first possible reservation/reveal.
        self.transition('selected')

    def selected(self):
        if self.v11:
            return self.batch_selected()
        s = self.state
        if s['selected']:
            # Resume may be just after a durable reveal. Rehydrate that exact
            # package, never select again and never charge it again.
            self.broker._offered.add(s['selected'])
            package = self.broker.acquire(s['selected'])
            self.journal.append('request_reveal_link', round=s['round'], step=s['step'], query_id=package.query_id,
                ledger_reveal_sequence=next(e['sequence'] for e in self.ledger.events
                                           if e['kind'] == 'reveal' and e['query_id'] == package.query_id))
            with self.scope('pending_source_sampling'):
                s['new_targets'], s['new_chi'] = self.draw_slots([package], package_only=True)
                if self.config.get('source_estimator') == 'cv':
                    s['new_controls'] = s.pop('draw_controls')
            if not s['calibrated']:
                if not s['old_noop']:
                    # A round with no inner teacher carries eta as specified.
                    # A deferred pilot may only change an identity reference.
                    raise ValueError('first-purchase pilot would change a nonidentity reference')
                self.calibrate([package])
                s['step_rule'] = FrozenStep(s['diagonal'], s['eta'], f"r{s['round']}", s['preconditioner_metadata'])
                s['reference'] = InsertionReference(s['parameters'], None, s['step_rule'], old_slots=self.slots,
                    exact_noop=True, smoke=s['smoke'])
                if s['reference_feedback'].parameter_hash != s['reference'].reference_hash:
                    raise ValueError('pilot changed the identity reference')
        self.transition('revealed')

    def revealed(self):
        if self.alpha_d:
            return self.alpha_revealed()
        if self.v11:
            return self.batch_revealed()
        s = self.state
        ref = s['reference']
        if s['selected']:
            with self.scope('new_package_gradient'):
                gq = self.gradient(s['new_targets'], s['new_chi'], ref.start, controls=s.get('new_controls'))
                actual, label, accounting = ref.insert(s['selected'], None, s['reference_feedback'],
                    label_type=LabelType.PENDING_NEW, owned_before=s['owned_before'], pending_ids={s['selected']},
                    new_slots=self.slots, new_gradient=gq)
                s['actual'], s['label'], s['accounting'] = snapshot(actual), label, accounting
        else:
            actual, accounting = ref.empty()
            s['actual'], s['accounting'] = snapshot(actual), accounting
        self.transition('actual')

    def actual(self):
        if self.alpha_d:
            return self.alpha_actual()
        if self.v11:
            return self.batch_actual()
        s = self.state
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
                        vjp = streamed_gate_vjp(s['old_targets'], s['old_chi'], s['phi'], self.backend,
                                                s['reference'].start, s['step_rule'], feedback, gate=self.gate,
                                                **self.estimator_options(s.get('old_controls')))
                    if s['selected']:
                        vjp = .75*vjp + .25*streamed_gate_vjp(s['new_targets'], s['new_chi'], s['phi'], self.backend,
                                                s['reference'].start, s['step_rule'], feedback, gate=self.gate,
                                                **self.estimator_options(s.get('new_controls')))
                    s['next_phi'], meta = s['controller'].update(s['phi'], vjp)
                self.journal.append('gate_update', round=s['round'], step=s['step'], vjp=vjp.tolist(),
                    next_phi=s['next_phi'].detach().tolist(), identifiable=feedback.metadata['identifiable'], **meta)
            else:
                s['next_phi'] = s['phi']
        else:
            s['next_phi'] = s['phi']
        self.transition('feedback')

    def feedback_commit(self):
        if self.alpha_d:
            return self.alpha_feedback_commit()
        if self.v11:
            return self.batch_feedback_commit()
        s = self.state
        commit_step(self.backend.model, s['actual'], expected_start_hash=s['reference'].start_hash)
        s['parameters'] = snapshot(lora_parameters(self.backend.model))
        s['phi'] = s['next_phi'].detach().requires_grad_(True)
        if s['selected']:
            q = s['selected']
            s['owned'].append(q)
            s['posterior'].observe(s['selected_features'], s['label'])
            package = self.broker.acquire(q)
            s['cost_model'].observe_revealed(s['candidate_specs'][q], s['selected_features'], cost=package.cost,
                confidence=package.cost_confidence, revealed_ids=self.ledger.owned_ids)
        exact_noop = s['old_noop'] and s['selected'] is None
        row = dict(round=s['round'], step=s['step'], decision=s['decision'], inner_fold=(s['round']-1)%2,
            selected=s['selected'], slots=self.slots, fixed_evidence=self.fixed,
            exact_noop=exact_noop, start_hash=s['reference'].start_hash, actual_hash=tensor_state_hash(s['parameters']),
            audit_passed=s['audit_passed'], raw_old_slots=0 if s['old_noop'] else self.slots,
            raw_new_slots=self.slots if s['selected'] else 0, weighted_new_slots=.25*self.slots if s['selected'] else 0,
            weighted_old_slots=(.75 if s['selected'] else 1)*self.slots if not s['old_noop'] else 0,
            authorized_budget=self.ledger.budget, actual_spend=self.ledger.spent, eta=s['eta'],
            selection=s['selection'], label=asdict(s['label']) if s['label'] else None)
        def exposure(targets):
            def teacher_tokens(teacher):
                ids = self.backend.tokenizer.encode(teacher.text, add_special_tokens=False)
                return len(ids) + int(not ids or ids[-1] != self.backend.tokenizer.eos_token_id)
            return dict(source_action_tokens=sum(src.length for src, _ in targets),
                teacher_action_tokens=sum(teacher_tokens(t) for _, t in targets if t is not None),
                prompt_tokens=sum(len(self.backend.tokenizer.encode(src.behavior.state.prompt, add_special_tokens=False))
                                  * (2 if t is not None else 1) for src, t in targets),
                teacher_slots=sum(t is not None for _, t in targets))
        row['old_exposure'] = exposure(()) if s['old_noop'] else exposure(s['old_targets'])
        row['new_exposure'] = exposure(s['new_targets'] or ())
        if s['decision']:
            reference, actual = s['reference_feedback'], s['actual_feedback']
            def counts(feedback):
                return {key: feedback.metadata.get(key, 0) for key in
                        ('rollouts', 'truncated_rollouts', 'truncated_actions')}
            reused = actual.parameter_hash == reference.parameter_hash
            ref_counts, actual_counts = counts(reference), counts(actual)
            row['truncation'] = dict(reference=ref_counts, actual=actual_counts, actual_feedback_reused=reused,
                **{key: ref_counts[key] + (0 if reused else actual_counts[key]) for key in ref_counts})
            self.journal.append('window_truncation', round=s['round'], step=s['step'], **row['truncation'])
            def malformed_counts(feedback):
                return {key: feedback.metadata.get(key, 0) for key in
                        ('rollouts', 'malformed_rollouts', 'malformed_actions')}
            ref_bad, actual_bad = malformed_counts(reference), malformed_counts(actual)
            exceptions = dict(reference.metadata.get('malformed_exception_types', {}))
            if not reused:
                for name, count in actual.metadata.get('malformed_exception_types', {}).items():
                    exceptions[name] = exceptions.get(name, 0) + count
            row['malformed_feedback'] = dict(reference=ref_bad, actual=actual_bad,
                actual_feedback_reused=reused, malformed_exception_types=exceptions,
                **{key: ref_bad[key] + (0 if reused else actual_bad[key]) for key in ref_bad})
            self.journal.append('window_malformed', round=s['round'], step=s['step'], **row['malformed_feedback'])
        row['source_truncated_slots'] = sum(src.truncated for src, _ in
            (*s['old_targets'], *(s['new_targets'] or ())))
        s['steps'].append(row)
        assert_run_invariants(s, self.ledger)
        # Recovery uses this checkpoint as the commit record. Auxiliary JSON
        # artifacts are rebuilt from it and cannot advance the experiment.
        self.transition('committed')

    def committed(self):
        s = self.state
        if self.v11:
            if s['decision'] and not self.fixed:
                self.ledger.close_window(s['window_id'])
        row = s['steps'][-1]
        path = self.directory / 'steps' / f"r{s['round']}-s{s['step']:02d}.json"
        atomic_json(path, row | dict(old_slots=[dict(source=asdict(src), teacher=asdict(t) if t else None)
                                               for src, t in s['old_targets']],
                                   new_slots=[dict(source=asdict(src), teacher=asdict(t) if t else None)
                                               for src, t in (s['new_targets'] or ())], selector_trace=s['trace']))
        # All fields belong to the completed finite update, and can now be freed.
        for key in ('reference', 'reference_feedback', 'actual_feedback', 'actual', 'old_targets', 'old_chi',
                    'new_targets', 'new_chi', 'candidate_specs', 'selected_features', 'next_phi'):
            s.pop(key, None)
        if self.v11:
            for key in ('feedback_rollouts', 'batch_targets', 'batch_chi', 'batch_gradients', 'batch_rows',
                        'batch_specs', 'value_statistics', 'reliability_check',
                        'old_controls', 'batch_controls', 'draw_controls'):
                s.pop(key, None)
        if self.alpha_d:
            from .conventions import export_schedule
            export_schedule(self)
            for key in ('alpha_pairs', 'alpha_chi', 'alpha_start', 'd_reference', 'd_reference_feedback',
                        'd_solution', 'alpha_d_control', 'old_reference', 'old_reference_pairs', 'replay_exposure',
                        'acquisition_feedback_statistic'):
                s.pop(key, None)
        print(f"[rtd] {self.manifest['arm']} round={s['round']} step={s['step']} "
              f"package={row['selected']} spend={self.ledger.spent}/{self.ledger.budget}", flush=True)
        if s['smoke'] or s['step'] == 12:
            self.transition('round_end')
        else:
            s['step'] += 1
            self.transition('step_start')

    def round_end(self):
        import os
        import shutil
        s = self.state
        if self.alpha_d:
            from .metrics_v11 import options as metric_options
            if metric_options(self.config)['enabled']:
                from .experiment_metrics_v11 import round_metrics
                metric_path = self.directory/'metrics'/f"round-{s['round']}.json"
                if metric_path.exists():
                    metric = json.loads(metric_path.read_text())
                    if (metric['parameter_hash'] != tensor_state_hash(s['parameters']) or
                            metric['fixed_task_set_hash'] != self.manifest['fixed_task_set_v11']['hash']):
                        raise ValueError('saved round metrics belong to another frozen checkpoint/task set')
                else:
                    round_metrics(self)
        directory = self.directory / f"round-{s['round']}"
        expected_hash = tensor_state_hash(s['parameters'])
        if directory.exists():
            meta = json.loads((directory / 'checkpoint.json').read_text())
            if (meta['parameter_hash'] != expected_hash or meta['adapter_hash'] != tree_hash(directory / 'lora')
                    or meta['round_state_hash'] != file_hash(directory / 'round_state.pt')):
                raise ValueError('saved round checkpoint was changed')
        else:
            temporary = directory.with_name(directory.name + '.tmp')
            if temporary.exists():
                shutil.rmtree(temporary)
            temporary.mkdir()
            self.backend.model.save_pretrained(temporary / 'lora', safe_serialization=True)
            self.backend.tokenizer.save_pretrained(temporary / 'lora')
            meta = dict(round=s['round'], parameter_hash=expected_hash, adapter_hash=tree_hash(temporary / 'lora'),
                manifest_hash=manifest_hash(self.manifest), source_id=s['source_id'], authorized_budget=self.ledger.budget,
                actual_spend=self.ledger.spent, owned=sorted(s['owned']), config_hash=self.manifest['config_hash'])
            # Source traces/moments are retained independently of rolling recovery.
            with (temporary / 'round_state.pt').open('wb') as stream:
                round_state = {k: s[k] for k in ('source', 'source_cache', 'diagonal', 'standardizer', 'eta',
                                               'phi', 'posterior', 'cost_model', 'controller')}
                if self.v11:
                    round_state['batch_state'] = {k: v for k, v in s.items() if k not in round_state}
                torch.save(round_state, stream)
                stream.flush(); os.fsync(stream.fileno())
            meta['round_state_hash'] = file_hash(temporary / 'round_state.pt')
            atomic_json(temporary / 'checkpoint.json', meta)
            for path in (temporary / 'lora').rglob('*'):
                if path.is_file():
                    with path.open('rb') as stream:
                        os.fsync(stream.fileno())
            fsync_directory(temporary / 'lora')
            fsync_directory(temporary)
            os.replace(temporary, directory)
            fsync_directory(self.directory)
        s['checkpoints'].append(meta)
        atomic_json(self.directory / 'trajectory.json', dict(checkpoints=s['checkpoints'], steps=s['steps']))
        if s['smoke'] or s['round'] == s.get('rounds', 3):
            self.transition('complete')
        else:
            s['round'] += 1
            s['step'] = 1
            self.transition('round_start')

    def initialize_fixed(self):
        """Section 10.1: charge the final ledger collection, then retrain all cells."""
        s = self.state
        self.ledger.authorize(max(self.ceilings))
        pending = set(s['fixed_ids']) - self.ledger.owned_ids
        while pending:
            view = StudentSnapshot('fixed-evidence-initialization', frozenset(self.support.parents))
            candidates = self.broker.list_candidates(view, self.ledger.owned_ids, self.ledger.remaining)
            legal = sorted(q.query_id for q in candidates if q.query_id in pending)
            if not legal:
                raise ValueError('fixed ledger has missing dependencies, unavailable packages or insufficient caps')
            self.broker.acquire(legal[0])
            pending.remove(legal[0])
        s['owned'] = sorted(self.ledger.owned_ids)
        self.transition('round_start')

    def run(self, *, stop_after_round=False):
        phases = dict(round_start=self.round_start, step_start=self.step_start, reference=self.reference,
            selected=self.selected, revealed=self.revealed, actual=self.actual, feedback=self.feedback_commit,
            committed=self.committed, round_end=self.round_end, initialize_fixed=self.initialize_fixed)
        while self.state['phase'] != 'complete':
            if stop_after_round and self.state['checkpoints'] and self.state['checkpoints'][-1]['round'] >= int(stop_after_round):
                return dict(round_ready=self.state['checkpoints'][-1]['round'], complete=False)
            phase = self.state['phase']
            if (phase == 'round_end' and self.config.get('replay_mode') == 'streaming'
                    and self.manifest['arm'] == 'V1'
                    and (self.state['smoke'] or self.state['round'] == self.state['rounds'])):
                # Final integrity must succeed before publishing the last round
                # checkpoint; coordinator resume must not skip this barrier.
                self.replay_schedule.finish()
            if phase in {'initialize_fixed', 'round_end'}:
                print(f"[rtd] round={self.state['round']} step={self.state['step']} phase={phase}", flush=True)
                phases[phase]()
                continue
            round_number, step_number = self.state['round'], self.state['step']
            with self.journal.measure_step(round=round_number, step=step_number, start_phase=phase):
                while ((self.state['round'], self.state['step']) == (round_number, step_number)
                       and self.state['phase'] not in {'round_end', 'complete'}):
                    phase = self.state['phase']
                    print(f"[rtd] round={round_number} step={step_number} phase={phase}", flush=True)
                    phases[phase]()
        if self.config.get('replay_mode') == 'streaming' and self.manifest['arm'] == 'V1':
            self.replay_schedule.finish()
        result = assert_run_invariants(self.state, self.ledger, complete=True)
        atomic_json(self.directory / 'audit.json', result)
        return result
