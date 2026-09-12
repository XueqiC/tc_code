"""P1 fixed-pool runner on the durable v1.1 scheduler and benchmark harnesses.

D1 delegates alpha/d to the existing mixin. Other arms build the P0 affine
problem with streamed gradients, solve it locally, and commit once. P1 disables
acquisition fitting/extra validation rollouts for ALL arms: purchases and slots
are replayed, while the three feedback roles retain matched compute protocols.
"""
from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import torch

from ..experiment import RTDExperiment
from ..functional_step import FrozenStep, commit_step, lora_parameters, snapshot
from ..persistence import atomic_json, digest, file_hash
from ...behavior.deltas import tensor_state_hash
from .arms import solve_arm
from .config import runtime_config
from .engine import UnifiedConfig
from .execution import commit_distillation
from .immutable import Parameters, Array
from .objective import TeachingObjective
from .problem import TeachingContext, build_teaching_problem
from .profiling import profile
from .scoring import score_streamed


class P1Experiment(RTDExperiment):
    def __init__(self, config, manifest, directory, backend, support, **kwargs):
        if not config.get('replay_schedule') and not kwargs.get('smoke', False):
            raise ValueError('P1 requires a recorded V0 exposure schedule')
        self.p1_config = config
        self.unified = UnifiedConfig.from_config(config)
        super().__init__(runtime_config(config), manifest, directory, backend, support, **kwargs)

    @property
    def slots(self):
        return self.p1_config['p1']['smoke_slots'] if self.state['smoke'] else super().slots

    @property
    def max_new_packages(self):
        return self.p1_config['p1']['smoke_packages'] if self.state['smoke'] else super().max_new_packages

    def stage(self, name):
        return profile(self.journal, name, round=self.state['round'], step=self.state['step'], arm=self.manifest['arm'])

    def save(self):
        with self.stage('save'):
            super().save()

    @contextmanager
    def scope(self, label):
        stage = {'source_sampling': 'student_sampling', 'source_preconditioner': 'sketch_metric_gram',
            'alpha_d_commit_source_sampling': 'student_sampling',
            'alpha_d_old_virtual_reference': 'source_teacher_soft_gradients',
            'alpha_d_same_batch_reference': 'source_teacher_soft_gradients',
            'alpha_d_blocked_gate_vjp': 'controller', 'commit': 'commit',
            'alpha_d_qp': 'qp', 'alpha_d_gram': 'sketch_metric_gram'}.get(label)
        if stage:
            with self.stage(stage), super().scope(label):
                yield
        else:
            with super().scope(label):
                yield

    def alpha_calibrate(self, packages):
        # A common predeclared eta avoids arm-dependent pilot-selected doses.
        self.state.update(eta=self.p1_config['initial_eta'], calibrated=True)
        self.journal.append('p1_shared_eta', eta=self.state['eta'], selection='preregistered_shared_fixed_eta')

    def choose_feedback_tasks(self):
        s = self.state
        rng = np.random.default_rng(self.role_seed('feedback_tasks'))
        tasks = []
        for multi in (False, True):
            group = sorted(h for h in s['feedback'] if h in self.support.states and
                self.support.categories[self.support.parents[h]].startswith('multi_turn') == multi)
            limit = self.config['meta_tasks_multi_turn' if multi else 'meta_tasks_per_feedback']
            count = self.config['rollouts_multi_turn' if multi else 'rollouts_per_meta_task']
            if s['smoke']:
                limit, count = self.p1_config['p1']['smoke_feedback_tasks'], self.p1_config['p1']['smoke_rollouts']
            tasks.extend((str(h), count) for h in rng.choice(group, min(limit, len(group)), replace=False))
        if not tasks:
            raise ValueError('no legal P1 feedback tasks')
        return tasks

    def role_seed(self, role):
        return int(digest([self.config['training_seed'], self.state['round'], self.state['step'], role])[:15], 16)

    def alpha_draw_pairs(self, records, *, role):
        self.sampling_rng.manual_seed(self.role_seed('sources/'+role))
        return super().alpha_draw_pairs(records, role=role)

    def alpha_feedback(self, parameters, role, *, scores=None, isolated=False):
        previous = self.sampling_rng
        self.sampling_rng = torch.Generator(device=self.device).manual_seed(self.role_seed('feedback/'+role))
        try:
            with self.stage('feedback_rollouts'):
                return super().alpha_feedback(parameters, role, scores=scores, isolated=False)
        finally:
            self.sampling_rng = previous

    def selected(self):
        with self.stage('teacher_replay'):
            return super().selected()

    def alpha_acquisition_labels(self):
        self.state.update(labels={}, value_statistics=dict(query_ids=[], values=[], covariance=[]))

    def alpha_validate_pairs(self):
        pass  # Fixed P1 protocol: no extra arm-specific fitting/diagnostic rollouts.

    def make_problem(self, pairs, weights, role):
        s, c = self.state, self.unified
        layout = Parameters.of(s['parameters'])
        context = TeachingContext(layout, s['source'], s['source_id'], layout.flatten(s['diagonal']),
            loss=c.loss, eta=s['eta'], a_ref_default=c.a_ref, regularization=c.regularization,
            qp_tolerance=c.qp_tolerance, qp_max_iterations=c.qp_max_iterations,
            psd_tolerance=c.psd_tolerance, estimator=c.estimator,
            exposure_version=s['window_id']+'/'+role, sampling_version=s['window_id']+'/'+role,
            optimizer_version=f"r{s['round']}", owned_query_ids=tuple(sorted(self.ledger.owned_ids)),
            inner_parent_hashes=tuple(sorted(s['inner'])))
        exposures, evidence, bindings = [], {}, {}
        with self.stage('source_teacher_soft_gradients'):
            for index, (pair, weight) in enumerate(zip(pairs, weights)):
                slot = f'{role}/{index}'
                e, t = score_streamed(self.backend, pair, context, s['parameters'], s['source'],
                                     slot_id=slot, weight=float(weight))
                exposures.append(e); bindings[slot] = None if t is None else t.key
                if t is not None:
                    evidence[t.key] = t
        with self.stage('sketch_metric_gram'):
            return build_teaching_problem(context, evidence.values(), exposures, evidence_by_slot=bindings)

    def baseline_parameters(self, pairs, weights):
        from appworld_train import rtd_sft_kl_gradient
        s = self.state
        d0 = self.p1_config['p1']['d0']
        with self.stage('source_teacher_soft_gradients'):
            g = rtd_sft_kl_gradient(self.backend, pairs, weights, s['parameters'], s['source'],
                mode=d0['mode'], teacher_weight=d0['teacher_weight'], kl_weight=d0['kl_weight'],
                normalization=self.unified.loss.normalization)
        with self.stage('sketch_metric_gram'):
            return {n: (p-s['eta']*s['diagonal'][n]*g[n]).detach().requires_grad_(True)
                    for n, p in s['parameters'].items()}

    def alpha_revealed(self):
        if self.manifest['arm'] == 'D1':
            return super().alpha_revealed()
        s = self.state
        pending = [self.broker.acquire(q) for q in s['selected']]
        pending = [p for p in pending if p.behaviors and self.broker.training_eligible(p.query_id)]
        self.alpha_calibrate(pending)
        old = self.alpha_old_pool()
        # Identical public record choices to the alpha/d procedure, then exact
        # replay overrides choices/weights from the one recorded schedule.
        new = list(pending)
        if len(new) > self.slots:
            new = [new[i] for i in sorted(self.rng.choice(len(new), self.slots, replace=False))]
        records = []
        for package in new:
            choices = tuple(self.supervision_records(package, is_new=True))
            records.append(choices[int(self.rng.integers(len(choices)))])
        old_records = [old[int(self.rng.integers(len(old)))] for _ in range(self.slots)]
        records.extend(old_records[len(records):])
        weights = old_weights = [1/self.slots]*self.slots
        if s['replay_exposure']:
            replay = s['replay_exposure']
            available = list(old)+[r for p in pending for r in self.supervision_records(p, is_new=True)]
            records, weights = self.alpha_replay_records(replay['records'], available)
            old_records, old_weights = self.alpha_replay_records(replay['reference_records'], old)
        if len(records) != self.slots or len(old_records) != self.slots:
            raise ValueError('P1 replay exposure count changed')
        with self.stage('student_sampling'):
            s['alpha_pairs'] = self.alpha_draw_pairs(records, role='commit')
            s['old_reference_pairs'] = self.alpha_draw_pairs(old_records, role='virtual_reference')
        s['step_rule'] = FrozenStep(s['diagonal'], s['eta'], f"r{s['round']}", s['preconditioner_metadata'])
        d0 = self.manifest['arm'] == 'D0'
        if d0:
            theta_ref = self.baseline_parameters(s['alpha_pairs'], weights)
            old_theta = self.baseline_parameters(s['old_reference_pairs'], old_weights)
            problem = None
        else:
            problem = self.make_problem(s['alpha_pairs'], weights, 'commit')
            theta_ref = problem.reference_parameters.tensors(device=self.device)
            old_problem = self.make_problem(s['old_reference_pairs'], old_weights, 'virtual_reference')
            old_theta = old_problem.reference_parameters.tensors(device=self.device)
            del old_problem
        alpha = [self.unified.a_ref if r.teacher is not None else 0. for r in records]
        s['d_reference'] = SimpleNamespace(weights=torch.tensor(weights, dtype=torch.float64),
            alpha=torch.tensor(alpha, dtype=torch.float64), theta0=theta_ref,
            start_hash=tensor_state_hash(s['parameters']))
        s['old_reference'] = SimpleNamespace(weights=torch.tensor(old_weights, dtype=torch.float64))
        s['reference'] = SimpleNamespace(updated=old_theta, reference_hash=tensor_state_hash(old_theta))
        if s['decision']:
            s['feedback_tasks'] = self.choose_feedback_tasks()
            s['reference_feedback'] = self.alpha_feedback(old_theta, 'acquisition_reference_feedback')
            scores = []
            feedback = self.alpha_feedback(theta_ref, 'same_batch_reference_feedback', scores=scores)
            s['d_reference_feedback'] = feedback
            s['d_feedback'] = self.alpha_statistic(feedback, scores, 'same_batch_reference_feedback')
        statistic = s['d_feedback']
        age = (s['round']-1)*12+s['step']-statistic.refreshed_step
        if not d0:
            with self.stage('sketch_metric_gram'):
                layout = problem.context.theta
                directions = [layout.tensors(values=v, requires_grad=False) for v in problem.U.numpy().T]
                _, error = statistic.project(directions, 'loo')
                problem = replace(problem, h=layout.flatten(statistic.gradient), epsilon=Array.of(error),
                    feedback_version=f"{statistic.refreshed_step}", feedback_hash=digest(dict(
                        parameter_hash=statistic.parameter_hash, rewards=statistic.rewards, tasks=statistic.task_ids)))
                if s['decision'] and statistic.parameter_hash != problem.reference_parameters.hash:
                    raise ValueError('P1 feedback reference mismatch')
        with self.stage('qp'):
            if d0:
                coefficients, permutation, solution = [], [], None
            else:
                solution, coefficients, permutation = solve_arm(problem, self.manifest['arm'],
                                                                 seed=self.role_seed('source_permutation'))
        with self.stage('controller'):
            s['alpha_d_control'] = dict(arm=self.manifest['arm'], warmup=False, feedback_age=age,
                coefficients=list(map(float, coefficients)), permutation=list(map(int, permutation)),
                solver=None if solution is None else dict(status=solution.status, iterations=solution.iterations,
                    stationarity=solution.stationarity, feasibility=solution.feasibility,
                    full_objective=solution.value, numerical_scale=solution.numerical_scale),
                estimator='teacher_sft_plus_source_prefix_kl' if d0 else self.unified.estimator,
                loss_normalization=self.unified.loss.normalization, controller='exact_only_no_predictor')
        with self.stage('commit'):
            s['alpha_start'] = s['parameters']
            if d0:
                commit_step(self.backend.model, theta_ref, expected_start_hash=s['d_reference'].start_hash)
            else:
                commit = commit_distillation(problem, coefficients, model=self.backend.model)
                s['alpha_d_control'].update(problem_id=problem.identity, increment_error=commit.increment_error,
                    full_objective_committed=TeachingObjective(problem).value(coefficients),
                    psd_repair=vars(problem.psd_repair))
            s['parameters'] = snapshot(lora_parameters(self.backend.model))
            s['actual'] = s['parameters']
            self.journal.append('p1_commit', round=s['round'], step=s['step'], backbone_updates=1,
                                rl_updates=0, **s['alpha_d_control'])
            self.transition('actual')

    def alpha_actual(self):
        if self.manifest['arm'] == 'D1':
            return super().alpha_actual()
        s = self.state
        s['next_phi'] = s['phi']
        if s['decision']:
            feedback = self.alpha_feedback(s['actual'], 'post_commit_feedback')
            s['actual_feedback'] = feedback
            s['support_return'] = float(np.mean(feedback.metadata['rewards']))
        self.alpha_acquisition_labels()
        self.transition('feedback')

    def round_end(self):
        self.finish_complete_replay()
        with self.stage('save'):
            return super().round_end()

    def finish_complete_replay(self):
        """Publish the same completed replay evidence for either launch mode."""
        if (self.config.get('replay_mode') != 'complete'
                or 'replay_schedule_identity' not in self.manifest):
            return
        rows = self.state['steps']
        if len(rows) != (1 if self.state['smoke'] else 12*self.state['rounds']):
            return
        if file_hash(self.config['replay_schedule']) != self.config['replay_schedule_hash']:
            raise ValueError('P1 recorded schedule changed before completion')
        self.manifest.update(replay_mode='complete', replay_consumed_steps=[dict(
            round=row['round'], step=row['step'], content_hash=digest(row['exposure_schedule'])) for row in rows])
        atomic_json(self.directory/'manifest.json', self.manifest)

    def run(self, **kwargs):
        if (self.config.get('replay_schedule') and self.config.get('replay_mode') != 'streaming'
                and file_hash(self.config['replay_schedule']) != self.config['replay_schedule_hash']):
            raise ValueError('P1 recorded schedule changed before execution')
        result = super().run(**kwargs)
        self.finish_complete_replay()
        return result
