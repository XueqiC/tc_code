"""Config-selected P0 engine with injectable scoring/feedback; no model loader."""
from dataclasses import dataclass
import math

from .estimators import LossDefinition
from .execution import trial_distillation, commit_distillation
from .feedback import FeedbackCheckpoint, collect_task_feedback
from .interfaces import validate_and_report
from .problem import TeachingContext, build_teaching_problem, attach_feedback
from .arms import ARMS, solve_arm


@dataclass(frozen=True)
class UnifiedConfig:
    student: str
    m: int = 2
    a_ref: float = .5
    regularization: float = 1.
    qp_tolerance: float = 1e-8
    qp_max_iterations: int = 10000
    psd_tolerance: float = 1e-10
    estimator: str = 'cv'
    loss: LossDefinition = LossDefinition()

    @classmethod
    def from_config(cls, config):
        if config.get('method') != 'rtd_unified' or config.get('protocol_version') != 'unified-p0-1':
            raise ValueError('explicit method=rtd_unified and unified-p0-1 required; v1.1 is separate')
        u = config['unified']
        if (u['m'] != 2 or config.get('source_samples_per_state') != 2
                or config.get('source_temperature') != 1 or config.get('source_top_p') != 1
                or u['H'] != 'inverse_diagonal_preconditioner'
                or u['uncertainty_label'] != 'uncertainty regulariser'
                or u['epsilon'] != 'feedback_standard_errors'
                or config.get('optimizer') != 'fixed_preconditioned_single_step'
                or config.get('new_teacher_calls') is not False or config.get('new_teacher_tokens') != 0):
            raise ValueError('unsupported unified P0 sampling/metric/feedback/access contract')
        return cls(config['student'], m=u['m'], a_ref=u['a_ref'], regularization=u['lambda'],
            qp_tolerance=u['qp_tolerance'], qp_max_iterations=u['qp_max_iterations'],
            psd_tolerance=u['psd_tolerance'], estimator=u['estimator'],
            loss=LossDefinition(**u['loss_definition']))

    def __post_init__(self):
        if (not self.student or self.m != 2 or not 0 <= self.a_ref <= 1 or self.regularization < 0
                or self.qp_tolerance <= 0 or self.psd_tolerance <= 0 or self.qp_max_iterations < 1
                or self.estimator not in {'cv', 'raw'} or not all(math.isfinite(v) for v in
                    (self.a_ref, self.regularization, self.qp_tolerance, self.psd_tolerance))):
            raise ValueError('invalid unified configuration')


class UnifiedEngine:
    def __init__(self, config):
        self.config = UnifiedConfig.from_config(config)
        self.arm = config.get('arm', 'D3')
        if self.arm not in ARMS or ARMS[self.arm].trainer != 'unified':
            raise ValueError('D0/D1 use the P1 benchmark runner and their existing trainers')

    def build_problem(self, *, parameters, source_snapshot, source_policy_hash, preconditioner,
                      owned_evidence, frozen_exposure, **context_options):
        c = self.config
        context = TeachingContext(parameters, source_snapshot, source_policy_hash, preconditioner,
            loss=c.loss, a_ref_default=c.a_ref, regularization=c.regularization,
            qp_tolerance=c.qp_tolerance, qp_max_iterations=c.qp_max_iterations,
            psd_tolerance=c.psd_tolerance, estimator=c.estimator, **context_options)
        return build_teaching_problem(context, owned_evidence, frozen_exposure)

    def run_step(self, problem, *, backend, rollout, feedback_split, feedback_version,
                 optimizer=None, generators=(), state=None):
        c = self.config
        if (problem.context.loss != c.loss or problem.context.estimator != c.estimator
                or problem.context.regularization != c.regularization
                or problem.context.qp_tolerance != c.qp_tolerance):
            raise ValueError('problem does not match selected unified config')
        _, feedback = trial_distillation(problem, problem.a_ref, model=backend.model,
            optimizer=optimizer, generators=generators, state=state,
            evaluate=lambda parameters: collect_task_feedback(
                FeedbackCheckpoint(parameters, backend, rollout, feedback_version), feedback_split))
        problem = attach_feedback(problem, feedback)
        solution, coefficients, permutation = solve_arm(problem, self.arm)
        commit = commit_distillation(problem, coefficients, model=backend.model)
        run = dict(problem=problem, solution=solution, commit=commit, permutation=permutation)
        return run | dict(report=validate_and_report(run))
