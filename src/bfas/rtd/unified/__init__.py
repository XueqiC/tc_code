"""RTD unified replacement P0; importing never starts an experiment."""
from .engine import UnifiedConfig, UnifiedEngine
from .estimators import LossDefinition, alpha_d, empirical_target, estimate, hard_loss, soft_retention_loss
from .execution import commit_distillation, trial_distillation, restored_state
from .feedback import collect_task_feedback, FeedbackCheckpoint, FeedbackSplit, FeedbackEstimate
from .immutable import Array, Parameters
from .interfaces import (build_prequery_features, purchase_or_reveal, predict_control,
    build_acquisition_labels, score_query_batch, validate_and_report,
    PrequeryContext, PrequeryFeatures, ControlObservables, RevealedEvidence)
from .objective import TeachingObjective
from .problem import (TeachingContext, TeachingProblem, Exposure, Evidence,
                      build_teaching_problem, attach_feedback, checked_psd)
from .scoring import score_sources, score_source_samples, score_teacher
from .solver import solve_exact
