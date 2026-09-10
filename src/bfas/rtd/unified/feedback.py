"""Adapter to v1.1's action-only score-function estimator and LOO uncertainty."""
from dataclasses import dataclass

import numpy as np

from ..alpha_d import FeedbackStatistic
from ..persistence import digest
from ..return_gradient import collect_feedback, reinforce_gradient
from .immutable import Array, Parameters


@dataclass(frozen=True)
class FeedbackCheckpoint:
    parameters: Parameters
    backend: object
    rollout: object  # (task_id, parameters) -> full TaskRollout; independent calls
    version: str


@dataclass(frozen=True)
class FeedbackSplit:
    tasks: tuple[tuple[str, str], ...]
    allowed_parents: frozenset[str]
    inner_parents: frozenset[str]
    confirmation_parents: frozenset[str] = frozenset()
    task_weights: tuple[float, ...] = ()
    rollouts_per_task: int = 4
    baseline: str = 'leave_one_out_same_task'
    error_mode: str = 'loo'


@dataclass(frozen=True)
class FeedbackEstimate:
    gradient: Array
    scores: tuple[Array, ...]
    rewards: tuple[float, ...]
    task_ids: tuple[str, ...]
    task_weights: tuple[tuple[str, float], ...]
    parameter_hash: str
    version: str
    baseline: str
    error_mode: str
    action_tokens: int

    def __post_init__(self):
        object.__setattr__(self, 'rewards', tuple(float(r) for r in self.rewards))
        object.__setattr__(self, 'task_weights', tuple((t, float(w)) for t, w in self.task_weights))
        if (not isinstance(self.gradient, Array) or type(self.scores) is not tuple
                or any(not isinstance(s, Array) or s.shape != self.gradient.shape for s in self.scores)
                or type(self.rewards) is not tuple or type(self.task_ids) is not tuple
                or type(self.task_weights) is not tuple or any(type(w) is not tuple for w in self.task_weights)
                or len(self.scores) != len(self.rewards) or len(self.scores) != len(self.task_ids)
                or not all(type(t) is str and t for t in (*self.task_ids, self.parameter_hash, self.version))
                or self.baseline not in {'leave_one_out_same_task', 'independent_zero'}
                or self.error_mode not in {'loo', 'split_half'}
                or any(not np.isfinite(r) or not 0 <= r <= 1 for r in self.rewards)
                or any(not np.isfinite(w) or w < 0 for _, w in self.task_weights)
                or len(dict(self.task_weights)) != len(self.task_weights)
                or set(dict(self.task_weights)) != set(self.task_ids)
                or abs(sum(w for _, w in self.task_weights)-1) > 1e-10
                or any(self.task_ids.count(t) < 2 for t, _ in self.task_weights)):
            raise ValueError('immutable, aligned feedback statistics required')

    @property
    def identity(self):
        return digest(dict(gradient=self.gradient.hash, scores=[s.hash for s in self.scores],
            rewards=self.rewards, tasks=self.task_ids, weights=self.task_weights,
            checkpoint=self.parameter_hash, version=self.version, baseline=self.baseline,
            error_mode=self.error_mode, action_tokens=self.action_tokens))

    def project(self, layout, U):
        # Reuse the v1.1 jackknife, including re-fit of LOO baselines on deletion.
        directions = tuple(layout.tensors(values=v, requires_grad=False) for v in U.T)
        variance = np.zeros(U.shape[1])
        for task, weight in self.task_weights:
            ids = [i for i, t in enumerate(self.task_ids) if task == t]
            statistic = FeedbackStatistic(
                layout.tensors(values=self.gradient.numpy(), requires_grad=False),
                tuple(layout.tensors(values=self.scores[i].numpy(), requires_grad=False) for i in ids),
                tuple(self.rewards[i] for i in ids), tuple(task for _ in ids), self.parameter_hash,
                0, 'smoke_zero' if self.baseline == 'independent_zero' else self.baseline)
            if directions:
                _, se = statistic.project(directions, self.error_mode)
                variance += weight**2*se**2
        return self.gradient.numpy(), np.sqrt(variance)


def collect_task_feedback(checkpoint, feedback_split):
    """On-policy temperature=1 full-task return gradient; no optimizer step.

    Each task's independent rollouts get a same-task LOO baseline. Explicit
    task weights multiply task means; environment/tool tokens remain prompts,
    never score terms. `independent_zero` is a legal fixed baseline.
    """
    s = feedback_split
    if (s.allowed_parents & (s.inner_parents | s.confirmation_parents)
            or s.baseline not in {'leave_one_out_same_task', 'independent_zero'}
            or s.error_mode not in {'loo', 'split_half'} or not s.tasks):
        raise ValueError('disjoint training feedback, legal baseline and uncertainty mode required')
    weights = np.asarray(s.task_weights or (1/len(s.tasks),)*len(s.tasks), dtype=float)
    if weights.shape != (len(s.tasks),) or not np.isfinite(weights).all() or (weights < 0).any() or abs(weights.sum()-1) > 1e-10:
        raise ValueError('fixed task sampling weights must sum to one')
    backend, layout = checkpoint.backend, checkpoint.parameters
    device = next(iter(backend.model.parameters())).device
    parameters = layout.tensors(device=device)
    rollouts = collect_feedback(s.tasks, lambda task: checkpoint.rollout(task, parameters),
        feedback_parent_hashes=s.allowed_parents, rollouts_per_task=s.rollouts_per_task)
    if any(r.policy_id != backend.identity(parameters) for r in rollouts):
        raise ValueError('off-policy feedback is unsupported')
    total = np.zeros_like(layout.values.numpy())
    all_scores, rewards, task_ids, tokens = [], [], [], 0
    for (task, _), weight in zip(s.tasks, weights):
        rows = tuple(r for r in rollouts if r.task_id == task)
        scores = []
        result = reinforce_gradient(rows, backend, parameters, trajectory_scores=scores,
            baseline='smoke_zero' if s.baseline == 'independent_zero' else s.baseline)
        total += weight*layout.flatten(result.gradient).numpy()
        all_scores.extend(layout.flatten(score) for score in scores)
        rewards.extend(r.reward for r in rows)
        task_ids.extend(r.task_id for r in rows)
        tokens += result.metadata['action_tokens']
    return FeedbackEstimate(Array.of(total), tuple(all_scores), tuple(rewards), tuple(task_ids),
        tuple((task, float(w)) for (task, _), w in zip(s.tasks, weights)), layout.hash,
        checkpoint.version, s.baseline, s.error_mode, tokens)
