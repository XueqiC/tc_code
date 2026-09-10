"""Dense exact local problems, independent of benchmark/GPU resources."""
from dataclasses import replace

import numpy as np
import torch

from bfas.rtd.unified import (Array, Parameters, TeachingContext, LossDefinition, Exposure,
    Evidence, FeedbackEstimate, build_teaching_problem, attach_feedback)


def from_directions(U, *, h=None, epsilon=None, a_ref=.5, regularization=1., P=None):
    U = np.asarray(U, dtype=float)
    p, n = U.shape
    assert n % 2 == 0
    parameters = Parameters.of({'lora_weight': torch.zeros(p, dtype=torch.float64)})
    P = np.ones(p) if P is None else np.asarray(P)
    context = TeachingContext(parameters, parameters, 'source-policy', Array.of(P),
        loss=LossDefinition('total_token_nll'), eta=1., a_ref_default=a_ref,
        regularization=regularization, owned_query_ids=tuple(f'q{i}' for i in range(n//2)),
        inner_parent_hashes=('inner',))
    exposure, evidence = [], []
    for i in range(n//2):
        exposure.append(Exposure(f'slot{i}', f'state{i}', 'inner', (f'y{i}0', f'y{i}1'),
            (f'sample{i}0', f'sample{i}1'), (f'draw{i}0', f'draw{i}1'), 'source-policy',
            parameters.hash, context.loss.hash, Array.of(U[:, 2*i:2*i+2].T*n/P),
            Array.of(np.zeros(p)), 2/n))
        evidence.append(Evidence(f'q{i}', 'teacher-1', f'state{i}', 'inner', 'teacher text',
            f'teacher{i}', parameters.hash, context.loss.hash, Array.of(np.zeros(p))))
    problem = build_teaching_problem(context, evidence, exposure)
    if h is not None:
        problem = with_oracle_feedback(problem, h)
    if epsilon is not None:
        problem = replace(problem, epsilon=Array.of(epsilon))
    return problem


def with_oracle_feedback(problem, h):
    h = Array.of(h)
    # A known local linear functional for QP/identity tests; not a claimed
    # sampled policy gradient. Integration tests use actual scored rollouts.
    feedback = FeedbackEstimate(h, (h, h), (1., 1.), ('oracle', 'oracle'),
        (('oracle', 1.),), problem.reference_parameters.hash, 'analytic-toy-oracle',
        'independent_zero', 'loo', 0)
    return attach_feedback(problem, feedback)
