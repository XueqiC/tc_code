"""Full convex QP with auxiliary absolute-value variables and KKT validation."""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import LinearConstraint, minimize, nnls

from .immutable import Array
from .objective import TeachingObjective


@dataclass(frozen=True)
class ExactSolution:
    problem_id: str
    coefficients: Array
    x: Array
    auxiliary: Array
    multipliers: Array
    value: float
    iterations: int
    stationarity: float
    feasibility: float
    complementarity: float
    numerical_scale: float
    status: str = 'local_convex_QP_KKT_verified'


def auxiliary_constraints(problem):
    n = len(problem.coordinates)
    I, Z = np.eye(n), np.zeros((n, n))
    # A y <= b, y=(x,t). t<=1 is redundant for the canonical t=|x|.
    A = np.vstack([np.c_[I, Z], np.c_[-I, Z], np.c_[I, -I],
                   np.c_[-I, -I], np.c_[Z, -I], np.c_[Z, I]])
    b = np.r_[problem.upper, -problem.lower, np.zeros(3*n), np.ones(n)]
    for group in problem.groups:
        if len(group) > 1:
            row = np.zeros(2*n)
            row[list(group)] = 1
            A = np.vstack([A, row])
            b = np.r_[b, 1-problem.a_ref.numpy()[list(group)].sum()]
    return A, b


def kkt_certificate(objective, y, A, b, scale, tolerance):
    slack = b-A @ y
    grad = objective.auxiliary_value_gradient(y)[1]/scale
    active = slack <= max(tolerance*5, 1e-10)
    dual = np.zeros(len(b))
    if active.any():
        dual[active] = nnls(A[active].T, -grad, maxiter=max(1000, 20*len(b)))[0]
    stationarity = float(np.max(np.abs(grad+A.T @ dual), initial=0))
    feasibility = float(np.max(-slack, initial=0))
    complementarity = float(np.max(np.abs(dual*slack), initial=0))
    return dual*scale, stationarity, feasibility, complementarity


def solve_exact(problem, *, equality=None, control_gram=None):
    """Optional homogeneous constraints on x; report value under the full F.

    Equality rows are encoded as paired inequalities so the same independent
    KKT certificate verifies unrestricted and restricted local problems.
    """
    if control_gram is not None:
        from .problem import checked_psd
        control_gram = checked_psd(control_gram, problem.context.psd_tolerance)[0].numpy()
        if control_gram.shape != problem.K.shape:
            raise ValueError('control Gram shape mismatch')
    objective = TeachingObjective(problem, control_gram)
    n = len(problem.coordinates)
    if n == 0:
        empty = Array.of([])
        return ExactSolution(problem.identity, empty, empty, empty, empty, objective.value([]), 0, 0., 0., 0., 1.)
    Q, linear = objective.quadratic()
    if not np.isfinite(Q).all() or not np.isfinite(linear).all():
        raise ValueError('QP coefficients overflowed; rescale the frozen problem explicitly')
    scale = max(float(np.linalg.norm(Q, 2)), float(np.max(np.abs(linear))),
                float(np.max(problem.epsilon.numpy())), 1e-12)
    if not np.isfinite(scale):
        raise ValueError('QP numerical scale overflowed')
    A, b = auxiliary_constraints(problem)
    if equality is not None:
        equality = np.asarray(equality, dtype=float)
        if equality.ndim != 2 or equality.shape[1] != n or not np.isfinite(equality).all():
            raise ValueError('finite equality rows on x required')
        E = np.c_[equality, np.zeros_like(equality)]
        A, b = np.vstack([A, E, -E]), np.r_[b, np.zeros(2*len(E))]
    def fun(y):
        value, grad = objective.auxiliary_value_gradient(y)
        return value/scale, grad/scale
    result = minimize(fun, np.zeros(2*n), jac=True, method='SLSQP',
        constraints=LinearConstraint(A, -np.inf, b), options=dict(
            ftol=max(1e-15, min(1e-12, problem.context.qp_tolerance**2)),
            maxiter=problem.context.qp_max_iterations))
    # Only remove floating-point constraint error; this is not a decision rule.
    a = np.clip(result.x[:n]+problem.a_ref.numpy(), 0, 1)
    for group in problem.groups:
        ids = list(group)
        if a[ids].sum() > 1:
            a[ids] /= a[ids].sum()
    x = a-problem.a_ref.numpy()
    y = np.r_[x, np.abs(x)]
    dual, stationarity, feasibility, complementarity = kkt_certificate(
        objective, y, A, b, scale, problem.context.qp_tolerance)
    if (not np.isfinite((stationarity, feasibility, complementarity)).all()
            or max(stationarity, feasibility, complementarity) > problem.context.qp_tolerance):
        raise RuntimeError(f'QP failed KKT tolerance: {stationarity=}, {feasibility=}, '
                           f'{complementarity=}; SLSQP: {result.message}')
    value = TeachingObjective(problem).value(a)
    if objective.value(a) < -problem.context.qp_tolerance*scale:
        raise RuntimeError('QP degraded its feasible reference')
    return ExactSolution(problem.identity, Array.of(a), Array.of(x), Array.of(np.abs(x)),
        Array.of(dual), value, int(result.nit), stationarity, feasibility, complementarity, scale)
