"""Feasible capability recipes and local dose control (NumPy/SciPy only).

Conventions match ``response`` and ``lowrank``: B_hat is (probe, source),
X is (source, H-orthonormal coordinate), and u = X.T @ a is a *code*, not
a parameter vector. Pass ``ResponsePredictor(model).predict_from_codes(X)``;
decode u with ``whiten.Basis.decode`` outside this module. Missing responses
must be estimated or excluded explicitly; NaN is never replaced by zero.
Optional nonnegative E tightens retention and the missing-behaviour gap using
B_hat - E. An empirical held-out residual quantile is not a confidence bound:
conservative feasibility is conditional on E, not a guarantee of true behaviour.

All optimizers return dictionaries with explicit success/status diagnostics.
The QP implements the signed, weighted hinge damage in the theory derivation,
not a quadratic damage surrogate. Its ball constraint makes it a convex QCQP;
auxiliary epigraph variables allow a small smooth SLSQP solve.
"""
from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
from scipy.optimize import linprog, lsq_linear, minimize


def _array(value, name, ndim):
    value = np.asarray(value, dtype=float)
    if value.ndim != ndim or not np.isfinite(value).all():
        raise ValueError(f"{name} must be a finite {ndim}D array")
    return value


def _nonnegative(value, name):
    value = float(value)
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


def _controls(tol, maxiter=None):
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError("tol must be positive and finite")
    if maxiter is not None and (isinstance(maxiter, bool) or
                               not isinstance(maxiter, (int, np.integer)) or maxiter < 1):
        raise ValueError("maxiter must be a positive integer")


def _indices(value, size, name):
    value = np.asarray(sorted(value) if isinstance(value, (set, frozenset)) else value)
    if value.ndim != 1 or (value.size and value.dtype.kind not in "iu"):
        raise ValueError(f"{name} must be a 1D set of integer probe indices")
    value = value.astype(int)
    if np.any(value < 0) or np.any(value >= size) or len(np.unique(value)) != len(value):
        raise ValueError(f"{name} contains duplicate or out-of-range indices")
    return value


def _error_matrix(E, shape):
    if E is None:
        return np.zeros(shape)
    E = np.asarray(E, dtype=float)
    if not np.isfinite(E).all() or np.any(E < 0):
        raise ValueError("E must be finite and nonnegative")
    if E.ndim == 1 and E.shape == (shape[0],):
        E = E[:, None]
    if E.ndim != 0 and E.shape not in {(shape[0], 1), shape}:
        raise ValueError("E must be a scalar, per-probe vector/column, or have B_hat's shape")
    return np.broadcast_to(E, shape).copy()


@dataclass
class FeasibleSet:
    """A = {a >= 0, sum(a) = 1, (B_hat[R] - E[R]) @ a >= 0}.

    r follows the supplied order of M (sets are sorted). K may be singular.
    If omitted, K=X@X.T when X is available, otherwise K=I explicitly assumes
    a Euclidean *coefficient* metric. Supply the true Gram for an H-dose bound.
    A provided K takes precedence: truncated X can omit orthogonal dose.
    R and M must be disjoint. Inputs are copied and never modified by solves.
    E accepts a scalar, a per-probe vector/column, or a (probe, source) matrix;
    omitted E is zero. B_R/B_M remain nominal; conservative_B_R/B_M subtract E.
    B_hat must come from one response operator applied to the source updates;
    E describes recipe-dependent uncertainty, not a different nominal predictor.
    """

    B_hat: np.ndarray
    R: np.ndarray
    M: np.ndarray
    r: np.ndarray
    K: np.ndarray | None = None
    X: np.ndarray | None = None
    E: np.ndarray | None = None

    def __post_init__(self):
        self.B_hat = _array(self.B_hat, "B_hat", 2).copy()
        p, n = self.B_hat.shape
        self.E = _error_matrix(self.E, (p, n))
        self.R, self.M = _indices(self.R, p, "R"), _indices(self.M, p, "M")
        if np.intersect1d(self.R, self.M).size:
            raise ValueError("R and M must be disjoint")
        self.r = _array(self.r, "r", 1).copy()
        if self.r.shape != (len(self.M),) or np.any(self.r < 0):
            raise ValueError("r must be nonnegative and have one entry per missing probe")
        if self.X is not None:
            self.X = _array(self.X, "X", 2).copy()
            if self.X.shape[0] != n:
                raise ValueError("X must have one row per source")
        self.metric_source = "provided" if self.K is not None else ("codes" if self.X is not None else "identity")
        gram = self.K if self.K is not None else (self.X @ self.X.T if self.X is not None else np.eye(n))
        gram = _array(gram, "K", 2)
        if gram.shape != (n, n) or not np.allclose(gram, gram.T, rtol=1e-10, atol=1e-12):
            raise ValueError("K must be a symmetric (n_sources, n_sources) Gram matrix")
        self.K = (gram + gram.T) / 2
        values, vectors = np.linalg.eigh(self.K)
        cutoff = 1e-10 * max(1.0, np.max(np.abs(values), initial=0.0))
        if np.min(values, initial=0.0) < -cutoff:
            raise ValueError("K must be positive semidefinite")
        if np.any(values < 0):  # Remove roundoff, never add an unrequested ridge.
            self.K = (vectors * np.maximum(values, 0)) @ vectors.T

    @property
    def n_sources(self):
        return self.B_hat.shape[1]

    @property
    def B_R(self):
        return self.B_hat[self.R]

    @property
    def B_M(self):
        return self.B_hat[self.M]

    @property
    def conservative_B_R(self):
        return self.B_R - self.E[self.R]

    @property
    def conservative_B_M(self):
        return self.B_M - self.E[self.M]

    def contains(self, a, *, tol=1e-8):
        _controls(tol)
        a = np.asarray(a, dtype=float)
        return bool(a.shape == (self.n_sources,) and np.isfinite(a).all()
                    and np.all(a >= -tol) and abs(a.sum() - 1) <= tol
                    and np.all(self.conservative_B_R @ a >= -tol))

    def feasibility_report(self, *, tol=1e-8):
        """Compare normalized LP max gains with/without E (not geometric volume)."""
        _controls(tol)
        return _feasibility_report(self, tol=tol)


def _problem(B_hat, R, M, r, K, X, E=None):
    if isinstance(B_hat, FeasibleSet):
        if any(v is not None for v in (R, M, K, X)):
            raise ValueError("pass either a FeasibleSet or raw inputs, not both")
        if r is None and E is None:
            return B_hat
        result = FeasibleSet(B_hat.B_hat, B_hat.R, B_hat.M, B_hat.r if r is None else r,
                             B_hat.K, B_hat.X, B_hat.E if E is None else E)
        result.metric_source = B_hat.metric_source
        return result
    if any(v is None for v in (R, M, r)):
        raise ValueError("raw B_hat requires R, M and r")
    return FeasibleSet(B_hat, R, M, r, K, X, E)


def _lp(c, B_R, *, tol, A_eq=None, b_eq=None):
    return linprog(c, A_ub=-B_R if len(B_R) else None,
                   b_ub=np.zeros(len(B_R)) if len(B_R) else None,
                   A_eq=np.ones((1, len(c))) if A_eq is None else A_eq,
                   b_eq=[1.] if b_eq is None else b_eq, bounds=(0, None),
                   method="highs-ds", options={
                       "primal_feasibility_tolerance": max(1e-10, min(tol, 1e-7)),
                       "dual_feasibility_tolerance": max(1e-10, min(tol, 1e-7))})


def _feasibility_report(fs, *, tol, conservative_solved=None):
    """Three LPs separate retention-set shrinkage from the gain error penalty.

    Raw max gains are None for an empty simplex or solver failure. The volume
    proxy allows the no-update action: max(0, LP gain), zero for an empty set.
    Gain loss depends on r and is not a measure of geometric volume. Unknown
    solver outcomes stay None, rather than being reported as infeasibility.
    """
    nominal_c, conservative_c = fs.B_M.T @ fs.r, fs.conservative_B_M.T @ fs.r

    def summarize(c, retention, solved=None):
        if not fs.n_sources:
            return dict(status="infeasible", feasible=False, max_gain=None, value=0.)
        solved = _lp(-c, retention, tol=tol) if solved is None else solved
        if solved.success:
            gain = float(c @ solved.x)
            return dict(status="optimal", feasible=True, max_gain=gain, value=max(0., gain))
        infeasible = solved.status == 2
        return dict(status="infeasible" if infeasible else "solver_failed",
                    feasible=False if infeasible else None, max_gain=None,
                    value=0. if infeasible else None)

    conservative = summarize(conservative_c, fs.conservative_B_R, conservative_solved)
    same_gain = np.array_equal(nominal_c, conservative_c)
    same_retention = not np.any(fs.E[fs.R])
    retention_only = conservative if same_gain else summarize(nominal_c, fs.conservative_B_R)
    nominal = retention_only if same_retention else summarize(nominal_c, fs.B_R)

    def loss(after):
        if nominal["value"] is None or after["value"] is None:
            return None
        return max(0., nominal["value"] - after["value"])

    gain_loss = loss(conservative)
    return dict(nominal=nominal, retention_only=retention_only, conservative=conservative,
                infeasible=conservative["feasible"] is False,
                became_infeasible=nominal["feasible"] is True and conservative["feasible"] is False,
                gain_loss=gain_loss, retention_gain_loss=loss(retention_only),
                relative_gain_loss=(gain_loss / nominal["value"] if gain_loss is not None
                                    and nominal["value"] else None),
                proxy="LP max gain with zero-update option; retention_only holds the nominal objective fixed")


def _support(a, B_R, tol):
    support = np.flatnonzero(a > tol)
    active = np.flatnonzero(np.abs(B_R @ a) <= tol)
    rows = B_R[np.ix_(active, support)]
    rank = int(np.linalg.matrix_rank(rows, tol=tol)) if rows.size else 0
    return support, active, rank


def find_recipe(B_hat, R=None, M=None, r=None, *, K=None, X=None, E=None,
                tol=1e-8, ridge=1e-8):
    """Maximize r.T B_M a on A with HiGHS dual simplex.

    Accept a FeasibleSet (optionally override r or E), or raw construction inputs.
    Both objective and retention use B_hat-E; b remains the nominal B_hat @ a,
    and conservative_b is its lower estimate. feasibility_report compares LP
    max gains with/without E and explicitly flags an empty normalized simplex.
    Active retention indices refer to the original probe rows. The sparsity
    check uses the rank of active rows restricted to the recipe support.
    A nonvertex numerical result triggers a warning and a small ridge solve
    on the *exact LP optimal face*, then a HiGHS vertex crossover. A ridge
    alone can make recipes dense. Neither tie-breaking stage may sacrifice
    the original objective. Persistent violations are reported as failure.
    """
    _controls(tol)
    ridge = _nonnegative(ridge, "ridge")
    fs = _problem(B_hat, R, M, r, K, X, E)
    c = fs.conservative_B_M.T @ fs.r
    if not fs.n_sources:
        return dict(success=False, status="infeasible", message="empty source simplex", a=None,
                    feasible=False, infeasible=True, feasibility_report=_feasibility_report(fs, tol=tol))
    solved = _lp(-c, fs.conservative_B_R, tol=tol)
    report = _feasibility_report(fs, tol=tol, conservative_solved=solved)
    if not solved.success:
        return dict(success=False, status="infeasible" if solved.status == 2 else "solver_failed",
                    message=solved.message, a=None, solver_status=solved.status,
                    feasible=False if solved.status == 2 else None, infeasible=solved.status == 2,
                    feasibility_report=report)
    a = solved.x.copy()
    support, active, rank = _support(a, fs.conservative_B_R, tol)
    degenerate = len(support) > rank + 1
    tie_used = False
    if degenerate:
        warnings.warn("LP sparsity bound violated; ridge tie-break and vertex crossover required",
                      RuntimeWarning, stacklevel=2)
        eq = np.ones((1, fs.n_sources))
        rhs = np.array([1.])
        # Constant objectives add no independent equality to the optimal face.
        if np.ptp(c) > tol:
            eq = np.vstack([eq, c])
            rhs = np.r_[rhs, c @ a]
        constraints = [{"type": "eq", "fun": lambda z: eq @ z - rhs, "jac": lambda z: eq}]
        if len(fs.R):
            constraints.append({"type": "ineq", "fun": lambda z: fs.conservative_B_R @ z,
                                "jac": lambda z: fs.conservative_B_R})
        regularized = minimize(lambda z: .5 * ridge * (z @ z), a,
                               jac=lambda z: ridge * z, bounds=[(0, None)] * fs.n_sources,
                               constraints=constraints, method="SLSQP",
                               options={"ftol": max(1e-14, ridge * tol), "maxiter": 200})
        center = regularized.x if regularized.success else a
        tie = center + np.arange(fs.n_sources) / max(1, fs.n_sources)
        crossed = _lp(tie, fs.conservative_B_R, tol=tol, A_eq=eq, b_eq=rhs)
        if crossed.success and abs(c @ (crossed.x - a)) <= tol * (1 + abs(c @ a)):
            a, tie_used = crossed.x.copy(), True
            support, active, rank = _support(a, fs.conservative_B_R, tol)
    violation = len(support) > rank + 1
    success = fs.contains(a, tol=10 * tol) and not violation
    # Max-LP dual certificate: c + B_R.T y <= upper_bound, y >= 0.
    dual_retention = -np.asarray(solved.ineqlin.marginals)
    upper = -float(solved.eqlin.marginals[0])
    return dict(success=success, status="optimal" if success else "numerical_failure",
                feasible=fs.contains(a, tol=10 * tol), infeasible=False, feasibility_report=report,
                message=solved.message, a=a, u=a.copy() if fs.X is None else fs.X.T @ a,
                b=fs.B_hat @ a, conservative_b=(fs.B_hat - fs.E) @ a,
                objective=float(c @ a), nominal_objective=float(fs.r @ (fs.B_M @ a)), support=support,
                support_size=len(support), active_retention=fs.R[active], active_rank=rank,
                sparsity_bound=rank + 1, sparsity_violation=violation,
                degeneracy_detected=degenerate, tie_break_used=tie_used,
                dual_retention=dual_retention, upper_bound=upper,
                dual_slack=upper - c - fs.conservative_B_R.T @ dual_retention)


def iterate_atoms(B_hat, R=None, M=None, r=None, *, k=5, eta=1., K=None, X=None, E=None,
                  tol=1e-8, ridge=1e-8):
    """Greedily extract at most k (a, u_code, b) triples; retain all LP reports.

    Update r <- maximum(r - eta B_M a, 0). Stop on a closed gap, no positive
    direction, or an LP failure. Triples may repeat when dose eta is small.
    This is a fixed local model, not a claim of finite-dose task improvement.
    """
    _controls(tol)
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k < 0:
        raise ValueError("k must be a nonnegative integer")
    eta = _nonnegative(eta, "eta")
    if eta == 0:
        raise ValueError("eta must be positive")
    fs = _problem(B_hat, R, M, r, K, X, E)
    residual = fs.r.copy()
    atoms, reports, history = [], [], [residual.copy()]
    status, success = "max_atoms", True
    for _ in range(k):
        if np.max(residual, initial=0.) <= tol:
            status = "gap_closed"
            break
        result = find_recipe(fs, r=residual, tol=tol, ridge=ridge)
        reports.append(result)
        if not result["success"]:
            status, success = result["status"], False
            break
        if result["objective"] <= tol:
            status = "no_positive_direction"
            break
        atoms.append((result["a"], result["u"], result["b"]))
        residual = np.maximum(residual - eta * result["conservative_b"][fs.M], 0)
        history.append(residual.copy())
    if np.max(residual, initial=0.) <= tol:
        status = "gap_closed"
    return dict(success=success, status=status, atoms=atoms, recipes=reports,
                support_sizes=[len(np.flatnonzero(a > tol)) for a, _, _ in atoms],
                residual_gap=residual, gap_history=np.asarray(history),
                sparsity_violations=[i for i, result in enumerate(reports)
                                     if result.get("degeneracy_detected", False)])


def positive_direction_exists(B_M, B_R, r, *, E=None, tol=1e-8):
    """Return a normalized positive-direction witness or a dual certificate.

    ``exists`` is True/False for certified results, None for numerical failure.
    A feasible optimum <= tol certifies no gain *above tol*, not exact strict
    positivity in floating point. ``feasible`` distinguishes an empty simplex
    from a nonpositive optimum. Infeasibility has a Farkas witness y >= 0 with
    B_R.T y <= -1; otherwise y and upper_bound bound every feasible gain.
    E uses rows in stacked [B_M; B_R] order (or scalar/per-probe broadcasting).
    With E, witnesses and certificates apply to B_hat-E, not the unknown true B.
    """
    B_M, B_R = _array(B_M, "B_M", 2), _array(B_R, "B_R", 2)
    if B_M.shape[1] != B_R.shape[1]:
        raise ValueError("B_M and B_R must have the same source axis")
    m, q = len(B_M), len(B_R)
    errors = _error_matrix(E, (m + q, B_M.shape[1]))
    result = find_recipe(np.vstack([B_M, B_R]), np.arange(m, m + q), np.arange(m), r, E=errors, tol=tol)
    retention = B_R - errors[m:]
    if result["success"]:
        certified = bool(np.all(result["dual_slack"] >= -10 * tol))
        exists = bool(result["objective"] > tol)
        if not exists and (not certified or result["upper_bound"] > tol):
            exists = None
        return dict(success=exists is not None, exists=exists, feasible=True,
                    infeasible=False, feasibility_report=result["feasibility_report"],
                    status="positive" if exists else ("nonpositive" if exists is False else "numerical_failure"),
                    w=result["a"], gain=result["objective"], upper_bound=result["upper_bound"],
                    certificate={"type": "primal" if exists else "dual",
                                 "retention_multipliers": result["dual_retention"],
                                 "upper_bound": result["upper_bound"], "slack": result["dual_slack"]})
    infeasible = result["status"] == "infeasible"
    certificate = None
    if infeasible and B_M.shape[1] and q:
        farkas = linprog(np.ones(q), A_ub=retention.T, b_ub=-np.ones(B_M.shape[1]),
                         bounds=(0, None), method="highs")
        if farkas.success:
            certificate = {"type": "farkas", "retention_multipliers": farkas.x,
                           "source_bounds": retention.T @ farkas.x}
    elif infeasible and not B_M.shape[1]:
        certificate = {"type": "empty_simplex"}
    return dict(success=infeasible, exists=False if infeasible else None,
                infeasible=infeasible, feasibility_report=result["feasibility_report"],
                feasible=False if infeasible else None, status=result["status"],
                w=None, gain=None, certificate=certificate, message=result["message"])


def _kkt(gradient, slack, jacobian, tol, eq_jac=None, eq_residual=None):
    """Recover nonnegative multipliers for c(z)>=0; equality multipliers free."""
    eq_jac = np.empty((0, len(gradient))) if eq_jac is None else eq_jac
    if not all(np.isfinite(v).all() for v in (gradient, slack, jacobian, eq_jac)) or (
            eq_residual is not None and not np.isfinite(eq_residual).all()):
        return dict(verified=False, primal_violation=np.inf, stationarity_residual=np.inf,
                    complementarity_residual=np.inf, multipliers=np.zeros(len(slack)),
                    equality_multipliers=np.zeros(len(eq_jac)), active_constraints=np.array([], dtype=int))
    active = np.flatnonzero(slack <= max(1e-7, 10 * tol))
    matrix = np.vstack([jacobian[active], eq_jac]).T
    multipliers = np.zeros(len(slack))
    equality = np.zeros(len(eq_jac))
    if matrix.shape[1]:
        lower = np.r_[np.zeros(len(active)), np.full(len(eq_jac), -np.inf)]
        fit = lsq_linear(matrix, gradient, bounds=(lower, np.full(matrix.shape[1], np.inf)),
                         method="bvls", tol=min(tol, 1e-10), max_iter=1000)
        multipliers[active] = fit.x[:len(active)]
        equality = fit.x[len(active):]
    stationarity = gradient - jacobian.T @ multipliers - eq_jac.T @ equality
    primal = max(0., float(-np.min(slack, initial=0.)),
                 float(np.max(np.abs(eq_residual), initial=0.)) if eq_residual is not None else 0.)
    dual = float(np.max(np.abs(stationarity), initial=0.))
    comp = float(np.max(np.abs(multipliers * slack), initial=0.))
    scale = 1 + float(np.max(np.abs(gradient), initial=0.))
    verified = primal <= 10 * tol and dual <= max(1e-6, 20 * tol) * scale and comp <= 10 * tol * scale
    return dict(verified=verified, primal_violation=primal, stationarity_residual=dual,
                complementarity_residual=comp, multipliers=multipliers,
                equality_multipliers=equality, active_constraints=active)


def distill_recipe(B_hat, R=None, M=None, r=None, *, K=None, X=None, E=None,
                   lam=0., eps=1., tol=1e-8, maxiter=1000):
    """Constrained squared-hinge NNLS with analytic-gradient SLSQP.

    Minimize .5||[r-B_M w]_+||² + .5 lam w.T K w subject to w>=0,
    B_R w>=0 and w.T K w<=eps². Check primal feasibility, KKT stationarity
    and complementary slackness independently of the solver termination flag.
    Convexity makes verified KKT conditions sufficient (up to tolerance).
    For eps=0, impose the Gram range as linear equalities to avoid a vanishing
    quadratic constraint gradient. Singular Grams preserve null-space updates.

    If the initial solve fails validation, retry from a scaled LP direction;
    otherwise return an explicit ``solver_failed`` zero-update fallback with
    success=False. For valid inputs this homogeneous problem is always feasible
    at zero, even if A is empty: solver failure is never called infeasibility.
    Negative/nonfinite eps is invalid input, rather than a fabricated solution.
    predicted_gain is B_M w (per probe); weighted_gain is r.T B_M w.
    With E, the objective/gap and retention constraints use B_hat-E. Nominal
    predictions keep their meaning; conservative_gain/retention_change report
    the lower estimates. simplex_feasible and feasibility_report flag when no
    nonzero feasible direction exists, even though w=0 remains feasible.
    """
    _controls(tol, maxiter)
    lam, eps = _nonnegative(lam, "lam"), _nonnegative(eps, "eps")
    fs = _problem(B_hat, R, M, r, K, X, E)
    n, q = fs.n_sources, len(fs.R)
    eq = np.empty((0, n))
    if eps == 0 and n:
        values, vectors = np.linalg.eigh(fs.K)
        eq = vectors[:, values > np.finfo(float).eps * max(1., values[-1]) * n].T

    def objective(w):
        gap = np.maximum(fs.r - fs.conservative_B_M @ w, 0)
        return float(.5 * (gap @ gap) + .5 * lam * (w @ fs.K @ w))

    def gradient(w):
        return -fs.conservative_B_M.T @ np.maximum(fs.r - fs.conservative_B_M @ w, 0) + lam * (fs.K @ w)

    def slack(w):
        return np.r_[w, fs.conservative_B_R @ w, [eps * eps - w @ fs.K @ w] if eps > 0 else []]

    def jacobian(w):
        rows = [np.eye(n), fs.conservative_B_R]
        if eps > 0:
            rows.append((-2 * fs.K @ w)[None, :])
        return np.vstack(rows)

    constraints = [{"type": "ineq", "fun": slack, "jac": jacobian}]
    if len(eq):
        constraints.append({"type": "eq", "fun": lambda w: eq @ w, "jac": lambda w: eq})
    attempts = []
    zero = np.zeros(n)
    start = zero.copy()
    if not n or len(eq) == n:
        w, success, status = zero, True, "optimal"
        diagnostics = _kkt(gradient(w), slack(w), jacobian(w), tol, eq, eq @ w)
    else:
        success = False
        for attempt in range(2):
            solved = minimize(objective, start, jac=gradient, constraints=constraints,
                              method="SLSQP", options={"ftol": min(1e-12, tol * .01), "maxiter": maxiter})
            candidate = solved.x
            diagnostics = _kkt(gradient(candidate), slack(candidate), jacobian(candidate), tol, eq, eq @ candidate)
            # Squared-norm feasibility alone is too permissive near zero dose.
            candidate_dose = np.sqrt(max(0., candidate @ fs.K @ candidate))
            diagnostics["dose_violation"] = max(0., candidate_dose - eps)
            diagnostics["verified"] &= diagnostics["dose_violation"] <= 10 * tol
            attempts.append(dict(solver_success=bool(solved.success), message=solved.message,
                                 iterations=solved.nit, **diagnostics))
            if diagnostics["verified"] and objective(candidate) <= objective(zero) + tol:
                w, success = candidate, True
                break
            if attempt == 0:
                recipe = find_recipe(fs, tol=tol)
                if recipe["success"]:
                    direction = recipe["a"]
                    norm = np.sqrt(max(0., direction @ fs.K @ direction))
                    scale = min(1., eps / norm) if norm else 1.
                    start = .5 * scale * direction
                else:
                    start = zero.copy()
        if not success:
            w = zero
            diagnostics = _kkt(gradient(w), slack(w), jacobian(w), tol, eq, eq @ w)
        status = "optimal" if success else "solver_failed"
    gain, retention = fs.B_M @ w, fs.B_R @ w
    conservative_gain, conservative_retention = fs.conservative_B_M @ w, fs.conservative_B_R @ w
    report = _feasibility_report(fs, tol=tol)
    dose = float(np.sqrt(max(0., w @ fs.K @ w)))
    return dict(success=success, status=status, feasible=True, fallback=not success,
                simplex_feasible=report["conservative"]["feasible"],
                simplex_status=report["conservative"]["status"], feasibility_report=report,
                message="KKT conditions verified" if success else "optimization failed; feasible zero-update fallback",
                w=w, u=w.copy() if fs.X is None else fs.X.T @ w, b=fs.B_hat @ w,
                predicted_gain=gain, weighted_gain=float(fs.r @ gain),
                predicted_retention_change=retention, conservative_gain=conservative_gain,
                conservative_weighted_gain=float(fs.r @ conservative_gain),
                conservative_retention_change=conservative_retention,
                conservative_b=(fs.B_hat - fs.E) @ w,
                residual_gap=np.maximum(fs.r - conservative_gain, 0),
                effective_dose=dose, objective=objective(w), initial_objective=objective(zero),
                active_set=np.flatnonzero(w > tol), active_retention=fs.R[np.abs(conservative_retention) <= tol],
                trust_region_active=abs(dose - eps) <= 10 * tol,
                metric_source=fs.metric_source, kkt=diagnostics, attempts=attempts)


def _qp_inputs(x0, B_plus, B_minus, w_weights):
    x0 = _array(x0, "x0", 1)
    B_plus, B_minus = _array(B_plus, "B_plus", 2), _array(B_minus, "B_minus", 2)
    if B_plus.shape[1] != len(x0) or B_minus.shape[1] != len(x0):
        raise ValueError("response matrices must have one column per coordinate")
    # Separate weights are needed when missing and retained probe counts differ.
    if isinstance(w_weights, dict):
        plus, minus = w_weights["plus"], w_weights["minus"]
    elif isinstance(w_weights, tuple) and len(w_weights) == 2:
        plus, minus = w_weights
    else:
        plus = minus = w_weights
    def weights(value, count, name):
        value = np.asarray(value, dtype=float)
        if value.ndim == 0:
            value = np.full(count, value)
        if value.shape != (count,) or not np.isfinite(value).all() or np.any(value < 0):
            raise ValueError(f"{name} must be nonnegative, with one entry per probe")
        return value
    return x0, B_plus, B_minus, weights(plus, len(B_plus), "plus weights"), weights(minus, len(B_minus), "minus weights")


def trust_region_qp(x0, B_plus, B_minus, w_weights, alpha, tau, rho, perp_norm,
                    *, tol=1e-8, maxiter=1000):
    """Solve min ||x-x0||²-alpha J+(x), J-(x)<=tau, ||x||²+perp_norm²<=rho².

    J+=p.T B_plus x and J-=q.T maximum(-B_minus x, 0); B_minus contains
    signed *retention changes*. w_weights is a shared scalar/vector or a
    (plus, minus) tuple / dict with those keys. Coordinates are unrestricted
    in sign. The perpendicular update is held fixed and consumes the radius.

    Use a tiny convex epigraph QCQP with analytic objective/constraint
    Jacobians and SLSQP. Return independently recovered nonnegative KKT
    multipliers: damage budget, hinge inequalities, epigraph nonnegativity,
    and squared ball. Stationarity is
    2(x-x0)-alpha B_plus.T p-B_minus.T hinge+2 ball*x=0.
    Thus the unconstrained shift is alpha/2 times the gain gradient.

    A zero remaining radius is a singleton. The squared-ball gradient then
    vanishes and ordinary finite KKT multipliers may not exist; report the
    singleton's normal explicitly, without inventing a finite ball multiplier.
    """
    _controls(tol, maxiter)
    x0, B_plus, B_minus, p, q = _qp_inputs(x0, B_plus, B_minus, w_weights)
    alpha, tau = _nonnegative(alpha, "alpha"), _nonnegative(tau, "tau")
    rho, perp_norm = _nonnegative(rho, "rho"), _nonnegative(perp_norm, "perp_norm")
    if perp_norm > rho:
        return dict(success=False, status="infeasible", feasible=False, x=None, multipliers=None,
                    message="fixed perpendicular norm exceeds trust-region radius")
    radius2 = (rho - perp_norm) * (rho + perp_norm)
    d, m = len(x0), len(q)
    gain_gradient = B_plus.T @ p

    def summarize(x, success, status, multipliers, kkt, message, **extra):
        return dict(success=success, status=status, x=x, multipliers=multipliers, kkt=kkt,
                    gain=float(gain_gradient @ x), damage=float(q @ np.maximum(-B_minus @ x, 0)),
                    effective_dose=float(np.sqrt(x @ x + perp_norm ** 2)),
                    objective=float((x-x0) @ (x-x0) - alpha * (gain_gradient @ x)),
                    feasible=bool(np.hypot(np.linalg.norm(x), perp_norm) <= rho + 10 * tol
                                  and q @ np.maximum(-B_minus @ x, 0) <= tau + 10 * tol),
                    message=message, **extra)

    if radius2 == 0 or d == 0:
        normal = 2 * x0 + alpha * gain_gradient
        return summarize(np.zeros(d), True, "optimal", {"damage": 0., "hinge": np.zeros(m),
                         "epigraph": np.zeros(m), "ball": None if np.any(normal) else 0.,
                         "singleton_normal": normal},
                         {"verified": True, "constraint_qualification": "singleton; use its normal cone"},
                         "zero remaining radius fixes x=0")

    def objective(z):
        delta = z[:d] - x0
        return float(delta @ delta - alpha * (gain_gradient @ z[:d]))

    def gradient(z):
        return np.r_[2 * (z[:d] - x0) - alpha * gain_gradient, np.zeros(m)]

    def slack(z):
        x, t = z[:d], z[d:]
        return np.r_[B_minus @ x + t, t, tau - q @ t, radius2 - x @ x]

    def jacobian(z):
        return np.vstack([np.c_[B_minus, np.eye(m)], np.c_[np.zeros((m, d)), np.eye(m)],
                          np.r_[np.zeros(d), -q], np.r_[-2 * z[:d], np.zeros(m)]])

    scale = min(1., np.sqrt(radius2) / np.linalg.norm(x0)) if np.linalg.norm(x0) else 1.
    damage0 = float(q @ np.maximum(-B_minus @ x0, 0))
    if damage0:
        scale = min(scale, tau / damage0)
    start = np.r_[scale * x0, np.maximum(-B_minus @ (scale * x0), 0)]
    constraints = [{"type": "ineq", "fun": slack, "jac": jacobian}]
    attempts = []
    for _ in range(2):
        solved = minimize(objective, start, jac=gradient, constraints=constraints, method="SLSQP",
                          options={"ftol": min(1e-12, tol * .01), "maxiter": maxiter})
        z = solved.x
        kkt = _kkt(gradient(z), slack(z), jacobian(z), tol)
        kkt["dose_violation"] = max(0., np.hypot(np.linalg.norm(z[:d]), perp_norm) - rho)
        kkt["verified"] &= kkt["dose_violation"] <= 10 * tol
        attempts.append(dict(solver_success=bool(solved.success), message=solved.message,
                             iterations=solved.nit, **kkt))
        if kkt["verified"]:
            mu = kkt["multipliers"]
            return summarize(z[:d], True, "optimal", {"hinge": mu[:m], "epigraph": mu[m:2*m],
                             "damage": float(mu[-2]), "ball": float(mu[-1])}, kkt,
                             "KKT conditions verified", attempts=attempts)
        start = np.zeros(d + m)
    kkt = _kkt(gradient(start), slack(start), jacobian(start), tol)
    return summarize(np.zeros(d), False, "solver_failed", None, kkt,
                     "optimization failed; feasible zero-coordinate fallback", fallback=True, attempts=attempts)


def compare_global_kl(x0, B_plus, B_minus, w_weights, alpha, tau, rho, perp_norm,
                      *, tol=1e-8, maxiter=1000):
    """Compare shaping with s*delta0 at matched *realised linear-model damage*.

    Global scaling scales both x0 and its perpendicular component. Choose the
    largest 0<=s<=1 consistent with rho and the shaped solution's damage. If
    that damage is unattainable on the global ray, matched_damage is False;
    do not label it a Pareto comparison. No true model rollout is performed.

    The plan's unconditional gain-dominance / strict-iff claim does not follow
    from a proximity-regularized objective for arbitrary B. Report the actual
    gains and dominates flag. Separated gain/damage modes with alpha=0 and an
    inactive ball give the intended strict improvement (or no-damage tie).
    """
    shaped = trust_region_qp(x0, B_plus, B_minus, w_weights, alpha, tau, rho, perp_norm,
                             tol=tol, maxiter=maxiter)
    if not shaped["success"]:
        return dict(success=False, shaped=shaped, matched_damage=False, global_x=None)
    x0, B_plus, B_minus, p, q = _qp_inputs(x0, B_plus, B_minus, w_weights)
    dose0 = float(np.hypot(np.linalg.norm(x0), perp_norm))
    scale = min(1., rho / dose0) if dose0 else 1.
    damage0 = float(q @ np.maximum(-B_minus @ x0, 0))
    if damage0 > 0:
        scale = min(scale, max(0., shaped["damage"]) / damage0)
    global_x = scale * x0
    global_damage = float(q @ np.maximum(-B_minus @ global_x, 0))
    global_gain = float(p @ (B_plus @ global_x))
    matched = abs(global_damage - shaped["damage"]) <= 10 * tol
    improvement = shaped["gain"] - global_gain
    return dict(success=True, shaped=shaped, global_x=global_x, global_scale=scale,
                global_perp_norm=scale * perp_norm, global_effective_dose=scale * dose0,
                global_damage=global_damage, global_gain=global_gain,
                matched_damage=matched, gain_improvement=improvement,
                dominates=bool(matched and improvement >= -10 * tol),
                strictly_better=bool(matched and improvement > 10 * tol))


def finite_step_bound(g_u, L, eps):
    """Largest eta>=0 with eta*g_u - .5*L*eta² >= -eps, elementwise.

    g_u is the signed directional derivative. Use a unit-norm direction, or
    pass L_j*||u||² as L; all norms and smoothness constants must use the same
    metric. For several probes take min(returned bounds). This is a local
    smoothness guarantee only within the region where L applies.
    L=0 allows infinity for a nonnegative derivative. Broadcasting is allowed.
    """
    g, curvature, budget = np.broadcast_arrays(np.asarray(g_u, dtype=float),
                                              np.asarray(L, dtype=float), np.asarray(eps, dtype=float))
    if not all(np.isfinite(v).all() for v in (g, curvature, budget)) or np.any(curvature < 0) or np.any(budget < 0):
        raise ValueError("g_u must be finite; L and eps must be finite and nonnegative")
    bound = np.full(g.shape, np.inf)
    curved = curvature > 0
    root = np.hypot(g[curved], np.sqrt(2.) * np.sqrt(curvature[curved]) * np.sqrt(budget[curved]))
    signed = g[curved]
    values = np.empty_like(root)
    positive = signed >= 0
    values[positive] = (signed[positive] + root[positive]) / curvature[curved][positive]
    # Rationalized root avoids cancellation for large negative derivatives.
    values[~positive] = 2 * budget[curved][~positive] / (root[~positive] - signed[~positive])
    bound[curved] = values
    linear_loss = ~curved & (g < 0)
    bound[linear_loss] = budget[linear_loss] / -g[linear_loss]
    return float(bound) if bound.ndim == 0 else bound
