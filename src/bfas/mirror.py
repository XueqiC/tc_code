"""Capability-constrained mirror distillation (CRCD Block III, spec §5).

This module holds the *math* of the mirror target, the CMD loss, the two
regularisers and the primal-dual bookkeeping.  It knows nothing about model
loading; ``tools/crcd_mirror_train.py`` wires it to the student.

Notation follows docs/2026-09-04-crcd-next-tasks-from-user.md §2:

* event ``i`` has decision state ``s_i`` and a candidate set
  ``Y_i = {y_i^S, y_i^T, y_i^(1), ...}`` (owner tags ``"S"``, ``"T"``, ``"X0"`` ...);
* ``score0(y) = log pi_0(y | s_i) / |y|`` is the length-normalised frozen-base
  sequence score (cached once per pool);
* ``Q_i(y)`` is the stored branch outcome of ``y`` (``_event_u_plus`` for the
  teacher continuation, ``_event_u_minus`` for the student one);
* ``zbar_ki = |z_ki|`` are the non-negative atom loadings, ``Lambda_i`` is the
  normalised capability pressure and ``q_i`` the mirror target
  ``q_i(y) ∝ pi_0(y|s_i) exp(Lambda_i Q_i(y) / eta)`` over ``Y_i``.

Definitions that the spec leaves open (stated here so the trainer, the tests
and the report agree):

``J_k(pi_theta)``
    Atom-weighted *best-utility mass*: for event ``i`` let ``y_i*`` be the
    candidate with the highest ``Q_i`` (ties -> teacher first) and
    ``p_theta(y | s_i; Y_i)`` the student's length-normalised-score softmax
    restricted to ``Y_i``.  Then ``J_k = sum_i w_ki p_theta(y_i* | s_i; Y_i)``
    with ``w_ki = zbar_ki / sum_j zbar_kj`` over the events in the estimation
    window.  ``J_k ∈ [0, 1]``; 1 means the student always prefers the
    best-known continuation on atom-``k`` events.  ``j_mode="expected_utility"``
    swaps the indicator for ``Q_i`` itself (``sum_y p_theta(y) Q_i(y)``).

``rho_k``
    Required target level of ``J_k`` in ``[0, 1]``.  It is an input; the
    trainer can set it as ``J_k(pi_0) + offset_k`` so that atoms start out
    satisfied or violated by construction (used by the smoke).

``xi_k`` (slack)
    Minimising ``kappa d_k xi_k - lambda_k xi_k`` over ``xi_k >= 0`` gives
    ``xi_k = 0`` while ``lambda_k < kappa d_k`` and an unbounded slack beyond
    it, so the dual variable is clamped to ``[0, kappa d_k]`` and the slack is
    read off as ``xi_k = [rho_k - eps_k - J_k]_+`` only when the clamp is
    active.

``sketch``
    A fixed count-sketch (seed 0, 256 buckets): every LoRA-B coordinate ``j`` is
    hashed to bucket ``h(j)`` with sign ``s(j)``; ``sketch(v)_b = sum_{h(j)=b}
    s(j) v_j``.  It is norm-preserving in expectation and needs no
    ``dim x |theta|`` matrix.  The penalty is
    ``||(I - U U^+) sketch(F^{1/2} (theta_B - theta_B,0))||^2`` (whiten, then
    sketch, then project off the atom span; ``U = 0`` -> full displacement).
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F


TEACHER = "T"
STUDENT = "S"


# --------------------------------------------------------------------------
# candidate sets
# --------------------------------------------------------------------------


@dataclass
class Candidate:
    owner: str
    text: str
    utility: float
    logp0: float | None = None  # summed log pi_0(y | s)
    length: int | None = None  # response tokens (incl. EOS) used for the sum

    @property
    def score0(self) -> float:
        if self.logp0 is None or not self.length:
            raise ValueError(f"candidate {self.owner!r} has no cached base score")
        return self.logp0 / self.length

    def key(self) -> str:
        digest = hashlib.sha1(self.text.encode("utf-8")).hexdigest()[:16]
        return f"{self.owner}:{digest}"


@dataclass
class Event:
    event_id: str
    prompt: str
    candidates: list[Candidate]
    meta: dict[str, Any] = field(default_factory=dict)

    def owners(self) -> list[str]:
        return [c.owner for c in self.candidates]

    def utilities(self) -> np.ndarray:
        return np.asarray([c.utility for c in self.candidates], dtype=np.float64)

    def scores0(self) -> np.ndarray:
        return np.asarray([c.score0 for c in self.candidates], dtype=np.float64)

    def index_of(self, owner: str) -> int:
        for j, c in enumerate(self.candidates):
            if c.owner == owner:
                return j
        raise KeyError(owner)

    def best_index(self) -> int:
        """Highest-utility candidate; ties resolve to the teacher, then order."""
        u = self.utilities()
        best = float(u.max())
        tied = [j for j, v in enumerate(u) if v >= best - 1e-12]
        for j in tied:
            if self.candidates[j].owner == TEACHER:
                return j
        return tied[0]


def event_id_for_row(row: dict[str, Any]) -> str:
    if row.get("_traj"):
        return str(row["_traj"])
    return f"{row['task_id']}#t{row.get('turn_index', 0)}"


def candidates_from_row(
    row: dict[str, Any], utility_mode: str = "auto"
) -> Event:
    """Build ``Y_i`` from a pool row (prompt/response/_rejected/_event_u_*).

    ``utility_mode``:
      * ``"stored"``   – ``Q^T = _event_u_plus``, ``Q^S = _event_u_minus``;
      * ``"gt_teacher"`` – ``Q^T = 1.0`` (teacher continuation is ground truth),
        ``Q^S = _event_u_minus``;
      * ``"auto"``     – ``gt_teacher`` for BFCL single-turn oracle rows
        (``teacher == "oracle_gt"`` and ``turn_index == 0``), else ``stored``.
    Optional extra continuations come from ``row["_extra_candidates"]`` as
    ``[{"response": str, "utility": float}, ...]``.
    """
    if "_rejected" not in row or not row["_rejected"]:
        raise ValueError(f"row {event_id_for_row(row)} has no _rejected continuation")
    mode = utility_mode
    if mode == "auto":
        # single-turn rows whose teacher continuation is verified-correct by construction:
        # benchmark GT (accounting-inconsistent, ablation only), teacher-authored GT of a
        # generated task, or a verified paid teacher demo (2026-09-04 relabelling)
        is_gt = int(row.get("turn_index", 0)) == 0 and (
            row.get("teacher") in ("oracle_gt", "teacher_authored_gt", "teacher_authored_abstain")
            or float(row.get("_event_u_plus", 0.0)) >= 1.0)
        mode = "gt_teacher" if is_gt else "stored"
    if mode == "gt_teacher":
        q_teacher = 1.0
    elif mode == "stored":
        q_teacher = float(row["_event_u_plus"])
    else:
        raise ValueError(f"unknown utility_mode {utility_mode!r}")
    q_student = float(row["_event_u_minus"])
    cands = [
        Candidate(TEACHER, row["response"], q_teacher),
        Candidate(STUDENT, row["_rejected"], q_student),
    ]
    for n, extra in enumerate(row.get("_extra_candidates") or []):
        cands.append(Candidate(f"X{n}", extra["response"], float(extra["utility"])))
    return Event(
        event_id_for_row(row),
        row["prompt"],
        cands,
        meta={"utility_mode": mode, "task_id": row.get("task_id"),
              "turn_index": row.get("turn_index")},
    )


# --------------------------------------------------------------------------
# capability pressure and mirror target
# --------------------------------------------------------------------------


def normalized_loadings(Z: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """``W[i, k] = |z_ki| / (sum_j |z_kj| + eps)`` (shape n x K)."""
    zbar = np.abs(np.asarray(Z, dtype=np.float64))
    if zbar.ndim != 2:
        raise ValueError("Z must be n x K")
    return zbar / (zbar.sum(axis=0, keepdims=True) + eps)


def capability_pressure(W: np.ndarray, lam: np.ndarray) -> np.ndarray:
    """``Lambda_i = sum_k lambda_k W[i, k]`` (shape n)."""
    lam = np.asarray(lam, dtype=np.float64)
    if W.shape[1] != lam.shape[0]:
        raise ValueError(f"W has {W.shape[1]} atoms but lambda has {lam.shape[0]}")
    return W @ lam


def mirror_target(
    score0: np.ndarray, utility: np.ndarray, pressure: float, eta: float
) -> np.ndarray:
    """``q(y) ∝ exp(score0(y) + Lambda * Q(y) / eta)`` over the candidate set."""
    if eta <= 0:
        raise ValueError("eta must be positive")
    logits = np.asarray(score0, dtype=np.float64) + pressure * np.asarray(
        utility, dtype=np.float64
    ) / eta
    logits = logits - logits.max()
    q = np.exp(logits)
    return q / q.sum()


def restricted_policy(score: np.ndarray) -> np.ndarray:
    """Softmax of length-normalised scores over ``Y_i`` (``Lambda = 0`` target)."""
    return mirror_target(score, np.zeros_like(score, dtype=np.float64), 0.0, 1.0)


def log_odds(
    score0: np.ndarray, utility: np.ndarray, pressure: float, eta: float, a: int, b: int
) -> float:
    """Closed form ``log q(y_a)/q(y_b) = score0_a - score0_b + Lambda (Q_a - Q_b)/eta``."""
    return float(score0[a] - score0[b] + pressure * (utility[a] - utility[b]) / eta)


def entropy(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64)
    nz = p[p > 0]
    return float(-(nz * np.log(nz)).sum())


def target_stats(q: np.ndarray, event: Event) -> dict[str, float]:
    owners = event.owners()
    stats = {"entropy": entropy(q), "best_mass": float(q[event.best_index()])}
    stats["teacher_mass"] = float(q[owners.index(TEACHER)]) if TEACHER in owners else float("nan")
    stats["student_mass"] = float(q[owners.index(STUDENT)]) if STUDENT in owners else float("nan")
    return stats


# --------------------------------------------------------------------------
# losses
# --------------------------------------------------------------------------


def cmd_loss(q: np.ndarray | torch.Tensor, scores_theta: torch.Tensor) -> torch.Tensor:
    """``KL(q || softmax(scores_theta))`` over the candidate set.

    ``scores_theta`` are the student's length-normalised sequence scores of the
    candidates (with grad).  Up to the constant ``-H(q)`` this is the
    cross-entropy against the target; its gradient wrt the scores is
    ``p_theta - q``.
    """
    q_t = torch.as_tensor(np.asarray(q, dtype=np.float64), device=scores_theta.device,
                          dtype=torch.float64)
    logp = F.log_softmax(scores_theta.double(), dim=0)
    nz = q_t > 0
    return (q_t[nz] * (torch.log(q_t[nz]) - logp[nz])).sum()


def event_policy_mass(scores_theta: torch.Tensor, index: int) -> float:
    """``p_theta(y_index | s; Y)`` under the restricted softmax (detached)."""
    return float(F.softmax(scores_theta.detach().double(), dim=0)[index].item())


def token_kl_theta_base(
    logits_theta: torch.Tensor, logits_base: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Mean token-level ``KL(pi_theta || pi_0)`` on the masked positions.

    Both logits are ``(1, L, V)``; ``mask`` is ``(1, L)`` bool over positions
    whose *next* token is a response token.  Full-vocabulary KL, computed in
    float32.
    """
    lt = F.log_softmax(logits_theta.float(), dim=-1)
    lb = F.log_softmax(logits_base.float(), dim=-1)
    kl_pos = (lt.exp() * (lt - lb)).sum(dim=-1)  # (1, L)
    m = mask.to(kl_pos.dtype)
    return (kl_pos * m).sum() / m.sum().clamp_min(1.0)


class DisplacementPenalty:
    """``||(I - U U^+) sketch(F^{1/2} (theta - theta_0))||^2`` on LoRA-B params.

    ``params`` – list of ``(name, tensor)`` (the live trainable LoRA-B tensors);
    ``fisher`` – ``float`` (scalar diagonal Fisher, 1.0 for the smoke) or
    ``{name: tensor}`` matching each param's shape;
    ``U`` – ``(dim, K)`` array spanning the sketched capability subspace or
    ``None``/zeros (penalty on the full displacement).
    """

    def __init__(
        self,
        params: Sequence[tuple[str, torch.Tensor]],
        fisher: float | dict[str, torch.Tensor] = 1.0,
        U: np.ndarray | None = None,
        dim: int = 256,
        seed: int = 0,
    ) -> None:
        self.dim = dim
        self.params = sorted(params, key=lambda kv: kv[0])
        if not self.params:
            raise ValueError("no parameters to penalise")
        device = self.params[0][1].device
        gen = torch.Generator(device="cpu").manual_seed(seed)
        self.buckets: list[torch.Tensor] = []
        self.signs: list[torch.Tensor] = []
        self.theta0: list[torch.Tensor] = []
        self.fsqrt: list[torch.Tensor | float] = []
        for name, p in self.params:
            n = p.numel()
            self.buckets.append(
                torch.randint(0, dim, (n,), generator=gen).to(device=device, dtype=torch.long))
            self.signs.append(
                (torch.randint(0, 2, (n,), generator=gen).float() * 2 - 1).to(device))
            self.theta0.append(p.detach().clone())
            if isinstance(fisher, dict):
                f = fisher[name]
                if tuple(f.shape) != tuple(p.shape):
                    raise ValueError(f"fisher[{name}] shape {tuple(f.shape)} != {tuple(p.shape)}")
                self.fsqrt.append(f.to(device=device, dtype=torch.float32).sqrt().reshape(-1))
            else:
                self.fsqrt.append(math.sqrt(float(fisher)))
        self.device = device
        self.set_subspace(U)

    def set_subspace(self, U: np.ndarray | None) -> None:
        """Install ``(I - U U^+)``; ``U=None`` or all-zero -> identity."""
        dim = self.dim
        if U is None:
            proj = np.eye(dim)
        else:
            U = np.asarray(U, dtype=np.float64)
            if U.shape[0] != dim:
                raise ValueError(f"U must be {dim} x K, got {U.shape}")
            proj = np.eye(dim) - U @ np.linalg.pinv(U)
        self.proj = torch.as_tensor(proj, device=self.device, dtype=torch.float32)

    def sketch(self) -> torch.Tensor:
        out = torch.zeros(self.dim, device=self.proj.device, dtype=torch.float32)
        for (name, p), b, s, t0, fs in zip(
            self.params, self.buckets, self.signs, self.theta0, self.fsqrt
        ):
            delta = (p - t0).reshape(-1).float() * fs
            out = out.index_add(0, b, delta * s)
        return out

    def __call__(self) -> torch.Tensor:
        return (self.proj @ self.sketch()).pow(2).sum()


# --------------------------------------------------------------------------
# primal-dual bookkeeping
# --------------------------------------------------------------------------


def event_J_value(
    p_theta: np.ndarray, event: Event, j_mode: str = "best_mass"
) -> float:
    """Per-event contribution to ``J``: best-utility mass or expected utility."""
    p = np.asarray(p_theta, dtype=np.float64)
    if j_mode == "best_mass":
        return float(p[event.best_index()])
    if j_mode == "expected_utility":
        return float((p * event.utilities()).sum())
    raise ValueError(f"unknown j_mode {j_mode!r}")


class JEstimator:
    """Running atom-weighted estimate of ``J_k`` over a window of events."""

    def __init__(self, K: int) -> None:
        self.num = np.zeros(K)
        self.den = np.zeros(K)
        self.count = 0

    def add(self, zbar_row: np.ndarray, value: float) -> None:
        z = np.abs(np.asarray(zbar_row, dtype=np.float64))
        self.num += z * value
        self.den += z
        self.count += 1

    def value(self, fallback: np.ndarray | None = None) -> np.ndarray:
        J = np.full_like(self.num, np.nan)
        ok = self.den > 0
        J[ok] = self.num[ok] / self.den[ok]
        if fallback is not None:
            J[~ok] = np.asarray(fallback, dtype=np.float64)[~ok]
        return J

    def reset(self) -> None:
        self.num[:] = 0.0
        self.den[:] = 0.0
        self.count = 0


def atom_J_from_events(
    Z: np.ndarray, values: Sequence[float]
) -> np.ndarray:
    """``J_k = sum_i |z_ki| v_i / sum_i |z_ki|`` over a full set of events."""
    est = JEstimator(np.asarray(Z).shape[1])
    for z, v in zip(np.asarray(Z), values):
        est.add(z, float(v))
    return est.value()


@dataclass
class DualState:
    lam: np.ndarray
    rho: np.ndarray
    eps: np.ndarray
    kappa: float = 1.0
    d: np.ndarray | None = None
    eta_dual: float = 1.0

    def __post_init__(self) -> None:
        K = len(self.lam)
        self.lam = np.asarray(self.lam, dtype=np.float64).copy()
        self.rho = np.broadcast_to(np.asarray(self.rho, dtype=np.float64), (K,)).copy()
        self.eps = np.broadcast_to(np.asarray(self.eps, dtype=np.float64), (K,)).copy()
        self.d = (np.ones(K) if self.d is None
                  else np.broadcast_to(np.asarray(self.d, dtype=np.float64), (K,)).copy())

    @property
    def cap(self) -> np.ndarray:
        return self.kappa * self.d

    def residual(self, J: np.ndarray) -> np.ndarray:
        """``r_k = [rho_k - J_k]_+``."""
        return np.maximum(self.rho - np.asarray(J, dtype=np.float64), 0.0)

    def slack(self, J: np.ndarray) -> np.ndarray:
        """``xi_k`` – non-zero only where the dual clamp ``lambda_k = kappa d_k`` binds."""
        gap = np.maximum(self.rho - self.eps - np.asarray(J, dtype=np.float64), 0.0)
        return np.where(self.lam >= self.cap - 1e-12, gap, 0.0)

    def update(self, J: np.ndarray) -> dict[str, np.ndarray]:
        """``lambda_k <- clip(lambda_k + eta_dual (rho_k - eps_k - J_k), 0, kappa d_k)``."""
        J = np.asarray(J, dtype=np.float64)
        if np.any(np.isnan(J)):
            raise ValueError("J has NaN entries; give the estimator a fallback")
        before = self.lam.copy()
        gap = self.rho - self.eps - J
        self.lam = np.clip(self.lam + self.eta_dual * gap, 0.0, self.cap)
        return {
            "lambda_before": before,
            "lambda_after": self.lam.copy(),
            "J": J,
            "gap": gap,
            "r": self.residual(J),
            "xi": self.slack(J),
            "satisfied": J >= self.rho - self.eps,
        }


def as_list(x: Any) -> Any:
    if isinstance(x, np.ndarray):
        return [as_list(v) for v in x.tolist()] if x.ndim else float(x)
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, dict):
        return {k: as_list(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [as_list(v) for v in x]
    return x
