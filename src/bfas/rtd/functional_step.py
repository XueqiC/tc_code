"""The actual RTD one-step map, in ordered LoRA coordinates (equation 3).

No Adam state, clipping, weight decay, or implicit step-size recalibration.
All tensor operations preserve float64 for the CPU derivative oracle.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import torch

from ..behavior.deltas import canonical_hash, parameter_layout, tensor_state_hash

Parameters = Mapping[str, torch.Tensor]


def lora_parameters(model) -> dict[str, torch.Tensor]:
    layout = parameter_layout(model)
    if any("lora_" not in e["name"].lower() for e in layout):
        raise ValueError("RTD updates LoRA trainables only; freeze all other parameters")
    return {e["name"]: dict(model.named_parameters())[e["name"]] for e in layout}


def snapshot(parameters: Parameters):
    return {n: p.detach().clone().requires_grad_(True) for n, p in parameters.items()}


def policy_identity(parameters, *, base_checkpoint_hash, harness_hash):
    if not base_checkpoint_hash or not harness_hash:
        raise ValueError("base checkpoint and harness content hashes required")
    return canonical_hash(dict(base=base_checkpoint_hash, harness=harness_hash,
                               trainables=tensor_state_hash(parameters)))


def gradients(loss, parameters, *, create_graph=False):
    if loss.ndim != 0 or not torch.isfinite(loss):
        raise ValueError("finite scalar loss required")
    if not loss.requires_grad:
        return {n: torch.zeros_like(p) for n, p in parameters.items()}
    values = torch.autograd.grad(loss, tuple(parameters.values()), create_graph=create_graph,
                                 allow_unused=True)
    return {n: torch.zeros_like(p) if g is None else g
            for (n, p), g in zip(parameters.items(), values)}


def _matching(parameters, tensors):
    if tuple(parameters) != tuple(tensors):
        raise ValueError("ordered parameter layouts differ")
    if any(tensors[n].shape != p.shape or tensors[n].device != p.device or
           not torch.isfinite(tensors[n]).all() for n, p in parameters.items()):
        raise ValueError("parameter shape/device mismatch or nonfinite values")


@dataclass(frozen=True)
class FrozenStep:
    diagonal: Parameters
    eta: float
    round_id: str
    metadata: dict

    def __post_init__(self):
        # Explicit stop-gradient even when supplied by a differentiable pilot.
        eta = float(torch.as_tensor(self.eta).detach()) if torch.is_tensor(self.eta) else float(self.eta)
        if not math.isfinite(eta) or eta <= 0 or not self.round_id or not self.diagonal:
            raise ValueError("positive frozen eta and round identity required")
        diagonal = {n: p.detach().clone() for n, p in self.diagonal.items()}
        if any(not torch.isfinite(p).all() or (p <= 0).any() for p in diagonal.values()):
            raise ValueError("P must be finite and strictly positive")
        object.__setattr__(self, "eta", eta)
        object.__setattr__(self, "diagonal", diagonal)
        object.__setattr__(self, "metadata", dict(self.metadata))

    def update(self, parameters, gradient):
        _matching(parameters, self.diagonal)
        _matching(parameters, gradient)
        return {n: p - self.eta * self.diagonal[n].detach() * gradient[n]
                for n, p in parameters.items()}


def rms_diagonal(parameters, source_gradients, *, inner_parent_hashes, source_snapshot_id,
                 relative_damping=0.01):
    """Stream (parent_hash, complete-source-rollout gradient), never an N×P array.

    r_j=sqrt(mean_n g_nj²), raw P_j=1/(r_j+.01*mean_j r_j),
    P=raw P/mean_j raw P. Means weight coordinates, not parameter tensors.
    A completely zero source second moment uses identity, recorded explicitly.
    """
    if relative_damping != 0.01 or not source_snapshot_id:
        raise ValueError("v1 requires relative damping .01 and source identity")
    second = {n: torch.zeros_like(p, dtype=torch.float64 if p.dtype == torch.float64 else torch.float32)
              for n, p in parameters.items()}
    count, parents = 0, set()
    for parent, gradient in source_gradients:
        if parent not in inner_parent_hashes:
            raise ValueError("preconditioner source outside inner parents")
        _matching(parameters, gradient)
        for n in parameters:
            second[n].add_(gradient[n].detach().to(second[n]).square())
        count += 1
        parents.add(parent)
    if not count:
        raise ValueError("source rollout gradients required")
    rms = {n: (s / count).sqrt() for n, s in second.items()}
    size = sum(s.numel() for s in rms.values())
    mean_rms = sum(s.sum() for s in rms.values()) / size
    damping = (relative_damping * mean_rms).detach()
    raw = {n: torch.ones_like(s) if mean_rms == 0 else (s + damping).reciprocal()
           for n, s in rms.items()}
    mean_diagonal = (sum(s.sum() for s in raw.values()) / size).detach()
    diagonal = {n: (s / mean_diagonal).to(parameters[n]).detach() for n, s in raw.items()}
    return diagonal, dict(source_snapshot_id=source_snapshot_id, source_rollouts=count,
        inner_parents=sorted(parents), relative_damping=relative_damping, damping=float(damping),
        raw_mean_diagonal=float(mean_diagonal), mean_diagonal=1., zero_rms_identity=bool(mean_rms == 0))


def functional_step(loss, parameters, step: FrozenStep, *, exact_noop=False, create_graph=True):
    if exact_noop:
        return dict(parameters)  # no gradient evaluation, no weight decay
    return step.update(parameters, gradients(loss, parameters, create_graph=create_graph))


def commit_step(model, updated, *, expected_start_hash):
    """Commit once at the saved start; a virtual reference is never committed first."""
    parameters = lora_parameters(model)
    _matching(parameters, updated)
    if tensor_state_hash(parameters) != expected_start_hash:
        raise ValueError("student moved since the saved common start")
    with torch.no_grad():
        for n, p in parameters.items():
            p.copy_(updated[n].detach())


def kl_pilot(parameters, diagonal, loss_at_alpha, source_kl, *, candidates,
             owned_ids, evidence_ids, inner_parent_hashes, evidence_parents, source_parents,
             target=0.005):
    """Calibrate a fixed alpha=.5 update using legal purchased evidence only.

    ``source_kl(updated)`` sums conditional forward KL over each frozen source
    action including EOS, then averages source slots. It sees train-side samples
    only. Return the complete trial log for separate pilot-compute accounting.
    No reward/checker or feedback labels are inputs. Empty D must defer pilot.
    """
    if (not evidence_ids or not set(evidence_ids) <= set(owned_ids)
            or not evidence_parents or not source_parents
            or not set(evidence_parents) <= set(inner_parent_hashes)
            or not set(source_parents) <= set(inner_parent_hashes)):
        raise ValueError("pilot requires purchased evidence and sources in the inner fold")
    if target != 0.005 or not candidates:
        raise ValueError("v1 KL target .005 and predeclared candidates required")
    gradient = {n: g.detach() for n, g in gradients(loss_at_alpha(0.5), parameters).items()}
    trials = []
    for eta in candidates:
        trial = FrozenStep(diagonal, eta, "pilot", {})
        with torch.no_grad():
            kl = float(source_kl(trial.update(parameters, gradient)).detach())
        if not math.isfinite(kl) or kl < 0:
            raise ValueError("pilot KL must be finite and nonnegative")
        trials.append(dict(eta=trial.eta, kl=kl))
    chosen = min(trials, key=lambda t: (abs(t["kl"] - target), t["eta"]))
    return chosen["eta"], dict(target=target, alpha=0.5, trials=trials, selected=dict(chosen),
        evidence_ids=sorted(evidence_ids), source_parents=sorted(set(source_parents)),
        trial_updates=len(trials), gradient_evaluations=1, committed_updates=0)
