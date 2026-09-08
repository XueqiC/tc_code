"""Exposure-conserving insertion at one saved starting model (7)--(8)."""
from dataclasses import dataclass
from enum import Enum

import torch

from ..behavior.deltas import tensor_state_hash
from .functional_step import FrozenStep, _matching, gradients, snapshot
from .return_gradient import ReturnGradient


class LabelType(str, Enum):
    PENDING_NEW = "pending_new"
    REWEIGHT_EXISTING = "reweight_existing"


@dataclass(frozen=True)
class InsertionLabel:
    query_id: str
    label_type: LabelType
    value: float
    round_id: str
    start_hash: str
    reference_hash: str

    def __post_init__(self):
        if not isinstance(self.label_type, LabelType) or not self.query_id or not self.round_id:
            raise ValueError("explicit request-level insertion label type required")
        if not torch.isfinite(torch.tensor(self.value)):
            raise ValueError("finite insertion label required")


def insertion_derivative(g_q, g_D, g_J_ref, step: FrozenStep):
    _matching(g_D, g_q)
    _matching(g_D, g_J_ref)
    _matching(g_D, step.diagonal)
    return -step.eta * sum((g_J_ref[n].detach() * step.diagonal[n].detach() *
                           (g_q[n].detach() - g_D[n].detach())).sum() for n in g_D)


class InsertionReference:
    """Virtual old-data update plus actual mixture, both from the same theta.

    Loss callables receive the saved start, preventing a second update at theta_D+.
    An empty old pool explicitly supplies zero gD (identity reference). The caller
    must freeze eta after its first legal purchase before constructing this block.
    """
    def __init__(self, parameters, old_loss, step: FrozenStep, *, old_slots=8, exact_noop=False,
                 smoke=False, old_gradient=None):
        if old_slots != (2 if smoke else 8):
            raise ValueError("v1 reference requires the fixed eight-slot list")
        self.start = snapshot(parameters)
        self.start_hash = tensor_state_hash(self.start)
        self.step, self.exact_noop = step, exact_noop
        self.slots = old_slots
        self.g_D = ({n: torch.zeros_like(p) for n, p in self.start.items()} if exact_noop else
                    {n: g.detach() for n, g in (old_gradient if old_gradient is not None else
                     gradients(old_loss(self.start), self.start)).items()})
        _matching(self.start, self.g_D)
        self.updated = snapshot(step.update(self.start, self.g_D))
        self.reference_hash = tensor_state_hash(self.updated)
        self._labeled = set()

    def insert(self, query_id, new_loss, feedback: ReturnGradient, *, label_type,
               owned_before, pending_ids, new_slots=8, epsilon=0.25, new_gradient=None):
        if feedback.parameter_hash != self.reference_hash:
            raise ValueError("insertion needs gJ at the same-start reference update")
        if new_slots != self.slots or epsilon != 0.25:
            raise ValueError("v1 insertion uses two fixed eight-slot lists and epsilon=.25")
        if tensor_state_hash(self.start) != self.start_hash:
            raise ValueError("saved insertion start was modified")
        if label_type is LabelType.PENDING_NEW:
            if query_id in owned_before or query_id not in pending_ids:
                raise ValueError("pending_new must be newly purchased at this window")
        elif label_type is LabelType.REWEIGHT_EXISTING:
            if query_id not in owned_before or query_id in pending_ids:
                raise ValueError("reweight_existing requires previously trained owned evidence")
        else:
            raise ValueError("explicit label type required")
        if query_id in self._labeled:
            raise ValueError("one observation per request, not per event/slot")
        g_q = {n: g.detach() for n, g in (new_gradient if new_gradient is not None else
                gradients(new_loss(self.start), self.start)).items()}
        value = insertion_derivative(g_q, self.g_D, feedback.gradient, self.step)
        mixture = {n: (1 - epsilon) * self.g_D[n] + epsilon * g_q[n] for n in self.start}
        actual = self.step.update(self.start, mixture)
        self._labeled.add(query_id)
        label = InsertionLabel(query_id, label_type, float(value), self.step.round_id,
                               self.start_hash, self.reference_hash)
        accounting = dict(raw_old_slots=0 if self.exact_noop else self.slots, raw_new_slots=self.slots,
            weighted_old_slots=0 if self.exact_noop else .75*self.slots, weighted_new_slots=.25*self.slots,
            committed_updates=1, reference_updates=0 if self.exact_noop else 1)
        return actual, label, accounting

    def empty(self):
        """empty has L_empty=L_D, v=0, c=0; it still consumes the decision window."""
        return dict(self.updated), dict(query_id=None, value=0., cost=0, decision_windows=1,
            raw_old_slots=0 if self.exact_noop else self.slots, weighted_slots=0 if self.exact_noop else self.slots,
            committed_updates=0 if self.exact_noop else 1)
