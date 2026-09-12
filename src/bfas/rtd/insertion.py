"""Exposure-conserving insertion at one saved starting model (7)--(8)."""
from dataclasses import dataclass, field
from enum import Enum
import math

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


@dataclass(frozen=True)
class BatchInsertionLabel(InsertionLabel):
    # value remains the unit-exposure derivative v_q; predicted position gain
    # is value/E and its variance is variance/E**2. Never scale twice.
    variance: float = 1.
    position_share: float = 1./40
    window_id: str = ''
    approximation: str = 'batch_common_reference'
    variance_method: str = 'stratified_rollout_bootstrap'
    position_value: float = field(init=False)
    position_variance: float = field(init=False)

    def __post_init__(self):
        super().__post_init__()
        if not math.isfinite(self.variance) or self.variance <= 0 or not 0 < self.position_share <= 1:
            raise ValueError('positive label noise and exposure share required')
        object.__setattr__(self, 'position_value', self.value*self.position_share)
        object.__setattr__(self, 'position_variance', self.variance*self.position_share**2)


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
                 smoke=False, old_gradient=None, exposure_slots=None):
        if exposure_slots is None and old_slots != (2 if smoke else 8):
            raise ValueError("v1 reference requires the fixed eight-slot list")
        if exposure_slots is not None:
            if type(exposure_slots) is not int or exposure_slots < 1 or old_slots != exposure_slots:
                raise ValueError('batch reference needs exactly E old slots')
            self.exposure_slots = exposure_slots
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

    def batch_labels(self, package_gradients, feedback, *, owned_before, pending_ids, window_id,
                     variances, label_type=LabelType.PENDING_NEW):
        """Label paid requests at a shared reference; never load a broker here."""
        if not hasattr(self, 'exposure_slots') or feedback.parameter_hash != self.reference_hash:
            raise ValueError('batch labels need the shared reference feedback')
        if tensor_state_hash(self.start) != self.start_hash:
            raise ValueError('saved insertion start was modified')
        labels = {}
        for q, gradient in package_gradients.items():
            if label_type is LabelType.PENDING_NEW:
                legal = q not in owned_before and q in pending_ids
            elif label_type is LabelType.REWEIGHT_EXISTING:
                legal = q in owned_before and q not in pending_ids
            else:
                legal = False
            if not legal or q in self._labeled:
                raise ValueError('one matching paid request-level label required')
            value = float(insertion_derivative(gradient, self.g_D, feedback.gradient, self.step))
            labels[q] = BatchInsertionLabel(q, label_type, value, self.step.round_id, self.start_hash,
                self.reference_hash, variance=variances[q], position_share=1./self.exposure_slots, window_id=window_id)
            self._labeled.add(q)
        return labels

    def insert_batch(self, package_gradients, feedback, *, owned_before, pending_ids, window_id,
                     variances, max_new_packages=20):
        if len(package_gradients) > min(max_new_packages, self.exposure_slots) or set(package_gradients) != set(pending_ids):
            raise ValueError('batch exceeds exposure/package capacity or pending set mismatch')
        labels = self.batch_labels(package_gradients, feedback, owned_before=owned_before,
            pending_ids=pending_ids, window_id=window_id, variances=variances)
        E, m = self.exposure_slots, len(labels)
        if not m:
            actual = dict(self.updated)  # bit-identical to the old-data reference
        else:
            mixture = {n: (1-m/E)*self.g_D[n] + sum(g[n] for g in package_gradients.values())/E for n in self.start}
            actual = self.step.update(self.start, mixture)
        accounting = dict(exposure_slots=E, new_packages=m, old_coefficient=1-m/E, position_share=1/E,
            raw_old_slots=0 if self.exact_noop else E, raw_new_slots=m,
            weighted_old_slots=0 if self.exact_noop else E-m, weighted_new_slots=m,
            unfilled_old_slots=E-m, committed_updates=0 if self.exact_noop and not m else 1,
            predicted_additive_gain=sum(label.value/E for label in labels.values()))
        return actual, labels, accounting
