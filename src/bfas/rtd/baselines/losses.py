"""Full-action B1/B2/B3 objectives and teacher-only B4 derivatives.

B2 reuses arms.pair_unit_margins (also imported by pair_unit.py), with ordinary
logistic loss, not the CC two-condition squared hinge. Reference theta0 never
refreshes. Counting follows pair_unit's train_input/response and reference-input
semantics: an input traversing forward+backward is one training-token exposure;
forward/backward counts are also written separately.
"""
import torch
import torch.nn.functional as F

from ...arms import pair_unit_margins
from ..functional_step import gradients
from ..return_gradient import gate_vjp
from ..transport import positive_mixture_loss


def pairwise_loss(chosen, rejected, base_chosen, base_rejected, *, beta=.1):
    return -F.logsigmoid(pair_unit_margins(chosen, base_chosen.detach(), rejected, base_rejected.detach(), beta=beta))


def teacher_weights(u, query_ids, query_mass):
    """p-weighted mean-one softplus on the ENTIRE legal inner pool, not a batch.

    query_ids are table indices only. No source scores, lengths, projections,
    semantics, labels, or teacher features enter the weight map. The derivative
    contains normalized teacher gradients, with no gT-gS transport term.
    """
    if len(query_ids) != len(u) or len(set(query_ids)) != len(query_ids) or not set(query_mass) <= set(query_ids):
        raise ValueError("query scalar layout mismatch")
    if not query_mass or any(p <= 0 for p in query_mass.values()):
        raise ValueError("positive frozen legal query mass required")
    mass = u.new_tensor([query_mass.get(q, 0.) for q in query_ids]).detach()
    v = F.softplus(u)
    denominator = (mass * v).sum()
    # Outside-inner coordinates have exactly zero influence on this denominator.
    return {q: v[i]/denominator for i, q in enumerate(query_ids) if q in query_mass}


def slot_loss(slot, backend, parameters, recipe, *, reference=None, weight=None):
    """Differentiable CPU oracle; production uses one action graph at a time."""
    if slot.teacher is None and recipe != "b3_mix":
        return next(iter(parameters.values())).new_zeros(())
    teacher = backend.score_behavior(slot.teacher, parameters) if slot.teacher else None
    if recipe in {"b1", "b3_sft"}:
        return -teacher
    if recipe == "b4":
        if weight is None:
            raise ValueError("B4 requires the normalized query weight")
        return -teacher * weight
    source = backend.score_source(slot.source, parameters)
    if recipe == "b2":
        if reference is None:
            raise ValueError("B2 requires a frozen theta0 reference")
        return pairwise_loss(teacher, source, *reference)
    if recipe == "b3_mix":
        return positive_mixture_loss(source.reshape(1), teacher.reshape(1) if teacher is not None else None,
                                     source.new_tensor(.5))
    raise ValueError("unknown recipe")


def supervised_gradient(slots, rhos, backend, parameters, recipe, *, references=None, weights=None, journal=None,
                        role="update", context=None):
    """Streaming exact first derivatives, including pairwise cross-side margin."""
    if not slots or len(slots) != len(rhos):
        raise ValueError("aligned, nonempty predetermined draws required")
    result = {n: torch.zeros_like(p) for n, p in parameters.items()}
    total_loss = 0.
    for index, (slot, rho) in enumerate(zip(slots, rhos)):
        context = context or {}
        def side(teacher):
            score = backend.score_behavior(slot.teacher, parameters) if teacher else backend.score_source(slot.source, parameters)
            value = score.detach()
            gradient = {n: g.detach() for n, g in gradients(score, parameters).items()}
            if journal:
                action = slot.teacher_tokens if teacher else slot.source.length
                journal.usage(role, input_tokens=slot.prompt_tokens+action, action_tokens=action,
                              side="teacher" if teacher else "source", **context)
            return value, gradient
        if slot.teacher is None and recipe != "b3_mix":
            continue
        chosen, gc = side(True) if slot.teacher else (None, None)
        if recipe in {"b2", "b3_mix"}:
            rejected, gr = side(False)
        if recipe == "b2":
            bp, bm = references[index]
            margin = pair_unit_margins(chosen, bp, rejected, bm, beta=.1)
            factor = -.1 * torch.sigmoid(-margin)
            loss = -F.logsigmoid(margin)
            g = {n: factor * (gc[n]-gr[n]) for n in parameters}
        elif recipe == "b3_mix":
            loss = positive_mixture_loss(rejected.reshape(1), chosen.reshape(1) if chosen is not None else None,
                                         rejected.new_tensor(.5))
            g = {n: -.5*(gc[n]+gr[n]) if gc is not None else -gr[n] for n in parameters}
        else:
            weight = weights[slot.query_id].detach() if recipe == "b4" else 1.
            loss = -chosen * weight
            g = {n: -gc[n] * weight for n in parameters}
        total_loss += float(loss) * rho / len(slots)
        for n in result:
            result[n].add_(g[n] * (rho / len(slots)))
    return result, total_loss


def teacher_weight_vjp(slots, rhos, backend, parameters, u, query_ids, query_mass, step, feedback,
                       *, journal=None, context=None):
    """Exact teacher-only streaming reduction via public general gate_vjp.

    The LM gradient is detached in a linear proxy in theta, retaining the exact
    cross derivative in u. This avoids second derivatives of the LM kernels
    and an N x LoRA-coordinate gradient matrix. No source score is evaluated.
    feedback must be sampled at the actual fixed-step theta+; the caller checks
    its parameter hash before this reduction, never uses streamed_gate_vjp.
    """
    total = torch.zeros_like(u)
    for slot, rho in zip(slots, rhos):
        if slot.teacher is None:
            continue
        teacher_g = {n: g.detach() for n, g in gradients(-backend.score_behavior(slot.teacher, parameters), parameters).items()}
        weight = teacher_weights(u, query_ids, query_mass)[slot.query_id]
        proxy = sum((p * teacher_g[n]).sum() for n, p in parameters.items()) * weight * (rho / len(slots))
        total.add_(gate_vjp(proxy, parameters, u, step, feedback.gradient))
        if journal:
            journal.usage("aux", input_tokens=slot.prompt_tokens+slot.teacher_tokens,
                          action_tokens=slot.teacher_tokens, operation="teacher_weight_vjp", **(context or {}))
    return total.detach()
