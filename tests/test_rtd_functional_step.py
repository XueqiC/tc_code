"""Float64 CPU checks of the actual one-step map and both derivative directions."""
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bfas.behavior.deltas import tensor_state_hash
from bfas.rtd.functional_step import (FrozenStep, commit_step, functional_step, gradients,
    kl_pilot, lora_parameters, rms_diagonal, snapshot)
from bfas.rtd.insertion import InsertionReference, LabelType, insertion_derivative
from bfas.rtd.return_gradient import GateController, ReturnGradient, actual_gate_vjp, gate_vjp
from bfas.rtd.transport import positive_mixture_loss

DT = torch.float64


def problem():
    theta = {"lora_weight": torch.tensor([.3, -.2, .1], dtype=DT, requires_grad=True)}
    phi = torch.tensor([.2, -.15], dtype=DT, requires_grad=True)
    chi = torch.tensor([[1., .4], [-.3, .7], [.2, -.6]], dtype=DT, requires_grad=True)
    step = FrozenStep({"lora_weight": torch.tensor([.5, 1., 1.5], dtype=DT, requires_grad=True)},
                      torch.tensor(.12, dtype=DT, requires_grad=True), "r1", {})
    def loss(params, gate=phi):
        lp = params["lora_weight"].log_softmax(0)
        return positive_mixture_loss(lp[[0, 1, 2]], lp[[1, 2, 1]], (chi.detach() @ gate).sigmoid())
    def reward(params):
        p = params["lora_weight"].softmax(0)
        return p[1] + .3 * p[0].square()
    return theta, phi, chi, step, loss, reward


def test_gate_vjp_matches_central_differences_and_free_gate_formula():
    theta, phi, chi, step, loss, reward = problem()
    actual = functional_step(loss(theta), theta, step)
    at_actual = snapshot(actual)
    gJ = gradients(reward(at_actual), at_actual, create_graph=True)
    vjp = gate_vjp(loss(theta), theta, phi, step, gJ)
    eps, fd = 1e-5, []
    for j in range(phi.numel()):
        perturb = torch.zeros_like(phi); perturb[j] = eps
        def value(gate):
            start = snapshot(theta)
            return reward(functional_step(loss(start, gate), start, step))
        fd.append((value(phi.detach() + perturb) - value(phi.detach() - perturb)) / (2 * eps))
    torch.testing.assert_close(vjp, torch.stack(fd), atol=2e-10, rtol=2e-8)
    assert not vjp.requires_grad and not step.diagonal['lora_weight'].requires_grad
    assert phi.grad is chi.grad is None
    a = torch.tensor([.3, .5, .7], dtype=DT, requires_grad=True)
    lp = theta['lora_weight'].log_softmax(0)
    free_loss = positive_mixture_loss(lp[[0, 1, 2]], lp[[1, 2, 1]], a)
    free = gate_vjp(free_loss, theta, a, step, gJ)
    predicted = []
    for s, t in zip([0, 1, 2], [1, 2, 1]):
        lp = theta['lora_weight'].log_softmax(0)
        diff = gradients(-lp[t] + lp[s], theta)['lora_weight']
        predicted.append(-step.eta / 3 * (gJ['lora_weight'].detach() * step.diagonal['lora_weight'] * diff).sum())
    torch.testing.assert_close(free, torch.stack(predicted))


def test_rms_is_train_only_streamed_damped_and_coordinate_normalized():
    p = {'lora_A': torch.zeros(2, dtype=DT), 'lora_B': torch.zeros(1, dtype=DT)}
    gs = [('inner', {'lora_A': torch.tensor([1., 0.], dtype=DT, requires_grad=True),
                     'lora_B': torch.tensor([3.], dtype=DT)})] * 2
    P, audit = rms_diagonal(p, iter(gs), inner_parent_hashes={'inner'}, source_snapshot_id='source')
    rms = torch.tensor([1., 0., 3.], dtype=DT)
    expected = 1 / (rms + .01 * rms.mean()); expected /= expected.mean()
    torch.testing.assert_close(torch.cat(list(P.values())), expected)
    assert all(not t.requires_grad for t in P.values()) and audit['source_rollouts'] == 2
    with pytest.raises(ValueError, match='outside inner'):
        rms_diagonal(p, gs, inner_parent_hashes={'feedback'}, source_snapshot_id='source')
    zero, audit = rms_diagonal(p, [('inner', p)], inner_parent_hashes={'inner'}, source_snapshot_id='source')
    assert audit['zero_rms_identity'] and all(t.eq(1).all() for t in zero.values())


def test_legitimate_kl_pilot_fixed_alpha_target_and_no_commit():
    theta, _, _, step, loss, _ = problem()
    initial = tensor_state_hash(theta)
    alphas = []
    def pilot_loss(alpha):
        alphas.append(alpha)
        lp = theta['lora_weight'].log_softmax(0)
        return positive_mixture_loss(lp[[0]], lp[[1]], lp.new_tensor(alpha))
    p = theta['lora_weight'].detach().softmax(0)
    def kl(updated):
        return (p * (p.log() - updated['lora_weight'].log_softmax(0))).sum()
    kwargs = dict(candidates=[.01, .05, .1, .3, .5], owned_ids={'paid'}, evidence_ids={'paid'},
        inner_parent_hashes={'i'}, evidence_parents={'i'}, source_parents={'i'})
    eta, audit = kl_pilot(theta, step.diagonal, pilot_loss, kl, **kwargs)
    assert alphas == [.5] and audit['target'] == .005
    assert eta == min(audit['trials'], key=lambda r: abs(r['kl'] - .005))['eta']
    assert tensor_state_hash(theta) == initial and audit['committed_updates'] == 0
    for override in [dict(evidence_ids={'sealed'}), dict(source_parents={'feedback'}), dict(evidence_ids=set())]:
        with pytest.raises(ValueError, match='purchased evidence'):
            kl_pilot(theta, step.diagonal, pilot_loss, kl, **(kwargs | override))


def test_commit_only_lora_exact_noop_and_stale_start_rejection():
    model = torch.nn.Module()
    model.register_parameter('lora_A', torch.nn.Parameter(torch.tensor([1., 2.], dtype=DT)))
    model.register_parameter('base', torch.nn.Parameter(torch.tensor([7.], dtype=DT), requires_grad=False))
    p = lora_parameters(model); initial = tensor_state_hash(p)
    step = FrozenStep({'lora_A': torch.ones(2, dtype=DT)}, .1, 'r1', {})
    noop = functional_step(None, p, step, exact_noop=True)
    commit_step(model, noop, expected_start_hash=initial)
    assert tensor_state_hash(p) == initial
    changed = functional_step(p['lora_A'].square().sum(), p, step)
    commit_step(model, changed, expected_start_hash=initial)
    torch.testing.assert_close(model.lora_A, torch.tensor([.8, 1.6], dtype=DT))
    assert model.base.item() == 7
    with pytest.raises(ValueError, match='moved'):
        commit_step(model, changed, expected_start_hash=initial)
    model.base.requires_grad_(True)
    with pytest.raises(ValueError, match='LoRA'):
        lora_parameters(model)


def test_insertion_derivative_central_difference_same_start_and_empty():
    theta, _, _, step, old, reward = problem()
    new = lambda p: -p['lora_weight'].log_softmax(0)[0]
    ref = InsertionReference(theta, old, step)
    gJ = gradients(reward(ref.updated), ref.updated)
    feedback = ReturnGradient(gJ, ref.reference_hash, {})
    actual, label, audit = ref.insert('q', new, feedback, label_type=LabelType.PENDING_NEW,
                                     owned_before=set(), pending_ids={'q'})
    gq = gradients(new(ref.start), ref.start)
    def value(epsilon):
        gradient = {n: (1 - epsilon) * ref.g_D[n] + epsilon * gq[n] for n in theta}
        return reward(step.update(ref.start, gradient))
    fd = (value(1e-5) - value(-1e-5)) / 2e-5
    assert label.value == pytest.approx(float(fd.detach()), abs=2e-11)
    torch.testing.assert_close(reward(actual), value(.25))
    assert audit['raw_old_slots'] + audit['raw_new_slots'] == 16
    assert audit['weighted_old_slots'] + audit['weighted_new_slots'] == 8
    assert tensor_state_hash(theta) == ref.start_hash
    empty, meta = ref.empty()
    assert tensor_state_hash(empty) == ref.reference_hash and meta['value'] == meta['cost'] == 0
    with pytest.raises(ValueError, match='one observation'):
        ref.insert('q', new, feedback, label_type=LabelType.PENDING_NEW, owned_before=set(), pending_ids={'q'})
    bad = ReturnGradient(gJ, tensor_state_hash(actual), {})
    with pytest.raises(ValueError, match='same-start reference'):
        ref.insert('other', new, bad, label_type=LabelType.PENDING_NEW, owned_before=set(), pending_ids={'other'})


def test_equal_gradients_zero_insertion_and_no_gD_control_is_wrong():
    theta, _, _, step, old, reward = problem()
    ref = InsertionReference(theta, old, step)
    gJ = gradients(reward(ref.updated), ref.updated)
    correct = insertion_derivative(ref.g_D, ref.g_D, gJ, step)
    wrong = -step.eta * sum((gJ[n] * step.diagonal[n] * ref.g_D[n]).sum() for n in theta)
    assert correct.item() == 0 and abs(float(wrong.detach())) > 1e-4
    f = lambda e: reward(step.update(ref.start, {n: (1-e)*ref.g_D[n]+e*ref.g_D[n] for n in theta}))
    assert float(((f(1e-5) - f(-1e-5)) / 2e-5).detach()) == 0


def test_label_types_and_identity_reference():
    theta, _, _, step, old, _ = problem()
    ref = InsertionReference(theta, None, step, exact_noop=True)
    assert ref.start_hash == ref.reference_hash
    fb = ReturnGradient({n: torch.ones_like(p) for n, p in theta.items()}, ref.reference_hash, {})
    with pytest.raises(ValueError, match='newly purchased'):
        ref.insert('q', old, fb, label_type=LabelType.PENDING_NEW, owned_before={'q'}, pending_ids={'q'})
    _, label, _ = ref.insert('q', old, fb, label_type=LabelType.REWEIGHT_EXISTING,
                             owned_before={'q'}, pending_ids=set())
    assert label.label_type is LabelType.REWEIGHT_EXISTING
    assert ref.empty()[1]['weighted_slots'] == 0


def test_controller_uses_past_rms_and_regularization_in_zero_reward_case():
    control = GateController()
    phi = torch.tensor([.3, -.4], dtype=DT, requires_grad=True)
    updated, meta = control.update(phi, torch.zeros_like(phi))
    torch.testing.assert_close(updated, .9 * phi)
    assert meta['past_rms_scale'] == 1e-8
    control.past_sum_squares, control.past_coordinates = 8., 2
    _, meta = control.update(phi, torch.full_like(phi, 100.))
    assert meta['past_rms_scale'] == 2.  # does not see the current large derivative


def test_actual_gate_rejects_reference_gradient_at_a_different_model():
    theta, phi, _, step, loss, _ = problem()
    actual = functional_step(loss(theta), theta, step)
    wrong = ReturnGradient({n: torch.ones_like(p) for n, p in theta.items()}, tensor_state_hash(theta), {})
    with pytest.raises(ValueError, match='actual updated'):
        actual_gate_vjp(loss(theta), theta, phi, step, actual, wrong)
