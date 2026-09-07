from dataclasses import replace

import pytest
import torch

from bfas.rtd.baselines.feedback import combine_pg
from bfas.rtd.baselines.losses import pairwise_loss, slot_loss, supervised_gradient, teacher_weight_vjp, teacher_weights
from bfas.rtd.baselines.pool import Slot
from bfas.rtd.functional_step import FrozenStep, gradients, lora_parameters, snapshot
from bfas.rtd.return_gradient import ReturnGradient, gate_vjp
from bfas.rtd.transport import Behavior, FullState, SourceSample
from test_rtd_baselines_helpers import tiny_backend


def slots_and_backend():
    backend = tiny_backend()
    p = lora_parameters(backend.model)
    state = FullState.create(dict(question="synthetic"), [dict(role="user", content="synthetic")], "prompt0", "00")
    # Legal empty action contains exactly EOS. It must not be dropped as a reject.
    source = SourceSample(Behavior(state, ""), backend.identity(p), (3,), 3, -1.)
    return backend, p, [Slot(source, Behavior(state, "A"), "q1", .25, 2, 2),
                        Slot(source, Behavior(state, "BBB"), "q2", .75, 2, 4)]


def test_b2_matches_base_centered_logistic_and_keeps_empty_reject():
    backend, p, slots = slots_and_backend()
    initial = snapshot(p)
    with torch.no_grad():
        p["lora_transition"][1, 3] += .3
    refs = [(backend.score_behavior(s.teacher, initial).detach(), backend.score_source(s.source, initial).detach()) for s in slots]
    expected = sum(slot_loss(s, backend, p, "b2", reference=ref) for s, ref in zip(slots, refs))/2
    expected_g = gradients(expected, p)
    actual, _ = supervised_gradient(slots, [1., 1.], backend, p, "b2", references=refs)
    torch.testing.assert_close(actual["lora_transition"], expected_g["lora_transition"])
    identical = pairwise_loss(torch.tensor(-3.), torch.tensor(-3.), torch.tensor(-4.), torch.tensor(-4.))
    assert float(identical) == pytest.approx(.69314718056)
    assert all(s.source.length == 1 for s in slots)
    assert slots[0].costs("b2")["input_tokens"] == 7


@pytest.mark.parametrize("recipe", ["b1", "b3_sft", "b3_mix", "b4"])
def test_streamed_importance_gradients_match_complete_action_oracle(recipe):
    backend, p, slots = slots_and_backend()
    weights = {"q1": torch.tensor(.5), "q2": torch.tensor(1.5)}
    rhos = [.5, 1.5]
    loss = sum(rho*slot_loss(s, backend, p, recipe, weight=weights[s.query_id]) for s, rho in zip(slots, rhos))/2
    expected = gradients(loss, p)
    actual, value = supervised_gradient(slots, rhos, backend, p, recipe, weights=weights)
    torch.testing.assert_close(actual["lora_transition"], expected["lora_transition"])
    assert value == pytest.approx(float(loss.detach()))


def test_b4_teacher_only_vjp_matches_general_interface_and_finite_difference(monkeypatch):
    backend, p, slots = slots_and_backend()
    u = torch.tensor([.2, -.3, 2.], dtype=torch.float64, requires_grad=True)
    query_ids, mass = ["q1", "q2", "heldout"], {"q1": .25, "q2": .75}
    weights = teacher_weights(u, query_ids, mass)
    assert sum(mass[q]*weights[q] for q in mass).item() == pytest.approx(1.)
    # Deliberately change the third (feedback-fold) coordinate: no effect.
    changed = u.detach().clone(); changed[2] = 100.
    assert teacher_weights(changed, query_ids, mass)["q1"] == weights["q1"]
    step = FrozenStep({n: torch.ones_like(x)*1.3 for n, x in p.items()}, .02, "r1", {})
    gJ = {n: torch.arange(x.numel(), dtype=x.dtype).reshape_as(x)/x.numel() for n, x in p.items()}
    feedback = ReturnGradient(gJ, "virtual-policy", {})
    def loss(values):
        w = teacher_weights(values, query_ids, mass)
        return sum(s.p*slot_loss(s, backend, p, "b4", weight=w[s.query_id]) for s in slots)
    expected = gate_vjp(loss(u), p, u, step, gJ)
    monkeypatch.setattr(backend, "score_source", lambda *a: (_ for _ in ()).throw(AssertionError("B4 transport/source loss")))
    import bfas.rtd.runtime as runtime
    monkeypatch.setattr(runtime, "streamed_gate_vjp", lambda *a, **k: (_ for _ in ()).throw(AssertionError("transport gate VJP")))
    # Two draws with rho=.5/1.5 represent p=.25/.75 exactly in this oracle.
    got = teacher_weight_vjp(slots, [.5, 1.5], backend, p, u, query_ids, mass, step, feedback)
    torch.testing.assert_close(got, expected)
    def outer(values):
        updated = step.update(p, gradients(loss(values), p))
        return sum((updated[n]*gJ[n]).sum() for n in p).detach()
    for i in range(3):
        plus, minus = u.detach().clone(), u.detach().clone()
        plus[i] += 1e-5; minus[i] -= 1e-5
        assert float(got[i]) == pytest.approx(float((outer(plus)-outer(minus))/2e-5), abs=1e-9)
    assert got[2] == 0


def test_pg_direction_and_stale_policy_refusal():
    feedback = ReturnGradient({"lora_x": torch.tensor([2.])}, "theta", {})
    combined = combine_pg({"lora_x": torch.tensor([.5])}, feedback, parameter_hash="theta")
    assert combined["lora_x"].item() == -1.5
    assert (torch.tensor([0.]) - .1*combined["lora_x"]).item() > 0  # return ASCENT
    with pytest.raises(ValueError, match="same pre-commit"):
        combine_pg({"lora_x": torch.tensor([0.])}, feedback, parameter_hash="already-sft-updated")


def test_teacher_source_full_state_and_parent_must_both_match():
    _, _, slots = slots_and_backend()
    bad = replace(slots[0].teacher.state, parent_hash="01")
    with pytest.raises(ValueError, match="mismatch"):
        replace(slots[0], teacher=Behavior(bad, "A"))
