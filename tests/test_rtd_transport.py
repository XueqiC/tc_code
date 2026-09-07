"""CPU oracles for the full-space target, sequence scoring and frozen features."""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    sys.path.insert(0, str(path))

from bfas.rtd.transport import (Behavior, FullState, SamplingRequest, SourceSample,
    TransportSlot, complete_sequence_logprob, finite_transport_q, is_exact_noop,
    positive_mixture_loss, sample_sources, score_behaviors)
from bfas.rtd.features import FeatureRow, FrozenProjection, FrozenStandardizer, linear_sigmoid_gate
from bfas.rtd.selector import PublicFeatures


def state(history=None):
    return FullState.create({"question": "x", "tools": []}, history or [{"role": "user", "content": "x"}],
                            "prompt<think>\n\n</think>\n\n", "parent")


def test_q_normalization_identity_teacher_endpoint_and_outside_teacher_support():
    p = torch.tensor([.1, .2, .3, .4], dtype=torch.double)
    nu = torch.tensor([0., .25, .75, 0.], dtype=torch.double)
    a = torch.tensor([.2, .4, .6, .8], dtype=torch.double)
    q = finite_transport_q(p, a, nu)
    assert q.sum() == pytest.approx(1.)
    assert (q >= 0).all()
    assert q[3] == pytest.approx(.08)  # mass outside teacher support survives
    assert torch.equal(finite_transport_q(p, torch.tensor(0.), nu), p)
    assert torch.equal(finite_transport_q(p, torch.tensor(1.), nu), nu)
    assert torch.equal(finite_transport_q(p, a, None), p)
    with pytest.raises(ValueError, match="sum to one"):
        finite_transport_q(p[:2], a[:2], nu[:2])
    with pytest.raises(ValueError):
        finite_transport_q(p, a * 2, nu)


def test_positive_mixture_mc_expectation_equals_explicit_q_cross_entropy():
    p = torch.tensor([.12, .23, .31, .34], dtype=torch.double)
    nu = torch.tensor([.4, .6, 0., 0.], dtype=torch.double)
    a = torch.tensor([.1, .8, .3, .6], dtype=torch.double)
    logpi = torch.tensor([.15, .2, .25, .4], dtype=torch.double).log()
    expected = -(finite_transport_q(p, a, nu) * logpi).sum()
    exact = sum(p[i] * nu[j] * positive_mixture_loss(logpi[i], logpi[j], a[i])
                for i in range(4) for j in range(4))
    assert exact == pytest.approx(expected.item(), abs=1e-12)
    generator = torch.Generator().manual_seed(142)
    source = torch.multinomial(p, 150000, replacement=True, generator=generator)
    teacher = torch.multinomial(nu, 150000, replacement=True, generator=generator)
    empirical = positive_mixture_loss(logpi[source], logpi[teacher], a[source])
    assert empirical == pytest.approx(expected.item(), abs=.004)


def test_mixture_positive_gradients_availability_and_full_sequence_not_average():
    source = torch.tensor([-12., -3.], requires_grad=True)
    teacher = torch.tensor([-5., float("nan")], requires_grad=True)
    a = torch.tensor([.25, .75], requires_grad=True)
    loss = positive_mixture_loss(source, teacher, a, teacher_available=torch.tensor([True, False]), reduction="none")
    assert loss.tolist() == pytest.approx([10.25, 3.])
    loss.sum().backward()
    assert source.grad.tolist() == pytest.approx([-.75, -1.])
    assert teacher.grad.tolist() == pytest.approx([-.25, 0.])
    assert a.grad.tolist() == pytest.approx([-7., 0.])
    assert positive_mixture_loss(source, None, a).item() == pytest.approx(7.5)
    with pytest.raises(ValueError):
        positive_mixture_loss(source, teacher[:1], a)


def test_noop_requires_no_evidence_and_exact_same_policy():
    assert is_exact_noop(has_teacher_evidence=False, student_snapshot_id="frozen", frozen_snapshot_id="frozen")
    assert not is_exact_noop(has_teacher_evidence=True, student_snapshot_id="frozen", frozen_snapshot_id="frozen")
    assert not is_exact_noop(has_teacher_evidence=False, student_snapshot_id="updated", frozen_snapshot_id="frozen")


def test_full_state_history_and_random_source_contract():
    s = state()
    changed = state([{"role": "user", "content": "x"}, {"role": "assistant", "content": "old action"},
                     {"role": "tool", "content": "new observation"}])
    with pytest.raises(ValueError, match="full state mismatch"):
        s.assert_matches(changed)
    with pytest.raises(ValueError):
        FullState.create({"x": 1}, [], "prompt", "parent")
    with pytest.raises(ValueError, match="random"):
        SamplingRequest(s, "frozen", do_sample=False)
    sample = SourceSample(Behavior(s, "answer"), "frozen", (1, 2), 2, -4.)
    request = SamplingRequest(s, "frozen")
    received = []
    def sampler(req, generator):
        received.append((req.do_sample, req.temperature, req.top_p, generator.initial_seed()))
        return (sample, sample)
    assert len(sample_sources(request, sampler, torch.Generator().manual_seed(7))) == 2
    assert received == [(True, 1., 1., 7)]
    with pytest.raises(ValueError, match="source policy"):
        sample_sources(SamplingRequest(s, "other"), sampler, torch.Generator())
    with pytest.raises(ValueError, match="full state mismatch"):
        TransportSlot(sample, (Behavior(changed, "teacher"),))
    with pytest.raises(ValueError, match="termination"):
        SourceSample(Behavior(s, "answer"), "frozen", (1,), 2, -4.)
    slot = TransportSlot(sample, (Behavior(s, "a"), Behavior(s, "b")))
    generator = torch.Generator().manual_seed(8)
    assert {slot.sample_teacher(generator).text for _ in range(30)} == {"a", "b"}
    assert TransportSlot(sample).sample_teacher(generator) is None


def test_native_sequence_ce_sums_eos_excludes_observations_and_padding():
    generator = torch.Generator().manual_seed(91)
    logits = torch.randn(2, 6, 7, generator=generator, dtype=torch.double, requires_grad=True)
    ids = torch.tensor([[1, 4, 3, 6, -100, -100], [1, 2, 3, 4, 5, 6]])
    mask = torch.tensor([[False, False, True, True, False, False], [False, False, False, False, True, True]])
    scores = complete_sequence_logprob(logits, ids, mask, eos_token_id=6)
    expected = torch.stack([logits[0, 1].log_softmax(0)[3] + logits[0, 2].log_softmax(0)[6],
                            logits[1, 3].log_softmax(0)[5] + logits[1, 4].log_softmax(0)[6]])
    torch.testing.assert_close(scores, expected)  # C24 preserves the float64 CPU oracle
    scores.sum().backward()
    assert not logits.grad[0, 0].any() and not logits.grad[0, 3:].any()
    bad_mask = mask.clone(); bad_mask[1, -1] = False
    with pytest.raises(ValueError, match="EOS"):
        complete_sequence_logprob(logits, ids, bad_mask, eos_token_id=6)


def test_margin_scorer_is_reused_without_reinterpreting_state(monkeypatch):
    from bfas.behavior import margin
    calls = []
    def score(student, jobs, protocol):
        calls.append(jobs)
        return [{"logp_plus": -9.}]
    monkeypatch.setattr(margin, "score_pairs", score)
    result = score_behaviors(None, [Behavior(state(), "full action")], {"micro_update": {"allow_prompt_truncation": False}})
    assert result == [{"logp_plus": -9.}]
    assert calls[0][0]["y_plus"] == "full action" and calls[0][0]["y_minus"] is None


def test_frozen_projection_train_statistics_and_gate_stop_gradient():
    hidden = torch.arange(8., requires_grad=True)
    projection = FrozenProjection(8, initial_snapshot_id="theta0", seed=9)
    projected = projection(hidden, snapshot_id="theta0")
    assert projected == FrozenProjection(8, initial_snapshot_id="theta0", seed=9)(hidden, snapshot_id="theta0")
    with pytest.raises(ValueError, match="INITIAL"):
        projection(hidden, snapshot_id="round2")
    first = FeatureRow("inner", "s1", "theta0", PublicFeatures(projected, -10., 4))
    second = FeatureRow("inner", "s2", "theta0", PublicFeatures(tuple(x + 2 for x in projected), -6., 8))
    stats = FrozenStandardizer.fit([first, second], inner_parent_hashes={"inner"},
                                  allowed_state_hashes={"s1", "s2"}, initial_snapshot_id="theta0")
    chi = stats.transform(first)
    assert chi.shape == (35,) and chi[-1] == 1 and not chi.requires_grad
    torch.testing.assert_close(chi[:34], -torch.ones(34, dtype=torch.double))
    phi = torch.zeros(35, dtype=torch.double, requires_grad=True)
    chi.requires_grad_()
    gate = linear_sigmoid_gate(phi, chi)
    assert gate.item() == .5
    gate.backward()
    assert phi.grad is not None and chi.grad is None and hidden.grad is None
    with pytest.raises(ValueError, match="legal inner"):
        FrozenStandardizer.fit([first], inner_parent_hashes={"feedback"}, allowed_state_hashes={"s1"}, initial_snapshot_id="theta0")
    with pytest.raises(ValueError):
        stats.transform(FeatureRow("inner", "certificate", "theta0", first.features))
    with pytest.raises(ValueError, match="placeholders"):
        FeatureRow("inner", "s1", "theta0", PublicFeatures()).tensor()
