"""Numerical masking, weighting, discriminator and policy-gradient checks."""
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/"src"), str(ROOT)]
from bfas.rtd.baselines.paper_losses import (SmallDiscriminator, discriminator_loss,
    group_advantages, grpo_loss, segment_spans, span_ce, token_kinds)


def test_spans_cover_reason_action_observation_final_without_overlap():
    text = "[REASON]Think.[/REASON]\n[ACT]go[OBS]environment[FINAL]done"
    spans = segment_spans(text, benchmark="bfcl")
    assert spans[0].start == 0 and spans[-1].end == len(text)
    assert all(a.end == b.start for a, b in zip(spans, spans[1:]))
    def at(word):
        index = text.index(word)
        return next(s.kind for s in spans if s.start <= index < s.end)
    assert [at(w) for w in ("Think", "go", "environment", "done")] == ["reason", "action", "observation", "final"]


def test_native_call_and_generated_handoff_are_action_not_observation():
    text = "<|channel>thought\n<channel|><|tool_call>call:f{}<|tool_response>observed"
    spans = segment_spans(text)
    for word in ("call:f", "<|tool_response>"):
        assert next(s.kind for s in spans if s.start <= text.index(word) < s.end) == "action"
    assert spans[-1].kind == "observation"


@pytest.mark.parametrize("text", ["go to table 1", "ACTION: go to table 1"])
def test_alfworld_terminal_decision_weight(text):
    assert segment_spans(text, benchmark="alfworld")[-1].kind == "action"
    assert segment_spans(text, benchmark="alfworld", final_step=True)[-1].kind == "final"


def test_segment_weighted_ce_and_masked_gradient():
    logp = torch.tensor([-1., -2., -3., -100.], requires_grad=True)
    loss = span_ce(logp, ["reason", "action", "final", "observation"], "smartad")
    assert float(loss.detach()) == pytest.approx((1+3+6)/3)
    loss.backward()
    assert torch.allclose(logp.grad, torch.tensor([-1/3, -.5, -2/3, 0.]))


def test_sad_balances_spans_instead_of_tokens_and_omits_missing_group():
    logp = torch.tensor([-2., -4., -10., -900.])
    assert span_ce(logp, ["reason", "reason", "action", "observation"], "sad") == 6.5
    assert span_ce(logp[:2], ["action", "final"], "sad") == 3
    with pytest.raises(ValueError):
        span_ce(logp, ["observation"]*4, "sad")


def test_offset_boundary_touching_observation_is_masked():
    text = "[ACT]a[OBS]b"
    class Tokens:
        def __call__(self, text, **kwargs):
            return dict(input_ids=[7, 8], offset_mapping=[(0, 5), (5, len(text))])
    ids, kinds = token_kinds(Tokens(), text, benchmark="bfcl")
    assert ids == (7, 8) and kinds == ["action", "observation"]


def test_discriminator_bt_learns_teacher_preference_on_same_prompt():
    torch.manual_seed(0)
    model = SmallDiscriminator(width=4, hidden=6)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.03, weight_decay=0.)
    def loss():
        return discriminator_loss(model(["same prompt"], ["teacher"]), model(["same prompt"], ["student"]))
    before = float(loss().detach())
    for _ in range(8):
        optimizer.zero_grad(set_to_none=True)
        value = loss()
        value.backward()
        optimizer.step()
    assert float(loss().detach()) < before
    assert not torch.equal(model(["prompt A"], ["teacher"]), model(["prompt B"], ["teacher"]))


def test_advantages_detached_constant_groups_safe():
    rewards = torch.tensor([1., 2., 3., 4.], requires_grad=True)
    advantages = group_advantages(rewards)
    assert not advantages.requires_grad
    assert float(advantages.mean()) == pytest.approx(0., abs=1e-6)
    assert float(advantages.std(unbiased=False)) == pytest.approx(1.)
    assert torch.equal(group_advantages(torch.ones(4)), torch.zeros(4))
    with pytest.raises(ValueError):
        group_advantages(torch.tensor([float("nan"), 0.]))


def test_grpo_reward_moves_student_and_never_discriminator():
    student = torch.tensor([.2, -.2], requires_grad=True)
    rewards = torch.tensor([2., 0.], requires_grad=True)
    advantages = group_advantages(rewards)
    logps = student.log_softmax(0)
    loss = sum(grpo_loss(logps[i:i+1], logps[i:i+1].detach(), advantages[i]) for i in range(2))/2
    loss.backward()
    assert student.grad[0] < 0 and student.grad[1] > 0
    assert rewards.grad is None
    before = student.softmax(0)[0].detach()
    assert (student-.1*student.grad).softmax(0)[0] > before


def test_grpo_clips_positive_overlarge_policy_ratio():
    value = torch.tensor([1.], requires_grad=True)
    loss = grpo_loss(value, torch.zeros(1), torch.tensor(1.))
    loss.backward()
    assert float(loss.detach()) == pytest.approx(-1.2)
    assert float(value.grad) == 0
