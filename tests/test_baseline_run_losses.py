"""Numerical masking, weighting, discriminator and policy-gradient checks."""
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/"src"), str(ROOT)]
from bfas.rtd.baselines.paper_losses import (Span, SmallDiscriminator, discriminator_loss,
    group_advantages, grpo_loss, segment_spans, span_ce, token_kinds)


def test_spans_cover_reason_action_observation_final_without_overlap():
    text = "[REASON]Think.[/REASON]\n[ACT]go[OBS]environment[FINAL]done"
    spans = segment_spans(text, benchmark="bfcl")
    assert [(text[s.start:s.end], s.kind) for s in spans] == [
        ("[REASON]Think.", "reason"), ("[/REASON]", "reason"), ("\n", "final"),
        ("[ACT]go", "action"), ("[OBS]environment", "observation"), ("[FINAL]done", "final")]
    assert spans[0].start == 0 and spans[-1].end == len(text)
    assert all(a.end == b.start for a, b in zip(spans, spans[1:]))
    def at(word):
        index = text.index(word)
        return next(s.kind for s in spans if s.start <= index < s.end)
    assert [at(w) for w in ("Think", "go", "environment", "done")] == ["reason", "action", "observation", "final"]


def test_native_call_and_generated_handoff_are_action_not_observation():
    text = "<|channel>thought\n<channel|><|tool_call>call:f{}<|tool_response>observed"
    spans = segment_spans(text)
    assert [(text[s.start:s.end], s.kind) for s in spans] == [
        ("<|channel>thought\n", "reason"), ("<channel|>", "reason"),
        ("<|tool_call>call:f{}", "action"), ("<|tool_response>", "action"),
        ("observed", "observation")]
    for word in ("call:f", "<|tool_response>"):
        assert next(s.kind for s in spans if s.start <= text.index(word) < s.end) == "action"
    assert spans[-1].kind == "observation"


@pytest.mark.parametrize("text", ["go to table 1", "ACTION: go to table 1"])
def test_alfworld_terminal_decision_weight(text):
    assert segment_spans(text, benchmark="alfworld") == [Span(0, len(text), "action")]
    assert segment_spans(text, benchmark="alfworld", final_step=True) == [Span(0, len(text), "final")]


@pytest.mark.parametrize("prefix", ["", "THOUGHT: "])
@pytest.mark.parametrize("final_step", [False, True])
def test_alfworld_react_reasoning_with_or_without_thought_label(prefix, final_step):
    text = (prefix + "I need to find the alarm clock, likely on one of the nearby sidetables. "
            "I'll check sidetable 1 first.\n\nACTION: go to sidetable 1")
    boundary = text.index("ACTION:")
    action_kind = "final" if final_step else "action"
    assert segment_spans(text, benchmark="alfworld", final_step=final_step) == [
        Span(0, boundary, "reason"), Span(boundary, len(text), action_kind)]
    # Unequal reason/action token counts distinguish SAD from plain token CE.
    tokenizer = lambda *a, **kw: dict(input_ids=[1, 2, 3],
        offset_mapping=[(0, 10), (10, boundary), (boundary, len(text))])
    ids, kinds = token_kinds(tokenizer, text, benchmark="alfworld", final_step=final_step)
    assert ids == (1, 2, 3) and kinds == ["reason", "reason", action_kind]
    logp = torch.tensor([-2., -4., -10.])
    assert span_ce(logp, kinds, "sad").item() == pytest.approx(6.5)
    assert span_ce(logp, kinds, "smartad").item() == pytest.approx((6 + (20 if final_step else 15))/3)


@pytest.mark.parametrize("marker", ["ACTION:", "act:", "[ACT]", "<|tool_call>"])
def test_alfworld_unlabelled_prefix_uses_existing_action_markers(marker):
    text = "Inspect the table.\n" + marker + "go to table 1"
    boundary = text.index(marker)
    assert segment_spans(text, benchmark="alfworld") == [
        Span(0, boundary, "reason"), Span(boundary, len(text), "action")]


def test_alfworld_reasoning_prefix_keeps_observations_masked_and_spans_exhaustive():
    pieces = [("Plan.\n", "reason"), ("[OBS]environment", "observation"),
              ("[/OBS]", "observation"), ("\nInspect the table.\n", "reason"),
              ("ACTION: go to table 1\n", "action"), ("OBSERVATION: a table", "observation")]
    text = "".join(part for part, _ in pieces)
    spans = segment_spans(text, benchmark="alfworld")
    assert [(text[s.start:s.end], s.kind) for s in spans] == pieces
    assert spans[0].start == 0 and spans[-1].end == len(text)
    assert all(a.end == b.start for a, b in zip(spans, spans[1:]))
    tokenizer = lambda *a, **kw: dict(input_ids=list(range(len(text))),
        offset_mapping=[(i, i+1) for i in range(len(text))])
    _, kinds = token_kinds(tokenizer, text, benchmark="alfworld")
    assert kinds == [kind for part, kind in pieces for _ in part]
    for method in ("smartad", "sad"):
        logp = torch.full((len(text),), -2., requires_grad=True)
        span_ce(logp, kinds, method).backward()
        assert all(logp.grad[i] == 0 for i, kind in enumerate(kinds) if kind == "observation")


def test_segment_weighted_ce_and_masked_gradient():
    logp = torch.tensor([-1., -2., -3., -100.], requires_grad=True)
    loss = span_ce(logp, ["reason", "action", "final", "observation"], "smartad")
    assert float(loss.detach()) == pytest.approx((1+3+6)/3)
    loss.backward()
    assert torch.allclose(logp.grad, torch.tensor([-1/3, -.5, -2/3, 0.]))


@pytest.mark.parametrize("method", ["sad", "sad_mean"])
def test_sad_balances_spans_instead_of_tokens_and_omits_missing_group(method):
    logp = torch.tensor([-2., -4., -10., -900.])
    assert span_ce(logp, ["reason", "reason", "action", "observation"], method) == 6.5
    assert span_ce(logp[:2], ["action", "final"], method) == 3
    with pytest.raises(ValueError):
        span_ce(logp, ["observation"]*4, method)


def test_sad_sum_is_generated_token_ce_and_mean_ablation_has_different_weights():
    # Three reasoning tokens versus two action/final tokens, with different NLLs.
    kinds = ["reason", "reason", "reason", "action", "final", "observation"]
    logp = torch.tensor([-2., -4., -6., -10., -14., -900.], requires_grad=True)
    total = span_ce(logp, kinds, "sad_sum")
    mean = span_ce(logp, kinds, "sad_mean")
    torch.testing.assert_close(total, -logp[:5].sum()/5)
    torch.testing.assert_close(total, span_ce(logp, kinds, "sft"))
    assert total.item() == pytest.approx(36/5)
    assert mean.item() == pytest.approx((12/3 + 24/2)/2)
    assert total.item() != pytest.approx(mean.item())
    torch.testing.assert_close(mean, span_ce(logp, kinds, "sad"))
    torch.testing.assert_close(torch.autograd.grad(total, logp)[0],
                               torch.tensor([-.2, -.2, -.2, -.2, -.2, 0.]))
    torch.testing.assert_close(torch.autograd.grad(mean, logp)[0],
                               torch.tensor([-1/6, -1/6, -1/6, -.25, -.25, 0.]))
    changed = logp.detach().clone()
    changed[-1] = -1e6
    for method in ("sad_sum", "sad_mean"):
        torch.testing.assert_close(span_ce(changed, kinds, method), span_ce(logp, kinds, method))


@pytest.mark.parametrize("kinds", [["reason", "reason"], ["action", "final"]])
def test_sad_sum_single_group_and_no_generated_tokens(kinds):
    logp = torch.tensor([-2., -4., -900.])
    assert span_ce(logp, kinds + ["observation"], "sad_sum") == 3
    with pytest.raises(ValueError, match="no generated supervision"):
        span_ce(logp, ["observation"]*3, "sad_sum")


@pytest.mark.parametrize("benchmark", ["bfcl", "alfworld"])
def test_offset_boundary_touching_observation_is_masked(benchmark):
    text = "[ACT]a[OBS]b"
    class Tokens:
        def __call__(self, text, **kwargs):
            return dict(input_ids=[7, 8], offset_mapping=[(0, 5), (5, len(text))])
    ids, kinds = token_kinds(Tokens(), text, benchmark=benchmark)
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
