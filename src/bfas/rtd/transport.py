"""RTD equations (1)/(2), independent of acquisition and the student optimizer."""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Protocol

import torch
import torch.nn.functional as F

from ..cc_pairs import digest


def _json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


@dataclass(frozen=True)
class FullState:
    """Immutable task plus ordered, already-observed history and exact prompt.

    Task JSON includes the environment's initial configuration and tools.
    History preserves the interleaving of observations and previous actions.
    No observation is a supervised output; teacher bridge states are only
    constructed after the package and its prefix dependencies are owned.
    """
    task_json: str
    history_json: str
    prompt: str
    parent_hash: str

    @classmethod
    def create(cls, task, history, prompt, parent_hash):
        state = cls(_json(task), _json(history), prompt, parent_hash)
        state.validate()
        return state

    def validate(self):
        task, history = json.loads(self.task_json), json.loads(self.history_json)
        if not isinstance(task, dict) or not task or not self.prompt or not self.parent_hash:
            raise ValueError("complete task, prompt and parent identity required")
        if not isinstance(history, list) or not history:
            raise ValueError("full observation/action history required")
        for message in history:
            if (not isinstance(message, dict) or message.get("role") not in {"system", "user", "assistant", "tool"}
                    or "content" not in message):
                raise ValueError("invalid full-state history message")
        if self.task_json != _json(task) or self.history_json != _json(history):
            raise ValueError("state JSON must be canonical")
        return self

    @property
    def state_hash(self):
        # Parent identity is not part of state equality: identical states alias.
        return digest(dict(task=json.loads(self.task_json), history=json.loads(self.history_json), prompt=self.prompt))

    def assert_matches(self, other: FullState):
        self.validate()
        other.validate()
        if self.state_hash != other.state_hash or self.parent_hash != other.parent_hash:
            raise ValueError("teacher/source full state mismatch")


@dataclass(frozen=True)
class Behavior:
    state: FullState
    text: str

    def __post_init__(self):
        self.state.validate()
        if not isinstance(self.text, str):
            raise ValueError("complete action text required (empty text is allowed)")


@dataclass(frozen=True)
class SamplingRequest:
    state: FullState
    frozen_snapshot_id: str
    samples: int = 2
    temperature: float = 1.0
    top_p: float = 1.0
    do_sample: bool = True

    def __post_init__(self):
        self.state.validate()
        if not self.frozen_snapshot_id or self.samples < 1:
            raise ValueError("frozen source snapshot and positive sample count required")
        if self.temperature != 1 or self.top_p != 1 or not self.do_sample:
            raise ValueError("v1 requires random temperature=1, top_p=1 sampling")


@dataclass(frozen=True)
class SourceSample:
    behavior: Behavior
    frozen_snapshot_id: str
    token_ids: tuple[int, ...]
    eos_token_id: int
    logprob: float
    truncated: bool = False

    def __post_init__(self):
        validate_sampled_action(self.token_ids, self.eos_token_id, self.truncated)
        if not math.isfinite(self.logprob) or self.logprob > 0 or not self.frozen_snapshot_id:
            raise ValueError("finite full-sequence frozen-source log probability required")

    @property
    def length(self):
        return len(self.token_ids)


class RandomSourceSampler(Protocol):
    def __call__(self, request: SamplingRequest, generator: torch.Generator) -> tuple[SourceSample, ...]:
        """Independent frozen-policy samples, through EOS or the configured cap."""
        ...


def sample_sources(request: SamplingRequest, sampler: RandomSourceSampler, generator: torch.Generator):
    samples = tuple(sampler(request, generator))
    if len(samples) != request.samples:
        raise ValueError("sampler returned the wrong number of independent source slots")
    for sample in samples:
        request.state.assert_matches(sample.behavior.state)
        if sample.frozen_snapshot_id != request.frozen_snapshot_id:
            raise ValueError("sampler used a different source policy")
    return samples


@dataclass(frozen=True)
class TransportSlot:
    source: SourceSample
    teachers: tuple[Behavior, ...] = ()

    def __post_init__(self):
        for teacher in self.teachers:
            self.source.behavior.state.assert_matches(teacher.state)

    def sample_teacher(self, generator: torch.Generator):
        """Uniform empirical teacher distribution; costs stay at package level."""
        if not self.teachers:
            return None
        return self.teachers[int(torch.randint(len(self.teachers), (), generator=generator))]


def is_exact_noop(*, has_teacher_evidence: bool, student_snapshot_id: str, frozen_snapshot_id: str):
    if not student_snapshot_id or not frozen_snapshot_id:
        raise ValueError("snapshot identities required for exact no-op")
    return not has_teacher_evidence and student_snapshot_id == frozen_snapshot_id


def positive_mixture_loss(source_logprobs: torch.Tensor, teacher_logprobs: torch.Tensor | None,
                          gate: torch.Tensor, *, teacher_available: torch.Tensor | None = None,
                          reduction: str = "mean"):
    """Equation (2), per complete source slot; no token-length normalization.

    Gate remains differentiable. Frozen-source features/samples are detached by
    features.py. These log-probs are scored under the *current* student. Missing
    evidence sets a=0. The trainer must skip the optimizer for is_exact_noop().
    """
    source = source_logprobs
    if not source.is_floating_point() or not torch.isfinite(source).all() or (source > 0).any():
        raise ValueError("finite nonpositive full-sequence source log-probs required")
    if not torch.isfinite(gate).all() or ((gate < 0) | (gate > 1)).any():
        raise ValueError("gate must be in [0,1]")
    if gate.shape not in (torch.Size([]), source.shape):
        raise ValueError("one gate per source slot required")
    if teacher_logprobs is None:
        if teacher_available is not None and teacher_available.any():
            raise ValueError("teacher marked available without a scored response")
        losses = -source
    else:
        if teacher_logprobs.shape != source.shape:
            raise ValueError("teacher/source slots must match")
        mask = torch.ones_like(source, dtype=torch.bool) if teacher_available is None else teacher_available
        if mask.shape != source.shape or mask.dtype != torch.bool:
            raise ValueError("one boolean availability flag per source slot required")
        if not torch.isfinite(teacher_logprobs[mask]).all() or (teacher_logprobs[mask] > 0).any():
            raise ValueError("finite nonpositive teacher log-probs required")
        a = torch.where(mask, gate, torch.zeros_like(source))
        teacher = torch.where(mask, teacher_logprobs, torch.zeros_like(source))
        losses = (1 - a) * (-source) + a * (-teacher)
    if reduction == "none":
        return losses
    if reduction == "sum":
        return losses.sum()
    if reduction != "mean" or losses.numel() == 0:
        raise ValueError("nonempty slots and mean/sum/none reduction required")
    return losses.mean()


def finite_transport_q(source_probs: torch.Tensor, gate: torch.Tensor,
                       teacher_probs: torch.Tensor | None):
    """Exact equation (1) on an explicitly complete finite action universe.

    This is a toy oracle, never a candidate-set renormalizer. Teacher and source
    vectors use the same full universe, including actions outside teacher support.
    """
    def distribution(p):
        if p.ndim != 1 or not p.numel() or not torch.isfinite(p).all() or (p < 0).any():
            raise ValueError("invalid finite distribution")
        if not torch.allclose(p.sum(), p.new_tensor(1.), atol=1e-7, rtol=1e-7):
            raise ValueError("complete distribution must already sum to one")
    distribution(source_probs)
    if gate.shape not in (torch.Size([]), source_probs.shape) or not torch.isfinite(gate).all() or ((gate < 0) | (gate > 1)).any():
        raise ValueError("invalid source-dependent gate")
    if teacher_probs is None:
        return source_probs.clone()
    distribution(teacher_probs)
    if teacher_probs.shape != source_probs.shape:
        raise ValueError("teacher and source must share the full action universe")
    return source_probs * (1 - gate) + (source_probs * gate).sum() * teacher_probs


def validate_sampled_action(ids, eos_token_id, truncated=False):
    if (not len(ids) or type(truncated) is not bool or eos_token_id in ids[:-1]
            or (ids[-1] == eos_token_id) == truncated):
        raise ValueError("sampled action must end at first EOS termination or be explicitly truncated without EOS")


def complete_token_logprobs(logits, token_ids, action_mask, *, eos_token_id, truncated=False):
    """Differentiable native CE per selected token, in batch/action order.

    logits[b,t] predicts token_ids[b,t+1]. The boolean mask selects exactly one
    contiguous sampled action, including EOS only when sampled, and excludes
    prompt/observations/padding. Explicit truncation never adds an EOS term.
    This tensor primitive is also usable by the future T3 student scorer.
    """
    if logits.ndim != 3 or token_ids.shape != logits.shape[:2] or action_mask.shape != token_ids.shape or action_mask.dtype != torch.bool:
        raise ValueError("invalid logits/token/mask shapes")
    if action_mask[:, 0].any():
        raise ValueError("the first action token needs a preceding prompt token")
    flags = [truncated]*len(token_ids) if type(truncated) is bool else truncated
    if len(flags) != len(token_ids):
        raise ValueError("one truncation flag per action required")
    for ids, mask, capped in zip(token_ids, action_mask, flags):
        positions = mask.nonzero().flatten()
        if not len(positions) or int(positions[-1] - positions[0] + 1) != len(positions):
            raise ValueError("one contiguous complete action required")
        action = ids[mask]
        validate_sampled_action(action, eos_token_id, capped)
    shifted = action_mask[:, 1:]
    # Select positions BEFORE CE, as margin.py does: irrelevant pad labels may
    # be -100, and no environment token is accidentally scored.
    accumulation_dtype = torch.float64 if logits.dtype == torch.float64 else torch.float32
    return -F.cross_entropy(logits[:, :-1][shifted].to(accumulation_dtype),
                            token_ids[:, 1:][shifted], reduction="none")


def complete_sequence_logprob(logits, token_ids, action_mask, *, eos_token_id, truncated=False):
    """Sum the score-audit CE, including EOS only when it was sampled."""
    values = complete_token_logprobs(logits, token_ids, action_mask, eos_token_id=eos_token_id, truncated=truncated)
    scores = []
    offset = 0
    for count in action_mask[:, 1:].sum(dim=1).tolist():
        scores.append(values[offset:offset + count].sum())
        offset += count
    return torch.stack(scores)


def score_behaviors(student, behaviors, protocol):
    """Inference scoring reuses the existing EOS-inclusive margin implementation.

    No length truncation is allowed. The future trainer uses differentiable
    log-probs rather than the inference scorer's detached floats.
    """
    from ..behavior.margin import score_pairs
    from ..cc_pairs import thinking_off
    if protocol["micro_update"]["allow_prompt_truncation"]:
        raise ValueError("RTD forbids semantic prompt truncation")
    jobs = []
    for i, behavior in enumerate(behaviors):
        prompt = behavior.state.prompt
        # margin applies this preprocessing; refuse to silently change the state.
        if thinking_off(prompt) != prompt:
            raise ValueError("state must already use the frozen thinking-off harness prompt")
        jobs.append(dict(probe_id=str(i), prompt=prompt, y_plus=behavior.text, y_minus=None))
    return score_pairs(student, jobs, protocol)
