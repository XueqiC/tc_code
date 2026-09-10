"""Small-model, full-vocabulary gradient adapter using the existing sampler.

No production model is loaded here. This P0 reference path allocates full
action logits; streaming/large-model adapters are a measured P1 engineering task.
"""
from dataclasses import asdict

import torch

from ..functional_step import gradients
from ..persistence import digest
from ..source_scoring import sampled_prefix_positions
from ..transport import SamplingRequest, sample_sources
from .estimators import hard_loss, soft_retention_loss
from .immutable import Parameters, Array
from .problem import Exposure, Evidence


def behavior_identity(state_hash, token_ids, eos_token_id, truncated):
    return digest((state_hash, tuple(token_ids), eos_token_id, bool(truncated)))


def _action_logits(backend, prompt, token_ids, parameters):
    device = next(iter(parameters.values())).device
    ids = torch.tensor([tuple(prompt)+tuple(token_ids)], device=device)
    return backend._logits(parameters, ids)[0, len(prompt)-1:len(prompt)-1+len(token_ids)]


def score_sources(backend, state, context, generator, *, slot_id, weight, inverse_length_provider=None):
    """Refresh m=2 via the frozen v1.1 categorical protocol, without filtering."""
    device = next(iter(backend.model.parameters())).device
    source = context.source_snapshot.tensors(device=device)
    if backend.identity(source) != context.source_policy_hash:
        raise ValueError('source snapshot identity mismatch')
    samples = sample_sources(SamplingRequest(state, context.source_policy_hash, samples=2),
                             backend.source_sampler(source), generator)
    return score_source_samples(backend, samples, context, slot_id=slot_id, weight=weight,
                                inverse_length_provider=inverse_length_provider)


def score_source_samples(backend, samples, context, *, slot_id, weight, inverse_length_provider=None):
    if len(samples) != 2:
        raise ValueError('P0 requires two source draws')
    layout = context.theta
    device = next(iter(backend.model.parameters())).device
    parameters, source = layout.tensors(device=device), context.source_snapshot.tensors(device=device)
    state = samples[0].behavior.state
    prompt = tuple(backend.tokenizer.encode(state.prompt, add_special_tokens=False))
    hard, soft, behavior_hashes, sample_hashes = [], [], [], []
    for sample in samples:
        state.assert_matches(sample.behavior.state)
        if sample.frozen_snapshot_id != context.source_policy_hash or backend.identity(source) != sample.frozen_snapshot_id:
            raise ValueError('source snapshot identity mismatch')
        sampled_prefix_positions(sample, prompt)  # exact EOS/cap and prompt checks
        labels = torch.tensor(sample.token_ids, device=device)
        new = _action_logits(backend, prompt, sample.token_ids, parameters)
        hard.append(layout.flatten(gradients(hard_loss(new, labels, context.loss), parameters)).numpy())
        new = _action_logits(backend, prompt, sample.token_ids, parameters)
        with torch.no_grad():
            old = _action_logits(backend, prompt, sample.token_ids, source)
        rho = inverse_length_provider(sample) if inverse_length_provider is not None else None
        loss = soft_retention_loss(new, old, labels, context.loss, inverse_lengths=rho)
        soft.append(layout.flatten(gradients(loss, parameters)).numpy())
        behavior_hashes.append(behavior_identity(state.state_hash, sample.token_ids, sample.eos_token_id, sample.truncated))
        sample_hashes.append(digest(asdict(sample)))
    return Exposure(slot_id, state.state_hash, state.parent_hash, tuple(behavior_hashes), tuple(sample_hashes),
        tuple(digest((context.sampling_version, slot_id, j)) for j in range(2)),
        context.source_policy_hash, layout.hash, context.loss.hash, Array.of(hard),
        Array.of((soft[0]+soft[1])/2), weight)


def score_teacher(backend, behavior, context, *, query_id, version):
    if query_id not in context.owned_query_ids or behavior.state.parent_hash not in context.inner_parent_hashes:
        raise ValueError('teacher text must be purchased and in the inner fold before scoring')
    device = next(iter(backend.model.parameters())).device
    layout, tokenizer = context.theta, backend.tokenizer
    parameters = layout.tensors(device=device)
    prompt = tuple(tokenizer.encode(behavior.state.prompt, add_special_tokens=False))
    tokens = tuple(tokenizer.encode(behavior.text, add_special_tokens=False))
    eos = tokenizer.eos_token_id
    if not tokens or tokens[-1] != eos:
        tokens += (eos,)
    from ..transport import validate_sampled_action
    validate_sampled_action(tokens, eos, False)
    if not prompt:
        raise ValueError('complete teacher prompt required')
    logits = _action_logits(backend, prompt, tokens, parameters)
    gradient = gradients(hard_loss(logits, torch.tensor(tokens, device=device), context.loss), parameters)
    return Evidence(query_id, version, behavior.state.state_hash, behavior.state.parent_hash, behavior.text,
        behavior_identity(behavior.state.state_hash, tokens, eos, False), layout.hash, context.loss.hash,
        layout.flatten(gradient))
