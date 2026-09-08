"""Full-vocabulary source gradients with one position batch of vocabulary data.

Two teacher-forced backbone forwards per source slot. Capture the input of the
LM output head, before allocating sequence x vocabulary logits. Head VJPs run
on position batches; only hidden-state cotangents survive between batches.
"""
from contextlib import nullcontext

import torch
from torch.func import functional_call

from .functional_step import _matching, lora_parameters
from .transport import validate_sampled_action


class _HeadInput(Exception):
    def __init__(self, hidden):
        self.hidden = hidden


def sampled_prefix_positions(source, prompt_ids):
    """Shared hard/soft positions, including sampled EOS and never adding cap EOS."""
    if not prompt_ids:
        raise ValueError('complete prompt/state required')
    validate_sampled_action(source.token_ids, source.eos_token_id, source.truncated)
    return tuple(range(len(prompt_ids)-1, len(prompt_ids)-1+len(source.token_ids)))


def _hidden_at_head(backend, head, parameters, ids):
    def capture(module, args):
        raise _HeadInput(args[0])
    hook = head.register_forward_pre_hook(capture)
    try:
        backend._logits(parameters, ids)
    except _HeadInput as captured:
        return captured.hidden
    finally:
        hook.remove()
    raise ValueError('teacher-forced model did not call its output embedding head')


def _position_vjps(head, current_head, frozen_head, hidden, frozen_hidden, labels):
    """All vocabulary-sized temporaries die on return from this function."""
    leaf = hidden.detach().requires_grad_(True)
    with torch.no_grad():
        old = functional_call(head, frozen_head, (frozen_hidden,))
        dtype = torch.float64 if old.dtype == torch.float64 else torch.float32
        logp = old.to(dtype).log_softmax(-1)
    new = functional_call(head, current_head, (leaf,))
    logq = new.to(dtype).log_softmax(-1)
    hard = -logq.gather(-1, labels[:, None]).sum()
    # Analytic CE/KL logit derivative q-p. Integrate this cotangent through the
    # head/backbone, without differentiating a sampled label or storing a graph
    # of source probabilities. The residual is exactly zero at equal policies.
    residual = (logq.exp() - logp.exp()).detach()
    soft_surrogate = (new.to(dtype) * residual).sum()
    inputs = (leaf, *current_head.values())
    gh = torch.autograd.grad(hard, inputs, allow_unused=True, retain_graph=True)
    gs = torch.autograd.grad(soft_surrogate, inputs, allow_unused=True)
    # KL is for reporting only; direct form avoids exp(delta) overflow.
    kl = float((logp.exp() * (logp-logq.detach())).sum())
    return tuple(None if g is None else g.detach() for g in gh), tuple(
        None if g is None else g.detach() for g in gs), kl


def source_gradient_pair(backend, source, parameters, source_parameters, *, position_batch_size=1):
    """Return detached (G_hard, G_soft, metadata) on the exact sampled prefixes.

    Backends must expose a standard, positionwise ``get_output_embeddings()``
    head. Qwen/PEFT use this path without materializing full-sequence logits.
    No candidate normalization, EOS insertion, or prompt truncation occurs.
    Frozen base weights/buffers are shared; frozen LoRA coordinates are bound
    by functional_call. Gradient checkpoint replays retain that binding.
    """
    _matching(lora_parameters(backend.model), parameters)
    _matching(parameters, source_parameters)
    if source.frozen_snapshot_id != backend.identity(source_parameters):
        raise ValueError('soft source snapshot mismatch')
    if type(position_batch_size) is not int or position_batch_size < 1:
        raise ValueError('positive position batch size required')
    if not hasattr(backend.model, 'get_output_embeddings'):
        raise ValueError('soft/CV scoring requires get_output_embeddings for streamed vocabulary projection')
    head = backend.model.get_output_embeddings()
    if head is None:
        raise ValueError('soft/CV scoring requires a positionwise output head')
    # Models with post-head nonlinear logit transforms need an explicit adapter.
    config = getattr(backend.model, 'config', None)
    if getattr(config, 'final_logit_softcapping', None):
        raise ValueError('post-head logit softcapping requires a source-scoring adapter')
    head_name = next(n for n, module in backend.model.named_modules() if module is head)
    prefix = head_name + '.' if head_name else ''
    head_names = {n: prefix+n for n, _ in head.named_parameters() if prefix+n in parameters}
    current_head = {n: parameters[full] for n, full in head_names.items()}
    frozen_head = {n: source_parameters[full].detach() for n, full in head_names.items()}
    prompt = tuple(backend.tokenizer.encode(source.behavior.state.prompt, add_special_tokens=False))
    positions = sampled_prefix_positions(source, prompt)
    ids = torch.tensor([prompt + source.token_ids], device=next(iter(parameters.values())).device)
    scope = backend.measured('source_teacher_forced_pair', forward_passes=2,
        prompt_tokens=2*len(prompt), action_tokens=2*source.length) if hasattr(backend, 'measured') else nullcontext()
    with scope:
        with torch.no_grad():
            frozen_hidden = _hidden_at_head(backend, head, source_parameters, ids).detach()
        current_hidden = _hidden_at_head(backend, head, parameters, ids)
        hard_cotangent, soft_cotangent = torch.zeros_like(current_hidden), torch.zeros_like(current_hidden)
        gh = {n: torch.zeros_like(p) for n, p in parameters.items()}
        gs = {n: torch.zeros_like(p) for n, p in parameters.items()}
        kl = 0.
        for start in range(0, source.length, position_batch_size):
            end = min(start + position_batch_size, source.length)
            sl = slice(positions[start], positions[end-1]+1)
            hard, soft, value = _position_vjps(head, current_head, frozen_head,
                current_hidden[0, sl], frozen_hidden[0, sl], ids[0, len(prompt)+start:len(prompt)+end])
            hard_cotangent[0, sl] = hard[0]
            soft_cotangent[0, sl] = soft[0]
            for i, full in enumerate(head_names.values(), 1):
                if hard[i] is not None:
                    gh[full].add_(hard[i])
                if soft[i] is not None:
                    gs[full].add_(soft[i])
            kl += value
        if current_hidden.requires_grad:
            for result, cotangent, retain in ((gh, hard_cotangent, True), (gs, soft_cotangent, False)):
                values = torch.autograd.grad(current_hidden, tuple(parameters.values()), grad_outputs=cotangent,
                                             allow_unused=True, retain_graph=retain)
                for name, value in zip(parameters, values):
                    if value is not None:
                        result[name].add_(value.detach())
    return gh, gs, dict(source_kl=kl, action_positions=source.length,
                        position_batch_size=position_batch_size, backbone_forwards=2)
