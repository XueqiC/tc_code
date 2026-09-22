"""Full-vocabulary source gradients with one position batch of vocabulary data.

Two teacher-forced backbone forwards per source slot. Capture the input of the
LM output head, before allocating sequence x vocabulary logits. Head VJPs run
on position batches; only hidden-state cotangents survive between batches.
"""
from contextlib import nullcontext

import torch
from torch.func import functional_call
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .functional_step import _matching, lora_parameters
from .transport import validate_sampled_action


class _HeadInput(Exception):
    def __init__(self, hidden):
        self.hidden = hidden


def require_device(device, **tensors):
    """Fail before a projection/loss can silently run on an offloaded tensor."""
    for name, tensor in tensors.items():
        if tensor.device != device:
            raise ValueError(f'{name} device {tensor.device} differs from scoring device {device}')


def generated_token_scores(scores, ids, *, device, position_chunk_size=32):
    """Reduce HF generation scores on device; copy only chosen-token scalars."""
    if len(scores) != len(ids) or not ids or position_chunk_size < 1:
        raise ValueError('aligned generation scores and positive chunk size required')
    values = []
    with torch.no_grad():
        for start in range(0, len(ids), position_chunk_size):
            end = start+position_chunk_size
            logits = torch.cat(scores[start:end], dim=0)
            require_device(device, generation_logits=logits)
            labels = torch.tensor(ids[start:end], device=device)
            dtype = torch.float64 if logits.dtype == torch.float64 else torch.float32
            values.append(-F.cross_entropy(logits.to(dtype), labels, reduction='none'))
    return tuple(torch.cat(values).tolist())


def teacher_forcing_inputs(prompt_ids, action_ids, *, device):
    """One complete turn per forward: no padding, collation, or truncation."""
    return dict(input_ids=torch.tensor([tuple(prompt_ids) + tuple(action_ids)], device=device),
                labels=torch.tensor([(-100,) * len(prompt_ids) + tuple(action_ids)], device=device))


def chunked_token_scores(backend, prompt_ids, action_ids, parameters, *, eos_token_id,
                         truncated=False, position_chunk_size=32):
    """Native hard-label CE with bounded vocabulary tensors, also under backward.

    Capture the head input once, then project only action prediction positions.
    Non-reentrant checkpoints retain hidden states/labels, never every chunk's
    vocabulary-sized CE graph. The base-equivalent SmartAD reference uses this
    same path under no_grad; reference logits never leave the model device.
    """
    if type(position_chunk_size) is not int or position_chunk_size < 1:
        raise ValueError('positive position chunk size required')
    if not prompt_ids:
        raise ValueError('complete prompt/state required')
    validate_sampled_action(action_ids, eos_token_id, truncated)
    _matching(lora_parameters(backend.model), parameters)
    device = next(iter(parameters.values())).device
    head = backend.model.get_output_embeddings()
    if head is None:
        raise ValueError('chunked scoring requires a positionwise output head')
    prefix = next(n for n, module in backend.model.named_modules() if module is head)
    prefix = prefix + '.' if prefix else ''
    bound = {n[len(prefix):]: p for n, p in parameters.items() if n.startswith(prefix)}
    inputs = teacher_forcing_inputs(prompt_ids, action_ids, device=device)
    ids = inputs["input_ids"]
    hidden = _hidden_at_head(backend, head, parameters, ids)
    require_device(device, hidden=hidden, input_ids=ids, **dict(head.named_parameters()))
    softcap = logit_softcap(backend.model)
    logits_dtype = None

    def score_chunk(h, labels, head_parameters):
        nonlocal logits_dtype
        require_device(device, hidden=h, labels=labels, **head_parameters)
        logits = cap_logits(functional_call(head, head_parameters, (h,)), softcap)
        require_device(device, logits=logits)
        logits_dtype = str(logits.dtype)
        dtype = torch.float64 if logits.dtype == torch.float64 else torch.float32
        return -F.cross_entropy(logits.to(dtype), labels, reduction='none')

    chunks = []
    for start in range(0, len(action_ids), position_chunk_size):
        end = min(start+position_chunk_size, len(action_ids))
        h = hidden[0, len(prompt_ids)-1+start:len(prompt_ids)-1+end]
        labels = inputs["labels"][0, len(prompt_ids)+start:len(prompt_ids)+end]
        values = (checkpoint(score_chunk, h, labels, bound, use_reentrant=False)
                  if torch.is_grad_enabled() else score_chunk(h, labels, bound))
        require_device(device, token_logprobs=values)
        chunks.append(values)
    values = torch.cat(chunks)
    score = values.sum()
    from .scoring import attention_implementation
    return score, values, dict(implementation='torch-functional-chunked-native-ce-v1',
        use_cache=False, cache_type=None, attention=attention_implementation(backend.model),
        logits_dtype=logits_dtype, logprob_dtype=str(values.dtype), reduction_dtype=str(score.dtype),
        parameter_dtypes=sorted({str(p.dtype) for p in parameters.values()}),
        model_class=type(backend.model).__name__, torch_version=torch.__version__,
        position_chunk_size=position_chunk_size, logical_device=str(device))


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


def _position_vjps(head, current_head, frozen_head, hidden, frozen_hidden, labels, *, softcap=None):
    """All vocabulary-sized temporaries die on return from this function."""
    device = hidden.device
    require_device(device, reference_hidden=frozen_hidden, labels=labels,
                   **{**dict(head.named_parameters()), **current_head})
    require_device(device, **frozen_head)
    leaf = hidden.detach().requires_grad_(True)
    with torch.no_grad():
        old = functional_call(head, frozen_head, (frozen_hidden,))
        require_device(device, reference_logits=old)
        old = cap_logits(old, softcap)
        dtype = torch.float64 if old.dtype == torch.float64 else torch.float32
        logp = old.to(dtype).log_softmax(-1)
    new = cap_logits(functional_call(head, current_head, (leaf,)), softcap)
    require_device(device, student_logits=new)
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
    softcap = logit_softcap(backend.model)
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
                current_hidden[0, sl], frozen_hidden[0, sl], ids[0, len(prompt)+start:len(prompt)+end], softcap=softcap)
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


def logit_softcap(model):
    config = getattr(model, 'config', None)
    config = config.get_text_config() if hasattr(config, 'get_text_config') else config
    return getattr(config, 'final_logit_softcapping', None)


def cap_logits(logits, softcap):
    # Same operations and dtype as the model's native forward.
    return (logits / softcap).tanh() * softcap if softcap else logits
