"""Decoder checkpointing for eval-mode, functional LoRA scoring.

HF's training-only checkpoint switch is insufficient here: RTD stays in eval
mode, and functional_call restores the resident parameters before backward.
Each replay must therefore explicitly bind the original scoring snapshot.
"""
from functools import wraps

import torch
from torch.func import functional_call
from torch.utils.checkpoint import checkpoint


class _LayerForward(torch.nn.Module):
    """Use public functional_call while bypassing the installed forward wrapper.

    This proxy lives in the wrapper closure, outside the model's module tree;
    adapter names, state_dicts, and generation are unchanged.
    """
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
        self.original_forward = layer.forward

    def forward(self, *args, **kwargs):
        return self.original_forward(*args, **kwargs)


def checkpoint_layer(layer):
    if getattr(layer.forward, '_rtd_checkpoint', False):
        return
    proxy = _LayerForward(layer)

    @wraps(proxy.original_forward)
    def forward(*args, **kwargs):
        if not torch.is_grad_enabled():
            return proxy.original_forward(*args, **kwargs)
        if kwargs.get('use_cache') or any(kwargs.get(k) is not None for k in
                                         ('past_key_values', 'past_key_value', 'cache_params')):
            raise ValueError('RTD checkpointed backward requires use_cache=False and no KV state')
        # These are the tensors installed by the enclosing functional_call,
        # not the resident student's weights. Only LoRA coordinates are kept.
        parameters = {f'layer.{n}': p for n, p in layer.named_parameters() if p.requires_grad}
        if any('lora_' not in n.lower() for n in parameters):
            raise ValueError('RTD checkpointing requires frozen non-LoRA weights')

        def replay(parameters, *args, **kwargs):
            return functional_call(proxy, parameters, args, kwargs)

        # Non-reentrant checkpointing supports autograd.grad and frozen
        # embeddings; no input-requires-grad hook or training-mode toggle.
        return checkpoint(replay, parameters, *args, use_reentrant=False,
                          preserve_rng_state=True, **kwargs)

    forward._rtd_checkpoint = True
    layer.forward = forward


def enable_gradient_checkpointing(model):
    from transformers.modeling_layers import GradientCheckpointingLayer

    layers = [m for m in model.modules() if isinstance(m, GradientCheckpointingLayer)]
    if not layers:
        raise ValueError('RTD requires decoder layers supporting gradient checkpointing')
    for layer in layers:
        checkpoint_layer(layer)
    return len(layers)
