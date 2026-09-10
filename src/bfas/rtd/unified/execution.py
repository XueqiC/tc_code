"""One frozen-P distillation commit; trials restore all caller-owned state."""
from contextlib import contextmanager
from dataclasses import dataclass
import copy
import random

import numpy as np
import torch

from ..functional_step import commit_step, lora_parameters
from .immutable import Array, Parameters
from .objective import TeachingObjective


@contextmanager
def restored_state(model, *, optimizer=None, generators=(), state=None):
    weights = copy.deepcopy(model.state_dict())
    grads = {n: None if p.grad is None else p.grad.detach().clone() for n, p in model.named_parameters()}
    requires_grad = {n: p.requires_grad for n, p in model.named_parameters()}
    modes = [(module, module.training) for module in model.modules()]
    optimizer_state = None if optimizer is None else copy.deepcopy(optimizer.state_dict())
    saved_state = None if state is None else copy.deepcopy(state)
    python_rng, numpy_rng, torch_rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
    # Do not initialize CUDA merely to capture state in CPU-only P0.
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    explicit = []
    for rng in generators:
        if isinstance(rng, torch.Generator):
            explicit.append((rng, rng.get_state().clone()))
        elif isinstance(rng, np.random.Generator):
            explicit.append((rng, copy.deepcopy(rng.bit_generator.state)))
        else:
            raise TypeError('only torch/numpy explicit generators supported')
    try:
        yield
    finally:
        model.load_state_dict(weights)
        for n, p in model.named_parameters():
            p.grad = grads[n]
            p.requires_grad_(requires_grad[n])
        for module, mode in modes:
            module.training = mode
        if optimizer is not None:
            optimizer.load_state_dict(optimizer_state)
        if state is not None:
            state.clear()
            state.update(saved_state)
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        torch.set_rng_state(torch_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
        for rng, value in explicit:
            if isinstance(rng, torch.Generator):
                rng.set_state(value)
            else:
                rng.bit_generator.state = value


@dataclass(frozen=True)
class DistillationStep:
    problem_id: str
    coefficients: Array
    parameters: Parameters
    predicted_increment: Array
    actual_increment: Array
    increment_error: float
    objective_value: float
    committed: bool
    backbone_updates: int
    rl_updates: int = 0


def commit_distillation(problem, coefficients, *, model=None):
    objective = TeachingObjective(problem)
    coefficients = problem.check_coefficients(coefficients)
    updated = objective.parameters(coefficients)
    predicted = objective.increment(coefficients)
    actual = updated.values.numpy()-problem.context.theta.values.numpy()
    if model is not None:
        device = next(iter(lora_parameters(model).values())).device
        commit_step(model, updated.tensors(device=device), expected_start_hash=problem.theta_hash)
    return DistillationStep(problem.identity, Array.of(coefficients), updated, Array.of(predicted),
        Array.of(actual), float(np.max(np.abs(actual-predicted), initial=0)),
        objective.value(coefficients), model is not None, int(model is not None))


def trial_distillation(problem, coefficients, *, model, evaluate=None, optimizer=None, generators=(), state=None):
    with restored_state(model, optimizer=optimizer, generators=generators, state=state):
        step = commit_distillation(problem, coefficients, model=model)
        result = evaluate(step.parameters) if evaluate else None
    return step, result
