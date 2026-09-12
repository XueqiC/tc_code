from dataclasses import replace
import copy
import random

import numpy as np
import pytest
import torch
import yaml

from bfas.rtd.functional_step import gradients, lora_parameters
from bfas.rtd.return_gradient import TaskRollout
from bfas.rtd.transport import FullState, Behavior
from bfas.rtd.unified import (Array, Parameters, LossDefinition, TeachingContext, build_teaching_problem,
    TeachingObjective, commit_distillation, trial_distillation, score_sources, score_teacher,
    FeedbackCheckpoint, FeedbackSplit, collect_task_feedback, attach_feedback, solve_exact,
    UnifiedConfig, UnifiedEngine, validate_and_report)
from test_rtd_return_gradient import backend


def actual_problem(normalization='per_sequence_mean'):
    b = backend()
    b.max_action_tokens = 3
    theta = Parameters.of(lora_parameters(b.model))
    context = TeachingContext(theta, theta, b.identity(theta.tensors()), Array.of(np.linspace(.5, 1.5, 16)),
        loss=LossDefinition(normalization, retention_scale=1/3), eta=.03,
        owned_query_ids=('paid',), inner_parent_hashes=('inner',))
    state = FullState.create({'task': 'toy'}, [{'role': 'user', 'content': 'toy'}], 'prompt', 'inner')
    exposure = score_sources(b, state, context, torch.Generator().manual_seed(12), slot_id='slot', weight=1.)
    evidence = score_teacher(b, Behavior(state, 'teacher'), context, query_id='paid', version='T1')
    return b, build_teaching_problem(context, [evidence], [exposure])


@pytest.mark.parametrize('normalization', ['total_token_nll', 'per_sequence_mean'])
def test_frozen_P_affine_update_matches_independent_gradient_commit_and_trial(normalization):
    b, problem = actual_problem(normalization)
    a = [.3, .8]
    slot, teacher = problem.exposure[0], problem.evidence[0]
    # Independent estimator accumulation, not just two calls to the map.
    raw_gradient = slot.soft.numpy()+sum(a[j]*(teacher.gradient.numpy()-slot.hard.numpy()[j])/2 for j in range(2))
    direct = -problem.context.eta*problem.context.preconditioner.numpy()*raw_gradient
    predicted = TeachingObjective(problem).increment(a)
    np.testing.assert_allclose(direct, predicted, atol=1e-17)
    start = Parameters.of(lora_parameters(b.model)).hash
    trial, _ = trial_distillation(problem, a, model=b.model)
    assert Parameters.of(lora_parameters(b.model)).hash == start
    commit = commit_distillation(problem, a, model=b.model)
    assert trial.parameters.hash == commit.parameters.hash
    np.testing.assert_array_equal(trial.actual_increment.numpy(), commit.actual_increment.numpy())
    assert commit.increment_error < 1e-15 and commit.rl_updates == 0 and commit.backbone_updates == 1
    with pytest.raises(ValueError, match='saved common start'):
        commit_distillation(problem, a, model=b.model)


@pytest.mark.parametrize('raises', [False, True])
def test_trial_restores_optimizer_rng_buffers_grads_modes_and_caller_state(raises):
    b, problem = actual_problem()
    optimizer = torch.optim.AdamW(b.model.parameters(), lr=.01)
    for p in b.model.parameters():
        p.grad = torch.ones_like(p)
    optimizer.step()  # initialize a real stateful optimizer before snapshot
    # Restore theta, keeping populated optimizer state for the preservation check.
    with torch.no_grad():
        for n, p in lora_parameters(b.model).items():
            p.copy_(problem.context.theta.tensors()[n])
    before = copy.deepcopy(optimizer.state_dict())
    initial_model = copy.deepcopy(b.model.state_dict())
    initial_grad = b.model.lora_transition.grad.clone()
    torch_rng = torch.Generator().manual_seed(91)
    numpy_rng = np.random.default_rng(29)
    rng_before = torch_rng.get_state().clone(), copy.deepcopy(numpy_rng.bit_generator.state)
    global_before = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    state = {'round': 1, 'nested': {'feedback': []}}
    def mutate(_):
        random.random(); np.random.rand(); torch.rand(2)
        torch.rand(2, generator=torch_rng); numpy_rng.random()
        state['nested']['feedback'].append(1)
        b.model.train()
        b.model.frozen.add_(5)
        b.model.lora_transition.grad.zero_()
        optimizer.param_groups[0]['lr'] = .9
        optimizer.state[b.model.lora_transition]['exp_avg'].add_(1)
        if raises:
            raise RuntimeError('test failure during feedback')
    def run():
        return trial_distillation(problem, [.4, .7], model=b.model, optimizer=optimizer,
            generators=(torch_rng, numpy_rng), state=state, evaluate=mutate)
    if raises:
        with pytest.raises(RuntimeError, match='test failure'):
            run()
    else:
        run()
    assert state == {'round': 1, 'nested': {'feedback': []}} and not b.model.training
    for name, value in b.model.state_dict().items():
        torch.testing.assert_close(value, initial_model[name], atol=0, rtol=0)
    torch.testing.assert_close(b.model.lora_transition.grad, initial_grad, atol=0, rtol=0)
    assert optimizer.state_dict()['param_groups'] == before['param_groups']
    for index, values in before['state'].items():
        for name, value in values.items():
            torch.testing.assert_close(optimizer.state_dict()['state'][index][name], value, atol=0, rtol=0)
    torch.testing.assert_close(torch_rng.get_state(), rng_before[0], atol=0, rtol=0)
    assert numpy_rng.bit_generator.state == rng_before[1]
    assert random.getstate() == global_before[0]
    np.testing.assert_equal(np.random.get_state(), global_before[1])
    torch.testing.assert_close(torch.get_rng_state(), global_before[2], atol=0, rtol=0)


def test_feedback_matches_action_only_loo_gradient_and_task_weights():
    b, problem = actual_problem('total_token_nll')
    parameters = problem.reference_parameters
    rng = torch.Generator().manual_seed(37)
    captured = []
    counts = {'a': 0, 'b': 0}
    def rollout(task, p):
        actions = tuple(b.sample_action(prompt, p, rng) for prompt in ('prompt', 'observation'))
        reward = float(counts[task] % 2)
        counts[task] += 1
        row = TaskRollout(task, actions, reward, b.identity(p))
        captured.append(row)
        return row
    split = FeedbackSplit((('a', 'feedback'), ('b', 'feedback')), frozenset({'feedback'}),
        frozenset({'inner'}), task_weights=(.2, .8), rollouts_per_task=4)
    feedback = collect_task_feedback(FeedbackCheckpoint(parameters, b, rollout, 'feedback-1'), split)
    assert Parameters.of(lora_parameters(b.model)).hash == problem.theta_hash  # no RL update
    p = parameters.tensors()
    objective = 0.
    for row in captured:
        others = [r.reward for r in captured if r.task_id == row.task_id and r is not row]
        advantage = row.reward-np.mean(others)
        # Complete observations appear in prompt_ids but are never score terms.
        score = sum(b.score_action(a, p) for a in row.actions)
        objective += (.2 if row.task_id == 'a' else .8)*advantage*score/4
    expected = parameters.flatten(gradients(objective, p)).numpy()
    np.testing.assert_allclose(feedback.gradient.numpy(), expected, atol=1e-15)
    assert feedback.action_tokens == sum(len(a.action_ids) for r in captured for a in r.actions)
    problem = attach_feedback(problem, feedback)
    assert (problem.epsilon.numpy() >= 0).all() and np.isfinite(problem.epsilon.numpy()).all()
    assert problem.uncertainty_label == 'uncertainty regulariser'
    assert feedback.baseline == 'leave_one_out_same_task'
    with pytest.raises(ValueError, match='reference checkpoint'):
        attach_feedback(problem, replace(feedback, parameter_hash='wrong'))
    with pytest.raises(ValueError, match='disjoint'):
        collect_task_feedback(FeedbackCheckpoint(parameters, b, rollout, 'feedback-2'),
                              replace(split, inner_parents=frozenset({'feedback'})))


def test_config_selection_and_integrated_feedback_solve_one_commit(monkeypatch):
    from pathlib import Path
    config = yaml.safe_load(Path('configs/rtd/unified_bfcl_gemma4.yaml').read_text())
    selected = UnifiedConfig.from_config(config)
    assert selected.student == 'google/gemma-4-12B-it' and selected.m == 2 and selected.a_ref == .5
    with pytest.raises(ValueError, match='v1.1 is separate'):
        UnifiedConfig.from_config(yaml.safe_load(Path('configs/rtd/v1_1_bfcl.yaml').read_text()))
    b, problem = actual_problem()
    config['unified']['loss_definition']['retention_scale'] = 1/3
    engine = UnifiedEngine(config)
    rng = torch.Generator().manual_seed(17)
    optimizer = torch.optim.AdamW(b.model.parameters())
    def forbidden_step(*args, **kwargs):
        raise AssertionError('extra optimizer/RL step')
    monkeypatch.setattr(optimizer, 'step', forbidden_step)
    def rollout(task, p):
        action = b.sample_action('prompt', p, rng)
        return TaskRollout(task, (action,), float(action.action_ids[0] == 3), b.identity(p))
    split = FeedbackSplit((('task', 'feedback'),), frozenset({'feedback'}), frozenset({'inner'}))
    run = engine.run_step(problem, backend=b, rollout=rollout, feedback_split=split,
                          feedback_version='feedback-1', optimizer=optimizer, generators=(rng,))
    assert run['report']['rl_updates'] == 0 and run['report']['backbone_updates'] == 1
    assert run['commit'].parameters.hash == Parameters.of(lora_parameters(b.model)).hash
    assert run['report']['feedback_version'] == 'feedback-1'
    assert optimizer.state_dict()['state'] == {}
    assert validate_and_report(run) == run['report']


def test_raw_path_uses_same_frozen_update_map():
    b, cv = actual_problem()
    raw = build_teaching_problem(replace(cv.context, estimator='raw'), cv.evidence, cv.exposure)
    a = [.1, .9]
    p, e, t = raw.context.preconditioner.numpy(), raw.exposure[0], raw.evidence[0]
    expected = sum(((1-a[j])*e.hard.numpy()[j]+a[j]*t.gradient.numpy())/2 for j in range(2))
    step = commit_distillation(raw, a, model=b.model)
    np.testing.assert_allclose(step.actual_increment.numpy(), -raw.context.eta*p*expected, atol=1e-15)


def test_independent_loo_baseline_is_unbiased_self_reward_baseline_is_not():
    # Enumerate two independent Bernoulli(.5) task trajectories, reward=action.
    # Score gradient is action-.5; true d E[R]/d logit = .25.
    from itertools import product
    loo_mean = self_baseline_mean = 0.
    for first, second in product([0., 1.], repeat=2):
        loo_mean += .25*((first-second)*(first-.5)+(second-first)*(second-.5))/2
        self_baseline_mean += .25*((first-first)*(first-.5)+(second-second)*(second-.5))/2
    assert loo_mean == .25 and self_baseline_mean == 0.
