"""Same-backend CPU sampling, EOS-inclusive REINFORCE and official BFCL wiring."""
from dataclasses import replace
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / 'src'):
    sys.path.insert(0, str(p))
from bfas.rtd.functional_step import FrozenStep, gradients, kl_pilot, lora_parameters, rms_diagonal, snapshot
from bfas.rtd.return_gradient import (ActionTrace, IncompleteRolloutError, TaskRollout,
    TorchPolicyBackend, bfcl_task_rollout, collect_feedback, freeze_slot_targets, gate_vjp,
    leave_one_out, loss_on_slots, reinforce_gradient, source_rollout_gradients)
from bfas.rtd.transport import Behavior, FullState, SamplingRequest, TransportSlot, sample_sources


class TinyLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_transition = torch.nn.Parameter(torch.tensor([
            [.1, -.1, .2, 1.], [.3, .4, .1, 1.1], [-.2, .3, .1, 1.2], [.1, .2, .1, 1.3]], dtype=torch.float64))
        self.register_buffer('frozen', torch.tensor(1., dtype=torch.float64))

    def forward(self, input_ids, attention_mask, use_cache=False, output_hidden_states=False):
        logits = self.lora_transition[input_ids]
        return SimpleNamespace(logits=logits, hidden_states=(logits * 2,) if output_hidden_states else None)


class TinyTokenizer:
    eos_token_id = 3
    def encode(self, prompt, add_special_tokens=False):
        return [0, 1] if prompt != 'observation' else [2, 0, 1]
    def decode(self, ids, skip_special_tokens=False):
        return ''.join(str(i) for i in ids)


def backend():
    model = TinyLM().eval()
    return TorchPolicyBackend(model, TinyTokenizer(), base_checkpoint_hash='base', harness_hash='harness',
        tokenizer_hash='tokenizer', max_action_tokens=100, max_context_tokens=200)


def test_stochastic_sampling_and_eos_scores_match_to_double_precision():
    b = backend(); params = lora_parameters(b.model)
    rng = torch.Generator().manual_seed(10)
    actions = [b.sample_action('prompt', params, rng) for _ in range(25)]
    assert len({a.action_ids for a in actions}) > 2
    for action in actions:
        score = b.score_action(action, params)
        assert score.dtype == torch.float64
        assert float(score.detach()) == pytest.approx(action.generation_logprob, abs=1e-12)
        all_ids = action.prompt_ids + action.action_ids
        expected = sum(params['lora_transition'][all_ids[i-1]].log_softmax(0)[all_ids[i]]
                       for i in range(len(action.prompt_ids), len(all_ids)))
        torch.testing.assert_close(score, expected)
    altered = snapshot(params); altered['lora_transition'].data[0, 0] += .1
    with pytest.raises(ValueError, match='mismatch'):
        b.score_action(actions[0], altered)
    with pytest.raises(ValueError, match='mismatch'):
        b.score_action(replace(actions[0], backend_id='vllm'), params)
    with pytest.raises(ValueError, match='temperature'):
        b.sample_action('prompt', params, rng, temperature=.7)


def test_no_forced_eos_capped_outcomes_and_frozen_round_sources():
    b = backend(); params = lora_parameters(b.model)
    rng = torch.Generator().manual_seed(1)
    state = FullState.create({'question': 'hi'}, [{'role': 'user', 'content': 'hi'}], 'prompt', 'inner')
    identity = b.identity(params)
    sampler = b.source_sampler(params)
    request = SamplingRequest(state, identity)
    samples = sample_sources(request, sampler, rng)
    assert len(samples) == 2 and all(s.frozen_snapshot_id == identity for s in samples)
    params['lora_transition'].data.add_(.1)
    assert sample_sources(request, sampler, rng)[0].frozen_snapshot_id == identity
    b.max_action_tokens = 1
    params['lora_transition'].data[:, 3] = -10000.
    action = b.sample_action('prompt', params, rng)
    assert action.truncated and len(action.action_ids) == 1 and 3 not in action.action_ids
    score, diagnostic = b.checked_score_action(action, params)
    assert float(score.detach()) == pytest.approx(action.generation_logprob, abs=1e-12)
    assert diagnostic['truncated'] and not diagnostic['eos']['included_in_both_scores']
    assert diagnostic['eos']['action_positions'] == []
    assert action.text == b.tokenizer.decode(action.action_ids)
    # The context's remaining room is also a deterministic cap, not forced EOS.
    b.max_action_tokens = 10
    b.max_context_tokens = 4
    assert len(b.sample_action('prompt', params, rng).action_ids) == 2
    b.max_context_tokens = 1
    with pytest.raises(IncompleteRolloutError, match='no semantic truncation'):
        b.sample_action('prompt', params, rng)


def test_capped_fake_backend_runs_official_checker_once_and_preserves_text():
    class CappedBackend(ScriptedBackend):
        def sample_action(self, *args, **kwargs):
            action = super().sample_action(*args, **kwargs)
            return replace(action, action_ids=(1, 2), truncated=True)
    entry = {'id': 'simple_python_999997', 'question': [[{'role': 'user', 'content': 'add 2 and 3'}]],
        'function': [{'name': 'add', 'description': 'sum', 'parameters': {'type': 'dict',
            'properties': {'a': {'type': 'integer'}, 'b': {'type': 'integer'}}, 'required': ['a', 'b']}}]}
    truth = [{'add': {'a': [2], 'b': [3]}}]
    call = '<tool_call>\n{"name":"add","arguments":{"a":2,"b":3}}\n</tool_call>'
    # A capped but valid call can pass: the checker, not truncation, determines R.
    for text, reward in [(call[:35], 0.), (call, 1.)]:
        b = CappedBackend([text])
        rollout = bfcl_task_rollout(entry, 'simple_python', truth, b, {}, None)
        assert rollout.truncated and rollout.reward == reward
        assert len(b.prompts) == len(rollout.actions) == 1
        assert rollout.actions[0].text == text and rollout.actions[0].action_ids == (1, 2)


def test_capped_reinforce_includes_only_sampled_tokens_and_retains_loo_denominator():
    b = backend()
    b.max_action_tokens = 2
    p = lora_parameters(b.model)
    p['lora_transition'].data[:, 3] = -10000.
    rng = torch.Generator().manual_seed(2)
    actions = [b.sample_action('prompt', p, rng) for _ in range(2)]
    rollouts = [TaskRollout('task', (a,), float(i), b.identity(p)) for i, a in enumerate(actions)]
    estimate = reinforce_gradient(rollouts, b, p)
    def sampled_score(action):
        ids = action.prompt_ids + action.action_ids
        return sum(p['lora_transition'][ids[i-1]].log_softmax(0)[ids[i]]
                   for i in range(len(action.prompt_ids), len(ids)))
    expected = gradients((sampled_score(actions[1])-sampled_score(actions[0]))/2, p)
    torch.testing.assert_close(estimate.gradient['lora_transition'], expected['lora_transition'])
    assert estimate.metadata['truncated_rollouts'] == estimate.metadata['truncated_actions'] == 2
    assert estimate.metadata['action_tokens'] == 4
    assert estimate.metadata['baselines'] == [1., 0.]
    with pytest.raises(ValueError, match='prompt/state'):
        replace(actions[0], prompt_ids=())
    with pytest.raises(ValueError, match='EOS'):
        replace(actions[0], action_ids=(1, 3))


def test_reinforce_complete_action_sums_loo_detachment_and_finite_difference():
    b = backend(); params = lora_parameters(b.model)
    rng = torch.Generator().manual_seed(7)
    trajectories = []
    for task, reward in [('a', 1.), ('a', 0.), ('b', .25), ('b', .75)]:
        actions = tuple(b.sample_action(prompt, params, rng) for prompt in ['prompt', 'observation'])
        trajectories.append(TaskRollout(task, actions, reward, b.identity(params)))
    estimate = reinforce_gradient(trajectories, b, params, score_atol=1e-12, score_rtol=1e-12)
    rewards = torch.tensor([1., 0., .25, .75], dtype=torch.float64, requires_grad=True)
    baseline = leave_one_out(['a', 'a', 'b', 'b'], rewards)
    torch.testing.assert_close(baseline, torch.tensor([0., 1., .75, .25], dtype=torch.float64))
    assert not baseline.requires_grad
    advantage = (rewards.detach() - baseline).detach()
    def objective(p):
        return sum(advantage[i] * sum(b.score_action(a, p, verify_policy=False) for a in r.actions)
                   for i, r in enumerate(trajectories)) / 4
    expected = gradients(objective(params), params)
    torch.testing.assert_close(estimate.gradient['lora_transition'], expected['lora_transition'])
    plus, minus = snapshot(params), snapshot(params)
    plus['lora_transition'].data[1, 3] += 1e-5
    minus['lora_transition'].data[1, 3] -= 1e-5
    fd = (objective(plus) - objective(minus)) / 2e-5
    assert float(fd.detach()) == pytest.approx(float(estimate.gradient['lora_transition'][1, 3]), abs=1e-10)
    assert estimate.metadata['action_tokens'] == sum(len(a.action_ids) for r in trajectories for a in r.actions)
    assert estimate.metadata['generation_backend'] == estimate.metadata['score_backend']
    assert all(not g.requires_grad for g in estimate.gradient.values()) and rewards.grad is None
    bad = replace(trajectories[0].actions[0], generation_logprob=-100.)
    corrupted = [replace(trajectories[0], actions=(bad,))] + trajectories[1:]
    with pytest.raises(ValueError, match='differs'):
        reinforce_gradient(corrupted, b, params)
    with pytest.raises(ValueError, match='two independent'):
        leave_one_out(['a', 'b'], torch.ones(2))


class ScriptedBackend:
    """Harness-only fixture. Native stochastic probability checks are above."""
    def __init__(self, outputs):
        self.outputs, self.prompts = iter(outputs), []
    def identity(self, parameters):
        return 'toy-policy'
    def sample_action(self, prompt, parameters, generator, **settings):
        assert settings == {'temperature': 1., 'top_p': 1.}
        self.prompts.append(prompt)
        return ActionTrace((0,), (1, 3), 3, next(self.outputs), -.5, 'scripted', 'toy-policy')


def test_single_turn_complete_generation_uses_official_ast_checker():
    call = '<tool_call>\n{"name":"add","arguments":{"a":2,"b":3}}\n</tool_call>'
    entry = {'id': 'simple_python_999999', 'question': [[{'role': 'user', 'content': 'add 2 and 3'}]],
        'function': [{'name': 'add', 'description': 'sum', 'parameters': {'type': 'dict',
            'properties': {'a': {'type': 'integer'}, 'b': {'type': 'integer'}}, 'required': ['a', 'b']}}]}
    truth = [{'add': {'a': [2], 'b': [3]}}]
    for output, expected in [(call, 1.), (call.replace('"b":3', '"b":4'), 0.)]:
        b = ScriptedBackend([output])
        r = bfcl_task_rollout(entry, 'simple_python', truth, b, {}, None)
        assert r.reward == expected and len(r.actions) == 1 and r.from_task_start
    class BrokenChecker:
        def check(self, *args):
            raise RuntimeError('infrastructure failure')
    with pytest.raises(RuntimeError, match='infrastructure'):
        bfcl_task_rollout(entry, 'simple_python', truth, ScriptedBackend([call]), {}, None, checker=BrokenChecker())


def test_multi_turn_reuses_official_environment_and_terminal_evaluation_from_start():
    entry = {'id': 'multi_turn_base_999999', 'question': [
        [{'role': 'user', 'content': 'Add 2 and 3.'}], [{'role': 'user', 'content': 'Multiply 4 and 5.'}]],
        'initial_config': {}, 'involved_classes': ['MathAPI']}
    outputs = ['<tool_call>\n{"name":"add","arguments":{"a":2,"b":3}}\n</tool_call>', 'Done.',
        '<tool_call>\n{"name":"multiply","arguments":{"a":4,"b":5}}\n</tool_call>', 'Done.']
    original = __import__('copy').deepcopy(entry)
    for _ in range(2):
        b = ScriptedBackend(outputs)
        result = bfcl_task_rollout(entry, 'multi_turn_base', [['add(a=2, b=3)'], ['multiply(a=4, b=5)']], b, {}, None)
        assert result.reward == 1. and len(result.actions) == 4
        assert '<tool_response>' not in b.prompts[0] and '<tool_response>' in b.prompts[1]
        assert 'Multiply 4 and 5.' not in b.prompts[0] and 'Multiply 4 and 5.' in b.prompts[2]
        assert entry == original  # no teacher prefix or state carried across rollouts


def test_feedback_fold_validation_and_all_zero_rewards():
    b = backend(); params = lora_parameters(b.model); rng = torch.Generator().manual_seed(3)
    def rollout(task_id):
        return TaskRollout(task_id, (b.sample_action('prompt', params, rng),), 0., b.identity(params))
    rs = collect_feedback([('task', 'feedback')], rollout, feedback_parent_hashes={'feedback'})
    estimate = reinforce_gradient(rs, b, params)
    assert not estimate.metadata['identifiable'] and all(g.eq(0).all() for g in estimate.gradient.values())
    with pytest.raises(ValueError, match='illegal'):
        collect_feedback([('task', 'inner')], rollout, feedback_parent_hashes={'feedback'})


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_low_precision_model_still_scores_the_same_float32_categorical_policy(dtype):
    b = backend(); b.model.to(dtype)
    params = lora_parameters(b.model)
    a = b.sample_action('prompt', params, torch.Generator().manual_seed(8))
    score = b.score_action(a, params)
    assert score.dtype == torch.float32
    assert float(score.detach()) == pytest.approx(a.generation_logprob, abs=2e-6)


def test_native_source_rms_pilot_slot_loss_and_initial_hidden_integration():
    b = backend(); params = lora_parameters(b.model); rng = torch.Generator().manual_seed(9)
    state = FullState.create({'question': 'hi'}, [{'role': 'user', 'content': 'hi'}], 'prompt', 'inner')
    initial_id = b.identity(params)
    hidden = b.initial_hidden('prompt', params, initial_snapshot_id=initial_id)
    assert hidden.shape == (4,) and not hidden.requires_grad
    sources = sample_sources(SamplingRequest(state, initial_id), b.source_sampler(params), rng)
    for source in sources:
        assert float(b.score_source(source, params).detach()) == pytest.approx(source.logprob, abs=1e-12)
    targets = freeze_slot_targets([TransportSlot(s, (Behavior(state, 'observation'),)) for s in sources], rng)
    chi = torch.tensor([[1., .2], [-.4, 1.]], dtype=torch.float64, requires_grad=True)
    phi = torch.zeros(2, dtype=torch.float64, requires_grad=True)
    loss = lambda p: loss_on_slots(targets, chi, phi, b, p)
    actions = [b.sample_action('prompt', params, rng) for _ in range(2)]
    rollouts = [TaskRollout('source_task', (a,), 0., initial_id) for a in actions]
    gs = source_rollout_gradients(rollouts, b, params, parent_by_task={'source_task': 'inner'}, inner_parent_hashes={'inner'})
    P, audit = rms_diagonal(params, gs, inner_parent_hashes={'inner'}, source_snapshot_id=initial_id)
    eta, pilot = kl_pilot(params, P, lambda alpha: loss(params),
        lambda updated: b.source_kl(actions, params, updated), candidates=[.01, .1, .5],
        owned_ids={'paid'}, evidence_ids={'paid'}, inner_parent_hashes={'inner'},
        evidence_parents={'inner'}, source_parents={'inner'})
    assert audit['source_rollouts'] == 2 and pilot['committed_updates'] == 0
    step = FrozenStep(P, eta, 'r1', audit)
    gJ = {n: torch.ones_like(p) for n, p in params.items()}
    vjp = gate_vjp(loss(params), params, phi, step, gJ)
    assert torch.isfinite(vjp).all() and chi.grad is None
    assert b.identity(params) == initial_id


def test_stateful_simulator_and_official_relevance_paths_are_reset():
    entry = {'id': 'multi_turn_base_999998', 'question': [[{'role': 'user', 'content': 'Create folder fresh.'}]],
             'initial_config': {}, 'involved_classes': ['GorillaFileSystem']}
    outputs = ['<tool_call>\n{"name":"mkdir","arguments":{"dir_name":"fresh"}}\n</tool_call>', 'Done.']
    from tools.behavior_atom.checker_bridge import CheckerBridge
    # Reuse a persistent worker as the runner does: both generation and checker
    # must start from fresh filesystem state on the second same-task rollout.
    with CheckerBridge() as checker:
        for _ in range(2):
            b = ScriptedBackend(outputs)
            r = bfcl_task_rollout(entry, 'multi_turn_base', [["mkdir(dir_name='fresh')"]], b, {}, None, checker=checker)
            assert r.reward == 1 and 'File exists' not in b.prompts[1]
        from bfcl_eval.eval_checker.multi_turn_eval import multi_turn_utils
        assert not any('_rtd_' in k and k.endswith('_instance') for k in vars(multi_turn_utils))
        single = {'id': 'live_relevance_999998', 'question': [[{'role': 'user', 'content': 'Do something.'}]],
                  'function': []}
        for category, text, expected in [('live_relevance', outputs[0], 1.), ('live_relevance', 'No.', 0.),
                                          ('live_irrelevance', 'No.', 1.)]:
            single['id'] = category + '_999998'
            r = bfcl_task_rollout(single, category, None, ScriptedBackend([text]), {}, None, checker=checker)
            assert r.reward == expected
