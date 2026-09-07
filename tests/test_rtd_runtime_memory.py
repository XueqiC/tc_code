"""C25c: CPU-only device boundary, activation lifetime, and gradient oracles."""
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT)]

from bfas.behavior.deltas import tensor_state_hash
from bfas.rtd import cli
from bfas.rtd.checkpointing import checkpoint_layer, enable_gradient_checkpointing
from bfas.rtd.functional_step import FrozenStep, gradients, lora_parameters, rms_diagonal, snapshot
from bfas.rtd.persistence import ComputeJournal
from bfas.rtd.return_gradient import ReturnGradient, TorchPolicyBackend, gate_vjp
from bfas.rtd.runtime import slot_loss, streamed_gate_vjp, streamed_gradient
from bfas.rtd.transport import Behavior, FullState, SourceSample


@pytest.mark.parametrize('visible', ['2', 'GPU-20b20454-ae9f-7860-6801-430d68842a27', '2,0', '', None])
def test_training_worker_command_preserves_parent_environment(tmp_path, monkeypatch, visible):
    # No CUDA call or subprocess launch: test the real coordinator boundary.
    from bfas.rtd import evaluation
    if visible is None:
        monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    else:
        monkeypatch.setenv('CUDA_VISIBLE_DEVICES', visible)
    monkeypatch.setenv('CUDA_DEVICE_ORDER', 'PCI_BUS_ID')
    monkeypatch.setenv('RTD_TEST_INHERIT', 'unchanged')
    parent_env = os.environ.copy()
    monkeypatch.setattr(cli, 'hardware_identity', lambda: dict(uuid='parent-uuid', gpu='mock', memory=48*2**30))
    monkeypatch.setattr(evaluation, 'evaluate', lambda *a, **kw: None)
    monkeypatch.setattr(evaluation, 'report', lambda *a, **kw: None)
    launches = []
    monkeypatch.setattr(cli.subprocess, 'run', lambda command, **kw: launches.append((command, kw)))
    args = SimpleNamespace(command='run', run_dir=tmp_path/'run', arm='R0', port=9170,
                           config=ROOT/'configs/rtd/v1_bfcl_c25.yaml')
    cli.run_campaign(args, {'output_root': 'unused'})
    assert len(launches) == 3
    for round_number, (command, options) in enumerate(launches, 1):
        assert command[:3] == [sys.executable, str(ROOT/'tools/rtd_experiment.py'), 'run']
        assert '--training-worker' in command
        assert command[command.index('--through-round')+1] == str(round_number)
        assert command[command.index('--expected-gpu-uuid')+1] == 'parent-uuid'
        assert options['env'] == parent_env and options['env'] is not os.environ
        assert options['cwd'] == ROOT and options['check'] is True
    assert os.environ == parent_env


@pytest.mark.parametrize('ordering', [None, 'FASTEST_FIRST'])
def test_entrypoint_sets_order_before_importing_cli_without_changing_visibility(ordering):
    # Stub the CLI so this subprocess never imports torch or initializes CUDA.
    script = '''
import json, os, runpy, sys, types
cli = types.ModuleType('bfas.rtd.cli')
cli.main = lambda: 0
sys.modules['bfas.rtd.cli'] = cli
runpy.run_path('tools/rtd_experiment.py', run_name='test_entry')
print(json.dumps([os.environ['CUDA_DEVICE_ORDER'], os.environ['CUDA_VISIBLE_DEVICES']]))
'''
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = '2'
    env.pop('CUDA_DEVICE_ORDER', None)
    if ordering:
        env['CUDA_DEVICE_ORDER'] = ordering
    result = subprocess.run([sys.executable, '-c', script], env=env, cwd=ROOT,
                            capture_output=True, text=True, check=True)
    assert json.loads(result.stdout) == [ordering or 'PCI_BUS_ID', '2']


def test_worker_refuses_coordinator_gpu_mismatch_before_model_load(tmp_path, monkeypatch):
    from bfas.rtd import runtime
    config = cli.load_config(ROOT/'configs/rtd/v1_bfcl_c25.yaml')
    monkeypatch.setattr(cli, 'load_config', lambda path: config)
    monkeypatch.setattr(cli, 'bank_audit', lambda config: {})
    monkeypatch.setattr(cli, 'make_manifest', lambda *a, **kw: {'hardware': {'uuid': 'wrong-gpu'}})
    def forbidden(*a, **kw):
        pytest.fail('model must not load on a mismatched GPU')
    monkeypatch.setattr(runtime, 'load_backend', forbidden)
    args = SimpleNamespace(command='run', config='unused', run_dir=tmp_path/'run', arm='R0',
                           training_worker=True, expected_gpu_uuid='parent-gpu')
    with pytest.raises(ValueError, match='GPU UUID differs'):
        cli.run_command(args)


class Layer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base = torch.nn.Linear(5, 5, bias=False).requires_grad_(False)
        self.lora_A = torch.nn.Parameter(torch.randn(2, 5)*.2)
        self.lora_B = torch.nn.Parameter(torch.randn(5, 2)*.2)

    def forward(self, hidden, *, scale=1.):
        assert not self.training
        return hidden + scale*torch.tanh(self.base(hidden) + hidden @ self.lora_A.T @ self.lora_B.T)


class LM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(7, 5).requires_grad_(False)
        self.layers = torch.nn.ModuleList([Layer(), Layer()])
        self.head = torch.nn.Linear(5, 7, bias=False).requires_grad_(False)

    def forward(self, input_ids, attention_mask, use_cache=False):
        assert not use_cache and not self.training
        hidden = self.embed(input_ids)
        for layer in self.layers:
            hidden = layer(hidden, scale=.7)
        return SimpleNamespace(logits=self.head(hidden))


class Tokenizer:
    eos_token_id = 6
    def encode(self, text, add_special_tokens=False):
        return [2, 4] if text == 'teacher' else [0, 1, 3]


def problem():
    torch.manual_seed(25)
    b = TorchPolicyBackend(LM().double().eval(), Tokenizer(), base_checkpoint_hash='tiny',
        harness_hash='c25c', tokenizer_hash='tiny')
    parameters = snapshot(lora_parameters(b.model))
    # Force replay to differ from the resident student; otherwise a checkpoint
    # closure accidentally using restored parameters would pass the oracle.
    with torch.no_grad():
        for p in parameters.values():
            p.add_(.15)
    state = FullState.create({'question': 'q'}, [{'role': 'user', 'content': 'q'}], 'prompt', 'inner')
    targets = [(SourceSample(Behavior(state, 'source'), b.identity(parameters), (i % 5, 6), 6, -2.),
                None if i % 3 == 0 else Behavior(state, 'teacher')) for i in range(8)]
    chi = torch.randn(8, 3, dtype=torch.float64)
    phi = torch.tensor([.2, -.3, .1], dtype=torch.float64, requires_grad=True)
    return b, parameters, targets, chi, phi


@pytest.mark.parametrize('gate', ['linear_sigmoid', 'scalar_sigmoid', 'fixed_half', 'teacher_only'])
@pytest.mark.parametrize('checkpointed', [False, True])
def test_eight_sequential_slots_equal_batched_loss_gradients_and_release_graphs(gate, checkpointed):
    b, p, targets, chi, phi = problem()
    if gate == 'scalar_sigmoid':
        chi, phi = chi[:, :1], phi[:1].detach().requires_grad_(True)
    def loss():
        return sum(slot_loss(t, c, phi, b, p, gate=gate) for t, c in zip(targets, chi))/len(targets)
    expected = gradients(loss(), p)
    resident_hash = tensor_state_hash(lora_parameters(b.model))
    if checkpointed:
        for layer in b.model.layers:
            checkpoint_layer(layer)
    alive = 0
    peak = 0
    forwards = []
    class Saved:
        def __init__(self, tensor):
            nonlocal alive, peak
            self.tensor = tensor
            alive += 1
            peak = max(peak, alive)
        def __del__(self):
            nonlocal alive
            alive -= 1
    def before_forward(*args):
        # Every source/teacher forward starts after all previous saved tensors
        # have been released, including checkpoint inputs and LoRA references.
        assert alive == 0
        forwards.append(1)
    hook = b.model.register_forward_pre_hook(before_forward)
    with torch.autograd.graph.saved_tensors_hooks(Saved, lambda saved: saved.tensor):
        actual = streamed_gradient(targets, chi, phi, b, p, gate=gate)
    hook.remove()
    assert alive == 0 and peak > 0
    assert len(forwards) == sum(1 + (t is not None) for _, t in targets)
    assert tuple(actual) == tuple(p) and sum(g.numel() for g in actual.values()) == 40
    for name in p:
        torch.testing.assert_close(actual[name], expected[name], rtol=1e-12, atol=1e-12)
        assert actual[name].grad_fn is None and not actual[name].requires_grad
    assert all(t.grad is None for t in b.model.parameters())
    assert tensor_state_hash(lora_parameters(b.model)) == resident_hash


def test_checkpointed_sequential_gate_vjp_and_rms_match_batched_oracles():
    b, p, targets, chi, phi = problem()
    source_gradients = [gradients(-b.score_source(src, p), p) for src, _ in targets]
    expected_p, _ = rms_diagonal(p, (('inner', g) for g in source_gradients),
                                 inner_parent_hashes={'inner'}, source_snapshot_id=b.identity(p))
    step = FrozenStep(expected_p, .01, 'r1', {})
    feedback = ReturnGradient({n: torch.randn_like(t) for n, t in p.items()}, b.identity(p), {})
    loss = sum(slot_loss(t, c, phi, b, p) for t, c in zip(targets, chi))/len(targets)
    expected_vjp = gate_vjp(loss, p, phi, step, feedback.gradient)
    del source_gradients, loss
    for layer in b.model.layers:
        checkpoint_layer(layer)
    actual_p, _ = rms_diagonal(p, (('inner', gradients(-b.score_source(src, p), p)) for src, _ in targets),
                               inner_parent_hashes={'inner'}, source_snapshot_id=b.identity(p))
    for n in p:
        torch.testing.assert_close(actual_p[n], expected_p[n], rtol=0, atol=0)
    torch.testing.assert_close(streamed_gate_vjp(targets, chi, phi, b, p, step, feedback),
                               expected_vjp, rtol=1e-12, atol=1e-12)


def test_checkpointed_bf16_qwen_gradient_replays_scoring_snapshot_in_eval_mode():
    from test_rtd_score_consistency import qwen_backend, PROMPT
    b = qwen_backend()
    p = snapshot(lora_parameters(b.model))
    with torch.no_grad():
        for value in p.values():
            value.add_(.03)
    resident = tensor_state_hash(lora_parameters(b.model))
    ids = b.tokenizer.encode(PROMPT)
    saved_bytes = []
    def pack(tensor):
        saved_bytes.append(tensor.numel()*tensor.element_size())
        return tensor
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        expected_score = b.score_tokens(ids, (1, 2, 3), p, eos_token_id=3)
    uncheckpointed_bytes = sum(saved_bytes)
    expected = gradients(expected_score, p)
    rng = torch.Generator().manual_seed(5)
    expected_action = b.sample_action(PROMPT, p, rng)
    assert enable_gradient_checkpointing(b.model) == 4
    saved_bytes.clear()
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        score = b.score_tokens(ids, (1, 2, 3), p, eos_token_id=3)
    # This measures tensors saved for backward on a real hybrid-attention LM,
    # not GPU peak usage. It catches an eval-mode checkpoint switch doing nothing.
    assert sum(saved_bytes) < uncheckpointed_bytes / 2
    # A second outstanding functional graph uses another snapshot. Replays of
    # the first must remain bound to p even after that functional call returns.
    other = snapshot(p)
    with torch.no_grad():
        for value in other.values():
            value.add_(.05)
    other_score = b.score_tokens(ids, (2, 3), other, eos_token_id=3)
    actual = gradients(score, p)
    assert any(g.ne(0).any() for g in actual.values())
    torch.testing.assert_close(score, expected_score, rtol=0, atol=0)
    for n in p:
        torch.testing.assert_close(actual[n], expected[n], rtol=0, atol=0)
    gradients(other_score, other)
    assert b.sample_action(PROMPT, p, torch.Generator().manual_seed(5)) == expected_action
    assert not any(m.training for m in b.model.modules())
    assert all(t.grad is None for t in b.model.parameters())
    assert tensor_state_hash(lora_parameters(b.model)) == resident


@pytest.mark.parametrize('fail', [False, True])
def test_step_peak_survives_nested_timers_and_failures(tmp_path, monkeypatch, fail):
    resets = []
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda *a: None)
    monkeypatch.setattr(torch.cuda, 'reset_peak_memory_stats', lambda device: resets.append(device))
    peak = {'allocated': 0, 'reserved': 0}
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda device: peak['allocated'])
    monkeypatch.setattr(torch.cuda, 'max_memory_reserved', lambda device: peak['reserved'])
    journal = ComputeJournal(tmp_path/'compute.jsonl', cuda=True)
    try:
        with journal.measure_step(round=1, step=1, start_phase='round_start'):
            with journal.measure('source_geometry'):
                with journal.measure('forward'):
                    peak.update(allocated=10, reserved=12)
                peak.update(allocated=20, reserved=24)
            if fail:
                raise RuntimeError('simulated failure')
    except RuntimeError:
        assert fail
    assert resets == ['cuda:0']
    row = journal.events[-1]
    assert row['kind'] == 'step_memory'
    assert row['peak_allocated_bytes'] == 20 and row['peak_reserved_bytes'] == 24
    assert row['start_phase'] == 'round_start' and row['round'] == row['step'] == 1
    assert row['status'] == ('failed' if fail else 'complete')
    assert ComputeJournal(journal.path).events == journal.events
