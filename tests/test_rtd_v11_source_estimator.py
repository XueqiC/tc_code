"""D1 CPU oracles: full-vocabulary conditioning, independent CV and new VJP."""
from dataclasses import replace
import itertools
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT)]

from bfas.behavior.deltas import tensor_state_hash
from bfas.rtd import cli
from bfas.rtd.functional_step import FrozenStep, gradients, lora_parameters, snapshot
from bfas.rtd.persistence import ComputeJournal, digest
from bfas.rtd.return_gradient import ReturnGradient, TorchPolicyBackend
from bfas.rtd.runtime import streamed_gate_vjp, streamed_gradient
from bfas.rtd.source_estimator import (SourceControl, estimator_coefficients, gradient_norm,
    source_variance_diagnostic, validate_source_config)
from bfas.rtd.source_scoring import source_gradient_pair
from bfas.rtd.transport import Behavior, FullState, SourceSample
from test_rtd_manifest_tolerance import manifest_inputs
from test_rtd_checks import toy_bank

DT = torch.float64


class Tokenizer:
    eos_token_id = 3
    def encode(self, text, add_special_tokens=False):
        return [0, 1] if text.startswith('prompt') else [2]
    def decode(self, ids, skip_special_tokens=False):
        return ''.join(str(i) for i in ids)


class CategoricalLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_transition = torch.nn.Parameter(torch.tensor([
            [.1, -.1, .2, 1.], [.3, .4, .1, 1.1], [-.2, .3, .1, 1.2], [.1, .2, .1, 1.3]], dtype=DT))
        self.head = torch.nn.Identity()
    def get_output_embeddings(self):
        return self.head
    def forward(self, input_ids, attention_mask=None, use_cache=False, output_hidden_states=False):
        hidden = self.lora_transition[input_ids]
        return SimpleNamespace(logits=self.head(hidden), hidden_states=(hidden,))


def problem():
    backend = TorchPolicyBackend(CategoricalLM().eval(), Tokenizer(), base_checkpoint_hash='toy',
        harness_hash='d1', tokenizer_hash='toy', max_action_tokens=2, max_context_tokens=20)
    frozen = snapshot(lora_parameters(backend.model))
    current = snapshot(frozen)
    with torch.no_grad():
        current['lora_transition'].add_(torch.tensor([.4, -.2, .1, -.3], dtype=DT))
    state = FullState.create({'q': 'q'}, [{'role': 'user', 'content': 'q'}], 'prompt', 'inner')
    def source(tokens):
        return SourceSample(Behavior(state, 'source'), backend.identity(frozen), tokens, 3, -2., tokens[-1] != 3)
    sources = (source((0, 3)), source((2, 1)), source((3,)))
    targets = [(s, Behavior(state, 'teacher')) for s in sources]
    chi = torch.tensor([[1., -.3], [1., .8], [1., 1.4]], dtype=DT)
    phi = torch.tensor([.2, -.6], dtype=DT, requires_grad=True)
    return backend, current, frozen, targets, chi, phi


def full_soft_loss(backend, source, current, frozen):
    prompt = tuple(backend.tokenizer.encode(source.behavior.state.prompt))
    ids = torch.tensor([prompt + source.token_ids])
    sl = slice(len(prompt)-1, ids.shape[1]-1)
    with torch.no_grad():
        p = backend._logits(frozen, ids)[0, sl].softmax(-1)
    return -(p * backend._logits(current, ids)[0, sl].log_softmax(-1)).sum()


@pytest.mark.parametrize('tokens', [(3,), (0, 3), (2, 1)])
@pytest.mark.parametrize('dtype', [torch.float64, torch.float32])
def test_each_prefix_has_exact_zero_soft_gradient_at_snapshot(tokens, dtype):
    b, _, frozen, targets, _, _ = problem()
    b.model.to(dtype)
    frozen = snapshot(lora_parameters(b.model))
    with torch.no_grad():
        frozen['lora_transition'].add_(torch.tensor([.15, -.2, .3, -.4], dtype=dtype))
    assert tensor_state_hash(frozen) != tensor_state_hash(lora_parameters(b.model))
    source = replace(targets[0][0], token_ids=tokens, truncated=tokens[-1] != 3,
                     frozen_snapshot_id=b.identity(frozen))
    # Check each prefix separately too: sequence summation cannot hide errors.
    for length in range(1, len(tokens)+1):
        prefix = tokens[:length]
        s = replace(source, token_ids=prefix, truncated=prefix[-1] != 3)
        hard, soft, meta = source_gradient_pair(b, s, snapshot(frozen), frozen)
        assert gradient_norm(hard) > .1
        assert gradient_norm(soft) == 0.
        assert meta['source_kl'] == 0.
    assert all(p.grad is None for p in frozen.values())


def test_streamed_pair_matches_hard_and_full_vocabulary_soft_oracles():
    b, p, frozen, targets, _, _ = problem()
    for source, _ in targets:
        expected_hard = gradients(-b.score_source(source, p), p)
        expected_soft = gradients(full_soft_loss(b, source, p, frozen), p)
        for size in (1, 2):
            hard, soft, _ = source_gradient_pair(b, source, p, frozen, position_batch_size=size)
            for n in p:
                torch.testing.assert_close(hard[n], expected_hard[n], atol=1e-12, rtol=1e-12)
                torch.testing.assert_close(soft[n], expected_soft[n], atol=1e-12, rtol=1e-12)
                assert not hard[n].requires_grad and not soft[n].requires_grad
    with pytest.raises(ValueError, match='snapshot mismatch'):
        source_gradient_pair(b, targets[0][0], p, p)


def enumerable(b, p, frozen, template):
    rows = []
    for tokens in [(3,), *((first, second) for first in range(3) for second in range(4))]:
        source = replace(template, token_ids=tokens, truncated=tokens[-1] != 3)
        probability = b.score_source(source, frozen).detach().exp()
        hard, soft, _ = source_gradient_pair(b, source, p, frozen)
        rows.append((source, probability, hard['lora_transition'].flatten(), soft['lora_transition'].flatten()))
    return rows


def test_enumeration_and_monte_carlo_equal_expected_hard_and_soft_sequence_gradients():
    b, p, frozen, targets, _, _ = problem()
    rows = enumerable(b, p, frozen, targets[0][0])
    probs = torch.stack([r[1] for r in rows])
    hard, soft = (torch.stack([r[i] for r in rows]) for i in (2, 3))
    torch.testing.assert_close(probs.sum(), probs.new_tensor(1.))
    torch.testing.assert_close(probs @ hard, probs @ soft, atol=1e-12, rtol=1e-12)
    indices = torch.multinomial(probs, 100000, replacement=True, generator=torch.Generator().manual_seed(719))
    differences = (hard-soft)[indices]
    assert torch.all(differences.mean(0).abs() <= 6*differences.std(0)/len(indices)**.5 + 1e-12)


def controls_for(targets, chi, mode):
    sources = tuple(s for s, _ in targets)
    if mode == 'loo':
        return [SourceControl(sources, chi, i) for i in range(len(sources))]
    # Separate draw objects may legitimately realize exactly the same tokens.
    auxiliary = tuple(replace(s) for s in sources)
    auxiliary_chi = chi + chi.new_tensor([0., .3])
    return [SourceControl(auxiliary, auxiliary_chi) for _ in sources]


def test_loo_excludes_current_draw_and_keeps_auxiliary_gate_differentiable():
    b, p, frozen, targets, chi, phi = problem()
    controls = controls_for(targets, chi, 'loo')
    _, cs = estimator_coefficients(targets, chi, phi, gate='linear_sigmoid', source_estimator='cv', controls=controls)
    expected = (1-(chi[1:] @ phi).sigmoid()).mean()
    torch.testing.assert_close(cs[0], expected)
    torch.testing.assert_close(torch.autograd.grad(cs[0], phi, retain_graph=True)[0],
                               torch.autograd.grad(expected, phi)[0])
    changed = chi.clone(); changed[0] += 20
    _, changed_cs = estimator_coefficients(targets, changed, phi, gate='linear_sigmoid', source_estimator='cv',
                                          controls=controls_for(targets, changed, 'loo'))
    torch.testing.assert_close(cs[0], changed_cs[0], atol=0, rtol=0)
    assert cs[1] != changed_cs[1]
    with pytest.raises(ValueError, match='index'):
        SourceControl(tuple(s for s, _ in targets), chi, 1).features(targets[0][0], 'loo')
    with pytest.raises(ValueError, match='exclude'):
        SourceControl(tuple(s for s, _ in targets), chi).features(targets[0][0], 'independent')
    with pytest.raises(ValueError, match='distinct'):
        SourceControl((targets[0][0], targets[0][0]), chi[:2], 0).features(targets[0][0], 'loo')


def test_cv_unbiased_on_enumerable_independent_source_pairs():
    b, p, frozen, targets, _, phi = problem()
    rows = enumerable(b, p, frozen, targets[0][0])
    features = torch.tensor([[1., float(r[0].token_ids[0])-1] for r in rows], dtype=DT)
    gates = (features @ phi.detach()).sigmoid()
    gt = gradients(-b.score_behavior(targets[0][1], p), p)['lora_transition'].flatten()
    expected = sum(r[1]*((1-a)*r[2]+a*gt) for r, a in zip(rows, gates))
    cv = torch.zeros_like(expected)
    for i, j in itertools.product(range(len(rows)), repeat=2):
        # j is an independent source draw (also the two-sample LOO law).
        ri, rj = rows[i], rows[j]
        cv += ri[1]*rj[1]*((1-gates[i])*ri[2]+gates[i]*gt-(1-gates[j])*(ri[2]-ri[3]))
    torch.testing.assert_close(cv, expected, atol=1e-12, rtol=1e-12)
    naive_soft = sum(r[1]*((1-a)*r[3]+a*gt) for r, a in zip(rows, gates))
    assert (naive_soft-expected).norm() > .01


@pytest.mark.parametrize('gate', ['fixed_half', 'scalar_sigmoid', 'teacher_only'])
def test_fixed_coefficient_cv_reduces_bit_exactly_to_soft(gate):
    b, p, frozen, targets, _, _ = problem()
    chi = torch.ones(len(targets), 1, dtype=DT)
    phi = torch.tensor([.3], dtype=DT, requires_grad=True)
    kw = dict(gate=gate, source_parameters=frozen, cv_cs_mode='fixed_one_minus_a')
    soft = streamed_gradient(targets, chi, phi, b, p, source_estimator='soft', **kw)
    cv = streamed_gradient(targets, chi, phi, b, p, source_estimator='cv', **kw)
    assert tensor_state_hash(soft) == tensor_state_hash(cv)


@pytest.mark.parametrize('estimator,mode,gate', [
    ('cv', 'loo', 'linear_sigmoid'), ('cv', 'independent', 'linear_sigmoid'),
    ('cv', 'fixed_one_minus_a', 'scalar_sigmoid'), ('soft', 'loo', 'scalar_sigmoid')])
@pytest.mark.parametrize('diagonal_dtype', [torch.float32, torch.float64])
def test_new_update_gate_vjp_matches_nonlinear_outer_finite_difference(estimator, mode, gate, diagonal_dtype):
    b, p, frozen, targets, chi, phi = problem()
    if gate == 'scalar_sigmoid':
        chi, phi = torch.ones(len(targets), 1, dtype=DT), phi[:1].detach().requires_grad_(True)
    # One missing-teacher slot tests the effective zero gate and its derivative.
    targets[-1] = (targets[-1][0], None)
    controls = controls_for(targets, chi, mode) if gate == 'linear_sigmoid' else None
    kw = dict(gate=gate, source_estimator=estimator, source_parameters=frozen,
              cv_cs_mode=mode, source_controls=controls)
    step = FrozenStep({n: torch.linspace(.4, 1.6, t.numel(), dtype=diagonal_dtype).reshape_as(t)
                       for n, t in p.items()}, .17, 'r1', {})
    def updated(control):
        return step.update(p, streamed_gradient(targets, chi, control, b, p, **kw))
    def reward(parameters):
        return parameters['lora_transition'].sin().sum()
    actual = snapshot(updated(phi))
    feedback = ReturnGradient(gradients(reward(actual), actual), tensor_state_hash(actual), {})
    vjp = streamed_gate_vjp(targets, chi, phi, b, p, step, feedback, **kw)
    finite = []
    for i in range(len(phi)):
        delta = torch.zeros_like(phi); delta[i] = 1e-5
        finite.append((reward(updated(phi+delta))-reward(updated(phi-delta)))/2e-5)
    torch.testing.assert_close(vjp, torch.stack(finite), atol=2e-10, rtol=1e-7)
    legacy = streamed_gate_vjp(targets, chi, phi, b, p, step, feedback, gate=gate)
    assert (vjp-legacy).norm() > 1e-4


def test_full_gate_rejects_biased_soft_or_current_sample_coefficient():
    b, p, frozen, targets, chi, phi = problem()
    for estimator in ('soft', 'cv'):
        with pytest.raises(ValueError, match='fixed/scalar'):
            streamed_gradient(targets, chi, phi, b, p, source_parameters=frozen,
                              source_estimator=estimator, cv_cs_mode='fixed_one_minus_a')
    with pytest.raises(ValueError, match='controls'):
        streamed_gradient(targets, chi, phi, b, p, source_parameters=frozen, source_estimator='cv')


def test_vocab_lifetime_two_backbone_forwards_and_head_trainable_gradients():
    from test_rtd_runtime_memory import LM
    from bfas.rtd.checkpointing import checkpoint_layer
    torch.manual_seed(122)
    model = LM().double().eval()
    class Head(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lora_weight = torch.nn.Parameter(torch.randn(19, 5, dtype=DT))
        def forward(self, x):
            return x @ self.lora_weight.T
    model.head = Head()
    model.get_output_embeddings = lambda: model.head
    b = TorchPolicyBackend(model, Tokenizer(), base_checkpoint_hash='head', harness_hash='d1', tokenizer_hash='toy')
    frozen = snapshot(lora_parameters(model)); p = snapshot(frozen)
    with torch.no_grad():
        for value in p.values():
            value.add_(.12)
    source = replace(problem()[3][0][0], frozen_snapshot_id=b.identity(frozen))
    hard_expected = gradients(-b.score_source(source, p), p)
    soft_expected = gradients(full_soft_loss(b, source, p, frozen), p)
    for layer in model.layers:
        checkpoint_layer(layer)
    forwards, projections = [], []
    hook = model.register_forward_pre_hook(lambda *args: forwards.append(1))
    # Ordinary post hooks never see the aborted full-sequence head invocation.
    head_hook = model.head.register_forward_hook(lambda m, args, out: projections.append(out.shape))
    resident = tensor_state_hash(lora_parameters(model))
    hard, soft, _ = source_gradient_pair(b, source, p, frozen)
    hook.remove(); head_hook.remove()
    assert len(forwards) == 2
    assert projections == [torch.Size([1, 19])] * (2*source.length)
    for n in p:
        torch.testing.assert_close(hard[n], hard_expected[n], atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(soft[n], soft_expected[n], atol=1e-12, rtol=1e-12)
    assert tensor_state_hash(lora_parameters(model)) == resident
    assert not model.head._forward_pre_hooks
    assert all(t.grad is None for t in model.parameters())


def test_real_cpu_hf_qwen_snapshot_identity_with_checkpointed_source_forwards():
    from test_rtd_score_consistency import qwen_backend, PROMPT
    from bfas.rtd.checkpointing import enable_gradient_checkpointing
    b = qwen_backend()
    assert enable_gradient_checkpointing(b.model) == 4
    frozen = snapshot(lora_parameters(b.model))
    with torch.no_grad():
        for p in frozen.values():
            p.add_(.03)
    state = FullState.create({'q': 'q'}, [{'role': 'user', 'content': 'q'}], PROMPT, 'inner')
    source = SourceSample(Behavior(state, 'source'), b.identity(frozen), (2, 3), 3, -2.)
    hard, soft, _ = source_gradient_pair(b, source, snapshot(frozen), frozen)
    assert gradient_norm(hard) > 0 and gradient_norm(soft) == 0
    assert all(p.grad is None for p in b.model.parameters())


def test_diagnostic_resamples_sources_and_journals_each_slot(tmp_path):
    b, _, frozen, targets, _, _ = problem()
    b.journal = ComputeJournal(tmp_path/'compute.jsonl')
    b.context = 'variance'
    phi = torch.tensor([.2], dtype=DT)
    kwargs = dict(generator=torch.Generator().manual_seed(66), K=30, gate='scalar_sigmoid',
                  teacher=targets[0][1], cs_mode='fixed_one_minus_a', source_samples_per_state=2)
    report = source_variance_diagnostic(targets[0][0].behavior.state, b, snapshot(frozen), frozen, phi,
                                       lambda s: phi.new_ones(1), **kwargs)
    assert report['estimators']['hard2']['gradient_norm_variance'] > .001
    assert report['estimators']['soft'] == report['estimators']['cv']
    assert report['estimators']['soft']['gradient_norm_variance'] < 1e-28
    rows = [e for e in b.journal.events if e['kind'] == 'source_estimator_slot']
    assert len(rows) == 60
    for row in rows:
        assert row['c_s'] == pytest.approx(float(1-phi.sigmoid()))
        assert row['soft_source_gradient_norm'] == 0
        assert row['cv_gradient_norm'] == row['soft_gradient_norm']
        assert row['hard_gradient_norm'] > 0


@pytest.mark.parametrize('options', [dict(source_estimator='bad'), dict(cv_cs_mode='bad'),
    dict(source_samples_per_state=0), dict(source_samples_per_state=True),
    dict(source_estimator='cv', source_samples_per_state=1)])
def test_invalid_config_is_rejected(options):
    with pytest.raises(ValueError):
        validate_source_config(options)


def test_hard2_config_manifest_and_reference_gradient_are_byte_identical(manifest_inputs, tmp_path, monkeypatch):
    from test_rtd_runtime_memory import problem as legacy_problem
    c = manifest_inputs
    # Pin values captured by executing main HEAD's unmodified cli/runtime.
    monkeypatch.setattr(cli, 'ROOT', ROOT)
    legacy_config = cli.load_config(ROOT/'configs/rtd/v1_bfcl_c25.yaml')
    # D15 records one new default; compare the historical declaration to its oracle.
    historical_config = json.loads(json.dumps(legacy_config))
    assert historical_config['score_consistency_tolerance'].pop('min_tokens_for_mean') == 8
    assert digest(historical_config) == '96c08a0e77c1c260caf65cc4e5b8e761992ae8754e938e3af8fa2a23c0956d09'
    path = tmp_path/'explicit-hard2.yaml'
    path.write_text(yaml.safe_dump(dict(legacy_config, source_estimator='hard2', cv_cs_mode='loo')))
    explicit = cli.load_config(path)
    assert json.dumps(explicit, sort_keys=True) == json.dumps(legacy_config, sort_keys=True)
    explicit['student'] = c.config['student']
    monkeypatch.setattr(cli, 'ROOT', c.root)
    before = cli.make_manifest(c.config, 'R1', c.audit)
    after = cli.make_manifest(explicit, 'R1', c.audit)
    assert json.dumps(after, sort_keys=True).encode() == json.dumps(before, sort_keys=True).encode()
    # Entire pre-D1 manifest oracle; only the temporary checkout path and its
    # derived config hash are normalized. Code provenance still uses real files
    # in this fixed synthetic BFCL checkout, rather than a stubbed source hash.
    normalized = json.loads(json.dumps(after).replace(str(c.root), '{ROOT}'))
    normalized['config']['score_consistency_tolerance'].pop('min_tokens_for_mean')
    normalized['score_consistency']['tolerance'].pop('min_tokens_for_mean')
    normalized['config_hash'] = digest(normalized['config'])
    # D7 adds report labels outside the frozen scoring projection. Preserve
    # honest current source provenance, then normalize this one operational
    # file to the pre-D7 hash for the historical manifest-byte oracle.
    assert after['rtd_source'] == cli.source_identity(c.root)
    normalized['rtd_source']['files']['src/bfas/rtd/evaluation.py'] = 'e2310d926fae8dba3f516d43c5cc90bb8032a9bca6d7ae4a13b918299f94ad2e'
    # D5 adds a separate syntax diagnostic endpoint. The official verdict
    # projection remains pinned by test_rtd_identity_update; real manifests
    # retain the current raw source hash, as asserted above.
    normalized['rtd_source']['files']['tools/behavior_atom/checker_bridge.py'] = 'e928dcb48ed58c47d4c37fa234821e40fc645cc76208560e20de1b303f3a3bfd'
    # The pre-P0 Azure BFCL merge changed operational adapter code. Its frozen
    # scoring projection is unchanged; pin that fact before normalizing only
    # the raw file hash to c3300f2 for this historical manifest-byte oracle.
    adapter = 'src/bfas/adapters/bfcl.py'
    assert after['evaluation_harness']['tools'][adapter] == 'f6d6d341e937d3330a9ae2a1030713085a411c53ef727517cd6ac64c79ef677a'
    normalized['rtd_source']['files'][adapter] = 'fd26057630d1d37b04d27bf94be0e058b5890fe1debea6f0934b2aeb85fa5537'
    normalized['rtd_source']['hash'] = digest(normalized['rtd_source']['files'])
    assert digest(normalized) == '428e3d9bfd56011060e763bb6cc781e7c9007115b393cd12a77980a16c4ba3af'
    b, p, targets, chi, phi = legacy_problem()
    actual = streamed_gradient(targets, chi, phi, b, p, source_estimator='hard2')
    assert tensor_state_hash(actual) == '8ea394b94365283bf4b954871b5721715518307b174464d52878f245a89d5358'


@pytest.mark.parametrize('mode', ['loo', 'independent', 'fixed_one_minus_a'])
def test_runner_refreshes_cv_draws_and_recovers_exact_controls(toy_bank, tmp_path, mode):
    from test_rtd_checks import experiment, engine_config
    config = dict(engine_config(), source_estimator='cv', cv_cs_mode=mode, source_samples_per_state=3)
    if mode == 'fixed_one_minus_a':
        config['gate'] = 'scalar_sigmoid'
    e = experiment(tmp_path/mode, toy_bank, config=config, smoke=True)
    # Give the tiny runner its explicit positionwise output-head interface.
    e.backend.model.head = torch.nn.Identity()
    e.backend.model.get_output_embeddings = lambda: e.backend.model.head
    original = e.backend.model.forward
    def forward(*args, **kw):
        out = original(*args, **kw)
        out.logits = e.backend.model.head(out.logits)
        return out
    e.backend.model.forward = forward
    e.round_start()
    cached = tuple(s for group in e.state['source_cache'].values() for s in group)
    e.step_start()
    assert len(e.state['old_controls']) == e.slots
    assert all(all(s is not old for old in cached) for s, _ in e.state['old_targets'])
    for source, _ in e.state['old_targets']:
        assert len(e.state['source_cache'][source.behavior.state.state_hash]) == 3
    e.reference(); e.selected(); e.revealed()
    loaded = e.store.load(e.ledger)
    for target, control in zip(loaded['new_targets'], loaded['new_controls']):
        if mode != 'fixed_one_minus_a':
            assert len(control.features(target[0], mode)) == (2 if mode == 'loo' else 3)
    options = e.estimator_options(e.state['new_controls'])
    expected = streamed_gradient(e.state['new_targets'], e.state['new_chi'], e.state['phi'], e.backend,
                                e.state['reference'].start, gate=e.gate, **options)
    replay = streamed_gradient(loaded['new_targets'], loaded['new_chi'], loaded['phi'], e.backend,
        loaded['reference'].start, gate=e.gate, source_estimator='cv', source_parameters=loaded['source'],
        cv_cs_mode=mode, source_controls=loaded['new_controls'])
    assert tensor_state_hash(expected) == tensor_state_hash(replay)
    e.actual()  # The experiment's meta-update must consume those same controls.
    assert e.state['phase'] == 'feedback'
