"""Section 13's seven checks plus CPU transaction/evaluation regression cases.

Synthetic finite policies test mathematics and orchestration, never benchmark
performance. No production model, teacher API, vLLM server or GPU is launched.
"""
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT)]
from bfas.behavior.deltas import tensor_state_hash
from bfas.cc_pairs import digest
from bfas.rtd.broker import RequestRecord, SealedReplayBroker, seal_bank
from bfas.rtd.caps import affordability, public_cap
from bfas.rtd.evaluation import report, validate_evaluation
from bfas.rtd.experiment import RTDExperiment, assert_run_invariants
from bfas.rtd.features import FeatureRow, FrozenProjection
from bfas.rtd.functional_step import FrozenStep, gradients, lora_parameters, snapshot
from bfas.rtd.insertion import insertion_derivative
from bfas.rtd.ledger import Ledger, LedgerError
from bfas.rtd.persistence import StateStore, atomic_json, tree_hash
from bfas.rtd.return_gradient import ReturnGradient, TaskRollout, TorchPolicyBackend, gate_vjp
from bfas.rtd.runtime import HFGenerateBackend, installed_parameters, slot_loss, streamed_gate_vjp
from bfas.rtd.selector import PublicFeatures, PublicQuerySpec, StudentSnapshot, select_public
from bfas.rtd.transport import (Behavior, FullState, SourceSample, complete_sequence_logprob,
                                finite_transport_q, positive_mixture_loss)

DT = torch.float64


def test_check1_finite_q_normalization_identity_endpoints_and_outside_mass():
    p = torch.tensor([.1, .2, .3, .4], dtype=DT)
    nu = torch.tensor([0., 1., 0., 0.], dtype=DT)
    a = torch.tensor([.1, .4, .7, .8], dtype=DT)
    q = finite_transport_q(p, a, nu)
    assert float(q.sum()) == pytest.approx(1.) and (q >= 0).all()
    torch.testing.assert_close(q[[0, 2, 3]], (p*(1-a))[[0, 2, 3]])
    for gate, teacher, expected in [(a*0, nu, p), (a*0+1, nu, nu), (a, None, p)]:
        torch.testing.assert_close(finite_transport_q(p, gate, teacher), expected)
    with pytest.raises(ValueError, match='sum to one'):
        finite_transport_q(p[:2], a[:2], nu[:2])


def test_check2_mc_positive_mixture_matches_explicit_q_ce():
    p = torch.tensor([.1, .2, .3, .4], dtype=DT)
    nu = torch.tensor([.4, .6, 0., 0.], dtype=DT)
    a = torch.tensor([.8, .2, .5, .1], dtype=DT)
    logp = torch.tensor([.25, .15, .5, .1], dtype=DT).log()
    rng = torch.Generator().manual_seed(88)
    sources = torch.multinomial(p, 150000, replacement=True, generator=rng)
    teachers = torch.multinomial(nu, len(sources), replacement=True, generator=rng)
    losses = positive_mixture_loss(logp[sources], logp[teachers], a[sources], reduction='none')
    exact = -(finite_transport_q(p, a, nu)*logp).sum()
    assert abs(float(losses.mean()-exact)) < 5*float(losses.std()/len(sources)**.5)
    assert (losses >= 0).all()


def derivative_problem():
    theta = {'lora_w': torch.tensor([.2, -.4, .1], dtype=DT, requires_grad=True)}
    phi = torch.tensor([.1, -.3], dtype=DT, requires_grad=True)
    chi = torch.tensor([[1., .2], [.3, 1.]], dtype=DT)
    step = FrozenStep({'lora_w': torch.tensor([.3, 1., 1.7], dtype=DT)}, .13, 'r1', {})
    def loss(p, control):
        lp = p['lora_w'].log_softmax(0)
        return positive_mixture_loss(lp[[0, 1]], lp[[1, 2]], (chi@control).sigmoid())
    def reward(p):
        return p['lora_w'].softmax(0)[2] + .2*p['lora_w'].softmax(0)[0].square()
    return theta, phi, step, loss, reward


def test_check3_gate_vjp_and_insertion_match_double_central_differences():
    theta, phi, step, loss, reward = derivative_problem()
    gd = gradients(loss(theta, phi), theta)
    reference = snapshot(step.update(theta, gd))
    gj = gradients(reward(reference), reference)
    vjp = gate_vjp(loss(theta, phi), theta, phi, step, gj)
    def outer(control):
        p = snapshot(theta)
        return reward(step.update(p, gradients(loss(p, control), p)))
    finite = []
    for i in range(len(phi)):
        delta = torch.zeros_like(phi); delta[i] = 1e-5
        finite.append((outer(phi+delta)-outer(phi-delta))/2e-5)
    torch.testing.assert_close(vjp, torch.stack(finite), atol=1e-10, rtol=1e-7)
    gq = gradients(-theta['lora_w'].log_softmax(0)[0], theta)
    insertion = insertion_derivative(gq, gd, gj, step)
    def inserted(e):
        return reward(step.update(theta, {n: (1-e)*gd[n]+e*gq[n] for n in theta}))
    assert float(insertion) == pytest.approx(float((inserted(1e-5)-inserted(-1e-5))/2e-5), abs=1e-10)


def test_check4_equal_gradients_zero_and_missing_old_gradient_is_wrong():
    theta, phi, step, loss, reward = derivative_problem()
    gd = gradients(loss(theta, phi), theta)
    ref = snapshot(step.update(theta, gd))
    gj = gradients(reward(ref), ref)
    assert float(insertion_derivative(gd, gd, gj, step)) == 0.
    wrong = -step.eta*sum((gj[n]*step.diagonal[n]*gd[n]).sum() for n in gd)
    assert abs(float(wrong)) > 1e-5


class TinyTokenizer:
    eos_token_id = 3
    bos_token_id = 0
    def encode(self, text, add_special_tokens=False):
        return [0, 1] if text.startswith('prompt') else [{'a': 0, 'b': 1, 'c': 2}[c] for c in text]
    def decode(self, ids, skip_special_tokens=False):
        return ''.join('abc'[i] for i in ids)
    def save_pretrained(self, directory):
        Path(directory).mkdir(parents=True, exist_ok=True)
        (Path(directory)/'tokenizer.json').write_text('{"toy":true}')


class TinyLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_w = torch.nn.Parameter(torch.tensor([[.2, .5, -.3, 2.]]*4, dtype=DT))
    def forward(self, input_ids, attention_mask=None, use_cache=False, output_hidden_states=False):
        logits = self.lora_w[input_ids]
        return SimpleNamespace(logits=logits, hidden_states=(logits,) if output_hidden_states else None)
    def save_pretrained(self, directory, **kwargs):
        Path(directory).mkdir(parents=True, exist_ok=True)
        from safetensors.torch import save_file
        save_file({'lora_w': self.lora_w.detach().contiguous()}, str(Path(directory)/'adapter_model.safetensors'))


def tiny_backend():
    return TorchPolicyBackend(TinyLM().eval(), TinyTokenizer(), base_checkpoint_hash='toybase',
        harness_hash='toyharness', tokenizer_hash='toytokenizer', max_action_tokens=40, max_context_tokens=100)


def test_check5_source_sampling_scores_eos_masks_and_hf_generation_restore():
    b = tiny_backend()
    params = snapshot(lora_parameters(b.model))
    rng = torch.Generator().manual_seed(9)
    actions = [b.sample_action('prompt', params, rng) for _ in range(30)]
    assert len({a.action_ids for a in actions}) > 1
    for action in actions:
        assert float(b.score_action(action, params)) == pytest.approx(action.generation_logprob, abs=1e-12)
        ids = torch.tensor([action.prompt_ids+action.action_ids])
        logits = b._logits(params, ids)
        mask = torch.zeros_like(ids, dtype=torch.bool); mask[:, len(action.prompt_ids):] = True
        value = complete_sequence_logprob(logits, ids, mask, eos_token_id=3)
        assert float(value) == pytest.approx(action.generation_logprob, abs=1e-12)
        mask[:, -1] = False
        with pytest.raises(ValueError):
            complete_sequence_logprob(logits, ids, mask, eos_token_id=3)
    initial = tensor_state_hash(lora_parameters(b.model))
    changed = snapshot(params); changed['lora_w'].data.add_(.2)
    with pytest.raises(RuntimeError):
        with installed_parameters(b.model, changed):
            assert tensor_state_hash(lora_parameters(b.model)) != initial
            raise RuntimeError('generation failed')
    assert tensor_state_hash(lora_parameters(b.model)) == initial


@pytest.fixture
def toy_bank(tmp_path):
    parents = [f'{i:064x}' for i in (1, 2, 3, 4)]
    records, payloads, states = [], {}, {}
    for i, parent in enumerate(parents):
        state = FullState.create({'question': str(i)}, [{'role':'user', 'content':str(i)}], f'prompt {i}', parent)
        states[parent] = state
        q = digest(['query', i])
        cap, provenance = public_cap('generator_item')
        spec = PublicQuerySpec(q, state.state_hash, PublicFeatures(), 1, cap, 'exact', provenance)
        records.append(RequestRecord(spec, parent))
        payloads[q] = dict(cost=10+i, cost_confidence='exact', usage={'output_tokens':10+i},
            provenance={'kind':'generator_item'}, historical_response='b',
            historical_events=[{'alias':1}, {'alias':2}], behaviors=[{'state':asdict(state), 'text':'b'}])
    bank = tmp_path/'bank'
    seal_bank(bank, records, payloads)
    return bank, records, payloads, states


def test_check6_sealing_features_same_package_events_and_dependencies(toy_bank, tmp_path):
    bank, records, payloads, states = toy_bank
    parent = records[0].parent_hash
    first = records[0]
    second = replace(records[1], parent_hash=parent, dependencies=(first.spec.query_id,))
    payloads[second.spec.query_id]['behaviors'] = payloads[first.spec.query_id]['behaviors']
    bank = seal_bank(tmp_path/'dependency-bank', [first, second],
                     {r.spec.query_id: payloads[r.spec.query_id] for r in (first, second)})
    ledger = Ledger(20000, tmp_path/'ledger.jsonl')
    broker = SealedReplayBroker(bank, ledger, inner_parent_hashes={parent})
    view = StudentSnapshot('source', frozenset({parent}))
    public = broker.list_candidates(view, (), ledger.remaining)
    assert [c.query_id for c in public] == [first.spec.query_id]
    with pytest.raises(PermissionError):
        select_public(public, lambda _: (bank/'sealed'/f'{first.spec.query_id}.json').read_text())
    with pytest.raises(ValueError, match='deferred'):
        FeatureRow(parent, first.spec.state_hash, 'initial', public[0].features).tensor()
    package = broker.acquire(first.spec.query_id)
    assert len(package.historical_events) == 2
    assert broker.acquire(first.spec.query_id) is package
    broker.list_candidates(view, ledger.owned_ids, ledger.remaining)
    broker.acquire(second.spec.query_id)
    restored = Ledger.resume(20000, tmp_path/'ledger.jsonl')
    assert restored.spent == sum(payloads[r.spec.query_id]['cost'] for r in (first, second))
    assert [e['query_id'] for e in restored.events if e['kind']=='reveal'] == [first.spec.query_id, second.spec.query_id]
    payloads[first.spec.query_id]['cost'] = 5000
    assert affordability([first, second], payloads) == affordability([first, second])


class TinySupport:
    def __init__(self, states):
        self.states = states
        self.parents = {h: 'simple_python_'+str(i) for i, h in enumerate(states)}
        self.categories = {t:'simple_python' for t in self.parents.values()}
    def feedback(self, parent, backend, parameters, generator, checker):
        action = backend.sample_action(self.states[parent].prompt, parameters, generator)
        return TaskRollout(self.parents[parent], (action,), float(action.action_ids[0] == 1), backend.identity(parameters))


def engine_config():
    import yaml
    return yaml.safe_load((ROOT/'configs/rtd/v1_bfcl_c25.yaml').read_text())


def experiment(directory, toy_bank, *, resume=False, after_save=None, arm='R1', smoke=False, config=None):
    bank, _, _, states = toy_bank
    config = config or engine_config()
    manifest = dict(config_hash=digest(config), config=config, arm=arm, bank_path=str(bank),
                    budget_ceilings=[100000,200000,300000], bank_public_cap_sum=32768, hardware_hash='toy-cpu')
    return RTDExperiment(config, manifest, directory, tiny_backend(), TinySupport(states),
                         resume=resume, after_save=after_save, smoke=smoke)


def test_check7_noop_initial_restore_and_complete_evaluation(toy_bank, tmp_path):
    # A closed budget enforces empty; exact identity must survive all 36 steps.
    e = experiment(tmp_path/'noop', toy_bank)
    initial = tensor_state_hash(e.state['initial'])
    e.ceilings[:] = [0, 0, 0]
    e.ledger.budget = 0
    result = e.run()
    assert result['decision_windows'] == 12 and result['steps'] == 36
    assert all(s['exact_noop'] for s in e.state['steps'])
    assert tensor_state_hash(e.state['parameters']) == initial
    assert e.state['checkpoints'][-1]['parameter_hash'] == initial
    # Official score files list failures plus a summary, so total coverage and
    # checkpoint provenance are required before passes may be inferred.
    expected = dict(generation={'simple_python':['t1','t2']}, scoring={'simple_python':['t1','t2']})
    resultdir, scoredir = tmp_path/'results', tmp_path/'scores'
    resultdir.mkdir(); scoredir.mkdir()
    generated = resultdir/'BFCL_v4_simple_python_result.json'
    generated.write_text('{"id":"t1"}\n')
    score = scoredir/'BFCL_v4_simple_python_score.json'
    score.write_text('{"correct_count":1,"total_count":2}\n{"id":"t2","valid":false}\n')
    with pytest.raises(ValueError, match='incomplete evaluation generation'):
        validate_evaluation(expected, resultdir, scoredir)
    generated.write_text('{"id":"t1"}\n{"id":"t2"}\n')
    assert validate_evaluation(expected, resultdir, scoredir)['verdicts'] == {'t1':True, 't2':False}
    score.unlink()
    with pytest.raises(ValueError, match='score categories'):
        validate_evaluation(expected, resultdir, scoredir)


@pytest.mark.parametrize('arm', ['R0', 'R1'])
def test_full_loop_rotation_budget_windows_and_idempotent_resume(toy_bank, tmp_path, arm):
    directory = tmp_path/arm
    e = experiment(directory, toy_bank, arm=arm)
    e.run()
    assert e.ledger.spent > 0
    assert len(e.state['checkpoints']) == 3
    assert [s['step'] for s in e.state['steps'] if s['decision']] == [1,4,7,10]*3
    memory = [event for event in e.journal.events if event['kind'] == 'step_memory']
    assert [(row['round'], row['step']) for row in memory] == [(r, s) for r in (1, 2, 3) for s in range(1, 13)]
    assert all(row['status'] == 'complete' and row['peak_allocated_bytes'] == row['peak_reserved_bytes'] == 0
               for row in memory)
    assert [row['start_phase'] for row in memory if row['step'] == 1] == ['round_start']*3
    original = tensor_state_hash(e.state['parameters'])
    reveals = e.ledger.events.copy()
    restored = experiment(directory, toy_bank, resume=True, arm=arm)
    restored.run()
    assert tensor_state_hash(restored.state['parameters']) == original
    assert restored.ledger.events == reveals
    if arm == 'R0':
        assert restored.state['phi'].eq(0).all()


@pytest.mark.parametrize('phase', ['selected', 'revealed', 'actual', 'feedback', 'committed', 'round_end'])
def test_crash_recovery_keeps_choice_slots_rng_and_model(toy_bank, tmp_path, phase):
    clean = experiment(tmp_path/'clean', toy_bank, smoke=True)
    clean.run()
    class Crash(Exception): pass
    def interrupt(current):
        if current == phase:
            raise Crash()
    with pytest.raises(Crash):
        experiment(tmp_path/'crash', toy_bank, smoke=True, after_save=interrupt).run()
    recovered = experiment(tmp_path/'crash', toy_bank, smoke=True, resume=True)
    recovered.run()
    assert recovered.state['steps'] == clean.state['steps']
    assert tensor_state_hash(recovered.state['parameters']) == tensor_state_hash(clean.state['parameters'])
    assert recovered.ledger.charges == clean.ledger.charges


def test_crash_after_durable_reveal_does_not_double_charge(toy_bank, tmp_path, monkeypatch):
    clean = experiment(tmp_path/'clean', toy_bank)
    clean.run()
    original = Ledger.settle
    class Crash(Exception): pass
    def fail(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        if result:
            raise Crash('lost process after fsync')
        return result
    with monkeypatch.context() as m:
        m.setattr(Ledger, 'settle', fail)
        with pytest.raises(Crash):
            experiment(tmp_path/'crash', toy_bank).run()
    recovered = experiment(tmp_path/'crash', toy_bank, resume=True)
    recovered.run()
    assert recovered.state['steps'] == clean.state['steps']
    assert recovered.ledger.charges == clean.ledger.charges


def test_checkpoint_and_ledger_corruption_rejected(toy_bank, tmp_path):
    e = experiment(tmp_path/'run', toy_bank, smoke=True)
    e.run()
    changed = dict(e.manifest, config_hash='changed')
    with pytest.raises(ValueError, match='binding mismatch'):
        StateStore(e.directory/'recovery', changed).load(e.ledger)
    ptr = json.loads(e.store.pointer.read_text())
    path = e.store.directory/ptr['file']
    with path.open('ab') as stream:
        stream.write(b'corruption')
    with pytest.raises(ValueError, match='binding mismatch'):
        e.store.load(e.ledger)
    ledger = Ledger(10, tmp_path/'bad.jsonl'); ledger.reserve('q', 5)
    raw = ledger.path.read_text().replace('"cap": 5', '"cap": 4')
    ledger.path.write_text(raw)
    with pytest.raises(LedgerError, match='hash/sequence'):
        Ledger.resume(10, ledger.path)


def test_hf_generate_matches_teacher_forcing_on_local_tiny_model():
    from transformers import GPT2Config, GPT2LMHeadModel
    from peft import LoraConfig, get_peft_model
    torch.manual_seed(8)
    model = GPT2LMHeadModel(GPT2Config(vocab_size=4, n_positions=128, n_embd=8, n_layer=1,
                                     n_head=1, eos_token_id=3, bos_token_id=0, attn_implementation='eager'))
    model = get_peft_model(model, LoraConfig(r=2, lora_alpha=2, lora_dropout=0., task_type='CAUSAL_LM',
                                           target_modules=['c_attn'])).eval()
    backend = HFGenerateBackend(model, TinyTokenizer(), base_checkpoint_hash='random-tiny', harness_hash='test',
                                 tokenizer_hash='test', max_action_tokens=80, max_context_tokens=128)
    params = snapshot(lora_parameters(model))
    original = tensor_state_hash(lora_parameters(model))
    rng = torch.Generator().manual_seed(12)
    for _ in range(3):
        action = backend.sample_action('prompt', params, rng)
        assert float(backend.score_action(action, params)) == pytest.approx(action.generation_logprob, abs=1e-5)
    assert tensor_state_hash(lora_parameters(model)) == original


def test_streamed_sigmoid_vjp_matches_general_autograd_and_finite_difference(toy_bank):
    b = tiny_backend(); p = snapshot(lora_parameters(b.model))
    state = next(iter(toy_bank[3].values()))
    action = b.sample_action(state.prompt, p, torch.Generator().manual_seed(2))
    source = SourceSample(Behavior(state, action.text), b.identity(p), action.action_ids, 3, action.generation_logprob)
    targets = [(source, Behavior(state, 'b')), (source, None)]
    chi = torch.tensor([[1., .2], [.3, 1.]], dtype=DT)
    phi = torch.tensor([.1, -.2], dtype=DT, requires_grad=True)
    step = FrozenStep({n: torch.ones_like(t) for n,t in p.items()}, .1, 'r1', {})
    def loss(params, control):
        return sum(slot_loss(t, c, control, b, params) for t,c in zip(targets,chi))/len(targets)
    def reward(params):
        return params['lora_w'][1].softmax(0)[1]
    updated = snapshot(step.update(p, gradients(loss(p,phi),p)))
    gj = gradients(reward(updated),updated)
    fb = ReturnGradient(gj, tensor_state_hash(updated), {})
    streamed = streamed_gate_vjp(targets,chi,phi,b,p,step,fb)
    torch.testing.assert_close(streamed, gate_vjp(loss(p,phi),p,phi,step,gj), rtol=1e-10, atol=1e-12)
    def outer(control):
        theta=snapshot(p)
        return reward(step.update(theta,gradients(loss(theta,control),theta)))
    fd=[]
    for i in range(2):
        d=torch.zeros_like(phi); d[i]=1e-5
        fd.append((outer(phi+d)-outer(phi-d))/2e-5)
    torch.testing.assert_close(streamed,torch.stack(fd),rtol=1e-7,atol=1e-10)


@pytest.mark.parametrize('arm', ['R0','R1'])
def test_fixed_collection_retrains_from_initial_and_empty_collection_is_legal(toy_bank, tmp_path, arm):
    config=engine_config()
    ids=[r.spec.query_id for r in toy_bank[1]]
    config.update(mode='fixed_evidence',fixed_evidence_ids=ids,
                  fixed_initial_parameter_hash=tensor_state_hash(lora_parameters(tiny_backend().model)))
    e=experiment(tmp_path/'fixed',toy_bank,arm=arm,config=config)
    e.run()
    assert e.ledger.owned_ids==set(ids) and len(e.state['steps'])==36
    assert all(s['fixed_evidence'] and s['selected'] is None for s in e.state['steps'])
    assert len([ev for ev in e.ledger.events if ev['kind']=='reveal'])==len(ids)
    config.update(fixed_evidence_ids=[])
    empty=experiment(tmp_path/'empty',toy_bank,arm=arm,config=config,smoke=True)
    empty.run()
    assert empty.ledger.spent==0 and empty.state['steps'][0]['exact_noop']


def test_scalar_gate_override_and_round_barrier_resume(toy_bank, tmp_path):
    config=engine_config(); config['gate']='scalar_sigmoid'
    e=experiment(tmp_path/'scalar',toy_bank,config=config)
    first=e.run(stop_after_round=1)
    assert first=={'round_ready':1,'complete':False} and len(e.state['steps'])==12
    restored=experiment(tmp_path/'scalar',toy_bank,resume=True,config=config)
    restored.run(stop_after_round=2)
    assert len(restored.state['steps'])==24 and restored.state['phi'].shape==(1,)
    again=experiment(tmp_path/'scalar',toy_bank,resume=True,config=config)
    again.run(stop_after_round=3)
    assert again.state['phase']=='complete'


def test_torn_ledger_tail_and_resume_before_initial_snapshot(tmp_path, toy_bank):
    ledger=Ledger(100,tmp_path/'ledger.jsonl'); ledger.reserve('q',10)
    with ledger.path.open('ab') as stream: stream.write(b'{"kind":"reveal"')
    restored=Ledger.resume(100,ledger.path)
    assert restored.reservations=={'q':10} and restored.spent==0
    assert ledger.path.with_suffix('.torn').exists()
    e=experiment(tmp_path/'initial-crash',toy_bank,resume=True,smoke=True)
    e.run()
    assert_run_invariants(e.state,e.ledger,complete=True)


def test_budget_report_uses_actual_spend_and_refuses_tampered_checkpoint(tmp_path, toy_bank):
    e=experiment(tmp_path/'run',toy_bank,smoke=True)
    atomic_json(e.directory/'manifest.json',e.manifest)
    e.run()
    e.journal.append('evaluation_lock_wait', round=1, idle_seconds=12., accounting='idle',
                     gpu_seconds=0., gpu_reserved_seconds=0., status='timeout')
    e.journal.append('evaluation_lock_wait', round=2, idle_seconds=99., accounting='idle',
                     gpu_seconds=0., gpu_reserved_seconds=0., status='acquired')
    rows=report([e.directory],tmp_path/'report')
    assert len(rows)==1 and rows[0]['evaluation_status']=='missing'
    assert rows[0]['actual_spend_x']==e.ledger.spent
    assert rows[0]['actual_spend_x'] < rows[0]['authorized_cap_budget']
    assert rows[0]['evaluation_lock_idle_seconds'] == 12. and rows[0]['gpu_seconds'] == 0.
    adapter=e.directory/'round-1/lora/adapter_model.safetensors'
    with adapter.open('ab') as stream: stream.write(b'bad')
    with pytest.raises(ValueError,match='artifacts changed'):
        report([e.directory],tmp_path/'bad-report')


@pytest.mark.parametrize('reuse', [False, True])
def test_capped_rollouts_survive_runner_ledger_trajectory_and_report(tmp_path, toy_bank, monkeypatch, reuse):
    original = tiny_backend
    def capped_backend():
        backend = original()
        backend.max_action_tokens = 1
        lora_parameters(backend.model)['lora_w'].data[:, 3] = -10000.
        return backend
    monkeypatch.setattr(sys.modules[__name__], 'tiny_backend', capped_backend)
    e = experiment(tmp_path/'run', toy_bank, smoke=True)
    if reuse:
        e.ceilings[:] = [0, 0, 0]
        e.ledger.budget = 0
    atomic_json(e.directory/'manifest.json', e.manifest)
    e.run()
    rollouts = [r for r in e.journal.events if r['kind'] == 'feedback_rollout']
    assert rollouts and all(r['rollout']['truncated'] for r in rollouts)
    assert all(r['rollout']['actions'][0]['truncated'] for r in rollouts)
    window = e.state['steps'][0]['truncation']
    assert window['actual_feedback_reused'] == reuse
    assert window['truncated_rollouts'] == window['truncated_actions'] == len(rollouts)
    assert window['reference']['truncated_rollouts'] == len(rollouts) // (1 if reuse else 2)
    assert all(src.truncated for sources in e.state['source_cache'].values() for src in sources)
    trajectory = json.loads((e.directory/'trajectory.json').read_text())
    assert trajectory['steps'][0]['truncation'] == window
    assert any(r['kind'] == 'window_truncation' and r['truncated_rollouts'] == len(rollouts) for r in e.journal.events)
    row = report([e.directory], tmp_path/'report')[0]
    assert row['sampled_truncated_rollouts'] == row['committed_truncated_rollouts'] == len(rollouts)
    assert row['truncation_by_window'][0]['truncated_actions'] == len(rollouts)
    assert {'source_sampling', 'source_preconditioner'} <= row['subphase_memory_peaks'].keys()
    assert 'Truncation per committed decision window' in (tmp_path/'report/budget_curve.md').read_text()


def test_state_batch_size_preserves_samples_gradients_pilot_and_updates(tmp_path, toy_bank):
    engines = []
    for size in (1, 3):
        config = engine_config() | {'max_state_batch_size': size}
        engine = experiment(tmp_path/f'batch-{size}', toy_bank, smoke=True, config=config)
        engine.run()
        engines.append(engine)
    first, second = engines
    assert first.state['steps'] == second.state['steps']
    assert first.state['source_cache'] == second.state['source_cache']
    assert first.state['eta'] == second.state['eta']
    assert tensor_state_hash(first.state['parameters']) == tensor_state_hash(second.state['parameters'])
    assert tensor_state_hash(first.state['diagonal']) == tensor_state_hash(second.state['diagonal'])
    torch.testing.assert_close(first.state['phi'], second.state['phi'], rtol=0, atol=0)
    torch.testing.assert_close(first.sampling_rng.get_state(), second.sampling_rng.get_state(), rtol=0, atol=0)
