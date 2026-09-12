"""Batch math, guarded optimization, hard reservations and durable CPU windows."""
import copy
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from bfas.behavior.deltas import tensor_state_hash
from bfas.rtd.acquisition import BatchAcquisitionPolicy, CostRegressor, ValuePosterior, window_budget
from bfas.rtd.functional_step import FrozenStep, gradients, snapshot
from bfas.rtd.insertion import InsertionReference
from bfas.rtd.ledger import Ledger, BudgetError
from bfas.rtd.persistence import StateStore
from bfas.rtd.persistence import atomic_json
from bfas.rtd.evaluation import report
from bfas.rtd.broker import seal_bank
from bfas.cc_pairs import digest
from bfas.rtd.return_gradient import ReturnGradient
from bfas.rtd.selector import select_public_batch
from test_rtd_acquisition import spec, features
from test_rtd_checks import toy_bank, experiment, tiny_backend, TinySupport
from test_rtd_checks import DT
from bfas.rtd.runtime import slot_loss, streamed_gate_vjp
from bfas.rtd.return_gradient import gate_vjp
from bfas.rtd.transport import Behavior, SourceSample
from test_rtd_v11_posterior import batch_config


@pytest.mark.parametrize('m', [0, 1, 20])
def test_batch_same_start_finite_difference_exposure_and_empty(m):
    p = {'x': torch.tensor([.2, -.1], dtype=torch.float64, requires_grad=True)}
    step = FrozenStep({'x': torch.tensor([.7, 1.3], dtype=torch.float64)}, .1, 'r1', {})
    old = {'x': torch.tensor([.3, -.8], dtype=torch.float64)}
    ref = InsertionReference(p, None, step, old_slots=40, exposure_slots=40, old_gradient=old)
    feedback = ReturnGradient({'x': 2*ref.updated['x']}, ref.reference_hash, {})
    gs = {str(i): {'x': torch.tensor([i+.7, .4], dtype=torch.float64)} for i in range(m)}
    actual, labels, accounting = ref.insert_batch(gs, feedback, owned_before=(), pending_ids=set(gs),
        window_id='w1', variances=dict.fromkeys(gs, .1))
    expected = step.update(p, {'x': (1-m/40)*old['x']+sum(g['x'] for g in gs.values())/40})
    torch.testing.assert_close(actual['x'], expected['x'])
    assert accounting['raw_new_slots'] == accounting['weighted_new_slots'] == m
    assert accounting['weighted_old_slots'] + accounting['weighted_new_slots'] == 40
    assert accounting['old_coefficient'] >= 0
    def reward(eps):
        g = {'x': old['x']+eps*sum((g['x']-old['x'])/40 for g in gs.values())}
        return float(step.update(p, g)['x'].square().sum())
    finite_difference = (reward(1e-5)-reward(-1e-5))/(2e-5)
    assert sum(v.value/40 for v in labels.values()) == pytest.approx(finite_difference, abs=1e-9)
    assert all(v.position_share == .025 for v in labels.values())
    if not m:
        assert tensor_state_hash(actual) == ref.reference_hash
    else:
        with pytest.raises(ValueError, match='one matching'):
            ref.insert_batch(gs, feedback, owned_before=(), pending_ids=set(gs), window_id='w1', variances=dict.fromkeys(gs, .1))


def test_batch_random_fills_k_without_half_empty_and_learned_can_buy_nothing():
    p = ValuePosterior(2, round_id='r1')
    policy = BatchAcquisitionPolicy(p, CostRegressor(2), rng=np.random.default_rng(5))
    specs = [spec(str(i), 1) for i in range(100)]
    rows = {c.query_id: features(c.query_id, values=(1., 0.)) for c in specs}
    chosen, trace = select_public_batch(specs, lambda public: policy.choose_batch(public, rows,
        remaining_budget=100, random_control=True))
    assert len(chosen) == len(set(chosen)) == 20 and all(t['kind'] == 'public' for t in trace)
    assert policy.last_decision['stop_reason'] == 'package_limit'
    p.model.observe((1., 0.), -10., noise_variance=1e-9)
    assert policy.choose_batch(specs, rows, remaining_budget=100) == []
    assert policy.last_decision['predicted_additive_gain'] == 0
    assert policy.last_decision['replay_prior_mass'] is None


def test_expected_cost_knapsack_and_public_caps_are_separate():
    p = ValuePosterior(2, round_id='r1')
    p.model.observe((1., 0.), 5., noise_variance=1e-9)
    c = CostRegressor(2)
    policy = BatchAcquisitionPolicy(p, c, rng=np.random.default_rng(1))
    specs = [spec('a', 2), spec('b', 2), spec('too_expensive', 5)]
    rows = {s.query_id: features(s.query_id, values=(1., 0.)) for s in specs}
    assert len(policy.choose_batch(specs, rows, remaining_budget=3)) == 1
    assert policy.last_decision['budget_binding']
    assert policy.last_decision['predicted_cost'] <= 3
    assert 'too_expensive' not in policy.last_decision['query_ids']
    assert policy.last_decision['predicted_additive_gain'] > 0
    assert window_budget(5537, 4, [spec('demo', 2048), spec('item', 512)]) == 2048
    assert window_budget(8306, 4, [spec('demo', 2048)]) == 2077


def test_unified_batch_gate_vjp_matches_full_autograd(toy_bank):
    backend = tiny_backend()
    p = snapshot(dict(backend.model.named_parameters()))
    state = next(iter(toy_bank[3].values()))
    action = backend.sample_action(state.prompt, p, torch.Generator().manual_seed(2))
    source = SourceSample(Behavior(state, action.text), backend.identity(p), action.action_ids, 3,
                          action.generation_logprob, truncated=action.truncated)
    old = [(source, Behavior(state, 'a')), (source, None)]
    new = [[(source, Behavior(state, 'b'))], [(source, Behavior(state, 'c'))]]
    old_chi = torch.tensor([[1., .2], [.3, 1.]], dtype=DT)
    new_chi = [torch.tensor([[.7, .4]], dtype=DT), torch.tensor([[.1, .8]], dtype=DT)]
    phi = torch.tensor([.1, -.2], dtype=DT, requires_grad=True)
    step = FrozenStep({n: torch.ones_like(v) for n, v in p.items()}, .1, 'r1', {})
    def loss(targets, chi, params):
        return sum(slot_loss(t, c, phi, backend, params) for t, c in zip(targets, chi))/len(targets)
    def mixed(params):
        return .95*loss(old, old_chi, params)+sum(loss(t, c, params)/40 for t, c in zip(new, new_chi))
    updated = snapshot(step.update(p, gradients(mixed(p), p)))
    gj = gradients(updated['lora_w'][1].softmax(0)[1], updated)
    fb = ReturnGradient(gj, tensor_state_hash(updated), {})
    vjp = .95*streamed_gate_vjp(old, old_chi, phi, backend, p, step, fb)
    vjp += sum(streamed_gate_vjp(t, c, phi, backend, p, step, fb)/40 for t, c in zip(new, new_chi))
    torch.testing.assert_close(vjp, gate_vjp(mixed(p), p, phi, step, gj), rtol=1e-10, atol=1e-12)


def test_global_and_window_ledger_invariants_release_dependency_count_and_resume(tmp_path):
    ledger = Ledger(100, tmp_path/'ledger.jsonl')
    ledger.open_window('w', 50, 2)
    ledger.reserve('a', 40)
    assert ledger.spent + sum(ledger.reservations.values()) <= ledger.budget
    with pytest.raises(BudgetError):
        ledger.reserve('b', 11)
    ledger.settle('a', 5, confidence='exact', usage={})
    assert ledger.window_remaining == 45 and ledger.remaining == 95
    with pytest.raises(BudgetError, match='dependency'):
        ledger.reserve_chain({'b': 10, 'c': 10})
    assert not ledger.reservations
    ledger.reserve('b', 40); ledger.release('b')
    ledger.reserve('b', 40); ledger.settle('b', 0, confidence='exact', usage={})
    with pytest.raises(BudgetError):
        ledger.reserve('c', 0)  # free packages still consume K
    assert ledger.reserve('a', 40) is False
    ledger.close_window('w')
    ledger.authorize(200)
    restored = Ledger.resume(100, ledger.path)
    assert restored.charges == {'a': 5, 'b': 0} and restored.window is None
    assert restored.events == ledger.events and restored.remaining == 195


def test_v10_one_window_byte_identical_to_prechange_cpu_fixture(toy_bank, tmp_path):
    e = experiment(tmp_path/'v10', toy_bank, smoke=True); e.run()
    actual = json.dumps(e.state['steps'][0], sort_keys=True, indent=2)+'\n'
    assert actual == (Path(__file__).parent/'fixtures/rtd_v11/v10_window.json').read_text()


@pytest.mark.parametrize('phase,paid', [('reference', 0), ('selected', 0), ('selected', 1), ('selected', 2),
                                      ('revealed', 2), ('actual', 2), ('feedback', 2), ('committed', 2)])
def test_batch_resume_at_every_window_phase_preserves_draws_labels_and_one_commit(toy_bank, tmp_path, phase, paid):
    config = batch_config(rounds=2)
    clean = experiment(tmp_path/'clean', toy_bank, config=config, smoke=True); clean.run()
    broken = experiment(tmp_path/'broken', toy_bank, config=config, smoke=True)
    def stop(current):
        if current == phase and len(broken.ledger.owned_ids) == paid:
            raise RuntimeError('injected crash')
    broken.after_save = stop
    with pytest.raises(RuntimeError, match='injected'):
        broken.run()
    restored = experiment(tmp_path/'broken', toy_bank, config=config, smoke=True, resume=True)
    restored.run()
    assert restored.state['steps'] == clean.state['steps']
    assert restored.ledger.charges == clean.ledger.charges
    assert tensor_state_hash(restored.state['parameters']) == tensor_state_hash(clean.state['parameters'])
    np.testing.assert_array_equal(restored.state['posterior'].model.precision, clean.state['posterior'].model.precision)
    np.testing.assert_array_equal(restored.state['cost_model'].model.precision, clean.state['cost_model'].model.precision)
    assert restored.state['source_cache'] == clean.state['source_cache']
    assert len(restored.state['steps']) == 1


@pytest.mark.parametrize('crash_at', ['reserve', 'reveal', 'window_close'])
def test_resume_when_ledger_leads_checkpoint_only_current_transaction(toy_bank, tmp_path, monkeypatch, crash_at):
    config = batch_config(rounds=2)
    clean = experiment(tmp_path/'clean', toy_bank, config=config, smoke=True); clean.run()
    broken = experiment(tmp_path/'broken', toy_bank, config=config, smoke=True)
    method = {'reserve': 'reserve', 'reveal': 'settle', 'window_close': 'close_window'}[crash_at]
    original = getattr(broken.ledger, method)
    class PowerLoss(BaseException):
        pass
    def crash(*args, **kwargs):
        original(*args, **kwargs)
        raise PowerLoss()
    monkeypatch.setattr(broken.ledger, method, crash)
    with pytest.raises(PowerLoss):
        broken.run()
    restored = experiment(tmp_path/'broken', toy_bank, config=config, smoke=True, resume=True)
    restored.run()
    assert restored.state['steps'] == clean.state['steps']
    assert restored.ledger.charges == clean.ledger.charges


def test_recovery_rejects_a_future_planned_transaction(toy_bank, tmp_path):
    e = experiment(tmp_path/'run', toy_bank, config=batch_config(), smoke=True)
    def stop(phase):
        if phase == 'selected':
            raise RuntimeError('stop')
    e.after_save = stop
    with pytest.raises(RuntimeError):
        e.run()
    first, future = e.state['selected']
    assert e.state['transaction_query'] == first
    e.ledger.reserve(future, e.broker._records[future].spec.cost_upper_bound)
    with pytest.raises(ValueError, match='durably selected'):
        e.store.load(e.ledger)


def test_two_round_schedule_and_round_checkpoint_resume(toy_bank, tmp_path):
    config = batch_config(rounds=2)
    e = experiment(tmp_path/'run', toy_bank, config=config)
    e.run(stop_after_round=1)
    resumed = experiment(tmp_path/'run', toy_bank, config=config, resume=True)
    audit = resumed.run()
    assert audit['steps'] == 24 and audit['decision_windows'] == 8
    assert [c['round'] for c in resumed.state['checkpoints']] == [1, 2]
    assert not (e.directory/'round-3').exists()
    atomic_json(resumed.directory/'manifest.json', resumed.manifest)
    rows = report([resumed.directory], tmp_path/'report')
    assert len(rows) == 2 and rows[-1]['actual_spend_x'] == resumed.ledger.spent
    for row in resumed.state['steps']:
        if row['decision']:
            assert row['joint_vs_additive_error'] == pytest.approx(row['realised_joint_update_gain']-row['predicted_additive_gain'])
        if not row['selected']:
            assert row['actual_hash'] == row['reference_hash']


def test_default_forty_slot_window_can_commit_twenty_distinct_packages(toy_bank, tmp_path):
    _, original, payloads, states = toy_bank
    records, expanded = [], {}
    for record in original:
        for i in range(10):
            q = digest([record.spec.query_id, i])
            records.append(replace(record, spec=replace(record.spec, query_id=q)))
            expanded[q] = payloads[record.spec.query_id]
    bank = seal_bank(tmp_path/'large-bank', records, expanded)
    config = batch_config(exposure_slots_per_window=40, max_new_packages_per_window=20)
    e = experiment(tmp_path/'run', (bank, records, expanded, states), config=config, smoke=True,
                   budget_ceilings=[1000000, 2000000, 3000000])
    e.run()
    row, = e.state['steps']
    assert row['slots'] == 40 and len(row['selected']) == 20
    assert row['raw_new_slots'] == row['weighted_new_slots'] == 20
    assert row['old_coefficient'] == .5 and len(row['labels']) == 20
    assert all(len(v) == 1 for v in row['new_slots_by_query'].values())
    assert len(e.state['posterior'].observations) == len(e.ledger.owned_ids) == 20


def test_batch_hard_cap_tail_never_reveals_or_redraws_unaffordable_choice(toy_bank, tmp_path):
    e = experiment(tmp_path/'tail', toy_bank, config=batch_config(), smoke=True,
                   budget_ceilings=[8192, 16384, 32768])
    e.round_start(); e.step_start()
    # Simulate an optimistic cost posterior without reading any sealed usage.
    # Both public caps remain 8192 and must still be reserved.
    e.state['cost_model'].model.information[-1] = -0.999
    e.run()
    row, = e.state['steps']
    assert len(row['selection']['planned_selected']) == 2 and len(row['selected']) == 1
    assert row['stop_reason'] == 'hard_cap_tail' and row['budget_binding']
    assert len(e.ledger.owned_ids) == 1 and not e.ledger.reservations
