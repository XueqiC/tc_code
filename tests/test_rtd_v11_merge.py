"""D1 estimators through the D2/D3 batch executor (CPU, temporary banks)."""
from pathlib import Path

import numpy as np
import pytest
import torch

from bfas.behavior.deltas import tensor_state_hash
from bfas.rtd.cli import load_config
from bfas.rtd.return_gradient import ReturnGradient
from bfas.rtd.selector import StudentSnapshot
from test_rtd_checks import experiment, toy_bank
from test_rtd_v11_posterior import batch_config


def merged_experiment(path, bank, estimator, mode='loo', *, resume=False):
    config = batch_config(rounds=2, source_estimator=estimator, cv_cs_mode=mode,
                          source_samples_per_state=3)
    if estimator == 'soft' or mode == 'fixed_one_minus_a':
        config['gate'] = 'scalar_sigmoid'
    engine = experiment(path, bank, config=config, smoke=True, resume=resume)
    # Expose the tiny model's existing positionwise logits as an output head,
    # exactly as in D1's runner test; no production model or GPU is loaded.
    model = engine.backend.model
    model.head = torch.nn.Identity()
    model.get_output_embeddings = lambda: model.head
    original = model.forward

    def forward(*args, **kwargs):
        out = original(*args, **kwargs)
        out.logits = model.head(out.logits)
        return out

    model.forward = forward
    return engine


@pytest.mark.parametrize('estimator,mode', [('soft', 'loo'), ('cv', 'loo'),
    ('cv', 'independent'), ('cv', 'fixed_one_minus_a')])
def test_batch_estimator_commit_and_vjp_use_identical_exposure_and_controls(toy_bank, tmp_path, estimator, mode):
    e = merged_experiment(tmp_path/'run', toy_bank, estimator, mode)
    e.round_start()
    s = e.state
    # Seed one paid inner package so this window has BOTH an old gradient
    # and a new package. The seed still goes through the ordinary ledger.
    view = StudentSnapshot(s['source_id'], frozenset(s['inner']))
    specs = e.broker.list_candidates(view, e.ledger.owned_ids, e.ledger.remaining)
    package = e.broker.acquire(specs[0].query_id)
    s['owned'].append(package.query_id)
    e.calibrate([package])
    with torch.no_grad():
        s['phi'].copy_(torch.linspace(-.3, .4, s['phi'].numel(), dtype=e.dtype))
    e.step_start()
    assert not s['old_noop']
    e.reference()
    while s['phase'] == 'selected':
        e.selected()
    e.revealed()
    assert len(s['selected']) == 1
    m, E = len(s['selected']), e.slots

    def actual_for(phi):
        saved = s['phi']
        s['phi'] = phi
        try:
            old = e.gradient(s['old_targets'], s['old_chi'], s['reference'].start,
                             controls=s.get('old_controls'))
            new = [e.gradient(s['batch_targets'][q], s['batch_chi'][q], s['reference'].start,
                              controls=s.get('batch_controls', {}).get(q)) for q in s['selected']]
            mixed = {n: (1-m/E)*old[n]+sum(g[n]/E for g in new) for n in old}
            return s['step_rule'].update(s['reference'].start, mixed)
        finally:
            s['phi'] = saved

    expected = actual_for(s['phi'])
    for n in expected:
        torch.testing.assert_close(s['actual'][n], expected[n], rtol=1e-12, atol=1e-12)
    # J(theta)=||theta||^2 gives a direct finite-difference oracle for the
    # complete mixed update, including c_s and its cross-draw derivatives.
    feedback = ReturnGradient({n: 2*p.detach() for n, p in s['actual'].items()},
                              tensor_state_hash(s['actual']), {})
    vjp = (1-m/E)*e.batch_gate_vjp(s['old_targets'], s['old_chi'], feedback,
                                  controls=s.get('old_controls'))
    for q in s['selected']:
        vjp += e.batch_gate_vjp(s['batch_targets'][q], s['batch_chi'][q], feedback,
                               controls=s.get('batch_controls', {}).get(q))/E
    direction = torch.linspace(.2, 1., s['phi'].numel(), dtype=e.dtype)
    delta = 1e-5

    def reward(phi):
        return sum(p.square().sum() for p in actual_for(phi).values())

    finite = (reward(s['phi']+delta*direction)-reward(s['phi']-delta*direction))/(2*delta)
    torch.testing.assert_close((vjp*direction).sum(), finite, atol=1e-8, rtol=1e-6)
    e.actual()
    e.feedback_commit()
    e.committed()
    assert not {'batch_controls', 'old_controls', 'draw_controls'} & s.keys()


@pytest.mark.parametrize('mode', ['loo', 'independent', 'fixed_one_minus_a'])
@pytest.mark.parametrize('phase,paid', [('selected', 1), ('revealed', 2), ('actual', 2), ('feedback', 2)])
def test_cv_batch_resume_preserves_pilot_pending_controls_and_posteriors(toy_bank, tmp_path, mode, phase, paid):
    clean = merged_experiment(tmp_path/'clean', toy_bank, 'cv', mode)
    clean.run()
    broken = merged_experiment(tmp_path/'broken', toy_bank, 'cv', mode)

    def stop(current):
        if current == phase and len(broken.ledger.owned_ids) == paid:
            raise RuntimeError('merge recovery probe')

    broken.after_save = stop
    with pytest.raises(RuntimeError, match='merge recovery probe'):
        broken.run()
    if 'batch_controls' in broken.state:
        for q, controls in broken.state['batch_controls'].items():
            assert len(controls) == len(broken.state['batch_targets'][q]) == 1
            source = broken.state['batch_targets'][q][0][0]
            if mode != 'fixed_one_minus_a':
                assert len(controls[0].features(source, mode)) == (2 if mode == 'loo' else 3)
    restored = merged_experiment(tmp_path/'broken', toy_bank, 'cv', mode, resume=True)
    restored.run()
    assert restored.state['steps'] == clean.state['steps']
    # Independent runs have different wall times and therefore different hash
    # chains. Compare every transaction field while retaining each run's audit.
    def transactions(ledger):
        return [{k: v for k, v in event.items() if k not in {'timestamp', 'previous_hash', 'event_hash'}}
                for event in ledger.events]
    assert transactions(restored.ledger) == transactions(clean.ledger)
    assert tensor_state_hash(restored.state['parameters']) == tensor_state_hash(clean.state['parameters'])
    np.testing.assert_array_equal(restored.state['posterior'].model.precision, clean.state['posterior'].model.precision)
    np.testing.assert_array_equal(restored.state['cost_model'].model.precision, clean.state['cost_model'].model.precision)


def test_canonical_v11_config_combines_source_and_batch_options():
    config = load_config(Path(__file__).resolve().parents[1]/'configs/rtd/v1_1_bfcl.yaml')
    assert config['rounds'] == 2
    assert config['budget_checkpoints_bank_fraction'] == [.1, .25]
    assert config['exposure_slots_per_window'] == config['slots_per_step'] == 40
    assert config['max_new_packages_per_window'] == 20
    assert config['replay_bank_path'] == 'data/rtd/v1_1_bfcl'
    assert config['acquisition_posterior_refresh'] == 'persistent'
    assert (config['source_estimator'], config['cv_cs_mode'], config['gate']) == ('cv', 'loo', 'linear_sigmoid')
