"""D7 frozen conventions: real CPU executor, independent schedule/ledger oracles."""
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest
import torch

from bfas.behavior.deltas import tensor_state_hash
from bfas.rtd.alpha_d import BatchReference, FeedbackStatistic, SourcePair
from bfas.rtd.conventions import (ARMS, BASE_CHECKPOINT, CORE_CONTROL, ADAPTIVE_EFFECT,
    DEVELOPMENT, CERTIFICATION, arm_config, certification_metadata, fold_roles)
from bfas.rtd.functional_step import FrozenStep
from bfas.rtd.joint_surrogate import marginal_values
from bfas.rtd.ledger import Ledger, BudgetError
from bfas.rtd.persistence import digest
from bfas.rtd.return_gradient import ReturnGradient
from bfas.rtd.source_estimator import validate_source_config
from bfas.rtd.source_scoring import sampled_prefix_positions
from test_rtd_checks import toy_bank, engine_config
from test_rtd_manifest_tolerance import manifest_inputs
from test_rtd_v11_alpha_d import engine, finish_step, pair_problem

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('balanced', [False, True], ids=['checked-in-22-18', 'record-shape-20-20'])
def test_bfcl_record_folds_export_parent_hashes_in_manifest(manifest_inputs, balanced, monkeypatch):
    from bfas.rtd import cli

    support_path = ROOT/'configs/rtd/v1_bfcl_support.json'
    support = json.loads(support_path.read_text())
    if balanced:
        support = support | {'parents': [p | {'fold': i % 2} for i, p in enumerate(support['parents'])]}
        support_path = manifest_inputs.root/'support.json'
        support_path.write_text(json.dumps(support))
    parents = support['parents']
    assert support['m'] == len(parents) == 40
    expected = {str(f): sorted(p['parent_hash'] for p in parents if p['fold'] == f) for f in (0, 1)}
    assert [len(expected[str(f)]) for f in (0, 1)] == ([20, 20] if balanced else [22, 18])
    roles = fold_roles(parents)
    for f in (0, 1):
        assert roles[str(f)]['inner_parent_groups'] == expected[str(f)]
        assert roles[str(f)]['feedback_parent_groups'] == expected[str(1-f)]
    # Explicit BFCL assignments are authoritative even if they differ from parity.
    reassigned = [p | {'fold': 1-p['fold']} for p in parents]
    assert fold_roles(reassigned)['0'] == roles['1']

    with monkeypatch.context() as m:
        m.setattr(cli, 'ROOT', ROOT)
        config = cli.load_config(ROOT/'configs/rtd/v1_1_bfcl.yaml', arm='V0')
    config.update(student=manifest_inputs.config['student'], support_manifest=str(support_path))
    # This test covers the fold-role export only; the fixed task set (metrics_v11) needs the sealed support/bank pair.
    config['metrics_v11'] = dict(config.get('metrics_v11', {}), enabled=False)
    audit = manifest_inputs.audit | dict(budget_denominator=100, cost_scope='cached content',
        cap_certificate_sha256='cert', public_cost_assumption='class cap', m=40)
    manifest = cli.make_manifest(config, 'V0', audit, smoke=True)
    assert manifest['parent_group_roles_by_fold'] == roles


@pytest.mark.parametrize('fold', [-1, 2, '0', None, True])
def test_bfcl_record_invalid_fold_is_rejected(fold):
    with pytest.raises(ValueError, match='explicit fold'):
        fold_roles([dict(parent_hash='00', fold=fold)])


def test_complete_v0_v1_schedule_replay_keeps_windows_weights_repetitions_but_redraws(toy_bank, tmp_path):
    v0 = engine(tmp_path/'V0', toy_bank, arm='V0', smoke=False)
    # Exercise a nonuniform frozen schedule, not only the uniform default.
    freeze = v0.alpha_freeze
    def nonuniform(records, *, role, **kw):
        if role in {'commit', 'virtual_reference'}:
            kw['weights'] = [.1, .2, .3, .4]
        return freeze(records, role=role, **kw)
    v0.alpha_freeze = nonuniform
    v0.run()
    schedule = json.loads((v0.directory/'exposure_schedule.json').read_text())
    assert schedule['complete'] and len(schedule['steps']) == 24
    v1 = engine(tmp_path/'V1', toy_bank, arm='V1', smoke=False, replay_schedule=str(v0.directory))
    v1.run()
    actual = json.loads((v1.directory/'exposure_schedule.json').read_text())
    assert schedule['steps'] == actual['steps']
    assert v0.ledger.charges == v1.ledger.charges
    for s0, s1 in zip(v0.state['steps'], v1.state['steps']):
        assert s0['exposure_schedule'] == s1['exposure_schedule']
        assert [r['weight'] for r in s1['exposure_records']] == [.1, .2, .3, .4]
    hashes = lambda e: [r['sample_hashes'] for r in e.journal.events if r['kind'] == 'alpha_d_source_pair' and r['role'] == 'commit']
    assert hashes(v0) != hashes(v1)
    assert all(set(r) == {'query_id', 'record_index', 'state_hash', 'has_teacher', 'is_new', 'weight', 'repetition_count'}
               for step in schedule['steps'] for r in step['records'])
    # A resumed completed run neither changes the schedule nor charges again.
    restored = engine(v1.directory, toy_bank, arm='V1', smoke=False, resume=True, replay_schedule=str(v0.directory))
    restored.run()
    assert restored.state['steps'] == v1.state['steps']


def test_all_three_arms_execute_one_estimator_and_prefix_implementation(toy_bank, tmp_path, monkeypatch):
    from bfas.rtd import alpha_d, source_scoring
    calls, positions = {}, {}
    active = None
    estimate = alpha_d.estimate
    prefixes = source_scoring.sampled_prefix_positions
    def estimator(*a, **kw):
        calls[active] = calls.get(active, 0)+1
        return estimate(*a, **kw)
    def prefix(*a, **kw):
        positions[active] = positions.get(active, 0)+1
        return prefixes(*a, **kw)
    monkeypatch.setattr(alpha_d, 'estimate', estimator)
    monkeypatch.setattr(source_scoring, 'sampled_prefix_positions', prefix)
    for active in ('V0', 'V1', 'V2'):
        extra = {'replay_schedule': str(tmp_path/'V0')} if active == 'V1' else {}
        e = engine(tmp_path/active, toy_bank, arm=active, **extra)
        e.run()
        draws = [r for r in e.journal.events if r['kind'] == 'alpha_d_source_pair' and r['role'] == 'commit']
        assert len(draws) == 4
        assert all(r['temperature'] == r['top_p'] == 1. and not r['cache_reused'] for r in draws)
        assert all(r['rng_before'] != r['rng_after'] and len(r['sample_hashes']) == 2 for r in draws)
    assert set(calls) == set(positions) == {'V0', 'V1', 'V2'}
    assert all(calls[a] >= 8 and positions[a] >= 2*calls[a] for a in calls)


@pytest.mark.parametrize('arm', ['V0', 'V2'])
def test_cold_start_identity_and_pool_membership_after_commit_only(toy_bank, tmp_path, arm):
    e = engine(tmp_path/arm, toy_bank, arm=arm, smoke=False, slots_per_step=40)
    e.round_start(); e.step_start(); e.reference()
    while e.state['phase'] == 'selected':
        e.selected()
    pending = set(e.state['selected'])
    assert pending and e.ledger.owned_ids == pending and e.state['owned'] == []
    assert all(r.teacher is None for r in e.alpha_old_pool())
    e.revealed()
    assert e.state['owned'] == []
    assert tensor_state_hash(e.state['reference'].updated) == e.state['reference'].start_hash
    old = e.state['old_reference']
    assert old.alpha.tolist() == [0.]*40
    e.actual(); e.feedback_commit(); e.committed()
    assert set(e.state['owned']) == pending
    assert {r.query_id for r in e.alpha_old_pool() if r.teacher is not None} == pending
    first = e.state['steps'][0]
    assert first['exposure_units'] == 40 and first['source_actions'] == 80 and first['microbatches'] == 10
    assert first['teacher_evidence_units'] == len(pending)
    finish_step(e)
    second = e.state['steps'][1]
    assert not second['decision'] and second['selected'] == []
    assert all(r['query_id'] in pending | {None} and not r['is_new'] for r in second['exposure_records'])
    assert second['exposure_schedule']['pool_before'] == sorted(pending)


def test_no_purchase_is_full_40_unit_source_only_identity_without_forced_spend(toy_bank, tmp_path):
    e = engine(tmp_path/'zero', toy_bank, arm='V0', slots_per_step=40, max_new_packages_per_window=0)
    e.run()
    row = e.state['steps'][0]
    assert row['exposure_units'] == row['reference_pool_units'] == 40
    assert row['teacher_evidence_units'] == e.ledger.spent == 0 and row['exact_noop']
    assert row['alpha_d']['alpha'] == row['alpha_d']['d_star'] == [0.]*40
    gradients = [r for r in e.journal.events if r['kind'] == 'return_gradient']
    assert [r['role'] for r in gradients] == ['acquisition_reference_feedback', 'same_batch_reference_feedback', 'post_commit_feedback']
    assert gradients[0]['parameter_hash'] == gradients[1]['parameter_hash']


def test_virtual_and_actual_feedback_different_states_and_separate_compute(toy_bank, tmp_path):
    e = engine(tmp_path/'feedback', toy_bank, arm='V0')
    e.round_start(); e.step_start(); e.reference()
    while e.state['phase'] == 'selected':
        e.selected()
    e.revealed()
    ref_hash = e.state['reference'].reference_hash
    actual_hash = tensor_state_hash(e.state['parameters'])
    assert ref_hash != actual_hash
    assert e.state['d_feedback'].parameter_hash == tensor_state_hash(e.state['d_reference'].theta0)
    e.actual()
    assert e.state['actual_feedback'].parameter_hash == actual_hash
    assert all(label.reference_hash == ref_hash for label in e.state['labels'].values())
    roles = ['acquisition_reference_feedback', 'same_batch_reference_feedback', 'post_commit_feedback']
    for role in roles:
        assert any(r['kind'] == 'feedback_rollout' and r['role'] == role for r in e.journal.events)
        assert any(r['kind'] == 'compute_begin' and r.get('operation') == role for r in e.journal.events)
    assert not any(r['kind'] == 'feedback_reused' for r in e.journal.events)


@pytest.mark.parametrize('substitute', ['alpha', 'insertion'])
def test_feedback_substitution_is_rejected_even_when_parameters_can_coincide(toy_bank, tmp_path, substitute):
    e = engine(tmp_path/substitute, toy_bank, arm='V0')
    e.round_start(); e.step_start(); e.reference()
    while e.state['phase'] == 'selected':
        e.selected()
    e.revealed()
    if substitute == 'alpha':
        e.feedback = lambda *a, **kw: e.state['reference_feedback']
        with pytest.raises(ValueError, match='post_commit_feedback'):
            e.actual()
    else:
        e.state['actual_feedback'] = e.state['reference_feedback']
        with pytest.raises(ValueError, match='substituted'):
            e.alpha_acquisition_labels()


def test_cached_source_objects_are_rejected_and_frozen_draws_are_journaled(toy_bank, tmp_path):
    e = engine(tmp_path/'cache', toy_bank, arm='V0')
    e.round_start()
    records = list(e.alpha_old_pool())[:1]
    cached = e.state['source_cache'][records[0].state.state_hash]
    def fake(*a, **kw):
        assert kw == dict(refresh=True, cache=False)
        torch.rand(1, generator=e.sampling_rng)
        return cached
    e.sample_state = fake
    with pytest.raises(AssertionError, match='cache'):
        e.alpha_draw_pairs(records, role='commit')


@pytest.mark.parametrize('arm', ['V0', 'V2'])
def test_solver_plan_is_filtered_by_shared_public_cap_executor(toy_bank, tmp_path, arm):
    e = engine(tmp_path/arm, toy_bank, arm=arm)
    e.ceilings[:] = [8192, 16384]
    e.ledger.budget = 8192
    e.round_start()
    e.state['cost_model'].model.information[-1] = -.999
    e.run()
    row = e.state['steps'][0]
    execution = next(r for r in e.journal.events if r['kind'] == 'acquisition_execution')
    assert len(execution['planned']) == 2 and len(execution['acquired']) == 1
    assert execution['planned'] == sorted(execution['planned'])
    assert execution['drops'][0]['reason'] == 'hard_cap_tail'
    assert execution['acquired'] == row['selected'] == row['trained_new_packages']
    assert {r['query_id'] for r in row['exposure_records'] if r['is_new']} == set(execution['acquired'])
    assert e.ledger.spent < e.ledger.budget and not e.ledger.reservations


def test_schedule_tampering_and_unfinished_replay_are_rejected(toy_bank, tmp_path):
    v0 = engine(tmp_path/'V0', toy_bank, arm='V0'); v0.run()
    path = v0.directory/'exposure_schedule.json'
    config, _ = arm_config(v0.config, 'V1', path)
    data = json.loads(path.read_text())
    data['complete'] = False
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='changed'):
        arm_config(config, 'V1')
    with pytest.raises(ValueError, match='completed V0'):
        engine(tmp_path/'V1', toy_bank, arm='V1', replay_schedule=str(path))


def test_shared_positions_include_eos_and_keep_cap_truncation():
    _, _, _, pairs, _ = pair_problem()
    source = pairs[0].sources[0]
    eos = source.eos_token_id
    for tokens, truncated in [((eos,), False), ((0, eos), False), ((0, 1), True)]:
        sample = replace(source, token_ids=tokens, truncated=truncated)
        assert sampled_prefix_positions(sample, (5, 6, 7)) == tuple(range(2, 2+len(tokens)))
    assert len(sampled_prefix_positions(sample, (5, 6))) == 2  # no fabricated EOS


def test_online_billed_output_is_settled_before_task_failure_and_local_precheck_is_free(tmp_path):
    ledger = Ledger(100, tmp_path/'online.jsonl')
    class FakeOnlineBroker:
        calls = 0
        def request(self):
            self.calls += 1
            return dict(cost=17, confidence='exact', usage={'output_tokens': 17}, output='invalid task result')
    broker = FakeOnlineBroker()
    def fail(*a):
        raise ValueError('task failed')
    with pytest.raises(ValueError):
        ledger.online_request('local', 30, precheck=fail, request=broker.request, validate=fail)
    assert broker.calls == ledger.spent == 0 and not ledger.events
    with pytest.raises(ValueError):
        ledger.online_request('online', 30, precheck=lambda: None, request=broker.request, validate=fail)
    assert broker.calls == 1 and ledger.spent == 17 and ledger.remaining == 83
    assert not ledger.reservations
    assert Ledger.resume(100, ledger.path).spent == 17


def test_joint_surrogate_retains_teacher_effect_when_all_d_zero():
    _, _, _, pairs, _ = pair_problem()
    pairs = tuple(replace(p, record=replace(p.record, query_id=q, is_new=True)) for p, q in zip(pairs, ['good', 'bad']))
    start = {'x': torch.tensor([0.], dtype=torch.float64)}
    step = FrozenStep({'x': torch.ones(1, dtype=torch.float64)}, 1., 'r1', {})
    zero = {'x': torch.zeros(1, dtype=torch.float64)}
    dirs = (zero, zero)
    weights = torch.tensor([.5, .5], dtype=torch.float64)
    alpha = weights.clone()
    batch = BatchReference('start', start, dirs, weights, alpha,
                           ({'x': torch.tensor([-1.])}, {'x': torch.tensor([1.])}))
    old = BatchReference('start', start, dirs, weights, torch.zeros(2), (zero, zero))
    fb = ReturnGradient({'x': torch.ones(1)}, tensor_state_hash(start), {})
    statistic = FeedbackStatistic(fb.gradient, (zero, zero), (0., 1.), ('a', 'a'), fb.parameter_hash, 1, 'smoke_zero')
    values, _ = marginal_values(pairs, batch, old, start, start, step, fb, statistic, revealed_ids={'good', 'bad'}, zero=True, d_lambda=0.)
    assert values['good'] == pytest.approx(.5) and values['bad'] == pytest.approx(-.5)
    # Redundant teacher steps incur a shared quadratic cost even when d=0.
    redundant = replace(batch, baseline_gradients=({'x': torch.tensor([-1.])},)*2)
    values, diagnostic = marginal_values(pairs, redundant, old, start, start, step, fb, statistic, revealed_ids={'good', 'bad'},
                                         zero=True, d_lambda=1.)
    assert diagnostic['full_set_value'] == pytest.approx(.5)
    assert list(values.values()) == pytest.approx([.125, .125])


def test_arm_labels_manifest_certification_and_legacy_hard2(manifest_inputs):
    from bfas.rtd import cli
    from bfas.rtd.alpha_d import ALPHA_D_DEFAULTS
    c = manifest_inputs.config | dict(protocol_version='1.1.0', rounds=2, exposure_slots_per_window=40,
        max_new_packages_per_window=20, slots_per_step=40) | ALPHA_D_DEFAULTS
    # Replace the v1.0 fixture's bank fractions with one absolute cap per round.
    c.pop('budget_checkpoints_bank_fraction')
    c['budget_checkpoints_tokens'] = [15000, 30000]
    for arm in ('V0', 'V2'):
        config, resolved = arm_config(c, arm)
        assert resolved == arm and all(config[k] == v for k, v in ARMS[arm].items())
        assert validate_source_config(config) == ('alpha_d', None, 2)
    with pytest.raises(ValueError, match='replay-schedule'):
        arm_config(c, 'V1')
    with pytest.raises(ValueError, match='legacy arms'):
        arm_config(c, 'R1')
    with pytest.raises(ValueError, match='v1.0 regression'):
        validate_source_config({'protocol_version': '1.1.0', 'source_estimator': 'hard2'})
    assert validate_source_config(engine_config())[0] == 'hard2'
    audit = manifest_inputs.audit | dict(budget_denominator=100, cost_scope='cached content',
                                         cap_certificate_sha256='cert', public_cost_assumption='class cap')
    config, _ = arm_config(c, 'V2')
    manifest = cli.make_manifest(config, 'V2', audit)
    assert manifest['distillation_protocol'] == 'alpha_d_rev31_same_batch'
    assert manifest['acquisition_label'].startswith('acquisition_surrogate')
    control_manifest = cli.make_manifest(config | {'acquisition_value_mode': 'independent'}, 'V2', audit)
    assert control_manifest['acquisition_label'] == 'independent_control'
    assert manifest['base_checkpoint_description'] == BASE_CHECKPOINT
    assert manifest['comparison_labels'] == {'V1−V0': ADAPTIVE_EFFECT, 'same_alpha_d_vs_zero': CORE_CONTROL}
    assert manifest['evaluation_label'] == DEVELOPMENT and manifest['certification_label'] == CERTIFICATION
    assert manifest['support_roles']['bfcl_m'] == 40
    assert manifest['support_roles']['alfworld_folds']['0']['inner'] == 79
    assert manifest['support_roles']['alfworld_folds']['1']['feedback'] == 79
    metadata = certification_metadata(frozen_checkpoint_hash='frozen', held_out_data_hash='heldout')
    assert metadata['bounds'] is None and metadata['status'] == 'plumbing_only'
    parents = json.loads((ROOT/'configs/rtd/v1_alfworld_support_c26.json').read_text())['parents']
    roles = fold_roles(parents)
    assert [len(roles[str(f)]['inner_parent_groups']) for f in (0, 1)] == [79, 56]
    assert roles['0']['inner_parent_groups'] == roles['1']['feedback_parent_groups']


def test_launcher_argv_cache_and_resume_relay_without_launching_gpu(tmp_path):
    script = ROOT/'scripts/rtd_v11_run_hpg.slurm'
    subprocess.run(['bash', '-n', str(script)], check=True)
    text = script.read_text()
    for directive in ('--mem=64G', '--cpus-per-task=8', '--time=24:00:00'):
        assert directive in text
    fake = tmp_path/'capture'
    fake.write_text('#!/usr/bin/env python3\nimport json,os,sys\nprint(json.dumps(dict(args=sys.argv[1:],hf=os.environ["HF_HUB_CACHE"],cache=os.environ["XDG_CACHE_HOME"])))\n')
    fake.chmod(0o755)
    env = os.environ | dict(SLURM_SUBMIT_DIR=str(ROOT), SLURM_JOB_ID='1', CUDA_VISIBLE_DEVICES='0',
        RTD_PYTHON=str(fake), RTD_RUN_DIR=str(tmp_path/'run'), RTD_EXTRA_ARGS='--training-worker --through-round 1')
    out = subprocess.check_output(['bash', str(script), 'V1', '--replay-schedule', str(tmp_path/'V0')], env=env, text=True)
    row = json.loads(out)
    assert row['args'][1:4] == ['run', '--arm', 'V1'] and '--replay-schedule' in row['args']
    assert '--smoke-deadline-seconds' not in row['args']
    assert row['hf'].endswith('/hf-cache/hub') and row['cache'].startswith(str(ROOT))
    env['RTD_COMMAND'] = 'resume'
    row = json.loads(subprocess.check_output(['bash', str(script), 'V1'], env=env, text=True))
    assert row['args'][1] == 'resume' and '--config' not in row['args']
    assert '--smoke-deadline-seconds' not in row['args']
    env['RTD_COMMAND'] = 'smoke'
    row = json.loads(subprocess.check_output(['bash', str(script), 'V0'], env=env, text=True))
    assert row['args'][1] == 'smoke'
    assert row['args'][row['args'].index('--smoke-deadline-seconds')+1] == '7200'
