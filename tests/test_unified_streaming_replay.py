"""Unified D14 parity using a tiny CPU policy and deterministic V0 publishing."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from bfas.behavior.deltas import tensor_state_hash
from bfas.rtd import cli, streaming_replay
from bfas.rtd.conventions import hashed_step
from bfas.rtd.identity import verified_checkpoint
from bfas.rtd.persistence import ComputeJournal, atomic_json, file_hash, manifest_hash
from bfas.rtd.unified.arms import ARMS
from test_rtd_checks import toy_bank
from test_rtd_manifest_tolerance import manifest_inputs
from test_rtd_v11_alpha_d import engine, finish_step
from test_rtd_v11_streaming_replay import Clock, schedule
from test_unified_p1 import p1_engine


def replay_fields(manifest):
    return {k: v for k, v in manifest.items() if k.startswith('replay_')}


@pytest.mark.parametrize('arm', list(ARMS))
def test_unified_streaming_matches_complete_replay_all_arms(toy_bank, tmp_path, monkeypatch, arm):
    clock = Clock(monkeypatch)
    v0 = engine(tmp_path/'V0', toy_bank, arm='V0', slots_per_step=2, max_new_packages_per_window=1)
    v0.run()
    final = schedule(v0)
    growing = tmp_path/'growing.json'
    atomic_json(growing, final | dict(steps=[], complete=False))
    target = p1_engine(tmp_path/'streaming', toy_bank, arm, replay_schedule=growing,
        replay_mode='streaming', replay_poll_seconds=2., replay_timeout_seconds=10.)

    def publish():
        if len(clock.sleeps) == 1:
            assert not target.state['steps'] and not target.ledger.charges
            atomic_json(growing, final | dict(complete=False))
        else:
            # The committed exposure is trained while the producer is still live.
            assert len(target.state['steps']) == 1
            assert not (target.directory/'round-1').exists()
            atomic_json(growing, final)

    clock.advance = publish
    target.run()
    complete = p1_engine(tmp_path/'complete', toy_bank, arm, replay_schedule=v0.directory)
    complete.run()
    exposures = lambda e: [r['exposure_schedule'] for r in e.state['steps']]
    assert exposures(target) == exposures(complete) == exposures(v0)
    assert target.ledger.charges == complete.ledger.charges == v0.ledger.charges
    assert replay_fields(target.manifest) == replay_fields(complete.manifest)
    assert target.manifest['replay_mode'] == 'complete'
    assert target.manifest['config']['replay_mode'] == 'streaming'
    assert target.manifest['replay_schedule_hash'] == file_hash(growing)
    assert len(target.manifest['replay_consumed_steps']) == 1
    assert tensor_state_hash(target.state['parameters']) == tensor_state_hash(complete.state['parameters'])
    assert clock.sleeps == [2., 2.]
    assert target.store.binding == manifest_hash(target.manifest)
    verified_checkpoint(target.directory, target.manifest, 1)


def test_D3_streaming_growing_schedule_two_rounds_and_resume(toy_bank, tmp_path, monkeypatch):
    clock = Clock(monkeypatch)
    # A V0 stub finishes first so the complete and incremental readers can be
    # compared against the exact same producer bytes without threads or GPUs.
    v0 = engine(tmp_path/'V0', toy_bank, arm='V0', smoke=False,
        slots_per_step=2, max_new_packages_per_window=1)
    freeze = v0.alpha_freeze

    def nonuniform(records, *, role, **kwargs):
        if role in {'commit', 'virtual_reference'}:
            kwargs['weights'] = [.25, .75]
        return freeze(records, role=role, **kwargs)

    v0.alpha_freeze = nonuniform
    v0.run()
    final = schedule(v0)
    growing = tmp_path/'growing.json'
    atomic_json(growing, final | dict(steps=[], complete=False))
    target = p1_engine(tmp_path/'streaming', toy_bank, replay_schedule=growing, smoke=False,
        replay_mode='streaming', replay_poll_seconds=.25, replay_timeout_seconds=30.)
    published = 0

    def publish():
        nonlocal published
        assert len(target.state['steps']) == published
        if published < len(final['steps']):
            published += 1
            atomic_json(growing, final | dict(steps=final['steps'][:published], complete=False))
        else:
            assert not (target.directory/'round-2').exists()
            atomic_json(growing, final)

    clock.advance = publish
    target.run(stop_after_round=1)
    assert len(target.state['steps']) == published == 12
    checkpoint = verified_checkpoint(target.directory, target.manifest, 1)
    target = p1_engine(target.directory, toy_bank, resume=True, smoke=False)
    target.run()
    complete = p1_engine(tmp_path/'complete', toy_bank, replay_schedule=v0.directory, smoke=False)
    complete.run()
    exposures = lambda e: [r['exposure_schedule'] for r in e.state['steps']]
    assert exposures(target) == exposures(complete) == exposures(v0)
    assert target.ledger.charges == complete.ledger.charges == v0.ledger.charges
    assert len(target.ledger.charges) > 1
    assert replay_fields(target.manifest) == replay_fields(complete.manifest)
    assert target.manifest['replay_schedule_identity'] == final['identity']
    assert target.manifest['replay_consumed_steps'] == [
        {k: r[k] for k in ('round', 'step', 'content_hash')} for r in final['steps']]
    draws = lambda e: [ev['sample_hashes'] for ev in e.journal.events
        if ev['kind'] == 'alpha_d_source_pair' and ev['role'] == 'commit']
    assert draws(target) == draws(complete)
    assert tensor_state_hash(target.state['parameters']) == tensor_state_hash(complete.state['parameters'])
    assert clock.sleeps == [.25]*25
    assert verified_checkpoint(target.directory, target.manifest, 1) == checkpoint
    verified_checkpoint(target.directory, target.manifest, 2)
    restored = p1_engine(target.directory, toy_bank, resume=True, smoke=False)
    restored.run()
    assert restored.state['steps'] == target.state['steps']
    assert replay_fields(restored.manifest) == replay_fields(target.manifest)


@pytest.mark.parametrize('arm', list(ARMS))
def test_unified_streaming_startup_accepts_unpublished_schedule_and_keeps_guard(tmp_path, arm):
    path = Path('configs/rtd/unified_bfcl_gemma4.yaml')
    config, selected = cli.startup_config(path, arm, tmp_path, smoke=False, replay_mode='streaming')
    assert selected == arm
    assert config['replay_schedule'] == str(tmp_path/'exposure_schedule.json')
    assert config['replay_poll_seconds'] == 60.
    assert config['replay_timeout_seconds'] == 36*3600.
    assert 'replay_schedule_hash' not in config
    assert cli.validate_config(config) == config
    with pytest.raises(ValueError, match='requires --replay-schedule'):
        cli.startup_config(path, arm, smoke=False)
    with pytest.raises(ValueError, match='requires --replay-schedule'):
        cli.startup_config(path, arm, smoke=False, replay_mode='streaming')


@pytest.mark.parametrize('key', ['replay_poll_seconds', 'replay_timeout_seconds'])
@pytest.mark.parametrize('value', [0, -1, float('nan'), float('inf'), True])
def test_unified_streaming_invalid_wait(tmp_path, key, value):
    with pytest.raises(ValueError, match='positive finite seconds'):
        cli.load_config('configs/rtd/unified_bfcl_gemma4.yaml', replay_schedule=tmp_path,
            replay_mode='streaming', **{key: value})


def test_unified_streaming_manifest_before_publication(tmp_path, manifest_inputs, monkeypatch):
    from bfas.rtd.identity import validate_resume
    root = Path(__file__).resolve().parents[1]
    with monkeypatch.context() as m:
        m.setattr(cli, 'ROOT', root)
        config = cli.load_config(root/'configs/rtd/unified_bfcl_gemma4.yaml', replay_schedule=tmp_path,
            replay_mode='streaming')
    config.update(student=manifest_inputs.config['student'], student_call_format='qwen')
    audit = manifest_inputs.audit | dict(budget_denominator=100, cost_scope='cached content',
        cap_certificate_sha256='cert', public_cost_assumption='class cap', m=40)
    manifest = cli.make_manifest(config, 'D3', audit)
    assert manifest['replay_mode'] == 'streaming'
    assert manifest['replay_consumed_steps'] == [] and manifest['replay_schedule_hash'] is None
    assert cli.resume_config(None, manifest) == config
    completed = manifest | dict(replay_mode='complete', replay_schedule_identity={'frozen': 'identity'},
        replay_consumed_steps=[dict(round=1, step=1, content_hash='consumed')], replay_schedule_hash='final')
    validate_resume(tmp_path, tmp_path, completed, manifest)
    with pytest.raises(ValueError, match='resume config/data/base metadata changed'):
        validate_resume(tmp_path, tmp_path, completed, manifest | {'config_hash': 'changed'})


def test_unified_streaming_timeout_resume_and_consumed_tamper(toy_bank, tmp_path, monkeypatch):
    clock = Clock(monkeypatch)
    v0 = engine(tmp_path/'V0', toy_bank, arm='V0', slots_per_step=2, max_new_packages_per_window=1)
    with pytest.raises(TimeoutError, match='resume D3'):
        p1_engine(tmp_path/'D3', toy_bank, replay_schedule=v0.directory,
            replay_mode='streaming', replay_poll_seconds=2., replay_timeout_seconds=5.)
    assert clock.sleeps == [2., 2., 1.]
    assert (tmp_path/'D3/recovery/latest.json').exists()
    v0.run()
    target = p1_engine(tmp_path/'D3', toy_bank, resume=True)
    target.round_start()
    finish_step(target)
    changed = schedule(v0)
    changed['steps'][0]['records'][0]['weight'] += .125
    changed['steps'][0] = hashed_step(changed['steps'][0])
    atomic_json(v0.directory/'exposure_schedule.json', changed)
    with pytest.raises(ValueError, match='consumed step changed'):
        target.run()
    assert not (target.directory/'round-1').exists()
    before = file_hash(target.store.pointer)
    with pytest.raises(ValueError, match='consumed step changed'):
        p1_engine(target.directory, toy_bank, resume=True)
    assert file_hash(target.store.pointer) == before


def test_unified_complete_rejects_incomplete_schedule_and_streaming_rebinding(toy_bank, tmp_path):
    v0 = engine(tmp_path/'V0', toy_bank, arm='V0', slots_per_step=2, max_new_packages_per_window=1)
    v0.run()
    data = schedule(v0)
    atomic_json(v0.directory/'exposure_schedule.json', data | dict(complete=False))
    with pytest.raises(ValueError, match='completed V0'):
        p1_engine(tmp_path/'D3', toy_bank, replay_schedule=v0.directory)
    config = cli.load_config('configs/rtd/unified_bfcl_gemma4.yaml', replay_schedule=v0.directory)
    with pytest.raises(ValueError, match='complete-mode config hash'):
        cli.validate_config(config, replay_mode='streaming')
    with pytest.raises(ValueError, match='complete or streaming'):
        cli.validate_config(config, replay_mode='invalid')


def test_unified_streaming_completed_coordinator_revalidates_before_evaluation(toy_bank, tmp_path, monkeypatch):
    from bfas.rtd import evaluation
    v0 = engine(tmp_path/'V0', toy_bank, arm='V0', slots_per_step=2, max_new_packages_per_window=1)
    v0.run()
    target = p1_engine(tmp_path/'D3', toy_bank, replay_schedule=v0.directory, replay_mode='streaming')
    target.run()
    assert target.manifest['replay_mode'] == 'complete'
    assert streaming_replay.enabled(target.manifest)
    path = v0.directory/'exposure_schedule.json'
    path.write_text(path.read_text() + '\n')
    monkeypatch.setattr(cli, 'hardware_identity', lambda: dict(uuid='mock', gpu='mock', memory=0))
    monkeypatch.setattr(cli, 'bank_audit', lambda config: {})
    monkeypatch.setattr(cli, 'make_manifest', lambda *a, **kw: target.manifest)
    monkeypatch.setattr(cli, 'validate_resume', lambda *a, **kw: None)
    monkeypatch.setattr(evaluation, 'evaluate', lambda *a, **kw: pytest.fail('must validate before evaluation'))
    args = SimpleNamespace(command='resume', run_dir=target.directory, arm='D3')
    with pytest.raises(ValueError, match='final whole-file hash changed'):
        cli.run_campaign(args, target.p1_config)
    assert ComputeJournal(target.directory/'compute.jsonl').events[-1]['kind'] == 'replay_schedule_diff'
