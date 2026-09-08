"""D14 CPU replay/integrity tests. Fake clock drives real tiny V0/V1 engines."""
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest
import torch

from bfas.behavior.deltas import tensor_state_hash
from bfas.rtd import cli, streaming_replay
from bfas.rtd.conventions import arm_config, hashed_step
from bfas.rtd.experiment import RTDExperiment
from bfas.rtd.identity import verified_checkpoint
from bfas.rtd.persistence import ComputeJournal, atomic_json, digest, file_hash, manifest_hash
from test_rtd_checks import toy_bank, tiny_backend
from test_rtd_manifest_tolerance import manifest_inputs
from test_rtd_v11_alpha_d import engine, finish_step


class Clock:
    def __init__(self, monkeypatch):
        self.now, self.sleeps, self.advance = 0., [], lambda: None
        # Replace this module's clock, not Python's global time module.
        monkeypatch.setattr(streaming_replay, 'time', self)

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds
        self.advance()


def schedule(v0):
    return json.loads((v0.directory/'exposure_schedule.json').read_text())


def publish(v0, data):
    atomic_json(v0.directory/'exposure_schedule.json', data)


def follower(path, toy_bank, v0, **options):
    return engine(path, toy_bank, arm='V1', smoke=v0.state['smoke'], replay_schedule=str(v0.directory),
                  replay_mode='streaming', **options)


def resume(e):
    manifest = json.loads((e.directory/'manifest.json').read_text())
    backend = tiny_backend()
    model = backend.model
    model.head = torch.nn.Identity()
    model.get_output_embeddings = lambda: model.head
    forward = model.forward
    def logits(*args, **kwargs):
        out = forward(*args, **kwargs)
        out.logits = model.head(out.logits)
        return out
    model.forward = logits
    return RTDExperiment(manifest['config'], manifest, e.directory, backend, e.support,
                         resume=True, smoke=e.state['smoke'])


def test_lag_one_real_run_matches_complete_replay_and_redraws(toy_bank, tmp_path, monkeypatch):
    clock = Clock(monkeypatch)
    v0 = engine(tmp_path/'V0', toy_bank, arm='V0', smoke=False)
    freeze = v0.alpha_freeze
    def nonuniform(records, *, role, **kwargs):
        if role in {'commit', 'virtual_reference'}:
            kwargs['weights'] = [.1, .2, .3, .4]
        return freeze(records, role=role, **kwargs)
    v0.alpha_freeze = nonuniform
    v0.round_start()
    finish_step(v0)
    v1 = follower(tmp_path/'V1', toy_bank, v0)
    assert not schedule(v0)['complete']
    v1.round_start()
    finish_step(v1)
    assert len(v1.state['steps']) == len(v0.state['steps']) == 1
    assert not clock.sleeps

    def next_v0_step():
        # V1 is waiting for exactly one next step; V0 may now commit it.
        assert len(v0.state['steps']) == len(v1.state['steps'])
        if v0.state['phase'] == 'round_end':
            v0.round_end()
            v0.round_start()
        finish_step(v0)
    clock.advance = next_v0_step
    v1.run(stop_after_round=1)
    assert len(v1.state['steps']) == 12
    first_checkpoint = verified_checkpoint(v1.directory, v1.manifest, 1)
    restored = resume(v1)
    v1 = restored
    restored.run()
    assert len(clock.sleeps) == 23 and set(clock.sleeps) == {60.}
    assert v1.state['phase'] == 'complete' and schedule(v0)['complete']
    assert schedule(v0)['steps'] == schedule(v1)['steps']
    assert v0.ledger.charges == v1.ledger.charges
    draws = lambda e: [x['sample_hashes'] for x in e.journal.events
                       if x['kind'] == 'alpha_d_source_pair' and x['role'] == 'commit']
    assert draws(v0) != draws(v1)
    manifest = json.loads((v1.directory/'manifest.json').read_text())
    assert manifest['replay_mode'] == 'streaming'
    assert manifest['replay_schedule_identity'] == schedule(v0)['identity']
    assert manifest['replay_schedule_hash'] == file_hash(v0.directory/'exposure_schedule.json')
    assert 'replay_schedule_hash' not in manifest['config']
    assert [r['content_hash'] for r in manifest['replay_consumed_steps']] == [r['content_hash'] for r in schedule(v0)['steps']]
    assert verified_checkpoint(v1.directory, manifest, 1) == first_checkpoint
    verified_checkpoint(v1.directory, manifest, 2)
    assert v1.store.binding == manifest_hash(manifest)
    waits = [e for e in v1.journal.events if e.get('operation') == 'replay_schedule_wait' and e['kind'] == 'compute_end']
    assert len(waits) == 23 and sum(e['wall_seconds'] for e in waits) == 23*60
    assert all(e['gpu_seconds'] == 0 and e['compute_type'] == 'idle' for e in waits)
    complete = engine(tmp_path/'complete', toy_bank, arm='V1', smoke=False, replay_schedule=str(v0.directory))
    complete.run()
    assert draws(complete) == draws(v1)
    assert tensor_state_hash(complete.state['parameters']) == tensor_state_hash(v1.state['parameters'])
    resumed_complete = resume(v1)
    resumed_complete.run()
    assert resumed_complete.state['steps'] == v1.state['steps']


@pytest.mark.parametrize('torn', [b'{"steps": [', b'\xff'])
def test_torn_read_retries_same_bytes_hash_without_binding_partial_file(toy_bank, tmp_path, monkeypatch, torn):
    clock = Clock(monkeypatch)
    v0 = engine(tmp_path/'V0', toy_bank, arm='V0'); v0.run()
    path = v0.directory/'exposure_schedule.json'
    original = Path.read_bytes
    reads = []
    def read(self):
        if self == path:
            reads.append(self)
            if len(reads) == 1:
                return torn
        return original(self)
    monkeypatch.setattr(Path, 'read_bytes', read)
    v1 = follower(tmp_path/'V1', toy_bank, v0)
    v1.run()
    assert clock.sleeps == [60.]
    assert v1.manifest['replay_schedule_hash'] == file_hash(path)
    assert any('torn read' in e.get('reason', '') for e in v1.journal.events)


def test_missing_first_export_waits_and_timeout_is_resumable(toy_bank, tmp_path, monkeypatch):
    clock = Clock(monkeypatch)
    v0 = engine(tmp_path/'V0', toy_bank, arm='V0')
    with pytest.raises(TimeoutError, match='V0 may have stopped.*resume V1'):
        follower(tmp_path/'V1', toy_bank, v0, replay_poll_seconds=2., replay_timeout_seconds=5.)
    assert clock.sleeps == [2., 2., 1.]
    directory = tmp_path/'V1'
    assert (directory/'recovery/latest.json').exists()
    v0.run()
    saved = json.loads((directory/'manifest.json').read_text())
    # Resume from the initial durable state, with a fresh per-attempt deadline.
    restored = resume(SimpleNamespace(directory=directory, support=v0.support, state={'smoke': True}))
    restored.run()
    assert restored.manifest['config'] == saved['config']
    assert restored.state['phase'] == 'complete'
    journal = ComputeJournal(directory/'compute.jsonl')
    assert len([e for e in journal.events if e['kind'] == 'replay_schedule_timeout']) == 1


@pytest.mark.parametrize('change', ['rehash', 'unhashed', 'identity', 'missing'])
def test_resume_revalidates_consumed_step_before_training(toy_bank, tmp_path, monkeypatch, change):
    Clock(monkeypatch)
    v0 = engine(tmp_path/'V0', toy_bank, arm='V0'); v0.run()
    v1 = follower(tmp_path/'V1', toy_bank, v0)
    v1.round_start(); v1.step_start()
    data = schedule(v0)
    if change in {'rehash', 'unhashed'}:
        data['steps'][0]['records'][0]['weight'] += .125
        if change == 'rehash':
            data['steps'][0] = hashed_step(data['steps'][0])
    elif change == 'identity':
        data['identity']['training_seed'] += 1
    else:
        data.update(steps=[], complete=False)
    publish(v0, data)
    before = file_hash(v1.store.pointer)
    with pytest.raises(ValueError, match='streaming replay divergence'):
        resume(v1)
    assert file_hash(v1.store.pointer) == before
    assert not v1.ledger.charges
    diff = ComputeJournal(v1.directory/'compute.jsonl').events[-1]
    assert diff['kind'] == 'replay_schedule_diff'
    assert diff['diff'][0]['expected'] != diff['diff'][0]['actual']


def test_final_barrier_detects_reexport_divergence_before_last_checkpoint(toy_bank, tmp_path, monkeypatch):
    Clock(monkeypatch)
    v0 = engine(tmp_path/'V0', toy_bank, arm='V0'); v0.run()
    v1 = follower(tmp_path/'V1', toy_bank, v0)
    v1.round_start(); finish_step(v1)
    data = schedule(v0)
    data['steps'][0]['reference_records'][0]['weight'] += .1
    data['steps'][0] = hashed_step(data['steps'][0])
    publish(v0, data)
    with pytest.raises(ValueError, match='consumed step changed'):
        v1.run()
    assert v1.state['phase'] == 'round_end'
    assert not (v1.directory/'round-1').exists()
    assert v1.manifest['replay_schedule_hash'] is None


def test_final_marker_timeout_then_resume_and_journal_leading_manifest(toy_bank, tmp_path, monkeypatch):
    clock = Clock(monkeypatch)
    v0 = engine(tmp_path/'V0', toy_bank, arm='V0'); v0.run()
    data = schedule(v0); data['complete'] = False; publish(v0, data)
    v1 = follower(tmp_path/'V1', toy_bank, v0, replay_poll_seconds=2., replay_timeout_seconds=5.)
    v1.round_start(); finish_step(v1)
    with pytest.raises(TimeoutError, match='final complete=true'):
        v1.run()
    assert clock.sleeps == [2., 2., 1.]
    assert v1.state['phase'] == 'round_end' and not (v1.directory/'round-1').exists()
    # Simulate crash between consumption event fsync and manifest replacement.
    v1.manifest['replay_consumed_steps'] = []
    atomic_json(v1.directory/'manifest.json', v1.manifest)
    data['complete'] = True; publish(v0, data)
    restored = resume(v1); restored.run()
    assert len(restored.manifest['replay_consumed_steps']) == 1
    assert restored.ledger.charges == v1.ledger.charges


@pytest.mark.parametrize('phase', ['reference', 'selected', 'revealed', 'actual', 'feedback', 'committed'])
def test_streaming_resume_each_durable_phase_has_same_sources_and_one_purchase(toy_bank, tmp_path, monkeypatch, phase):
    Clock(monkeypatch)
    v0 = engine(tmp_path/'V0', toy_bank, arm='V0'); v0.run()
    clean = follower(tmp_path/'clean', toy_bank, v0); clean.run()
    broken = follower(tmp_path/'broken', toy_bank, v0)
    def crash(current):
        if current == phase:
            raise RuntimeError('crash probe')
    broken.after_save = crash
    with pytest.raises(RuntimeError, match='crash probe'):
        broken.run()
    restored = resume(broken); restored.run()
    assert restored.state['steps'] == clean.state['steps']
    assert restored.ledger.charges == clean.ledger.charges
    assert tensor_state_hash(restored.state['parameters']) == tensor_state_hash(clean.state['parameters'])
    assert len(restored.manifest['replay_consumed_steps']) == 1


def test_v0_resume_reexports_identical_committed_hashes(toy_bank, tmp_path):
    v0 = engine(tmp_path/'V0', toy_bank, arm='V0', smoke=False)
    v0.round_start(); finish_step(v0)
    original = (v0.directory/'exposure_schedule.json').read_bytes()
    # Auxiliary schedule can be lost; durable V0 steps remain authoritative.
    (v0.directory/'exposure_schedule.json').unlink()
    restored = engine(v0.directory, toy_bank, arm='V0', smoke=False, resume=True)
    assert (v0.directory/'exposure_schedule.json').read_bytes() == original
    first = schedule(restored)['steps'][0]
    finish_step(restored)
    assert schedule(restored)['steps'][0] == first
    assert first['content_hash'] == digest(v0.state['steps'][0]['exposure_schedule'])
    assert 'content_hash' not in v0.state['steps'][0]['exposure_schedule']


def test_complete_mode_keeps_whole_file_binding_and_legacy_schedule(toy_bank, tmp_path):
    v0 = engine(tmp_path/'V0', toy_bank, arm='V0'); v0.run()
    data = schedule(v0)
    data['steps'] = [{k: v for k, v in row.items() if k != 'content_hash'} for row in data['steps']]
    publish(v0, data)
    v1 = engine(tmp_path/'V1', toy_bank, arm='V1', replay_schedule=str(v0.directory)); v1.run()
    assert v1.config['replay_schedule_hash'] == file_hash(v0.directory/'exposure_schedule.json')
    assert 'replay_mode' not in v1.config
    assert v1.store.binding == digest(v1.manifest)
    data['complete'] = False; publish(v0, data)
    with pytest.raises(ValueError, match='replay schedule changed'):
        arm_config(v1.config, 'V1')
    with pytest.raises(ValueError, match='completed V0'):
        engine(tmp_path/'incomplete', toy_bank, arm='V1', replay_schedule=str(v0.directory))


@pytest.mark.parametrize('key', ['replay_poll_seconds', 'replay_timeout_seconds'])
@pytest.mark.parametrize('bad', [0, -1, float('nan'), float('inf'), True])
def test_streaming_config_rejects_invalid_wait(key, bad, tmp_path):
    config = cli.load_config(cli.ROOT/'configs/rtd/v1_1_bfcl.yaml', arm='V0')
    with pytest.raises(ValueError, match='positive finite seconds'):
        arm_config(config, 'V1', tmp_path, replay_mode='streaming', **{key: bad})


def test_cli_config_manifest_and_resume_without_opening_schedule(tmp_path, manifest_inputs, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    with monkeypatch.context() as m:
        m.setattr(cli, 'ROOT', root)
        config = cli.load_config(root/'configs/rtd/v1_1_bfcl.yaml', arm='V1', replay_schedule=tmp_path,
                                 replay_mode='streaming')
    assert config['replay_poll_seconds'] == 60 and config['replay_timeout_seconds'] == 36*3600
    assert config['replay_schedule'] == str(tmp_path/'exposure_schedule.json')
    assert 'replay_schedule_hash' not in config
    config.update(student=manifest_inputs.config['student'], metrics_v11={'enabled': False})
    audit = manifest_inputs.audit | dict(budget_denominator=100, cost_scope='cached content',
        cap_certificate_sha256='cert', public_cost_assumption='class cap', m=40)
    manifest = cli.make_manifest(config, 'V1', audit)
    assert manifest['replay_mode'] == 'streaming'
    assert manifest['replay_consumed_steps'] == [] and manifest['replay_schedule_hash'] is None
    assert cli.resume_config(None, manifest) == config
    received = []
    monkeypatch.setattr(cli, 'run_command', lambda args: received.append(args))
    for command in ('run', 'smoke', 'resume'):
        cli.main([command, '--arm', 'V1', '--replay-schedule', str(tmp_path), '--replay-mode', 'streaming',
                  '--replay-poll-seconds', '2', '--replay-timeout-seconds', '5'])
    assert all(args.replay_mode == 'streaming' and args.replay_poll_seconds == 2 and args.replay_timeout_seconds == 5
               for args in received)


def test_overall_timeout_is_shared_across_steps_and_wait_never_bills_gpu(toy_bank, tmp_path, monkeypatch):
    clock = Clock(monkeypatch)
    v0 = engine(tmp_path/'V0', toy_bank, arm='V0', smoke=False)
    v0.round_start(); finish_step(v0); finish_step(v0)
    data = schedule(v0)
    publish(v0, data | {'steps': data['steps'][:1]})
    v1 = follower(tmp_path/'V1', toy_bank, v0, replay_poll_seconds=2., replay_timeout_seconds=5.)
    reader = v1.replay_schedule
    reader.get((1, 1))
    # Exercise the wait on a CUDA-accounting journal without any CUDA call.
    v1.journal.cuda = True
    clock.advance = lambda: publish(v0, data)
    reader.get((1, 2))
    with pytest.raises(TimeoutError, match='timed out after 5'):
        reader.get((1, 3))
    assert clock.sleeps == [2., 2., 1.]
    ends = [e for e in v1.journal.events if e.get('operation') == 'replay_schedule_wait' and e['kind'] == 'compute_end']
    assert sum(e['wall_seconds'] for e in ends) == 5
    assert all(e['gpu_seconds'] == 0 for e in ends)


def test_completed_coordinator_resume_checks_source_before_evaluation(toy_bank, tmp_path, monkeypatch):
    from bfas.rtd import evaluation
    Clock(monkeypatch)
    v0 = engine(tmp_path/'V0', toy_bank, arm='V0'); v0.run()
    v1 = follower(tmp_path/'V1', toy_bank, v0); v1.run()
    manifest = v1.manifest | {'smoke': True}
    atomic_json(v1.directory/'manifest.json', manifest)
    # Even a formatting-only rewrite changes the final whole-file binding.
    path = v0.directory/'exposure_schedule.json'
    path.write_text(path.read_text() + '\n')
    monkeypatch.setattr(cli, 'hardware_identity', lambda: dict(uuid='mock', gpu='mock', memory=0))
    monkeypatch.setattr(cli, 'bank_audit', lambda config: {})
    monkeypatch.setattr(cli, 'make_manifest', lambda *a, **kw: manifest)
    monkeypatch.setattr(cli, 'validate_resume', lambda *a, **kw: None)
    monkeypatch.setattr(evaluation, 'evaluate', lambda *a, **kw: pytest.fail('must validate before evaluation'))
    args = SimpleNamespace(command='resume', run_dir=v1.directory, arm='V1')
    with pytest.raises(ValueError, match='final whole-file hash changed'):
        cli.run_campaign(args, v1.config)


def test_coordinator_forwards_streaming_options_to_worker(tmp_path, monkeypatch):
    from bfas.rtd import evaluation
    monkeypatch.setattr(cli, 'hardware_identity', lambda: dict(uuid='mock', gpu='mock', memory=0))
    monkeypatch.setattr(evaluation, 'evaluate', lambda *a, **kw: None)
    monkeypatch.setattr(evaluation, 'report', lambda *a, **kw: None)
    launches = []
    monkeypatch.setattr(cli.subprocess, 'run', lambda command, **kw: launches.append(command))
    config = dict(output_root='unused', rounds=2, replay_mode='streaming', replay_schedule=str(tmp_path/'V0'),
                  replay_poll_seconds=2., replay_timeout_seconds=5.)
    args = SimpleNamespace(command='run', run_dir=tmp_path/'V1', arm='V1', port=None,
                           config=cli.ROOT/'configs/rtd/v1_1_bfcl.yaml')
    cli.run_campaign(args, config)
    assert len(launches) == 2
    for command in launches:
        for key, value in config.items():
            if key.startswith('replay_'):
                assert command[command.index('--'+key.replace('_', '-'))+1] == str(value)


def test_launcher_streaming_extra_args_and_saved_config_resume(tmp_path):
    root = Path(__file__).resolve().parents[1]
    capture = tmp_path/'capture'
    capture.write_text('#!/usr/bin/env python3\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n')
    capture.chmod(0o755)
    directory = tmp_path/'V1'; directory.mkdir()
    env = os.environ | dict(SLURM_SUBMIT_DIR=str(root), SLURM_JOB_ID='1', CUDA_VISIBLE_DEVICES='0',
        RTD_PYTHON=str(capture), RTD_RUN_DIR=str(directory), RTD_COMMAND='auto',
        RTD_EXTRA_ARGS=f'--replay-schedule {tmp_path}/V0 --replay-mode streaming --replay-poll-seconds 2 --replay-timeout-seconds 5')
    script = root/'scripts/rtd_v11_run_hpg.slurm'
    argv = json.loads(subprocess.check_output(['bash', str(script), 'V1'], env=env, text=True))
    assert argv[1:4] == ['run', '--arm', 'V1']
    assert argv[-8:] == env['RTD_EXTRA_ARGS'].split()
    (directory/'manifest.json').write_text('{}')
    env['RTD_EXTRA_ARGS'] = ''
    argv = json.loads(subprocess.check_output(['bash', str(script), 'V1'], env=env, text=True))
    assert argv[1:4] == ['resume', '--arm', 'V1'] and '--config' not in argv


def test_streaming_control_archive_and_report_survive_progress(toy_bank, tmp_path, monkeypatch):
    from bfas.rtd.controls_v11 import load_window, save_window
    from test_rtd_v11_metrics_controls import report_fixture
    from tools.rtd_v11_report import collect
    Clock(monkeypatch)
    v0 = engine(tmp_path/'V0', toy_bank, arm='V0'); v0.run()
    v1 = follower(tmp_path/'V1', toy_bank, v0)
    v1.round_start(); v1.step_start(); v1.reference()
    while v1.state['phase'] == 'selected':
        v1.selected()
    v1.revealed(); v1.actual()
    archive = save_window(v1)
    frozen = load_window(archive)['manifest']
    binding = json.loads(archive.with_suffix('.json').read_text())
    assert frozen['replay_schedule_hash'] is None
    assert binding['manifest_hash'] == manifest_hash(frozen)
    v1.run()
    assert v1.manifest['replay_schedule_hash'] is not None
    assert manifest_hash(v1.manifest) == manifest_hash(frozen)
    assert load_window(archive)['manifest'] == frozen

    root = report_fixture(tmp_path/'report')
    path = root/'results/rtd_v1_1/V1/manifest.json'
    manifest = json.loads(path.read_text()) | dict(replay_mode='streaming',
        replay_schedule_identity=frozen['replay_schedule_identity'], replay_consumed_steps=[], replay_schedule_hash=None)
    control = dict(schema='rtd-v11-controls-rev31-1', arm='V1', manifest_hash=manifest_hash(manifest))
    manifest.update(replay_consumed_steps=v1.manifest['replay_consumed_steps'],
                    replay_schedule_hash=v1.manifest['replay_schedule_hash'])
    atomic_json(path, manifest)
    report = root/'control.json'; atomic_json(report, control)
    assert collect(root, controls=[report])['controls'] == [control]
    manifest['replay_schedule_identity'] = {'changed': True}; atomic_json(path, manifest)
    with pytest.raises(ValueError, match='different run manifest'):
        collect(root, controls=[report])
