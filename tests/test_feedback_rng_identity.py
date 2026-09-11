"""CPU stream derivation and fail-closed BFCL resume/replay version binding."""
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

from bfas.rtd import cli
from bfas.rtd.conventions import hashed_step, load_schedule, schedule_identity
from bfas.rtd.feedback_rng import (
    FEEDBACK_RNG_VERSION, FeedbackRNG, feedback_rng_identity, guard_feedback_rng_comparison,
)
from bfas.rtd.identity import validate_resume
from bfas.rtd.persistence import ComputeJournal, atomic_json, digest, file_hash, manifest_hash
from bfas.rtd.streaming_replay import StreamingSchedule, prepare_manifest
from bfas.rtd.unified.config import arm_config, manifest_fields
from bfas.rtd.unified.arms import ARMS
from test_rtd_manifest_tolerance import manifest_inputs


def test_feedback_rng_v2_keys_all_six_fields_and_pins_seed_derivation():
    context = FeedbackRNG(19, 2, 4, 'post_commit_feedback')
    expected = int(digest(dict(feedback_rng_version=2, run_seed=19, round=2, step=4,
        feedback_role='post_commit_feedback', meta_task_id='multi_turn_base_3', rollout_index=1))[:16], 16) % (2**63)
    assert FEEDBACK_RNG_VERSION == 2
    assert expected == 4280134842326760438
    assert context.episode_seed('multi_turn_base_3', 1) == expected
    variants = [context.episode_seed('multi_turn_base_3', 1),
        replace(context, run_seed=20).episode_seed('multi_turn_base_3', 1),
        replace(context, round=3).episode_seed('multi_turn_base_3', 1),
        replace(context, step=5).episode_seed('multi_turn_base_3', 1),
        replace(context, feedback_role='acquisition_reference_feedback').episode_seed('multi_turn_base_3', 1),
        context.episode_seed('multi_turn_base_4', 1), context.episode_seed('multi_turn_base_3', 2)]
    assert len(set(variants)) == 7
    assert context.generator('multi_turn_base_3', 1, device='cpu').initial_seed() == expected


@pytest.mark.parametrize('version', [None, 1])
def test_feedback_rng_old_shared_version_refused_on_resume(manifest_inputs, version):
    c = manifest_inputs
    current = cli.make_manifest(c.config, 'R1', c.audit)
    assert current['feedback_rng_version'] == 2
    saved = dict(current)
    if version is None:
        saved.pop('feedback_rng_version')
    else:
        saved['feedback_rng_version'] = version
    assert manifest_hash(saved) != manifest_hash(current)
    with pytest.raises(ValueError, match='resume config/data/base metadata changed'):
        validate_resume(c.root, c.root, saved, current, acknowledge=True)
    assert not (c.root/'code_drift.jsonl').exists()


@pytest.mark.parametrize('version', [None, 1, 2])
@pytest.mark.parametrize('streaming', [False, True])
def test_feedback_rng_replay_identity_requires_same_version(tmp_path, version, streaming):
    bank = tmp_path/'bank'
    atomic_json(bank/'public/requests.json', [])
    atomic_json(bank/'sealed/integrity.json', {})
    path = tmp_path/'exposure_schedule.json'
    config = dict(benchmark='bfcl', training_seed=0, rounds=2, replay_schedule=str(path),
                  replay_poll_seconds=1., replay_timeout_seconds=1.)
    support = SimpleNamespace(parents={'00': 'multi_turn_base_0'})
    manifest = dict(config=config, arm='D3', bank_path=str(bank), budget_ceilings=[10, 25],
                    feedback_rng_version=2)
    old = dict(manifest)
    if version is None:
        old.pop('feedback_rng_version')
    else:
        old['feedback_rng_version'] = version
    identity = schedule_identity(old, config, support)
    if version is None:
        # Actual pre-V2 schedules had no version field at all.
        identity.pop('feedback_rng_version')
    atomic_json(path, dict(version='rtd-v11-exposure-v1', arm='V0', complete=True,
        smoke=True, identity=identity, steps=[hashed_step(dict(round=1, step=1, decision=True))]))
    config['replay_schedule_hash'] = file_hash(path)
    if streaming:
        holder = SimpleNamespace(directory=tmp_path/'D3', config=config, manifest=manifest,
            support=support, journal=ComputeJournal(tmp_path/'D3/compute.jsonl'))
        prepare_manifest(holder)
        reader = StreamingSchedule(holder, smoke=True)
        check = lambda: reader.snapshot(check_initial_parameters=False)
    else:
        check = lambda: load_schedule(config, manifest, support, smoke=True, check_initial_parameters=False)
    if version == 2:
        check()
    else:
        with pytest.raises(ValueError, match='identity'):
            check()
        if streaming:
            assert holder.journal.events[-1]['kind'] == 'replay_schedule_diff'


def test_feedback_rng_p1_matching_binds_one_version_for_every_arm():
    config = cli.load_config('configs/rtd/unified_bfcl_gemma4.yaml')
    manifest = dict(base_checkpoint_hash='model', tokenizer_hash='tokenizer', data_hash='bank',
                    hardware_hash='cpu', smoke=False, feedback_rng_version=2)
    matched = []
    for arm in ARMS:
        selected, _ = arm_config(config, arm)
        manifest['config_hash'] = digest(selected)
        row = manifest_fields(selected, manifest)
        matched.append(row['p1_matching'])
        old = manifest_fields(selected, manifest | dict(feedback_rng_version=1))
        assert row['p1_matching']['feedback_rng_version'] == 2
        assert row['p1_matching'] != old['p1_matching']
        assert row['campaign_identity'] != old['campaign_identity']
    assert all(row == matched[0] for row in matched)
    assert feedback_rng_identity(dict(benchmark='alfworld')) == {}


@pytest.mark.parametrize('old_version', [None, 1])
def test_feedback_rng_p1_report_refuses_mixed_versions_before_reading_runs(tmp_path, old_version):
    from bfas.rtd.evaluation import report
    runs, manifests = [], []
    for arm in ('V0', *ARMS):
        manifest = dict(config=dict(benchmark='bfcl'), arm=arm, feedback_rng_version=2)
        if arm == 'V0':
            if old_version is None:
                manifest.pop('feedback_rng_version')
            else:
                manifest['feedback_rng_version'] = old_version
        run = tmp_path/arm
        atomic_json(run/'manifest.json', manifest)
        runs.append(run)
        manifests.append(manifest)
    with pytest.raises(ValueError, match='feedback RNG versions differ'):
        report(runs, tmp_path/'report')
    assert not (tmp_path/'report').exists()
    for manifest in manifests:
        manifest['feedback_rng_version'] = 2
    guard_feedback_rng_comparison(manifests)


def test_feedback_rng_v11_report_refuses_mixed_versions(tmp_path):
    from test_rtd_v11_metrics_controls import report_fixture
    from tools.rtd_v11_report import collect
    root = report_fixture(tmp_path)
    path = root/'results/rtd_v1_1/V1/manifest.json'
    manifest = json.loads(path.read_text())
    atomic_json(path, manifest | dict(feedback_rng_version=2))
    with pytest.raises(ValueError, match='feedback RNG versions differ'):
        collect(root)
