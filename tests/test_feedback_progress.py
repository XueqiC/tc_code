"""Read-only live JSONL progress: physical calls, retries and stage boundaries."""
import json

import pytest

from tools.rtd_progress import feedback_progress, format_progress


CONFIG = dict(benchmark='alfworld', method='rtd_v1_1', meta_tasks_per_feedback=4,
              rollouts_per_meta_task=2, alfworld_max_episode_steps=40)


def write_journal(path, rows, *, torn=False):
    data = b''.join((json.dumps(dict(sequence=i, **r))+'\n').encode() for i, r in enumerate(rows))
    data += b'{"sequence":' if torn else b''
    (path/'compute.jsonl').write_bytes(data)
    return data


def test_feedback_progress_counts_calls_once_and_ignores_torn_tail(tmp_path):
    rows = [dict(kind='compute_begin', operation='acquisition_reference_feedback', round=1, step=1),
        dict(kind='feedback_plan', role='acquisition_reference_feedback', round=1, step=1, episodes=8),
        dict(kind='compute_begin', operation='generation', parent_sequence=0, sequences=8),
        *[dict(kind='generated_tokens', context='r1/s1/acquisition_reference_feedback') for _ in range(8)],
        dict(kind='compute_end', begin_sequence=2, operation='generation', status='complete', wall_seconds=7.),
        dict(kind='subphase_memory', operation='generation', sequences=8, status='complete'),
        dict(kind='alfworld_episode', task_id='game/trial', rollout_index=1, excluded=True),
        dict(kind='alfworld_episode_retry', task_id='game/trial', rollout_index=1),
        dict(kind='alfworld_episode', task_id='game/trial', rollout_index=1, excluded=False),
        dict(kind='feedback_rollout', role='acquisition_reference_feedback', round=1, step=1,
             rollout=dict(actions=[{}])),
        dict(kind='compute_begin', operation='generation', parent_sequence=0, sequences=7)]
    before = write_journal(tmp_path, rows, torn=True)
    report = feedback_progress(tmp_path, CONFIG)
    assert report['episodes_done'] == report['accepted_rollouts'] == report['retries'] == 1
    assert report['episodes_total'] == 8 and report['action_horizon_bound'] == 320
    assert report['generations_done'] == report['generations_in_flight'] == 1
    assert report['generation_batch_sizes'] == {8: 1} and report['sampled_actions'] == 8
    assert report['estimated_actions'] is None
    assert (tmp_path/'compute.jsonl').read_bytes() == before
    assert len(list(tmp_path.iterdir())) == 1


def test_feedback_progress_uses_actual_completed_window_total(tmp_path):
    rows = [dict(kind='compute_begin', operation='acquisition_reference_feedback', round=1, step=1)]
    rows.extend(dict(kind='feedback_rollout', role='acquisition_reference_feedback', round=1, step=1,
                     rollout=dict(actions=[{}]*i)) for i in range(1, 33))
    rows.extend([dict(kind='compute_end', begin_sequence=0, operation='acquisition_reference_feedback', status='complete'),
        dict(kind='compute_begin', operation='same_batch_reference_feedback', round=1, step=1),
        dict(kind='compute_begin', operation='generation', parent_sequence=34, sequences=4),
        *[dict(kind='generated_tokens', context='r1/s1/same_batch_reference_feedback') for _ in range(4)],
        dict(kind='compute_end', begin_sequence=35, operation='generation', status='complete', wall_seconds=8.)])
    write_journal(tmp_path, rows)
    report = feedback_progress(tmp_path, CONFIG)
    assert report['episodes_total'] == 32 and report['configured_episodes'] == 8
    assert report['generations_done'] == 1 and report['generations_in_flight'] == 0
    assert report['estimated_actions'] == sum(range(1, 33))
    assert report['generation_eta_seconds'] == (sum(range(1, 33))-4)*2
    assert 'config requests 8' in format_progress(report)


def test_feedback_progress_legacy_first_stage_and_new_attempt(tmp_path):
    rows = [dict(kind='round_frozen', round=1, feedback=['a', 'b', 'c']),
        dict(kind='compute_begin', operation='acquisition_reference_feedback', round=1, step=1),
        dict(kind='compute_begin', operation='generation', parent_sequence=1, sequences=4),
        dict(kind='compute_end', begin_sequence=2, operation='generation', status='failed', wall_seconds=3.),
        dict(kind='compute_end', begin_sequence=1, operation='acquisition_reference_feedback', status='failed'),
        dict(kind='compute_begin', operation='acquisition_reference_feedback', round=1, step=1),
        dict(kind='compute_begin', operation='generation', parent_sequence=5, sequences=4)]
    write_journal(tmp_path, rows)
    report = feedback_progress(tmp_path, CONFIG)
    assert report['episodes_total'] == 12  # legacy selects up to 8 parents x 4
    assert report['generations_done'] == 0 and report['generations_in_flight'] == 1


def test_feedback_progress_rejects_corrupt_complete_record(tmp_path):
    (tmp_path/'compute.jsonl').write_bytes(b'{broken}\n')
    with pytest.raises(json.JSONDecodeError):
        feedback_progress(tmp_path, CONFIG)
