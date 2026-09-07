"""C26-E config/identity tests: synthetic CPU archives only, all writes in tmp_path."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path

import pytest
import yaml

from bfas.rtd.benchmarks import alfworld_config as config
from bfas.rtd.benchmarks import alfworld_bank as bank
from bfas.rtd.benchmarks.alfworld_support import (
    environment_identity, freeze_support, seal_verified_bank, verify_package,
)
from bfas.rtd.persistence import digest
from rtd_alfworld_evaluation_fixtures import campaign, put, ROOT
from test_rtd_alfworld_state import FakeStepper, make_payload
from test_rtd_alfworld_rollout import render


@pytest.fixture
def sealed_campaign(campaign):
    """Production-sized *synthetic* counts; no patched validator/auditor or real bank."""
    c = campaign
    for name in ('src/bfas/rtd/transport.py', 'src/bfas/rtd/benchmarks/alfworld_state.py'):
        put(c.root / name, (ROOT / name).read_text())
    put(c.model / 'chat_template.jinja', 'fixture {{messages}}')
    tids = [f'pick_and_place_simple-Apple-None-Table-{i:03d}/trial-1' for i in range(135)]
    put(c.root / bank.SUPPORT, dict(demand=tids, calibration=[]))
    put(c.root / bank.PROBE, '')
    put(c.root / bank.LEDGER, ''.join(json.dumps(dict(task_id=t))+'\n' for t in tids))
    for tid in tids:
        world = c.data / 'train' / tid
        put(world / 'game.tw-pddl', dict(grammar={'task': [{'rhs': 'Your task is to: put apple on table.'}]}))
        put(world / 'traj_data.json', dict(task_id=tid))
        put(world / 'initial_state.pddl', 'unique fixture ' + tid)
    support = freeze_support(c.root, environment_identity(c.root, c.model))
    records, payloads, requests, resets = [], {}, {}, {}
    for i, tid in enumerate(tids):
        request = support['tasks'][tid]['request']
        p = make_payload(request)
        verified = verify_package(p, request, FakeStepper(request), render)
        resets[tid] = verified['verification']['states'][0]
        if i >= 107:
            continue
        cost = 360 if i == 0 else 339
        p['historical_response']['tokens_spent'] = p['cost'] = cost
        p['raw_ledger_line'] = json.dumps(p['historical_response'])
        p['usage']['estimated_output_tokens'] = cost
        verified = verify_package(p, request, FakeStepper(request), render)
        q = p['query_id']
        records.append(bank.public_record(q, request)); payloads[q] = verified; requests[q] = request
    for i in range(112):
        tid = tids[i]
        request = support['tasks'][tid]['request']
        p = make_payload(request)
        row = p['historical_response']
        cost = 1399 if i == 0 else 1368
        row.update(verified=False, attempt_index=1, tokens_spent=cost)
        row.pop('demo')
        p['provenance'].update(attempt_index=1, line=i+2)
        q = bank.query_id(p['provenance']['ledger_sha256'], row, i+2)
        p.update(query_id=q, success=False, status='unavailable', cost=cost, commands=[],
            payload_kind=None, raw_ledger_line=json.dumps(row), behaviors=[],
            unavailable_reason='failed attempt: response/trajectory missing')
        p['usage']['estimated_output_tokens'] = cost
        records.append(bank.public_record(q, request)); payloads[q] = p; requests[q] = request
    historical = dict(source_files=[], successful_legacy_token_estimate=36294,
        failed_legacy_token_estimate=153247, failed_attempts=112,
        historical_attempt_gaps=[dict(task_id=tids[0], missing_attempt_indices=[0], cost=None, outcome=None)])
    c.config = config.default_config()
    c.config['student'] = str(c.model)
    c.bank = c.root / c.config['replay_bank_path']
    archive = bank.Archive(records, payloads, requests, [], historical)
    seal_verified_bank(c.bank, archive, support, payloads, resets)
    put(c.root / c.config['support_manifest'], support)
    c.support = support
    return c


def test_yaml_defaults_roundtrip_and_scalar_identity(tmp_path):
    path = ROOT / 'configs/rtd/v1_alfworld_c26.yaml'
    linear = config.load_config(path)
    assert yaml.safe_load(path.read_text()) == linear == config.default_config()
    assert config.validate_config({'benchmark': 'alfworld'}) == linear
    scalar = config.load_config(ROOT / 'configs/rtd/v1_alfworld_c26_scalar_gate.yaml')
    assert {k for k in linear if linear[k] != scalar[k]} == {'gate'}
    assert scalar['gate'] == 'scalar_sigmoid' and digest(linear) != digest(scalar)
    path = tmp_path / 'roundtrip.yaml'
    path.write_text(yaml.safe_dump(scalar, sort_keys=False))
    assert config.load_config(path) == scalar
    assert json.loads(json.dumps(scalar)) == scalar
    copy = config.default_config(); copy['pilot_eta_candidates'].append(99)
    assert copy != config.default_config()


def test_shared_math_is_exact_bfcl_configuration():
    bfcl = yaml.safe_load((ROOT / 'configs/rtd/v1_bfcl_c25.yaml').read_text())
    alf = config.default_config()
    differences = {'benchmark', 'support_parent_tasks_m', 'support_manifest', 'replay_bank_path',
        'replay_public_cap_output_tokens_by_class', 'replay_cap_scope', 'meta_tasks_per_feedback',
        'rollouts_per_meta_task', 'max_action_tokens_by_benchmark', 'max_state_batch_size'}
    assert all(alf[k] == v for k, v in bfcl.items() if k in alf and k not in differences)
    assert 'meta_tasks_multi_turn' not in alf and 'historical_demo_output_tokens_exact' not in alf


@pytest.mark.parametrize('key,value', [
    ('rounds', 2), ('committed_steps_per_round', 11), ('decision_steps_per_round', [1,4,7]),
    ('slots_per_step', 4), ('source_temperature', .9), ('source_top_p', .9),
    ('insertion_fraction', .5), ('baseline', 'smoke_zero'), ('meta_tasks_per_feedback', 8),
    ('rollouts_per_meta_task', 1), ('gate_initial_logit', 1.), ('initial_eta', .01),
    ('pilot_eta_candidates', [.01]), ('support_parent_tasks_m', 134), ('new_teacher_calls', True),
    ('benchmark', 'bfcl'), ('alfworld_train_split', 'valid_seen'),
    ('alfworld_evaluation_split', 'valid_unseen'), ('alfworld_expected_eval_tasks', 139),
    ('alfworld_max_episode_steps', 41), ('alfworld_student_react', False),
    ('max_action_tokens_by_benchmark', {'alfworld': {'single_turn': 256}}),
    ('max_state_batch_size', True), ('max_context_tokens', 256),
    ('memory_peak_budget_gb', float('nan')), ('memory_reserve_gb', -1),
    ('memory_state_estimate_gb', 0), ('memory_reserve_gb', 38), ('gate', 'fixed_half'),
    ('score_consistency_tolerance', {'mean_abs': .05, 'max_abs': True, 'max_abs_outlier_tokens': 2, 'max_abs_hard': 8.}),
    ('pilot_eta_candidates', [float('nan')]),
    ('alfworld_environment_root', '/different/environment'),
    ('protocol_version', '1.0.7'), ('benchmark_schema_version', 'future'),
    ('meta_tasks_multi_turn', 4), ('historical_demo_output_tokens_exact', 1233607),
    ('historical_output_tokens_estimated', 1), ('source_backend', 'vllm'),
])
def test_invalid_protocol_is_rejected(key, value):
    value_config = config.default_config(); value_config[key] = value
    with pytest.raises(ValueError):
        config.validate_config(value_config)


def test_duplicate_yaml_and_bfcl_loader_refuse(tmp_path):
    p = tmp_path / 'bad.yaml'; p.write_text('benchmark: alfworld\nbenchmark: bfcl\n')
    with pytest.raises(ValueError, match='duplicate'):
        config.load_config(p)
    from bfas.rtd.cli import load_config
    with pytest.raises(ValueError, match='frozen protocol'):
        load_config(ROOT / 'configs/rtd/v1_alfworld_c26.yaml')


def test_manifest_section_determinism_and_cap_denominator(sealed_campaign):
    c = sealed_campaign
    a = config.manifest_section(c.root, c.config)
    assert config.manifest_section(c.root, c.config) == a
    assert a['available_packages'] == 107 and a['m'] == 135
    assert a['bank_public_cap_sum'] == 107 * 1310720
    assert a['budget_ceilings'] == [14024704, 35061760, 70123520]
    assert a['bank_audit']['unavailable_packages'] == 112
    assert a['recorded_bank_usage']['output_tokens_estimated'] == 189541
    assert a['recorded_bank_usage']['provider_usage_exact'] is False
    assert a['recorded_bank_usage']['historical_attempt_gaps'][0]['cost'] is None
    assert a['config_hash'] == digest(c.config)
    assert a['harness_hash'] == digest(a['evaluation_harness'])
    assert 'hardware' not in a and 'initial_parameter_hash' not in a
    # Scalar component alters scientific config identity, not the frozen data.
    scalar = config.manifest_section(c.root, dict(c.config, gate='scalar_sigmoid'))
    assert scalar['config_hash'] != a['config_hash'] and scalar['data_hash'] == a['data_hash']


def test_data_identity_binds_support_bank_and_valid_seen(sealed_campaign):
    c = sealed_campaign
    original = config.data_identity(c.root, c.config, c.bank)
    paths = [c.bank / 'sealed/manifest.json', c.root / c.config['support_manifest'],
        next((c.data / 'valid_seen').glob('*/*/initial_state.pddl'))]
    for path in paths:
        data = path.read_bytes(); path.write_bytes(data + b'\n')
        assert config.data_identity(c.root, c.config, c.bank) != original
        path.write_bytes(data)
    (c.root / c.config['support_manifest']).write_text('{}')
    with pytest.raises(ValueError):
        config.bank_audit(c.root, c.config)


def test_bank_corruption_and_train_identity_drift_rejected(sealed_campaign):
    c = sealed_campaign
    p = next((c.data / 'train').glob('*/*/initial_state.pddl'))
    original = p.read_bytes(); p.write_bytes(original + b'changed')
    with pytest.raises(ValueError, match='training world'):
        config.manifest_section(c.root, c.config)
    p.write_bytes(original)
    p = c.bank / 'sealed/integrity.json'; p.write_text('{}')
    with pytest.raises(ValueError, match='integrity'):
        config.bank_audit(c.root, c.config)


def test_preparation_cli_never_starts_run_or_gpu(sealed_campaign, capsys):
    from tools.rtd_alfworld_experiment import main, smoke_plan
    c = sealed_campaign
    path = c.root / 'config.yaml'; path.write_text(yaml.safe_dump(c.config))
    assert main(['audit', '--root', str(c.root), '--config', str(path)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['status'] == 'passed' and not result['gpu_used'] and not result['environment_started']
    plan = smoke_plan(c.root, c.config)
    assert len(plan['train_replays']) == 2
    assert plan['resources']['maximum_feedback_episodes'] == 16
    assert plan['resources']['rollouts_per_meta_task'] == 2
    for command in ('run', 'resume'):
        assert main([command, '--config', '/missing/config']) == 2
        assert 'not yet integrated: run/resume require the C26-F edits' in capsys.readouterr().out
