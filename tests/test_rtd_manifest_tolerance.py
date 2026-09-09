"""C25q-b: declared manifest tolerances and effective resume audit records."""
from dataclasses import asdict
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT)]
from bfas.rtd import cli, identity
from bfas.rtd.persistence import ComputeJournal, digest
from bfas.rtd.scoring import ScoreTolerance
from rtd_identity_fixtures import hardware_fixture, put_tools


@pytest.fixture
def manifest_inputs(tmp_path, monkeypatch):
    config = cli.load_config(ROOT/'configs/rtd/v1_bfcl_c25.yaml')
    put_tools(tmp_path)
    leaderboard = tmp_path/identity.LEADERBOARD/'bfcl_eval'
    (leaderboard/'data').mkdir(parents=True)
    (leaderboard/'utils.py').write_text('fixture')
    (leaderboard/'data/tasks.json').write_text('{}')
    model = tmp_path/'model'
    model.mkdir()
    (model/'model.safetensors').write_text('fixture weights')
    (model/'tokenizer.json').write_text('{}')
    config['student'] = str(model)
    monkeypatch.setattr(cli, 'ROOT', tmp_path)
    monkeypatch.setattr(cli, 'hardware_identity', hardware_fixture)
    monkeypatch.setattr(cli, 'data_identity', lambda *a: 'fixture-data')
    monkeypatch.setattr(cli, 'evaluation_harness_metadata', lambda *a: {})
    audit = dict(bank_path='fixture-bank', bank_public_cap_sum=100,
                 budget_ceilings=[10, 25, 50], recorded_bank_usage={}, available_packages=1, m=1)
    return SimpleNamespace(root=tmp_path, config=config, audit=audit)


@pytest.mark.parametrize('config_source', ['manifest', 'saved_yaml'])
@pytest.mark.parametrize('declared', [
    {'max_abs': 1.0, 'mean_abs': .05},
    {'mean_abs': .05, 'max_abs': 1},
])
def test_legacy_manifest_resumes_identically_and_audits_defaults(
        manifest_inputs, config_source, declared):
    c = manifest_inputs
    config = dict(c.config, score_consistency_tolerance=declared)
    saved = cli.make_manifest(config, 'R1', c.audit)
    # Reproduce the pre-C25q manifest independently of today's defaults.
    saved['score_consistency']['tolerance'] = dict(declared)
    original = json.dumps(saved, indent=2)
    path = c.root/'manifest.json'
    path.write_text(original)
    saved = json.loads(path.read_text())
    config_path = c.root/'saved.yaml'
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    resumed = cli.resume_config(config_path if config_source == 'saved_yaml' else None, saved)
    current = cli.make_manifest(resumed, 'R1', c.audit)

    assert current == saved
    assert json.dumps(current, indent=2) == original
    assert list(current['score_consistency']['tolerance']) == list(declared)
    for _ in range(2):
        identity.validate_resume(c.root, c.root, saved, current)
    # Every successful resume is audited, including repeated resumes with no drift.
    events = ComputeJournal(c.root/'code_drift.jsonl').events
    assert len(events) == 2
    for event in events:
        assert event['kind'] == 'resume_score_consistency'
        assert event['context'] == 'training_resume'
        assert event['manifest_hash'] == digest(saved)
        assert event['rtd_source_hash'] == current['rtd_source']['hash']
        assert event['tolerance'] == dict(
            mean_abs=.05, max_abs=1., max_abs_outlier_tokens=2, max_abs_hard=8., min_tokens_for_mean=8)
    assert path.read_text() == original
    assert json.dumps(saved, indent=2) == original


@pytest.mark.parametrize('name', ['v1_bfcl_c25.yaml', 'v1_bfcl_c25_scalar_gate.yaml'])
def test_updated_yaml_records_all_explicit_tolerance_keys(manifest_inputs, name):
    c = manifest_inputs
    declared = yaml.safe_load((ROOT/'configs/rtd'/name).read_text())['score_consistency_tolerance']
    assert set(declared) == {'mean_abs', 'max_abs', 'max_abs_outlier_tokens', 'max_abs_hard'}
    config = dict(c.config, score_consistency_tolerance=declared)
    manifest = cli.make_manifest(config, 'R1', c.audit)
    assert json.dumps(manifest['score_consistency']['tolerance']) == json.dumps(declared)
    identity.validate_resume(c.root, c.root, manifest, manifest, training=False)
    event, = ComputeJournal(c.root/'code_drift.jsonl').events
    assert event['context'] == 'evaluation_resume'
    assert event['tolerance'] == asdict(ScoreTolerance.from_config(config)) == dict(
        declared, min_tokens_for_mean=8)


@pytest.mark.parametrize('field', ['generation', 'scoring', 'reinforce_likelihood'])
def test_saved_score_backend_change_still_refuses_resume(manifest_inputs, field):
    c = manifest_inputs
    current = cli.make_manifest(c.config, 'R1', c.audit)
    saved = json.loads(json.dumps(current))
    saved['score_consistency'][field] = 'different-saved-backend'
    with pytest.raises(ValueError, match='resume config/data/base metadata changed'):
        identity.validate_resume(c.root, c.root, saved, current, acknowledge=True)
    assert not (c.root/'code_drift.jsonl').exists()
