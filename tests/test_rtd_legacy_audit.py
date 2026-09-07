"""C25f: legacy audit CLI with controlled filesystem evidence, no GPU work."""
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT)]
from bfas.rtd import cli, identity
from bfas.rtd.persistence import atomic_json, digest, file_hash
from rtd_identity_fixtures import put_tools, tool_content


def put(path, content='fixture'):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


def snapshot(directory):
    return {str(p.relative_to(directory)): (file_hash(p), p.stat().st_mtime_ns)
            for p in directory.rglob('*') if p.is_file()}


@pytest.fixture
def legacy(tmp_path, monkeypatch):
    root, directory = tmp_path, tmp_path/'results/fake_R1'
    files = [identity.LEADERBOARD+'/bfcl_eval/utils.py',
             identity.LEADERBOARD+'/bfcl_eval/templates/prompt.txt',
             identity.LEADERBOARD+'/bfcl_eval/data/tasks.json',
             *identity.EVALUATION_TOOLS]
    manifest_ns = 1_700_000_000_000_000_000
    mtimes = {}
    for i, name in enumerate(files):
        path = put(root/name, tool_content(name) if name in identity.EVALUATION_TOOLS else 'fixture')
        mtimes[name] = manifest_ns - 100 + i
        os.utime(path, ns=(mtimes[name], mtimes[name]))
    put(root/'envs/bfcl/.venv/bin/bfcl').chmod(0o755)
    put(root/'envs/bfcl/.venv/lib/python3.11/site-packages/bfcl_eval-1.dist-info/METADATA')
    # These newer files are outside evaluation identity; no mtime veto.
    for name in ('bfcl_eval/__pycache__/utils.pyc', 'result_p9170/tasks.json'):
        put(root/identity.LEADERBOARD/name)
    put(root/'src/bfas/rtd/experiment.py', 'new RTD implementation')
    config = dict(evaluation_temperature=0., max_action_tokens=4096)
    manifest = dict(arm='R1', config=config, config_hash=digest(config), harness_hash='old-combined-hash')
    manifest_path = directory/'manifest.json'
    atomic_json(manifest_path, manifest)
    os.utime(manifest_path, ns=(manifest_ns, manifest_ns))
    put(directory/'recovery/latest.json', '{"untouched": true}')
    put(directory/'round-1/checkpoint.json', '{"untouched": true}')
    before = snapshot(directory)

    def forbidden(*a, **kw):
        pytest.fail('audit must not load config defaults, query hardware, train, or initialize CUDA')

    def git_only(command, **kw):
        assert command == ['git', '-C', str(root/identity.LEADERBOARD), 'rev-parse', 'HEAD']
        return SimpleNamespace(stdout='fixture-revision\n')

    monkeypatch.setattr(cli, 'ROOT', root)
    monkeypatch.setattr(cli, 'load_config', forbidden)
    monkeypatch.setattr(cli, 'hardware_identity', forbidden)
    monkeypatch.setattr(cli, 'run_command', forbidden)
    monkeypatch.setattr(torch.cuda, '_lazy_init', forbidden)
    monkeypatch.setattr(identity.subprocess, 'run', git_only)
    monkeypatch.setenv('RTD_CONFIG', 'must-not-read-current-config.yaml')
    return SimpleNamespace(root=root, directory=directory, manifest=manifest, manifest_ns=manifest_ns,
                           mtimes=mtimes, before=before,
                           path=root/'configs/rtd/legacy_identities'/f'{digest(manifest)}.json')


def test_cli_writes_full_manifest_binding_and_prints_exact_evidence(legacy, capsys):
    c = legacy
    assert cli.main(['audit-legacy', '--run-dir', str(c.directory)]) == 0
    printed = json.loads(capsys.readouterr().out)
    saved = json.loads(c.path.read_text())
    assert printed == dict(supplement_path=str(c.path), created=True, **saved)
    assert c.path.name == digest(c.manifest)+'.json'
    assert saved['manifest_hash'] == digest(c.manifest)
    assert saved['legacy_harness_hash'] == c.manifest['harness_hash']
    assert saved['rtd_source'] == dict(kind='legacy-mixed-harness', hash='old-combined-hash', files=None)
    assert saved['harness_hash'] == digest(saved['evaluation_harness'])
    assert saved['evaluation_harness']['config'] == {'evaluation_temperature': 0.}
    audit = saved['audit']
    assert audit['file_mtimes_ns'] == c.mtimes
    assert audit['harness_files_checked'] == len(c.mtimes)
    assert audit['manifest_mtime_ns'] == c.manifest_ns
    assert audit['latest_harness_mtime_ns'] == max(c.mtimes.values())
    assert 'not cryptographic proof' in audit['limitation']
    assert identity.guard_harness(c.root, c.directory, c.manifest) == saved
    assert snapshot(c.directory) == c.before
    # A different full manifest cannot inherit this supplement.
    with pytest.raises(ValueError, match='audited, manifest-bound'):
        identity.saved_identities(c.root, c.directory, dict(c.manifest, arm='R0'))
    first = snapshot(c.path.parent)
    assert cli.main(['audit-legacy', '--run-dir', str(c.directory)]) == 0
    assert json.loads(capsys.readouterr().out)['created'] is False
    assert snapshot(c.path.parent) == first
    assert snapshot(c.directory) == c.before


@pytest.mark.parametrize('name', [identity.LEADERBOARD+'/bfcl_eval/utils.py',
    identity.LEADERBOARD+'/bfcl_eval/templates/prompt.txt',
    identity.LEADERBOARD+'/bfcl_eval/data/tasks.json',
    *identity.EVALUATION_TOOLS])
def test_cli_refuses_any_newer_harness_file_without_writing(legacy, name):
    c = legacy
    os.utime(c.root/name, ns=(c.manifest_ns+1, c.manifest_ns+1))
    with pytest.raises(ValueError, match='newer or equal') as error:
        cli.main(['audit-legacy', '--run-dir', str(c.directory)])
    assert name in str(error.value) and str(c.manifest_ns+1) in str(error.value)
    assert not c.path.parent.exists()
    assert snapshot(c.directory) == c.before


def test_equal_mtime_cannot_establish_that_harness_predates_manifest(legacy):
    c = legacy
    os.utime(c.root/identity.EVALUATION_TOOLS[0], ns=(c.manifest_ns, c.manifest_ns))
    with pytest.raises(ValueError, match='strictly predate'):
        cli.main(['audit-legacy', '--run-dir', str(c.directory)])
    assert not c.path.exists()


@pytest.mark.parametrize('fields', [{'evaluation_harness': {}}, {'rtd_source': {}},
                                  {'evaluation_harness': {}, 'rtd_source': {}}])
def test_cli_refuses_split_identity_even_if_partial(legacy, fields):
    c = legacy
    atomic_json(c.directory/'manifest.json', dict(c.manifest, **fields))
    before = snapshot(c.directory)
    with pytest.raises(ValueError, match='already carries a split identity'):
        cli.main(['audit-legacy', '--run-dir', str(c.directory)])
    assert not c.path.parent.exists()
    assert snapshot(c.directory) == before


def test_audit_identity_and_mtime_inventory_ignore_local_venv(legacy):
    c = legacy
    before = identity.evaluation_harness_identity(c.root, c.manifest['config'])
    (c.root/'envs/bfcl/.venv').rename(c.root/'envs/bfcl-venv')
    result = identity.audit_legacy(c.root, c.directory)
    assert result['evaluation_harness'] == before
    assert result['audit']['file_mtimes_ns'] == c.mtimes
    metadata = c.root/'envs/bfcl-venv/lib/python3.11/site-packages/bfcl_eval-1.dist-info/METADATA'
    os.utime(metadata, ns=(c.manifest_ns+1, c.manifest_ns+1))
    saved = snapshot(c.path.parent)
    identity.audit_legacy(c.root, c.directory)
    assert snapshot(c.path.parent) == saved


def test_reaudit_cannot_rebind_existing_supplement_even_with_old_mtimes(legacy):
    c = legacy
    identity.audit_legacy(c.root, c.directory)
    before = snapshot(c.path.parent)
    name = identity.EVALUATION_TOOLS[0]
    put(c.root/name, 'different evaluation contents')
    os.utime(c.root/name, ns=(c.mtimes[name], c.mtimes[name]))
    with pytest.raises(ValueError, match='refusing to overwrite'):
        cli.main(['audit-legacy', '--run-dir', str(c.directory)])
    assert snapshot(c.path.parent) == before
    assert snapshot(c.directory) == c.before


@pytest.mark.parametrize('target', ['harness', 'inventory', 'manifest'])
def test_audit_refuses_changes_during_hashing(legacy, monkeypatch, target):
    c = legacy
    original = identity.evaluation_harness_identity

    def mutate(root, config):
        result = original(root, config)
        path = {'harness': root/identity.EVALUATION_TOOLS[0],
                'inventory': root/identity.LEADERBOARD/'bfcl_eval/new.py',
                'manifest': c.directory/'manifest.json'}[target]
        put(path, 'changed during audit')
        return result

    monkeypatch.setattr(identity, 'evaluation_harness_identity', mutate)
    with pytest.raises(ValueError, match='changed during legacy audit'):
        identity.audit_legacy(c.root, c.directory)
    assert not c.path.parent.exists()


def test_invalid_saved_config_is_refused(legacy):
    c = legacy
    c.manifest['config']['evaluation_temperature'] = 1.
    atomic_json(c.directory/'manifest.json', c.manifest)
    with pytest.raises(ValueError, match='valid saved config hash'):
        identity.audit_legacy(c.root, c.directory)
    assert not c.path.parent.exists()


def test_missing_harness_file_is_refused(legacy):
    c = legacy
    (c.root/identity.EVALUATION_TOOLS[0]).unlink()
    with pytest.raises(FileNotFoundError):
        identity.audit_legacy(c.root, c.directory)
    assert not c.path.parent.exists()
