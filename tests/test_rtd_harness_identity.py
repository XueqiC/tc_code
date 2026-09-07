"""C25g: portable content identity and audited v1 supplement migration, CPU only."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT)]
from bfas.rtd import cli, identity
from bfas.rtd.persistence import atomic_json, digest, file_hash
from rtd_identity_fixtures import put_tools, tool_content


def put(path, content='fixture'):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


@pytest.fixture
def checkout(tmp_path):
    root = tmp_path/'rai'
    for name in ('bfcl_eval/utils.py', 'bfcl_eval/templates/prompt.txt',
                 'bfcl_eval/data/tasks.json', 'bfcl_eval/data/nested/answers.json', 'pyproject.toml'):
        put(root/identity.LEADERBOARD/name)
    put_tools(root)
    return root


def test_identical_copies_with_and_without_git_have_identical_identity(checkout, tmp_path, monkeypatch):
    other = tmp_path/'different/absolute/path/hpg'
    shutil.copytree(checkout, other)
    put(checkout/Path(identity.LEADERBOARD).parent/'.git/HEAD', 'revision on rai')
    # Even differing absolute shebangs, install layout, and mtimes are immaterial.
    put(checkout/'envs/bfcl/.venv/bin/bfcl', f'#!{checkout}/python')
    put(other/'envs/bfcl-venv/bin/bfcl', f'#!{other}/python')
    def forbidden(*a, **kw):
        pytest.fail('content identity must not execute git (or any subprocess)')
    monkeypatch.setattr(identity.subprocess, 'run', forbidden)
    config = dict(evaluation_temperature=0.)
    left = identity.evaluation_harness_identity(checkout, config)
    right = identity.evaluation_harness_identity(other, config)
    assert left == right and cli.harness_hash(checkout, config) == cli.harness_hash(other, config)
    assert identity.evaluation_harness_metadata(other) == {}
    for name, row in left['data_manifest'].items():
        path = checkout/identity.LEADERBOARD/name
        assert not Path(name).is_absolute()
        assert row == dict(size=path.stat().st_size, sha256=file_hash(path))
    assert str(checkout) not in json.dumps(left) and str(other) not in json.dumps(right)


@pytest.mark.parametrize('git_location', ['', '..'])
@pytest.mark.parametrize('outcome', ['ok', 'empty', 'missing-executable', 'exit-128', 'timeout'])
def test_optional_git_metadata_never_changes_content_or_raises(checkout, monkeypatch, git_location, outcome):
    before = identity.evaluation_harness_identity(checkout, {})
    (checkout/identity.LEADERBOARD/git_location/'.git').mkdir()
    def git(command, **kw):
        assert command == ['git', '-C', str(checkout/identity.LEADERBOARD), 'rev-parse', 'HEAD']
        assert kw['timeout'] == 5
        if outcome == 'missing-executable':
            raise FileNotFoundError('git')
        if outcome == 'exit-128':
            raise subprocess.CalledProcessError(128, command)
        if outcome == 'timeout':
            raise subprocess.TimeoutExpired(command, 5)
        return SimpleNamespace(stdout='test-head\n' if outcome == 'ok' else '')
    monkeypatch.setattr(identity.subprocess, 'run', git)
    assert identity.evaluation_harness_metadata(checkout) == (
        {'checkout_revision': 'test-head'} if outcome == 'ok' else {})
    assert identity.evaluation_harness_identity(checkout, {}) == before


@pytest.mark.parametrize('change', ['source', 'data-bytes', 'data-size', 'data-rename', 'data-add',
                                  'data-delete', 'tool', 'config'])
def test_each_content_component_changes_identity(checkout, change):
    config = dict(evaluation_temperature=0.)
    before = identity.evaluation_harness_identity(checkout, config)
    data = checkout/identity.LEADERBOARD/'bfcl_eval/data/tasks.json'
    if change == 'source':
        put(checkout/identity.LEADERBOARD/'bfcl_eval/utils.py', 'changed')
    elif change == 'data-bytes':
        put(data, 'changed')  # same length as fixture: bytes matter independently of size
    elif change == 'data-size':
        put(data, 'longer fixture')
    elif change == 'data-rename':
        data.rename(data.with_name('renamed.json'))
    elif change == 'data-add':
        put(data.with_name('new.json'))
    elif change == 'data-delete':
        data.unlink()
    elif change == 'tool':
        p = checkout/identity.EVALUATION_TOOLS[0]
        put(p, p.read_text().replace('handler = QwenFCHandler(model, 1., model, True)',
                                   'handler = QwenFCHandler(model, 0., model, True)'))
    else:
        config['evaluation_temperature'] = 1.
    assert digest(identity.evaluation_harness_identity(checkout, config)) != digest(before)


@pytest.fixture
def v1_supplement(checkout):
    root, directory = checkout, checkout/'run'
    config = dict(evaluation_temperature=0., replay_bank_path='bank', support_manifest='support.json')
    put(root/'bank/public/requests.json')
    put(root/'bank/sealed/integrity.json')
    put(root/'support.json')
    manifest = dict(config=config, config_hash=digest(config), harness_hash='old-mixed',
                    data_hash=cli.data_identity(root, config, root/'bank'))
    atomic_json(directory/'manifest.json', manifest)
    content = identity._content_v2_identity(root, config, identity.EVALUATION_TOOLS)
    old = dict(version='bfcl-evaluation-harness-v1', leaderboard=identity.LEADERBOARD,
               checkout_revision='historical-head',
               checkout_hash=digest({n: row['sha256'] for n, row in content['checkout_files'].items()}),
               executable='envs/bfcl/.venv/bin/bfcl', executable_hash='historical-wrapper',
               package_metadata={'historical/METADATA': 'historical-package'},
               tools=content['tools'], config=content['config'])
    saved = dict(version='rtd-c25e-audited-legacy-identity-v1', manifest_hash=digest(manifest),
                 legacy_harness_hash=manifest['harness_hash'], evaluation_harness=old,
                 harness_hash=digest(old), rtd_source=dict(kind='legacy-mixed-harness', hash='old-mixed', files=None),
                 audit=dict(evidence='original C25e/C25f audit'))
    path = root/'configs/rtd/legacy_identities'/f'{digest(manifest)}.json'
    atomic_json(path, saved)
    return SimpleNamespace(root=root, directory=directory, manifest=manifest, saved=saved, path=path)


def test_v1_supplement_migrates_without_git_with_old_binding_and_audit_retained(v1_supplement, monkeypatch):
    c = v1_supplement
    original = (c.directory/'manifest.json').read_bytes()
    def forbidden(*a, **kw):
        pytest.fail('git/GPU must not be needed for legacy migration')
    monkeypatch.setattr(identity.subprocess, 'run', forbidden)
    with pytest.raises(ValueError, match='audit-legacy'):
        identity.guard_harness(c.root, c.directory, c.manifest)
    result = identity.audit_legacy(c.root, c.directory)
    saved = json.loads(c.path.read_text())
    assert not result['created']
    assert saved['evaluation_harness']['version'] == 'bfcl-evaluation-harness-content-v2'
    for key in ('manifest_hash', 'legacy_harness_hash', 'rtd_source', 'audit'):
        assert saved[key] == c.saved[key]
    note, = saved['identity_migrations']
    assert note['previous_evaluation_harness'] == c.saved['evaluation_harness']
    assert note['previous_harness_hash'] == c.saved['harness_hash']
    assert note['verified_manifest_data_hash'] == c.manifest['data_hash']
    assert (c.directory/'manifest.json').read_bytes() == original
    with pytest.raises(ValueError, match='update-identity'):
        identity.guard_harness(c.root, c.directory, c.manifest)


@pytest.mark.parametrize('change', ['source', 'tool', 'data', 'config', 'binding', 'digest'])
def test_migration_refuses_drift_without_rewriting_evidence(v1_supplement, change):
    c = v1_supplement
    if change in {'source', 'tool', 'data'}:
        name = {'source': identity.LEADERBOARD+'/bfcl_eval/utils.py',
                'tool': identity.EVALUATION_TOOLS[0],
                'data': identity.LEADERBOARD+'/bfcl_eval/data/tasks.json'}[change]
        put(c.root/name, 'changed')
    else:
        saved = json.loads(c.path.read_text())
        if change == 'config':
            saved['evaluation_harness']['config']['evaluation_temperature'] = 1.
            saved['harness_hash'] = digest(saved['evaluation_harness'])
        elif change == 'binding':
            saved['manifest_hash'] = 'other-manifest'
        else:
            saved['harness_hash'] = 'invalid-hash'
        atomic_json(c.path, saved)
    before = c.path.read_bytes()
    with pytest.raises(ValueError, match='differs|mismatch'):
        identity.audit_legacy(c.root, c.directory)
    assert c.path.read_bytes() == before
