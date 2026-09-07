"""C25j scoring scopes and audited updates; all production boundaries forbidden."""
import copy
import json
import os
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT)]
from bfas.rtd import cli, identity, identity_update as update
from bfas.rtd.persistence import atomic_json, digest, file_hash, tree_hash
from bfas.rtd.scoring_scope import scoring_hash
from rtd_identity_fixtures import put_tools


def put(path, text='fixture'):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def snapshot(directory):
    return {p.relative_to(directory).as_posix(): (file_hash(p), p.stat().st_mtime_ns)
            for p in directory.rglob('*') if p.is_file()}


@pytest.fixture
def run(tmp_path, monkeypatch):
    root = tmp_path/'checkout'
    put_tools(root)
    for name in ('bfcl_eval/utils.py', 'bfcl_eval/extra.py', 'bfcl_eval/data/tasks.json', 'pyproject.toml'):
        put(root/identity.LEADERBOARD/name)
    for name in ('src/bfas/rtd/evaluation_lock.py', 'tools/bfcl_campaign_lock.py',
                 'scripts/rtd_run_hpg.slurm', 'src/bfas/rtd/cli.py'):
        put(root/name)
    put(root/'base/model.safetensors', 'tiny fake weights')
    put(root/'base/tokenizer.json', 'tiny fake tokenizer')
    ns = 1_700_000_000_000_000_000
    for p in root.rglob('*'):
        if p.is_file():
            os.utime(p, ns=(ns-1, ns-1))
    config = dict(evaluation_temperature=0., student='fake-student')
    old = identity._content_v2_identity(root, config, [*identity.EVALUATION_TOOLS,
                                      'src/bfas/rtd/evaluation_lock.py', 'tools/bfcl_campaign_lock.py'])
    manifest = dict(config=config, config_hash=digest(config), harness_hash=digest(old),
                    evaluation_harness=old, rtd_source=identity.source_identity(root),
                    model_path=str(root/'base'), base_checkpoint_hash=tree_hash(root/'base'),
                    tokenizer_hash=digest([('tokenizer.json', file_hash(root/'base/tokenizer.json'))]))
    directory = root/'results/rtd_v1/fake'
    atomic_json(directory/'manifest.json', manifest)
    os.utime(directory/'manifest.json', ns=(ns, ns))
    put(directory/'recovery/latest.json', '{"untouched":true}')
    put(directory/'round-1/lora/adapter.safetensors', 'trained weights')
    put(directory/'round-1/round_state.pt', 'state')
    atomic_json(directory/'round-1/checkpoint.json', dict(round=1, manifest_hash=digest(manifest),
        config_hash=manifest['config_hash'], adapter_hash=tree_hash(directory/'round-1/lora'),
        round_state_hash=file_hash(directory/'round-1/round_state.pt')))
    def forbidden(*args, **kwargs):
        pytest.fail('identity audit must not launch subprocesses, query hardware, load a model or initialize CUDA')
    monkeypatch.setattr(identity.subprocess, 'run', forbidden)
    monkeypatch.setattr(torch.cuda, '_lazy_init', forbidden)
    monkeypatch.setattr(cli, 'hardware_identity', forbidden)
    monkeypatch.setattr(cli, 'load_config', forbidden)
    monkeypatch.setattr(cli, 'run_command', forbidden)
    monkeypatch.setattr(cli, 'ROOT', root)
    return SimpleNamespace(root=root, directory=directory, manifest=manifest, ns=ns,
        path=root/'configs/rtd/legacy_identities'/f'{digest(manifest)}.json', before=snapshot(directory))


@pytest.mark.parametrize('path', ['src/bfas/rtd/evaluation_lock.py', 'tools/bfcl_campaign_lock.py',
    'scripts/rtd_run_hpg.slurm', 'src/bfas/rtd/cli.py', identity.LEADERBOARD+'/pyproject.toml'])
def test_plumbing_changes_only_source_metadata(run, path):
    c = run
    before = identity.evaluation_harness_identity(c.root, c.manifest['config'])
    source = identity.source_identity(c.root)
    put(c.root/path, 'new plumbing')
    assert identity.evaluation_harness_identity(c.root, c.manifest['config']) == before
    if not path.startswith(identity.LEADERBOARD):
        assert identity.source_identity(c.root) != source


@pytest.mark.parametrize('path,before,after,changes_score', [
    ('tools/behavior_atom/checker_bridge.py', 'fixture', 'new checker', True),
    ('tools/bfcl_std_campaign.sh', 'Qwen/Qwen3.5-4B-FC', 'Qwen/different-FC', True),
    ('tools/bfcl_std_campaign.sh', '${BFCLSTD_TEMPERATURE:-0.001}', '${BFCLSTD_TEMPERATURE:-0.1}', True),
    ('tools/bfcl_std_campaign.sh', '["Overall Acc"]', '["Another Acc"]', True),
    ('tools/bfcl_std_campaign.sh', 'tools/bfcl_campaign_lock.py', 'tools/other_lock.py', False),
    ('tools/bfcl_std_campaign.sh', 'LOCAL_SERVER_PORT="$PORT"', 'LOCAL_SERVER_PORT="9999"', False),
    ('tools/bfcl_std_campaign.sh', '${GPU_UTIL:-0.85}', '${GPU_UTIL:-0.75}', False),
    ('tools/bfcl_std_campaign.sh', 'CAMPAIGN COMPLETE', 'CAMPAIGN FINISHED', False),
    ('tools/bfcl_generation_check.py', 'if is_format_sensitivity(category):', 'if False:', True),
    ('tools/bfcl_generation_check.py', 'GEN COUNT', 'COUNT LOG', False),
    ('src/bfas/rtd/evaluation.py', 'torch.bfloat16', 'torch.float32', True),
    ('src/bfas/rtd/evaluation.py', 'safe_merge=True', 'safe_merge=False', True),
    ('src/bfas/rtd/evaluation.py', "torch_device='cpu'", "torch_device='cuda'", False),
    ('src/bfas/rtd/evaluation.py', 'include_prereq=False', 'include_prereq=True', True),
    ('src/bfas/rtd/evaluation.py', "['Overall Acc']", "['Other Acc']", True),
    ('src/bfas/rtd/evaluation.py', "--verify", "--not-verify", True),
    ('src/bfas/rtd/evaluation.py', 'evaluation resources tag=', 'resources tag=', False),
    ('src/bfas/adapters/bfcl.py', 'task_ids - failures', 'task_ids', True),
    ('tools/bfcl_event_mine_single.py', 'if "java"', 'if "different"', True),
])
def test_scoring_projections_distinguish_operations(run, path, before, after, changes_score):
    c = run
    initial = identity.evaluation_harness_identity(c.root, c.manifest['config'])
    text = (c.root/path).read_text()
    assert before in text
    put(c.root/path, text.replace(before, after))
    assert (identity.evaluation_harness_identity(c.root, c.manifest['config']) != initial) == changes_score


def test_update_split_manifest_reports_old_diff_and_binds_guard_without_run_writes(run, capsys):
    c = run
    lock = 'src/bfas/rtd/evaluation_lock.py'
    previous = file_hash(c.root/lock)
    put(c.root/lock, 'changed lock')
    assert cli.main(['update-identity', '--run-dir', str(c.directory)]) == 0
    result = json.loads(capsys.readouterr().out)
    saved = json.loads(c.path.read_text())
    assert result['updated'] and result['created']
    change, = result['audit']['changed_old_identity_files']
    assert change == dict(path=lock, old_sha256=previous, new_sha256=file_hash(c.root/lock),
                          new_file_sha256=file_hash(c.root/lock), hash_kind='file', in_new_scoring_set=False)
    assert saved['manifest_hash'] == digest(c.manifest)
    assert saved['rtd_source'] == c.manifest['rtd_source']
    assert identity.guard_harness(c.root, c.directory, c.manifest) == saved
    assert snapshot(c.directory) == c.before
    before = snapshot(c.path.parent)
    assert not update.update_identity(c.root, c.directory)['updated']
    assert snapshot(c.path.parent) == before


@pytest.mark.parametrize('change', ['checker', 'deleted-checker', 'backdated-checker', 'data', 'deleted-source',
                                 'added-source', 'same-time', 'newer-unchanged', 'model', 'tokenizer', 'checkpoint'])
def test_update_refuses_scoring_changes_and_timestamp_gaps_without_supplement(run, change, capsys):
    c = run
    checker = c.root/'tools/behavior_atom/checker_bridge.py'
    target = {'checker': checker, 'deleted-checker': checker, 'backdated-checker': checker, 'same-time': checker,
              'newer-unchanged': checker, 'data': c.root/identity.LEADERBOARD/'bfcl_eval/data/tasks.json',
              'deleted-source': c.root/identity.LEADERBOARD/'bfcl_eval/extra.py',
              'added-source': c.root/identity.LEADERBOARD/'bfcl_eval/added.py',
              'model': c.root/'base/model.safetensors', 'tokenizer': c.root/'base/tokenizer.json',
              'checkpoint': c.directory/'round-1/lora/adapter.safetensors'}[change]
    if change in {'deleted-source', 'deleted-checker'}:
        target.unlink()
    elif change == 'same-time':
        os.utime(target, ns=(c.ns, c.ns))
    elif change == 'newer-unchanged':
        os.utime(target, ns=(c.ns+1, c.ns+1))
    else:
        put(target, 'changed contents')
        if change in {'backdated-checker', 'added-source'}:
            os.utime(target, ns=(c.ns-1, c.ns-1))
    before = snapshot(c.directory)
    assert cli.main(['update-identity', '--run-dir', str(c.directory)]) == 1
    result = json.loads(capsys.readouterr().out)
    assert result['refusals'] and not result['updated']
    if change in {'checker', 'deleted-checker', 'backdated-checker', 'data', 'deleted-source'}:
        assert any(row['path'] == target.relative_to(c.root).as_posix()
                   for row in result['audit']['changed_old_identity_files'])
    assert not c.path.exists() and snapshot(c.directory) == before


def test_config_and_supplement_tampering_refused(run):
    c = run
    update.update_identity(c.root, c.directory)
    saved = json.loads(c.path.read_text())
    saved['manifest_hash'] = 'different'
    atomic_json(c.path, saved)
    before = snapshot(c.path.parent)
    with pytest.raises(ValueError, match='binding mismatch'):
        update.update_identity(c.root, c.directory)
    assert snapshot(c.path.parent) == before
    manifest = copy.deepcopy(c.manifest)
    manifest['config']['evaluation_temperature'] = 9.
    atomic_json(c.directory/'manifest.json', manifest)
    with pytest.raises(ValueError, match='valid saved config'):
        update.update_identity(c.root, c.directory)


@pytest.mark.parametrize('target', ['manifest', 'supplement', 'scoring', 'inventory'])
def test_concurrent_mutations_during_audit_refused(run, monkeypatch, target):
    c = run
    original = update._model_evidence
    def mutate(manifest):
        result = original(manifest)
        paths = dict(manifest=c.directory/'manifest.json', supplement=c.path,
                     scoring=c.root/'tools/behavior_atom/checker_bridge.py',
                     inventory=c.root/identity.LEADERBOARD/'bfcl_eval/new.py')
        put(paths[target], 'concurrent writer')
        return result
    monkeypatch.setattr(update, '_model_evidence', mutate)
    with pytest.raises(ValueError, match='changed during identity audit'):
        update.update_identity(c.root, c.directory)
    assert not c.path.exists() if target != 'supplement' else c.path.read_text() == 'concurrent writer'


def test_reviewed_pre_manifest_sources_admit_plumbing_only_changes_without_git(run):
    c = run
    # Recreate the real C25g legacy inventory using the reviewed historical bytes.
    bundle = json.loads((ROOT/update.EVIDENCE_PATH).read_text())
    for row in bundle['files']:
        put(c.root/row['path'], row['content'])
    ns = max(row['observed_at_ns'] for row in bundle['files']) + 10**9
    for p in c.root.rglob('*'):
        if p.is_file():
            os.utime(p, ns=(ns-1, ns-1))
    config = c.manifest['config']
    old = identity._content_v2_identity(c.root, config,
        [n for n in identity.EVALUATION_TOOLS if n not in {'src/bfas/rtd/evaluation.py', 'src/bfas/adapters/bfcl.py'}])
    manifest = {k: v for k, v in c.manifest.items() if k not in {'evaluation_harness', 'rtd_source'}}
    manifest['harness_hash'] = 'historical-mixed'
    atomic_json(c.directory/'manifest.json', manifest)
    os.utime(c.directory/'manifest.json', ns=(ns, ns))
    # Checkpoint must retain the exact original manifest binding.
    cp = c.directory/'round-1/checkpoint.json'
    meta = json.loads(cp.read_text()); meta['manifest_hash'] = digest(manifest); atomic_json(cp, meta)
    path = c.root/'configs/rtd/legacy_identities'/f'{digest(manifest)}.json'
    saved = dict(manifest_hash=digest(manifest), legacy_harness_hash='historical-mixed',
                 evaluation_harness=old, harness_hash=digest(old),
                 rtd_source=dict(kind='legacy-mixed-harness', hash='historical-mixed', files=None),
                 audit=dict(manifest_mtime_ns=ns, evidence='original audit'), identity_migrations=[{'task': 'C25g'}])
    atomic_json(path, saved)
    dest = c.root/update.EVIDENCE_PATH; dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT/update.EVIDENCE_PATH, dest)
    for row in bundle['files']:
        put(c.root/row['path'], (ROOT/row['path']).read_text())
    before = snapshot(c.directory)
    result = update.update_identity(c.root, c.directory)
    assert result['updated']
    assert [r['path'] for r in result['audit']['changed_old_identity_files']] == ['tools/bfcl_std_campaign.sh']
    supplement = json.loads(path.read_text())
    assert supplement['audit'] == saved['audit'] and supplement['identity_migrations'] == saved['identity_migrations']
    assert identity.guard_harness(c.root, c.directory, manifest) == supplement
    assert snapshot(c.directory) == before
    # Archived evidence cannot authorize a new scoring edit, even with an old mtime.
    p = c.root/'src/bfas/rtd/evaluation.py'
    put(p, p.read_text().replace('torch.bfloat16', 'torch.float32'))
    os.utime(p, ns=(ns-1, ns-1))
    saved_bytes = path.read_bytes()
    with pytest.raises(update.IdentityUpdateRefused):
        update.update_identity(c.root, c.directory)
    assert path.read_bytes() == saved_bytes


def test_mixed_manifest_without_supplement_records_unrecoverable_inventory(run):
    c = run
    manifest = {k: v for k, v in c.manifest.items() if k not in {'evaluation_harness', 'rtd_source'}}
    atomic_json(c.directory/'manifest.json', manifest)
    os.utime(c.directory/'manifest.json', ns=(c.ns, c.ns))
    cp = c.directory/'round-1/checkpoint.json'
    meta = json.loads(cp.read_text()); meta['manifest_hash'] = digest(manifest); atomic_json(cp, meta)
    result = update.update_identity(c.root, c.directory)
    assert result['audit']['old_inventory_unavailable'] and result['audit']['old_identity_files_checked'] == 0
    assert identity.guard_harness(c.root, c.directory, manifest)['rtd_source']['files'] is None
