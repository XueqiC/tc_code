"""C25n: CPU-only scheduler class guards, audited migration and local pinning."""
import copy
import json
import os
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT)]
from bfas.rtd import cli, hardware, identity, evaluation
from bfas.rtd.persistence import ComputeJournal, atomic_json, digest, file_hash
from rtd_identity_fixtures import hardware_fixture
from test_rtd_evaluation_resume import campaign, checkpoint


@pytest.fixture(autouse=True)
def no_cuda(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('hardware identity tests must not initialize CUDA')
    monkeypatch.setattr(torch.cuda, '_lazy_init', forbidden)


@pytest.mark.parametrize('hostname,gpu,env,expected', [
    ('rai', 'NVIDIA B200', {}, 'rai'),
    ('rai', 'NVIDIA A100', {}, 'rai'),
    ('c1100a-s25', 'NVIDIA B200', {}, 'hpg-b200'),
    ('c1100a-s26', 'NVIDIA B200', {'SLURM_CLUSTER_NAME': 'hipergator'}, 'hpg-b200'),
    ('node42', 'NVIDIA B200', {'SLURM_JOB_PARTITION': 'hpg-b200'}, 'hpg-b200'),
    ('node42', 'NVIDIA B200', {'SLURM_PARTITION': 'hpg-b200'}, 'hpg-b200'),
    ('node42', 'NVIDIA A100', {'SLURM_CLUSTER_NAME': 'cluster-a'}, 'cluster-a'),
    ('node43', 'NVIDIA A100', {'SLURM_CLUSTER_NAME': 'cluster-a'}, 'cluster-a'),
])
def test_host_classes(hostname, gpu, env, expected):
    assert hardware.host_class(hostname, gpu, env) == expected


def test_live_identity_separates_class_from_instance(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 1)
    props = SimpleNamespace(name='NVIDIA B200', total_memory=192_000_000_000,
                            major=10, minor=0, uuid='GPU-original')
    monkeypatch.setattr(torch.cuda, 'get_device_properties', lambda n: props)
    monkeypatch.setattr(hardware, 'driver_version', lambda: '580.95.05')
    monkeypatch.setattr(hardware, 'device_inventory', lambda: [dict(uuid='original', pci_bus_id='0000:41:00.0')])
    monkeypatch.setattr(hardware.socket, 'gethostname', lambda: 'c1100a-s25')
    monkeypatch.setenv('SLURM_JOB_PARTITION', 'hpg-b200')
    monkeypatch.setenv('CUDA_DEVICE_ORDER', 'PCI_BUS_ID')
    first = hardware.hardware_identity()
    props.uuid = 'GPU-next'
    monkeypatch.setattr(hardware.socket, 'gethostname', lambda: 'c1100a-s26')
    monkeypatch.setenv('CUDA_DEVICE_ORDER', 'FASTEST_FIRST')
    second = hardware.hardware_identity()
    assert hardware.checked_hardware(first)['hard'] == second['hard']
    assert first['metadata'] != second['metadata']
    assert first['metadata']['pci_bus_id'] == '0000:41:00.0'


def test_same_class_resume_accepts_uuid_host_pci_changes_and_records_transitions(campaign):
    c = campaign
    current = copy.deepcopy(cli.make_manifest(None))
    original = current['hardware']['metadata'].copy()
    current['hardware']['metadata'].update(uuid='6b5cf600-next', hostname='c1100a-s26',
                                           pci_bus_id='0000:81:00.0', cuda_device_order='FASTEST_FIRST')
    before = (c.directory/'manifest.json').read_bytes()
    identity.validate_resume(c.root, c.directory, c.manifest, current)
    identity.validate_resume(c.root, c.directory, c.manifest, current)
    journal = ComputeJournal(c.directory/'device_instances.jsonl')
    event, = journal.events
    assert event['kind'] == 'device_instance_changed'
    assert event['old_metadata'] == event['original_metadata'] == original
    assert event['new_metadata'] == current['hardware']['metadata']
    assert event['hardware_hash'] == c.manifest['hardware_hash']
    identity.validate_resume(c.root, c.directory, c.manifest, cli.make_manifest(None))
    assert len(ComputeJournal(c.directory/'device_instances.jsonl').events) == 2
    assert (c.directory/'manifest.json').read_bytes() == before and not c.calls


@pytest.mark.parametrize('field,value', [('gpu', 'NVIDIA H100'), ('capability', [9, 0]),
    ('memory', 80_000_000_000), ('cuda', '12.8'), ('driver', 'different'),
    ('python', '3.13'), ('machine', 'aarch64'), ('host_class', 'different-cluster'),
    *[('versions.' + name, 'changed') for name in hardware.PACKAGES]])
def test_resume_refuses_each_hard_class_change_even_with_code_ack(campaign, field, value):
    c = campaign
    current = copy.deepcopy(cli.make_manifest(None))
    hard = current['hardware']['hard']
    if '.' in field:
        hard['versions'][field.split('.')[1]] = value
    else:
        hard[field] = value
    current['hardware_hash'] = digest(hard)
    with pytest.raises(ValueError, match='hardware class differs'):
        identity.validate_resume(c.root, c.directory, c.manifest, current, acknowledge=True)
    assert not c.calls and not (c.directory/'device_instances.jsonl').exists()


def test_evaluation_accepts_same_class_instance_change_and_reuses_score(campaign, monkeypatch):
    c = campaign
    first = evaluation.evaluate(c.root, c.directory, 1, port=0)
    current = copy.deepcopy(c.manifest['hardware'])
    current['metadata'].update(uuid='6b5cf600-next', hostname='c1100a-s26')
    monkeypatch.setattr(cli, 'hardware_identity', lambda: current)
    calls = len(c.calls)
    second = evaluation.evaluate(c.root, c.directory, 1, port=0)
    assert first['identity'] == second['identity'] and len(c.calls) == calls
    event, = ComputeJournal(c.directory/'device_instances.jsonl').events
    assert event['context'] == 'evaluation-1' and event['new_metadata'] == current['metadata']
    current['hard']['gpu'] = 'NVIDIA H100'
    with pytest.raises(ValueError, match='hardware class differs'):
        evaluation.evaluate(c.root, c.directory, 1, port=0)


def test_base_comparison_rejects_another_class_before_campaign(campaign):
    c = campaign
    base = evaluation.evaluate(c.root, c.directory, 1, port=0)
    base['hardware_class'] = dict(base['hardware_class'], gpu='NVIDIA H100')
    base['hardware_class_hash'] = digest(base['hardware_class'])
    atomic_json(c.root/'base-evaluation.json', base)
    calls = len(c.calls)
    with pytest.raises(ValueError, match='same audited hardware class'):
        evaluation.evaluate(c.root, c.directory, 1, base_evaluation=c.root/'base-evaluation.json')
    assert len(c.calls) == calls


@pytest.fixture
def legacy(campaign, monkeypatch):
    c = campaign
    current = c.manifest['hardware']
    c.manifest['hardware'] = {**{k: current['hard'][k] for k in hardware.LEGACY_HARD},
                             **{k: current['metadata'][k] for k in ('hostname', 'uuid')}}
    c.manifest['hardware_hash'] = digest(c.manifest['hardware'])
    atomic_json(c.directory/'manifest.json', c.manifest)
    checkpoint(c.directory, c.manifest, 1)
    monkeypatch.setattr(hardware.socket, 'gethostname', lambda: 'migration-login-node')
    c.reference = current
    return c


def snapshot(directory):
    return {str(p.relative_to(directory)): (file_hash(p), p.stat().st_mtime_ns)
            for p in directory.rglob('*') if p.is_file()}


def test_migration_lists_old_new_preserves_run_and_is_idempotent_without_git(legacy, capsys):
    c = legacy
    before = snapshot(c.directory)
    args = ['update-hardware-identity', '--run-dir', str(c.directory), '--host-class', 'hpg-b200',
            '--driver-version', '580.95.05']
    assert cli.main(args) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['updated'] and result['old_hardware'] == c.manifest['hardware']
    assert result['old_hardware_hash'] == c.manifest['hardware_hash']
    assert result['hardware']['hard'] == c.reference['hard']
    assert not result['audit']['driver_was_recorded']
    assert result['hardware']['metadata']['cuda_device_order'] is None
    assert result['audit']['verified_checkpoints']['round-1']['manifest_hash'] == digest(c.manifest)
    assert snapshot(c.directory) == before
    path = Path(result['supplement_path'])
    stamp, data = path.stat().st_mtime_ns, path.read_bytes()
    assert cli.main(args) == 0
    assert not json.loads(capsys.readouterr().out)['updated']
    assert path.stat().st_mtime_ns == stamp and path.read_bytes() == data
    current = copy.deepcopy(c.manifest)
    current.update(hardware=c.reference, hardware_hash=digest(c.reference['hard']))
    identity.validate_resume(c.root, c.directory, c.manifest, current)
    assert hardware.comparison_hash(c.root, c.manifest) == current['hardware_hash']


@pytest.mark.parametrize('change', ['host', 'model', 'memory', 'driver', 'checkpoint', 'hash'])
def test_migration_refuses_changed_class_or_corrupt_binding_without_writes(legacy, change):
    c = legacy
    reference = copy.deepcopy(c.reference)
    label = 'hpg-b200'
    if change == 'host':
        label = 'rai'
    elif change in {'model', 'memory', 'driver'}:
        reference['hard'][{'model': 'gpu'}.get(change, change)] = 'different'
    elif change == 'checkpoint':
        (c.directory/'round-1/lora/adapter.safetensors').write_text('corruption')
    else:
        c.manifest['hardware']['gpu'] = 'different'
        atomic_json(c.directory/'manifest.json', c.manifest)
    before = snapshot(c.directory)
    with pytest.raises(ValueError, match='differs|mismatch'):
        hardware.update_hardware_identity(c.root, c.directory, expected_host_class=label,
                                          driver='580.95.05', reference=reference)
    assert not hardware.supplement_path(c.root, c.manifest).exists()
    assert snapshot(c.directory) == before


def test_existing_migration_refuses_new_driver_and_tampering(legacy):
    c = legacy
    result = hardware.update_hardware_identity(c.root, c.directory, driver='580.95.05')
    path = Path(result['supplement_path'])
    before = path.read_bytes()
    with pytest.raises(ValueError, match='class differs'):
        hardware.update_hardware_identity(c.root, c.directory, driver='different')
    assert path.read_bytes() == before
    saved = json.loads(before)
    saved['hardware']['hard']['gpu'] = 'NVIDIA H100'
    saved['hardware_hash'] = digest(saved['hardware']['hard'])
    saved['audit']['new_hardware_hash'] = saved['hardware_hash']
    atomic_json(path, saved)
    with pytest.raises(ValueError, match='binding mismatch'):
        hardware.bound_hardware(c.root, c.manifest)


def test_unmigrated_legacy_fails_closed(legacy):
    c = legacy
    with pytest.raises(ValueError, match='update-hardware-identity'):
        hardware.guard_hardware(c.root, c.directory, c.manifest, c.reference, context='test')


@pytest.mark.parametrize('change', ['none', 'gpu', 'software', 'concurrent-manifest'])
def test_local_cpu_migration_observations_and_race_refusal(legacy, monkeypatch, change):
    c = legacy
    old = c.manifest['hardware']
    monkeypatch.setattr(hardware.socket, 'gethostname', lambda: old['hostname'])
    monkeypatch.setattr(hardware, 'driver_version', lambda: '580.95.05')
    monkeypatch.setattr(hardware.importlib.metadata, 'version', lambda name: 'changed' if change == 'software' else 'fixture')
    monkeypatch.setattr(hardware.platform, 'python_version', lambda: old['python'])
    monkeypatch.setattr(hardware.platform, 'machine', lambda: old['machine'])
    monkeypatch.setattr(torch.version, 'cuda', old['cuda'])
    def inventory():
        if change == 'concurrent-manifest':
            atomic_json(c.directory/'manifest.json', dict(c.manifest, concurrent='writer'))
        return [dict(uuid=old['uuid'], gpu='NVIDIA H100' if change == 'gpu' else old['gpu'],
                     pci_bus_id='0000:41:00.0')]
    monkeypatch.setattr(hardware, 'device_inventory', inventory)
    if change == 'none':
        result = hardware.update_hardware_identity(c.root, c.directory)
        assert result['audit']['driver_basis'] == 'local-driver-version'
        assert result['audit']['observations']['matching_devices'] == inventory()
    else:
        with pytest.raises(ValueError, match='differs|changed during'):
            hardware.update_hardware_identity(c.root, c.directory)
        assert not hardware.supplement_path(c.root, c.manifest).exists()


@pytest.mark.parametrize('scheduler,explicit,expected', [
    (False, None, 'GPU-4200c43f-original'), (False, 'GPU-override', 'GPU-override'),
    (False, '', ''), (True, None, None), (True, '0', '0'),
])
@pytest.mark.parametrize('nested', [False, True])
def test_launcher_defaults_local_run_to_uuid_before_import(scheduler, explicit, expected, nested, tmp_path, monkeypatch):
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    monkeypatch.delenv('SLURM_JOB_ID', raising=False)
    monkeypatch.delenv('SLURM_CLUSTER_NAME', raising=False)
    if scheduler:
        monkeypatch.setenv('SLURM_JOB_ID', '41293740')
    if explicit is not None:
        monkeypatch.setenv('CUDA_VISIBLE_DEVICES', explicit)
    hw = hardware_fixture()
    atomic_json(tmp_path/'manifest.json', dict(hardware=hw if nested else hw['metadata']))
    launcher = runpy.run_path(str(ROOT/'tools/rtd_experiment.py'), run_name='test_entry')
    launcher['pin_local_resume'](['resume', '--run-dir', str(tmp_path)])
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == expected
