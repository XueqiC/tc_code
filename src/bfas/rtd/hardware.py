"""Protocol 1.0.4 device classes and manifest-bound, CPU-only migration."""
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import socket
import subprocess

from .persistence import ComputeJournal, atomic_json, digest


VERSION = 'rtd-hardware-class-v1'
PACKAGES = ('torch', 'transformers', 'peft', 'numpy')
LEGACY_HARD = ('gpu', 'capability', 'memory', 'cuda', 'versions', 'python', 'machine')


def host_class(hostname, gpu, env=None):
    """Use a scheduler class; a local machine retains its literal hostname.

    The explicit historical HPG mapping also works on copied, pre-SLURM-metadata
    manifests and on login nodes. It cannot map an arbitrary host/model to HPG.
    """
    env = os.environ if env is None else env
    partition = env.get('SLURM_JOB_PARTITION', env.get('SLURM_PARTITION', ''))
    cluster = env.get('SLURM_CLUSTER_NAME', '')
    hpg = (partition == 'hpg-b200' or cluster.lower() in {'hpg', 'hipergator'}
           or re.fullmatch(r'c\d+[a-z]-s\d+(?:\..*)?', hostname) is not None)
    if hpg and re.search(r'\bB200\b', gpu, re.I):
        return 'hpg-b200'
    return cluster or partition or hostname


def driver_version():
    # Kernel files remain readable when the process has no CUDA device access.
    path = Path('/sys/module/nvidia/version')
    if path.is_file():
        return path.read_text().strip()
    path = Path('/proc/driver/nvidia/version')
    if path.is_file():
        match = re.search(r'Kernel Module.*?\s(\d+\.\d+(?:\.\d+)?)\s', path.read_text())
        if match:
            return match[1]
    result = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
                            check=True, capture_output=True, text=True, timeout=10)
    versions = set(result.stdout.split())
    if len(versions) != 1:
        raise ValueError('cannot identify a unique NVIDIA driver version')
    return versions.pop()


def device_inventory():
    """Read driver-exported model/UUID/PCI inventory; never initialize CUDA."""
    rows = []
    for path in sorted(Path('/proc/driver/nvidia/gpus').glob('*/information')):
        fields = dict(line.split(':', 1) for line in path.read_text().splitlines() if ':' in line)
        fields = {k.strip(): v.strip() for k, v in fields.items()}
        rows.append(dict(gpu=fields['Model'], uuid=fields['GPU UUID'].removeprefix('GPU-'),
                         pci_bus_id=fields['Bus Location']))
    return rows


def hardware_identity():
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError('exactly one CUDA GPU is required')
    props = torch.cuda.get_device_properties(0)
    hostname = socket.gethostname()
    uuid = str(getattr(props, 'uuid', 'unknown')).removeprefix('GPU-')
    pci = next((row['pci_bus_id'] for row in device_inventory() if row['uuid'] == uuid), None)
    return dict(version=VERSION, hard=dict(
        gpu=props.name, capability=[props.major, props.minor], memory=props.total_memory,
        cuda=torch.version.cuda, driver=driver_version(),
        versions={name: importlib.metadata.version(name) for name in PACKAGES},
        python=platform.python_version(), machine=platform.machine(),
        host_class=host_class(hostname, props.name)), metadata=dict(
        hostname=hostname, uuid=uuid, pci_bus_id=pci,
        cuda_device_order=os.environ.get('CUDA_DEVICE_ORDER', 'FASTEST_FIRST')))


def checked_hardware(hardware):
    if (hardware.get('version') != VERSION
            or set(hardware.get('hard', {})) != {*LEGACY_HARD, 'driver', 'host_class'}
            or set(hardware['hard'].get('versions', {})) != set(PACKAGES)
            or any(hardware['hard'][k] in (None, '', 'unknown') for k in ('driver', 'host_class'))
            or not {'hostname', 'uuid', 'pci_bus_id', 'cuda_device_order'} <= set(hardware.get('metadata', {}))):
        raise ValueError('invalid hardware class identity')
    return hardware


def instance(hardware):
    return hardware.get('metadata', hardware)


def device_class(hardware):
    return hardware.get('hard', hardware)


def supplement_path(root, manifest):
    return Path(root)/'configs/rtd/hardware_identities'/f'{digest(manifest)}.json'


def _require_same_class(old, new, context):
    if old != new:
        differences = {k: dict(old=old.get(k), new=new.get(k))
                       for k in sorted(set(old) | set(new)) if old.get(k) != new.get(k)}
        raise ValueError(f'hardware class differs {context}: ' + json.dumps(differences, sort_keys=True))


def _legacy_projection(old, *, driver, expected_host_class=None):
    if not all(k in old for k in (*LEGACY_HARD, 'hostname', 'uuid')):
        raise ValueError('incomplete legacy hardware identity')
    # Never use today's scheduler environment to reinterpret a historical host.
    label = old.get('host_class') or host_class(old['hostname'], old['gpu'], {})
    if expected_host_class is not None and label != expected_host_class:
        raise ValueError(f'hardware class differs: saved host class {label!r}, requested {expected_host_class!r}')
    if old.get('driver', driver) != driver:
        raise ValueError('hardware class differs: saved driver version')
    return checked_hardware(dict(version=VERSION,
        hard=dict({k: old[k] for k in LEGACY_HARD}, host_class=label, driver=driver),
        metadata=dict(hostname=old['hostname'], uuid=old['uuid'].removeprefix('GPU-'),
                      pci_bus_id=old.get('pci_bus_id'), cuda_device_order=old.get('cuda_device_order'))))


def bound_hardware(root, manifest):
    hardware = manifest['hardware']
    if hardware.get('version') == VERSION:
        checked_hardware(hardware)
        if manifest['hardware_hash'] != digest(hardware['hard']):
            raise ValueError('hardware class hash mismatch')
        return hardware
    if manifest['hardware_hash'] != digest(hardware):
        raise ValueError('legacy hardware hash mismatch')
    path = supplement_path(root, manifest)
    if not path.is_file():
        raise ValueError('legacy hardware needs update-hardware-identity --run-dir before resume/evaluation')
    saved = json.loads(path.read_text())
    new = checked_hardware(saved['hardware'])
    if (saved.get('version') != VERSION or saved.get('manifest_hash') != digest(manifest)
            or saved.get('old_hardware') != hardware or saved.get('old_hardware_hash') != manifest['hardware_hash']
            or new != _legacy_projection(hardware, driver=new['hard']['driver'])
            or saved.get('hardware_hash') != digest(new['hard'])
            or saved.get('audit', {}).get('new_hardware_hash') != saved['hardware_hash']):
        raise ValueError('hardware supplement binding mismatch')
    return new


def guard_hardware(root, directory, manifest, current, *, context):
    saved = bound_hardware(root, manifest)
    checked_hardware(current)
    _require_same_class(saved['hard'], current['hard'], 'from training')
    journal = ComputeJournal(Path(directory)/'device_instances.jsonl')
    previous = journal.events[-1]['new_metadata'] if journal.events else saved['metadata']
    if previous != current['metadata']:
        journal.append('device_instance_changed', protocol_version='1.0.4', context=context,
            manifest_hash=digest(manifest), hardware_hash=digest(saved['hard']),
            original_metadata=saved['metadata'], old_metadata=previous, new_metadata=current['metadata'])
    return saved


def comparison_hash(root, manifest):
    # Unmigrated historical reports retain their conservative instance grouping.
    if ('hardware' not in manifest or (manifest['hardware'].get('version') != VERSION
                                      and not supplement_path(root, manifest).exists())):
        return manifest['hardware_hash']
    return digest(bound_hardware(root, manifest)['hard'])


def update_hardware_identity(root, directory, *, expected_host_class=None, driver=None, reference=None):
    """Project saved facts, optionally compare a reference class, write no run files.

    Driver versions absent in old manifests are explicitly established now;
    neither current driver files nor an operator value prove the historical one.
    All previously recorded class fields must be retained exactly.
    """
    from .identity import _file_stamp, verified_checkpoint
    from .identity_update import _supplement_directory_lock
    root, directory = Path(root).resolve(), Path(directory).resolve()
    path = directory/'manifest.json'
    original, stamp = path.read_bytes(), _file_stamp(path)
    manifest = json.loads(original)
    if digest(manifest['config']) != manifest['config_hash']:
        raise ValueError('saved config hash mismatch')
    old = manifest['hardware']
    dest = supplement_path(root, manifest)
    previous = dest.read_bytes() if dest.exists() else None
    if old.get('version') == VERSION or previous is not None:
        new = bound_hardware(root, manifest)
        requested = dict(new['hard'])
        if expected_host_class is not None:
            requested['host_class'] = expected_host_class
        if driver is not None:
            requested['driver'] = driver
        _require_same_class(new['hard'], requested, 'from existing binding')
        if reference is not None:
            _require_same_class(new['hard'], checked_hardware(reference)['hard'], 'from reference identity')
        return dict(updated=False, supplement_path=str(dest), hardware_hash=digest(new['hard']), hardware=new)
    if digest(old) != manifest['hardware_hash']:
        raise ValueError('legacy hardware hash mismatch')
    supplied_driver = driver is not None
    same_host = socket.gethostname() == old['hostname']
    if driver is None and not same_host:
        raise ValueError('migration on another host requires --driver-version from the target compute node')
    driver = driver or driver_version()
    new = _legacy_projection(old, driver=driver, expected_host_class=expected_host_class)
    if reference is not None:
        _require_same_class(new['hard'], checked_hardware(reference)['hard'], 'from reference identity')
    observations = {}
    if same_host:
        import torch
        observed = dict(versions={name: importlib.metadata.version(name) for name in PACKAGES},
                        python=platform.python_version(), machine=platform.machine(), cuda=torch.version.cuda)
        _require_same_class({k: new['hard'][k] for k in observed}, observed, 'in local software environment')
        rows = [row for row in device_inventory() if row['uuid'] == new['metadata']['uuid']]
        if rows and any(row['gpu'] != new['hard']['gpu'] for row in rows):
            raise ValueError('hardware class differs: local GPU inventory')
        observations = dict(software=observed, matching_devices=rows)
    checkpoints = {p.name: verified_checkpoint(directory, manifest, int(p.name.removeprefix('round-')))
                   for p in sorted(directory.glob('round-[123]')) if p.is_dir()}
    audit = dict(task='C25n', protocol_version='1.0.4', recorded_at=datetime.now(timezone.utc).isoformat(),
        method='immutable-manifest-device-class-projection-v1', new_hardware_hash=digest(new['hard']),
        retained_hard_fields=list(LEGACY_HARD), observations=observations,
        driver_basis='operator-supplied' if supplied_driver else 'local-driver-version',
        driver_was_recorded='driver' in old, verified_checkpoints=checkpoints,
        limitation='Legacy driver version and PCI order were not recorded. Driver is established at migration, '
                   'not proven historically; missing instance metadata remains null. Saved model, capability, '
                   'memory and software fields are preserved; the full live class is checked before GPU work.')
    supplement = dict(version=VERSION, manifest_hash=digest(manifest), old_hardware=old,
                      old_hardware_hash=manifest['hardware_hash'], hardware=new,
                      hardware_hash=digest(new['hard']), audit=audit)
    def unchanged():
        if (original != path.read_bytes() or stamp != _file_stamp(path)
                or previous != (dest.read_bytes() if dest.exists() else None)
                or checkpoints != {p.name: verified_checkpoint(directory, manifest, int(p.name.removeprefix('round-')))
                                   for p in sorted(directory.glob('round-[123]')) if p.is_dir()}):
            raise ValueError('manifest, checkpoint or supplement changed during hardware audit')
    with _supplement_directory_lock(dest.parent):
        unchanged()
        atomic_json(dest, supplement)
    return dict(updated=True, supplement_path=str(dest), **supplement)
