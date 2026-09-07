"""Frozen evaluation provenance and append-only RTD code-drift metadata.

The original manifest is immutable: recovery and round checkpoints hash it.
Legacy supplements are audited, content-bound records, never inferred from a
run directory name or silently initialized from the current environment.
"""
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

from .persistence import ComputeJournal, atomic_json, digest, file_hash, tree_hash
from .scoring import ScoreTolerance
from .scoring_scope import scoring_hash


LEADERBOARD = 'envs/bfcl/gorilla/berkeley-function-call-leaderboard'
EVALUATION_TOOLS = ('tools/behavior_atom/checker_bridge.py',
                    'tools/bfcl_event_mine_single.py',
                    'tools/bfcl_std_campaign.sh', 'tools/bfcl_generation_check.py',
                    'tools/bfcl_hub_merge_export.py', 'src/bfas/rtd/evaluation.py',
                    'src/bfas/adapters/bfcl.py')
SOURCE_PATHS = ('src', 'tools', 'scripts', *(LEADERBOARD+'/'+n for n in
                ('pyproject.toml', 'setup.py', 'setup.cfg', 'requirements.txt')))
CONTENT_VERSION = 'bfcl-evaluation-harness-scoring-v4'


def source_identity(root):
    root = Path(root)
    files = {}
    for name in SOURCE_PATHS:
        path = root / name
        paths = [path] if path.is_file() else sorted(p for p in path.rglob('*')
                    if p.is_file() and p.suffix in {'.py', '.sh', '.slurm'})
        for p in paths:
            if '__pycache__' not in p.parts:
                files[str(p.relative_to(root))] = file_hash(p)
    if not files:
        raise ValueError('missing RTD source tree')
    return dict(kind='rtd-source-v2', hash=digest(files), files=files)


def _evaluation_harness_paths(root):
    """One inventory for both the frozen identity and legacy mtime evidence."""
    root = Path(root)
    leaderboard = root / LEADERBOARD
    # Source and package resources belong to the checkout; venv wrappers can
    # contain absolute shebangs and must not enter a portable content identity.
    package = {p.relative_to(leaderboard).as_posix(): p
             for p in sorted((leaderboard/'bfcl_eval').rglob('*'))
             if p.is_file() and not {'.git', '__pycache__'} & set(p.relative_to(leaderboard).parts)
             and p.suffix not in {'.pyc', '.pyo'}}
    data = {n: p for n, p in package.items() if n.startswith('bfcl_eval/data/')}
    files = {n: p for n, p in package.items() if n not in data}
    if not files:
        raise ValueError('missing BFCL leaderboard checkout')
    if not data:
        raise ValueError('missing BFCL leaderboard data')
    return files, data


def _content_manifest(paths):
    return {name: dict(size=p.stat().st_size, sha256=file_hash(p))
            for name, p in sorted(paths.items())}


def evaluation_harness_identity(root, config):
    """Only relative names and file/config contents; no git or install state."""
    root = Path(root)
    files, data = _evaluation_harness_paths(root)
    checkout, data_manifest = _content_manifest(files), _content_manifest(data)
    return dict(version=CONTENT_VERSION, leaderboard=LEADERBOARD,
                checkout_files=checkout, checkout_hash=digest(checkout),
                data_manifest=data_manifest, data_hash=digest(data_manifest),
                tools={p: scoring_hash(p, (root/p).read_bytes().decode('utf-8')) for p in EVALUATION_TOOLS},
                config={k: v for k, v in config.items() if k.startswith('evaluation_')})


def _content_v2_identity(root, config, tool_names):
    """Historical format, only for verifying C25g's explicit v1 migration."""
    files, data = _evaluation_harness_paths(root)
    for name in ('pyproject.toml', 'setup.py', 'setup.cfg', 'requirements.txt'):
        if (Path(root)/LEADERBOARD/name).is_file():
            files[name] = Path(root)/LEADERBOARD/name
    checkout, data_manifest = _content_manifest(files), _content_manifest(data)
    return dict(version='bfcl-evaluation-harness-content-v2', leaderboard=LEADERBOARD,
                checkout_files=checkout, checkout_hash=digest(checkout),
                data_manifest=data_manifest, data_hash=digest(data_manifest),
                tools={p: file_hash(Path(root)/p) for p in tool_names},
                config={k: v for k, v in config.items() if k.startswith('evaluation_')})


def evaluation_harness_metadata(root):
    """Best-effort HEAD, kept outside every evaluation/resume hard guard.

    Gorilla normally owns .git one directory above the leaderboard. Do not
    accidentally attribute the enclosing tc-alignment repository to BFCL.
    """
    leaderboard = Path(root) / LEADERBOARD
    if not any((p/'.git').is_dir() for p in (leaderboard, leaderboard.parent)):
        return {}
    try:
        revision = subprocess.run(['git', '-C', str(leaderboard), 'rev-parse', 'HEAD'],
                                  capture_output=True, text=True, check=True, timeout=5).stdout.strip()
        return dict(checkout_revision=revision) if revision else {}
    except (OSError, subprocess.SubprocessError):
        return {}


def _file_stamp(path):
    stat = path.stat()
    return stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size, stat.st_dev, stat.st_ino


def _harness_file_stamps(root):
    root = Path(root)
    files, data = _evaluation_harness_paths(root)
    paths = [*files.values(), *data.values(), *(root/p for p in EVALUATION_TOOLS)]
    return {str(p.relative_to(root)): _file_stamp(p) for p in sorted(paths)}


def _migrate_legacy_content(root, directory, manifest, path, saved):
    """Explicit v1 -> content audit; never infer a binding during evaluation."""
    manifest_path = directory/'manifest.json'
    original_manifest, original_supplement = manifest_path.read_bytes(), path.read_bytes()
    if json.loads(original_manifest) != manifest or json.loads(original_supplement) != saved:
        raise ValueError('manifest or supplement changed during legacy audit')
    before = _harness_file_stamps(root)
    old = saved['evaluation_harness']
    harness = _content_v2_identity(root, manifest['config'], old['tools'])
    # v1 already pinned source/package and tool contents. Its git HEAD and
    # installed wrapper/metadata are historical provenance only after C25g.
    old_checkout_hash = digest({n: f['sha256'] for n, f in harness['checkout_files'].items()})
    if (old['leaderboard'] != harness['leaderboard'] or old['checkout_hash'] != old_checkout_hash
            or old['tools'] != harness['tools'] or old['config'] != harness['config']):
        raise ValueError('evaluation harness content differs from legacy supplement; migration refused')
    # Data was outside v1's harness but bound by the original manifest. Verify
    # that binding before incorporating data into the new content identity.
    from .cli import data_identity
    bank = root / manifest['config']['replay_bank_path']
    if data_identity(root, manifest['config'], bank) != manifest['data_hash']:
        raise ValueError('legacy manifest data hash differs; content migration refused')
    if before != _harness_file_stamps(root):
        raise ValueError('evaluation harness files changed during legacy audit')
    if original_manifest != manifest_path.read_bytes() or original_supplement != path.read_bytes():
        raise ValueError('manifest or supplement changed during legacy audit')
    note = dict(task='C25g', method='verified-v1-content-and-manifest-data-v2',
                recorded_at=datetime.now(timezone.utc).isoformat(),
                previous_harness_hash=saved['harness_hash'], previous_evaluation_harness=old,
                content_harness_hash=digest(harness), verified_manifest_data_hash=manifest['data_hash'],
                evidence='The v1 checkout content hash, tool hashes and evaluation config match; '
                         'the original manifest data hash matches before adding the data manifest.',
                note='Migrated to relative-path content identity. Git HEAD and installed venv '
                     'wrapper/package metadata are retained above as historical provenance only. '
                     'Original run manifest, checkpoints, RTD source identity and audit are unchanged.')
    migrated = dict(saved, evaluation_harness=harness, harness_hash=digest(harness),
                    evaluation_harness_metadata=evaluation_harness_metadata(root),
                    identity_migrations=[*saved.get('identity_migrations', []), note])
    atomic_json(path, migrated)
    return dict(supplement_path=str(path), created=False, **migrated)


def audit_legacy(root, directory):
    """Explicit CPU audit; only the manifest-bound supplement may be written.

    Mtimes corroborate historical continuity, not historical content hashes.
    Re-audits must pass again. A v1 supplement may migrate after verifying its
    content and the original data binding, retaining the old identity as evidence.
    """
    root, directory = Path(root), Path(directory)
    manifest_path = directory/'manifest.json'
    manifest_stamp = _file_stamp(manifest_path)
    original = manifest_path.read_bytes()
    manifest = json.loads(original)
    if {'evaluation_harness', 'rtd_source'} & manifest.keys():
        raise ValueError('manifest already carries a split identity; legacy audit refused')
    if (not isinstance(manifest.get('harness_hash'), str) or not manifest['harness_hash']
            or not isinstance(manifest.get('config'), dict)
            or digest(manifest['config']) != manifest.get('config_hash')):
        raise ValueError('legacy manifest needs a combined harness hash and a valid saved config hash')
    path = root/'configs/rtd/legacy_identities'/f'{digest(manifest)}.json'
    if path.exists():
        saved = saved_identities(root, directory, manifest)
        if saved['evaluation_harness']['version'] == 'bfcl-evaluation-harness-v1':
            return _migrate_legacy_content(root, directory, manifest, path, saved)
    before = _harness_file_stamps(root)
    mtimes = {name: stamp[0] for name, stamp in before.items()}
    too_recent = {name: ns for name, ns in mtimes.items() if ns >= manifest_stamp[0]}
    if too_recent:
        raise ValueError('evaluation harness files must strictly predate manifest '
                         f'(manifest_mtime_ns={manifest_stamp[0]}); newer or equal file mtimes: '
                         + json.dumps(too_recent, sort_keys=True))
    harness = evaluation_harness_identity(root, manifest['config'])
    if before != _harness_file_stamps(root):
        raise ValueError('evaluation harness files changed during legacy audit')
    if manifest_stamp != _file_stamp(manifest_path) or original != manifest_path.read_bytes():
        raise ValueError('manifest changed during legacy audit')
    supplement = dict(
        version='rtd-c25e-audited-legacy-identity-v1',
        manifest_hash=digest(manifest), legacy_harness_hash=manifest['harness_hash'],
        evaluation_harness=harness, harness_hash=digest(harness),
        evaluation_harness_metadata=evaluation_harness_metadata(root),
        rtd_source=dict(kind='legacy-mixed-harness', hash=manifest['harness_hash'], files=None),
        audit=dict(
            method='harness-mtime-before-manifest-v1',
            recorded_at=datetime.now(timezone.utc).isoformat(),
            manifest_mtime_ns=manifest_stamp[0], harness_files_checked=len(mtimes),
            latest_harness_mtime_ns=max(mtimes.values()),
            latest_harness_file=max(mtimes, key=mtimes.get), file_mtimes_ns=mtimes,
            evidence='All current evaluation harness file mtimes strictly predate the saved manifest; '
                     'the manifest and harness file inventory/stats remained unchanged during hashing.',
            limitation='The legacy combined digest cannot reconstruct a separate historical evaluation-harness '
                       'or RTD source digest. This supplement pins the audited harness for this exact original '
                       'manifest; mtimes are corroboration, not cryptographic proof of historical contents.',
            training_resume_requires='--acknowledge-code-drift'))
    created = not path.exists()
    if created:
        atomic_json(path, supplement)
    else:
        saved = saved_identities(root, directory, manifest)
        if any(saved[key] != supplement[key] for key in ('evaluation_harness', 'harness_hash', 'rtd_source')):
            raise ValueError('existing legacy supplement differs from audited identity; refusing to overwrite')
    return dict(supplement_path=str(path), created=created, **supplement)


def saved_identities(root, directory, manifest):
    path = Path(root)/'configs/rtd/legacy_identities'/f'{digest(manifest)}.json'
    if 'evaluation_harness' in manifest:
        if digest(manifest['evaluation_harness']) != manifest['harness_hash']:
            raise ValueError('manifest evaluation harness hash mismatch')
        if not path.is_file():
            return manifest
    # C25e's legacy audit is keyed by the FULL original manifest hash. A copied
    # directory or another checkpoint cannot inherit its authorization by name.
    if not path.is_file():
        raise ValueError('legacy run needs an audited, manifest-bound evaluation identity supplement')
    saved = json.loads(path.read_text())
    if saved['manifest_hash'] != digest(manifest) or saved['legacy_harness_hash'] != manifest['harness_hash']:
        raise ValueError('legacy identity supplement binding mismatch')
    if saved['harness_hash'] != digest(saved['evaluation_harness']):
        raise ValueError('legacy evaluation harness hash mismatch')
    if 'evaluation_harness' in manifest:
        if (saved.get('version') != 'rtd-c25j-audited-identity-v1'
                or saved.get('rtd_source') != manifest.get('rtd_source')):
            raise ValueError('split identity supplement binding mismatch')
    if saved.get('version') == 'rtd-c25j-audited-identity-v1':
        if saved.get('model_identity') != dict(base_checkpoint_hash=manifest.get('base_checkpoint_hash'),
                tokenizer_hash=manifest.get('tokenizer_hash'), student=manifest['config'].get('student')):
            raise ValueError('identity update model/tokenizer binding mismatch')
        updates = saved.get('identity_updates', [])
        if not updates or updates[-1].get('new_harness_hash') != saved['harness_hash']:
            raise ValueError('identity update audit binding mismatch')
        if ('evaluation_harness' in manifest and updates[0].get('previous_identity') !=
                dict(harness_hash=manifest['harness_hash'], evaluation_harness=manifest['evaluation_harness'])):
            raise ValueError('identity update original manifest binding mismatch')
        for index, note in enumerate(updates):
            previous = note.get('previous_identity', {})
            mixed = (index == 0 and 'evaluation_harness' not in manifest
                     and previous.get('evaluation_harness') is None
                     and previous.get('harness_hash') == manifest['harness_hash'])
            if (note.get('manifest_hash') != digest(manifest)
                    or (not mixed and previous.get('harness_hash') != digest(previous.get('evaluation_harness')))
                    or (index and previous['harness_hash'] != updates[index-1]['new_harness_hash'])):
                raise ValueError('identity update history binding mismatch')
    return saved


def guard_harness(root, directory, manifest, *, current=None):
    saved = saved_identities(root, directory, manifest)
    current = current or evaluation_harness_identity(root, manifest['config'])
    if saved['evaluation_harness'] != current:
        if ('evaluation_harness' not in manifest
                and saved['evaluation_harness'].get('version') == 'bfcl-evaluation-harness-v1'):
            raise ValueError('evaluation harness uses a git-based legacy supplement; '
                             'run audit-legacy --run-dir to verify and migrate its content identity')
        raise ValueError('evaluation harness differs from training/audited legacy binding; '
                         'run update-identity --run-dir for an explicit scoring-continuity audit')
    return saved


def record_code_drift(root, directory, manifest, *, context, training=False,
                      acknowledge=False, current=None, identities=None):
    identities = identities or saved_identities(root, directory, manifest)
    old = identities['rtd_source']
    current = current or source_identity(root)
    journal = ComputeJournal(Path(directory)/'code_drift.jsonl')
    if old['hash'] != current['hash']:
        files = (sorted(p for p in set(old['files']) | set(current['files'])
                        if old['files'].get(p) != current['files'].get(p))
                 if old.get('files') is not None else None)
        record = dict(manifest_hash=digest(manifest), old_hash=old['hash'], new_hash=current['hash'],
                      old_identity_kind=old['kind'], new_identity_kind=current['kind'],
                      files=files, context=context, acknowledged=bool(training and acknowledge))
        if files is None:
            record['files_unavailable_reason'] = 'legacy manifest retained only the combined harness/source hash'
        if not any(all(e.get(k) == v for k, v in record.items()) for e in journal.events):
            journal.append('code_drift', **record)
        if training and not acknowledge:
            raise ValueError('RTD source changed; training resume requires --acknowledge-code-drift')
    return journal.events


def validate_resume(root, directory, saved, current, *, acknowledge=False, training=True):
    from .hardware import bound_hardware, guard_hardware
    ignored = {'initial_parameter_hash', 'harness_hash', 'evaluation_harness',
               'evaluation_harness_metadata', 'rtd_source', 'hardware', 'hardware_hash'}
    if ({k: v for k, v in saved.items() if k not in ignored}
            != {k: v for k, v in current.items() if k not in ignored}):
        raise ValueError('resume config/data/base metadata changed')
    guard_hardware(root, directory, saved, bound_hardware(root, current),
                   context='training_resume' if training else 'evaluation_resume')
    identities = guard_harness(root, directory, saved, current=current['evaluation_harness'])
    record_code_drift(root, directory, saved, context='training_resume' if training else 'evaluation_resume',
                      training=training,
                      acknowledge=acknowledge, current=current['rtd_source'], identities=identities)
    # Record every accepted resume, even when source drift has already been acknowledged.
    ComputeJournal(Path(directory)/'code_drift.jsonl').append('resume_score_consistency',
        context='training_resume' if training else 'evaluation_resume', manifest_hash=digest(saved),
        rtd_source_hash=current['rtd_source']['hash'],
        tolerance=asdict(ScoreTolerance.from_config(saved['config'])))


def verified_checkpoint(directory, manifest, round_number):
    checkpoint = Path(directory)/f'round-{round_number}'
    meta = json.loads((checkpoint/'checkpoint.json').read_text())
    if (meta['round'] != round_number or meta['manifest_hash'] != digest(manifest)
            or meta['config_hash'] != manifest['config_hash']
            or digest(manifest['config']) != manifest['config_hash']
            or meta['adapter_hash'] != tree_hash(checkpoint/'lora')
            or meta['round_state_hash'] != file_hash(checkpoint/'round_state.pt')):
        raise ValueError('evaluation checkpoint/manifest hash mismatch')
    return meta
