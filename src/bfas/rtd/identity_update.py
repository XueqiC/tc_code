"""Explicit, CPU-only scoring-continuity audit for immutable run manifests."""
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path

from . import identity as ids
from .persistence import manifest_digest, atomic_json, digest, file_hash, tree_hash
from .scoring_scope import PYTHON_SCOPES, SHELL, scoring_hash


EVIDENCE_PATH = 'configs/rtd/identity_evidence/c25r.json'
EVIDENCE_SHA256 = '959e79cf5321467d09d5c24517b3dea27e956dcbfb162f6a49a150c4ed7a2598'


class IdentityUpdateRefused(ValueError):
    def __init__(self, evidence):
        self.evidence = evidence
        super().__init__('scoring identity update refused: ' + json.dumps(evidence, sort_keys=True))


def _historical_sources(root):
    path = root/EVIDENCE_PATH
    if not path.exists():
        return {}
    if file_hash(path) != EVIDENCE_SHA256:
        raise ValueError('reviewed scoring evidence digest mismatch')
    result = {}
    for row in json.loads(path.read_text())['files']:
        if hashlib.sha256(row['content'].encode()).hexdigest() != row['sha256']:
            raise ValueError('historical source digest mismatch')
        result[row['path']] = row
    return result


def _old_files(harness):
    # C25j/v3 projected the existing mixed files but hashed the bridge RAW.
    # Interpret saved hashes by their historical format, never today's scopes.
    scoped = {*PYTHON_SCOPES, SHELL}
    if harness.get('version') == 'bfcl-evaluation-harness-scoring-v3':
        scoped.remove('tools/behavior_atom/checker_bridge.py')
    elif harness.get('version') != ids.CONTENT_VERSION:
        scoped = set()
    result = {ids.LEADERBOARD+'/'+n: dict(sha256=row['sha256'], kind='file')
              for field in ('checkout_files', 'data_manifest')
              for n, row in harness.get(field, {}).items()}
    result.update({n: dict(sha256=sha, kind='scoring_projection' if n in scoped else 'file')
        for n, sha in harness.get('tools', {}).items()})
    return result


def _model_evidence(manifest):
    """Read bytes only; never deserialize weights, query hardware or use CUDA."""
    model = Path(manifest['model_path'])
    before = {str(p): ids._file_stamp(p) for p in sorted(model.rglob('*')) if p.is_file()}
    base = tree_hash(model)
    tokenizer = digest([(p.name, file_hash(p)) for p in sorted(model.glob('*'))
                        if p.is_file() and any(k in p.name for k in ('token', 'vocab', 'merges', 'chat_template'))])
    after = {str(p): ids._file_stamp(p) for p in sorted(model.rglob('*')) if p.is_file()}
    if before != after:
        raise ValueError('model/tokenizer files changed during identity audit')
    if base != manifest['base_checkpoint_hash'] or tokenizer != manifest['tokenizer_hash']:
        raise ValueError('model/tokenizer content differs from original manifest')
    return dict(base_checkpoint_hash=base, tokenizer_hash=tokenizer,
                student=manifest['config'].get('student')), before


@contextmanager
def _supplement_directory_lock(parent):
    # Serialize writers using the directory inode, without creating run locks
    # or a non-supplement lock file. Recheck the original supplement under it.
    parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)


def update_identity(root, directory):
    root, directory = Path(root).resolve(), Path(directory).resolve()
    manifest_path = directory/'manifest.json'
    stamp, original = ids._file_stamp(manifest_path), manifest_path.read_bytes()
    manifest = json.loads(original)
    if (not isinstance(manifest.get('config'), dict)
            or digest(manifest['config']) != manifest.get('config_hash')
            or not isinstance(manifest.get('harness_hash'), str)):
        raise ValueError('identity update requires a valid saved config and harness hash')
    if ('evaluation_harness' in manifest) != ('rtd_source' in manifest):
        raise ValueError('incomplete split identity in manifest')
    manifest_hash = manifest_digest(manifest)
    path = root/'configs/rtd/legacy_identities'/f'{manifest_hash}.json'
    previous_bytes = path.read_bytes() if path.exists() else None
    if previous_bytes is not None or 'evaluation_harness' in manifest:
        saved = ids.saved_identities(root, directory, manifest)
    else:
        # No invented per-file history for the original mixed digest (e.g. HPG).
        saved = dict(harness_hash=manifest['harness_hash'], evaluation_harness=None,
                     rtd_source=dict(kind='legacy-mixed-harness', hash=manifest['harness_hash'], files=None))
    old = saved.get('evaluation_harness') or {}
    old_files = _old_files(old)
    # Include removed old members in the race check and exact old-set diff.
    old_stamps = {n: ids._file_stamp(root/n) if (root/n).is_file() else None for n in old_files}
    changed = []
    for name, row in sorted(old_files.items()):
        p = root/name
        raw = file_hash(p) if p.is_file() else None
        try:
            now = (scoring_hash(name, p.read_text()) if raw and row['kind'] == 'scoring_projection' else raw)
        except (SyntaxError, ValueError):
            now = None  # malformed selected source has no recoverable new projection
        if now != row['sha256']:
            changed.append(dict(path=name, old_sha256=row['sha256'], new_sha256=now,
                                hash_kind=row['kind'], new_file_sha256=raw))
    try:
        before = ids._harness_file_stamps(root)
        current = ids.evaluation_harness_identity(root, manifest['config'])
    except (OSError, SyntaxError, ValueError) as exc:
        raise IdentityUpdateRefused(dict(supplement_path=str(path), updated=False,
            audit=dict(manifest_hash=manifest_hash, changed_old_identity_files=changed,
                       old_identity_files_checked=len(old_files)),
            refusals=[dict(reason='cannot construct current scoring inventory: ' + str(exc))])) from exc
    new_files = _old_files(current)
    for row in changed:
        row['in_new_scoring_set'] = row['path'] in new_files
    # On copied runs an older manifest mtime retained in its audit remains the
    # stricter bound. Copy time cannot expand the authorized historical window.
    cutoff = min(stamp[0], saved.get('audit', {}).get('manifest_mtime_ns', stamp[0]),
                 *(note['manifest_mtime_ns'] for note in saved.get('identity_updates', [])))
    history = _historical_sources(root)
    evidence, refusals = {}, []
    if old and old.get('config') != current['config']:
        refusals.append(dict(reason='saved evaluation config differs'))
    if old and old.get('leaderboard') != current['leaderboard']:
        refusals.append(dict(reason='saved leaderboard path differs'))
    for field in ('checkout_files', 'data_manifest'):
        if field in old:
            old_names = {n for n in old[field] if n.startswith('bfcl_eval/')}
            if old_names != set(current[field]):
                refusals.append(dict(reason='scoring package inventory changed', section=field,
                                     added=sorted(set(current[field])-old_names),
                                     removed=sorted(old_names-set(current[field]))))
    for name, row in sorted(new_files.items()):
        ns = before[name][0]
        prior = old_files.get(name)
        scoped = row['kind'] == 'scoring_projection'
        item = dict(mtime_ns=ns, scoring_sha256=row['sha256'], file_sha256=file_hash(root/name))
        evidence[name] = item
        if prior and prior['kind'] == row['kind'] and prior['sha256'] != row['sha256']:
            refusals.append(dict(path=name, reason='saved scoring content changed',
                                 old_sha256=prior['sha256'], new_sha256=row['sha256']))
            continue
        if scoped and prior and prior == row:
            item['basis'] = 'manifest-bound-scoring-projection-unchanged'
            continue
        needs_old_projection = scoped and prior and prior['kind'] == 'file' and prior['sha256'] != item['file_sha256']
        if ns < cutoff and not needs_old_projection:
            item['basis'] = 'file-mtime-strictly-before-manifest'
            continue
        # A mixed file may be newer solely because its excluded plumbing was
        # edited. Require actual historical bytes, never a hash allowlist that
        # bypasses comparison of the selected scoring source.
        archived = history.get(name) if scoped else None
        if archived and archived['observed_at_ns'] < cutoff:
            historical_hash = scoring_hash(name, archived['content'])
            if (historical_hash == row['sha256'] and
                    (not prior or prior['kind'] != 'file' or archived['sha256'] == prior['sha256'])):
                item.update(basis='reviewed-pre-manifest-source-projection',
                            conclusion='scoring projection unchanged',
                            historical_file_sha256=archived['sha256'],
                            historical_scoring_sha256=historical_hash,
                            observed_at_ns=archived['observed_at_ns'], evidence_bundle=EVIDENCE_PATH,
                            evidence_bundle_sha256=EVIDENCE_SHA256)
                continue
        refusals.append(dict(path=name, reason='scoring continuity lacks pre-manifest evidence',
                             mtime_ns=ns, manifest_mtime_ns=cutoff))
    audit = dict(task='C25r', method='scoring-scope-and-pre-manifest-evidence-v1',
                 recorded_at=datetime.now(timezone.utc).isoformat(), manifest_hash=manifest_hash,
                 manifest_mtime_ns=cutoff, current_manifest_mtime_ns=stamp[0],
                 previous_identity=dict(harness_hash=saved['harness_hash'], evaluation_harness=saved.get('evaluation_harness')),
                 new_harness_hash=digest(current), changed_old_identity_files=changed,
                 old_identity_files_checked=len(old_files), scoring_files_checked=len(new_files),
                 old_inventory_unavailable=(None if old.get('checkout_files') is not None else
                     'Original identity did not retain a recoverable per-file checkout inventory; no exact checkout diff claimed.'),
                 scoring_file_evidence=evidence,
                 limitation='Mtimes and archived source timestamps corroborate continuity; they are not cryptographic proof of historical contents.',
                 training_resume_requires='--acknowledge-code-drift')
    def refuse():
        raise IdentityUpdateRefused(dict(supplement_path=str(path), updated=False, audit=audit, refusals=refusals))
    if refusals:
        refuse()
    try:
        model, model_stamps = _model_evidence(manifest)
        checkpoints = {p.name: ids.verified_checkpoint(directory, manifest, int(p.name.removeprefix('round-')))
                       for p in sorted(directory.glob('round-[123]')) if p.is_dir()}
    except (ValueError, KeyError, OSError) as exc:
        refusals.append(dict(reason=str(exc)))
        refuse()
    audit.update(model_identity=model, verified_checkpoints=checkpoints,
                 current_source=ids.source_identity(root))
    def unchanged():
        if (stamp != ids._file_stamp(manifest_path) or original != manifest_path.read_bytes()
                or before != ids._harness_file_stamps(root)
                or (history and file_hash(root/EVIDENCE_PATH) != EVIDENCE_SHA256)
                or old_stamps != {n: ids._file_stamp(root/n) if (root/n).is_file() else None for n in old_files}
                or previous_bytes != (path.read_bytes() if path.exists() else None)
                or model_stamps != {str(p): ids._file_stamp(p) for p in sorted(Path(manifest['model_path']).rglob('*')) if p.is_file()}
                or checkpoints != {p.name: ids.verified_checkpoint(directory, manifest, int(p.name.removeprefix('round-')))
                                   for p in sorted(directory.glob('round-[123]')) if p.is_dir()}):
            raise ValueError('manifest, supplement, checkpoint or scoring files changed during identity audit')
    unchanged()
    if old == current:
        return dict(supplement_path=str(path), updated=False, created=False, audit=audit)
    supplement = dict(saved, version='rtd-c25j-audited-identity-v1',
                      manifest_hash=manifest_hash, legacy_harness_hash=manifest['harness_hash'],
                      evaluation_harness=current, harness_hash=digest(current), model_identity=model,
                      evaluation_harness_metadata=ids.evaluation_harness_metadata(root),
                      identity_updates=[*saved.get('identity_updates', []), audit])
    # Keep the original legacy audit, migrations and source identity intact.
    with _supplement_directory_lock(path.parent):
        unchanged()
        atomic_json(path, supplement)
    return dict(supplement_path=str(path), updated=True, created=previous_bytes is None,
                harness_hash=digest(current), audit=audit)
