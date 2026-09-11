"""Execution-only, prefix-bound V0 schedule reader for V1 and unified arms."""
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace

from .conventions import schedule_identity, schedule_path, step_content
from .persistence import ComputeJournal, atomic_json, digest


def enabled(config):
    from .unified.arms import ARMS
    # Unified manifests finalize their replay evidence as complete; the frozen
    # launch configuration still selects streaming validation on every resume.
    settings = config.get('config', config)
    return (config.get('arm') in {'V1', *ARMS}
            and settings.get('replay_mode', config.get('replay_mode', 'complete')) == 'streaming')


def prepare_manifest(engine):
    """Freeze identity before creating StateStore; progress is journal-backed."""
    manifest = engine.manifest
    identity = schedule_identity(manifest, engine.config, engine.support)
    if manifest.get('replay_schedule_identity', identity) != identity:
        engine.journal.append('replay_schedule_diff', reason='manifest identity changed',
            diff=[dict(field='identity', expected=manifest.get('replay_schedule_identity'), actual=identity)])
        raise ValueError('V0/V1 exposure schedule identity differs')
    manifest['replay_schedule_identity'] = identity
    manifest.setdefault('replay_mode', engine.config.get('replay_mode', 'complete'))
    manifest.setdefault('replay_consumed_steps', [])
    manifest.setdefault('replay_schedule_hash', engine.config.get('replay_schedule_hash'))
    atomic_json(engine.directory/'manifest.json', manifest)


def validate_saved_replay(directory, manifest):
    """Coordinator resume also checks progress when no training worker runs."""
    directory = Path(directory)
    holder = SimpleNamespace(directory=directory, manifest=manifest, config=manifest['config'],
                             journal=ComputeJournal(directory/'compute.jsonl'))
    reader = StreamingSchedule(holder, smoke=manifest['smoke'])
    reader.snapshot()
    final_round = 1 if manifest['smoke'] else manifest['config']['rounds']
    if (directory/f'round-{final_round}').exists():
        reader.finish()


class StreamingSchedule:
    """One deadline per execution attempt, shared by all reads and finalization.

    The hash-chained compute journal is the durable consumption record. The
    manifest is its atomic projection and may lag it across a process crash.
    """
    def __init__(self, engine, *, smoke=False):
        self.engine, self.manifest, self.journal = engine, engine.manifest, engine.journal
        self.path = schedule_path(engine.config['replay_schedule'])
        self.identity = self.manifest['replay_schedule_identity']
        self.smoke = smoke
        self.expected = ([(1, 1)] if smoke else
                         [(r, s) for r in range(1, engine.config['rounds']+1) for s in range(1, 13)])
        self.poll = engine.config['replay_poll_seconds']
        self.timeout = engine.config['replay_timeout_seconds']
        self.started = time.monotonic()
        self.deadline = self.started + self.timeout
        self.consumed = [dict(round=e['round'], step=e['step'], content_hash=e['content_hash'])
                         for e in self.journal.events if e['kind'] == 'replay_step_consumed']
        saved = self.manifest['replay_consumed_steps']
        if saved != self.consumed[:len(saved)]:
            self.fail('manifest/journal consumption differs', 'replay_consumed_steps', saved, self.consumed)
        if [(r['round'], r['step']) for r in self.consumed] != self.expected[:len(self.consumed)]:
            self.fail('consumed steps are not a prefix', 'replay_consumed_steps', self.expected, self.consumed)
        if saved != self.consumed:
            self.persist()

    def fail(self, reason, field, expected, actual):
        self.journal.append('replay_schedule_diff', schedule=str(self.path), reason=reason,
                            diff=[dict(field=field, expected=expected, actual=actual)])
        raise ValueError(f'streaming replay divergence: {reason} ({field}); see replay_schedule_diff in compute.jsonl')

    def persist(self):
        self.manifest['replay_consumed_steps'] = list(self.consumed)
        atomic_json(self.engine.directory/'manifest.json', self.manifest)

    def wait(self, reason, key):
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            message = (f'streaming replay timed out after {self.timeout:g} seconds waiting for V0 '
                       f'schedule {self.path} ({reason}, step={key}); V0 may have stopped; '
                       f"resume {self.manifest['arm']} once V0 has progressed or completed; resume state retained")
            self.journal.append('replay_schedule_timeout', message=message, reason=reason,
                                elapsed_seconds=time.monotonic()-self.started)
            raise TimeoutError(message)
        seconds = min(self.poll, remaining)
        print(f'[rtd] streaming replay waiting {seconds:g}s: {reason}; step={key}; V0={self.path}', flush=True)
        sequence = self.journal.append('compute_begin', operation='replay_schedule_wait',
            parent_sequence=None, compute_type='idle', schedule=str(self.path), reason=reason,
            round=key[0] if key else None, step=key[1] if key else None, poll_seconds=seconds)
        started, status = time.monotonic(), 'failed'
        try:
            time.sleep(seconds)
            status = 'complete'
        finally:
            self.journal.append('compute_end', operation='replay_schedule_wait', begin_sequence=sequence,
                compute_type='idle', status=status, wall_seconds=time.monotonic()-started, gpu_seconds=0.,
                peak_allocated_bytes=0, peak_reserved_bytes=0)

    def snapshot(self, key=None, *, wait=True, check_initial_parameters=True):
        while True:
            try:
                # Parse and hash the SAME bytes/open inode: atomic replacement
                # between read and a separate file_hash must not mix versions.
                raw = self.path.read_bytes()
                data = json.loads(raw)
            except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError) as error:
                if not wait:
                    raise
                self.wait(f'schedule unavailable/torn read: {type(error).__name__}', key)
                continue
            if not isinstance(data, dict):
                self.fail('invalid schedule', 'document', 'object', data)
            for field, value in [('version', 'rtd-v11-exposure-v1'), ('arm', 'V0'),
                                 ('identity', self.identity), ('smoke', self.smoke)]:
                actual = data.get(field)
                if field == 'identity' and not check_initial_parameters and isinstance(actual, dict):
                    value = {k: v for k, v in value.items() if k != 'initial_parameter_hash'}
                    actual = {k: v for k, v in actual.items() if k != 'initial_parameter_hash'}
                if actual != value:
                    self.fail('schedule identity/header changed', field, value, actual)
            rows = data.get('steps')
            if not isinstance(rows, list) or any(not isinstance(r, dict) for r in rows):
                self.fail('invalid steps', 'steps', 'list of step objects', rows)
            keys = [(r.get('round'), r.get('step')) for r in rows]
            if keys != self.expected[:len(rows)] or len(rows) > len(self.expected):
                self.fail('missing/duplicate/out-of-order steps', 'steps', self.expected[:len(rows)], keys)
            if type(data.get('complete')) is not bool or (data['complete'] and len(rows) != len(self.expected)):
                self.fail('invalid completion marker', 'complete', len(rows) == len(self.expected), data.get('complete'))
            for row in rows:
                computed = digest(step_content(row))
                if row.get('content_hash') != computed:
                    self.fail('step content hash missing or changed', f"r{row['round']}/s{row['step']}.content_hash",
                              row.get('content_hash'), computed)
                decision = row['step'] in (1, 4, 7, 10)
                if row.get('decision') != decision or (not decision and
                        (row.get('selected') != [] or row.get('window_budget') is not None)):
                    self.fail('purchase timing changed', f"r{row['round']}/s{row['step']}.decision", decision, row)
            for index, consumed in enumerate(self.consumed):
                actual = rows[index]['content_hash'] if index < len(rows) else None
                if actual != consumed['content_hash']:
                    self.fail('consumed step changed or disappeared', f"r{consumed['round']}/s{consumed['step']}.content_hash",
                              consumed['content_hash'], actual)
            return data, hashlib.sha256(raw).hexdigest()

    def start(self):
        """Revalidate all consumed hashes before any resumed training work."""
        self.snapshot()
        state = self.engine.state
        rows = [r['exposure_schedule'] for r in state['steps']]
        if state.get('replay_exposure') is not None:
            rows.append(state['replay_exposure'])
        consumed = {(r['round'], r['step']): r['content_hash'] for r in self.consumed}
        for row in rows:
            key = row['round'], row['step']
            actual = digest(step_content(row))
            if consumed.get(key) != actual:
                self.fail('checkpoint consumption differs', str(key), consumed.get(key), actual)
        self.journal.append('replay_schedule_validated', schedule=str(self.path), identity=self.identity,
                            consumed_steps=len(self.consumed))

    def get(self, key):
        if key not in self.expected:
            self.fail('unexpected requested step', 'step', self.expected, key)
        index = self.expected.index(key)
        while True:
            data, _ = self.snapshot(key)
            if len(data['steps']) > index:
                row = data['steps'][index]
                if index == len(self.consumed):
                    consumed = dict(round=key[0], step=key[1], content_hash=row['content_hash'])
                    self.journal.append('replay_step_consumed', schedule=str(self.path), **consumed)
                    self.consumed.append(consumed)
                    self.persist()
                elif index > len(self.consumed):
                    self.fail('step consumption skipped', 'step', self.expected[len(self.consumed)], key)
                # Existing exposure/purchase/weight/repetition checks see exactly
                # their historical payload; source actions are never replayed.
                return step_content(row)
            self.wait('step not yet committed/exported', key)

    def finish(self):
        while True:
            data, final_hash = self.snapshot()
            if data['complete']:
                if len(self.consumed) != len(self.expected):
                    self.fail('incomplete replay consumption', 'steps', len(self.expected), len(self.consumed))
                previous = self.manifest.get('replay_schedule_hash')
                if previous is not None and previous != final_hash:
                    self.fail('final whole-file hash changed', 'replay_schedule_hash', previous, final_hash)
                self.manifest['replay_schedule_hash'] = final_hash
                if self.manifest['config'].get('method') == 'rtd_unified':
                    self.manifest['replay_mode'] = 'complete'
                self.persist()
                if previous is None:
                    self.journal.append('replay_schedule_complete', schedule=str(self.path), sha256=final_hash,
                                        consumed_steps=len(self.consumed))
                return
            self.wait('final complete=true schedule', None)
