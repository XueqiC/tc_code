"""Content-bound RTD snapshots and durable compute accounting (no model loading)."""
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

import torch


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def manifest_identity(manifest):
    """Observed diagnostics are a journal projection, not a run identity knob."""
    return {k: v for k, v in manifest.items() if k != 'score_consistency_observed'}


def manifest_digest(manifest):
    return digest(manifest_identity(manifest))


def manifest_hash(manifest):
    """Streaming V1 binds immutable identity; its journal binds replay progress.

    Historical manifests without observations retain their original digest.
    """
    if manifest.get('arm') == 'V1' and manifest.get('replay_mode') == 'streaming':
        manifest = {k: v for k, v in manifest.items()
                    if k not in {'replay_consumed_steps', 'replay_schedule_hash'}}
    return manifest_digest(manifest)


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def tree_hash(path):
    path = Path(path)
    files = [path] if path.is_file() else sorted(p for p in path.rglob('*') if p.is_file()
                                               and '__pycache__' not in p.parts and '.git' not in p.parts)
    if not files:
        raise ValueError(f'no files to hash: {path}')
    return digest([(str(p.relative_to(path)) if path.is_dir() else p.name, file_hash(p)) for p in files])


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)
    fsync_directory(path.parent)


def fsync_directory(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class StateStore:
    """Write payload, fsync, then atomically publish a hash-bound pointer.

    The teacher journal may lead the pointer only across a recorded selection.
    Two snapshot generations survive; round LoRAs are retained separately.
    Never load arbitrary external pickle files: verify our snapshot hash first.
    """
    def __init__(self, directory, manifest):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.binding = manifest_hash(manifest)
        self.pointer = self.directory / 'latest.json'

    def save(self, state, ledger):
        old = json.loads(self.pointer.read_text()) if self.pointer.exists() else {'generation': -1}
        generation = old['generation'] + 1
        name = f'state-{generation:06d}.pt'
        path = self.directory / name
        temporary = path.with_suffix('.tmp')
        with temporary.open('wb') as stream:
            torch.save(state, stream)
            stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
        atomic_json(self.pointer, dict(generation=generation, file=name, sha256=file_hash(path),
            binding=self.binding, ledger_sequence=len(ledger.events),
            ledger_hash=ledger.events[-1]['event_hash'] if ledger.events else None))
        for obsolete in self.directory.glob('state-*.pt'):
            if int(obsolete.stem.split('-')[1]) < generation - 1:
                obsolete.unlink()

    def load(self, ledger, *, device='cpu'):
        p = json.loads(self.pointer.read_text())
        path = self.directory / p['file']
        if Path(p['file']).name != p['file'] or p['binding'] != self.binding or file_hash(path) != p['sha256']:
            raise ValueError('checkpoint/config/data/hardware binding mismatch')
        count = p['ledger_sequence']
        if count > len(ledger.events) or (count and ledger.events[count-1]['event_hash'] != p['ledger_hash']):
            raise ValueError('checkpoint ledger prefix mismatch')
        state = torch.load(path, map_location=device, weights_only=False)
        tail = ledger.events[count:]
        if state.get('batch_schema_version'):
            allowed_q = {state.get('transaction_query')}
        else:
            allowed_q = {state.get('selected')}
        if state['phase'] == 'initialize_fixed':
            allowed_q = set(state.get('fixed_ids', ()))
        for event in tail:
            if event['kind'] == 'authorize' and state['phase'] in {'round_start', 'initialize_fixed'}:
                continue
            if state.get('batch_schema_version') and event['kind'] == 'window_open':
                if (state['phase'] != 'selected' or event['window_id'] != state['window_id']
                        or event['budget'] != state['window_budget']
                        or event['max_packages'] != state['selection']['max_new_packages']):
                    raise ValueError('ledger opened outside the durably selected window')
                continue
            if state.get('batch_schema_version') and event['kind'] == 'window_close':
                if state['phase'] != 'committed' or event['window_id'] != state['window_id']:
                    raise ValueError('ledger closed outside the durably committed window')
                continue
            if state.get('batch_schema_version') and (event['kind'] not in {'reserve', 'reveal', 'release'}
                    or event['query_id'] is None):
                raise ValueError('ledger advanced outside a durably selected transaction')
            if state['phase'] not in {'selected', 'initialize_fixed'} or event['query_id'] not in allowed_q:
                raise ValueError('ledger advanced outside a durably selected transaction')
        return state


class ComputeJournal:
    """Hash-chained events retain failed/repeated work instead of rolling it back."""
    def __init__(self, path, *, cuda=False, deadline=None, deadline_seconds=None):
        self.path = Path(path)
        self.cuda, self.deadline = cuda, deadline
        self.deadline_seconds = deadline_seconds
        self.events = []
        self._score_manifest = None
        self._score_summary = None
        self._measure_stack = []
        self._step_peaks = None
        self._phase_peaks = []
        if self.path.exists():
            raw = self.path.read_bytes()
            if raw and not raw.endswith(b'\n'):
                cut = raw.rfind(b'\n') + 1
                with self.path.with_suffix('.torn').open('ab') as stream:
                    stream.write(raw[cut:] + b'\n')
                with self.path.open('r+b') as stream:
                    stream.truncate(cut); stream.flush(); os.fsync(stream.fileno())
                raw = raw[:cut]
            for line in raw.splitlines():
                e = json.loads(line)
                if e['sequence'] != len(self.events) or e['previous_hash'] != (self.events[-1]['hash'] if self.events else None):
                    raise ValueError('compute journal sequence mismatch')
                if e['hash'] != digest({k: v for k, v in e.items() if k != 'hash'}):
                    raise ValueError('compute journal hash mismatch')
                self.events.append(e)

    def bind_score_manifest(self, manifest, path):
        """Rebuild from the verified journal, repairing a stale/crash-lagged summary."""
        from .scoring import ScoreConsistencySummary
        self._score_manifest = (manifest, Path(path))
        self._score_summary = ScoreConsistencySummary()
        for event in self.events:
            if event['kind'] == 'score_consistency':
                self._score_summary.add(event)
        self._publish_score_summary()

    def _publish_score_summary(self):
        manifest, path = self._score_manifest
        manifest['score_consistency_observed'] = self._score_summary.record()
        atomic_json(path, manifest)

    def append(self, kind, **values):
        event = dict(sequence=len(self.events), kind=kind, timestamp=datetime.now(timezone.utc).isoformat(),
                     previous_hash=self.events[-1]['hash'] if self.events else None, **values)
        event['hash'] = digest(event)
        with self.path.open('a') as stream:
            stream.write(json.dumps(event, allow_nan=False) + '\n'); stream.flush(); os.fsync(stream.fileno())
        self.events.append(event)
        if kind == 'score_consistency' and self._score_manifest is not None:
            self._score_summary.add(event)
            self._publish_score_summary()  # Durable before enforce_score_diagnostic raises.
        return event['sequence']

    def gpu_memory(self):
        if not self.cuda:
            return dict(peak_allocated_bytes=0, peak_reserved_bytes=0)
        return dict(peak_allocated_bytes=torch.cuda.max_memory_allocated('cuda:0'),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved('cuda:0'))

    def _capture_peaks(self):
        peaks = self.gpu_memory()
        for accumulator in [self._step_peaks, *self._phase_peaks]:
            if accumulator is not None:
                for key, value in peaks.items():
                    accumulator[key] = max(accumulator.get(key, 0), value)
        return peaks

    @contextmanager
    def measure_phase(self, operation, **counts):
        """Local peaks, with ancestor/whole-step maxima preserved across resets."""
        from .memory import release_device_cache
        device = 'cuda:0' if self.cuda else 'cpu'
        self._capture_peaks()
        release_device_cache(device)
        if self.cuda:
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        peaks = dict(peak_allocated_bytes=0, peak_reserved_bytes=0)
        self._phase_peaks.append(peaks)
        status = 'failed'
        try:
            with self.measure(operation, _peaks=peaks, **counts):
                yield
            status = 'complete'
        finally:
            self._capture_peaks()
            self._phase_peaks.pop()
            self.append('subphase_memory', operation=operation, status=status,
                        logical_device=device, **counts, **peaks)
            release_device_cache(device)
            if self.cuda:
                torch.cuda.reset_peak_memory_stats(device)

    @contextmanager
    def measure_step(self, *, round, step, start_phase):
        """Aggregate all subphase peaks for a step attempt, including failures.

        Round setup belongs to step 1. On resume, start_phase identifies the
        measured suffix; failures retain their own peak instead of disappearing.
        """
        if self.cuda:
            torch.cuda.synchronize('cuda:0')
            torch.cuda.reset_peak_memory_stats('cuda:0')
        self._step_peaks = dict(peak_allocated_bytes=0, peak_reserved_bytes=0)
        status = 'failed'
        try:
            yield
            status = 'complete'
        finally:
            if self.cuda:
                torch.cuda.synchronize('cuda:0')
            self._capture_peaks()
            self.append('step_memory', round=round, step=step, start_phase=start_phase,
                status=status, logical_device='cuda:0' if self.cuda else 'cpu', **self._step_peaks)
            self._step_peaks = None

    @contextmanager
    def measure(self, operation, *, _peaks=None, **counts):
        if self.deadline is not None and time.monotonic() >= self.deadline:
            message = ('compute deadline exceeded' if self.deadline_seconds is None else
                       f'smoke exceeded {self.deadline_seconds} seconds')
            raise TimeoutError(f'{message}; resume state retained')
        if self.cuda:
            torch.cuda.synchronize()
        start = time.monotonic()
        sequence = self.append('compute_begin', operation=operation,
            parent_sequence=self._measure_stack[-1] if self._measure_stack else None, **counts)
        self._measure_stack.append(sequence)
        status = 'failed'
        try:
            yield
            status = 'complete'
        finally:
            if self.cuda:
                torch.cuda.synchronize()
            seconds = time.monotonic() - start
            current = self._capture_peaks()
            self.append('compute_end', begin_sequence=sequence, operation=operation, status=status,
                        wall_seconds=seconds, gpu_seconds=seconds if self.cuda else 0.,
                        **(current if _peaks is None else _peaks))
            self._measure_stack.pop()


@contextmanager
def exclusive_run(directory):
    import fcntl
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / '.lock').open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
