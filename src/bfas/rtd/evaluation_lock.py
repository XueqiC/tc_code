"""CPU-only BFCL resource leases, shared by RTD and the standalone campaign.

Lock files are permanent inodes. Never unlink them on release: another waiter
may already have opened the inode. Children inherit the leases so a coordinator
exit cannot expose a still-running campaign's resources to another run.
"""
from contextlib import contextmanager
import errno
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import time


def wait_settings(timeout=None, log_interval=None):
    timeout = float(os.environ.get('RTD_EVALUATION_LOCK_TIMEOUT_SECONDS', 6 * 3600)
                    if timeout is None else timeout)
    log_interval = float(os.environ.get('RTD_EVALUATION_LOCK_LOG_INTERVAL_SECONDS', 60)
                         if log_interval is None else log_interval)
    if not math.isfinite(timeout) or timeout < 0:
        raise ValueError('evaluation lock timeout must be finite and nonnegative')
    if not math.isfinite(log_interval) or log_interval <= 0:
        raise ValueError('evaluation lock log interval must be finite and positive')
    return timeout, log_interval


def tag_lock_path(root, tag, *, benchmark='bfcl', output_root=None):
    if not isinstance(tag, str) or not tag or tag in {'.', '..'} or '/' in tag or '\\' in tag:
        raise ValueError(f'invalid campaign tag: {tag!r}')
    if benchmark == 'alfworld':
        if output_root is None:
            raise ValueError('ALFWorld campaign lock requires output_root')
        # Arms share round tags, but only writers to one resolved campaign
        # directory share artifacts. Canonicalize aliases before taking a lease.
        campaign = (Path(output_root) / tag).resolve()
        key = hashlib.sha256(str(campaign).encode()).hexdigest()
        return Path(root) / 'results/alfworld_std/.locks' / key / '.lock'
    if benchmark != 'bfcl':
        raise ValueError('unknown evaluation benchmark')
    key = hashlib.sha256(tag.encode()).hexdigest()
    return Path(root) / 'results/bfcl_std/.locks' / key / '.lock'


def port_lock_path(root, port):
    # Also coordinate with pre-C25h campaigns that already hold this inode.
    return Path(root) / 'results/rtd_v1' / f'eval-port-{port}' / '.lock'


def _holder(stream):
    stream.seek(0)
    try:
        owner = json.load(stream)
        if not isinstance(owner, dict):
            raise ValueError('invalid lock owner')
        return owner
    except (ValueError, OSError):
        # Older exclusive_run locks had no owner metadata.
        stat = os.fstat(stream.fileno())
        try:
            for line in Path('/proc/locks').read_text().splitlines():
                fields = line.split()
                if len(fields) < 6 or fields[1] != 'FLOCK':
                    continue
                major, minor, inode = fields[5].split(':')
                if (int(major, 16), int(minor, 16), int(inode)) == (
                        os.major(stat.st_dev), os.minor(stat.st_dev), stat.st_ino):
                    return dict(pid=int(fields[4]), tag='legacy/unknown')
        except (OSError, ValueError):
            pass
        return dict(pid='unknown', tag='unknown')


@contextmanager
def evaluation_lock(path, *, tag, timeout=None, log_interval=None, on_wait=None):
    """Wait up to a monotonic deadline; journal each idle interval, even on error.

    LOCK_NB is only the polling primitive for a bounded blocking acquisition.
    on_wait is called before entering the protected body, never during GPU work.
    """
    timeout, log_interval = wait_settings(timeout, log_interval)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+') as stream:
        start = accounted = time.monotonic()
        next_log = start
        holder = None
        status = 'interrupted'
        try:
            while True:
                try:
                    fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    status = 'acquired'
                    break
                except BlockingIOError:
                    holder = _holder(stream)
                now = time.monotonic()
                if now >= next_log:
                    print(f"waiting for evaluation lock held by {holder.get('pid', 'unknown')}/"
                          f"{holder.get('tag', 'unknown')} ({path}; waited {now-start:.1f}s)", flush=True)
                    if on_wait and now > accounted:
                        on_wait(wall_seconds=now-accounted, lock=str(path), holder=holder, status='waiting')
                    accounted = now
                    next_log = now + log_interval
                if now - start >= timeout:
                    status = 'timeout'
                    raise TimeoutError(f'evaluation lock timeout after {timeout:g}s: {path}; holder={holder}')
                time.sleep(min(0.1, next_log-now, timeout-(now-start)))
        finally:
            if holder is not None and on_wait:
                on_wait(wall_seconds=time.monotonic()-accounted, lock=str(path), holder=holder, status=status)
        # Only the owner writes metadata. A waiter must never truncate it.
        stream.seek(0)
        stream.truncate()
        json.dump(dict(pid=os.getpid(), tag=tag, hostname=socket.gethostname()), stream)
        stream.flush()
        # Close this descriptor, but don't explicitly LOCK_UN: an inherited
        # descriptor in a surviving child must keep the lease alive.
        yield stream.fileno()


def preferred_port(gpu_uuid=None):
    job = os.environ.get('SLURM_JOB_ID')
    if job:
        return 20000 + int(job) % 30000
    gpu_uuid = gpu_uuid or os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if gpu_uuid.startswith(('GPU-', 'MIG-')) or gpu_uuid and not gpu_uuid.isdigit():
        return 20000 + int(hashlib.sha256(gpu_uuid.encode()).hexdigest()[:8], 16) % 30000
    return 0  # physical ordinals aren't portable identities; ask the kernel


@contextmanager
def reserve_port(root, *, tag, port=None, gpu_uuid=None, timeout=None, log_interval=None, on_wait=None):
    """Atomically lease a free port among campaigns, falling back on collisions.

    The socket probe closes before vLLM binds; the inherited flock spans that
    handoff. An unrelated service is refused, never adopted or killed.
    """
    automatic = port in (None, 0)
    candidate = preferred_port(gpu_uuid) if automatic else port
    if not 0 <= candidate <= 65535:
        raise ValueError('campaign port must be in 1..65535, or 0/auto')
    timeout, log_interval = wait_settings(timeout, log_interval)
    deadline = time.monotonic() + timeout
    while True:
        if not candidate:
            with socket.socket() as probe:
                probe.bind(('0.0.0.0', 0))
                candidate = probe.getsockname()[1]
        # Automatic choices should skip collisions without a six-hour wait.
        # A zero-time attempt is still accounted and reports its holder.
        manager = evaluation_lock(port_lock_path(root, candidate), tag=tag,
            timeout=0 if automatic else max(0, deadline-time.monotonic()),
            log_interval=log_interval, on_wait=on_wait)
        try:
            fd = manager.__enter__()
        except TimeoutError:
            if not automatic or time.monotonic() >= deadline:
                raise
            candidate = 0
            continue
        try:
            try:
                with socket.socket() as probe:
                    probe.bind(('0.0.0.0', candidate))
            except OSError as error:
                if error.errno != errno.EADDRINUSE or not automatic or time.monotonic() >= deadline:
                    raise
                candidate = 0
                continue
            yield candidate, fd
            return
        finally:
            manager.__exit__(None, None, None)


def inherited_lock(fd, path):
    """Validate that a child received this resource's already-locked inode."""
    fd = int(fd)
    actual, expected = os.fstat(fd), Path(path).stat()
    if (actual.st_dev, actual.st_ino) != (expected.st_dev, expected.st_ino):
        raise ValueError('inherited evaluation lock belongs to another resource')
    # A separate open must contend, whereas flock on the inherited descriptor
    # must succeed (same open file description as its owner).
    with Path(path).open('a+') as probe:
        try:
            fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return fd
    raise ValueError('inherited evaluation descriptor is not locked')
