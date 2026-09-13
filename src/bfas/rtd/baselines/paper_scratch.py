"""Job-local intermediate exports; durable receipts stay in the run directory."""
from contextlib import contextmanager
import os
from pathlib import Path
import re
import tempfile


@contextmanager
def export_directory(directory):
    """Prefer usable node scratch, retaining the shared export path as fallback.

    A random suffix isolates concurrent phases even within one Slurm job/array.
    TemporaryDirectory unwinds on errors and on the worker's TERM/SystemExit.
    """
    job = os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOBID") or f"pid-{os.getpid()}"
    job = re.sub(r"[^a-zA-Z0-9_-]", "_", job)[:80]
    temporary = None
    for candidate in (os.environ.get("SLURM_TMPDIR"), os.environ.get("TMPDIR"), "/tmp"):
        if not candidate:
            continue
        try:
            base = Path(candidate).resolve()
            if not base.is_dir() or not os.access(base, os.W_OK | os.X_OK):
                continue
            temporary = tempfile.TemporaryDirectory(prefix=f"baseline-export-{job}-", dir=base)
        except OSError:
            # A read-only mount, ACL, quota, or a race can defeat os.access().
            continue
        break
    if temporary is None:
        yield Path(directory)/"export"
    else:
        with temporary as name:
            yield Path(name)
