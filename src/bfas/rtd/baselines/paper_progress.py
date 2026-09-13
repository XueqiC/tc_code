"""Flushed worker progress, including heartbeats during long blocking stages."""
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import sys
import threading
import time

HEARTBEAT_SECONDS = 30


def progress(stage, event, **fields):
    print(json.dumps(dict(timestamp=datetime.now(timezone.utc).isoformat(),
        stage=stage, event=event, **fields)), file=sys.stderr, flush=True)


@contextmanager
def stage(name, *, gpu_held=True, **fields):
    # Logical model/server ownership, including startup; not kernel utilisation
    # or the enclosing scheduler allocation. Training keeps its model resident.
    fields = dict(gpu_held=gpu_held, **fields)
    start = time.monotonic()
    stop = threading.Event()
    progress(name, "begin", **fields)

    def heartbeat():
        while not stop.wait(HEARTBEAT_SECONDS):
            progress(name, "running", elapsed_seconds=round(time.monotonic()-start, 3), **fields)

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    status = "failed"
    try:
        yield
        status = "complete"
    finally:
        stop.set()
        thread.join()
        progress(name, status, elapsed_seconds=round(time.monotonic()-start, 3), **fields)
