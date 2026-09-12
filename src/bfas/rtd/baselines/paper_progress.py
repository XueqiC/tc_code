"""Flushed worker progress, including heartbeats during long blocking stages."""
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import sys
import threading
import time


def progress(stage, event, **fields):
    print(json.dumps(dict(timestamp=datetime.now(timezone.utc).isoformat(),
        stage=stage, event=event, **fields)), file=sys.stderr, flush=True)


@contextmanager
def stage(name, **fields):
    start = time.monotonic()
    stop = threading.Event()
    progress(name, "begin", **fields)

    def heartbeat():
        while not stop.wait(30):
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
