"""Cleanup for subprocess trees launched with start_new_session=True (setsid)."""
import os
import signal
import subprocess


def stop_process_group(process, timeout=30):
    # The leader may already have exited while an EngineCore is still alive.
    # Its pid remains the process-group id; do not gate killpg on poll().
    def send(sig):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            pass

    try:
        send(signal.SIGTERM)
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            pass
    finally:
        send(signal.SIGKILL)  # Also stop descendants that outlived the leader.
        process.wait()
