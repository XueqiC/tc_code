"""Stage heartbeats identify GPU ownership without GPU inspection."""
import json
from pathlib import Path
import sys
import threading

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from bfas.rtd.baselines import paper_progress


@pytest.mark.parametrize("gpu_held", [False, True])
@pytest.mark.parametrize("failure", [False, True])
def test_stage_logs_gpu_ownership_on_begin_heartbeat_and_exit(monkeypatch, capsys, gpu_held, failure):
    assert paper_progress.HEARTBEAT_SECONDS == 30
    monkeypatch.setattr(paper_progress, "HEARTBEAT_SECONDS", .001)
    heartbeat = threading.Event()
    emit = paper_progress.progress
    def progress(name, event, **fields):
        emit(name, event, **fields)
        if event == "running":
            heartbeat.set()
    monkeypatch.setattr(paper_progress, "progress", progress)
    def work():
        with paper_progress.stage("example", gpu_held=gpu_held, benchmark="bfcl"):
            assert heartbeat.wait(timeout=2), "stage did not emit a heartbeat"
            if failure:
                raise RuntimeError("stage failed")
    if failure:
        with pytest.raises(RuntimeError, match="stage failed"):
            work()
    else:
        work()
    events = [json.loads(line) for line in capsys.readouterr().err.splitlines()]
    assert events[0]["event"] == "begin"
    assert any(event["event"] == "running" for event in events)
    assert events[-1]["event"] == ("failed" if failure else "complete")
    assert all(event["gpu_held"] is gpu_held and event["benchmark"] == "bfcl" for event in events)
    assert all(event["elapsed_seconds"] >= 0 for event in events[1:])
