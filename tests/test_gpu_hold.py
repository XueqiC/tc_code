"""CPU-only GPU holder checks: fake nvidia-smi, CUDA allocator, and children."""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools import gpu_hold as hold


UID = os.getuid()
GPU0 = hold.GPU("0", "GPU-aaaa")
GPU4 = hold.GPU("4", "GPU-bbbb")


def snapshot(gpu=GPU0, used=100, apps=()):
    return hold.Snapshot(int(gpu.label), gpu.uuid, used, 10_000,
                         tuple(hold.ComputeApp(pid, mb) for pid, mb in apps))


class FakeChild:
    def __init__(self, pid):
        self.pid = pid
        self.returncode = None
        self.signals = []
        self.stdin = SimpleNamespace(closed=False, close=self.close_pipe)

    def close_pipe(self):
        self.stdin.closed = True

    def poll(self):
        return self.returncode

    def terminate(self):
        self.signals.append(signal.SIGTERM)
        self.returncode = -signal.SIGTERM

    def kill(self):
        self.signals.append(signal.SIGKILL)
        self.returncode = -signal.SIGKILL

    def wait(self, timeout):
        if self.returncode is None:
            raise subprocess.TimeoutExpired("fake child", timeout)
        return self.returncode


@pytest.fixture
def rig(tmp_path, monkeypatch):
    clock = [1_000.0]
    owners = {}
    snapshots = {gpu.uuid: snapshot(gpu) for gpu in (GPU0, GPU4)}
    children = []
    launches = []
    queries = []

    def query(gpu, timeout=10):
        queries.append(gpu)
        value = snapshots[gpu]
        if isinstance(value, Exception):
            raise value
        return value

    def popen(args, **kwargs):
        child = FakeChild(30_000 + len(children))
        owners[child.pid] = UID
        children.append(child)
        launches.append((args, kwargs))
        return child

    monkeypatch.setattr(hold, "query_gpu", query)
    monkeypatch.setattr(hold, "process_uid", lambda pid: owners.get(pid))
    monkeypatch.setattr(hold.subprocess, "Popen", popen)
    monkeypatch.setattr(hold.time, "time", lambda: clock[0])
    holder = hold.Holder([GPU0, GPU4], tmp_path, until=1_100)
    return SimpleNamespace(holder=holder, snapshots=snapshots, owners=owners,
                           children=children, launches=launches, queries=queries,
                           clock=clock, state_dir=tmp_path)


@pytest.mark.parametrize("contents,expected", [
    ("Name:\ttest\nUid:\t123\t456\t789\t789\n", 123),
    ("Name:\ttest\n", None), ("Uid:\tnonsense\n", None), ("Uid:\n", None),
])
def test_proc_status_uid(tmp_path, contents, expected):
    (tmp_path / "42").mkdir()
    (tmp_path / "42/status").write_text(contents)
    assert hold.process_uid(42, tmp_path) == expected
    assert hold.process_uid(43, tmp_path) is None
    assert hold.process_uid(-1, tmp_path) is None


def test_unreadable_uid_is_unknown(monkeypatch):
    def denied(*args, **kwargs):
        raise PermissionError("hidden /proc")
    monkeypatch.setattr(Path, "read_text", denied)
    assert hold.process_uid(42) is None


@pytest.mark.parametrize("apps,used,held_pid,released,expected", [
    ((), 100, None, False, "free"),
    (((10, 100),), 100, None, False, "our job running"),
    (((20, 100),), 100, None, False, "other users present"),
    (((99, 100),), 100, None, False, "other users present"),
    ((), 1500, None, False, "other users present"),
    ((), 1499, None, False, "free"),
    (((10, 9000),), 9000, 10, False, "held"),
    ((), 100, 10, False, "held"),  # Child may still be importing torch.
    (((10, 9000), (11, 100)), 9100, 10, False, "our job running"),
    (((10, 100), (20, 100)), 200, None, False, "our job running"),
    (((10, 9000),), 9000, 10, True, "released"),
])
def test_classification(monkeypatch, apps, used, held_pid, released, expected):
    owners = {10: UID, 11: UID, 20: UID + 1}
    monkeypatch.setattr(hold, "process_uid", owners.get)
    assert hold.classify(snapshot(used=used, apps=apps), UID, 1500,
                         held_pid, released) == expected


def test_query_parses_compute_memory_and_physical_gpu(monkeypatch):
    calls = []
    def run(args, **kwargs):
        calls.append((args, kwargs))
        output = "10, 512\n20, [N/A]\n" if "--query-compute-apps=pid,used_gpu_memory" in args else \
                 "4, GPU-bbbb, 600, 24576\n"
        return SimpleNamespace(stdout=output)
    monkeypatch.setattr(hold.subprocess, "run", run)
    result = hold.query_gpu("GPU-bbbb")
    assert result == hold.Snapshot(4, "GPU-bbbb", 600, 24576,
                                   (hold.ComputeApp(10, 512), hold.ComputeApp(20, None)))
    assert len(calls) == 2
    assert all("--id=GPU-bbbb" in args and kwargs["check"] and kwargs["timeout"] <= 10
               for args, kwargs in calls)


@pytest.mark.parametrize("output", ["not a CSV\n", "4, GPU-bbbb, [N/A], 24576\n",
                                        "4, GPU-bbbb, 30000, 24576\n", ""])
def test_query_bad_memory_fails_closed(monkeypatch, output):
    def run(args, **kwargs):
        return SimpleNamespace(stdout="" if "--query-compute-apps=pid,used_gpu_memory" in args
                               else output)
    monkeypatch.setattr(hold.subprocess, "run", run)
    with pytest.raises(ValueError):
        hold.query_gpu("4")


def test_claim_release_resume_and_grab_when_free(rig):
    holder = rig.holder
    rig.owners[200] = UID + 1
    rig.snapshots[GPU4.uuid] = snapshot(GPU4, used=8000, apps=((200, 7900),))
    holder.check_once()
    assert len(rig.children) == 1
    first = rig.children[0]
    assert holder.pid_path(GPU0).read_text() == f"{first.pid}\n"
    assert (rig.state_dir / "hold_0.log").exists()
    args, kwargs = rig.launches[0]
    assert args[:4] == [sys.executable, "-u", str(hold.SCRIPT), "_hold"]
    assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == GPU0.uuid
    assert kwargs["stdin"] == subprocess.PIPE and kwargs["close_fds"]
    assert "0.92" in args

    # Idempotent even before nvidia-smi sees the starting child's CUDA context.
    holder.check_once()
    assert len(rig.children) == 1
    rig.snapshots[GPU0.uuid] = snapshot(used=9200, apps=((first.pid, 9100),))
    holder.release_path(GPU0).touch()
    holder.check_once()
    assert first.signals == [signal.SIGTERM]
    assert first.stdin.closed
    assert not holder.pid_path(GPU0).exists()

    rig.snapshots[GPU0.uuid] = snapshot()
    holder.check_once()
    assert len(rig.children) == 1  # Release persists even on an empty GPU.
    holder.release_path(GPU0).unlink()
    rig.owners[100] = UID
    rig.snapshots[GPU0.uuid] = snapshot(used=200, apps=((100, 100),))
    holder.check_once()
    assert len(rig.children) == 1  # Our real job can use less than the threshold.

    rig.snapshots[GPU0.uuid] = snapshot()
    rig.snapshots[GPU4.uuid] = snapshot(GPU4)
    holder.check_once()
    assert len(rig.children) == 3
    assert set(holder.children) == {GPU0.uuid, GPU4.uuid}
    assert all(not child.signals for child in rig.children[1:])


def test_foreign_small_process_and_memory_only_usage_block_claims(rig):
    rig.owners[200] = UID + 1
    rig.snapshots[GPU0.uuid] = snapshot(used=100, apps=((200, 1),))
    rig.snapshots[GPU4.uuid] = snapshot(GPU4, used=1500)
    rig.holder.check_once()
    assert not rig.children


def test_release_precedes_failed_queries_and_does_not_restart(rig):
    rig.holder.check_once()
    rig.holder.release_path(GPU4).touch()
    for gpu in (GPU0, GPU4):
        rig.snapshots[gpu.uuid] = subprocess.TimeoutExpired("nvidia-smi", 10)
    rig.holder.check_once()
    assert rig.children[1].signals == [signal.SIGTERM]
    assert not rig.children[0].signals
    assert len(rig.children) == 2


def test_release_created_during_query_stops_child_in_same_check(rig, monkeypatch):
    rig.holder.check_once()
    def query(gpu, **kwargs):
        rig.holder.release_path(GPU0).touch()
        return rig.snapshots[gpu]
    monkeypatch.setattr(hold, "query_gpu", query)
    rig.holder.check_once()
    assert rig.children[0].signals == [signal.SIGTERM]
    assert len(rig.children) == 2


def test_reap_crashed_child_and_replace(rig):
    rig.holder.check_once()
    rig.children[0].returncode = 1
    rig.holder.check_once()
    assert len(rig.children) == 3
    assert not rig.children[0].signals
    assert rig.holder.pid_path(GPU0).read_text() == f"{rig.children[2].pid}\n"


def test_our_real_job_causes_only_our_placeholder_to_stop(rig):
    rig.holder.check_once()
    rig.owners[100] = UID
    rig.snapshots[GPU0.uuid] = snapshot(used=9500, apps=((rig.children[0].pid, 9100), (100, 300)))
    rig.holder.check_once()
    assert rig.children[0].signals == [signal.SIGTERM]
    assert rig.children[1].signals == []


@pytest.mark.parametrize("owner", [None, UID + 1])
def test_never_signal_child_with_unverified_ownership(rig, owner):
    rig.holder.check_once()
    child = rig.children[0]
    rig.owners[child.pid] = owner
    rig.holder.stop_placeholder(GPU0)
    assert not child.signals


def test_stale_pid_file_never_signals_an_unrelated_job(rig):
    rig.owners[100] = UID
    rig.holder.pid_path(GPU0).write_text("100\n")
    rig.holder.release_path(GPU0).touch()
    rig.holder.check_once()
    assert GPU0.uuid not in rig.holder.children
    assert rig.holder.pid_path(GPU0).read_text() == "100\n"
    assert len(rig.children) == 1  # Only GPU4 was claimed.


def test_run_clips_sleep_to_until_and_stops_children(rig, monkeypatch):
    rig.holder.until = 1_035
    waits = []
    def wait(seconds):
        waits.append(seconds)
        rig.clock[0] += seconds
    monkeypatch.setattr(rig.holder.stop_event, "wait", wait)
    rig.holder.run()
    assert waits == [30, 5]
    assert rig.clock[0] == 1_035
    assert len(rig.children) == 2
    assert all(child.signals == [signal.SIGTERM] for child in rig.children)
    assert not list(rig.state_dir.glob("hold_*.pid"))


def test_deadline_during_query_prevents_claim(rig, monkeypatch):
    def query(*args, **kwargs):
        rig.clock[0] = rig.holder.until
        return snapshot()
    monkeypatch.setattr(hold, "query_gpu", query)
    rig.holder.check_once()
    assert not rig.children


def test_past_deadline_exits_without_query_or_state_files(tmp_path, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("expired daemon must not query GPUs or start children")
    monkeypatch.setattr(hold, "query_gpu", unexpected)
    monkeypatch.setattr(hold.subprocess, "Popen", unexpected)
    assert hold.main(["--gpus", "0,4", "--until", "2000-01-01T00:00",
                      "--state-dir", str(tmp_path / "state")]) == 0
    assert not (tmp_path / "state").exists()


def test_local_deadline_and_validation():
    assert hold.local_deadline("2026-09-26T23:59") == hold.datetime(2026, 9, 26, 23, 59).timestamp()
    with pytest.raises(ValueError, match="local time"):
        hold.local_deadline("2026-09-26T23:59+00:00")
    assert hold.parse_gpus(["0,4", "GPU-aaaa", "04"]) == ["0", "4", "GPU-aaaa"]
    with pytest.raises(ValueError):
        hold.parse_gpus(["../../file"])


def test_resolution_deduplicates_gpu_aliases(monkeypatch):
    monkeypatch.setattr(hold, "query_gpu", lambda *args, **kwargs: snapshot())
    assert hold.resolve_gpus(["0", GPU0.uuid]) == [GPU0]


def test_resolution_stops_at_deadline(rig, monkeypatch):
    calls = []
    def query(gpu, timeout):
        calls.append((gpu, timeout))
        rig.clock[0] += timeout
        raise subprocess.TimeoutExpired("nvidia-smi", timeout)
    monkeypatch.setattr(hold, "query_gpu", query)
    assert hold.resolve_gpus(["0", "4"], deadline=1001) == []
    assert calls == [("0", 1)]


def test_lock_excludes_another_process_and_can_be_reacquired(tmp_path):
    probe = """
from pathlib import Path
import sys
from tools.gpu_hold import AlreadyRunning, DaemonLock
try:
    with DaemonLock(Path(sys.argv[1])):
        pass
except AlreadyRunning:
    sys.exit(17)
"""
    with hold.DaemonLock(tmp_path):
        assert (tmp_path / "daemon.lock").read_text() == f"{os.getpid()}\n"
        result = subprocess.run([sys.executable, "-c", probe, str(tmp_path)], cwd=ROOT,
                                capture_output=True, text=True, timeout=10)
        assert result.returncode == 17, result.stderr
    assert (tmp_path / "daemon.lock").exists()
    result = subprocess.run([sys.executable, "-c", probe, str(tmp_path)], cwd=ROOT,
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr


def test_duplicate_daemon_returns_without_gpu_queries(tmp_path, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("second daemon must not query GPUs")
    monkeypatch.setattr(hold, "query_gpu", unexpected)
    with hold.DaemonLock(tmp_path):
        assert hold.main(["--gpus", "0", "--until", "2099-01-01T00:00",
                          "--state-dir", str(tmp_path)]) == 0


def test_status_prints_all_states_without_starting_children(rig, monkeypatch, capsys):
    gpus = [hold.GPU(str(i), f"GPU-{i:04x}") for i in range(5)]
    (rig.state_dir / "config.json").write_text(json.dumps({
        "gpus": [hold.asdict(gpu) for gpu in gpus], "free_threshold_mb": 1500,
    }))
    rig.owners.update({100: UID, 200: UID + 1})
    for gpu in gpus:
        rig.snapshots[gpu.uuid] = snapshot(gpu)
    rig.snapshots[gpus[1].uuid] = snapshot(gpus[1], used=200, apps=((100, 100),))
    rig.snapshots[gpus[2].uuid] = snapshot(gpus[2], used=8000, apps=((200, 7900),))
    (rig.state_dir / "release_3").touch()
    monkeypatch.setattr(hold, "recorded_placeholder", lambda state, gpu: 300 if gpu == gpus[0] else None)
    assert hold.main(["status", "--state-dir", str(rig.state_dir)]) == 0
    lines = capsys.readouterr().out.splitlines()
    for line, expected in zip(lines, ["held", "our job running", "other users present", "released", "free"]):
        assert f": {expected} |" in line
    assert len(lines) == 5
    assert not rig.children


def test_status_does_not_trust_pid_file_alone(tmp_path):
    (tmp_path / "hold_0.pid").write_text(f"{os.getpid()}\n")
    assert hold.recorded_placeholder(tmp_path, GPU0) is None


def test_child_allocates_once_and_sleeps_with_fake_torch(tmp_path, monkeypatch):
    clock = [1_000.0]
    allocations = []
    sleeps = []
    threads = []
    monkeypatch.setattr(hold.time, "time", lambda: clock[0])
    monkeypatch.setattr(hold, "query_gpu", lambda *args, **kwargs: snapshot())
    monkeypatch.setattr(hold.sys, "stdin", SimpleNamespace(fileno=lambda: 99))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4")

    def select_(read, write, error, timeout):
        if timeout:
            sleeps.append(timeout)
            clock[0] += timeout
        return [], [], []

    monkeypatch.setattr(hold.select, "select", select_)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        set_num_threads=threads.append, uint8="uint8",
        cuda=SimpleNamespace(mem_get_info=lambda device: (1_000_000, 2_000_000)),
        empty=lambda size, **kwargs: allocations.append((size, kwargs)),
    ))
    assert hold.placeholder(["--gpu", GPU0.uuid, "--label", "0", "--state-dir", str(tmp_path),
                             "--until-epoch", "1002.5", "--claim-fraction", "0.92",
                             "--free-threshold-mb", "1500"]) == 0
    assert allocations == [(920_000, {"dtype": "uint8", "device": "cuda:0"})]
    assert threads == [1]
    assert sleeps == [1.0, 1.0, 0.5]
    assert os.environ["CUDA_VISIBLE_DEVICES"] == GPU0.uuid


@pytest.mark.parametrize("reason", ["release", "deadline", "parent_exit"])
def test_child_exits_before_import_or_cuda_if_no_longer_needed(tmp_path, monkeypatch, reason):
    monkeypatch.setitem(sys.modules, "torch", None)  # Any import would fail.
    if reason == "release":
        (tmp_path / "release_0").touch()
    monkeypatch.setattr(hold.time, "time", lambda: 1001 if reason == "deadline" else 999)
    monkeypatch.setattr(hold.sys, "stdin", SimpleNamespace(fileno=lambda: 99))
    monkeypatch.setattr(hold.select, "select", lambda *args: ([hold.sys.stdin], [], []))
    monkeypatch.setattr(hold.os, "read", lambda *args: b"")
    assert hold.placeholder(["--gpu", GPU0.uuid, "--label", "0", "--state-dir", str(tmp_path),
                             "--until-epoch", "1000", "--claim-fraction", "0.92",
                             "--free-threshold-mb", "1500"]) == 0


def test_child_rechecks_before_allocating_when_another_user_arrives(tmp_path, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("busy GPU must not initialize CUDA or allocate")
    monkeypatch.setattr(hold.time, "time", lambda: 999)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4")
    monkeypatch.setattr(hold.select, "select", lambda *args: ([], [], []))
    monkeypatch.setattr(hold, "query_gpu", lambda *args, **kwargs: snapshot(used=100, apps=((20, 1),)))
    monkeypatch.setattr(hold, "process_uid", lambda pid: UID + 1)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        set_num_threads=lambda count: None,
        cuda=SimpleNamespace(mem_get_info=unexpected), empty=unexpected,
    ))
    assert hold.placeholder(["--gpu", GPU0.uuid, "--label", "0", "--state-dir", str(tmp_path),
                             "--until-epoch", "1000", "--claim-fraction", "0.92",
                             "--free-threshold-mb", "1500"]) == 0


def test_module_import_does_not_import_torch():
    result = subprocess.run([sys.executable, "-c",
                             "import sys; from tools import gpu_hold; assert 'torch' not in sys.modules"],
                            cwd=ROOT, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr


@pytest.fixture
def guard_rig(tmp_path, monkeypatch):
    """Exercise the real query parser and child loop without CUDA or subprocesses."""
    gb = hold.GB_MB
    state = SimpleNamespace(clock=1000.0, total=128 * gb, job=30 * gb, job_alive=True,
                            foreign=[], live=0, cached=0, allocations=[], events=[],
                            ticks=[], queries=[], fail_query=False, query_hook=None)
    release = tmp_path / "release_0"
    release.touch()
    monkeypatch.delenv("GPU_HOLD_RESERVE_GB", raising=False)
    monkeypatch.setattr(hold.time, "time", lambda: state.clock)
    monkeypatch.setattr(hold.time, "monotonic", lambda: state.clock)
    monkeypatch.setattr(hold, "process_uid",
                        lambda pid: UID if pid in (100, os.getpid()) else UID + 1)
    monkeypatch.setattr(hold.sys, "stdin", SimpleNamespace(fileno=lambda: 99))

    def free_mb():
        return state.total - state.job - sum(mb for _, mb in state.foreign) - state.live - state.cached

    def smi(args, **kwargs):
        assert args[0] == "nvidia-smi" and "--id=GPU-aaaa" in args
        state.queries.append((state.clock, kwargs["timeout"]))
        if state.fail_query:
            raise subprocess.TimeoutExpired("nvidia-smi", kwargs["timeout"])
        if state.query_hook:
            state.query_hook()
        if "--query-compute-apps=pid,used_gpu_memory" in args:
            apps = ([(100, state.job)] if state.job_alive else []) + state.foreign
            if state.live + state.cached:
                apps.append((os.getpid(), state.live + state.cached))
            output = "".join(f"{pid}, {mb}\n" for pid, mb in apps)
        else:
            output = f"0, GPU-aaaa, {state.total - free_mb()}, {state.total}\n"
        return SimpleNamespace(stdout=output)

    def select_(read, write, error, timeout):
        state.clock += timeout
        while state.ticks and state.ticks[0][0] <= state.clock:
            _, action = state.ticks.pop(0)
            action()
        return [], [], []

    class Tensor:
        def __init__(self, size):
            self.mb = size // hold.MIB
            assert self.mb <= free_mb()
            state.live += self.mb

        def __del__(self):
            state.live -= self.mb
            state.cached += self.mb
            state.events.append(("delete", state.clock, self.mb))

    def empty(size, **kwargs):
        assert kwargs == {"dtype": "uint8", "device": "cuda:0"}
        state.allocations.append(size // hold.MIB)
        state.events.append(("allocate", state.clock, size // hold.MIB))
        return Tensor(size)

    def empty_cache():
        state.events.append(("empty_cache", state.clock, state.cached))
        state.cached = 0

    monkeypatch.setattr(hold.subprocess, "run", smi)
    monkeypatch.setattr(hold.select, "select", select_)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        set_num_threads=lambda _: None, uint8="uint8", empty=empty,
        cuda=SimpleNamespace(empty_cache=empty_cache,
                             mem_get_info=lambda _: (free_mb() * hold.MIB, state.total * hold.MIB)),
    ))

    def run(duration=2.5, interval=1):
        return hold.placeholder([
            "--gpu", GPU0.uuid, "--label", "0", "--state-dir", str(tmp_path),
            "--until-epoch", str(1000 + duration), "--claim-fraction", "0.92",
            "--free-threshold-mb", "1500", "--guard", "--check-seconds", str(interval),
        ])

    state.run = run
    state.release = release
    return state


@pytest.mark.parametrize("total_gb", [96, 128])
def test_guard_sizing_excludes_own_placeholder_and_reserves_job_budget(guard_rig, total_gb):
    state = guard_rig
    state.total = total_gb * hold.GB_MB
    state.ticks = [(1001, lambda: setattr(state, "job", 60 * hold.GB_MB))]
    assert state.run() == 0
    # 128 total - 30 job - (66 budget - 30 job) = 62 guard.
    # Growing to 60 consumes the reserved headroom, without inflating the guard.
    assert state.allocations == [(total_gb - 66) * hold.GB_MB]
    assert state.live == state.cached == 0


def test_guard_shrinks_on_pressure_and_returns_cached_tensor_to_cuda(guard_rig):
    state = guard_rig
    state.ticks = [(1001, lambda: setattr(state, "job", int(64.5 * hold.GB_MB)))]
    assert state.run() == 0
    assert state.allocations == [62 * hold.GB_MB, int(61.5 * hold.GB_MB)]
    assert state.events[2:5] == [
        ("delete", 1001, 62 * hold.GB_MB),
        ("empty_cache", 1001, 62 * hold.GB_MB),
        ("allocate", 1001, int(61.5 * hold.GB_MB)),
    ]


def test_guard_grows_only_when_slack_exceeds_four_gb(guard_rig):
    state = guard_rig
    state.job = 70 * hold.GB_MB
    state.ticks = [
        (1001, lambda: setattr(state, "job", 66 * hold.GB_MB)),  # Exactly 4 GB slack.
        (1002, lambda: setattr(state, "job", 60 * hold.GB_MB)),
    ]
    assert state.run(duration=3) == 0
    assert state.allocations == [56 * hold.GB_MB, 62 * hold.GB_MB]
    assert [t for event, t, _ in state.events if event == "allocate"] == [1000, 1002]


@pytest.mark.parametrize("foreign_mb", [1, 63 * hold.GB_MB])
def test_guard_does_not_allocate_with_foreign_process(guard_rig, foreign_mb):
    state = guard_rig
    state.foreign = [(200, foreign_mb)]
    assert state.run() == 0
    assert state.allocations == []


def test_existing_guard_leaves_allocation_unchanged_with_foreign_process(guard_rig):
    state = guard_rig
    def other_arrives():
        state.foreign = [(200, hold.GB_MB)]
        state.release.write_text("50")  # Would otherwise grow the guard.
    state.ticks = [(1001, other_arrives)]
    assert state.run() == 0
    assert state.allocations == [62 * hold.GB_MB]


@pytest.mark.parametrize("foreign", [False, True])
def test_guard_frees_entirely_within_interval_under_one_gb(guard_rig, foreign):
    state = guard_rig
    def pressure():
        if foreign:
            state.foreign = [(200, int(35.5 * hold.GB_MB))]
        else:
            state.job = int(65.5 * hold.GB_MB)
    state.ticks = [(1000.5, pressure)]
    assert state.run(duration=60, interval=30) == 0
    assert state.clock == 1001  # Local pressure check doesn't wait for nvidia-smi.
    assert state.allocations == [62 * hold.GB_MB]
    assert state.live == state.cached == 0


@pytest.mark.parametrize("contents,env,expected_gb", [
    ("72\n", None, 72), (" 70.5 \n", "80", 70.5), ("", "80", 80),
    ("not a number", None, 66), ("nan", "inf", 66), ("-5", "-1", 66),
    ("1e308", "1e309", 66),
])
def test_guard_release_file_numeric_reserve_and_env_default(guard_rig, monkeypatch,
                                                          contents, env, expected_gb):
    state = guard_rig
    state.release.write_text(contents)
    if env is not None:
        monkeypatch.setenv("GPU_HOLD_RESERVE_GB", env)
    assert state.run() == 0
    assert state.allocations == [int((128 - expected_gb) * hold.GB_MB)]


@pytest.mark.parametrize("reason", ["job_exit", "release_removed", "query_failure"])
def test_guard_yields_when_no_longer_safe_or_needed(guard_rig, reason):
    state = guard_rig
    def change():
        if reason == "job_exit":
            state.job_alive = False
            state.job = 0
        elif reason == "release_removed":
            state.release.unlink()
        else:
            state.fail_query = True
    state.ticks = [(1001, change)]
    assert state.run() == 0
    assert state.clock == 1001
    assert state.live == state.cached == 0


def test_guard_rechecks_foreign_process_before_reallocation(guard_rig):
    state = guard_rig
    def arrive_after_drop():
        if state.allocations and not state.live:
            state.foreign = [(200, hold.GB_MB)]
    state.query_hook = arrive_after_drop
    state.ticks = [(1001, lambda: state.release.write_text("80"))]
    assert state.run() == 0
    assert state.allocations == [62 * hold.GB_MB]
    assert state.clock == 1001
    assert state.live == state.cached == 0


def test_daemon_starts_guard_reaps_it_on_pressure_and_keeps_full_mode(rig):
    holder = rig.holder
    holder.release_path(GPU0).write_text("66")
    rig.owners[100] = UID
    rig.snapshots[GPU0.uuid] = hold.Snapshot(
        0, GPU0.uuid, 30 * hold.GB_MB, 128 * hold.GB_MB,
        (hold.ComputeApp(100, 30 * hold.GB_MB),))
    holder.check_once()
    guard, full = rig.children
    assert "--guard" in rig.launches[0][0]
    assert "--guard" not in rig.launches[1][0]
    assert rig.launches[0][0][-2:] == ["--check-seconds", "30"]
    holder.check_once()  # Preserve even before the child's CUDA context appears.
    assert len(rig.children) == 2
    assert not guard.signals and not full.signals
    rig.snapshots[GPU0.uuid] = hold.Snapshot(
        0, GPU0.uuid, int(127.5 * hold.GB_MB), 128 * hold.GB_MB,
        (hold.ComputeApp(100, int(65.5 * hold.GB_MB)),
         hold.ComputeApp(guard.pid, 62 * hold.GB_MB)))
    holder.check_once()
    assert guard.signals == [signal.SIGTERM]
    assert not full.signals
    assert GPU0.uuid not in holder.guards
    assert not holder.pid_path(GPU0).exists()


def test_status_reports_guard_job_and_others(rig, monkeypatch, capsys):
    (rig.state_dir / "config.json").write_text(json.dumps({"gpus": [hold.asdict(GPU0)]}))
    rig.owners.update({100: UID, 200: UID + 1})
    rig.holder.release_path(GPU0).touch()
    rig.snapshots[GPU0.uuid] = hold.Snapshot(
        0, GPU0.uuid, 94 * hold.GB_MB, 128 * hold.GB_MB,
        (hold.ComputeApp(100, 30 * hold.GB_MB), hold.ComputeApp(200, 2 * hold.GB_MB),
         hold.ComputeApp(300, 62 * hold.GB_MB)))
    monkeypatch.setattr(hold, "recorded_placeholder", lambda *_: 300)
    assert hold.print_status(rig.state_dir, [GPU0.uuid], None) == 0
    assert "guarded (guard 62.0 GB, job 30.0 GB, others 2.0 GB)" in capsys.readouterr().out
    assert not rig.children
