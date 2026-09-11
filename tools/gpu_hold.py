#!/usr/bin/env python3
"""Cooperatively hold idle GPUs; only the private child imports torch."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime
import fcntl
import json
import logging
import math
import os
from pathlib import Path
import re
import select
import signal
import subprocess
import sys
import threading
import time
from typing import Callable, TextIO


LOG = logging.getLogger("gpu_hold")
SCRIPT = Path(__file__).resolve()
DEFAULT_STATE_DIR = SCRIPT.parents[1] / ".gpu_hold"
MIB = 1024 ** 2
GB_MB = 1024  # GB settings use GiB, matching nvidia-smi's MiB counters.


@dataclass(frozen=True)
class ComputeApp:
    pid: int
    used_mb: int | None


@dataclass(frozen=True)
class Snapshot:
    index: int
    uuid: str
    used_mb: int
    total_mb: int
    apps: tuple[ComputeApp, ...]


@dataclass(frozen=True)
class GPU:
    label: str
    uuid: str


def query_gpu(selector: str, timeout: float = 10.0) -> Snapshot:
    """Query compute PIDs and physical memory; any query failure fails closed."""
    deadline = time.monotonic() + timeout

    def query(fields: str) -> list[list[str]]:
        result = subprocess.run(
            ["nvidia-smi", f"--id={selector}", fields, "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True,
            timeout=max(0.001, deadline - time.monotonic()),
        )
        return [row for row in csv.reader(result.stdout.splitlines(), skipinitialspace=True)
                if row and any(cell.strip() for cell in row)]

    apps = []
    for row in query("--query-compute-apps=pid,used_gpu_memory"):
        pid, memory = row
        # Unsupported per-process accounting must not hide the process itself.
        apps.append(ComputeApp(int(pid), int(memory) if memory.strip().isdigit() else None))
    rows = query("--query-gpu=index,uuid,memory.used,memory.total")
    if len(rows) != 1:
        raise ValueError(f"expected exactly one GPU for {selector!r}")
    index, uuid, used, total = rows[0]
    snapshot = Snapshot(int(index), uuid.strip(), int(used), int(total), tuple(apps))
    if not 0 <= snapshot.used_mb <= snapshot.total_mb or snapshot.total_mb <= 0:
        raise ValueError(f"invalid GPU memory counters for {selector!r}")
    return snapshot


def process_uid(pid: int, proc_root: Path = Path("/proc")) -> int | None:
    """Return the real owner UID, or unknown if the PID vanished/is unreadable."""
    if pid <= 0:
        return None
    try:
        for line in (proc_root / str(pid) / "status").read_text().splitlines():
            if line.startswith("Uid:"):
                return int(line.split()[1])
    except (OSError, ValueError, IndexError):
        pass
    return None


@dataclass(frozen=True)
class MemoryUsage:
    guard_mb: int
    job_mb: int
    others_mb: int
    job_alive: bool
    foreign: bool
    known: bool


def memory_usage(snapshot: Snapshot, uid: int, held_pid: int | None = None) -> MemoryUsage:
    guard = job = others = 0
    job_alive = foreign = False
    known = True
    for app in snapshot.apps:
        known &= app.used_mb is not None
        memory = app.used_mb or 0
        if app.pid == held_pid:
            guard += memory
        elif process_uid(app.pid) == uid:
            job_alive = True
            job += memory
        else:
            foreign = True  # Unknown owners also block new guard allocations.
            others += memory
    others += max(0, snapshot.used_mb - guard - job - others)
    return MemoryUsage(guard, job, others, job_alive, foreign, known)


def read_reserve_mb(release: Path) -> int:
    """A numeric release file takes precedence over the environment default."""
    for value in (release.read_text().strip(), os.environ.get("GPU_HOLD_RESERVE_GB", "66"), "66"):
        try:
            number = float(value) * GB_MB
            if math.isfinite(number) and number > 0:
                return int(number)
        except ValueError:
            pass
    raise AssertionError("the built-in reserve is valid")


def guard_headroom_mb(usage: MemoryUsage, reserve_mb: int) -> int:
    # Preserve the remaining job budget, with room to exceed it without fighting
    # the guard at the allocation-pressure boundary.
    return max(2 * GB_MB, reserve_mb - usage.job_mb)


def guard_target_mb(snapshot: Snapshot, usage: MemoryUsage, current_mb: int,
                    reserve_mb: int) -> int:
    free_mb = snapshot.total_mb - snapshot.used_mb
    if not usage.job_alive or not usage.known or free_mb < GB_MB:
        return 0
    if usage.foreign:
        # Do not compete with an existing foreign process. Pressure still yields
        # our allocation, including when the foreign process caused the pressure.
        return 0 if free_mb < 2 * GB_MB else current_mb
    headroom = guard_headroom_mb(usage, reserve_mb)
    target = max(0, current_mb + free_mb - headroom)
    if current_mb == 0 or target < current_mb or free_mb > headroom + 4 * GB_MB:
        return target
    return current_mb


def classify(snapshot: Snapshot, uid: int, free_threshold_mb: int,
             held_pid: int | None = None, released: bool = False) -> str:
    if released:
        usage = memory_usage(snapshot, uid, held_pid)
        if held_pid is not None and usage.job_alive and usage.guard_mb > 0:
            return (f"guarded (guard {usage.guard_mb / GB_MB:.1f} GB, "
                    f"job {usage.job_mb / GB_MB:.1f} GB, "
                    f"others {usage.others_mb / GB_MB:.1f} GB)")
        return "released"
    owners = {app.pid: process_uid(app.pid) for app in snapshot.apps}
    if any(owner == uid and pid != held_pid for pid, owner in owners.items()):
        return "our job running"
    if held_pid is not None:
        return "held"
    if owners or snapshot.used_mb >= free_threshold_mb:
        # Unknown UIDs and memory without a visible compute PID also block claims.
        return "other users present"
    return "free"


class AlreadyRunning(RuntimeError):
    pass


class DaemonLock:
    """The lock inode persists; unlinking it would allow two independent locks."""

    def __init__(self, state_dir: Path):
        self.path = state_dir / "daemon.lock"
        self.file: TextIO | None = None

    def __enter__(self) -> DaemonLock:
        self.file = self.path.open("a+")
        try:
            fcntl.flock(self.file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.file.close()
            self.file = None
            raise AlreadyRunning(f"a daemon already holds {self.path}") from exc
        self.file.seek(0)
        self.file.truncate()
        self.file.write(f"{os.getpid()}\n")
        self.file.flush()
        return self

    def __exit__(self, *_: object) -> None:
        if self.file is not None:
            self.file.close()
            self.file = None


def remove_pid_file(path: Path, pid: int) -> None:
    try:
        if path.read_text().strip() == str(pid):
            path.unlink()
    except FileNotFoundError:
        pass


class Holder:
    def __init__(self, gpus: list[GPU], state_dir: Path, until: float,
                 check_seconds: float = 30, claim_fraction: float = 0.92,
                 free_threshold_mb: int = 1500):
        self.gpus = gpus
        self.state_dir = state_dir
        self.until = until
        self.check_seconds = check_seconds
        self.claim_fraction = claim_fraction
        self.free_threshold_mb = free_threshold_mb
        self.uid = os.getuid()
        self.children: dict[str, subprocess.Popen] = {}
        self.guards: set[str] = set()
        self.stop_event = threading.Event()

    def release_path(self, gpu: GPU) -> Path:
        return self.state_dir / f"release_{gpu.label}"

    def pid_path(self, gpu: GPU) -> Path:
        return self.state_dir / f"hold_{gpu.label}.pid"

    def start_placeholder(self, gpu: GPU, guard: bool = False) -> None:
        if (self.stop_event.is_set() or time.time() >= self.until
                or self.release_path(gpu).exists() != guard):
            return
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu.uuid,
                   OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        args = [sys.executable, "-u", str(SCRIPT), "_hold", "--gpu", gpu.uuid,
                "--label", gpu.label, "--state-dir", str(self.state_dir),
                "--until-epoch", str(self.until), "--claim-fraction", str(self.claim_fraction),
                "--free-threshold-mb", str(self.free_threshold_mb)]
        if guard:
            args += ["--guard", "--check-seconds", str(self.check_seconds)]
        with (self.state_dir / f"hold_{gpu.label}.log").open("a") as log:
            child = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=log,
                                     stderr=subprocess.STDOUT, env=env, close_fds=True)
        # Keep the pipe open: the child observes EOF even if this daemon is SIGKILLed.
        self.children[gpu.uuid] = child
        if guard:
            self.guards.add(gpu.uuid)
        self.pid_path(gpu).write_text(f"{child.pid}\n")
        LOG.info("GPU %s (%s): started %s PID %s", gpu.label, gpu.uuid,
                 "guard" if guard else "placeholder", child.pid)

    def stop_placeholder(self, gpu: GPU) -> None:
        child = self.children.get(gpu.uuid)
        if child is None:
            return
        # Never signal PIDs read from state files, other users, or unrelated jobs.
        if child.poll() is None and process_uid(child.pid) == self.uid:
            child.terminate()
        if child.stdin is not None and not child.stdin.closed:
            child.stdin.close()
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            if process_uid(child.pid) != self.uid:
                LOG.error("GPU %s: cannot verify child PID %s ownership; no signal sent",
                          gpu.label, child.pid)
                return
            child.kill()
            child.wait(timeout=5)
        self.children.pop(gpu.uuid, None)
        self.guards.discard(gpu.uuid)
        remove_pid_file(self.pid_path(gpu), child.pid)
        LOG.info("GPU %s: placeholder PID %s stopped", gpu.label, child.pid)

    def check_once(self) -> None:
        # Releases come first, including when nvidia-smi fails on another GPU.
        for gpu in self.gpus:
            child = self.children.get(gpu.uuid)
            if child is not None and (child.poll() is not None
                    or self.release_path(gpu).exists() != (gpu.uuid in self.guards)):
                self.stop_placeholder(gpu)
        for gpu in self.gpus:
            remaining = self.until - time.time()
            if remaining <= 0 or self.stop_event.is_set():
                break
            try:
                snapshot = query_gpu(gpu.uuid, timeout=min(10, remaining))
            except (OSError, ValueError, subprocess.SubprocessError) as exc:
                LOG.warning("GPU %s: query failed; skipping claim: %s", gpu.label, exc)
                if gpu.uuid in self.guards:
                    self.stop_placeholder(gpu)
                continue
            child = self.children.get(gpu.uuid)
            released = self.release_path(gpu).exists()
            if released:
                if child is not None and gpu.uuid not in self.guards:
                    self.stop_placeholder(gpu)
                    continue  # Query again after the full placeholder has gone.
                usage = memory_usage(snapshot, self.uid, child.pid if child else None)
                if child is not None:
                    if (not usage.job_alive or not usage.known
                            or snapshot.total_mb - snapshot.used_mb < GB_MB):
                        self.stop_placeholder(gpu)
                else:
                    try:
                        reserve = read_reserve_mb(self.release_path(gpu))
                    except OSError as exc:
                        LOG.warning("GPU %s: cannot read release file: %s", gpu.label, exc)
                        continue
                    if guard_target_mb(snapshot, usage, 0, reserve) > 0:
                        self.start_placeholder(gpu, guard=True)
                continue
            if gpu.uuid in self.guards:
                self.stop_placeholder(gpu)
                continue
            state = classify(snapshot, self.uid, self.free_threshold_mb,
                             child.pid if child else None)
            if state == "our job running" and child is not None:
                self.stop_placeholder(gpu)
            elif state == "free":
                self.start_placeholder(gpu)

    def run(self) -> None:
        try:
            while not self.stop_event.is_set() and time.time() < self.until:
                self.check_once()
                self.stop_event.wait(max(0, min(self.check_seconds, self.until - time.time())))
        finally:
            for gpu in self.gpus:
                self.stop_placeholder(gpu)


def guard_placeholder(torch, args: argparse.Namespace, release: Path,
                      should_exit: Callable[[float], bool]) -> int:
    """Resize in the child so a slow query on another GPU cannot delay yielding."""
    allocation = None
    allocation_mb = 0
    try:
        while not should_exit(0):
            next_check = time.monotonic() + args.check_seconds

            def query() -> Snapshot:
                return query_gpu(args.gpu, timeout=max(0.001, min(
                    10, args.check_seconds / 2, args.until_epoch - time.time())))

            snapshot = query()
            usage = memory_usage(snapshot, os.getuid(), os.getpid())
            reserve = read_reserve_mb(release)
            target_mb = guard_target_mb(snapshot, usage, allocation_mb, reserve)
            if target_mb == 0:
                return 0
            if target_mb != allocation_mb:
                # Dropping a tensor alone leaves it in PyTorch's caching allocator.
                # Return the old block to CUDA before measuring or allocating again.
                allocation = None
                allocation_mb = 0
                torch.cuda.empty_cache()
                # Recheck ownership and budget after freeing: a job or foreign
                # process can arrive while torch imports or an old block is freed.
                snapshot = query()
                usage = memory_usage(snapshot, os.getuid(), os.getpid())
                reserve = read_reserve_mb(release)
                target_mb = min(target_mb, guard_target_mb(snapshot, usage, 0, reserve))
                if target_mb <= 0 or should_exit(0):
                    return 0
                free_bytes, _ = torch.cuda.mem_get_info(0)
                target_mb = min(target_mb, free_bytes // MIB - guard_headroom_mb(usage, reserve))
                if target_mb <= 0 or should_exit(0):
                    return 0
                allocation = torch.empty(target_mb * MIB, dtype=torch.uint8, device="cuda:0")
                allocation_mb = target_mb
                print(f"Guard holding {allocation_mb / GB_MB:.1f} GB on {args.gpu}; "
                      f"job {usage.job_mb / GB_MB:.1f} GB; PID {os.getpid()}", flush=True)
            while not should_exit(0):
                # Check pressure locally at least every second, even between the
                # more expensive nvidia-smi sizing checks. Exiting drops the CUDA
                # context too, and the daemon will reap us on its next check.
                if torch.cuda.mem_get_info(0)[0] < GB_MB * MIB:
                    return 0
                remaining = next_check - time.monotonic()
                if remaining <= 0:
                    break
                if should_exit(min(1.0, remaining)):
                    return 0
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f"Guard query failed; yielding GPU {args.gpu}: {exc}", flush=True)
    finally:
        allocation = None
        torch.cuda.empty_cache()
    return 0


def placeholder(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Private GPU holder child")
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--until-epoch", type=float, required=True)
    parser.add_argument("--claim-fraction", type=float, required=True)
    parser.add_argument("--free-threshold-mb", type=int, required=True)
    parser.add_argument("--guard", action="store_true")
    parser.add_argument("--check-seconds", type=float, default=30)
    args = parser.parse_args(argv)
    release = args.state_dir / f"release_{args.label}"

    def should_exit(timeout: float = 0) -> bool:
        remaining = args.until_epoch - time.time()
        if remaining <= 0 or release.exists() != args.guard:
            return True
        readable, _, _ = select.select([sys.stdin], [], [], min(timeout, remaining))
        return bool(readable) and os.read(sys.stdin.fileno(), 1) == b""

    if should_exit():
        return 0
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    import torch  # Only this child needs torch or initializes CUDA.

    torch.set_num_threads(1)
    # Recheck after slow imports and before initializing a CUDA context.
    if should_exit():
        return 0
    if args.guard:
        return guard_placeholder(torch, args, release, should_exit)
    snapshot = query_gpu(args.gpu, timeout=min(10, args.until_epoch - time.time()))
    if classify(snapshot, os.getuid(), args.free_threshold_mb) != "free" or should_exit():
        print("GPU is no longer free (or release/deadline reached); exiting", flush=True)
        return 0
    free_bytes, _ = torch.cuda.mem_get_info(0)
    allocation_bytes = int(free_bytes * args.claim_fraction)
    if should_exit() or allocation_bytes <= 0:
        return 0
    # uint8 makes numel == bytes. No fill, kernels, or periodic CUDA operations.
    allocation = torch.empty(allocation_bytes, dtype=torch.uint8, device="cuda:0")
    print(f"Holding {allocation_bytes} bytes on {args.gpu}; PID {os.getpid()}", flush=True)
    while not should_exit(1.0):
        pass
    del allocation
    return 0


def parse_gpus(values: list[str]) -> list[str]:
    labels = []
    for value in values:
        for token in value.split(","):
            token = token.strip()
            if not re.fullmatch(r"(?:[0-9]+|GPU-[0-9a-fA-F-]+)", token):
                raise ValueError(f"invalid GPU index or UUID: {token!r}")
            label = str(int(token)) if token.isdigit() else token
            if label not in labels:
                labels.append(label)
    return labels


def resolve_gpus(labels: list[str], deadline: float | None = None) -> list[GPU]:
    gpus = []
    seen = set()
    for label in labels:
        remaining = 10 if deadline is None else deadline - time.time()
        if remaining <= 0:
            break
        try:
            snapshot = query_gpu(label, timeout=min(10, remaining))
        except subprocess.TimeoutExpired:
            if deadline is not None and time.time() >= deadline:
                break
            raise
        if snapshot.uuid not in seen:
            gpus.append(GPU(label, snapshot.uuid))
            seen.add(snapshot.uuid)
    return gpus


def recorded_placeholder(state_dir: Path, gpu: GPU) -> int | None:
    """Validate status metadata; PID files are never an authority to kill."""
    try:
        pid = int((state_dir / f"hold_{gpu.label}.pid").read_text().strip())
        if process_uid(pid) != os.getuid():
            return None
        argv = (Path("/proc") / str(pid) / "cmdline").read_bytes().split(b"\0")
        expected = [str(SCRIPT), "_hold", "--gpu", gpu.uuid, "--label", gpu.label,
                    "--state-dir", str(state_dir)]
        if argv[2:2 + len(expected)] != [value.encode() for value in expected]:
            return None
        return pid
    except (OSError, ValueError):
        return None


def print_status(state_dir: Path, labels: list[str] | None, threshold: int | None) -> int:
    config_path = state_dir / "config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    configured = [GPU(**gpu) for gpu in config.get("gpus", [])]
    if labels is None:
        gpus = configured
    else:
        # Reuse the daemon's filename label even if status uses the UUID alias.
        by_uuid = {gpu.uuid: gpu for gpu in configured}
        gpus = [by_uuid.get(gpu.uuid, gpu) for gpu in resolve_gpus(labels)]
    if not gpus:
        raise ValueError("status requires --gpus or an existing state-dir/config.json")
    threshold = threshold if threshold is not None else config.get("free_threshold_mb", 1500)
    result = 0
    for gpu in gpus:
        released = (state_dir / f"release_{gpu.label}").exists()
        try:
            snapshot = query_gpu(gpu.uuid)
            state = classify(snapshot, os.getuid(), threshold,
                             recorded_placeholder(state_dir, gpu), released)
            detail = f"{snapshot.used_mb}/{snapshot.total_mb} MiB"
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            state = "released" if released else "unknown"
            detail = f"query failed: {exc}"
            result = 1
        print(f"{gpu.label} ({gpu.uuid}): {state} | {detail}")
    return result


def local_deadline(value: str) -> float:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is not None or "T" not in value:
        raise ValueError("--until must be local time, e.g. 2026-09-26T23:59 (no UTC offset)")
    return parsed.timestamp()


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "_hold":
        return placeholder(argv[1:])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", choices=("run", "status"), default="run")
    parser.add_argument("--gpus", nargs="+", help="physical indices or UUIDs, comma/space separated")
    parser.add_argument("--until", help="local deadline, e.g. 2026-09-26T23:59")
    parser.add_argument("--check-seconds", type=float, default=30)
    parser.add_argument("--claim-fraction", type=float, default=0.92)
    parser.add_argument("--free-threshold-mb", type=int, default=None,
                        help="claim only below this used MiB threshold (default: 1500)")
    parser.add_argument("--state-dir", type=Path, default=DEFAULT_STATE_DIR)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        if not math.isfinite(args.check_seconds) or args.check_seconds <= 0:
            raise ValueError("--check-seconds must be finite and positive")
        if not 0 < args.claim_fraction < 1:
            raise ValueError("--claim-fraction must be between 0 and 1, exclusive")
        if args.free_threshold_mb is not None and args.free_threshold_mb < 0:
            raise ValueError("--free-threshold-mb must be nonnegative")
        labels = parse_gpus(args.gpus) if args.gpus else None
        state_dir = args.state_dir.expanduser().resolve()
        if args.command == "status":
            return print_status(state_dir, labels, args.free_threshold_mb)
        if not labels or not args.until:
            raise ValueError("run requires --gpus and --until")
        until = local_deadline(args.until)
        if time.time() >= until:
            LOG.info("Deadline already passed; nothing started")
            return 0
        state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        if state_dir.stat().st_uid != os.getuid():
            raise ValueError("state directory must belong to our UID")
        with DaemonLock(state_dir):
            gpus = resolve_gpus(labels, deadline=until)
            if time.time() >= until:
                return 0
            holder = Holder(gpus, state_dir, until, args.check_seconds, args.claim_fraction,
                            args.free_threshold_mb if args.free_threshold_mb is not None else 1500)
            config = dict(gpus=[asdict(gpu) for gpu in gpus], until=args.until,
                          free_threshold_mb=holder.free_threshold_mb)
            temporary = state_dir / "config.json.tmp"
            temporary.write_text(json.dumps(config, indent=2) + "\n")
            temporary.replace(state_dir / "config.json")
            daemon_pid = state_dir / "daemon.pid"
            daemon_pid.write_text(f"{os.getpid()}\n")
            previous_handlers = {}
            try:
                for signum in (signal.SIGTERM, signal.SIGINT):
                    previous_handlers[signum] = signal.signal(
                        signum, lambda *_: holder.stop_event.set())
                LOG.info("Watching %s until %s (local)", ", ".join(labels), args.until)
                holder.run()
            finally:
                for signum, handler in previous_handlers.items():
                    signal.signal(signum, handler)
                remove_pid_file(daemon_pid, os.getpid())
        return 0
    except AlreadyRunning as exc:
        LOG.info("%s; nothing started", exc)
        return 0
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        LOG.error("%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
