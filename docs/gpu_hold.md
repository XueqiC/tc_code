# GPU holder on shared rai

`tools/gpu_hold.py` watches selected physical GPUs and holds idle memory with a
separate Python child per GPU. The daemon uses only the standard library; only
the child imports PyTorch. This requires Linux `/proc`, `flock`, `nvidia-smi`, and
a CUDA-enabled PyTorch installation for actual holding.

## Launch and inspect

From `/home/xueqi/hq/projects/tc-alignment` on rai:

```bash
tools/gpu_hold.sh --gpus 0,4 --until 2026-09-26T23:59
tools/gpu_hold.sh status
```

The launcher uses `/home/xueqi/hq/projects/tc-alignment/.venv/bin/python`, starts
the daemon with `nohup`, and always uses
`/home/xueqi/hq/projects/tc-alignment/.gpu_hold` for state. `status` and `--help`
run in the foreground. Check `.gpu_hold/daemon.log` for startup errors; the
launcher printing a PID does not establish that initialization succeeded.

Equivalent foreground command with every setting explicit:

```bash
.venv/bin/python tools/gpu_hold.py run \
  --gpus 0,4 \
  --until 2026-09-26T23:59 \
  --check-seconds 30 \
  --claim-fraction 0.92 \
  --free-threshold-mb 1500 \
  --state-dir /home/xueqi/hq/projects/tc-alignment/.gpu_hold
```

`run` is optional. `--gpus` and `--until` are required when running. GPUs may be
comma-separated or space-separated physical `nvidia-smi` indices or NVIDIA GPU
UUIDs, including a mixture. Duplicate physical GPUs are watched once. Indices
are resolved to UUIDs at startup, and children use `CUDA_VISIBLE_DEVICES=<UUID>`
with local CUDA device 0, so inherited CUDA visibility does not remap the target.

`--until` is local wall time on the machine running the daemon, using its `TZ`
environment or system timezone. Supply an ISO date and time without an offset.
A past deadline exits successfully without querying or allocating GPUs. Polling
defaults to 30 seconds, the allocation fraction to 0.92, and the used-memory
threshold to 1500 MiB (`nvidia-smi` units). The fraction must be strictly between
0 and 1; the interval must be positive.

## Claiming and yielding

Each check reads compute-app PIDs and their used memory, plus GPU used/total
memory. A PID belongs to us only when the first `Uid:` field in
`/proc/<pid>/status` equals the daemon's UID. Unreadable or vanished PIDs are
treated as unknown and block claiming for that check.

A claim requires no live placeholder, no compute-app PIDs, and used memory
strictly below the threshold. In particular, a visible foreign process blocks
claiming even if it uses less than 1500 MiB. Memory above the threshold also
blocks claiming when no compute PID is visible. Query failures skip claims and
are logged. Each GPU query has a bounded timeout.

GPU4-style “grab when free” needs no special flag: an occupied GPU is left alone,
then claimed on the first check after it becomes free. After importing torch,
the child checks again before creating its CUDA context. It then measures current
free bytes with `torch.cuda.mem_get_info` and performs one `torch.empty` allocation
of `int(free_bytes * claim_fraction)` bytes using `uint8`. It does no tensor
initialization or recurring GPU computation. Its keepalive waits up to one second
on a pipe, using little CPU. Allocation failures appear in the child's log; the
daemon reaps failed children and can retry on the next eligible check.

Before starting a real job, create the release file and wait for the holder PID
file to disappear:

```bash
STATE=/home/xueqi/hq/projects/tc-alignment/.gpu_hold
touch "$STATE/release_4"
while [ -e "$STATE/hold_4.pid" ]; do sleep 1; done
# Run your job on physical GPU 4 here, keeping release_4 in place.
# When the job is finished:
rm "$STATE/release_4"
```

The daemon stops its placeholder on the next check and will not restart it
while the release file exists. The child also checks the release file itself,
so an established holder usually exits within one second. On release removal,
our remaining real jobs still block claiming; once they finish, holding resumes.
If a real job is detected alongside a placeholder, the daemon also yields its
placeholder. Other users' processes are never signaled.

These are cooperative memory holds, not scheduler reservations: another process
can start between a free-memory check and allocation. There is no eviction of
other users, and there is no atomic GPU reservation. Use release files for your
own job handoff; `released` reports the request, so wait for PID-file removal
before assuming the placeholder has finished exiting.

## State, status, and shutdown

Per-GPU filenames use the first requested selector, normalized for indices:
`--gpus 4` uses `hold_4.pid`, `hold_4.log`, and `release_4`;
`--gpus GPU-<uuid>` uses the corresponding UUID in each name. Keep selectors
consistent across launches. `config.json` records the mapping, allowing status
without repeating GPU arguments:

```bash
.venv/bin/python tools/gpu_hold.py status --state-dir .gpu_hold
.venv/bin/python tools/gpu_hold.py status --gpus 0,4 --state-dir .gpu_hold
```

Status prints one row per GPU, including its UUID and used/total MiB:

| State | Meaning |
| --- | --- |
| `held` | A verified placeholder process is alive (possibly initializing). |
| `our job running` | Another process with our UID is using the GPU. |
| `other users present` | Foreign/unknown compute PIDs or threshold-level memory use blocks claiming. |
| `released` | A release file exists; holding is disabled. |
| `free` | Eligible for a claim, but no placeholder currently exists. |
| `unknown` | The GPU query failed; status returns a nonzero exit code. |

Release takes precedence, followed by our real jobs, our holder, and other usage.
Status reads live GPU data and validates holder PID ownership and command line;
it never starts children or takes the daemon lock. A release still prints as
`released` when its GPU query fails, with the error alongside it.

`daemon.lock` provides an exclusive advisory lock for this state directory.
Starting again with the same directory exits successfully without creating
another daemon. Always use the same state directory for this watcher on rai;
different directories are independent. Do not remove the lock file while a
daemon is running. `daemon.pid` identifies the current daemon, and
`daemon.log` contains its launcher output. Runtime state is Git-ignored.

At the deadline, or on SIGTERM/SIGINT, the daemon stops and reaps its own children
and removes their PID files. Children independently enforce the deadline and
exit when their parent pipe closes, including after a daemon crash. A crash can
leave stale PID files; they are never used as targets for signals. The daemon
signals only live children it created whose UID it can verify. The lock is
released automatically when its process exits; its file remains for reuse.

## CPU tests

```bash
.venv/bin/python -m pytest -q tests/test_gpu_hold.py
```

The tests replace GPU queries and child launches, exercise the real lock across
processes, and use a fake torch allocator to verify the single allocation and
sleeping keepalive. They do not start a daemon or initialize CUDA.
