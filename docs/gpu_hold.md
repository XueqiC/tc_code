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

The daemon stops its full placeholder on the next check. That child also checks
the release file itself, so an established full holder usually exits within one
second. While the release file exists and the GPU has no job with our UID, it
stays released. Once our job appears, the daemon can start a **guard** in the
same `hold_4.pid`/`hold_4.log` slot. Wait for the initial full holder to disappear
**before** launching the job; the PID file can reappear for its guard afterward.

Without a release file, the existing full-placeholder behavior is unchanged:
our real jobs block claiming, and a full placeholder yields if our job appears.
Removing the release file stops any guard; full holding resumes once our jobs
finish and the GPU meets the idle-claim rules. Other users' processes are never
signaled.

## Guarding a released job

A guard holds memory beyond our job's total budget, allowing a job that starts
small to grow. `R` is the **total job budget**, including its current usage. It
defaults to 66 GB. A finite positive number in `release_<gpu>` overrides that
GPU's budget; an empty or nonnumeric file uses `GPU_HOLD_RESERVE_GB` from the
daemon's environment, or 66 if that variable is absent or invalid. Values use
GiB (1024 MiB), labeled GB in settings and status. For example:

```bash
# Set GPU 4's total job budget; the guard rereads it at every sizing check.
echo 66 > .gpu_hold/release_4

# Alternatively set the default for empty release files when launching a daemon.
GPU_HOLD_RESERVE_GB=72 tools/gpu_hold.sh --gpus 0,4 --until 2026-09-26T23:59
```

The guard sums all our compute processes except its own PID. It leaves free
headroom of `max(R - job_usage, 2 GB)` and allocates the remaining free memory.
On a 96 GB card with a 30 GB job and R=66, this means a 30 GB guard and 36 GB
free for the job. At 60 GB job usage, the same guard leaves 6 GB free. Physical
memory counters also account for other usage and CUDA overhead, so actual
allocations can be smaller. `--claim-fraction` applies only to full placeholders.

The guard child checks sizing independently every `--check-seconds` (default
30), so queries on another GPU do not delay it:

- If available headroom is too small, including free memory below 2 GB, it
  shrinks by deleting its tensor, calling `torch.cuda.empty_cache()`, and
  reallocating a smaller tensor after another ownership/memory check.
- If free memory exceeds the required headroom by **more than 4 GB**, it grows.
  This margin avoids repeated resizing for small fluctuations. Budget changes
  in the release file use the same rules.
- If any foreign or unknown-owner compute process is present, it makes no new
  allocation and leaves an existing guard alone. This also covers foreign jobs
  already using more than the memory outside our budget. If free memory drops
  below 2 GB in that situation, it yields entirely instead of reallocating.
- If free memory drops **below 1 GB**, it frees the entire guard, including its
  CUDA context, within one check interval. Local free-memory checks run at
  least once a second between sizing checks; the daemon also stops a guard if
  it observes this pressure. Pressure yielding takes precedence over foreign
  process handling and does not reallocate in that check.

A guard exits when our job disappears, the release file is removed, memory
accounting is unavailable, or a query fails. It also obeys the deadline and
parent-pipe cleanup rules. The daemon can start another guard on a later check
when the job is still present and there is safe surplus memory.

These are cooperative memory holds, not scheduler reservations: another process
can start between a free-memory check and allocation. There is no eviction of
other users, and there is no atomic GPU reservation. Use release files for your
own job handoff and allow enough budget for large allocations between checks;
polling cannot prevent an OOM from an allocation that outruns those checks.
`released` reports the request, so wait for the initial full holder's PID-file
removal before launching the job.

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
| `guarded (guard X GB, job Y GB, others Z GB)` | A release file exists, our job is alive, and a verified placeholder is holding guard memory. |
| `released` | A release file exists, with no active guard allocation reported. |
| `free` | Eligible for a claim, but no placeholder currently exists. |
| `unknown` | The GPU query failed; status returns a nonzero exit code. |

Release/guard status takes precedence, followed by our real jobs, our holder,
and other usage. Guard/job/others values exclude the guard PID from job usage;
`others` includes foreign processes and memory not attributed to compute PIDs.
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
processes, and use fake `nvidia-smi` responses and a fake torch caching allocator
to verify full holding, guard sizing, shrinking, growth hysteresis, pressure
release, foreign-process handling, numeric/environment budgets, and status.
They do not start a daemon or initialize CUDA.
