# C25h: concurrent RTD official evaluation

**C25j follow-up:** the operational lock/port implementation below remains in
use. Its whole-file harness inventory is superseded by the
[protocol v1.0.2 scoring scope](rtd_v1_protocol.md). Existing runs can explicitly
audit scoring continuity with `update-identity --run-dir`; see
[C25j evidence and commands](rtd_v1_status.md).

The C25g rai R1 failure was contention on
`results/rtd_v1/eval-port-9170/.lock`. This was a **port lock**, not a
leaderboard-wide lock: `cli.run_campaign` assigned 9170 to every non-SLURM
run, and `exclusive_run` immediately raised `BlockingIOError` for the second
caller. The lock covered CPU export, the entire campaign, and publication.

## Shared resource audit

The checked-in BFCL implementation uses these paths:

| Resource | Existing BFCL behavior | C25h protection |
| --- | --- | --- |
| Benchmark prompts, ground truth, function documents | `utils.load_dataset_entry` / `load_ground_truth_entry` read package data; no campaign data copy or rewrite | Shared reads; BFCL retains its short per-file read locks |
| Generation files and memory snapshots | `_llm_response_generation.collect_test_cases`, `MemoryAPI._prepare_snapshot`, and handler writes use `--result-dir` | Exclusively created `result_p<port>_<attempt UUID>` per invocation |
| Scores and leaderboard CSVs | `eval_runner.runner` passes `--score-dir` to `generate_leaderboard_csv`; `data_overall.csv` is under that directory | Exclusively created `score_p<port>_<attempt UUID>` per invocation |
| Export, binding, campaign logs, copied results | Paths include the tag: `results/appworld_students/<tag>`, `logs/bfclstd_*_<tag>.log`, `results/bfcl_std/<tag>` | One tag lock from export/cache check through validation and publication |
| Local vLLM listener | Uses `LOCAL_SERVER_PORT` | Port lease plus a free-port probe; lease descriptors inherited by campaign children |

There is no shared leaderboard CSV write or shared-data copy in this campaign,
so no global copy lock is needed. Different tags with different ports can run
at the same time. RTD's run/coordinator locks remain fail-fast: they protect one
run's mutable checkpoints and journals, not resources shared between arms.

Both RTD and direct `bfcl_std_campaign.sh` calls use the same tag and port lock
inodes through `tools/bfcl_campaign_lock.py`. Tag lock files live outside the
hashed output tree, at `results/bfcl_std/.locks/<SHA256(tag)>/.lock`. Port locks
retain the old `results/rtd_v1/eval-port-<port>/.lock` path, so new campaigns
respect leases held by a still-running C25g process. Lock files are never
unlinked. Inherited descriptors preserve ownership if a coordinator exits
before a campaign child does.

## Waiting and accounting

Acquisition waits until the lock is available or a monotonic deadline expires;
nonblocking `flock` attempts are the polling mechanism, not a fail-fast API.
The default bound is **21600 seconds (six hours)** per resource acquisition.
Contention logs immediately and every 60 seconds by default:

```text
waiting for evaluation lock held by <pid>/<tag> (<lock path>; waited ...s)
```

Owner metadata also includes the hostname. For old empty lock files, the code
attempts to find the owner PID in `/proc/locks` and labels the tag
`legacy/unknown`.

`run`, `resume`, and `evaluate` accept:

```text
--evaluation-lock-timeout SECONDS
--evaluation-lock-log-interval SECONDS
```

Environment defaults, also honored by the standalone shell campaign:

```bash
export RTD_EVALUATION_LOCK_TIMEOUT_SECONDS=21600
export RTD_EVALUATION_LOCK_LOG_INTERVAL_SECONDS=60
```

These are operational options, outside the saved scientific configuration.
Timeout and interval values must be finite; timeout may be zero, interval must
be positive. RTD appends periodic `evaluation_lock_wait` records to its
hash-chained `compute.jsonl`, including on timeout/interruption. Records contain
`accounting: "idle"`, `idle_seconds`, lock/holder/status/round/tag, and zero
`gpu_seconds` and `gpu_reserved_seconds`. The campaign timer begins only after
the locks are acquired. Reports expose cumulative
`evaluation_lock_idle_seconds`, including failed attempts.

## Ports and cleanup

Omitting `--port` (or using 0) selects a preferred port from `SLURM_JOB_ID` on
SLURM, otherwise from the GPU UUID. Physical GPU ordinals without a UUID use a
kernel-selected free port. A collision on an automatic choice selects another
free port under a lease; it never attaches to the existing service. An explicit
port waits for another campaign's lease, then fails if an unrelated service
still occupies the port. Direct shell callers can pass `auto` or 0.

Every shell invocation creates fresh result/score directories with exclusive
`mkdir`, even if a port was used previously. Only completeness retries *inside
that invocation* reuse its generation files and memory snapshots. Legacy
`result_p<port>` directories and other attempts are never cleaned or reused.
`BFCL_PROJECT_ROOT` is fixed to this checkout so relative paths cannot resolve
against another run's environment override. Because BFCL loads `.env` with
`override=True`, the wrapper rejects `.env` entries that override the leased
port/GPU, project root, or ownership fields, and pins the local endpoint to
`127.0.0.1`. Listener cleanup additionally
requires the process to carry this invocation's `BFCLSTD_RUN_ID`; matching a
port or the text `vllm` alone no longer authorizes cleanup.

The free-port socket closes before vLLM binds; cooperating campaigns are
protected across that handoff by the port lease. An unrelated process that
ignores these leases can still race to bind, and is never adopted or killed.

## Validation and existing run identities

CPU tests exercise two real concurrent fake shell campaigns, overlapping
different-tag generation, same-tag serialization, same-port wait/timeout,
periodic PID/tag logs, old lock compatibility, inherited descriptors, new
namespaces on a reused port, occupied-port rejection, and RTD idle accounting
on both timeout and successful acquisition. The managed test sandbox prohibits
local sockets, so **only socket probes are simulated** using test fixtures;
flocks, file descriptors, subprocesses, copies, retries, and journals are real.
Production has no socket-probe bypass. The full RTD/campaign suite passed
**216 CPU tests** before the final `.env` guard; the final guard and reporting
changes were checked again with the focused regression suite. No GPU jobs
were launched.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m pytest -q tests/test_rtd_*.py tests/test_bfcl_std_campaign.py
```

C25h changes the pinned campaign file and adds the wrapper and lock helper to
the evaluation harness inventory. Existing frozen C25g identities therefore
**require an explicit audited identity update before deployment/resume**.
The existing `audit-legacy` content-preserving C25g migration is not an approval
for changed C25h tool contents. This patch preserves the content guard and does
not relabel saved scores or alter run manifests, checkpoints, or legacy
supplements. An already-running old campaign can also reject changed harness
contents at its post-campaign guard; its generated artifacts must not be
silently admitted under a different identity.
