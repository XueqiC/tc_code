# ALFWorld feedback worker failures

ALFWorld adapter and RTD workers use these RPC limits, in seconds:

| Environment variable | Default | Scope |
| --- | ---: | --- |
| `BFAS_ALFWORLD_STEP_TIMEOUT` | 120 | One step request and response |
| `BFAS_ALFWORLD_RESET_TIMEOUT` | 300 | Worker startup, ready message and initial observation together |

Values must be positive finite numbers. They are read when a new worker is
created. Explicit Python `timeout` and `reset_timeout` arguments take precedence;
the legacy `timeout` argument applies to both operations unless `reset_timeout`
is also supplied.

Feedback no longer has a default whole-episode wall-clock deadline. Time spent
generating student actions therefore cannot exhaust an environment deadline.
Callers that explicitly pass `episode_timeout` still get that additional limit.
The 40-action horizon, 256-token action cap, task selection and return estimator
are unchanged.

On an RPC failure, feedback closes its owned worker and retries that episode
once with a new worker, the same reset request, policy seed and owned prefix.
Already sampled actions (including a batched first action) and their RNG
transitions are replayed exactly. Further actions continue the same sampling
stream. Only the completed attempt reaches feedback scoring. Terminal losses
are scored normally; policy and state contract errors are not retried. If the
retry also fails, `IncompleteFeedbackError` still excludes the incomplete
measurement without substituting zero or computing partial-task LOO.

Both attempts are retained as `alfworld_episode` journal records with `attempt`
0 or 1, linked by an `alfworld_episode_retry` event. Failures include the stage,
exception text, worker traceback when available, exit code, worker PID, the last
8 KiB of stderr, elapsed call/RPC seconds and configured timeouts. An exit code
of `null` means the process had not exited when diagnostics were captured (or
could not be spawned). Diagnostics are captured before temporary files are
removed and emitted to the run log at ERROR level.

## Lockstep feedback generation

ALFWorld feedback with `generation_batch` enabled now schedules the full
selected feedback block across parents. Each live episode owns a `RealStepper`
and its existing bounded CPU environment subprocess. The caller drives the
episode state machines to their next unsampled prompt, generates the live
prompts together, then steps each environment. Completed episodes drop out.
No model threads, teacher collector, API calls or persistent worker pool are
needed. All workers close on success, retry, contract error or cancellation.

The natural worker count is `meta_tasks_per_feedback * rollouts_per_meta_task`,
capped by `generation_batch.prompts_per_batch` and the reset-prompt token
budget. Each later generate call also respects the padded prompt plus action
token budget; differing effective action limits require separate calls. D12's
oversized-single-prompt rule remains in force. A legacy same-prompt task-start
RNG group stays intact even if it has more sequences than the worker cap (D12
counts distinct prompts); those episodes execute in bounded cohorts.

Set `BFAS_FEEDBACK_LOCKSTEP=0` to select the original serial feedback driver.
The selector, selected tasks, number of rollouts and LOO estimator are unchanged.
In particular, legacy `RTDExperiment.choose_feedback_tasks` uses fixed counts
(ALFWorld: up to eight parents, four rollouts each), whereas the unified selector
uses config counts. The Luna config's natural concurrency is eight; an already
selected 32-episode block still executes all 32 episodes.

Sampling preserves the existing task-start tickets, their original logical HF
batches, and the subsequent registry seed draws in parent-major order. Each
episode continuation retains its own generator and singleton D12 batch seed.
A thread-local `TorchFunctionMode` routes HF's `torch.multinomial` calls through
those original sampling generators while the model handles all live rows in
one physical batch. Retry replay consumes cached actions and restores their RNG
transitions without issuing new generation calls. No RNG, action or KV cache
survives the feedback scope.

Each attempt still produces `alfworld_episode`/`alfworld_episode_retry` records;
accepted `feedback_rollout` and `validation_rollout` records retain input order.
`generated_tokens` remains one entry per fresh action, while `compute_begin` and
`compute_end` describe actual physical generation calls. Physical batch sizes,
padding, timings, journal ordering and hashes can change. Per-action metadata
also records the original sampling batch size/width and its RNG end state.
`feedback_plan` records the actual selected episode count before sampling.
Teacher-forced scoring, score tolerances, failure enforcement and gradient
construction are unchanged.

CPU tests compare full stub-environment trajectories, seeds, rewards, token
scores, gradients and accounting, including early termination, the 40-step
horizon, reset/step retries and generation failures. Tiny real HF CPU tests
compare sampled tokens and RNG end states for fused starts and continuations.
This verifies RNG independence; GPU kernel rounding can still vary with batch
shape. GPU token equivalence and speed must be measured separately before
claiming bitwise production equivalence or a measured A100 speedup.

## Read-only progress

```bash
/home/xueqi/hq/projects/tc-alignment/.venv/bin/python tools/rtd_progress.py \
  results/rtd_unified/p1_alfworld_luna/V0 \
  --config configs/rtd/v1_1_alfworld_luna.yaml
```

The tool snapshots `compute.jsonl` without constructing a `ComputeJournal`,
loading checkpoints or repairing an unfinished final line. It reports the
latest feedback stage, accepted/completed episodes, fresh actions, completed
generate calls, batch sizes and retries. It prefers `feedback_plan`, then a
completed same-window measurement, over config-only workload estimates.
`subphase_memory` duplicates generation metadata and is never counted as a
second generate call. Future action/call totals depend on early termination;
the report labels horizon bounds and estimates explicitly. Generation ETA
excludes environment and scoring time. `--json` exposes the same counters.
