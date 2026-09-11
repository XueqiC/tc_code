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
