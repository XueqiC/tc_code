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
prompts together, then steps the environments in parallel RPC threads.
Completed episodes drop out. Policy generation, per-episode RNG bookkeeping,
state validation, rendering and journal writes remain on the caller thread.
No model threads, teacher collector, API calls or persistent worker pool are
needed. All workers close on success, retry, contract error or cancellation.

Set `BFAS_FEEDBACK_LOCKSTEP_EPISODES=32` to run all 32 episodes of a selected
feedback block concurrently. This positive integer defaults to
`meta_tasks_per_feedback * rollouts_per_meta_task` (eight in the current plan).
It controls the live episode limit K and physical feedback batch size independently
of task selection and `generation_batch.prompts_per_batch`. At most
`min(K, selected episode count)` CPU environment subprocesses are live, each
with its own temporary working directory and the existing CPU thread limits.
Larger blocks execute in cohorts of up to K episodes; workers and RPC threads
close when each cohort ends, including retries and failure cleanup.

Within each lockstep round, physical generate calls split the live prompts to
respect `generation_batch.max_batch_tokens`, including padding and action caps.
Differing effective action limits also require separate calls. The token budget
does not reduce the number of live workers. D12's oversized-single-prompt rule
remains in force. Task-start logical RNG groups still use the original generation
configuration; a group straddling a K-episode boundary is sampled intact and its
unused first actions wait for the next cohort.

Continuation sub-batches dispatch their environment steps as soon as generation
returns, overlapping those RPCs with generation of later sub-batches. After all
steps and any cached retry replay return, the next lockstep round starts without
another queue or polling delay. No episode samples its next action before its
observation is available. Task-start generation precedes worker reset as before.

Set `BFAS_FEEDBACK_LOCKSTEP=0` to select the original serial feedback driver.
The selector, selected tasks, number of rollouts and LOO estimator are unchanged.
In particular, legacy `RTDExperiment.choose_feedback_tasks` uses fixed counts
(ALFWorld: up to eight parents, four rollouts each), whereas the unified selector
uses config counts. The Luna config's default concurrency is eight; an already
selected 32-episode block still executes all 32 episodes, with the environment
override allowing all of them to share the same lockstep rounds.

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

## Lockstep greedy window diagnostics

The D16 path is `AlphaDExperimentMixin.alpha_actual` →
`experiment_metrics_v11.window_metrics` → `metrics_v11.greedy_success` →
`GreedyBackend`. Previously each unique fixed parent ran to completion through
`interactive_diagnostics.alfworld_greedy`, with one HF generate call per action
under `r*/s*/v11_window_metrics` / `diagnostic_greedy_generation`.

With `generation_batch` enabled, ALFWorld now sends those unique parents through
`ALFWorldExperimentSupport.diagnostic_batch`. Diagnostics and feedback share
`generation_batch.run_episode_streams_lockstep`: live prompts generate together,
each returned sub-batch immediately dispatches its environment steps, and
completed episodes leave the next sampling barrier. Diagnostic reset and step
RPCs run in parallel threads; generation, rendering and cleanup run on the caller.
Workers and threads close before a cohort returns or an exception propagates.

`BFAS_DIAGNOSTIC_LOCKSTEP_EPISODES` sets the positive live-episode limit, falling
back to `BFAS_FEEDBACK_LOCKSTEP_EPISODES`, then **32**. This overrides
`generation_batch.prompts_per_batch` for diagnostics only. D12 length grouping,
equal effective action caps, left padding and `generation_batch.max_batch_tokens`
still bound physical calls; an oversized single prompt runs alone. Use
`BFAS_DIAGNOSTIC_LOCKSTEP=0` for the original serial driver. Backends without
generation batching or a diagnostic batch provider retain serial execution.

HF uses `do_sample=False`, one beam and argmax (recorded temperature 0), retaining
the serial stop tokens, likelihoods, decoded text, truncation flags, policy IDs
and diagnostic metadata. No sampling tickets, episode seed draws or RNG updates
are introduced. The 40 fixed slots retain their order and repeated-parent cache;
each unique parent is evaluated once at each checkpoint. The original diagnostic
40-action horizon, binary reward and exception behavior remain in force, without
adding feedback retries, fold restrictions or feedback episode journal records.
Only physical compute records change: `diagnostic_greedy_generation` now records
`sequences`, prompt-token count and padded prompt-token count for every call.

CPU fake-policy/stub-environment tests assert exact serial/lockstep `TaskRollout`
records, including every token and likelihood, early EOS/native turn stops,
episode early termination, horizon exhaustion, fixed-slot success/repair/damage,
and unchanged RNG states. They also exercise bounded cohorts, token budgets,
context caps, parallel reset/step RPCs and failure cleanup. Production bf16
batch-shape drift has the same qualification as D12; these CPU checks do not
claim bitwise GPU equivalence or measured production speedup.

## Generation-stage audit

Audited direct `sample_action`, `sample_actions`, `prefetch_actions`, HF
`generate` and `sequences=1` producers in `src/bfas/rtd`. A call with
`num_return_sequences=1` can still contain many prompt rows; it is not evidence
of serial execution. Token-budget limits, distinct effective caps or the last
live episode can also produce a legitimate singleton physical call.

| Stage / producer | Current execution | Converted here? |
| --- | --- | --- |
| D16 window before/full/control greedy success (`metrics_v11.greedy_success`) | Shared lockstep episode driver and batched argmax, default K=32 | Yes, for ALFWorld with generation batching |
| Offline controls calling the same greedy-success helper (`controls_v11.run_controls`) | Uses the same ALFWorld diagnostic dispatch when available | Yes, through the shared helper |
| `alpha_d` virtual-reference, same-batch-reference, post-commit feedback (`AlphaDExperimentMixin.alpha_feedback`, `RTDExperiment.feedback`) | Already uses `feedback_rollout_tasks` and ALFWorld lockstep; each virtual checkpoint remains a separate policy evaluation | Already batched; scheduler extracted for reuse |
| Paired validation and z-direction/acquisition validation probes (`alpha_validation_return`) | Already uses ALFWorld lockstep feedback on its isolated stream | Already batched; scheduler extracted for reuse |
| Commit and virtual-reference source pairs (`alpha_draw_pairs`) | Already prefetches two draws per state across requests through D12 batches | No change |
| Source/feature acquisition pool and pending-purchase states (`RTDExperiment.pool`, `sample_state`) | Iterates states; draws for one state batch together, normally two, with no cross-state prefetch at this entry point | No change |
| Acquisition candidate values (`joint_surrogate.marginal_values`) | Uses existing gradients/statistics; no per-candidate generation or environment probe | No conversion needed |
| Round-end parse battery (`metrics_v11.parse_battery`) | Already prefetches four draws per fixed state across requests | No change |
| Round-end variance and offline source-estimator diagnostics (`controls_v11.sampler`, `metrics_v11.estimator_batches`) | Calls `sample_action` once per draw: singleton HF batches, eight draws per state per resample | Remains serial |
| Legacy baseline source collection (`baselines/runner.py`) | Two separate `sample_action` calls per state | Remains serial |
| Round-end official ALFWorld v1.1/unified evaluation (`registry.alfworld_evaluate` → `webshop_evaluation.evaluate_adapter`) | Separate adapter campaign, concurrent episode threads making individual requests to the vLLM serving lane; outside in-process HF generation | No change |
| Legacy ALFWorld official evaluation (`alfworld_evaluation.official_episode` / `HFBackend.generate`) | Separate campaign with singleton local HF generation | Remains serial |
| BFCL/WebShop continuation and diagnostic providers; explicit serial fallback (`return_gradient.bfcl_task_rollout`, `webshop_rollout`, `GreedyBackend.sample_action`) | Individual continuation/diagnostic actions; BFCL feedback starts can already batch | No change |

This change requires a newly started process to use the edited modules. Existing
ALFWorld V0/D3 and BFCL V0 processes were not restarted or modified; production
results and Luna configurations were not edited.

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
