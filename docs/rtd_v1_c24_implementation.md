# RTD v1 C24 implementation contracts

This implements governing spec equations (3)–(9) and section 13 T3/T4. It is a
CPU-tested implementation, not a completed BFCL training experiment. No GPU,
teacher request or new evaluation job was launched. Section 14's 4 feedback
parents × 2 rollouts remains frozen; the execution plan's 8×4 adjustment remains
unadopted, as in C23. Existing unrelated worktree changes were preserved.

## Public caps and historical evidence

`caps.py` separates `demo_attempt` from `generator_item`. `RequestLimits` computes
the product of configured max output, max actions and total attempts. Archived
`request_parameters` / `request_config` envelopes, including those inside
`metadata`, are the only record fields admitted. `max_retries` means additional
retries, whereas `max_attempts` includes the first. Conflicting settings fail.
Teacher-authored function schemas with a parameter named `max_tokens` are never
request configuration. Demo a1/a2/a3 are independently billed packages, not a
threefold multiplier on each package.

The actual audit found:

* `tools/bfcl_teacher_demos.sh` passes temperature and thread count to `bfcl
  generate`, with no output limit. Its `max_tokens=5` is a credential probe,
  not a demo request. The DeepSeek registry resolves to
  `OpenAICompletionsHandler._query_FC`, which supplies messages/model/temperature/
  store/tools but no `max_tokens` or `max_completion_tokens`. The rate-limit
  decorator has no finite stop. Demo records have usage/latency and sometimes
  inference history, but no request limit configuration.
* `tools/bfcl_generate.py::ask` supplies model/messages/temperature and **no
  max-output setting**. This is also true in its committed HEAD version. The
  three-attempt repair allowance does not provide a token cap. Generator rows
  lack provider request configuration and batch/repair boundaries. The nearby
  `bfcl_gen_loop.py` max_tokens=512 is another student-generation path; the OSS
  handler's 4096 limit is another local-model path. Neither bounds these teacher
  archives.

Consequently both classes use the existing **131,072-token class-uniform public
fallback**. This is a declared conservative replay convention, not a recovered
provider limit or an online billing guarantee. The cap was not lowered merely
to make the budget points affordable. Each public request includes JSON cap
provenance with class, basis, configured limits or null, and audited source
paths/content hashes. The rebuilt public manifest SHA256 is
`b12b241d4662ffb4e1466b98d9a2ad6dd319a4b1e2f70a5b2b576b248112dabe`.

The cost, eligibility and budget table are in [status](rtd_v1_status.md).
`sealed/audit.json` records the same affordability calculation. Individual
affordability at empty ownership is not a claim that all those packages can be
bought together. Dependencies remain locked and both fold counts are recorded.

## One actual update and frozen statistics

`functional_step.py` reuses `behavior.deltas` for ordered LoRA layout and exact
tensor content hashes. `lora_parameters` rejects any unfrozen non-LoRA weight.
`snapshot` creates independent differentiable trainable tensors;
`commit_step` checks the saved start hash before copying exactly one resulting
update. No Adam moments, clipping, decay, or second reference update intervene.
The explicit no-op branch does not evaluate a loss/gradient or mutate parameters.

`rms_diagonal` consumes an iterator of `(inner_parent_hash, source_rollout_gradient)`
and accumulates coordinate second moments without storing all gradients. With
`r_j = sqrt(mean_n g_nj²)`, it uses `raw_P_j = 1/(r_j + .01*mean_j r_j)` and
`P_j = raw_P_j / mean_j raw_P_j`. Means weight coordinates, including both LoRA
factors, rather than equally weighting parameter tensors. An entirely zero
second moment uses identity and records that fallback. This is the same ordered
diagonal geometry used by the existing tools; it does not reuse the old Fisher
whitening as the new optimizer. All input gradients/statistics are detached.

`FrozenStep` clones/detaches P and converts eta to a detached scalar. It records
round identity and accepts no optimizer state. `kl_pilot` checks owned evidence,
inner evidence parents and inner source parents, computes a fixed alpha=.5
gradient once, tries the predeclared eta grid, and picks the trial nearest KL
**0.005**. It saves all candidate KLs, the choice, evidence IDs, and virtual
compute counts. The native backend's pilot KL sums conditional forward KL over
the full source action (including EOS), then averages slots. No reward/checker
is available to the pilot. A missing legal teacher pool defers calibration;
the runner must carry a previous eta or handle the first identity reference,
then freeze eta after the first legal purchase before evaluating derivatives.
No derivative call recalibrates it. The grid/observed mismatch remains auditable.

## Sampling, scoring, BFCL rewards and gate VJP

`TorchPolicyBackend` accepts an already-loaded eval-mode causal LM and tokenizer;
it loads neither and never starts a server. It uses raw categorical sampling at
temperature=1/top_p=1, with no top-k, repetition penalty or other logits warper.
The same torch functional model recomputes likelihood on the actual sampled
token IDs. Non-double logits use float32 log-softmax/CE on both paths; float64
is preserved for CPU checks. Prompt/observation tokens are excluded, action
tokens through first EOS are summed without length normalization. Action/context
limit violations raise `IncompleteRolloutError`; they are not silently truncated,
force-terminated with a fabricated EOS, dropped, or retried until short enough.

`source_sampler` takes an independent round-frozen parameter snapshot.
`initial_hidden` checks the separately frozen initial policy for projection.
`freeze_slot_targets` draws cached teachers uniformly once and `loss_on_slots`
scores the fixed source/teacher texts at the **current block theta**, preserving
positive mixture weights. Feature tensors are explicitly detached. Source rollout
gradients for P are unrewarded and checked against the inner parents.

`bfcl_task_rollout` uses the existing BFCL Qwen handler from a fresh task copy,
replacing only its model-query callback. A single-turn generation is a complete
task rollout; official AST checking supplies the reward. BFCL relevance categories
use the official relevance evaluator. Multi-turn execution follows the existing
handler observation loop and `_evaluate_single_multi_turn_entry`. The existing
checker bridge now supports these official evaluators, including its BFCL-venv
worker fallback; failures remain errors, never reward zero. Unique simulator
namespaces and cleanup prevent BFCL's global simulator cache from carrying state
between same-task rollouts. Teacher prefixes never seed feedback rollouts.

Memory/web-search agentic categories still require their separate initial-state
and official agentic-checker adapter and are rejected explicitly; this does not
unlock their unavailable historical teacher packages. Multi-turn CPU tests use
synthetic tasks and the real official simulator/evaluator. No certification tasks
are used. There is no tested vLLM-generation/torch-scoring substitution.

`reinforce_gradient` computes same-task leave-one-out baselines, detaches rewards,
baselines and action traces, and accumulates equation (4) one action graph at a
time with **1/N rollouts**, not 1/tokens or 1/actions. It checks generation versus
recomputed log probabilities and saves maximum discrepancy, tokens, rewards,
baselines, backend and model identities. All-zero advantages are reported as
unidentifiable and yield zero return gradient. An action-independent score
baseline would require an explicitly different protocol.

`gate_vjp` differentiates the scalar contraction of the inner loss gradient with
`stop(P*gJ)`. It does not construct a demonstrations-by-parameters matrix.
`actual_gate_vjp` also checks the actual updated model hash. Source samples,
normalization, P, eta and gJ are explicitly stopped. `GateController` performs
ascent and ridge shrinkage to phi=0; its RMS scale uses only past feedback and
records the numerical-epsilon fallback. The new phi is for the next block.

## Insertion and acquisition

`InsertionReference` saves theta and computes the virtual old-data update using
one fixed eight-slot list. It requires reference-model gJ, evaluates the new
eight-slot loss at the saved theta, and returns the actual `.75*gD+.25*gq` update
at that same start. The label is `-eta*<gJ_ref,P*(gq-gD)>`. Empty returns the
reference update with exactly zero value/cost and consumes one decision window.
Two lists cost 16 raw slots with weighted exposure 6+2 when old evidence exists;
identity slots never claim supervised exposure. Raw tokens remain runner accounting.

Labels are typed `pending_new` or `reweight_existing`, with request ID, round,
start and reference hashes. One request cannot be replicated into event-level
observations. `ValuePosterior` fits only pending-new labels; old-package labels
are kept in a separate diagnostic store. Each round resets the value posterior
to precision=1/noise variance=1. Historical labels are rejected, rather than
silently reused without progress/staleness reporting.

Acquisition features contain only the pre-purchase projection, student logprob/
length, L, owned coverage, training progress, support return summary and intercept.
Missing model features raise rather than inventing zeros. Bayesian linear
regression samples one beta per decision from its full posterior; fixed-noise
metadata remains inspectable. The cost regressor uses public caps initially and
only revealed usage thereafter, retaining exact/estimated flags. It fits cost/cap
residuals around 1, with fixed noise 1 for exact and 4 for estimated observations;
these are expected costs, never hard reservation amounts.

`budget_distribution` assigns empty prior .5 and distributes the remaining .5
uniformly over deduplicated real requests. It normalizes values by the common
RMS prior predictive standard deviation and costs by the mean legal nonempty
cost, recording both scales. A one-dimensional dual bracket/bisection enforces
expected cost <= remaining budget / remaining windows at tau=1. Zero budget uses
the analytic zero-cost support limit. Negative values remain possible choices.
`AcquisitionPolicy` samples once, including empty, with no free redraw. The R0
control uses zero values and the same prior/dual. `select_and_acquire` passes only
public values through the C23 guard, then calls the broker; the ledger reserves
the full cap before reveal regardless of the predicted cost. It does not itself
fit a posterior: the runner reveals, measures a pending-new label, and then fits.

## Remaining integration boundary

C25 owns the three-round schedule, deduplicated inner state/slot sampling,
initial/source snapshot persistence, past-feedback timing, first-purchase pilot
sequencing, transactional pending/owned/checkpoint recovery, full compute ledger,
GPU smoke/production checks and official complete endpoint evaluation. The CPU
tests prove local numerical and data-flow properties, not training performance.
