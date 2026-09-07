# RTD protocol v1.0.2 — C25j scoring identity (2026-09-07)

This addendum scopes evaluation provenance. It preserves the v1.0.1 scientific
configuration, budgets, training method and immutable run manifests below.
Existing saved `config.protocol_version: 1.0.1` remains valid; resuming a run
must not rewrite its config or checkpoint bindings to rename the protocol.

The scoring content identity is `bfcl-evaluation-harness-scoring-v3`. The
**exact source inventory** is the following table. Mixed files use the explicit
source selectors in `src/bfas/rtd/scoring_scope.py`; entire files are not hashed
when they also contain campaign coordination or training code.

| File(s), relative to the repository | Included scoring content |
| --- | --- |
| `envs/bfcl/gorilla/berkeley-function-call-leaderboard/bfcl_eval/**` | All package source and resources, with relative names, sizes and SHA-256s. `bfcl_eval/data/**` has its own complete manifest. Exclude `.git`, `__pycache__`, `.pyc`, `.pyo`. The current inventory is 112 package files plus 71 data files. |
| `tools/behavior_atom/checker_bridge.py` | Whole checker bridge and wire/verdict contract. |
| `tools/bfcl_event_mine_single.py` | `language_for`, the bridge's language dispatch dependency. Event mining, CLI and teacher-data plumbing are metadata. |
| `tools/bfcl_std_campaign.sh` | `MODEL_DIR`; `generate_args` and its local-weight append; both `VLLM_USE_FLASHINFER_SAMPLER` assignments; both BFCL `generate` command argument lines; the BFCL `evaluate` invocation; the `overall=$(python3 -c ...)` CSV aggregate expression and CSV input. Exclude `--num-gpus`, `--num-threads`, `--gpu-memory-utilization`, and `--result-dir` from the generation argument array. Handler, backend, temperature, category selection and evaluated local weights remain pinned. |
| `tools/bfcl_generation_check.py` | `check_generation`, including official category expansion, FC exclusion, missing/duplicate/unexpected ID checks and incomplete-category output. Diagnostic `print(..., file=sys.stderr)` calls are excluded. |
| `tools/bfcl_hub_merge_export.py` | Whole Hub-faithful tensor overlay, shard export, tokenizer/resource copying and verification helper. |
| `src/bfas/rtd/evaluation.py` | `official_expectations`, `validate_evaluation`, `_flatten_adapter`; the argument list passed to the Hub merge subprocess; the `data_overall.csv` read and finite/range check in `evaluate`. Only `device_map` and `torch_device` keywords on `_flatten_adapter`'s `from_pretrained` calls are excluded as device placement. Dtype, base/adapter selection, merge options and tokenizer export remain pinned. |
| `src/bfas/adapters/bfcl.py` | `BFCLScoreError`, `_score_category`, `_generated_by_category`, `read_score_summaries`, `extract_verdicts`: complete-summary reconciliation and per-task verdict aggregation used by evaluation. |

Python projections hash canonical ASTs (without locations, comments or
docstrings), including module imports referenced by the selected source.
Shell selectors require their exact expected multiplicities and hash selected
text after trimming line-edge whitespace. Missing/ambiguous selectors fail
closed. Changes to these selectors require a protocol review: they are an
explicit boundary, not automatic semantic equivalence detection. The two whole
helpers are conservative scoring units; their internal edits remain guarded.

C25m adds one explicit CSV representation equivalence: the exact
`_parse_percentage` helper that strips surrounding whitespace and a trailing
`%`, returns `None` for `N/A`, and otherwise calls `float` projects to the
historical `float` reader. Its nullable finite/range guard projects to the
historical numeric guard. The helper's complete AST must match the reviewed
template; other implementations remain scoring content. Column selection,
numeric scale, finite checks and bounds remain guarded, preserving the frozen
scoring identity while admitting BFCL's real CSV format.

The **effective evaluation identity** also binds the saved `evaluation_*`
configuration, original `base_checkpoint_hash`, `tokenizer_hash`, and the
round checkpoint's adapter/round-state hashes. Base/tokenizer content lives in
the immutable manifest, is verified from local bytes, and is explicitly retained
in the C25j supplement and endpoint identity. It is not identified by an absolute
installation path or model alias alone. Checkpoint, config, data and hardware
guards continue independently of the scoring-source hash.

Exclude evaluation locks (`src/bfas/rtd/evaluation_lock.py`,
`tools/bfcl_campaign_lock.*`), port/UUID selection, executable discovery,
launchers, SLURM scripts, cache directories, output namespaces, retry/resume
coordination, cleanup, campaign logs and report formatting from the hard scoring
guard. BFCL package build files (`pyproject.toml`, `setup.py`, `setup.cfg`,
`requirements.txt`), git HEAD and local virtualenv wrappers are also outside it.
These operational edits do not change the scoring function or tensor overlay.
They must not strand an intact trained checkpoint at its evaluation boundary.
Changes to actual BFCL package code/data, checker semantics, sampling, merge
math or aggregation must still invalidate evaluation.

C25e's code-drift policy remains: `rtd_source` records full content hashes for
all `.py`, `.sh`, `.slurm` files under `src/`, `tools/`, `scripts/`, plus the four
BFCL build files when present (`rtd-source-v2`). Python caches are excluded.
Changes are recorded in `code_drift.jsonl` during evaluation/resume; continued
training requires `--acknowledge-code-drift`. That acknowledgment cannot bypass
a scoring mismatch. Optional BFCL git HEAD remains separate metadata. Identity
construction requires neither git nor an installed BFCL executable.

## C25j audited update for parked runs

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python tools/rtd_experiment.py update-identity --run-dir results/rtd_v1/rai_R0
CUDA_VISIBLE_DEVICES='' .venv/bin/python tools/rtd_experiment.py update-identity --run-dir results/rtd_v1/rai_R1
```

The command reads the saved config, reports every changed member of the **old**
recoverable inventory with old/new SHA-256s and hash kind, and checks every member
of the **new** scoring inventory. It refuses changed saved scoring hashes even
with backdated mtimes; package/data additions and deletions also fail. For whole
files, integer-nanosecond mtimes must strictly predate the manifest (equality
fails). On a copied run, earlier manifest mtimes retained in an existing audit
remain the stricter cutoff.

For newer mixed files, accept only an unchanged manifest-bound scoring projection
or equality with actual reviewed source recorded before the manifest. An old
whole-file binding must match those archived bytes before projecting it. The
portable reviewed bundle `configs/rtd/identity_evidence/c25j.json` retains the
pre-run campaign/export source and successful source-edit record timestamps and
hashes. Its own SHA-256 is pinned in `identity_update.py`. The archived campaign's
full hash independently matches both rai supplements. The archived export source
has timestamp corroboration, because the original mixed manifest did not record
its independent hash. Mtimes and source-history timestamps are corroboration,
not cryptographic proof of historical contents; unavailable old inventories are
explicitly labeled unavailable, never invented.

The CPU audit verifies base/tokenizer bytes and every present round checkpoint,
checks for manifest, supplement, scoring inventory, model and checkpoint changes
during hashing, and writes only
`configs/rtd/legacy_identities/<full-original-manifest-hash>.json`. It preserves
the original legacy audit, C25g migrations and source identity, and appends an
`identity_updates` record with old/new identities, exact file differences,
per-file evidence, model/checkpoint checks and current source metadata. Split
manifests and legacy manifests use the same full-manifest-bound lookup. Repeating
an unchanged successful update does not rewrite the supplement. Refusal emits
JSON evidence and a nonzero CLI status.

After syncing this code **and the reviewed evidence bundle**, the exact HPG
command, from that checkout's repository root, is:

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python tools/rtd_experiment.py update-identity --run-dir results/rtd_v1/R0
```

This command does not resume training or launch evaluation. A process already
running old code cannot acquire the new scoping rules in memory. Use a fresh
process for subsequent resume/evaluation. Old completed scores and generated
artifacts keep their original identities; the supplement never relabels or
adopts them. Manifests, recovery journals and round-1 checkpoints stay intact.

# RTD protocol v1.0.1 — C25 decision (2026-09-06)

This addendum supersedes the C23/C24 cap, budget-denominator and feedback-dose
statements below. The original records remain as history. Authority: the project
lead's explicit C25 instruction, under governing spec §6.2 and §9.2.

* Demo attempt packages: **65,536 output tokens**, uniformly by public class.
* Generator item packages: **8,192 output tokens**, uniformly by public class.
* The cumulative round checkpoints authorize **10%, 25%, 50% of the sum of public
  caps over the usable bank**, flooring to integer tokens. Hidden usage is never
  used to set a cap or the budget denominator.
* Hard reservation checks the cap and unpaid dependencies before reveal. After
  reveal, deduct recorded actual usage, retaining **exact/estimated** flags.
  Release unused reservation. Reports place actual recorded spend on the x-axis
  and show cap-authorized ceilings alongside it.
* At each decision, **b = remaining authorized tokens / remaining windows in
  this round**. Empty consumes one window. b constrains expected cost; a package
  need not individually fit b if it fits the hard remaining budget.
* Feedback uses **8 parents × 4 full rollouts for single-turn BFCL**, and
  **4 × 2 for multi-turn**, limited to available feedback parents. Both arms use
  the same procedure. One single-turn rollout is one stochastic complete action
  scored by the official checker. Smoke alone uses one rollout and the
  action-independent zero baseline (LOO is undefined for one sample).

Rationale: archived demo calls have no max_tokens setting and no finite retry
bound; generator calls lack output settings and original batch/repair boundaries.
The project lead therefore selects the §6.2 class-uniform public conservative
replay convention, using the documented reasoning output maximum (64 Ki tokens)
and chat-completion maximum (8 Ki tokens). These are retrospective **package
conventions**, not recovered retry-wide provider billing guarantees. No output
length, hidden usage, success label or affordability target determined the caps.
The historical provider references are [reasoning model documentation](https://api-docs.deepseek.com/guides/reasoning_model)
and [chat API upgrade documentation](https://api-docs.deepseek.com/news/news0725/).
Provider model aliases/documentation can change; the numeric convention above is
frozen by this decision. Generator rejected drafts/repairs remain unrecorded and
their spend remains an explicitly incomplete estimate. No online calls authorized.

The usable 84 demos + 336 generator items have public-cap sum **8,257,536**.
Cumulative authorized ceilings are **825,753 / 2,064,384 / 4,128,768**.
Recorded bank usage **55,370 = 21,203 exact + 34,167 estimated** is reported
separately. Rebuild into `data/rtd/v1_bfcl_c25`; preserve C23/C24 bank directories.
`configs/rtd/v1_bfcl_c25.yaml` is the executable version; the original section-14
template remains in `v1_bfcl.yaml` for historical reproducibility.

**C25 executable conventions:** use one local HF model for KV-cached unwarped
generation and teacher-forced scoring (two source samples, T=1/top_p=1/top_k=0).
This avoids loading a separate vLLM copy for each virtual/actual parameter state;
GPU throughput has not been measured. Freeze initial projection, round sources,
train-only standardization and RMS P. Newly revealed states use the frozen
statistics. Stream complete-slot gradients and the exact frozen-sigmoid gate VJP.
The selected first legal package supplies the deferred .005 KL pilot; no teacher
payload is opened for calibration before selection. Save both raw old/new slot
counts and weighted exposure. Every round has a LoRA checkpoint and a separate
official full BFCL v4 endpoint via the existing campaign/Hub export conventions,
with explicit temperature 0 for RTD. Evaluation cannot influence the schedule,
loss or configuration. Resume validates checkpoint/config/data/hardware hashes;
reports never turn missing task IDs or missing score categories into passes.
See [C25 status and exact commands](rtd_v1_status.md) for implementation, evidence,
budget tables and the distinction between CPU validation and unrun GPU work.

# RTD v1 protocol — C23 / T0–T2, with C24 implementation addendum

**C24 addendum (2026-09-06):** The frozen method/configuration values below remain
unchanged. T3/T4 are now implemented and CPU tested; see
[current status](rtd_v1_status.md) and
[implementation contracts](rtd_v1_c24_implementation.md).
The active rebuild is `data/rtd/v1_bfcl_c24/`; the original C23 bank is preserved.
The blanket-cap interpretation below is superseded by the class-specific
configuration audit: both archived writer paths omit max-output settings, and
the demo rate-limit retry wrapper has no finite stop. Records have no request
configuration envelopes. Each class therefore retains the 131,072-token public
fallback, explicitly labelled unknown configuration. Affordable packages remain
0/0/0 at the three ceilings; no smaller bound was inferred from hidden usage.
Known archived metadata now takes precedence through
`max_output_tokens * max_actions * max_attempts`, with per-item generator scope.
Source/LoRA/KL/rollout interfaces described below as future are implemented by
C24; runner scheduling, persistence and production GPU validation remain C25.

The rest of this document is the original C23 record.

Frozen on 2026-09-06. The governing definitions are in
[RTD_END_TO_END_METHOD_AND_EXECUTION_V1.md](RTD_END_TO_END_METHOD_AND_EXECUTION_V1.md),
read in full before implementation. The subsequent
[execution plan](2026-09-06-rtd-v1-execution-plan-zh.md) supplies practical data
choices. Its increased BFCL feedback dose is not adopted: the section 14
configuration remains **4 feedback parents × 2 stochastic complete rollouts**.
All template keys and values are preserved in
[configs/rtd/v1_bfcl.yaml](../configs/rtd/v1_bfcl.yaml); a test checks them against
the governing document. Changes require a new version. C23 does no training,
GPU work, teacher requests, or new evaluation submissions.

## Resource and evidence access

This is **exploratory sealed_replay**, with text-only historical evidence,
training seed 0, fixed harness and identical hardware for future comparisons.
There is no pretrained cross-task controller. New teacher tokens and new
monetary spend in C23 are both zero. Historical cost remains nonzero.

[The data-access table](rtd_v1_data_access.md) records every selected source,
exclusion source and the broader historical generator account. Raw archives,
labels, request integrity hashes, usage, event correspondence, and aggregate
bank accounting belong to the ingestion/broker side. They must not be passed to
the selector. A future run manifest must additionally record all unlabeled task,
environment and generator access; support feedback counts; development,
confirmation and certification instances; student tokens, GPU time and rollouts.
C23 records zero feedback/environment interactions.

The source audit found discrepancies in the preliminary task description:

* `demos_ds.json` contains **34 merged verified examples**, not 120 raw responses.
  The raw attempt directories contain the **40 demand tasks × 3 attempts**,
  including failures. Those 120 records cost **70,318 output tokens exactly**.
* The archived **1,233,607 exact output tokens** cover all 255 raw records in the
  three attempt directories: a1 = 465,825, a2 = 344,800, a3 = 422,982. This includes
  memory prerequisites and stale earlier-split tasks. The additional 1,163,289
  tokens remain in historical accounting; they are not spread over the 120
  current attempts or silently discarded.
* **142,727 estimated generator tokens (~143k)** cover 18 archived generator
  files. The selected v3 and OOS files account for **33,452 + 10,613 = 44,065** of
  that estimate. The 302 v3t event rows are derived evidence, not 302 extra bills.

## Support parents and isolation

A parent is the distinct **official task content** behind a demand demo or
selected generator seed. Use `cc_pairs.digest` over the official entry excluding
`id`, `ground_truth`, and `possible_answer`; group by that hash. Generated task
IDs are provenance, not identities. Prompt hashes and
`cc_pairs.question_content_hash` identify generated states and authored items;
16 legacy generator IDs occur more than once in the selected files.

The measured union is **m = 40**, not 45: v3 has five seed parents, OOS has 28,
and all these seeds already belong to the 40 demand parents. Dependency requests
and generated variants do not increase m. The explicit 40-parent mapping is
[configs/rtd/v1_bfcl_support.json](../configs/rtd/v1_bfcl_support.json).

Two-fold assignment is `int(parent_hash, 16) % 2`: fold 0 has **22** parents,
fold 1 has **18**. Rounds 1/2/3 use inner folds 0/1/0 and feedback folds 1/0/1.
Before building anything for a round, set its fold roles. Candidates, pending
packages, teacher supervision, source replay, feature statistics and the future
preconditioner use only that inner fold. Ownership persists across rounds;
feedback after rotation is **meta-training**, never untouched validation.
With no legal request, the decision is continue training; do not borrow from the
feedback fold.

All `calibration_ids*.json` exclusions, heldout sections of
`configs/support_split.json` and `configs/bfcl_support_split.json`, and available
registered probe content are applied through the existing `cc_pairs.Exclusions`
implementation. Content takes priority over reused generator IDs. Official heldout
parents remain excluded. The official 65-task calibration set and the 60 archived
generated calibration prompt hashes **never enter training, gradients,
acquisition, feature fitting, preconditioning or hyperparameter selection**.
Previously used development data retain their development label.

## Packages, sealed files and costs

A demo attempt is one package identified by `(attempt, official task id)`, with
its exact nested input/output usage and full raw response retained. Attempts are
independent unless the original request has prefix prerequisites: attempt 2 does
not acquire an invented dependency on attempt 1. Memory `depends_on` chains retain
same-attempt request IDs and costs. An absent prerequisite is unavailable, not
free. Reusing a purchased response for multiple events or source slots never
charges it again.

The plan's generator unit is one archived authored item. Its opaque package key
includes file position and a content hash; a reused `gen_*` ID never merges two
items. A generator payload contains the authored question and ground truth,
the deterministic action rendering, and matched v3t rows. Its pre-purchase state
is the official seed state: the generated question is itself purchased teacher
output and is not exposed as a candidate embedding. v3t aliases are matched by
complete prompt and response/content, without correctness, Delta-U or error-type
filtering. A serialization difference is preserved for audit without replacing
the independently archived authored action. Empty authored ground truth renders
an empty action plus termination, not invented teacher refusal prose.

**Request provenance limitation:** the old generator made batched writer calls
and could make repair/verification calls, but did not retain those provider
request boundaries or complete usage. The item package is the plan's explicit
historical approximation; it is not evidence of one recovered provider API call.
Per-item costs are marked **estimated**, allocating each archived file total
proportionally to UTF-8 bytes of stored teacher-authored fields, with deterministic
integer remainders. This differs from recovering provider usage and remains a
lower estimate that misses rejected drafts, repairs and verification. No provider
request IDs, raw generated prose, or unrecorded short/long variants are invented.
`L=1` for these single-action item replays does not claim that the generator made
one API call. A strict provider-request replay/online claim remains unavailable
until those histories exist.

The local ingestion creates `data/rtd/v1_bfcl/public/requests.json` and a separate
`data/rtd/v1_bfcl/sealed/` directory of JSON payloads, integrity information and
research accounting. There are no payload Python modules and no payload imports
from `bfas.rtd.selector`. The selector receives only immutable `PublicQuerySpec`:
opaque query ID, pre-purchase state hash, nullable frozen 32-dimensional projection,
student source log probability/length, L, public upper bound, cost confidence and
cap provenance. Missing model features remain **None**, never zero-valued
fabricated features. `select_public` runs the selector under a cooperative guard
that denies data reads, privileged imports, subprocesses and network access.
This guards the normal Python path; it is not a security sandbox against hostile
reflection/native code. A runner executing untrusted policies must use an OS
process/filesystem boundary and must not mount raw archives there.

`list_candidates(snapshot, owned_ids, remaining_budget)` checks authoritative
ownership, the inner fold, dependencies and public cap feasibility. A request
containing teacher prefix state is invisible until its dependencies are owned.
`acquire(query_id)` rechecks these constraints, reserves before opening a payload,
verifies integrity, then records one charge/reveal event with sequence and UTC
time. Raw usage is revealed only then. Duplicate acquisition returns the owned
package without a second charge. `assert_no_hidden_access(trace)` checks the
instrumented public/purchased accesses and reveal ordering; an uninstrumented
empty trace alone is not a proof of process isolation.

Output tokens are the replay budget currency. Input tokens, output tokens,
reasoning tokens and monetary cost are distinct usage fields; unknown fields are
null. A separately known reasoning component must not be added twice to a
provider's output total. `Ledger.reserve_chain` checks the sum of all unpaid
caps before reserving any of them. Dependencies settle first. Usage above a cap
is an accounting error, with no silent overspend. The future expected-cost
constraint in equation (9) cannot replace this hard ledger.

A uniform **131,072-output-token retrospective cap** is frozen independently of
individual hidden usage. It is a conservative declared replay convention, not
an archived provider cap and not authorization for online spend. No candidate
cap is reconstructed from the actual response length. With the present legal
bank this cap makes all three fractional budget points **purchase-infeasible**;
report replay-only rather than lowering caps after inspecting usage. A later
version needs defensible archived/provider caps to make these purchases feasible.

The current build retains 698 archival entries: 120 demand attempts, 443 generator
items, and 135 prerequisite/earlier-split records for accounting. Of these, 420
are structurally available before budgeting: 84 single-turn demo attempts plus
336 generator items. There are 36 unavailable stateful demand attempts, 135
out-of-support/prerequisite records and 107 protected generator items. Stateful
raw inference logs survive, but exact handler bridge-state reconstruction and
request caps are not yet validated; no flattened trajectory is a training target.
247 of 302 v3t rows attach to packages; 55 have no matching complete-state source
in the selected files and remain unavailable. They do not create free packages.

`B_bank = 55,370` recorded/estimated output tokens for the available subset:
21,203 exact demo tokens and 34,167 estimated generator tokens. The 10/25/50%
ceilings are 5,537 / floor(13,842.5) / 27,685 integer output tokens. Report actual
spend and the ceiling separately. Broader historical costs above remain in the
end-to-end account even when evidence is not eligible for this bank.

## Source transport and features

`FullState` stores immutable canonical task/environment JSON, ordered observation
and action history, the exact frozen-harness prompt, and parent hash. Source and
teacher records must agree on the full state and parent, not merely task ID.
Environment observations are context only. The BFCL adapter renders prompts;
the existing thinking-off preprocessing is applied once when constructing states.

`SamplingRequest`/`SourceSample`/`RandomSourceSampler` provide the trainer boundary:
frozen policy snapshot, two independent source actions per state by default,
`do_sample=True`, temperature 1, top_p 1, full token IDs through the first EOS,
and the log probability under that frozen policy. No model sampler is launched
by C23. The trainer must bind snapshot IDs to saved parameter/harness hashes,
verify that scores match the sampling policy, and keep source sampling fixed for
the round. `TransportSlot` checks teacher/source state equality and samples the
cached teacher texts uniformly. Missing teacher evidence defines identity.

`finite_transport_q` implements equation (1) over a declared **complete finite
universe**, without renormalizing candidate outputs:

```
q = p * (1 - a) + sum(p * a) * nu
```

Without evidence it returns p, at a=0 it is p, and at a=1 it is nu. Source mass
outside teacher support survives. This finite oracle is only for correctness tests.

`positive_mixture_loss` implements equation (2) from current-student per-sequence
log probabilities: `(1-a)*(-logp_source) + a*(-logp_teacher)`. It retains positive
weights, gate gradients, and per-slot mean reduction. It does not sample a
Bernoulli or divide a sequence likelihood by output length. Missing teacher slots
use a=0. `is_exact_noop` tells the trainer to skip **the entire optimizer step**
when no teacher evidence exists and current student equals the frozen source;
this includes skipping weight decay. Once the student changes, the identity CE
still preserves the source distribution. A sampled identity CE is not a substitute
for that exact initial no-op rule.

`score_behaviors` delegates inference likelihood to `behavior.margin.score_pairs`:
authored full response plus EOS, native CE, fp32 sum, no prompt truncation. The
tensor primitive `complete_sequence_logprob` provides the sum with gradients,
requiring one contiguous action mask and excluding prompt/observations/padding.
For sampled actions, C25d permits an explicit `truncated=True` flag: score every
actually sampled token and include EOS only if sampled. A configured cap is a
legitimate outcome, scored as-is by the official checker without retry, filtering,
forced termination or an assumed reward. `TorchPolicyBackend` and
`HFGenerateBackend` preserve the sampled token IDs for this training path.
Full prompts/states are never semantically truncated. The per-benchmark caps,
truncation accounting and memory controls are documented in `rtd_v1_status.md`.

`FrozenProjection` uses a CPU-local seeded Gaussian map of **initial-student**
hidden states into 32 dimensions and detaches them. Round source refreshes do not
refresh this initial representation. The raw 34 varying gate features are that
projection, source sequence log probability, and source length including EOS.
`FrozenStandardizer.fit` accepts only declared legal inner parents/states; it
records fit-state hashes and initial snapshot identity, uses population standard
deviation, and gives constant columns scale 1. `transform` appends an unscaled
constant 1. Thus chi has 35 entries. `linear_sigmoid_gate` computes
`sigmoid(chi.detach() @ phi)`, initially .5 at phi=0. No benchmark ID, function-name
one-hot, error label, checker result or unpurchased teacher feature is admitted.

## Frozen future schedule and reuse boundaries

Three rounds, each with 12 actual updates, 8 complete source slots per update,
and decision windows 1/4/7/10. Empty is an explicit continue-training choice with
prior mass .5, consumes a decision window and does not trigger a free re-draw.
At most one new package per window, at most 12 overall. Budget may remain unused.
Source policy, feature statistics and future preconditioner freeze each round.

For a purchase, the future trainer must compute old/new gradients from separate
fixed eight-slot lists at the same starting parameters, submit `.75*gD+.25*gq`,
and count the raw 16 slots and their tokens separately from the weighted eight
slots of exposure. No-op windows cannot claim supervised exposure. Later replay
samples inner tasks uniformly, then their available deduplicated full states;
new-package slots sample its states uniformly. Those training/update samplers
belong to T3/T5, not this pure target interface.

T3 reuses `behavior.deltas` for actual LoRA parameter layouts, content hashes,
restoration and deltas, and `behavior.whiten` for compatible blocked diagonal
geometry. Its train-only RMS preconditioner still needs the spec's damping and
mean-diagonal normalization; the old Fisher whitening is not silently equated to
that optimizer. `tools/behavior_atom/checker_bridge.py` remains the BFCL official
single-turn scoring bridge for T3 feedback; no checker is consulted for T1
eligibility or features. T3 must retain infrastructure failures as errors rather
than assigning reward zero. A/B/C checkpoints, configs and jobs remain separate.

## Existing A/B/C result intake

The local record (`PROJECT_STATE.md`, entries through 2026-09-07 01:10Z) says
training of A/B/C is complete, official eval 41241770 was running, heldout eval
41241772 failed in a merge-directory race, and its duplicate 41241773 was
cancelled. A read-only squeue/sacct attempt during C23 failed because this
workspace's network sandbox denied connecting to the HPG tunnel. Therefore this
is **recorded status, not freshly verified job status**. No complete new official
or heldout score artifacts were available locally to accept as final scores.
No job was submitted, cancelled or duplicated. The future result intake must
check expected task coverage and model/checkpoint hashes before reporting any
aggregate; historical A/B/C scores are not a prerequisite or a result of RTD.
