# HotpotQA-ReAct RTD bank and provider

HotpotQA is registered in the shared RTD runner for v1.1 V0/V1/V2 and unified
D0/D1/D2/D3 (including attribution arms). The existing WebShop registration is
preserved. The paper baseline bank reader needs no schema conversion.

## Frozen protocol

- Student: `google/gemma-4-12B-it`; teacher: `openai/gpt-5.6-luna`.
- `configs/hotpotqa_support_split.json` supplies **all 200 train questions** to
  RTD, including questions without a verified teacher demo. Each question has
  one public reset state and an explicit `int(parent_hash, 16) % 2` fold. The
  original BFAS 160/40 demand/calibration partition remains in the manifest as
  provenance; RTD uses the requested full 200-question support.
- Feedback always starts at a support reset in the opposite fold. It uses
  temperature 1, top-p 1, the full episode's terminal answer EM, and
  `FeedbackRNG` version 2 (run seed, round, step, feedback role, task ID and
  rollout index). Serial and lockstep execution use the same episode streams
  and retain every sampled token and its likelihood. ReAct format retries can
  recover within a step and do not force the episode reward to zero.
- The adapter and RTD provider share `bfas.hotpotqa.episode_stream`: seven
  environment steps, one action-only retry per step, at most 14 model calls,
  and 100 generated tokens per call. ReAct stop markers delimit environment
  input; complete sampled actions remain the units of RTD likelihood scoring.
- Every episode owns a separate Wikipedia cursor. Only immutable cache JSON
  files are shared. The lockstep driver batches ready model requests and runs
  cached tool work in its CPU executor. Cache misses abort incomplete feedback;
  they never become fabricated zero-reward measurements. Set
  `BFAS_FEEDBACK_LOCKSTEP=0` for the serial driver or
  `BFAS_FEEDBACK_LOCKSTEP_EPISODES` to bound concurrent episodes.
- Greedy support diagnostics use the same loop, temperature 0, both support
  folds when requested, and no stochastic feedback RNG draws.
- Round-end evaluation uses the adapter campaign on the fixed first **500 dev
  questions**, temperature 0. **Answer EM** is the headline; answer F1 is
  reported separately. F1 does not verify teacher demos. Campaign receipts
  bind the checkpoint, spend, hardware, data, tokenizer, prompt, evaluator and
  cached Wikipedia snapshots. The record validator recomputes EM/F1 and checks
  the complete ordered inventory and frozen gold/question identity.

The prompt is byte-identical to the sibling worktree's
`prompts/hotpotqa_react_6shot.txt`; SHA-256:
`e52ba17b32d98144c4ee3f95f3cd5a2fec2016629219427f8aa9ddcfbdcc3fdf`.
The accompanying source description and MIT license are included. See
[hotpotqa_setup.md](hotpotqa_setup.md) for the underlying adapter protocol.

## Configurations

`configs/rtd/v1_1_hotpotqa_luna.yaml`,
`configs/rtd/unified_hotpotqa_gemma4_luna.yaml`, and
`configs/rtd/unified_hotpotqa_gemma4_luna_d0.yaml` copy the corresponding
ALFWorld Luna frozen values: E=40 exposure slots, K=20 new packages per window,
two rounds, seed 0, 60 GB, two source samples per state,
and the same D15 score-tolerance block (`mean_abs=0.15`, `max_abs=2.0`,
`max_abs_outlier_tokens=4`, `max_abs_hard=8.0`). HotpotQA uses absolute cumulative
checkpoints of **15,000 / 30,000 output tokens**, plus its own benchmark paths,
support size, evaluation protocol and ReAct action horizons. The alias
`configs/rtd/unified_hotpotqa_gemma4_d0_luna.yaml` also supports the existing
ALFWorld filename ordering.

## Build and accounting

Run from this worktree, with the populated local tokenizer and HotpotQA
JSON/cache available:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES='' \
PYTHONPATH=src .venv/bin/python tools/rtd_bank_build.py \
  --benchmark hotpotqa \
  --pool /home/xueqi/hq/projects/tc-alignment-ws/envs/hotpotqa/teacher_pool_v2/demos.json \
  --ledger /home/xueqi/hq/projects/tc-alignment-ws/envs/hotpotqa/teacher_pool_v2/teacher_ledger.jsonl \
  --out data/rtd/v1_1_hotpotqa_luna
```

The default config for `--benchmark hotpotqa` is `v1_1_hotpotqa_luna.yaml`.
Builds refuse to overwrite an existing bank. Snapshot the pool locally and copy
its cache **after** the ledger when collecting concurrently. The builder reads
`demos.json`, the completed ledger prefix and any adjacent `usage.jsonl`,
`attempts.jsonl`, and `identity.json`. It records hashes and trailing partial-line
byte counts. A complete malformed JSONL record fails validation. The source
pool is never locked, imported into another ledger, or modified by this tool.

A collecting pool may publish `demos.json` only at worker shutdown. Successful
ledger rows already contain paid demo payloads: those are eligible after an
offline replay against cached Wikipedia, with every message context, target,
worked example and terminal EM checked. A pool demo must match a verified paid
attempt exactly; the collector may publish a later success. Missing attempts
archives are explicitly recorded as absent;
they are not reconstructed. The request journal is reconciled with completed
episode usage; in-flight requests outside that ledger prefix are disclosed but
are not falsely labeled completed episodes.

The layout is the same as the ALFWorld/BFCL v1.1 banks:

```text
public/requests.json
public/support.json
public/reset_states.json
public/cap_certificate.json
public/task_attempts.json
sealed/<query_id>.json
sealed/integrity.json
sealed/audit.json
sealed/audit_v11.json
```

There is **one package per task with recorded attempts**, with ordered
`behaviors[].state` and `behaviors[].text` from the **earliest verified attempt
index**. Every verified attempt is replayed, including later successes. The
selected original ledger row is retained in `historical_response`; all original
rows, sorted by attempt index, are retained in `historical_attempts`. Provenance
records the selected attempt, all charged indices and ledger row numbers,
teacher, ledger snapshot and student rendering. Empty replies in verified
episodes are rendered as explicit native
EOS targets. RTD's `teacher_tokens` already appends that same EOS to an empty
reply, so the supervised token sequence is unchanged and the paper reader gets
a nonempty target. Original empty replies stay in `historical_response`, and
`provenance.empty_targets_rendered_with_native_eos` records their turn indices.
Tasks with no verified attempt have empty behaviors and an explicit unavailable
reason. Their costs remain available to privileged historical accounting and
paper-baseline charging. The RTD broker charges failed-only tasks too; they produce no training rows.

Package costs sum the task's ledger `tokens_spent == usage.completion_tokens`
across **every attempt**, including failures, later successes, hidden reasoning
and collector estimates. Each attempt is charged exactly once. No target-length
estimate or per-state allocation replaces recorded cost. A package is `exact`
only when all its attempts are exact; any collector row marked `estimated`
makes the entire package cost confidence `estimated`. Estimated usage does not
invalidate a replay-verified demo. Missing/unknown tasks, negative token counts,
duplicate attempt identities and inconsistent usage or replay evidence still
fail validation; retry count and index gaps do not. Cached input tokens remain
a subset of prompt tokens and are not counted again as output tokens.

The usable recorded-output-token sum is the v1.1 budget denominator. The cap
certificate publishes the next-power-of-two usable task-cost maximum for the
`hotpotqa_demo_episode` class; it is an archived cached-content envelope, not an
online provider guarantee or the runtime reservation. Full historical teacher usage, including failed
attempts, is separately retained in the accounting block. No teacher calls or
new teacher tokens are incurred by building or preflight.

The completed `teacher_pool_v2` snapshot has **419 attempts: 113 verified,
306 failed**, covering all 200 support tasks. It produces **200 packages:
113 usable, 87 unavailable**. This snapshot has no duplicate successes;
synthetic tests exercise selection among multiple verified attempts.

| Accounting scope | Exact output tokens | Estimated output tokens | Total |
| --- | ---: | ---: | ---: |
| Usable packages, classified by whole-task confidence | 125,173 | 8,033 | **133,206** |
| All historical attempts, classified by ledger-row confidence | 507,176 | 28,390 | **535,566** |

There are **111 exact and two estimated usable packages**. Eight historical
attempts have estimated usage (two verified, six failed); all their recorded
costs are retained. Failed attempts account for **468,766** historical output
tokens, including **66,406** charged to tasks that eventually succeeded.
Historical prompt/cached tokens are 6,422,285 / 402,477. The bank's historical
cost confidence is `estimated`. `demos.json` has all 113 demos, and
`attempts.jsonl` covers all 419 attempts. The request journal reconciles with
the complete ledger; there are zero outside-prefix calls or trailing bytes.
All 113 successes were replayed against cached Wikipedia without model calls.

The usable denominator is **133,206**. Both V0 and D3 resolve their configured
cumulative caps to **15,000 / 30,000**, independent of the generic bank audit's
10%/25% reference amounts (13,321 / 33,302). The maximum usable task cost is
**8,889**, giving a uniform class reservation cap of **16,384**. Consequently,
the old broker could not reserve any package at 15k. This cap originates in
`bank_build.seal_v11`: `1 << max(0, (class_maximum - 1).bit_length())`.
`SealedReplayBroker.list_candidates` filtered against it, and `acquire` passed
it to `Ledger.reserve`. It is unrelated to `generation_batch.max_batch_tokens`,
which happens to also be 16,384, or the 100-token ReAct action limit.

## Task purchase accounting correction (2026-09-11)

Final requested CPU suite: **314 passed, 1 skipped**, 104.93 seconds.

The purchase unit is now **one task containing all recorded attempts**, including
failed attempts, later successes, and verified attempts excluded from training.
`public/task_attempts.json` records task ID, attempt index, ledger tokens,
verification, confidence, and archived query IDs; the certificate binds this
file. Builders publish it from the sealed ledger rows. The broker reads only
this public summary to price purchases and rechecks every member payload's
integrity and ledger summary on acquisition. Completion tokens include reasoning
exactly as the ledger records it; reasoning is never added again. Collector
estimates remain estimates. ALFWorld's legacy `estimated` confidence labels are
retained, even where its archived collector row contains reported usage.

The frozen order is **sorted task IDs, shuffled once with `random.Random(0)`**,
independent of training seed. A task is atomic: reserve and charge the sum of
all its attempt tokens, or stop before the first overflow. Never remove a
failed-only task or skip to a cheaper task. A failed-only or excluded task is
ledger-owned but supplies no training row. For tasks with usable evidence, the
earliest usable attempt supplies the trajectory (HotpotQA already selected the
earliest verified attempt). Other attempts are still charged. The representative
query ID remains an archived ID; member IDs cannot be bought separately.

V0 and unified acquisition preserve this order instead of sorting the selected
IDs or running a cost knapsack. The per-window package limit remains; task
purchases use the remaining cumulative authorization, avoiding artificial
per-window fractional quotas. Failed purchases stay in the durable ledger and
V0/D3 replay schedule, but receive no teacher exposure. Existing fold guards
remain: training restricts the one frozen order to the current legal inner
parents, without reshuffling. The audit table below buys across **all recorded
parents**, including unavailable/protected tasks, without training folds or
window quotas; it is not a predicted training schedule.

The three accounting indexes were rebuilt in place from their existing sealed
rows. Before replacement, the rebuild checked byte identity of **every existing
artifact except the requests/certificate/accounting metadata**. In particular,
all support manifests, task IDs, parents, folds, reset states, sealed payloads,
integrity indexes, and original audits stayed byte-identical. A failed comparison
aborts replacement. No raw pool, tokenizer, model, API, or GPU was needed.
Rebuild/audit command:

```bash
PYTHONPATH=src:. CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
/home/xueqi/hq/projects/tc-alignment/.venv/bin/python tools/rtd_task_purchase_audit.py \
  --bank data/rtd/v1_1_alfworld_luna --bank data/rtd/v1_1_bfcl_luna \
  --bank data/rtd/v1_1_hotpotqa_luna --rebuild-accounting \
  --out docs/rtd_task_purchase_validation.json
```

Omit `--rebuild-accounting` for a read-only audit. The default audit budgets are
7,500 / 15,000 / 30,000 / 60,000; they do **not** modify configured checkpoints.
ALFWorld retains the archived usable-attempt denominator **118,792** and its
configured fraction caps **11,879 / 29,698**. BFCL retains denominator **1,418**,
HotpotQA **133,206**; both keep absolute caps **15,000 / 30,000**. Denominators
are archival fraction bases, not the new task purchase prices.

| Bank | Budget | Tasks purchased | Usable packages | Attempts | Tokens charged |
| --- | ---: | ---: | ---: | ---: | ---: |
| alfworld | 7,500 | 0 | 0 | 0 | 0 |
| alfworld | 15,000 | 0 | 0 | 0 | 0 |
| alfworld | 30,000 | 1 | 0 | 3 | 21,457 |
| alfworld | 60,000 | 8 | 4 | 16 | 58,314 |
| bfcl | 7,500 | 9 | 4 | 19 | 2,860 |
| bfcl | 15,000 | 9 | 4 | 19 | 2,860 |
| bfcl | 30,000 | 9 | 4 | 19 | 2,860 |
| bfcl | 60,000 | 9 | 4 | 19 | 2,860 |
| hotpotqa | 7,500 | 5 | 3 | 9 | 6,690 |
| hotpotqa | 15,000 | 8 | 4 | 16 | 12,670 |
| hotpotqa | 30,000 | 11 | 5 | 23 | 20,944 |
| hotpotqa | 60,000 | 26 | 13 | 58 | 57,428 |

ALFWorld at 30,000 does **not** match the cited 5 usable / 29,229 baseline
receipt. Direct execution of the read-only sibling's `paper_data.load_purchased`
reproduced that receipt: it shuffles **233 archived attempt query IDs**, and its
first 13 purchases are 13 separate attempts from 13 tasks. It does not collect
all attempts for each of those tasks. The new task order shuffles **142 task
IDs**. Its first task is
`pick_and_place_simple-Cloth-None-Cart-401/trial_T20190909_054512_021256`:
three failed attempts cost **8,061 + 6,536 + 6,860 = 21,457**. The next task,
`pick_clean_then_place_in_recep-DishSponge-None-Drawer-427/trial_T20190909_095203_563442`,
costs **3,116 + 3,256 + 3,257 = 9,629**. Their sum **31,086** exceeds 30,000,
so only the first task is purchased, with **zero usable packages**. The same
prefix holds at the unchanged configured cap 29,698. This discrepancy comes
from the baseline reader's attempt unit and query-ID order, not rounding,
missing costs, reasoning subtraction, or support/fold changes. The read-only
baseline worktree and paper were not modified.

BFCL's tenth frozen task costs **80878** tokens, so it blocks all four reported budgets
after 2,860 tokens. Filling those budgets with later inexpensive tasks would
violate the prefix rule.

Full V0/D3 CPU preflight passed for ALFWorld and HotpotQA. BFCL full preflight
was run for both arms and failed at the existing local harness missing
`memory_kv_141-notetaker-11`; its V0/D3 acquisition-only preflights passed.
Machine-readable evidence: [task purchase audit](rtd_task_purchase_validation.json),
[exact baseline comparison](rtd_alfworld_task_baseline_comparison.json), and
[CPU validation](rtd_task_cpu_validation.json). Older recorded-cost receipts
below or in linked historical reports are superseded for purchase accounting.


`public/support.json` is byte-identical to
`_trash/v1_1_hotpotqa_luna_partial_09120141Z/public/support.json`, including all
200 IDs and folds. Its SHA-256 is
`164ee60ec28d12447b44a21334b36e86dd1e71b9121e7d28bcaefdc62dff3808`.

The pool remains read-only. `envs/hotpotqa/cache` links to the shared ws cache;
offline reads validate immutable, atomically published snapshots without
creating cache locks. Large runtime datasets/cache and the pre-existing
`.cache` remain untracked. The certified bank needs explicit staging with
`git add -f data/rtd/v1_1_hotpotqa_luna`.
Copy the same Wikipedia snapshots with the bank when running elsewhere. A
cache warmed only by this teacher pool may lack queries generated by students;
this offline protocol will report those missing snapshots explicitly.

## Paper baseline integration in the read-only base worktree

Inspected files in `/home/xueqi/hq/projects/tc-alignment-base` were left
unchanged. Its `src/bfas/rtd/baselines/paper_data.py::load_purchased` accepts this
bank's certificate, episode payloads, teacher IDs, dependencies, and native
Gemma prompt/target boundary without changes. The reader was exercised on the
archived partial bank at fractions 0.1, 0.25 and 1.0; that historical receipt
does not measure purchases from the completed task-package bank.

The entrypoint and execution hooks in that worktree still need these edits:

1. Add `hotpotqa` to `tools/baseline_run.py::arguments` benchmark choices; port
   this adapter/protocol/evaluator, registry/config changes and
   `configs/rtd/v1_1_hotpotqa_luna.yaml`. Add the HotpotQA prompt, split and
   evaluator files to its manifest's `source_files` identity list.
2. In `paper_train.py`, map HotpotQA rows to `agent_action` alongside ALFWorld;
   the current `else` derives a BFCL category from a task ID. Keep the 100-token
   HotpotQA action cap and native bank prompts unchanged. In
   `paper_losses.segment_spans`/`token_kinds`, add HotpotQA handling for numbered
   `Thought n:`/`Action n:` markers, standalone `search`/`lookup`/`finish`, and
   the inherited `Thought n:` prefill. The current marker regex only recognizes
   unnumbered labels; treating a whole ReAct completion as one action would
   misapply SmartAD weights and SAD's reason/action grouping. Use the row prompt
   to distinguish thought continuations from action-only retries; keep observed
   Wikipedia text masked and classify terminal `finish` as the final decision.
3. Add an explicit HotpotQA branch in `paper_evaluation.protocol` and
   `evaluate_run`; its current fallback routes all non-ALFWorld benchmarks to
   BFCL. Use `HotpotQAAdapter(seed=0, port=port, offline=True)` through
   `bfas.run.serving_lane` on the merged student export, then
   `adapter.evaluate(str(merged), out)`. Release the policy in `finally` and
   preserve the baseline's GPU-time and checkpoint receipts.
4. Validate the resulting `records.jsonl`/metrics with
   `bfas.rtd.benchmarks.hotpotqa_evaluation.validate_records`. Obtain the
   expected 500 IDs and question/gold hash from
   `hotpotqa_identity.evaluation_harness_identity(root, config)['expected']`.
   Return `overall_accuracy_percent=100*metrics['em']`, `overall_metric='em'`,
   and `em`, `f1`, `validation`, `expected`, and `tasks=500`. Evaluation must
   use temperature 0, seven steps, a 100-token cap, the frozen six-shot prompt
   and the same offline cache.
5. Kang's extra SAG branch requires a separate HotpotQA implementation if
   desired: vote over normalized final answers from complete sampled ReAct
   episodes. The ported official evaluator is greedy. Do not label a greedy
   rerun as SAG or route HotpotQA to the ALFWorld/BFCL SAG wrappers.

The paper reader's fixed seed-zero prefix purchases include unavailable packages
and stop before the first overflow. On **the archived partial snapshot**, both 10% and
25% purchase one failed package (1,005 tokens) and **zero positive rows**; even
100% of the usable denominator purchases only three failed packages (6,755
tokens). Those purchase numbers apply only to the archived bank. The completed
task-package bank is a new snapshot that all compared arms must share. Its
packages retain the full task cost, including failures; purchase receipts must
be recomputed against its certificate and the chosen budget configuration.

## CPU verification

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES='' \
/home/xueqi/hq/projects/tc-alignment/.venv/bin/python -m pytest -q tests/ \
  -k 'hotpotqa or acquisition or broker or preflight or caps'

PYTHONPATH=src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES='' \
.venv/bin/python tools/rtd_preflight.py \
  --config configs/rtd/v1_1_hotpotqa_luna.yaml --arm V0

PYTHONPATH=src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES='' \
.venv/bin/python tools/rtd_preflight.py \
  --config configs/rtd/unified_hotpotqa_gemma4_luna.yaml --arm D3
```

CPU tests cover tiny synthetic paid pools with all 200 support resets, failed
attempt costs, earliest-success selection despite later published demos,
estimated success/retry cost propagation, byte-identical support manifests,
stale demos snapshots, corruption rejection, native rendering,
serial/cohort trace equality under RNG v2, per-episode wiki cursors, fold guards,
cache failures, greedy diagnostics, synthetic V0/D3 preflight, and a stubbed
500-question round-end adapter campaign with separate EM/F1. Real-bank V0 and
D3 preflight both passed with 113 usable packages, 200 rendered states and zero
persistent ledger spend. Broker regression tests cover recorded reservations,
exact fits, stale offers, window/package limits, integrity checks, durable
resume, confidence preservation, and stopping before an overflow even when a
later cheaper package fits. Two stale BFCL assertions were updated from
2.5k/5k to the existing 15k/30k configuration. See
[rtd_hotpotqa_validation.json](rtd_hotpotqa_validation.json) for
the final certificate hash and CPU preflight receipts. No GPU, network or live
model/teacher API was used.

No commit was created. Staging status is recorded in the validation receipt.
