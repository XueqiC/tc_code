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

There is **one package per recorded teacher attempt**. Each verified attempt
has its own ordered `behaviors[].state` / `behaviors[].text` trajectory and
original `historical_response` ledger row. Every success is replayed, including
later successes; the collector's selected `demos.json` entry does not remove
other attempts. Failed attempts have empty behaviors and an explicit unavailable
reason. Empty replies in verified episodes retain the native EOS rendering,
with original text and affected turn indices preserved in provenance.

Each package charges its own `tokens_spent == usage.completion_tokens`, including
reasoning already recorded there. Confidence is per attempt, so an estimated
failure does not change a sibling success's confidence. No target-length
estimate or per-state allocation replaces ledger usage. Prompt and cached input
tokens are not added to output charges. The per-task public ledger retains the
attempt-to-task dependency without forcing joint purchases.

The usable recorded-output-token sum is the v1.1 budget denominator. The cap
certificate publishes the next-power-of-two usable attempt-cost maximum for the
`hotpotqa_demo_episode` class; it is an archived cached-content envelope, not an
online provider guarantee or the runtime reservation. Full historical teacher usage, including failed
attempts, is separately retained in the accounting block. No teacher calls or
new teacher tokens are incurred by building or preflight.

The completed `teacher_pool_v2` snapshot has **419 attempts: 113 verified,
306 failed**, covering all 200 support tasks. It produces **419 packages:
113 usable, 306 unavailable**. This snapshot has no duplicate successes;
synthetic tests exercise separate purchases of multiple verified attempts.

| Accounting scope | Exact output tokens | Estimated output tokens | Total |
| --- | ---: | ---: | ---: |
| Usable attempts, classified by ledger-row confidence | 58,767 | 8,033 | **66,800** |
| All historical attempts, classified by ledger-row confidence | 507,176 | 28,390 | **535,566** |

There are **111 exact and two estimated usable packages**. Eight historical
attempts have estimated usage (two verified, six failed); all their recorded
costs are retained. Failed attempts account for **468,766** historical output
tokens, including **66,406** from tasks that eventually succeeded; those failures
remain separate packages, outside the usable denominator.
Historical prompt/cached tokens are 6,422,285 / 402,477. The bank's historical
cost confidence is `estimated`. `demos.json` has all 113 demos, and
`attempts.jsonl` covers all 419 attempts. The request journal reconciles with
the complete ledger; there are zero outside-prefix calls or trailing bytes.
All 113 successes were replayed against cached Wikipedia without model calls.

The usable denominator is **66,800**. Both V0 and D3 keep absolute cumulative
caps **15,000 / 30,000**. The generic fraction reference is 6,680 / 16,700.
Runtime reservations use individual recorded costs; the archived class envelope
is not the runtime price and does not control affordability.

## Attempt purchase correction (2026-09-11)

The purchase unit is **one teacher attempt**, verified or failed, at its exact
recorded ledger token cost, including reasoning already counted in completion
usage. Estimates retain their original confidence; reasoning is never added
again. Sort the original opaque attempt query IDs, then apply
`random.Random(0).shuffle` once. Buy that global prefix and **stop before the
first overflow**. Failures and excluded attempts are charged and yield no positive.
Every purchased usable attempt contributes its own trajectory; buying a success
does not buy its sibling retries.

`public/task_attempts.json` stays as the certificate-bound per-task ledger: it
maps each attempt to its task and parent. Existing ALFWorld/BFCL v1 metadata is
accepted as archival accounting, but its obsolete task order/prices do not drive
purchases. New banks publish v2 with the attempt order. The broker prices from
public ledger rows and validates only the purchased sealed payload on reveal.
Old task-cost ledgers and non-prefix resumes are rejected.

Purchases span the same whole inventory as the paper baseline, independently of
training seed or current fold. Fold membership still gates **training exposure**;
purchased other-fold evidence waits for its task to rotate into training.
Failed and other-fold purchases remain in V0/D3 replay receipts. Frozen purchases
need no student features or cost-model fitting. The per-window package limit
remains; windows consume the cumulative authorization and continue the same
prefix, never skipping a blocker. The audit below exhausts that prefix at each
cap without the training window limit. Usable counts mean bank-usable purchased
attempts, before selecting the current training fold.

The read-only sibling `paper_data.load_purchased` was executed in a separate
Python process against each exact same bank. Assertions compare ordered attempt
IDs, usable IDs/counts, tokens, next blocker and the entire shuffle hash at all
four budgets. Full receipts with **both code paths' IDs** are in
[attempt parity validation](rtd_attempt_purchase_validation.json).

| Bank | Budget | Attempts | Usable | Tokens charged |
| --- | ---: | ---: | ---: | ---: |
| alfworld | 7,500 | 4 | 2 | 6,535 |
| alfworld | 15,000 | 9 | 5 | 13,342 |
| alfworld | 30,000 | 13 | 5 | 29,229 |
| alfworld | 60,000 | 24 | 9 | 58,602 |
| bfcl | 7,500 | 11 | 3 | 5,289 |
| bfcl | 15,000 | 11 | 3 | 5,289 |
| bfcl | 30,000 | 11 | 3 | 5,289 |
| bfcl | 60,000 | 23 | 8 | 59,951 |
| hotpotqa | 7,500 | 6 | 1 | 7,207 |
| hotpotqa | 15,000 | 11 | 3 | 12,134 |
| hotpotqa | 30,000 | 17 | 4 | 29,663 |
| hotpotqa | 60,000 | 35 | 8 | 59,859 |

ALFWorld 30k matches the cited **13 attempts / 5 usable / 29,229 tokens**.
The previous task-package comparison and its zero-usable result are superseded.
ALFWorld and BFCL banks were **not rebuilt**: every file, including their
per-task ledgers, certificates and sealed payloads, remains byte-identical.
Their denominators remain 118,792 / 1,418; ALFWorld configured fraction caps stay
11,879 / 29,698, and BFCL absolute caps stay 15,000 / 30,000.

HotpotQA required a sealed rebuild because its prior 200 packages combined
419 attempts. Rebuilding used the exact same input hashes/completed ledger
snapshot and the offline cached replay builder. The replacement gate compared
all support/reset bytes, all 419 original ledger rows and the original verified
trajectories before replacing the bank. The result has 419 attempt packages,
113 usable and 306 charged failures. Total historical output remains 535,566;
the usable denominator becomes **66,800** (58,767 exact + 8,033 estimated).
The configured absolute caps stay **15,000 / 30,000**. The former 133,206 task
basis had included 66,406 failed-retry tokens from eventually successful tasks.
The previous bank is retained in
`_trash/v1_1_hotpotqa_luna_task_accounting_before_attempt_correction`.
All arms and baselines must consume the rebuilt HotpotQA certificate.

Reproduce the read-only parity audit:

```bash
PYTHONPATH=src:. CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
/home/xueqi/hq/projects/tc-alignment/.venv/bin/python tools/rtd_task_purchase_audit.py \
  --bank data/rtd/v1_1_alfworld_luna --bank data/rtd/v1_1_bfcl_luna \
  --bank data/rtd/v1_1_hotpotqa_luna \
  --baseline-reader /home/xueqi/hq/projects/tc-alignment-base/src/bfas/rtd/baselines/paper_data.py \
  --out docs/rtd_attempt_purchase_validation.json
```

Full V0/D3 CPU preflight passes for all three banks. BFCL requires
`BFCL_PROJECT_ROOT=/tmp/rtd-attempt-bfcl` for writable runtime locks in this
read-only shared checkout. `memory_kv_141-notetaker-11` was **not removed or
renamed**: the raw `memory_141-notetaker-11` row is expanded by the official
loader into backend-specific IDs and prerequisite chains. Without the override,
`load_file` raises EROFS creating `.file_locks`; `BFCLAdapter._load_memory_entries`
swallows the exception, and `BFCLSupport` then fails with the missing-ID KeyError.
With the override, the expanded entry has the exact frozen support parent hash
and fold. Data still comes from `PACKAGE_ROOT/data`; no harness or support edit
was needed. Details: [harness diagnosis](rtd_bfcl_memory_harness_diagnosis.json).

The requested CPU filter finished with **316 passed, 1 skipped, 3,069 deselected,
1 existing warning**, in 112.72 seconds. See
[CPU receipts and byte-preservation checks](rtd_attempt_cpu_validation.json).
No GPU, model, teacher/API call or commit was used.

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
does not measure purchases from the completed attempt-package bank.

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
attempt-package bank is a new snapshot that all compared arms must share. Its
individual failed attempts are paid separately; current purchase receipts are
asserted against the baseline reader above.

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
attempt costs, separate successes despite later published demos,
per-attempt estimated success/retry costs, byte-identical support manifests,
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
