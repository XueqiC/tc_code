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
two rounds, checkpoints `[0.1, 0.25]`, seed 0, 60 GB, two source samples per state,
and the same D15 score-tolerance block (`mean_abs=0.15`, `max_abs=2.0`,
`max_abs_outlier_tokens=4`, `max_abs_hard=8.0`). Only benchmark paths, support
size, evaluation protocol and ReAct action horizons differ. The alias
`configs/rtd/unified_hotpotqa_gemma4_d0_luna.yaml` also supports the existing
ALFWorld filename ordering.

## Build and accounting

Run from this worktree, with the populated local tokenizer and HotpotQA
JSON/cache available:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES='' \
PYTHONPATH=src .venv/bin/python tools/rtd_bank_build.py \
  --benchmark hotpotqa \
  --pool /path/to/frozen-pool/demos.json \
  --ledger /path/to/frozen-pool/teacher_ledger.jsonl \
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
worked example and terminal EM checked. Any demo present in both files must
match exactly. Missing attempts archives are explicitly recorded as absent;
they are not reconstructed. The request journal is reconciled with completed
episode usage; in-flight requests outside that ledger prefix are disclosed but
are not falsely labeled completed episodes.

The layout is the same as the ALFWorld/BFCL v1.1 banks:

```text
public/requests.json
public/support.json
public/reset_states.json
public/cap_certificate.json
sealed/<query_id>.json
sealed/integrity.json
sealed/audit.json
sealed/audit_v11.json
```

There is **one package per paid teacher episode**, with ordered
`behaviors[].state` and `behaviors[].text`. Its original ledger row is retained
in `historical_response`; provenance identifies the task, attempt, teacher,
ledger snapshot and student rendering. Two empty replies in one verified episode are rendered as explicit native
EOS targets. RTD's `teacher_tokens` already appends that same EOS to an empty
reply, so the supervised token sequence is unchanged and the paper reader gets
a nonempty target. Original empty replies stay in `historical_response`, and
`provenance.empty_targets_rendered_with_native_eos` records their turn indices.
Failed attempts have their original
costs, empty behaviors and an explicit unavailable reason. They remain
available to privileged historical accounting and paper-baseline charging.
The RTD broker offers only usable packages.

Costs are the ledger's `tokens_spent == usage.completion_tokens`, including
hidden reasoning and paid failed attempts. No target-length estimate or
per-state allocation replaces an episode's recorded cost. Reported usage is
`exact`; uncertain reservations already marked `estimated` by the collector
remain estimated and cannot produce a usable demo. Cached input tokens remain
a subset of prompt tokens and are not counted again as output tokens.

The usable recorded-output-token sum is the v1.1 budget denominator. The cap
certificate publishes the next-power-of-two usable episode maximum for the
`hotpotqa_demo_episode` class; it is a cached-content reservation bound, not an
online provider guarantee. Full historical teacher usage, including failed
attempts, is separately retained in the accounting block. No teacher calls or
new teacher tokens are incurred by building or preflight.

The built bank snapshot has 82 completed attempts: **25 verified, 57 failed**.
Its usable denominator is **10,719**, with cumulative 10%/25% caps of
**1,072 / 2,680**, and a class cap of **2,048**. Historical output accounting is
**105,091**: 98,316 reported exact tokens and 6,775 estimated tokens already
marked uncertain in the source ledger. Failed attempts account for 94,372 of
the total. `demos.json` had four entries; the other 21 verified demos came from
ledger payloads. `attempts.jsonl` was absent. All 25 successes were replayed
against the locally copied cache without model calls.

The workspace originally linked all of `envs` into the shared checkout. Local
setup replaced that workspace-only symlink with a directory of links to the
same existing environments and a copied `envs/hotpotqa/{data,cache}`. Thus cache
lock files stay in this writable tree. Large runtime datasets/cache and the
pre-existing `.cache` remain untracked. The certified bank needs explicit staging
with `git add -f data/rtd/v1_1_hotpotqa_luna`.
Copy the same Wikipedia snapshots with the bank when running elsewhere. A
cache warmed only by this teacher pool may lack queries generated by students;
this offline protocol will report those missing snapshots explicitly.

## Paper baseline integration in the read-only base worktree

Inspected files in `/home/xueqi/hq/projects/tc-alignment-base` were left
unchanged. Its `src/bfas/rtd/baselines/paper_data.py::load_purchased` accepts this
bank's certificate, episode payloads, teacher IDs, dependencies, and native
Gemma prompt/target boundary without changes. The reader was exercised directly
on the real bank at fractions 0.1, 0.25 and 1.0.

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

The paper reader's fixed seed-zero prefix purchases include failed attempts
and stop before the first overflow. On **this partial snapshot**, both 10% and
25% purchase one failed package (1,005 tokens) and **zero positive rows**; even
100% of the usable denominator purchases only three failed packages (6,755
tokens). The existing baseline trainer consequently refuses training until
its frozen purchase yields usable rows. This is a budget/inventory limitation,
not a layout error. Do not reorder query IDs, drop paid failures, or silently
change the denominator to improve that outcome. A later collection snapshot
must be a new, jointly frozen bank for all arms.

## CPU verification

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES='' \
/home/xueqi/hq/projects/tc-alignment/.venv/bin/python -m pytest -q tests/ \
  -k 'hotpotqa or bank_build or registry or preflight'

PYTHONPATH=src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES='' \
.venv/bin/python tools/rtd_preflight.py \
  --config configs/rtd/v1_1_hotpotqa_luna.yaml --arm V0

PYTHONPATH=src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES='' \
.venv/bin/python tools/rtd_preflight.py \
  --config configs/rtd/unified_hotpotqa_gemma4_luna.yaml --arm D3
```

CPU tests cover tiny synthetic paid pools with all 200 support resets, failed
attempt costs, stale demos snapshots, corruption rejection, native rendering,
serial/cohort trace equality under RNG v2, per-episode wiki cursors, fold guards,
cache failures, greedy diagnostics, synthetic V0/D3 preflight, and a stubbed
500-question round-end adapter campaign with separate EM/F1. Real-bank V0 and
D3 preflight both passed with 25 usable packages, 200 rendered states and zero
ledger spend. The requested test selection passed **151 tests** (3,128
deselected); shared ledger, RNG-identity and WebShop regressions passed another
31 tests. See [rtd_hotpotqa_validation.json](rtd_hotpotqa_validation.json) for
the final certificate hash and CPU preflight receipts. No GPU, network or live
model/teacher API was used.

Staging in this sandbox was blocked: Git could not create the linked worktree's
`/home/xueqi/hq/projects/tc-alignment/.git/worktrees/tc-alignment-uni/index.lock`
on its read-only filesystem. Source changes and the bank remain in this
worktree; no commit was created.
