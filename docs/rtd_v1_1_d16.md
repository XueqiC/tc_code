# RTD v1.1 D16: Gemma 4, GPT-5.4, and three benchmarks

The governing method remains `RTD_V1_1_METHOD_ZH.md` and
`RTD_END_TO_END_METHOD_AND_EXECUTION_V1.md`; student/teacher/deployment settings
follow `protocol_v2_gemma4_gpt54.md`. D12 generation tickets, D14 replay scheduling,
D15 source scoring consistency, exposure units, injection, feedback roles, fold
rotation, and reserve/reveal/settle ledger accounting are unchanged.

The shared `tools/rtd_experiment.py` runner now dispatches support, action limits,
feedback, evaluation, and identities through the benchmark registry. All three
production configurations use `google/gemma-4-12B-it`, two cumulative budget rounds,
and a 60 GB memory budget. The original configs use 10% and 25%; Luna overrides
are specified below. No teacher acquisition is launched by bank building
or training. The current checkout contains historical banks; the new GPT-5.4
banks must be built from the corresponding paid pool and ledger.

## P1 absolute teacher-output-token budgets

Declare exactly one budget key: `budget_checkpoints_bank_fraction: [0.1, 0.25]`
or `budget_checkpoints_tokens: [B_half, B]`. Both validators reject both keys or
neither key. Token caps must be strictly increasing positive integers, one per
round (P1 has two; v1.1 also permits three). They are cumulative recorded teacher
output token caps, independent of bank size, with no scaling, rounding, or
requirement to spend the full cap. Fractions retain half-up integer rounding of
the certified usable recorded-output-token denominator.

Both forms use the same sealed-pool purchase order and accounting: charge paid
failed attempts, and stop before the first overflow. Do not drop failures,
reorder purchases, or skip an overflowing item to fill the cap. Existing hard
reservations and replay ledger checks still apply; these caps authorize access
to cached content, with no new teacher API calls.

| Luna benchmark | V0 / unified configs | Cumulative caps |
|---|---|---|
| BFCL | `v1_1_bfcl_luna.yaml`, `unified_bfcl_gemma4_luna.yaml` | `[2500, 5000]` tokens |
| HotpotQA | `v1_1_hotpotqa_luna.yaml`, `unified_hotpotqa_gemma4_luna.yaml`, both D0 filename variants | `[10000, 20000]` tokens |
| ALFWorld | All existing Luna configs and the running V0 remain unchanged | `[0.1, 0.25]` bank fractions |

ALFWorld Luna's 25% cap is **29,698 tokens**, reported as **B=30k**; the actual
running cap is not changed to 30,000. In each new manifest,
`budget_checkpoint_form` is `bank_fraction` or `tokens`, `budget_ceilings` gives
the resolved absolute caps, and `config` retains the declaration. Token manifests
use `budget_rounding=none_absolute_tokens`. Fraction and token configs remain
different config/campaign/replay identities even when their resolved caps match.
Complete and streaming replay both enforce this distinction; historical fraction
schedule identities remain compatible.

CPU verification on the real Luna banks passed with `tools/rtd_preflight.py`
for BFCL V0/D3 (20 available packages, 33 rendered support states) and HotpotQA
V0/D3 (25 available packages, 200 rendered support states). Each audit resolved
the caps above and each preflight reported zero ledger spend. These use the
default preflight `smoke` mode, local tokenizer files, disabled CUDA, offline HF
settings, and `BFCL_PROJECT_ROOT=/tmp/rtd-p1-budget-bfcl-runtime`; they perform
startup checks without training, teacher calls, or writes to live run directories.

```bash
export CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export BFCL_PROJECT_ROOT=/tmp/rtd-p1-budget-bfcl-runtime
export PYTHONPATH=src:. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
RTD_PY=/home/xueqi/hq/projects/tc-alignment/.venv/bin/python
"$RTD_PY" tools/rtd_preflight.py --config configs/rtd/v1_1_bfcl_luna.yaml --arm V0
"$RTD_PY" tools/rtd_preflight.py --config configs/rtd/unified_bfcl_gemma4_luna.yaml --arm D3
"$RTD_PY" tools/rtd_preflight.py --config configs/rtd/v1_1_hotpotqa_luna.yaml --arm V0
"$RTD_PY" tools/rtd_preflight.py --config configs/rtd/unified_hotpotqa_gemma4_luna.yaml --arm D3
"$RTD_PY" -m pytest -q tests/ -k 'unified_p1 or preflight or config or budget'
```

## Build the banks (CPU, cached tokenizer, no API)

The GPT-5.4 BFCL harness recovery build, its actual certificate totals, and the
missing-usage/stateful exclusions are documented in
[rtd_v1_1_bfcl_gpt54_bank.md](rtd_v1_1_bfcl_gpt54_bank.md).

From the repository root:

```bash
export PYTHONPATH=src:.
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
RTD_PY=/home/xueqi/hq/projects/tc-alignment/.venv/bin/python

"$RTD_PY" tools/rtd_bank_build.py --benchmark bfcl \
  --pool data/bfcl_sft/pool_gpt54_sft.jsonl \
  --ledger data/teacher_ledger/bfcl_gpt54.jsonl \
  --out data/rtd/v1_1_bfcl_gpt54

"$RTD_PY" tools/rtd_bank_build.py --benchmark alfworld \
  --pool data/rtd/v1_alfworld_c26 \
  --ledger data/teacher_ledger/alfworld.jsonl \
  --out data/rtd/v1_1_alfworld

"$RTD_PY" tools/rtd_bank_build.py --benchmark webshop \
  --pool data/webshop_sft/demos_gpt54.json \
  --ledger data/teacher_ledger/webshop.jsonl \
  --out data/rtd/v1_1_webshop
```

The BFCL and WebShop pool filenames above designate the new collection artifacts;
building refuses missing files, unpaid evidence, mismatched pool/ledger targets,
and existing output directories. `--config PATH` overrides the corresponding
production configuration. The ALFWorld-specific equivalent is:

```bash
"$RTD_PY" tools/rtd_alfworld_bank_v11.py \
  --source data/rtd/v1_alfworld_c26 \
  --ledger data/teacher_ledger/alfworld.jsonl \
  --config configs/rtd/v1_1_alfworld.yaml --out data/rtd/v1_1_alfworld
```

BFCL input is the standard per-state demo JSONL (`task_id`, `response`, and
`_render_context` containing messages/functions, or a recoverable legacy prompt).
The paired ledger uses the BFAS teacher gateway schema: `task_id`, `teacher`,
`attempt_index`, `verified`, `tokens_spent`, and a verified `demo.turns` payload.
The builder calls `tools/bfcl_pool_render_gemma4.py:render_row(..., verify=True)`
for both the pool and the ledger evidence. Native targets and prompts are its
unaltered output bytes. Stateful rows require their actual per-step context;
whole episodes cannot be flattened into a fabricated single-turn target.
The unchanged BFCL RTD configuration retains its existing 40-parent manifest;
other paid tasks remain unavailable inventory. The certificate records the pool
and ledger hashes and does not inherit C25's 55,370-token denominator.

WebShop input is the JSON mapping of task IDs to serialized `Demo` objects returned
by `WebShopAdapter.teacher_demo`, or equivalent JSONL with `task_id` and `turns`.
Its ledger is the adapter gateway's `data/teacher_ledger/webshop.jsonl`.
The builder uses the adapter's stratified split over sessions 500–6909, freezes
S_d=200 and S_c=50, and records a reset state for **every** S_d task, including tasks
with no verified demo. The full environment/index is required for this build.
Teacher commands are replayed through the bridge and official episode driver;
live observations must agree with the archived demo. States are rendered from
the verified command replay with the deployment prompt. This removes archived
teacher thoughts and format-only no-ops from command histories, while retaining
the paid original turn in sealed provenance. No teacher call is made.

A WebShop package contains one state and one teacher action. Later states require
ownership of the preceding prefix package. If the ledger has only an episode
cost, integer target-token allocations sum exactly to that recorded cost and are
labelled estimated; the full original episode bill, including failed attempts,
is also retained in the sealed audit. No prefix repeats the full episode charge.
WebShop ledger rows without a cost-confidence field remain estimated, since the
adapter can fall back to text estimates when Azure usage is absent.
ALFWorld and BFCL retain paid episode boundaries; D12 enumerates their ordered
state supervision records. ALFWorld conversion audits the original C26-B replay
and ledger and changes only the student rendering in a new bank.

New certificates bind student, benchmark, public support/reset artifacts, sealed
integrity, source hashes, usable set, recorded-cost denominator, and uniform
next-power-of-two class caps. The broker reserves those certified public caps.
The original C25 certificate validator and converter remain available for old
banks; old payloads are never rewritten.

### Parent fold contract

`fold` is an integer 0/1 on each **support parent** in `public/support.json`.
BFCL and WebShop store parent record lists; ALFWorld stores a mapping keyed by
parent hash, with `selected_task_id` identifying the representative trial.
Paid `public/requests.json` packages carry `parent_hash`; they do not duplicate
the support fold. Both current ALFWorld and BFCL GPT-5.4 banks already contain
all support folds. The D3 `KeyError: 'fold'` came from the manifest adapter
discarding this field while reconstructing ALFWorld parent records.

The builders assign `int(parent_hash, 16) % 2` and preserve frozen explicit
assignments during conversion. For ALFWorld the hash is SHA256 of canonical JSON
`{benchmark: alfworld, split: train, parent_game: first task_id segment}`; related
trials stay together. These are rotating inner-training/feedback folds, with
calibration/probe parents excluded separately, not calibration/confirmation
assignments. Explicit folds remain authoritative, including BFCL assignments
that differ from parity; malformed explicit values are rejected.

Manifest creation now reads these support records directly. For legacy parent
records without `fold`, `fold_roles` uses the same hash parity. The run manifest
records the rule and sorted affected hashes in `parent_group_fold_derivation`
only when derivation was needed. Existing signed support files, certificates,
and packages are not rewritten. Both ALFWorld builder entrypoints already copy
the complete audited support, including folds, into the new certificate-bound
bank; this also covers the Luna collector's C26-format output.

## Run each benchmark

Select one physical GPU; the same hardware class remains mandatory for all arms
being compared. RTX PRO 6000 Blackwell on rai has an explicit hardware class;
GPU1/GPU4 UUIDs remain instance metadata and GPU model, memory, capability, CUDA,
driver, Python and library versions remain hard comparison fields.
Student dispatch and Gemma export change the scoring identity. Use fresh run
directories; the historical C25 evidence cannot authorize these changes as
operational-only updates to existing campaigns.

```bash
export CUDA_VISIBLE_DEVICES=1 CUDA_DEVICE_ORDER=PCI_BUS_ID

# BFCL v4
"$RTD_PY" tools/rtd_experiment.py run --arm V0 --config configs/rtd/v1_1_bfcl_gemma4.yaml --run-dir results/rtd_v1_1/bfcl_gemma4/V0
"$RTD_PY" tools/rtd_experiment.py run --arm V1 --config configs/rtd/v1_1_bfcl_gemma4.yaml --replay-schedule results/rtd_v1_1/bfcl_gemma4/V0 --run-dir results/rtd_v1_1/bfcl_gemma4/V1
"$RTD_PY" tools/rtd_experiment.py run --arm V2 --config configs/rtd/v1_1_bfcl_gemma4.yaml --run-dir results/rtd_v1_1/bfcl_gemma4/V2

# ALFWorld
"$RTD_PY" tools/rtd_experiment.py run --arm V0 --config configs/rtd/v1_1_alfworld.yaml --run-dir results/rtd_v1_1/alfworld/V0
"$RTD_PY" tools/rtd_experiment.py run --arm V1 --config configs/rtd/v1_1_alfworld.yaml --replay-schedule results/rtd_v1_1/alfworld/V0 --run-dir results/rtd_v1_1/alfworld/V1
"$RTD_PY" tools/rtd_experiment.py run --arm V2 --config configs/rtd/v1_1_alfworld.yaml --run-dir results/rtd_v1_1/alfworld/V2

# WebShop
"$RTD_PY" tools/rtd_experiment.py run --arm V0 --config configs/rtd/v1_1_webshop.yaml --run-dir results/rtd_v1_1/webshop/V0
"$RTD_PY" tools/rtd_experiment.py run --arm V1 --config configs/rtd/v1_1_webshop.yaml --replay-schedule results/rtd_v1_1/webshop/V0 --run-dir results/rtd_v1_1/webshop/V1
"$RTD_PY" tools/rtd_experiment.py run --arm V2 --config configs/rtd/v1_1_webshop.yaml --run-dir results/rtd_v1_1/webshop/V2
```

V1 can use the unchanged D14 streaming options when its V0 run is still producing
checkpoints. Resume uses `tools/rtd_experiment.py resume --run-dir RUN` and the
saved configuration. An audit uses `tools/rtd_experiment.py audit --config CONFIG`.

Feedback is temperature-one, top-p-one, from full task resets; malformed outcomes
consume their original sampled actions. WebShop reuses its bridge/index between
episodes. Official evaluation is separate: BFCL uses `gemma4_fc`, all 5,217 entries,
and the frozen official 0.001 temperature; ALFWorld uses vLLM ReAct, valid_seen 140,
40 steps, greedy; WebShop uses the adapter's test sessions 0–499, full index,
15 steps, greedy, 128 tokens, and the exact shared `tools/webshop_eval.py` prompt.
WebShop success rate is primary and 100×mean reward is diagnostic. Missing tasks
or environment failures prevent publication. Each `evaluation-<round>.json`
binds campaign identity, checkpoint spend, hardware, data, and artifact hashes,
with `overall_accuracy_percent` for the common reporter.

Gemma native turn/handoff tokens are scored termination events, including in
batched generation. Teacher targets that already contain a native terminator do
not receive an extra EOS. Streamed source gradients apply Gemma's native logit
softcap before probabilities and differentiate through the same transform.
Production forward batching remains disabled as in the existing BFCL config.

## D15 follow-up: ALFWorld Luna bf16 score consistency (2026-09-11)

**The production likelihood discrepancy remains unresolved by the CPU audit.**
The earlier explanation of ordinary bf16 drift was a hypothesis, not an isolated
cause. The stronger CPU tests below already pass against the original numerical
implementation. They cannot establish a production mean delta below 0.01 or
justify relaxing the guard. This follow-up changes diagnostic reporting and
tests, not model arithmetic or any Luna config.

The earlier audit recorded a V0 action mean/max of 0.05232027349/0.54536819458
(43 tokens, 629-token unpadded prompt); the subsequent reported failure is
approximately 0.097/1.27. During this investigation the live V0 log contained
startup/preflight output and its journal contained no score-consistency records,
so those failed attempts could not be re-audited from the current files. The
available `results/rtd_unified/smoke_alf_D3/compute.jsonl` contains:

| Scope | Checks | Tokens | Token-weighted mean absolute delta | Largest action mean | Largest token delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| Available D3 smoke | 184 | 14,507 | 0.02238173576 | 0.04732709961 | 1.53363204002 |
| Smoke sequence 133 | 1 | 125 | 0.04732709961 | 0.04732709961 | 0.50670385361 |
| Smoke sequence 932 | 1 | 69 | 0.04581571836 | 0.04581571836 | 0.58704185486 |

Sequence 133 has prompt length 486 and padded width 677; its first token differs
by 0.27753853798 nats, before incremental decoding. Sequence 932 has a single
529-token prompt with no padding. Both actions finish below 1024. Window
crossing, incremental cache updates, and left padding therefore cannot each
explain all the observed discrepancies.

The local checkpoint config and installed source were inspected without loading
12B weights: `Gemma4UnifiedForConditionalGeneration`, Transformers 5.14.1, Torch
2.13.0+cu130. The checkpoint has 48 dense layers, five sliding layers followed by
one full-attention layer per block, a **1024-token** sliding window, global K=V,
`num_kv_shared_layers=0`, and `final_logit_softcapping=30`. There is no 4096-token
window in this checkpoint.

| Stage | Production path | Attention implementation | Dtype | Cache / positions |
| --- | --- | --- | --- | --- |
| Batched sampler | `HFGenerateBackend` → `HFGenerationBatchMixin._generate_group` → PEFT `model.generate` → native HF Gemma wrapper; no vLLM or custom decode loop | Loader explicitly requests `eager`; all text layers share the model's text attention config | Base projections, hidden states and native final softcap: bf16; LoRA: FP32; HF sampling softmax and recorded log-softmax: FP32 | HF `DynamicCache`: sliding layers retain the window, global layers retain full K/V. Left-padded prefill, then one token per row; HF derives real-token position IDs from the mask |
| Single-action HF sampler | `runtime.HFGenerateBackend.sample_action` when generation batching is disabled | Same model and eager setting | Same as batched sampler | Same HF KV-cache path; unpadded prompt |
| Source / REINFORCE scorer | `return_gradient.score_tokens` → `torch.func.functional_call` → native Gemma forward → `transport.complete_token_logprobs`; `scoring.py` only compares the resulting scores | Same model and eager setting | Same bf16 base/softcap and FP32 LoRA; selected logits upcast to FP32 **before** native CE; FP32 sum | Full prompt + sampled action, `use_cache=False`, no retained KV state, positions start at zero. Each preceding logit predicts its sampled label, including termination |
| Optional batched scorer | `forward_batch._forward_hidden` plus chunked native head projection and `source_scoring.cap_logits`; **disabled** by `forward_prompts_per_batch: 0` in the supplied config | Same model and eager setting | Same bf16 head/softcap, FP32 CE | No KV cache, right padding preserves real-token positions |
| Differentiable scoring | Same serial scorer with `checkpointing.enable_gradient_checkpointing` wrappers | Same eager setting | Same forward dtypes; gradients with respect to FP32 LoRA | Non-reentrant layer recomputation in eval mode; cache forbidden during checkpointed backward |

The six hypotheses were checked in the requested order:

1. **Attention implementation.** Neither path switches implementations. One
   actual reporting bug was reproduced: sampler metadata and backend identity
   hard-coded `attention='eager'`, even when constructed with SDPA; scorer
   metadata omitted attention entirely. These now read the text config's
   effective `_attn_implementation` (null for models that do not expose it).
   Both HF samplers also record the returned cache type; scorers record null.
   This fixes misleading evidence, **not the production likelihood gap**.
   The installed Unified attention passes scaling 1 and the sliding-window
   setting; it does not pass an attention-score softcap. Its final vocabulary
   logit softcap is a separate native transform.
2. **Sliding/global attention and KV cache.** CPU comparisons cover prompts
   before, at, and beyond a small window, and actual 1024-window crossing. Tests
   include both the production no-sharing setting and shared-KV layers. Native
   cached generation and uncached teacher forcing agree in these cases.
3. **Final softcap and embedding scale.** Both normal paths call the same native
   scaled embedding, vocabulary head and divide/tanh/multiply softcap. HF's
   `logits_to_keep` only selects the positions sent to that head. The optional
   batched scorer explicitly applies the same softcap after projection.
4. **Dtype.** HF upcasts next-token logits before sampling; the sampler records
   FP32 log-softmax. CE also upcasts before normalization. There is no bf16 vs
   FP32 log-softmax mismatch. In eager attention, QK matmul produces bf16,
   attention softmax computes in FP32 then casts to query dtype, and the AV
   matmul produces bf16. Native final softcap also runs in bf16. Shape-dependent
   arithmetic remains an unisolated possibility; the audit does not establish
   that it explains a 0.1-nat production mean.
5. **LoRA and dropout.** `installed_parameters` copies the requested FP32
   snapshot into the same PEFT model and restores resident values afterward;
   scoring binds that snapshot functionally. Tests use nonzero snapshot changes
   on every production LoRA target and configured dropout 0.2 with all modules
   in eval mode. Production config uses dropout zero. Checkpointing is enabled
   in the tests as in the production loader.
6. **Left-padding positions/mask.** Tests inspect HF's prefill and every decode
   position: real prompt tokens start at zero, cached tokens continue at the
   real prompt length, and shorter rows are independently rescored without
   padding. Both the first generated token and later cached tokens are checked.

`tests/test_rtd_gemma4_scoring.py` now has 14 numerical cases and six reporting
cases. Each numerical case generates real actions through the production HF
sampler and calls the production scorer on the exact sampled IDs. It requires
per-token FP32 agreement within **1e-4** and bf16 **mean absolute delta <0.01**,
in addition to the existing guard. Hooks verify returned HF scores equal raw
forward logits and match the actual multinomial probabilities. LoRA parameters
must be restored after generation. The reporting cases exercise eager/SDPA,
serial/batched generation, and optional batched scoring.

Before the reporting fix, **all 14 numerical cases passed**, while the four
initial eager/SDPA reporting cases failed (missing scorer field or a false
`eager` label). After the fix, **all 20 tests passed**. An additional exploratory
48-layer, width-32 CPU probe also passed: largest FP32 per-token delta
6.4373e-6; largest bf16 action mean 0.00092757 and token delta 0.00197673. These
are randomly initialized tiny models, not the trained 12B checkpoint. No
numerical root-cause fix or production improvement is claimed. Isolating the
remaining discrepancy requires comparing layer outputs for the failing tokens
on the production numerical backend; no GPU or API calls were made here.

The supplied Luna config currently declares mean/max tolerances 0.15/2.0,
four allowed outliers, and hard max 8.0. Those settings and the running chain's
results were left untouched. Changing or widening tolerances is not a fix for
generation/scoring disagreement.

This follow-up's requested regression command passed **207 tests** (2,799
deselected) in 125.33 seconds:

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
BFCL_PROJECT_ROOT=/tmp/rtd-gemma-cpu-bfcl PYTHONPATH=src:. \
/home/xueqi/hq/projects/tc-alignment/.venv/bin/python -m pytest -q tests/ \
  -k 'scoring or generation_batch or return_gradient or unified'
```

The existing Luna override regression had pinned an obsolete 0.08/1.0 guard;
it now verifies that the actual YAML declaration survives validation, while
retaining independent fixed checks of partial overrides and strict defaults.
No configuration was edited to make the test pass.

The manifest's `score_consistency.tolerance` records the declared guard;
`score_consistency_tolerance_override` additionally records the complete effective
guard whenever it differs from `ScoreTolerance()` (including stricter overrides).
Each source/feedback `score_consistency` journal append updates
`manifest.json: score_consistency_observed` **before enforcement can raise**:

- `mean_abs_difference`: token-weighted mean over comparable journaled checks;
  `max_abs_difference`: largest token delta; `max_mean_abs_difference`: largest
  per-action mean, the statistic relevant to the mean guard.
- `checks`, `failed_checks`, `comparable_checks`, `compared_tokens`, and
  `outlier_token_count` identify the coverage. Failed and repeated attempts count;
  checks lacking a finite, aligned comparison never contribute a fabricated zero.
- `last_score_sequence` and `last_score_hash` bind the summary to the durable
  `compute.jsonl` prefix. Resume rebuilds it from the verified journal, repairing
  a crash between journal persistence and manifest publication without counting
  old records twice. No observations means null means/maxima and zero counts.

Only this derived observation block is excluded from manifest identity digests,
checkpoint bindings and resume configuration comparisons. Tolerances, explicit
overrides, backend identity, config hashes and the other frozen fields remain
bound. Historical manifests without observations retain their old digest; the
existing streaming-replay identity rules remain intact. Changing the configured
tolerance requires a fresh run, rather than silently relaxing a saved run during
resume. For paper reporting, include the effective guard and all three delta
statistics with their check/token counts and failed/repeated-attempt scope.

The requested regression command passed **156 tests** (2,832 deselected) in
37.82 seconds, using the shared venv, CPU-only visibility and offline Hub mode:

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
BFCL_PROJECT_ROOT=/tmp/rtd-score-consistency-final-bfcl PYTHONPATH=src:. \
/home/xueqi/hq/projects/tc-alignment/.venv/bin/python -m pytest -q tests/ \
  -k 'scoring or score_consistency or unified_p1 or d15'
```

## CPU validation

The merged, untouched pre-D16 HEAD reproduced a stale historical manifest oracle
in `test_rtd_v11_source_estimator.py`. The test now uses the independently
reproduced HEAD fingerprint, normalizing only D16 code bindings; its frozen
configuration and numerical gradient oracles remain unchanged.

For a read-only shared BFCL installation, its supported `BFCL_PROJECT_ROOT`
override moves generated files and file locks to a writable runtime directory;
the package data and official functions remain in the original checkout.

```bash
export CUDA_VISIBLE_DEVICES=''
export BFCL_PROJECT_ROOT=/tmp/rtd-d16-bfcl-runtime
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PYTHONPATH=src:. /home/xueqi/hq/projects/tc-alignment/.venv/bin/python -m pytest -q tests
```

The complete C26 bank was also converted and audited offline under
`/tmp/rtd-d16-alfworld-bank-v11`: 135 support parents, 107 usable packages,
36,294 recorded output tokens, and checkpoint caps 3,629 / 9,074. The configured
`data/rtd` output resolves to the shared checkout, which this session cannot
write. The production build command above creates the same bank when run with
write access to that directory.

Full CPU suite result: **2,360 passed, 4 skipped, 6 warnings** in 913.24 seconds.
GPU visibility was disabled and Hub access forced offline. The final WebShop
certificate-affordability check also passed in the focused nine-test suite.

The 2026-09-10 fold regression check used the main tree's venv, empty
`CUDA_VISIBLE_DEVICES`, offline Hub settings, and a `/tmp` BFCL runtime:

```bash
PYTHONPATH=src:. .venv/bin/python -m pytest -q tests/ -k 'unified or bank or conventions or manifest'
```

Result: **214 passed, 1 failed**. The failure is the pre-existing historical hash
oracle in `test_hard2_config_manifest_and_reference_gradient_are_byte_identical`;
an untouched `git archive HEAD` reproduced the identical `4d286905…` actual hash
against its expected `e251f010…`. The fold/converter regressions passed, including
both ALFWorld command entrypoints and the fake-teacher Luna pipeline. A CPU-only
`make_manifest` call with arm D3 and the real `data/rtd/v1_1_alfworld` bank passed
with 135 parents split 79/56. Only the mandatory training-GPU identity probe was
replaced with a CPU fixture; local checkpoint/tokenizer/harness/data identities
and the bank audit ran normally. No GPU computation or teacher API call ran.
