# RTD v1.1 D16: Gemma 4, GPT-5.4, and three benchmarks

The governing method remains `RTD_V1_1_METHOD_ZH.md` and
`RTD_END_TO_END_METHOD_AND_EXECUTION_V1.md`; student/teacher/deployment settings
follow `protocol_v2_gemma4_gpt54.md`. D12 generation tickets, D14 replay scheduling,
D15 source scoring consistency, exposure units, injection, feedback roles, fold
rotation, and reserve/reveal/settle ledger accounting are unchanged.

The shared `tools/rtd_experiment.py` runner now dispatches support, action limits,
feedback, evaluation, and identities through the benchmark registry. All three
production configurations use `google/gemma-4-12B-it`, two budget rounds at 10% and
25%, and a 60 GB memory budget. No teacher acquisition is launched by bank building
or training. The current checkout contains historical banks; the new GPT-5.4
banks must be built from the corresponding paid pool and ledger.

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
