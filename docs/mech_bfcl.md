# BFCL mechanism validation

For the coverage expansion, use [Round 2](#round-2-coverage-expansion) below.
The first-round workflow and its held-out layers remain supported.

Implements [the user specification](2026-09-12-gpt6-mechanism-validation-from-user.md)
for **google/gemma-4-12B-it**, comparing ordinary local practice **C** with
gap-conditioned practice **D**. Entrypoint: `tools/mech_bfcl.py`. Implementation:
`src/bfas/mech_bfcl/`. No GPU training, student inference, or paid API calls were
performed while building this pipeline. CPU tests use stubs and the installed
official checker; compatibility with an actual A100 training/serving run remains
to be established by the commands below.

The committed split has 24 support tasks, 16 calibration tasks, and 256 full
evaluation questions. Both training-side sets come exclusively from the union
of `demand` and `calibration` in `configs/bfcl_support_split.json`. Evaluation
excludes that entire 50-ID pool, shared parents, and training-side API families.
IDs, source hashes, exclusions, category counts, and seed 0 are fixed in
`configs/mech_bfcl_splits.json`; rerunning `splits` verifies the same contents.

Memory parents are whole personas across backends, because their write chains
share information. Stateful tasks with identical initial configurations share
a parent. Stateless API families are normalized names/namespaces; executable
families are official executor classes. The input pool touches every memory
backend, so rec_sum is reserved for evaluation and excluded from the 40 selected
training-side tasks. WebSearchAPI is also reserved for evaluation. Evaluation
contains 39 memory, 59 multi-turn, 49 irrelevance, 59 other live, 40 non-live, and
10 web-search questions. Support emphasizes weakness: 2 memory, 7 multi-turn,
and 5 irrelevance tasks. Calibration includes 3 memory and 1 multi-turn task.
There are 24 / 14 / 190 distinct parent groups, respectively. These constrained
subsets are **mechanism validation**, not official BFCL Overall. Historical
base context (not a newly measured result): Overall 45.6; NL 82 / Live 80 /
MT 53 / Memory 30 / Irrel 75.

The official harness is used without editing its checkout or datasets.
`tools/bfcl_cli.py` enables an opt-in Gemma capture handler through
`MECH_BFCL_CAPTURE=1`. The native `gemma4_fc` renderer/parser and official
agent loop, memory prerequisites, AST checker, executor, and full-task evaluator
remain authoritative. Each subprocess receives a private `BFCL_PROJECT_ROOT`.
All query histories, exact rendered prompts, responses, tool returns, executor
snapshots, raw results, scores, and logs are retained. Successful completed runs
are reusable. On rerun, a harness directory containing `run.json` but no
`items.json` is moved to `<name>.failed-<UTC timestamp>` before a fresh attempt
starts; timestamp collisions receive a numeric suffix. No failed artifacts are
deleted. For example, the incomplete `evaluation/base/main/full` moves to
`evaluation/base/main/full.failed-20260911T230001.123456Z`. Directories without
`run.json` and completed runs with changed inputs are still rejected.

Each support, calibration, or full-evaluation batch expands memory questions
through their official `depends_on` chains, deduplicating shared prerequisites.
One generation invocation runs the entire batch; the official scheduler orders
each persona/backend's chain. The matching checker invocation scores questions
while excluding write-phase prerequisites. `run.json` records requested and
expanded IDs. Extra episodes retain their student captures in `trajectories/`
and are indexed as prerequisite trajectories in `prerequisites.json`; only
requested items enter `items.json`, and only support IDs can supply seeds.
Memory initial configurations use the same reversible packing as executor
snapshots so the harness's filesystem paths survive capture.

Layer 3 evaluates **246 of the 256 frozen subset questions**, using the same
memory expansion. `LAYER3_EXCLUDED_CATEGORIES = ("web_search",)` in
`src/bfas/mech_bfcl/pipeline.py` excludes the ten `web_search` IDs when building
the full-task ID list for **every arm and repeat**. This project has no SERPAPI
key: web search scores 0/200 for every model, including the base student, cannot
discriminate arms, and costs roughly 20 retries per item. Also, the official
generator writes `BFCL_v4_web_search_result.json`, while the evaluator recognizes
only `web_search_base` and `web_search_no_snippet` and silently skips that file.
The exclusion changes only layer-3 selection; `configs/mech_bfcl_splits.json`
and its split hash remain unchanged, so completed rollout, diagnose, C/D
generation, and local evaluation artifacts remain reusable. Layers 1/2 keep
their original held-out items. BFCL data, checker, and server settings are unchanged.

The full run's `run.json` records `excluded_categories`, `excluded_ids`,
`subset_questions`, `evaluated_questions`, and `exclusion_reason`. Each layer-3
row in both `full/items.json` and the combined evaluation `items.json` carries
the same fields under `evaluation_scope`, preserving the existing list format.
Support/calibration metadata keeps its existing format for cache compatibility.
`report.json` includes `layer_3_scope`; layer-3 paired intervals carry
`evaluation_scope` and the actual item count. `report.md` names the exclusions,
lists their IDs, and states the 246/256 coverage. Pairing validates the selected
IDs and scope metadata for each arm. Checker category omissions now report the
missing, expected, and scored category sets plus the evaluation log path.

Prerequisites do not enlarge the scored subset. Each harness batch's `cost.json` totals captured
generations and exact input/output tokens, with requested/prerequisite
breakdowns; evaluation exposes these totals under `cost.json`'s `layer_3` key.
Completed queries in failed episodes are included. Missing result and completed
trajectory IDs are reported together after the batch and retained cost records.

Run this sequence from the project root on **one A100 80GB**. The cached model
and tokenizer, BFCL environment, vLLM environment, and training environment must
already be installed. `envs/bfcl/.venv/bin/python` has the harness's provider
imports; `.venv/bin/python` has the training dependencies.

```bash
cd /home/xueqi/hq/projects/tc-alignment-mech
set -euo pipefail
export PYTHONPATH=src:.
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0
export BFAS_TEACHER=openai/gpt-5.6-luna
export BFAS_OPENAI_SERVICE_TIER=flex
export OPENAI_BASE_URL=https://api.openai.com/v1
# OPENAI_API_KEY must already be exported by your credential setup.
# Use a fresh directory; an existing single-pass run cannot adopt the loop.
MECH_RUN="$PWD/results/mech_bfcl_loop"
BFCL_PY="$PWD/envs/bfcl/.venv/bin/python"
TRAIN_PY="$PWD/.venv/bin/python"
VLLM="$PWD/envs/vllm-serve/.venv/bin/vllm"
mkdir -p "$MECH_RUN"

"$TRAIN_PY" -m pytest -q tests/test_mech_bfcl*.py
"$BFCL_PY" tools/mech_bfcl.py splits --run-dir "$MECH_RUN"

# The API parent and engine children share a process group. Stop all of them
# and wait until the GPU is empty before allowing training to load weights.
wait_server() {
  until curl --fail --silent http://127.0.0.1:8901/v1/models >/dev/null; do
    kill -0 "$MECH_SERVING_PID"
    sleep 2
  done
}
stop_server() {
  kill -- -"$MECH_SERVING_PID"
  wait "$MECH_SERVING_PID" || true
  unset MECH_SERVING_PID
  while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do
    sleep 2
  done
}

setsid "$VLLM" serve google/gemma-4-12B-it \
  --served-model-name google/gemma-4-12B-it \
  --dtype bfloat16 --max-model-len 32768 --gpu-memory-utilization 0.85 \
  --port 8901 >"$MECH_RUN/serve-base.log" 2>&1 &
export MECH_SERVING_PID=$!
wait_server

# Exactly 24 support rollouts determine seeds. The fixed 16 calibration
# rollouts supply held-out contexts and never contribute failure diagnoses.
"$BFCL_PY" tools/mech_bfcl.py rollout --run-dir "$MECH_RUN" --base-url http://127.0.0.1:8901/v1
"$BFCL_PY" tools/mech_bfcl.py diagnose --run-dir "$MECH_RUN"
"$BFCL_PY" tools/mech_bfcl.py generate --run-dir "$MECH_RUN" --arm C --target-exercises 64 --max-output-tokens 24000 --base-url http://127.0.0.1:8901/v1
"$BFCL_PY" tools/mech_bfcl.py generate --run-dir "$MECH_RUN" --arm D --target-exercises 64 --max-output-tokens 24000 --base-url http://127.0.0.1:8901/v1

# Every evaluate invocation runs all three layers. Repeat base once before
# training; report records paired correctness stability on the selected items.
"$BFCL_PY" tools/mech_bfcl.py evaluate --run-dir "$MECH_RUN" --arm base --base-url http://127.0.0.1:8901/v1
"$BFCL_PY" tools/mech_bfcl.py evaluate --run-dir "$MECH_RUN" --arm base --repeat repeat --base-url http://127.0.0.1:8901/v1
stop_server

"$TRAIN_PY" tools/mech_bfcl.py train --run-dir "$MECH_RUN" --arm C
setsid "$VLLM" serve google/gemma-4-12B-it \
  --served-model-name google/gemma-4-12B-it \
  --dtype bfloat16 --max-model-len 32768 --gpu-memory-utilization 0.85 \
  --enable-lora --max-lora-rank 16 \
  --lora-modules "mech-C=$MECH_RUN/C/training/adapter" \
  --port 8901 >"$MECH_RUN/serve-C.log" 2>&1 &
export MECH_SERVING_PID=$!
wait_server
"$BFCL_PY" tools/mech_bfcl.py evaluate --run-dir "$MECH_RUN" --arm C --base-url http://127.0.0.1:8901/v1
stop_server

"$TRAIN_PY" tools/mech_bfcl.py train --run-dir "$MECH_RUN" --arm D
setsid "$VLLM" serve google/gemma-4-12B-it \
  --served-model-name google/gemma-4-12B-it \
  --dtype bfloat16 --max-model-len 32768 --gpu-memory-utilization 0.85 \
  --enable-lora --max-lora-rank 16 \
  --lora-modules "mech-D=$MECH_RUN/D/training/adapter" \
  --port 8901 >"$MECH_RUN/serve-D.log" 2>&1 &
export MECH_SERVING_PID=$!
wait_server
"$BFCL_PY" tools/mech_bfcl.py evaluate --run-dir "$MECH_RUN" --arm D --base-url http://127.0.0.1:8901/v1
stop_server

"$BFCL_PY" tools/mech_bfcl.py report --run-dir "$MECH_RUN"
```

All inference uses T=0.001, greedy `top_k=1`, the frozen split seed (0 in the
committed split), and explicit
`enable_thinking=False`, through raw completions. No native stop/handoff marker
is inserted into the short supervision target. The renderer reuses
`tools/bfcl_pool_render_gemma4.py`, matching the luna bank's dotted tool names,
argument quoting, and prompt conventions. The bank supplies conventions only;
none of its historical examples or cached-content cost estimates are charged
as new teacher calls or used as new training examples. `--served-model` can
override the default-seed adapter alias; training-seed repeats require the
exact `mech-<arm>-s<seed>` alias. `/models` must identify the matching local adapter.
This command sequence uses vLLM LoRA, avoiding a second set of merged weights.

`diagnose` spends at most 2,000 of the shared 8,000 output tokens. Preparation
uses at most 6,000 for approximately 48 local exercises and 24 actual natural
states from calibration, in distinct generation groups. Natural states are
never fabricated or edited. At most eight support failures seed each arm.
Generation rotates through those seeds repeatedly, requesting nine short
examples per call with a fresh diversity instruction and the already accepted
examples for that context. Both arms use `--target-exercises` (default 64) and
`--max-output-tokens` (default 24,000), frozen together with the seed order in
`generation_protocol.json`. Each call reserves 3,000 output tokens. The loop
stops at the distinct validated target, or before actual output tokens spent
plus the next full reservation would exceed the cap; the reservation is never
shrunk to spend the remainder. Caps below 3,000 permit no arm calls.

De-duplication happens before admission, across the whole arm and within each
batch. It compares student messages, tools, executor state, and demonstration,
ignoring IDs and variant labels. Failed neighbours, malformed or inconsistent
examples, duplicates, and unused proposals from the final call remain charged.
`generation.json` records the stop reason, actual count, call count, output
tokens, and any target shortfall. Training never repeats examples to compensate.
No replacement seeds are mined from evaluation. Cached attempts replay without
new teacher purchases on resume; changing settings or adopting the loop from
an existing single-pass generation requires a new `--run-dir`.

C and D receive identical task/interface/state context and format requirements.
Only D receives the current student response, execution failure, and testable
gap hypothesis. Prior natural history is identical and may contain imperfect
student actions. The system prompt, seed/diversity rotation, reservation policy,
de-duplication, and admission checks are shared. Both arms must retain condition,
surface, and base-verified neighbouring examples globally, including a changed
condition action; this preserves D's requirements across all loop iterations.
Per-seed coverage failures are reported without requiring a strict contrast
pair for every seed. Both arms apply the same AST/executor checks and frozen-base
neighbourhood probes. Reaching the count alone does not make a bank ready for
training if variant coverage fails. The student input is built from
whitelisted history/tool fields; diagnosis, variant labels, gold arguments,
validation metadata, and costs remain outside the rendered prompt.

Every teacher request goes through `src/appworld_teacher.py` with explicit
official endpoint enforcement, Luna, Flex, `reasoning_effort=none`, and zero
automatic retries. These options follow the [official Luna model documentation](https://developers.openai.com/api/docs/models/gpt-5.6-luna)
and [Chat Completions reference](https://developers.openai.com/api/reference/python/resources/chat/subresources/completions/methods/create).
The append-only `teacher_ledger.jsonl` records reservations, raw-response events,
and final dispositions. `teacher_raw/` retains every request and provider
envelope, returned model ID, fingerprint when supplied, request/response IDs,
returned service tier, complete usage details, and failure/truncation status.
An exercise's `cost_call_id` links to its full generation charge; shared batches
are counted once, not fractionally estimated per retained example. Hidden
reasoning, cached input, discarded generations, and purchased-but-unused
training data remain charged. USD is left unpriced rather than inferred from
the requested tier; the returned tier can differ.

A network timeout or error without provider usage cannot be assigned an exact
cost. It is recorded as unresolved, retains its reservation, and blocks all
further purchases. Obtain authoritative provider usage before continuing; never
replace missing usage with zero or delete that attempt. Reconciliation is an
append-only ledger row with the same `call_id`, `event: "reconciled"`, the exact
Chat Completions `usage` object, and a billing evidence reference. Do not rerun
the unresolved request. Known failed attempts remain paid and are reused as
failed inventory on rerun. There is no unbudgeted repair path.

The trainer ports the useful LoRA/encoding pattern from the read-only base
repository's `src/bfas/rtd/baselines/paper_train.py`. It implements this experiment's
different loss and exposure rules: BF16, r16/alpha32/dropout0, all actual attention
and MLP linear projections, AdamW, micro-batch 1, and checkpointing. The GPT-6
design's one pass at learning rate 1e-5 remains the default. `train --passes N`
(positive integer, default 1) and `--learning-rate LR` (positive finite float,
default 1e-5) allow a larger training dose because BFCL supervised spans are
short: 611 supervised tokens yield only two updates at 512 tokens per step.
For example, use `--passes 20 --learning-rate 1e-4 --tokens-per-step 512` on
**both** C and D training commands to obtain 12,220 supervised token exposures
and 40 optimizer steps per arm when the common cap is 611 tokens.

No GPU is held by an inference server during training. The common supervision
cap is `min(16000, usable_C_tokens, usable_D_tokens)` **per pass**. Each pass
starts from the arm's full usable exercise set in ID order and reshuffles it
using `train_seed + pass_index` (zero-based), preserving the original first-pass
order for the default training seed. Each arm receives exactly the common cap per pass and the same number
of passes. Updates accumulate `--tokens-per-step` supervised tokens (default
512), except the final update of **each pass**; accumulation resets at pass
boundaries while AdamW state persists. Sequences can cross update boundaries
without repeating token targets within a pass.
Whole contexts use 4k or 8k; over-8k examples are excluded and audited. A final
partial supervision span can meet the common cap without appending an EOS.

Before either arm trains, `training_plan.json` freezes the common per-pass cap,
passes, learning rate, tokens per step, exercise/encoded hashes, exclusions,
row IDs, and both arms' canonical schedules for the **split seed**. The plan
stays unchanged across training seeds, including when a non-default seed trains
first. Each training output's `schedules.json` records its actual pass numbers,
shuffle seeds, row orders, and exact token segments; row indices refer to the
plan's `row_ids`. Changing either arm's dose or data across arms **or seeds**
after freezing raises `C/D exposure plan changed after it was frozen`.
Choose the dose before starting either arm; existing training directories
cannot be resumed or overwritten in place.

`audit-loss --arm C|D [--train-seed K] [--checkpoint /path/to/adapter]`
decomposes the training objective on every valid, encoded exercise in that arm.
For example, with the caller's `CUDA_VISIBLE_DEVICES` selecting the GPU:

```bash
"$TRAIN_PY" tools/mech_bfcl.py audit-loss --run-dir "$MECH_RUN" --arm C
"$TRAIN_PY" tools/mech_bfcl.py audit-loss --run-dir "$MECH_RUN" --arm D --train-seed 1
```

The default checkpoint is the selected seed's `training[_sK]/adapter`. The
audit uses the saved model/tokenizer paths and projection chunk size; optional
`--model-path`, `--tokenizer`, and `--position-chunk` overrides support relocated
caches and memory limits. It verifies the exercise and encoding hashes against
`training_plan.json`, reuses training's native encoding, causal target positions,
frozen hidden states, and full-vocabulary softcapped projection, and runs BF16
on CUDA in evaluation mode under `torch.no_grad()`. Every encoded target is
counted once; the training cap, repeated passes, and shuffle order do not weight
this exercise census. Invalid and over-context rows remain excluded.

The sole result written is the selected training directory's `loss_audit.json`,
including with an explicit checkpoint override. It contains per-token values,
per-exercise means, and token-weighted overall means for adapter-disabled
**before** and adapter-enabled **after**: teacher NLL, reference cross-entropy,
base entropy, forward KL, the 0.5/0.5 mixture, and target-argmax correctness.
The console prints overall before/after rows and one line per exercise, plus
the fraction of targets the base already predicts. Losses use natural logs.
For the frozen base distribution `q`, reference CE is `H(q) + KL(q || p)`;
its floor is `H(q)`, so a small mixture alone does not establish learning.
`steps.json` records mixtures during optimization on each step's token subset;
the audit records both fixed models on the entire exercise set. It does not
rewrite training artifacts, protocol files, locks, or evaluation results.

`train --train-seed K` repeats training on the same generated C/D exercises.
It defaults to the split seed, preserving existing behavior. It controls the
per-pass shuffle, PyTorch CPU/CUDA manual seeds, adapter initialization, and
any dropout randomness (configured LoRA dropout remains zero). It does not
change support/calibration selection, diagnoses, generated banks, held-out
sets, the frozen split, or the layer-3 subset. Keep `--seed` and `--splits`
unchanged; `--seed` is still the split and inference seed.

For the default training seed, output remains `<run-dir>/<arm>/training/`.
For any different seed K, output is `<run-dir>/<arm>/training_sK/`, including
its `adapter/`, `config.json`, `metrics.json`, and `schedules.json`. Both config
and metrics record `train_seed`; all seeds retain the same `training_plan_hash`.
The same seed cannot overwrite an existing training directory.

`evaluate --train-seed K` selects the corresponding adapter. Non-default seeds
must be served as `mech-<arm>-sK`, with `/models` pointing to that seed's local
`training_sK/adapter/`. Their three-layer evaluations go to
`evaluation/<arm>-sK/<repeat>/`; default seeds keep `evaluation/<arm>/<repeat>/`.
The evaluation protocol and full harness metadata record `train_seed` while
decoding still uses the split seed. A changed training seed in an existing
fingerprint rejects reuse. Legacy default-seed fingerprints without the field
remain reusable. Base ignores `--train-seed`, keeps its served name and paths,
and records `train_seed: null` in the evaluation protocol; base harness metadata
stays unchanged. `--repeat repeat` repeats evaluation of the selected checkpoint;
it does not retrain it or contribute another training seed to reporting.

To add seed 1 after the default C/D training and evaluations above, reuse
`MECH_RUN`, the shell helpers, and the existing split. There are no new teacher
calls. Read the exact dose from the frozen plan rather than assuming defaults:

```bash
MECH_TRAIN_SEED=1  # Must differ from the split seed for these suffixed paths.
read -r MECH_PASSES MECH_LR MECH_TPS MECH_SPLIT_SEED < <(
  "$TRAIN_PY" -c 'import json, sys; p=json.load(open(sys.argv[1])); print(p["passes"], p["learning_rate"], p["tokens_per_step"], p["seed"])' \
    "$MECH_RUN/training_plan.json"
)
for arm in C D; do
  "$TRAIN_PY" tools/mech_bfcl.py train --run-dir "$MECH_RUN" --arm "$arm" \
    --seed "$MECH_SPLIT_SEED" --train-seed "$MECH_TRAIN_SEED" \
    --passes "$MECH_PASSES" --learning-rate "$MECH_LR" --tokens-per-step "$MECH_TPS"
  setsid "$VLLM" serve google/gemma-4-12B-it \
    --served-model-name google/gemma-4-12B-it \
    --dtype bfloat16 --max-model-len 32768 --gpu-memory-utilization 0.85 \
    --enable-lora --max-lora-rank 16 \
    --lora-modules "mech-$arm-s$MECH_TRAIN_SEED=$MECH_RUN/$arm/training_s$MECH_TRAIN_SEED/adapter" \
    --port 8901 >"$MECH_RUN/serve-$arm-s$MECH_TRAIN_SEED.log" 2>&1 &
  export MECH_SERVING_PID=$!
  wait_server
  "$BFCL_PY" tools/mech_bfcl.py evaluate --run-dir "$MECH_RUN" --arm "$arm" \
    --seed "$MECH_SPLIT_SEED" --train-seed "$MECH_TRAIN_SEED" \
    --base-url http://127.0.0.1:8901/v1
  stop_server
done
"$BFCL_PY" tools/mech_bfcl.py report --run-dir "$MECH_RUN" --seed "$MECH_SPLIT_SEED"
```

The API teacher provides a demonstration, not Gemma-vocabulary logits. For each
demonstration token `y`, the distillation target is explicitly
`q = 0.5 * one_hot(y) + 0.5 * p_base`; loss is `-sum(q * log(p_adapter))`.
`p_base` comes from the same backbone with the adapter disabled and gradients
off. The backbone's hidden states are projected in chunks of 32 supervised
positions over the entire vocabulary, including Gemma's final softcap.
Chunk graphs are freed after computing the hidden-state gradient, followed by
one decoder backward pass. No full-sequence vocabulary tensor or top-k target
cache is retained. Both arms record passes, learning rate, tokens per step, and
training seed in their training directory's `config.json` and `metrics.json`. Metrics record
`supervised_tokens_per_pass` as the common cap and `supervised_tokens` as total
exposures across all passes (`passes * supervised_tokens_per_pass`), along with
input tokens, optimizer updates, wall time, and memory. Config records exact
module names and model revision; `steps.json` labels every update with its pass.
The report shows training seed, passes, learning rate, optimizer steps, and total
supervised exposure for every variant with a completed `main` evaluation.

Local validation checks format, official AST argument validity, and available
official execution. A valid AST is **not** independent verification of teacher
task semantics. Stateless tool implementations may be unavailable; abstention
wording is not semantically graded. Those limits are attached to every exercise.
For executable cases the evaluator compares official final state and execution
responses, allowing equivalent calls rather than requiring exact teacher text.
Layer 2 checks the next local decision in an unedited natural context; layer 3
runs the complete official task. Tool-call totals, execution/format errors,
repeats, truncations, and early stops are saved. Full-task early stop is explicitly
a proxy: a failed task whose final decision contains no call.

Final artifacts: `report.md` and `report.json`, `paired/{base_vs_C,base_vs_D,C_vs_D}.json`,
per-category breakdowns, all four correctness outcomes, and paired 95% bootstrap
intervals over independent parent groups. The estimate weights each parent
equally; item-weighted deltas are also provided. Single-parent intervals are
unavailable. The report includes the five section-nine interpretations and a
leave-one-parent-out check before suggesting a broad positive targeting signal.

The mechanism table and category breakdowns include every completed `(arm,
train_seed)` variant, reading its corresponding training metrics. Default C/D
rows retain their labels and infer the split seed for legacy metrics. Base/C/D
default evaluations remain required. Seed K adds `base_vs_C_sK`,
`base_vs_D_sK`, `C_sK_vs_D_sK`, and within-arm `C_s0_vs_C_sK` /
`D_s0_vs_D_sK` comparisons (replace 0 by the split seed for a different split).
Within-arm signed deltas and catches/regressions measure training noise;
`mean_absolute_item_delta` is the fraction of questions whose correctness flips.

With at least two seeds completed for both C and D, `C_vs_D_pooled` averages
correctness **per question over the same matched seeds**, subtracts C from D,
then averages question deltas within parents and weights parents equally.
The parent bootstrap operates on those means; training seeds do not multiply
the item count or independent-parent count. Pooled rows retain fractional
`left_mean_correct`, `right_mean_correct`, and `delta`; binary outcome counts
are inapplicable and shown as unavailable. The report records
`pooled_training_seeds` and `unpooled_training_seeds`; an unmatched seed remains
in the table and its available comparisons, but is excluded from both arms'
pool. All comparisons require identical question IDs, parent/category labels,
and layer-3 scope. Available evaluation protocols must also share held-out
hashes, split hashes, and decoding settings.

The interpretation uses the pooled C/D comparison when available. Read it
alongside the individual seed effects and within-arm noise: its intervals
measure parent uncertainty conditional on the observed training seeds, not
uncertainty over future training seeds. Generated-token cost is shared across
seed variants and must not be summed again as new generation spending.

CPU-only verification (all model/server work is mocked or uses tiny CPU tensors):

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_mech_bfcl*.py
```

## Round 2: coverage expansion

R2 reuses the five R1 diagnoses and **every** accepted R1 exercise. R1 had 22
exercises per arm, including repeated teaching signals, and no accepted
multi-turn exercises. The commands below use a new run directory outside
`results/`. The sibling R1 run is read-only; its frozen training plan,
exercises, heldout set, ledgers, and evaluations are never updated. These are
execution instructions for a later authorized experiment: implementation and
unit verification do not run the teacher, student inference, or GPU training.

`prepare-r2` copies support/calibration evidence, seeds, diagnoses, and the
byte-identical `heldout.json` and audit. `round2.json` records input hashes and
R1 bank identities. It imports **only shared** R1 teacher ledger events and
their raw envelopes. This preserves the single 8,000-output-token allowance
for diagnosis, heldout preparation, and confirmation. The observed R1 shared
spend was 4,363 tokens, leaving at most 3,637 for confirmation. It imports no
training plan, adapter, arm generation ledger, or evaluation directory.
Use copies rather than symlinks for these inputs. Preparation can be verified
again with the same command; changed source or copied evidence is rejected.

`generate --arm C|D --target 64 --extend-from <R1 exercises.json>` freezes the
source path, content hash, and all original exercise IDs in
`<arm>/extension.json`. The output starts with the original rows, with only
`origin="round1"` and `round1_exercise_id` added. Historical duplicates remain
in this frozen subset. New rows have `origin="round2"` and noncolliding IDs.
Both arms must use extension mode in a prepared R2 run. `--target` is an alias
of `--target-exercises`; the common 24,000-output-token cap, target, seed
contexts, and coverage rules are frozen in `generation_protocol.json`.

Each new candidate is deduplicated against the entire accepted pool of its
seed and direction, using the normalized correct call **set**. Call order,
dictionary-key order, duplicated calls, and no-call wording do not create
new teaching signals; argument values and array order still matter. The
existing full-exercise deduplication also remains. Both sides of the following
relation must be represented **within each direction** (`condition`,
`surface`, `neighbour_correct`) of **each** diagnosed seed:

| Interface / R1 seeds | Positive side | Negative side |
| --- | --- | --- |
| Archival/core memory, seed_0 and seed_3 | Archival retrieval/search needed | Core memory or visible observations suffice |
| Parallel calls, seed_2 | Several distinct calls genuinely needed | Exactly one call necessary |
| Vehicle prerequisites, seed_1 | Authorized door-lock prerequisite required | Door-lock prerequisite unnecessary or unauthorized |
| Further action/completion, seed_4 | A further tool action required | No further action; answer or clarify |

These interface-derived predicates and authoring rules are shared by C/D.
Only D sees the diagnosis and observed failure. Teachers provide condition
side and evidence as metadata; the validator checks that the side agrees
with the correct calls. Missing seed/direction sides get generation priority,
and the remaining target slots are reserved for them. All candidates still
pass native rendering and the same official AST/executor validation;
neighbours still require a correct frozen-base response. `surface` requires a
new call set within its direction: paraphrasing the same target does not
expand R2 coverage. A changed requirement may be expressed in the last user
request; tools, actual state, and historical observations cannot be invented.

The loop stops at 64 retained examples, or when the next full 3,000-token
reservation cannot fit under the arm cap. Invalid proposals, paid failures,
repaired proposals in later batches, duplicates, and unused proposals are all
charged. Resuming replays cached purchases and probes. No unbudgeted repair
call exists. `generation_audit.json` retains the existing rejection reasons
and new coverage/deduplication failures. `generation.json` reports actual
spend per arm, origins, R1 IDs, counts, shortfall, snapshot exclusions, and
`decision_coverage` down to missing sides. `ready` requires every seed and
direction to cover both sides, even if the count target is reached. A
budget-limited smaller pool can be ready when coverage is complete; training
does not fill missing counts by repetition.

### Multi-turn executor reconstruction

Snapshots now losslessly pack and restore `random.Random.getstate()`, which
caused both R1 multi-turn exclusions. For a missing legacy multi-turn
snapshot, reconstruction loads the official entry and ground truth, creates
the official executor classes through `execute_multi_turn_func_call`, loads
`initial_config`, and replays all prior turns' ground-truth calls. The official
long-context/composite flag is preserved. Prior observed student calls are
replayed separately: public executor state and RNG state must agree with
the gold replay. Calls already observed in the current failure turn are then
replayed on the gold state, and their outputs must match the captured tool
observations. The current turn's future gold calls are **never** replayed.
Private executor instances are removed after success or failure.

A CPU check against the actual read-only R1 evidence successfully reconstructed
`multi_turn_base_79:2` (VehicleControlAPI) and
`multi_turn_long_context_175:7` (TwitterAPI/TravelAPI). R1 seed files remain
unchanged; reconstructed contexts and provenance belong to R2 generation.
New captures retain RNG state directly. Remaining exclusions are explicit:
unavailable official entry/gold, turn alignment failure, errors in prior gold
replay, divergent prior student state/RNG, mismatched current-turn outputs,
or other unserializable executor attributes. Live web and filesystem memory
prerequisite executors are excluded from this **multi-turn reconstruction**;
existing captured memory snapshots remain executable as before. Generated
examples can still fail normal AST/execution, base-neighbour, or 8k training
context checks. Teacher task semantics, including necessity of an action,
remain teacher-authored; successful execution is not an independent proof of
natural-language intent.

### Confirmation and dose

Run `confirm --target 24` before R2 evaluation. It pre-registers at most eight
independent calibration parents and three fixed slots per parent: an anchor,
a changed-condition request, and an unchanged-decision surface paraphrase.
The source selection is deterministic under the split seed, excludes the
whole support split and every training exercise's task/parent, and takes at
most one source task per parent. No adapter score or baseline success filters
selection. `confirmation_plan.json` freezes sources, exclusions, roles, pair
relations, and equal per-parent output reservations from the shared remaining
budget **before** teacher authoring. Parent shortages and authoring failures
produce explicit shortfalls without replacement tasks or extra spending.

Each member receives the same materialization, native rendering, and official
validation as exercises. A valid condition pair has different natural requests,
opposite relation sides, and different correct call sets; a valid surface pair
has different wording and the identical correct call set. Pair metadata and
authoring cues never enter student prompts. Invalid or unavailable members
are audited; valid surviving members of invalid pairs remain independent
items. The existing `heldout` layer 1 and natural layer 2 stay exactly as they
were, including their original IDs and hash.

Evaluation saves separate `confirmation.jsonl` and `confirmation.json` under
each variant/repeat, including per-item correctness, valid/invalid pairs,
unpaired items, and the fraction of valid pairs with **both members correct**.
As a diagnostic only, it teacher-forces the exact native target tokens through
the served model's completion `echo`/`logprobs` interface, records sequence
probability and log probability, and compares each adapter against base.
This is the probability of the particular authored call sequence (or authored
no-call text), not summed probability over all equivalent calls. A server
that cannot return prompt token log probabilities records an explicit
unavailable diagnostic; correctness remains scored. Raw responses and usage
are retained. Reports recompute the deltas regardless of evaluation order.

`train --supervised-budget 15900` derives
`passes = max(1, round(15900 / supervised_tokens_per_pass))`, using Python's
round-to-even convention and the existing common C/D per-pass token cap.
It overrides `--passes`. Omitting the budget preserves the original behavior.
Actual exposure is `passes * supervised_tokens_per_pass`; its rounding error
is normally at most half a pass (the minimum-one-pass rule can overshoot a
very small budget). Use the **same** learning rate and `--tokens-per-step` in
C/D. The full-vocabulary 0.5 teacher/0.5 frozen-base mixture, normalization by
supervised tokens per optimizer step, and accumulation reset per pass are
unchanged. `training_plan.json` freezes the requested budget as well as the
derived passes and hashes of both full pools/encodings. Changing the budget
even when it rounds to the same number of passes, or changing pool provenance,
fails the cross-arm/cross-seed guard. R1's plan is unaffected.

Every new training run saves `adapter_mid` immediately after
`ceil(total_optimizer_steps / 2)` optimizer steps, and saves `adapter` at the
end. Both paths and their actual optimizer-step counts and token exposures
are in `metrics.json.checkpoints`; midpoint exposure is not guessed as half
the final token count. `evaluate --checkpoint mid` requires the exact alias
`mech-C-mid` / `mech-D-mid` (or `mech-C-sK-mid` for a training-seed repeat),
checks its `/models` adapter root, and writes `evaluation/C-mid/<repeat>/`
or `evaluation/C-sK-mid/<repeat>/`. `--checkpoint end` remains the default,
with the original alias, adapter path, and evaluation directory. Base keeps
its original name/path and ignores checkpoint selection.

### R2 commands, end to end

Do not run R1 rollout or diagnosis again. This sequence leaves both worktrees'
`results/` contents unchanged; choose a persistent new run directory for actual
training artifacts. The server/training commands require the previously
described A100 and installed environments.

```bash
cd /home/xueqi/hq/projects/tc-alignment-mech2
set -euo pipefail
export PYTHONPATH=src:. PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export BFAS_TEACHER=openai/gpt-5.6-luna BFAS_OPENAI_SERVICE_TIER=flex
export OPENAI_BASE_URL=https://api.openai.com/v1
# OPENAI_API_KEY is supplied by the existing credential setup.
MECH_R1=/home/xueqi/hq/projects/tc-alignment-mech/results/mech_bfcl
MECH_R2="$PWD/runs/mech_bfcl_r2"
BFCL_PY="$PWD/envs/bfcl/.venv/bin/python"
TRAIN_PY="$PWD/.venv/bin/python"
VLLM="$PWD/envs/vllm-serve/.venv/bin/vllm"

CUDA_VISIBLE_DEVICES='' BFCL_PROJECT_ROOT=/tmp/mech-bfcl-cpu-runtime \
  "$TRAIN_PY" -m pytest -q -p no:cacheprovider tests/test_mech_bfcl*.py
"$BFCL_PY" tools/mech_bfcl.py prepare-r2 --run-dir "$MECH_R2" --round1-run-dir "$MECH_R1"
# Confirmation authoring requires the teacher but no GPU or student server.
"$BFCL_PY" tools/mech_bfcl.py confirm --run-dir "$MECH_R2" --target 24

export CUDA_VISIBLE_DEVICES=0  # Select the assigned A100.
wait_server() {
  until curl --fail --silent http://127.0.0.1:8901/v1/models >/dev/null; do
    kill -0 "$MECH_SERVING_PID"
    sleep 2
  done
}
stop_server() {
  kill -- -"$MECH_SERVING_PID"
  wait "$MECH_SERVING_PID" || true
  unset MECH_SERVING_PID
  while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do
    sleep 2
  done
}
setsid "$VLLM" serve google/gemma-4-12B-it \
  --served-model-name google/gemma-4-12B-it --dtype bfloat16 \
  --max-model-len 32768 --gpu-memory-utilization 0.85 --port 8901 \
  >"$MECH_R2/serve-base.log" 2>&1 &
export MECH_SERVING_PID=$!
wait_server
for arm in C D; do
  "$BFCL_PY" tools/mech_bfcl.py generate --run-dir "$MECH_R2" --arm "$arm" \
    --target 64 --max-output-tokens 24000 \
    --extend-from "$MECH_R1/$arm/exercises.json" --base-url http://127.0.0.1:8901/v1
done
# Inspect both generation.json files: ready must be true; count/spend/shortfall
# and every seed/direction's missing_sides are explicit. No count padding.
"$BFCL_PY" tools/mech_bfcl.py evaluate --run-dir "$MECH_R2" --arm base \
  --base-url http://127.0.0.1:8901/v1
"$BFCL_PY" tools/mech_bfcl.py evaluate --run-dir "$MECH_R2" --arm base --repeat repeat \
  --base-url http://127.0.0.1:8901/v1
stop_server

for arm in C D; do
  "$TRAIN_PY" tools/mech_bfcl.py train --run-dir "$MECH_R2" --arm "$arm" \
    --supervised-budget 15900 --learning-rate 1e-4 --tokens-per-step 512
  setsid "$VLLM" serve google/gemma-4-12B-it \
    --served-model-name google/gemma-4-12B-it --dtype bfloat16 \
    --max-model-len 32768 --gpu-memory-utilization 0.85 \
    --enable-lora --max-lora-rank 16 --max-loras 2 \
    --lora-modules "mech-$arm=$MECH_R2/$arm/training/adapter" \
                   "mech-$arm-mid=$MECH_R2/$arm/training/adapter_mid" \
    --port 8901 >"$MECH_R2/serve-$arm.log" 2>&1 &
  export MECH_SERVING_PID=$!
  wait_server
  for checkpoint in mid end; do
    "$BFCL_PY" tools/mech_bfcl.py evaluate --run-dir "$MECH_R2" --arm "$arm" \
      --checkpoint "$checkpoint" --base-url http://127.0.0.1:8901/v1
  done
  stop_server
done
"$BFCL_PY" tools/mech_bfcl.py report --run-dir "$MECH_R2" --round1-run-dir "$MECH_R1"
```

The report source defaults to `round2.json`'s R1 run when prepared this way.
It reads every completed R1 main evaluation, including `C-sK`/`D-sK` seed
repeats that are present at report time, and requires R2 C/D at midpoint and
end. One table contains R1 arms/repeats, R2 midpoint/end arms, exposure at each
checkpoint, and confirmation accuracy/both-sides scores. R1 confirmation
cells are unavailable, not invented. Existing paired-by-parent C/D and
seed-pooled statistics remain; midpoint/end and cross-round within-arm
comparisons are added. R1 statistics are written under **R2**'s `paired/round1/`.
Confirmation has separate parent-cluster intervals for per-item correctness
and valid-pair success; shared anchors do not multiply independent parents.
Per-item probability changes are diagnostic only. Generated-token costs in
the R2 rows are actual **new** arm spending; R1 generation and shared costs
are retained separately, and neither checkpoint rows nor seed repeats imply
another generation purchase.
