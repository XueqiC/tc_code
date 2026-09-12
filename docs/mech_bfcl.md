# BFCL first-round mechanism validation

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

All inference uses T=0.001, greedy `top_k=1`, seed 0, and explicit
`enable_thinking=False`, through raw completions. No native stop/handoff marker
is inserted into the short supervision target. The renderer reuses
`tools/bfcl_pool_render_gemma4.py`, matching the luna bank's dotted tool names,
argument quoting, and prompt conventions. The bank supplies conventions only;
none of its historical examples or cached-content cost estimates are charged
as new teacher calls or used as new training examples. `--served-model` can
override the adapter alias; `/models` must identify the matching local adapter.
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
and MLP linear projections, AdamW at 1e-5, micro-batch 1, checkpointing, one pass.
No GPU is held by an inference server during training. The common supervision
cap is `min(16000, usable_C_tokens, usable_D_tokens)`, frozen before either arm
trains. Updates accumulate exactly 512 supervised tokens (except the final
update); sequences can cross update boundaries without repeating token targets.
Whole contexts use 4k or 8k; over-8k examples are excluded and audited. A final
partial supervision span can meet the common cap without appending an EOS.

The API teacher provides a demonstration, not Gemma-vocabulary logits. For each
demonstration token `y`, the distillation target is explicitly
`q = 0.5 * one_hot(y) + 0.5 * p_base`; loss is `-sum(q * log(p_adapter))`.
`p_base` comes from the same backbone with the adapter disabled and gradients
off. The backbone's hidden states are projected in chunks of 32 supervised
positions over the entire vocabulary, including Gemma's final softcap.
Chunk graphs are freed after computing the hidden-state gradient, followed by
one decoder backward pass. No full-sequence vocabulary tensor or top-k target
cache is retained. Both arms record input tokens, actual supervised tokens,
optimizer updates, wall time, memory, exact module names, and model revision.

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
