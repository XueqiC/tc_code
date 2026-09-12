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
are reusable; incomplete harness directories are preserved and rejected rather
than silently mixing attempts. Use a new run directory for an abandoned run.

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

The 256-question evaluation subset uses the same expansion. Prerequisites do
not enlarge the scored subset. Each harness batch's `cost.json` totals captured
generations and exact input/output tokens, with requested/prerequisite
breakdowns; evaluation exposes these totals under `cost.json`'s `layer_3` key.
Completed queries in failed episodes are included. Missing result and completed
trajectory IDs are reported together after the batch and retained cost records.

Run this sequence from the project root on **one A100 80GB**. The cached model
and tokenizer, BFCL environment, vLLM environment, and training environment must
already be installed. `envs/bfcl/.venv/bin/python` has the harness's provider
imports; `.venv/bin/python` has the training dependencies. The BFCL web-search
executor needs the same runtime credentials/resources as the existing harness.
Keep any required keys in the environment, never in command-line arguments.

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
MECH_RUN="$PWD/results/mech_bfcl"
BFCL_PY="$PWD/envs/bfcl/.venv/bin/python"
TRAIN_PY="$PWD/.venv/bin/python"
VLLM="$PWD/envs/vllm-serve/.venv/bin/vllm"
mkdir -p "$MECH_RUN"

"$TRAIN_PY" -m pytest -q tests/test_mech_bfcl*.py
"$BFCL_PY" tools/mech_bfcl.py splits

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
"$BFCL_PY" tools/mech_bfcl.py rollout --base-url http://127.0.0.1:8901/v1
"$BFCL_PY" tools/mech_bfcl.py diagnose
"$BFCL_PY" tools/mech_bfcl.py generate --arm C --base-url http://127.0.0.1:8901/v1
"$BFCL_PY" tools/mech_bfcl.py generate --arm D --base-url http://127.0.0.1:8901/v1

# Every evaluate invocation runs all three layers. Repeat base once before
# training; report records paired correctness stability, including web effects.
"$BFCL_PY" tools/mech_bfcl.py evaluate --arm base --base-url http://127.0.0.1:8901/v1
"$BFCL_PY" tools/mech_bfcl.py evaluate --arm base --repeat repeat --base-url http://127.0.0.1:8901/v1
stop_server

"$TRAIN_PY" tools/mech_bfcl.py train --arm C
setsid "$VLLM" serve google/gemma-4-12B-it \
  --served-model-name google/gemma-4-12B-it \
  --dtype bfloat16 --max-model-len 32768 --gpu-memory-utilization 0.85 \
  --enable-lora --max-lora-rank 16 \
  --lora-modules "mech-C=$MECH_RUN/C/training/adapter" \
  --port 8901 >"$MECH_RUN/serve-C.log" 2>&1 &
export MECH_SERVING_PID=$!
wait_server
"$BFCL_PY" tools/mech_bfcl.py evaluate --arm C --base-url http://127.0.0.1:8901/v1
stop_server

"$TRAIN_PY" tools/mech_bfcl.py train --arm D
setsid "$VLLM" serve google/gemma-4-12B-it \
  --served-model-name google/gemma-4-12B-it \
  --dtype bfloat16 --max-model-len 32768 --gpu-memory-utilization 0.85 \
  --enable-lora --max-lora-rank 16 \
  --lora-modules "mech-D=$MECH_RUN/D/training/adapter" \
  --port 8901 >"$MECH_RUN/serve-D.log" 2>&1 &
export MECH_SERVING_PID=$!
wait_server
"$BFCL_PY" tools/mech_bfcl.py evaluate --arm D --base-url http://127.0.0.1:8901/v1
stop_server

"$BFCL_PY" tools/mech_bfcl.py report
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
never fabricated or edited. At most eight support failures seed each arm;
each seed requests nine examples, with at most 24,000 output tokens per arm.
Counts yield to budget and validation. Failed neighbours and malformed or
inconsistent examples are retained in the generation audit. An arm may have
fewer than 48 valid examples; this is reported explicitly and training never
repeats examples to compensate. No replacement seeds are mined from evaluation.

C and D receive identical task/interface/state context and format requirements.
Only D receives the current student response, execution failure, and testable
gap hypothesis. Prior natural history is identical and may contain imperfect
student actions. D must retain condition, surface, and base-verified neighbouring
examples globally; per-seed coverage failures are reported without requiring a
strict contrast pair for every seed. Both arms apply the same AST/executor
checks and frozen-base neighbourhood probes. The student input is built from
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
