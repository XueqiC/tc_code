# ALFWorld K=32 REPAIR-CE material

The three tools produce material for the existing `pi1_ce` trainer. They do not
change the trainer or `pi1.load_bank`. Collection and verification run on CPU;
the rollout client talks to an **already served** merged student through the
same launch-bound `VLLMBackend` contract as evaluation. A real rollout still
uses the server's inference compute. No server is launched by these tools.

## Takeover and supervision

Run every one of D0's 32 historical support tasks, including tasks without a
usable D0 package, in lexicographic task-ID order. Read both frozen
`public/reset_requests.json` and `public/reset_states.json`; compare the live
reset observation and admissible commands before generating. Use the unchanged
`alfworld_evaluation.official_episode` with a `RealStepper` bridge and
`EpisodeParserDiagnostics`: temperature 0, 256 generated tokens per action,
40 environment steps. Its internal evaluation-loop guard still names
`valid_seen`; the injected worlds, persisted records and campaign identity are
explicitly **train**. No evaluation task selection or second parser is used.
The server model/tokenizer identity must match the local paths supplied to the
client. The client does not load model weights or initialize CUDA.

Each hashed task filename contains `task_id`, `request_state_hash` (canonical
hash of the frozen reset request), `reset_state_hash` (the complete frozen
`FullState` hash), `commands`, post-step `observations`, `steps`, `success`,
`horizon_reached`, and the evaluation `turns`. Parser fields are exactly
`parser_fallback`, `parser_candidate`, and, only on fallback,
`parser_fallback_reason`. The index binds every file hash, source manifest,
server identity and completion status. Environment failures propagate; an
incomplete index cannot be used for purchase.

Only completed episodes with `success: false` yield a request. Indices are
**zero based**. Scan from the first executed command and select the first index
`k` where either:

1. The command equals any earlier executed command (exact, case-sensitive text).
2. Its post-command observation starts with `Nothing happens.` after
   case folding. No whitespace stripping or substring search is applied.

If both occur at the same index, `rule` is `repeated_command` and `triggers`
records both. Otherwise `rule` is `nothing_happens`; when neither occurs,
`rule` is `midpoint` and `k = steps // 2`. Successful episodes never yield a
request, even if they contain repetitions or Nothing-happens observations.

Replay **`commands[0:k]`**, excluding the triggering command, through
`replay_commands` on the original frozen reset request. Compare the replay's
full history against the student's saved prefix before any paid call. `k=0`
means the reset state. Execute student parser fallback commands literally,
including `look` when it was absent from the admissible list. Only that masked
student prefix has relaxed admissibility; teacher commands remain strict.
The teacher continues from the reached cursor in the same live worker, using
D0's `prompt_messages` ReAct format, for at most **`40-k`** new commands.

A successful purchase is verified in a fresh worker by replaying student
prefix plus teacher suffix and requiring terminal environment success.
`payload_kind` stays `teacher_react_turns`. The package stores student commands
as `student_prefix_commands`; `commands`, `teacher_react_turns` and `behaviors`
contain only the teacher suffix. Verification retains all reset/prefix/suffix
states. Consequently `load_bank` creates one row per teacher turn, with student
commands and observations only in the masked prompt. Student generated prose
is never a target. Existing packages retain their original format and strict
verification semantics.

For every repair request, record `k`, `rule`, trigger flags, remaining horizon,
actual takeover state/hash, teacher commands/responses, usage, sweep share,
filter flag and final availability. Sweep share is the number of **teacher
suffix** commands matching the existing sub-bank regex
`^(go to|open|close) (cabinet|drawer) \d+$`, divided by all teacher suffix
commands. Matching is case-sensitive. Empty suffixes have share zero. Filter
strictly when share **> 0.5**; exactly 0.5 is retained.

## Accounting and frozen banks

The REPAIR collector refuses an existing collection, bank or registration.
The D0 collection identity supplies the teacher and the three prices; its
support hash and ledger hash are checked against the D0 bank. Explicit budget
caps are recorded. The collector uses `PoolAdapter.purchase`, `Budget`,
write-ahead `usage.jsonl`, `teacher_ledger.jsonl`, rate-limit retries, reserved
charges for missing usage, and orphan-attempt recovery. Failed calls still
cost money. One suffix attempt is made per failed student episode; HTTP 429
retries follow the existing transport policy. No extra trajectory retries or
new takeover decisions are made.

The new collection contains:

- `identity.json`, `tasks.json`: frozen inputs, rules, prices and per-task decisions.
- `requests/<task-hash>.json`: reached context, suffix, costs and availability.
- `usage.jsonl`, `teacher_ledger.jsonl`: durable purchase accounting.
- `packages/<query-id>.json`: all attempted packages, including failures and
  `unavailable_reason: sweep_filtered` packages, with paid evidence intact.
- `candidate_sets.json`, `summary.json`: totals and the exported-bank binding.

Only usable packages go into the REPAIR frozen bank. Its support lists all 32
historical tasks and parents (`m=32`), while `training_task_ids` lists exactly
those with usable packages. A signed `training_selection: usable_packages`
marker makes this distinction auditable; existing support manifests keep their
old semantics. If no package survives, the empty bank and accounting are still
written, but no pi1 YAML is written because the unchanged recipe requires
positive demonstration and token counts. Inspect `summary.json` for that case.
An interrupted collection is not resumed by this tool: preserve its paid
journals; do not point a new purchase at that directory.

Union inputs must share the same frozen support, reset requests, environment
identity and reset states. REPAIR preserves D0's identity for this reason.
Every selected sealed package is copied byte for byte. Complete source
historical accounting is retained without prorating unselected purchases.
`--mixing all` retains all usable packages, deduplicating byte-identical IDs.
`--mixing tokens-1:1` requires disjoint IDs, keeps the smaller token side whole,
and selects whole packages from the larger side. The exact subset-sum search
minimizes distance to the smaller-side total, then the selected total; seeded
package ordering breaks remaining ties. The required integer bound is
`95*T <= 100*selected_larger <= 105*T`. It fails if no whole-package subset
meets that bound. No reply is truncated, duplicated or reweighted.

The union manifest records source hashes, mixing mode, seed, exact per-package
token costs, original/selected totals, and IDs per source. Token counts for
mixing and registration use the same rule as training preflight:

```python
sum(len(pi1.encode_teacher_turn(tokenizer, row, 32768)['target_ids']) for row in rows)
```

This includes each complete teacher reply plus native `<turn|>` ID 106, and
excludes masked prompts. YAML registration uses
`configs/rtd/pi1_alfworld_k32_kang.yaml` with only bank fields/counts changed.
Registration and preflight load the tokenizer locally and use no GPU.

## Real run: rollouts → repair → union → preflight

Run from this worktree. Set `STUDENT_MODEL` and `SERVER_JSON` to an already
running `tools/alf_vllm_server.py` launch. `SERVER_TOKENIZER` must name that
launch's tokenizer (the model directory by default). Use a fresh `RUN` parent;
none of these paths are under `results/` or `artifacts/`. The following budget
caps are explicit run choices, not guaranteed sufficient spend for 32 tasks.
The repair command is the only command in this sequence that buys teacher calls.

```bash
PY=.venv/bin/python
D0=/home/xueqi/hq/projects/tc-alignment/data/rtd/v1_alfworld_k32_d0
RUN="$PWD/data/alfworld_k32_repair_ce"
STUDENT_MODEL=/absolute/path/to/served-merged-student
SERVER_JSON=/absolute/path/to/server-state/server.json
SERVER_TOKENIZER="$STUDENT_MODEL"
TOKENIZER=/home/xueqi/.cache/huggingface/hub/models--google--gemma-4-12B-it/snapshots/707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7

CUDA_VISIBLE_DEVICES='' "$PY" tools/alf_pi1_support_rollouts.py \
  --source "$D0" --output "$RUN/rollouts" \
  --server-json "$SERVER_JSON" --model-path "$STUDENT_MODEL" \
  --tokenizer-path "$SERVER_TOKENIZER"

CUDA_VISIBLE_DEVICES='' "$PY" tools/alf_repair_collect.py \
  --rollouts "$RUN/rollouts" --source "$D0" --source-collection "$D0.collection" \
  --collection "$RUN/repair.collection" --output "$RUN/repair" \
  --config "$RUN/repair.yaml" --model-path "$TOKENIZER" \
  --max-tokens 2000000 --max-usd 10 --attempt-index 0 --temperature 0 \
  --rate-limit-retries 8

CUDA_VISIBLE_DEVICES='' "$PY" tools/alf_bank_union.py \
  --left "$D0" --right "$RUN/repair" --output "$RUN/d0_plus_repair_1to1" \
  --mixing tokens-1:1 --seed 0 --config "$RUN/d0_plus_repair_1to1.yaml" \
  --model-path "$TOKENIZER"

CUDA_VISIBLE_DEVICES='' "$PY" tools/alf_pi1_train.py \
  --config "$RUN/repair.yaml" --model-path "$TOKENIZER" --preflight
CUDA_VISIBLE_DEVICES='' "$PY" tools/alf_pi1_train.py \
  --config "$RUN/d0_plus_repair_1to1.yaml" --model-path "$TOKENIZER" --preflight
```

For an all-package union, use `--mixing all` and fresh output/YAML paths.
The CLI does not silently fall back from an impossible token-balanced union.

## ADD-DEMO using the existing collector

This uses fresh collection accounting and buys a target of one additional
verified full ReAct demonstration per support task. D0's attempt indices are
0–2; the new attempt indices are 3–5 and the explicit temperature is 0.7.
`--candidates-per-task 1` stops purchases after the first verified demo per task;
failed attempts remain charged. Budget/attempt exhaustion may leave a shortfall.
The sweep filter is applied at export, so a filtered verified demo is retained
as paid material and is not automatically replaced by another purchase.

```bash
ADD="$PWD/data/alfworld_k32_add_demo"
test ! -e "$ADD" && test ! -e "$ADD.collection" && \
CUDA_VISIBLE_DEVICES='' BFAS_TEACHER=gpt-5.6-luna "$PY" tools/alfworld_teacher_pool.py \
  --source "$D0" --out "$ADD" --method plain --candidates-per-task 1 \
  --attempt-start 3 --attempts-per-task 3 --temperature 0.7 \
  --sweep-filter --workers 1 --rate-limit-retries 8 \
  --max-tokens 2000000 --max-usd 10 \
  --usd-per-mtok-in 0.20 --usd-per-mtok-out 1.20 --usd-per-mtok-cached 0.01
```

The price flags above reproduce D0's recorded prices; they are not a statement
about current provider pricing. The `test` guard makes this a new ADD-DEMO
collection, although the existing collector still supports its usual explicit
resume workflow. Omitting the new options preserves the old attempt schedule,
temperature schedule and export behavior. To register just the usable ADD-DEMO
packages under plain CE:

```bash
CUDA_VISIBLE_DEVICES='' "$PY" tools/alf_bank_subset.py \
  --source "$ADD" --output "$ADD.usable" --selection all-usable \
  --config "$ADD.yaml" --model-path "$TOKENIZER"
CUDA_VISIBLE_DEVICES='' "$PY" tools/alf_pi1_train.py \
  --config "$ADD.yaml" --model-path "$TOKENIZER" --preflight
```

## CPU tests

The synthetic two-task fixture uses a stub student backend, teacher transport
and stepper. Network connections, real environment startup and CUDA queries
are forbidden. Tests cover takeover ordering, strict sweep threshold, masked
fallback prefixes, actual replay success, charged failed calls, hard budgets,
replay divergence, byte-preserving union, exact token arithmetic and seeded
selection. The installed cached tokenizer is also used for real registration
and `alf_pi1_train.py --preflight` on the synthetic REPAIR and union banks.

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest -q \
  tests/test_alf_repair_pipeline.py tests/test_alfworld_teacher_pool.py \
  tests/test_rtd_alfworld_support.py tests/test_rtd_alfworld_state.py \
  tests/test_alf_bank_subset.py tests/test_alfworld_boundary_diagnostics.py \
  tests/test_rtd_alfworld_evaluation.py \
  -m 'not integration'
```

No real teacher purchase, served rollout, ALFWorld replay or training is part
of the implementation validation run.

Validation (2026-09-22, `CUDA_VISIBLE_DEVICES=''`):
`256 passed, 2 deselected in 227.64s (0:03:47)`.
The two deselected tests require real ALFWorld; synthetic REPAIR and union
registration and CPU preflights are included in the passing tests. A read-only
D0 audit also confirmed 32 support tasks, 32 matching portable reset states and
30 usable packages.
