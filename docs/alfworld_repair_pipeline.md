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

Union inputs must share the same support tasks, parent/fold metadata, every
world-file hash and combined world hash, and every reset request/state field:
goal, horizon, complete initial observation, ordered admissible actions, prompt,
history flags and parent identity. Differences name the task and field.
Environment identity may differ only in the collector-code fingerprints for
`tools/alfworld_teacher_pool.py` and `alfworld_support.py`, plus the environment
and historical-environment hashes. Other environment metadata, including the
tokenizer and renderer settings, must match. Both source banks still pass the
unchanged auditor, which checks their signatures and derived hashes.

The union inherits the left/D0 environment, requests and resets, preserving
its exact support manifest/hash when it describes the complete support. If the
left bank has a `training_selection` marker, only that training inventory is
recomputed and the derived support is re-signed. This keeps D0's registered
identity for the D0 + ADD-DEMO unions. The manifest's `derivation` records both
source environment/support hashes, every compared field, the statement
`code-identity-only difference`, and the chosen identity and reason.

Packages already using the left identity are copied byte for byte. Packages
using the other identity are normalized **only in the union copy**: update the
request hash and state task identities, recompute the transcript and sealing
hashes, and record both original and derived package hashes. Historical ledger
evidence, teacher replies, commands, observations, admissible lists and stored
prompts remain unchanged. The tool also checks that the same renderer produces
identical prompts before and after normalization for every verified state.
The unchanged auditor and `pi1.load_bank` must accept the result. Complete source
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

### D0 + ADD-DEMO unions (Codex#19, 2026-09-22)

Built from `/home/xueqi/hq/projects/tc-alignment/data/rtd/v1_alfworld_k32_d0`
and `data/alfworld_k32_add_demo`, using the Kang template and local tokenizer
snapshot `/home/xueqi/.cache/huggingface/hub/models--google--gemma-4-12B-it/snapshots/707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`.
All 32 task/reset comparisons passed. Both unions preserve D0's exact support
hash `6a3f856d466fce1d63e34bc74440988f7abae52bec62ca69efc5ae64f39b2a9c`
and environment hash
`d5747372239d8b530fb154a32102238a2e62c8f3c299d85c4cdaf8987d0bc301`.
The union manifests record both original identities and the normalization of
the 20 selected ADD-DEMO packages. Source banks and pinned modules are unchanged.

| Bank under `artifacts/` | Packages | Turns | Supervised tokens | D0 / ADD tokens |
| --- | ---: | ---: | ---: | ---: |
| `alfworld_k32_d0_plus_add_1to1` | 37 | 516 | 12,900 | 6,450 / 6,450 |
| `alfworld_k32_d0_plus_add_all` | 50 | 673 | 16,099 | 9,649 / 6,450 |

The 1:1 selection uses seed 0 and retains 17 D0 plus all 20 ADD-DEMO packages.
Both `configs/rtd/pi1_alfworld_k32_d0_plus_add_{1to1,all}.yaml` registrations
use counts from `pi1.load_bank` and its encoder, including one native turn
boundary per turn. Every selected rendered training row equals its source row.
Both `tools/alf_pi1_train.py --preflight` runs passed with CUDA hidden and
offline tokenization: exact exposure targets are 38,700 / 129,000 for 1:1 and
48,297 / 160,990 for all. No GPU, teacher calls or simulator replay were used
to materialize or preflight these banks.

Regression command:

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
HF_DATASETS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q \
  tests/test_alf_bank_union.py tests/test_alf_repair_pipeline.py \
  tests/test_alf_pi1_train.py tests/test_rtd_alfworld_support.py \
  tests/test_rtd_alfworld_state.py tests/test_alfworld_boundary_identity.py \
  tests/test_rtd_identity_update.py -m 'not integration'
```

`195 passed, 1 skipped, 1 deselected, 1 warning in 58.46s`

The skipped test needs the separate original pi1 bank; real-environment
integration was deselected. The warning is PEFT's missing-config vocabulary
assumption in its synthetic export test. New tests include audited synthetic
banks with changed world bytes or admissible actions, identity-only unions in
both mixing modes, unchanged loaded rows and source bytes, and rejection of a
renderer whose output depends on the environment hash.

## Targeted ADD-DEMO (Codex#20)

Choose task **types** from the student's support failures, then buy only tasks
of those types within the frozen support. The default selection is:

| Task-type prefix | K=32 tasks |
| --- | ---: |
| `pick_cool_then_place_in_recep` | 5 |
| `pick_heat_then_place_in_recep` | 4 |
| `pick_two_obj_and_place` | 7 |
| Total | 16 |

`configs/alfworld_k32_targeted_tasks.json` contains that sorted 16-ID list.
The selector reads historical support, including tasks without a usable D0
demo. It matches the complete type before the first hyphen; a partial prefix
or unknown type is rejected. To generate a new list (the output must not exist):

```bash
PY="$PWD/.venv/bin/python"
D0=/home/xueqi/hq/projects/tc-alignment/data/rtd/v1_alfworld_k32_d0
CUDA_VISIBLE_DEVICES='' "$PY" tools/alf_targeted_tasks.py \
  --support "$D0/public/support.json" \
  --task-types pick_cool_then_place_in_recep pick_heat_then_place_in_recep pick_two_obj_and_place \
  --output "$PWD/data/alfworld_k32_targeted_tasks.json"
```

The default `--task-types` is the list shown above, and the default `--support`
is the D0 path shown above. The tool prints counts per type and writes only a
JSON list; it never contacts the teacher or runs a student.

The collector's `--only-tasks FILE` accepts unique support task IDs, rejects
unknown IDs before any purchase or collection write, and freezes the sorted
selection in `identity.json`. Changing that selection when resuming is rejected.
An empty list buys nothing. Tasks outside the list never enter the acquisition
queue; ledger and request-journal validation also rejects outside tasks.
`summary.json` and `candidate_sets.json` count only selected tasks, so unselected
tasks do not inflate candidate shortfalls.

The exported `public/support.json` and reset states retain **all 32 historical
tasks**, parent groups, folds and world identities. Its signed
`training_selection: usable_packages` inventory includes only selected tasks
with usable purchased packages. Failed or sweep-filtered purchases retain their
paid ledger/package evidence. This supports the existing D0 union's
`code-identity-only difference` rule without changing that rule.

Run this **dry-run** from the baselines worktree. It audits D0 and prints the
selection, attempt indices, temperatures, candidate/attempt ceilings and budget
caps, with no collection/bank writes, teacher calls, environment startup or GPU:

```bash
PY="$PWD/.venv/bin/python"
D0=/home/xueqi/hq/projects/tc-alignment/data/rtd/v1_alfworld_k32_d0
TASKS="$PWD/configs/alfworld_k32_targeted_tasks.json"
TARGETED="$PWD/data/alfworld_k32_add_targeted"
CUDA_VISIBLE_DEVICES='' BFAS_TEACHER=gpt-5.6-luna "$PY" tools/alfworld_teacher_pool.py \
  --source "$D0" --out "$TARGETED" --only-tasks "$TASKS" --dry-run \
  --method smartad --candidates-per-task 2 --attempt-start 6 --attempts-per-task 4 \
  --temperature 0.7 --sweep-filter --workers 1 --rate-limit-retries 8 \
  --max-tokens 2000000 --max-usd 10 \
  --usd-per-mtok-in 0.20 --usd-per-mtok-out 1.20 --usd-per-mtok-cached 0.01
```

Expected plan: 16 selected / 32 historical tasks, at most 32 verified demos
and 64 trajectory attempts, indices **6–9**, temperature **0.7** throughout.
D0 used 0–2 and the earlier ADD-DEMO used 3–5. `--method smartad` enables the
existing multiple-candidate collector; exported targets remain full ReAct
demonstrations for `pi1_ce`. It stops at two collection-verified demos per task.
CPU replay and the export-time sweep filter can reduce the usable count;
filtered demos are charged and are not automatically replaced. Budget caps
also apply. The quoted prices reproduce D0's accounting, not current pricing.

**Purchase recipe for the user to run later; not part of the dry-run.** These
guards refuse an existing bank or collection (including dangling symlinks).
Run in a shell with `set -e` so refusal also prevents the subsequent union:

```bash
set -e
for path in "$TARGETED" "$TARGETED.collection"; do
  if [ -e "$path" ] || [ -L "$path" ]; then
    echo "Refusing existing targeted collection: $path" >&2
    exit 1
  fi
done
CUDA_VISIBLE_DEVICES='' BFAS_TEACHER=gpt-5.6-luna "$PY" tools/alfworld_teacher_pool.py \
  --source "$D0" --out "$TARGETED" --only-tasks "$TASKS" \
  --method smartad --candidates-per-task 2 --attempt-start 6 --attempts-per-task 4 \
  --temperature 0.7 --sweep-filter --workers 1 --rate-limit-retries 8 \
  --max-tokens 2000000 --max-usd 10 \
  --usd-per-mtok-in 0.20 --usd-per-mtok-out 1.20 --usd-per-mtok-cached 0.01

TOKENIZER=/home/xueqi/.cache/huggingface/hub/models--google--gemma-4-12B-it/snapshots/707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7
UNION="$PWD/data/alfworld_k32_d0_plus_targeted"
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  "$PY" tools/alf_bank_union.py \
  --left "$D0" --right "$TARGETED" --output "$UNION" --mixing all \
  --config "$UNION.registration.yaml" --model-path "$TOKENIZER"
```

The union and registration also refuse existing outputs. Keep D0 on the left
to inherit its exact registered support identity. The intermediate registration
records the actual union manifest, demonstration/turn counts and supervised
tokens; these cannot be known until collection finishes.

**Token-endpoint config template instructions only.** The token-endpoint
trainer/configs live in `../tc-alignment-taskeq`, not this baselines tree.
After the union above, run the following to create
`../tc-alignment-taskeq/configs/rtd/pi1_alfworld_k32_d0_plus_targeted_tok.yaml`.
It uses the existing token-endpoint template, actual union bank fields, an
absolute bank path valid in that tree, seeds 0/1/2, and fixed
`exposure_tokens: [28947, 96490]`. It refuses an existing config. No training
command is included, and no config in the taskeq tree is written by this change.

```bash
TASKEQ="$(realpath ../tc-alignment-taskeq)"
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 "$PY" - \
  "$TASKEQ" "$UNION" "$UNION.registration.yaml" <<'PY'
from pathlib import Path
import sys
import yaml

tree, bank, registration = map(Path, sys.argv[1:])
template = tree / 'configs/rtd/pi1_alfworld_k32_d0_plus_add_all_tok.yaml'
output = tree / 'configs/rtd/pi1_alfworld_k32_d0_plus_targeted_tok.yaml'
config = yaml.safe_load(template.read_text())
measured = yaml.safe_load(registration.read_text())
for key in ('sealed_manifest_sha256', 'support_size', 'demonstrations',
            'supervised_turns', 'bank_supervised_tokens'):
    config[key] = measured[key]
config['bank'] = str(bank.resolve())
config.pop('exposure_passes', None)
config.update(exposure_tokens=[28947, 96490], training_seeds=[0, 1, 2])
with output.open('x') as stream:
    stream.write('# Targeted ADD-DEMO union; matched supervised-token endpoints.\n')
    yaml.safe_dump(config, stream, sort_keys=False)
sys.path.insert(0, str(tree / 'src'))
from bfas.rtd.baselines.pi1 import load_config
load_config(output)
print(output)
PY
```

Validation (2026-09-23): the real D0 dry-run selected 16/32 tasks with
`teacher_calls: 0` and left no targeted bank or collection directory. A read-only
comparison of all 32 D0 requests/resets against the current collector's derived
support passed the union's `code-identity-only difference` rule. The synthetic
tests prohibit network, real environment startup and CUDA access, and exercise
restricted purchases with one/three workers, rejection of outside IDs, frozen
resume selection, failed/empty selections, all four attempt indices, read-only
dry-run, exact type selection, and audited D0 union with unchanged training rows.

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
HF_DATASETS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q \
  tests/test_alf_targeted_tasks.py tests/test_alfworld_teacher_pool.py \
  tests/test_alf_bank_union.py tests/test_alf_repair_pipeline.py \
  tests/test_rtd_alfworld_support.py tests/test_rtd_alfworld_state.py \
  tests/test_alf_bank_subset.py -m 'not integration'
```

`198 passed, 1 deselected in 112.11s (0:01:52)`

The deselected test requires real ALFWorld. No real teacher calls, training,
GPU use, or writes under `results/` or `artifacts/` were performed.
