# ALFWorld Luna teacher pool and v1.1 bank

Run from the repository root on `rtd-unified`. Collection requires the local
ALFWorld/TextWorld environment and train world files under `envs/alfworld`, plus
`OPENAI_API_KEY` in the environment. It runs the CPU environment and remote
teacher; it loads no student weights. The commands below describe a future
collection run; implementation and tests did not purchase demonstrations.

Use this tree's `.venv/bin/python`. If absent, substitute
`/home/xueqi/hq/projects/tc-alignment/.venv/bin/python`.

## 1. Collect the pool

```bash
BFAS_TEACHER=openai/gpt-5.6-luna \
BFAS_OPENAI_SERVICE_TIER=flex \
CUDA_VISIBLE_DEVICES='' \
.venv/bin/python tools/alfworld_teacher_pool.py \
  --source data/rtd/v1_alfworld_c26 \
  --out data/rtd/v1_alfworld_luna \
  --max-tokens 400000 \
  --max-usd 5.0 \
  --usd-per-mtok-in 0.10 \
  --usd-per-mtok-out 0.60 \
  --usd-per-mtok-cached 0.01
```

Those are the defaults. `BFAS_TEACHER` selects the teacher; the command sets
`BFAS_OPENAI_SERVICE_TIER=flex` before loading its configuration. The supplied
Luna Flex price assumptions are dollars per million tokens. If selecting a
different teacher, supply that teacher's rates; the command does not fetch prices.

Add `--workers N` to run up to N episodes concurrently (default: `1`). Workers
pull tasks from a shared queue and each owns a separate adapter and ALFWorld
environment subprocess, with its own working directory and `TMPDIR`. Token and
USD caps apply to the entire collection, regardless of worker count. Worker count
can change on resume; it does not change the pool layout or collection identity.

The source's `sealed/manifest.json` and artifact hashes are audited before any
purchase. Task IDs must agree across `public/support.json`,
`public/reset_requests.json`, `public/requests.json`, and the sealed payloads.
The current source contains **142 historical task IDs**. Every ID is retained,
including tasks without a successful source demonstration. The existing training
folds and protected parent exclusions are preserved; availability does not select
the task set. Protected attempts remain unavailable for training.

The collector uses the BFAS ledger format and purchase locks with **attempts=3**:
attempt 0 uses the gateway's greedy setting, and attempts 1–2 use its sampled
setting. `BFAS_TEACHER_MIN_INTERVAL_S` spaces episode starts across all workers.
The shared client applies provider-specific sampling restrictions. A verified
attempt ends acquisition for that task. No quota probes or automatic HTTP retries
are issued by this collector. Each failed network call consumes one episode
attempt and remains charged; failed episodes are retained in the sealed pool.

`--max-tokens` caps **prompt plus completion tokens**, including cached prompt
and hidden reasoning tokens. Ledger `tokens_spent` retains the BFAS convention of
completion tokens, so v1.1 replay costs remain compatible. Before each request,
the collector reserves an uncached prompt envelope (UTF-8 bytes plus message
framing allowance) and up to 2048 output tokens. The output request limit shrinks
to the remaining token and USD budget. When even one output token cannot fit,
collection stops and exports the partial pool; a cap can leave unused headroom.
Reported prompt/completion usage releases unused reservation. Cached prompt
usage is a subset of prompt usage and is priced once:

```text
USD = ((prompt - cached) * input_rate
       + cached * cached_rate + completion * output_rate) / 1_000_000
```

These are local hard admission caps using the configured price assumptions and
text-token envelope. If a provider violates that envelope, the command records
its reported usage and stops; it cannot undo a provider charge. Unknown usage,
malformed responses, timeouts, and interrupted requests retain their full
reservation, including uncached input. The summary identifies uncertain calls.
Reservation and settlement each reload the request journal under its file lock.
Workers wait when an active request may release enough headroom; uncertain
reservations from ended or crashed requests remain charged. Interrupting the
collector stops new requests, waits for in-flight calls and environment cleanup,
then exports the partial pool.

## 2. Resume and inspect

Repeat the same collection command to resume. Caps may be raised. Keep these
paths together:

```text
data/rtd/v1_alfworld_luna/
  public/{requests,reset_requests,reset_states,support}.json
  sealed/{<query_id>,integrity,manifest,audit,event_aliases}.json

data/rtd/v1_alfworld_luna.collection/
  identity.json
  teacher_ledger.jsonl
  usage.jsonl
  summary.json
  collection.lock
```

The sibling directory contains the BFAS episode ledger and fsynced request
reservations. A process lock serializes collection/export for the same output.
Within a collection, short file locks serialize request accounting and episode
appends; no purchase lock is held during teacher or environment calls. Export
runs after all workers finish, under the collection and episode-ledger locks.
The ledger deduplicates task/attempt purchases. If a process dies after reserving
a call but before recording its episode, resume conservatively records that
attempt as failed and charged instead of buying it again, even when the cap is
already exhausted. Missing or inconsistent accounting fails closed. Keep the
same source, teacher, prices, and collector
code for a resume; a changed identity requires a new output directory.

The printed JSON and `summary.json` report task count, attempted tasks, verified
usable packages, total tokens, prompt/cached/completion tokens, estimated USD,
uncertain calls, and stop reason. `verified` counts successful CPU replay checks.
All purchased attempts are exported, including failures. The public support
still names all 142 tasks when a cap stops collection early. With no successful
replay, the partial pool is valid but cannot yet form a usable v1.1 bank.

The source's world bytes and task folds are preserved. A new environment identity
binds the collector code and portable message rendering. Successful commands
are replayed against those frozen world bytes, with every deployment context
checked and full ordered states sealed. Pool export and resume may repeat these
CPU replays; neither purchases teacher output. The bank audit's
`new_teacher_calls=0` refers to this offline sealing phase; acquisition totals
are in the sibling ledger and summary.

## 3. Convert to the student-rendered v1.1 bank

After collection finishes, choose **one** of these equivalent offline commands.
These use the completed production pool `v1_alfworld_luna2` and its exact ledger.
The destination must be new. Conversion needs the configured student's tokenizer
in the local Hugging Face cache; it loads no model weights. The config's
`support_manifest` must already exist: it is the frozen GPT-5.4 support shared
by both teacher banks and all arms.

```bash
PYTHONPATH=src CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
.venv/bin/python tools/rtd_alfworld_bank_v11.py \
  --source data/rtd/v1_alfworld_luna2 \
  --ledger data/rtd/v1_alfworld_luna2.collection/teacher_ledger.jsonl \
  --config configs/rtd/v1_1_alfworld.yaml \
  --out data/rtd/v1_1_alfworld_luna.frozen_support
```

Or use the shared bank builder:

```bash
PYTHONPATH=src CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
.venv/bin/python tools/rtd_bank_build.py \
  --benchmark alfworld \
  --pool data/rtd/v1_alfworld_luna2 \
  --ledger data/rtd/v1_alfworld_luna2.collection/teacher_ledger.jsonl \
  --config configs/rtd/v1_1_alfworld.yaml \
  --out data/rtd/v1_1_alfworld_luna.frozen_support
```

Both commands audit the C26-format pool, bind the exact supplied ledger rows,
render stored states with the configured student tokenizer, and write the v1.1
bank and public cost certificate. Finish collection before conversion: adding
ledger rows changes the pool's sealed identity and requires a fresh conversion.

Both entrypoints now use the config's frozen support. They allow the collector's
documented environment identity change only when its historical environment hash
matches the frozen environment and every nonderived support field is identical.
Task inventory, ordering, worlds, goals, parents, folds, exclusions and round
assignments cannot silently change. Other differences fail conversion.

The converter regenerates runtime reset and behavior states with the frozen
requests and configured student renderer, then reseals the integrity index and
cost certificate. Copying only `support.json` would leave incompatible state
hashes. Original collector verification transcripts, ledger rows, commands,
costs and outcomes remain archived unchanged; `support_binding` in the audit and
certificate records the source environment and both support identities.

Folds use `int(parent_hash, 16) % 2`, grouping all trials of the same game.
Protected calibration/probe parents remain excluded. Request packages refer to
the parent hash without duplicating its fold. See the
[D16 fold contract](rtd_v1_1_d16.md#parent-fold-contract).

## 4. Point the experiment configs at the bank

The Luna configs (`v1_1_alfworld_luna.yaml`, `unified_alfworld_gemma4_luna.yaml`
and `unified_alfworld_gemma4_d0_luna.yaml`) retain these settings:

```yaml
replay_bank_path: data/rtd/v1_1_alfworld_luna
support_manifest: data/rtd/v1_1_alfworld/public/support.json
```

Their `student` must match the tokenizer used at conversion (currently the
configured Gemma student). Preserve `mode: sealed_replay`,
`new_teacher_calls: false`, and `new_teacher_tokens: 0` for offline replay.
The original configs continue to use the GPT-5.4 replay bank. Both sets reference
the same external support; do not redirect `support_manifest` to a collector's
support just to bypass preflight.

## 5. Production repair audit (2026-09-11)

`src/bfas/rtd/benchmarks/config.py:data_identity` raises
`external support manifest differs from bank` when the parsed JSON at
`config['support_manifest']` differs from `replay_bank_path/public/support.json`.
For V0/V2/D3/D0 the external file is
`data/rtd/v1_1_alfworld/public/support.json`, the GPT-5.4 bank's frozen manifest.
Object key ordering is irrelevant to this comparison; list ordering is significant.
The older v1.0 check in `alfworld_config.py:bank_audit` uses a similar message
ending in `sealed bank`, but was not the failing path.

The Luna pool and old converted bank have **exactly the same task inventory and
split** as GPT-5.4. Source-only task IDs: `[]`; frozen-only task IDs: `[]`.
All task IDs, parent hashes, folds, selected trials, list ordering, world hashes,
world-file hashes and 429 source-file entries agree. The apparent 135-versus-142
discrepancy compares training **parents** against historical **task IDs**:

| Quantity | GPT-5.4 and Luna support |
|---|---:|
| Historical task IDs / parent games | 142 / 139 |
| Excluded historical tasks / parents | 4 / 4 |
| Training task IDs / parents (`m`) | 138 / 135 |
| Fold 0 training tasks / parents | 81 / 79 |
| Fold 1 training tasks / parents | 57 / 56 |
| Protected calibration / probe task IDs | 35 / 4 |
| Protected parent hashes (including parents outside historical inventory) | 36 |

The manifest names the protected sets `calibration_task_ids` and `probe_task_ids`;
there is no separate `confirmation_task_ids` field. Those sets, every exclusion,
and the full round/fold assignments remain identical across banks and arms.

Only three top-level fields differed: `environment`, `tasks`, and
`manifest_hash`. In all 142 task entries, only `request.environment_hash` and its
derived `request_hash` differed. `tools/alfworld_teacher_pool.py:collection_support`
intentionally replaces the historical environment/tokenizer identity with one
binding collector code and portable messages. Its `historical_environment_hash`
matches the GPT-5.4 environment. The old converter copied that collection
manifest instead of using the config's frozen support.

The bank was regenerated offline from `data/rtd/v1_alfworld_luna2` and its
233-row ledger using the command above. The original bank was retained at
`data/rtd/v1_1_alfworld_luna.pre_frozen_support_20260911`; the validated replacement
occupies `data/rtd/v1_1_alfworld_luna`. No packages were dropped or added, and no
frozen task was missing recorded attempts. Package availability does not define
support: 34 training tasks have no usable Luna package but remain in support.

| Bank audit | Before and after |
|---|---:|
| Recorded attempts / attempted tasks | 233 / 142 |
| Usable / unavailable packages | 104 / 129 |
| Fold 0 / fold 1 usable packages | 62 / 42 |
| Fold 0 / fold 1 parents with usable packages | 60 / 41 |
| Usable command/behavior states | 1,535 |
| Public resets / selected training resets | 142 / 135 |
| Usable recorded output-token denominator | 118,792 |
| 10% / 25% budget ceilings | 11,879 / 29,698 |
| Per-package class cap / sum of usable caps | 16,384 / 1,703,936 |
| All-attempt recorded output-token estimate | 521,643 |
| New teacher calls / GPU use during repair | 0 / 0 |

The former usable-attempt reservation audit (7/24 packages at 11,879/29,698)
is superseded by attempt purchase accounting. The class cap and denominator remain
archival; the runtime charge now sums every attempt for the selected task.
See the correction and exact baseline comparison below.

All 142 reset states also match the GPT-5.4 bank exactly, so feedback and fixed
diagnostic task selection share the same requests, histories, prompts and hashes.
All 1,535 converted behavior states pass the frozen support's ownership/fold
guards, and all 233 sealed payload hashes validate. The source pool and ledger
hashes, recorded attempts, original replay evidence, commands and costs are unchanged.

The frozen and repaired `support.json` files are byte-identical, with SHA256
`d6f3dbf6fddca162472982b14ab55888e3b55862fa64c4abc1e5d753290c161b` and internal
manifest hash `c9e94738a072548a1b61333ef67b324005c995dd713d870dda035651067ab0f6`.
The old Luna manifest hash was
`38f4fa7f7a4a31d37244cfa43fe717a40b9613c1e647db87b6ed2c5a17983c5c`.
Full before/after certificate hashes and machine-readable inventory checks are
in [the support audit](rtd_alfworld_luna_support_audit.json).

All four CPU preflights passed with 104 available packages, 135 support states,
135 rendered states, zero ledger spend and no deferred checks:

```bash
.venv/bin/python tools/rtd_preflight.py --config configs/rtd/v1_1_alfworld_luna.yaml --arm V0 --mode run
.venv/bin/python tools/rtd_preflight.py --config configs/rtd/unified_alfworld_gemma4_luna.yaml --arm D3 --mode smoke
.venv/bin/python tools/rtd_preflight.py --config configs/rtd/unified_alfworld_gemma4_d0_luna.yaml --arm D0 --mode smoke
.venv/bin/python tools/rtd_preflight.py --config configs/rtd/v1_1_alfworld_luna.yaml --arm V2 --mode run
```

The tool masks CUDA and forces offline tokenizer loading before imports. D3/D0
standalone checks use its default smoke mode because production mode requires
`--replay-schedule <completed V0 run>`; V0 failed before publishing a schedule.
Their initial schedule-free `--mode run` checks correctly rejected that missing
prerequisite. This repair does not waive replay requirements or fabricate a
schedule. Successful logs are `logs/alfworld_luna_frozen_support_preflight_V0.log`,
`_V2.log`, `_D3_smoke.log` and `_D0_smoke.log` (same filename prefix).
GPU3's placeholder and the stopped production chain were left in place.

Regression coverage checks both offline builder entrypoints against original
and collector-derived identities, frozen-support guards on purchased prefixes,
rejection of split/world/inventory changes, and collector resume/conversion.
The combined CPU regression run passed **67 tests** across
`test_rtd_v11_alfworld_conversion.py`, `test_alfworld_teacher_pool.py` and
`test_rtd_preflight.py`; output is in
`logs/alfworld_luna_frozen_support_tests.log`.

## Attempt purchase correction (2026-09-11)

Each teacher attempt is a separate package at its original ledger token cost,
including failed attempts and recorded reasoning. Sort the original query IDs,
shuffle once with `random.Random(0)`, and stop before the first overflow.
The per-task ledger retains each attempt's task/parent dependency; it does not
combine retry costs. All bank bytes, including support/folds, per-task ledgers,
sealed payloads and certificates, are unchanged; no rebuild was needed.

RTD and the read-only paper reader bought exactly the same ordered attempt IDs
and usable IDs at all four budgets, with equal spend and next blocker:

| Budget | Attempts | Usable | Tokens charged |
| ---: | ---: | ---: | ---: |
| 7,500 | 4 | 2 | 6,535 |
| 15,000 | 9 | 5 | 13,342 |
| 30,000 | 13 | 5 | 29,229 |
| 60,000 | 24 | 9 | 58,602 |

The cited 30k receipt is reproduced: **13 attempts, 5 usable, 29,229 tokens**.
The task-level zero-usable result is superseded. Denominator 118,792 and configured
fraction caps 11,879 / 29,698 remain unchanged. V0/D3 full CPU preflight passes.

Purchases use the whole bank prefix, independently of training seed/fold.
Only exposure is restricted to the active parent fold. Failed and other-fold
purchases remain in V0/D3 receipts. Window limits pause and resume the same order.

[Shared protocol](rtd_hotpotqa.md#attempt-purchase-correction-2026-09-11),
[both code paths' attempt IDs](rtd_attempt_purchase_validation.json),
[CPU and byte-preservation receipts](rtd_attempt_cpu_validation.json).
No GPU/API calls were made. Older task-purchase receipts are superseded.
