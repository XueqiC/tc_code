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

The source's `sealed/manifest.json` and artifact hashes are audited before any
purchase. Task IDs must agree across `public/support.json`,
`public/reset_requests.json`, `public/requests.json`, and the sealed payloads.
The current source contains **142 historical task IDs**. Every ID is retained,
including tasks without a successful source demonstration. The existing training
folds and protected parent exclusions are preserved; availability does not select
the task set. Protected attempts remain unavailable for training.

Purchases go through `bfas.ledger.acquire_demos` with **attempts=3**: attempt 0
uses the gateway's greedy setting, and attempts 1–2 use its sampled setting.
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
The ledger deduplicates task/attempt purchases. If a process dies after reserving
a call but before recording its episode, resume conservatively records that
attempt as failed and charged instead of buying it again. Missing or inconsistent
accounting fails closed. Keep the same source, teacher, prices, and collector
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
The destination must be new. Conversion needs the configured student's tokenizer
in the local Hugging Face cache; it loads no model weights.

```bash
PYTHONPATH=src CUDA_VISIBLE_DEVICES='' \
.venv/bin/python tools/rtd_alfworld_bank_v11.py \
  --source data/rtd/v1_alfworld_luna \
  --ledger data/rtd/v1_alfworld_luna.collection/teacher_ledger.jsonl \
  --config configs/rtd/v1_1_alfworld.yaml \
  --out data/rtd/v1_1_alfworld_luna
```

Or use the shared bank builder:

```bash
PYTHONPATH=src CUDA_VISIBLE_DEVICES='' \
.venv/bin/python tools/rtd_bank_build.py \
  --benchmark alfworld \
  --pool data/rtd/v1_alfworld_luna \
  --ledger data/rtd/v1_alfworld_luna.collection/teacher_ledger.jsonl \
  --config configs/rtd/v1_1_alfworld.yaml \
  --out data/rtd/v1_1_alfworld_luna
```

Both commands audit the C26-format pool, bind the exact supplied ledger rows,
render stored states with the configured student tokenizer, and write the v1.1
bank and public cost certificate. Finish collection before conversion: adding
ledger rows changes the pool's sealed identity and requires a fresh conversion.

## 4. Point the experiment configs at the bank

Set these keys in **both** `configs/rtd/unified_alfworld_gemma4.yaml` and
`configs/rtd/v1_1_alfworld.yaml` after the new bank exists:

```yaml
replay_bank_path: data/rtd/v1_1_alfworld_luna
support_manifest: data/rtd/v1_1_alfworld_luna/public/support.json
```

Their `student` must match the tokenizer used at conversion (currently the
configured Gemma student). Preserve `mode: sealed_replay`,
`new_teacher_calls: false`, and `new_teacher_tokens: 0` for offline replay.
The checked-in configs continue to reference the existing bank until collection
and conversion have been performed.
