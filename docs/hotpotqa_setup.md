# HotpotQA-ReAct

This adapter uses the ReAct HotpotQA protocol of
[Yao et al. (2022)](https://arxiv.org/abs/2210.03629), published at ICLR 2023.
`src/bfas/hotpotqa.py` owns the prompt construction, Wikipedia environment,
action parser, episode loop and answer verifier used by both the BFAS adapter
and standalone evaluation. Nothing in setup or the CPU tests calls an LLM.

## Install data

From the repository root:

```bash
scripts/setup_hotpotqa.sh
```

The script uses `.venv/bin/python -B` and first reuses valid JSON already in
`envs/hotpotqa/data`. For missing JSON it uses local HuggingFace parquet shards
when present; otherwise it tries the official
[training JSON](https://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json)
(90,447 questions) and
[distractor dev JSON](https://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json)
(7,405 questions), linked by the [HotpotQA site](https://hotpotqa.github.io/).
If the official download fails, setup falls back to the
[`hotpotqa/hotpot_qa` HuggingFace mirror](https://huggingface.co/datasets/hotpotqa/hotpot_qa),
config `distractor`. To skip the official host, place these shards directly in
`envs/hotpotqa/data` with these basenames before running setup:

- `train-00000-of-00002.parquet`
- `train-00001-of-00002.parquet`
- `validation-00000-of-00001.parquet`

If any shard for a split is present, setup uses the mirror for that split and
downloads only its missing shards from
`https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/main/distractor/<basename>`.
Finish local downloads before running setup. Conversion preserves shard, row,
title and sentence order, renames `id` to `_id`, and reconstructs the official
`supporting_facts` (`[[title, sentence_index], ...]`) and `context`
(`[[title, [sentence, ...]], ...]`) arrays. The resulting JSON records also
retain `question`, `answer`, `type` and `level`.

All downloads, temporary files and checksum manifests stay under
`envs/hotpotqa/data`. Before publishing JSON, setup validates the record format,
counts, unique IDs, full source ID order and resolution of every frozen support,
demand, calibration and eval ID. Downloads use HTTPS, a 60-second socket timeout,
three attempts per URL, and an atomic rename after validation. Re-running setup
reuses valid files. A malformed existing file fails validation instead of being
silently overwritten. The dataset is distributed under CC BY-SA 4.0.

`SOURCE.json` records each split's actual source, output count and SHA-256; mirror
entries include the dataset/config and each input shard's URL, checksum and
whether it was local or downloaded. Setup preserves that provenance on reruns;
JSON without matching provenance is labeled `existing_json`. `manifest.json`
also contains this inventory after both splits succeed.

Runtime dependencies are already in this workspace's `.venv`: `requests`,
`beautifulsoup4`, `openai`; BFAS rendering also uses its existing `transformers`
installation. Mirror conversion uses `pyarrow`, already present in `.venv`.
If it is absent in another checkout, install only into that environment:

```bash
uv pip install --python .venv/bin/python pyarrow
```

Official JSON setup needs only Python's standard library. Setup does not install
packages or use external dataset cache directories.

The first execution in this development sandbox failed DNS resolution for
`curtis.ml.cmu.edu`; no official JSON was downloaded. The checked-in ID manifests
were extracted from the existing local `hotpotqa/hotpot_qa` distractor cache at
revision `1908d6afbbead072334abe2965f91bd2709910ab`, preserving its full source
order. Setup checks the official downloads against those complete ID-order
hashes before the adapter will use them. Run the command above after the parquet
downloads finish, or in a shell with network access to download missing data.

## Frozen protocol

- `configs/hotpotqa_eval_split.json`: the **first 500 dev questions in file
  order**, with IDs recorded. `--start` and `--n` select within that inventory;
  they cannot expand evaluation beyond question 499. This is the requested
  fixed-order protocol; the upstream notebook shuffled dev indices with seed
  233 before selecting 500, so those published numbers are not directly matched.
- `configs/hotpotqa_support_split.json`: 200 train IDs selected by
  `random.Random(0).sample(source_ids, 200)`, retaining sample order. BFAS demand
  and calibration contain 160 and 40 IDs. Calibration is
  `random.Random(0).sample(support_ids, 40)`; the split is fixed across BFAS seeds.
  HotpotQA overrides BFAS's generic 5%/seed-50 support selection.
- `prompts/hotpotqa_react_6shot.txt` is the **verbatim concatenation** of the
  upstream notebook's `instruction` and `prompts_naive.json["webthink_simple6"]`.
  Source URLs, extraction description and SHA-256 are in the adjacent
  `.source.json`; the original MIT license is included. Original whitespace and
  encoding artifacts are deliberately retained. Attribution is outside the
  model input. The six examples do not occur in dev (checked against the local
  source inventory).
- A single user message contains that prompt, the current question, the full
  episode transcript and `Thought i:`. The dataset's distractor paragraphs,
  gold answer and supporting facts are never sent to either model.
- At most **7 environment steps**. As in the upstream notebook, a malformed
  response gets one action-only completion retry within that step (at most 14
  model requests total). Invalid actions consume a step. There is no extra
  model-generated answer after the step limit. The student uses temperature 0
  during evaluation and the notebook's 100-token completion cap.
- `finish[answer]` terminates the episode. Only official normalized **answer
  EM == 1** verifies a rollout/demo; answer F1 never makes a demo eligible.
  EM/F1 include the official yes/no/noanswer special cases and token multiplicity.
  Scores are fractions in `[0,1]`; supporting-fact/joint metrics are not used.

## Wikipedia and offline operation

The original [ReAct wrapper](https://github.com/ysymyth/ReAct/blob/master/wikienv.py)
uses `https://en.wikipedia.org/w/index.php?search=...`, an HTML search endpoint,
rather than the MediaWiki JSON API. We preserve that transport and behavior:
`search[entity]` returns the first five sentences of an article or up to five
similar titles; `lookup[keyword]` returns successive matching sentences from the
current page. An unsuccessful search keeps the previous page and lookup cursor.
Successful search resets lookup. Disambiguation retries bracket the entity,
bounded to three queries. HTML is decoded as Unicode without the original
wrapper's brittle double-decoding routine.

`envs/hotpotqa/cache/<sha256(query UTF-8)>.json` stores an immutable parsed
snapshot, exact query, source URL, wrapper version and content checksum. A file
lock covers each first fetch and atomic cache write, including across workers.
Both successful pages and search misses are cached; timeouts, invalid pages and
HTTP errors are not. Requests have a 20-second timeout and two retries.
Every evaluation record stores its queries and snapshot hashes.

Use the **same preserved cache** for model comparisons. Live Wikipedia can
change before a query's first fetch; the cache does not claim to reproduce a
historical 2022 Wikipedia snapshot. `--offline` disables Wikipedia fetches,
while still allowing the explicitly configured model endpoint. A missing query
raises an error containing the query and cache path. Evaluation leaves that
question pending and writes incomplete metrics. A teacher pool records the
failed episode and any already incurred model usage, then continues. To replay
a run elsewhere, copy the cache together with its query snapshots. No cache
eviction or refresh is performed automatically.

## Evaluation and BFAS

The same standalone evaluator also accepts `--dataset 2wiki`, `musique`, or
`bamboogle`. These use frozen local OOD inventories, with no new training or
teacher collection path. See [the multi-hop construction report](multihop_evaluation.md)
for selection, alias/abstention scoring, annotation granularity, and the EM/F1
versus LLM-judge comparability limit.

```bash
.venv/bin/python tools/hotpotqa_eval.py \
  --base-url http://localhost:8900/v1 --model bfas-policy \
  --start 0 --n 500 --out results/hotpotqa/base

# Add --offline once all queries for this policy are cached.
PYTHONPATH=src .venv/bin/python -m bfas.run \
  --benchmark hotpotqa --arm base --seeds 1 --gpu 0
```

The adapter is server-backed like WebShop. BFAS owns the student server;
standalone evaluation addresses an existing OpenAI-compatible endpoint.
The BFAS default student is `Qwen/Qwen3.5-4B`, overridable with
`BFAS_HOTPOTQA_MODEL`. Evaluation uses live Wikipedia retrieval by default,
including RTD and paper-baseline campaigns. `BFAS_HOTPOTQA_OFFLINE=1` or the
standalone `--offline` flag explicitly selects cache replay; `offline=False`
overrides the environment for Python callers. No evaluation module forces
offline mode. The shared campaign module is `bfas.rtd.benchmarks.adapter_evaluation`.
Student authentication can be supplied with `BFAS_STUDENT_API_KEY`.

Before training workers, model export, or serving, a retrieval preflight checks
the configured mode. Live mode makes one uncached request (through the same
fetch/parser) and checks cache writability. Offline mode requires a nonempty,
readable cache and validates snapshot identities/checksums. This cannot prove
coverage of future student queries: explicitly offline evaluations still fail
on any later cache miss. Preflight does not call a model or buy teacher tokens.

Each output directory has `identity.json`, `records.jsonl`, and `metrics.json`.
Records include ID, source index, question, prediction, gold, EM/F1, steps,
model call count, responses, actions, observations, query hashes and errors.
Metrics include EM, F1, mean steps, per-category EM, requested/completed counts,
ordered IDs and decoding settings. Incomplete evaluations, infrastructure
errors, and interruptions stop the run and write `status="failed"`,
`complete=false`, and an error reason. EM/F1, headline, mean score, and category
scores are null; diagnostic partial scores live under `partial_metrics`.
Completed records resume without model calls. Changing the model, endpoint,
retrieval mode, range or protocol identity requires a new output directory. Use a new directory
if the weights behind an unchanged served model alias change.

Student turn prompts come from the policy tokenizer's chat template. Guidance
is collection-only; guided rollouts also retain deployment turns without the
task-specific demonstration for BFAS pool rendering and importance scoring.

## Teacher pool and accounting

The following command **does purchase teacher generations**; it was not run
during implementation:

```bash
BFAS_TEACHER=openai/gpt-5.6-luna \
.venv/bin/python tools/hotpotqa_teacher_pool.py \
  --workers 4 --max-tokens 400000 --max-usd 5 \
  --out envs/hotpotqa/teacher_pool
```

`BFAS_TEACHER` defaults to `openai/gpt-5.6-luna`, using the shared
`appworld_teacher` configuration/client and `OPENAI_API_KEY`. Pool service tier
defaults to `flex`, overridable through the shared client's
`BFAS_OPENAI_SERVICE_TIER`. There are at most three attempts per support task:
requested temperatures 0, 0.7, 0.7. The shared client omits temperature for Luna.
Teachers have a 2,048-token completion limit including hidden reasoning.
Reasoning models do not receive unsupported `stop`; output is truncated locally
at the observation boundary, after recording the entire provider usage.

Every request reserves an upper bound for UTF-8 prompt tokens plus its output
cap **before sending**, under a shared worker lock and with an fsynced journal.
Automatic transport retries are disabled for these calls. Prompt plus completion
tokens count toward `--max-tokens`; cached tokens are already included in prompt
tokens. Cost caps use configurable assumed USD/Mtok rates (defaults: 0.10 input,
0.01 cached input, 0.60 output); these are budget parameters, not a live price
lookup. Override them using `--usd-per-mtok-{in,cached,out}` to match your route.
The next request must fit both caps at its full reservation. Idle workers wait
for in-flight reservations to settle when they may release enough headroom.
`BFAS_TEACHER_MIN_INTERVAL_S` optionally spaces episode attempts.

When request usage is missing (including HTTP errors and timeouts), the pool
allows three total tries of the same request, with 1- and 2-second backoffs or
the server's `Retry-After`, whichever is longer. Every try reserves separately;
each unknown call is charged its full upper bound as `estimated`, even if a
later try succeeds. After three failures the task is exhausted and its worker
continues with the next task. HTTP 401 stops collection immediately without a
retry. Other request failures do not stop collection while the hard caps allow
further requests. Interruptions and accounting/identity errors still fail closed.

The simpler pool format follows the accounting and identity conventions in
`tc-alignment-uni/tools/alfworld_teacher_pool.py`, without its sealed RTD bank:

- `identity.json`: support manifest, teacher, endpoint overrides, service tier,
  prompt/protocol versions, code hashes, prices and decoding limits. Incompatible
  resumes fail before requests. Raising explicit token/cost caps on resume is
  allowed; existing spend still counts.
- `usage.jsonl`: append-only request journal. The latest row per call ID is
  authoritative: a reservation, exact `reported` usage, or an `estimated` failed
  call with its error, HTTP status when available, and retry-exhaustion flag.
  Cached input and hidden reasoning are never double-counted.
- `teacher_ledger.jsonl`: standard BFAS episode rows keyed by task, teacher and
  zero-based attempt index. Both failures and successes retain usage and
  `tokens_spent` (completion tokens). `usage_status=reported` identifies exact
  counts. Missing/uncertain usage retains its full reservation, is marked
  `estimated`, and is included alongside any exact retry usage. A successful
  retry can produce a verified demo, with the combined usage still marked
  `estimated`. Orphan reservations become estimated charges on resume and are
  recovered as a paid failed episode without replaying its requests.
- `attempts.jsonl`: append-only audit rows keyed by task and zero-based attempt
  index, including failed attempts. Each row retains the question, step history
  with thoughts/actions/observations, raw responses, predicted and gold answers,
  EM/F1, termination reason and token usage. A purchase whose trajectory was lost
  during interruption is marked `trajectory_available=false`; it is never
  replayed to reconstruct the missing output.
- `demos.json`: verified demos with portable prompts, exact message contexts,
  targets and worked examples. `summary.json`: coverage, spend, limits,
  uncertain requests, completion flag and stop reason.

There is one collection process per output directory, enforced with `flock`;
`--workers` gives concurrent episodes with separate Wikipedia lookup cursors.
Verified tasks, exhausted requests and purchased attempts are not repurchased
on resume. Keep the same output directory to preserve accounting; distinct
directories are distinct purchase inventories. A crash leaving a partial
journal line fails closed.

To reuse that pool in BFAS without repurchasing its attempts:

```bash
BFAS_HOTPOTQA_TEACHER_POOL="$PWD/envs/hotpotqa/teacher_pool" \
PYTHONPATH=src .venv/bin/python -m bfas.run \
  --benchmark hotpotqa --arm sft --seeds 1 --gpu 0
```

The adapter validates pool identity, renders archived message contexts through
the student tokenizer, and imports all episode rows into the shared BFAS
HotpotQA ledger while preserving spend and timestamps. Conflicting purchases
fail instead of being overwritten. The standard BFAS gateway then reuses the
verified demos and attempt counts. Standard BFAS acquisition follows the shared
three-attempt protocol; the pool command owns the aggregate hard caps. The pool
and BFAS ledger are two representations of the same imported spend; do not add
their totals together.

## CPU validation

```bash
.venv/bin/python -m pytest -q tests/test_hotpotqa*.py
```

Tests stub model clients, teacher requests and Wikipedia. They cover prompt
integrity, action parsing and fallback, EM/F1, seven-step limits, deployment
rendering, deterministic concurrent caching, offline failure, split identity,
atomic/idempotent setup, synthetic parquet conversion and mirror fallback,
exact ledger usage, hard caps and pool resume/import.
