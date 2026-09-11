# tau2 on rai: Gemma 12B / Luna

The official [sierra-research/tau2-bench](https://github.com/sierra-research/tau2-bench)
checkout is pinned to
[`a2c024725189473d2d7cea3a5cfdbcc67478e41f`](https://github.com/sierra-research/tau2-bench/commit/a2c024725189473d2d7cea3a5cfdbcc67478e41f).
Source and core data live in `envs/tau2/repo/`; the independent uv virtual
environment is **`envs/tau2/.venv/`**, not `repo/.venv`.
The core data is tracked by the official repository and arrives with the clone;
no separate data download is needed.

## Install and offline validation

From this checkout, with `uv`, Git, and Python 3.12 available:

```bash
bash scripts/setup_tau2_rai.sh
```

All installer writes stay in `envs/tau2`: source, data, environment, uv and XDG
caches, temporary build files, `installed-commit.txt`, and
`requirements-installed.txt`. The script does not write this document, modify
global Python packages, start vLLM, or run a model request. It rejects an
`envs/tau2` path that resolves outside this checkout. Repeated runs retain the
environment and refuse modified tracked vendor files. Dependencies satisfy the
pinned project's constraints; their exact installed versions are captured in
`requirements-installed.txt`. The upstream lock file is older than its current
project metadata, so installation resolves from `pyproject.toml`.

This workspace originally had `envs` symlinked to
`/home/xueqi/hq/projects/tc-alignment/envs`. For this installation it was replaced
with a local directory; its other entries remain links to their original
environments. The original tau2 installation was left untouched. This workspace
preparation is separate from the confined installer.

On 2026-09-10 the shell's GitHub lookup failed (`Could not resolve host:
github.com`). The official commit was verified through the GitHub web page, and
installation succeeded using a read-only local source/cache seed. The final
reproducible command used here was:

```bash
TAU2_SOURCE_CACHE=/home/xueqi/hq/projects/tc-alignment/envs/tau2/repo \
TAU2_UV_CACHE_SEED=/home/xueqi/.cache/uv \
TAU2_PYTHON=/home/xueqi/.local/share/uv/python/cpython-3.12-linux-x86_64-gnu/bin/python3.12 \
TAU2_OFFLINE=1 \
bash scripts/setup_tau2_rai.sh > envs/tau2/setup.log 2>&1
```

The source seed uses `git clone --no-hardlinks` and resets origin to the
official repository. The cache seed copies wheels and indexes into the sandbox,
with archive links pointing inside it. It never uses the shared cache in place.
The initial network failure is recorded in `envs/tau2/setup-network.log`.
Installed versions include Python 3.12.13, tau2 1.0.1, LiteLLM 1.82.6,
OpenAI SDK 3.13.0, and pytest 9.1.1.

The installer runs these offline checks (all passed):

```bash
export TAU2_DATA_DIR="$PWD/envs/tau2/repo/data"
export LITELLM_LOCAL_MODEL_COST_MAP=True
export PYTHONDONTWRITEBYTECODE=1
envs/tau2/.venv/bin/tau2 --help
envs/tau2/.venv/bin/tau2 check-data
envs/tau2/.venv/bin/python - <<'PY'
from tau2.run import get_tasks
for domain in ('retail', 'airline', 'telecom'):
    tasks = get_tasks(domain, task_split_name='test')
    assert tasks and len({task.id for task in tasks}) == len(tasks)
    print(domain, len(tasks))
PY
(cd envs/tau2/repo && ../.venv/bin/python -m pytest tests/test_tasks.py -q)
```

Task counts are retail **40**, airline **20**, telecom **40** (100 total).
Tau2's three task-loading unit tests passed. These are the official `test`
splits requested for this experiment. Upstream's public leaderboard convention
uses `base` (114 retail, 50 airline, 114 telecom); these results must be labeled
`test`, and should not be presented as full leaderboard scores.

Project CPU checks, with the CLI stubbed and HTTP requests mocked:

```bash
PYTHONDONTWRITEBYTECODE=1 LITELLM_LOCAL_MODEL_COST_MAP=True \
envs/tau2/.venv/bin/python -m pytest \
  tests/test_tau2_adapter.py tests/test_tau2_eval.py \
  tests/test_tau2_litellm.py tests/test_bfas_ledger.py -q
bash -n scripts/setup_tau2_rai.sh
envs/tau2/.venv/bin/python tools/tau2_eval.py --help
envs/tau2/.venv/bin/python tools/tau2_teacher_probe.py --help
```

The LiteLLM test uses `httpx.MockTransport` and forbids socket connections. It
checks the actual serialized Chat Completions body and cached-token response.
No GPU workload or real API completion was run during setup.

## Pair configuration

```bash
source configs/tau2_rai.env
# OPENAI_API_KEY must already be exported by the caller.
# This config defaults BFAS_OPENAI_SERVICE_TIER to flex; a preset value is kept.
```

| Role | Model / endpoint | Decoding |
| --- | --- | --- |
| Student | `google/gemma-4-12B-it`, served as `gemma4-12b-base` at `http://localhost:8950/v1` | temperature 0, max output 2048 |
| User simulator | `openai/gpt-5.6-luna`, official OpenAI API | temperature omitted, `service_tier=flex` |
| Teacher agent | `openai/gpt-5.6-luna`, official OpenAI API | temperature omitted, `service_tier=flex` |
| Required native NL judge | `openai/gpt-5.6-luna`, official OpenAI API | temperature omitted, `service_tier=flex` |

Credentials are inherited through `OPENAI_API_KEY`, never embedded in command
arguments. The official endpoint is fixed at `https://api.openai.com/v1`, so an
ambient Ollama/Azure/`OPENAI_BASE_URL` setting cannot redirect Luna. The local
student receives `EMPTY`, or `BFAS_STUDENT_API_KEY` in the bounded tools.
`BFAS_TAU2_TEACHER_MODEL` takes precedence over legacy `BFAS_TAU2_TEACHER` and
`BFAS_TEACHER`. Explicit `azure/` and legacy unprefixed Ollama models remain
supported; mixed official/OpenAI-compatible metered routes must use separate
credentials.

Official OpenAI documentation describes
[Luna](https://developers.openai.com/api/docs/models/gpt-5.6-luna) and
[Flex processing](https://developers.openai.com/api/docs/guides/flex-processing).
The rates below are the requested experiment estimates, not dynamically fetched
prices. The first live run encountered an unmapped LiteLLM Luna price and a
student server missing automatic tool-choice support. The fixes below are
validated offline; no live evaluation was repeated to validate them.

## Student server: native Gemma 4 tool calls

The installed vLLM **0.27.1** in
`/home/xueqi/hq/projects/tc-alignment/envs/vllm-serve/.venv` provides the parser
name **`gemma4`**. Inspection of `vllm/tool_parsers/__init__.py` shows that it
registers `Gemma4EngineToolParser` from `gemma4_engine_tool_parser.py`; that
adapter delegates to `vllm/parser/gemma4.py`. This parser recognizes the native
format also parsed by the gorilla checkout's
`bfcl_eval/model_handler/local_inference/gemma4_fc.py`:

```text
<|tool_call>call:lookup{key:<|"|>value<|"|>}<tool_call|>
```

It supports the native string delimiters and nested objects/arrays. Use the
built-in `gemma4` parser; no `--tool-parser-plugin` file is needed for this
installation. Add **both** flags to the existing student launch command:

```bash
--enable-auto-tool-choice --tool-call-parser gemma4
```

For example (launch only when ready for GPU serving; retain the site's other
model, memory, and parallelism settings):

```bash
envs/vllm-serve/.venv/bin/vllm serve google/gemma-4-12B-it \
  --served-model-name gemma4-12b-base --port 8950 \
  --enable-auto-tool-choice --tool-call-parser gemma4
```

`tools/tau2_eval.py` checks the student once before starting tasks or paid
simulator/judge calls: a Chat Completions request with a tiny function schema,
`tool_choice="auto"`, `max_tokens=1`, no streaming, a 30-second timeout, and no
retries. It uses `BFAS_STUDENT_API_KEY` or `EMPTY`. Rejection (including the
missing-flags HTTP 400) stops the command immediately with the server error and
the required flags. Acceptance verifies endpoint capability, not successful
generation of a complete tool call. Nothing executes the probe's tool. Either
zero cap skips this check and all model calls; the teacher-only probe also skips
the student check. The evaluation tools never launch or restart vLLM.

## Evaluation commands (prepared, not executed)

These commands use an already-running student server. They never start vLLM.
They make paid API calls when run with nonzero caps.

```bash
source configs/tau2_rai.env
envs/tau2/.venv/bin/python tools/tau2_eval.py \
  --base-url http://localhost:8950/v1 --model gemma4-12b-base \
  --max-usd 3 --max-tasks 100 --seed 0 \
  --out-dir results/tau2/gemma4-12b-test

envs/tau2/.venv/bin/python tools/tau2_teacher_probe.py \
  --tasks-per-domain 5 --max-usd 3 --max-tasks 15 --seed 0 \
  --out-dir results/tau2/luna-probe
```

Each task has one trial (pass^1), concurrency 1, and native `--max-steps 30`.
Steps use tau2's orchestrator counting, including tool/environment steps; 30
does not mean 30 paired agent/user exchanges. Task order is deterministic and
interleaves retail, airline, telecom; the probe selects the first N test IDs per
domain. `--max-tasks` is a total cap across domains, including failed attempts.
Its default is all selected tasks; `--max-usd` defaults to 3. Either zero cap
makes no calls. Both paid participants and the judge have a default per-request
`--max-completion-tokens 2048` ceiling (including reasoning tokens).

The tools invoke the unchanged vendored CLI through
`tools/tau2_guarded_cli.py`. Its runtime hook wraps tau2's LiteLLM call to
enforce the budget. The pinned retail tasks include required natural-language
assertions; the hook preserves the native rubric/evaluator and selects Luna as
its judge, with separate usage accounting. The adapter also applies this judge
selection when its configured teacher is Luna. Vendor source and task data are
not patched.

## Outputs and budget semantics

Every run needs a new output directory; existing results cannot be overwritten
or silently resumed. It contains:

- `tasks.jsonl`: one record for every attempted task, with verdict, termination,
  role-specific usage, and estimated cost.
- `tasks/NNNN-native.json`: native results for completed CLI runs.
- `metrics.json`: per-domain and task-weighted overall pass^1, task counts,
  simulator/teacher/judge usage, cost, coverage, and stop reason. Scores are
  fractions; no attempted tasks produces `null`. Infrastructure failures count
  as non-passes. `complete=false` identifies capped evaluations.
- `ledger/tau2.jsonl`: standard BFAS ledger rows for every reported paid call.
  Purposes are `user_sim`, `teacher_probe`, and `teacher_judge`; probe and judge
  records do not consume demo attempts. Pipeline demo acquisition continues to
  record teacher episodes with `purpose="teacher"` in its normal shared ledger.
- `budget.json`: durable request reservations and settled usage, including
  incomplete/unknown charges. `request-config.json` holds no credentials.

Usage contains `prompt_tokens`, `cached_tokens`, and `completion_tokens`.
`cached_tokens` comes from the returned
`usage.prompt_tokens_details.cached_tokens` (zero when absent); prompt and
completion counters come directly from returned usage. Cached input is a
subset of total prompt input. `BFAS_OPENAI_SERVICE_TIER` selects this local
Luna price table, in USD per million tokens:

| Tier | Uncached input | Cached input | Output |
| --- | ---: | ---: | ---: |
| Unset, `default`, or `standard` (sent as `default`) | 0.20 | 0.02 | 1.20 |
| `flex` | 0.10 | 0.01 | 0.60 |

`configs/tau2_rai.env` selects flex unless a tier was already set. Override it
with `export BFAS_OPENAI_SERVICE_TIER=default` for standard service. Other
values (including `auto` and `priority`) fail before requests because they have
no price in this experiment. The chosen tier is recorded in `budget.json`,
`request-config.json`, and `metrics.json` and applied to every paid role.
Estimated USD is:

```text
((prompt_tokens - cached_tokens) * input_rate
 + cached_tokens * cached_input_rate
 + completion_tokens * output_rate) / 1_000_000
```

At guarded CLI startup, the same table registers `openai/gpt-5.6-luna` and the
bare response name `gpt-5.6-luna` with `litellm.register_model`. This prevents
LiteLLM's missing built-in mapping from aborting Luna calls or breaking native
tau2 cost reporting. Budget reservations, settlements, and the BFAS usage
ledger use our table and returned counters independently of LiteLLM's cost map
or response-cost metadata. A missing LiteLLM price is not an unknown charge;
missing/invalid provider usage and transport failures still retain reservations.

`teacher_usage` includes teacher-agent and judge calls; `judge_usage` is its
separately reported subset. Total cost includes all paid roles, including
failed-task usage. Ledger temperature fields remain numeric for compatibility;
they do not imply that a temperature was sent to Luna.

The dollar cap is enforced before each paid request by reserving uncached input
at a conservative serialized-text byte bound, plus a framing allowance and the
enforced output-token maximum. Requests whose input bound exceeds 272,000 tokens
are refused. Reservations are persisted before sending; successful usage settles
them at the requested rates. Cache hits can release funds. This can stop before
the nominal remaining dollars are exhausted. Internal provider retries are
disabled. Missing usage, timeouts, and interrupted requests retain their full
reservation and stop further calls. `charged_upper_bound_usd` includes these
reservations; `estimated_usd` includes only reported usage. The cap is local to
this run and these rates, not an account-wide billing limit.
