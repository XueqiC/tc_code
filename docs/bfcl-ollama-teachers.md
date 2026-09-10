# BFCL Ollama and OpenRouter FC teachers

The tracked `src/bfas/bfcl_ollama.py` extension registers these API models in
BFCL's model mapping, API model map, and supported list:

| BFCL teacher | Model sent to Ollama |
| --- | --- |
| `ollama/gpt-oss:120b-FC` | `gpt-oss:120b` |
| `ollama/mistral-large-3:675b-FC` | `mistral-large-3:675b` |

Both use Chat Completions with native `tools` and structured `tool_calls`.
`tools/bfcl_cli.py` registers them before importing the official CLI. The adapter
selects that wrapper for generation and evaluation; no harness source edits or
local vLLM server are needed.

Set `OLLAMA_API_KEY`. If unset, the handler checks `~/.ollama_api_key2`, then
`~/.ollama_api_key`. `OLLAMA_BASE_URL` defaults to `https://ollama.com/v1`;
both a host URL and an existing `/v1` URL work. Unrelated `OPENAI_*` credentials
do not override these settings. Collection makes no credential-probe API calls.

```bash
export BFAS_BFCL_TEACHER='ollama/gpt-oss:120b-FC'
export BFAS_BFCL_SKIP_MEMORY_PREREQ=1
bash tools/bfcl_teacher_demos.sh
```

## OpenRouter

The same tracked extension registers these OpenRouter API FC teachers:

| BFCL teacher | Model sent to OpenRouter |
| --- | --- |
| `openrouter/openai/gpt-5.4-FC` | `openai/gpt-5.4` |
| `openrouter/anthropic/claude-sonnet-5-FC` | `anthropic/claude-sonnet-5` |
| `openrouter/google/gemini-3.1-pro-preview-FC` | `google/gemini-3.1-pro-preview` |
| `openrouter/openai/gpt-5.6-luna-FC` | `openai/gpt-5.6-luna` |

Set `OPENROUTER_API_KEY`; an unset, empty, or whitespace-only value raises an
error before client construction. There is no file fallback, and neither
`OLLAMA_API_KEY` nor `OPENAI_API_KEY` satisfies this requirement.
`OPENROUTER_BASE_URL` defaults to `https://openrouter.ai/api/v1`; overrides use
the supplied API base path with trailing slashes removed.

```bash
export OPENROUTER_API_KEY='your-openrouter-key'
export BFAS_BFCL_TEACHER='openrouter/openai/gpt-5.4-FC'
export BFAS_BFCL_SKIP_MEMORY_PREREQ=1
bash tools/bfcl_teacher_demos.sh
```

OpenRouter and Ollama share `OpenAICompatibleHandler`, using OpenAI Chat
Completions native `tools`, exact `usage.completion_tokens`, and the same usage
journal and ledger importer. The full BFCL teacher name labels ledger rows and
separates attempt budgets, including the `openrouter/` prefix and `-FC` suffix.
Registration and CLI model listing do not require credentials or call APIs.

Provider routing and credentials use the `PROVIDERS` table in
`src/bfas/bfcl_teacher.py`: each prefix maps to its base URL environment variable,
API key environment variable, and default URL. Adding a compatible provider
requires one table row plus its model entries in `register_api_models()` in
`src/bfas/bfcl_ollama.py`; the handler and adapter recognize it automatically.
`OllamaOpenAIHandler` and `register_ollama_models()` remain compatibility aliases.
Ollama alone retains its historical file fallback and `/v1` normalization.

## Collection and accounting

A positional model argument overrides `BFAS_BFCL_TEACHER`. The same environment
works with `BFCLAdapter().teacher_demo(task_ids, attempts=3)`. The adapter uses
the acquisition gateway, reusing that teacher's existing verified demos and
buying only unresolved attempts. Changing teachers starts a separate attempt
budget in the shared ledger. The shell driver retains its three batch attempts
and official result/score directories, then merges the first verified result
per demand task. It writes both a model-specific verified-ID file and the
historical `data/bfcl_demos_ds_verified.json` alias used by pool tools.

Both paths append gateway-schema rows to `data/teacher_ledger/bfcl.jsonl`.
`tokens_spent` sums provider `usage.completion_tokens`, recorded by BFCL as
`output_token_count`, including nested multi-turn steps and generated memory
prerequisites. Reasoning is already included and is never added twice. Failed
inference, parsing, rendering, and evaluation attempts retain measured usage.
SDK retries are disabled for both providers so collection owns the attempt budget.

Each inference journals its usage before parsing to `bfas_usage.jsonl`
inside the result directory. A unique attempt ID lets the batch importer recover
usage after an interrupted worker, charge a resumed call separately, and avoid
double charging when the same artifacts are imported again. Raw batch artifacts
remain available for auditing. Historical batch rows without provider usage
stop import with an error instead of silently estimating their cost.

`BFAS_BFCL_SKIP_MEMORY_PREREQ=1` excludes every `memory_*` demand ID before
prerequisite expansion. These IDs are recorded as `unavailable inventory` in
`data/teacher_ledger/bfcl_inventory.jsonl`; they consume no teacher attempts or
tokens and do not enter the merged demo inventory. Unset the flag to collect
memory tasks with their prerequisite chains. Student evaluation and the support
split remain unchanged.

CPU validation, with stubbed APIs and a fake shell-collector checkout:

```bash
HF_HUB_OFFLINE=1 .venv/bin/python -m pytest -q \
  tests/test_bfcl_ollama_teacher.py tests/test_bfcl*.py
```
