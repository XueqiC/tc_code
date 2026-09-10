# BFCL Ollama FC teachers

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
Ollama SDK retries are disabled so collection owns the attempt budget.

Each Ollama inference journals its usage before parsing to `bfas_usage.jsonl`
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
HF_HUB_OFFLINE=1 .venv/bin/python -m pytest -q tests/test_bfcl*.py \
  tests/test_bfas_ledger.py tests/test_bfas_conformance.py
```
