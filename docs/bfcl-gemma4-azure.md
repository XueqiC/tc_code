# BFCL: Azure GPT-5.4 teacher and Gemma 4 rows

Collect with the existing driver (credentials are never probed against Ollama
for Azure models):

```bash
BFAS_BFCL_TEACHER=azure/gpt-5.4-FC bash tools/bfcl_teacher_demos.sh
```

The positional model argument still works and takes precedence over the env var.
`AZURE_LLM_ENDPOINT` and `AZURE_LLM_KEY` come from the environment, with missing
values filled from `~/.azure_llm_api`, using `appworld_teacher._azure_credentials`.
`AZURE_OPENAI_API_VERSION` overrides the AppWorld default, `2024-10-21`.
The SDK receives `gpt-5.4` as the Azure deployment name.

`BFAS_BFCL_AZURE_REASONING_EFFORT` defaults to `none`, preserving the driver's
greedy/sampled temperatures. With `low`, `medium`, `high`, or `xhigh`, temperature
is omitted because GPT-5.4 supports it only with reasoning disabled. See the
[official parameter compatibility documentation](https://developers.openai.com/api/docs/guides/latest-model?model=gpt-5.4#GPT-5.4-parameter-compatibility).

The harness is an ignored, shared external checkout in this worktree. The tracked
extension `src/bfas/bfcl_azure.py` registers `azure/gpt-5.4-FC` in the harness's
`model_config.MODEL_CONFIG_MAPPING`, API model map, and supported-model list.
`tools/bfcl_cli.py` performs that registration before invoking the official CLI;
the adapter and collection driver automatically use it for Azure. Direct CLI use:

```bash
envs/bfcl/.venv/bin/python tools/bfcl_cli.py models
```

BFCL writes `usage.completion_tokens` as `output_token_count`. The BFAS ledger
uses those counts for every teacher episode, including failed attempts and any
generated memory prerequisites. Multi-turn step counts are summed. Reasoning
tokens are already part of the provider completion total and are not added twice.
Legacy results with no usage retain the existing text-estimate fallback.

Convert an existing demo pool:

```bash
HF_HUB_OFFLINE=1 .venv/bin/python tools/bfcl_pool_render_gemma4.py \
  --in data/bfcl_sft/pool_bfcl_ds_sft.jsonl \
  --out data/bfcl_sft/pool_gemma4_ds_sft.jsonl --verify
```

Only the cached `google/gemma-4-12B-it` tokenizer is loaded. The renderer calls
`Gemma4FCHandler._format_prompt` for both prompt generation and tool-schema
conversion. Assistant calls are rendered by the same tokenizer's chat template,
including sorted arguments, native string quoting, parallel calls, dotted BFCL
names, and the emitted handoff token. No-call responses use text with the emitted
turn terminator; empty no-call responses carry `_native_fc_empty_response=true`.

The cached template puts an empty thought channel in its generation prompt but
omits it when rendering a completed assistant. The renderer aligns that prefix
explicitly while taking every target byte from the assistant template rendering.
`--verify` checks prompt/target equality and decodes calls with the evaluation
handler to check argument fidelity.

Rows retain metadata and store structured context in `_render_context`. The
top-level `messages` key is removed so the trainer uses the serialized prompt
instead of retemplating without tools. Context can come from `_render_context`,
explicit messages and functions/tools, a legacy Qwen FC prompt, or the original
single-turn BFCL task referenced by an old demo-pool row. Whole-episode targets
without per-step context, malformed calls, and unknown/ambiguous tool names fail
with the input line number; they are never silently flattened or dropped.

CPU validation (no model weights or API calls):

```bash
HF_HUB_OFFLINE=1 .venv/bin/python -m pytest -q \
  tests/test_bfcl_azure_teacher.py tests/test_bfcl_gemma4_rows.py
```
