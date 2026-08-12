# BFCL v4 local-HuggingFace evaluation notes

Date: 2026-08-11

## Outcome and evidence boundary

The official evaluator is the Berkeley Function Calling Leaderboard package:

- PyPI distribution: [`bfcl-eval`](https://pypi.org/project/bfcl-eval/)
- Python import: `bfcl_eval`
- command-line entry point: `bfcl`
- source: [`ShishirPatil/gorilla`, `berkeley-function-call-leaderboard/`](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard)

Do **not** install the unrelated PyPI project named `bfcl`.

This machine has a read-only, externally installed `bfcl_eval==2025.11.19.1`. Its packaged
README, metadata, category mapping, evaluator, and model-handler source were inspected without
importing it or writing to its directories. No network operation, package installation, model
download, dataset download, or BFCL evaluation was run for this setup. The current upstream
version/commit as of 2026-08-11 is **TO-VERIFY** when network access is available; all exact
version-specific statements below refer to the locally inspectable `2025.11.19.1` package.

The wheel contains the v4 questions and ground truth, so there is no separate BFCL dataset download
step for this version. Local OSS inference is performed through an OpenAI-compatible server that
the CLI starts with vLLM or SGLang. The evaluator does not directly accept an arbitrary
`AutoModelForCausalLM` object.

With the recommended wheel install, immutable benchmark fixtures reside inside the explicitly
allowed `envs/bfcl-venv` package tree. All mutable/downloaded model data and HuggingFace assets are
redirected to `envs/bfcl/data`; results, caches, configuration, state, and temporary files remain
elsewhere under `envs/bfcl/`.

## BFCL v4 categories

These are the active category names in the inspected v4 `category_mapping.py`. Counts are the
number of scored question records in the bundled files unless noted otherwise.

| Group | CLI category | Records | What it tests |
|---|---|---:|---|
| Non-live, single-turn | `simple_python` | 400 | One Python function call |
| | `simple_java` | 100 | One Java-style function call |
| | `simple_javascript` | 50 | One JavaScript-style function call |
| | `multiple` | 200 | Multiple candidate functions; select/call the needed one |
| | `parallel` | 200 | Multiple calls that can be issued in parallel |
| | `parallel_multiple` | 200 | Parallel calls with multiple candidate functions |
| | `irrelevance` | 240 | Do not call a tool when none is relevant |
| Live, single-turn | `live_simple` | 258 | Simple calls over the live-data snapshot |
| | `live_multiple` | 1,053 | Function selection over the live-data snapshot |
| | `live_parallel` | 16 | Parallel calls over the live-data snapshot |
| | `live_parallel_multiple` | 24 | Parallel calls plus function selection over the live snapshot |
| | `live_irrelevance` | 884 | Abstention on irrelevant live-snapshot prompts |
| | `live_relevance` | 16 | Detect that a relevant tool exists |
| Multi-turn | `multi_turn_base` | 200 | Multi-turn, multi-step tool use |
| | `multi_turn_miss_func` | 200 | Recover when a needed function is initially unavailable |
| | `multi_turn_miss_param` | 200 | Recover when a required parameter is initially missing |
| | `multi_turn_long_context` | 200 | Multi-turn calls with long tool/context inputs |
| Agentic web | `web_search_base` | 100 | Web-search tool use with snippets |
| | `web_search_no_snippet` | 100 | The same questions without result snippets |
| Agentic memory | `memory_kv` | 155 | Key/value and BM25-backed memory |
| | `memory_vector` | 155 | Vector memory using `all-MiniLM-L6-v2` |
| | `memory_rec_sum` | 155 | Recursive-summarization memory |
| Diagnostic, non-scoring | `format_sensitivity` | expanded dynamically | Prompt/tool-format variations; skipped for native-FC model configurations |

Each memory backend also generates 37 prerequisite conversation records (192 generations per
backend, 155 scored memory questions). Each web-search variant reuses the same 100 underlying
questions with a different tool configuration. The format-sensitivity record total is generated
from sampled IDs and a configuration grid; its expanded count is **TO-VERIFY** for any other BFCL
revision.

Useful collection aliases accepted by `--test-category` are `all`, `all_scoring`, `single_turn`,
`non_live`, `live`, `python`, `non_python`, `multi_turn`, `agentic`, `memory`, and `web_search`.
`all` adds the non-scoring `format_sensitivity` diagnostic to `all_scoring`.

The installed tree also contains old/unused files named `exec_simple`, `exec_multiple`,
`exec_parallel`, `exec_parallel_multiple`, `rest`, `sql`, `chatable`, and
`multi_turn_composite`. They are commented out of the active v4 category mapping and must not be
reported as results from the standard `all_scoring` run for this version.

## Metrics

There is not one uniform “tool-call accuracy” calculation for every v4 category.

- Active non-live and live function-call categories use **AST accuracy**. The handler decodes a
  response into function names and typed argument structures, and the AST checker compares them to
  acceptable answers. This is structural matching, not raw string exact match.
- `irrelevance` and `live_irrelevance` score whether the model correctly emits no function call;
  `live_relevance` scores whether it emits a call when one is appropriate.
- Multi-turn entries are executable and state-based: decoded calls are run against BFCL's mock
  classes, and an entry passes only if per-turn state and required execution responses agree with
  ground truth. This is distinct from the older standalone `exec_*` suite.
- Web-search and memory entries are agentic tasks. Their final response must contain an accepted
  answer after tool interaction; comparison is case/whitespace/punctuation normalized.
- `format_sensitivity` reports maximum accuracy delta and standard deviation across prompt formats;
  it is diagnostic and not included in the v4 overall score.

For `bfcl_eval==2025.11.19.1`, **Overall Acc** is a fixed-weight macro-composite:

```text
10% non-live AST + 10% live AST + 10% irrelevance detection
+ 30% multi-turn + 40% agentic (web + memory)
```

Within that calculation, non-live AST macro-averages simple/multiple/parallel/parallel-multiple,
live AST is count-weighted across its four call categories, multi-turn macro-averages its four
categories, and agentic gives equal weight to the web-search and memory summaries. Relevance is
reported separately. The source treats categories omitted from a run as zero when building
summary columns, so an `Overall Acc` produced from only a subset is **not** comparable to the
official leaderboard. Use the category score JSON files for partial runs.

Thus, for a conventional local tool-calling check, report AST accuracy per single-turn category.
For a full BFCL v4 submission, report the official weighted Overall Acc plus its multi-turn and
agentic components. Do not label the inactive `exec_*` metric as the v4 standard executable score.

## Strict-offline limitations

A fully offline local-model run can use non-live, live, multi-turn, `memory_kv`,
`memory_rec_sum`, and `format_sensitivity` (subject to model context length). There are two special
cases:

- `web_search_base` and `web_search_no_snippet` call SerpAPI and fetch public URLs. They cannot run
  in strict offline mode. Consequently, a strict-offline run cannot produce the official complete
  v4 Overall Acc.
- `memory_vector` hardcodes `SentenceTransformer("all-MiniLM-L6-v2")`. It is offline only if that
  encoder has already been staged in the isolated HuggingFace/sentence-transformers cache. Exclude
  it otherwise.

The “live” categories are bundled snapshot data and do not themselves require live web access.

## Storage estimate

The inspected v4 package has:

- 12,296,985 bytes (reported as 12 MiB by `du`) under `bfcl_eval/data`;
- 10,929,183 bytes of active data/support files when `unused_datasets/` is excluded;
- 4,696 raw top-level scored-question records before web/memory expansion and memory prerequisites
  (the separate format-sensitivity configuration file is not a question JSONL file).

Budget roughly 12 MiB for benchmark fixtures. This does not include the Python environment,
vLLM/SGLang, CUDA/Triton caches, model weights, generated responses, or scores. The backend
environment can consume multiple GiB. BF16 weights alone are approximately two bytes per
parameter (about 16 GB for an 8B model), and adapter pre-merging creates another full checkpoint.
Exact environment/model sizes are **TO-VERIFY** for the selected backend and model.

## Commands to run later — do not run from the repository root

The following commands are recommendations for the user to execute. They were **not** executed
during this task. They keep the working directory away from the repository root and redirect known
mutable caches and data roots under `envs/bfcl/`. The virtual environment is the explicitly allowed
exception at `envs/bfcl-venv`.

### 1. Establish isolated paths

Run this block first in every shell/job that installs or evaluates BFCL:

```bash
export TC_ALIGNMENT_ROOT="/home/xueqi/hq/projects/tc-alignment"
export BFCL_ROOT="$TC_ALIGNMENT_ROOT/envs/bfcl"
export BFCL_VENV="$TC_ALIGNMENT_ROOT/envs/bfcl-venv"

mkdir -p \
  "$BFCL_ROOT/data/huggingface/hub" \
  "$BFCL_ROOT/data/huggingface/datasets" \
  "$BFCL_ROOT/data/huggingface/assets" \
  "$BFCL_ROOT/data/huggingface/modules" \
  "$BFCL_ROOT/data/sentence-transformers" \
  "$BFCL_ROOT/data/models" \
  "$BFCL_ROOT/data/merged-models" \
  "$BFCL_ROOT/data/xdg" \
  "$BFCL_ROOT/cache/pip" \
  "$BFCL_ROOT/cache/uv" \
  "$BFCL_ROOT/cache/xdg" \
  "$BFCL_ROOT/cache/torch" \
  "$BFCL_ROOT/cache/vllm" \
  "$BFCL_ROOT/cache/triton" \
  "$BFCL_ROOT/cache/torchinductor" \
  "$BFCL_ROOT/cache/numba" \
  "$BFCL_ROOT/cache/cuda" \
  "$BFCL_ROOT/cache/pycache" \
  "$BFCL_ROOT/cache/torch-extensions" \
  "$BFCL_ROOT/config/xdg" \
  "$BFCL_ROOT/state/xdg" \
  "$BFCL_ROOT/wandb" \
  "$BFCL_ROOT/tmp" \
  "$BFCL_ROOT/runs"

cd "$BFCL_ROOT"
test "$(pwd -P)" = "$BFCL_ROOT"

export BFCL_PROJECT_ROOT="$BFCL_ROOT"
export HF_HOME="$BFCL_ROOT/data/huggingface"
export HF_HUB_CACHE="$BFCL_ROOT/data/huggingface/hub"
export HF_DATASETS_CACHE="$BFCL_ROOT/data/huggingface/datasets"
export HF_ASSETS_CACHE="$BFCL_ROOT/data/huggingface/assets"
export HF_MODULES_CACHE="$BFCL_ROOT/data/huggingface/modules"
export SENTENCE_TRANSFORMERS_HOME="$BFCL_ROOT/data/sentence-transformers"
export XDG_CACHE_HOME="$BFCL_ROOT/cache/xdg"
export XDG_CONFIG_HOME="$BFCL_ROOT/config/xdg"
export XDG_DATA_HOME="$BFCL_ROOT/data/xdg"
export XDG_STATE_HOME="$BFCL_ROOT/state/xdg"
export PIP_CACHE_DIR="$BFCL_ROOT/cache/pip"
export UV_CACHE_DIR="$BFCL_ROOT/cache/uv"
export UV_PYTHON_INSTALL_DIR="$BFCL_ROOT/data/uv-python"
export TORCH_HOME="$BFCL_ROOT/cache/torch"
export VLLM_CACHE_ROOT="$BFCL_ROOT/cache/vllm"
export TRITON_CACHE_DIR="$BFCL_ROOT/cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$BFCL_ROOT/cache/torchinductor"
export NUMBA_CACHE_DIR="$BFCL_ROOT/cache/numba"
export CUDA_CACHE_PATH="$BFCL_ROOT/cache/cuda"
export PYTHONPYCACHEPREFIX="$BFCL_ROOT/cache/pycache"
export TORCH_EXTENSIONS_DIR="$BFCL_ROOT/cache/torch-extensions"
export TMPDIR="$BFCL_ROOT/tmp"
export MPLCONFIGDIR="$BFCL_ROOT/cache/matplotlib"
export WANDB_DIR="$BFCL_ROOT/wandb"
export WANDB_CACHE_DIR="$BFCL_ROOT/cache/wandb"
export WANDB_CONFIG_DIR="$BFCL_ROOT/config/wandb"
export WANDB_MODE=disabled
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
```

These variables cover the writers used by the inspected version and common vLLM/Transformers
dependencies. Backend cache variables can change; **TO-VERIFY** this list when changing the pinned
BFCL/vLLM version. An OS-level sandbox/container with only these two project paths writable is the
strongest enforcement if a new backend is introduced.

### 2. Install the pinned evaluator and vLLM backend

This downloads packages when the user runs it. It must only be run after the isolation block above:

```bash
uv venv --no-project --python 3.11 "$BFCL_VENV"
uv pip install \
  --python "$BFCL_VENV/bin/python" \
  "bfcl-eval[oss-eval-vllm]==2025.11.19.1" \
  peft

touch "$BFCL_ROOT/.env"
test ! -s "$BFCL_ROOT/.env"
"$BFCL_VENV/bin/python" -c \
  'from importlib.metadata import version; print(version("bfcl-eval"))'
"$BFCL_VENV/bin/bfcl" test-categories
"$BFCL_VENV/bin/bfcl" models > "$BFCL_ROOT/supported-models.txt"
```

`2025.11.19.1` is the locally verified version, not a claim that it is the newest. Before choosing a
newer pin, **TO-VERIFY** its categories, aggregate weights, model registry, backend versions, and
cache variables. For Ampere-or-newer GPUs, upstream documentation says SGLang is faster on
multi-turn tasks; the alternative extra is `bfcl-eval[oss-eval-sglang]`. vLLM is the more broadly
compatible recommendation for T4/V100-era hardware.

The inspected CLI loads `$BFCL_PROJECT_ROOT/.env` with `override=True`. Keep this file empty for
strict-offline evaluation, or audit every entry: values there can override the safe path/offline
environment exported by the shell.

### 3. Confirm strict offline mode and identify the model

The model directory must already contain `config.json`, `tokenizer_config.json`, tokenizer files,
and weights. Place/copy it under `envs/bfcl/data/models/`; none of the commands below download a
model.

```bash
test "$HF_HUB_OFFLINE" = 1
test "$TRANSFORMERS_OFFLINE" = 1
test "$HF_DATASETS_OFFLINE" = 1
export LOCAL_SERVER_ENDPOINT=127.0.0.1
export LOCAL_SERVER_PORT=1053

export MODEL_KEY="meta-llama/Llama-3.1-8B-Instruct"
export MODEL_DIR="$BFCL_ROOT/data/models/Meta-Llama-3.1-8B-Instruct"
export BFCL_TAG="llama31-8b-local"

test -f "$MODEL_DIR/config.json"
test -f "$MODEL_DIR/tokenizer_config.json"
```

`MODEL_KEY` is not an arbitrary display name. It must appear in `supported-models.txt` and selects
BFCL's model-specific prompt formatter and output decoder. `--local-model-path` changes where that
registered handler loads weights; it does not register a new architecture. For a model absent from
the registry, the official route is to add a handler and `ModelConfig` in an editable Gorilla
checkout. That extension interface is version-sensitive and is **TO-VERIFY** against the chosen
upstream commit.

Use the `-FC` registry key only when evaluating the corresponding native function-calling format;
the key without `-FC` uses BFCL's prompting path. This choice materially changes formatting and is
part of the benchmark configuration.

### 4. One-case offline smoke generation and evaluation

This creates one explicit test ID and uses partial evaluation. It verifies plumbing only and is not
a leaderboard score:

```bash
printf '%s\n' \
  '{"simple_python":["simple_python_0"]}' \
  > "$BFCL_ROOT/test_case_ids_to_generate.json"

"$BFCL_VENV/bin/bfcl" generate \
  --model "$MODEL_KEY" \
  --run-ids \
  --backend vllm \
  --num-gpus 1 \
  --num-threads 1 \
  --gpu-memory-utilization 0.90 \
  --temperature 0.001 \
  --local-model-path "$MODEL_DIR" \
  --result-dir "runs/$BFCL_TAG/result"

"$BFCL_VENV/bin/bfcl" evaluate \
  --model "$MODEL_KEY" \
  --test-category simple_python \
  --partial-eval \
  --result-dir "runs/$BFCL_TAG/result" \
  --score-dir "runs/$BFCL_TAG/score"
```

Expected locations are
`envs/bfcl/runs/$BFCL_TAG/result/<model>/non_live/BFCL_v4_simple_python_result.json`
and the parallel tree under `score/`. BFCL also creates `result/`, `score/`, and `.file_locks/` at
`BFCL_PROJECT_ROOT` while importing its configuration; all remain under `envs/bfcl/` here.

### 5. Substantial strict-offline run

The following covers the standard non-agentic suite without web access. It deliberately excludes
all agentic categories, so use its component scores and not its zero-filled `Overall Acc`:

```bash
export BFCL_CATEGORIES="non_live,live,multi_turn"

"$BFCL_VENV/bin/bfcl" generate \
  --model "$MODEL_KEY" \
  --test-category "$BFCL_CATEGORIES" \
  --backend vllm \
  --num-gpus 1 \
  --gpu-memory-utilization 0.90 \
  --temperature 0.001 \
  --local-model-path "$MODEL_DIR" \
  --result-dir "runs/$BFCL_TAG/result"

"$BFCL_VENV/bin/bfcl" evaluate \
  --model "$MODEL_KEY" \
  --test-category "$BFCL_CATEGORIES" \
  --result-dir "runs/$BFCL_TAG/result" \
  --score-dir "runs/$BFCL_TAG/score"
```

Add `memory_kv,memory_rec_sum` if desired. Add `memory_vector` only after staging
`all-MiniLM-L6-v2` in the isolated cache and confirming it resolves with offline variables set.
Never add `web_search` or `all_scoring` to a strict-offline job.

To use a pre-existing local vLLM/SGLang endpoint instead, set the two `LOCAL_SERVER_*` variables
and add `--skip-server-setup`. The endpoint must serve the same model represented by `MODEL_KEY`.

### 6. Optional PEFT adapter pre-merge

The BFCL CLI has no `--adapter` option, and its internally constructed vLLM command does not expose
vLLM's LoRA flags. Mirror `src/appworld_eval.py` by merging the adapter first, but save the merged
checkpoint under `envs/bfcl/data/merged-models/`. This requires enough CPU RAM and another full
checkpoint's disk space:

```bash
export BASE_MODEL_DIR="$MODEL_DIR"
export ADAPTER_DIR="$BFCL_ROOT/data/adapters/my-adapter"
export MERGED_MODEL_DIR="$BFCL_ROOT/data/merged-models/$BFCL_TAG"

test -d "$BASE_MODEL_DIR"
test -d "$ADAPTER_DIR"
test ! -e "$MERGED_MODEL_DIR"

"$BFCL_VENV/bin/python" - <<'PY'
import os
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_path = Path(os.environ["BASE_MODEL_DIR"]).resolve()
adapter_path = Path(os.environ["ADAPTER_DIR"]).resolve()
output_path = Path(os.environ["MERGED_MODEL_DIR"]).resolve()

tokenizer = AutoTokenizer.from_pretrained(
    base_path,
    local_files_only=True,
    trust_remote_code=False,
)
base = AutoModelForCausalLM.from_pretrained(
    base_path,
    torch_dtype=torch.bfloat16,
    device_map="cpu",
    local_files_only=True,
    trust_remote_code=False,
)
merged = PeftModel.from_pretrained(
    base,
    adapter_path,
    local_files_only=True,
).merge_and_unload()

output_path.mkdir(parents=True, exist_ok=False)
merged.save_pretrained(output_path, safe_serialization=True)
tokenizer.save_pretrained(output_path)
print(output_path)
PY

export MODEL_DIR="$MERGED_MODEL_DIR"
```

Some custom architectures require `trust_remote_code=True`; the repository convention is `False`,
so changing it should be an explicit reviewed exception. BFCL's own inspected local-server launcher
passes `--trust-remote-code` to the backend even when the tokenizer/config were local.

## Why `src/bfcl_eval.py` was not added

The documented, supported API is the `bfcl generate` / `bfcl evaluate` CLI. The Python helpers that
load cases, build model handlers, write results, and aggregate scores are private implementation
details; importing their configuration also creates directories. More importantly, every supported
local model is tied to a registry entry whose handler controls tool-document formatting and output
decoding. A generic Transformers loop modeled on `src/appworld_eval.py` would bypass those handlers
and could silently produce non-comparable scores. PEFT adapters are also not part of BFCL's public
API.

Therefore the package API is not clear/stable enough to create the requested generic
`--model/--adapter/--category/--tag` Python skeleton without inventing an integration contract.
`src/bfcl_eval.py` is intentionally omitted. The CLI commands above preserve the official prompt,
decode, output, and scoring paths; `BFCL_TAG` supplies the run-tag behavior through isolated result
and score directories.
