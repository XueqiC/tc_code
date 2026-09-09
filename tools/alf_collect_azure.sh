#!/bin/bash
# ALFWorld teacher collection with the Azure gpt-5.6-luna teacher (ollama weekly quota is
# exhausted on both keys, 2026-09-02). Credentials come from ~/hq/secrets/llm_apis.env
# (the panel's working keys; ~/.azure_llm_api holds stale ones) and never touch argv/logs.
set -uo pipefail
cd "$(dirname "$0")/.."
SECRETS=~/hq/secrets/llm_apis.env
val() { grep -E "^$1=" "$SECRETS" | head -1 | cut -d= -f2- | tr -d '"' | tr -d "'"; }
HOST=$(val AZURE_APIM_ENDPOINT | awk -F/ '{print $1"//"$3}')
export AZURE_LLM_ENDPOINT="$HOST" AZURE_LLM_KEY="$(val AZURE_${AZURE_SLOT:-P2}_PRIMARY)"   # P1 budget exhausted 2026-09-02 03:40; AZURE_SLOT=P1 to switch back
export BFAS_TEACHER=${ALF_TEACHER:-gpt-5.4}   # separate Azure deployment from AppWorld (gpt-5.6-luna) to split the per-deployment rate limit
export OLLAMA_BASE_URL=${OLLAMA_BASE_URL:-https://ollama.com} OLLAMA_API_KEY="${OLLAMA_API_KEY:-$(cat ~/.ollama_api_key${OLLAMA_KEY_SUFFIX:-})}"   # set OLLAMA_BASE_URL=http://localhost:8999 OLLAMA_API_KEY=proxy to use tools/ollama_proxy.py
export BFAS_DEMO_WORKERS=${BFAS_DEMO_WORKERS:-3} BFAS_TEACHER_MIN_INTERVAL_S=1 BFAS_TEACHER_QUOTA_WAIT_S=180
# ALFWorld prompts (goal + admissible commands + 8-line history) run past the 1024 default cap
export AW_MAX_PROMPT_TOKENS=${AW_MAX_PROMPT_TOKENS:-2048}
UUID=$(nvidia-smi --query-gpu=index,gpu_uuid --format=csv,noheader | awk -F", " -v i="${ALF_GPU:-2}" '$1==i{print $2}')
# pin by UUID: plain indices follow CUDA order, which differs from nvidia-smi order on this mixed box
export GPU_UTIL=${GPU_UTIL:-0.45} CUDA_VISIBLE_DEVICES="$UUID" BFAS_PORT_BASE=${BFAS_PORT_BASE:-8930} PYTHONPATH=src
# MODE=train runs collect -> train -> official eval; anything else stays a collection dry-run.
# (a "${X:---dry-run}" default treated MODE="" as unset, so train launches silently stayed dry-run; fixed 2026-09-02)
DRY=--dry-run; [ "${MODE:-collect}" = train ] && DRY=""
exec .venv/bin/python -m bfas.run --benchmark alfworld --arm ours --seeds ${ALF_SEEDS:-1} --gpu "$UUID" $DRY
