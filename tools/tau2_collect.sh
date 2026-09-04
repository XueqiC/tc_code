#!/bin/bash
# tau2 collection: student via vllm (GPU by UUID through --gpu), teacher demos via ollama key1
# (already cached), user simulator via Azure gpt-5.4 on the P2 project (both ollama keys
# rate-limited the simulator on 2026-09-02). Credentials env-only.
set -uo pipefail
cd "$(dirname "$0")/.."
SECRETS=~/hq/secrets/llm_apis.env
val() { grep -E "^$1=" "$SECRETS" | head -1 | cut -d= -f2- | tr -d '"' | tr -d "'"; }
HOST=$(val AZURE_APIM_ENDPOINT | awk -F/ '{print $1"//"$3}')
export AZURE_LLM_ENDPOINT="$HOST" AZURE_LLM_KEY="$(val AZURE_${AZURE_SLOT:-P2}_PRIMARY)"
export BFAS_TAU2_USER_MODEL=${TAU2_USER:-azure/gpt-5.4}
export OLLAMA_BASE_URL=${OLLAMA_BASE_URL:-https://ollama.com} OLLAMA_API_KEY="${OLLAMA_API_KEY:-$(cat ~/.ollama_api_key${OLLAMA_KEY_SUFFIX:-})}"   # set OLLAMA_BASE_URL=http://localhost:8999 OLLAMA_API_KEY=proxy to use tools/ollama_proxy.py
UUID=$(nvidia-smi --query-gpu=index,gpu_uuid --format=csv,noheader | awk -F", " -v i="${TAU2_GPU:-4}" '$1==i{print $2}')
export GPU_UTIL=${GPU_UTIL:-0.5} BFAS_PORT_BASE=${BFAS_PORT_BASE:-8940} PYTHONPATH=src
export BFAS_DEMO_WORKERS=4 BFAS_TEACHER_MIN_INTERVAL_S=1 BFAS_TEACHER_QUOTA_WAIT_S=180
# MODE=train runs collect -> train -> official eval; anything else stays a collection dry-run.
# (a "${X:---dry-run}" default treated MODE="" as unset, so train launches silently stayed dry-run; fixed 2026-09-02)
DRY=--dry-run; [ "${MODE:-collect}" = train ] && DRY=""
exec .venv/bin/python -m bfas.run --benchmark tau2 --arm ours --seeds ${TAU2_SEEDS:-1} --gpu "$UUID" $DRY
