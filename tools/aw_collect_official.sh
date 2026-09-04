#!/bin/bash
# AppWorld collection under the OFFICIAL scaffold (adapters/appworld_official.py), single seed,
# teacher = Azure gpt-5.4 (gpt-5.6-luna deployment is saturated department-wide, 2026-09-02).
# Credentials come from ~/hq/secrets/llm_apis.env via env only. GPU pinned by UUID through --gpu
# (bfas ServingLane uses --gpu as CUDA_VISIBLE_DEVICES).
set -uo pipefail
cd "$(dirname "$0")/.."
SECRETS=~/hq/secrets/llm_apis.env
val() { grep -E "^$1=" "$SECRETS" | head -1 | cut -d= -f2- | tr -d '"' | tr -d "'"; }
HOST=$(val AZURE_APIM_ENDPOINT | awk -F/ '{print $1"//"$3}')
export AZURE_LLM_ENDPOINT="$HOST" AZURE_LLM_KEY="$(val AZURE_${AZURE_SLOT:-P2}_PRIMARY)"   # P1 budget exhausted 2026-09-02 03:40; AZURE_SLOT=P1 to switch back
export BFAS_TEACHER=${AW_TEACHER:-gpt-5.4}
export OLLAMA_BASE_URL=${OLLAMA_BASE_URL:-https://ollama.com} OLLAMA_API_KEY="${OLLAMA_API_KEY:-$(cat ~/.ollama_api_key${OLLAMA_KEY_SUFFIX:-})}"   # set OLLAMA_BASE_URL=http://localhost:8999 OLLAMA_API_KEY=proxy to use tools/ollama_proxy.py
UUID=$(nvidia-smi --query-gpu=index,gpu_uuid --format=csv,noheader | awk -F", " -v i="${AW_GPU:-1}" '$1==i{print $2}')
export GPU_UTIL=${GPU_UTIL:-0.5} BFAS_PORT_BASE=${BFAS_PORT_BASE:-8950} PYTHONPATH=src
export BFAS_APPWORLD_PROCESSES=${BFAS_APPWORLD_PROCESSES:-4}
export BFAS_TEACHER_MIN_INTERVAL_S=1 BFAS_TEACHER_QUOTA_WAIT_S=180
# MODE=train runs collect -> train -> official eval; anything else stays a collection dry-run.
# (a "${X:---dry-run}" default treated MODE="" as unset, so train launches silently stayed dry-run; fixed 2026-09-02)
DRY=--dry-run; [ "${MODE:-collect}" = train ] && DRY=""
exec .venv/bin/python -m bfas.run --benchmark appworld --arm ours --seeds ${AW_SEEDS:-1} --gpu "$UUID" $DRY
