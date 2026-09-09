#!/bin/bash
# CRCD B-ladder on rai (daytime; Qwen jobs on hpg get cancelled during the day):
# train each arm on the intervention-event pool, then the 507-task PROXY eval.
# Proxy numbers are only comparable with each other (base proxy in results/bfcl_fast/base).
set -uo pipefail
cd "$(dirname "$0")/.."
UUID=$(nvidia-smi --query-gpu=index,gpu_uuid --format=csv,noheader | awk -F", " -v i="${CRCD_GPU:-4}" '$1==i{print $2}')
PORT=${CRCD_PORT:-8990}
export AW_MAX_PROMPT_TOKENS=4096 AW_TRUNCATE_SIDE=tail PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True AW_GRAD_CKPT=1
export CUDA_VISIBLE_DEVICES=$UUID
for ARM in ${CRCD_ARMS:-b2ce b3pref b4cepref}; do
  TAG=crcdr_${ARM}_s0
  case $ARM in
    b2ce)     POOL=data/bfcl_sft/pool_events_ce.jsonl;   EXTRA="" ;;
    b3pref)   POOL=data/bfcl_sft/pool_events_pref.jsonl; EXTRA="AW_DISTILL=ddpo AW_SFT_SKIP=1" ;;
    b4cepref) POOL=data/bfcl_sft/pool_events_pref.jsonl; EXTRA="AW_DISTILL=ddpo" ;;
  esac
  echo "[crcd-rai] === train $TAG ($EXTRA) ==="
  env AW_POOL_PATH=$PWD/$POOL $EXTRA .venv/bin/python src/appworld_train.py --selection full --budget 999999 \
    --seed 0 --student Qwen/Qwen3.5-4B --tag "$TAG" > "logs/${TAG}_train.log" 2>&1 \
    || { echo "[crcd-rai] $TAG TRAIN FAILED"; continue; }
  echo "[crcd-rai] === proxy eval $TAG ==="
  GPU_UTIL=0.45 bash tools/bfcl_fast_eval.sh "$UUID" "$PORT" "$TAG" 2>&1 | grep -E "PROXY|FAILED|NO SCORE"
done
echo "[crcd-rai] CHAIN DONE"
