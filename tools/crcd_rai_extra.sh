#!/bin/bash
# Extra CRCD arms on rai (controls for the B-ladder): 1-epoch variants to separate
# "CE on corrections collapses" from "3 epochs on 137 rows overfits".
set -uo pipefail
cd "$(dirname "$0")/.."
UUID=$(nvidia-smi --query-gpu=index,gpu_uuid --format=csv,noheader | awk -F", " -v i="${CRCD_GPU:-4}" '$1==i{print $2}')
PORT=${CRCD_PORT:-8991}
export AW_MAX_PROMPT_TOKENS=4096 AW_TRUNCATE_SIDE=tail PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True AW_GRAD_CKPT=1
export CUDA_VISIBLE_DEVICES=$UUID
for ARM in ${CRCD_ARMS:-b2ce1ep b4cepref1ep}; do
  TAG=crcdr_${ARM}_s0
  case $ARM in
    b2ce1ep)     POOL=data/bfcl_sft/pool_events_ce.jsonl;   EXTRA="AW_EPOCHS=1" ;;
    b4cepref1ep) POOL=data/bfcl_sft/pool_events_pref.jsonl; EXTRA="AW_DISTILL=ddpo AW_EPOCHS=1" ;;
    r3union)     POOL=data/bfcl_sft/pool_events_pref_v3.jsonl; EXTRA="AW_DISTILL=ddpo AW_SFT_SKIP=1" ;;
    c1anchor)    POOL=data/bfcl_sft/pool_events_pref_c1.jsonl; EXTRA="AW_DISTILL=ddpo AW_SFT_SKIP=1" ;;
    r3uc3)       POOL=data/bfcl_sft/pool_events_pref_v3c3.jsonl; EXTRA="AW_DISTILL=ddpo AW_SFT_SKIP=1" ;;
    b3reffree)   POOL=data/bfcl_sft/pool_events_pref_v2.jsonl; EXTRA="AW_DISTILL=ddpo AW_SFT_SKIP=1 AW_DDPO_REF_FREE=1" ;;
    b3span)      POOL=data/bfcl_sft/pool_events_pref_v2.jsonl; EXTRA="AW_DISTILL=ddpo AW_SFT_SKIP=1 AW_DDPO_SPAN_ONLY=1" ;;
    r3reffree)   POOL=data/bfcl_sft/pool_events_pref_v3.jsonl; EXTRA="AW_DISTILL=ddpo AW_SFT_SKIP=1 AW_DDPO_REF_FREE=1" ;;
  esac
  echo "[crcd-rai] === train $TAG ($EXTRA) ==="
  env AW_POOL_PATH=$PWD/$POOL $EXTRA .venv/bin/python src/appworld_train.py --selection full --budget 999999 \
    --seed 0 --student Qwen/Qwen3.5-4B --tag "$TAG" > "logs/${TAG}_train.log" 2>&1 \
    || { echo "[crcd-rai] $TAG TRAIN FAILED"; continue; }
  echo "[crcd-rai] === proxy eval $TAG ==="
  GPU_UTIL=${GPU_UTIL:-0.45} bash tools/bfcl_fast_eval.sh "$UUID" "$PORT" "$TAG" 2>&1 | grep -E "PROXY|FAILED|NO SCORE"
done
echo "[crcd-rai] EXTRA CHAIN DONE"
