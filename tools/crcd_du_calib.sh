#!/bin/bash
# dU calibration study, step 2 (rai, proxy eval): train one preference-only model per dU bucket
# (equal rows, step-matched epochs from data/bfcl_sft/du_calib/plan.json) and score each on the
# 507-task proxy. Gain per bucket vs the base proxy (37.46) is the calibration curve.
set -uo pipefail
cd "$(dirname "$0")/.."
UUID=$(nvidia-smi --query-gpu=index,gpu_uuid --format=csv,noheader | awk -F", " -v i="${CRCD_GPU:-3}" '$1==i{print $2}')
PORT=${CRCD_PORT:-8994}
D=${DU_DIR:-data/bfcl_sft/du_calib}
export AW_MAX_PROMPT_TOKENS=4096 AW_TRUNCATE_SIDE=tail PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True AW_GRAD_CKPT=1
export CUDA_VISIBLE_DEVICES=$UUID
for B in ${BUCKETS:-neg zero small medium large}; do
  [ -f $D/pool_$B.jsonl ] || { echo "[du-calib] $B: no pool"; continue; }
  EP=$(.venv/bin/python -c "import json;print(json.load(open('$D/plan.json'))['$B']['epochs'])")
  TAG=${DU_TAG:-crcddu}_${B}_s0
  [ -f results/bfcl_fast/$TAG/score/data_overall.csv ] && { echo "[du-calib] $B already scored"; continue; }
  echo "[du-calib] === train $TAG ($(wc -l < $D/pool_$B.jsonl) rows, $EP ep) ==="
  env AW_POOL_PATH=$PWD/$D/pool_$B.jsonl AW_DISTILL=ddpo AW_SFT_SKIP=1 AW_DDPO_EPOCHS=$EP .venv/bin/python src/appworld_train.py \
    --selection full --budget 999999 --seed 0 --student Qwen/Qwen3.5-4B --tag "$TAG" > logs/${TAG}_train.log 2>&1 \
    || { echo "[du-calib] $TAG TRAIN FAILED"; continue; }
  GPU_UTIL=${GPU_UTIL:-0.4} bash tools/bfcl_fast_eval.sh "$UUID" "$PORT" "$TAG" 2>&1 | grep -E "PROXY|FAILED|NO SCORE"
  rm -rf results/appworld_students/$TAG/hub_merged
done
echo "[du-calib] CHAIN DONE"
