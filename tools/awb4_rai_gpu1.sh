#!/bin/bash
# GPU1 chain (2026-09-01 evening): AppWorld cluster labeling (gate evidence on the
# 21-task v32 pool) -> local awb4_cp_s1 (curr + lambda0.5, seed 1) -> dev40 eval.
set -uo pipefail
cd "$(dirname "$0")/.."
UUID=$(nvidia-smi --query-gpu=index,gpu_uuid --format=csv,noheader | awk -F', ' '$1==1{print $2}')
export CUDA_VISIBLE_DEVICES=$UUID
export AW_MAX_PROMPT_TOKENS=4096 AW_TRUNCATE_SIDE=tail PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "[gpu1] using $UUID"
echo "[gpu1] === cluster labeling appworld pool_v32_adv ==="
: skip cluster labeling, already written; PYTHONPATH=src .venv/bin/python tools/cluster_pool.py --generic \
  --pool data/appworld_sft/pool_v32_adv.jsonl --out results/analysis/appworld_v32_clusters.json \
  || echo "[gpu1] CLUSTER FAILED"
TAG=awb4_cp_s1
echo "[gpu1] === train $TAG ==="
AW_POOL_PATH=$PWD/data/appworld_sft/pool_v32_adv.jsonl AW_V3=1 AW_CURRICULUM=turn AW_PRESERVE=0.5 AW_GRAD_CKPT=1 \
  .venv/bin/python src/appworld_train.py --selection full --budget 999999 \
  --seed 1 --student Qwen/Qwen3.5-4B --tag "$TAG" > "logs/${TAG}_train.log" 2>&1 \
  || { echo "[gpu1] $TAG TRAIN FAILED"; exit 1; }
echo "[gpu1] === eval $TAG ==="
.venv/bin/python src/appworld_eval.py --model "results/appworld_students/$TAG/adapter" \
  --split dev --max-tasks 40 --max-steps 24 --seed 0 --tag "${TAG}_eval" > "logs/${TAG}_eval.log" 2>&1 \
  || { echo "[gpu1] $TAG EVAL FAILED"; exit 1; }
grep -E 'success|TGC|pooled|score' "logs/${TAG}_eval.log" | tail -5
echo "[gpu1] $TAG DONE"
