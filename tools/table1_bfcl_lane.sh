#!/usr/bin/env bash
# Table 1 BFCL baselines under the unified budget (sealed-pool random acquisition, B=<cap>):
# train with the bfclv2 recipe on the audited pool, then run the standard official campaign.
# Usage: bash tools/table1_bfcl_lane.sh <pci_gpu_index> <gpu_uuid> <cap> <arm:seed> [...]
set -uo pipefail
PROJ=/home/xueqi/hq/projects/tc-alignment-table1; cd "$PROJ"
GPU_IDX=$1; GPU_UUID=$2; CAP=$3; shift 3
for spec in "$@"; do
  ARM=${spec%%:*}; SEED=${spec##*:}; tag="bfclB${CAP}_${ARM}_s${SEED}"
  if [ -f "results/bfcl_std/$tag/data_overall.csv" ]; then echo "[t1] $tag already scored, skip"; continue; fi
  case $ARM in
    star) POOL=$PROJ/data/bfcl_sft/pool_bfcl_star.jsonl; EXTRA=() ;;
    pbsd_insp) POOL=$PROJ/results/table1_audit/pools_native/bfcl/B${CAP}_seed${SEED}/pool_pbsd_insp.jsonl; EXTRA=(AW_DISTILL=pbsd) ;;
    sad) POOL=$PROJ/results/table1_audit/pools_native/bfcl/B${CAP}_seed${SEED}/pool_sad.jsonl; EXTRA=(AW_DISTILL=sad) ;;
    pbsd_agent) POOL=$PROJ/results/table1_audit/pools_native/bfcl/B${CAP}_seed${SEED}/pool_pbsd_agent_costed.jsonl; EXTRA=(AW_DISTILL=pbsd_agent AW_GRAD_CKPT=1) ;;
    *) POOL=$PROJ/results/table1_audit/pools_native/bfcl/B${CAP}_seed${SEED}/pool_${ARM}.jsonl; EXTRA=() ;;
  esac
  echo "[t1] $(date -u +%FT%TZ) TRAIN $tag pool=$(basename $POOL) rows=$(wc -l < $POOL) gpu=$GPU_UUID"
  if [ ! -d "results/appworld_students/$tag/adapter" ]; then
    env CUDA_VISIBLE_DEVICES=$GPU_UUID PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True AW_MAX_PROMPT_TOKENS=4096 AW_TRUNCATE_SIDE=tail "${EXTRA[@]}" \
      AW_POOL_PATH=$POOL PYTHONPATH=src:. \
      .venv/bin/python src/appworld_train.py --selection full --budget 999999 --seed $SEED --student Qwen/Qwen3.5-4B --tag "$tag" \
      > logs/table1_train_$tag.log 2>&1 && echo "[t1] $tag trained" || { echo "[t1] $tag TRAIN FAILED"; continue; }
  else echo "[t1] $tag adapter exists, skip training"; fi
  echo "[t1] $(date -u +%FT%TZ) EVAL $tag on pci $GPU_IDX"
  bash tools/bfcl_std_campaign.sh $GPU_IDX auto "$tag" > logs/table1_eval_$tag.log 2>&1 && echo "[t1] $tag EVAL DONE $(head -2 results/bfcl_std/$tag/data_overall.csv | tail -1 | cut -d, -f1-3)" || echo "[t1] $tag EVAL FAILED"
done
echo "[t1] LANE COMPLETE"
