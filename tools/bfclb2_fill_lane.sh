#!/usr/bin/env bash
# Fill missing truncation-fixed BFCL baseline cells (bfclb2_<arm>_s<seed>): train with the
# bfclv2 recipe, then run the standard official campaign. Usage:
#   bash tools/bfclb2_fill_lane.sh <pci_gpu_index> <gpu_uuid> <arm:seed> [<arm:seed> ...]
set -uo pipefail
PROJ=/home/xueqi/hq/projects/tc-alignment; cd "$PROJ"
GPU_IDX=$1; GPU_UUID=$2; shift 2
declare -A POOL=( [sft]=pool_bfcl_ds_sft.jsonl [sad]=pool_bfcl_ds_sft.jsonl [agentkd]=pool_bfcl_ds_sft.jsonl \
  [ddpo]=pool_bfcl_ds_ddpo.jsonl [pbsd]=pool_bfcl_ds_pbsd.jsonl [bbopd]=pool_bfcl_ds_bbopd.jsonl [star]=pool_bfcl_star.jsonl )
for spec in "$@"; do
  ARM=${spec%%:*}; SEED=${spec##*:}; tag="bfclb2_${ARM}_s${SEED}"
  if [ -f "results/bfcl_std/$tag/data_overall.csv" ]; then echo "[fill] $tag already scored, skip"; continue; fi
  EXTRA=(); case $ARM in sad|agentkd|ddpo|pbsd) EXTRA=(AW_DISTILL=$ARM) ;; esac
  echo "[fill] $(date -u +%FT%TZ) TRAIN $tag on $GPU_UUID"
  if [ ! -f "results/appworld_students/$tag/adapter/adapter_config.json" ] && [ ! -f "results/appworld_students/$tag/adapter/model.safetensors" ]; then
    env CUDA_VISIBLE_DEVICES=$GPU_UUID PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True AW_MAX_PROMPT_TOKENS=4096 AW_TRUNCATE_SIDE=tail "${EXTRA[@]}" \
      AW_POOL_PATH=$PROJ/data/bfcl_sft/${POOL[$ARM]} \
      .venv/bin/python src/appworld_train.py --selection full --budget 999999 --seed $SEED --student Qwen/Qwen3.5-4B --tag "$tag" \
      > logs/bfclb2_fill_train_$tag.log 2>&1 && echo "[fill] $tag trained" || { echo "[fill] $tag TRAIN FAILED"; continue; }
  else echo "[fill] $tag adapter exists, skip training"; fi
  echo "[fill] $(date -u +%FT%TZ) EVAL $tag on pci $GPU_IDX"
  bash tools/bfcl_std_campaign.sh $GPU_IDX auto "$tag" > logs/bfclb2_fill_eval_$tag.log 2>&1 && echo "[fill] $tag EVAL DONE $(grep -h 'Overall' results/bfcl_std/$tag/data_overall.csv 2>/dev/null | head -1 | cut -c1-40)" || echo "[fill] $tag EVAL FAILED"
done
echo "[fill] LANE COMPLETE"
