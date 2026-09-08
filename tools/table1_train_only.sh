#!/usr/bin/env bash
# Train-only helper for Table 1 arms (evaluation runs later on the Blackwell lanes).
# Usage: bash tools/table1_train_only.sh <gpu_uuid> <cap> <arm:seed> [...]
set -uo pipefail
PROJ=/home/xueqi/hq/projects/tc-alignment-table1; cd "$PROJ"
GPU_UUID=$1; CAP=$2; shift 2
for spec in "$@"; do
  ARM=${spec%%:*}; SEED=${spec##*:}; tag="bfclB${CAP}_${ARM}_s${SEED}"
  case $ARM in
    pbsd_agent) POOL=$PROJ/results/table1_audit/pools_native/bfcl/B${CAP}_seed${SEED}/pool_pbsd_agent_costed.jsonl; EXTRA=(AW_DISTILL=pbsd_agent AW_GRAD_CKPT=1) ;;
    sad) POOL=$PROJ/results/table1_audit/pools_native/bfcl/B${CAP}_seed${SEED}/pool_sad.jsonl; EXTRA=(AW_DISTILL=sad) ;;
    pbsd_insp) POOL=$PROJ/results/table1_audit/pools_native/bfcl/B${CAP}_seed${SEED}/pool_pbsd_insp.jsonl; EXTRA=(AW_DISTILL=pbsd) ;;
    star) POOL=$PROJ/data/bfcl_sft/pool_bfcl_star.jsonl; EXTRA=() ;;
    *) POOL=$PROJ/results/table1_audit/pools_native/bfcl/B${CAP}_seed${SEED}/pool_${ARM}.jsonl; EXTRA=() ;;
  esac
  if [ -d "results/appworld_students/$tag/adapter" ]; then echo "[t1train] $tag exists, skip"; continue; fi
  echo "[t1train] $(date -u +%FT%TZ) TRAIN $tag rows=$(wc -l < $POOL)"
  env CUDA_VISIBLE_DEVICES=$GPU_UUID PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True AW_MAX_PROMPT_TOKENS=4096 AW_TRUNCATE_SIDE=tail "${EXTRA[@]}" \
    AW_POOL_PATH=$POOL PYTHONPATH=src:. \
    .venv/bin/python src/appworld_train.py --selection full --budget 999999 --seed $SEED --student Qwen/Qwen3.5-4B --tag "$tag" \
    > logs/table1_train_$tag.log 2>&1 && echo "[t1train] $tag trained" || echo "[t1train] $tag TRAIN FAILED"
done
echo "[t1train] DONE"
