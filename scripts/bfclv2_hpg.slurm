#!/bin/bash
#SBATCH --job-name=bfclv2
#SBATCH --account=fsu-compsci-dept
#SBATCH --qos=fsu-compsci-dept
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=240gb
#SBATCH --time=02:00:00
#SBATCH --array=0-26
#SBATCH --output=logs/bfclv2_%a.log
cd /blue/fsu-compsci-dept/xc25.fsu/hq/tc-alignment
ARMS=(sft sad agentkd ddpo pbsd bbopd star oursadv oursmt)
ARM=${ARMS[$((SLURM_ARRAY_TASK_ID / 3))]}
SEED=$((SLURM_ARRAY_TASK_ID % 3))
tag="bfclb2_${ARM}_s${SEED}"
declare -A POOL=( [sft]=pool_bfcl_ds_sft.jsonl [sad]=pool_bfcl_ds_sft.jsonl \
  [agentkd]=pool_bfcl_ds_sft.jsonl [ddpo]=pool_bfcl_ds_ddpo.jsonl \
  [pbsd]=pool_bfcl_ds_pbsd.jsonl [bbopd]=pool_bfcl_ds_bbopd.jsonl \
  [star]=pool_bfcl_star.jsonl [oursadv]=pool_bfcl_ds_v3.jsonl \
  [oursmt]=pool_bfcl_ds_v3mt_v3.jsonl )
EXTRA=""
case $ARM in
  sad|agentkd|ddpo|pbsd) EXTRA="AW_DISTILL=$ARM" ;;
  oursadv|oursmt) EXTRA="AW_V3=1" ;;
esac
env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True AW_MAX_PROMPT_TOKENS=4096 AW_TRUNCATE_SIDE=tail $EXTRA \
  AW_POOL_PATH=$PWD/data/bfcl_sft/${POOL[$ARM]} \
  .venv/bin/python src/appworld_train.py --selection full --budget 999999 \
  --seed $SEED --student Qwen/Qwen3.5-4B --tag "$tag" \
  && echo "[bfclv2] $tag trained" || echo "[bfclv2] $tag TRAIN FAILED"
