#!/bin/bash
#SBATCH --job-name=oursmt
#SBATCH --account=fsu-compsci-dept
#SBATCH --qos=fsu-compsci-dept
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=240gb
#SBATCH --time=04:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/oursmt_s%a.log
cd /blue/fsu-compsci-dept/xc25.fsu/hq/tc-alignment
SEED=$SLURM_ARRAY_TASK_ID
tag="bfclb_oursmt_s${SEED}"
AW_POOL_PATH=$PWD/data/bfcl_sft/pool_bfcl_ds_v3mt_v3.jsonl AW_V3=1 \
  .venv/bin/python src/appworld_train.py --selection full --budget 999999 \
  --seed $SEED --student Qwen/Qwen3.5-4B --tag "$tag" \
  && echo "[oursmt] $tag trained" || echo "[oursmt] $tag TRAIN FAILED"
