#!/bin/bash
#SBATCH --job-name=awb3
#SBATCH --account=fsu-compsci-dept
#SBATCH --qos=fsu-compsci-dept
#SBATCH --partition=hpg-b200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=120gb
#SBATCH --time=04:00:00
#SBATCH --array=0-23
#SBATCH --output=logs/%x_%A_%a.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export HF_HOME=/blue/fsu-compsci-dept/xc25.fsu/hq/tools/hf-cache
source .venv/bin/activate
export APPWORLD_ROOT=$PWD/envs/appworld-data
export AW_MAX_PROMPT_TOKENS=4096 AW_TRUNCATE_SIDE=tail
ARMS=(sft sad agentkd ddpo pbsd bbopd star oursadv)
ARM=${ARMS[$((SLURM_ARRAY_TASK_ID / 3))]}
SEED=$((SLURM_ARRAY_TASK_ID % 3))
tag="awb3_${ARM}_s${SEED}"
case $ARM in
  sft)     export AW_POOL_PATH=$PWD/data/appworld_sft/pool.jsonl ;;
  sad)     export AW_POOL_PATH=$PWD/data/appworld_sft/pool.jsonl AW_DISTILL=sad ;;
  agentkd) export AW_POOL_PATH=$PWD/data/appworld_sft/pool_agentkd.jsonl AW_DISTILL=agentkd ;;
  ddpo)    export AW_POOL_PATH=$PWD/data/appworld_sft/pool_ddpo.jsonl AW_DISTILL=ddpo ;;
  pbsd)    export AW_POOL_PATH=$PWD/data/appworld_sft/pool_pbsd.jsonl AW_DISTILL=pbsd ;;
  bbopd)   export AW_POOL_PATH=$PWD/data/appworld_sft/pool_bbopd.jsonl ;;
  star)    export AW_POOL_PATH=$PWD/data/appworld_sft/pool_star.jsonl ;;
  oursadv) export AW_POOL_PATH=$PWD/data/appworld_sft/pool_v32_adv.jsonl AW_V3=1 ;;
esac
python src/appworld_train.py --selection full --budget 999999 \
  --seed $SEED --student Qwen/Qwen3.5-4B --tag "$tag"
python src/appworld_eval.py --model "results/appworld_students/$tag/adapter" \
  --split dev --max-tasks 40 --max-steps 24 --seed 0 --tag "${tag}_eval"
echo "[awb3] $tag DONE"
