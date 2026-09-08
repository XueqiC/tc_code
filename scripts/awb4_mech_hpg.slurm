#!/bin/bash
#SBATCH --job-name=awb4
#SBATCH --account=fsu-compsci-dept
#SBATCH --qos=fsu-compsci-dept
#SBATCH --partition=hpg-b200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=120gb
#SBATCH --time=04:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/%x_%A_%a.out
# AppWorld transfer test of the BFCL-validated training mechanisms on the SAME pool
# awb3_oursadv used (pool_v32_adv, AW_V3=1, seed 0 -> 7/40). Arms: +curriculum(turn),
# +preserve lambda=0.5, +both (mirrors bfclb11/presev/bfclb16). Single seed, dev40 eval.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export HF_HOME=/blue/fsu-compsci-dept/xc25.fsu/hq/tools/hf-cache
source .venv/bin/activate
export APPWORLD_ROOT=$PWD/envs/appworld-data
export AW_MAX_PROMPT_TOKENS=4096 AW_TRUNCATE_SIDE=tail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ARMS=(curr pres cp)
ARM=${ARMS[$SLURM_ARRAY_TASK_ID]}
tag="awb4_${ARM}_s0"
case $ARM in
  curr) export AW_CURRICULUM=turn ;;
  pres) export AW_PRESERVE=0.5 ;;
  cp)   export AW_CURRICULUM=turn AW_PRESERVE=0.5 ;;
esac
AW_POOL_PATH=$PWD/data/appworld_sft/pool_v32_adv.jsonl AW_V3=1 \
  python src/appworld_train.py --selection full --budget 999999 \
  --seed 0 --student Qwen/Qwen3.5-4B --tag "$tag"
python src/appworld_eval.py --model "results/appworld_students/$tag/adapter" \
  --split dev --max-tasks 40 --max-steps 24 --seed 0 --tag "${tag}_eval"
echo "[awb4] $tag DONE"
