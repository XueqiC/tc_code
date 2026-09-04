#!/bin/bash
#SBATCH --job-name=bfclb10
#SBATCH --account=fsu-compsci-dept
#SBATCH --qos=fsu-compsci-dept
#SBATCH --partition=hpg-b200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=200gb
#SBATCH --time=08:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/%x_%A_%a.out
# pool v9: all four boundary behaviour classes (call / close / silent / refuse),
# three seeds, each with the full official BFCL evaluation.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export HF_HOME=/blue/fsu-compsci-dept/xc25.fsu/hq/tools/hf-cache
source .venv/bin/activate
export AW_MAX_PROMPT_TOKENS=4096 AW_TRUNCATE_SIDE=tail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SEED=$SLURM_ARRAY_TASK_ID
tag="bfclb10_full_s${SEED}"
AW_POOL_PATH=$PWD/data/bfcl_sft/pool_bfcl_v9.jsonl AW_V3=1 \
  python src/appworld_train.py --selection full --budget 999999 \
  --seed "$SEED" --student Qwen/Qwen3.5-4B --tag "$tag"
bash tools/bfcl_std_campaign.sh 0 $((9000 + SEED)) "$tag"
if [ ! -f "results/bfcl_std/$tag/data_overall.csv" ]; then
  echo "[bfclb10] $tag NO SCORE PRODUCED -- evaluation did not run"
  exit 1
fi
echo "[bfclb10] $tag DONE"
