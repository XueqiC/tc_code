#!/bin/bash
#SBATCH --job-name=bfclpres
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
# Preservation-weight sweep with FULL evaluations, run in parallel.
# weight = (1 - p-hat) + AW_PRESERVE * p-hat. AW_PRESERVE=0 is the deficit-only
# behaviour already covered by the bfclb7 array, so this sweeps the rest.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export HF_HOME=/blue/fsu-compsci-dept/xc25.fsu/hq/tools/hf-cache
source .venv/bin/activate
export AW_MAX_PROMPT_TOKENS=4096 AW_TRUNCATE_SIDE=tail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
VALUES=(0.25 0.50 1.00)
P=${VALUES[$SLURM_ARRAY_TASK_ID]}
tag="bfclp_${P/./}_s0"
AW_POOL_PATH=$PWD/data/bfcl_sft/pool_bfcl_v6.jsonl AW_V3=1 AW_PRESERVE=$P \
  python src/appworld_train.py --selection full --budget 999999 \
  --seed 0 --student Qwen/Qwen3.5-4B --tag "$tag"
bash tools/bfcl_std_campaign.sh 0 $((8995 + SLURM_ARRAY_TASK_ID)) "$tag"
# a campaign that skips evaluation still exits 0, so the job must check that a
# score actually exists before calling itself done
if [ ! -f "results/bfcl_std/$tag/data_overall.csv" ]; then
  echo "[$SLURM_JOB_NAME] $tag NO SCORE PRODUCED -- evaluation did not run"
  exit 1
fi
echo "[bfclpres] $tag (preserve=$P) DONE"
