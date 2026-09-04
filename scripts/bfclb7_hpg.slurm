#!/bin/bash
#SBATCH --job-name=bfclb7
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
# Three seeds of the generated-stateful pool, each with a FULL official BFCL
# evaluation. rai cannot run these in parallel; the point of putting them here
# is seed variance on the reportable number while rai sweeps the preservation
# weight on the fast proxy.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export HF_HOME=/blue/fsu-compsci-dept/xc25.fsu/hq/tools/hf-cache
source .venv/bin/activate
export AW_MAX_PROMPT_TOKENS=4096 AW_TRUNCATE_SIDE=tail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SEED=$SLURM_ARRAY_TASK_ID
tag="bfclb7_gen_s${SEED}"
AW_POOL_PATH=$PWD/data/bfcl_sft/pool_bfcl_v6.jsonl AW_V3=1 \
  python src/appworld_train.py --selection full --budget 999999 \
  --seed "$SEED" --student Qwen/Qwen3.5-4B --tag "$tag"
bash tools/bfcl_std_campaign.sh 0 $((8990 + SEED)) "$tag"
# a campaign that skips evaluation still exits 0, so the job must check that a
# score actually exists before calling itself done
if [ ! -f "results/bfcl_std/$tag/data_overall.csv" ]; then
  echo "[$SLURM_JOB_NAME] $tag NO SCORE PRODUCED -- evaluation did not run"
  exit 1
fi
echo "[bfclb7] $tag DONE"
