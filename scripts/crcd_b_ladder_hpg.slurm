#!/bin/bash
#SBATCH --job-name=crcdB
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
# CRCD P0-2 distillation-object ladder on BFCL (single seed, full official eval):
#   B2  corrected-span CE only            (pool_events_ce, plain SFT)
#   B3  local preference only             (pool_events_pref, AW_DISTILL=ddpo, AW_SFT_SKIP=1)
#   B4' corrected-span CE + preference    (pool_events_pref, AW_DISTILL=ddpo: SFT stage then base-centered DPO)
# B0 = v9 SFT (bfclb10, 36.70) and base (46.27 / CI rerun) are the references.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export HF_HOME=/blue/fsu-compsci-dept/xc25.fsu/hq/tools/hf-cache
source .venv/bin/activate
export AW_MAX_PROMPT_TOKENS=4096 AW_TRUNCATE_SIDE=tail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ARMS=(b2ce b3pref b4cepref)
ARM=${ARMS[$SLURM_ARRAY_TASK_ID]}
tag="crcd_${ARM}_s0"
case $ARM in
  b2ce)     export AW_POOL_PATH=$PWD/data/bfcl_sft/pool_events_ce.jsonl ;;
  b3pref)   export AW_POOL_PATH=$PWD/data/bfcl_sft/pool_events_pref.jsonl AW_DISTILL=ddpo AW_SFT_SKIP=1 ;;
  b4cepref) export AW_POOL_PATH=$PWD/data/bfcl_sft/pool_events_pref.jsonl AW_DISTILL=ddpo ;;
esac
python src/appworld_train.py --selection full --budget 999999 \
  --seed 0 --student Qwen/Qwen3.5-4B --tag "$tag"
bash tools/bfcl_std_campaign.sh 0 $((9080 + SLURM_ARRAY_TASK_ID)) "$tag"
if [ ! -f "results/bfcl_std/$tag/data_overall.csv" ]; then
  echo "[crcdB] $tag NO SCORE PRODUCED"; exit 1
fi
echo "[crcdB] $tag DONE"
