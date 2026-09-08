#!/bin/bash
#SBATCH --job-name=crcdN2
#SBATCH --account=fsu-compsci-dept
#SBATCH --qos=fsu-compsci-dept
#SBATCH --partition=hpg-b200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=200gb
#SBATCH --time=06:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/%x_%A_%a.out
# CRCD night batch 2: preference DOSE on the round-2 pool (does more DPO buy gain without damage?).
#   0  r1_b2ce      round-1 pool, corrected-span CE (official score for the rai proxy 8.17)
#   1  r1_b4cepref  round-1 pool, CE -> base-centered DPO (official for proxy 7.32)
#   2  r2_b3pref    round-2 pool (+abstain events, 200 rows), preference only
#   3  r2_b2ce      round-2 pool, CE only (does adding abstain targets stop the always-call collapse?)
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export HF_HOME=/blue/fsu-compsci-dept/xc25.fsu/hq/tools/hf-cache
source .venv/bin/activate
export AW_MAX_PROMPT_TOKENS=4096 AW_TRUNCATE_SIDE=tail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ARMS=(r2_b3pref_ep3 r2_b3pref_ep5)
ARM=${ARMS[$SLURM_ARRAY_TASK_ID]}
tag="crcd_${ARM}_s0"
case $ARM in
  r2_b3pref_ep3) export AW_POOL_PATH=$PWD/data/bfcl_sft/pool_events_pref_v2.jsonl AW_DISTILL=ddpo AW_SFT_SKIP=1 AW_DDPO_EPOCHS=3 ;;
  r2_b3pref_ep5) export AW_POOL_PATH=$PWD/data/bfcl_sft/pool_events_pref_v2.jsonl AW_DISTILL=ddpo AW_SFT_SKIP=1 AW_DDPO_EPOCHS=5 ;;
esac
python src/appworld_train.py --selection full --budget 999999 \
  --seed 0 --student Qwen/Qwen3.5-4B --tag "$tag"
bash tools/bfcl_std_campaign.sh 0 $((9095 + SLURM_ARRAY_TASK_ID)) "$tag"
if [ ! -f "results/bfcl_std/$tag/data_overall.csv" ]; then
  echo "[crcdN2] $tag NO SCORE PRODUCED"; exit 1
fi
echo "[crcdN2] $tag DONE"
