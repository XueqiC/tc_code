#!/bin/bash
#SBATCH --job-name=crcdEV
#SBATCH --account=fsu-compsci-dept
#SBATCH --qos=fsu-compsci-dept
#SBATCH --partition=hpg-b200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=100gb
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%A_%a.out
# Official BFCL eval only, for tags whose training finished but whose generation failed
# (hpg $HOME quota: vllm compile cache could not be created). Pass tags via --array and TAGS.
#   sbatch --array=0-1 --export=ALL,TAGS="crcd_r2_b3pref_ep3_s0 crcd_r2_b3pref_lr1x_s0" scripts/crcd_evalonly_hpg.slurm
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export HF_HOME=/blue/fsu-compsci-dept/xc25.fsu/hq/tools/hf-cache
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export HOME=/blue/fsu-compsci-dept/xc25.fsu/hq/tools/fakehome; mkdir -p "$HOME"
source .venv/bin/activate
read -ra ARR <<< "${TAGS:?set TAGS}"
tag=${ARR[$SLURM_ARRAY_TASK_ID]}
[ -f "results/appworld_students/$tag/adapter/model.safetensors" ] || { echo "[crcdEV] $tag NO ADAPTER"; exit 1; }
rm -rf "results/bfcl_std/$tag"
bash tools/bfcl_std_campaign.sh 0 $((9080 + SLURM_ARRAY_TASK_ID)) "$tag"
if [ ! -f "results/bfcl_std/$tag/data_overall.csv" ]; then
  echo "[crcdEV] $tag NO SCORE PRODUCED"; exit 1
fi
echo "[crcdEV] $tag DONE"
