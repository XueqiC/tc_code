#!/bin/bash
#SBATCH --job-name=acq
#SBATCH --account=fsu-compsci-dept
#SBATCH --qos=fsu-compsci-dept
#SBATCH --partition=hpg-b200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=200gb
#SBATCH --time=05:00:00
#SBATCH --array=0-6
#SBATCH --output=logs/%x_%A_%a.out
# BF-6 acquisition replay arms (T11, docs/2026-09-04-acquisition-replay.md): subsets of the teacher-consistent round-3 pool
# selected by different acquisition policies at equal teacher-token budget (b25) or equal rows (u50); base student,
# pref-only 1 epoch (BF-1 recipe), official BFCL. Single seed.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export HF_HOME=/blue/fsu-compsci-dept/xc25.fsu/hq/tools/hf-cache
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export HOME=/blue/fsu-compsci-dept/xc25.fsu/hq/tools/fakehome; mkdir -p "$HOME"
source .venv/bin/activate
export VLLM_CACHE_ROOT=/blue/fsu-compsci-dept/xc25.fsu/hq/tools/vllm-cache TORCHINDUCTOR_CACHE_DIR=/blue/fsu-compsci-dept/xc25.fsu/hq/tools/inductor-cache
export TRITON_CACHE_DIR=/blue/fsu-compsci-dept/xc25.fsu/hq/tools/triton-cache XDG_CACHE_HOME=/blue/fsu-compsci-dept/xc25.fsu/hq/tools/xdg-cache
mkdir -p "$VLLM_CACHE_ROOT" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME"
export AW_MAX_PROMPT_TOKENS=4096 AW_TRUNCATE_SIDE=tail AW_DISTILL=ddpo AW_SFT_SKIP=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ARMS=(random_b25 consequential_first_b25 residual_per_token_b25 cheapest_first_b25 random_u50 consequential_first_u50 residual_per_token_u50)
ARM=${ARMS[$SLURM_ARRAY_TASK_ID]}; tag="crcd_acq_${ARM}_s0"
export AW_POOL_PATH=$PWD/data/bfcl_sft/pool_acq_${ARM}.jsonl
[ -f "$AW_POOL_PATH" ] || { echo "[acq] missing pool $AW_POOL_PATH"; exit 1; }
if [ ! -f results/appworld_students/$tag/hub_merged/config.json ]; then
  python src/appworld_train.py --selection full --budget 999999 --seed 0 --student Qwen/Qwen3.5-4B --tag "$tag"
fi
bash tools/bfcl_std_campaign.sh 0 $((9160 + SLURM_ARRAY_TASK_ID)) "$tag"
[ -f "results/bfcl_std/$tag/data_overall.csv" ] || { echo "[acq] $tag NO SCORE PRODUCED"; exit 1; }
echo "[acq] $tag DONE"
