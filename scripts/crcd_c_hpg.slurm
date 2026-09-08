#!/bin/bash
#SBATCH --job-name=crcdC
#SBATCH --account=fsu-compsci-dept
#SBATCH --qos=fsu-compsci-dept
#SBATCH --partition=hpg-b200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=100gb
#SBATCH --time=06:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/%x_%A_%a.out
# CRCD C-ladder: risk-triggered retention anchors (base student's own correct behaviour, dU = 0)
# mixed into the round-2 preference pool. Single seed, official BFCL.
#   0  c1_anchor      1 DPO epoch   (compare with r2_b3pref 47.34, Irrel 80.57)
#   1  c1_anchor_ep3  3 DPO epochs  (compare with ep3 47.61, Irrel 73.45 -- does the anchor stop the retention slide?)
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
export AW_POOL_PATH=$PWD/data/bfcl_sft/pool_events_pref_c1.jsonl
ARMS=(c1_anchor c1_anchor_ep3)
ARM=${ARMS[$SLURM_ARRAY_TASK_ID]}
tag="crcd_${ARM}_s0"
[ "$ARM" = c1_anchor_ep3 ] && export AW_DDPO_EPOCHS=3
python src/appworld_train.py --selection full --budget 999999 --seed 0 --student Qwen/Qwen3.5-4B --tag "$tag"
bash tools/bfcl_std_campaign.sh 0 $((9100 + SLURM_ARRAY_TASK_ID)) "$tag"
if [ ! -f "results/bfcl_std/$tag/data_overall.csv" ]; then
  echo "[crcdC] $tag NO SCORE PRODUCED"; exit 1
fi
echo "[crcdC] $tag DONE"
