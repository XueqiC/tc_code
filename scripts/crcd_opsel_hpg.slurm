#!/bin/bash
#SBATCH --job-name=crcdOp
#SBATCH --account=fsu-compsci-dept
#SBATCH --qos=fsu-compsci-dept
#SBATCH --partition=hpg-b200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=100gb
#SBATCH --time=06:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/%x_%A_%a.out
# Operator-selection test (2026-09-03 evening): CE vs pairwise on base-INCOMPETENT task families (JS, live_parallel_multiple, java; base rate <0.1)
# and on base-COMPETENT families (rate >= 0.25), equal ~44 steps. Prediction: CE wins where the base cannot do the family, collapses where it can.
# mixed into the round-2 preference pool. Single seed, official BFCL.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export HF_HOME=/blue/fsu-compsci-dept/xc25.fsu/hq/tools/hf-cache
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export HOME=/blue/fsu-compsci-dept/xc25.fsu/hq/tools/fakehome; mkdir -p "$HOME"
source .venv/bin/activate
export VLLM_CACHE_ROOT=/blue/fsu-compsci-dept/xc25.fsu/hq/tools/vllm-cache TORCHINDUCTOR_CACHE_DIR=/blue/fsu-compsci-dept/xc25.fsu/hq/tools/inductor-cache
export TRITON_CACHE_DIR=/blue/fsu-compsci-dept/xc25.fsu/hq/tools/triton-cache XDG_CACHE_HOME=/blue/fsu-compsci-dept/xc25.fsu/hq/tools/xdg-cache
mkdir -p "$VLLM_CACHE_ROOT" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME"
export AW_MAX_PROMPT_TOKENS=4096 AW_TRUNCATE_SIDE=tail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ARMS=(ce_incomp pref_incomp ce_comp pref_comp)
ARM=${ARMS[$SLURM_ARRAY_TASK_ID]}
case $ARM in
  ce_incomp)   export AW_POOL_PATH=$PWD/data/bfcl_sft/pool_incomp_v2.jsonl AW_EPOCHS=7 ;;
  pref_incomp) export AW_POOL_PATH=$PWD/data/bfcl_sft/pool_incomp_v2.jsonl AW_DISTILL=ddpo AW_SFT_SKIP=1 AW_DDPO_EPOCHS=7 ;;
  ce_comp)     export AW_POOL_PATH=$PWD/data/bfcl_sft/pool_comp_v2.jsonl AW_EPOCHS=3 ;;   # 135 rows -> ~51 steps
  pref_comp)   export AW_POOL_PATH=$PWD/data/bfcl_sft/pool_comp_v2.jsonl AW_DISTILL=ddpo AW_SFT_SKIP=1 AW_DDPO_EPOCHS=3 ;;
esac
ARM=${ARMS[$SLURM_ARRAY_TASK_ID]}
tag="crcd_${ARM}_s0"
:
python src/appworld_train.py --selection full --budget 999999 --seed 0 --student Qwen/Qwen3.5-4B --tag "$tag"
bash tools/bfcl_std_campaign.sh 0 $((9150 + SLURM_ARRAY_TASK_ID)) "$tag"
if [ ! -f "results/bfcl_std/$tag/data_overall.csv" ]; then
  echo "[crcdOp] $tag NO SCORE PRODUCED"; exit 1
fi
echo "[crcdOp] $tag DONE"
