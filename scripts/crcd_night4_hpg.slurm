#!/bin/bash
#SBATCH --job-name=crcdN4
#SBATCH --account=fsu-compsci-dept
#SBATCH --qos=fsu-compsci-dept
#SBATCH --partition=hpg-b200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=100gb
#SBATCH --time=06:00:00
#SBATCH --array=0-4
#SBATCH --output=logs/%x_%A_%a.out
# CRCD night 4 (user prompt P0-4 / P1-8): c4_adaptive = C3 pool with risk-triggered anchors (primal-dual per side);
# b3_reffree = reference-free pairwise on the r2 pool (vs base-DPO 47.34); r3uc3_adaptive = r3 union + C3 anchors, adaptive.
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
export AW_MAX_PROMPT_TOKENS=4096 AW_TRUNCATE_SIDE=tail AW_DISTILL=ddpo AW_SFT_SKIP=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ARMS=(c4_adaptive b3_reffree r3uc3_adaptive b3_reffree_ep3 b3_span)
case ${ARMS[$SLURM_ARRAY_TASK_ID]} in
  c4_adaptive)     export AW_POOL_PATH=$PWD/data/bfcl_sft/pool_events_pref_c3.jsonl AW_ANCHOR_ADAPTIVE=1 AW_ANCHOR_TRACE=$PWD/results/analysis/anchor_trace_c4.json ;;
  b3_reffree)      export AW_POOL_PATH=$PWD/data/bfcl_sft/pool_events_pref_v2.jsonl AW_DDPO_REF_FREE=1 ;;
  b3_reffree_ep3)  export AW_POOL_PATH=$PWD/data/bfcl_sft/pool_events_pref_v2.jsonl AW_DDPO_REF_FREE=1 AW_DDPO_EPOCHS=3 ;;
  b3_span)         export AW_POOL_PATH=$PWD/data/bfcl_sft/pool_events_pref_v2.jsonl AW_DDPO_SPAN_ONLY=1 ;;
  r3uc3_adaptive)  export AW_POOL_PATH=$PWD/data/bfcl_sft/pool_events_pref_v3c3.jsonl AW_ANCHOR_ADAPTIVE=1 AW_ANCHOR_TRACE=$PWD/results/analysis/anchor_trace_r3uc3.json ;;
esac
ARM=${ARMS[$SLURM_ARRAY_TASK_ID]}
tag="crcd_${ARM}_s0"
:
python src/appworld_train.py --selection full --budget 999999 --seed 0 --student Qwen/Qwen3.5-4B --tag "$tag"
bash tools/bfcl_std_campaign.sh 0 $((9130 + SLURM_ARRAY_TASK_ID)) "$tag"
if [ ! -f "results/bfcl_std/$tag/data_overall.csv" ]; then
  echo "[crcdN4] $tag NO SCORE PRODUCED"; exit 1
fi
echo "[crcdN4] $tag DONE"
