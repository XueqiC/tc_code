#!/bin/bash
#SBATCH --job-name=crcdSeed
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
# CRCD BFCL seeds 1-2 for the frozen headline arms (2026-09-03 23:00): c4_adaptive = C3 pool with risk-triggered anchors (primal-dual per side);
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
ARMS=(r3_union_s1 r3_union_s2 r3uc3_adaptive_s1 r3uc3_adaptive_s2)
ARMNAME=${ARMS[$SLURM_ARRAY_TASK_ID]}
SEED=${ARMNAME##*_s}
case ${ARMNAME%_s*} in
  # seeds 1-2 of the frozen BFCL headline arms (seed 0: r3_union 47.82, r3uc3_adaptive 47.81 / Irrel 82.02)
  r3_union)        export AW_POOL_PATH=$PWD/data/bfcl_sft/pool_events_pref_v3.jsonl ;;
  r3uc3_adaptive)  export AW_POOL_PATH=$PWD/data/bfcl_sft/pool_events_pref_v3c3.jsonl AW_ANCHOR_ADAPTIVE=1 AW_ANCHOR_TRACE=$PWD/results/analysis/anchor_trace_${ARMNAME}.json ;;
esac
tag="crcd_${ARMNAME}"
:
python src/appworld_train.py --selection full --budget 999999 --seed $SEED --student Qwen/Qwen3.5-4B --tag "$tag"
bash tools/bfcl_std_campaign.sh 0 $((9170 + SLURM_ARRAY_TASK_ID)) "$tag"
if [ ! -f "results/bfcl_std/$tag/data_overall.csv" ]; then
  echo "[crcdSeed] $tag NO SCORE PRODUCED"; exit 1
fi
echo "[crcdSeed] $tag DONE"
