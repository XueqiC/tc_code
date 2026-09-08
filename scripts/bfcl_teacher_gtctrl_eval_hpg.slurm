#!/bin/bash
#SBATCH --job-name=bfclTge
#SBATCH --account=fsu-compsci-dept
#SBATCH --qos=fsu-compsci-dept
#SBATCH --partition=hpg-b200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=200gb
#SBATCH --time=04:00:00
#SBATCH --array=0
#SBATCH --output=logs/%x_%A_%a.out
# BF-1 teacher-consistent rerun (2026-09-04): round-3 union pool with official-task y^T taken from verified deepseek demos,
# calibration ids removed (302 rows, pool_events_pref_v3t). Same recipe as crcd_r3_union_s0 (base student, pref-only 1 ep). Single seed.
#   0  r3_union  base student, preference-only on round-2 ∪ round-3 events (pool_events_pref_v3)
#   1  r3_cont   continue from the round-2 model (merged adapter), preference-only on round-3 events only
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
# eval-only resubmission of the oracle-GT control (41108408 trained + merged, cancelled by signal during scoring).
tag=crcd_r3_union_tgt_s0
[ -f results/appworld_students/$tag/hub_merged/config.json ] || { echo "[bfclTe] no merged model"; exit 1; }
bash tools/bfcl_std_campaign.sh 0 $((9135 + SLURM_ARRAY_TASK_ID)) "$tag"
if [ ! -f "results/bfcl_std/$tag/data_overall.csv" ]; then echo "[bfclTe] $tag NO SCORE PRODUCED"; exit 1; fi
echo "[bfclTe] $tag DONE"
