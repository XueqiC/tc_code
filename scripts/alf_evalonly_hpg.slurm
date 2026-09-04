#!/bin/bash
#SBATCH --job-name=alfEV
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
# ALFWorld official eval only (bfas adapter, valid_unseen) for tags whose training + hub-merge finished
# but whose vllm eval died on the full $HOME (vllm usage-stats writer). Usage:
#   sbatch --array=0-2 --export=ALL,TAGS="alfb1_ours_s0 alfb1_curr_s0 alfb1_pres_s0" scripts/alf_evalonly_hpg.slurm
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export HF_HOME=/blue/fsu-compsci-dept/xc25.fsu/hq/tools/hf-cache
export HOME=/blue/fsu-compsci-dept/xc25.fsu/hq/tools/fakehome; mkdir -p "$HOME"
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export VLLM_CACHE_ROOT=/blue/fsu-compsci-dept/xc25.fsu/hq/tools/vllm-cache TORCHINDUCTOR_CACHE_DIR=/blue/fsu-compsci-dept/xc25.fsu/hq/tools/inductor-cache
export TRITON_CACHE_DIR=/blue/fsu-compsci-dept/xc25.fsu/hq/tools/triton-cache XDG_CACHE_HOME=/blue/fsu-compsci-dept/xc25.fsu/hq/tools/xdg-cache
mkdir -p "$VLLM_CACHE_ROOT" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME"
source .venv/bin/activate
export PATH=$PWD/envs/vllm/.venv/bin:$PATH
export AW_MAX_PROMPT_TOKENS=2048 AW_TRUNCATE_SIDE=tail PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
read -ra ARR <<< "${TAGS:?set TAGS}"
tag=${ARR[$SLURM_ARRAY_TASK_ID]}
ARM=${tag#alfb1_}; ARM=${ARM%_s0}
[ -f "results/appworld_students/$tag/hub_merged/config.json" ] || { echo "[alfEV] $tag NO MERGED MODEL"; exit 1; }
rm -rf "results/bfas/alfworld/${ARM}4b_s0"
PYTHONPATH=src GPU_UTIL=0.6 BFAS_PORT_BASE=$((8935 + SLURM_ARRAY_TASK_ID)) VLLM_USE_FLASHINFER_SAMPLER=0 \
  .venv/bin/python tools/bfas_eval_ckpt.py --benchmark alfworld --checkpoint "results/appworld_students/$tag/hub_merged" \
  --out "results/bfas/alfworld/${ARM}4b_s0" --gpu 0 --port $((8935 + SLURM_ARRAY_TASK_ID)) --seed 0
echo "[alfEV] $tag DONE"
