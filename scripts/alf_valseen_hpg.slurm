#!/bin/bash
#SBATCH --job-name=alfVS
#SBATCH --account=fsu-compsci-dept
#SBATCH --qos=fsu-compsci-dept
#SBATCH --partition=hpg-b200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=100gb
#SBATCH --time=03:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/%x_%A_%a.out
# ALFWorld valid_seen (140 games) reference scores for the development split (unified-CRCD brief §8.1):
# base / CE-consequential (seed 0) / best pairwise (C_56step seed 0). No tuning; reference only.
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
export BFAS_ALFWORLD_EVAL_SPLIT=valid_seen
ARMS=(base ce_c pref_c56)
CKPTS=(Qwen/Qwen3.5-4B results/appworld_students/alfabl_CE_conseq_s0/hub_merged results/appworld_students/alfctl_C_56step_s0/hub_merged)
ARM=${ARMS[$SLURM_ARRAY_TASK_ID]}; CK=${CKPTS[$SLURM_ARRAY_TASK_ID]}
case $CK in results/*) [ -f "$CK/config.json" ] || { echo "[alfVS] $ARM NO MERGED MODEL"; exit 1; } ;; esac
rm -rf "results/bfas/alfworld/vs_${ARM}_s0"
PYTHONPATH=src GPU_UTIL=0.6 BFAS_PORT_BASE=$((8955 + SLURM_ARRAY_TASK_ID)) VLLM_USE_FLASHINFER_SAMPLER=0 \
  .venv/bin/python tools/bfas_eval_ckpt.py --benchmark alfworld --checkpoint "$CK" \
  --out "results/bfas/alfworld/vs_${ARM}_s0" --gpu 0 --port $((8955 + SLURM_ARRAY_TASK_ID)) --seed 0
echo "[alfVS] $ARM DONE"
