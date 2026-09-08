#!/bin/bash
#SBATCH --job-name=alf3
#SBATCH --account=fsu-compsci-dept
#SBATCH --qos=fsu-compsci-dept
#SBATCH --partition=hpg-b200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=200gb
#SBATCH --time=04:00:00
#SBATCH --array=0-4
#SBATCH --output=logs/%x_%A_%a.out
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
# ALF-3: capability-constrained mirror distillation on ALFWorld (T8 prep, docs/2026-09-04-alf3-mirror-prep.md).
# Arms: C_rho1 / C_rhoJ0 / C_k1 (85-row consequential pool, 44 steps), A_rho1 (385 rows, 49 steps), C_strong (gt_teacher utility, kappa 20).
# Eval: valid_seen (dev split). Single seed.
ARMS=(C_rho1 C_rhoJ0 C_k1 A_rho1 C_strong); ARM=${ARMS[$SLURM_ARRAY_TASK_ID]}; tag=crcd_alf3_${ARM}_s0
if [ ! -f results/appworld_students/$tag/hub_merged/config.json ]; then
  PYTHONPATH=src .venv/bin/python tools/crcd_mirror_train.py --config configs/mirror_alf3_${ARM}.json
  .venv/bin/python tools/bfcl_hub_merge_export.py --adapter results/appworld_students/$tag/adapter --out results/appworld_students/$tag/hub_merged --model Qwen/Qwen3.5-4B --verify
fi
rm -rf "results/bfas/alfworld/$tag"
PYTHONPATH=src GPU_UTIL=0.6 BFAS_PORT_BASE=$((8965 + SLURM_ARRAY_TASK_ID)) VLLM_USE_FLASHINFER_SAMPLER=0 \
  .venv/bin/python tools/bfas_eval_ckpt.py --benchmark alfworld --checkpoint results/appworld_students/$tag/hub_merged \
  --out "results/bfas/alfworld/$tag" --gpu 0 --port $((8965 + SLURM_ARRAY_TASK_ID)) --seed 0
echo "[alf3] $tag DONE"
