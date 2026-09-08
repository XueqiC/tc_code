#!/bin/bash
#SBATCH --job-name=alfdose
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
#SBATCH --output=logs/%x_%j_%a.out
# ALFWorld ablation CONTROLS (user 2026-09-03): permuted weights (global / within-category), step-matched B/C/A: same candidate pool, A all divergences / B first divergence /
# C consequential-only / D utility-weighted (tanh(dU/0.3)) / CE on consequential; 1 epoch each; official valid_unseen eval.
# Train with the shared trainer, hub-merge, evaluate with the bfas adapter (tools/bfas_eval_ckpt.py).
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export HF_HOME=/blue/fsu-compsci-dept/xc25.fsu/hq/tools/hf-cache
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export HOME=/blue/fsu-compsci-dept/xc25.fsu/hq/tools/fakehome; mkdir -p "$HOME"
source .venv/bin/activate
export VLLM_CACHE_ROOT=/blue/fsu-compsci-dept/xc25.fsu/hq/tools/vllm-cache TORCHINDUCTOR_CACHE_DIR=/blue/fsu-compsci-dept/xc25.fsu/hq/tools/inductor-cache
export TRITON_CACHE_DIR=/blue/fsu-compsci-dept/xc25.fsu/hq/tools/triton-cache XDG_CACHE_HOME=/blue/fsu-compsci-dept/xc25.fsu/hq/tools/xdg-cache
mkdir -p "$VLLM_CACHE_ROOT" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME"
export PATH=$PWD/envs/vllm/.venv/bin:$PATH
export AW_MAX_PROMPT_TOKENS=2048 AW_TRUNCATE_SIDE=tail PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ARMS=(C_88step C_110step)
ARM=${ARMS[$SLURM_ARRAY_TASK_ID]}
tag="alfdose_${ARM}_s0"
case $ARM in
  # dose curve of the consequential pool under pairwise: 85 rows x 8 / 10 ep (C_44 17.2, C_56 28.4 so far)
  C_88step)  export AW_DISTILL=ddpo AW_SFT_SKIP=1 AW_DDPO_EPOCHS=8;  POOL=pool_C_conseq.jsonl ;;
  C_110step) export AW_DISTILL=ddpo AW_SFT_SKIP=1 AW_DDPO_EPOCHS=10; POOL=pool_C_conseq.jsonl ;;
esac
AW_POOL_PATH=$PWD/data/alf_sft/$POOL \
  .venv/bin/python src/appworld_train.py --selection full --budget 999999 --seed 0 --student Qwen/Qwen3.5-4B --tag "$tag"
rm -rf "results/appworld_students/$tag/hub_merged"
.venv/bin/python tools/bfcl_hub_merge_export.py --adapter "results/appworld_students/$tag/adapter" \
  --out "results/appworld_students/$tag/hub_merged" --model Qwen/Qwen3.5-4B --verify
PYTHONPATH=src GPU_UTIL=0.6 BFAS_PORT_BASE=$((8975 + SLURM_ARRAY_TASK_ID)) VLLM_USE_FLASHINFER_SAMPLER=0 \
  .venv/bin/python tools/bfas_eval_ckpt.py --benchmark alfworld --checkpoint "results/appworld_students/$tag/hub_merged" \
  --out "results/bfas/alfworld/dose_${ARM}_s0" --gpu 0 --port $((8975 + SLURM_ARRAY_TASK_ID)) --seed 0
echo "[alfdose] $tag DONE"
