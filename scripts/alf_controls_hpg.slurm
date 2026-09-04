#!/bin/bash
#SBATCH --job-name=alfctl
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
ARMS=(D_perm D_perm_cat B_44step C_56step A_85)
ARM=${ARMS[$SLURM_ARRAY_TASK_ID]}
tag="alfctl_${ARM}_s0"
case $ARM in
  # controls requested 2026-09-03: same 85 events / steps / seed as D; only the weight alignment changes
  D_perm)     export AW_DISTILL=ddpo AW_SFT_SKIP=1 AW_DDPO_EPOCHS=4 AW_DDPO_WEIGHT=tanh AW_DDPO_TAU=0.3 AW_DDPO_WEIGHT_PERMUTE=global;   POOL=pool_D_weighted.jsonl ;;
  D_perm_cat) export AW_DISTILL=ddpo AW_SFT_SKIP=1 AW_DDPO_EPOCHS=4 AW_DDPO_WEIGHT=tanh AW_DDPO_TAU=0.3 AW_DDPO_WEIGHT_PERMUTE=category; POOL=pool_D_weighted.jsonl ;;
  # step-matched controls: B at C's steps, C at B's steps, A subsampled to 85 events at 44 steps
  B_44step)   export AW_DISTILL=ddpo AW_SFT_SKIP=1 AW_DDPO_EPOCHS=3; POOL=pool_B_first.jsonl ;;
  C_56step)   export AW_DISTILL=ddpo AW_SFT_SKIP=1 AW_DDPO_EPOCHS=5; POOL=pool_C_conseq.jsonl ;;
  A_85)       export AW_DISTILL=ddpo AW_SFT_SKIP=1 AW_DDPO_EPOCHS=4; POOL=pool_A_85.jsonl ;;
esac
AW_POOL_PATH=$PWD/data/alf_sft/$POOL \
  .venv/bin/python src/appworld_train.py --selection full --budget 999999 --seed 0 --student Qwen/Qwen3.5-4B --tag "$tag"
rm -rf "results/appworld_students/$tag/hub_merged"
.venv/bin/python tools/bfcl_hub_merge_export.py --adapter "results/appworld_students/$tag/adapter" \
  --out "results/appworld_students/$tag/hub_merged" --model Qwen/Qwen3.5-4B --verify
PYTHONPATH=src GPU_UTIL=0.6 BFAS_PORT_BASE=$((8960 + SLURM_ARRAY_TASK_ID)) VLLM_USE_FLASHINFER_SAMPLER=0 \
  .venv/bin/python tools/bfas_eval_ckpt.py --benchmark alfworld --checkpoint "results/appworld_students/$tag/hub_merged" \
  --out "results/bfas/alfworld/ctl_${ARM}_s0" --gpu 0 --port $((8960 + SLURM_ARRAY_TASK_ID)) --seed 0
echo "[alfctl] $tag DONE"
