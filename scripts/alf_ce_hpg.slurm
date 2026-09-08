#!/bin/bash
#SBATCH --job-name=alfce
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
# ALFWorld CE controls (2026-09-03 evening): same candidate pool, A all divergences / B first divergence /
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
ARMS=(CE_A CE_B CE_zero CE_C1ep CEpair_C)
ARM=${ARMS[$SLURM_ARRAY_TASK_ID]}
tag="alfce_${ARM}_s0"
case $ARM in
  # CE controls after CE_conseq = 74.6%: does consequentiality / pool matter for CE, dose, and CE->pairwise
  CE_A)     export AW_EPOCHS=1; POOL=pool_A_all.jsonl ;;                    # 385 rows, 49 steps
  CE_B)     export AW_EPOCHS=3; POOL=pool_B_first.jsonl ;;                  # 107 rows, 42 steps
  CE_zero)  export AW_EPOCHS=4; POOL=pool_zero85.jsonl ;;                   # 85 dU=0 rows, 44 steps
  CE_C1ep)  export AW_EPOCHS=1; POOL=pool_C_conseq.jsonl ;;                 # 11 steps
  CEpair_C) export AW_EPOCHS=4 AW_DISTILL=ddpo AW_DDPO_EPOCHS=1; POOL=pool_C_conseq.jsonl ;;   # CE then base-centred DPO
esac
AW_POOL_PATH=$PWD/data/alf_sft/$POOL \
  .venv/bin/python src/appworld_train.py --selection full --budget 999999 --seed 0 --student Qwen/Qwen3.5-4B --tag "$tag"
rm -rf "results/appworld_students/$tag/hub_merged"
.venv/bin/python tools/bfcl_hub_merge_export.py --adapter "results/appworld_students/$tag/adapter" \
  --out "results/appworld_students/$tag/hub_merged" --model Qwen/Qwen3.5-4B --verify
PYTHONPATH=src GPU_UTIL=0.6 BFAS_PORT_BASE=$((8970 + SLURM_ARRAY_TASK_ID)) VLLM_USE_FLASHINFER_SAMPLER=0 \
  .venv/bin/python tools/bfas_eval_ckpt.py --benchmark alfworld --checkpoint "results/appworld_students/$tag/hub_merged" \
  --out "results/bfas/alfworld/ce_${ARM}_s0" --gpu 0 --port $((8970 + SLURM_ARRAY_TASK_ID)) --seed 0
echo "[alfce] $tag DONE"
