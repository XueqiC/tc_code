#!/bin/bash
#SBATCH --job-name=alfr2b
#SBATCH --account=fsu-compsci-dept
#SBATCH --qos=fsu-compsci-dept
#SBATCH --partition=hpg-b200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=100gb
#SBATCH --time=06:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/%x_%j_%a.out
# ALFWorld round-2 from the competent CE model (2026-09-04): same candidate pool, A all divergences / B first divergence /
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
ARMS=(r2b_pref_C_ep3 r2b_pref_union r2b_CE_union)
ARM=${ARMS[$SLURM_ARRAY_TASK_ID]}
tag="alfr2b_${ARM}_s0"
BASE=$PWD/results/appworld_students/alfabl_CE_conseq_s0/hub_merged   # the competent student (74.6%)
case $ARM in
  # round-2 dose-matched BY EVENTS (the 14-epoch arms over-repeated 25 events): from the CE model,
  # pairwise on the 25 residual events x 3 ep (~10 steps); pairwise / CE on round-1 ∪ round-2 consequential (110 rows) x 3 / 1 ep
  r2b_pref_C_ep3) export AW_DISTILL=ddpo AW_SFT_SKIP=1 AW_DDPO_EPOCHS=3; POOL=r2/pool_C_conseq.jsonl ;;
  r2b_pref_union) export AW_DISTILL=ddpo AW_SFT_SKIP=1 AW_DDPO_EPOCHS=3; POOL=r2/pool_union_C.jsonl ;;
  r2b_CE_union)   export AW_EPOCHS=1; POOL=r2/pool_union_C.jsonl ;;
esac
AW_POOL_PATH=$PWD/data/alf_sft/$POOL \
  .venv/bin/python src/appworld_train.py --selection full --budget 999999 --seed 0 --student "$BASE" --tag "$tag"
rm -rf "results/appworld_students/$tag/hub_merged"
.venv/bin/python tools/bfcl_hub_merge_export.py --adapter "results/appworld_students/$tag/adapter" \
  --out "results/appworld_students/$tag/hub_merged" --model "$BASE" --verify
PYTHONPATH=src GPU_UTIL=0.6 BFAS_PORT_BASE=$((8993 + SLURM_ARRAY_TASK_ID)) VLLM_USE_FLASHINFER_SAMPLER=0 \
  .venv/bin/python tools/bfas_eval_ckpt.py --benchmark alfworld --checkpoint "results/appworld_students/$tag/hub_merged" \
  --out "results/bfas/alfworld/r2b_${ARM}_s0" --gpu 0 --port $((8993 + SLURM_ARRAY_TASK_ID)) --seed 0
echo "[alfr2b] $tag DONE"
