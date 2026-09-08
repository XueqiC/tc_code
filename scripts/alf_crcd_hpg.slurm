#!/bin/bash
#SBATCH --job-name=alfcrcd
#SBATCH --account=fsu-compsci-dept
#SBATCH --qos=fsu-compsci-dept
#SBATCH --partition=hpg-b200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=100gb
#SBATCH --time=06:00:00
#SBATCH --array=0
#SBATCH --output=logs/%x_%j_%a.out
# ALFWorld CRCD: base-centred preference training on demo-replay intervention events (data/alf_sft/pool_events_pref_v1.jsonl),
# then hub-merge and the official valid_unseen eval (compare with base 8.96%, pres 11.94%).
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
ARMS=(crcd)
ARM=${ARMS[$SLURM_ARRAY_TASK_ID]}
tag="alfcrcd_${ARM}_s0"
case $ARM in
  crcd) export AW_DISTILL=ddpo AW_SFT_SKIP=1 AW_DDPO_EPOCHS=${ALF_EPOCHS:-1} ;;
esac
AW_POOL_PATH=$PWD/data/alf_sft/pool_events_pref_v1.jsonl \
  .venv/bin/python src/appworld_train.py --selection full --budget 999999 --seed 0 --student Qwen/Qwen3.5-4B --tag "$tag"
rm -rf "results/appworld_students/$tag/hub_merged"
.venv/bin/python tools/bfcl_hub_merge_export.py --adapter "results/appworld_students/$tag/adapter" \
  --out "results/appworld_students/$tag/hub_merged" --model Qwen/Qwen3.5-4B --verify
PYTHONPATH=src GPU_UTIL=0.6 BFAS_PORT_BASE=$((8945 + SLURM_ARRAY_TASK_ID)) VLLM_USE_FLASHINFER_SAMPLER=0 \
  .venv/bin/python tools/bfas_eval_ckpt.py --benchmark alfworld --checkpoint "results/appworld_students/$tag/hub_merged" \
  --out "results/bfas/alfworld/crcd4b_s0" --gpu 0 --port $((8945 + SLURM_ARRAY_TASK_ID)) --seed 0
echo "[alfcrcd] $tag DONE"
