#!/bin/bash
#SBATCH --job-name=alfabl
#SBATCH --account=fsu-compsci-dept
#SBATCH --qos=fsu-compsci-dept
#SBATCH --partition=hpg-b200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=100gb
#SBATCH --time=06:00:00
#SBATCH --array=0-6
#SBATCH --output=logs/%x_%j_%a.out
# ALFWorld consequentiality ablation (user prompt §4): same candidate pool, A all divergences / B first divergence /
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
ARMS=(A_all B_first C_conseq D_weighted CE_conseq D_conf E_negteacher)
ARM=${ARMS[$SLURM_ARRAY_TASK_ID]}
tag="alfabl_${ARM}_s0"
case $ARM in
  # step-matched: A (385 rows) = 48 steps at 1 epoch; the smaller pools get 4 epochs (~44-54 steps)
  A_all) export AW_DISTILL=ddpo AW_SFT_SKIP=1 AW_DDPO_EPOCHS=1; POOL=pool_A_all.jsonl ;;
  B_first|C_conseq|E_negteacher) export AW_DISTILL=ddpo AW_SFT_SKIP=1 AW_DDPO_EPOCHS=${ALF_EPOCHS:-4}; POOL=pool_${ARM}.jsonl ;;
  D_weighted) export AW_DISTILL=ddpo AW_SFT_SKIP=1 AW_DDPO_EPOCHS=${ALF_EPOCHS:-4} AW_DDPO_WEIGHT=tanh AW_DDPO_TAU=0.3; POOL=pool_D_weighted.jsonl ;;
  CE_conseq) export AW_EPOCHS=${ALF_EPOCHS:-4}; POOL=pool_C_conseq.jsonl ;;
  D_conf) export AW_DISTILL=ddpo AW_SFT_SKIP=1 AW_DDPO_EPOCHS=1 AW_DDPO_WEIGHT=conf; POOL=pool_A_all.jsonl ;;
esac
AW_POOL_PATH=$PWD/data/alf_sft/$POOL \
  .venv/bin/python src/appworld_train.py --selection full --budget 999999 --seed 0 --student Qwen/Qwen3.5-4B --tag "$tag"
rm -rf "results/appworld_students/$tag/hub_merged"
.venv/bin/python tools/bfcl_hub_merge_export.py --adapter "results/appworld_students/$tag/adapter" \
  --out "results/appworld_students/$tag/hub_merged" --model Qwen/Qwen3.5-4B --verify
PYTHONPATH=src GPU_UTIL=0.6 BFAS_PORT_BASE=$((8950 + SLURM_ARRAY_TASK_ID)) VLLM_USE_FLASHINFER_SAMPLER=0 \
  .venv/bin/python tools/bfas_eval_ckpt.py --benchmark alfworld --checkpoint "results/appworld_students/$tag/hub_merged" \
  --out "results/bfas/alfworld/abl_${ARM}_s0" --gpu 0 --port $((8950 + SLURM_ARRAY_TASK_ID)) --seed 0
echo "[alfabl] $tag DONE"
