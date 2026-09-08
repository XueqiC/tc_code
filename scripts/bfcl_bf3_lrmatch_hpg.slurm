#!/bin/bash
#SBATCH --job-name=bf3l
#SBATCH --account=fsu-compsci-dept
#SBATCH --qos=fsu-compsci-dept
#SBATCH --partition=hpg-b200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=200gb
#SBATCH --time=05:00:00
#SBATCH --array=0
#SBATCH --output=logs/%x_%A_%a.out
# BF-3c lr-matched mirror arm: as gentle (kappa 1, eta_dual .5, 32 probes gamma_probe 1, gamma_perp 1e-2) but lr 5e-6 = the pairwise anchor lr (the K=32 arms used 1e-4, 20x larger; all three collapsed monotonically in pressure).
# Fisher psi (data/atoms/bfcl_r3t_K32.npz), 38 steps, dual every 4. Arms: rho1 (rho_k=1), rhoJ0 (rho_k=J0+0.3), k1 (global mirror K=1 baseline).
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export HF_HOME=/blue/fsu-compsci-dept/xc25.fsu/hq/tools/hf-cache
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export HOME=/blue/fsu-compsci-dept/xc25.fsu/hq/tools/fakehome; mkdir -p "$HOME"
source .venv/bin/activate
export VLLM_CACHE_ROOT=/blue/fsu-compsci-dept/xc25.fsu/hq/tools/vllm-cache TORCHINDUCTOR_CACHE_DIR=/blue/fsu-compsci-dept/xc25.fsu/hq/tools/inductor-cache
export TRITON_CACHE_DIR=/blue/fsu-compsci-dept/xc25.fsu/hq/tools/triton-cache XDG_CACHE_HOME=/blue/fsu-compsci-dept/xc25.fsu/hq/tools/xdg-cache
mkdir -p "$VLLM_CACHE_ROOT" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME"
export AW_MAX_PROMPT_TOKENS=4096 AW_TRUNCATE_SIDE=tail PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ARMS=(lrmatch); ARM=${ARMS[$SLURM_ARRAY_TASK_ID]}; tag=crcd_bf3_${ARM}_s0
if [ ! -f results/appworld_students/$tag/hub_merged/config.json ]; then
  PYTHONPATH=src python tools/crcd_mirror_train.py --config configs/mirror_bf3_${ARM}.json
  python tools/bfcl_hub_merge_export.py --adapter results/appworld_students/$tag/adapter --out results/appworld_students/$tag/hub_merged --model Qwen/Qwen3.5-4B
fi
bash tools/bfcl_std_campaign.sh 0 $((9185 + SLURM_ARRAY_TASK_ID)) "$tag"
[ -f "results/bfcl_std/$tag/data_overall.csv" ] || { echo "[bf3] $tag NO SCORE PRODUCED"; exit 1; }
echo "[bf3] $tag DONE"
