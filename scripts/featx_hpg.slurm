#!/bin/bash
#SBATCH --job-name=tc-featx
#SBATCH --account=yd24f.fsu
#SBATCH --qos=yd24f.fsu
#SBATCH --partition=hpg-b200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=240gb
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out

# Usage: sbatch --export=ALL,TC_MODEL=Qwen/Qwen3.5-4B,TC_TAG=_qwen35-4b scripts/featx_hpg.slurm
set -e
cd "$SLURM_SUBMIT_DIR"
export HF_HOME=/blue/yd24f.fsu/xc25.fsu/hq/tools/hf-cache
mkdir -p "$HF_HOME" logs results/pilot
source .venv/bin/activate
echo "=== $(hostname) $(date) model=$TC_MODEL tag=$TC_TAG ==="
srun python src/grad_features.py
