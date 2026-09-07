#!/bin/bash
#SBATCH --job-name=behavior-margin
#SBATCH --account=fsu-compsci-dept
#SBATCH --qos=fsu-compsci-dept
#SBATCH --partition=hpg-b200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=240G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
cd /blue/fsu-compsci-dept/xc25.fsu/hq/tc-alignment
source .venv/bin/activate
export HF_HOME=/blue/fsu-compsci-dept/xc25.fsu/hq/tools/hf-cache
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=src
# Archive-only pair selection; sampling requires explicit --sample-base K.
python tools/behavior_atom_experiment.py margin --config configs/behavior_atom/margin_v1.json --step pairs
# Step 2: one resident bf16 student, exact init restore before every saved delta.
python tools/behavior_atom_experiment.py margin --config configs/behavior_atom/margin_v1.json --step score
# CPU diagnostics and docs-ready table, with the pre-fixed three verdict rules.
python tools/behavior_atom_experiment.py margin --config configs/behavior_atom/margin_v1.json --step analyze
