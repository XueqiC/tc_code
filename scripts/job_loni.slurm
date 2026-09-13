#!/bin/bash
# LONI QB-D template (A100 80GB PCIe nodes; gpu2 has 2 GPUs per node, gpu4 has 4).
# Allocation loni_depedlab03. See docs/loni_setup.md for the cluster survey.
#SBATCH --job-name=tc-alignment
#SBATCH --account=loni_depedlab03
#SBATCH --partition=gpu2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out

set -e
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs results

echo "=== $(hostname) $(date -u) ==="
nvidia-smi || true

# Environment lives on /work (home quotas on LONI are small).
source .venv/bin/activate 2>/dev/null || true

srun python "$@"
