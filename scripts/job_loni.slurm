#!/bin/bash
# LONI QB-D template. Partition/account/gres are placeholders until the cluster
# survey lands in docs/loni_setup.md; do not submit before they are filled in.
#SBATCH --job-name=tc-alignment
#SBATCH --account=PLACEHOLDER_ALLOCATION
#SBATCH --partition=PLACEHOLDER_GPU_PARTITION
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
