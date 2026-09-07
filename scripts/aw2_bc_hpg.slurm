#!/bin/bash
#SBATCH --job-name=aw2_bc
#SBATCH --account=fsu-compsci-dept
#SBATCH --qos=fsu-compsci-dept
#SBATCH --partition=hpg-b200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=240G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
# Submit from the project root after sync/setup; logs/ must exist before sbatch.
# Select AW2_ARMS=B, C, 'B C' (default), comma-separated names, or full arm tags.
# Switch both account and qos with: sbatch --account=yd24f.fsu --qos=yd24f.fsu ...
set -euo pipefail
cd -- "${SLURM_SUBMIT_DIR:?Submit from the project root}"
exec bash tools/aw2_runner_bc_hpg.sh
