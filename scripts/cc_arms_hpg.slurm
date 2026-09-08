#!/bin/bash
#SBATCH --job-name=cc_arms
#SBATCH --account=fsu-compsci-dept
#SBATCH --qos=fsu-compsci-dept
#SBATCH --partition=hpg-b200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=240G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
# Submit from the project root with logs/ present.
# AW-style selection: CC_ARMS='A B C' (default), B, or A,C; sequential on this GPU.
# CC_STEPS must cover complete passes; CC_EPOCHS=1 by default.
set -euo pipefail
cd -- "${SLURM_SUBMIT_DIR:?Submit from the project root}"
exec bash tools/cc_three_arms.sh
