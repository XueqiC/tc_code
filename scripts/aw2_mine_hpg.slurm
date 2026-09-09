#!/bin/bash
#SBATCH --job-name=aw2_mine
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
# Switch both account and qos with: sbatch --account=yd24f.fsu --qos=yd24f.fsu ...
# Mines independent midpoint states, then builds C and its offline exposure report.
set -euo pipefail
cd -- "${SLURM_SUBMIT_DIR:?Submit from the project root}"
ROOT=$(pwd -P)
source tools/aw_hpg_common.sh
aw_hpg_environment
bash tools/aw2_mine_more_hpg.sh
.venv/bin/python tools/aw2_expand_pool.py --mined data/appworld_events/aw2_spread_more_k3.jsonl
