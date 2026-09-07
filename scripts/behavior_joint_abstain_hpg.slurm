#!/bin/bash
#SBATCH --job-name=bajointA
#SBATCH --account=fsu-compsci-dept
#SBATCH --qos=fsu-compsci-dept
#SBATCH --partition=hpg-b200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=200gb
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%A.out
# P2 collect: 24 formal source micro-updates + 32 P_disc probe evaluations, sharded across the two accounts
# (SHARD env = i/n). Requires results/behavior_atom_v1/pilot_r1/pilot_pass.json (frozen config hash) copied from rai.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export HF_HOME=/blue/fsu-compsci-dept/xc25.fsu/hq/tools/hf-cache
export HOME=/blue/fsu-compsci-dept/xc25.fsu/hq/tools/fakehome; mkdir -p "$HOME"
source .venv/bin/activate
export AW_MAX_PROMPT_TOKENS=4096 AW_TRUNCATE_SIDE=tail PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SHARD=${SHARD:-0/2}
PYTHONPATH=src python tools/behavior_atom_experiment.py collect --config configs/behavior_atom/joint_abstain_r1.json --resume
echo "[bacol] shard $SHARD DONE"
