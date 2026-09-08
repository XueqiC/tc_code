#!/bin/bash
#SBATCH --job-name=bapilot2d
#SBATCH --account=fsu-compsci-dept
#SBATCH --qos=fsu-compsci-dept
#SBATCH --partition=hpg-b200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=200gb
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%A.out
# P2 pilot: 4 pilot sources + 2 no-op controls, independent micro-updates from the shared init state, 32 P_disc probes,
# numerical-floor / zero-delta / reload checks -> results/behavior_atom_v1/pilot_r2/pilot_pass.json. Fisher may be 'pending'.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export HF_HOME=/blue/fsu-compsci-dept/xc25.fsu/hq/tools/hf-cache
export HOME=/blue/fsu-compsci-dept/xc25.fsu/hq/tools/fakehome; mkdir -p "$HOME"
source .venv/bin/activate
export AW_MAX_PROMPT_TOKENS=4096 AW_TRUNCATE_SIDE=tail PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PYTHONPATH=src python tools/behavior_atom_experiment.py pilot --config configs/behavior_atom/pilot_r2.json --dry-run
PYTHONPATH=src python tools/behavior_atom_experiment.py pilot --config configs/behavior_atom/pilot_r2.json --profile --resume
echo "[bapilot] DONE"; cat results/behavior_atom_v1/pilot_r2/pilot_pass.json 2>/dev/null | head -c 600
