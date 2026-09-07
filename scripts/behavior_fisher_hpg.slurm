#!/bin/bash
#SBATCH --job-name=bafish
#SBATCH --account=fsu-compsci-dept
#SBATCH --qos=fsu-compsci-dept
#SBATCH --partition=hpg-b200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=200gb
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%A.out
# Training-layout diagonal Fisher for the P2 metric H = diag(F)+eps I (student continuations of the v3t pool,
# ell = full_sequence, same LoRA layout/init seed as the micro-updates). Output copied back to rai by the caller.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export HF_HOME=/blue/fsu-compsci-dept/xc25.fsu/hq/tools/hf-cache
export HOME=/blue/fsu-compsci-dept/xc25.fsu/hq/tools/fakehome; mkdir -p "$HOME"
source .venv/bin/activate
export AW_MAX_PROMPT_TOKENS=4096 AW_TRUNCATE_SIDE=tail PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p data/behavior_atom_v1
PYTHONPATH=src python tools/behavior_atom/fisher_train_layout.py --benchmark-rows 3 --pool data/bfcl_sft/pool_events_pref_v3t.jsonl --owner student --pool-role training --ell full_sequence --seed 0 --device cuda --output data/behavior_atom_v1/fisher_bench.npz
PYTHONPATH=src python tools/behavior_atom/fisher_train_layout.py --pool data/bfcl_sft/pool_events_pref_v3t.jsonl --owner student --pool-role training --ell full_sequence --seed 0 --device cuda --init-state data/behavior_atom_v1/init_state_seed0.pt --output data/behavior_atom_v1/fisher_v3t_student_full.npz
echo "[bafish] DONE"
