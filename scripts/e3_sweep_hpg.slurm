#!/bin/bash
#SBATCH --job-name=tc-e3-sweep
#SBATCH --account=yd24f.fsu
#SBATCH --qos=yd24f.fsu
#SBATCH --partition=hpg-b200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=240gb
#SBATCH --time=03:00:00
#SBATCH --array=0-11
#SBATCH --output=logs/%x_%A_%a.out

# Usage: sbatch --export=ALL,TC_USE_TMP=1 scripts/e3_sweep_hpg.slurm
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

# TC_USE_TMP=1 -> cache the model on node-local scratch (survives group-quota
# pressure on /blue; re-downloads each run, fine for one-off big models)
if [ "${TC_USE_TMP:-0}" = "1" ]; then
  export HF_HOME="$TMPDIR/hf-cache"
else
  export HF_HOME=/blue/yd24f.fsu/xc25.fsu/hq/tools/hf-cache
fi
mkdir -p "$HF_HOME" logs results
source .venv/bin/activate

manifest=configs/e3_sweep_manifest.txt
line_number=$((SLURM_ARRAY_TASK_ID + 1))
sweep_line=$(sed -n "${line_number}p" "$manifest")
if [ -z "$sweep_line" ]; then
  echo "No sweep configuration at $manifest line $line_number" >&2
  exit 1
fi

# Prevent inherited E3_* values from contaminating lines that use defaults.
unset E3_STUDENT E3_BUDGET E3_SEED E3_CONDITIONS E3_TASK E3_TASK_FILTER E3_TAG
read -r -a sweep_environment <<< "$sweep_line"
export "${sweep_environment[@]}"

echo "=== $(hostname) $(date) array=$SLURM_ARRAY_TASK_ID $sweep_line ==="
srun python src/e3_tier1.py
