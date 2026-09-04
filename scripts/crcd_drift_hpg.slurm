#!/bin/bash
#SBATCH --job-name=crcdDrift
#SBATCH --account=fsu-compsci-dept
#SBATCH --qos=fsu-compsci-dept
#SBATCH --partition=hpg-b200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60gb
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
# Drift-by-round / drift-by-anchor diagnostics (user prompt 11 & 19): KL(theta||base) on the 149
# mastered prompts for the official checkpoints kept on hpg.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export HF_HOME=/blue/fsu-compsci-dept/xc25.fsu/hq/tools/hf-cache
export HOME=/blue/fsu-compsci-dept/xc25.fsu/hq/tools/fakehome; mkdir -p "$HOME"
source .venv/bin/activate
M="base=Qwen/Qwen3.5-4B"
for t in crcd_b3pref_s0 crcd_r2_b3pref_s0 crcd_r2_b3pref_ep3_s0 crcd_r3_union_s0 crcd_r3u_ep2_s0 crcd_r3_cont_s0 crcd_c1_anchor_s0 crcd_c2_selfanchor_s0 crcd_c3_balanced_s0; do
  [ -f results/appworld_students/$t/adapter/model.safetensors ] && M="$M ${t#crcd_}=results/appworld_students/$t/adapter"
done
PYTHONPATH=src python tools/crcd_drift_diag.py --models $M \
  --prompts data/bfcl_sft/anchors_single_base.jsonl data/bfcl_sft/anchors_oos_base.jsonl \
  --out results/analysis/crcd_drift_rounds.json
echo "[crcdDrift] DONE"
