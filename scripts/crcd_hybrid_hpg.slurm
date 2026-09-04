#!/bin/bash
#SBATCH --job-name=crcdHyb
#SBATCH --account=fsu-compsci-dept
#SBATCH --qos=fsu-compsci-dept
#SBATCH --partition=hpg-b200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=100gb
#SBATCH --time=06:00:00
#SBATCH --array=0
#SBATCH --output=logs/%x_%A_%a.out
# Residual-conditional operator (candidate CRCD v2, 2026-09-03 23:10): absolute correction (CE) on the
# consequential events of task families the base cannot do (pool_incomp_v2: JS / live_parallel_multiple /
# java, base rate < 0.1), then relative repair (base-centred DPO, reference = the CE stage) on the events of
# families the base can already do (pool_comp_v2). Compare with ce_incomp / pref_comp alone (41032704),
# r2 pref 47.34 and base 46.06 on the official suite.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export HF_HOME=/blue/fsu-compsci-dept/xc25.fsu/hq/tools/hf-cache
export HOME=/blue/fsu-compsci-dept/xc25.fsu/hq/tools/fakehome; mkdir -p "$HOME"
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export VLLM_CACHE_ROOT=/blue/fsu-compsci-dept/xc25.fsu/hq/tools/vllm-cache TORCHINDUCTOR_CACHE_DIR=/blue/fsu-compsci-dept/xc25.fsu/hq/tools/inductor-cache
export TRITON_CACHE_DIR=/blue/fsu-compsci-dept/xc25.fsu/hq/tools/triton-cache XDG_CACHE_HOME=/blue/fsu-compsci-dept/xc25.fsu/hq/tools/xdg-cache
mkdir -p "$VLLM_CACHE_ROOT" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME"
source .venv/bin/activate
export AW_MAX_PROMPT_TOKENS=4096 AW_TRUNCATE_SIDE=tail PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
S1=crcd_hyb_stage1_s0; S2=crcd_hybrid_s0
# stage 1: CE on incompetent-family events (50 rows x 7 ep ~ 44 steps)
AW_POOL_PATH=$PWD/data/bfcl_sft/pool_incomp_v2.jsonl AW_EPOCHS=7 \
  python src/appworld_train.py --selection full --budget 999999 --seed 0 --student Qwen/Qwen3.5-4B --tag $S1
rm -rf results/appworld_students/$S1/hub_merged
python tools/bfcl_hub_merge_export.py --adapter results/appworld_students/$S1/adapter --out results/appworld_students/$S1/hub_merged --model Qwen/Qwen3.5-4B
# stage 2: base-centred pairwise on competent-family events, starting from (and referenced to) the stage-1 model
AW_POOL_PATH=$PWD/data/bfcl_sft/pool_comp_v2.jsonl AW_DISTILL=ddpo AW_SFT_SKIP=1 AW_DDPO_EPOCHS=3 \
  python src/appworld_train.py --selection full --budget 999999 --seed 0 --student $PWD/results/appworld_students/$S1/hub_merged --tag $S2
export BFCLSTD_BASE_MODEL=$PWD/results/appworld_students/$S1/hub_merged
bash tools/bfcl_std_campaign.sh 0 9160 "$S2"
[ -f "results/bfcl_std/$S2/data_overall.csv" ] || { echo "[crcdHyb] $S2 NO SCORE PRODUCED"; exit 1; }
echo "[crcdHyb] $S2 DONE"
