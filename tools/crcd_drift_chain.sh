#!/bin/bash
# Corrective-overwrite study (rai GPU by index): retrain CE (3 ep) and pairwise (1 ep) on the same
# round-2 pool (no eval), then run the drift diagnostics against the base on mastered prompts.
set -uo pipefail
cd "$(dirname "$0")/.."
UUID=$(nvidia-smi --query-gpu=index,gpu_uuid --format=csv,noheader | awk -F", " -v i="${CRCD_GPU:-1}" '$1==i{print $2}')
export CUDA_VISIBLE_DEVICES=$UUID AW_MAX_PROMPT_TOKENS=4096 AW_TRUNCATE_SIDE=tail PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True AW_GRAD_CKPT=1
POOL=$PWD/data/bfcl_sft/pool_events_pref_v2.jsonl
[ -f results/appworld_students/crcdr_ce_r2_s0/adapter/model.safetensors ] || \
  env AW_POOL_PATH=$PWD/data/bfcl_sft/pool_events_ce_v2.jsonl .venv/bin/python src/appworld_train.py --selection full --budget 999999 --seed 0 --student Qwen/Qwen3.5-4B --tag crcdr_ce_r2_s0 > logs/crcdr_ce_r2_s0_train.log 2>&1 || echo "[drift] CE TRAIN FAILED"
[ -f results/appworld_students/crcdr_pref_r2_s0/adapter/model.safetensors ] || \
  env AW_POOL_PATH=$POOL AW_DISTILL=ddpo AW_SFT_SKIP=1 .venv/bin/python src/appworld_train.py --selection full --budget 999999 --seed 0 --student Qwen/Qwen3.5-4B --tag crcdr_pref_r2_s0 > logs/crcdr_pref_r2_s0_train.log 2>&1 || echo "[drift] PREF TRAIN FAILED"
[ -f results/appworld_students/crcdr_ce1ep_r2_s0/adapter/model.safetensors ] || \
  env AW_POOL_PATH=$PWD/data/bfcl_sft/pool_events_ce_v2.jsonl AW_EPOCHS=1 .venv/bin/python src/appworld_train.py --selection full --budget 999999 --seed 0 --student Qwen/Qwen3.5-4B --tag crcdr_ce1ep_r2_s0 > logs/crcdr_ce1ep_r2_s0_train.log 2>&1 || echo "[drift] CE1EP TRAIN FAILED"
echo "[drift] training done"
PYTHONPATH=src .venv/bin/python tools/crcd_drift_diag.py --models base=Qwen/Qwen3.5-4B \
  ce=results/appworld_students/crcdr_ce_r2_s0/adapter ce1ep=results/appworld_students/crcdr_ce1ep_r2_s0/adapter \
  pref=results/appworld_students/crcdr_pref_r2_s0/adapter \
  --prompts data/bfcl_sft/anchors_single_base.jsonl data/bfcl_sft/anchors_oos_base.jsonl \
  --out results/analysis/crcd_drift_v1.json 2>&1 | grep -E '\[drift\]|Error|Traceback'
echo "[drift] DONE"
