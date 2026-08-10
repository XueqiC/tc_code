#!/usr/bin/env bash
# Run a small experiment on rai (no scheduler).
# Usage: bash scripts/run_local.sh <gpu_ids> <exp_id> <cmd...>
# e.g.:  bash scripts/run_local.sh 2,3 pilot01 python src/train.py --config configs/pilot01.yaml
set -e
GPUS="$1"; EXP="$2"; shift 2
[ -z "$EXP" ] && { echo "usage: run_local.sh <gpu_ids> <exp_id> <cmd...>"; exit 1; }
PROJ_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ_DIR"
mkdir -p logs "results/$EXP"
echo "[run_local] CUDA_VISIBLE_DEVICES=$GPUS exp=$EXP cmd: $*"
CUDA_VISIBLE_DEVICES="$GPUS" EXP_ID="$EXP" nohup "$@" > "logs/$EXP.log" 2>&1 &
PID=$!
echo "[run_local] started pid=$PID, log: logs/$EXP.log"
echo "$PID"
