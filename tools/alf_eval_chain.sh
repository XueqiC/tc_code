#!/bin/bash
# ALFWorld official eval of the ours checkpoint, then the base student, on one GPU (by UUID).
# (The in-lane eval vllm failed to start right after training while GPU memory was still held.)
set -uo pipefail
cd "$(dirname "$0")/.."
UUID=$(nvidia-smi --query-gpu=index,gpu_uuid --format=csv,noheader | awk -F", " -v i="${ALF_GPU:-1}" '$1==i{print $2}')
export PYTHONPATH=src GPU_UTIL=${GPU_UTIL:-0.35} BFAS_PORT_BASE=${BFAS_PORT_BASE:-8930}
echo "[alfeval] ours"
.venv/bin/python tools/bfas_eval_ckpt.py --benchmark alfworld --checkpoint results/bfas/alfworld/ours_s0/checkpoint \
  --out results/bfas/alfworld/ours_s0 --gpu "$UUID" --port $BFAS_PORT_BASE --seed 0
echo "[alfeval] base"
.venv/bin/python tools/bfas_eval_ckpt.py --benchmark alfworld --checkpoint Qwen/Qwen3.5-2B \
  --out results/bfas/alfworld/base_s0 --gpu "$UUID" --port $BFAS_PORT_BASE --seed 0
echo "[alfeval] CHAIN DONE"
