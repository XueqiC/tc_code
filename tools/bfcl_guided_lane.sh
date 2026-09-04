#!/usr/bin/env bash
# Serve the base student and run the guided pass against it.
# The guided pass itself expects an already-running server (--skip-server-setup),
# which is why it needs this wrapper rather than being launched directly.
# Usage: bfcl_guided_lane.sh <gpu> <port> [repeats] [tag]
set -u
GPU=${1:-4}
PORT=${2:-8953}
REPEATS=${3:-4}
TAG=${4:-ds}
# resolve from this script's own location: hardcoding a home path made
# every hpg run skip evaluation silently ("NO ADAPTER") while still
# printing DONE, because /home/xueqi does not exist there
PROJ=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
STUDENT=Qwen/Qwen3.5-4B
cd "$PROJ"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=$GPU
export VLLM_USE_FLASHINFER_SAMPLER=0
"$PROJ/envs/vllm-serve/.venv/bin/vllm" serve "$STUDENT" \
  --served-model-name "$STUDENT" --port "$PORT" \
  --gpu-memory-utilization "${GPU_UTIL:-0.85}" --max-model-len 32768 \
  > "$PROJ/logs/bfcl_guided_vllm.log" 2>&1 &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null' EXIT

for _ in $(seq 1 300); do
  curl -s -m 3 "http://localhost:$PORT/v1/models" > /dev/null && break
  sleep 5
done
curl -s -m 3 "http://localhost:$PORT/v1/models" > /dev/null || {
  echo "[guidedlane] server never came up"; exit 1; }
echo "[guidedlane] server up on $PORT (gpu $GPU)"

PYTHONPATH=$PROJ/src "$PROJ/.venv/bin/python" tools/bfcl_guided_pass.py \
  --port "$PORT" --repeats "$REPEATS" --tag "$TAG"
echo "[guidedlane] GUIDED PASS COMPLETE"
