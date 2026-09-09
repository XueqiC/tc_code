#!/usr/bin/env bash
# Serve the base student, then run the generation loop against it.
# Usage: bfcl_gen_lane.sh <gpu> <port> [extra args for bfcl_gen_loop.py]
set -u
GPU=${1:-4}; PORT=${2:-8971}; shift 2 || true
# resolve from this script's own location: hardcoding a home path made
# every hpg run skip evaluation silently ("NO ADAPTER") while still
# printing DONE, because /home/xueqi does not exist there
PROJ=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
STUDENT=Qwen/Qwen3.5-4B
cd "$PROJ"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU
export VLLM_USE_FLASHINFER_SAMPLER=0
"$PROJ/envs/vllm-serve/.venv/bin/vllm" serve "$STUDENT" \
  --served-model-name "$STUDENT" --port "$PORT" \
  --gpu-memory-utilization "${GPU_UTIL:-0.85}" --max-model-len 32768 \
  > "$PROJ/logs/bfcl_gen_vllm.log" 2>&1 &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null' EXIT
for _ in $(seq 1 300); do
  curl -s -m 3 "http://localhost:$PORT/v1/models" > /dev/null && break
  sleep 5
done
curl -s -m 3 "http://localhost:$PORT/v1/models" > /dev/null || {
  echo "[genlane] server never came up"; exit 1; }
echo "[genlane] student up on $PORT (gpu $GPU)"
OLLAMA_API_KEY="${OLLAMA_API_KEY:-$(cat ~/.ollama_api_key2 | tr -d '\n')}" \
PYTHONPATH=$PROJ/src "$PROJ/envs/bfcl/.venv/bin/python" -u \
  "$PROJ/tools/bfcl_gen_loop.py" --port "$PORT" "$@"
echo "[genlane] GEN LOOP COMPLETE"
