#!/bin/bash
# ALFWorld dU re-estimation lane (extra continuations for dU=0 events) (rai, no API): serve the base 4B student under the deployment
# name, mine demo-replay divergence events with K continuations, then build the preference pool.
set -uo pipefail
cd "$(dirname "$0")/.."
UUID=$(nvidia-smi --query-gpu=index,gpu_uuid --format=csv,noheader | awk -F", " -v i="${MINE_GPU:-1}" '$1==i{print $2}')
PORT=${MINE_PORT:-8981}
export CUDA_VISIBLE_DEVICES=$UUID PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True BFAS_ALFWORLD_STUDENT_REACT=1
serve_ok=0
for attempt in 1 2 3 4; do
  VLLM_USE_FLASHINFER_SAMPLER=0 nohup envs/vllm-serve/.venv/bin/vllm serve ${STUDENT:-Qwen/Qwen3.5-4B} \
    --served-model-name bfas-policy --port $PORT --gpu-memory-utilization ${GPU_UTIL:-0.4} \
    --max-model-len 16384 --max-num-seqs ${MAX_SEQS:-64} > logs/vllm_alfmine_$PORT.log 2>&1 &
  VPID=$!
  for i in $(seq 1 120); do
    curl -s -m 3 localhost:$PORT/v1/models | grep -q 'bfas-policy' && { serve_ok=1; break; }
    kill -0 $VPID 2>/dev/null || break
    sleep 5
  done
  [ $serve_ok = 1 ] && break
  echo "[alf-lane] vllm attempt $attempt failed"; kill $VPID 2>/dev/null; sleep 20
done
[ $serve_ok = 1 ] || { echo "[alf-lane] vllm failed"; exit 1; }
echo "[alf-lane] vllm up ($VPID)"
PYTHONPATH=src .venv/bin/python tools/alf_event_reestimate.py --port $PORT --k-extra ${K_EXTRA:-5} --select ${SELECT:-zero} ${REEST_EXTRA:-} --out ${OUT:-data/alf_sft/events_v1_k8.jsonl} > logs/alf_event_reest.log 2>&1
  --out ${OUT:-data/alf_sft/events_v1.jsonl} ${MINE_EXTRA:-} > logs/alf_event_mine.log 2>&1
echo "[alf-lane] reestimate done: $(wc -l < ${OUT:-data/alf_sft/events_v1_k8.jsonl}) rows"
kill $VPID 2>/dev/null; sleep 5
echo "[alf-lane] DONE"
