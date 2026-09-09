#!/bin/bash
# CRCD A-ladder validity gate (overnight, rai GPU by UUID):
#   for each fingerprint cluster c: train preference-only on that cluster's events
#   (1 DPO epoch), hub-merge, serve, and measure the student's pass rate on EVERY
#   single-turn task of the round-2 pool (K=4). Together with the base rates this
#   gives the gain matrix G[c, c'] (train on c, gain on c'); crcd_gate_analyze.py
#   compares it with the fingerprint-predicted transfer (centroid cosine).
set -uo pipefail
cd "$(dirname "$0")/.."
GATE=data/bfcl_sft/gate
UUID=$(nvidia-smi --query-gpu=index,gpu_uuid --format=csv,noheader | awk -F", " -v i="${GATE_GPU:-4}" '$1==i{print $2}')
PORT=${GATE_PORT:-8977}
export CUDA_VISIBLE_DEVICES=$UUID AW_MAX_PROMPT_TOKENS=4096 AW_TRUNCATE_SIDE=tail AW_GRAD_CKPT=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
serve() {  # $1 model path
  VLLM_USE_FLASHINFER_SAMPLER=0 nohup envs/vllm-serve/.venv/bin/vllm serve "$1" --served-model-name Qwen/Qwen3.5-4B Qwen/Qwen3.5-4B-FC \
    --port $PORT --gpu-memory-utilization ${GPU_UTIL:-0.4} --max-model-len 32768 --max-num-seqs 256 > logs/vllm_gate.log 2>&1 &
  VPID=$!
  for i in $(seq 1 120); do curl -s -m 3 localhost:$PORT/v1/models | grep -q 'Qwen3.5-4B' && return 0; sleep 5; done
  echo "[gate] vllm failed for $1"; return 1
}
measure() {  # $1 rates file
  PYTHONPATH=src:tools envs/bfcl/.venv/bin/python tools/bfcl_event_mine_single.py --port $PORT --k 4 --no-support \
    --gen data/bfcl_sft/gen_pool_v3.jsonl data/bfcl_sft/gen_pool_v4.jsonl data/bfcl_sft/gen_oos.jsonl --irrelevance \
    --only-ids $GATE/measure_ids.json --rates-out "$1" --out /dev/null 2>&1 | grep -E 'rates|FINAL' | cut -c1-160
}
# support single-turn tasks are part of the measurement set too: include them by not passing --no-support
measure_all() {
  PYTHONPATH=src:tools envs/bfcl/.venv/bin/python tools/bfcl_event_mine_single.py --port $PORT --k 4 \
    --gen data/bfcl_sft/gen_pool_v3.jsonl data/bfcl_sft/gen_pool_v4.jsonl data/bfcl_sft/gen_oos.jsonl --irrelevance \
    --only-ids $GATE/measure_ids.json --rates-out "$1" --out /dev/null 2>&1 | grep -E 'rates|FINAL' | cut -c1-160
}
echo "[gate] === base rates ==="
if [ ! -f $GATE/rates_base.json ]; then
  serve Qwen/Qwen3.5-4B || exit 1
  measure_all $GATE/rates_base.json
  kill $VPID; sleep 10
fi
for POOL in $GATE/pool_c*.jsonl; do
  C=$(basename $POOL .jsonl); C=${C#pool_}
  TAG=crcdgate_${C}_s0
  [ -f $GATE/rates_${C}.json ] && { echo "[gate] $C already measured"; continue; }
  echo "[gate] === train $TAG on $(wc -l < $POOL) events ==="
  env AW_POOL_PATH=$PWD/$POOL AW_DISTILL=ddpo AW_SFT_SKIP=1 AW_DDPO_EPOCHS=1 .venv/bin/python src/appworld_train.py \
    --selection full --budget 999999 --seed 0 --student Qwen/Qwen3.5-4B --tag "$TAG" > logs/${TAG}_train.log 2>&1 \
    || { echo "[gate] $TAG TRAIN FAILED"; continue; }
  rm -rf results/appworld_students/$TAG/hub_merged
  .venv/bin/python tools/bfcl_hub_merge_export.py --adapter results/appworld_students/$TAG/adapter \
    --out results/appworld_students/$TAG/hub_merged --model Qwen/Qwen3.5-4B > logs/${TAG}_merge.log 2>&1 \
    || { echo "[gate] $TAG MERGE FAILED"; continue; }
  serve results/appworld_students/$TAG/hub_merged || continue
  measure_all $GATE/rates_${C}.json
  kill $VPID; sleep 10
  rm -rf results/appworld_students/$TAG/hub_merged  # 8GB each; adapters kept
done
echo "[gate] GATE DONE"
