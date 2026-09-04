#!/bin/bash
# Round-2 event mining (after the rai control arms free GPU4): serve the base student,
# resume stateful mining at episode 30 (multi_turn / web_search) and mine irrelevance
# (abstain) events for the single-turn set.
set -uo pipefail
cd "$(dirname "$0")/.."
UUID=$(nvidia-smi --query-gpu=index,gpu_uuid --format=csv,noheader | awk -F", " -v i="${MINE_GPU:-4}" '$1==i{print $2}')
CUDA_VISIBLE_DEVICES=$UUID VLLM_USE_FLASHINFER_SAMPLER=0 nohup envs/vllm-serve/.venv/bin/vllm serve Qwen/Qwen3.5-4B \
  --served-model-name Qwen/Qwen3.5-4B Qwen/Qwen3.5-4B-FC --port 8975 --gpu-memory-utilization 0.4 \
  --max-model-len 32768 --max-num-seqs 256 > logs/vllm_eventmine_8975.log 2>&1 &
VPID=$!
for i in $(seq 1 120); do curl -s -m 3 localhost:8975/v1/models | grep -q 'Qwen3.5-4B' && break; sleep 5; done
echo "[round2] vllm up ($VPID)"
PYTHONPATH=src:tools envs/bfcl/.venv/bin/python tools/bfcl_event_mine_single.py --port 8975 --k 4 --irrelevance \
  --out data/bfcl_sft/events_single_irrel_v2.jsonl > logs/bfcl_event_mine_single_v2.log 2>&1
echo "[round2] single/irrelevance done: $(wc -l < data/bfcl_sft/events_single_irrel_v2.jsonl) events"
PYTHONPATH=src:tools envs/bfcl/.venv/bin/python tools/bfcl_event_mine.py --port 8975 --start-index 30 --rollouts 3 --k 4 \
  --out data/bfcl_sft/events_v1.jsonl > logs/bfcl_event_mine_v1b.log 2>&1
echo "[round2] stateful resume done: $(wc -l < data/bfcl_sft/events_v1.jsonl) stateful events total"
kill $VPID 2>/dev/null
echo "[round2] DONE"
