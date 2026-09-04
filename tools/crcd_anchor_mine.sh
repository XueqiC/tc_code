#!/bin/bash
# CRCD C-ladder, step 1 (ollama-free): risk-triggered retention anchors from the BASE student's own
# behaviour. Serve the base, sample K replies on every single-turn task (+irrelevance, +out-of-scope);
# where the base is right, keep (chosen = its own correct reply, rejected = its own failing reply or a
# minimal flip). Anchors carry dU = 0: they add no new capability, they pin what must not move.
# Then build the C1 pool = round-2 preference pool ∪ anchors and hand off training
# (night -> hpg scripts/crcd_c_hpg.slurm, official; day -> rai proxy).
set -uo pipefail
cd "$(dirname "$0")/.."
UUID=$(nvidia-smi --query-gpu=index,gpu_uuid --format=csv,noheader | awk -F", " -v i="${MINE_GPU:-2}" '$1==i{print $2}')
PORT=${MINE_PORT:-8978}
export CUDA_VISIBLE_DEVICES=$UUID PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# 0. wait for the round-3 chain to release the GPU (its vllm on 8976)
[ "${SKIP_WAIT:-0}" = 1 ] || for i in $(seq 1 360); do
  grep -q '\[r3\] DONE\|\[r3\] submitted\|\[r3\] vllm failed\|\[r3\] MERGE FAILED' logs/crcd_r3_mine.log 2>/dev/null && break
  sleep 30
done
serve_ok=0
for attempt in 1 2 3 4; do
  VLLM_USE_FLASHINFER_SAMPLER=0 nohup envs/vllm-serve/.venv/bin/vllm serve Qwen/Qwen3.5-4B \
    --served-model-name Qwen/Qwen3.5-4B Qwen/Qwen3.5-4B-FC --port $PORT --gpu-memory-utilization ${GPU_UTIL:-0.5} \
    --max-model-len 32768 --max-num-seqs ${MAX_SEQS:-128} > logs/vllm_anchor_$PORT.log 2>&1 &
  VPID=$!
  for i in $(seq 1 120); do
    curl -s -m 3 localhost:$PORT/v1/models | grep -q 'Qwen3.5-4B' && { serve_ok=1; break; }
    kill -0 $VPID 2>/dev/null || break
    sleep 5
  done
  [ $serve_ok = 1 ] && break
  echo "[anchor] vllm attempt $attempt failed"; kill $VPID 2>/dev/null; sleep 20
done
[ $serve_ok = 1 ] || { echo "[anchor] vllm failed"; exit 1; }
echo "[anchor] vllm up ($VPID)"
PYTHONPATH=src:tools envs/bfcl/.venv/bin/python tools/bfcl_event_mine_single.py --port $PORT --k 4 --irrelevance \
  --anchors-out data/bfcl_sft/anchors_single_base.jsonl --anchor-cap ${ANCHOR_CAP:-40} \
  --out data/bfcl_sft/events_single_base_rerun.jsonl > logs/bfcl_anchor_single.log 2>&1
echo "[anchor] single: $(wc -l < data/bfcl_sft/anchors_single_base.jsonl) anchors"
PYTHONPATH=src:tools envs/bfcl/.venv/bin/python tools/bfcl_event_mine_single.py --port $PORT --k 4 --irrelevance --no-support \
  --gen data/bfcl_sft/gen_oos.jsonl --anchors-out data/bfcl_sft/anchors_oos_base.jsonl --anchor-cap 60 \
  --out data/bfcl_sft/events_single_oos_base_rerun.jsonl > logs/bfcl_anchor_oos.log 2>&1
echo "[anchor] oos: $(wc -l < data/bfcl_sft/anchors_oos_base.jsonl) anchors"
kill $VPID 2>/dev/null; sleep 10
cat data/bfcl_sft/pool_events_pref_v2.jsonl data/bfcl_sft/anchors_single_base.jsonl data/bfcl_sft/anchors_oos_base.jsonl \
  > data/bfcl_sft/pool_events_pref_c1.jsonl
echo "[anchor] C1 pool: $(wc -l < data/bfcl_sft/pool_events_pref_c1.jsonl) rows (v2 210 + anchors)"
H=$(date +%H)
if [ "$H" -ge 22 ] || [ "$H" -lt 6 ]; then
  rsync -az -e "ssh -o BatchMode=yes" data/bfcl_sft/pool_events_pref_c1.jsonl data/bfcl_sft/anchors_single_base.jsonl \
    data/bfcl_sft/anchors_oos_base.jsonl hpg:/blue/fsu-compsci-dept/xc25.fsu/hq/tc-alignment/data/bfcl_sft/ \
    && bash scripts/sync_to_hpg.sh >/dev/null 2>&1
  JOB=$(ssh -o BatchMode=yes hpg "cd /blue/fsu-compsci-dept/xc25.fsu/hq/tc-alignment && sbatch --parsable scripts/crcd_c_hpg.slurm" 2>/dev/null | tail -1)
  echo "[anchor] submitted hpg $JOB"
else
  echo "[anchor] daytime -> rai train c1 (proxy eval)"
  CRCD_GPU=${MINE_GPU:-2} CRCD_PORT=$PORT CRCD_ARMS=c1anchor bash tools/crcd_rai_extra.sh 2>&1 | tail -3
fi
echo "[anchor] DONE"
