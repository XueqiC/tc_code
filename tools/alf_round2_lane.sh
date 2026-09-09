#!/bin/bash
# ALFWorld round-2 residual mining from the CE model (alfabl_CE_conseq_s0, official 74.6%): fetch its
# weights from hpg, merge on rai, serve under the deployment name, mine demo-replay events (K=3,
# <=3 probes/demo) -> data/alf_sft/events_r2ce.jsonl. Tests the refined operator rule: once the student
# is competent, does relative repair beat CE on the residual?
set -uo pipefail
cd "$(dirname "$0")/.."
M=results/appworld_students/alfabl_CE_conseq_s0
if [ ! -f $M/adapter/model.safetensors ]; then
  rsync -az -e "ssh -o BatchMode=yes" hpg:/blue/fsu-compsci-dept/xc25.fsu/hq/tc-alignment/$M/adapter $M/ || { echo "[alf-r2] rsync failed"; exit 1; }
fi
UUID=$(nvidia-smi --query-gpu=index,gpu_uuid --format=csv,noheader | awk -F", " -v i="${MINE_GPU:-3}" '$1==i{print $2}')
export CUDA_VISIBLE_DEVICES=$UUID PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True BFAS_ALFWORLD_STUDENT_REACT=1
if [ ! -f $M/hub_merged/config.json ]; then
  .venv/bin/python tools/bfcl_hub_merge_export.py --adapter $M/adapter --out $M/hub_merged --model Qwen/Qwen3.5-4B > logs/alf_ce_merge.log 2>&1 || { echo "[alf-r2] MERGE FAILED"; exit 1; }
fi
echo "[alf-r2] model ready"
PORT=${MINE_PORT:-8984}; serve_ok=0
for attempt in 1 2 3 4; do
  VLLM_USE_FLASHINFER_SAMPLER=0 nohup envs/vllm-serve/.venv/bin/vllm serve $M/hub_merged --served-model-name bfas-policy --port $PORT \
    --gpu-memory-utilization ${GPU_UTIL:-0.45} --max-model-len 16384 --max-num-seqs 64 > logs/vllm_alfr2_$PORT.log 2>&1 &
  VPID=$!
  for i in $(seq 1 120); do curl -s -m 3 localhost:$PORT/v1/models | grep -q 'bfas-policy' && { serve_ok=1; break; }; kill -0 $VPID 2>/dev/null || break; sleep 5; done
  [ $serve_ok = 1 ] && break; echo "[alf-r2] vllm attempt $attempt failed"; kill $VPID 2>/dev/null; sleep 20
done
[ $serve_ok = 1 ] || { echo "[alf-r2] vllm failed"; exit 1; }
echo "[alf-r2] vllm up ($VPID)"
rm -f data/alf_sft/events_r2ce.jsonl
PYTHONPATH=src .venv/bin/python tools/alf_event_mine.py --port $PORT --k 3 --max-probes 3 --student $M/hub_merged \
  --out data/alf_sft/events_r2ce.jsonl > logs/alf_event_mine_r2ce.log 2>&1
echo "[alf-r2] mining done: $(wc -l < data/alf_sft/events_r2ce.jsonl) events"
kill $VPID 2>/dev/null
echo "[alf-r2] DONE"
