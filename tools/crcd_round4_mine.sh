#!/bin/bash
# CRCD round-4 self-iteration (mine the residual of the round-3 union model) (ollama-free): mine the residual intervention events of the
# round-2 preference model (crcd_r2_b3pref_s0, official 47.34) with the same GT-checker miners
# used for rounds 1-2, build the round-3 pools, and hand off training:
#   night (22:00-05:59)  -> sync pools + sbatch scripts/crcd_r4_hpg.slurm (official eval)
#   day                  -> train base on the union pool here (proxy eval)
# Arms defined in the slurm: r3_union (base on r2 ∪ r3 events) and r3_cont (continue from r2 model on r3 events).
set -uo pipefail
cd "$(dirname "$0")/.."
UUID=$(nvidia-smi --query-gpu=index,gpu_uuid --format=csv,noheader | awk -F", " -v i="${MINE_GPU:-2}" '$1==i{print $2}')
PORT=${MINE_PORT:-8976}
R2=results/appworld_students/crcd_r3_union_s0
export CUDA_VISIBLE_DEVICES=$UUID PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# 0. fetch the round-3 union model (seed 0, official 47.82) from hpg
mkdir -p $R2
[ -f $R2/adapter/model.safetensors ] || rsync -az -e "ssh -o BatchMode=yes" hpg:/blue/fsu-compsci-dept/xc25.fsu/hq/tc-alignment/$R2/adapter $R2/ || { echo "[r4] rsync failed"; exit 1; }
[ -f $R2/adapter/model.safetensors ] || { echo "[r4] adapter missing"; exit 1; }
# 1. merge
if [ ! -f $R2/hub_merged/config.json ]; then
  .venv/bin/python tools/bfcl_hub_merge_export.py --adapter $R2/adapter --out $R2/hub_merged --model Qwen/Qwen3.5-4B \
    > logs/crcd_r3_union_merge.log 2>&1 || { echo "[r4] MERGE FAILED"; exit 1; }
fi
echo "[r4] merged r2 model"
# 2. serve the r2 model under the base names the miners expect (retry: shared GPU, neighbours
#    releasing memory during vllm's profiling trips its free-memory assertion)
serve_ok=0
for attempt in 1 2 3 4; do
  VLLM_USE_FLASHINFER_SAMPLER=0 nohup envs/vllm-serve/.venv/bin/vllm serve $R2/hub_merged \
    --served-model-name Qwen/Qwen3.5-4B Qwen/Qwen3.5-4B-FC --port $PORT --gpu-memory-utilization ${GPU_UTIL:-0.3} \
    --max-model-len 32768 --max-num-seqs ${MAX_SEQS:-128} > logs/vllm_r4mine_$PORT.log 2>&1 &
  VPID=$!
  for i in $(seq 1 120); do
    curl -s -m 3 localhost:$PORT/v1/models | grep -q 'Qwen3.5-4B' && { serve_ok=1; break; }
    kill -0 $VPID 2>/dev/null || break
    sleep 5
  done
  [ $serve_ok = 1 ] && break
  echo "[r4] vllm attempt $attempt failed"; kill $VPID 2>/dev/null; sleep 20
done
[ $serve_ok = 1 ] || { echo "[r4] vllm failed"; exit 1; }
echo "[r4] vllm up ($VPID)"
# 3. mine: single-turn (+irrelevance), out-of-scope abstain, stateful first-divergence
PYTHONPATH=src:tools envs/bfcl/.venv/bin/python tools/bfcl_event_mine_single.py --port $PORT --k 4 --irrelevance \
  --out data/bfcl_sft/events_single_r4.jsonl > logs/bfcl_event_mine_single_r4.log 2>&1
echo "[r4] single done: $(wc -l < data/bfcl_sft/events_single_r4.jsonl) events"
PYTHONPATH=src:tools envs/bfcl/.venv/bin/python tools/bfcl_event_mine_single.py --port $PORT --k 4 --irrelevance --no-support \
  --gen data/bfcl_sft/gen_oos.jsonl --out data/bfcl_sft/events_single_oos_r4.jsonl > logs/bfcl_event_mine_single_oos_r4.log 2>&1
echo "[r4] oos done: $(wc -l < data/bfcl_sft/events_single_oos_r4.jsonl) events"
PYTHONPATH=src:tools envs/bfcl/.venv/bin/python tools/bfcl_event_mine.py --port $PORT --rollouts 3 --k 4 \
  --out data/bfcl_sft/events_r4.jsonl > logs/bfcl_event_mine_r4.log 2>&1
echo "[r4] stateful done: $(wc -l < data/bfcl_sft/events_r4.jsonl) events"
kill $VPID 2>/dev/null; sleep 10
# 4. pools: r3-only and union with round-2
.venv/bin/python tools/bfcl_events_to_pools.py --stateful data/bfcl_sft/events_r4.jsonl \
  --single data/bfcl_sft/events_single_r4.jsonl data/bfcl_sft/events_single_oos_r4.jsonl \
  --out-ce data/bfcl_sft/pool_events_ce_r4.jsonl --out-pref data/bfcl_sft/pool_events_pref_r4.jsonl 2>&1 | tail -3
.venv/bin/python - <<'EOF'
import json
seen=set(); out=[]
for f in ("data/bfcl_sft/pool_events_pref_v3.jsonl","data/bfcl_sft/pool_events_pref_r4.jsonl"):
    for l in open(f):
        if not l.strip(): continue
        r=json.loads(l); k=(r["task_id"], r.get("turn_index",0), r["response"][:200], str(r.get("_rejected",""))[:200])
        if k in seen: continue
        seen.add(k); out.append(l if l.endswith("\n") else l+"\n")
open("data/bfcl_sft/pool_events_pref_v4.jsonl","w").writelines(out)
print(f"[r4] union pool v4: {len(out)} rows")
EOF
echo "[r4] pools: r4=$(wc -l < data/bfcl_sft/pool_events_pref_r4.jsonl) union=$(wc -l < data/bfcl_sft/pool_events_pref_v4.jsonl)"
# 5. hand off
H=$(date +%H)
if [ "$H" -ge 22 ] || [ "$H" -lt 6 ]; then
  rsync -az -e "ssh -o BatchMode=yes" data/bfcl_sft/pool_events_pref_r4.jsonl data/bfcl_sft/pool_events_pref_v4.jsonl \
    hpg:/blue/fsu-compsci-dept/xc25.fsu/hq/tc-alignment/data/bfcl_sft/ && bash scripts/sync_to_hpg.sh >/dev/null 2>&1
  JOB=$(ssh -o BatchMode=yes hpg "cd /blue/fsu-compsci-dept/xc25.fsu/hq/tc-alignment && sbatch --parsable scripts/crcd_r4_hpg.slurm" 2>/dev/null | tail -1)
  echo "[r4] submitted hpg $JOB"
else
  echo "[r4] daytime -> rai train (union arm, proxy eval)"
  CRCD_GPU=${MINE_GPU:-2} CRCD_PORT=$PORT CRCD_ARMS=r4union bash tools/crcd_rai_extra.sh 2>&1 | tail -3
fi
echo "[r4] DONE"
