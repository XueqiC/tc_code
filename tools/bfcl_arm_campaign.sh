#!/usr/bin/env bash
# Sequentially score trained arms on BFCL v4: merge LoRA -> serve -> generate -> evaluate.
# Usage: bash tools/bfcl_arm_campaign.sh <gpu> <tag1> [tag2 ...]
set -u
GPU=$1; shift
PROJ=/home/xueqi/hq/projects/tc-alignment
BFCL_DIR=$PROJ/envs/bfcl
PORT=8901
cd "$PROJ"
export CUDA_DEVICE_ORDER=PCI_BUS_ID

for TAG in "$@"; do
  ADAPTER=$PROJ/results/appworld_students/$TAG/adapter
  MERGED=$PROJ/results/appworld_students/$TAG/merged
  MODEL_NAME="Qwen/Qwen3.5-4B"
  if [ ! -d "$ADAPTER" ]; then echo "[bfclarm] $TAG NO ADAPTER, skip"; continue; fi
  if [ ! -d "$MERGED" ]; then
    CUDA_VISIBLE_DEVICES=$GPU .venv/bin/python - "$ADAPTER" "$MERGED" << 'PYEOF'
import sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
adapter, out = sys.argv[1], sys.argv[2]
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-4B", dtype=torch.bfloat16, device_map="cpu")
m = PeftModel.from_pretrained(base, adapter)
m = m.merge_and_unload()
m.save_pretrained(out)
AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B").save_pretrained(out)
print("merged ->", out)
PYEOF
    [ -d "$MERGED" ] || { echo "[bfclarm] $TAG MERGE FAILED"; continue; }
  fi
  # stop any previous server on the port
  PREV=$(lsof -ti tcp:$PORT 2>/dev/null | head -1)
  if [ -n "${PREV:-}" ]; then kill $PREV; sleep 10; fi
  CUDA_VISIBLE_DEVICES=$GPU VLLM_USE_FLASHINFER_SAMPLER=0 nohup \
    $PROJ/envs/vllm-serve/.venv/bin/vllm serve "$MERGED" \
    --served-model-name "$MODEL_NAME" --port $PORT \
    --gpu-memory-utilization 0.85 --max-model-len 32768 \
    > "$PROJ/logs/vllm_${TAG}.log" 2>&1 &
  SPID=$!
  ok=""
  for i in $(seq 1 120); do
    if curl -s -o /dev/null -w "%{http_code}" http://localhost:$PORT/v1/models 2>/dev/null | grep -q 200; then ok=1; break; fi
    kill -0 $SPID 2>/dev/null || break
    sleep 5
  done
  [ -n "$ok" ] || { echo "[bfclarm] $TAG SERVER FAILED"; continue; }
  cd "$BFCL_DIR"
  . .venv/bin/activate
  export LOCAL_SERVER_ENDPOINT=localhost LOCAL_SERVER_PORT=$PORT
  RESULT_ROOT=$BFCL_DIR/gorilla/berkeley-function-call-leaderboard
  rm -rf "$RESULT_ROOT/result/Qwen_Qwen3.5-4B-FC"
  bfcl generate --model Qwen/Qwen3.5-4B-FC --test-category all --skip-server-setup --num-threads 8 \
    > "$PROJ/logs/bfcl_gen_${TAG}.log" 2>&1
  bfcl evaluate --model Qwen/Qwen3.5-4B-FC > "$PROJ/logs/bfcl_eval_${TAG}.log" 2>&1
  mkdir -p "$PROJ/results/bfcl/$TAG"
  cp "$RESULT_ROOT/score/data_overall.csv" "$PROJ/results/bfcl/$TAG/" 2>/dev/null
  cp -r "$RESULT_ROOT/score/Qwen_Qwen3.5-4B-FC" "$PROJ/results/bfcl/$TAG/scoredir" 2>/dev/null
  rm -rf "$RESULT_ROOT/score/Qwen_Qwen3.5-4B-FC" "$RESULT_ROOT/result/Qwen_Qwen3.5-4B-FC"
  deactivate
  cd "$PROJ"
  OV=$(python3 -c "import csv;print([r['Overall Acc'] for r in csv.DictReader(open('results/bfcl/$TAG/data_overall.csv'))][0])" 2>/dev/null)
  echo "[bfclarm] $TAG OVERALL=$OV"
  kill $SPID 2>/dev/null
  sleep 8
done
echo "[bfclarm] CAMPAIGN COMPLETE"
