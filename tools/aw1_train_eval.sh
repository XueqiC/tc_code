#!/bin/bash
# AW-1: train one arm on an AppWorld event pool, merge, serve on a rai GPU, run the OFFICIAL dev split
# (simplified_react_code_agent, same jsonnet as qwen35-4b-base_dev with only the served model name changed).
# Usage: aw1_train_eval.sh <tag> <pool.jsonl> <ce|ddpo> [gpu_index=4] [port=8950]
set -uo pipefail
cd "$(dirname "$0")/.."; PROJ=$PWD
TAG=$1; POOL=$2; MODE=$3; GPU=${4:-4}; PORT=${5:-8950}
UUID=$(nvidia-smi --query-gpu=index,gpu_uuid --format=csv,noheader | awk -F', ' -v g="$GPU" '$1==g{print $2}')
export AW_POOL_PATH=$PROJ/$POOL AW_MAX_PROMPT_TOKENS=4096 AW_TRUNCATE_SIDE=tail PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
case $MODE in ddpo) export AW_DISTILL=ddpo AW_SFT_SKIP=1 ;; ce) export AW_EPOCHS=${AW_EPOCHS:-1} ;; *) echo "mode?"; exit 2 ;; esac
OUT=results/appworld_students/$TAG
if [ ! -f $OUT/hub_merged/config.json ]; then
  CUDA_VISIBLE_DEVICES=$UUID .venv/bin/python src/appworld_train.py --selection full --budget 999999 --seed 0 --student Qwen/Qwen3.5-4B --tag "$TAG" > logs/aw1_train_$TAG.log 2>&1 || { echo "[aw1] train failed $TAG"; exit 1; }
  [ -f $OUT/hub_merged/config.json ] || CUDA_VISIBLE_DEVICES=$UUID .venv/bin/python tools/bfcl_hub_merge_export.py --adapter $OUT/adapter --out $OUT/hub_merged --model Qwen/Qwen3.5-4B >> logs/aw1_train_$TAG.log 2>&1
fi
[ -f $OUT/hub_merged/config.json ] || { echo "[aw1] no merged model for $TAG"; exit 1; }
# experiment config: copy of the base dev config with the served name swapped
CFG=envs/appworld-repo/experiments/configs/simplified_react_code_agent/local
sed -e "s/qwen35-4b-base/$TAG/g" -e "s#localhost:8950#localhost:$PORT#" $CFG/qwen35-4b-base_dev.jsonnet > $CFG/${TAG}_dev.jsonnet
if ss -ltn | grep -q ":$PORT "; then echo "[aw1] port $PORT busy"; exit 3; fi
export OPENAI_API_KEY=EMPTY OPENAI_BASE_URL=http://localhost:$PORT/v1
CUDA_VISIBLE_DEVICES=$UUID VLLM_USE_FLASHINFER_SAMPLER=0 nohup envs/vllm-serve/.venv/bin/vllm serve $PROJ/$OUT/hub_merged \
  --served-model-name "$TAG" --port $PORT --gpu-memory-utilization 0.45 --max-model-len 32768 --max-num-seqs 256 \
  --default-chat-template-kwargs '{"enable_thinking": false}' > logs/vllm_aw1_$TAG.log 2>&1 &
VPID=$!; trap 'kill $VPID 2>/dev/null' EXIT
for i in $(seq 1 120); do curl -s -m 3 localhost:$PORT/v1/models | grep -q "$TAG" && break; sleep 5; done
curl -s -m 3 localhost:$PORT/v1/models | grep -q "$TAG" || { echo "[aw1] vllm did not come up"; exit 1; }
(cd envs/appworld-repo && ../appworld-official/.venv/bin/appworld run simplified_react_code_agent/local/${TAG}_dev --root . --without-setup --num-processes 4 > $PROJ/logs/awoff_${TAG}_dev.log 2>&1)
echo "[aw1] $TAG dev run exit=$?"; grep -aA6 'Text Evaluation Report' logs/awoff_${TAG}_dev.log | tail -7
kill $VPID 2>/dev/null; sleep 10; echo "[aw1] $TAG DONE"
