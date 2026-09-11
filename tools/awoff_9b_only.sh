#!/bin/bash
# 9B leg of the official-scaffold base re-measure (standalone copy of the logic in
# tools/awoff_base_chain.sh), rerun after the Mamba-cache/max_num_seqs failure with
# --max-num-seqs 256.
set -uo pipefail
cd "$(dirname "$0")/.."
PROJ=$PWD
UUID=$(nvidia-smi --query-gpu=index,gpu_uuid --format=csv,noheader | awk -F', ' '$1==4{print $2}')
export OPENAI_API_KEY=EMPTY OPENAI_BASE_URL=http://localhost:8950/v1
NAME=qwen35-9b-base
CUDA_VISIBLE_DEVICES=$UUID VLLM_USE_FLASHINFER_SAMPLER=0 nohup envs/vllm-serve/.venv/bin/vllm serve Qwen/Qwen3.5-9B \
  --served-model-name $NAME --port 8950 --gpu-memory-utilization 0.45 --max-model-len 32768 --max-num-seqs 256 \
  --default-chat-template-kwargs '{"enable_thinking": false}' > "logs/vllm_awoff_$NAME.log" 2>&1 &
VPID=$!
up=0
for i in $(seq 1 120); do
  curl -s -m 3 localhost:8950/v1/models | grep -q $NAME && { up=1; break; }
  sleep 5
done
[ $up = 1 ] || { echo "[awoff] vllm for $NAME did not come up (2nd try)"; kill $VPID 2>/dev/null; exit 1; }
EXP=simplified_react_code_agent/local/qwen35-9b-base_dev
(cd envs/appworld-repo && ../appworld-official/.venv/bin/appworld run "$EXP" --root . --without-setup --num-processes 4 \
   > "$PROJ/logs/awoff_qwen35-9b-base_dev.log" 2>&1)
echo "[awoff] $EXP exit=$?"
grep -aA6 'Text Evaluation Report' "$PROJ/logs/awoff_qwen35-9b-base_dev.log" | tail -7
kill $VPID 2>/dev/null
echo "[awoff] CHAIN DONE (9B)"
