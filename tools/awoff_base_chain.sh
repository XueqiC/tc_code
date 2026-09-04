#!/bin/bash
# Official AppWorld ReAct scaffold (stonybrooknlp/appworld 0.2, simplified_react_code_agent,
# unchanged prompt/max_steps=50) over the full dev split for Qwen3.5 base models served by
# vllm on GPU4 (non-thinking). Purpose (2026-09-01, user): our own harness gives base 0/40;
# measure base under the official scaffold before choosing the AppWorld base model.
set -uo pipefail
cd "$(dirname "$0")/.."
PROJ=$PWD
UUID=$(nvidia-smi --query-gpu=index,gpu_uuid --format=csv,noheader | awk -F', ' '$1==4{print $2}')
export OPENAI_API_KEY=EMPTY OPENAI_BASE_URL=http://localhost:8950/v1
serve() {  # $1 hf model, $2 served name
  CUDA_VISIBLE_DEVICES=$UUID VLLM_USE_FLASHINFER_SAMPLER=0 nohup envs/vllm-serve/.venv/bin/vllm serve "$1" \
    --served-model-name "$2" --port 8950 --gpu-memory-utilization 0.45 --max-model-len 32768 --max-num-seqs 256 \
    --default-chat-template-kwargs '{"enable_thinking": false}' > "logs/vllm_awoff_$2.log" 2>&1 &
  VPID=$!
  for i in $(seq 1 120); do curl -s -m 3 localhost:8950/v1/models | grep -q "$2" && return 0; sleep 5; done
  echo "[awoff] vllm for $2 did not come up"; return 1
}
run_dev() {  # $1 experiment name
  (cd envs/appworld-repo && ../appworld-official/.venv/bin/appworld run "$1" --root . --without-setup --num-processes 4 \
     > "$PROJ/logs/awoff_$(basename "$1").log" 2>&1)
  echo "[awoff] $1 exit=$?"
  grep -aA6 'Text Evaluation Report' "$PROJ/logs/awoff_$(basename "$1").log" | tail -7
}
# 4B: server already up from the smoke test if alive, else start it
curl -s -m 3 localhost:8950/v1/models | grep -q qwen35-4b-base || serve Qwen/Qwen3.5-4B qwen35-4b-base || exit 1
run_dev simplified_react_code_agent/local/qwen35-4b-base_dev
pkill -f 'served-model-name qwen35-4b-base' ; sleep 15
serve Qwen/Qwen3.5-9B qwen35-9b-base || exit 1
run_dev simplified_react_code_agent/local/qwen35-9b-base_dev
pkill -f 'served-model-name qwen35-9b-base'
echo "[awoff] CHAIN DONE"
