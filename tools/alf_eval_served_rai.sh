#!/bin/bash
# rai mirror of LONI slurm/eval_gpu.slurm: served vLLM 0.27.1 evaluation of one model on one GPU (PCI index).
# usage: bash tools/alf_eval_served_rai.sh <model_dir> <tag> <gpu_pci_index> [clients]
set -euo pipefail
R=/home/xueqi/hq/projects/tc-alignment-vllm
MODEL="${1:?model dir}"; TAG="${2:?tag}"; GPU="${3:?gpu pci index}"; N="${4:-14}"
RUN=$R/runs/$TAG-$(date +%s)
PORT=$(( 21000 + RANDOM % 9000 ))
mkdir -p "$RUN"
export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONPATH="$R/src:$R" PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_ATTENTION_BACKEND=FLASH_ATTN VLLM_USE_FLASHINFER_SAMPLER=0
echo "[$(date)] starting server for $TAG on gpu $GPU port $PORT run $RUN"
$R/envs/vllm-serve/.venv/bin/python -B $R/tools/alf_vllm_server.py \
  --model "$MODEL" --gpu "$GPU" --port "$PORT" --state-dir "$RUN/server" > "$RUN/server.stdout" 2>&1 &
SERVER_PID=$!
for i in $(seq 1 120); do
  [ -f "$RUN/server/server.json" ] && break
  kill -0 $SERVER_PID 2>/dev/null || { echo "server died"; tail -20 "$RUN/server/server.log" 2>/dev/null; tail -20 "$RUN/server.stdout"; exit 1; }
  sleep 15
done
[ -f "$RUN/server/server.json" ] || { echo "server never became ready"; kill $SERVER_PID 2>/dev/null || true; exit 1; }
echo "[$(date)] server ready"
CUDA_VISIBLE_DEVICES="" $R/.venv/bin/python -B $R/tools/rtd_alfworld_evaluate.py prepare \
  --model "$MODEL" --data-root $R/envs/alfworld/data/json_2.1.1 \
  --server-json "$RUN/server/server.json" --out "$RUN/binding.json" > "$RUN/prepare.log" 2>&1
echo "[$(date)] binding prepared"
pids=()
for ((i=0;i<N;i++)); do
  CUDA_VISIBLE_DEVICES="" $R/.venv/bin/python -B $R/tools/alf_eval_shard.py --root "$R" \
    --binding "$RUN/binding.json" --server-json "$RUN/server/server.json" \
    --output-root "$RUN/campaigns" --tag "$TAG" --shard $i --of $N --device cuda:0 \
    > "$RUN/shard_$i.log" 2>&1 &
  pids+=($!)
done
status=0
for p in "${pids[@]}"; do wait $p || status=1; done
echo "[$(date)] shards done (status $status)"
CUDA_VISIBLE_DEVICES="" $R/.venv/bin/python -B $R/tools/rtd_alfworld_evaluate.py run \
  --binding "$RUN/binding.json" --output-root "$RUN/campaigns" --tag "$TAG" > "$RUN/finalise.log" 2>&1 || true
kill $SERVER_PID 2>/dev/null || true
echo "[$(date)] EVAL_DONE $TAG"
grep -o "\"successes\": [0-9]*" "$RUN/finalise.log" | head -1 || tail -5 "$RUN/finalise.log"
