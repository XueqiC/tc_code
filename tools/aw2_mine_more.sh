#!/bin/bash
# Run from the project root:
#   CUDA_VISIBLE_DEVICES=2 bash tools/aw2_mine_more.sh 8989
# Inspect the exact per-task commands without starting anything:
#   bash tools/aw2_mine_more.sh 8989 --dry-run
# Then build C/report (offline tokenizer):
#   .venv/bin/python tools/aw2_expand_pool.py --mined data/appworld_events/aw2_spread_more_k3.jsonl
# The miner itself starts its bridge with envs/appworld-official/.venv/bin/python,
# cwd/APPWORLD_ROOT=envs/appworld-repo. No global AppWorld installation or setup.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
[[ $(pwd -P) == "$ROOT" ]] || { echo "Run this script from the project root: $ROOT" >&2; exit 2; }
if [[ $# -lt 1 || $# -gt 2 || ! $1 =~ ^[0-9]{1,5}$ ]]; then
  echo "Usage: CUDA_VISIBLE_DEVICES=<one GPU> bash tools/aw2_mine_more.sh <port> [--dry-run]" >&2
  exit 2
fi
PORT=$((10#$1))
[[ $PORT -ge 1 && $PORT -le 65535 ]] || { echo "Invalid port" >&2; exit 2; }
DRY_RUN=0
if [[ $# == 2 ]]; then
  [[ $2 == --dry-run ]] || { echo "Unknown option: $2" >&2; exit 2; }
  DRY_RUN=1
fi
OUT=data/appworld_events/aw2_spread_more_k3.jsonl
MODEL=Qwen/Qwen3.5-4B
SERVED=bfas-policy
PLAN=$(.venv/bin/python tools/aw2_expand_pool.py --plan-probes)
[[ -n $PLAN ]] || { echo "No independent midpoint states remain" >&2; exit 2; }
miner_command() {
  local task=$1 probes=$2 positions=$3
  MINER=(.venv/bin/python tools/appworld_event_mine.py
    --demos results/bfas/appworld/collect_shared/demos.json
    --tasks "$task" --split demand --probe-select spread --probe-positions "$positions"
    --max-probes "$probes" --k 3 --probe-samples 3 --max-steps 50
    --max-cont-steps 20 --max-tokens 4096 --context-len 32768
    --thinking off --equality api_effects --temperature 0.7 --seed 0 --env-seed 1
    --student "$MODEL" --served-model "$SERVED" --port "$PORT"
    --run-tag aw2_spread_more_k3 --out "$OUT" --resume)
}
if (( DRY_RUN )); then
  while IFS=$'\t' read -r task probes positions; do
    miner_command "$task" "$probes" "$positions"
    printf '%q ' "${MINER[@]}"
    printf '\n'
  done <<< "$PLAN"
  exit 0
fi
[[ -n ${CUDA_VISIBLE_DEVICES:-} && $CUDA_VISIBLE_DEVICES != *,* && $CUDA_VISIBLE_DEVICES != -1 ]] || {
  echo "Set CUDA_VISIBLE_DEVICES to exactly one GPU index or UUID" >&2; exit 2;
}
for executable in .venv/bin/python envs/vllm-serve/.venv/bin/vllm envs/appworld-official/.venv/bin/python; do
  [[ -x $executable ]] || { echo "Missing project environment: $executable" >&2; exit 2; }
done
[[ -d envs/appworld-repo && -f results/bfas/appworld/collect_shared/demos.json ]] || {
  echo "Missing official AppWorld repo or archived demos" >&2; exit 2;
}
mkdir -p logs data/appworld_events
exec 9>logs/aw2_mine_more.lock
flock -n 9 || { echo "AW-2 mining is already running" >&2; exit 3; }
# appworld_event_mine.py appends. Never silently append a new run to old output.
[[ ! -e $OUT && ! -e $OUT.done ]] || {
  echo "Output already exists: $OUT (or .done); inspect/archive it before a fresh run" >&2; exit 2;
}
.venv/bin/python - "$PORT" <<'PY'
import socket, sys
with socket.socket() as sock:
    sock.bind(("127.0.0.1", int(sys.argv[1])))
PY
exec > >(tee -a logs/aw2_mine_more.log) 2>&1
export CUDA_VISIBLE_DEVICES OPENAI_API_KEY=EMPTY OPENAI_BASE_URL=http://localhost:$PORT/v1
export APPWORLD_ROOT=$ROOT/envs/appworld-repo PYTHONPATH=$ROOT/src
export VLLM_USE_FLASHINFER_SAMPLER=0
VPID=
cleanup() {
  if [[ -n $VPID ]]; then
    kill "$VPID" 2>/dev/null || true
    wait "$VPID" 2>/dev/null || true
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
nohup envs/vllm-serve/.venv/bin/vllm serve "$MODEL" \
  --served-model-name "$SERVED" --host 127.0.0.1 --port "$PORT" \
  --gpu-memory-utilization 0.45 --max-model-len 32768 --max-num-seqs 256 \
  --default-chat-template-kwargs '{"enable_thinking": false}' > logs/vllm_aw2_mine_more.log 2>&1 &
VPID=$!
ready=0
for ((i=0; i<120; i++)); do
  kill -0 "$VPID" 2>/dev/null || { echo "AW-2 vllm exited; see logs/vllm_aw2_mine_more.log"; exit 1; }
  if curl -fsS -m 3 "http://localhost:$PORT/v1/models" 2>/dev/null | grep -q "$SERVED"; then
    ready=1
    break
  fi
  sleep 5
done
(( ready )) || { echo "AW-2 vllm did not come up"; exit 1; }
while IFS=$'\t' read -r task probes positions; do
  miner_command "$task" "$probes" "$positions"
  printf '[aw2-mine] '
  printf '%q ' "${MINER[@]}"
  printf '\n'
  "${MINER[@]}"
  # The miner logs task errors but may return zero. Require its success sidecar.
  grep -Fxq -- "$task" "$OUT.done" || { echo "Mining incomplete for $task; inspect logs/aw2_mine_more.log"; exit 1; }
done <<< "$PLAN"
echo "[aw2-mine] DONE -> $OUT"
