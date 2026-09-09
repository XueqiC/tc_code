#!/usr/bin/env bash
# AW-1 on one SLURM GPU: train, merge, serve, OFFICIAL simplified ReAct dev57.
# Usage: aw1_train_eval_hpg.sh <tag> <pool.jsonl> <ce|ddpo> [gpu=slurm] [port=job-derived]
# The legacy fourth GPU argument is accepted for contract compatibility and ignored.
# CUDA_VISIBLE_DEVICES is inherited unchanged from SLURM; the fifth port overrides
# 20000 + SLURM_JOB_ID % 30000. Logs and model/config names match rai AW-1.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
cd -- "$ROOT"
[[ $# -ge 3 && $# -le 5 ]] || {
  echo "Usage: aw1_train_eval_hpg.sh <tag> <pool.jsonl> <ce|ddpo> [gpu=slurm] [port]" >&2; exit 2;
}
TAG=$1; POOL=$2; MODE=$3
[[ $TAG =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || { echo "Invalid tag" >&2; exit 2; }
source tools/aw_hpg_common.sh
PORT=$(aw_hpg_port "${5:-}")
aw_hpg_require_gpu
aw_hpg_environment
aw_hpg_check_official
[[ -s $POOL ]] || { echo "Missing pool: $POOL" >&2; exit 2; }
for executable in .venv/bin/python envs/vllm-serve/.venv/bin/vllm; do
  [[ -x $executable ]] || { echo "Missing environment: $executable" >&2; exit 2; }
done
export AW_POOL_PATH=$(realpath -e -- "$POOL") AW_MAX_PROMPT_TOKENS=4096 AW_TRUNCATE_SIDE=tail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
case $MODE in
  ddpo) export AW_DISTILL=ddpo AW_SFT_SKIP=1 ;;
  ce) unset AW_DISTILL AW_SFT_SKIP; export AW_EPOCHS=${AW_EPOCHS:-1} ;;
  *) echo "Mode must be ce or ddpo" >&2; exit 2 ;;
esac
mkdir -p logs
OUT=results/appworld_students/$TAG
CFG=envs/appworld-repo/experiments/configs/simplified_react_code_agent/local
[[ -f $CFG/qwen35-4b-base_dev.jsonnet ]] || { echo "Missing official base dev config" >&2; exit 2; }
if [[ ! -f $OUT/hub_merged/config.json ]]; then
  .venv/bin/python src/appworld_train.py --selection full --budget 999999 --seed 0 \
    --student Qwen/Qwen3.5-4B --tag "$TAG" > "logs/aw1_train_$TAG.log" 2>&1 || {
    echo "[aw1] train failed $TAG"; exit 1;
  }
  if [[ ! -f $OUT/hub_merged/config.json ]]; then
    .venv/bin/python tools/bfcl_hub_merge_export.py --adapter "$OUT/adapter" \
      --out "$OUT/hub_merged" --model Qwen/Qwen3.5-4B >> "logs/aw1_train_$TAG.log" 2>&1
  fi
fi
[[ -f $OUT/hub_merged/config.json ]] || { echo "[aw1] no merged model for $TAG"; exit 1; }
sed -e "s/qwen35-4b-base/$TAG/g" -e "s#localhost:8950#localhost:$PORT#" \
  "$CFG/qwen35-4b-base_dev.jsonnet" > "$CFG/${TAG}_dev.jsonnet"
# Job-derived ports can still collide modulo the range: fail if occupied.
.venv/bin/python - "$PORT" <<'PY'
import socket, sys
with socket.socket() as sock:
    sock.bind(("127.0.0.1", int(sys.argv[1])))
PY
export OPENAI_API_KEY=EMPTY OPENAI_BASE_URL=http://localhost:$PORT/v1 VLLM_USE_FLASHINFER_SAMPLER=0
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
nohup envs/vllm-serve/.venv/bin/vllm serve "$ROOT/$OUT/hub_merged" \
  --served-model-name "$TAG" --host 127.0.0.1 --port "$PORT" \
  --gpu-memory-utilization 0.45 --max-model-len 32768 --max-num-seqs 256 \
  --default-chat-template-kwargs '{"enable_thinking": false}' > "logs/vllm_aw1_$TAG.log" 2>&1 &
VPID=$!
for i in $(seq 1 120); do
  kill -0 "$VPID" 2>/dev/null || { echo "[aw1] vllm exited; see logs/vllm_aw1_$TAG.log"; exit 1; }
  curl -s -m 3 "localhost:$PORT/v1/models" | grep -Fq "$TAG" && break
  sleep 5
done
curl -s -m 3 "localhost:$PORT/v1/models" | grep -Fq "$TAG" || { echo "[aw1] vllm did not come up"; exit 1; }
status=0
(cd envs/appworld-repo && ../appworld-official/.venv/bin/appworld run \
  "simplified_react_code_agent/local/${TAG}_dev" --root . --without-setup --num-processes 4 \
  > "$ROOT/logs/awoff_${TAG}_dev.log" 2>&1) || status=$?
echo "[aw1] $TAG dev run exit=$status"
grep -aA6 'Text Evaluation Report' "logs/awoff_${TAG}_dev.log" | tail -7 || true
(( status == 0 )) || exit "$status"
grep -aqE '^[[:space:]]*aggregate[[:space:]]*\|' "logs/awoff_${TAG}_dev.log" || {
  echo "[aw1] missing official aggregate for $TAG"; exit 1;
}
cleanup
VPID=
sleep 10
echo "[aw1] $TAG DONE"
