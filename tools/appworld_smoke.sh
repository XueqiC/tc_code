#!/bin/bash
# Deterministic AppWorld base-agent smoke test (CRCD spec §11 P0 item 3).
# Serves the base student with the project vLLM lane on ONE pinned GPU, runs the
# official scaffold through AppWorldOfficialAdapter on N train tasks R times at
# temperature 0, checks run-to-run identity + offline evaluator, kills the server.
#
#   bash tools/appworld_smoke.sh                 # GPU index 4, 3 tasks, 2 repeats
#   AW_GPU=1 N_TASKS=5 REPEATS=3 bash tools/appworld_smoke.sh
#   TASKS=82e2fac_1,07b42fd_1 bash tools/appworld_smoke.sh
#
# Outputs: logs/t4_appworld_smoke.log (this script), logs/t4_appworld_vllm.log,
#          results/analysis/appworld_audit_smoke.json, official run logs under
#          logs/bfas/appworld_official/s0_smoke*.log
set -uo pipefail
cd "$(dirname "$0")/.."
POLICY=${POLICY:-Qwen/Qwen3.5-4B}
PORT=${PORT:-8988}
GPU_UTIL=${GPU_UTIL:-0.4}
N_TASKS=${N_TASKS:-3}
REPEATS=${REPEATS:-2}
PROCESSES=${PROCESSES:-1}
KEEP=${KEEP:-1}
LOG=logs/t4_appworld_smoke.log
VLLM_LOG=logs/t4_appworld_vllm.log
VLLM=envs/vllm-serve/.venv/bin/vllm
UUID=$(nvidia-smi --query-gpu=index,gpu_uuid --format=csv,noheader | awk -F", " -v i="${AW_GPU:-4}" '$1==i{print $2}')
[ -n "$UUID" ] || { echo "GPU index ${AW_GPU:-4} not found" >&2; exit 2; }
mkdir -p logs results/analysis
exec > >(tee -a "$LOG") 2>&1
echo "=== $(date '+%F %T') appworld_smoke start policy=$POLICY port=$PORT gpu=${AW_GPU:-4} ($UUID) util=$GPU_UTIL tasks=$N_TASKS repeats=$REPEATS processes=$PROCESSES"

if ss -ltn 2>/dev/null | grep -q ":$PORT "; then echo "port $PORT already in use; refusing to start" >&2; exit 2; fi
# same flags as bfas.run.VLLMServer (shared student lane) so the smoke reflects the real pipeline
CUDA_VISIBLE_DEVICES="$UUID" VLLM_USE_FLASHINFER_SAMPLER=0 nohup "$VLLM" serve "$POLICY" \
  --served-model-name bfas-policy --port "$PORT" --gpu-memory-utilization "$GPU_UTIL" \
  --max-model-len 32768 --enable-auto-tool-choice --tool-call-parser hermes \
  > "$VLLM_LOG" 2>&1 &
VPID=$!
echo "vllm pid $VPID (log $VLLM_LOG)"
cleanup() {
  if kill -0 "$VPID" 2>/dev/null; then
    echo "killing vllm $VPID"; kill "$VPID"; sleep 5; kill -9 "$VPID" 2>/dev/null; pkill -9 -P "$VPID" 2>/dev/null
  fi
}
trap cleanup EXIT INT TERM
for i in $(seq 1 300); do
  kill -0 "$VPID" 2>/dev/null || { echo "vllm exited early; tail of $VLLM_LOG:"; tail -20 "$VLLM_LOG"; exit 2; }
  curl -sf "http://localhost:$PORT/v1/models" >/dev/null 2>&1 && break
  sleep 2
done
curl -sf "http://localhost:$PORT/v1/models" >/dev/null || { echo "vllm not ready after 600s"; exit 2; }
echo "vllm ready after ~$((i*2))s"
VLLM_VERSION=$("$VLLM" --version 2>/dev/null | tail -1)

KEEP_FLAG=""; [ "$KEEP" = 1 ] && KEEP_FLAG=--keep
TASK_FLAG=""; [ -n "${TASKS:-}" ] && TASK_FLAG="--tasks $TASKS"
PYTHONPATH=src .venv/bin/python tools/appworld_audit.py smoke --policy "$POLICY" --port "$PORT" \
  --n-tasks "$N_TASKS" --repeats "$REPEATS" --processes "$PROCESSES" --vllm-version "$VLLM_VERSION" $KEEP_FLAG $TASK_FLAG
RC=$?
echo "=== $(date '+%F %T') appworld_smoke done rc=$RC (0 = deterministic, 3 = runs differ)"
exit $RC
