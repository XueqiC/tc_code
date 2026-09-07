#!/usr/bin/env bash
# hpg: $HOME is full (quota) -> keep vllm/torch/triton compile caches on /blue, never under ~/.cache
if [ -d /blue/fsu-compsci-dept/xc25.fsu/hq/tools ]; then
  export VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-/blue/fsu-compsci-dept/xc25.fsu/hq/tools/vllm-cache}
  export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-/blue/fsu-compsci-dept/xc25.fsu/hq/tools/inductor-cache}
  export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/blue/fsu-compsci-dept/xc25.fsu/hq/tools/triton-cache}
  export XDG_CACHE_HOME=${XDG_CACHE_HOME:-/blue/fsu-compsci-dept/xc25.fsu/hq/tools/xdg-cache}
  mkdir -p "$VLLM_CACHE_ROOT" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME"
  export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1   # vllm usage-stats writer targets ~/.config (home is full)
  export HOME=/blue/fsu-compsci-dept/xc25.fsu/hq/tools/fakehome; mkdir -p "$HOME"
fi

set -u
if [ "$#" -lt 3 ]; then
  echo "Usage: bash tools/bfcl_std_campaign.sh <GPU> <PORT|auto> <tag1> [tag2 ...]" >&2
  exit 2
fi
GPU=$1 PORT=$2
shift 2
# resolve from this script's own location: hardcoding a home path made
# every hpg run skip evaluation silently ("NO ADAPTER") while still
# printing DONE, because /home/xueqi does not exist there
PROJ=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
if [ "${BFCLSTD_LOCKED:-0}" != 1 ]; then
  exec python3 "$PROJ/tools/bfcl_campaign_lock.py" "$GPU" "$PORT" "$@"
fi
LEADERBOARD=$PROJ/envs/bfcl/gorilla/berkeley-function-call-leaderboard
# the bfcl entry point sits in a different venv on each machine
# (rai: envs/bfcl/.venv, hpg: envs/bfcl-venv), and a wrong guess surfaces only
# as "env: No such file or directory" buried in a generate log
BFCL=""
for cand in "$PROJ/envs/bfcl/.venv/bin/bfcl" "$PROJ/envs/bfcl-venv/bin/bfcl"; do
  [ -x "$cand" ] && { BFCL=$cand; break; }
done
if [ -z "$BFCL" ]; then
  echo "no bfcl executable under $PROJ/envs (looked in bfcl/.venv and bfcl-venv)" >&2
  exit 3
fi
# The lease wrapper exclusively creates fresh directories for every attempt.
# Retries within this invocation retain them; no later run may reuse them.
RES_SUB=${BFCLSTD_RESULT_SUB:?missing campaign result namespace}
SCORE_SUB=${BFCLSTD_SCORE_SUB:?missing campaign score namespace}
MODEL_DIR=Qwen_Qwen3.5-4B-FC
mkdir -p "$PROJ/logs" "$PROJ/results/bfcl_std" "$PROJ/_trash"
cleanup_harness_outputs() {
  rm -rf -- "$LEADERBOARD/$SCORE_SUB/$MODEL_DIR" \
    "$LEADERBOARD/$RES_SUB/$MODEL_DIR"
}
cleanup_listener() {
  local tag=$1 pid cmdline
  local killed=0 waited=0
  while IFS= read -r pid; do
    [ -n "$pid" ] || continue
    [ -r "/proc/$pid/cmdline" ] || continue
    # Port ownership alone is insufficient: never kill a different run's
    # server, even if it appeared after our original free-port probe.
    grep -zFxq "BFCLSTD_RUN_ID=${BFCLSTD_RUN_ID:?}" "/proc/$pid/environ" 2>/dev/null || continue
    cmdline=$(tr '\0' ' ' < "/proc/$pid/cmdline")
    case "$cmdline" in
      *"hub_merged/$tag"*|*vllm*)
        kill "$pid" 2>/dev/null
        killed=1
        ;;
    esac
  done < <(lsof -tiTCP:"$PORT" -sTCP:LISTEN 2>/dev/null | sort -u)
  if [ "$killed" -eq 1 ]; then
    while lsof -tiTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; do
      if [ "$waited" -ge 60 ]; then
        echo "[bfclstd] $tag PORT $PORT STILL BUSY"
        return
      fi
      sleep 1
      waited=$((waited + 1))
    done
  fi
}
run_tag() (
  local tag=$1 adapter merged="" trash out overall
  local attempt incomplete category counts gen_log
  local -a generate_args
  case "$tag" in
    ""|.|..|*/*) echo "[bfclstd] $tag INVALID TAG"; return 1 ;;
  esac
  if [ -n "${BFCLSTD_PREMERGED:-}" ]; then
    merged=$BFCLSTD_PREMERGED
    if [ "$tag" = base ] || [ ! -f "$merged/model.safetensors.index.json" ]; then
      echo "[bfclstd] $tag INVALID PREMERGED CHECKPOINT"
      return 1
    fi
  elif [ "$tag" != base ]; then
    adapter=$PROJ/results/appworld_students/$tag/adapter
    merged=$PROJ/results/appworld_students/$tag/hub_merged
    if [ ! -f "$adapter/model.safetensors" ]; then
      echo "[bfclstd] $tag NO ADAPTER, skip"
      return
    fi
    if [ -e "$merged" ] || [ -L "$merged" ]; then
      trash=$PROJ/_trash/hub_merged_${tag}_$(date +%s)
      while [ -e "$trash" ] || [ -L "$trash" ]; do
        sleep 1
        trash=$PROJ/_trash/hub_merged_${tag}_$(date +%s)
      done
      if ! mv -- "$merged" "$trash"; then
        echo "[bfclstd] $tag MERGE FAILED"
        return
      fi
    fi
    if ! "$PROJ/.venv/bin/python" "$PROJ/tools/bfcl_hub_merge_export.py" \
      --adapter "$adapter" --out "$merged" --verify ${BFCLSTD_BASE_MODEL:+--model "$BFCLSTD_BASE_MODEL"} || \
      [ ! -f "$merged/model.safetensors.index.json" ]; then
      echo "[bfclstd] $tag MERGE FAILED"
      return
    fi
  fi
  cd "$LEADERBOARD" || { echo "[bfclstd] $tag GENERATE FAILED"; return; }
  generate_args=(--model Qwen/Qwen3.5-4B-FC --backend vllm --num-gpus 1
    --temperature "${BFCLSTD_TEMPERATURE:-0.001}"
    --gpu-memory-utilization "${GPU_UTIL:-0.85}"
    --num-threads 8 --result-dir "$RES_SUB")
  if [ "$tag" != base ]; then
    generate_args+=(--local-model-path "$merged")
  fi
  if ! env CUDA_VISIBLE_DEVICES="$GPU" CUDA_DEVICE_ORDER=PCI_BUS_ID \
    VLLM_USE_FLASHINFER_SAMPLER=0 \
    LOCAL_SERVER_PORT="$PORT" PATH="$(dirname "$BFCL"):$PROJ/envs/vllm-serve/.venv/bin:$PATH" \
    "$BFCL" generate "${generate_args[@]}" --test-category all --allow-overwrite \
    > "$PROJ/logs/bfclstd_gen_${tag}.log" 2>&1; then
    echo "[bfclstd] $tag GENERATE FAILED"
    cleanup_harness_outputs
    return
  fi
  gen_log=$PROJ/logs/bfclstd_gen_${tag}.log
  for attempt in 0 1 2; do
    echo "[bfclstd] $tag GEN COMPLETENESS CHECK after $attempt retries" >> "$gen_log"
    if ! incomplete=$("$(dirname "$BFCL")/python" "$PROJ/tools/bfcl_generation_check.py" \
      "$LEADERBOARD" "$LEADERBOARD/$RES_SUB/$MODEL_DIR" "$tag" 2>> "$gen_log"); then
      echo "[bfclstd] $tag GENERATION COMPLETENESS CHECK FAILED (see $gen_log)" | tee -a "$gen_log"
      cleanup_harness_outputs
      return 1
    fi
    [ -n "$incomplete" ] || break
    if [ "$attempt" -eq 2 ]; then
      {
        echo "[bfclstd] $tag GENERATION INCOMPLETE after 2 retries; refusing evaluate:"
        printf '%s\n' "$incomplete"
      } | tee -a "$gen_log"
      cleanup_harness_outputs
      return 1
    fi
    while IFS=$'\t' read -r category counts; do
      echo "[bfclstd] $tag GEN RETRY $((attempt + 1))/2 $counts" | tee -a "$gen_log"
      cleanup_listener "$tag"
      # collect_test_cases resumes missing IDs without --allow-overwrite;
      # existing category results and memory snapshots must stay intact.
      if ! env CUDA_VISIBLE_DEVICES="$GPU" CUDA_DEVICE_ORDER=PCI_BUS_ID \
        VLLM_USE_FLASHINFER_SAMPLER=0 \
        LOCAL_SERVER_PORT="$PORT" PATH="$(dirname "$BFCL"):$PROJ/envs/vllm-serve/.venv/bin:$PATH" \
        "$BFCL" generate "${generate_args[@]}" --test-category "$category" \
        >> "$gen_log" 2>&1; then
        echo "[bfclstd] $tag GEN RETRY $category FAILED; rechecking counts" | tee -a "$gen_log"
      fi
    done <<< "$incomplete"
  done
  if ! "$BFCL" evaluate --model Qwen/Qwen3.5-4B-FC \
    --result-dir "$RES_SUB" --score-dir "$SCORE_SUB" \
    > "$PROJ/logs/bfclstd_eval_${tag}.log" 2>&1; then
    echo "[bfclstd] $tag EVALUATE FAILED"
    cleanup_harness_outputs
    return
  fi
  out=$PROJ/results/bfcl_std/$tag
  mkdir -p "$out"
  if ! cp "$SCORE_SUB/data_overall.csv" "$out/" || \
    ! cp -rT "$SCORE_SUB/$MODEL_DIR" "$out/scoredir"; then
    echo "[bfclstd] $tag SCORE COPY FAILED"
    cleanup_harness_outputs
    return
  fi
  # RTD binds complete generation IDs to the exact checkpoint before admitting
  # aggregate scores. Opt-in retention leaves existing campaign behavior intact.
  if [ "${BFCLSTD_PRESERVE_GENERATION:-0}" = 1 ]; then
    if ! cp -rT "$RES_SUB/$MODEL_DIR" "$out/resultdir"; then
      echo "[bfclstd] $tag GENERATION COPY FAILED"
      return 1
    fi
  fi
  cleanup_harness_outputs
  if [ "$tag" != base ] && [ -z "${BFCLSTD_PREMERGED:-}" ] && [ -f "$out/data_overall.csv" ]; then
    case "$merged" in
      "$PROJ"/results/appworld_students/*/hub_merged) rm -rf -- "$merged" ;;
      *) echo "[bfclstd] $tag REFUSING UNSAFE MERGED CLEANUP" ;;
    esac
  fi
  overall=$(python3 -c 'import csv,sys; print(next(csv.DictReader(open(sys.argv[1], newline="")))["Overall Acc"])' \
    "$out/data_overall.csv" 2>/dev/null)
  echo "[bfclstd] $tag OVERALL=$overall"
)
campaign_status=0
for TAG in "$@"; do
  run_tag "$TAG" || campaign_status=1
  cleanup_listener "$TAG"
done
if [ "$campaign_status" -ne 0 ]; then
  echo "[bfclstd] CAMPAIGN FAILED"
  exit "$campaign_status"
fi
echo "[bfclstd] CAMPAIGN COMPLETE"
