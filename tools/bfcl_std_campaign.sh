#!/usr/bin/env bash
set -u
if [ "$#" -lt 3 ]; then
  echo "Usage: bash tools/bfcl_std_campaign.sh <GPU> <PORT> <tag1> [tag2 ...]" >&2
  exit 2
fi
GPU=$1 PORT=$2
shift 2
PROJ=/home/xueqi/hq/projects/tc-alignment
LEADERBOARD=$PROJ/envs/bfcl/gorilla/berkeley-function-call-leaderboard
BFCL=$PROJ/envs/bfcl/.venv/bin/bfcl
RES_SUB=result_p$PORT SCORE_SUB=score_p$PORT
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
  local -a generate_args
  case "$tag" in
    ""|*/*) echo "[bfclstd] $tag INVALID TAG"; return ;;
  esac
  if [ "$tag" != base ]; then
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
      --adapter "$adapter" --out "$merged" --verify || \
      [ ! -f "$merged/model.safetensors.index.json" ]; then
      echo "[bfclstd] $tag MERGE FAILED"
      return
    fi
  fi
  cd "$LEADERBOARD" || { echo "[bfclstd] $tag GENERATE FAILED"; return; }
  generate_args=(--model Qwen/Qwen3.5-4B-FC --backend vllm --num-gpus 1
    --gpu-memory-utilization "${GPU_UTIL:-0.85}" --test-category all
    --num-threads 8 --result-dir "$RES_SUB" --allow-overwrite)
  if [ "$tag" != base ]; then
    generate_args+=(--local-model-path "$merged")
  fi
  if ! env CUDA_VISIBLE_DEVICES="$GPU" CUDA_DEVICE_ORDER=PCI_BUS_ID \
    VLLM_USE_FLASHINFER_SAMPLER=0 \
    LOCAL_SERVER_PORT="$PORT" PATH="$PROJ/envs/vllm-serve/.venv/bin:$PATH" \
    "$BFCL" generate "${generate_args[@]}" \
    > "$PROJ/logs/bfclstd_gen_${tag}.log" 2>&1; then
    echo "[bfclstd] $tag GENERATE FAILED"
    cleanup_harness_outputs
    return
  fi
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
  cleanup_harness_outputs
  if [ "$tag" != base ] && [ -f "$out/data_overall.csv" ]; then
    case "$merged" in
      "$PROJ"/results/appworld_students/*/hub_merged) rm -rf -- "$merged" ;;
      *) echo "[bfclstd] $tag REFUSING UNSAFE MERGED CLEANUP" ;;
    esac
  fi
  overall=$(python3 -c 'import csv,sys; print(next(csv.DictReader(open(sys.argv[1], newline="")))["Overall Acc"])' \
    "$out/data_overall.csv" 2>/dev/null)
  echo "[bfclstd] $tag OVERALL=$overall"
)
for TAG in "$@"; do
  run_tag "$TAG"
  cleanup_listener "$TAG"
done
echo "[bfclstd] CAMPAIGN COMPLETE"
