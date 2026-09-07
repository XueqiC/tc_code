#!/usr/bin/env bash
# Fast proxy evaluation for iterating on the mechanism.
#
# The full official flow is 5217 entries and roughly three hours, so a single
# design change costs half a day to judge. This runs the same harness over the
# fixed proxy subset in configs/bfcl_proxy_ids.json (~500 entries, equal weight
# per category, support tasks excluded), which turns an iteration into minutes.
#
# The number it prints is a PROXY, comparable across runs of this script and
# nothing else. Reported figures still come from tools/bfcl_std_campaign.sh.
#
# Measured against the full run on the base model (20m34s here vs ~3h there):
#   Multi Turn  proxy 50.00  full 51.12   usable
#   Live        proxy 76.71  full 77.65   usable
#   Non-Live    proxy 75.42  full 79.67   usable, biased low
#   Memory      proxy  0.00  full 24.95   NOT USABLE -- 1 task per backend
#   Web Search  proxy   N/A  full 10.00   NOT USABLE -- too few entries
# So this decides the balance between Multi Turn and the single-turn axes, which
# is what the preservation weight trades off. Memory and Web Search verdicts must
# come from the full run.
#
# Usage: bfcl_fast_eval.sh <GPU> <PORT> <tag> [tag2 ...]
set -u
if [ "$#" -lt 3 ]; then
  echo "Usage: bash tools/bfcl_fast_eval.sh <GPU> <PORT> <tag> [tag2 ...]" >&2
  exit 2
fi
GPU=$1 PORT=$2
shift 2
# resolve from this script's own location: hardcoding a home path made
# every hpg run skip evaluation silently ("NO ADAPTER") while still
# printing DONE, because /home/xueqi does not exist there
PROJ=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
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
RES_SUB=result_fast_p$PORT
SCORE_SUB=score_fast_p$PORT
SELECTION=$LEADERBOARD/test_case_ids_to_generate.json

for tag in "$@"; do
  echo "[fast] === $tag ==="
  merged=""
  if [ "$tag" != base ]; then
    adapter=$PROJ/results/appworld_students/$tag/adapter
    merged=$PROJ/results/appworld_students/$tag/hub_merged
    if [ ! -f "$adapter/model.safetensors" ]; then
      echo "[fast] $tag NO ADAPTER, skip"; continue
    fi
    # reuse an existing merge; merging is minutes and ~8G of writes
    if [ ! -f "$merged/model.safetensors.index.json" ]; then
      rm -rf -- "$merged"
      if ! "$PROJ/.venv/bin/python" "$PROJ/tools/bfcl_hub_merge_export.py" \
        --adapter "$adapter" --out "$merged" --verify \
        > "$PROJ/logs/bfclfast_merge_${tag}.log" 2>&1; then
        echo "[fast] $tag MERGE FAILED"; continue
      fi
    fi
  fi

  cp "$PROJ/configs/bfcl_proxy_ids.json" "$SELECTION"
  rm -rf -- "$LEADERBOARD/$RES_SUB" "$LEADERBOARD/$SCORE_SUB"
  cd "$LEADERBOARD" || { echo "[fast] $tag GENERATE FAILED"; continue; }
  args=(--model Qwen/Qwen3.5-4B-FC --backend vllm --num-gpus 1
    --gpu-memory-utilization "${GPU_UTIL:-0.85}" --run-ids
    --num-threads 8 --result-dir "$RES_SUB" --allow-overwrite)
  [ "$tag" != base ] && args+=(--local-model-path "$merged")
  if ! env CUDA_VISIBLE_DEVICES="$GPU" CUDA_DEVICE_ORDER=PCI_BUS_ID \
    VLLM_USE_FLASHINFER_SAMPLER=0 LOCAL_SERVER_PORT="$PORT" \
    PATH="$(dirname "$BFCL"):$PROJ/envs/vllm-serve/.venv/bin:$PATH" \
    "$BFCL" generate "${args[@]}" \
    > "$PROJ/logs/bfclfast_gen_${tag}.log" 2>&1; then
    echo "[fast] $tag GENERATE FAILED (see logs/bfclfast_gen_${tag}.log)"
    rm -f "$SELECTION"; continue
  fi
  "$BFCL" evaluate --model Qwen/Qwen3.5-4B-FC --result-dir "$RES_SUB" \
    --score-dir "$SCORE_SUB" --partial-eval \
    > "$PROJ/logs/bfclfast_eval_${tag}.log" 2>&1 || true
  rm -f "$SELECTION"

  out=$PROJ/results/bfcl_fast/$tag
  mkdir -p "$out"
  cp "$SCORE_SUB/data_overall.csv" "$out/" 2>/dev/null
  "$PROJ/.venv/bin/python" "$PROJ/tools/bfcl_print_scores.py" \
    "$out/data_overall.csv" "$tag" --prefix '[fast]' --proxy
done
echo "[fast] DONE"
