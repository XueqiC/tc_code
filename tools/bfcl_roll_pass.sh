#!/usr/bin/env bash
# Unguided student rollout pass over the BFCL demand split.
#
# Produces result_roll_<kind>_r{0..K-1} / score_roll_<kind>_r{0..K-1} inside the
# leaderboard root, which tools/bfcl_v3_pool.py reads to compute each task's
# empirical success rate (_task_phat) and to emit training rows.
#
# The selective-generation id file is built through the adapter rather than by
# data-file membership: memory ids only resolve once the group is expanded into
# its per-backend categories and the prerequisite write-chain rides along.
#
# That id file lives at one fixed path inside the leaderboard package, so only
# one selective run may be in flight at a time: a concurrent teacher-demo or
# guided pass silently rewrites the selection out from under this one.
#
# Usage: bfcl_roll_pass.sh <base|mt> <gpu> <port> [repeats]
set -u
KIND=${1:-base}
GPU=${2:-4}
PORT=${3:-8951}
REPEATS=${4:-4}
# resolve from this script's own location: hardcoding a home path made
# every hpg run skip evaluation silently ("NO ADAPTER") while still
# printing DONE, because /home/xueqi does not exist there
PROJ=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BFCL_DIR=$PROJ/envs/bfcl
ROOT=$BFCL_DIR/gorilla/berkeley-function-call-leaderboard
STUDENT=Qwen/Qwen3.5-4B
cd "$PROJ"

PYTHONPATH=$PROJ/src .venv/bin/python - <<PY
import json, sys
sys.path.insert(0, "$PROJ/src")
from bfas.adapters.bfcl import BFCLAdapter
split = json.load(open("configs/bfcl_support_split.json"))
adapter = BFCLAdapter()
demand = list(split["demand"])
if "$KIND" == "mt":
    # the mt pass exists to record message streams for the stateful categories;
    # single-turn tasks carry no extra turns, so running them again buys nothing
    cats = adapter.task_categories()
    demand = [
        t for t in demand
        if cats[t].startswith(("multi_turn", "web_search", "memory"))
    ]
by_cat = adapter._selective_file(demand)
json.dump(by_cat, open("$ROOT/test_case_ids_to_generate.json", "w"), indent=1)
n = sum(len(v) for v in by_cat.values())
print("[bfclroll] ids file:", n, "entries in", len(by_cat), "categories")
PY

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=$GPU
export VLLM_USE_FLASHINFER_SAMPLER=0
"$PROJ/envs/vllm-serve/.venv/bin/vllm" serve "$STUDENT" \
  --served-model-name "$STUDENT" --port "$PORT" \
  --gpu-memory-utilization "${GPU_UTIL:-0.85}" --max-model-len 32768 \
  > "$PROJ/logs/bfcl_roll_${KIND}_vllm.log" 2>&1 &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null' EXIT

for _ in $(seq 1 300); do
  curl -s -m 3 "http://localhost:$PORT/v1/models" > /dev/null && break
  sleep 5
done
curl -s -m 3 "http://localhost:$PORT/v1/models" > /dev/null || {
  echo "[bfclroll] server never came up; see logs/bfcl_roll_${KIND}_vllm.log"; exit 1; }
echo "[bfclroll] server up on $PORT (gpu $GPU)"

# A repeat counts as done only when every id of the CURRENT selection has a
# result in it; a non-empty directory can just as easily be a partial run that
# was interrupted, or leftovers from an earlier split.
have_all_selected() {
  PYTHONPATH=$PROJ/src "$PROJ/.venv/bin/python" - "$1" "$ROOT/test_case_ids_to_generate.json" <<'PYCHK'
import json, sys
from pathlib import Path
target, selection = Path(sys.argv[1]), Path(sys.argv[2])
wanted = {i for ids in json.load(selection.open()).values() for i in ids}
have = set()
if target.is_dir():
    for f in target.rglob("*_result.json"):
        for line in f.open():
            if line.strip():
                have.add(json.loads(line)["id"])
sys.exit(0 if wanted <= have else 1)
PYCHK
}
cd "$BFCL_DIR"; . .venv/bin/activate
export LOCAL_SERVER_ENDPOINT=localhost LOCAL_SERVER_PORT=$PORT
# the mt pass records the full evaluation-time message stream so the pool
# builder can emit one training row per assistant turn; the handler treats
# BFCL_DUMP_MESSAGES as the destination path, so it is set per repeat
DUMP_DIR=$PROJ/data/bfcl_dumps
[ "$KIND" = "mt" ] && mkdir -p "$DUMP_DIR"

for R in $(seq 0 $((REPEATS - 1))); do
  if (cd "$PROJ" && have_all_selected "$ROOT/result_roll_${KIND}_r${R}"); then
    echo "[bfclroll] $KIND r$R already covers the selection, skipping"
    continue
  fi
  if [ "$KIND" = "mt" ]; then
    export BFCL_DUMP_MESSAGES="$DUMP_DIR/roll_dump_r${R}.jsonl"
    : > "$BFCL_DUMP_MESSAGES"
  fi
  bfcl generate --model "$STUDENT-FC" --run-ids --skip-server-setup \
    --temperature 0.7 --num-threads 4 \
    --result-dir "result_roll_${KIND}_r${R}" \
    > "$PROJ/logs/bfcl_roll_${KIND}_r${R}_gen.log" 2>&1
  bfcl evaluate --model "$STUDENT-FC" \
    --result-dir "result_roll_${KIND}_r${R}" \
    --score-dir "score_roll_${KIND}_r${R}" --partial-eval \
    > "$PROJ/logs/bfcl_roll_${KIND}_r${R}_eval.log" 2>&1
  echo "[bfclroll] $KIND r$R done"
done
echo "[bfclroll] PASS COMPLETE $KIND"
