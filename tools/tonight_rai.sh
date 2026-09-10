#!/usr/bin/env bash
# Tonight's rai queue, run strictly one at a time so the box stays shareable.
#
# The question this answers: the deficit-only weighting buys what the student
# lacks and holds nothing back for what it already does, which is why pool v6
# went to 97% stateful mass. AW_PRESERVE adds (preserve * p-hat) to each row's
# weight, making preservation explicit. The sweep asks the proxy where the two
# families balance -- the value is measured, not chosen.
set -u
# resolve from this script's own location: hardcoding a home path made
# every hpg run skip evaluation silently ("NO ADAPTER") while still
# printing DONE, because /home/xueqi does not exist there
PROJ=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
GPU=${1:-4}
PORT=${2:-8985}
cd "$PROJ"

# A 4B at a 4096-token cap needs more than a 48GB card: the 47GB devices OOM
# part-way through the first epoch. Fail here with a clear reason instead of
# after several minutes of loading.
FREE_MB=$(nvidia-smi --id="$GPU" --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null || echo 0)
if [ "${FREE_MB:-0}" -lt 80000 ]; then
  echo "[tonight] GPU $GPU has ${FREE_MB}MB total; this training needs a ~97GB card (1 or 4)."
  exit 2
fi

for P in 0.25 0.50 1.00; do
  TAG="bfclb7_p${P/./}"
  echo "[tonight] === train $TAG (AW_PRESERVE=$P) ==="
  AW_POOL_PATH=$PWD/data/bfcl_sft/pool_bfcl_v6.jsonl AW_V3=1 \
    AW_PRESERVE=$P AW_MAX_PROMPT_TOKENS=4096 AW_TRUNCATE_SIDE=tail \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU \
    .venv/bin/python src/appworld_train.py --selection full --budget 999999 \
    --seed 0 --student Qwen/Qwen3.5-4B --tag "$TAG" \
    > "logs/${TAG}_train.log" 2>&1 || { echo "[tonight] $TAG TRAIN FAILED"; continue; }
  echo "[tonight] === fast eval $TAG ==="
  GPU_UTIL=0.80 bash tools/bfcl_fast_eval.sh "$GPU" "$PORT" "$TAG" \
    2>&1 | grep -E "PROXY|FAILED|NO SCORE"
done
echo "[tonight] RAI QUEUE DONE"
