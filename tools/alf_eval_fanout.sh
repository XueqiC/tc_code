#!/usr/bin/env bash
# Fan one ALFWorld campaign across shards on the two Blackwell cards.
# Only GPU1 and GPU4 match the binding's hardware class on rai; the Ada and A100
# cards are a different class and the harness refuses them by design.
set -euo pipefail
WT="$(cd "$(dirname "$0")/.." && pwd)"
BINDING="${1:?usage: alf_eval_fanout.sh <binding.json> <output-root> <tag> [shards-per-gpu]}"
OUT="${2:?}"; TAG="${3:?}"; PER="${4:-3}"
GPUS=(1 4)
mkdir -p "$WT/logs"
i=0
for g in "${GPUS[@]}"; do
  for ((k=0; k<PER; k++)); do
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$g" \
    PYTHONPATH="$WT/src:$WT" PYTHONDONTWRITEBYTECODE=1 \
    nohup "$WT/.venv/bin/python" -B "$WT/tools/alf_eval_shard.py" \
      --root "$WT" --binding "$BINDING" --output-root "$OUT" --tag "$TAG" \
      --shard "$i" --of $(( ${#GPUS[@]} * PER )) --device cuda:0 \
      > "$WT/logs/shard_${TAG}_${i}_gpu${g}.log" 2>&1 &
    echo "shard $i -> GPU$g pid $!"
    i=$((i+1))
  done
done
