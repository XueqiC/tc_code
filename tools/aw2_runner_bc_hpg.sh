#!/bin/bash
# AW-2 exposure vs coverage, sequential on one GPU; official dev57 via AW-1.
# Usage inside a SLURM allocation: bash tools/aw2_runner_bc_hpg.sh [port=job-derived]
# AW2_ARMS='B C' (default), 'B', 'C', or comma-separated/full arm tags.
# B can run before mining C: AW2_ARMS=B bash tools/aw2_runner_bc_hpg.sh
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
[[ $(pwd -P) == "$ROOT" ]] || { echo "Run this script from the project root: $ROOT" >&2; exit 2; }
[[ $# -le 1 ]] || {
  echo "Usage: bash tools/aw2_runner_bc_hpg.sh [port=job-derived]" >&2; exit 2;
}
source tools/aw_hpg_common.sh
PORT=$(aw_hpg_port "${1:-}")
aw_hpg_require_gpu
aw_hpg_environment
selection=${AW2_ARMS-B C}
selection=${selection//,/ }
read -r -a requested <<< "$selection"
[[ ${#requested[@]} -gt 0 ]] || { echo "AW2_ARMS must select B and/or C" >&2; exit 2; }
run_b=0
run_c=0
for arm in "${requested[@]}"; do
  case $arm in
    B|b|aw2_pref_spread_e2_s0) run_b=1 ;;
    C|c|aw2_pref_spread_x2_s0) run_c=1 ;;
    *) echo "Unknown AW2_ARMS entry: $arm" >&2; exit 2 ;;
  esac
done
mkdir -p logs
exec 9>logs/aw2_runner_bc.lock
flock -n 9 || { echo "AW-2 B/C runner is already running" >&2; exit 3; }
exec > >(tee -a logs/aw2_runner_bc.log) 2>&1
# Pin the A settings, including the DDPO controls that can otherwise leak in
# from another experiment's environment. AW-1 supplies seed 0 and dev config.
export AW_GRAD_CKPT=1 AW_DISTILL=ddpo AW_SFT_SKIP=1 AW_DDPO_LR=5e-6
export AW_DDPO_BETA=0.1 AW_DDPO_WEIGHT=none AW_DDPO_REF_FREE=0 AW_DDPO_SPAN_ONLY=0
export AW_ANCHOR_ADAPTIVE=0
unset AW_DDPO_WEIGHT_PERMUTE
for arm in B C; do
  case $arm in
    B)
      (( run_b )) || continue
      TAG=aw2_pref_spread_e2_s0
      POOL=data/appworld_events/pool_aw2_spread_matched.jsonl
      export AW_DDPO_EPOCHS=2 ;;
    C)
      (( run_c )) || continue
      TAG=aw2_pref_spread_x2_s0
      POOL=data/appworld_events/pool_aw2_spread_x2.jsonl
      export AW_DDPO_EPOCHS=1 ;;
  esac
  echo "[aw2-runner] === $TAG (ddpo on $POOL; arm=$arm epochs=$AW_DDPO_EPOCHS) $(date '+%F %T')"
  [[ -s $POOL ]] || { echo "[aw2-runner] missing pool: $POOL"; exit 2; }
  status=0
  ARM_LOG=logs/aw2_runner_bc_${TAG}.log
  bash tools/aw1_train_eval_hpg.sh "$TAG" "$POOL" ddpo slurm "$PORT" 2>&1 | tee "$ARM_LOG" || status=$?
  # Require this invocation's success marker too, not just a report in an old log.
  EVAL_LOG=logs/awoff_${TAG}_dev.log
  if (( status == 0 )); then
    if ! grep -aFq "[aw1] $TAG dev run exit=0" "$ARM_LOG" ||
       ! grep -aqE '^[[:space:]]*aggregate[[:space:]]*\|' "$EVAL_LOG"; then
      echo "[aw2-runner] missing successful official evaluation/aggregate for $TAG"
      status=1
    fi
  fi
  echo "[aw2-runner] === $TAG exit=$status $(date '+%F %T')"
  (( status == 0 )) || exit "$status"
done
echo "[aw2-runner] ALL DONE $(date '+%F %T')"
