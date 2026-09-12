#!/usr/bin/env bash
# Same-machine CC comparison. CC_ARMS='A B C' (default), B, or A,C.
# A: flattened logistic DDPO; B: joint correct pairs; C: joint shuffled pairs.
# Usage (one visible GPU): bash tools/cc_three_arms.sh [port=job-derived or 9160]
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
cd -- "$ROOT"
[[ $# -le 1 ]] || { echo "Usage: $0 [port]" >&2; exit 2; }
source tools/aw_hpg_common.sh
if [[ -n ${SLURM_JOB_ID:-} ]]; then
  aw_hpg_require_gpu
  aw_hpg_environment
  PORT=$(aw_hpg_port "${1:-}")
else
  PORT=$(aw_hpg_port "${1:-9160}")
fi
[[ -n ${CUDA_VISIBLE_DEVICES:-} && $CUDA_VISIBLE_DEVICES != *,* && $CUDA_VISIBLE_DEVICES != -1 ]] || {
  echo "Set CUDA_VISIBLE_DEVICES to exactly one GPU; all arms train and evaluate there" >&2; exit 2;
}
PYTHON=${CC_PYTHON:-$ROOT/.venv/bin/python}
STUDENT=${CC_STUDENT:-Qwen/Qwen3.5-4B}
SEED=${CC_SEED:-0}
PREFIX=${CC_TAG_PREFIX:-cc_v1}
[[ $PREFIX =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ && $SEED =~ ^[0-9]+$ ]] || {
  echo "Invalid CC_TAG_PREFIX or CC_SEED" >&2; exit 2;
}
selection=${CC_ARMS-A B C}
selection=${selection//,/ }
read -r -a requested <<< "$selection"
[[ ${#requested[@]} -gt 0 ]] || { echo "CC_ARMS must select A, B and/or C" >&2; exit 2; }
declare -A wanted=()
for arm in "${requested[@]}"; do
  case $arm in
    A|a) wanted[A]=1 ;;
    B|b) wanted[B]=1 ;;
    C|c) wanted[C]=1 ;;
    *) echo "Unknown CC_ARMS entry: $arm" >&2; exit 2 ;;
  esac
done
mkdir -p logs
exec 9>logs/cc_three_arms.lock
flock -n 9 || { echo "CC runner already running on this checkout" >&2; exit 3; }
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export AW_CC_PAIRS_PATH=${CC_PAIRS_PATH:-$ROOT/data/cc_pairs_v1/confused_pairs.json}
export AW_CC_SPLIT_PATH=${CC_SPLIT_PATH:-$(dirname -- "$AW_CC_PAIRS_PATH")/split.json}
export AW_CC_ANCHOR_PATH=${CC_ANCHOR_PATH:-$ROOT/data/bfcl_sft/anchors_single_base.jsonl}
# Pin every recipe control that could otherwise leak from another AW campaign.
export AW_DDPO_LR=${CC_LR:-5e-6} AW_DDPO_BETA=${CC_BETA:-0.1}
export AW_DDPO_EPOCHS=${CC_EPOCHS:-1} AW_PAIR_STEPS=${CC_STEPS:-0}
export AW_PAIR_GAMMA=${CC_GAMMA:-1.0} AW_PAIR_REDUCTION=${CC_REDUCTION:-worst}
export AW_PAIR_SHUFFLE_SEED=${CC_SHUFFLE_SEED:-0} AW_PAIR_BATCH_UNITS=${CC_BATCH_UNITS:-4}
export AW_DDPO_WEIGHT=none AW_DDPO_REF_FREE=0 AW_DDPO_SPAN_ONLY=0 AW_ANCHOR_ADAPTIVE=0
export AW_SFT_SKIP=1 AW_GRAD_CKPT=1 AW_MAX_PROMPT_TOKENS=4096 AW_TRUNCATE_SIDE=tail
export AW_EPOCHS=1 AW_LR=$AW_DDPO_LR AW_PRESERVE=0 AW_LAMBDA_PREF=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 BFCLSTD_BASE_MODEL=$STUDENT
unset AW_DDPO_WEIGHT_PERMUTE AW_TEACHER AW_V3 AW_UNIORPO AW_TOKSEL AW_CURRICULUM
unset AW_ANCHOR_TRACE AW_DDPO_TAU AW_DDPO_PERM_SEED
export AW_DISTILL=pair_unit AW_PAIR_PAIRING=correct
# CPU validation only; no teacher requests or synthetic replacement sides.
"$PYTHON" - "${wanted[C]:-0}" <<'PY'
import json, os, sys
from pathlib import Path
import appworld_train as trainer
from bfas.pair_unit import configuration, load_pairs, load_anchors, step_schedule
from bfas.arms import same_seed_pair_counts, shuffle_pair_sides
import numpy as np

pairs = load_pairs(os.environ['AW_CC_PAIRS_PATH'], os.environ['AW_CC_SPLIT_PATH'])
anchors = load_anchors(os.environ['AW_CC_ANCHOR_PATH'], trainer)
types = {p['type'] for p in pairs}
if len(pairs) < int(os.environ.get('CC_MIN_PAIRS', '20')) or len(types) < int(os.environ.get('CC_MIN_TYPES', '2')):
    raise SystemExit(f'CC gate: {len(pairs)} train pairs, {len(types)} types; need >=20 pairs and >=2 types')
if sys.argv[1] == '1':
    shuffled = shuffle_pair_sides(pairs, seed=int(os.environ['AW_PAIR_SHUFFLE_SEED']))
    print('[cc] arm C shuffle_same_seed_by_type=' +
          json.dumps(same_seed_pair_counts(shuffled), sort_keys=True), flush=True)
steps = len(list(step_schedule(len(pairs) + len(anchors), configuration(), np.random.default_rng(0))))
print(f'[cc] train pairs={len(pairs)} types={len(types)} anchors={len(anchors)} steps={steps}', flush=True)
PY
tags=()
for arm in A B C; do
  [[ ${wanted[$arm]:-0} == 1 ]] || continue
  tag=${PREFIX}_${arm}_s${SEED}
  [[ ! -e results/appworld_students/$tag && ! -e results/bfcl_std/$tag ]] || {
    echo "Output already exists for $tag; choose a new CC_TAG_PREFIX" >&2; exit 2;
  }
  tags+=("$tag")
done
# Complete all selected training first, then check exposure before official eval.
for arm in A B C; do
  [[ ${wanted[$arm]:-0} == 1 ]] || continue
  tag=${PREFIX}_${arm}_s${SEED}
  export AW_DISTILL=pair_unit AW_PAIR_PAIRING=correct
  case $arm in
    A) export AW_DISTILL=ddpo ;;
    C) export AW_PAIR_PAIRING=shuffled ;;
  esac
  "$PYTHON" src/appworld_train.py --selection full --budget 999999 \
    --seed "$SEED" --student "$STUDENT" --tag "$tag" 2>&1 | tee "logs/${tag}_train.log"
done
"$PYTHON" - "${tags[@]}" <<'PY'
import json, sys
from pathlib import Path

manifests = [json.loads((Path('results/appworld_students') / tag / 'cc_manifest.json').read_text()) for tag in sys.argv[1:]]
keys = ('student', 'seed', 'hostname', 'exposure_sha256', 'optimizer_steps', 'tokens', 'reference_tokens',
        'anchor_adaptive', 'anchor_source', 'retention_environment', 'prompt_truncation', 'max_response_tokens')
for tag, manifest in zip(sys.argv[2:], manifests[1:]):
    for key in keys:
        if manifest[key] != manifests[0][key]:
            raise SystemExit(f'{tag}: unmatched {key}')
    for key in ('learning_rate', 'beta', 'epochs', 'steps', 'units_per_step', 'gamma', 'reduction', 'shuffle_seed'):
        if manifest['config'][key] != manifests[0]['config'][key]:
            raise SystemExit(f'{tag}: unmatched {key}')
print('[cc] matched settings, side/anchor exposure, optimizer steps, actual training tokens and machine')
PY
for tag in "${tags[@]}"; do
  [[ -f results/appworld_students/$tag/adapter/model.safetensors ]] || {
    echo "Missing campaign adapter for $tag" >&2; exit 1;
  }
  eval_log=logs/${tag}_official.log
  bash tools/bfcl_std_campaign.sh "$CUDA_VISIBLE_DEVICES" "$PORT" "$tag" 2>&1 | tee "$eval_log"
  # The existing campaign can return zero on a failed tag. Require this run's marker and score.
  if ! grep -Eq "^\[bfclstd\] $tag OVERALL=.+" "$eval_log" || \
     [[ ! -s results/bfcl_std/$tag/data_overall.csv ]] || \
     [[ ! -d results/bfcl_std/$tag/scoredir ]]; then
    echo "Official BFCL v4 full evaluation failed for $tag" >&2; exit 1;
  fi
done
echo "[cc] COMPLETE ${tags[*]}"
