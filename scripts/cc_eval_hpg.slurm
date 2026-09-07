#!/usr/bin/env bash
#SBATCH --job-name=cc_eval
#SBATCH --account=fsu-compsci-dept
#SBATCH --qos=fsu-compsci-dept
#SBATCH --partition=hpg-b200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=200gb
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x_%j.out
# Submit from the checkout with logs/ already present:
#   sbatch scripts/cc_eval_hpg.slurm cc_v1_A_s0 cc_v1_B_s0 cc_v1_C_s0
# Or set CC_EVAL_TAGS='tag1 tag2'. Each evaluation has a hard 30-minute limit.
# Request 30 minutes per tag plus merge time, e.g. sbatch --time=02:00:00 ...
# Local fallback: CUDA_VISIBLE_DEVICES=0 bash scripts/cc_eval_hpg.slurm TAG
set -euo pipefail
if [[ -v SLURM_SUBMIT_DIR ]]; then
    # SLURM executes a spooled copy, so BASH_SOURCE is not the checkout path.
    CC_ROOT=$(cd -- "${SLURM_SUBMIT_DIR:?}" && pwd)
else
    CC_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
fi
if [[ ! -f "$CC_ROOT/tools/cc_pairs.py" || ! -d "$CC_ROOT/data" ]]; then
    echo "Invalid CC_ROOT '$CC_ROOT': expected tools/cc_pairs.py and data/. Submit from the tc-alignment checkout." >&2
    exit 1
fi
cd "$CC_ROOT"
CC_PYTHON=${CC_PYTHON:-.venv/bin/python}
CC_SEED=${CC_SEED:-0}
CC_PAIRS_PATH=${CC_PAIRS_PATH:-data/cc_pairs_v1_stage1/confused_pairs.json}
CC_SPLIT_PATH=${CC_SPLIT_PATH:-$(dirname -- "$CC_PAIRS_PATH")/split.json}
CC_BASE_RESULTS=${CC_BASE_RESULTS:-data/cc_pairs_v1/probe_results.jsonl}
CC_EVAL_OUT=${CC_EVAL_OUT:-results/cc_eval}
export PYTHONPATH="$CC_ROOT/src:$CC_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false CUBLAS_WORKSPACE_CONFIG=:4096:8
if [[ -n ${SLURM_JOB_ID:-} ]]; then
    # Keep the probe launcher's FlashInfer/ninja PATH convention.
    CC_VLLM_VENV="$CC_ROOT/envs/vllm/.venv"
    if [[ -d "$CC_ROOT/envs/vllm-serve/.venv" ]]; then
        CC_VLLM_VENV=$(realpath -e -- "$CC_ROOT/envs/vllm-serve/.venv")
    fi
    export PATH="$CC_VLLM_VENV/bin:$CC_ROOT/.venv/bin:$PATH"
    export HF_HOME=${HF_HOME:-/blue/fsu-compsci-dept/xc25.fsu/hq/tools/hf-cache}
    export VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-/blue/fsu-compsci-dept/xc25.fsu/hq/tools/vllm-cache}
fi
if [[ -z ${BFCL_VENV_PYTHON:-} ]]; then
    if [[ -x envs/bfcl-venv/bin/python ]]; then
        export BFCL_VENV_PYTHON="$CC_ROOT/envs/bfcl-venv/bin/python"
    else
        export BFCL_VENV_PYTHON="$CC_ROOT/envs/bfcl/.venv/bin/python"
    fi
fi
if [[ $# -gt 0 ]]; then
    tags=("$@")
else
    selection=${CC_EVAL_TAGS:-cc_v1_A_s0 cc_v1_B_s0 cc_v1_C_s0}
    selection=${selection//,/ }
    read -r -a tags <<< "$selection"
fi
[[ ${#tags[@]} -gt 0 ]] || { echo 'No evaluation tags supplied' >&2; exit 2; }
models=()
adapters=()
for tag in "${tags[@]}"; do
    [[ $tag =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || { echo "Invalid tag: $tag" >&2; exit 2; }
    model=
    adapter=
    # Official eval deletes hub_merged; adapter/ may only be a training export
    # without config.json. Prefer existing native HF checkpoints in either tree.
    candidates=("results/appworld_students/$tag/hub_merged"
                "results/bfcl_students/$tag/hub_merged"
                "results/appworld_students/$tag/adapter"
                "results/bfcl_students/$tag/adapter")
    for candidate in "${candidates[@]}"; do
        if [[ -f "$candidate/config.json" ]]; then
            model=$candidate
            break
        fi
    done
    if [[ -z $model ]]; then
        for candidate in "results/appworld_students/$tag/adapter" \
                         "results/bfcl_students/$tag/adapter"; do
            if [[ -f "$candidate/model.safetensors" ]]; then
                adapter=$candidate
                break
            fi
        done
        [[ -n $adapter ]] || {
            echo "Missing merged HF checkpoint (config.json) or mergeable adapter (model.safetensors) for $tag. Tried directories: ${candidates[*]}" >&2
            exit 2
        }
    fi
    models+=("$model")
    adapters+=("$adapter")
done
tmp_model=
trash_tmp_model() {
    if [[ -n $tmp_model && ( -e $tmp_model || -L $tmp_model ) ]]; then
        local trash
        mkdir -p "$CC_ROOT/_trash"
        trash=$(mktemp -d "$CC_ROOT/_trash/cc_eval_${tag}_XXXXXX")
        mv -- "$tmp_model" "$trash/"
        echo "[cc eval] moved temporary checkpoint to $trash/hub_merged_tmp"
    fi
}
# Also archive partial exports and checkpoints from failed/timed-out evaluations.
trap trash_tmp_model EXIT
mkdir -p logs
for i in "${!tags[@]}"; do
    tag=${tags[$i]}
    model=${models[$i]}
    if [[ -z $model ]]; then
        adapter=${adapters[$i]}
        tmp_model="$CC_EVAL_OUT/$tag/hub_merged_tmp"
        trash_tmp_model  # Preserve stale output before exporting into an empty directory.
        echo "[cc eval] $tag merging $adapter into $tmp_model"
        "$CC_PYTHON" tools/bfcl_hub_merge_export.py \
            --adapter "$adapter" --out "$tmp_model" --model Qwen/Qwen3.5-4B --verify \
            2>&1 | tee "logs/${tag}_cc_merge_${SLURM_JOB_ID:-local}.log"
        [[ -f "$tmp_model/config.json" ]] || {
            echo "Merge produced no config.json for $tag. Tried directories: $tmp_model" >&2
            exit 2
        }
        model=$tmp_model
    fi
    echo "[cc eval] $tag model=$model (30-minute limit)"
    # Native generation and probe-native CE share this process/GPU. timeout
    # also terminates checker children; no background model server is left.
    timeout --signal=TERM --kill-after=15s 30m \
        "$CC_PYTHON" tools/cc_pairs.py evaluate --model "$model" --tag "$tag" \
        --pairs "$CC_PAIRS_PATH" --split "$CC_SPLIT_PATH" --base-results "$CC_BASE_RESULTS" \
        --seed "$CC_SEED" --out "$CC_EVAL_OUT/$tag" \
        2>&1 | tee "logs/${tag}_cc_eval_${SLURM_JOB_ID:-local}.log"
    trash_tmp_model
    tmp_model=
done
echo "[cc eval] COMPLETE ${tags[*]}"
