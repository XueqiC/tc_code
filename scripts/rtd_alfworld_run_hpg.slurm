#!/usr/bin/env bash
#SBATCH --job-name=rtd_alfworld_v1
#SBATCH --account=fsu-compsci-dept
#SBATCH --qos=fsu-compsci-dept
#SBATCH --partition=hpg-b200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=results/c26f/hpg/%x_%j.out
set -euo pipefail
cd -- "${SLURM_SUBMIT_DIR:?Submit from the project root}"
ROOT=$(pwd -P)
source tools/aw_hpg_common.sh
aw_hpg_require_gpu
aw_hpg_environment
# The shared helper defaults caches to envs/.cache. This checkout has read-only
# asset symlinks, so all newly generated caches belong to its results/ tree.
export XDG_CACHE_HOME="$ROOT/results/rtd_alfworld_cache"
export APPWORLD_CACHE="$XDG_CACHE_HOME/appworld" VLLM_CACHE_ROOT="$XDG_CACHE_HOME/vllm"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton" TORCH_HOME="$XDG_CACHE_HOME/torch"
export TORCHINDUCTOR_CACHE_DIR="$XDG_CACHE_HOME/torchinductor" CUDA_CACHE_PATH="$XDG_CACHE_HOME/nv"
export NUMBA_CACHE_DIR="$XDG_CACHE_HOME/numba" IPYTHONDIR="$XDG_CACHE_HOME/ipython"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
export TMPDIR="$ROOT/results/rtd_alfworld_tmp/${SLURM_JOB_ID}"
mkdir -p -- "$XDG_CACHE_HOME" "$TMPDIR"
case ${RTD_ARM:?Set RTD_ARM to R0 or R1} in R0|R1) ;; *) echo 'RTD_ARM must be R0 or R1' >&2; exit 2 ;; esac
RTD_CONFIG=${RTD_CONFIG:-configs/rtd/v1_alfworld_c26.yaml}
export PYTHONPATH="$ROOT/src:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export ALFWORLD_DATA="$ROOT/envs/alfworld/data"
export ALFWORLD_DATA_ROOT="$ROOT/envs/alfworld/data"
export ALFRED_DATA="$ROOT/envs/alfworld/data/json_2.1.1"
export BFAS_ALFWORLD_MAX_STEPS=40 BFAS_ALFWORLD_STUDENT_REACT=1
export OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
RTD_COMMAND=${RTD_COMMAND:-run}
case $RTD_COMMAND in run|resume|smoke) ;; *) echo 'RTD_COMMAND must be run, resume or smoke' >&2; exit 2 ;; esac
args=("$RTD_COMMAND" --arm "$RTD_ARM" --config "$RTD_CONFIG")
if [[ -n ${RTD_RUN_DIR:-} ]]; then args+=(--run-dir "$RTD_RUN_DIR"); fi
# Optional whitespace-separated flags, preserving literal characters.
IFS=$' \t\n' read -r -d '' -a extra_args < <(printf '%s\0' "${RTD_EXTRA_ARGS:-}")
args+=("${extra_args[@]}")
exec "${RTD_PYTHON:-$ROOT/.venv/bin/python}" tools/rtd_experiment.py "${args[@]}"
