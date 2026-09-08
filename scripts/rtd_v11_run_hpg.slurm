#!/usr/bin/env bash
#SBATCH --job-name=rtd_v11
#SBATCH --account=fsu-compsci-dept
#SBATCH --qos=fsu-compsci-dept
#SBATCH --partition=hpg-b200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
cd -- "${SLURM_SUBMIT_DIR:?Submit from the project root}"
ROOT=$(pwd -P)
source tools/aw_hpg_common.sh
aw_hpg_require_gpu
aw_hpg_environment
RTD_ARM=${1:-${RTD_ARM:-}}
if [[ $# -gt 0 ]]; then shift; fi
case $RTD_ARM in V0|V1|V2) ;; *) echo 'arm must be V0, V1 or V2' >&2; exit 2 ;; esac
RTD_RUN_DIR=${RTD_RUN_DIR:-results/rtd_v1_1/$RTD_ARM}
RTD_COMMAND=${RTD_COMMAND:-auto}
if [[ $RTD_COMMAND == auto ]]; then
  RTD_COMMAND=run
  if [[ -f $RTD_RUN_DIR/manifest.json ]]; then RTD_COMMAND=resume; fi
fi
case $RTD_COMMAND in run|resume|smoke) ;; *) echo 'RTD_COMMAND must be auto, run, resume or smoke' >&2; exit 2 ;; esac
export PYTHONPATH="$ROOT/src:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export PYTHONDONTWRITEBYTECODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Keep writable runtime caches in this checkout; data/ and envs/ may be symlinks.
export XDG_CACHE_HOME="$ROOT/.cache/rtd-v11" APPWORLD_CACHE="$ROOT/.cache/rtd-v11/appworld"
export VLLM_CACHE_ROOT="$XDG_CACHE_HOME/vllm" TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export TORCH_HOME="$XDG_CACHE_HOME/torch" TORCHINDUCTOR_CACHE_DIR="$XDG_CACHE_HOME/torchinductor"
export CUDA_CACHE_PATH="$XDG_CACHE_HOME/nv" NUMBA_CACHE_DIR="$XDG_CACHE_HOME/numba"
export IPYTHONDIR="$XDG_CACHE_HOME/ipython" MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
args=("$RTD_COMMAND" --arm "$RTD_ARM" --run-dir "$RTD_RUN_DIR")
# Resume uses its saved config; relay jobs may omit RTD_CONFIG entirely.
if [[ $RTD_COMMAND != resume ]]; then
  args+=(--config "${RTD_CONFIG:-configs/rtd/v1_1_bfcl.yaml}")
fi
IFS=$' \t\n' read -r -d '' -a extra_args < <(printf '%s\0' "${RTD_EXTRA_ARGS:-}")
args+=("${extra_args[@]}" "$@")
exec "${RTD_PYTHON:-$ROOT/.venv/bin/python}" tools/rtd_experiment.py "${args[@]}"
