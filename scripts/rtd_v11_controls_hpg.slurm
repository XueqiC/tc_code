#!/usr/bin/env bash
#SBATCH --job-name=rtd_v11_controls
#SBATCH --account=fsu-compsci-dept
#SBATCH --qos=fsu-compsci-dept
#SBATCH --partition=hpg-b200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
cd -- "${SLURM_SUBMIT_DIR:?Submit from this worktree root}"
ROOT=$(pwd -P)
source tools/aw_hpg_common.sh
aw_hpg_require_gpu
aw_hpg_environment
export PYTHONPATH="$ROOT/src:$ROOT"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export XDG_CACHE_HOME="$ROOT/.cache/rtd-v11-controls"
export APPWORLD_CACHE="$XDG_CACHE_HOME/appworld" VLLM_CACHE_ROOT="$XDG_CACHE_HOME/vllm"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton" TORCH_HOME="$XDG_CACHE_HOME/torch"
export TORCHINDUCTOR_CACHE_DIR="$XDG_CACHE_HOME/torchinductor" CUDA_CACHE_PATH="$XDG_CACHE_HOME/nv"
export NUMBA_CACHE_DIR="$XDG_CACHE_HOME/numba" IPYTHONDIR="$XDG_CACHE_HOME/ipython"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib" BFCL_PROJECT_ROOT="$XDG_CACHE_HOME/bfcl"
RTD_CONTROL_RUN=${1:-results/rtd_v1_1/V2}
RTD_CONTROL_OUT=${2:-results/rtd_v1_1_controls}
# Two predetermined late windows; no selecting windows by observed gain.
for window in r1-s10 r2-s10; do
  "${RTD_PYTHON:-$ROOT/.venv/bin/python}" tools/rtd_v11_controls.py \
    --window "$RTD_CONTROL_RUN/controls/windows/$window.pt" \
    --out "$RTD_CONTROL_OUT/$window" --seed 0 --resamples 4 --z-probes 4
done
