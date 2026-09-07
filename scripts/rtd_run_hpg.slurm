#!/usr/bin/env bash
#SBATCH --job-name=rtd_v1
#SBATCH --account=fsu-compsci-dept
#SBATCH --qos=fsu-compsci-dept
#SBATCH --partition=hpg-b200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=240G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
cd -- "${SLURM_SUBMIT_DIR:?Submit from the project root}"
ROOT=$(pwd -P)
source tools/aw_hpg_common.sh
aw_hpg_require_gpu
aw_hpg_environment
case ${RTD_ARM:?Set RTD_ARM to R0 or R1} in R0|R1) ;; *) echo 'RTD_ARM must be R0 or R1' >&2; exit 2 ;; esac
RTD_CONFIG=${RTD_CONFIG:-configs/rtd/v1_bfcl_c25.yaml}
export PYTHONPATH="$ROOT/src:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
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
