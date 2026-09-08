#!/usr/bin/env bash
#SBATCH --job-name=cc_probe
#SBATCH --account=fsu-compsci-dept
#SBATCH --qos=fsu-compsci-dept
#SBATCH --partition=hpg-b200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=200gb
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
# Submit from the checkout with logs/ already present:
#   sbatch scripts/cc_probe_hpg.slurm
# rai fallback (same sequential serve/score workflow):
#   CUDA_VISIBLE_DEVICES=0 CC_PYTHON=.venv/bin/python bash scripts/cc_probe_hpg.slurm
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
CC_VLLM=${CC_VLLM:-envs/vllm-serve/.venv/bin/vllm}
CC_PORT=${CC_PORT:-8976}
CC_OUT=${CC_OUT:-data/cc_pairs_v1}
CC_SEED=${CC_SEED:-0}
export PYTHONPATH="$CC_ROOT/src:$CC_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false CUBLAS_WORKSPACE_CONFIG=:4096:8
if [[ -n ${SLURM_JOB_ID:-} ]]; then
    # Match aw_hpg_environment: FlashInfer JIT locates ninja through PATH.
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
mkdir -p logs
if [[ ! -f $CC_OUT/candidates.jsonl ]]; then
    "$CC_PYTHON" tools/cc_pairs.py candidates --out "$CC_OUT" --seed "$CC_SEED"
fi
# Refuse an occupied port so an unrelated or adapted server cannot supply data.
"$CC_PYTHON" - "$CC_PORT" <<'PY'
import socket, sys
with socket.socket() as sock:
    sock.bind(('127.0.0.1', int(sys.argv[1])))
PY
CC_SERVER_PID=
cleanup() {
    if [[ -n $CC_SERVER_PID ]]; then
        kill "$CC_SERVER_PID" 2>/dev/null || true
        wait "$CC_SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT
"$CC_VLLM" serve Qwen/Qwen3.5-4B --served-model-name Qwen/Qwen3.5-4B \
    --host 127.0.0.1 --port "$CC_PORT" --seed "$CC_SEED" \
    --dtype bfloat16 --gpu-memory-utilization 0.85 --max-model-len 32768 \
    > "logs/cc_vllm_${SLURM_JOB_ID:-rai}.log" 2>&1 &
CC_SERVER_PID=$!
CC_READY=0
for ((i=0; i<180; i++)); do
    if ! kill -0 "$CC_SERVER_PID" 2>/dev/null; then
        echo 'Base vLLM server exited; inspect logs/cc_vllm_*.log' >&2
        exit 1
    fi
    if curl --fail --silent --max-time 3 "http://127.0.0.1:$CC_PORT/v1/models" >/dev/null; then
        CC_READY=1
        break
    fi
    sleep 5
done
[[ $CC_READY == 1 ]] || { echo 'Base vLLM server timed out' >&2; exit 1; }
"$CC_PYTHON" tools/cc_pairs.py probe --phase generate --out "$CC_OUT" --seed "$CC_SEED" \
    --base-url "http://127.0.0.1:$CC_PORT/v1"
cleanup
CC_SERVER_PID=
# Native CE scoring now has the GPU to itself, using the unadapted Hub base.
"$CC_PYTHON" tools/cc_pairs.py probe --phase score --out "$CC_OUT" --seed "$CC_SEED"
"$CC_PYTHON" tools/cc_pairs.py select --out "$CC_OUT" --seed "$CC_SEED"
