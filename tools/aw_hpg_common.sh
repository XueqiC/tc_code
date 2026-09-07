#!/usr/bin/env bash
# Shared by the HPG launchers. Caller sets ROOT to the physical project root.
# Sourcing this file has no side effects; no GPU discovery or reassignment.
aw_hpg_port() {
  local port=${1:-}
  if [[ -z $port ]]; then
    [[ ${SLURM_JOB_ID:-} =~ ^[0-9]{1,12}$ ]] || {
      echo "A numeric SLURM_JOB_ID is required to choose the vllm port" >&2; return 2;
    }
    port=$((20000 + 10#$SLURM_JOB_ID % 30000))
  fi
  [[ $port =~ ^[0-9]{1,5}$ ]] || { echo "Invalid port: $port" >&2; return 2; }
  port=$((10#$port))
  [[ $port -ge 1 && $port -le 65535 ]] || { echo "Invalid port: $port" >&2; return 2; }
  printf '%s\n' "$port"
}

aw_hpg_require_gpu() {
  [[ ${SLURM_JOB_ID:-} =~ ^[0-9]{1,12}$ && -n ${CUDA_VISIBLE_DEVICES:-} && \
     $CUDA_VISIBLE_DEVICES != *,* && $CUDA_VISIBLE_DEVICES != -1 ]] || {
    echo "Run inside a SLURM allocation with exactly one assigned GPU" >&2; return 2;
  }
}

aw_hpg_environment() {
  # FlashInfer JIT finds ninja via PATH, even when vllm is launched by full path.
  local vllm_venv=$ROOT/envs/vllm/.venv
  if [[ -d $ROOT/envs/vllm-serve/.venv ]]; then
    vllm_venv=$(realpath -e -- "$ROOT/envs/vllm-serve/.venv")
  fi
  export PATH="$vllm_venv/bin:$ROOT/.venv/bin:$PATH"
  export HF_HOME=/blue/fsu-compsci-dept/xc25.fsu/hq/tools/hf-cache
  export HF_HUB_CACHE=$HF_HOME/hub HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
  export XDG_CACHE_HOME=$ROOT/envs/.cache
  export APPWORLD_ROOT=$ROOT/envs/appworld-repo APPWORLD_CACHE=$XDG_CACHE_HOME/appworld
  export VLLM_CACHE_ROOT=$XDG_CACHE_HOME/vllm TRITON_CACHE_DIR=$XDG_CACHE_HOME/triton
  export TORCH_HOME=$XDG_CACHE_HOME/torch TORCHINDUCTOR_CACHE_DIR=$XDG_CACHE_HOME/torchinductor
  export CUDA_CACHE_PATH=$XDG_CACHE_HOME/nv NUMBA_CACHE_DIR=$XDG_CACHE_HOME/numba
  export IPYTHONDIR=$XDG_CACHE_HOME/ipython MPLCONFIGDIR=$XDG_CACHE_HOME/matplotlib
  export PYTHONNOUSERSITE=1
}

aw_hpg_check_official() {
  [[ -x envs/appworld-official/.venv/bin/python && -x envs/appworld-official/.venv/bin/appworld && \
     -d envs/appworld-repo/data/tasks && \
     $(realpath -e -- envs/appworld-repo) == "$ROOT/envs/appworld-repo" && \
     $(realpath -e -- envs/appworld-repo/data) == "$ROOT/envs/appworld-repo/data" ]] || {
    echo "Missing/unsafe official AppWorld environment; run scripts/hpg_appworld_official_setup.sh" >&2; return 2;
  }
  envs/appworld-official/.venv/bin/python - <<'PY'
from importlib.metadata import version
from pathlib import Path
import sys

assert sys.version_info[:2] == (3, 12), sys.version
assert version("appworld") == "0.2.0.dev0"
assert version("appworld-agents") == "0.1.0.dev0"
data = Path("envs/appworld-repo/data")
assert (data / "version.txt").read_text().strip() == "0.2.0"
assert all((data / name).is_dir() for name in ("api_docs", "base_dbs", "datasets", "tasks"))
ids = [line.strip() for line in (data / "datasets/dev.txt").read_text().splitlines()
       if line.strip() and not line.startswith("#")]
assert len(ids) == len(set(ids)) == 57, "Official evaluation requires all dev57 tasks"
assert all((data / "tasks" / task / "specs.json").is_file() for task in ids)
PY
}
