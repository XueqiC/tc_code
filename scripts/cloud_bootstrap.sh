#!/usr/bin/env bash
# Run on the rented Ubuntu box, after rsync; see docs/cloud_bootstrap.md.
set -Eeuo pipefail

log() { printf '[%s] %s\n' "$(date -u +%FT%TZ)" "$*"; }
die() { log "ERROR: $*" >&2; exit 1; }
trap 'log "ERROR: bootstrap failed at line $LINENO; fix the cause and rerun." >&2' ERR

if [[ ${1:-} == --help ]]; then
    printf 'Usage: VOLUME_ROOT=/workspace HF_TOKEN=... bash scripts/cloud_bootstrap.sh\n'
    exit 0
fi
[[ $# == 0 ]] || die 'Only --help is accepted; configure VOLUME_ROOT and HF_TOKEN in the environment.'
[[ $(uname -s) == Linux && $(uname -m) == x86_64 ]] || die 'Linux x86_64 is required.'
# shellcheck source=/dev/null
source /etc/os-release
[[ ${ID:-} == ubuntu ]] || die 'This bootstrap supports Ubuntu.'
export VOLUME_ROOT=${VOLUME_ROOT:-/workspace}
[[ $VOLUME_ROOT == /* && $VOLUME_ROOT != / ]] || die 'VOLUME_ROOT must be an absolute persistent-volume path other than /.'
[[ -d $VOLUME_ROOT && -w $VOLUME_ROOT ]] || die 'Mount a writable persistent volume at VOLUME_ROOT first.'
VOLUME_ROOT=$(realpath -e -- "$VOLUME_ROOT")
REPO_ROOT=$VOLUME_ROOT/tc-alignment
[[ $(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P) == "$REPO_ROOT" ]] ||
    die "Rsync this checkout to $REPO_ROOT and run its script."
[[ -n ${HF_TOKEN:-} ]] || die 'Export HF_TOKEN with access to google/gemma-4-12B-it before starting.'
for req in cloud_train cloud_alfworld; do
    [[ -f $REPO_ROOT/requirements/$req.txt ]] || die "Missing requirements/$req.txt."
done

mkdir -p -- "$VOLUME_ROOT/logs" "$VOLUME_ROOT/tmp" "$VOLUME_ROOT/bin" "$VOLUME_ROOT/envs"
exec > >(tee -a "$VOLUME_ROOT/logs/cloud_bootstrap.log") 2>&1
command -v flock >/dev/null || die 'Install util-linux (flock) first.'
exec 9>"$VOLUME_ROOT/.cloud-bootstrap.lock"
flock -n 9 || die 'Another bootstrap is running on this volume.'
log "Preparing $REPO_ROOT; log: $VOLUME_ROOT/logs/cloud_bootstrap.log"

# Never overwrite a copied venv, a real envs directory, or an unrelated link.
link_directory() {
    local target=$1 link=$2
    if [[ -L $link ]]; then
        [[ $(realpath -m -- "$link") == "$target" ]] || die "Conflicting link: $link"
    elif [[ -e $link ]]; then
        die "$link already exists; move it aside before creating the volume link."
    else
        ln -s -- "$target" "$link"
    fi
}
TRAIN_VENV=$VOLUME_ROOT/venvs/train
ALF_ROOT=$VOLUME_ROOT/envs/alfworld
SERVE_VENV=$VOLUME_ROOT/envs/vllm-serve/.venv
mkdir -p -- "$TRAIN_VENV" "$ALF_ROOT" "$(dirname -- "$SERVE_VENV")"
link_directory "$VOLUME_ROOT/envs" "$REPO_ROOT/envs"
link_directory "$TRAIN_VENV" "$REPO_ROOT/.venv"
mkdir -p -- "$VOLUME_ROOT/envs/vllm"
link_directory "$SERVE_VENV" "$VOLUME_ROOT/envs/vllm/.venv"

log 'Checking Ubuntu build/download tools for TextWorld and vLLM JIT.'
packages=(build-essential ca-certificates curl git rsync unzip cmake pkg-config libffi-dev libssl-dev)
missing=()
for package in "${packages[@]}"; do
    if [[ $(dpkg-query -W -f='${Status}' "$package" 2>/dev/null || true) != 'install ok installed' ]]; then
        missing+=("$package")
    fi
done
if ((${#missing[@]})); then
    elevate=()
    if ((EUID != 0)); then
        command -v sudo >/dev/null && sudo -n true || die 'Root or passwordless sudo is needed to install Ubuntu packages.'
        elevate=(sudo -n)
    fi
    "${elevate[@]}" env DEBIAN_FRONTEND=noninteractive apt-get update
    "${elevate[@]}" env DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "${missing[@]}"
else
    log 'Ubuntu packages already installed.'
fi

# Keep uv, managed Python, wheel caches and download scratch on the volume.
export UV_CACHE_DIR=$VOLUME_ROOT/.cache/uv UV_PYTHON_INSTALL_DIR=$VOLUME_ROOT/python
export TMPDIR=$VOLUME_ROOT/tmp PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PATH="$VOLUME_ROOT/bin:$PATH"
UV=$VOLUME_ROOT/bin/uv
if [[ ! -x $UV ]] || [[ $("$UV" --version) != 'uv 0.11.7'* ]]; then
    log 'Installing uv 0.11.7 on the volume.'
    installer=$(mktemp "$TMPDIR/uv-install.XXXXXX")
    curl --fail --location --retry 3 https://astral.sh/uv/0.11.7/install.sh -o "$installer"
    UV_INSTALL_DIR="$VOLUME_ROOT/bin" UV_NO_MODIFY_PATH=1 sh "$installer"
    rm -f -- "$installer"
else
    log 'uv 0.11.7 already installed.'
fi
PYTHON_VERSION=3.12.13 # .venv/pyvenv.cfg in the source checkout.
log "Ensuring managed Python $PYTHON_VERSION is available."
"$UV" python install "$PYTHON_VERSION"
ensure_venv() {
    local venv=$1 version
    if [[ -f $venv/pyvenv.cfg ]]; then
        [[ -x $venv/bin/python ]] || die "Broken venv at $venv; move it aside and rerun."
        version=$("$venv/bin/python" -I -c 'import platform; print(platform.python_version())')
        [[ $version == "$PYTHON_VERSION" ]] || die "$venv uses Python $version, expected $PYTHON_VERSION."
        log "Reusing $venv."
    else
        log "Creating $venv."
        "$UV" venv --managed-python --python "$PYTHON_VERSION" "$venv"
    fi
}
ensure_venv "$TRAIN_VENV"
log 'Installing the filtered, pinned ALFWorld unified training dependencies.'
"$UV" pip install --python "$TRAIN_VENV/bin/python" -r "$REPO_ROOT/requirements/cloud_train.txt"
"$UV" pip check --python "$TRAIN_VENV/bin/python"

ensure_venv "$ALF_ROOT/.venv"
log 'Installing the isolated ALFWorld 0.4.2 / TextWorld 1.7.0 CPU worker.'
"$UV" pip install --python "$ALF_ROOT/.venv/bin/python" -r "$REPO_ROOT/requirements/cloud_alfworld.txt"
"$UV" pip check --python "$ALF_ROOT/.venv/bin/python"
export ALFWORLD_DATA=$ALF_ROOT/data ALFWORLD_DATA_ROOT=$ALF_ROOT/data
export ALFRED_DATA=$ALF_ROOT/data/json_2.1.1
mkdir -p -- "$ALFWORLD_DATA" "$ALF_ROOT/tmp"
cd -- "$ALF_ROOT"
data_complete() {
    "$ALF_ROOT/.venv/bin/python" -I - <<'PY'
import os
import json
from pathlib import Path
root = Path(os.environ['ALFWORLD_DATA'])
assert all((root / 'logic' / name).is_file() for name in ('alfred.pddl', 'alfred.twl2'))
task_types = {'pick_and_place_simple', 'look_at_obj_in_light', 'pick_clean_then_place_in_recep',
              'pick_heat_then_place_in_recep', 'pick_cool_then_place_in_recep', 'pick_two_obj_and_place'}
for split, count in (('train', 3553), ('valid_seen', 140), ('valid_unseen', 134)):
    games = []
    # Match ALFWorldAdapter._game_ids, including solvability and task types.
    for p in (root / 'json_2.1.1' / split).rglob('game.tw-pddl'):
        if 'movable' in str(p) or 'Sliced' in str(p):
            continue
        game = json.loads(p.read_text())
        trajectory = json.loads((p.parent / 'traj_data.json').read_text())
        if game.get('solvable') is True and trajectory.get('task_type') in task_types:
            games.append(p)
    assert len(games) == count, (split, len(games), count)
    assert all(p.stat().st_size and all((p.parent / n).is_file() and (p.parent / n).stat().st_size
               for n in ('traj_data.json', 'initial_state.pddl')) for p in games)
PY
}
if [[ -f $ALF_ROOT/.cloud-data-complete ]] && data_complete; then
    log 'ALFWorld data already complete; skipping download.'
else
    log "Downloading ALFWorld data into $ALFWORLD_DATA (working directory: $ALF_ROOT)."
    # --force repairs partial extracted files after an interrupted first run.
    CUDA_VISIBLE_DEVICES='' TMPDIR="$ALF_ROOT/tmp" \
        "$ALF_ROOT/.venv/bin/alfworld-download" --data-dir "$ALFWORLD_DATA" --force
    data_complete
    touch "$ALF_ROOT/.cloud-data-complete"
fi

ensure_venv "$SERVE_VENV"
log 'Installing vLLM 0.27.1, Transformers 5.14.1 and ninja for official evaluation.'
"$UV" pip install --python "$SERVE_VENV/bin/python" \
    'vllm==0.27.1' 'transformers==5.14.1' 'torch==2.13.0' 'ninja==1.13.0'
"$UV" pip check --python "$SERVE_VENV/bin/python"

export HF_HOME=$VOLUME_ROOT/hf-cache
export HF_HUB_CACHE=$HF_HOME/hub HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
mkdir -p -- "$HF_HUB_CACHE"
log 'Caching google/gemma-4-12B-it; HF_TOKEN is read from the environment and is not saved.'
# snapshot_download resumes partial downloads. Persist the first revision so a
# later bootstrap does not silently advance a model used by frozen identities.
HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 "$TRAIN_VENV/bin/python" -I - <<'PY'
import os
from pathlib import Path
from huggingface_hub import snapshot_download
cache = Path(os.environ['HF_HOME'])
revision_file = cache / 'gemma-4-12B-it.revision'
revision = revision_file.read_text().strip() if revision_file.exists() else 'main'
snapshot = Path(snapshot_download('google/gemma-4-12B-it', revision=revision,
    cache_dir=os.environ['HF_HUB_CACHE'], token=os.environ['HF_TOKEN']))
assert (snapshot / 'config.json').is_file() and (snapshot / 'tokenizer.json').is_file()
if not revision_file.exists():
    tmp = revision_file.with_suffix('.tmp')
    tmp.write_text(snapshot.name + '\n')
    tmp.replace(revision_file)
# Configs resolve refs/main with local_files_only=True, including on a rerun.
refs = snapshot.parent.parent / 'refs'
refs.mkdir(exist_ok=True)
(refs / 'main').write_text(snapshot.name + '\n')
print(f'Cached model snapshot: {snapshot}')
PY

log 'Writing the reusable hpg-equivalent activation file.'
ENV_FILE=$VOLUME_ROOT/cloud_env.sh
{
    printf '# Generated by scripts/cloud_bootstrap.sh; source before every run.\n'
    printf 'export VOLUME_ROOT=%q\n' "$VOLUME_ROOT"
    printf 'export ROOT=%q\n' "$REPO_ROOT"
    printf 'source %q\n' "$TRAIN_VENV/bin/activate"
    cat <<'ENV'
export RTD_PYTHON="$ROOT/.venv/bin/python"
# aw_hpg_environment puts the serving tools first so FlashInfer can find ninja.
# Always use RTD_PYTHON explicitly for training; bare python is the serving venv.
export PATH="$ROOT/envs/vllm-serve/.venv/bin:$ROOT/.venv/bin:$VOLUME_ROOT/bin:$PATH"
export PYTHONPATH="$ROOT/src:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export HF_HOME="$VOLUME_ROOT/hf-cache"
export HF_HUB_CACHE="$HF_HOME/hub" HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export ALFWORLD_DATA="$VOLUME_ROOT/envs/alfworld/data"
export ALFWORLD_DATA_ROOT="$ALFWORLD_DATA" ALFRED_DATA="$ALFWORLD_DATA/json_2.1.1"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0}"
export XDG_CACHE_HOME="$VOLUME_ROOT/.cache/rtd-v11"
export APPWORLD_ROOT="$ROOT/envs/appworld-repo" APPWORLD_CACHE="$XDG_CACHE_HOME/appworld"
export VLLM_CACHE_ROOT="$XDG_CACHE_HOME/vllm" TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export TORCH_HOME="$XDG_CACHE_HOME/torch" TORCHINDUCTOR_CACHE_DIR="$XDG_CACHE_HOME/torchinductor"
export CUDA_CACHE_PATH="$XDG_CACHE_HOME/nv" NUMBA_CACHE_DIR="$XDG_CACHE_HOME/numba"
export IPYTHONDIR="$XDG_CACHE_HOME/ipython" MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
export UV_CACHE_DIR="$VOLUME_ROOT/.cache/uv" UV_PYTHON_INSTALL_DIR="$VOLUME_ROOT/python"
export TMPDIR="$VOLUME_ROOT/tmp"
ENV
} > "$ENV_FILE.tmp"
bash -n "$ENV_FILE.tmp"
mv -- "$ENV_FILE.tmp" "$ENV_FILE"
mkdir -p -- "$REPO_ROOT/logs" "$REPO_ROOT/results" "$VOLUME_ROOT/.cache/rtd-v11"
if [[ ! -f $REPO_ROOT/data/rtd/v1_1_alfworld/public/support.json ]]; then
    log 'Replay bank missing: transfer it separately and audit it as described in docs/cloud_bootstrap.md.'
fi
log 'Setup complete. No GPU workload was launched. Audit the sealed replay bank before smoke.'
printf '\n# Smoke command (run after the bank audit passes):\n'
printf 'source %q\ncd -- %q\n' "$ENV_FILE" "$REPO_ROOT"
cat <<'SMOKE'
"$RTD_PYTHON" tools/rtd_experiment.py smoke \
  --config configs/rtd/unified_alfworld_gemma4.yaml --arm D3 \
  --run-dir results/rtd_unified/smoke_alf_D3 --smoke-deadline-seconds 7200
SMOKE
