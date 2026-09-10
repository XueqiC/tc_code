#!/usr/bin/env bash
# Run ON HPG from the project root, after scripts/sync_appworld_to_hpg.sh on rai.
# Inspected rai metadata: CPython 3.12.13, INSTALLER=uv; both direct_url.json
# files have dir_info.editable=true, pointing to this checkout and experiments/.
# appworld 0.2.0.dev0, appworld-agents 0.1.0.dev0; checkout a072b7a86e7c1d5b1d7175659d750ebb9b79f10a.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
[[ $# == 0 && $(pwd -P) == "$ROOT" ]] || {
  echo "Run from the project root: bash scripts/hpg_appworld_official_setup.sh" >&2; exit 2;
}
REPO=$ROOT/envs/appworld-repo
VENV=$ROOT/envs/appworld-official/.venv
[[ -f $REPO/pyproject.toml && -f $REPO/experiments/pyproject.toml ]] || {
  echo "Missing official checkout; run scripts/sync_appworld_to_hpg.sh on rai first" >&2; exit 2;
}

# HARD SAFETY GUARD: AppWorld download_data() recursively deletes root/data.
# Check the PHYSICAL cwd immediately before every AppWorld CLI operation.
assert_appworld_cwd() {
  local here path
  here=$(pwd -P)
  [[ $here == "$ROOT/envs/"* && $here == "$REPO" ]] || {
    echo "REFUSE AppWorld operation: cwd resolves outside envs/appworld-repo: $here" >&2; exit 2;
  }
  for path in data .tmp; do
    [[ $(realpath -m -- "$here/$path") == "$here/$path" && ! -L $here/$path ]] || {
      echo "REFUSE AppWorld operation: $path must stay inside the checkout (no symlinks)" >&2; exit 2;
    }
  done
}
cd -- "$REPO"
assert_appworld_cwd
[[ $ROOT == /blue/* ]] || { echo "Setup must run ON HPG with the project on /blue" >&2; exit 2; }

# Keep interpreters, virtualenv, installers and application caches on /blue.
export XDG_CACHE_HOME=$ROOT/envs/.cache
export UV_CACHE_DIR=$XDG_CACHE_HOME/uv UV_PYTHON_INSTALL_DIR=$ROOT/envs/.python
export UV_PYTHON_BIN_DIR=$ROOT/envs/.python/bin PIP_CACHE_DIR=$XDG_CACHE_HOME/pip
export APPWORLD_ROOT=$REPO APPWORLD_CACHE=$XDG_CACHE_HOME/appworld
export IPYTHONDIR=$XDG_CACHE_HOME/ipython PYTHONNOUSERSITE=1
export UV_NO_CONFIG=1 PIP_CONFIG_FILE=/dev/null
for path in "$VENV" "$XDG_CACHE_HOME" "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" \
            "$UV_PYTHON_BIN_DIR" "$PIP_CACHE_DIR" "$APPWORLD_CACHE" "$IPYTHONDIR"; do
  [[ $(realpath -m -- "$path") == "$path" ]] || {
    echo "REFUSE setup: environment/cache path resolves elsewhere: $path" >&2; exit 2;
  }
done
mkdir -p "$(dirname -- "$VENV")" "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" "$PIP_CACHE_DIR" "$APPWORLD_CACHE"
exec 9>"$(dirname -- "$VENV")/setup.lock"
flock -n 9 || { echo "Official AppWorld setup is already running" >&2; exit 3; }
if [[ ! -e $VENV ]]; then
  if command -v uv >/dev/null 2>&1; then
    uv venv --no-project --managed-python --python 3.12.13 "$VENV"
  else
    # A site Python 3.12.x is the fallback; --copies keeps the executable on /blue.
    python3.12 -m venv --copies "$VENV"
  fi
fi
[[ -x $VENV/bin/python && $(realpath -e -- "$VENV/bin/python") == /blue/* ]] || {
  echo "Invalid /blue virtualenv; recreate it on HPG rather than copying rai's venv" >&2; exit 2;
}
"$VENV/bin/python" - "$VENV" "$REPO" <<'PY'
from pathlib import Path
import sys
import tomllib

assert sys.version_info[:2] == (3, 12), sys.version
assert Path(sys.prefix).resolve() == Path(sys.argv[1]), sys.prefix
repo = Path(sys.argv[2])
for directory, expected in [(repo, "0.2.0.dev0"), (repo / "experiments", "0.1.0.dev0")]:
    with (directory / "pyproject.toml").open("rb") as stream:
        assert tomllib.load(stream)["project"]["version"] == expected, directory
print(f"Official AppWorld Python: {sys.version.split()[0]}")
PY
if command -v uv >/dev/null 2>&1; then
  uv pip install --python "$VENV/bin/python" -e "$REPO" -e "$REPO/experiments"
else
  "$VENV/bin/python" -m pip install -e "$REPO" -e "$REPO/experiments"
fi
"$VENV/bin/python" - "$REPO" <<'PY'
from importlib.metadata import distribution
from pathlib import Path
import json
import sys

repo = Path(sys.argv[1])
for name, version, source in [("appworld", "0.2.0.dev0", repo),
                              ("appworld-agents", "0.1.0.dev0", repo / "experiments")]:
    dist = distribution(name)
    direct = json.loads(dist.read_text("direct_url.json"))
    assert dist.version == version, (name, dist.version)
    assert direct.get("dir_info", {}).get("editable") is True, direct
    assert direct["url"] == source.as_uri(), direct
PY
assert_appworld_cwd
"$VENV/bin/appworld" install --repo

data_ready() {
  "$VENV/bin/python" - <<'PY'
from pathlib import Path
import sys

data = Path("data")
try:
    assert all((data / part).is_dir() and any((data / part).iterdir())
               for part in ("api_docs", "base_dbs", "datasets", "tasks"))
    assert (data / "version.txt").read_text().strip() == "0.2.0"
    for split in ("train", "dev"):
        ids = [line.strip() for line in (data / "datasets" / f"{split}.txt").read_text().splitlines()
               if line.strip() and not line.startswith("#")]
        assert ids and (split != "dev" or len(ids) == len(set(ids)) == 57)
        assert all((data / "tasks" / task / "specs.json").is_file() for task in ids)
except (OSError, AssertionError):
    sys.exit(1)
PY
}
if data_ready; then
  echo "Official data 0.2.0 already complete; skipping download"
else
  assert_appworld_cwd
  "$VENV/bin/appworld" download data --root . --version 0.2.0 --without-setup
fi
assert_appworld_cwd
[[ -d $REPO/data/tasks ]] && data_ready || {
  echo "Official AppWorld data verification failed (requires complete train + dev57)" >&2; exit 1;
}
echo APPWORLD_OFFICIAL_OK
