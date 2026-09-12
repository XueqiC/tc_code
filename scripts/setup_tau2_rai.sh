#!/usr/bin/env bash
# All installation writes, including caches and temporary files, stay in envs/tau2.
set -euo pipefail
TASK_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
TAU_SANDBOX="$TASK_ROOT/envs/tau2"
TAU_COMMIT=a2c024725189473d2d7cea3a5cfdbcc67478e41f
TAU_ORIGIN=https://github.com/sierra-research/tau2-bench.git
if [[ $(realpath -m "$TAU_SANDBOX") != "$TAU_SANDBOX" ]]; then
    echo 'envs/tau2 must be a real directory inside this checkout, not an external symlink.' >&2
    exit 1
fi
mkdir -p "$TAU_SANDBOX"
cd "$TAU_SANDBOX"
# Refuse external links anywhere an installer could write.
while IFS= read -r -d '' link; do
    case "$(realpath -m "$link")" in
        "$TAU_SANDBOX"/*) ;;
        *) echo "External installation symlink: $link" >&2; exit 1 ;;
    esac
done < <(find . -type l ! -path './.venv/*' -print0)
export UV_CACHE_DIR="$TAU_SANDBOX/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$TAU_SANDBOX/.python"
export UV_PROJECT_ENVIRONMENT="$TAU_SANDBOX/.venv"
export UV_LINK_MODE=copy UV_NO_CONFIG=1
export XDG_CACHE_HOME="$TAU_SANDBOX/.cache"
export XDG_DATA_HOME="$TAU_SANDBOX/.local/share"
export XDG_CONFIG_HOME="$TAU_SANDBOX/.config"
export TMPDIR="$TAU_SANDBOX/.tmp"
export PYTHONDONTWRITEBYTECODE=1
export TAU2_DATA_DIR="$TAU_SANDBOX/repo/data"
export LITELLM_LOCAL_MODEL_COST_MAP=True
export HF_HUB_OFFLINE=1
mkdir -p "$TMPDIR" "$UV_CACHE_DIR" "$XDG_CONFIG_HOME"
if [[ ! -d repo/.git ]]; then
    # Optional read-only seed for hosts whose shell has no network. No hardlinks.
    if [[ -n ${TAU2_SOURCE_CACHE:-} ]]; then
        git clone --no-hardlinks "$TAU2_SOURCE_CACHE" repo
        git -C repo remote set-url origin "$TAU_ORIGIN"
    else
        git clone "$TAU_ORIGIN" repo
    fi
fi
if ! git -C repo cat-file -e "$TAU_COMMIT^{commit}"; then
    git -C repo fetch origin "$TAU_COMMIT"
fi
if [[ -n $(git -C repo status --porcelain --untracked-files=no) ]]; then
    echo 'Refusing to overwrite changes in envs/tau2/repo.' >&2
    exit 1
fi
git -C repo checkout --detach "$TAU_COMMIT"
test "$(git -C repo rev-parse HEAD)" = "$TAU_COMMIT"
printf '%s\n' "$TAU_COMMIT" > installed-commit.txt
# The pinned official repository tracks all core domain data; clone/checkout
# downloads it into repo/data. There is no separate external data downloader.
if [[ -n ${TAU2_UV_CACHE_SEED:-} ]]; then
    # Copy cached wheels, preserving uv's archive layout with LOCAL symlinks.
    # Reading a shared cache in place would let uv write outside the sandbox.
    python3 - <<'PY'
import importlib.metadata
import os
import shutil
import tomllib
from pathlib import Path

source = Path(os.environ['TAU2_UV_CACHE_SEED'])
target = Path(os.environ['UV_CACHE_DIR'])
lock = tomllib.loads(Path('repo/uv.lock').read_text())
packages = {}
for package in lock['package']:
    packages.setdefault(package['name'], []).append(package)
root = packages['tau2'][0]
todo = [d['name'] for d in root['dependencies']]
todo += ['pytest', 'iniconfig', 'pluggy', 'hatchling', 'packaging', 'pathspec',
         'trove-classifiers', 'editables', 'tomlkit']
# A prior install can seed newer compatible versions than upstream's old lock.
seed_repo = os.environ.get('TAU2_SOURCE_CACHE')
if seed_repo:
    for site in Path(seed_repo).glob('.venv/lib/python*/site-packages'):
        todo += [d.metadata['Name'].lower().replace('_', '-')
                 for d in importlib.metadata.distributions(path=[str(site)])]
seen = set()
while todo:
    name = todo.pop()
    if name in seen:
        continue
    seen.add(name)
    for package in packages.get(name, []):
        todo += [d['name'] for d in package.get('dependencies', [])]
    package_dir = source / 'wheels-v6/pypi' / name
    if not package_dir.is_dir():
        continue
    destination = target / 'wheels-v6/pypi' / name
    destination.mkdir(parents=True, exist_ok=True)
    for entry in package_dir.iterdir():
        output = destination / entry.name
        if output.exists():
            continue
        if entry.is_dir():
            archive = target / 'archive-v0' / entry.resolve().name
            if not archive.exists():
                shutil.copytree(entry, archive, symlinks=False)
            output.symlink_to(os.path.relpath(archive, output.parent))
        else:
            shutil.copy2(entry, output)
shutil.copytree(source / 'simple-v21', target / 'simple-v21', dirs_exist_ok=True)
print(f'Seeded {len(seen)} dependency names into the sandbox cache')
PY
fi
if [[ ! -x "$UV_PROJECT_ENVIRONMENT/bin/python" ]]; then
    uv venv --python "${TAU2_PYTHON:-3.12}" "$UV_PROJECT_ENVIRONMENT"
fi
uv pip install --python "$UV_PROJECT_ENVIRONMENT/bin/python" -e "$TAU_SANDBOX/repo" pytest ${TAU2_OFFLINE:+--offline}
uv pip check --python "$UV_PROJECT_ENVIRONMENT/bin/python"
uv pip freeze --python "$UV_PROJECT_ENVIRONMENT/bin/python" > requirements-installed.txt
"$UV_PROJECT_ENVIRONMENT/bin/tau2" --help
"$UV_PROJECT_ENVIRONMENT/bin/tau2" check-data
"$UV_PROJECT_ENVIRONMENT/bin/python" - <<'PY'
from tau2.run import get_tasks
for domain in ('retail', 'airline', 'telecom'):
    tasks = get_tasks(domain, task_split_name='test')
    assert tasks and len({task.id for task in tasks}) == len(tasks)
    print(f'{domain}: {len(tasks)} test tasks loaded offline')
PY
# Only task parsing tests: no agent, simulator, GPU, or API invocation.
cd repo
"$UV_PROJECT_ENVIRONMENT/bin/python" -m pytest tests/test_tasks.py -q
