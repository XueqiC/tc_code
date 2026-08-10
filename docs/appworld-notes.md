# AppWorld setup notes

Date: 2026-08-10

## Outcome

Setup is **incomplete because this execution shell has no DNS/network access**. The dedicated
Python 3.11 virtual environment exists, but the `appworld` PyPI package and benchmark data could
not be downloaded. Consequently, the smoke test did not start an episode and did not produce an
evaluation result object.

All paths selected for mutable AppWorld state are inside this project:

- virtual environment: `envs/appworld-venv`
- AppWorld root: `envs/appworld-data`
- AppWorld benchmark data, once downloaded: `envs/appworld-data/data`
- AppWorld episode outputs, once run: `envs/appworld-data/experiments/outputs`
- uv cache: `envs/.uv-cache`
- XDG cache: `envs/.cache`
- smoke runner: `envs/appworld-smoke.py`
- smoke output: `logs/appworld_smoke.log`

No files under `src/` or `paper/` were modified.

## Install steps and results

The isolation variables used were:

```bash
export APPWORLD_ROOT="$(pwd)/envs/appworld-data"
export XDG_CACHE_HOME="$(pwd)/envs/.cache"
export UV_CACHE_DIR="$(pwd)/envs/.uv-cache"
export UV_PYTHON_INSTALL_DIR="$(pwd)/envs/.uv-python"
```

### What worked

The host does not have `python3.12`. Attempting to let uv obtain Python 3.12 failed as recorded
below, so the requested Python 3.11 fallback was used:

```bash
uv venv --no-project --python 3.11 envs/appworld-venv
```

Result:

```text
Using CPython 3.11.7 interpreter at: /usr/local/anaconda3/bin/python3
Creating virtual environment at: envs/appworld-venv
Activate with: source envs/appworld-venv/bin/activate
```

The venv is intentionally minimal (`uv venv` without `--seed`), so it does not contain a separate
`pip` executable/package. `uv pip --python ...` is the intended installer.

### Exact failures

Python 3.12 creation command:

```bash
uv venv --no-project --python 3.12 envs/appworld-venv
```

Error (exit 2):

```text
error: Request failed after 3 retries in 6.2s
  Caused by: Failed to download https://github.com/astral-sh/python-build-standalone/releases/download/20260414/cpython-3.12.13%2B20260414-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz
  Caused by: error sending request for url (https://github.com/astral-sh/python-build-standalone/releases/download/20260414/cpython-3.12.13%2B20260414-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz)
  Caused by: client error (Connect)
  Caused by: dns error
  Caused by: failed to lookup address information: Temporary failure in name resolution
```

Package installation command:

```bash
uv pip install --python envs/appworld-venv/bin/python appworld
```

Error (exit 2):

```text
Using Python 3.11.7 environment at: envs/appworld-venv
error: Request failed after 3 retries in 7.3s
  Caused by: Failed to fetch: `https://pypi.org/simple/appworld/`
  Caused by: error sending request for url (https://pypi.org/simple/appworld/)
  Caused by: client error (Connect)
  Caused by: dns error
  Caused by: failed to lookup address information: Temporary failure in name resolution
```

No AppWorld package was found in the host's uv cache, pip cache, global Python installation, or
existing conda environments. Because the package was unavailable, each documented CLI-equivalent
command below exited 1:

```bash
envs/appworld-venv/bin/python -m appworld.cli --help
envs/appworld-venv/bin/python -m appworld.cli install
envs/appworld-venv/bin/python -m appworld.cli download data
```

Each emitted:

```text
/home/xueqi/hq/projects/tc-alignment/envs/appworld-venv/bin/python: Error while finding module specification for 'appworld.cli' (ModuleNotFoundError: No module named 'appworld')
```

The CLI forms selected above follow the current official installation instructions:
[`pip install appworld`, `appworld install`, and `appworld download data`](https://github.com/StonyBrookNLP/appworld#-installation).
The same documentation defines `APPWORLD_ROOT`/`--root` and places data at `<root>/data`.

### Resume when package-network access is available

From the project root, set the four isolation variables above and run:

```bash
uv pip install --python envs/appworld-venv/bin/python appworld
envs/appworld-venv/bin/appworld install
envs/appworld-venv/bin/appworld download data
APPWORLD_ROOT="$(pwd)/envs/appworld-data" \
  envs/appworld-venv/bin/python envs/appworld-smoke.py \
  > logs/appworld_smoke.log 2>&1
```

The official verification commands are then:

```bash
envs/appworld-venv/bin/appworld verify tests
envs/appworld-venv/bin/appworld verify tasks
```

## Data size on disk

`envs/appworld-data` contains no files. Its downloaded-data payload is therefore **0 bytes**
(the empty directory itself occupies 4 KiB according to `du`). At inspection time:

```text
4.0K  envs/appworld-data
100K  envs/appworld-venv
40K   envs/.uv-cache
```

Recompute the data size after a successful download with:

```bash
du -sh envs/appworld-data/data
```

## Smoke test

`envs/appworld-smoke.py` follows the documented minimal loop on the first `train` task. Its
trivial agent emits one Python code string, `apis.supervisor.complete_task()`, through
`world.execute(...)`, then serializes `world.evaluate().to_dict()` as `evaluation_result=...`.
The intent is environment plumbing validation, not task success.

This run exited 1 at import time:

```text
Traceback (most recent call last):
  File "/home/xueqi/hq/projects/tc-alignment/envs/appworld-smoke.py", line 7, in <module>
    from appworld import AppWorld, load_task_ids
ModuleNotFoundError: No module named 'appworld'
```

The exact combined stdout/stderr is in `logs/appworld_smoke.log`. Success remains unverified because
an episode did not execute and an evaluation object was not produced.

## API surface

The descriptions below are based on AppWorld's current official
[walkthrough](https://github.com/StonyBrookNLP/appworld#-appworld-walkthrough) and public source
for [`task.py`](https://github.com/StonyBrookNLP/appworld/blob/main/src/appworld/task.py) and
[`ground_truth.py`](https://github.com/StonyBrookNLP/appworld/blob/main/src/appworld/ground_truth.py).
They could not be runtime-verified in this environment.

### A. List tasks and app/scenario metadata

`load_task_ids("train")` lists task variants. A task ID has the form
`<generator_id>_<variant_number>`; the generator ID is the scenario ID shared by its variants.
For `train` and `dev`, load full ground truth to obtain `required_apps` and `required_apis`.

```python
from appworld import load_task_ids
from appworld.task import Task

rows = []
for task_id in load_task_ids("train"):
    task = Task.load(task_id, ground_truth_mode="full")
    try:
        ground_truth = task.ground_truth
        assert ground_truth is not None
        rows.append(
            {
                "task_id": task.id,
                "scenario_id": task.generator_id,
                "instruction": task.instruction,
                "required_apps": ground_truth.required_apps,
                "required_apis": ground_truth.required_apis,
                "difficulty": ground_truth.metadata["difficulty"],
                "num_apps": ground_truth.metadata["num_apps"],
                "num_apis": ground_truth.metadata["num_apis"],
                "num_solution_code_lines": ground_truth.metadata[
                    "num_solution_code_lines"
                ],
            }
        )
    finally:
        task.close()
```

`Task` is a module-level API rather than a top-level export. The documented public alternative is
to initialize `AppWorld(...)` and read the same object at `world.task`; that also creates an episode
output directory. Do not request full ground truth for test tasks. Test tasks expose only minimal
difficulty metadata by design, and task-wise test inspection should not be used for development.

### B. Run an agent that emits Python code strings

The agent contract can be a method returning the next Python string. `world.execute(code)` returns
printed output or an execution traceback and retains variables between calls.

```python
from appworld import AppWorld, load_task_ids

task_id = load_task_ids("train")[0]
with AppWorld(task_id=task_id, experiment_name="my_agent/train") as world:
    last_output = None
    for _ in range(20):
        code: str = agent.next_code_block(last_output, task=world.task)
        last_output = world.execute(code)
        if world.task_completed():
            break
```

Within emitted code, functional API calls use
`apis.<app_name>.<api_name>(**parameters)`. The agent signals completion with
`apis.supervisor.complete_task(...)`. The default safety guards should remain enabled, but AppWorld
notes that same-process execution is not a complete security boundary.

### C. Obtain pass/fail evaluation

For one live episode:

```python
result: dict = world.evaluate().to_dict()
passed: bool = result["success"]
report: str = world.evaluate().report(print_it=False, colorize=False)
```

The current result includes `success`, `passes`, `failures`, `difficulty`, and `num_tests`. For offline batch
evaluation, use `appworld evaluate <experiment_name> <dataset_name>` or
`appworld.evaluate_tasks(task_ids)`. Batch JSON adds aggregate `task_goal_completion` (TGC) and
`scenario_goal_completion` (SGC); SGC requires all variants of a scenario.

## First proposal for topic scoping

This proposal uses only public app/task metadata, not hidden test ground truth. The current app set
is `spotify`, `splitwise`, `todoist`, `venmo`, `phone`, `amazon`, `gmail`, `simple_note`, and
`file_system` (plus the `api_docs` and `supervisor` helper apps).

Public train/dev metadata inspected in the official
[Task Explorer](https://appworld.dev/appworld/task-explorer) included:

- `aa8502b_3`: train, medium, 7 APIs, required apps `{spotify}`.
- `27e1026_1`: train, easy, 8 APIs, required apps `{spotify}`.
- `ce359b5_3`: train, medium, 10 APIs, required apps `{spotify}`.
- `d0b1f43_3`: train, medium, 7 APIs, required apps `{phone, venmo}`.
- `fac291d_1`: dev, medium, 7 APIs, required apps `{spotify}`.

The explorer also explicitly states at the app level that Amazon and Gmail tasks occur only in
`test_challenge`.

A clean first scope is:

- **In-domain:** single-app media tasks (`spotify`) and personal/social-finance tasks
  (`phone` + `venmo`, later adding `splitwise`). These are represented in inspectable train
  metadata and exercise both one-app and cross-app behavior.
- **Bridge/dev:** personal organization (`todoist`, `simple_note`) and controlled file operations
  (`file_system`). Use these to determine whether failures come from new APIs or from cross-app
  composition before declaring a strict OOD result.
- **Out-of-domain:** communication/commerce workflows involving `gmail` and/or `amazon`, especially
  when composed with `file_system`, `phone`, or `venmo`. This is a defensible app-level OOD boundary
  because the official metadata withholds Amazon/Gmail scenarios to `test_challenge`.

For any custom split, group by `scenario_id` (`task.generator_id`), not task ID, so variants of the
same generator never cross the in-domain/OOD boundary. Report results both by required-app set and
by scenario, with difficulty, API count, app count, and solution-code length as balancing
covariates. The final clusters should be recomputed from all local train/dev `required_apps`
metadata after data download rather than inferred from the few public examples above.
