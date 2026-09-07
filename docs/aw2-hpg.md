# AW-2 on HiPerGator (C18)

These scripts prepare and run AW-2 B/C with the official simplified ReAct dev57
scaffold. The rai scripts are unchanged. The commands below are instructions;
creating these files does not transfer data, install packages or submit jobs.

On rai, from the project root, transfer code and then the AppWorld inputs:

```bash
bash scripts/sync_to_hpg.sh
bash scripts/sync_appworld_to_hpg.sh
```

The second script uses the existing `hpg` SSH alias. It copies the official
checkout, archived demos and whichever matched/x2 pools exist. It excludes
`.git`, `experiments/outputs`, data, virtualenvs and caches. Rai's checkout has a
data symlink; that link is deliberately excluded. No merged models are needed.

On HPG, prepare the official environment and data:

```bash
cd /blue/fsu-compsci-dept/xc25.fsu/hq/tc-alignment
bash scripts/hpg_appworld_official_setup.sh
```

Setup reproduces the inspected editable installs of `appworld==0.2.0.dev0`
from `envs/appworld-repo` and `appworld-agents==0.1.0.dev0` from its `experiments`
directory. Rai's installed metadata names uv as the installer, Python 3.12.13,
and checkout commit `a072b7a86e7c1d5b1d7175659d750ebb9b79f10a`. Setup uses uv
with managed Python 3.12.13 on `/blue`, or a site Python 3.12.x with `venv`/pip.
It checks the installed versions and editable source paths. Dependency ranges
come from the copied pyprojects; this is not a frozen transitive dependency lock.

Every AppWorld setup command runs inside the physical checkout. The downloader
gets `--root . --version 0.2.0 --without-setup`; the script refuses escaped cwd,
data or temporary paths. Complete data is retained on reruns. Success requires
train task specs and all 57 dev task specs and prints `APPWORLD_OFFICIAL_OK`.
Interpreters, environments and caches stay on `/blue` without changing `HOME`.

The matched pool is required for B and mining. If x2 is absent, B can run first:

```bash
sbatch --export=ALL,AW2_ARMS=B scripts/aw2_bc_hpg.slurm
sbatch scripts/aw2_mine_hpg.slurm
```

Mining uses the same midpoint plan, archived demos, official bridge, seeds and
continuation settings as rai. After successful mining, its SLURM script builds
`data/appworld_events/pool_aw2_spread_x2.jsonl` and the exposure report offline.
Existing mining output or its `.done` sidecar blocks a fresh run, as on rai.
After B and mining finish, C can run:

```bash
sbatch --export=ALL,AW2_ARMS=C scripts/aw2_bc_hpg.slurm
```

If both pools already exist, omit `AW2_ARMS` to run B then C in one allocation.
Comma-separated selections and full arm tags also work. Both job files request
one B200, 14 CPUs, 240G and six hours. To use the other account, add
`--account=yd24f.fsu --qos=yd24f.fsu` to `sbatch`. The B/C runner lock serializes
training runners that share this project; mining has its own lock.

The launchers preserve SLURM's `CUDA_VISIBLE_DEVICES`. The default vLLM port is
`20000 + SLURM_JOB_ID % 30000`, with an occupied-port check. They use the shared
HF cache at `/blue/fsu-compsci-dept/xc25.fsu/hq/tools/hf-cache` and the existing
training/vLLM environments. The official JSONNet is derived from
`qwen35-4b-base_dev.jsonnet`, replacing only `qwen35-4b-base` and `localhost:8950`.
Evaluation runs from `envs/appworld-repo` with `--root . --without-setup
--num-processes 4`. Log names match rai, including `aw1_train_<tag>.log`,
`vllm_aw1_<tag>.log`, `awoff_<tag>_dev.log` and the AW-2 runner/miner logs.
Failed official runs or missing aggregate reports fail the training job.

For an allocation-free preview of mining commands (no models or server):

```bash
SLURM_JOB_ID=12345 bash tools/aw2_mine_more_hpg.sh --dry-run
```

Local verification, including `bash -n` for every added shell/SLURM file:

```bash
.venv/bin/python -m pytest -q tests/test_aw2_hpg_scripts.py
```
