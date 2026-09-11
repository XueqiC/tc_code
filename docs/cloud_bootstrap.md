# Cloud ALFWorld setup

On an x86_64 Ubuntu GPU rental, mount a writable persistent volume at
`VOLUME_ROOT` (default `/workspace`) and put the repo at
`$VOLUME_ROOT/tc-alignment`. Use one 80–96 GB GPU, a driver compatible with the
installed Torch CUDA 13.0 runtime, and root/passwordless sudo for missing Ubuntu
build packages. Allow space for three venvs, model weights, caches and checkpoints
(start with 200 GB). Bootstrap launches no GPU workload and leaves the driver alone.

From **rai**, set an SSH alias `cloud` for the rental, then sync this checkout.
The exclusions mirror `scripts/sync_to_hpg.sh`; omit trailing `/` so they also
exclude this worktree's symlinks and `.git` file:

```bash
SRC=/home/xueqi/hq/projects/tc-alignment-uni
REMOTE_VOLUME=/workspace   # must match VOLUME_ROOT on the rental
ssh cloud "mkdir -p '$REMOTE_VOLUME/tc-alignment/logs'"
rsync -az --info=stats1 \
  --exclude '.git' --exclude 'data' --exclude 'results' --exclude 'logs' \
  --exclude '.venv' --exclude '__pycache__' --exclude 'wandb' \
  --exclude '_trash' --exclude 'envs' \
  "$SRC/" "cloud:$REMOTE_VOLUME/tc-alignment/"

# Both configs need this complete sealed bank; code sync deliberately excludes it.
ssh cloud "mkdir -p '$REMOTE_VOLUME/tc-alignment/data/rtd/v1_1_alfworld'"
rsync -az --info=stats1 "$SRC/data/rtd/v1_1_alfworld/" \
  "cloud:$REMOTE_VOLUME/tc-alignment/data/rtd/v1_1_alfworld/"
```

On the **rental**, export a Hugging Face token whose account has access to
`google/gemma-4-12B-it` (accept the model's access conditions first):

```bash
export VOLUME_ROOT=/workspace
read -rs -p 'HF token: ' HF_TOKEN; echo
export HF_TOKEN
bash "$VOLUME_ROOT/tc-alignment/scripts/cloud_bootstrap.sh"
unset HF_TOKEN
source "$VOLUME_ROOT/cloud_env.sh"
cd "$ROOT"
CUDA_VISIBLE_DEVICES='' "$RTD_PYTHON" tools/rtd_experiment.py audit \
  --config configs/rtd/unified_alfworld_gemma4.yaml
"$RTD_PYTHON" tools/rtd_experiment.py smoke \
  --config configs/rtd/unified_alfworld_gemma4.yaml --arm D3 \
  --run-dir results/rtd_unified/smoke_alf_D3 --smoke-deadline-seconds 7200
```

Use fresh run directories and the same GPU class for compared arms; cloud records
its own installation/hardware identity, so do not resume an hpg campaign here.
`audit` checks bank content, not GPU readiness or all runtime identities. If a
historical C26 environment/tokenizer identity is rejected, preserve the archive
and reverify into new outputs, then follow the [D16 conversion](rtd_v1_1_d16.md).
That requires the original paid archive/ledger; C26 verification also needs a
bind-mounted `envs/` view because it rejects symlinks escaping the repo. Bootstrap
does not change certificates, acquire teacher data or create the sealed bank.

The setup follows `scripts/rtd_v11_run_hpg.slurm`,
`scripts/rtd_v11_controls_hpg.slurm`, `tools/aw_hpg_common.sh`,
`docs/rtd_v1_1_d16.md`, both ALFWorld YAMLs, `PROJECT_STATE.md`, and local package
metadata. All paths below are relative to `VOLUME_ROOT`:

- `venvs/train`: uv-managed Python **3.12.13**, matching `.venv/pyvenv.cfg`.
  `requirements/cloud_train.txt` was generated from an offline `uv pip freeze`
  of the current `.venv`, filtered to unified ALFWorld imports and their runtime
  dependencies: Torch **2.13.0**, Transformers **5.14.1**, PEFT **0.20.0**,
  NumPy/SciPy, tokenizer/Hub dependencies and CUDA libraries.
- `envs/alfworld/.venv`: ALFWorld **0.4.2**, TextWorld **1.7.0**, PDDL engine and
  dependencies pinned in `requirements/cloud_alfworld.txt`. The local worker is
  Python 3.11.7; cloud uses 3.12.13 to follow the hpg 3.12 worker noted in
  `PROJECT_STATE.md`, with the training venv's setuptools pin. Downloads/scratch
  stay in `envs/alfworld`; `ALFWORLD_DATA` points at its `data/`.
- `envs/vllm-serve/.venv`: vLLM **0.27.1**, Transformers **5.14.1**, Torch
  **2.13.0**, ninja **1.13.0**; its remaining dependencies are resolver-managed.
  Training uses `hf_generate`; vLLM serves the official greedy ReAct evaluation
  (`valid_seen`, 140 tasks, 40 steps). The configs retain the 60 GB memory budget.
- `hf-cache`: complete Gemma snapshot. The first download records its revision in
  `hf-cache/gemma-4-12B-it.revision`; reruns keep that revision and resume missing
  files. To match an existing campaign, seed that file with its exact commit
  before the first bootstrap. `HF_TOKEN` is neither logged nor written to disk.
- `cloud_env.sh`: sources training activation, then reproduces hpg's serving-first
  `PATH`, `PYTHONPATH`, single-thread settings, allocator settings and runtime
  cache variables under the volume, both Hub cache aliases, offline model loading
  and GPU 0 by default, honoring explicit visibility.
  Always invoke **`"$RTD_PYTHON"`** for training: bare `python` resolves to serving.

Repo `.venv` and `envs` are compatibility links; their contents live on the
volume outside the repo. Conflicting existing paths are refused. Reruns reuse
venvs, check installed dependencies and skip verified ALFWorld downloads.
Keep the same absolute volume path across restarts. The bootstrap log is
`$VOLUME_ROOT/logs/cloud_bootstrap.log`; source `cloud_env.sh` in every new shell.

The idle watchdog is **opt-in**. Start it on the rental after bootstrap/downloads,
with a threshold longer than expected CPU-only intervals:

```bash
bash scripts/cloud_idle_stop.sh 30 --dry-run  # watch normally; only log the stop
# Or arm shutdown in the background (do not run alongside the dry run):
nohup bash scripts/cloud_idle_stop.sh 30 >/dev/null 2>&1 &
IDLE_STOP_PID=$!
# Cancel later in this shell: kill "$IDLE_STOP_PID"
```

It polls every 30 seconds, counts Python CUDA processes even at 0% utilization,
and recognizes renamed vLLM workers. Failed/ambiguous GPU queries reset the idle
timer. After 30 continuous idle minutes it rechecks, then calls
`runpodctl stop pod "$RUNPOD_POD_ID"` when available, otherwise `shutdown -h now`
(root or passwordless sudo). RunPod containers need `runpodctl` plus credentials
and `RUNPOD_POD_ID`; guest shutdown alone may not stop provider billing. Logs go to
`$VOLUME_ROOT/logs/cloud_idle_stop.log`; a volume lock prevents duplicate watchers.
