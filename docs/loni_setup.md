# LONI QB-D (qbd.loni.org) — access and cluster survey, 2026-09-12

## Access
- rai reaches `qbd.loni.org:22` **directly**; no tunnel (unlike HiPerGator).
- Key authentication works from rai: `ssh loni` (config in `~/.ssh/config`, user `xueqic`,
  `IdentityFile ~/.ssh/id_ed25519`, ControlMaster with `ControlPersist 8h`).
- Login nodes round-robin between qbd1 and qbd2. MFA (Microsoft Authenticator) is only needed for
  password logins, not for the installed key.
- The account password was shared in chat on 2026-09-12 and should be rotated; it is not stored in
  this workspace.

## Hardware and partitions
| Partition | Nodes | GPUs/node | Time limit |
|---|---|---|---|
| single (default) | 480 | none | 7 days |
| workq / checkpt | 480 | none | 3 days |
| bigmem | 5 | none | 3 days |
| gpu2 | 50 | 2 | 3 days |
| gpu4 | 10 | 4 | 3 days |

GPU node (qbd512): 64 CPU cores, 514 GB RAM, `Gres=gpu:4` (no type string; exact GPU model still
unverified because no job could be submitted, see below). Occupancy at survey time: gpu2 30 of 50
nodes idle, gpu4 7 of 10 idle. QOS names: normal, gpu2, gpu4, devel (30 min), limit1.

## Storage
| Path | Quota | Used |
|---|---|---|
| /home/xueqic | 10 GB | 959 MB |
| /work/xueqic (Lustre, 6.5 PB) | no MB quota, 4M files | 1.08 TB, 417k files |
| /scratch/xueqic, /project | same Lustre backend | |

Put venvs, HF caches and results on `/work`; home is far too small for model weights.

## Software
- Modules: `cuda/12.2.1` (default, up to 12.2), `gcc/11.4.0`, `gcc/13.2.0`, `python/3.11.5-anaconda`.
- System python3 is 3.13.2; `uv` is not installed; `rsync` and `git` are present.
- No `singularity` or `apptainer` on the login node.
- Outbound HTTPS works from the login node (huggingface.co returns 200), so model downloads are possible.

## Blocker: no active allocation
```
CPU Allocation SUs:  remaining   allocated  expiration
    loni_depedlab11:  -4538.83   150000.00  2026-10-01
```
`sbatch --test-only` is refused:
```
sbatch: error: No active CPU Allocation found. Verify by running: showquota
allocation failure: Invalid account or account/partition combination specified
```
The allocation is overdrawn, so **no job of any size can be submitted**. A new or renewed allocation
must be requested through the LONI portal (allocations.loni.org) before this cluster is usable.

## Prepared on our side
- `scripts/sync_to_loni.sh` — rsync of the project to `/work/xueqic/hq/<project>` (excludes data,
  results, logs, venvs, runs).
- `scripts/job_loni.slurm` — SLURM template; account and partition are placeholders until the
  allocation exists and the GPU model is confirmed.
- Survey script: scratchpad `loni_survey.sh` (read-only).

## Once an allocation exists
1. Confirm the GPU model and driver: `srun -A <alloc> -p gpu4 --gres=gpu:1 -t 00:05:00 nvidia-smi`.
2. Build the environment on /work (python 3.11 module, torch wheels matching the driver), stage the
   gemma-4-12B weights into `/work/xueqic/hf-cache`.
3. Move the slow, embarrassingly parallel work there first: the ALFWorld baseline curves (each cell
   is about 2 hours on one rai GPU, and four budgets times three methods times three benchmarks is
   far more than rai can absorb), then the mechanism-validation training and evaluation chains.
