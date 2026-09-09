#!/bin/bash
# Submit the P2 collect shards to both hpg accounts once the pilot has passed. Usage: bash tools/behavior_collect_submit.sh [pilot_pass.json path on hpg]
set -uo pipefail
cd "$(dirname "$0")/.."
R=/blue/fsu-compsci-dept/xc25.fsu/hq/tc-alignment
PP=${1:-$R/results/behavior_atom_v1/pilot_r1/pilot_pass.json}
bash scripts/sync_to_hpg.sh >/dev/null 2>&1
scp -q data/behavior_atom_v1/sources_pilot.json data/behavior_atom_v1/probes_disc.json data/behavior_atom_v1/folds.json data/behavior_atom_v1/collect_data_manifest.json hpg:$R/data/behavior_atom_v1/
ssh -o BatchMode=yes hpg "cd $R && python3 -c \"import json,sys; d=json.load(open('$PP')); print('pilot pass:', d.get('pass'), 'hash', d.get('frozen_config_hash','')[:12]); sys.exit(0 if d.get('pass') else 3)\" && SHARD=0/2 sbatch --parsable --export=ALL,SHARD=0/2 scripts/behavior_collect_fsu-compsci-dept_hpg.slurm && SHARD=1/2 sbatch --parsable --export=ALL,SHARD=1/2 scripts/behavior_collect_yd24f_hpg.slurm" 2>&1 | grep -v "cannot stat\|settings\|limit-remote"
