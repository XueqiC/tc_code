#!/bin/bash
# After the three collect_r2 shards finish on hpg: merge there, copy the merged run to rai, build codes, fit (flat + hierarchical, all targets), report.
# Usage: bash tools/behavior_postcollect.sh [shard_dir1 shard_dir2 shard_dir3]   (paths on hpg, relative to the project root;
#   default: results/behavior_atom_v1/collect_r2/shard-0000{0,1,2}-of-00003)
set -uo pipefail
cd "$(dirname "$0")/.."
R=/blue/fsu-compsci-dept/xc25.fsu/hq/tc-alignment
S0=${1:-results/behavior_atom_v1/collect_r2/shard-00000-of-00003}; S1=${2:-results/behavior_atom_v1/collect_r2/shard-00001-of-00003}; S2=${3:-results/behavior_atom_v1/collect_r2/shard-00002-of-00003}
ssh -o BatchMode=yes hpg "cd $R && source .venv/bin/activate && PYTHONPATH=src python tools/behavior_atom_experiment.py merge --shards $S0 $S1 $S2 --output-dir results/behavior_atom_v1/collect_r2_merged" || { echo "[postcollect] merge failed"; exit 1; }
mkdir -p results/behavior_atom_v1; rsync -a -e ssh hpg:$R/results/behavior_atom_v1/collect_r2_merged/ results/behavior_atom_v1/collect_r2_merged/ || exit 1
PYTHONPATH=src .venv/bin/python tools/behavior_atom_experiment.py codes --run-dir results/behavior_atom_v1/collect_r2_merged --fisher data/behavior_atom_v1/fisher_v3t_student_full.npz --sources data/behavior_atom_v1/sources_pilot.json --folds data/behavior_atom_v1/folds.json --output-dir results/behavior_atom_v1/collect_r2_codes || exit 1
echo '{"responses": "responses.jsonl", "codes": "codes_all.npz"}' > results/behavior_atom_v1/collect_r2_codes/fit_input.json
for m in flat hierarchical; do PYTHONPATH=src .venv/bin/python tools/behavior_atom_experiment.py fit --config configs/behavior_atom/fit_collect_r2.json --model $m --target all --output-dir results/behavior_atom_v1/collect_r2_fit_$m || echo "[postcollect] fit $m failed"; done
PYTHONPATH=src .venv/bin/python tools/behavior_atom_experiment.py report --config configs/behavior_atom/fit_collect_r2.json --output-dir results/behavior_atom_v1/collect_r2_fit_hierarchical 2>&1 | tail -3
echo "[postcollect] DONE"
