#!/bin/bash
#SBATCH --job-name=bfclbase
#SBATCH --account=fsu-compsci-dept
#SBATCH --qos=fsu-compsci-dept
#SBATCH --partition=hpg-b200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=200gb
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
# CRCD P0-1: second full official BFCL measurement of the untrained base (rai gave 46.27)
# to establish the base confidence interval before the B-ladder.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export HF_HOME=/blue/fsu-compsci-dept/xc25.fsu/hq/tools/hf-cache
source .venv/bin/activate
bash tools/bfcl_std_campaign.sh 0 9070 base
[ -f results/bfcl_std/base/data_overall.csv ] || { echo "[bfclbase] NO SCORE PRODUCED"; exit 1; }
python3 -c "
import csv; r=list(csv.DictReader(open('results/bfcl_std/base/data_overall.csv')))[0]
print('[bfclbase] base OVERALL=' + r['Overall Acc'])"
echo "[bfclbase] DONE"
