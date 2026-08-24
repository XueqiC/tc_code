#!/bin/bash
#SBATCH --job-name=tc-basebench
#SBATCH --account=fsu-compsci-dept
#SBATCH --qos=fsu-compsci-dept
#SBATCH --partition=hpg-b200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=120gb
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%A_%a.out

# Manifest line: ARM=<vanilla|sad|agentkd|star|bbopd|base> SEED=<n>
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export HF_HOME=/blue/fsu-compsci-dept/xc25.fsu/hq/tools/hf-cache
mkdir -p logs results
source .venv/bin/activate

manifest=${BB_MANIFEST:-configs/baseline_bench_manifest.txt}
line_number=$((SLURM_ARRAY_TASK_ID + 1))
line=$(sed -n "${line_number}p" "$manifest")
[ -n "$line" ] || { echo "no manifest line $line_number"; exit 1; }
unset ARM SEED
read -r -a kv <<< "$line"
export "${kv[@]}"
echo "[basebench] ARM=$ARM SEED=$SEED"

PROJ=$PWD
POOL=data/appworld_sft/pool.jsonl
unset AW_DISTILL AW_POOL_PATH
case "$ARM" in
  vanilla) ;;
  sad)     export AW_DISTILL=sad ;;
  agentkd) export AW_DISTILL=agentkd; POOL=data/appworld_sft/pool_agentkd.jsonl ;;
  star)    POOL=data/appworld_sft/pool_star.jsonl ;;
  bbopd)   POOL=data/appworld_sft/pool_bbopd.jsonl ;;
  base)    ;;
  *) echo "unknown ARM $ARM"; exit 1 ;;
esac

TAG="hb_${ARM}_s${SEED}"
if [ "$ARM" = "base" ]; then
  MODEL="Qwen/Qwen3.5-4B"
else
  export AW_POOL_PATH=$PROJ/$POOL
  MODEL="results/appworld_students/$TAG/adapter"
  if [ ! -d "$MODEL" ]; then
    python src/appworld_train.py --selection full --budget 999999 \
      --seed "$SEED" --student Qwen/Qwen3.5-4B --tag "$TAG"
  fi
fi

# ALFWorld unseen split with the official config shipped in the package.
export ALFWORLD_DATA=/blue/fsu-compsci-dept/xc25.fsu/hq/tc-alignment/envs/alfworld/data
ALF_VENV=envs/alfworld/.venv
ALF_CFG=$(ALFWORLD_DATA=$ALFWORLD_DATA $ALF_VENV/bin/python - << 'PY'
import alfworld, os, glob
base = os.path.dirname(alfworld.__file__)
hits = glob.glob(os.path.join(base, "**", "base_config.yaml"), recursive=True)
hits += glob.glob(os.path.join(os.environ["ALFWORLD_DATA"], "**", "base_config.yaml"), recursive=True)
print(hits[0] if hits else "")
PY
)
[ -n "$ALF_CFG" ] || { echo "no alfworld config found"; exit 1; }
# The eval driver needs torch+transformers (train venv) AND alfworld (bench
# venv). Run it in the train venv with the bench venv's site-packages added.
ALF_SITE=$($ALF_VENV/bin/python -c "import site;print(site.getsitepackages()[0])")
export ALFWORLD_CONFIG=$ALF_CFG
PYTHONPATH="$ALF_SITE:${PYTHONPATH:-}" python src/alfworld_eval.py \
  --model "$MODEL" --num-games 134 --max-steps 40 \
  --tag "${TAG}_alf" --seed 0
echo "[basebench] $TAG DONE"
