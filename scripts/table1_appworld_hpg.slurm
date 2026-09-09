#!/bin/bash
#SBATCH --job-name=t1aw
#SBATCH --account=fsu-compsci-dept
#SBATCH --qos=fsu-compsci-dept
#SBATCH --partition=hpg-b200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%A_%a.out
# Table 1 AppWorld baselines under the unified budget (B=22,633, deployment-exact pools), plus
# the initial student evaluated on the same node class. Array index: arms x 3 seeds, last = base.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export HF_HOME=/blue/fsu-compsci-dept/xc25.fsu/hq/tools/hf-cache
source .venv/bin/activate
export APPWORLD_ROOT=$PWD/envs/appworld-data
export AW_MAX_PROMPT_TOKENS=4096 AW_TRUNCATE_SIDE=tail PYTHONPATH=$PWD/src:$PWD
ARMS=(sft sad pbsd_insp pbsd_agent star)
CAP=22633
if [ "$SLURM_ARRAY_TASK_ID" -ge $(( ${#ARMS[@]} * 3 )) ]; then
  tag="awB${CAP}_base"
  python src/appworld_eval.py --model Qwen/Qwen3.5-4B --split dev --max-tasks 40 --max-steps 24 --seed 0 --tag "${tag}_eval"
  echo "[t1aw] $tag DONE"; exit 0
fi
ARM=${ARMS[$((SLURM_ARRAY_TASK_ID / 3))]}; SEED=$((SLURM_ARRAY_TASK_ID % 3)); tag="awB${CAP}_${ARM}_s${SEED}"
POOLDIR=$PWD/results/table1_audit/pools_native/appworld/B${CAP}_seed${SEED}
case $ARM in
  sft)        export AW_POOL_PATH=$POOLDIR/pool_sft.jsonl ;;
  sad)        export AW_POOL_PATH=$POOLDIR/pool_sad.jsonl AW_DISTILL=sad ;;
  pbsd_insp)  export AW_POOL_PATH=$POOLDIR/pool_pbsd_insp.jsonl AW_DISTILL=pbsd ;;
  pbsd_agent) export AW_POOL_PATH=$POOLDIR/pool_pbsd_agent_costed.jsonl AW_DISTILL=pbsd_agent AW_GRAD_CKPT=1 ;;
  star)       export AW_POOL_PATH=$PWD/data/appworld_sft/pool_star.jsonl ;;
esac
python src/appworld_train.py --selection full --budget 999999 --seed $SEED --student Qwen/Qwen3.5-4B --tag "$tag"
python src/appworld_eval.py --model "results/appworld_students/$tag/adapter" --split dev --max-tasks 40 --max-steps 24 --seed 0 --tag "${tag}_eval"
echo "[t1aw] $tag DONE"
