#!/bin/bash
# CPU/API chain: after the ADD-DEMO chain exits, REPAIR collections for seeds 1 and 2 (same rule and cap as seed 0), sequentially.
set -u
B=/home/xueqi/hq/projects/tc-alignment-baselines; D0=/home/xueqi/hq/projects/tc-alignment/data/rtd/v1_alfworld_k32_d0
TOK=/home/xueqi/.cache/huggingface/hub/models--google--gemma-4-12B-it/snapshots/707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7
SECRETS=~/hq/secrets/llm_apis.env; val() { grep -E "^$1=" "$SECRETS" | head -1 | cut -d= -f2- | tr -d '"' | tr -d "'"; }
export AZURE_LLM_ENDPOINT="$(val AZURE_APIM_ENDPOINT)" AZURE_LLM_KEY="$(val AZURE_${AZURE_SLOT:-P1}_PRIMARY)"
export CUDA_VISIBLE_DEVICES="" PYTHONPATH="$B/src:$B" PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
true
cd $B
for s in 1 2; do RUN=$B/data/alfworld_k32_repair_ce/seed-$s
  [ -f $RUN/rollouts/index.json ] || { echo "[$(date)] seed $s rollouts missing"; continue; }
  echo "[$(date)] REPAIR seed $s collection"
  $B/.venv/bin/python -B tools/alf_repair_collect.py --rollouts $RUN/rollouts --source $D0 --source-collection $D0.collection --collection $RUN/repair.collection --output $RUN/repair --config $RUN/repair.yaml --model-path $TOK --max-tokens 1500000 --max-usd 6 --attempt-index 0 --temperature 0 --rate-limit-retries 8 > $RUN/repair_collect.log 2>&1; st=$?
  echo "[$(date)] seed $s repair collect exit $st: $(grep -oE '"(requests|usable_packages)": ?[0-9]+' $RUN/repair_collect.log | tr '\n' ' ')"
  [ $st -eq 0 ] && $B/.venv/bin/python -B tools/alf_bank_union.py --left $D0 --right $RUN/repair --output $RUN/d0_plus_repair_1to1 --mixing tokens-1:1 --seed 0 --config $RUN/d0_plus_repair_1to1.yaml --model-path $TOK > $RUN/union.log 2>&1 && echo "[$(date)] seed $s union ok"
done
echo "[$(date)] REPAIR S1/S2 CHAIN DONE"
