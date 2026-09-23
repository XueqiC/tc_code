#!/bin/bash
# CPU/API chain: wait for seed-0 support rollouts, purchase REPAIR suffixes (Azure P1, luna), export bank, union with D0 (1:1 tokens), CPU preflight.
set -u
B=/home/xueqi/hq/projects/tc-alignment-baselines; D0=/home/xueqi/hq/projects/tc-alignment/data/rtd/v1_alfworld_k32_d0
RUN=$B/data/alfworld_k32_repair_ce/seed-0
TOK=/home/xueqi/.cache/huggingface/hub/models--google--gemma-4-12B-it/snapshots/707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7
SECRETS=~/hq/secrets/llm_apis.env
val() { grep -E "^$1=" "$SECRETS" | head -1 | cut -d= -f2- | tr -d '"' | tr -d "'"; }
export AZURE_LLM_ENDPOINT="$(val AZURE_APIM_ENDPOINT)" AZURE_LLM_KEY="$(val AZURE_${AZURE_SLOT:-P1}_PRIMARY)"
export CUDA_VISIBLE_DEVICES="" PYTHONPATH="$B/src:$B" PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
until [ -f $RUN/rollouts/index.json ] || ls $RUN/rollouts/*.json >/dev/null 2>&1 && grep -q "rollouts seed 0 exit 0" $B/logs/rai_chain_gpu4_rollouts_BC.log 2>/dev/null; do sleep 120; done
echo "[$(date)] rollouts present; collecting REPAIR suffixes (slot ${AZURE_SLOT:-P1})"
cd $B
$B/.venv/bin/python -B tools/alf_repair_collect.py --rollouts $RUN/rollouts --source $D0 --source-collection $D0.collection \
  --collection $RUN/repair.collection --output $RUN/repair --config $RUN/repair.yaml --model-path $TOK \
  --max-tokens 1500000 --max-usd 6 --attempt-index 0 --temperature 0 --rate-limit-retries 8 > $RUN/repair_collect.log 2>&1; st=$?
echo "[$(date)] repair collect exit $st"; tail -5 $RUN/repair_collect.log | cut -c1-300
[ $st -eq 0 ] || exit 1
$B/.venv/bin/python -B tools/alf_bank_union.py --left $D0 --right $RUN/repair --output $RUN/d0_plus_repair_1to1 --mixing tokens-1:1 --seed 0 \
  --config $RUN/d0_plus_repair_1to1.yaml --model-path $TOK > $RUN/union.log 2>&1; echo "[$(date)] union exit $?"; tail -3 $RUN/union.log | cut -c1-300
$B/.venv/bin/python -B tools/alf_pi1_train.py --config $RUN/d0_plus_repair_1to1.yaml --model-path $TOK --preflight > $RUN/preflight_union.log 2>&1; echo "[$(date)] union preflight exit $?"
echo "[$(date)] REPAIR S0 CHAIN DONE"
