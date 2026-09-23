#!/bin/bash
# CPU/API chain: ADD-DEMO material (one more verified ReAct demo per support task, attempts 3-5, T=0.7, sweep filter at export),
# same added-budget cap as the REPAIR arm (1.5M tokens / $6), then register the usable packages and CPU-preflight.
set -u
B=/home/xueqi/hq/projects/tc-alignment-baselines; D0=/home/xueqi/hq/projects/tc-alignment/data/rtd/v1_alfworld_k32_d0
ADD=$B/data/alfworld_k32_add_demo; TOK=/home/xueqi/.cache/huggingface/hub/models--google--gemma-4-12B-it/snapshots/707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7
SECRETS=~/hq/secrets/llm_apis.env; val() { grep -E "^$1=" "$SECRETS" | head -1 | cut -d= -f2- | tr -d '"' | tr -d "'"; }
export AZURE_LLM_ENDPOINT="$(val AZURE_APIM_ENDPOINT)" AZURE_LLM_KEY="$(val AZURE_${AZURE_SLOT:-P1}_PRIMARY)"
export CUDA_VISIBLE_DEVICES="" PYTHONPATH="$B/src:$B" PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 BFAS_TEACHER=gpt-5.6-luna
cd $B; mkdir -p data
[ -e "$ADD" ] || [ -e "$ADD.collection" ] && { echo "ADD-DEMO paths exist; refusing"; exit 1; }
echo "[$(date)] ADD-DEMO collection (slot ${AZURE_SLOT:-P1})"
$B/.venv/bin/python -B tools/alfworld_teacher_pool.py --source "$D0" --out "$ADD" --method plain --candidates-per-task 1 \
  --attempt-start 3 --attempts-per-task 3 --temperature 0.7 --sweep-filter --workers 1 --rate-limit-retries 8 \
  --max-tokens 1500000 --max-usd 6 --usd-per-mtok-in 0.20 --usd-per-mtok-out 1.20 --usd-per-mtok-cached 0.01 > $B/logs/add_demo_collect.log 2>&1; st=$?
echo "[$(date)] ADD-DEMO collect exit $st"; tail -4 $B/logs/add_demo_collect.log | cut -c1-300
[ $st -eq 0 ] || exit 1
$B/.venv/bin/python -B tools/alf_bank_subset.py --source "$ADD" --output "$ADD.usable" --selection all-usable --config "$ADD.yaml" --model-path $TOK > $B/logs/add_demo_register.log 2>&1; echo "[$(date)] register exit $?"; tail -2 $B/logs/add_demo_register.log | cut -c1-200
$B/.venv/bin/python -B tools/alf_pi1_train.py --config "$ADD.yaml" --model-path $TOK --preflight > $B/logs/add_demo_preflight.log 2>&1; echo "[$(date)] preflight exit $?"
echo "[$(date)] ADD-DEMO CHAIN DONE"
