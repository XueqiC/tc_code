#!/bin/bash
# GPU4 chain #2: after the evaluation chain exits, (1) pi1 support rollouts for seeds 0/1/2 (served CE 10-pass checkpoints),
# (2) four-cell B seed 0 (D0, task-equal CE), (3) C seed 0 (all87, plain CE, token endpoints) from the taskeq tree.
set -u
B=/home/xueqi/hq/projects/tc-alignment-baselines; T=/home/xueqi/hq/projects/tc-alignment-taskeq; V=/home/xueqi/hq/projects/tc-alignment-vllm
D0=/home/xueqi/hq/projects/tc-alignment/data/rtd/v1_alfworld_k32_d0
SNAP=/home/xueqi/.cache/huggingface/hub/models--google--gemma-4-12B-it/snapshots/707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7
GPU4=GPU-aaebd5af-4015-b475-52dd-5185ca7bdb52
export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_ATTENTION_BACKEND=FLASH_ATTN VLLM_USE_FLASHINFER_SAMPLER=0
while ps -p 2006245 >/dev/null 2>&1; do sleep 60; done
for p in 2089629 2089630 2089631; do while ps -p $p >/dev/null 2>&1; do sleep 30; done; done
echo "[$(date)] eval chain exited and CE merges done; CE 10-pass seeds 0/1 on the rai platform first"
bash $V/tools/alf_eval_served_rai.sh $V/merged_v2/seed0_pass10 rai_ces0p10 4 14
bash $V/tools/alf_eval_served_rai.sh $V/merged_v2/seed1_pass10 rai_ces1p10 4 14
echo "[$(date)] support rollouts"
for s in 0 1 2; do
  M=$V/merged_v2/seed${s}_pass10; RUN=$B/data/alfworld_k32_repair_ce/seed-$s; mkdir -p $RUN
  [ -f $M/config.json ] || { echo "merged model missing for seed $s"; continue; }
  PORT=$(( 22000 + RANDOM % 8000 ))
  PYTHONPATH="$V/src:$V" $V/envs/vllm-serve/.venv/bin/python -B $V/tools/alf_vllm_server.py --model "$M" --gpu 4 --port $PORT --state-dir $RUN/server > $RUN/server.stdout 2>&1 &
  SP=$!; for i in $(seq 1 120); do [ -f $RUN/server/server.json ] && break; kill -0 $SP 2>/dev/null || break; sleep 15; done
  [ -f $RUN/server/server.json ] || { echo "[$(date)] server failed for seed $s"; tail -5 $RUN/server.stdout; kill $SP 2>/dev/null; continue; }
  CUDA_VISIBLE_DEVICES="" PYTHONPATH="$B/src:$B" $B/.venv/bin/python -B $B/tools/alf_pi1_support_rollouts.py --source $D0 --output $RUN/rollouts --server-json $RUN/server/server.json --model-path "$M" --tokenizer-path "$M" > $RUN/rollouts.log 2>&1; st=$?
  kill $SP 2>/dev/null; wait $SP 2>/dev/null
  echo "[$(date)] rollouts seed $s exit $st: $(ls $RUN/rollouts/*.json 2>/dev/null | wc -l) files"; tail -2 $RUN/rollouts.log | cut -c1-200
done
echo "[$(date)] four-cell B seed 0 (taskeq tree)"
cd $T; export PYTHONPATH="$T/src:$T"
base="$T/.venv/bin/python -B tools/alf_pi1_train.py --config configs/rtd/pi1_alfworld_k32_taskeq.yaml --bank $D0 --model-path $SNAP --gpu-uuid $GPU4 --seed 0 --output results/taskeq_v2/seed-0"
CUDA_VISIBLE_DEVICES=4 $base --preflight > logs/B_s0_preflight.log 2>&1 && CUDA_VISIBLE_DEVICES=4 $base > logs/B_s0.log 2>&1; echo "[$(date)] B seed 0 exit $?"
echo "[$(date)] four-cell C seed 0 (all87 plain CE, token endpoints)"
base="$T/.venv/bin/python -B tools/alf_pi1_train.py --config configs/rtd/pi1_alfworld_k32_all87_tok.yaml --bank $T/artifacts/alfworld_k32_smartad_all87 --model-path $SNAP --gpu-uuid $GPU4 --seed 0 --output results/all87tok_v2/seed-0"
CUDA_VISIBLE_DEVICES=4 $base --preflight > logs/C_s0_preflight.log 2>&1 && CUDA_VISIBLE_DEVICES=4 $base > logs/C_s0.log 2>&1; echo "[$(date)] C seed 0 exit $?"
echo "[$(date)] GPU4 CHAIN #2 DONE"
