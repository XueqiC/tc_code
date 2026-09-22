#!/bin/bash
# GPU1 chain #2: after the SmartAD seed-2 chain exits, four-cell D seed 0 (all87 task-equal) then B seed 1 (taskeq tree).
set -u
T=/home/xueqi/hq/projects/tc-alignment-taskeq; D0=/home/xueqi/hq/projects/tc-alignment/data/rtd/v1_alfworld_k32_d0
SNAP=/home/xueqi/.cache/huggingface/hub/models--google--gemma-4-12B-it/snapshots/707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7
GPU1=GPU-97762062-28db-d85e-cb46-294b082f94df
export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONPATH="$T/src:$T" PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
while ps -p 2006246 >/dev/null 2>&1; do sleep 60; done
echo "[$(date)] GPU1 free; D seed 0"; cd $T
base="$T/.venv/bin/python -B tools/alf_pi1_train.py --config configs/rtd/pi1_alfworld_k32_all87_taskeq.yaml --bank $T/artifacts/alfworld_k32_smartad_all87 --model-path $SNAP --gpu-uuid $GPU1 --seed 0 --output results/all87taskeq_v2/seed-0"
CUDA_VISIBLE_DEVICES=1 $base --preflight > logs/D_s0_preflight.log 2>&1 && CUDA_VISIBLE_DEVICES=1 $base > logs/D_s0.log 2>&1; echo "[$(date)] D seed 0 exit $?"
base="$T/.venv/bin/python -B tools/alf_pi1_train.py --config configs/rtd/pi1_alfworld_k32_taskeq.yaml --bank $D0 --model-path $SNAP --gpu-uuid $GPU1 --seed 1 --output results/taskeq_v2/seed-1"
CUDA_VISIBLE_DEVICES=1 $base --preflight > logs/B_s1_preflight.log 2>&1 && CUDA_VISIBLE_DEVICES=1 $base > logs/B_s1.log 2>&1; echo "[$(date)] B seed 1 exit $?"
echo "[$(date)] GPU1 CHAIN #2 DONE"
