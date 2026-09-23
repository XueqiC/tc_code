#!/bin/bash
# GPU2 chain v2 (08:10 CDT): after 1alt s1 (pid 3476643) exits, train the min-id one-alternative bank seed 1 (second seed of the identity control).
set -u
T=/home/xueqi/hq/projects/tc-alignment-taskeq
SNAP=/home/xueqi/.cache/huggingface/hub/models--google--gemma-4-12B-it/snapshots/707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7; G2=GPU-20b20454-ae9f-7860-6801-430d68842a27
export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONPATH="$T/src:$T" PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
while ps -p 3476643 >/dev/null 2>&1; do sleep 60; done
cd $T; n=all87_1min
base="$T/.venv/bin/python -B tools/alf_pi1_train.py --config configs/rtd/pi1_alfworld_k32_${n}_tok.yaml --bank $T/artifacts/alfworld_k32_$n --model-path $SNAP --gpu-uuid $G2 --seed 1 --output results/${n}_v2/seed-1"
CUDA_VISIBLE_DEVICES=2 $base --preflight > logs/${n}_s1_preflight.log 2>&1 && CUDA_VISIBLE_DEVICES=2 $base > logs/${n}_s1.log 2>&1; echo "[$(date)] $n seed 1 exit $?"
echo "[$(date)] GPU2 CHAIN v2 DONE"
