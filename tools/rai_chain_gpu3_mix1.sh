#!/bin/bash
# GPU3 chain v4 (11:50 CDT): DIAGNOSTIC bank MIX1 = ALT1's picks on all tasks except the 4 heat tasks, which take 1min's picks (type-specific material quality test). Seed 0, training only; rai eval by chain #8 (MIX1s0).
set -u
T=/home/xueqi/hq/projects/tc-alignment-taskeq
SNAP=/home/xueqi/.cache/huggingface/hub/models--google--gemma-4-12B-it/snapshots/707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7; G3=GPU-8b270cf8-6bb4-cee0-7060-88eba83d2fb0
export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONPATH="$T/src:$T" PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd $T; n=all87_mix1
base="$T/.venv/bin/python -B tools/alf_pi1_train.py --config configs/rtd/pi1_alfworld_k32_${n}_tok.yaml --bank $T/artifacts/alfworld_k32_$n --model-path $SNAP --gpu-uuid $G3 --seed 0 --output results/${n}_v2/seed-0"
CUDA_VISIBLE_DEVICES=3 $base --preflight > logs/${n}_s0_preflight.log 2>&1 && CUDA_VISIBLE_DEVICES=3 $base > logs/${n}_s0.log 2>&1; echo "[$(date)] $n seed 0 exit $?"
echo "[$(date)] GPU3 CHAIN v4 DONE"
