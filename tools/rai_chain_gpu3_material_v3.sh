#!/bin/bash
# GPU3 (A100) chain v3 (06:20 CDT): after le50tok s2 (pid 3206637) exits, train the min-id one-alternative bank seed 0 (identity control for ALT1), then all87_2per seed 1.
set -u
T=/home/xueqi/hq/projects/tc-alignment-taskeq
SNAP=/home/xueqi/.cache/huggingface/hub/models--google--gemma-4-12B-it/snapshots/707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7; G3=GPU-8b270cf8-6bb4-cee0-7060-88eba83d2fb0
export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONPATH="$T/src:$T" PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
while ps -p 3206637 >/dev/null 2>&1; do sleep 120; done
cd $T
for spec in all87_1min:0 all87_2per:1; do n=${spec%%:*}; sd=${spec#*:}
  base="$T/.venv/bin/python -B tools/alf_pi1_train.py --config configs/rtd/pi1_alfworld_k32_${n}_tok.yaml --bank $T/artifacts/alfworld_k32_$n --model-path $SNAP --gpu-uuid $G3 --seed $sd --output results/${n}_v2/seed-$sd"
  CUDA_VISIBLE_DEVICES=3 $base --preflight > logs/${n}_s${sd}_preflight.log 2>&1 && CUDA_VISIBLE_DEVICES=3 $base > logs/${n}_s${sd}.log 2>&1; echo "[$(date)] $n seed $sd exit $?"
done
echo "[$(date)] GPU3 CHAIN v3 DONE"
