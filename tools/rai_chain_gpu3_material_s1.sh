#!/bin/bash
# GPU3 (A100) chain: after le50tok s2 exits, train the material-size arms seed 1 (1alt then 2per), then evaluate on GPU3? No — evaluation stays on the Blackwell class;
# the endpoints are evaluated later by the GPU1/GPU4 rai chains. Training only here.
set -u
T=/home/xueqi/hq/projects/tc-alignment-taskeq
SNAP=/home/xueqi/.cache/huggingface/hub/models--google--gemma-4-12B-it/snapshots/707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7; G3=GPU-8b270cf8-6bb4-cee0-7060-88eba83d2fb0
export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONPATH="$T/src:$T" PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
while ps -p 3206637 >/dev/null 2>&1; do sleep 120; done
cd $T
for n in all87_1alt all87_2per; do
  base="$T/.venv/bin/python -B tools/alf_pi1_train.py --config configs/rtd/pi1_alfworld_k32_${n}_tok.yaml --bank $T/artifacts/alfworld_k32_$n --model-path $SNAP --gpu-uuid $G3 --seed 1 --output results/${n}_v2/seed-1"
  CUDA_VISIBLE_DEVICES=3 $base --preflight > logs/${n}_s1_preflight.log 2>&1 && CUDA_VISIBLE_DEVICES=3 $base > logs/${n}_s1.log 2>&1; echo "[$(date)] $n seed 1 exit $?"
done
echo "[$(date)] GPU3 CHAIN DONE"
