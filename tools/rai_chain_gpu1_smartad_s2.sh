#!/bin/bash
# GPU1 chain: wait for le50 trainings, then SmartAD selection (rai tree identity) and SmartAD seed-2 training from scratch.
set -u
B=/home/xueqi/hq/projects/tc-alignment-baselines
SNAP=/home/xueqi/.cache/huggingface/hub/models--google--gemma-4-12B-it/snapshots/707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7
export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONPATH="$B/src:$B" PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
until [ -f $B/results/sweep_le50_v2/seed-0/pass-10/manifest.json ] && [ -f $B/results/sweep_le50_v2/seed-1/pass-10/manifest.json ]; do sleep 60; done
while ps -p 1919302 >/dev/null 2>&1 || ps -p 1919303 >/dev/null 2>&1; do sleep 30; done
echo "[$(date)] le50 trainings finished; SmartAD selection on GPU1"
cd $B
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -B tools/alf_baseline.py select --output results/smartad_selection_rai --bank artifacts/alfworld_k32_smartad_n3 --collection artifacts/alfworld_k32_smartad_n3.collection --config configs/rtd/pi1_alfworld_k32.yaml --model-path $SNAP --gpu-uuid GPU-97762062-28db-d85e-cb46-294b082f94df > logs/smartad_select_rai.log 2>&1; st=$?
[ $st -eq 0 ] && [ -f results/smartad_selection_rai/selection.json ] || { echo "[$(date)] selection failed ($st)"; exit 1; }
echo "[$(date)] selection done; training SmartAD seed 2 on GPU1"
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -B tools/alf_baseline.py train --method smartad --seed 2 --selection results/smartad_selection_rai/selection.json --config configs/rtd/pi1_alfworld_k32.yaml --model-path $SNAP --gpu-uuid GPU-97762062-28db-d85e-cb46-294b082f94df --output results/smartad_v2/seed-2 > logs/smartad_s2_rai.log 2>&1; st=$?
echo "[$(date)] SmartAD seed-2 training exit $st"
