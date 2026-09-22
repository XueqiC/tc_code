#!/bin/bash
# GPU1 chain #3: wait until GPU1 (labmate's job) frees >= 85 GB, then SmartAD seed-2 training from the rai selection.
set -u
B=/home/xueqi/hq/projects/tc-alignment-baselines
SNAP=/home/xueqi/.cache/huggingface/hub/models--google--gemma-4-12B-it/snapshots/707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7
export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONPATH="$B/src:$B" PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
free_gb() { nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits -i 1 | awk -F", " '{printf "%d", ($2-$1)/1024}'; }
until [ "$(free_gb)" -ge 85 ]; do sleep 120; done
echo "[$(date)] GPU1 free ($(free_gb) GB); SmartAD seed 2 training"; cd $B; rm -rf results/smartad_v2/seed-2
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -B tools/alf_baseline.py train --method smartad --seed 2 --selection results/smartad_selection_rai/selection.json --config configs/rtd/pi1_alfworld_k32.yaml --model-path $SNAP --gpu-uuid GPU-97762062-28db-d85e-cb46-294b082f94df --output results/smartad_v2/seed-2 > logs/smartad_s2_rai.log 2>&1; echo "[$(date)] SmartAD seed-2 training exit $?"
echo "[$(date)] GPU1 CHAIN #3 DONE"
