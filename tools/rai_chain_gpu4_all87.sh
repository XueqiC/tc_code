#!/bin/bash
# GPU4 follow-up chain: after the evaluation chain exits, train all87 seed 0 to its 3-pass endpoint only
# (our own trainer process is stopped once pass-3 is saved; pass-10 would need ~16 h), then merge + evaluate it on GPU4.
set -u
B=/home/xueqi/hq/projects/tc-alignment-baselines; V=/home/xueqi/hq/projects/tc-alignment-vllm
SNAP=/home/xueqi/.cache/huggingface/hub/models--google--gemma-4-12B-it/snapshots/707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7
export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONPATH="$B/src:$B" PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
while ps -p 2006245 >/dev/null 2>&1; do sleep 60; done
echo "[$(date)] GPU4 evaluation chain exited; all87 seed 0 preflight"
cd $B
base="$B/.venv/bin/python -B tools/alf_pi1_train.py --config configs/rtd/pi1_alfworld_k32_smartad_all87.yaml --bank artifacts/alfworld_k32_smartad_all87 --model-path $SNAP --gpu-uuid GPU-aaebd5af-4015-b475-52dd-5185ca7bdb52 --seed 0 --output results/all87_v2/seed-0"
CUDA_VISIBLE_DEVICES=4 $base --preflight > logs/all87_s0_preflight.log 2>&1 || { echo "[$(date)] all87 preflight failed"; exit 1; }
CUDA_VISIBLE_DEVICES=4 $base > logs/all87_s0_rai.log 2>&1 & T=$!
echo "[$(date)] all87 seed 0 training pid $T (stopping after pass-3)"
until [ -f results/all87_v2/seed-0/pass-3/manifest.json ] && [ -f results/all87_v2/seed-0/pass-3/lora/adapter_model.safetensors ]; do ps -p $T >/dev/null || { echo "[$(date)] trainer exited before pass-3"; exit 1; }; sleep 60; done
sleep 90; kill $T 2>/dev/null; wait $T 2>/dev/null; echo "[$(date)] pass-3 saved; trainer stopped"
CUDA_VISIBLE_DEVICES="" $V/.venv/bin/python -B $V/tools/alf_merge_lora.py --base $SNAP --adapter results/all87_v2/seed-0/pass-3/lora --out $V/merged_v2/all87_seed0_pass3 > $V/logs/merge_all87_s0p3.log 2>&1 || { echo "merge failed"; exit 1; }
bash $V/tools/alf_eval_served_rai.sh $V/merged_v2/all87_seed0_pass3 rai_all87s0p3 4 14
echo "[$(date)] ALL87 CHAIN DONE"
