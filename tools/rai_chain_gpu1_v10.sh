#!/bin/bash
# GPU1 chain #10 (10:15 CDT): evaluate the mike-trained ALT1 s2 and PER2 s2 endpoints on rai (adapters staged in tc-alignment-vllm/adapters_hpg),
# then train the min-id alternative bank seed 2 (third seed of the identity control) and evaluate it on rai. merge -> eval -> delete merge.
set -u
T=/home/xueqi/hq/projects/tc-alignment-taskeq; V=/home/xueqi/hq/projects/tc-alignment-vllm
SNAP=/home/xueqi/.cache/huggingface/hub/models--google--gemma-4-12B-it/snapshots/707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7; G1=GPU-97762062-28db-d85e-cb46-294b082f94df
export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONPATH="$T/src:$T" PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
ev() { a=$1; tag=$2; [ -f $a/adapter_model.safetensors ] || { echo "[$(date)] missing $a"; return; }; rm -rf $V/merged_v2/$tag; CUDA_VISIBLE_DEVICES="" $V/.venv/bin/python -B $V/tools/alf_merge_lora.py --base $SNAP --adapter $a --out $V/merged_v2/$tag > $V/logs/merge_$tag.log 2>&1 || { echo "[$(date)] merge failed $tag"; return; }; bash $V/tools/alf_eval_served_rai.sh $V/merged_v2/$tag rai_$tag 1 14; echo "[$(date)] EVAL_DONE rai_$tag"; rm -rf $V/merged_v2/$tag; }
ev $V/adapters_hpg/ALT1_s2/lora ALT1s2t96490
ev $V/adapters_hpg/PER2_s2/lora PER2s2t96490
cd $T; n=all87_1min
base="$T/.venv/bin/python -B tools/alf_pi1_train.py --config configs/rtd/pi1_alfworld_k32_${n}_tok.yaml --bank $T/artifacts/alfworld_k32_$n --model-path $SNAP --gpu-uuid $G1 --seed 2 --output results/${n}_v2/seed-2"
CUDA_VISIBLE_DEVICES=1 $base --preflight > logs/${n}_s2_preflight.log 2>&1 && CUDA_VISIBLE_DEVICES=1 $base > logs/${n}_s2.log 2>&1; echo "[$(date)] $n seed 2 exit $?"
ev $T/results/${n}_v2/seed-2/tokens-96490/lora ALT1MINs2t96490
echo "[$(date)] GPU1 CHAIN #10 DONE"
