#!/bin/bash
# GPU1 chain #9: after the cross-check chain #8 exits, train the material-size arms (seed 0) on GPU1: all87 1-alternative-per-task, then 2-per-task;
# then evaluate both high endpoints on the rai platform (merge -> eval -> delete merge).
set -u
T=/home/xueqi/hq/projects/tc-alignment-taskeq; V=/home/xueqi/hq/projects/tc-alignment-vllm
SNAP=/home/xueqi/.cache/huggingface/hub/models--google--gemma-4-12B-it/snapshots/707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7; G1=GPU-97762062-28db-d85e-cb46-294b082f94df
export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONPATH="$T/src:$T" PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
while ps -p 3174739 >/dev/null 2>&1; do sleep 120; done
cd $T
for n in all87_1alt all87_2per; do
  base="$T/.venv/bin/python -B tools/alf_pi1_train.py --config configs/rtd/pi1_alfworld_k32_${n}_tok.yaml --bank $T/artifacts/alfworld_k32_$n --model-path $SNAP --gpu-uuid $G1 --seed 0 --output results/${n}_v2/seed-0"
  CUDA_VISIBLE_DEVICES=1 $base --preflight > logs/${n}_s0_preflight.log 2>&1 && CUDA_VISIBLE_DEVICES=1 $base > logs/${n}_s0.log 2>&1; echo "[$(date)] $n seed 0 exit $?"
done
for n in all87_1alt all87_2per; do a=$T/results/${n}_v2/seed-0/tokens-96490/lora; [ -f $a/adapter_model.safetensors ] || continue
  rm -rf $V/merged_v2/${n}_s0; CUDA_VISIBLE_DEVICES="" $V/.venv/bin/python -B $V/tools/alf_merge_lora.py --base $SNAP --adapter $a --out $V/merged_v2/${n}_s0 > $V/logs/merge_${n}_s0.log 2>&1 && bash $V/tools/alf_eval_served_rai.sh $V/merged_v2/${n}_s0 rai_${n}s0t96490 1 14; rm -rf $V/merged_v2/${n}_s0
done
echo "[$(date)] GPU1 CHAIN #9 DONE"
