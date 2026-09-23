#!/bin/bash
# GPU4 chain #8: after chain #7 (pid 3488082) exits, evaluate on rai every material-arm 96,490 endpoint that appears in the taskeq tree
# (1alt s0/s1, 2per s0/s1; s1 trained on GPU2/GPU3), skipping tags already evaluated; poll every 10 min for up to 9 h. merge -> eval (util 0.80) -> delete merge.
set -u
T=/home/xueqi/hq/projects/tc-alignment-taskeq; V=/home/xueqi/hq/projects/tc-alignment-vllm
SNAP=/home/xueqi/.cache/huggingface/hub/models--google--gemma-4-12B-it/snapshots/707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7
export CUDA_DEVICE_ORDER=PCI_BUS_ID VLLM_GPU_MEMORY_UTILIZATION=0.80
while ps -p 3488082 >/dev/null 2>&1; do sleep 120; done
for i in $(seq 1 54); do
  left=0
  for spec in ALT1s0:all87_1alt_v2/seed-0 ALT1s1:all87_1alt_v2/seed-1 PER2s0:all87_2per_v2/seed-0 PER2s1:all87_2per_v2/seed-1 ALT1MINs0:all87_1min_v2/seed-0 ALT1MINs1:all87_1min_v2/seed-1 MIX1s0:all87_mix1_v2/seed-0; do
    tag=${spec%%:*}; rel=${spec#*:}; a=$T/results/$rel/tokens-96490/lora
    ls -d $V/runs/rai_${tag}t96490-* >/dev/null 2>&1 && continue
    [ -f $a/adapter_model.safetensors ] || { left=1; continue; }
    sleep 60; rm -rf $V/merged_v2/${tag}_t96490; CUDA_VISIBLE_DEVICES="" $V/.venv/bin/python -B $V/tools/alf_merge_lora.py --base $SNAP --adapter $a --out $V/merged_v2/${tag}_t96490 > $V/logs/merge_${tag}_t96490.log 2>&1 || { echo "[$(date)] merge failed $tag"; left=1; continue; }
    bash $V/tools/alf_eval_served_rai.sh $V/merged_v2/${tag}_t96490 rai_${tag}t96490 4 14; echo "[$(date)] EVAL_DONE rai_${tag}t96490"; rm -rf $V/merged_v2/${tag}_t96490
  done
  [ $left = 0 ] && break; sleep 600
done
echo "[$(date)] GPU4 CHAIN #8 DONE"
