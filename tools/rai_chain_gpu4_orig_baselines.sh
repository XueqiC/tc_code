#!/bin/bash
# GPU4 chain #7: after chain #6 (pid 3371415) exits, evaluate the ORIGINAL-protocol baseline pass-10 adapters on the rai platform
# (SmartAD original s0/s1 from LONI, SAD sad_sum s0/s1/s2, Kang s0/s1 from LONI + s2 rai-trained) -> merge -> eval (util 0.80) -> delete merge.
set -u
B=/home/xueqi/hq/projects/tc-alignment-baselines; V=/home/xueqi/hq/projects/tc-alignment-vllm
SNAP=/home/xueqi/.cache/huggingface/hub/models--google--gemma-4-12B-it/snapshots/707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7
export CUDA_DEVICE_ORDER=PCI_BUS_ID VLLM_GPU_MEMORY_UTILIZATION=0.80
while ps -p 3371417 >/dev/null 2>&1; do sleep 120; done
for spec in SMORIGs0:smartad_v2/seed-0 SMORIGs1:smartad_v2/seed-1 SADORIGs0:sad_sum_v2/seed-0 SADORIGs1:sad_sum_v2/seed-1 SADORIGs2:sad_sum_v2/seed-2 KANGs0:kang_v2/seed-0 KANGs1:kang_v2/seed-1 KANGs2:kang_v2/seed-2; do
  tag=${spec%%:*}; rel=${spec#*:}; a=$B/results/$rel/pass-10/lora; [ -f $a/adapter_model.safetensors ] || { echo "[$(date)] missing $a"; continue; }
  rm -rf $V/merged_v2/${tag}_p10; CUDA_VISIBLE_DEVICES="" $V/.venv/bin/python -B $V/tools/alf_merge_lora.py --base $SNAP --adapter $a --out $V/merged_v2/${tag}_p10 > $V/logs/merge_${tag}_p10.log 2>&1 || { echo "[$(date)] merge failed $tag"; continue; }
  bash $V/tools/alf_eval_served_rai.sh $V/merged_v2/${tag}_p10 rai_${tag}p10 4 14; echo "[$(date)] EVAL_DONE rai_${tag}p10"; rm -rf $V/merged_v2/${tag}_p10
done
echo "[$(date)] GPU4 CHAIN #7 DONE"
