#!/bin/bash
# GPU4 chain #3: after chain #2 (C s0 training) exits, evaluate mike-trained B s1/s2 and C s1 high endpoints on the rai platform; delete merges after use.
set -u
V=/home/xueqi/hq/projects/tc-alignment-vllm; X=$V/adapters_mike; mkdir -p $X
SNAP=/home/xueqi/.cache/huggingface/hub/models--google--gemma-4-12B-it/snapshots/707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7
while ps -p 2246098 >/dev/null 2>&1; do sleep 120; done
for spec in "taskeq_v2/seed-1:Bs1" "taskeq_v2/seed-2:Bs2" "all87tok_v2/seed-1:Cs1"; do d=${spec%%:*}; n=${spec##*:}
  rsync -a xueqic@smic.hpc.lsu.edu:/ddnA/work/xueqic/hq/tc-alf-taskeq/results/$d/tokens-96490/lora/ $X/${n}_t96490/ 2>/dev/null || { echo "[$(date)] pull failed $n"; continue; }
  rm -rf $V/merged_v2/mike_${n}_t96490; CUDA_VISIBLE_DEVICES="" $V/.venv/bin/python -B $V/tools/alf_merge_lora.py --base $SNAP --adapter $X/${n}_t96490 --out $V/merged_v2/mike_${n}_t96490 > $V/logs/merge_mike_${n}_t96490.log 2>&1 || { echo "[$(date)] merge failed $n"; continue; }
  bash $V/tools/alf_eval_served_rai.sh $V/merged_v2/mike_${n}_t96490 rai_${n}t96490 4 14; rm -rf $V/merged_v2/mike_${n}_t96490
done
echo "[$(date)] GPU4 CHAIN #3 DONE"
