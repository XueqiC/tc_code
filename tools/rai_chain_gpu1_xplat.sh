#!/bin/bash
# GPU1 chain #5: cross-platform check — pull mike-trained D s0/s2 and C s2 high endpoints via smic, merge, evaluate on rai GPU1; plus A s2 (rai_ces2p10).
set -u
V=/home/xueqi/hq/projects/tc-alignment-vllm; X=$V/adapters_mike; mkdir -p $X
SNAP=/home/xueqi/.cache/huggingface/hub/models--google--gemma-4-12B-it/snapshots/707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7
merge() { [ -f $V/merged_v2/$2/config.json ] || CUDA_VISIBLE_DEVICES="" $V/.venv/bin/python -B $V/tools/alf_merge_lora.py --base $SNAP --adapter $1 --out $V/merged_v2/$2 > $V/logs/merge_$2.log 2>&1; }
for spec in "all87taskeq_v2/seed-0:Ds0" "all87taskeq_v2/seed-2:Ds2" "all87tok_v2/seed-2:Cs2"; do d=${spec%%:*}; n=${spec##*:}
  rsync -a xueqic@smic.hpc.lsu.edu:/ddnA/work/xueqic/hq/tc-alf-taskeq/results/$d/tokens-96490/lora/ $X/${n}_t96490/ 2>/dev/null && echo "[$(date)] pulled $n" || { echo "[$(date)] pull failed $n"; continue; }
  merge $X/${n}_t96490 mike_${n}_t96490 && bash $V/tools/alf_eval_served_rai.sh $V/merged_v2/mike_${n}_t96490 rai_${n}t96490 1 14
done
bash $V/tools/alf_eval_served_rai.sh $V/merged_v2/seed2_pass10 rai_ces2p10 1 14
echo "[$(date)] GPU1 CHAIN #5 DONE"
