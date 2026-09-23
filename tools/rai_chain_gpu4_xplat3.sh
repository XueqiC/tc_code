#!/bin/bash
# GPU4 chain #5: after chain #4 exits, rai cross-check of the faithful SmartAD s0 pass-10 (mike-trained) and ADDALL s1 high endpoint.
set -u
V=/home/xueqi/hq/projects/tc-alignment-vllm; X=$V/adapters_mike; mkdir -p $X
SNAP=/home/xueqi/.cache/huggingface/hub/models--google--gemma-4-12B-it/snapshots/707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7
while ps -p 3209526 >/dev/null 2>&1; do sleep 120; done
run() { adapter=$1; name=$2; tag=$3; rm -rf $V/merged_v2/$name; CUDA_VISIBLE_DEVICES="" $V/.venv/bin/python -B $V/tools/alf_merge_lora.py --base $SNAP --adapter $adapter --out $V/merged_v2/$name > $V/logs/merge_$name.log 2>&1 || { echo "[$(date)] merge failed $name"; return 1; }; bash $V/tools/alf_eval_served_rai.sh $V/merged_v2/$name $tag 4 14; rm -rf $V/merged_v2/$name; }
for spec in "tc-alf-faithful/results/smartad_wn_v2/seed-0/pass-10:SMWNs0_p10:rai_SMWNs0p10" "tc-alf-taskeq/results/d0addall_v2/seed-1/tokens-96490:ADDALLs1_t96490:rai_ADDALLs1t96490"; do IFS=: read -r rel name tag <<< "$spec"
  rsync -a xueqic@smic.hpc.lsu.edu:/ddnA/work/xueqic/hq/$rel/lora/ $X/$name/ 2>/dev/null || { echo "[$(date)] pull failed $name"; continue; }
  run $X/$name mike_$name $tag
done
echo "[$(date)] GPU4 CHAIN #5 DONE"
