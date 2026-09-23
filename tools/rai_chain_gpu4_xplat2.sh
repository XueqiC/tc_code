#!/bin/bash
# GPU4 chain #4: after chain #2 (C s0 training) exits, rai-platform cross-checks: local C s0 t96490, mike-trained D s1 t96490, SADTC s0 pass-10, LE50 s1 t96490.
set -u
V=/home/xueqi/hq/projects/tc-alignment-vllm; T=/home/xueqi/hq/projects/tc-alignment-taskeq; X=$V/adapters_mike; mkdir -p $X
SNAP=/home/xueqi/.cache/huggingface/hub/models--google--gemma-4-12B-it/snapshots/707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7
while ps -p 2246098 >/dev/null 2>&1; do sleep 120; done
run() { adapter=$1; name=$2; tag=$3; rm -rf $V/merged_v2/$name; CUDA_VISIBLE_DEVICES="" $V/.venv/bin/python -B $V/tools/alf_merge_lora.py --base $SNAP --adapter $adapter --out $V/merged_v2/$name > $V/logs/merge_$name.log 2>&1 || { echo "[$(date)] merge failed $name"; return 1; }; bash $V/tools/alf_eval_served_rai.sh $V/merged_v2/$name $tag 4 14; rm -rf $V/merged_v2/$name; }
[ -f $T/results/all87tok_v2/seed-0/tokens-96490/lora/adapter_model.safetensors ] && run $T/results/all87tok_v2/seed-0/tokens-96490/lora local_Cs0_t96490 rai_Cs0t96490
for spec in "tc-alf-taskeq/results/all87taskeq_v2/seed-1/tokens-96490:Ds1_t96490:rai_Ds1t96490" "tc-alf-faithful/results/sad_tc_v2/seed-0/pass-10:SADTCs0_p10:rai_SADTCs0p10" "tc-alf-taskeq/results/le50tok_v2/seed-1/tokens-96490:LE50s1_t96490:rai_LE50s1t96490"; do IFS=: read -r rel name tag <<< "$spec"
  rsync -a xueqic@smic.hpc.lsu.edu:/ddnA/work/xueqic/hq/$rel/lora/ $X/$name/ 2>/dev/null || { echo "[$(date)] pull failed $name"; continue; }
  run $X/$name mike_$name $tag
done
echo "[$(date)] GPU4 CHAIN #4 DONE"
