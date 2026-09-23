#!/bin/bash
# GPU1 chain #7: after chain #6 exits, (re)merge C s2 high endpoint and evaluate it on rai.
set -u
V=/home/xueqi/hq/projects/tc-alignment-vllm; SNAP=/home/xueqi/.cache/huggingface/hub/models--google--gemma-4-12B-it/snapshots/707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7
while ps -p 3126137 >/dev/null 2>&1; do sleep 60; done
[ -f $V/adapters_mike/Cs2_t96490/adapter_model.safetensors ] || rsync -a xueqic@smic.hpc.lsu.edu:/ddnA/work/xueqic/hq/tc-alf-taskeq/results/all87tok_v2/seed-2/tokens-96490/lora/ $V/adapters_mike/Cs2_t96490/
rm -rf $V/merged_v2/mike_Cs2_t96490; CUDA_VISIBLE_DEVICES="" $V/.venv/bin/python -B $V/tools/alf_merge_lora.py --base $SNAP --adapter $V/adapters_mike/Cs2_t96490 --out $V/merged_v2/mike_Cs2_t96490 > $V/logs/merge_mike_Cs2_t96490.log 2>&1 && bash $V/tools/alf_eval_served_rai.sh $V/merged_v2/mike_Cs2_t96490 rai_Cs2t96490 1 14
echo "[$(date)] GPU1 CHAIN #7 DONE"
