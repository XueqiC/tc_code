#!/bin/bash
# GPU1 chain #4: rai-platform evaluations while GPU1 is free: SmartAD seed-2 pass-10 (original protocol) and B seed 0 high endpoint.
set -u
B=/home/xueqi/hq/projects/tc-alignment-baselines; T=/home/xueqi/hq/projects/tc-alignment-taskeq; V=/home/xueqi/hq/projects/tc-alignment-vllm
SNAP=/home/xueqi/.cache/huggingface/hub/models--google--gemma-4-12B-it/snapshots/707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7
merge() { [ -f $V/merged_v2/$2/config.json ] || CUDA_VISIBLE_DEVICES="" $V/.venv/bin/python -B $V/tools/alf_merge_lora.py --base $SNAP --adapter $1 --out $V/merged_v2/$2 > $V/logs/merge_$2.log 2>&1; }
merge $B/results/smartad_v2/seed-2/pass-10/lora smartad_seed2_pass10 && bash $V/tools/alf_eval_served_rai.sh $V/merged_v2/smartad_seed2_pass10 rai_smartads2p10 1 14
merge $T/results/taskeq_v2/seed-0/tokens-96490/lora taskeq_seed0_t96490 && bash $V/tools/alf_eval_served_rai.sh $V/merged_v2/taskeq_seed0_t96490 rai_Bs0t96490 1 14
echo "[$(date)] GPU1 CHAIN #4 DONE"
