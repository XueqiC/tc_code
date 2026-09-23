#!/bin/bash
# GPU1 chain #6: retry the two cross-platform evaluations whose vLLM server failed to start (C s2 high endpoint, A seed 2).
set -u
V=/home/xueqi/hq/projects/tc-alignment-vllm
bash $V/tools/alf_eval_served_rai.sh $V/merged_v2/mike_Cs2_t96490 rai_Cs2t96490 1 14
bash $V/tools/alf_eval_served_rai.sh $V/merged_v2/seed2_pass10 rai_ces2p10 1 14
echo "[$(date)] GPU1 CHAIN #6 DONE"
