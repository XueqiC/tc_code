#!/bin/bash
# GPU4 chain: wait for le50 trainings, merge le50 pass-10 adapters (CPU), then served evaluations on GPU4:
# base_rai, le50 s0/s1 pass-10, highsweep s0/s1 pass-10 (LONI adapters merged on rai).
set -u
B=/home/xueqi/hq/projects/tc-alignment-baselines; V=/home/xueqi/hq/projects/tc-alignment-vllm
SNAP=/home/xueqi/.cache/huggingface/hub/models--google--gemma-4-12B-it/snapshots/707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7
until [ -f $B/results/sweep_le50_v2/seed-0/pass-10/manifest.json ] && [ -f $B/results/sweep_le50_v2/seed-1/pass-10/manifest.json ]; do sleep 60; done
while ps -p 1919302 >/dev/null 2>&1 || ps -p 1919303 >/dev/null 2>&1; do sleep 30; done
echo "[$(date)] le50 trainings finished; merging"
for s in 0 1; do CUDA_VISIBLE_DEVICES="" $V/.venv/bin/python -B $V/tools/alf_merge_lora.py --base $SNAP --adapter $B/results/sweep_le50_v2/seed-$s/pass-10/lora --out $V/merged_v2/le50_seed${s}_pass10 > $V/logs/merge_le50_s$s.log 2>&1 || { echo "merge le50 s$s failed"; exit 1; }; done
echo "[$(date)] merges done; evaluating on GPU4"
bash $V/tools/alf_eval_served_rai.sh $SNAP rai_base 4 14
bash $V/tools/alf_eval_served_rai.sh $V/merged_v2/le50_seed0_pass10 rai_le50s0p10 4 14
bash $V/tools/alf_eval_served_rai.sh $V/merged_v2/le50_seed1_pass10 rai_le50s1p10 4 14
bash $V/tools/alf_eval_served_rai.sh $V/merged_v2/highsweep_seed0_pass10 rai_highs0p10 4 14
bash $V/tools/alf_eval_served_rai.sh $V/merged_v2/highsweep_seed1_pass10 rai_highs1p10 4 14
bash $V/tools/alf_eval_served_rai.sh $SNAP rai_base_rep 4 14
echo "[$(date)] GPU4 CHAIN DONE"
