#!/bin/bash
# Pull the LONI-trained original-protocol baseline pass-10 adapters (SmartAD original s0/s1, SAD sad_sum s0/s1/s2, Kang s0/s1) to rai,
# into the same results layout the rai-trained seed-2 cells use. Kang s2 is already on rai (rai-trained).
set -u
B=/home/xueqi/hq/projects/tc-alignment-baselines; L=/work/xueqic/hq
pull() { k=$1; src=$2; mkdir -p $B/results/$k/pass-10/lora; rsync -a --timeout=180 loni:$L/$src/pass-10/lora/ $B/results/$k/pass-10/lora/ 2>&1 | grep -v -i "loading requirement"; [ -f $B/results/$k/pass-10/lora/adapter_model.safetensors ] && echo "[$(date)] pulled $k" || echo "[$(date)] PULL FAILED $k"; }
pull smartad_v2/seed-0 tc-alf-baselines/results/smartad_v2/seed-0
pull smartad_v2/seed-1 tc-alf-baselines/results/smartad_v2/seed-1
pull sad_sum_v2/seed-0 tc-alf-baselines/results/sad_sum_v2/seed-0
pull sad_sum_v2/seed-1 tc-alf-baselines/results/sad_sum_v2/seed-1
pull sad_sum_v2/seed-2 tc-alf-baselines-s2/results/sad_sum_v2/seed-2
pull kang_v2/seed-0 tc-alf-baselines/results/kang_v2/seed-0
pull kang_v2/seed-1 tc-alf-baselines/results/kang_v2/seed-1
echo "[$(date)] PULL DONE"
