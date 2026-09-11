#!/bin/bash
# When the ALFWorld mining lane finishes, build the A/B/C/D/D_conf/CE/E pools, sync them to hpg and submit the ablation array.
cd "$(dirname "$0")/.."
while ! grep -q '\[alf-lane\] DONE' logs/alf_event_lane.log 2>/dev/null; do sleep 120; done
.venv/bin/python tools/alf_events_to_pools.py 2>&1 | tee -a logs/alf_ablation_waiter.log
rsync -az -e "ssh -o BatchMode=yes" data/alf_sft/pool_*.jsonl data/alf_sft/events_v1.jsonl data/alf_sft/events_negative.jsonl hpg:/blue/fsu-compsci-dept/xc25.fsu/hq/tc-alignment/data/alf_sft/ 2>/dev/null
J=$(ssh -o BatchMode=yes hpg "cd /blue/fsu-compsci-dept/xc25.fsu/hq/tc-alignment && sbatch --parsable scripts/alf_ablation_hpg.slurm" 2>/dev/null | tail -1)
echo "[alf-waiter] $(date +%H:%M) submitted alf_ablation $J" | tee -a logs/alf_ablation_waiter.log
