#!/bin/bash
# When round-2 mining (CE model) finishes: build pools under data/alf_sft/r2/, set epochs so that ~44 steps
# are used on the consequential pool, rsync to hpg and submit scripts/alf_round2_hpg.slurm.
cd "$(dirname "$0")/.."
while ! grep -q '\[alf-r2\] DONE' logs/alf_round2_lane.log 2>/dev/null; do sleep 120; done
mkdir -p data/alf_sft/r2
.venv/bin/python tools/alf_events_to_pools.py --events data/alf_sft/events_r2ce.jsonl --out-dir data/alf_sft/r2 2>&1 | tee -a logs/alf_round2_waiter.log
N=$(wc -l < data/alf_sft/r2/pool_C_conseq.jsonl); EP=$(python3 -c "import math;print(max(1, round(44*8/max($N,1))))")
echo "[alf-r2-waiter] consequential residual rows=$N -> epochs=$EP" | tee -a logs/alf_round2_waiter.log
ssh -o BatchMode=yes hpg "mkdir -p /blue/fsu-compsci-dept/xc25.fsu/hq/tc-alignment/data/alf_sft/r2" 2>/dev/null
rsync -az -e "ssh -o BatchMode=yes" data/alf_sft/r2/ data/alf_sft/events_r2ce.jsonl hpg:/blue/fsu-compsci-dept/xc25.fsu/hq/tc-alignment/data/alf_sft/r2/ 2>/dev/null
bash scripts/sync_to_hpg.sh >/dev/null 2>&1
J=$(ssh -o BatchMode=yes hpg "cd /blue/fsu-compsci-dept/xc25.fsu/hq/tc-alignment && sbatch --parsable --export=ALL,R2_EPOCHS=$EP scripts/alf_round2_hpg.slurm" 2>/dev/null | tail -1)
echo "[alf-r2-waiter] $(date +%H:%M) submitted alf_round2 $J" | tee -a logs/alf_round2_waiter.log
