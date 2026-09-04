#!/bin/bash
# When the ALFWorld training array 40952830 leaves the hpg queue, submit eval-only for the arms
# whose hub_merged exists but whose official score dir has no result (their in-job eval died on $HOME).
cd "$(dirname "$0")/.."
while ssh -o ConnectTimeout=30 -o BatchMode=yes hpg "squeue -h -j 40952830 -o %i 2>/dev/null | grep -q ." 2>/dev/null; do sleep 300; done
TAGS=$(ssh -o BatchMode=yes hpg "cd /blue/fsu-compsci-dept/xc25.fsu/hq/tc-alignment && for a in ours pres; do t=alfb1_\${a}_s0; [ -f results/appworld_students/\$t/hub_merged/config.json ] && [ ! -f results/bfas/alfworld/\${a}4b_s0/eval_metrics.json ] && echo -n \"\$t \"; done" 2>/dev/null)
echo "[alfEV-waiter] $(date +%H:%M) tags: $TAGS"
[ -z "$TAGS" ] && exit 0
N=$(echo $TAGS | wc -w)
ssh -o BatchMode=yes hpg "cd /blue/fsu-compsci-dept/xc25.fsu/hq/tc-alignment && sbatch --parsable --array=0-$((N-1)) --export=ALL,TAGS='$TAGS' scripts/alf_evalonly_hpg.slurm" 2>/dev/null | tail -1
