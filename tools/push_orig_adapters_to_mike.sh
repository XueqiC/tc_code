#!/bin/bash
# One rsync session (relative paths, parents created) pushes the 8 original-protocol pass-10 adapters to mike, then one ssh submits their evaluations.
set -u
B=/home/xueqi/hq/projects/tc-alignment-baselines; H=xueqic@mike.hpc.lsu.edu; D=/ddnA/work/xueqic/hq/tc-alf-orig/results
ssh -o BatchMode=yes $H "mkdir -p $D" || { echo "[$(date)] MKDIR FAILED"; exit 1; }
cd $B/results && rsync -aR --timeout=300 smartad_v2/seed-0/pass-10/lora smartad_v2/seed-1/pass-10/lora sad_sum_v2/seed-0/pass-10/lora sad_sum_v2/seed-1/pass-10/lora sad_sum_v2/seed-2/pass-10/lora kang_v2/seed-0/pass-10/lora kang_v2/seed-1/pass-10/lora kang_v2/seed-2/pass-10/lora $H:$D/ && echo "[$(date)] pushed 8 adapters" || { echo "[$(date)] PUSH FAILED"; exit 1; }
ssh -o BatchMode=yes $H "cd /ddnA/work/xueqic/hq; for spec in SMORIGs0:smartad_v2/seed-0 SMORIGs1:smartad_v2/seed-1 SADORIGs0:sad_sum_v2/seed-0 SADORIGs1:sad_sum_v2/seed-1 SADORIGs2:sad_sum_v2/seed-2 KANGs0:kang_v2/seed-0 KANGs1:kang_v2/seed-1 KANGs2:kang_v2/seed-2; do tag=\${spec%%:*}; rel=\${spec#*:}; a=$D/\$rel/pass-10/lora; test -f \$a/adapter_model.safetensors || { echo missing \$a; continue; }; j=\$(sbatch --parsable --export=ALL,ADAPTER=\$a,EVAL_TAG=mike_\${tag}p10 slurm/eval_adapter.slurm); echo \"[\$(date)] mike_\${tag}p10 -> \$j\"; echo mike_\${tag}p10 >> eval_submitted.txt; done; squeue -u xueqic -h | wc -l"
echo "[$(date)] PUSH+SUBMIT DONE"
