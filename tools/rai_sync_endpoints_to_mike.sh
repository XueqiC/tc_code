#!/bin/bash
# Every 10 min, push finished rai training endpoints (taskeq tree tokens-*/lora, baselines tree pass-*/lora for smartad seed 2)
# into the mirrored result trees on mike, where eval_when_ready.sh submits their evaluations.
T=/home/xueqi/hq/projects/tc-alignment-taskeq/results; B=/home/xueqi/hq/projects/tc-alignment-baselines/results
while true; do
  for d in $(ls -d $T/*/seed-*/tokens-*/lora 2>/dev/null); do rel=${d#$T/}; [ -f $d/adapter_model.safetensors ] && [ -f $(dirname $d)/manifest.json ] && rsync -a --ignore-existing $d/ xueqic@mike.hpc.lsu.edu:/ddnA/work/xueqic/hq/tc-alf-taskeq/results/$rel/ 2>/dev/null && echo "[$(date)] synced taskeq $rel"; done
  for d in $(ls -d $B/smartad_v2/seed-2/pass-*/lora 2>/dev/null); do rel=${d#$B/}; [ -f $d/adapter_model.safetensors ] && rsync -a --ignore-existing $d/ xueqic@mike.hpc.lsu.edu:/ddnA/work/xueqic/hq/tc-alf-faithful/results/smartad_orig_v2/seed-2/$(basename $(dirname $d))/lora/ 2>/dev/null && echo "[$(date)] synced $rel"; done
  sleep 600
done
