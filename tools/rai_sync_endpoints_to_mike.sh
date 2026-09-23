#!/bin/bash
# Every 10 min, push finished rai training endpoints into the mirrored result trees on the LSU shared /ddnA (via smic),
# where mike's eval_when_ready.sh submits their evaluations. Parent directories are created first (rsync makes only the last one).
T=/home/xueqi/hq/projects/tc-alignment-taskeq/results; B=/home/xueqi/hq/projects/tc-alignment-baselines/results; H=smic.hpc.lsu.edu
push() { src=$1; dst=$2; [ -f $src/adapter_model.safetensors ] || return 1; ssh -o BatchMode=yes xueqic@$H "test -f $dst/adapter_model.safetensors" 2>/dev/null && return 0; ssh -o BatchMode=yes xueqic@$H "mkdir -p $dst" 2>/dev/null && rsync -a $src/ xueqic@$H:$dst/ 2>/dev/null && echo "[$(date)] synced $dst"; }
while true; do
  for d in $(ls -d $T/*/seed-*/tokens-*/lora 2>/dev/null); do [ -f $(dirname $d)/manifest.json ] || continue; rel=${d#$T/}; push $d /ddnA/work/xueqic/hq/tc-alf-taskeq/results/$rel; done
  for d in $(ls -d $B/smartad_v2/seed-2/pass-*/lora 2>/dev/null); do push $d /ddnA/work/xueqic/hq/tc-alf-faithful/results/smartad_orig_v2/seed-2/$(basename $(dirname $d))/lora; done
  sleep 600
done
