#!/bin/bash
# Atomic push of one taskeq endpoint to mike's taskeq results tree (smic is over its login limit): rsync into lora.partial, then mv -> lora in one ssh.
# usage: push_endpoint_via_mike.sh <name>/seed-<k>/tokens-<N>
set -u
rel=$1; T=/home/xueqi/hq/projects/tc-alignment-taskeq/results; H=xueqic@mike.hpc.lsu.edu; D=/ddnA/work/xueqic/hq/tc-alf-taskeq/results
[ -f $T/$rel/lora/adapter_model.safetensors ] || { echo "[$(date)] no adapter at $rel"; exit 1; }
ssh -o BatchMode=yes $H "test -f $D/$rel/lora/adapter_model.safetensors" && { echo "[$(date)] already on mike: $rel"; exit 0; }
ssh -o BatchMode=yes $H "mkdir -p $D/$rel/lora.partial" && rsync -a --timeout=300 $T/$rel/lora/ $H:$D/$rel/lora.partial/ && rsync -a --timeout=120 $T/$rel/../manifest.json $H:$D/$rel/../manifest.json 2>/dev/null; ssh -o BatchMode=yes $H "cd $D/$rel && rm -rf lora && mv lora.partial lora && test -f lora/adapter_model.safetensors" && echo "[$(date)] pushed $rel" || echo "[$(date)] PUSH FAILED $rel"
