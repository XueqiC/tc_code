#!/bin/bash
# Endpoint sync loop v2 (09-23 05:30 CDT): smic is over its login limit, so push taskeq tokens-*/lora endpoints to mike directly,
# atomically (lora.partial -> mv lora) via push_endpoint_via_mike.sh, every 15 minutes. Only endpoints whose run has a manifest.json.
set -u
T=/home/xueqi/hq/projects/tc-alignment-taskeq/results; B=/home/xueqi/hq/projects/tc-alignment-baselines
while true; do
  for d in $(ls -d $T/*/seed-*/tokens-*/lora 2>/dev/null); do
    run=$(dirname $(dirname $d)); [ -f $run/manifest.json ] || continue
    rel=${d#$T/}; rel=${rel%/lora}
    grep -q "^$rel$" $B/logs/sync_via_mike_done.txt 2>/dev/null && continue
    out=$(bash $B/tools/push_endpoint_via_mike.sh $rel 2>&1 | grep -v -i "loading\|autoloading\|requirement"); echo "$out"
    echo "$out" | grep -qE "pushed|already on mike" && echo "$rel" >> $B/logs/sync_via_mike_done.txt
  done
  sleep 900
done
