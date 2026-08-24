#!/usr/bin/env bash
# Chunked idempotent push through the MTU-limited tunnel.
# Usage: bash tools/hpg_push.sh <local_file> <remote_abs_path> [chunk_chars]
set -u
LOCAL=$1; REMOTE=$2; CH=${3:-32000}
b64=$(base64 -w0 "$LOCAL")
len=${#b64}
timeout 20 ssh -o BatchMode=yes hpg "mkdir -p \$(dirname $REMOTE); : > $REMOTE.b64" || exit 1
off=0
while [ $off -lt $len ]; do
  chunk=${b64:$off:$CH}
  ok=""
  for try in 1 2 3; do
    if timeout 25 ssh -o BatchMode=yes hpg "printf %s '$chunk' >> $REMOTE.b64"; then ok=1; break; fi
    sleep 2
  done
  [ -n "$ok" ] || { echo "PUSH FAILED at offset $off"; exit 1; }
  off=$((off+CH))
done
lmd5=$(md5sum "$LOCAL" | cut -c1-32 | cut -d' ' -f1)
rmd5=$(timeout 25 ssh -o BatchMode=yes hpg "base64 -d $REMOTE.b64 > $REMOTE && rm $REMOTE.b64 && md5sum $REMOTE" | cut -d' ' -f1)
if [ "$lmd5" = "$rmd5" ]; then echo "PUSH OK $LOCAL -> $REMOTE ($len b64 chars)"; else echo "PUSH MD5 MISMATCH $lmd5 vs $rmd5"; exit 1; fi
