#!/bin/bash
# AW-1 runner v2 (2026-09-04 20:50). v1 matched the word FINAL in a handoff note and fired early; this one waits for the
# miner PROCESS to exit, re-mines ids that errored during the 17:00 server outage (server kept up), then stops the miner's
# vllm, builds the pools and runs the four arms sequentially on GPU 4 with the official dev split. Single seed, 0 teacher tokens.
cd "$(dirname "$0")/.."
MINER_PID=${MINER_PID:-3307538}; MINER_VLLM=${MINER_VLLM:-3690010}
EV=data/appworld_events/aw1_base_k3.jsonl
while kill -0 $MINER_PID 2>/dev/null; do sleep 60; done
echo "[aw1-runner] miner exited $(date '+%F %T'); done ids: $(wc -l < $EV.done)"
# repair pass: drop rows of errored ids, then resume (the .done sidecar lacks them, so the miner redoes exactly those)
ERR=$(grep -o "^[ ]*[0-9a-f]\{7\}_[0-9] ERROR" logs/t5_aw1_mine.log | awk '{print $1}' | sort -u | grep -v -x -F -f $EV.done | tr '\n' ' ')
if [ -n "$ERR" ]; then
  echo "[aw1-runner] repairing errored ids: $ERR"
  .venv/bin/python - "$EV" $ERR <<'PY'
import json, sys
ev, bad = sys.argv[1], set(sys.argv[2:])
rows = [l for l in open(ev) if l.strip() and json.loads(l)["task_id"] not in bad]
open(ev, "w").writelines(rows); print("[aw1-runner] rows kept", len(rows))
PY
  .venv/bin/python tools/appworld_event_mine.py --split demand --k 3 --max-probes 4 --max-cont-steps 20 --max-tokens 4096 --thinking off --equality api_effects --resume --port 8989 --run-tag aw1_base_k3 --out $EV >> logs/t5_aw1_mine_repair.log 2>&1
  echo "[aw1-runner] repair exit=$? done ids now $(wc -l < $EV.done)"
fi
kill $MINER_VLLM 2>/dev/null; sleep 20
.venv/bin/python tools/aw1_pools.py $EV
for arm in "aw1_ce_conseq_s0 data/appworld_events/pool_aw1_conseq.jsonl ce" "aw1_pref_conseq_s0 data/appworld_events/pool_aw1_conseq.jsonl ddpo" "aw1_ce_all_s0 data/appworld_events/pool_aw1_all.jsonl ce" "aw1_pref_all_s0 data/appworld_events/pool_aw1_all.jsonl ddpo"; do
  set -- $arm; echo "[aw1-runner] === $1 ($3 on $2) $(date '+%F %T')"; bash tools/aw1_train_eval.sh "$1" "$2" "$3" 4 8950; echo "[aw1-runner] === $1 exit=$? $(date '+%F %T')"
done
echo "[aw1-runner] ALL DONE $(date '+%F %T')"
