#!/bin/bash
# AW-1 runner v4 (spread-probe events): waits for the spread miner AND runner v3 (GPU 4) to exit, re-mines the ids that
# errored on context overflow with the adaptive cap (Codex fix), stops the miner vllm, builds pools (prefix aw1s) and runs
# the four arms on GPU 4 (official dev57). Pairwise arms use AW_GRAD_CKPT=1. Single seed, 0 teacher tokens.
cd "$(dirname "$0")/.."
MINER_PID=${MINER_PID:-4023432}; MINER_VLLM=${MINER_VLLM:-4018680}; V3_PID=${V3_PID:-4008040}
EV=data/appworld_events/aw1_spread_k3.jsonl
while kill -0 $MINER_PID 2>/dev/null; do sleep 60; done
echo "[aw1s-runner] spread miner exited $(date '+%F %T'); done ids: $(wc -l < $EV.done)"
ERR=$(grep -o "^[ ]*[0-9a-f]\{7\}_[0-9] ERROR" logs/aw1_mine_spread.log | awk '{print $1}' | sort -u | grep -v -x -F -f $EV.done | tr '\n' ' ')
if [ -n "$ERR" ]; then
  echo "[aw1s-runner] repairing errored ids with adaptive cap: $ERR"
  .venv/bin/python - "$EV" $ERR <<'PY'
import json, sys
ev, bad = sys.argv[1], set(sys.argv[2:])
rows = [l for l in open(ev) if l.strip() and json.loads(l)["task_id"] not in bad]
open(ev, "w").writelines(rows); print("[aw1s-runner] rows kept", len(rows))
PY
  .venv/bin/python tools/appworld_event_mine.py --split demand --k 3 --max-probes 4 --max-cont-steps 20 --max-tokens 4096 --thinking off --equality api_effects --probe-select spread --resume --port 8989 --run-tag aw1_spread --out $EV >> logs/aw1_mine_spread_repair.log 2>&1
  echo "[aw1s-runner] repair exit=$? done ids now $(wc -l < $EV.done)"
fi
kill $MINER_VLLM 2>/dev/null; sleep 20
.venv/bin/python tools/aw1_pools.py $EV --prefix aw1s
while kill -0 $V3_PID 2>/dev/null; do sleep 60; done
echo "[aw1s-runner] GPU 4 free $(date '+%F %T')"
for arm in "aw1s_ce_conseq_s0 data/appworld_events/pool_aw1s_conseq.jsonl ce" "aw1s_pref_conseq_s0 data/appworld_events/pool_aw1s_conseq.jsonl ddpo" "aw1s_ce_all_s0 data/appworld_events/pool_aw1s_all.jsonl ce" "aw1s_pref_all_s0 data/appworld_events/pool_aw1s_all.jsonl ddpo"; do
  set -- $arm; [ "$3" = ddpo ] && export AW_GRAD_CKPT=1 || unset AW_GRAD_CKPT
  echo "[aw1s-runner] === $1 ($3 on $2) $(date '+%F %T')"; bash tools/aw1_train_eval.sh "$1" "$2" "$3" 4 8950; echo "[aw1s-runner] === $1 exit=$? $(date '+%F %T')"
done
echo "[aw1s-runner] ALL DONE $(date '+%F %T')"
