#!/bin/bash
# AW-1 runner v3: the two pairwise arms OOM'd on rai GPU4 (95 GB; ddpo path with 4096-token AppWorld prompts).
# Waits for runner v2 to exit, then reruns them with gradient checkpointing (AW_GRAD_CKPT=1, as awb4_cp_s1 on rai GPU1).
cd "$(dirname "$0")/.."
while kill -0 ${RUNNER_V2_PID:-3695212} 2>/dev/null; do sleep 60; done
export AW_GRAD_CKPT=1
for arm in "aw1_pref_conseq_s0 data/appworld_events/pool_aw1_conseq.jsonl ddpo" "aw1_pref_all_s0 data/appworld_events/pool_aw1_all.jsonl ddpo"; do
  set -- $arm; rm -rf results/appworld_students/$1 2>/dev/null
  echo "[aw1-runner] === $1 ($3 on $2, grad-ckpt) $(date '+%F %T')"; bash tools/aw1_train_eval.sh "$1" "$2" "$3" 4 8950; echo "[aw1-runner] === $1 exit=$? $(date '+%F %T')"
done
echo "[aw1-runner] PREF DONE $(date '+%F %T')"
