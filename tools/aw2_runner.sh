#!/bin/bash
# AW-2 early-vs-spread matched control (T12 proposal): same 40 source tasks, 155 matched rows per arm, 20 CE steps;
# arms: CE early / CE spread / pairwise early / pairwise spread (grad-ckpt) / CE early half-dose. GPU 2, port 8951. Single seed.
cd "$(dirname "$0")/.."
for arm in "aw2_ce_early_s0 data/appworld_events/pool_aw2_early_matched.jsonl ce" "aw2_ce_spread_s0 data/appworld_events/pool_aw2_spread_matched.jsonl ce" "aw2_pref_early_s0 data/appworld_events/pool_aw2_early_matched.jsonl ddpo" "aw2_pref_spread_s0 data/appworld_events/pool_aw2_spread_matched.jsonl ddpo" "aw2_ce_early_half_s0 data/appworld_events/pool_aw2_early_halfdose.jsonl ce"; do
  set -- $arm; export AW_GRAD_CKPT=1  # GPU 2 is a 48 GB card: checkpointing for every arm
  echo "[aw2-runner] === $1 ($3 on $2) $(date '+%F %T')"; bash tools/aw1_train_eval.sh "$1" "$2" "$3" 2 8951; echo "[aw2-runner] === $1 exit=$? $(date '+%F %T')"
done
echo "[aw2-runner] ALL DONE $(date '+%F %T')"
