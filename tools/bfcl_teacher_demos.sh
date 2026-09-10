#!/usr/bin/env bash
# Collect three official BFCL FC attempts and import all usage into the BFAS ledger.
# Usage: BFAS_BFCL_TEACHER=ollama/gpt-oss:120b-FC bash tools/bfcl_teacher_demos.sh
# BFAS_BFCL_SKIP_MEMORY_PREREQ=1 marks memory_* demand as unavailable inventory.
set -euo pipefail
MODEL=${1:-${BFAS_BFCL_TEACHER:-deepseek-v4-pro-FC}}
export BFAS_BFCL_TEACHER="$MODEL"
PROJ=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BFCL_DIR=$PROJ/envs/bfcl
ROOT=$BFCL_DIR/gorilla/berkeley-function-call-leaderboard
SAFE=$(echo "$MODEL" | tr -c 'A-Za-z0-9' '_' | sed 's/_*$//')
cd "$PROJ"
mkdir -p logs
account() { "$PROJ/.venv/bin/python" "$PROJ/tools/bfcl_teacher_account.py" "$@" --model "$MODEL"; }
bfcl() { "$BFCL_DIR/.venv/bin/python" "$PROJ/tools/bfcl_cli.py" "$@"; }
# Restore the caller's selective-generation file even if collection fails.
SELECTION_BACKUP=$(mktemp)
HAD_SELECTION=0
if [[ -f "$ROOT/test_case_ids_to_generate.json" ]]; then
  cp "$ROOT/test_case_ids_to_generate.json" "$SELECTION_BACKUP"
  HAD_SELECTION=1
fi
restore_selection() {
  if [[ "$HAD_SELECTION" == 1 ]]; then
    cp "$SELECTION_BACKUP" "$ROOT/test_case_ids_to_generate.json"
  else
    rm -f "$ROOT/test_case_ids_to_generate.json"
  fi
  rm -f "$SELECTION_BACKUP"
}
trap restore_selection EXIT
account prepare
# Resolve credentials in the CLI/handler. Never spend tokens probing a key.
for A in 1 2 3; do
  TEMP=0.0; [[ "$A" -gt 1 ]] && TEMP=0.7
  if account complete --attempt "$A"; then
    # Import historical attempts without buying them again.
    account record --attempt "$A"
    echo "[bfcldemos] attempt $A already covers the demand split, skipping"
    continue
  fi
  export BFAS_BFCL_USAGE_LOG="$ROOT/result_demos_${SAFE}_a${A}/bfas_usage.jsonl"
  GEN_STATUS=0
  bfcl generate --model "$MODEL" --run-ids --num-threads 2 --temperature "$TEMP" \
    --result-dir "result_demos_${SAFE}_a${A}" > "$PROJ/logs/bfcl_demos_${SAFE}_a${A}_gen.log" 2>&1 || GEN_STATUS=$?
  EVAL_STATUS=0
  bfcl evaluate --model "$MODEL" --result-dir "result_demos_${SAFE}_a${A}" \
    --score-dir "score_demos_${SAFE}_a${A}" --partial-eval \
    > "$PROJ/logs/bfcl_demos_${SAFE}_a${A}_eval.log" 2>&1 || EVAL_STATUS=$?
  # Charge before handling subprocess failures: completed responses cost quota
  # even when generation or evaluation exits unsuccessfully.
  account record --attempt "$A"
  echo "[bfcldemos] attempt $A done (generate=$GEN_STATUS evaluate=$EVAL_STATUS)"
  [[ "$GEN_STATUS" == 0 ]] || exit "$GEN_STATUS"
done
account merge
echo "[bfcldemos] $MODEL 3-attempt collection done"
