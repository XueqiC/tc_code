#!/usr/bin/env bash
set -euo pipefail

PROJECT=/home/xueqi/hq/projects/tc-alignment
PYTHON="$PROJECT/.venv/bin/python"
STATE_DIR="$PROJECT/.gpu_hold"

case "${1:-}" in
  status|-h|--help)
    exec "$PYTHON" "$PROJECT/tools/gpu_hold.py" "$@" --state-dir "$STATE_DIR"
    ;;
esac

mkdir -p -m 700 "$STATE_DIR"
nohup "$PYTHON" -u "$PROJECT/tools/gpu_hold.py" "$@" --state-dir "$STATE_DIR" \
  >>"$STATE_DIR/daemon.log" 2>&1 </dev/null &
printf 'Launched GPU holder PID %s; startup/status log: %s/daemon.log\n' "$!" "$STATE_DIR"
