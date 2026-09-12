#!/usr/bin/env bash
# Push this project's code to LONI QB-D (excludes data/results/logs/envs).
# Requires rai's public key in ~/.ssh/authorized_keys on LONI (see docs/loni_setup.md).
set -e
LONI_BASE="${LONI_BASE:-/work/xueqic/hq}"
SRC="$(cd "$(dirname "$0")/.." && pwd)"
PROJ="$(basename "$SRC")"
ssh loni "mkdir -p $LONI_BASE/$PROJ/logs"
rsync -az --info=stats1 \
  --exclude '.git/' --exclude 'data/' --exclude 'results/' --exclude 'logs/' \
  --exclude '.venv/' --exclude '__pycache__/' --exclude 'wandb/' --exclude '_trash/' --exclude 'envs/' \
  --exclude 'runs/' \
  "$SRC/" loni:"$LONI_BASE/$PROJ/"
echo "synced -> loni:$LONI_BASE/$PROJ"
