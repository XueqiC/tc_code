#!/usr/bin/env bash
# Push this project's code to HiPerGator (excludes data/results/logs/envs).
set -e
HPG_BASE="/blue/fsu-compsci-dept/xc25.fsu/hq"
SRC="$(cd "$(dirname "$0")/.." && pwd)"
PROJ="$(basename "$SRC")"
ssh hpg "mkdir -p $HPG_BASE/$PROJ/logs"
rsync -az --info=stats1 \
  --exclude '.git/' --exclude 'data/' --exclude 'results/' --exclude 'logs/' \
  --exclude '.venv/' --exclude '__pycache__/' --exclude 'wandb/' --exclude '_trash/' \
  "$SRC/" hpg:"$HPG_BASE/$PROJ/"
echo "synced -> hpg:$HPG_BASE/$PROJ"
