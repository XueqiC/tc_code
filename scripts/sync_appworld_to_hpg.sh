#!/usr/bin/env bash
# Run on rai. Use scripts/sync_to_hpg.sh separately for the project code.
# Data is downloaded by hpg_appworld_official_setup.sh ON HPG. In particular,
# do not transfer rai's data symlink or its interpreter/virtualenv.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
cd -- "$ROOT"
DEST=/blue/fsu-compsci-dept/xc25.fsu/hq/tc-alignment
[[ $# == 0 ]] || { echo "Usage: bash scripts/sync_appworld_to_hpg.sh" >&2; exit 2; }
[[ -f envs/appworld-repo/pyproject.toml && -s results/bfas/appworld/collect_shared/demos.json ]] || {
  echo "Missing rai official checkout or archived demos" >&2; exit 2;
}
ssh hpg "mkdir -p '$DEST/envs/appworld-repo' '$DEST/results/bfas/appworld/collect_shared' '$DEST/data/appworld_events' '$DEST/logs'"
rsync -az --info=stats1 \
  --exclude '.git' --exclude '/experiments/outputs' --exclude '/data' \
  --exclude '.venv' --exclude '__pycache__' --exclude '.cache' --exclude '.tmp' --exclude '.env' \
  envs/appworld-repo/ "hpg:$DEST/envs/appworld-repo/"
rsync -az --info=stats1 results/bfas/appworld/collect_shared/demos.json \
  "hpg:$DEST/results/bfas/appworld/collect_shared/"
for pool in data/appworld_events/pool_aw2_spread_matched.jsonl data/appworld_events/pool_aw2_spread_x2.jsonl; do
  if [[ -f $pool ]]; then
    rsync -az --info=stats1 "$pool" "hpg:$DEST/data/appworld_events/"
  else
    echo "Skipping absent pool: $pool (matched is required for B/mining; x2 for C)"
  fi
done
echo "AppWorld inputs synced -> hpg:$DEST; run scripts/hpg_appworld_official_setup.sh there"
