#!/usr/bin/env bash
# Pull results and logs back from HiPerGator into results/hpg/ and logs/hpg/.
set -e
HPG_BASE="/blue/yd24f/xc25.fsu/hq"
SRC="$(cd "$(dirname "$0")/.." && pwd)"
PROJ="$(basename "$SRC")"
mkdir -p "$SRC/results/hpg" "$SRC/logs/hpg"
rsync -az hpg:"$HPG_BASE/$PROJ/results/" "$SRC/results/hpg/" 2>/dev/null || echo "(no results/ on hpg yet)"
rsync -az hpg:"$HPG_BASE/$PROJ/logs/"    "$SRC/logs/hpg/"    2>/dev/null || echo "(no logs/ on hpg yet)"
echo "fetched -> results/hpg/ , logs/hpg/"
