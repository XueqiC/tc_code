#!/usr/bin/env bash
# Generate stateful episodes for every stateful task in the demand split.
# These are the axes the student loses most on and the ones the single-call
# generator cannot draft for at all.
set -u
# resolve from this script's own location: hardcoding a home path made
# every hpg run skip evaluation silently ("NO ADAPTER") while still
# printing DONE, because /home/xueqi does not exist there
PROJ=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PER=${1:-6}
OUT=${2:-data/bfcl_sft/gen_stateful.jsonl}
cd "$PROJ"
: > "$OUT"
SEEDS=$(PYTHONPATH=src .venv/bin/python - <<'PY'
import json, sys
sys.path.insert(0, "src")
from bfas.adapters.bfcl import BFCLAdapter
cats = BFCLAdapter().task_categories()
split = json.load(open("configs/bfcl_support_split.json"))
# every demand seed: the unified generator routes each to its judge
print(" ".join(sorted(split["demand"])))
PY
)
export OLLAMA_API_KEY="${OLLAMA_API_KEY:-$(cat ~/.ollama_api_key2 | tr -d '\n')}"
for S in $SEEDS; do
  echo "=== $S ==="
  PYTHONPATH=src "$PROJ/envs/bfcl/.venv/bin/python" -u tools/bfcl_generate_mt.py \
    --only "$S" --per-seed "$PER" --out "$OUT" 2>&1 \
    | grep -vE "FutureWarning|ENCODER|Loading weights|⚠"
done
echo "[genstateful] total episodes: $(wc -l < "$OUT")"
echo "[genstateful] executable:     $(grep -c '"executes": true' "$OUT" || true)"
