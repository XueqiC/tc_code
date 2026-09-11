#!/usr/bin/env bash
# Out-of-scope items across the single-turn seeds: the half of the boundary
# where the tools cannot serve the request and the right answer is no call.
set -u
PROJ=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PER=${1:-5}
OUT=${2:-data/bfcl_sft/gen_oos.jsonl}
cd "$PROJ"
: > "$OUT"
SEEDS=$(PYTHONPATH=src .venv/bin/python - <<'PY'
import json, sys
sys.path.insert(0, "src")
from bfas.adapters.bfcl import BFCLAdapter
adapter = BFCLAdapter()
entries, cats = adapter._load_entries()
split = json.load(open("configs/bfcl_support_split.json"))
# single-turn seeds only: an out-of-scope item is one user turn and no call
print(" ".join(sorted(
    t for t in split["demand"]
    if "function" in entries.get(t, {})
    and not cats[t].startswith(("multi_turn", "memory", "web_search")))))
PY
)
export OLLAMA_API_KEY="${OLLAMA_API_KEY:-$(cat ~/.ollama_api_key2 | tr -d '\n')}"
TMP=data/bfcl_sft/_oos_one.jsonl
for S in $SEEDS; do
  echo "=== $S ==="
  PYTHONPATH=src "$PROJ/envs/bfcl/.venv/bin/python" -u tools/bfcl_generate.py \
    --out-of-scope --only "$S" --per-seed "$PER" --out "$TMP" 2>&1 \
    | grep -E "drafted|KEEP|drop"
  cat "$TMP" >> "$OUT" 2>/dev/null || true
done
rm -f "$TMP"
echo "[gen-oos-all] total: $(wc -l < "$OUT")"
echo "[gen-oos-all] verified: $(grep -c '"verified": true' "$OUT" || true)"
