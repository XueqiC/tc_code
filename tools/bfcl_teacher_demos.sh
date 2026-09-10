#!/usr/bin/env bash
# Collect verified teacher demos for the BFCL demand split via the
# official pipeline, with protocol parity to AppWorld: up to three
# attempts per task (greedy, then two sampled at temperature 0.7).
# Usage: bfcl_teacher_demos.sh [model, default: gpt-5.4]
set -u
MODEL=${1:-gpt-5.4}
# resolve from this script's own location: hardcoding a home path made
# every hpg run skip evaluation silently ("NO ADAPTER") while still
# printing DONE, because /home/xueqi does not exist there
PROJ=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BFCL_DIR=$PROJ/envs/bfcl
ROOT=$BFCL_DIR/gorilla/berkeley-function-call-leaderboard
cd "$PROJ"
SAFE=$(echo "$MODEL" | tr -c 'A-Za-z0-9' '_' | sed 's/_*$//')
# demand ids -> official selective-generation file (built by the adapter so
# memory ids resolve to their runnable per-backend category and drag their
# prerequisite write-chain along; data-file membership would emit the bare
# `memory` group name and crash generation with KeyError: MemoryAPI_)
PYTHONPATH=$PROJ/src .venv/bin/python - <<PY
import json, sys
sys.path.insert(0, "$PROJ/src")
from bfas.adapters.bfcl import BFCLAdapter
split = json.load(open("configs/bfcl_support_split.json"))
demand = list(split["demand"])
by_cat = BFCLAdapter()._selective_file(demand)
json.dump(by_cat, open("$ROOT/test_case_ids_to_generate.json", "w"), indent=1)
n = sum(len(v) for v in by_cat.values())
print("[bfcldemos] ids file:", n, "entries in", len(by_cat), "categories")
selected = {i for ids in by_cat.values() for i in ids}
missing = set(demand) - selected
assert not missing, f"demand ids missing: {sorted(missing)[:5]}"
PY
# An attempt/repeat counts as already collected only when every id of the
# CURRENT demand split has a result in it. Checking merely that the directory
# is non-empty silently accepts leftovers from an earlier split, which is how
# a stale attempt-3 directory from the previous split got merged in.
have_all_demand() {
  PYTHONPATH=$PROJ/src "$PROJ/.venv/bin/python" - "$1" <<'PYCHK'
import json, sys
from pathlib import Path
target = Path(sys.argv[1])
split = json.load(open("configs/bfcl_support_split.json"))
demand = set(split["demand"])
have = set()
if target.is_dir():
    for f in target.rglob("*_result.json"):
        for line in f.open():
            if line.strip():
                have.add(json.loads(line)["id"])
sys.exit(0 if demand <= have else 1)
PYCHK
}
cd "$BFCL_DIR"; . .venv/bin/activate
export OPENAI_BASE_URL="https://ollama.com/v1"
PROBE_PAYLOAD=$(python - "${MODEL%-FC}" <<'PYPROBE'
import json, sys
print(json.dumps({"model": sys.argv[1], "messages": [{"role": "user", "content": "hi"}], "max_tokens": 5}))
PYPROBE
)
for KF in ~/.ollama_api_key ~/.ollama_api_key2; do
  CODE=$(curl -s -m 20 -X POST "$OPENAI_BASE_URL/chat/completions" \
    -H "Authorization: Bearer $(cat $KF)" -H "Content-Type: application/json" \
    -d "$PROBE_PAYLOAD" \
    -o /dev/null -w "%{http_code}")
  [ "$CODE" = "200" ] && { export OPENAI_API_KEY="$(cat $KF)"; echo "[bfcldemos] using key file $KF"; break; }
done
[ -n "${OPENAI_API_KEY:-}" ] || { echo "[bfcldemos] NO WORKING KEY"; exit 1; }
for A in 1 2 3; do
  TEMP=0.001; [ "$A" -gt 1 ] && TEMP=0.7
  # teacher quota is the scarce resource here: an attempt already covering the
  # current demand is not re-bought when the run is resumed after a 429
  if (cd "$PROJ" && have_all_demand "$ROOT/result_demos_${SAFE}_a${A}"); then
    echo "[bfcldemos] attempt $A already covers the demand split, skipping"
    continue
  fi
  bfcl generate --model "$MODEL" --run-ids --num-threads 2 --temperature $TEMP \
    --result-dir "result_demos_${SAFE}_a${A}" > "$PROJ/logs/bfcl_demos_${SAFE}_a${A}_gen.log" 2>&1
  bfcl evaluate --model "$MODEL" --result-dir "result_demos_${SAFE}_a${A}" \
    --score-dir "score_demos_${SAFE}_a${A}" --partial-eval \
    > "$PROJ/logs/bfcl_demos_${SAFE}_a${A}_eval.log" 2>&1
  echo "[bfcldemos] attempt $A done"
done
# merge: verified union; per task keep the first verified attempt's entry
cd "$PROJ"
.venv/bin/python - <<PY
import json, glob, os, shutil
from pathlib import Path
ROOT = Path("$ROOT")
safe = "$SAFE"
merged = ROOT / f"result_demos_{safe}"
if merged.exists():
    shutil.rmtree(merged)
# scope the merge to the current demand split: an attempt directory can still
# hold results for ids from an earlier split, and letting those through would
# put off-split tasks into the demo library and the SFT baseline pool
demand = set(json.load(open("configs/bfcl_support_split.json"))["demand"])
verified_first: dict[str, tuple[int, dict, str]] = {}
for a in (1, 2, 3):
    failed = set()
    for f in (ROOT / f"score_demos_{safe}_a{a}").rglob("*_score.json"):
        lines = [json.loads(l) for l in f.open() if l.strip()]
        for e in lines[1:]:
            if isinstance(e, dict) and "id" in e:
                failed.add(e["id"])
    for f in (ROOT / f"result_demos_{safe}_a{a}").rglob("*_result.json"):
        rel = f.relative_to(ROOT / f"result_demos_{safe}_a{a}")
        for line in f.open():
            if not line.strip():
                continue
            e = json.loads(line)
            if (e["id"] in demand and e["id"] not in failed
                    and e["id"] not in verified_first):
                verified_first[e["id"]] = (a, e, str(rel))
by_file: dict[str, list[dict]] = {}
for a, e, rel in verified_first.values():
    by_file.setdefault(rel, []).append(e)
for rel, entries in by_file.items():
    dst = merged / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w") as fh:
        for e in entries:
            fh.write(json.dumps(e, ensure_ascii=False) + "\n")
json.dump(sorted(verified_first),
          open("data/bfcl_demos_ds_verified.json", "w"))
from collections import Counter
att = Counter(a for a, _, _ in verified_first.values())
print(f"[bfcldemos] verified={len(verified_first)}/{len(demand)} "
      f"by attempt {dict(att)}")
PY
echo "[bfcldemos] $MODEL 3-attempt collection done"
