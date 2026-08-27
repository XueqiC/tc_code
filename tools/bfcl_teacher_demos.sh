#!/usr/bin/env bash
# Collect verified teacher demos for the BFCL demand split via the
# official pipeline, with protocol parity to AppWorld: up to three
# attempts per task (greedy, then two sampled at temperature 0.7).
# Usage: bfcl_teacher_demos.sh <model e.g. deepseek-v4-pro-FC>
set -u
MODEL=${1:-deepseek-v4-pro-FC}
PROJ=/home/xueqi/hq/projects/tc-alignment
BFCL_DIR=$PROJ/envs/bfcl
ROOT=$BFCL_DIR/gorilla/berkeley-function-call-leaderboard
cd "$PROJ"
SAFE=$(echo "$MODEL" | tr -c 'A-Za-z0-9' '_' | sed 's/_*$//')
# demand ids -> official selective-generation file (membership by data file)
.venv/bin/python - <<PY
import json, collections
from pathlib import Path
split = json.load(open("configs/bfcl_support_split.json"))
demand = set(split["demand"])
data_dir = Path("$ROOT/bfcl_eval/data")
by_cat = collections.defaultdict(list)
for f in sorted(data_dir.glob("BFCL_v4_*.json")):
    cat = f.stem.replace("BFCL_v4_", "")
    for line in f.open():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            break
        if isinstance(row, dict) and row.get("id") in demand:
            by_cat[cat].append(row["id"])
json.dump(dict(by_cat), open("$ROOT/test_case_ids_to_generate.json", "w"), indent=1)
n = sum(len(v) for v in by_cat.values())
print("[bfcldemos] ids file:", n, "entries in", len(by_cat), "categories")
assert n == len(demand), f"membership mismatch {n} != {len(demand)}"
PY
cd "$BFCL_DIR"; . .venv/bin/activate
export OPENAI_BASE_URL="https://ollama.com/v1"
for KF in ~/.ollama_api_key ~/.ollama_api_key2; do
  CODE=$(curl -s -m 20 -X POST "$OPENAI_BASE_URL/chat/completions" \
    -H "Authorization: Bearer $(cat $KF)" -H "Content-Type: application/json" \
    -d '{"model":"deepseek-v4-pro","messages":[{"role":"user","content":"hi"}],"max_tokens":5}' \
    -o /dev/null -w "%{http_code}")
  [ "$CODE" = "200" ] && { export OPENAI_API_KEY="$(cat $KF)"; echo "[bfcldemos] using key file $KF"; break; }
done
[ -n "${OPENAI_API_KEY:-}" ] || { echo "[bfcldemos] NO WORKING KEY"; exit 1; }
for A in 1 2 3; do
  TEMP=0.001; [ "$A" -gt 1 ] && TEMP=0.7
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
            if e["id"] not in failed and e["id"] not in verified_first:
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
print(f"[bfcldemos] verified={len(verified_first)}/40 by attempt {dict(att)}")
PY
echo "[bfcldemos] $MODEL 3-attempt collection done"
