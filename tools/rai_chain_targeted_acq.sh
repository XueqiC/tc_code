#!/bin/bash
# CPU/API chain: coverage-targeted ADD-DEMO (cool/heat/two, 16 tasks, 2 extra verified demos each, attempts 6-9, T=0.7, sweep filter) on Azure P2,
# union with D0 (all), token-endpoint config in the taskeq tree, preflight, sync to mike and submit three seeds.
set -u
B=/home/xueqi/hq/projects/tc-alignment-baselines; T=/home/xueqi/hq/projects/tc-alignment-taskeq; D0=/home/xueqi/hq/projects/tc-alignment/data/rtd/v1_alfworld_k32_d0
TOK=/home/xueqi/.cache/huggingface/hub/models--google--gemma-4-12B-it/snapshots/707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7
TASKS=$B/configs/alfworld_k32_targeted_tasks.json; TARGETED=$B/data/alfworld_k32_add_targeted; UNION=$B/artifacts/alfworld_k32_d0_plus_targeted
SECRETS=~/hq/secrets/llm_apis.env; val() { grep -E "^$1=" "$SECRETS" | head -1 | cut -d= -f2- | tr -d '"' | tr -d "'"; }
export AZURE_LLM_ENDPOINT="$(val AZURE_APIM_ENDPOINT)" AZURE_LLM_KEY="$(val AZURE_${AZURE_SLOT:-P2}_PRIMARY)"
export CUDA_VISIBLE_DEVICES="" PYTHONPATH="$B/src:$B" PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 BFAS_TEACHER=gpt-5.6-luna
cd $B; PY=$B/.venv/bin/python
for p in "$TARGETED" "$TARGETED.collection"; do [ -e "$p" ] || [ -L "$p" ] && { echo "refusing existing $p"; exit 1; }; done
echo "[$(date)] targeted acquisition (slot ${AZURE_SLOT:-P2})"
$PY tools/alfworld_teacher_pool.py --source "$D0" --out "$TARGETED" --only-tasks "$TASKS" --method smartad --candidates-per-task 2 --attempt-start 6 --attempts-per-task 4 --temperature 0.7 --sweep-filter --workers 1 --rate-limit-retries 8 --max-tokens 2000000 --max-usd 10 --usd-per-mtok-in 0.20 --usd-per-mtok-out 1.20 --usd-per-mtok-cached 0.01 > $B/logs/targeted_collect.log 2>&1; st=$?
echo "[$(date)] targeted collect exit $st"; tail -3 $B/logs/targeted_collect.log | cut -c1-300; [ $st -eq 0 ] || exit 1
$PY tools/alf_bank_union.py --left "$D0" --right "$TARGETED" --output "$UNION" --mixing all --config $B/configs/rtd/pi1_alfworld_k32_d0_plus_targeted.yaml --model-path "$TOK" > $B/logs/union_targeted.log 2>&1; echo "[$(date)] union exit $?"; grep -E "^(demonstrations|supervised_turns|bank_supervised_tokens):" $B/configs/rtd/pi1_alfworld_k32_d0_plus_targeted.yaml
cd $T; python3 - <<'PYEOF'
from pathlib import Path
src = Path('configs/rtd/pi1_alfworld_k32_all87_tok.yaml').read_text()
ref = Path('/home/xueqi/hq/projects/tc-alignment-baselines/configs/rtd/pi1_alfworld_k32_d0_plus_targeted.yaml').read_text()
def field(txt, key):
    for line in txt.splitlines():
        if line.startswith(key + ':'): return line.split(':',1)[1].strip()
out = src
for key in ('bank','sealed_manifest_sha256','demonstrations','supervised_turns','bank_supervised_tokens'):
    out = out.replace(f"{key}: {field(src,key)}", f"{key}: {field(ref,key)}", 1)
out = out.replace("# Frozen subset under the unchanged pre-registered Kang pi1 recipe.", "# Coverage-targeted ADD-DEMO arm: D0 + extra verified demos for cool/heat/two tasks; plain CE; token-count endpoints.")
Path('configs/rtd/pi1_alfworld_k32_d0_plus_targeted_tok.yaml').write_text(out); print('config', field(out,'demonstrations'), field(out,'bank_supervised_tokens'))
PYEOF
ln -sfn ../../tc-alignment-baselines/artifacts/alfworld_k32_d0_plus_targeted $T/artifacts/alfworld_k32_d0_plus_targeted
PYTHONPATH="$T/src:$T" $T/.venv/bin/python -B tools/alf_pi1_train.py --config configs/rtd/pi1_alfworld_k32_d0_plus_targeted_tok.yaml --bank artifacts/alfworld_k32_d0_plus_targeted --model-path "$TOK" --preflight > $T/logs/targeted_preflight.log 2>&1; echo "[$(date)] preflight exit $?"
rsync -a "$UNION" xueqic@smic.hpc.lsu.edu:/ddnA/work/xueqic/hq/tc-alf-banks/ && scp -q $T/configs/rtd/pi1_alfworld_k32_d0_plus_targeted_tok.yaml xueqic@smic.hpc.lsu.edu:/ddnA/work/xueqic/hq/tc-alf-taskeq/configs/rtd/ && echo "[$(date)] synced to LSU"
ssh -o BatchMode=yes xueqic@mike.hpc.lsu.edu 'set -u; H=/ddnA/work/xueqic/hq; T=$H/tc-alf-taskeq; BK=$H/tc-alf-banks; SNAP=$H/hf-cache/hub/models--google--gemma-4-12B-it/snapshots/707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7; PY=$H/tc-alf-baselines/.venv/bin/python; TPL=$H/slurm/train_gpu.slurm
for s in 0 1 2; do part=$([ $((s%2)) -eq 0 ] && echo gpu || echo gpu4); out=$T/results/d0targeted_v2/seed-$s; base="$PY -B $T/tools/alf_pi1_train.py --config $T/configs/rtd/pi1_alfworld_k32_d0_plus_targeted_tok.yaml --bank $BK/alfworld_k32_d0_plus_targeted --model-path $SNAP --gpu-uuid \"\$GPU_UUID\" --seed $s --output $out"; J=$(sbatch -J d0targeted_v2-s$s -p $part -t 05:30:00 --exclude=mike183 --export=ALL,TRAIN_CMD="$base --preflight && $base",R=$T $TPL 2>&1 | grep -oE "Submitted batch job [0-9]+" | grep -oE "[0-9]+$"); echo "d0targeted_v2-s$s ($part) -> ${J:-FAILED}"; done
sed -i "s/\"all87_2per_v2:PER2\"; do/\"all87_2per_v2:PER2\" \"d0targeted_v2:TARG\"; do/" $H/eval_when_ready.sh; echo "loop has TARG: $(grep -c TARG $H/eval_when_ready.sh)"' 2>&1 | grep -vE "Autoloading|^$|Loading"
echo "[$(date)] TARGETED CHAIN DONE (restart the mike eval loop to pick up TARG)"
