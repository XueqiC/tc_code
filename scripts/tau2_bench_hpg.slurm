#!/bin/bash
#SBATCH --job-name=tc-tau2
#SBATCH --account=fsu-compsci-dept
#SBATCH --qos=fsu-compsci-dept
#SBATCH --partition=hpg-b200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=120gb
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%A_%a.out

# Manifest line: ARM=<...> SEED=<n>  (same manifest as basebench)
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export HF_HOME=/blue/fsu-compsci-dept/xc25.fsu/hq/tools/hf-cache
mkdir -p logs results/tau2
manifest=${BB_MANIFEST:-configs/baseline_bench_manifest.txt}
line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$manifest")
[ -n "$line" ] || { echo "no manifest line"; exit 1; }
unset ARM SEED; read -r -a kv <<< "$line"; export "${kv[@]}"
TAG="hb_${ARM}_s${SEED}"
echo "[tau2] ARM=$ARM SEED=$SEED"

if [ "$ARM" = "base" ]; then
  SERVE_PATH="Qwen/Qwen3.5-4B"
else
  ADAPTER=results/appworld_students/$TAG/adapter
  [ -d "$ADAPTER" ] || { echo "[tau2] $TAG missing student"; exit 1; }
  SERVE_PATH=results/appworld_students/$TAG/export_vllm
  if [ ! -f "$SERVE_PATH/model.safetensors" ]; then
    .venv/bin/python - "$ADAPTER" "$SERVE_PATH" << 'PYEOF'
import sys, shutil, os
from safetensors.torch import safe_open, save_file
src, dst = sys.argv[1], sys.argv[2]
os.makedirs(dst, exist_ok=True)
t = {}
with safe_open(os.path.join(src, "model.safetensors"), "pt") as f:
    for k in f.keys():
        t[k.replace("model.language_model.", "model.")] = f.get_tensor(k)
save_file(t, os.path.join(dst, "model.safetensors"), metadata={"format": "pt"})
for n in os.listdir(src):
    if n != "model.safetensors":
        shutil.copy(os.path.join(src, n), os.path.join(dst, n))
print("exported", dst)
PYEOF
  fi
fi

PORT=8903
VLLM_USE_FLASHINFER_SAMPLER=0 nohup envs/vllm/.venv/bin/vllm serve "$SERVE_PATH" \
  --served-model-name student --port $PORT \
  --gpu-memory-utilization 0.85 --max-model-len 32768 \
  > "logs/vllm_tau2_${TAG}.log" 2>&1 &
SPID=$!
ok=""
for i in $(seq 1 120); do
  curl -s -o /dev/null -w "%{http_code}" http://localhost:$PORT/v1/models 2>/dev/null | grep -q 200 && { ok=1; break; }
  kill -0 $SPID 2>/dev/null || break
  sleep 5
done
[ -n "$ok" ] || { echo "[tau2] $TAG SERVER FAILED"; exit 1; }

export OPENAI_API_KEY=local
OLLAMA_KEY=$(cat /blue/fsu-compsci-dept/xc25.fsu/hq/.ollama_api_key)
envs/tau2/.venv/bin/tau2 run \
  --domain retail --agent llm_agent --user user_simulator \
  --agent-llm openai/student \
  --agent-llm-args "{\"api_base\": \"http://localhost:$PORT/v1\", \"api_key\": \"local\", \"temperature\": 0.0}" \
  --user-llm openai/deepseek-v4-pro \
  --user-llm-args "{\"api_base\": \"https://ollama.com/v1\", \"api_key\": \"$OLLAMA_KEY\"}" \
  --num-trials 1 --seed 0 --max-concurrency 4 \
  --save-to "results/tau2/${TAG}" \
  > "logs/tau2_run_${TAG}.log" 2>&1 || { echo "[tau2] $TAG RUN FAILED"; tail -5 "logs/tau2_run_${TAG}.log"; kill $SPID; exit 1; }
kill $SPID 2>/dev/null || true
echo "[tau2] $TAG DONE"
