#!/bin/bash
#SBATCH --job-name=alfbase
#SBATCH --account=fsu-compsci-dept
#SBATCH --qos=fsu-compsci-dept
#SBATCH --partition=hpg-b200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=120gb
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
# ALFWorld official valid_unseen eval of the untrained 4B base under the ReAct deployment
# scaffold (bfas adapter, BFAS_ALFWORLD_STUDENT_REACT=1 default). Reference for the 4B lane.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export HF_HOME=/blue/fsu-compsci-dept/xc25.fsu/hq/tools/hf-cache
source .venv/bin/activate
export PYTHONPATH=src GPU_UTIL=0.6 BFAS_PORT_BASE=8930 VLLM_USE_FLASHINFER_SAMPLER=0
# vllm compiles a kernel at startup and needs ninja on PATH (hpg: FileNotFoundError ninja)
export PATH=$PWD/envs/vllm/.venv/bin:$PATH
python tools/bfas_eval_ckpt.py --benchmark alfworld --checkpoint Qwen/Qwen3.5-4B \
  --out results/bfas/alfworld/base4b_s0 --gpu 0 --port 8930 --seed 0
echo "[alfbase] DONE"
