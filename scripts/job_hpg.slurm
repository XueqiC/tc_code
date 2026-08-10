#!/bin/bash
#SBATCH --job-name=tc-alignment
#SBATCH --account=yd24f.fsu        # 组账号(8xB200,和组友共享); 系账号: fsu-compsci-dept (24xB200)
#SBATCH --qos=yd24f.fsu            # 与 account 同名; burst 低优先级: yd24f.fsu-b / fsu-compsci-dept-b
#SBATCH --partition=hpg-b200       # B200 节点: 112 cores / ~2TB / 8xB200 per node
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:b200:1          # 如报 gres 类型错误就改成 --gres=gpu:1
#SBATCH --cpus-per-task=14         # 约 14 cores / GPU
#SBATCH --mem=240gb                # 约 240GB / GPU
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out

set -e
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs results

echo "=== $(hostname) $(date) ==="
nvidia-smi

# 环境: 项目 venv 建在 /blue (home quota 很小)。B200=Blackwell, 需要 torch>=2.7 + cu128。
# 首次: cd 到项目目录后 uv venv && uv pip install torch --index-url https://download.pytorch.org/whl/cu128
source .venv/bin/activate 2>/dev/null || true

srun python src/train.py "$@"
