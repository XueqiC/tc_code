#!/bin/bash
#SBATCH --job-name=tc-smoke
#SBATCH --account=yd24f.fsu
#SBATCH --qos=yd24f.fsu
#SBATCH --partition=hpg-b200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=240gb
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x_%j.out

set -e
cd "$SLURM_SUBMIT_DIR"
echo "=== $(hostname) $(date) ==="
nvidia-smi
source .venv/bin/activate
python - <<'EOF'
import torch, time
print("torch", torch.__version__, "cuda", torch.version.cuda)
assert torch.cuda.is_available()
p = torch.cuda.get_device_properties(0)
print("device:", p.name, "capability:", p.major, p.minor, "mem GB:", p.total_memory // 2**30)
x = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
torch.cuda.synchronize(); t = time.time()
for _ in range(50):
    x = (x @ x).clamp(-1, 1)
torch.cuda.synchronize()
dt = time.time() - t
print(f"50x 8192^2 bf16 matmul: {dt:.2f}s ({50*2*8192**3/dt/1e12:.1f} TFLOPS)")
print("SMOKE_OK")
EOF
