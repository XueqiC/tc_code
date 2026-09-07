#!/bin/bash
#SBATCH --job-name=cc_eval_arms
#SBATCH --account=fsu-compsci-dept
#SBATCH --qos=fsu-compsci-dept
#SBATCH --partition=hpg-b200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=240G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
# Submit from the project root with logs/ present. Evaluation only (C22).
# For yd24f, override both --account=yd24f and --qos=yd24f at submission.
# CC_EVAL_TAGS='cc_s1_A_s0 cc_s1_B_s0 cc_s1_C_s0' by default; sequential.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project root}"
export HF_HOME=/blue/fsu-compsci-dept/xc25.fsu/hq/tools/hf-cache
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
source .venv/bin/activate
export VLLM_CACHE_ROOT=/blue/fsu-compsci-dept/xc25.fsu/hq/tools/vllm-cache TORCHINDUCTOR_CACHE_DIR=/blue/fsu-compsci-dept/xc25.fsu/hq/tools/inductor-cache
export TRITON_CACHE_DIR=/blue/fsu-compsci-dept/xc25.fsu/hq/tools/triton-cache XDG_CACHE_HOME=/blue/fsu-compsci-dept/xc25.fsu/hq/tools/xdg-cache
mkdir -p "$VLLM_CACHE_ROOT" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME"
export AW_MAX_PROMPT_TOKENS=4096 AW_TRUNCATE_SIDE=tail AW_DISTILL=ddpo AW_SFT_SKIP=1

read -r -a tags <<< "${CC_EVAL_TAGS:-cc_s1_A_s0 cc_s1_B_s0 cc_s1_C_s0}"
[ "${#tags[@]}" -gt 0 ] || { echo "[cc eval arms] No evaluation tags supplied" >&2; exit 2; }
tag_index=0
for tag in "${tags[@]}"; do
    [[ $tag =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || { echo "Invalid tag: $tag" >&2; exit 2; }
    port=$((10000 + (SLURM_JOB_ID + tag_index) % 50000))
    tag_index=$((tag_index + 1))
    if [ -f "results/bfcl_std/$tag/data_overall.csv" ]; then
        echo "[cc eval arms] $tag already scored, skip"
    else
        echo "[cc eval arms] $tag evaluate on port $port"
        bash tools/bfcl_std_campaign.sh 0 "$port" "$tag"
        if [ ! -f "results/bfcl_std/$tag/data_overall.csv" ]; then
            echo "[cc eval arms] $tag NO SCORE PRODUCED" >&2
            exit 1
        fi
    fi
    python3 tools/bfcl_print_scores.py "results/bfcl_std/$tag/data_overall.csv" "$tag" \
        --prefix '[cc eval arms]'
done
echo "[cc eval arms] COMPLETE ${tags[*]}"
