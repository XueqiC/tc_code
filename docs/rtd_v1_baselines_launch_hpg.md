# v1.0 强对照(§10.3)hpg 启动命令 — 2026-09-09(用户批准"v1.0 基线跑")

前提:configs/rtd/baselines/generated_hpg_v10(A0=R0、A1=R1,B200 class 01be985e…)已由 prepare 生成。

## 1. tune(B1/B2 各 6 trial,pool A1,slots 视角;单 B200 作业)
```bash
cd /blue/fsu-compsci-dept/xc25.fsu/hq/tc-alignment
sbatch --account=fsu-compsci-dept --qos=fsu-compsci-dept --partition=hpg-b200 --nodes=1 --ntasks=1 \
  --gres=gpu:b200:1 --cpus-per-task=8 --mem=64G --time=48:00:00 --job-name=bl_tune \
  --output=logs/%x_%j.out --wrap='source tools/aw_hpg_common.sh; aw_hpg_environment; export PYTHONPATH=src:. PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1; .venv/bin/python tools/rtd_baselines.py tune --config configs/rtd/baselines/bfcl_tuning.yaml --prepared configs/rtd/baselines/generated_hpg_v10 --out results/rtd_baselines/tuning_hpg_v10'
```

## 2. run(20 个配置;B1/B2 用 tuning 的 selection.json,B3/B4 固定配置;每个一张 B200,≤4 个同时排队)
```bash
for cfg in A0_b1_slots A0_b1_tokens A0_b2_slots A0_b2_tokens A1_b1_slots A1_b1_tokens A1_b2_slots A1_b2_tokens; do
  sbatch ... --job-name=bl_$cfg --wrap='... tools/rtd_baselines.py run --config configs/rtd/baselines/generated_hpg_v10/'$cfg'.yaml --selection results/rtd_baselines/tuning_hpg_v10/selection.json --run-dir results/rtd_baselines/'$cfg'_s0'
done
for cfg in A0_b3_sft_slots A0_b3_sft_tokens A0_b3_mix_slots A0_b3_mix_tokens A0_b4_slots A0_b4_tokens A1_b3_sft_slots A1_b3_sft_tokens A1_b3_mix_slots A1_b3_mix_tokens A1_b4_slots A1_b4_tokens; do
  sbatch ... --job-name=bl_$cfg --wrap='... tools/rtd_baselines.py run --config configs/rtd/baselines/generated_hpg_v10/'$cfg'.yaml --run-dir results/rtd_baselines/'$cfg'_s0'
done
```
B3/B4 不依赖 tuning,可与 tune 同时提交(先提 B3/B4 12 个,分批 ≤4 pending)。

## 3. 官方评测(每个 run 完成后,第 3 轮 checkpoint)
```bash
.venv/bin/python tools/rtd_experiment.py evaluate --run-dir results/rtd_baselines/<cfg>_s0 --round 3 --evaluation-lock-timeout 21600 --evaluation-lock-log-interval 60
```
可用 --dependency=afterok:<run job> 直接排在 run 作业后。
