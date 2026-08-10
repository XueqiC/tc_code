# tc-alignment — experiment log

| exp_id | config | where | metrics | artifacts |
|---|---|---|---|---|
pilot-boundary-v1 | 5域×120, Qwen2.5-1.5B LoRA-B grad JL64×196=12544d, subspace 90%能量+conformal α=0.1 | rai GPU1 | hard-AUROC: grad 1.000 / emb-traj 0.687 / emb-query 0.404; cross-AUROC grad 1.0, cov .93-.97, FA 0; k=5 时 grad 已 1.0 | results/pilot/summary.md, results/figs/pilot_*.png
pilot-controls-v1 | C1: grad 随机投影至768同维; C2: 剥离 def solution()/return 模板后重提梯度 | rai GPU1 | grad-768 hard-AUROC .999; stripped 1.000 (reject 100%); stripped+768 1.000 → 两个替代解释均排除 | logs/pilot_controls.log
pilot-distill-v1 | Qwen2.5-1.5B LoRA r16 SFT, 90条 gsm8k-code 伪轨迹, 33 步 | rai GPU1 | gen: exec 0%→37%, format 0%→100%, CoT-any 67%→37%; loss in-T 1.67→0.17, out-T 全部升 (cot .64→1.11, alpaca 1.50→2.22); gradnorm in-T 5.1→2.6, out-T 全升 | results/pilot_distill/metrics.json, results/figs/pilot_distill.png
e1-intra-v1 | 域内细分边界: sql join/plain (47/73), pandas loc/plain (45/75), fit20/cal10/test15, 5 seed, 复用缓存特征 | rai CPU | grad AUROC 仅 0.62-0.75, emb-traj 在 3/4 对上更高 (sql-join 0.866) — 但细分标签本身按关键词定义, 偏袒 surface 方法; 结论: 粗粒度边界 grad 压倒性, 细粒度需 per-step stacking + 行为定义的细分标签 | results/pilot/e1_intra.md
e1-refmodel-v1 | 同批600样本在 Qwen3.5-4B/9B 上重提梯度特征 (hpg B200, 各~7.5min), hard_eval | hpg | 两个模型 hard-AUROC 1.000 / cross 1.000 / reject 100% — 子空间结构对参考模型稳健; 9B coverage 0.83 (名义0.9, n_cal=15 波动) | results/pilot/features_grad_qwen35-{4b,9b}.npz
