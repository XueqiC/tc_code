# tc-alignment — experiment log

| exp_id | config | where | metrics | artifacts |
|---|---|---|---|---|
pilot-boundary-v1 | 5域×120, Qwen2.5-1.5B LoRA-B grad JL64×196=12544d, subspace 90%能量+conformal α=0.1 | rai GPU1 | hard-AUROC: grad 1.000 / emb-traj 0.687 / emb-query 0.404; cross-AUROC grad 1.0, cov .93-.97, FA 0; k=5 时 grad 已 1.0 | results/pilot/summary.md, results/figs/pilot_*.png
pilot-controls-v1 | C1: grad 随机投影至768同维; C2: 剥离 def solution()/return 模板后重提梯度 | rai GPU1 | grad-768 hard-AUROC .999; stripped 1.000 (reject 100%); stripped+768 1.000 → 两个替代解释均排除 | logs/pilot_controls.log
