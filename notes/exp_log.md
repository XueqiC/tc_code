# tc-alignment — experiment log

| exp_id | config | where | metrics | artifacts |
|---|---|---|---|---|
pilot-boundary-v1 | 5域×120, Qwen2.5-1.5B LoRA-B grad JL64×196=12544d, subspace 90%能量+conformal α=0.1 | rai GPU1 | hard-AUROC: grad 1.000 / emb-traj 0.687 / emb-query 0.404; cross-AUROC grad 1.0, cov .93-.97, FA 0; k=5 时 grad 已 1.0 | results/pilot/summary.md, results/figs/pilot_*.png
pilot-controls-v1 | C1: grad 随机投影至768同维; C2: 剥离 def solution()/return 模板后重提梯度 | rai GPU1 | grad-768 hard-AUROC .999; stripped 1.000 (reject 100%); stripped+768 1.000 → 两个替代解释均排除 | logs/pilot_controls.log
pilot-distill-v1 | Qwen2.5-1.5B LoRA r16 SFT, 90条 gsm8k-code 伪轨迹, 33 步 | rai GPU1 | gen: exec 0%→37%, format 0%→100%, CoT-any 67%→37%; loss in-T 1.67→0.17, out-T 全部升 (cot .64→1.11, alpaca 1.50→2.22); gradnorm in-T 5.1→2.6, out-T 全升 | results/pilot_distill/metrics.json, results/figs/pilot_distill.png
e1-intra-v1 | 域内细分边界: sql join/plain (47/73), pandas loc/plain (45/75), fit20/cal10/test15, 5 seed, 复用缓存特征 | rai CPU | grad AUROC 仅 0.62-0.75, emb-traj 在 3/4 对上更高 (sql-join 0.866) — 但细分标签本身按关键词定义, 偏袒 surface 方法; 结论: 粗粒度边界 grad 压倒性, 细粒度需 per-step stacking + 行为定义的细分标签 | results/pilot/e1_intra.md
e1-refmodel-v1 | 同批600样本在 Qwen3.5-0.8B/2B/4B/9B(instruct) 上重提梯度特征 (hpg B200) | hpg | 四个尺寸 hard-AUROC 全部 1.000 / cross 1.000 / reject 100% — 子空间结构对参考模型完全稳健; -Base 系与 Qwen3.6-27B 因组盘配额失败已改节点本地缓存重提 | results/pilot/features_grad_qwen35-*.npz
e2-gate-v1 | 66步SFT×8 checkpoint, 逐样本(残差梯度,exec成败), 预注册 pooled Spearman ρ≤-0.5 | rai GPU1 ~2.5h | GATE FAIL: pooled ρ=-0.403; 但 per-checkpoint 后期 ρ=-0.70/-0.52, 最低梯度 decile 成功率 96%; 梯度范数后期回升(风格漂移混淆) → 按预注册几何C不进训练环; E2-v2 改 within-checkpoint 归一 | results/e2_gate/records.json, results/figs/e2_gate.png
e1-intra-fine-v1 | per-line 块梯度, mean/max/top3 聚合, 域内复测 | rai GPU1+CPU | 整体未破 0.77: SQL 单行响应退化为 1 块(fine≡coarse), pandas 多行 max 聚合 0.60→0.68; 结论: 细尺度需子语句级切块(SQL 子句/AST), 排后续; 粗尺度主场景不受影响 | results/pilot/e1_intra_fine.md
e1-refmodel-27b | Qwen3.6-27B (跨代, 节点本地缓存) 特征提取 14min | hpg B200 | hard-AUROC 1.000 / cross 1.000 / coverage 0.97 — 稳健性覆盖 0.8B→27B 跨两代五个模型 | results/pilot/features_grad_qwen36-27b.npz
e2-gate-v2 | 修正案A1: within-checkpoint z归一主分析(CPU复用v1记录) | rai | FAIL: ρ=-0.303(比raw -0.403更弱; z归一移除跨checkpoint真实信号+早期全失败记录稀释) → gate 终审 FAIL, 几何C留在训练环外, 不再换指标; 自轨迹残差诊断(预注册副分析)GPU2在跑 | results/e2_gate/v2_analysis.json
e2-v2-secondary | 学生自轨迹残差 (风格漂移诊断, GPU2) | rai | ρ=-0.447: 优于 teacher-ref (-0.403) 与 z-norm (-0.303) 但仍未达 -0.5; 逐行确认漂移(如 row25 teacher 残差 15.06 vs own 1.35) → 风格漂移是真实混淆但非全部; 残差梯度=有信息但不完美的能力代理, 终审维持 FAIL | logs/e2_v2_self.log
e1-refmodel-base | Qwen3.5-{0.8,2,4,9}B-Base 特征 (节点本地缓存) | hpg B200 ×4 | 全部 hard-AUROC 1.000 — 稳健性矩阵完成: 9个参考模型(instruct×5含Qwen2.5-1.5B, base×4, 跨代27B)全1.000; RLHF与否不影响任务子空间 | results/pilot/features_grad_qwen35-*-base.npz
e3-tier1 | 6条件×40k budget, 1.5B instruct, 池=2teacher×3域+gold干扰 | rai GPU1 ~5h | 域内: D/B 90% 并列最高(饱和); 泄漏: D 域外loss最高(削最尖), F 域外拒答100%/域内误拒0%/-6.7pt; 行动空间俘获全条件100%; 梯度边界更紧(153 vs 264过阈) | results/e3_tier1/, figs/e3_tier1.png
atoms-pilot-v1 | 256原子 alpha=0.5 稀疏分解 2072块 | rai CPU | 超参失败(平均0.5激活/块, AUROC 0.55-0.65) 但可解释性命中: join判别原子top块全是JOIN语句; v2 (128原子 alpha=0.05) 在跑 | results/atoms_pilot/report.json
atoms-pilot-v2 | 128原子 alpha=0.05 (5.2激活/块) | rai CPU | 细粒度突破: sql-join 0.906±0.025 (密子空间0.615/emb 0.866全超); 粗粒度追平: code 0.996(hard 0.988)/sql 0.989/pandas 0.936; JOIN原子可解释性再确认 → 单一表示统一两尺度, 方法设计线收敛到 Capability Atoms | results/atoms_pilot/report.json
m1-interference | 核质量 vs E3实测Δloss (15 cells), 负内积占比 | rai GPU1 | Spearman ρ=0.529 p=0.043 (方向性预测成立); 负内积仅1-15% → 削尖≠梯度对抗, 是"断供+漂移"效应 → 泄漏控制=供给分配(选择时可控), 塑形控制项设计依据确立 | results/m1/report.json
m2m3-v3 | 双模型探针协议(参考提取器+W_eff拷贝), 8 checkpoint, 128原子空间 | rai GPU1 ~1.5h | M2: 原子需求单调排空 1.01→0.11 (-89%) 伴随 exec 0→57%, 且呈顺序波(原子0 step2前排空=格式候选, 原子8/15 接力达峰再排空); M3: 行为翻转区间(0→2)更新 85.5% 集中于 5/128 原子 — 低维载体证实; 前两版失败(stdin/基不匹配)已记方法论教训 | results/m2_m3/, figs/m2_m3.png
