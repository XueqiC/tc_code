# Pilot 实验设计:梯度谱子空间能否判任务边界?

日期:2026-08-09。前置:方法草图(gradient→spectral sketch)。状态:待用户确认后开跑。

## 要验证的三件事
- **P1 可分性**:不同 topic 的 query 在梯度特征空间形成可分的谱子空间。
- **P2 判别力**:投影能量比 + conformal 校准能判别 held-out query 的 in/out,
  且在"语义相近但所需能力不同"的困难对上**优于 semantic embedding 基线**
  ——这是方法必要性的关键证据。
- **P3 样本效率**:k(few-shot 数)从 5→100 的边界质量曲线。

## 设置
**特征提取模型**:Qwen2.5-1.5B-Instruct(单卡即可),LoRA(attn+MLP, r=8)。
每条样本:对其"轨迹"(query+代码解)算 loss → 反传 → 收集 LoRA 梯度 →
随机投影到 8192 维 → L2 归一(LESS 式)。

**任务域(3+1 个,都有现成代码解,pilot 不必调 teacher API)**:
- T1 math-with-code:GSM8K 题 + Python 解(现成 PoT/PAL 语料);
- T2 表格数据分析:DS-1000 / pandas 子集;
- T3 SQL 查询:Spider 子集;
- **困难对 T1′** math-with-CoT:同样的 GSM8K 题、自然语言推理解(无代码)。
  T1 vs T1′ 语义 embedding 上几乎不可分(题面相同),但行动空间完全不同
  ——梯度特征若能分开,就是"边界必须在能力空间定义"的直接证据(核心图)。

**数据划分**:每域 k=50 定子空间(top-r 由谱能量 90% 选),held-out 每域 30 做
in-domain 测试;out = 其他域 held-out + 通用指令池(Alpaca 抽样)。

**判别与指标**:
- 隶属分数 s(q) = ‖U_Tᵀ g(q)‖ / ‖g(q)‖;split-conformal 校准 α=0.1。
- P1:域间子空间主角谱、梯度特征聚类纯度(vs 域标签)。
- P2:AUROC(in vs out);conformal 实测覆盖(目标 ≥90%)与域外误纳率;
  **T1 边界对 T1′ 的拒绝率**(困难对单列)。
- P3:k ∈ {5,10,25,50} 重复 5 seed 的 AUROC 曲线。
- 基线:BGE-large embedding + kNN/Mahalanobis 分数走同一 conformal 流程;
  次基线:模型 last hidden state 均值特征。消融:投影维度、LoRA 位置、r。

**成功标准(预注册)**:
- P2 主判据:困难对 T1/T1′ 上梯度 AUROC ≥ 0.85 且高于 embedding 基线 ≥ 0.10;
  普通跨域上不低于 embedding 基线。
- conformal 覆盖在 [88%, 94%] 内(α=0.1 名义值附近)。
- 若失败:先检查梯度特征对 loss 归一/长度的敏感性(长度 confound 老问题),
  再考虑逐步(per code block)梯度;两轮都失败则边界表示需重新设计,不进蒸馏阶段。

**算力与时长**:~500 条样本 × 一次 1.5B LoRA 反传,rai GPU 1 单卡,
预计 2-4 小时(含 embedding 基线)。不用 hpg。

## 产出
- `src/grad_features.py`(特征提取)、`src/boundary.py`(子空间+conformal)、
  `configs/pilot.yaml`、`results/pilot/`,exp_log.md 记一行。
- 图:①主角谱热图 ②t-SNE/PCA of 梯度特征 vs embedding(四域着色)
  ③困难对 ROC ④k-效率曲线。
