# AgentOPSD (arXiv:2608.05987) 可借鉴点分析

来源:PI 推荐,2026-08-10。论文:turn 级 credit assignment 的递归自蒸馏
agentic RL(token 级 teacher-student log-prob gap → turn 级证据 → log-odds
空间递归贝叶斯信念更新 → 边际信念修正定位关键 turn;critic-free,
ALFWorld 89.1% @Qwen2.5-7B)。

## 与我们框架的映射
他们的 turn ≈ 我们的 block;他们的 log-prob gap ≈ 我们需求信号的
似然空间版本(我们的探针梯度是参数空间版本)。四个可借鉴点,按价值排序:

### A. Credit 加权的原子供给(修域内分辨率 + 选择质量)
现状:我们对轨迹所有 block 一视同仁(supply 求和、demand 取均值),
boilerplate 块和关键技能块权重相同——这正是域内分辨率弱(0.62-0.75,
细尺度 v0 失败)的另一面。借鉴:用 outcome-credit 给 block 重加权
(他们的边际信念修正,或我们更简的替代:leave-one-block-out 验证器差),
让 JOIN 块的原子在 supply/demand 里放大、样板块衰减。预期:域内分辨率
与选择质量同时受益,且与稀疏字典正交互补(稀疏管"拆出技能方向",
credit 管"哪个方向对成败要紧")。

### B. 递归信念更新稳需求估计(E2-v3 的降噪器)
我们逐块 demand 取 mean 聚合噪声大(30 题 ±9pt 的评测噪声也在此)。
借鉴:log-odds 空间的递归聚合替代算术平均,历史依赖的更新天然平滑
序列噪声。直接用于 E2-v3 的逐题能力预测聚合步,预注册 AUROC 门槛不变,
只换聚合器——干净的消融。

### C. Credit 加权 RFT(E4c 多轮塌缩的对症药)
E4c 诊断:多轮 RFT 均匀模仿自身成功轨迹的每个 token(含退化风格 token)
→ 漂移塌缩。他们的递归自蒸馏能 work 的关键正是 credit 把监督锚定在
outcome-pivotal 决策上而非均匀模仿。借鉴:RFT 的 SFT 步按 block-credit
加权 loss(pivotal 块全权重、其余衰减)。E4c-v2 的明确改法,一行公式:
L = Σ_b w_b · NLL_b,w_b = 信念修正幅度归一。

### D. 前向-only 需求代理(成本 3× 削减的可能)
他们的证据只需 teacher/student 两次前向的 log-prob gap,无反传。
若 block 级 log-prob gap 与我们的梯度 demand 高相关(可在现有工件上
直接验证:teacher 参考轨迹 + 学生 checkpoint 都在),边界/能力测量的
serving 侧就能用前向版,梯度版留给训练侧。验证实验半小时级。

## 适用域提醒(诚实)
他们是多轮 agentic RL(env 奖励),我们是蒸馏(验证器);turn↔block
是类比而非同构。但 tier-2 的 AppWorld/τ² 正是多轮设置——届时 turn 级
credit 是字面适用,这篇会成为 tier-2 方法节的直接引文。

## 排期
1. D 的相关性验证(现有工件,半小时,先做——若成立,B/C 都便宜);
2. C → E4c-v2(与 E4b/E4d 同批);
3. A → 域内分辨率复测 v3(credit × 稀疏 双因子);
4. bib 已核实(标题/作者/arXiv id 经 abs 页确认),写入 related work
   的 credit-assignment 句。
