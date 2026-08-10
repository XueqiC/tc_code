# Pilot 实验报告:任务边界与能力边界能否在梯度空间统一度量

日期:2026-08-09 | 硬件:rai 单卡(97GB)| 总耗时:约 1.5 小时
代码:`src/{data_prep,grad_features,boundary,controls,distill_pilot}.py`
工件:`results/pilot/`、`results/pilot_distill/`、`results/figs/`
配套:英文方法设计 `method-design-v1.md`;本报告含当晚问答中的原理澄清。

---

## 0. 背景:今天要回答的两个问题

项目核心命题(2026-08-09 重构后):给定 few-shot query 定义的任务域 T,
蒸馏出能力边界 C 恰好匹配 T 的小 agent。难点是 T(query 空间)和 C(参数
空间)天然不同空间,"对齐"无从度量。我们的提案:**把两者都嵌入学生模型的
梯度空间**。今天用三个 pilot 验证这个提案的两条前提:

- **Q1(实验一、二)**:任务边界能否从少量 few-shot 样例在梯度空间估计?
  为什么必须是梯度空间而不是语义空间?
- **Q2(实验三)**:学生的能力边界能否在同一空间读出?蒸馏如何移动它?

## 1. 实验一 pilot-boundary-v1:边界估计

### 1.1 目标
验证:k 条样例的梯度张成的子空间 + conformal 阈值,能judge新 query 的
in/out,且在语义方法原理上失效的场景下仍然有效。

### 1.2 数据构造(5 域 × 120 条)
| 域 | 内容 | 解法行动空间 |
|---|---|---|
| gsm8k-code | GSM8K 题 + Python 解 | 代码 |
| gsm8k-cot | **同一批题** + 原文字推理解 | 自然语言 |
| pandas | DS-1000 Pandas 子集 | 代码 |
| sql | Spider | SQL |
| alpaca | 通用指令(域外池) | 自然语言 |

**困难对构造**(整个实验的判决性设计):gsm8k-code 的代码解由原 CoT 中的
`<<48/2=24>>` 计算注释**机械转换**——每个算式变赋值语句、复用前面结果的
数字替换成变量、执行验证与金标准一致才保留(零 LLM 噪声)。例:

> 题:"40 题的考试,Ella 错 4 题,Marion 比 Ella 的一半多 6 分,Marion 几分?"
> code 域解法:`def solution(): v0 = 40-4; v1 = v0/2; v2 = v1+6; return v2`
> cot 域解法:"Ella 答对 40-4=36 题。Marion 是 36 的一半加 6 = 24 分。"

题面一字不差,唯一变量 = 解法的行动空间("脑区")。两种边界理论在此预测
相反:按语义划 → 同一任务,分不开;按所需能力划 → 两个任务,分得开。

### 1.3 特征提取:冻结模型 + 三层压缩
**模型全程冻结,无任何训练。** 梯度是测量仪器:每条(query+解)前向算
loss、反传一次,抄下梯度即清零。"如果要教会你这道题,参数会想往哪动?"

完整梯度 1.5B 维/条存不起,三层压缩(每层有依据):
1. **LoRA-B 草图**:挂 LoRA(r8),B=0 初始化 ⇒ ∂L/∂B = G·Aᵀ,即完整梯度
   G 被随机矩阵 A 投影的草图,免费获得(LESS 观察);~10M 维。
2. **JL 投影**:196 个模块各压到 64 维(固定种子高斯),拼接 12544 维。
3. **L2 归一**。JL 引理保证两两夹角/距离近似保留,而方法只消费几何关系。
最终:每条 12544 维,600 条共 ~30MB,单卡 5 分钟。

### 1.4 边界构造与评测协议
- 每域 120 条按 35 fit / 15 calibration / 30 test 划分。
- 子空间:fit 集梯度 SVD,取覆盖 90% 谱能量的 top-r(实测 r≈20-25)。
- 隶属分数 s(q) = ‖U_Tᵀ g(q)‖/‖g(q)‖(投影能量比)。
- 阈值:split-conformal,α=0.1,calibration 分数的下分位 → 域内覆盖 ≥90% 保证。
- 基线:BGE-base embedding(仅 query / query+解)走**同一套**子空间+conformal 流程。
- 评测:AUROC(域内 test vs 其余域 test)、conformal 实测覆盖、域外误纳率;
  困难对单列;k ∈ {5,10,25,50} × 5 seed 的样本效率曲线。

### 1.5 结果
![PCA](../results/figs/pilot_pca.png)
**图1**:梯度空间(左)五域五簇,code(蓝)与 cot(橙)远离;embedding
(中,query+解)蓝橙混叠;embedding(右,仅 query)蓝橙完全重合(按构造)。

![ROC](../results/figs/pilot_roc_hard.png)
**图2 困难对 ROC**:grad **1.000** / emb-traj 0.687 / emb-query 0.404(≈随机,
验证构造干净)。conformal 下 grad 对 cot 的拒绝率 100%。

![k曲线](../results/figs/pilot_k_curve.png)
**图3 样本效率**:grad 在 **k=5 即 1.000 且死平**——"低维小脑区一圈就中"
的形态;emb-traj 随 k **下降**(0.80→0.60):语义圈越大越把同语义的 cot
框进来,自证圈错了维度。

其余:三个域的跨域 AUROC 全 1.0,覆盖 93-97%(名义 90%),误纳 0%。
完整表:`results/pilot/summary.md`。

## 2. 实验二 pilot-controls-v1:排除替代解释

| 怀疑 | 控制 | 结果 | 结论 |
|---|---|---|---|
| 12544 维太高,任何两堆点天然互远 | 随机投影至 768 维(与 emb 同维)重测 | hard-AUROC 0.999 | 靠低维结构,非高维巧合 |
| 子空间只在认 `def solution():` 模板 | 剥掉模板/return/缩进只剩裸算式重提梯度 | 1.000,拒绝率 100% | 圈的是"解题",非"排版" |

## 3. 实验三 pilot-distill-v1:能力边界的移动

### 3.1 目标与设置
真的动一次参数,用同一把尺子量 C 怎么变。Qwen2.5-1.5B + LoRA r16,
90 条 gsm8k-code 轨迹,off-policy SFT(response 段 NLL,prompt mask,
lr 2e-4,batch 1×accum 8,3 epoch = 33 步)。训练前后各测:
① 五域 held-out 的 teacher-forced loss;② 同样本的**残差梯度范数**
(当前参数处梯度还有多大,小=已学会);③ gsm8k-code 生成评测:贪心解码
→ 抽代码 → 沙箱执行 → 比金标准。

### 3.2 结果
![蒸馏](../results/figs/pilot_distill.png)
**图4**:左 loss、右残差梯度中位数;蓝=前、橙=后;高亮列=目标域。

| 指标 | 训练前 | 训练后 |
|---|---|---|
| 写代码率(format) | 0% | **100%** |
| 代码执行正确率(exec) | 0% | 37% |
| 任意方式答对率(any) | **67%(全靠 CoT)** | 37%(CoT 消失) |
| 域内 loss / 残差梯度 | 1.67 / 5.09 | **0.17 / 2.55** |
| 域外(4 域)loss 与残差 | — | **全部上升**(如 cot loss 0.64→1.11) |

### 3.3 三个发现
1. **行为切换 ≫ 能力迁移**:33 步足以彻底翻转行动空间(0%→100% 代码),
   能力只迁移一半(exec 37% < 原 CoT 67%)。凡只看格式的蒸馏评估都会
   高估能力迁移;capability 与 behavior 必须分开度量——从审稿隐患变成
   我们主动展示的现象。
2. **C 在梯度空间可读**:域内残差如期减半且与 loss、执行率三读数同向;
   域外全部反向上升。几何读数追踪真实能力。
3. **窄蒸馏自发塑形**:未加任何抑制项,域外能力全面退化。C⊆T 方向
   不需要从零发明机制;研究问题升级为"塑形速率多快、如何随步数/数据/
   模型大小变化、能否精确控制"。

## 4. 证据链(防伪等级从弱到强)
聚簇(图1)→ 判决性困难对(图2,两理论预测相反)→ 曲线形状 + 同维 +
剥模板(图3 + 控制,排除三个替代解释)→ 因果干预(图4,真训练后梯度
读数随能力同向移动)。conformal 覆盖 93-97% 兑现了统计保证。

## 5. Insights(五条)
1. 任务边界必须定义在能力空间;语义空间在"同题面不同行动空间"处原理性失效。
2. few-shot 定边界成立:k=5 饱和,且 embedding 随 k 变差——优势来自表示而非数据量。
3. 行为切换远快于能力迁移(33 步 vs 未完成)。
4. 窄蒸馏自发塑形;要研究的是速率与控制,不是机制的存在性。
5. T 与 C 可在同一空间度量——方法中心命题的两侧均获 pilot 级验证。

## 6. 局限(如实)
四个域彼此差异较大(域内细分未测);伪轨迹无 teacher 采样噪声;单一
参考模型(特征对 θ_ref 的稳健性未测);单 seed;split 曾用进程加盐 hash
(distill_pilot 已改 sha256,框架统一修复排在 W1);exec 沙箱是简易版。

## 7. 下一步实验如何开展(执行版)

**W1(至 8/16)——五线并行,两个 gate:**
1. **框架加固**(codex 写,我 review):stable-hash split 全局统一、
   verifier 沙箱化(subprocess + 资源限制)、特征提取抽成库、config 驱动。
2. **E1 扩展**(我,rai):①域内细分困难对(pandas 内 filtering vs
   groupby、SQL 单表 vs join)测边界分辨率;②"语义不同/能力同"反向
   困难对(gpt-5.4-mini 预算内改写题面);③参考模型稳健性(同批数据在
   Qwen3.5-2B、Llama-3.2-1B 上重提特征,子空间结构是否保持)。
3. **E2 gate 实验**(我,rai):沿 SFT 轨迹存 checkpoint,逐条样本记
   (残差梯度, 执行成败),验证 Spearman ρ ≤ −0.5——**过了这个 gate,
   几何 C 才允许进训练环**;不过则几何 C 只作分析仪器。
4. **teacher 轨迹池 v0**(codex 写限速客户端,rai CPU 跑):
   deepseek-v4-pro + kimi-k2.7-code 各 ~1k 条(gsm8k/pandas/sql seed),
   执行通过率作质检 gate,不合格不入池。
5. **环境接入**(codex 骨架,我设计):AppWorld + BFCL v4 装机冒烟,
   设计 topic 划分方案(哪些 app/工具簇当 in-domain,哪些当 out)。

**W2-W3——闭环主实验 E3 tier-1(rai)+ 上 hpg:**
- E3 六条件(全量 A / 随机 B / embedding 过滤 C / U_T 过滤 D / D+投影
  更新 E / D+refusal F)在混合轨迹池上、matched token budget,先在
  1.5B/2B 学生跑通(rai),报 (R_cov, L) frontier;
- 同步把 Qwen3.5-4B/9B 学生的相同流程搬 hpg(sbatch 模板已验证),
  开始 student 尺寸扫描 → minimal-capacity 曲线。
**W4-W5**:全量 E3/E4 sweep(hpg)+ 消融(逐层/逐步 stacking、投影更新
强度、on-policy 蒸馏扩展条件)。**W6**:冻图写作。

每步产出进 exp_log.md,重要结果实时报 Discord;hpg 提交遵守 ≤4 pending。
