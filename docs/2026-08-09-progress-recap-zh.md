# 今日验证型研究 Recap · 方法全貌 · 下一步计划

2026-08-09 深夜。配套英文方法文档:`docs/method-design-v1.md`(ICLR 标准)。
所有数字可由 `src/` 复现,原始工件在 `results/`。

---

## 一、今天做了哪三个验证实验,结果怎么看

### 实验 1:pilot-boundary-v1 —— 任务边界能否从 few-shot query 在梯度空间估计?
**设置**:5 个域各 120 条(GSM8K-代码解 / GSM8K-CoT 文字解 / pandas /
SQL / 通用指令池)。GSM8K 两个域**题面完全相同**,只差解法的行动空间——
这是刻意构造的"困难对":语义特征按构造分不开,能力特征应该分得开。
特征 = 在 Qwen2.5-1.5B 上对每条(query+解)反传一次,取 LoRA-B 梯度
(数学上恰是全梯度的随机投影,LESS 技巧),12544 维。
边界 = k 条 few-shot 的梯度做谱分解取子空间 U_T,新 query 的隶属分数 =
投影能量比,conformal 校准出带覆盖保证的阈值。

**结果**:困难对 AUROC:**梯度 1.000** / embedding(query+解)0.687 /
embedding(仅 query)0.404≈随机。跨域全 1.0、误纳 0%、conformal 覆盖
93–97%(名义 90%)。**k=5 就饱和**;embedding 反而随 k 增大变差(0.80→0.60,
子空间大了把 CoT 也吸进去)。

**怎么看**:边界估计这条腿成立,且样本效率远超预期。

### 实验 2:pilot-controls-v1 —— 两个"结果太好"的替代解释排除
① 把 12544 维投到 768 维(和 embedding 同维)重测:AUROC 0.999——
不是高维几何白送分。
② 把代码里 `def solution():`/`return`/缩进全剥掉只剩裸算式,重提梯度:
AUROC 仍 1.000、拒绝率 100%——不是在认模板 token。

**怎么看**:梯度子空间编码的确实是"学这个任务需要动哪些参数",
不是表面统计量。

### 实验 3:pilot-distill-v1 —— 学生的能力边界 C 在同一空间可测吗?蒸馏怎么动它?
**设置**:90 条 gsm8k-code 轨迹 SFT 1.5B 学生(LoRA r16,33 步),
训练前后测五个域的 teacher-forced loss、逐条残差梯度范数、生成+执行准确率。

**结果**:
- 行动空间 33 步完全翻转:写代码率 0%→100%,CoT 输出消失;
- 但能力只迁移一半:代码执行正确率 0%→37%,而训练前用 CoT 能对 67%;
- 域内 loss 1.67→0.17、残差梯度中位数 5.1→2.6;
- **四个域外全部变差**(cot loss 0.64→1.11,alpaca 1.50→2.22;残差梯度全升)。

**怎么看**:C 的变化在梯度空间方向清晰可读;且窄蒸馏**自发**把学生
"削尖"进目标域,没加任何显式抑制。

## 二、五条核心 insight

1. **任务边界必须定义在能力空间,不是语义空间。** 同题面不同行动空间,
   embedding 原理上分不开,梯度空间干净分开——这是方法必要性的直接证据,
   也是论文动机图。
2. **few-shot 定边界是真的:k=5 够了。** 且 embedding 基线随 k 变差,
   优势不是"多数据碾压"而是表示本身对。
3. **行为切换 ≫ 能力迁移(速度)。** 33 步就换了行动空间,能力才到一半。
   capability vs behavior 必须分开度量——从"审稿人可能挑刺的点"
   变成了"我们主动展示的现象"。
4. **窄蒸馏自发塑形。** C⊆T(域外抑制)方向不必从零发明机制,
   要做的是**量化和控制塑形速率**——研究问题从"能不能"变成"多快、可控吗"。
5. **T 和 C 真的可以放进同一个空间度量。** T=子空间+conformal 半径,
   C=残差梯度低的区域,对齐=几何量。这是整个方法的中心命题,pilot 两侧都验了。

## 三、方法从头到尾(当前版本一页图景)

```
输入: teacher π_T(以代码行动) + k 条 few-shot query + 学生 base θ0
 1. 每条 query → teacher 轨迹 τ(q)(代码)
 2. g(q) = θ0 上 τ(q) 的 LoRA-B 梯度 → JL 投影 → 归一   [能力空间坐标]
 3. 谱分解 {g(qi)} → 任务子空间 U_T + conformal 阈值 t_α  [任务规格 = spec]
    → 边界判别器 B̂(q) = [投影能量比 ≥ t_α],带 (1−α) 覆盖保证
 4. teacher 生成候选轨迹池 D(含近域/离域杂质)
 5. 蒸馏 = 按 s(x) 选择/加权 D(主) ± 更新投影回 U_T(消融)
    ± 域外 refusal 增强(行为模块,单独报告)
 6. 学生能力读数: 残差梯度 ρ(q) ↓ + 执行验证 ✓  [C 的几何读法+行为读法]
 7. 评测: (R_cov, L) 双侧 frontier + 最小容量曲线,
    per-task 成败模式(IRT 对齐),全部预注册
```

## 四、下一步实验计划(已吸收今晚四点拍板)

### 环境面板(SOTA 对齐)
- **AppWorld**(主环境):agent 写 Python 调 457 个 API 完成 9 个模拟 app
  的任务,执行级验证,ACL 2024 → AppWorld-UL ICML 2026 一脉,是当前
  code-acting agent 研究的标准环境,和我们"LLM 用 code 行动"的设定原生一致。
- **BFCL v4**(agentic + multi-turn 子集):2026-04 改版后是 function-calling
  的社区标准榜。
- **τ²-bench**(retail/telecom 域切片):tool-agent-user 交互的 2025-26 标准,
  其域结构天然适合当我们的 topic 边界(retail=in,telecom=out 等)。
- Tier-1 静态四域 + 域内细分保留(便宜、可控、出机理图)。

### Teacher(2-3 个,已拿到两组凭证,均存 ~/hq/secrets/ 不进 git)
- **bulk 轨迹生产**:ollama cloud —— 首选 **deepseek-v4-pro**(通用 SOTA)
  + **kimi-k2.7-code**(代码专长);备选 **qwen3.5:397b**(与学生同家族,
  可作"同族 vs 跨族蒸馏"的受控变量,科学价值高)。
- **gpt-5.4-mini(Azure)只做小规模 probe**:3 把 key × 100k tokens/周 =
  共 300k/周,按每条轨迹 1–2k tokens 只够 150–300 条/周,当不了 bulk teacher;
  用于困难对构造、验证性对照、judge 类小任务。
- 兜底:若 ollama 配额/速率不够,hpg B200 上 vLLM 自托管开源 teacher
  (qwen3.5-27B/397B 或 deepseek 蒸馏版),无限量且完全可复现。
- 凭证安全:key 已在 Discord 消息里出现过,建议方便时轮换一次。

### 学生尺寸线(最新最丰富)
- 主线:**Qwen3.5 0.8B / 2B / 4B / 9B**(2026-03 发布,Apache 2.0,
  thinking/non-thinking 双模,当前最新的完整小模型线)。
- 跨家族对照点:1 个(Gemma-4 小杯或 Llama 系,按届时可得性定)。
- 加上 teacher 侧 qwen3.5:397b,可组成同族 397B→{0.8,2,4,9}B 的干净缩放线。

### 代码生产流程
- 此后实验代码默认委派 codex(模型 gpt-5.6-sol,reasoning **xhigh**,
  workspace-write 沙箱),`ops/codex_task.sh` 已配好;我负责实验设计、
  委派任务书、review diff、跑 sanity check、结果分析——按 CLAUDE.md
  的 Delegating to Codex 协议执行。

### 本周(W1)清单
1. codex:框架加固(stable-hash split、verifier 沙箱、feature 缓存层)
2. codex:AppWorld + BFCL v4 环境接入骨架;我设计 topic 划分方案
3. E1 扩展:域内细分困难对 + 参考模型稳健性(rai)
4. E2 gate 实验:残差梯度 vs 执行成功率的相关性(rai)
5. teacher 轨迹池 v0:ollama cloud 双 teacher 各 ~1k 条(限速下分批)
```
