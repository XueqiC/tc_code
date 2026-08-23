# 主实验设计 v2(2026-08-23,按定位重构)

**定位一句话**:我们是"面向 LLM base model 的预算化蒸馏方法"——
主表对手是 SOTA 蒸馏方法,不是 data selection;selection 与
distillation 两个部件各用一张对照表单独证明。

## Table 1 · 主表:蒸馏方法对决

**行(SOTA 蒸馏 baseline,全部同预算同 teacher)**:
1. Base student(零样本下界)
2. Vanilla SFT / 行为克隆(Orca、AgentTuning 风格:teacher 示范全量
   SFT——蒸馏领域的标准做法)
3. Structured Agent Distillation(liu2025structured,推理/动作分段
   损失)
4. SmartAD(tang2026smartad,容量对齐轨迹选择蒸馏)
5. On-policy 蒸馏(黑盒 hard-label 版:学生 rollout + teacher 重标注,
   DAgger 风格——代表 OPD 家族在无 logits 约束下的可实现形态)
6. Self-improvement(STaR 风格:自采样 + 验证过滤再训——无 teacher
   增益的对照)
7. **Ours(v2.0 全法)**
每格 3 种子 mean±std;粗体列最优;附 Δ vs base 行(正/负一目了然——
我们的核心卖点"唯一稳定为正"直接可见)。

**列(benchmark 面板,新+主流+对小学生可行)**:
- **AppWorld**(2024,多 App 日常事务 agent;官方 TGC/SGC,
  test_normal)——难度锚点,已有完整管线;
- **BFCL v4**(2025,Berkeley 函数调用榜;工具调用准确率)——最主流
  的 tool-use 评测,本地轻 harness,小模型有充分信号,行业对齐度
  最高;
- **τ²-bench**(2025,客服多轮工具 agent;pass^1)——最新主流对话
  agent 台;user-sim 用我们的 teacher API 兼任,费用计入预算,
  与设定自洽;
- **ALFWorld**(具身文本 agent;成功率)——经典可行台,保证小学生
  不全零,给"方法在不同 agent 形态上都成立"的广度。
落选说明:SWE-bench/WebArena/OSWorld(≤9B 学生近零分,无法展现
差异);GAIA(无干净 verifier 进训练环);WebShop(旧且信号弱于
ALFWorld);BIRD/DS-1000(非 agent,移作 Table 2/3 的开发台与
appendix 广度)。
**模型固定(正文全程唯一配置)**:teacher = deepseek-v4-pro(唯一
计费 teacher),student = **Qwen3.5-4B**——正文所有表格(1-5)只用
这一对;2B / 9B / 0.8B 的全部结果(规模趋势、9B 的 .321 等)移入
appendix 作规模研究,正文不混用。

## Table 2 · Data selection 对照(蒸馏部件固定为 Ours)

行 = random / embedding 检索 / LESS / SmartAD-选择 / Self-Instruct /
Evol-Instruct / LLM2LLM / **Ours(边际规则采购)**;
列 = 受控套件(外部 200 题)+ AppWorld + BFCL(轻量二台);
全部 + 我们的蒸馏(锚定分层 + 门控目标)。
主张:任何 selection 配我们的蒸馏都不掉,换成我们的 selection 再升
(尤其浪费/覆盖两个效率指标列入)。

## Table 3 · Distillation 对照(selection 固定为 Ours,行与 Table 1 对齐)

行 = **与 Table 1 完全相同的 SOTA 蒸馏家族**(Vanilla SFT 克隆 /
Structured AD / SmartAD 蒸馏部件 / 黑盒 on-policy / STaR)+ **Ours**,
但全部喂**同一批我们采购的数据**;列同 Table 2。
主张(与 Table 1 呼应,杀伤力最大):即便给这些 SOTA 蒸馏方法喂上
与我们完全相同的数据,表现仍不如我们的蒸馏——Table 1 输可能怪数据,
Table 3 把数据变量钉死,输的只能是蒸馏本身。

## Table 4 · Ablation(逐步累加)

克隆基线 → +自解锚(出处规则)→ +NLL 门控偏好(成功轴)→
+权重平均 → +边际采购(v(a) 含挤出成本)= 完整方法;
列 = AppWorld + 受控套件;每行相对上一行的 Δ。

## Table 5 / 分析节 · 预算转化与耦合

- **预算转化率**:in-task 提升 ÷ 消耗 token,各方法一条曲线
  (B*(τ)、浪费率、转化效率三联);
- **耦合案例研究**:三分型(解耦/建设性/破坏性)+ 双 κ 预报 vs 实测
  ΔL 散点 + 锚中和演示(cot ΔL +.38→−.40)——"off-task 走向是
  in-task 优化的可推导延伸"的图版;
- 零泄漏 + 门免费(残留 0.000,开门代价 ≈0)。

## Appendix

受控五域机制套件全量、纯度/证书、k/R/预算稳健性、负结果台账
(13 条,含机制)、harness 细节、BIRD/DS-1000 广度表。

## 与现有资产的映射(诚实盘点)

已在手:Table 1 的 AppWorld 列(base/克隆/ours 带 std;SAD/OPD/STaR
三行待实现)、Table 2 的受控套件列大半、Table 3 的 AppWorld 大半
(克隆/toksel/孤立偏好/ours 全有)、Table 4 的 AppWorld 链条全有、
Table 5 素材全有。
**缺口(按优先级)**:① BFCL v4 harness(轻,1-2 天,收益最大);
② SAD/黑盒 OPD/STaR 三个蒸馏 baseline 实现(主表行,2-3 天);
③ τ²-bench 接线(user-sim,2-3 天);④ ALFWorld(1 天);
⑤ 各台 teacher 示范池生成(计费,随台开通滚动)。
