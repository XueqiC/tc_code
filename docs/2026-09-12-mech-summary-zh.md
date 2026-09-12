# 新机制("靶向练习"蒸馏)验证汇总 — 2026-09-12(草稿,R2 种子 1 待填)

## 0. 一句话结论
在 BFCL 上、gemma-4-12B 学生、gpt-5.6-luna 教师、同预算同剂量的条件下,**围绕诊断缺口生成的靶向练习(D)相对通用局部练习(C)没有可重复的净收益**:
首轮 +8 题的优势在换训练种子后消失(R1 pooled D−C = 0),扩到 64 条练习后 D−C = +3(单种子),全部落在 ±8 题的训练随机性之内。
两类练习对 base 的平均提升约 +4~+6 题(+2 pp),也在噪声内。局部层(held-out 练习)七次评测全部 17±3/33,没有任何臂改变它。
唯一在所有 D 运行中稳定复现的差别是预登记的并行调用缺口(live_parallel_multiple 9/11 vs C 6–8 vs base 6,n=11),证据太薄。

## 1. 机制假设与设计(GPT-6 方案,docs/2026-09-12-gpt6-mechanism-validation-from-user.md)
核心问题:同样的教师预算与训练暴露下,围绕学生诊断出的缺口生成的短练习(D)是否比通用局部练习(C)更能提升未见完整任务?
三层评测:① held-out 局部练习;② 自然续接状态;③ 完整任务(固定子集),按 parent 配对统计。判定表见方案第 9 节。

## 2. 实际执行(tc-alignment-mech / mech2,分支 mechanism-validation / mech-r2)
- 划分:support 24 题(5 seed rollout → 19/24 成功 → 5 个失败 seed 诊断)、calibration 16 题、评测子集 256 题(分层),第 3 层排除 web_search(无 SERPAPI key,所有模型恒 0)→ **246 题**。
- 诊断(训练前登记):seed_0/3 archival 检索(memory_kv notetaker/healthcare)、seed_2 并行调用单位(live_parallel)、seed_1/4 多轮前置条件/收尾(multi_turn)。
- 练习:三方向(condition / surface / neighbour_correct),官方执行器验证。R1 每臂 22 条(multi_turn seed 0 条,快照不可用);R2 保留 22 条 + 两侧覆盖 + 去重 + multi_turn 快照重建 → C 63(触 24k 共同上限,21.0k token)/ D 64(9.7k token)。
- 训练(两臂完全一致):LoRA r16/α32,AdamW lr 1e-4,512 监督 token/步,soft target 0.5 teacher one-hot + 0.5 冻结 base;总监督 token ≈15.9k
  (R1:611/pass × 26 pass = 52 步;R2:1,585/pass × 10 pass = 40 步,中点+终点 checkpoint)。
  说明:方案原设计"单 pass、lr 1e-5"在 BFCL 上只有 2 步(611 监督 token),等于没训练;剂量按上限调整并告知用户。
- 评测:贪心;base 重复评测 246 题只翻 1 题(评测噪声可忽略);确认集 16 项 / 8 有效配对(R2 新增,来自独立 calibration 父任务)。
- 成本:教师总计 < $2(R1 诊断+练习 ~9.3k 输出 token;R2 练习 30.7k;确认集少量);GPU3 约 14 h。

## 3. 全部数字
### 3.1 完整任务(第 3 层,/246)
| 运行 | base | C | D | D−C |
|---|---|---|---|---|
| R1 seed 0(22 条) | 161 | 161 | 169 | +8 |
| R1 seed 1(22 条) | — | 169 | 161 | −8 |
| R1 pooled(两种子均值) | 161 | 165.0 | 165.0 | **0.0**(parent Δ −1.1 pp,CI [−4.2, +1.9]) |
| R2 seed 0 终点(64 条) | 160 | 165 | 168 | +3(翻转 10/7) |
| R2 seed 0 中点 | 160 | 161 | 170 | |
| R2 seed 1 终点 | — | {C_S1} | {D_S1} | {DC_S1} |
| R2 pooled | 160 | {C_POOL} | {D_POOL} | {DC_POOL} |

### 3.2 局部层与确认集
| 运行 | held-out (33) base/C/D | 续接 (9) base/C/D | 确认集 (16) base/C/D |
|---|---|---|---|
| R1 seed 0 | 17/17/17 | 5/4/6 | — |
| R1 seed 1 | 17/17/17 | 5/5/5 | — |
| R2 seed 0 终点 | 18/17/17 | 5/5/5 | 10/11/11 |
| R2 seed 0 中点 | 18/20/17 | 5/4/5 | 10/9/10 |
| R2 seed 1 | —/{C_S1_L1}/{D_S1_L1} | | —/{C_S1_CF}/{D_S1_CF} |

### 3.3 按预登记缺口的证据(完整任务答对数)
| 缺口(seed) | 类别 (n) | base | C(R1 s0/s1,R2) | D(R1 s0/s1,R2) |
|---|---|---|---|---|
| 并行调用(seed_2) | live_parallel_multiple (11) | 6 | 6 / 8 / 7 | **9 / 9 / 9** |
| archival 检索(seed_0、3) | memory_rec_sum (39,customer/finance 场景,与训练场景无重叠) | 8 / 7(R2 base) | 10 / 10 / 11 | 13 / 9 / 10 |
| 多轮前置/收尾(seed_1、4) | multi_turn ×4 (59) | 34 | 29 / 34 / 32 | 31 / 29 / 33 |
| 未诊断类别 | 其余 (137) | 113 | 116 / 117 / 115 | 116 / 114 / 116 |

### 3.4 loss 分项审计(R1 四个 adapter)
base 训练前已在 ≈96% 的监督位置输出 teacher 的 token(base 熵 ≈0.005 nats);训练后 argmax 命中 97.5–99.3%,KL(base‖student) 0.03–0.045 nats/token。
整个干预只翻转 ~10–20 个 token 级决策;"loss→0.001" 不是收敛证据。D 的难点(archival 检索)训练后仍有 2 条只到 81–87%。

## 4. 判断(对照方案第 9 节判定表)
- **"D 的优势在重复后消失"** 成立(R1 pooled 0);R2 单种子 +3 {R2_POOLED_SENTENCE}。→ 不稳定信号,不扩大叙事。
- **"全任务小幅改善、局部始终不变"**:两类练习对 base 平均 +4~+6 题,但 held-out 七次不动、确认集只 +1。不能宣称"局部掌握导致迁移";更像是十几个 token 决策的随机重排。
- **64 vs 22**:C 165 = 165(无收益),D 168 vs 165(噪声内)。多样性也没有超出噪声。
- 干预量级是根因:在 ≤16k 监督 token、base 已对 96% 的情况下,LoRA 只改十几个决策,与 246 题子集的 ±8 题种子波动同量级。要看见效应,要么把决策 token 的数量提高一个量级(几百条练习、或只监督决策 span),要么改评测(更大的子集、更弱的类别),两者都超出"廉价快速验证"的初衷。

## 5. 建议
1. 机制线收口:记录为"未确立",GPU3 回到主线(选择性替换 D3 vs D2 的 BFCL 实验,基线曲线继续)。
2. 若要再试一次,只有一个便宜且有信息量的变体:**只监督决策 span**(去掉 base 已确定的语法 token,把 15.9k 监督 token 全部花在 ~4% 的决策位置上),两臂同做;这直接针对根因。其余(A/B 臂、换 benchmark)不建议。
3. 可写进论文附录的收获:练习式蒸馏在 BFCL 上的干预量级分析(§3.4),以及"两种子是最低要求"的方法论。

附:docs/2026-09-12-mech-bfcl-round1-zh.md(首轮)、docs/2026-09-12-mech-bfcl-logcheck-zh.md(核查+loss 审计)、mech2/runs/mech_bfcl_r2/report.md(R2 报告)。
