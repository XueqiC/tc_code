# 执行计划 v3(2026-09-11 23:20Z;两张卡 GPU3/GPU4 至 9/26;对标 = SmartAD / SAD / Agent Distillation(Kang)/ GAD)

## 0. 已定
- pair:teacher gpt-5.6-luna(官方 API,Flex)+ student gemma-4-12B-it;seed 0。
- 故事:few-shot budgeted black-box agent distillation;根本问题 = 教师轨迹里哪些内容是当前学生需要学的
  (docs/2026-09-11-story-selective-replacement-zh.md + GPT-6 第二份建议
  docs/2026-09-11-gpt6-plan-suggestions-from-user.md)。
- 对标只做四个:SmartAD、SAD(文本适配版,标注)、Agent Distillation(Kang)、GAD。其余(SOPD/FutureBridge/
  SCoRe/STaR)写进 related work,不跑。

## 1. benchmark(便宜、简单、对标工作也用)
| benchmark | 谁在用 | 我们的状态 | 决定 |
|---|---|---|---|
| ALFWorld(valid_seen 140) | SAD | bank 建好,P1 在跑 | 主表 ✅ |
| HotpotQA-ReAct(维基搜索工具,EM/F1,500 dev 题) | SAD、SmartAD/Kang 的多跳 QA | 无 | **新增**,主表 ✅:episode 短(5–7 步)、教师便宜(~1–2k token/题)、verifier 确定(EM),三家共同的题型 |
| BFCL v4 | 我们 | bank 建好 | 保留为工具调用 benchmark(附表或主表第三列) |
| 数学工具题(SmartAD/Kang) | 对标 | 无 | 不做:非交互 agent,离我们设定远 |
| WebShop / τ² | — | 停 | 不做 |
HotpotQA 实现:ReAct 原版(Yao 2022)6-shot,Wikipedia 搜索/查找工具(在线 API,rai 可访问),
最多 7 步;support 集从 train 抽 200 题(few-shot 描述),评测 dev 500 题。

## 2. 方法臂(unified 为主线,v1.1 为附录)
- **D3** 行为级选择性替换(主方法);**D2** 状态级混合(关键消融:同剂量下"替换谁"是否重要);
  **D0** 同访问权限的 SFT+软 KL 对照;**D3-shuffle** 替换信号置换。
- v1.1 V0/V2:已在跑,作为附录/一致性证据;V1 视时间。
- **D4** 依赖一致的片段级替换(reason/act 分份额;改了动作就重跑环境,不拼接旧 observation)。
- **D5** "学生自己的推理路径上的动作蒸馏"(−log Σ_r π(r,a^T|s) 的采样近似)——作为独立 variant,
  能做多少做多少,不进主表承诺。
- 数据来源轴:示范来自教师轨迹 vs 学生访问状态(一次消融,ALFWorld 上做)。

## 3. 对标实现(同一教师池、同预算、同学生;Codex 实现,CPU 测试)
| 基线 | 做法 | 成本 |
|---|---|---|
| SmartAD | 每题采 K 条教师轨迹,留正确的,按学生 NLL 选最低;分段加权 CE(reason 1.0 / action 1.5 / final 2.0) | 训练 <1 h/臂 |
| SAD | [REASON]/[ACT] 分段损失(文本适配,无 logits) | <1 h/臂 |
| Agent Distillation(Kang) | first-thought 前缀改写 + 自洽动作生成解码 | <1 h 训练 + 评测略慢 |
| GAD | 判别器(小模型)区分教师/学生回复,学生按判别器奖励做 on-policy 更新;复用已购教师文本 | ~1 天/臂(实现 1–2 天) |
教师池:ALFWorld 已有;HotpotQA 新采(200 题 × ≤3,≈ $1);BFCL 已有。GAD 需要教师在学生 prompt 上的回复,
从同一池复用,不新增采购(超出部分计入预算)。

## 4. 两张卡的排程(每卡串行,两卡并行;新起作业一律用锁步 K=32 代码)
| 天 | GPU3 | GPU4 | CPU/Codex |
|---|---|---|---|
| 9/12 | ALFWorld V0 → V2(在跑) | D3 流式(跟 V0)+ BFCL V0(共存) | HotpotQA harness + adapter;SmartAD/SAD/Kang 训练脚本 |
| 9/13 | ALFWorld D2、D0 | ALFWorld 三个轨迹基线(各 <1 h)→ BFCL D3 | HotpotQA luna 池 + bank;GAD 实现 |
| 9/14 | HotpotQA V0 → D3 | ALFWorld GAD | D4 实现(片段替换) |
| 9/15 | HotpotQA D2、D0 | HotpotQA 三基线 + GAD | D5 原型 |
| 9/16 | ALFWorld D4(+D3-shuffle) | BFCL D2/D0 + 三基线 | 归因表/伤害表脚本 |
| 9/17 | HotpotQA D4 / 数据来源轴消融 | ALFWorld D5(若可行) | 论文 §5/§6 更新 |
| 9/18–19 | 补跑/复核(单种子) | 补跑 | 图表、附录实现开销 |
| 9/20–21 | 论文冻结前的空档(第二种子仅 D3/D2 若有时间) | | |
每个 P1 arm 现在 ≈ 8 h/step(旧代码);锁步 K=32 + 诊断锁步后目标 ≤ 3 h/step;若实测仍 >5 h/step,
启动"方法 A"(三角色共享 rollout)——它不改替换语义,只改反馈预算。

## 5. 汇报口径
- 每个 arm:主指标(SR / EM / Overall)、相对 base 的分类别伤害、教师 token、GPU 小时、D15 观测漂移。
- 关键结论句:同剂量、同数据下,D3 vs D2 的配对增益(含置换对照)。
