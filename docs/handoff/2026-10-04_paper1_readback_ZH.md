# 论文一交接文档的理解回读（2026-10-04）

原文存档：`docs/handoff/2026-10-04_paper1_context_ZH.md`。本文是我对它的理解、与现有资产的对照，以及开工前需要确认的点。尚未运行任何新实验。

## 1. 我理解的项目

- **核心问题**：给定 K 条可信、真实执行过的种子轨迹，**什么样的学习经验足以让受限学生（gemma-4-12B-it）形成能独立完成任务的策略**。
- **设置与以前不同的地方**：
  - K 是**可信种子轨迹数**（默认 10，之后 5/10/20，3 组 support 采样 seed），不是可用任务总数。
  - 允许自由生成新任务、新教师输出和新的真实环境交互，但所有经验来源与规模都要入账披露。
  - 不再以预算或最少教师调用为主贡献。
  - 教师是黑盒 **gpt-6-luna**，只用采样文本和动作，不用 logprobs。不用任何中国模型 checkpoint。只做 post-training。
- **先做发现，再定方法**：先区分学生失败的原因，至少分出两个待检验机制——
  1. 知识在经验里，但学生没有吸收或调用；
  2. 学生需要更明确的决策信息（更好的教学，或者学会合法地重新观察、查询）。

  同时排除格式、优化、任务覆盖、教师错误、接口缺陷这些混杂。
- **理论对象**：与当前证据相容的环境集合 V 上，学生可行成功策略的交集 I_ε。"每个环境单独可达，但交集为空"才是要隔离的信息障碍，并检验**容量限制会改变所需的环境信息**。存在性、可识别性、可学习性要分开讨论，不夸大成定理。
- **发现实验**（§8）：
  1. 先证明基础链路学得会，包括"学生能学会"的正对照；
  2. 四种经验来源的扩展控制：重采样、只扩任务覆盖、新任务+相关可信反馈、新任务+等量无关反馈；
  3. 知识吸收 × 信息表达的交叉干预，撤除辅助后再测；
  4. 容量改变知识需求的受控环境对；
  5. 带否证条件的核心假设表。
- **Baseline**：
  - 初轮：base、teacher、Seed-SFT、grounded Augmented-SFT、DAgger-hard。
  - 形成机制结论前要加入：SmartAD（10 候选、按 turn 平均 NLL、权重 1/1.5/2）和 DART-SD（CTB 后局部 mask）。
  - 另有黑盒 OPD、reward-only RL 对照，以及真值/可学习性上界。
- **环境**：先 ALFWorld，再从 AppWorld 和 ScienceWorld 中选一个主验证环境；BFCL V4 多轮作为补充。
- **数据**：种子 / 可扩展池 / dev / 封存 test 分开；逐事件 JSONL 记录来源；隐藏标签与可见信息分字段存放；测试集不参与任何生成、归因或 checkpoint 选择。
- **四个月路线**（§13）与首批交付物：environment_manifest、data_manifest、model_provenance、可重放 episode、loss mask 检查样本、base/teacher/SFT 逐实例结果、profiling_report。

## 2. 现有资产能复用什么（工程层面，不沿用旧结论）

| 资产 | 状态 | 需要做的 |
|---|---|---|
| gemma-4-12B-it 训练栈（LoRA r16、AdamW、assistant-only loss、turn 结束 token 106） | rai 与 hpg 上都跑通 | 锁定 revision；按文档对照 chat template、thinking 开关、loss mask |
| ALFWorld harness（vLLM 服务、14 并发、admissible commands 给所有方法） | 跑通，hpg B200 评测树稳定 | 审计 prompt 里没有 oracle 字段（expert_plan / PDDL facts / policy_commands）；当前只用 pddl 文件加载游戏，prompt 里只给 admissible commands，需逐字段确认 |
| reset + 动作前缀重放 | 支持集重放已在用 | 加状态一致性校验，用作分支起点 |
| 合并 adapter 后 vLLM rollout | 已用 | 文档要求做合并前后的固定输入一致性检查 |
| 教师采购工具（上限、dry run、按请求 ID 计费、Azure→OpenAI 回退） | 对 gpt-5.6-luna 跑通 | 换成 gpt-6-luna 需要先做账户 canary |
| HotpotQA K=32 管线、AppWorld 旧代码 | 有 | AppWorld 需要重新核对划分和快照语义 |

**旧项目里与新设置相关的工程教训**：这些不是论文一的结论，只用来避免踩坑。
- 训练量不足时，学生会全部掉到 base 以下，所以 §8.1 的正对照必须训练到稳定。
- 140 局 ALFWorld 只能分辨约 ±11 局；大训练量区间 seed 方差可达 14 局，需要多 seed 和配对统计。
- 同一权重跨 GPU 架构评测会差几局，所以每列只用一种机器。
- 测试 mock 曾经掩盖真实数据路径的 bug；集群上新作业类型要先在登录节点跑通数据加载。

## 3. 开工前需要你确认的点

1. **教师从 gpt-5.6-luna 换成 gpt-6-luna**：是否已开通？有没有本阶段的花费上限？我会先做 1 次调用的 canary。
2. **算力**：文档写的是 HiPerGator B200 + LONI。LONI 额度此前已透支，是否已经补充？rai 是否继续可用？
3. **第一周范围**：按 §14 的 1–9 步做接口审计和 smoke（ALFWorld、K=10、base/teacher/Seed-SFT），先不跑扩展实验。可以吗？
4. **代码位置**：在现有 tc_code 仓库里新开一个目录（例如 `paper1/`，按 §12 的结构），还是新开一个仓库？
