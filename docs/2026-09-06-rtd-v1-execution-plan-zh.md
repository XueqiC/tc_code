# RTD v1 执行计划与我的调整 — 2026-09-06 21:15 EDT

依据:docs/RTD_END_TO_END_METHOD_AND_EXECUTION_V1.md(用户 9/6 发来,视为主规范)。本文只写落地决定、我按理解做的优化,以及与规范的对应关系。单种子,同机(hpg)。

## 1. 停什么、留什么

- **停**:条件对照(cc)线不再开任何新臂、不买类型 2;P2/原子线、AppWorld 同配方扩量早已关闭。
- **留到完成**:cc_s1 A/B/C 官方评测(41241770,运行中)与随后的留出对评测——规范 §1 明确要求完成并独立归档,不重复提交。结果只作"既有 A/B/C 的独立结果及其解释边界"(T7),不作为 RTD 的前置条件。
- 现在 rai/hpg 没有其它本项目作业。

## 2. 分段交付(全部 Codex gpt-6-astra xhigh,我审查)

| 段 | 内容(规范 §13) | 复用 | 产物 |
|---|---|---|---|
| C23 | T0 协议/数据访问表/冻结配置;T1 封存 broker + ledger;T2 传输目标、门控特征、identity/no-op | cc_pairs 的哈希与泄漏规则、margin.py 的全序列似然、checker_bridge、adapters/bfcl | src/bfas/rtd/{broker,ledger,transport,features}.py,configs/rtd/v1_bfcl.yaml,docs/rtd_v1_protocol.md |
| C24 | T3 固定 P 一步实际更新 + 完整任务随机 rollout 回报 + 门控 VJP;T4 插入价值 + Bayesian 采购 + 成本模型 + 对偶 | behavior/{deltas,whiten}(LoRA 参数空间、对角预条件)、bfcl_std_campaign 的生成/评测、vllm 服务 | functional_step.py, return_gradient.py, insertion.py, acquisition.py |
| C25 | T5 runner(audit/smoke/run/resume/replay-ledger/swap-component/evaluate/report);T6 正确性检查 | cc_three_arms 的曝光匹配/manifest 习惯、campaign 完整性检查 | tools/rtd_experiment.py + tests |
| 然后 | hpg smoke(toy + 1 窗口)→ BFCL sealed_replay R0/R1 三轮 → 固定 ledger 2×2 → scalar gate 变体 → ALFWorld 同程序 | | |

## 3. 我的调整(规范允许的求解器/资源选择,不改定义)

1. **参数空间 = LoRA 可训练参数(21.2M)**。式 (3)(5)(8) 全在 LoRA 空间计算;P 用训练侧来源 rollout 梯度的 RMS 对角(规范默认),复用 P2 的 diag-Fisher 工具链;这让 per-slot 梯度 g_i^T − g_i^S 与 VJP 便宜可算(每步 8 槽 = 16 次反传)。
2. **回报反馈的样本量按任务类型分配**:BFCL 单轮题一次 forward 就是一次完整 rollout,便宜——反馈窗口用 8 个反馈父任务 × 4 次采样(而不是 4×2),多轮/记忆题保持 4×2。规范说"不足则用实际数量并记录",这是同一条款下的向上使用;所有臂相同,写进配置版本。
3. **sealed_replay 的 BFCL 教师 bank**:包 = 一次真实历史请求。演示 40 题 × 3 次尝试 = 120 包(精确 token,含失败尝试);生成池 303 + 140 项 = 各一包(成本 estimated 标记);v3t 事件行归属于对应生成项的包(同包不重复计费)。gen id 复用 → 一律按 prompt/内容哈希识别(cc_pairs 已有)。65 题校准集与 support_split 留出永不进入。
4. **支持父任务 m 与两折**:父任务 = 演示的 40 个官方题 + 生成种子(5 个);按父哈希分两折轮换;反馈折的回报用官方 checker(单轮)/官方多轮评测。**m 很小(≈45)**,首轮必然标 exploratory,如实报。
5. **回报**:BFCL 用官方 AST/执行判定 0/1(规范 §2:官方主要成功指标);ALFWorld 用 success;AppWorld 用 TGC。不用错误类型奖励。
6. **数值**:pilot 只用 KL 目标 0.005/步定 η(与 P2 微更新的 KL 尺度一致);φ=0 起(a=.5);ε=.25;每轮 12 步、窗口 {1,4,7,10};预算点 10/25/50% B_bank。
7. **官方评测**:沿用 bfcl_std_campaign(已加生成完整性检查与重试),240G,同机;每个终点 ~1 h B200。
8. **强对照排期**(§10.3):首轮先 R0/R1;第二批补同池 text-only SFT(短教师 continuation,即"A Few Teacher Steps"式)与 base-centred pairwise 终点,以及同反馈 policy-gradient 对照——后者复用 return_gradient 的 rollout 路径,不另写。
9. **不做**:capability dictionary、类别/错误类型特征、混淆筛选、Δ U 过滤、任何按 benchmark 手工切 loss。

## 4. 资源估计(BFCL,B200)
每轮 12 步 × 16 次 LoRA 反传 + 4 窗口 × (参考更新 + 2 组 rollout ≈ 60 次 forward) ≈ 30 min;三轮 + 两臂 ≈ 3 h 训练;官方评测 6 个终点(R0/R1 × 3 预算点)≈ 6 h;2×2 再 4 个终点。新 teacher token 0(sealed_replay);online 模式留到额度恢复。

## 5. 风险(先写)
- 回报梯度信噪比:0/1 回报、每窗口几十次 rollout,门控可能学不到排序——规范 §11 已列为可接受结果,如实报。
- 单函数生成池占比大:采购泛化受限,结论限定父函数覆盖(规范 §12)。
- 一步更新的线性化误差:用同起点插入实验测(§7.3),不当阈值。
