# 机制验证首轮 · 日志核查(不启动新训练)— 2026-09-12 08:00Z

对应用户 07:45Z 备忘的三项核查。数据来自 tc-alignment-mech/results/mech_bfcl(首轮,seed 0)。

## 1. Loss 的含义
训练目标(training.py `mixture_loss`),按监督 token 求和后除以 token 数记录:

    loss = 0.5 · [−log p_θ(target)]  +  0.5 · [−Σ_v q_base(v) · log p_θ(v)]

第一项 = teacher one-hot 的 NLL;第二项 = 对冻结 base 全词表分布 q_base 的交叉熵,其下界是 base 的熵 H(q_base),**不是 0**。
所以 steps.json 里的 `0.493→0.001`(D)/ `0.351→0.052`(C)是混合值,只能说明:在被监督的位置上,
(a) student 已把 teacher 目标 token 学到 p≈1,且 (b) base 在这些位置本来就近乎确定(H(q_base)≈0)——
即大部分监督 token 是函数调用语法/复述参数这类 base 已确定的 token;真正"改变决策"的 token 只占少数。
不能把接近零解释为"蒸馏已收敛"。分项(teacher NLL / 对 base 交叉熵 / base 熵下界 / KL(q_base‖p) / 目标 argmax 命中率,训练前后各一份)
由新加的 `audit-loss` 子命令(Codex 进行中,mech-r2 分支)在 GPU2 上对 C、D 两个 adapter 算出后补入本文件。

## 2. 实际多样性(22 条练习各来自哪里)
生成器本身已有三方向结构:`condition`(关键条件改变→正确行为随之改变,并验证 condition_action_changes=True)、
`surface`(表面改变→决策不变)、`neighbour_correct`(邻近正常情境→保留 base 原本正确的行为)。

| seed(诊断) | 父任务 | 类别 | C 练习(cond/surf/neigh) | D 练习(cond/surf/neigh) |
|---|---|---|---|---|
| seed_0 "只查 core memory,没意识到需要 archival 检索" | memory_kv_141-notetaker-11 | memory_kv | 6(2/3/1) | 7(3/3/1) |
| seed_1 "把引擎启动的安全错误当成拒绝而不是前置条件" | multi_turn_base_79 | multi_turn | **0**(9 候选全部 "Executor snapshot unavailable") | **0**(同) |
| seed_2 "饮品数量未指定时把单位映射成 piece" | live_parallel_11-7-0 | live_parallel | 8(3/3/2) | 9(3/3/3) |
| seed_3 "回答前没搜 archival memory,漏掉 healthcare 笔记" | memory_kv_35-healthcare-5 | memory_kv | 8(3/3/2) | 6(3/3/0) |
| seed_4 "转推后就当发帖完成" | multi_turn_long_context_175 | multi_turn | **0**(9 候选全部 "Executor snapshot unavailable") | **0**(同) |

- 候选/通过:C 45→22(拒绝:snapshot 不可用 18、base 在 neighbour 候选上本来就错 3、执行器报错 1);D 45→22(snapshot 18、neighbour 5)。
  未达 48–96 目标的原因是生成器首轮不循环("Budget first; no repetition to fill counts"),不是预算耗尽。
- **C 与 D 的真正差别在"练什么决策"**:D 的 demo 集中在诊断出的决策(seed_0/3 → `archival_memory_retrieve` / `archival_memory_key_search`;
  seed_2 → `log_food` 的 portion_unit),C 的 demo 分散到同一工具箱的其他操作(`core_memory_remove/replace/list_keys/retrieve_all`、`archival_memory_remove`)。
- **独立教学信号偏少**:D seed_0 的 7 条里 5 条正确行为完全相同(`archival_memory_retrieve{key: vendor_follow_up_friday}`);
  surface 变体多是同一答案的改写。22 条 ≠ 22 个独立信号,更接近 3 个决策 × 少量变体。
- 两条 multi_turn 诊断在两臂都没有练习(状态型任务无法为练习提供可执行快照),所以下面 multi_turn 上的任何变化都不是靶向效果。

## 3. 诊断 ↔ 练习 ↔ 评测修复/损伤(全部按训练前的诊断分组,不是看结果后命名)
评测子集的 memory 题全部来自 customer(22)/ finance(17)场景,与 support 的 notetaker/healthcare **无场景重叠**;live_parallel_multiple 与 support 的 live_parallel_11-7-0 无题重叠。

| 训练前缺口(seed) | 有无练习 | D−base 修复/损伤 | C−base 修复/损伤 | D−C |
|---|---|---|---|---|
| archival 检索(seed_0、seed_3;memory) | 有,两臂各 13–14 条 | memory:customer **+6/−1** | memory:customer +3/−1 | +3 |
| 并行调用单位(seed_2;live_parallel) | 有,8–9 条 | live_parallel_multiple **+3/0**;parallel +2 | parallel +2 | +3 |
| 多轮前置条件 / 收尾(seed_1、seed_4) | **无** | multi_turn 合计 +1/−4 | multi_turn 合计 +2/−7 | +5(全是 C 的损伤少于 D) |
| 未诊断类别 | — | live_irrel +1、live_multiple +1、live_relevance +1;irrelevance −1、simple_js −1 | live_relevance +1、simple_java +1;simple_js −1 | ±1 |

结论(限定于本配置:22 条、26 pass、lr 1e-4):
- D 对 base 的净增益(+15/−7)有 9 题落在训练前就登记的两个缺口所对应的类别上,且是跨场景迁移(notetaker/healthcare → customer);这部分可算机制相关证据。
- D−C 的 +8 中约 +5 来自 multi_turn:那里两臂都没练习,是 C 损伤更大(−7 vs −4),不是 D 的靶向收益。**去掉 multi_turn 后 D−C ≈ +3**。
- C 的零净效应更准确的说法是"收益没超过干扰":memory +3、parallel +2,但 multi_turn −7。

## 对 R2 的直接含义
1. 三方向结构已在生成器里,R2 要做的是把每个缺口的候选数循环补到目标(并保留 22 条原池、记录来源),不是新写规则。
2. multi_turn 缺口需要生成器能给状态型练习提供可执行快照(目前一律拒绝),否则 R2 仍然只覆盖 memory + live_parallel 两类缺口。
3. 剂量:总监督 token 仍以 15.9k 为参照,64 条时 pass 数按 15.9k / 每 pass 监督 token 数自动折算(≈9 pass),中点+终点各留 checkpoint。

## 1(补). Loss 分项审计结果(audit-loss,GPU3,10:04–10:07Z;单位 nats/token,监督 token 加权)
| 臂 | 监督 token | base argmax 已对 | teacher NLL 前→后 | 对 base 交叉熵 前→后 | base 熵下界 | KL(base‖student) 后 | 混合 loss 前→后 | argmax 后 |
|---|---|---|---|---|---|---|---|---|
| C_s0 | 611 | 95.9% | 0.638→0.048 | 0.0041→0.0365 | 0.0041 | 0.033 | 0.321→0.042 | 97.5% |
| C_s1 | 611 | 95.9% | 0.638→0.037 | 0.0041→0.0388 | 0.0041 | 0.035 | 0.321→0.038 | 99.3% |
| D_s0 | 706 | 95.5% | 0.855→0.047 | 0.0092→0.0465 | 0.0092 | 0.037 | 0.432→0.047 | 97.5% |
| D_s1 | 706 | 95.5% | 0.855→0.042 | 0.0092→0.0542 | 0.0092 | 0.045 | 0.432→0.048 | 97.9% |

解读:
- **训练前 base 已在 ≈96% 的监督位置给出 teacher 的 token**;base 熵 ≈0.004–0.009 nats,几乎确定。真正携带教学信号的位置只有 ~4%(C 25 个、D 32 个 token)。
- 训练后 argmax 命中 97.5–99.3%,即整个干预只翻转了 **~10–20 个 token 级决策**;对 base 的 KL 仅 0.03–0.045 nats/token。
- "loss→0.001" 是这两项都很小的自然结果,不是收敛证据;teacher NLL 0.04–0.05 说明 teacher 目标已基本记住。
- D 的难点练习(base argmax <90%)正是 archival 检索类(seed_0、seed_3),训练后仍有 2 条只到 81–87%——靶向练习的目标 token 学得并不彻底。
- 这解释了三层结果:层 1 五次评测全 17/33(贪心解码下十几个 token 的变化不改变这些题的输出);层 3 ±8 题的种子波动与 ~15 个 token 决策同量级。
