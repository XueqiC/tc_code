# 条件对照证据(Condition-Contrast Evidence)原型计划 — 2026-09-06 早

来源:用户转来的外部建议(inbox 1545966522843791380)+ 我的补充。单种子、同机训练评测(hpg dept)。

## 1. 对建议的采纳与我的修正

采纳全部核心判断:撤回"锚点 = 稀释";公共边界模式不投影掉;结束参数编码路线;AppWorld 暂停同配方扩量;论文不收缩为"受约束蒸馏 + 剂量控制 + 记账";联合 margin loss 是 worst-side margin(Group-DRO 类),创新只能由整个机制 + 效果建立;下一轮主产出 = 一个更好的学生。AppWorld 表述改为"A 的收益未被后续变体支持,同模型换评测机器 5 题翻转、换训练机器 7 题翻转,行为层不稳定"。

**两个硬约束(今晚查实)**:
1. **教师 API 全部耗尽**:ollama-1/2 429(5h/周窗口),azure-p1/p2 403(周预算 5M 用完)。所以"教师补足缺失一侧"现在买不了。原型第一阶段只能用**历史教师证据**——这恰好对应建议末尾"历史数据构造阶段只证明证据利用价值,不提前声称 teacher-token efficiency"。第二阶段(真实购买)等额度窗口恢复(我每小时汇报时顺带查)。
2. **现有可构造的对**:
   - 类型 1(同函数、用户约束变 → 参数绑定变):gen_pool_v3 303 个教师生成变体,5 个种子函数(simple_python_31 ×270、live_simple_26、live_multiple_923、live_parallel_multiple_10、simple_javascript_42),每个变体带教师给出的可执行答案(teacher_authored_gt,已在记账里);另有 oos 变体 13 组共享函数(oos_simple_java_6、oos_live_simple_26…)。**两侧教师文本都已有,0 新 token**。
   - 类型 2(多轮状态变化 → 下一步变):v3t 只有 13 条 multi_turn_miss_func + 11 条 memory,没有成对的另一侧;需要教师 → 等额度。现在只做候选状态清单。
   - 类型 3(调用/弃答):oos_irrelevance / live_irrelevance 变体,上限占 1/3。
   所以第一阶段 32 对主要来自类型 1 + 少量类型 3,函数多样性约 8–10 个;不够就如实报数量,不凑。

## 2. 我的补充(结合建议的设计)

- **混淆的可操作定义**:对候选对 (s₁,s₂)(同函数、条件不同、正确动作 a₁ ≠ a₂),学生"重复相同选择"= 学生在 s₂ 上的输出经 checker 判错,且其参数绑定等于/更接近 a₁(条件不敏感);连续版:κ(s₂) = log π(a₁|s₂) − log π(a₂|s₂) > 0。κ 既用于选对(只买/只训学生真混淆的对),也是**未见对上的主指标**之一(两侧同时正确的比例 + κ 是否翻负)。
- **对的来源分级**:P1 真实任务变体(gen/oos 变体,可重放);P2 官方题共享函数;不用文字编辑造不一致环境。
- **泄漏隔离**:排除 calibration/certification 官方题;P2 用过的 gen 探针(probes_disc/confirm)不进训练对;确认集按父任务(种子函数)隔离——整整留出 1 个种子函数 + 每个种子内留出对,两种都报。
- **三臂**(同一组对、同 token、同保留设置、同机 hpg dept):A 普通 pairwise(现调好的 base-centred);B 正确配对的联合 worst-side margin;C 在可比范围内打乱配对再联合。当前 BFCL 最好模型作性能锚点。记录实际 KL。
- **指标**:未见对两侧同时正确率(+ κ 翻转率);修复/损伤任务数(官方 v4 全量);官方整体分数分轴(NL/Live/MT/Memory/Irrel);整体调用率变化(排除"只是调用频率变了")。训练 margin 单独报,不作主指标。
- **判读**按建议的五条。

## 3. 执行

1. Codex C19:`tools/cc_pairs.py` — 候选对构造(gen_pool_v3 + gen_oos + oos 池;按种子函数配对;两侧教师文本从池中取,记 provenance 与历史 token;泄漏排除;父任务隔离划分)+ 学生混淆探测(base 学生贪心 + k=4 采样,checker 判定,κ 计算;hpg slurm + rai 回退)→ data/cc_pairs_v1/{candidates.jsonl, confused_pairs.json, split.json, report.md}。
2. Codex C20:BFCL 训练器加"pair unit"联合目标(L = max([γ−m₊]₊², [γ+m₋]₊²) 或按 v1.1 保留约束版)与打乱配对控制,配置开关,不改现有臂行为;三臂 runner + hpg slurm。
3. 混淆探测跑完 → 报可构造对数与混淆对数 → 若 ≥ 20 对且覆盖 ≥ 2 类型则训三臂;否则报数并等教师额度补类型 2。
4. AppWorld:离线检查失败轨迹里是否存在可验证的条件混淆(C19 之后,0 GPU)。

成本:混淆探测 ~10 min GPU;三臂训练各 ~10–20 min + 官方评测各 ~1 h(hpg dept);新教师 token 0(第一阶段)。
