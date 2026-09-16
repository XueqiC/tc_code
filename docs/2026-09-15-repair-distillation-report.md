# 2026-09-15 下午报告:学生思路条件下的动作蒸馏(教师修正包)——从决定到三臂结果

> 时间为 America/Chicago(CDT)。数字来自 LONI 树 `tc-hotpotqa-repair` 的冻结根与 rai 上的采购账本;路径在 §11。
> 本文只覆盖 15:53 你的决定之后的工作;当天上午到下午的设置探索、目标 2×2、控制器与探针检验见 `docs/2026-09-15-final-report.md`。

## 0. 一页结论

【待填:评测结果】

## 1. 你的决定与我的执行假设

15:53 附件(13KB)定了三件事:(1) 暂停"回报梯度→片段权重"路线,控制器冻结、旧候选 v 退役,不跑 A+B;(2) 冻结共同主干:react7 银行 + 纯 CE +
token 比例(U0:40.5 EM,+3.5 pp,两种子)+ 当前剂量;"参考 KL 有害"限定于已测配置;(3) 新主实验 = 学生思路条件下的动作蒸馏 + 教师修正包。

假设:学生学到的是"给定教师思路后怎样行动",部署时却只能依赖自己生成的思路 r^S。机制:在真实决策状态 s(任务 + 真实交互历史),冻结学生生成
当前回合,取其可见 Thought r^S 与拟议动作(不执行);教师返回 r^T(标准计划)/ c^T(接在学生思路后的修正,思路合理时为 NONE)/ a^T(动作);
a^T 执行核验,可解析 / 可执行 / 对任务有用分别记录,不筛。只替换当前回合的 Thought,过去的动作与 observation 一律不动。

三臂(同状态、同核验动作):A 标准 s → r^T, a^T;B 直接纠正 s, r^S → a^T;C 修正后行动 s, r^S → c^T, a^T;r^S 只作输入。正常训练长度、两配对种子、
完整自主评测(dev-500 作开发集)+ 局部机制指标。预注册判读:B 优于 A 且优于 C、完整任务同向 → 支持;B ≈ C → 只是减轻对教师 Thought 的依赖;
局部改善、任务不动 → 只支持局部;B 无稳定收益 → 结束该假设。采购:support-200 内,~32 父任务 / 48–64 状态,新增教师 output 上限 30k(失败与推理 token 计入),
先 8 状态干跑,不按救活筛、不用测试结果挑状态,达预算即停并报可用率。

我的执行假设(已在频道说明,可改):状态取自 U0 seed-0 学生的 greedy 执行记录;三臂从 U0 seed-0 的 LoRA 继续训练(修正包是本阶段全部材料),
两种子只改批次顺序;对照 = U0 原样合并后同批评测;训练 113 步(与 U0 同的"已证明能学习"的长度)。

## 2. 工具(三个 Codex 任务并行,16:00–16:55)

| 工具 | 作用 | 测试 |
|---|---|---|
| `tools/hotpotqa_student_states.py`(export-policy / collect / select / precheck) | 合并 U0 LoRA 并服务;U0 学生 greedy 跑 support-200 记每步;预注册规则选状态;三条件 NLL 前置检查 | 45 |
| `tools/hotpotqa_repair_pool.py` + `prompts/hotpotqa_repair_v1.txt` | 复用 teacher_pool 的预留式预算 / 账本 / 供应商机制;三行格式 Plan N / Repair / Action N;执行核验 | 66(+398 既有) |
| `tools/cr_repair_cell.py`(freeze / render / train / export / evaluate / local-check / analyze)+ `scripts/loni/repair_cell.slurm` | 三臂渲染与掩码、从 U0 起训、同批对照、留出状态局部检查、配对分析与四行判读 | 25(+90 回归) |

关键实现约束(均在文档 `docs/hotpotqa_{student_states,repair_pool,repair_cell}.md`):r^S 是零权重输入且不进分母;C 在 Repair=NONE 时与 B 逐 token 相同;
A 作批次参照,B/C 用同一包序列与步数,目标 token 量不匹配并在账本明示;λ=0 纯 CE、AdamW 新状态、lr 1e-5、clip 1;评测 binding 与冻结协议绑定。

## 3. 前置检查:训练条件确有差距(零教师调用)

128 条 react7 验证回合(seed 20260915 均匀抽样),U0 学生;对教师动作行 a^T 的 teacher-forced NLL(nats;每回合和 / 每 token):

| 条件 | 全部(128) | Search(84) | Lookup(4) | Finish(40) |
|---|---|---|---|---|
| T:教师 Thought r^T | 0.58 / 0.042 | 0.82 / 0.059 | 0.78 / 0.061 | 0.07 / 0.005 |
| S:学生自生成 Thought r^S | 3.03 / 0.238 | 2.71 / 0.214 | 6.66 / 0.502 | 3.36 / 0.261 |
| N:无 Thought | 22.9 / 1.80 | 22.8 / 1.77 | 25.7 / 2.09 | 22.8 / 1.84 |

配对差 S−T = +2.45 ± 0.50(题级 bootstrap [1.53, 3.40]);N−T = +22.3 ± 0.29。学生在 r^S 下自己生成的动作与 a^T 不同 32.8%(Search 36.9%、Finish 22.5%;
工具类型相同 86.7%);差距随步数增大(step 6–7:S−T 7–17 nats,动作不同 80%)。

读法(按你的口径):这只说明训练条件不同,不代表自主能力。给定教师 Thought,动作监督几乎"免费"(标准蒸馏 A 的动作项基本无可学);
学生自己的 Thought 下同一动作难 5 倍,三分之一状态学生会做别的动作;Thought 携带动作的几乎全部信息。B/C 确实在教一个不同的条件分布。
文件:`results/cr_diagnostics/thought_condition_precheck/summary.{md,json}`、`records.jsonl`、`student_turns.jsonl`。

## 4. 状态采集与预注册选择(零教师调用)

U0 seed-0 学生 greedy(7 步上限,实时 Wikipedia,4 并发 + 共享缓存)跑 support-200 全部 200 题:完成率 80%、EM 56%、平均 4.14 步。
预注册规则(写入每条记录):`parents = random.Random(20260915).shuffle(sorted(support ids)) 前 32;每父任务从其已执行步(含 Finish)均匀无放回抽 2 步;
前 8 个为干跑集`;不看结果。得 64 状态 / 32 父任务:步位置 1:16、2:18、3:11、4:8、5:6、6:4、7:1;学生拟议动作 Search 44 / Lookup 3 / Finish 17。
留出集(local-check 用):seed 20260916 二次选择减去已采购父任务 → 56 状态 / 28 父任务,无重叠。

## 5. 采购(gpt-5.6-luna 官方 API,flex)

| 批次 | 状态 | ok | 拒绝 | 计费 output(其中推理) | prompt | 花费 |
|---|---|---|---|---|---|---|
| 干跑 repair_dry8 | 8 | 8 | 0 | 677(243) | 11,465 | $0.002 |
| 全量 repair_v1(状态 9–64) | 56 | 54 | 2(OpenAI 429,零计费,确定性拒绝不重试) | 4,530(1,844) | 61,353 | $0.009 |
| 合计 | 64 | **62** | 2 | **5,207**(上限 30,000) | 72,818 | $0.011 |

62 包:可解析 62/62、可执行 62/62、对任务有用 37/62(60%;规则:Search/Lookup 观察含 supporting title 或 gold;Finish 则 EM=1);
Repair = NONE 51/62(82%);教师动作与学生拟议动作相同 53/62(85%);教师动作类型 Search 41 / Lookup 1 / Finish 20(含干跑)。
修正样例:两次把重复搜 "Cleveland Union Terminal" 改为搜 "Terminal Tower"(观察含 gold 街名);"Mark Lawrence (author)" → "(politician)"。
读法:随机抽的状态里学生多半走得对,所以 B/C 相对 A 的额外信息只有 11 条修正 + 9 处不同动作——这是这批材料的客观上限,按要求不筛。
未买到的 2 个状态(5a7475ae…:s3、5abf0ba6…:s3)不补(冻结已起;62 在预注册 48–64 内)。

## 6. 三臂冻结、渲染与曝光账本

冻结根 `runs/conditional_response/repair_r7_s200_u0`(LONI):62 包全部纳入(0 排除;核验标记不筛);初始 adapter = U0_s0 LoRA(sha 8bdd8e9a…),
六臂一致;AdamW 新状态、λ=0、lr 1e-5、113 步;两种子只改批次顺序;对照 = U0 原样合并同批评测。渲染核对(干跑包,逐 token):
A 目标 = `Thought N: {plan}\nAction N: {a^T}` + `<turn|>` + 格式尾;B 掩码输入 `Thought N: {r^S}\n`(零权重,不进分母),目标 `Action N: {a^T}` + 边界;
C 掩码 `Thought N: {r^S}`,目标 ` {c^T}\nAction N: {a^T}` + 边界;NONE 时 C ≡ B(51/62)。

曝光账本(seed 0,113 步全程;A 为批次参照,B/C 同包同序):

| 臂 | 目标 token | reasoning | action | 边界+格式 | 掩码输入 | 每批 N+ |
|---|---|---|---|---|---|---|
| A | 60,242 | 37,644(plan) | 19,484 | 3,114 | 0 | ≈ 533 |
| B | 21,041 | 0 | 17,927 | 3,114 | 53,612 | ≈ 186 |
| C | 27,616 | 6,300(repair) | 18,202 | 3,114 | 53,337 | ≈ 244 |

训练曲线(批均每 token CE,前 10 步 → 后 10 步):A 0.70 → 0.03;B 0.07 → 0.001;C 0.50 → 0.005;anchor_loss ≡ teacher_ce(λ=0 生效)。
113 步 ≈ 5.6 遍,三臂都把 62 包学到近零;B 起点只有 0.07 nats/token(几乎无可学),C 多学的是 6.3k 修正 token,A 学的是 37.6k 教师 plan。

## 7. 结果:dev-500 / conf-32(同批 U0 对照)

【待填:评测结果】

## 8. 局部机制检查(留出 56 状态,新生成 r^S)

【待填:评测结果】

## 9. 判读与分析

【待填:评测结果】

## 10. 过程中的失误与教训(如实)

- **冻结代码清单**:新树的代码取自 git HEAD,而 U0 的 v6 冻结根按哈希绑定 scan6 部署时的 75 个实现文件 + 4 个源文件(其中 3 个是未入库的生成物);
  10 个模块此后被改过 → 两作业 17:05 被门禁拒绝。修法:从 scan6 复制冻结清单,新工具改用冻结模块已有的 API(本地 is_objective_protocol;
  positive_terms 不传 reference;分析只比较冻结评测器写出的 binding 键;λ=0 时冻结模块仍记录 reference_kl 但不进损失,已在 journal 核实)。
- **并行拆分引入的竞争**:两作业从同一 U0_s0 run dir 导出,收据互相覆盖 → 精检作业身份校验失败;改回单作业串行。
- **Wikipedia 429**:16 并发 + 工具自建空缓存;改 4 并发、复用 17k 条共享缓存、采集阶段 resume 重试。
- **训练首提失败**:训练前的评测协议校验读归档 Table 1 的 metrics.json(results/ 不在 git archive);补软链后重提。
- **sbatch --parsable 被 LONI lua 插件的警告污染**:依赖链断;submit 阶段改取最后一行。
- rai 上 CPU 测试需 `CUDA_VISIBLE_DEVICES=''`(新 torch 的 AdamW.step 查询加速器,触发"禁 GPU"守卫);失败在无改动的树上同样出现,属环境。
- 采购:2 个 429 按零计费确定性拒绝处理、未重试;若要 64/64 需单独再买(≈ 200 token)。

## 11. 文件索引

| 内容 | 位置 |
|---|---|
| 代码(三个工具 + SLURM + 文档) | `tc_code` 分支 mech-hotpotqa,hq worktree(提交 edd25606 → 104846dc → 892b6030 及修补);LONI `tc-hotpotqa-repair`(代码 = 冻结 v6 清单 + 新工具) |
| 前置检查 | `results/cr_diagnostics/thought_condition_precheck/`(rai 已取回;LONI `results/repair_states/thought_condition_precheck/`) |
| 状态采集 / 选择 / 留出 | `results/repair_states/{collect.jsonl,states.jsonl,collect_summary.json,selection_summary.json}`;LONI `results/repair_heldout/heldout_states.jsonl` |
| 采购账本 | `tc-alignment-buy/envs/hotpotqa/teacher_pool_scan/{repair_dry8,repair_v1}/{packages.jsonl,journal.jsonl,summary.json}`;合并 `results/repair_states/packages_all.jsonl` |
| 冻结根与结果 | LONI `runs/conditional_response/repair_r7_s200_u0/`(frozen_protocol、exposure_ledger、filter_receipt、A/B/C_s{0,1}、control、analysis);rai `results/repair_r7_s200_u0/` |
| 作业 | states_all 1027773;freeze 1027814;train 1027823[0-5];evaluate+local 1027827[0-6];analyze 1027828 |
