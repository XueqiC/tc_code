# BFCL：RTD v1 §10.3 强基线执行计划

审计日期：2026-09-07；本地资产读取约截至 15:44 UTC。依据 `docs/RTD_END_TO_END_METHOD_AND_EXECUTION_V1.md` §10.1–10.3，尤其第 453–465 行。本文是执行计划，**本次仅创建本文；没有创建下述配置、修改实现、运行测试、提交作业、调用教师或运行评测**。正在运行的文件仍可能增长；下面的数量不是最终实验结果。`PROJECT_STATE.md` 中部分日期晚于本次系统时间，状态判断以读取到的文件为准，不据文字记录推断远端队列状态。

## 1. 必须交付什么

四类强基线全部保留：B1 随机学生状态／短教师 continuation + tuned text-only SFT；B2 同池 tuned base-centred pairwise；B3 teacher SFT 或固定混合 + **与 RTD 相同环境反馈的直接 policy gradient**；B4 干净支持反馈驱动的普通元重加权。B3 先同时保留 SFT、固定 0.5 混合两种初始化／监督目标，通过支持侧调优选择，报告两个结果。R0、learned scalar gate、更多 SFT 步数、离线 DPO 均不能代替 B3。

每类至少给出两个资源视角：V-S 固定教师预算与完整动作槽位；V-T 固定教师预算与学生训练 token／计算目标。每个表单列实际教师花费、监督曝光、反馈 rollout、总计算、调优开销。只赢 R0 不能支持“优于强学生基线”；单种子结果不提供种子稳健性证据。

第一优先完成 A1 最终 ledger 上五个配方（B1、B2、B3-SFT、B3-mix、B4）及固定证据 RTD 参照的两个视角；随后完整补 A0。另跑全 bank 候选空间中的随机短续写 B1，分开报告采购与训练器效应。§10.1 的 A0/A1 × D0/D1 四格均从 theta0 重训，不拿自适应 R0/R1 充当对角线。

## 2. 已有资产与真实缺口

### 2.1 本地快照

| 资产 | 读取到的事实 | 对计划的影响 |
| --- | --- | --- |
| `data/rtd/v1_bfcl_c25/public/requests.json` | 698 条目录记录；420 个结构可用包，覆盖 28 个父任务，**全部 L=1**；84 demo attempts、336 generator items | 可做单动作历史回放；不能配置出未记录的 L=4、L=8 或新学生状态续写 |
| 同 bank 不可用部分 | 135 历史前置／支持集外记录、36 未重建精确 stateful bridge 的请求、107 protected 内容 | 保留成本审计身份，不进入训练／调优，不从不可用包补数据 |
| `configs/rtd/v1_bfcl_support.json` | m=40，fold 0/1 为 22/18；轮次 inner fold 为 0、1、0 | 两折轮换属于 meta-training，不能称 untouched validation；memory/web-search 的反馈接口尚不可用 |
| R0/R1 manifest | usable public cap 总和 8,257,536；累计上限 825,753 / 2,064,384 / 4,128,768；可用 bank 历史输出成本合计 55,370 | 分母是 public cap 总和，不是 55,370；不能沿用旧文档 5,537 / 13,842 / 27,685 的预算值 |
| `results/rtd_v1/rai_R0/trajectory.json` | 仅 round 1 的 12 步与 checkpoint；该 checkpoint owned=2、实际支出 188 | `teacher.jsonl` 已有后续 round 2 reveal；当前 ledger 不能标 final A0 |
| `results/rtd_v1/rai_R1/trajectory.json` | 同样仅 round 1 的 12 步与 checkpoint；owned=2、实际支出 177 | round 2 compute 已存在，最终 A1 尚未冻结 |
| `results/rtd_v1/rai_R1s` | scalar gate 自适应运行，有第一轮 checkpoint | 是 §10.2 scalar transport 对照，不是 B4 |
| 第一轮 `evaluation-1.json` | R0 46.81%、R1 45.50%、R1s 45.71%，都不是最终三轮结果 | 不用于本次基线超参数选择；不同 GPU 的这些数值不构成匹配比较 |

55,370 为 21,203 exact demo output tokens + 34,167 estimated generator output tokens。历史总体开销仍另列 demo 1,233,607 exact、generator 142,727 estimated；这不是本次新增教师支出，也不能写成“教师成本为零”。input、reasoning、货币成本的未知字段保留 null。generator item 的成本是历史文件估算按教师内容分摊，不是恢复出的独立 provider 请求账单。本文没有读取未购买教师文本来设计选择器。

### 2.2 逐路径复用表

| 路径 | 可复用内容 | 必须补齐／不能直接套用的部分 |
| --- | --- | --- |
| `src/bfas/arms.py` | `build_pool('sft', ...)` 的 teacher demo 行格式、`trainer_environment('sft')`；`ddpo` 对应已有偏好训练；`pair_unit_margins` 的完整序列 base-centred margin | `build_pool` 不是 ledger adapter。`_first_failed_targets` 按 task 取失败文本，不能保证同完整 state，B2 不得直接用它生成 rejected。`star/ours` 的成功过滤不用于本组基线；`bbopd` 名称也不代表已有在线 PG |
| `src/appworld_train.py` | `train_ddpo`、`cache_ddpo_reference_log_probs` 的冻结 base reference；AdamW、LoRA r=16/alpha=32/dropout=0；SFT/偏好阶段与保存工具 | 旧 `encode` 截 prompt、截 response 后补 EOS；旧 SFT 用 model 的 token-mean CE。严格 RTD 比较应复用优化器配方而接 RTD 完整动作 scorer，不能无声沿用截断／长度归一化。旧 DDPO 还会丢空 `_rejected`，不能丢合法空文本/EOS 来源 |
| `src/bfas/pair_unit.py` | base-centred reference、AdamW、完整 side 的曝光/实际 token/reference token 计数、`step_schedule` 完整 pass 审计思路 | `pair_unit` 本身是两个**条件状态**的 squared-hinge worst/sum 联合损失，不是 B2 所需的普通 logistic pairwise；`load_pairs` 要两侧、同 seed function/type/train split，不为 bank 人造条件对 |
| `tools/cc_three_arms.sh` | A 是 flattened logistic DDPO；同 GPU、环境清理、曝光哈希、先训练再验证再官方评测的组织方式 | B/C 是正确／打乱条件配对对照。默认旧 confused pairs、anchors、4096 prompt cap、完整 pass 和 >=20 pairs/>=2 types gate 都不适合 RTD bank；不要直接跑默认脚本，也不为了通过 gate 编造 pair/type。旧 anchors 不能免费混入 |
| `src/bfas/rtd/transport.py` | `FullState`、`SourceSample`、`TransportSlot`、完整动作概率与 teacher/source state 一致性；B3-mix 直接用 `positive_mixture_loss` | B1/B2/B4 用同样状态与 scorer，但定义各自损失；B4 不调用 source-teacher transport 目标 |
| `src/bfas/rtd/functional_step.py` | `FrozenStep`、LoRA 参数快照、`gradients`、`commit_step`、训练侧 RMS P；固定 P 一步适用于同优化器归因；`kl_pilot` 可作数值校准 | 不能将一步 VJP 用到真实 AdamW 多步更新。强配方与归因配方分列；RTD pilot 固定 alpha=.5、KL=.005，不按 BFCL 最终分改 loss |
| `src/bfas/rtd/return_gradient.py` | `BFCLSupport.feedback` 所调用的 `bfcl_task_rollout`、`reinforce_gradient`、同任务 LOO、完整 rollout/action trace、`gate_vjp` 的一般一步链式法则 | 已有的是 gJ 估计和 gate 更新，**没有把 gJ 直接提交到学生的 B3 训练模式**；B4 需要自由标量参数及 teacher-only weighted loss。`GateController` 的 past-only RMS 可借用，不把 gate 差梯度照搬为重加权梯度 |
| `src/bfas/rtd/acquisition.py` | `AcquisitionPolicy.choose(random_control=True)`、public cost cap、empty prior、成本/预算约束，可给 B1 全 bank 随机选择复用 | B2/B3/B4 同池运行冻结 evidence、不学采购；`ValuePosterior.reweight_observations` 是采购诊断记录，不是 B4 的训练权重 |
| `src/bfas/rtd/experiment.py` | `pool/draw_slots` 的 parent→state→source→teacher 抽样、fold 隔离、round source refresh、`feedback`、checkpoint/compute journal、fixed-evidence 初始化 | 当前 `slots`、source 数、反馈日程等有硬编码；只新增 YAML 键不会生效。`fixed_evidence` 仍每决策窗口取得 reference feedback，不能把它当免费纯 SFT。新增基线执行器须显式跳过不需要的环节 |
| `tools/rtd_experiment.py` → `src/bfas/rtd/cli.py` | `replay-ledger`、R0/R1 fixed-evidence、`gate_override: teacher_only/fixed_half`、evaluate/report | `--arm` 仅 R0/R1；`load_config` 冻结步数、曝光、loss 等；B1 tuned AdamW、B2、B3、B4、V-T 均不是现成 CLI 模式。`teacher_only` 在无教师 state 仍保留 source identity，不能直接称纯 text-only SFT |
| `tools/bfcl_std_campaign.sh`；`src/bfas/rtd/{evaluation,evaluation_lock,identity,hardware}.py` | 官方 all-category 生成/checker、PEFT 导出、identity、tag/port lease、完整性检验与已完成 campaign 复用 | 复用评测实现；训练后导出必须有真实 baseline manifest/checkpoint，不能把旧 adapter 随意复制并伪装成 RTD checkpoint |

## 3. “同池”与“全 bank 等预算”的严格定义

### 3.1 A0/A1：最终已购完整集合

令 A0 来自**最终完成**的 R0，A1 来自最终完成的 R1。每个来源读取 `manifest.json`、`teacher.jsonl`、`trajectory.json`、`round-3/checkpoint.json`、`audit.json`；`compute.jsonl` 用于曝光/反馈日程。定义：

`A_j = teacher.jsonl 中合法 reveal 的 query_id 集合 = final checkpoint.owned`。

冻结前要求三轮 36 个 committed steps、12 个决策窗口、最终完成审计、无 reservation、无 pending/unmerged evidence，ledger 哈希链完整，checkpoint/trajectory/ledger 的 owned、spend 一致。运行中 `trajectory.json` 可能落后 reveal；不能取两者并集“修复”。保存原文件 SHA256、ledger event digest、最后 sequence、source manifest/base/tokenizer/data/hardware identity、完整 query ID 与依赖链。

同池包含：同请求包及依赖、同合法教师文本及经验重数、同完整 state、同支持父任务排除表、同 theta0；全 state hash 去重，包内 alias 不增加证据质量，独立付费请求的相同回答保留原经验重数。同一包拆出的多行不重复计教师费。teacher/source 必须同时匹配 state hash 和 parent hash。一次蒸馏状态的两个 source samples 共享 teacher 请求成本。

同池强配方按§10.1允许各训练器从自己的轮初学生重采来源，使用同样的两样本/状态规则，但不宣称来源文本逐字相同。若另报“同两侧文本/顺序”的严格归因，必须先冻结一份合法固定证据参照生成的外部 source snapshot/target schedule，所有归因格（包括RTD格）从theta0按它重训；这是额外的固定来源条件对照。不能一面各自重采，一面只凭相同seed声称source/action曝光哈希一致。共享source准备成本记录一次实际支出，并在各方法独立运行成本中列出其应承担份额。

固定集合实验从 theta0 重新训练，两折仍按 0/1/0；当前反馈折的教师、统计、监督梯度均不可进入当轮 inner。**最终集合在训练开始就可用**是 §10.1 的条件实验，不是在线采购：它的 round-1/2 checkpoint 也是“全最终教师成本下的训练进度”，不可重新标为自适应 10%/25% 教师预算点。若需真正三个教师预算点，分别冻结原 round-1/2/3 checkpoint 的 owned 集合，各从 theta0 训练；不能把最终包倒灌到较早预算。

每个 A_j 内所有基线可以精确匹配教师账单 B_j。A0 与 A1 的 B_j、包数、长度、覆盖可能不同；同一个上限不等于同一实际花费。两表分别报告，不能以裁掉某个昂贵请求的若干 token 强行相等。预算曲线同时显示实际 spend 与 authorized cap。

现有 `replay-ledger --out` 会检查 reservation，但**不会要求三轮已完成**，因此需先执行上面 completion gate。它通过 `Ledger.resume` 读取；后者遇到 torn 尾行会恢复/改写文件，故它不是可对活动 ledger 随时运行的纯只读命令。本次未执行它。后续只对冻结、验证无尾行损坏的来源执行，或对独立冻结副本执行。

### 3.2 全 bank 候选空间：另一个采购问题

“full-bank”指从 420 个合法 public 请求候选中选包，不是把 420 个 payload 全部训练后记作 A1 的成本。排除规则、support/fold、依赖、recorded L=1 不变。

优先给出两种清楚标记的比较：

1. **sealed 随机预算基线**：沿用相同 10/25/50% public-cap 上限、12 窗口、每窗最多一个新包、empty prior=.5、public reserve→reveal；关闭 learned value，记录 `AcquisitionPolicy` 返回的实际概率（成本约束可能使最终概率不同于裸 uniform prior）。短续写内容只在购买后解封；后续用 B1 tuned SFT。它与 RTD 是同授权预算／窗口，实际 spend 不保证相同。不得解封后退款或反复重抽直到花费好看。
2. **固定历史预算 B_j 的 retrospective allocation 补充表**：若确实需要从全 bank 抽出不超过最终 ledger 实际输出 token 的新池，先独立生成只含 query/dependency/历史 usage/置信标记的成本目录，明确这是公开历史成本的离线分配条件；所有方法均享有同一目录。按 seed=0 的、与文本/回报无关的随机顺序，在 dependency-closed 可负担包中抽取，分布写为 `Pr(q|owned,b)=1/|F(owned,b)|`，F 使用已公开历史成本；停于预算不能再购买。整包计费、报告剩余预算，不要求刚好花满。该表不运行或冒称原 sealed 选择协议；若要比较采购优劣，RTD 也须在同一历史成本公开条件下重跑。它不能替换第一张表。

必须保留这一限制：若 B_j 只有几百 tokens，而最小 public cap 是 8,192，严格 sealed broker 在这个实际支出上限下无法购买任何新包。不能偷偷改 cap 为已看见的实际长度。55,370-token 全可用 bank SFT 可作为另列上界锚点，不能标注为几百-token 同池强基线。

## 4. 四类基线的具体训练定义

### B1：随机学生状态 + 短教师 continuation + text-only SFT

真正在线版本应在当轮合法 inner 父任务中均匀抽任务，学生随机 rollout 到实际已观察状态，再均匀抽可请求 state，购买 L=1 教师完整动作。采样不能按成功、错误类型或 teacher 答案过滤。用 `FullState` 保存 task/history/prompt，续写动作不含 observation；前缀的教师依赖照价记账。

**现 bank 只能提供限制版**：对合法历史 state 请求目录随机选 L=1 包；在精确相同 state 上由冻结学生随机采两条 source action。`sample_state` 是给已知 state 采动作，并不产生任意新 occupancy 状态；generator 合成 state 也不能改称当前学生自然访问的失败状态。本文将此命名 `B1-replay-L1`。严格在线 B1 仍待实际 stateful 请求路径/教师授权/预算，不能用 nearest-neighbor 答案或伪造 bridge 补齐。缓存 B1 先做，不因 online 尚不可用而跳过全部基线。

监督目标 `L_SFT = E_i[-log pi_theta(y_i^T | s_i)]`，完整序列求和，教师多个回答按包重数均匀抽；无 gate、无 student identity loss、无成功过滤。强配方用教师可用状态条件分布（§5），AdamW；不免费加入旧 teacher demos、anchors、自训练成功池。另保留严格相同 slot list 的同优化器 CE 归因版：无教师 slot 的 CE 为零、分母仍按约定槽位计，并分别报告实际有教师槽位；不要把这种低教师命中率版当唯一强 SFT。

### B2：同池 tuned base-centred pairwise

每个已购 state 的 `y+` 是原教师完整文本，`y-` 是该 state 的冻结学生随机动作，两个 source samples 分别形成两槽；不是把不同 state 或同 task 的末次失败动作配起来。默认教师优先偏好不再调用 checker 给 source 打好坏标签；教师可能不优于来源是本基线的假设限制，不能用额外反馈只筛坏 source。相同 y+/y- 导致零区分信号仍保留和计曝光。

冻结 theta0 的完整序列 reference，目标为：

`L_pair(i) = -log sigmoid(beta * [(log pi_theta(y+|s)-log pi_0(y+|s)) - (log pi_theta(y-|s)-log pi_0(y-|s))])`。

beta 默认 .1；`AW_DDPO_REF_FREE=0`、`AW_DDPO_SPAN_ONLY=0`、`AW_DDPO_WEIGHT=none`、`AW_ANCHOR_ADAPTIVE=0`。实现复用 `train_ddpo` 的 logistic margin／reference 逻辑及 `pair_unit` 的计数方式，接 RTD scorer 与新曝光 scheduler。固定 reference 不随 round source refresh 更新。reference forward token/GPU 时间单列。强版 AdamW；同 P 一步、36 次提交另列归因。不默认跑 CC B/C、不造“两条件”结构，也不混入旧 anchors。

### B3：相同反馈的直接 PG，必须实施

这项是新训练模式，但 rollout、checker、score 与 gJ 路径全部复用：

`RTDExperiment.feedback` → `BFCLSupport.feedback` → `bfcl_task_rollout` → `reinforce_gradient`。

准备阶段从选定参照的**最终完整** compute/trajectory 抽取每窗 `(round, step, role, parent_hash, rollout_count, reused)`。主 A1 表绑定原最终 R1 的反馈日程，A0 表绑定 R0；fixed-evidence D1 也用相同外部日程作该表参照。这个日程必须真正注入执行器；仅设相同 seed 不够，因为 source/采购/模型采样共用 RNG，分支不同会改变后续 parent 选择。

原 `choose_feedback_tasks` 的上限是单轮任务最多 8×4、多轮最多 4×2，并受 fold 可用任务限制。当前第一轮每组实际 **34 rollout（8×4 + 1×2）**，第二轮见到每组 40；不是机械写成“每组 32”或“每窗固定 80”。若 reference 与 actual 参数哈希相同，RTD 复用反馈，该窗只算一组；否则两组。新方法对照按这份表生成同父任务、同组数、同每任务次数的**新 on-policy rollout**。不得把原 R1 的回答／reward 缓存当成 baseline 的 on-policy 数据，也不能分别计数 reference 和 reused actual 两次。

采用以下主配方，保持 36 个学生提交点：非反馈步仅监督；反馈步在当前 theta 计算监督梯度和所有约定组的 gJ，将它们合成一次更新。两组在同一个 baseline 当前 theta 采样，各组独立算 LOO，再按 rollout 数平均；“reference/actual”在 baseline 日志中只是资源配额来源，另存真实 sampling parameter hash，不声称是 RTD 的两个参数点。

`g_sup = grad L_SFT(theta)` 或 `grad L_mix_a=.5(theta)`；

`g_total = g_sup - lambda_pg * stop(gJ(theta))`；

`theta_next = OptimizerStep(theta, g_total)`。

方向是回报**上升**。强版 AdamW；归因版 `theta_next = theta - eta * P * g_total`，P/eta 训练侧冻结。不先执行 SFT 再把旧 theta 的 gJ 当新 theta 的 on-policy 梯度。若另加“先 SFT 后 PG”顺序版本，须重新 rollout 并增加真实提交步数/成本，不能写成相同步数。

反馈完全继承 RTD：从完整任务初态启动，官方终局 0/1 回报；temperature=1、top_p=1、top_k=0；同 BFCL caps（单轮 512、多轮 1024）、完整上下文上限 32768；只计真实 action token，EOS 仅在实际采到时计，cap 不强补 EOS。畸形 action 是失败样本，reward=0，保留 token、LOO 和分母；基础设施失败仍报错，不能变成 reward=0。score/generation 一致性及原 token IDs 的检查复用，不能改用 teacher likelihood 外层信号。

一个任务所有回报相同则 LOO advantage 为零，记录 `identifiable=false`，仍完成预定配额和监督训练；不追加 rollout 直到有正样本。B3 的 PG 参数必须出现实际非零 gradient/commit 证据；若所有反馈确实全零，应如实报告，不能因此删除该基线。

主表要求同反馈配额；另可给 B3 相同**总学生计算上限**，把 RTD 的 reference/VJP/acquisition 节约时间用于额外 on-policy PG，单列新增 rollout/steps。这是 equal-compute RL 扩展，不替换同反馈主表。action 长度受策略影响，“同 rollout 数”本身不保证等 token/GPU 时间。

### B4：features off 的普通元重加权

为消除“source”歧义，本计划把 source 定义为**训练证据来源包 query_id**。每个已购包有一个自由 scalar `u_q`，包内状态共享，同文本不同付费包仍有各自来源身份。初始化 u=0，`v_q=softplus(u_q)`；当前 inner 可用源中归一化得到 `w_q=v_q / sum_i p_i v_{q(i)}`，使 `sum_i p_i w_{q(i)}=1`。这里 p 是冻结的教师槽位基础分布，分母在整个合法 inner 池计算，不按 minibatch 随意重标定。query_id 仅作为参数表索引，不做语义特征。可另外报告 per-state scalar，但不把两个参数化混成一个结果。

目标只有 `L_RW(theta,u)=sum_i p_i w_{q(i)} [-log pi_theta(y_i^T|s_i)]`。**features off** 指不使用隐藏投影、source logprob、source length、类别、函数名、正确性或 teacher embedding 来预测权重；不使用 `(1-a)*ell_source+a*ell_teacher`、不对完整 source 分布保留／搬运概率质量。source 动作可以作为相同 P 的训练侧数值估计或曝光审计，但不进入 B4 监督目标。与 gate 的关键差别：RTD 权重导数含 `gT-gS`；B4 含归一化 weighted teacher gradients，没有 `gS` 差项。

每窗先按固定 P 的实际一步求 `theta+(u)`，在该参数点用与 B3/RTD 相同父任务/组数的**本模型完整 rollout**得到 stop(gJ)，再用一般 `gate_vjp` 链式法则计算 `d_u J(theta+(u))`。不得调用专门把 `gT-gS` 写死的 `streamed_gate_vjp` 当作 B4；需新增 teacher-weight VJP/streaming reduction。只更新 u，不再给学生额外直接 PG。u 的更新采用过去反馈 RMS 与向初始等权的 ridge；当轮反馈父任务不进入 inner loss、权重归一化或 P。跨轮已有训练历史不抹除，标 rotating meta-training。

这与 `scalar_sigmoid` 不同：后者只有一个全局 a，仍是 transport；本项有每包自由权重、只加权 teacher CE。B4 先用固定 P 一步真实更新保证求导正确；若将来给 AdamW 强版，必须真 unroll，包括 optimizer state，不用一步近似冒充。反馈组数向原完整 RTD 配额匹配，若原有两组则在当前 B4 theta+ 独立采两组并平均，不丢掉第二组。

## 5. 资源匹配与两个曝光视角

### 5.1 槽位分布和逐项计数

固定集合 RTD 的基础抽样为 `p0(i)=1/M * 1/K_parent * 1/2 * 1/T_state`：先在当前 inner 的 M 个可用父任务中均匀抽任务，再在该父任务 K 个去重完整 state 中均匀抽 state，再从两条冻结 source 中均匀抽一条、从 T 个教师经验项均匀抽一条；无教师时 teacher 是空项。历史独立请求重数按 §3 保留。B1/B2/B4 强配方使用 `pT(i)=p0(i|teacher_available)`，归一化常数与各父任务总质量必须写出；无教师 state 不抽成 teacher-only 训练样本。B3-SFT 同 pT，B3-mix 同 p0。强基线的教师曝光可能更多，必须列出；同列表归因版固定 p0/同 targets，单独标记。

动态购买步 RTD 为 `.75 L_D + .25 L_q`，8 个 old + 8 个 new 的 raw slots，对应加权 8 槽；后续 replay raw 8 槽。初始 no-op 不能记作8次有效教师学习。三轮 `36×8=288` 是名义加权槽位上限，绝非所有臂都已经执行 288 个 teacher slots 或完全相同两侧 token。固定最终集合比较不保留动态新包分支；若做逐次 reveal 比较则必须保留该 .75/.25 日程，另成表。

每步至少记录：`optimizer_commits`、supervised raw/weighted/source/teacher slots、query/state/source snapshot/teacher hash、sampling probability、importance weight、prompt tokens、source action tokens、teacher action tokens、每侧实际 forward/backward tokens、reference forward tokens、feedback generation/score tokens、virtual/pilot/VJP tokens、同机 GPU active/reserved seconds、CPU 秒、峰值显存。无损失的监督空槽不能计 backward；零权重却实际计算的分支计实际计算。重复阅读教师缓存不重复算新教师费，调优中的学生计算每次都算。

### 5.2 V-S：固定教师账单 + 固定完整动作槽位

同 A_j 的 B_j 完全相同，主目标 288 个监督完整动作槽位（三轮各96），每槽的概率由上节公开。36步×8槽作为标准；强配方可以在**不增288槽**的前提下比较72步×4槽，从而允许自己的 step tuning。两折边界仍每96槽；反馈窗口仍在每轮已处理监督槽位 0/24/48/72 处（36步版对应 step 1/4/7/10）。B3 每窗口把 PG 合入该次提交，不偷偷增加 optimizer steps；B4 likewise。

同优化器归因表固定 36 步、8 槽、同 P/eta 数值程序、同 slot targets/order。B2 计算 chosen+rejected，SFT只算teacher，B4可能还需元梯度，**相同槽位不会自动使这些不同目标等计算**；不做空 backward 浪费算力来伪造相等，由 V-T 补充比较。对两个不同采购池，既报告总槽位，也报告有教师槽位及实际 token，避免训练剂量被遮盖。

### 5.3 V-T：曝光校正的明确采样分布

令 p_i 为该配方上面声明的 p0 或 pT；c_i>0 是在当轮已冻结完整文本上，用同 tokenizer 得到的该训练模式一次 slot 所需**实际学生训练 token 成本**。至少同时保存 action-only 与 prompt+action；B2 包含两侧，B1 teacher侧，mixed包含两侧。模型生成、reference、反馈梯度、VJP 单列，不能把 prompt 长度或 PG backprop 藏掉。

采用正概率的 inverse-cost 抽样与重要性校正：

`Z = sum_j p_j/c_j;  q_i = (p_i/c_i)/Z;  rho_i = p_i/q_i = Z*c_i`。

每次 draw 的损失是 `rho_i * L_i`，以预定 draw 数平均；B4 的 w_i 再乘在 L_i 上。完整序列 log probability 始终是 token logprob 的**和**。不能把 `log pi(y|s)` 换成 `/len(y)`，不能丢 rho、裁 rho 或用 minibatch self-normalization 后仍声称原 p 目标。c_i 和 q_i 在 round 内固定且 stop-gradient；只有合法已购 state 可用于长度目录。动态 replay 每个 old/new 分支独立算 q/rho 后保持 .75/.25，不对未购文本估真实长度。

`E_q[c]=1/Z`，先据此为每轮/每步分配预定 draw 数和梯度累计，V-T 比较固定的 T_update 目标及相同提交步数。这里 T_update **包括产生学生更新的监督与 PG score/backprop**；RTD/B4 的外层 gJ/VJP、训练侧 pilot/reference 则另计 T_aux；总训练侧计算比较看 `T_update+T_aux` 和实测 C_train。B3 不因有 PG 就保留全部 SFT token 再免费加 PG。先从目标中留出实测/预估反馈梯度份额，剩余分给监督采样；若同反馈本身已经超出某个低预算，标该组合不可行并使用共同可行预算重跑参照，不能减少 B3 反馈配额。

完整动作不能截断来凑整数 token。预注册允许误差为“每轮最多一个完整微批次成本，且最终相对目标 <=1%”；长样本使 1% 无法满足时扩大所有臂共同目标或如实标 unmatched，不只为某个臂放宽。预定 draw 数下重要性梯度的期望成立；按实际 token 提前停止会引入有限样本停止效应，因此主实现使用预先冻结的 schedule，保存其实际预算误差，不能把随机停止后的平均值宣称严格无偏/逐 token 完全相等。超目标不重抽便宜样本、不按训练结果改 schedule。

同 token 仍不严格同 FLOPs（attention长度、两侧、checkpointing等不同）。若标 **equal compute**，必须在同硬件类上用未嵌套重复累计的训练 GPU 时间或统一 FLOP 计数校准公共 C_train，所有 rollout generation、参考模型、pilot、VJP、PG 都包括，官方评测/排队另列。同父任务/rollout 数与精确 token/时间不一定同时满足；主表优先保留同反馈，记录计算差额，等总计算 RL 另表，不声称三个条件天然全相等。

### 5.4 模型、机器与调优纪律

全部 theta0 为同一 Qwen/Qwen3.5-4B 本地 snapshot/tokenizer/chat template，LoRA r16/alpha32/相同 target modules、BF16、seed=0，完整状态与采样 scorer一致。新实验冻结当时有效完整配置；不要把旧 manifest 的缺省值用当前 YAML 覆盖，尤其 `rai_R0` 的早期 action/context/memory 设置。

当前 rai_R0 是 RTX 6000 Ada 48GB，rai_R1 是 A100 80GB PCIe，rai_R1s 是 RTX PRO 6000 Blackwell Max-Q 96GB；相同 hostname 不代表同 hardware class。按 `hardware.py` 的 hard fields 比较 GPU/内存/算力、driver、CUDA、依赖版本、host class；UUID/PCI/hostname 是实例元数据。单 run train/evaluate 绑定同硬件类，最好同卡；同类 hpg B200 的节点/UUID变化可以经现有 guard 接受。rai和hpg分别成表。

`replay-ledger` 会写入 `fixed_source_hardware_hash`，`make_manifest` 会验证。不能把 rai A0/A1 配置丢到 B200 后删除这一字段绕过约束。首选等待同一 hpg B200 硬件类上的 R0/R1 最终 ledgers，以同类训练四格和全部强基线。若专门研究 rai 导出的证据在 B200 上重训，需新建明确区分“采购来源硬件”和“重训硬件”的迁移实验协议，所有 D0/D1/B1–B4 都重训；原文件与 guard 保留，不能与原自适应 rai结果组成同机主表。

所有强基线只用同一支持侧 meta-training/development 反馈调优，不看 official all-category 分数、65题校准集或 protected generated hashes。最小网格如下；每个 trial 从 theta0/相同 source seed 起步，step候选通过 batch重分保持既定曝光。学习率单位不同的优化器不硬共用一个数值。

| 配方 | 小网格，seed=0 | 选择与限制 |
| --- | --- | --- |
| B1 AdamW SFT | lr=[3e-6,1e-5,3e-5] × commits=[36,72]，共6 | 不额外扩 epoch/teacher pool；完整序列 CE，weight decay=.01 固定 |
| B2 AdamW logistic | lr=[1e-6,5e-6,1e-5] × commits=[36,72]，共6；beta=.1 | 从旧5e-6配方起；不联合调 anchors、margin类型和过滤器 |
| B3-SFT / B3-mix | 每种监督目标 AdamW lr=[1e-6,3e-6,1e-5]，commits固定36，共6；lambda_pg=1固定 | 有自己的lr选择；不 token-normalize advantage；两种目标均留记录，可选最佳作为主B3。若另调lambda须事先增加所有相关trial与计算预算，不看认证分临时扩网格 |
| B4 | eta 为合法 KL pilot eta的[.3,1,3]倍 × meta_lr=[.03,.1]，共6；ridge=1，commits36 | 每trial eta整轮固定；真实固定P一步求导，初始等权，features off |

24个 trial 的额外支持评估日程相同：每轮末在该轮feedback折上按固定parent表生成一组，单轮最多8任务×4、多轮最多4任务×2；当前支持结构对应34/40/34、每trial合计108次额外完整rollout。使用独立预声明 sampling seed 流（仍是一个 training seed），逐任务平均回报后汇总三个轮末块；这些分数只用于选配，不反传、不替换B3/B4必须取得的训练反馈。若所有候选并列，先较少步，再较低lr/meta_lr。这些是支持开发反馈，不是独立泛化估计。没有足够干净反馈时报告选择不稳定/不可辨识，不从认证集借题。每个trial的额外调优rollout全部计调优预算，B1/B2不能冒称整个选配过程没有环境访问。

默认在预声明 A1 参考池/V-S 上完成一次这些小网格，冻结后迁移到 A0、V-T、full-bank；若需每池单独调参，所有基线获得相同额外trial配额并加算成本，不能只救某个输的臂。RTD始终保留规范固定loss/gate/采购程序及训练侧KL pilot，不按BFCL分数另挑 loss 或人工调错误类别；同优化器归因也不重调RTD。主强配方与归因配方分开，不以“相同lr”剥夺基线合理优化。

## 6. 评测必须通过原官方身份与 campaign

baseline执行器应产出与 `verified_checkpoint` 可核验的 manifest、`round-N/{lora,checkpoint.json,round_state.pt}`、trajectory、teacher/compute journal。存真实参数、optimizer/controller状态和哈希；采用独立 baseline 配置校验器/完成审计，不能篡改当前RTD的36步不变量来适配72步。评测直接调用现有 `tools/rtd_experiment.py evaluate`，不另写 BFCL评分实现。第N个保存点的含义在表中标明：固定最终池的训练进度 vs 原在线预算点。

每个baseline自己的 checkpoint/config identity必然不同；要相同的是 `evaluation_harness_hash`（或有证明的评分连续性链）、BFCL内容与expected任务集合、base/tokenizer、evaluation_temperature=0、硬件类。不能要求不同模型拥有同一个整体identity。`identity.py:evaluation_harness_identity` 以评分相关文件/数据内容和 evaluation_* 配置为依据，git HEAD是元数据。保存实际生效identity，不能拿旧文档hash硬贴新run；旧结果只有在该run已审计的identity链内、其余绑定及artifacts hash不变时才复用。

`evaluation.py:evaluate` 在CPU合并LoRA并经 `tools/bfcl_hub_merge_export.py` 导出，再调用 `tools/bfcl_std_campaign.sh`；后者使用 Qwen/Qwen3.5-4B-FC、vLLM、all categories、8 threads与官方checker。`BFCLSTD_TEMPERATURE` 由evaluate设为0；直接shell默认 .001，不能漏设。memory/web-search虽然暂不可用于训练反馈，官方全量评测仍按原harness覆盖，不能只报支持子集或proxy。

复用 `evaluation_lock.py` + `tools/bfcl_campaign_lock.py`：tag锁为 `results/bfcl_std/.locks/<SHA256(tag)>/.lock`，port锁为 `results/rtd_v1/eval-port-<port>/.lock`；自动选空闲端口，独立 `result_p<port>_<UUID>`/`score_p<port>_<UUID>`。默认每项锁最长21600秒、60秒进度日志；等待记idle，不计active GPU。不能删除锁、共用tag、沿用他人vLLM端口或关闭锁来赶进度。不同tag/端口可以并行，但资源空闲与共享运行的影响另审计。

正式终点必须同时有 `OVERALL`、`data_overall.csv`、完整 generation/scoring ID集合、无缺失/重复/unexpected IDs、checkpoint/merged/artifacts哈希绑定、`validation.complete=true`。退出码0或一个CSV均不够。completed campaign重用不重新生成，其历史首次campaign成本仍计；`campaign_seconds=0,reused_campaign=true`不表示原评测免费。已有 `tools/rtd_check_evaluation_identity.py` 可只读检查链成员（目前工作树未跟踪资产，后续需随执行版本固定）；它不代替全量coverage与hardware检查。

同harness认证证明评测实现/覆盖一致，不自动证明数据独立。支持40题来自官方任务，须核验它们与all-category expected IDs的交集并随报告列出；完整官方Overall保持原口径，独立确认/认证只使用预声明未训练、未调优的集合，必要时从同一campaign逐题verdict另报固定排除集合后的诊断值，不将其改名为官方Overall。65题校准与其他protected内容始终不参与选配。

## 7. 后续要创建的配置与实现契约

以下均为**待创建**，本次没有写入。建议隔离新入口 `tools/rtd_baselines.py` 和 `src/bfas/rtd/baselines.py`，优先复用上面接口，不在运行中的 `experiment.py` 热改逻辑。新入口所需能力：冻结完成ledger审计、合法pool导出、显式feedback/sampling schedule、SFT/pairwise/direct-PG/meta-weight四种更新、tuning、compatible checkpoint和资源报告。V-T sampler与B3/B4不能靠未知YAML键默默忽略；遇未支持键必须报错。

| `configs/rtd/baselines/` 下配置 | 内容 |
| --- | --- |
| `bfcl_common.yaml` | 固定本地模型、LoRA、support/data/harness identity、seed0、fold、完整动作scorer、硬件类、36步/288槽、反馈规则、no new teacher calls |
| `fixed_A0.yaml`, `fixed_A1.yaml` | 由**完成来源**的 `replay-ledger` 生成；owned IDs、来源哈希/依赖/实际成本；不是手填当前两个包 |
| `bfcl_b1_sft.yaml` | teacher-only CE、AdamW、pT；分别与A0/A1绑定 |
| `bfcl_b2_pairwise.yaml` | 同state teacher/source、base reference、beta=.1、AdamW、无anchor/过滤 |
| `bfcl_b3_sft_pg.yaml`, `bfcl_b3_mix_pg.yaml` | 监督目标SFT或a=.5、direct-PG合并更新、原RTD反馈日程、LOO、固定lambda=1与独立lr网格 |
| `bfcl_b4_meta_reweight.yaml` | features_off、per_query scalar、softplus/mean-one、teacher-only、固定P真实一步、相同return反馈 |
| `bfcl_fullbank_random_l1.yaml` | 420合法public候选、L=1、sealed cap预算/窗口、random_control、B1 SFT；不是全payload直接训练 |
| `bfcl_fullbank_cost_known.yaml` | 可选历史成本公开的retrospective补充协议；使用最终B_j，不能与sealed条件混报 |
| `exposure_slots.yaml`, `exposure_tokens.yaml` | V-S的槽位目标；V-T的q/rho、成本定义、误差阈值、T_update/T_aux/C_train目标与来源 |
| `bfcl_tuning.yaml` | 上述24trial网格、固定支持开发日程、选择/并列规则、总调优预算、冻结输出 |
| `bfcl_suite.yaml` | pool×recipe×view展开；主B3两目标都执行；D0/D1及同优化器归因开关；严格完成/硬件检查 |
| `generated/{A0,A1}_{b1,b2,b3_sft,b3_mix,b4,d0,d1}_{slots,tokens}.yaml` | 展开后的自包含、可哈希配置；不依赖当前CLI不支持的隐式YAML inheritance；D0为额外归因格 |
| `generated/fullbank_b1_{slots,tokens}.yaml` | 全bank sealed随机B1的两个曝光视角；采购日程和训练调优分开保存 |

调优后的effective config、pool索引、feedback schedule、曝光schedule分别存新 `results/rtd_baselines/<suite>/...`，记录来源hash。配置里只存路径与hash，数据不嵌进YAML。源文件仍保留，checkpoint保存实际state。主RTD参照D1通过同一执行器资源调度适配其冻结方法；V-T是明确标注的规范允许曝光变体，不修改既有运行。其监督/门控/采购公式不因benchmark改变。

## 8. 精确命令：已有能力与待实现能力分开

**下面是未来执行命令，不是本次已执行记录。** 从相应项目根目录执行。所有写配置/结果、GPU/Slurm操作都等后续执行阶段；活动checkout本次保持原样。

### 8.1 现有CLI：完成后冻结同池与固定四格

以下rai路径是当前实际存在的名字。只能在§3完成条件满足后运行；生成的A0/A1仍绑定各自来源GPU类，因此这两组单独可运行，不自动构成跨GPU四格主表。

```bash
mkdir -p configs/rtd/baselines
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. .venv/bin/python tools/rtd_experiment.py replay-ledger \
  --config configs/rtd/v1_bfcl_c25_frozen_v101.yaml \
  --run-dir results/rtd_v1/rai_R0 --out configs/rtd/baselines/fixed_A0.yaml
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. .venv/bin/python tools/rtd_experiment.py replay-ledger \
  --config configs/rtd/v1_bfcl_c25_frozen_v101.yaml \
  --run-dir results/rtd_v1/rai_R1 --out configs/rtd/baselines/fixed_A1.yaml

# 例：A1的两种训练器均回到A1来源对应的A100硬件类，从theta0重训。
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=GPU-8b270cf8-6bb4-cee0-7060-88eba83d2fb0 \
  PYTHONPATH=src:. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python tools/rtd_experiment.py run --arm R0 \
  --config configs/rtd/baselines/fixed_A1.yaml --run-dir results/rtd_baselines/rai_A1_D0
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=GPU-8b270cf8-6bb4-cee0-7060-88eba83d2fb0 \
  PYTHONPATH=src:. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python tools/rtd_experiment.py run --arm R1 \
  --config configs/rtd/baselines/fixed_A1.yaml --run-dir results/rtd_baselines/rai_A1_D1
```

这只是现有§10.1默认四格能力展示，尚没有注入原完整RTD的双组feedback schedule；最终强基线同反馈表需要8.2的扩展。不要把默认fixed-evidence的每窗一组反馈拿来与自适应RTD的两组配额等同。A0对应用 `--config .../fixed_A0.yaml`、新run目录、其Ada来源GPU类；HPG四格则在完成的同类B200来源上另生成两份fixed配置，不能直接搬上述rai配置。

### 8.2 待实现入口：prepare / tune / run 的明确命令契约

`tools/rtd_baselines.py` **目前不存在**，下面不是当前CLI支持的参数。后续实现必须按这些命令产出reviewable配置/目录，并通过离线正确性检查后才能启动。`prepare` 必须拒绝未完成ledger、硬件类不匹配、缺反馈组、缺token成本目录；不自动resume源run。

```bash
# suite里预先填完成的同硬件R0/R1来源、全部recipe和view配置路径。
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. .venv/bin/python tools/rtd_baselines.py prepare \
  --config configs/rtd/baselines/bfcl_suite.yaml \
  --out configs/rtd/baselines/generated

# 调优仅生成支持侧分数/selection，不跑全量官方认证。
PYTHONPATH=src:. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python tools/rtd_baselines.py tune \
  --config configs/rtd/baselines/bfcl_tuning.yaml \
  --prepared configs/rtd/baselines/generated \
  --out results/rtd_baselines/tuning

# run使用selection写出完整resolved manifest；seed/反馈日程/方法不可暗改。
for pool in A0 A1; do
  for view in slots tokens; do
    for recipe in b1 b2 b3_sft b3_mix b4 d1; do
      PYTHONPATH=src:. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
        .venv/bin/python tools/rtd_baselines.py run \
        --config "configs/rtd/baselines/generated/${pool}_${recipe}_${view}.yaml" \
        --selection results/rtd_baselines/tuning/selection.json \
        --run-dir "results/rtd_baselines/${pool}_${recipe}_${view}_s0"
    done
  done
done
for view in slots tokens; do
  PYTHONPATH=src:. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    .venv/bin/python tools/rtd_baselines.py run \
    --config "configs/rtd/baselines/generated/fullbank_b1_${view}.yaml" \
    --selection results/rtd_baselines/tuning/selection.json \
    --run-dir "results/rtd_baselines/fullbank_b1_${view}_s0"
done
```

GPU选择由调用环境明确设置；上述循环只适用于同一硬件类的suite，不能轮流拿空闲异型卡。`run`的终点评测默认关闭以便训练资源核验后统一认证；D1忽略selection中的baseline超参数且须验证其冻结方法指纹。fixed池三轮最终端点用round3，若tuned72步则每轮24步，checkpoint元数据如实记录，不伪造成原12步。

### 8.3 现有官方evaluate：新基线产物准备好后

```bash
for pool in A0 A1; do
  for view in slots tokens; do
    for recipe in b1 b2 b3_sft b3_mix b4 d1; do
      PYTHONPATH=src:. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
        .venv/bin/python tools/rtd_experiment.py evaluate \
        --run-dir "results/rtd_baselines/${pool}_${recipe}_${view}_s0" --round 3 \
        --evaluation-lock-timeout 21600 --evaluation-lock-log-interval 60
    done
  done
done
for view in slots tokens; do
  PYTHONPATH=src:. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    .venv/bin/python tools/rtd_experiment.py evaluate \
    --run-dir "results/rtd_baselines/fullbank_b1_${view}_s0" --round 3 \
    --evaluation-lock-timeout 21600 --evaluation-lock-log-interval 60
done
```

不传 `--port` 使用原自动lease，不传 `--config` 覆盖训练manifest。若只为调试旧trainer导出兼容性，现有shell命令是 `BFCLSTD_TEMPERATURE=0 bash tools/bfcl_std_campaign.sh "$CUDA_VISIBLE_DEVICES" auto "$BASELINE_TAG"`；它不自动补baseline的RTD identity/ledger证书，因此主报告仍走上述evaluate。

### 8.4 rai / hpg 调度

rai统一设置 `CUDA_DEVICE_ORDER=PCI_BUS_ID` 并使用已核验空闲的目标类GPU UUID，训练进程退出后再评测；不能占用运行中的R0/R1卡或因超时杀现有campaign。本文不检查/承诺现在哪张卡空闲。

HPG项目路径从该机checkout确认，先 `cd` 到它再提交。已有 `scripts/rtd_run_hpg.slurm` 是单B200、14 CPU、240G、8小时，且只接受RTD_ARM=R0/R1。现成D1可这样提交（需该机同B200来源的fixed_A1）：

```bash
sbatch --time=24:00:00 \
  --export=ALL,RTD_ARM=R1,RTD_CONFIG=configs/rtd/baselines/fixed_A1.yaml,RTD_RUN_DIR=results/rtd_baselines/hpg_A1_D1 \
  scripts/rtd_run_hpg.slurm
```

新baseline不要传 `RTD_ARM=B3` 给这个脚本。后续新增的 `scripts/rtd_baseline_hpg.slurm` 契约：同14CPU/240G/1×B200，source `tools/aw_hpg_common.sh`，接收 `RTDB_COMMAND`、`RTDB_CONFIG`、`RTDB_RUN_DIR`、`RTDB_SELECTION`，运行8.2入口；eval阶段改调8.3原入口，先释放训练模型。示例（脚本目前也待创建）：

```bash
sbatch --account=fsu-compsci-dept --qos=fsu-compsci-dept --partition=hpg-b200 \
  --gres=gpu:b200:1 --cpus-per-task=14 --mem=240G --time=24:00:00 \
  --export=ALL,RTDB_COMMAND=run,RTDB_CONFIG=configs/rtd/baselines/generated/A1_b3_sft_slots.yaml,RTDB_RUN_DIR=results/rtd_baselines/A1_b3_sft_slots_s0,RTDB_SELECTION=results/rtd_baselines/tuning/selection.json \
  scripts/rtd_baseline_hpg.slurm
```

只允许在不同完整run间利用空闲同类卡并行；不在一个run中换GPU类。代码/配置在新执行版本冻结，数据/模型可共享只读；不对活动checkout做覆盖式sync。调优及train/evaluate可拆作业，真实单项耗时超过walltime则按checkpoint恢复，不无限复制提交。

## 9. 计算估算：用当前账本校正旧估计

这些是排期范围，不是已测baseline速度或Slurm排队承诺。当前 `compute.jsonl` 的完整官方campaign耗时：rai_R0约4.97小时（cleanup失败后复用仍计这笔）、rai_R1成功全量约3.98小时，rai_R1s约2.52小时。原旧计划“每终点约1小时”不再采用。不同卡/模型行为/重试会改变时间。

训练计算不能把所有 `compute_end.wall_seconds` 相加：嵌套 generation/forward/feedback 会重复计数。按 `compute_begin.parent_sequence is None` 对应的完成end求和，失败/重试分列。读取时R0约13组reference/actual反馈已花6.10小时，R1约8组1.54小时，R1s约13组5.73小时；包含当前轮次的重试/不同长度，不能推断纯硬件加速比。多轮rollout是主要不确定性，监督本身较小。

原反馈配额每组第一/三轮34、第二轮最多40，若每窗只一组共432 rollout，若每窗两组最多864；实际按最终reused表落在其间。B3/B4同配额，额外tuning计另账。以下每条训练估计均以36/72步、小池、最多864反馈rollout为前提，不含官方终点。

| 项目 | rai 单卡预留 | hpg 1×B200暂定预留 | 说明 |
| --- | --- | --- | --- |
| B1/B2 选定配置训练 | .25–1.5 GPU h/条 | .15–1 GPU h/条 | 包含source/reference，池/长prompt变大可超；未实测 |
| B3/B4 或同反馈D1训练 | 4–12 GPU h/条 | 3–10 GPU h/条 | HF随机解码/多轮环境主导；B200区间只是排期假设，需要首条计时校准 |
| 一次官方全量终点 | 2.5–5 GPU h | 暂留2–5 GPU h | rai有上述依据；本地没有足以认定hpg速度的匹配计时 |
| 24trial支持侧调优 | 60–160 GPU h | 45–130 GPU h | 每trial独立训练及同支持评估；不对24trial逐个跑全量官方 |

最小A1包：5个baseline配方+D1，两个视角共12个最终终点；训练为4条监督类+8条反馈类，约33–102 rai GPU h、25–84 hpg GPU h，官方约30–60 / 24–60；合计约63–162 / 49–144，不含一次全局调优。

完整A0+A1+fullbank-B1包：24+2=26个终点；训练为10条监督类+16条反馈类，约66.5–207 rai GPU h、49.5–170 hpg GPU h，官方约65–130 / 52–130；再加一次24trial调优，合计约**192–497 rai GPU h / 147–430 hpg GPU h**（整数小时四舍五入）。四格D0、同优化器额外归因、cost-known补充表、独立三个教师预算集合重训、base重新认证均另加，不能塞进这26终点。两张同类卡在任务独立且无排队时墙钟约折半，总GPU h不减；同一套suite不拆到rai与hpg混汇总。

暂按每条job24小时上限，保留失败重试和锁等待的独立预算；不能把8小时Slurm模板当每条必然8小时内完成的证据。checkpoint/LoRA保留；每个4B合并BF16模型约8GB量级，26个若全留约208GB仅权重，另有raw BFCL结果/长token trace/源码环境；建议后续预留300–500GB并依据实际目录测量，不在本次清理任何结果。

## 10. 现在能准备什么，哪些必须等待

**本次现在完成的工作只有本审计计划。** 后续执行任务可以在不依赖最终ledger的隔离版本中先做：配置schema与上述小网格、B1/B2完整动作loss adapter、B3合并PG更新、B4每包标量VJP、q/rho曝光scheduler、资源journal、baseline checkpoint导出；只用合成小模型/固定fixture做CPU正确性检查。必要检查包括同state/rejected空动作、PG方向与sampling hash、LOO/畸形失败保留、B4有限差分VJP及features禁用、采样概率与预算误差、相同feedback组映射、同harness完整性。已有 `tests/test_rtd_{return_gradient,functional_step,transport,acquisition,evaluation_lock,evaluation_identity_reuse}.py` 和 `tests/test_pair_unit.py` 提供接口依据；后续测试应覆盖新增算法风险，不重跑生产GPU实验当单测。

必须等最终来源：A0/A1全部query IDs/依赖/实际账单、各预算checkpoint集合、完整的每窗feedback父任务与reused组数、实际raw/weighted曝光与token目标、完成审计及硬件/评测identity链；然后才能导出final pool、设最终匹配预算、完成tuning与正式baseline训练/认证。不能把当前round-1的两个包命名为final，也不从活动recovery临时state拼结果。

必须等同类机器/排队与执行版本冻结：正式同机suite、全量campaign和固定ledger四格；HPG状态/路径/空闲卡在调度时只读核验，本次未远程查询。当前已有R0/R1训练/评测继续，不追加重复作业。

严格在线随机学生state短续写还必须等待有效请求路径、完整历史依赖、实际可用教师及独立明确预算；当前 `new_teacher_calls=false` 不改。可完成缓存限制版、同池强训练器、**不可省的B3**和B4，再明确写出online覆盖仍未完成，不能把这项限制包装为已验证的teacher-token efficiency。

验收材料为：26个预定终点（分阶段完成也逐项列缺失）、每配置调优选择依据、两种曝光的实际分布/误差、教师/学生/反馈/调优计算账本、同硬件与官方identity/coverage验证。若B3/B4尚未实现、没有最终ledger或没有同机认证，则结论仍是“强基线未齐”，不能以R0/R1的局部结果补写性能主张。
