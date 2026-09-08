# BFCL sealed_replay：RTD protocol v1.1 预算、采购配额与曝光修订计划

**状态：待用户批准的协议修订提案，尚未实施。** 本次工作只创建本文；没有修改源码、配置、银行、结果、测试或运行作业，没有执行训练、评测、账本恢复或银行重建。现有 v1.0 三轮继续完成，所有旧 manifest、checkpoint 和账本保留原义。

建议批准的组合是：**按 recorded bank cost 定义累计预算；采用经离线审计、全类别统一的封存银行内容 cap；每窗最多 20 包、40 次有限采购子决策；40 个完整动作槽的训练块；共享参考点的批量一阶插入近似。** 这是回顾性有限银行实验的新协议，不是在线成本保证，也不是只修改两个 YAML 数值。10%/25% 为主要比较点，50% 为上限与饱和诊断。

## 1. 核对依据与结构性空结果的范围

治理文件是 `docs/RTD_END_TO_END_METHOD_AND_EXECUTION_V1.md`，尤其 §5–6、§9.2、§10.3、§14；执行沿革见 `docs/rtd_v1_protocol.md`。本文对照的本地源码 HEAD 为 `8b837c09872e38f8c56b6cb751f8d9570aacf365`。运行方法配置仍写 `protocol_version: 1.0.1`；v1.0.7 等修复版本与方法配置版本应分开理解。

只读核对了：

- `results/rtd_v1/rai_R0/{manifest.json,trajectory.json,teacher.jsonl,compute.jsonl}`；
- `results/rtd_v1/rai_R1s/`、`results/rtd_v1/rai_R1/` 的同名文件；
- `data/rtd/v1_bfcl_c25/public/requests.json`、`sealed/audit.json`，以及可用包 payload 的成本、类别、折、行为数量和依赖数量；没有把逐包成本或教师文本提供给运行中的 selector；
- `PROJECT_STATE.md` 的 HPG 状态记录。当前本地 `results/rtd_v1/` 没有 HPG R0/R1 镜像；本文未登录远端核验，也不把本地副本当作实时运行状态。

银行 public requests 的 SHA256 为 `2d27d0f4dcfd1076ac2fbdba6ff3d2789caf5971c14a5da61634d710d00084e8`。实施前应重新核验冻结输入，不能依据仍在追加的运行文件时间戳推断全局完成状态。

| 已核对的银行口径 | 包数 | recorded cost | 平均 tokens/包 | 最大 recorded cost |
|---|---:|---:|---:|---:|
| demo_attempt，exact | 84 | 21,203 | 252.4167 | 1,409 |
| generator_item，estimated | 336 | 34,167 | 101.6875 | 371 |
| 可用合计 | 420 | 55,370 | 131.8333 | 1,409 |
| fold 0：42 demo + 267 generator | 309 | 36,051 | 116.6699 | 1,097 |
| fold 1：42 demo + 69 generator | 111 | 19,319 | 174.0450 | 1,409 |

420 个可用包各有一个 `behaviors` 条目，均无未付依赖；不等于 420 个独立父任务。支持集仍是 40 个父任务，两折 22/18；三轮 inner fold 为 0/1/0。全部归档有 698 条，其他条目不能因提高配额而变为可用。

当前银行 public cap 总和为 `84×65,536 + 336×8,192 = 8,257,536`。`caps.affordability()` 以此计算 `[825753,2064384,4128768]`。最小预算已约为全部 recorded cost 的 14.91 倍，三个预算约为 14.91/37.28/74.57 倍。在本银行顺序结算下，即使买完所有可用包，实际支出也不足以使这些累计上限绑定。

`max_new_packages_per_decision=1` 和 12 个总窗口给出的是 **12 包的硬数量上限**；`12×131.8333=1,582` tokens、约银行的 2.86%，只是按均值推算，不能称为硬 token 上限。便宜包、昂贵包及 empty 会改变实际值。

本地 R0 已保存两轮、24 步、5 包、491 tokens；R1s 为两轮、24 步、4 包、582 tokens。用户给出的 R1 382 tokens/4 包与 `PROJECT_STATE.md` 的 HPG R1 第二轮记录对应；本地 rai_R1 副本仅保存第一轮、12 步、2 包、177 tokens，并已授权第二轮。HPG R0 的 188 tokens/2 包也来自运行记录，待收齐远端原件确认。上述数值不是第三轮最终值。

因此，v1.0 能说明小剂量闭环的运行情况，**不能据此判断预算受限采购器是否有效**。这不等于已经证明门控、插入价值或训练器无效。

## 2. 预算基础的两个选项与硬账本语义

### 2.1 两个选项共用的内容预算

这里的“内容”明确指 `payload.cost` 已记录的输出成本：demo 的归档 exact 输出计数，加 generator 按归档文件估计分摊的成本。**不是**对保存文本用学生 tokenizer 重新计数，也不是完整 provider 总账。`input_tokens`、`reasoning_tokens`、货币费用、未知字段及缺失 drafts/repairs/verification 仍按原样保留。

固定分母 `B=55,370`，只取可用银行一次，不按轮、折或当前剩余库存重算。累计预算用正整数 half-up：`C(p)=(B*p+50)//100`，`p∈{10,25,50}`。

| 轮/预算点 | 累计 ceiling | 若上一轮花满，本轮新增授权 | 按全池均值的累计包数 | 保持全池 20%/80% 数量混合的 demo/generator | 只买 demo 的均值容量 | 只买 generator 的均值容量 |
|---|---:|---:|---:|---:|---:|---:|
| 1 / 10% | 5,537 | 5,537 | 42.0 | 8.4 / 33.6 | 21.9 | 54.5 |
| 2 / 25% | 13,843 | 8,306 | 105.0 | 21.0 / 84.0 | 54.8 | 136.1 |
| 3 / 50% | 27,685 | 13,842 | 210.0 | 42.0 / 168.0 | 原始计算 109.7，库存最多 84 | 272.3 |

以上都是成本均值下的规划量，不是采购承诺、最优化解或逐类预留。§9.2 要求横轴是实际花费，不能把未花满的运行画到 ceiling 上。25% 的原始值是 13,842.5；旧 C23 文字采用 floor 得到 13,842，本文按用户指定取 **13,843**，必须显式冻结舍入规则。`budget_ceilings` 目前是 manifest/audit 产物，不是可随意填写的现有 YAML 键。

### 2.2 选项 A：保留原 public-cap reservation，仅换分母

保留 `caps.PUBLIC_CLASS_CAPS` 和 `public_cap()` 的现有含义：demo 65,536、generator 8,192，来自公开回放约定/独立归档配置；保留 public requests 及 sealed payload 原件。`caps.affordability()` 接受新的预算基础，`cli.bank_audit()` 使用 sealed audit 中的固定总成本；`Ledger` 的结算单位不变。

实际控制链是：

1. `broker.SealedReplayBroker.list_candidates()` 用 `spec.cost_upper_bound <= min(传入额度, ledger.remaining)` 过滤；
2. `acquisition.AcquisitionPolicy.choose()` 再按公开 cap 过滤；成本回归器的低预测值不能放行硬预算不够的包；
3. `broker.acquire()` 在读 payload **之前**调用 `Ledger.reserve()`；
4. `Ledger.settle()` 检验 `actual <= reserved cap`，追加 `reveal`，以 actual 入 `charges`，删除 reservation。正常结算的余额释放是隐式的，并没有另写一条 `release(cap-actual)`；`release()` 当前主要处理未完成预留/异常；
5. `experiment.round_start()` 调用 `authorize(C_r)` 提高累计 ceiling，绝不清零已花成本。

只改分母的后果：第一轮 5,537 小于两个 cap，零候选；第二、三轮 demo 仍完全不可行，只有 generator 在全局余额至少 8,192 时可行。即使取消窗口配额，顺序 reserve→settle 下，generator 最后一次获准前也要留 8,192；按当前 item 最大实际成本 371，空起点下第二轮可花至多 `13,843-8,192+371=6,022`，第三轮至多 `27,685-8,192+371=19,864`，实际还受库存、折及采购次数限制。不能用 `ceiling/cap` 推断顺序结算的最终包数：释放余额允许反复购买。

若再把 `C_r/4` 当严格窗口额度，三个起始值都小于 8,192，连 generator 也买不了。§3 推荐的公开 cap 最小窗口修正最多缓解后两轮，无法挽救第一轮或 demo。

**结论：A 保留 §6/§9.2 的原封存信息边界和硬预算，但不能完成三个有效预算点的类别公平采购比较，不推荐作为主实验。** 不得以“最终会释放”绕过第一步预留，也不得在 ceiling 外另借一个未计入的 reservation buffer；那已不是原来的 hard authorization。

### 2.3 选项 B：每类一个、按冻结银行内容认证的 cap（推荐，但需要正式例外）

建议 v1.1 BFCL bank cap 为 `demo_attempt: 2048`、`generator_item: 512`。确定规则：**对当前可用银行每类 recorded cost 的最大值向上取 2 的幂**；1,409→2,048，371→512。按类均值 253/102、P95、成本回归预测或单包实际长度设 cap 都不能保证上界，禁止这样做。

这些值是本次只读离线审计得到的**数据依赖银行上界**，不是原请求配置、provider guarantee，也不是从独立外部数据得到的先验。原 §6 明确禁止由隐藏 usage 反推 cap；因此不能声称 B 完全遵守未修订的 §6。请求用户批准的例外是：

> 只允许特权离线 ingestion 对冻结、可用的 BFCL 回放银行发布一个总成本和每请求类别一个统一上界；类别定义不变，不能细分到 state、任务、长度桶或请求。逐包 usage、排序、成本分位表与教师内容仍不进入购买前特征。该约定只对有哈希绑定的封存银行有效，不推广到真实在线请求。

本文的均值/最大值/折分布是研究审计信息；运行 selector 仍只接收 `PublicQuerySpec`、合法购买前特征、当前授权余额和已购观测，不额外输入这些逐折/逐类成本统计。尤其不能把真实逐包成本填进 `expected_costs`。公开两个 cap 本身仍泄露类别级上界，须作为方法限制报告。

实施时在新目录 `data/rtd/v1_1_bfcl/` 生成新 public manifest 和离线 cap certificate；复制冻结 payload，核验其内容与旧银行逐包一致。保留 request ID、支持父任务、成本与 confidence、失败记录、排除规则和依赖，不裁短长包、不删除高成本包来适配 cap。certificate 绑定旧/新 public hash、sealed integrity hash、类别、规则、可用集合和 `all actual <= class cap` 结果。可用集之外的不可用条目保留原始 cap/provenance，不能误认证为也满足新 cap。不能覆盖 `data/rtd/v1_bfcl_c25/`。

`spec.cost_upper_bound` 在该版本表示认证银行上界；`cap_provenance` 写清 `sealed_bank_class_content_envelope`、1.1.0、审计规则和 certificate hash，保留旧 request-cap provenance 作为历史字段。新可用 cap 总和为 `84×2048+336×512=344,064`，仅是诊断列；**预算仍是 55,370 的比例，绝不能再次以 344,064 为分母。**

硬预算证明保持原形。设已结算成本为 S，未结算预留和为 H，授权为 C：

```text
不变量：S + H <= C
reserve(q,c)：仅当 c <= C-S-H 才执行
settle(q,a)：必须 0 <= a <= c
结算后 S'=S+a，H'=H-c，所以 S'+H' <= S+H <= C
```

依赖链须对所有未付 cap 一次检查；零实际成本包仍占购买次数且不能绕过 cap；复用已购包不重复收费。若 actual 超 cap，应终止该次运行并标记 bank/accounting error，不能静默扩 cap、少收费、截断内容或继续把已看见的包视作未揭示。内容超过 cap 的异常在打开 payload 后才发现，所以必须在新银行上线前完成全量认证。

**B 保留对 recorded/estimated replay-token 货币的硬授权保证，但修订了上界来源的信息约定。** generator 的 estimated 值是硬约束计算所用的账本数，不是缺失 provider 账单的真实上界。无论 A/B，历史 demo 1,233,607 exact 与生成池 142,727 estimated 的更大成本仍分别列账，新教师调用/token 为零。

## 3. 每窗口 token 配额与 K：定义到可实现的状态机

### 3.1 累计预算不重置，窗口余额自动结转

推荐 `K=20`。20%/80% 全池混合下约 4 demo + 16 generator，每窗约 2,637 recorded tokens；实际两折混合约为 fold 0 的 2.72/17.28 包、2,333 tokens，fold 1 的 7.57/12.43 包、3,481 tokens。K=10/16/20 的全程数量上限分别为 120/192/240：K=10 很可能在 25% 前就受次数限制，K=16 连全池均值下的 210 包都容不下；20 是用户提出范围内余量最大的选择。**不据此强制 4:16 类别配额或重采样类别。**

第 r 轮开始授权 `C_r`。第 w 个外层窗口（步骤 1/4/7/10）开始时，前一窗口没有未决 reservation，记：

```text
R = C_r - ledger.spent                    # 所有前轮实际支出仍扣除
W = 本轮包含当前窗口的剩余外层窗口数       # 4,3,2,1
q_base = ceil(R / W)
c_max = 当前折、依赖已拥有、未购且全局可支付候选中的最大公开 cap；无候选则 0
Q_window = min(R, max(q_base, c_max))
```

`Q_window` 是窗口 **实际结算成本 + 未结算预留** 的硬额度，不是预计成本，也不是必须花掉的目标。`c_max` 修正可从本轮后续窗口提前分配额度，但不能提高累计 C_r；目的是避免 10% 首窗的 1,385-token 平分额结构性排除 demo。B 下首窗额度为 2,048，两个类别都可候选。后面窗口根据真实 R/W 重算，所以 early spend、empty 和 cap 余额都自然反映到后续额度。窗口内每笔成功结算同时释放全局和窗口 reservation 的剩余部分。

若上轮恰好花满，三轮未加 `c_max` 修正的首窗基准分别是 `ceil(5537/4)=1385`、`ceil(8306/4)=2077`、`ceil(13842/4)=3461`；修正后分别为 2,048/2,077/3,461。不能把第二、三轮额度简单设成 `C_r/4` 再当成新资金：C_r 是累计授权。若希望改用事先固定分配，则需另写 carry 规则；本提案只采用上面的动态定义。

购买条件同时满足：当前折/依赖合法、全局 `remaining>=cap`、窗口 `remaining>=cap`、本窗新付费请求数 `<20`。候选展示与 acquire 之间重新检查，不能只在 selector 过滤。未付依赖也分别收费并占 K；本 BFCL 可用集没有这种链，但通用账本测试必须覆盖。已拥有的包不占新的购买次数。

### 3.2 empty 必须有有限、明确的机会成本

只把 `max_new_packages_per_decision` 改成 20 并在首次 empty 时结束窗口，零价值、非约束成本下平均仍约买一个包：`E[N]=sum_{k=1..20}2^-k≈1`。这是另一个会保留结构性小剂量的实现陷阱。

推荐每外层窗口预先给 **M=40 次采购子决策机会**，新包无放回，最多成功购买 K=20。每次 empty 消耗一整次子决策，不能返还次数、重新设 RNG 或延长 M；仍按原时间表只提交一次学生更新。次数耗尽、K 满、无可行候选或额度耗尽时结束采购。若全部 empty，执行旧数据训练块/合法 exact no-op。子决策只做采购分配，不额外插入实际训练步或反馈 rollout。

这明确修订了 §5–6 的“empty 消耗整个单包窗口”：v1.1 中它消耗预先有限的子机会；不是保留原协议下的免费重抽。`replay_prior_mass=.5` 仍在每次子决策的 prior 中，成本对偶后的实际 empty 概率可能更高。只有成本不约束、价值为零时，成功次数约服从 `min(Binomial(40,.5),20)`，均值约 18.75，绝不是保证每窗 20 包。

R0、R1、R1s 使用同一个 M/K/Q、候选资格、空选项先验、无放回程序、成本模型及归一化；R0 的 learned value 设零，按相同成本对偶随机抽取，不能另改成不受成本倾斜的均匀抽取。三个臂结果花费不必相等，只要求授权与规则相同。

每个子决策的式 (9) 期望成本额度定义为 `b_sub = 窗口剩余实际授权 / 剩余子机会数`。这取代 v1 中“本轮余额/剩余外层窗口”的单次分配；外层该比例已用于 Q。必须区分字段 `remaining_outer_windows` 与 `remaining_subdecisions`，不能在日志里把 40 次都称作外层窗口。每次结算后重新计算 b_sub、合法候选、成本预测与对偶，不把窗口额度重复发放 40 次。

冷启动 `CostRegressor.predict()` 仍返回公开 cap，之后只用已购成本更新，exact/estimated 的观测噪声处理保持不变。保守 cap 加 b_sub 初期可能使 empty 增多；不能用本次审计的隐藏逐包成本绕过冷启动。应在批准后的合成公开成本测试中检查该效应，正式运行报告冷启动浪费，不能声称新规则必然花满。

### 3.3 K=20 的明确限制

即使按全池均值、前两轮完全花满，第三轮新增 13,842 tokens 约需 105 包，而 4K 只有 80 包。按前述两折均值，round 1/2 花满约买 47/48 包，第三轮再买 80 个 fold-0 包只能到约 23,177 tokens，即全池 41.9%。这是情景计算，策略可能选择更贵的包而更接近 50%，也可能更低。

因此本提案承诺“50% 为允许上限”，不承诺高预算一定绑定。要用典型混合充分探索最后新增预算，需 K 大于 20、更多窗口或改变 checkpoint；都超出本提案默认组合。应首先使 10%/25% 有可观察采购量，明确区分 token、K、M、cap tail 和库存各自的停止原因。

## 4. 多包选择与曝光守恒插入：采用什么近似

### 4.1 原协议允许什么

§5–6 与 §9.4 原文只允许每窗一个新包、同一个 theta、一次虚拟参考、一次真正提交。**原 v1 既没有授权多包逐次精确算法，也没有授权批量算法。** 本节请求正式扩展有限曝光目标及采购机会定义。

精确的逐包条件插入必须在已选集合更新后的训练分布 D_j 上重算 `g_Dj`、虚拟 `theta_Dj+` 和该点的 `g_J_ref,j`，仍从同一外层起点定义最终更新。不能把前一个包的实际 SGD 更新先提交，再声称只是一个外层更新；也不能只从 g_D 减一项却复用旧 g_J 并称为精确。每选一包重复完整参考反馈，最坏新增近 K 倍参考 rollout，当前速度不适合作为默认。

**推荐共享参考点的批量一阶近似。** 窗口内顺序选择、逐包 reserve/reveal，以维护硬额度和无放回，但不在每次选择后运行梯度/反馈来刷新价值。每外层窗口只抽一次 beta，冻结购买前 feature rows、旧数据参考和价值 posterior；成本回归可在每次已付 reveal 后更新。pending 内容仅由特权训练器接收，不能被用于尚未购买候选的评分。价值标签在本窗口实际提交后一次合入 posterior，下一窗口生效。

### 4.2 固定容量的插入块，避免买 K 包就获得 K 倍曝光

推荐 `slots_per_step=S=40`，在同一个起点 theta 抽取并固定 40 个旧槽计算 g_D。新块有 K=20 个位置，每位置使用两个完整动作来源槽；对选中的每个 q，在其合法状态内均匀抽状态/来源，得到两槽平均 g_q。没有被购买占用的位置代表继续旧数据训练，数学上复用同一个 g_D，不重复做旧梯度计算。

设本窗买到 m 包，`epsilon=.25`，`alpha=epsilon/K=.0125`：

```text
g_new_block = (1/K) * [sum_{q in pending} g_q + (K-m)*g_D]
g_actual    = .75*g_D + .25*g_new_block
            = (1-alpha*m)*g_D + alpha*sum_{q in pending} g_q
theta_plus  = theta - eta*P*g_actual
```

m=20 时旧/新曝光为 .75/.25；m=0 时严格回到旧数据参考。m<20 时未使用的新块位置回给旧数据，**不是把少数包重新放大到合计 .25**。所有系数非负，总和为 1。此处改变的是 §5 式 (7) 的多位置曝光分配，`FrozenStep` 的算子和 `insertion_fraction: .25` 的整个新块容量不变。不能既对每包给 .25、又声称总新曝光只有 .25。

对每个新请求记录：`v_q=-eta*<g_J_ref,P(g_q-g_D)>`，只用同一个 theta_D+ 的回报梯度；采购 logits 使用 `alpha*v_q`，不继续用单包 .25。共享参考的一阶预测增益为 `alpha*sum(v_q)`，与实际批量 perturbation 在 epsilon=0 的导数一致。由于没有在每次插入后更新 g_J，它**不是**有限 .25 下的逐包条件边际收益或联合真实 VOI；标签字段必须标记 `batch_common_reference`，不能冒充逐包精确重算。

一次请求一条 `pending_new` 标签，不按两个槽、多个别名或事件复制观测。标签与该请求的购买前 feature、round、start/reference hash 绑定；旧包重测仍单独记 `reweight_existing`。两个来源样本可能重复，不保证两次不同文本。共享 g_J 导致标签相关，仍用原固定噪声的 Bayesian 模型是一项近似，必须报告，不能据样本条数声称取得 m 份独立反馈。

真正提交只有一次；在 theta_plus 上取得 g_J_actual，并对上式以 `(1-alpha*m)` 和 alpha 做原门控 VJP 的线性组合。原 `streamed_gradient()`、`streamed_gate_vjp()` 支持任意非空槽列表，可复用；不需要改 `functional_step.py` 或 `return_gradient.py` 的算法。首次合法 reveal 的训练内 KL pilot 与本轮 eta 冻结规则保留；若 pilot 改 eta，只能在原来允许的 identity reference 情形重建参考，之后全窗共享该 eta。

### 4.3 必须持久化的审计与恢复信息

`teacher.jsonl` 继续是收费权威账本，扩展窗口事件/字段；`compute.jsonl` 记录决策与计算；`steps/*.json`、`trajectory.json` 保存提交后的摘要。至少包含：

- budget basis、分母、舍入、C_r、窗口 id、R/W/q_base/c_max/Q、cap certificate hash；每笔 reserve 的 cap、窗口剩余和全局剩余、reveal actual/confidence、释放余额、关联 sequence；
- M/K、子决策索引、剩余次数、候选 ID/公开 cap、empty 概率、beta/同窗复用标识、feature/候选视图 hash、成本预测、b_sub、对偶 lambda、全部归一化信息、随机状态、选择、停止原因；
- 有序 pending IDs、owned_before、当前 transaction query、已收费/已打标签/已并入训练的独立状态；每次 choice 与 RNG 必须先于 reserve/reveal 落盘；恢复不能为已收费请求重抽或重新收费；
- common-reference approximation、theta/start/reference/actual hashes、旧槽、新包各两槽的完整记录与请求归属、g_q reduction、alpha、m、每请求标签、预测批量增益、参考/实际回报差与有限插入误差诊断。误差带 rollout 噪声，不能称精确 Taylor 残差；
- 逐包/状态/父任务/类别/折的 raw slots、权重后曝光、包含门控 a 的教师权重曝光、第一次/最后一次曝光、购入后尚未实际提交的包数、轮内合法可重放次数；
- reference/actual rollout 是否复用、学生 prompt/action/teacher tokens、source refresh、梯度/VJP/pilot 的 GPU 时间、峰值显存、失败/重试计算和锁等待。嵌套 compute scope 不重复累加。

窗口结束须无未决 reservation；实际 commit 与 recovery pointer 仍原子关联。若 ledger 比 recovery 超前，仅容许已落盘的当前选择事务，不能因为 pending 列表有多个包就接受任意未知 reveal。

## 5. 曝光选项、推荐剂量与算力代价

现有每轮 `8×12=96` 是加权完整动作槽预算；有购买时原始计算还要加新槽，不能当成 96 次总前向。即使 bank 每包只有一个行为，parent→state→teacher 的分层有放回采样也不会保证包级均匀。购买多包不自动获得相同曝光。

| 选项 | 具体含义 | 限制与代价 |
|---|---|---|
| 增加每步槽数，推荐 | S=40、12 步、K=20；新包每包两槽，旧池仍按原 parent→state→teacher 分布 | 每包在购入窗口获得已定义的直接曝光；不是终身等次数。完整动作梯度工作量约增至原同类满购买块的 5 倍，反馈不按 K 倍增加 |
| 增加每轮步数 | 例如 8 槽×60 步得到 480 槽 | 多了 48 次实际学生更新，优化路径和计算量都改变；若仍只有四个反馈窗口，需重新冻结窗口位置及剩余曝光。若同步增加反馈次数，环境预算也增加。不推荐本轮同时改 |
| §10.3 V-T 曝光校正 | 固定教师预算和学生 token/compute；先明确目标完整槽分布 p，再用提议分布 q 采样、按 p/q 校正完整序列损失 | q 可依赖合法已购长度/可用成本代理；必须记录 q、权重、有效样本量和 token 计数。权重裁剪有偏须另声明。不能改成 logprob/output_length，也不能保证每包至少曝光一次。适合后续 V-T 对照，不能冒称与 V-S 同一个资源轴 |
| 保留 8 槽并接受欠曝光 | K 个新包中仅采部分进入固定新块，明确配额与包含概率 | m=20 时八槽有放回均匀采样的期望不同包仅 `20*(1-(19/20)^8)≈6.73`；至少 12 包不能在该块各占一个不同槽。需记录 bought-but-unexposed、覆盖和曝光差异，不适合以“买了很多”为剂量结论 |

推荐方案每轮 raw 训练梯度槽上限为 `12×40 + 2*该轮购买包数 <=640`，全程 `1440+2*N <=1920`，还不包含 gate VJP 等额外计算；exact no-op 的跳过需据实扣除。非 no-op 满窗 m=20 时等效旧/新曝光为 30/10 槽，每包新曝光 0.5 槽；两个 raw 新槽各带 .25 权重。门控之后实际教师质量份额还要乘 a，不能把两次 raw teacher 评分称为两份单位教师训练量。

新银行中一个包一个行为，所以这种分配能使每个已购买包在成功提交窗口都被处理；多行为银行则只能保证包级而非所有行为覆盖。较早购买的包可在后续 replay 多次出现，fold 1 的包第三轮完全不能进入 inner 训练；**40 槽也不能保证跨时间、跨折、跨父任务的总曝光相等**。保持旧池采样分布，报告实际差异；不要为达到均匀覆盖而跨折取证。

显存使用以 `runtime.streamed_gradient()` / `streamed_gate_vjp()` 的逐完整动作计算为基础，40 是顺序累计槽数，不是同时送入 GPU 的 batch=40。保留当前 gradient checkpointing、`max_state_batch_size=2` 和内存策略；峰值不会机械地乘五，但新购买的完整 prompt、更大的 source cache/round state、I/O 和长序列仍可能增加峰值与耗时。不能通过截断完整 state 或修改反馈动作 cap 来迁就显存。

硬件与时间按用户给出的设备和 round-1/2 吞吐规划，不根据显存大小宣称速度倍数：

| 平台 | 设备与实施约束 | 原剂量三轮训练参考 | 推荐方案每臂三轮暂估，含三次官方评测 |
|---|---|---:|---:|
| rai | RTX 6000 Ada 48 GB；内存余量最小，保持流式计算 | `36/3≈12 h` | 32–50 h |
| rai | A100 80 GB；须等已有占用释放，不能动其他用户进程 | 先借用约 3 steps/h；没有独立新剂量测速 | 先按 32–50 h，不承诺比 Ada 快 |
| rai | RTX PRO 6000，用户标称约 98 GB；硬身份按实际 bytes/name | 同上，不能仅按容量推算 | 先按 32–50 h；建议作为 rai 三臂固定设备类，前提是可独占 |
| hpg | B200；同类卡可分配多个独立单 GPU 作业 | `36/5≈7.2 h` | 21–33 h |

估算展开：v1.0 基线训练+评测分别为 rai `12+3*(1.5–2)=16.5–18 h`、B200 `7.2+4.5–6=11.7–13.2 h`。若旧训练时间中随槽数增长的部分占 f，则基础放大因子约 `1+4f`；采用 **未实测假设 f=.25–.5**，得到 2–3 倍训练时间，再加 rai 3–8 h / B200 2–5 h 的新增状态、source/geometry 和更少反馈复用余量，得上述区间。f 接近 1 时，压力情景是 rai 约 69–74 h、B200 约 43–47 h/臂，且长序列/失败可再增加；这些都不是硬上界。

提高采购率还会减少 empty 时 reference/actual feedback 的复用，新增状态会在后轮扩大预条件器构建；所以不能简单声称“40 槽只把耗时乘五”，也不能直接沿用 3/5 steps/h。批准后的短 GPU 验证必须分别计量 source/geometry、slot gradient、feedback 和 VJP，只据此调整排队/墙钟申请，不看评测分数调剂量。没有新测量前不承诺小时级完工时间。

## 6. 不变的部分与完整实施清单

### 6.1 冻结不变项

- 任务仍是 BFCL `sealed_replay`，`teacher_access: text_only`，无在线调用；可用包、support/exclusions、父任务 hash、fold 0/1/0 和数据隔离不变，训练 seed 仍仅 0。
- R0 固定 a=.5；R1 `linear_sigmoid`；R1s `scalar_sigmoid`。CLI 中 R1s 仍是 `--arm R1` 加 scalar 配置和独立 run-dir，不增加第三种算法 arm 字符串。门控初始化、特征、学习率、ridge、running RMS 规则不变。
- 正权重 source/teacher mixture、完整 action+termination 概率、source policy 每轮冻结、两次 source sampling、train-only 标准化/预条件器及原 KL pilot 原则不变。
- `functional_step.FrozenStep` 的固定预条件单步、同起点 actual/reference、exact identity/no-op（包括跳过 weight decay）、`return_gradient.py` 完整任务 return estimator、leave-one-out baseline、gate VJP 算法不变；仅 runner 使用已声明的多包线性权重。
- 三轮、每轮 12 个已安排步骤、决策/元反馈步骤 `[1,4,7,10]` 不变。沿用当前 C25 可执行反馈剂量：单轮任务窗口最多 8 tasks×4 rollouts，多轮 4×2；不能退回最早 §14 的 4×2 当作此次改动。评测后不增轮、不按错误类型改规则。
- 学生 Qwen/Qwen3.5-4B、LoRA rank/alpha/targets、source generation/scoring、benchmark action caps、context cap、score tolerance、官方评测温度 0、完整 task coverage、checkpoint/hash 检查、硬件类约束均不变。

已存在的 rai_R0 旧 manifest 还写有 legacy `max_action_tokens:4096`，其他臂已有 per-benchmark 512/1024；v1.1 必须从同一份**当前可执行 C25 配置**分叉并核对有效值，不能照抄各自历史 manifest 造成新臂差异。旧 manifest 不回填。v1.0 与 v1.1 只作带版本和配置差异说明的历史对照。

### 6.2 推荐分支的精确配置键

未来新增 `configs/rtd/v1_1_bfcl.yaml` 和 `configs/rtd/v1_1_bfcl_scalar_gate.yaml`；现有 v1/c25/frozen YAML 全部保留。除下面列出的变化外，继承选定 C25 canonical 的有效值；scalar 文件只额外把 gate 设为已有的 `scalar_sigmoid`。

| 键 | v1.1 值/动作 | 语义 |
|---|---|---|
| `protocol_version` | `1.1.0` | 预算、子机会和批量曝光协议 |
| `method` | `rtd_v1_1` | 与旧方法轨迹分离 |
| `budget_basis` | `usable_recorded_output_tokens` | 固定可用成本和 55,370 |
| `budget_checkpoint_rounding`（新增） | `half_up_integer` | 25% 为 13,843 |
| `budget_checkpoints_bank_fraction` | 保留 `[0.10,0.25,0.50]` | 对新分母生效，按整数百分比计算 |
| `replay_bank_path` | `data/rtd/v1_1_bfcl` | 新 bank public/certificate 版本 |
| `replay_public_cap_output_tokens_by_class` | `{demo_attempt: 2048, generator_item: 512}` | 仅可用 BFCL 银行的类别 envelope |
| `replay_cap_scope` | `sealed_bank_class_content_envelope` | 不伪装成 archived/provider cap |
| `replay_cap_certificate`（新增） | `data/rtd/v1_1_bfcl/sealed/cap_certificate.json` | 特权审计文件，hash 写入 manifest |
| `max_new_packages_per_decision` | v1.1 删除且拒绝出现 | 旧单包窗口仅在旧版本解释 |
| `max_new_packages_per_window`（新增） | `20` | 所有新付费请求含依赖的数量上限 |
| `acquisition_subdecisions_per_window`（新增） | `40` | 有限机会，包括 empty |
| `empty_consumes`（新增） | `one_subdecision` | 次数不返还 |
| `window_token_quota`（新增） | `remaining_over_windows_with_public_cap_floor` | §3 的 R/W、c_max 修正 |
| `window_budget_carry`（新增） | `recompute_from_cumulative_remaining` | 窗口实际余额自然结转，不重置账本 |
| `acquisition_batch_mode`（新增） | `common_reference_fixed_capacity` | 一次 beta、共享参考，按位置插入 |
| `pending_slots_per_package`（新增） | `2` | 每新包两个完整来源槽 |
| `slots_per_step` | `40` | 本版本要求等于 K×pending_slots_per_package |
| `output_root` | `results/rtd_v1_1` | 新输出，明确指定 host/arm run-dir |

`hard_cost_reservation:true`、`replay_is_explicit_choice:true`、`replay_prior_mass:.5`、`insertion_fraction:.25`、`acquisition_value: exposure_conserving_insertion`、`acquisition_posterior_refresh:round` 等保持；`alpha=.25/K` 为由配置推导并写日志的有效参数，不再新增可独立调节的 alpha 键。新银行分母/hash、预算 ceilings、certificate hash、版本字段属于 manifest 产物，不允许通过额外 YAML 数字覆盖审计结果。选项 A 只替换 budget basis/rounding，保留旧 bank/caps，不与推荐 B 的证书配置混用。

### 6.3 精确源码函数清单（未来实施，不是本次修改）

| 文件 | 要改/新增的符号 | 必须达到的行为 |
|---|---|---|
| `src/bfas/rtd/caps.py` | `public_cap()`、`affordability()`；新增 `checkpoint_ceilings()`、`content_class_caps()`、`verify_cap_certificate()` | 显式按 protocol/cap policy 分支；旧 `PUBLIC_CLASS_CAPS` 和 1.0.1 默认不全局覆盖；新 helper 只在特权离线银行侧读成本；affordability 区分 cap 库容、actual 分母、折、窗口可行性 |
| `src/bfas/rtd/bank.py` | `build_bfcl_bank()` 增加版本化 cap policy；新增 `rebind_bfcl_bank_caps()` | 推荐从冻结旧 bank 复制/重绑公开 envelope 与新 certificate；保留所有 payload、request ID、confidence、排除和历史记录；禁止覆盖旧目录 |
| `src/bfas/rtd/ledger.py` | `Ledger.__init__()`、`remaining`、`reserve()`、`reserve_chain()`、`settle()`、`release()`、`resume()`；新增 `open_window()`、`close_window()`、`window_remaining`、`window_purchases` | 全局/窗口两个余额在同一锁内检查；全依赖链检查 K 和两层 cap 后再预留；恢复窗口状态；authorize 累计增加的语义保持。正常 reveal 记录/可重建释放的 cap-actual |
| `src/bfas/rtd/broker.py` | `SealedReplayBroker.list_candidates()`、`acquire()` | 展示与读取前双重校验窗口/K；先全局+窗口 reserve 再读取；acquire 已购包不重复计数；无窗口的 v1.0 路径保留 |
| `src/bfas/rtd/acquisition.py` | `budget_distribution()`、`AcquisitionPolicy.choose()`、`select_and_acquire()`；新增 `begin_window()` | 新版接受位置曝光 alpha、有限子机会与 frozen beta；b_sub 不能填整个余额；零价值 R0 同程序；返回完整选择元数据。`CostRegressor` 数学不变，runner 将更新时机移到已付 reveal 后 |
| `src/bfas/rtd/insertion.py` | `InsertionReference.__init__()`；新增 `insert_batch()`，`InsertionLabel` 增加有默认值的 approximation/window 字段 | 旧 `insert()` 的单包兼容路径保留；新实现 S=40、每包两槽、共享起点/参考、每 ID 一标签、总曝光权重与批量更新一致；不改 `insertion_derivative()` |
| `src/bfas/rtd/experiment.py` | `assert_run_invariants()`；`RTDExperiment.__init__()`、`slots`、`draw_slots()`、`round_start()`、`step_start()`、`reference()`、`selected()`、`revealed()`、`actual()`、`feedback_commit()`、`committed()`、`round_end()`；新增 `window_quota()`、`draw_pending_slots()` | 不再硬编码 slots=8/单 selected；有限子事务循环、pending、alpha 权重、一次实际提交、批量标签/成本观测、每包曝光和窗口停止原因、checkpoint schema。`draw_slots()` 接受显式 count，旧池采样分布不改 |
| `src/bfas/rtd/persistence.py` | `StateStore.load()` | 识别 v1.1 窗口/子事务 phase，并严格验证 ledger 超前仅属于落盘选择；`save()` 的 manifest binding/原子写机制复用 |
| `src/bfas/rtd/cli.py` | `load_config()`、`bank_audit()`、`make_manifest()`、`data_identity()`、`replay_ledger()`、`main()` | 按版本选择 canonical/严格键表，拒绝 v1.0 与 v1.1 混搭；校验新 cap/certificate；audit 显示两个分母和真实配额；新 manifest/schema；data hash 对 v1.1 额外绑定 certificate，旧版本 hash 算法不变；更新 CLI 文案。`resume_config()`、`run_command()`、`run_campaign()` 原流程复用并回归测试 |
| `src/bfas/rtd/evaluation.py` | **仅 `report()`** | 新/旧 schema 分开读；增加 basis、实际成本/ceiling、配额停止原因、包/类别/折曝光、R1s gate 标签、近似标识和同硬件分组；保留 artifact/checkpoint 验证，不修改 `evaluate()` 或评分逻辑 |

`runtime.py` 的 streamed 梯度/VJP、`functional_step.py`、`return_gradient.py`、`transport.py`、门控/特征定义、`identity.py`、`scoring_scope.py`、`hardware.py`、`evaluation_lock.py`、BFCL adapter/checker/campaign/hub export **不需要修改算法或检查规则**。若未来实现发现非上述必要变更，应先更新本清单与版本说明，不能顺带修其他运行中的模块。

启动器 `scripts/rtd_run_hpg.slurm` 已支持 `RTD_CONFIG`、`RTD_ARM`、`RTD_RUN_DIR`、`RTD_COMMAND`、`RTD_EXTRA_ARGS`，不必改源脚本；scalar 仍传 `RTD_ARM=R1`。资源/时限可在提交参数中设置。未来同时补充治理文档 §5–6/§9.2/§14 的版本附录及 v1.1 protocol 文档，保留旧规范原文。本次不改这些文件。

### 6.4 批准后要添加的测试与验收

测试在隔离临时 fixture/新测试目录执行，不以生产银行重放训练，不运行会对活跃账本调用 `Ledger.resume()` 的诊断命令。该方法会修复 torn tail，不能当只读检查用。

1. `tests/test_rtd_budget_v11.py`（新增）：55,370 的 half-up 三点与累计增量；A 首轮零候选/两类 cap 不变；B 全量 certificate/hash/max 上界；均值/P95/超 cap/不同银行证书拒绝；审计分母变化不能改变 sealed payload 或把 hidden cost 添入 public features。
2. `tests/test_rtd_broker.py`（扩展）：全局与窗口余额都要 reserve、settle 后释放再买；cap>窗口/全局时在 payload read 前拒绝；库存/K/依赖链原子检查；重复 acquisition、零费用包、跨折、超 cap、损坏 payload；多笔并发预留无超支。断言 `S+H<=C` 和窗口同构不变量贯穿事件序列。
3. `tests/test_rtd_acquisition.py`（扩展）：40 子机会包括 empty、K=20、无放回；fixed beta 同窗复用；R0 与零 value R1 同成本/先验程序；b_sub 的 KKT/归一化；真实成本只在 reveal 后可拟合；成本冷启动导致的 empty 如实出现；禁止选中后免费重抽。
4. `tests/test_rtd_batch_insertion_v11.py`（新增）：小模型有限差分验证批量 epsilon=0 导数 `sum(v_q)/K`；m=0 identity、m=K 的 .75/.25、m<K 旧数据填位；两槽平均、请求级标签、单一实际 commit；runner 加权 streamed gate VJP 与完整 autograd 对照；故意换 reference/起点、重复标签必须失败。有限 .25 误差单独测试/报告，不要求它等于一阶预测。
5. `tests/test_rtd_checks.py`（扩展）：三轮/36 步/12 外层窗保持；slots=40 确实生效（当前 getter 硬编码 8）；每窗最多 20 个新付费包和 40 次选择；按 m 计算正确 raw/weighted exposure，旧池 empty/no-op 不伪造教师曝光；每包两槽、state/parent/request coverage、fold 1 第三轮隔离；每个子事务在 choice/reserve/reveal/label/commit 前后注入崩溃，resume 等价且不重复扣费/抽样/更新。
6. `tests/test_rtd_manifest_tolerance.py`、`test_rtd_checks.py`（扩展）：1.0.1 canonical/旧 config hash/data hash 不漂移，1.1.0 不接受旧单包键或错证书；新 config、manifest、恢复 state 的版本绑定；新版 report 正确区分 R1/R1s、按 actual spend 画点、旧结果可读，缺评测仍显示 missing。
7. `tests/test_rtd_harness_identity.py`、`test_rtd_identity_update.py`、`test_rtd_evaluation_identity_reuse.py`、`test_rtd_hardware_identity.py`（扩展/回归）：仅上述训练源码/`report()` 变化不改 scoring harness hash；确实修改评分内容应拒绝；v1.1 新 checkpoint 不复用 v1.0 score；不同 hardware class 拒绝 resume，汇总标注不匹配；旧审计链继续可读。复跑已有 functional-step、return-gradient、transport、score-consistency、evaluation resume/lock 的相关回归。

通过 CPU 语义/恢复测试后，在当前 v1.0 全部完成的前提下做小型独立 GPU 预检：同一起点的小窗口覆盖 m=0/1/20、长 state 内存、真实 feedback 路径，核验日志和哈希。该预检只用于正确性/资源定额，不能看官方分数选 K、cap、门控或预算。预检产物与正式 seed-0 六条轨迹分开，不复用其 posterior、已购集合或模型。

## 7. 评测 identity、harness 与版本迁移

**预期没有 evaluation harness 语义变化，也不需要 update-identity。** `identity.evaluation_harness_identity()` 绑定 BFCL checkout/data 内容、`EVALUATION_TOOLS` 的 scoring projection，以及所有 `evaluation_*` 配置；预算、K、槽数、采购函数不在该配置投影中。`scoring_scope.PYTHON_SCOPES['src/bfas/rtd/evaluation.py']` 与额外 evaluate AST 投影不包含 `report()`。因此报告新增字段也不应改变评分 hash；须以测试验证，不能扩大投影排除范围来强行使它相同。

“harness 不变”不等于“评测结果 identity 不变”。`evaluation.evaluate()` 的 campaign identity 还绑定 checkpoint metadata、config_hash、data_hash、base/tokenizer、hardware、expected task coverage；v1.1 的配置、公银行 cap/certificate 和学生 checkpoint 都是新的，**应该产生新的 campaign tag 并逐轮重评**。不能因相同 harness 就复用 v1.0 分数。新协议的 resume 只恢复新协议同一 immutable manifest；`--acknowledge-code-drift` 不是跨版本训练或解除 config/data/hardware binding 的通行证。

未来版本产物固定为：

- 方法/配置 `1.1.0`，manifest `version: rtd-v1.1.0-run`；新增 `trajectory_schema_version:2`、`ledger_schema_version:2`、`acquisition_protocol:quota_batch_v1` 与完整有效配置；
- 新 bank audit/certificate 版本 `rtd-v1.1.0-bfcl-bank`，含旧 bank 来源 hash、新 public/integrity/certificate hash、recorded/exact/estimated 成本、cap-policy 和 rounding；
- `results/rtd_v1_1/{rai_R0,rai_R1,rai_R1s,hpg_R0,hpg_R1,hpg_R1s}` 为计划目录；实际硬件类写 manifest，不能从目录名推断；每条轨迹都有空的初始账本和同一初始 base/LoRA 状态；
- `rtd_source=source_identity()` 自然改变；新训练从冻结 v1.1 实现启动。保留现有 `bfcl-evaluation-harness-scoring-v4` 和 hardware identity 格式，不制造无必要迁移。

## 8. 执行顺序与墙钟排期

本节是批准后的计划，不是本次提交、取消、同步或占用资源的授权执行。开始屏障 T0：当前 rai 与 hpg 的 v1.0 R0/R1/R1s（存在的各臂）**第三轮训练及其预定官方评测完成**，远端 manifest/trajectory/teacher 原件已收齐；仍排队或暂停的 v1.0 臂不算完成。若某臂最终失败，明确标注不完整并由用户决定是否解除屏障，不能悄悄改称完成。

1. 用户批准本文的 cap 信息例外、子机会/批量曝光定义与资源范围后，冻结 v1.1 配置和分析主点。实现与 CPU 测试预计约 2–3 个工作日，可在 T0 前使用独立副本推进，不能使活跃 v1.0 作业的代码/配置发生漂移。
2. T0 后才部署/同步冻结实现，创建新 bank、比对证书/输入/harness、完成 GPU 预检及 manifest 预检，预留约 3–6 h，不含尚未完成的实现时间。正式六臂从初始学生重跑三轮，不能从 v1.0 round 3 接续。
3. **两站点同时推进；每条臂内部三轮顺序运行，每轮结束做同一官方评测。** 一个臂的 round 2/3 不越过本臂评测屏障，不使用这些分数调参。

硬件公平性决定实际并行度。rai 现有 R0/Ada、R1/A100、R1s/RTX PRO 是三种 hardware class；将它们原样并行后称为同硬件比较，违反 `same_hardware_required:true`，现有 `evaluation.report()` 也会标为 mismatch。推荐的**六条正式轨迹排期**是：

| 站点 | 排队/执行安排 | 三轮端到端耗时，未含队列/外部争用 |
|---|---|---:|
| rai | 固定一个可独占的 GPU 类（建议 RTX PRO；若不可用则启动前统一选 Ada 或 A100），R0→R1→R1s 三条臂串行；每臂训练与评测均留在该类 | 每臂 32–50 h，总 96–150 h |
| hpg | 三个同软件/驱动/显存硬类的 B200 作业，R0/R1/R1s 同时提交并尽可能并行；R1s 用 scalar config | 三卡同时获配约 21–33 h；两卡需两波约 42–66 h；仅一卡串行约 63–99 h |

因此六臂从正式开始到全部完成的常规规划为 **约 4–6.3 天**，另加 3–6 h 准备及排队、评测锁等待和故障恢复；总 GPU 占用约 rai 96–150 + hpg 63–99 = **159–249 GPU·h**。共 18 个新官方评测，按每次 1.5–2 h 是 27–36 GPU·h，已含在表内，不重复相加。压力情景可使 rai 三臂到约 9 天，不能把常规区间当保证。

如果要求 rai 三张异类卡也同时跑，六臂第一波可缩至约 32–50 h，但 rai 结果只能作为 hardware-unmatched 的探索记录，不能解除原身份约束或并入 HPG matched 比较。要利用三卡同时得到 rai 同类完整三臂，需三波轮换，让每张卡各完成 R0/R1/R1s，即 rai 共九条轨迹；这增加实验数量与 GPU·h，**不是上述六轨迹默认授权范围**。同硬件、六轨迹、rai 三种卡同时各跑一臂，这三个条件不能同时满足。

HPG 现有 `scripts/rtd_run_hpg.slurm` 的默认 walltime 为 8 h，不能假设足够一次完成新剂量三轮。先依据预检与实际队列允许时限设置提交参数；若只能分段，使用同 manifest 的现有 checkpoint/resume，明确重提作业依赖和重复计算账单。不得在不同 GPU 类之间迁移来绕过等待。评测锁、端口、Hub merge/export 沿用既有隔离路径；不并发占同一 GPU 跑训练和评测，不触碰他人的进程。

## 9. 风险、报告口径与批准项

1. **上界与研究信息泄露。** 类别内容 cap 是特权银行统计的公开例外。若用户不批准，选项 A 仍合法但 10% 无法采购，应该如实交付不可行结论；没有“既不改信息约定又假装小 cap 来自请求配置”的第三条捷径。
2. **余额仍可能不绑定。** conservative cap tail、40 个子机会的成本冷启动、empty、K 和折限制都可能留下余额。逐窗记录 constraint binding/stop reason；不得运行后降低 cap、取消 empty 或加轮凑满预算。50% 是最大授权，不是测量到了 50% 剂量。
3. **银行耗尽。** 推荐 K=20 时最多 240/420 包，整池至少剩 180 包，故“第三轮必耗尽整池”不成立；fold 0 最多买 160/309、fold 1 最多买 80/111。真正可能耗尽的是某类别、某父任务、当前折的可支付子集：fold 1 只有 69 个 generator，42 个 demo 也可能被优先买完。可行集合为空后只继续训练，不跨折补货、不复制请求 ID、不开在线教师。
4. **类别与折不平衡。** 全局 20%/80% 不代表每折；fold 1 的 demo 数量占比 37.8%，成本明显高于 fold 0。类 cap 和剩余余额还会造成尾端 generator 优先。报告每轮可选/已购/未购类别数量、成本、父任务覆盖、cap exclusion；R0/R1/R1s 不加入额外类别配额来掩盖差异。
5. **高预算趋同。** 50% token 若主要花在 generator，按均值相当于约 272 个 generator，约全池 65%/该类 81%；推荐 K 会把总包数压至最多 240（57.1%）。即便达不到上述数字，多个包共享父任务和相似证据，覆盖也可能很早饱和。R1 与 R0 在顶部趋同是合理结果，不能单凭该点否定低预算选择。报告集合/父任务/状态重叠、曝光和完整曲线；10%/25% 两点预注册为主要结果，50% 为次要饱和点，不事后只挑最高分。
6. **预算轴同时带训练时间与折变化。** 三个点分别在 12/24/36 步、不同 fold 历史后评测，不是同一训练 checkpoint 上独立改变预算的纯因果曲线。同一轮跨臂可比较固定授权规则；实际 spend 不同要并列展示。只在共同实际成本区间做描述性连线/插值，不能把未花到的 50% 外推成实测结果。
7. **曝光与批量近似。** 每包两槽可以保证本银行购买后的初次处理，不能保证充分学习或长期同剂量；共同 g_J、有限 .25、两槽 g_q 噪声和 posterior 观测相关可能削弱采购信号。按真实位置份额使用 alpha=.25/20、同时保持 tau=1，也会减小相对 entropy 的价值差异，R1 可能仍接近随机；不能为了制造差距事后把 alpha 伪写回 .25。报告按包累计教师权重、有效覆盖、选择分布和近似误差。R0 同样计算/记录插入诊断并走同样反馈流程，采购不使用该价值，避免以省略诊断降低它的计算账户。
8. **历史成本不完整与单 seed。** 本实验仍是 estimated/exact 混合成本的 sealed replay，不能解释为节省完整真实教师账单；新教师成本仍为零。seed=0 的三臂/硬件块不是多 seed 重复，不能宣称统计显著性。仅胜过 R0 也不能替代 §10.3 的强对照；现有强基线安排不由本文改写。

批准应针对以下具体组合：B 的类别统计公开例外与 2,048/512 cap；55,370 分母及 5,537/13,843/27,685 ceiling；Q 的动态定义与公开 cap floor；K=20/M=40 的有限子机会；S=40/每包两槽/空位置回给旧数据的批量一阶目标；seed-0 六轨迹、站点并行但 rai 同类卡三臂串行的资源排期；10%/25% 为主要分析点。**本文不构成已经批准或已经执行这些变更的记录。**
