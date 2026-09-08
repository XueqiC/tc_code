
Capability-Residual Correction Distillation
面向预算受限黑盒 Teacher 的 Agent 能力蒸馏：新 Capability Atom、端到端方法与实验计划
Research Design Document · v6 · 2026-09-02
核心一句话
Existing distillation imitates what the teacher does. We distill only what the student needs to change: 在学生实际到达的状态上，定位能改变成败的局部行为差分，用 capability residual 决定是否值得学、学多强，并仅在预测会伤害已掌握能力时施加保护。

本文档是新的方法规格，不是 v5 的增量补丁。Acquisition、生成与审计均围绕“如何产生更高价值的蒸馏事件”服务；论文主贡献应落在蒸馏机制本身。

0. 执行摘要
本方法暂称 Capability-Residual Correction Distillation（CRCD，工作名）。它以 Differential Capability Atom（DCA，差分能力原子）作为学生侧的能力坐标，以局部、可验证的 fail→correct intervention 作为基本蒸馏事件，并把黑盒 teacher 的调用从“生成完整专家轨迹”改造成“在学生真正犯错的状态上提供最小必要纠正”。
方法的核心不是一个新的样本权重，而是四个相互咬合的机制：
蒸馏对象改变：从“复制 teacher trajectory”变为“蒸馏 student error → verified correction 的行为差分”。
蒸馏坐标改变：用 student-specific Differential Capability Atoms 表示一次纠正到底在补什么能力。
蒸馏强度改变：由 intervention utility × capability residual × teacher reliability × supervision-density mask 决定。
保护方式改变：只有当一次纠正预计会沿 signed capability graph 伤害已掌握 atoms 时，才启动 self-anchor / KL / 投影等保护。
由此，完整系统成为闭环：行为探测 → DCA 发现与校验 → capability ledger → 选择最有价值的 supervision event → 局部差分蒸馏 → 行为探针与账本刷新 → teacher 按 atom 成熟度逐步退场。
1. 问题设定与研究目标
给定任务分布 D、只有 query 的 support 集 S、小学生策略 πθ0、按输出 token 计费且只提供文本/行为的黑盒 teacher πφ*，在 teacher 预算 B 内改造学生，使其在与 support 不相交的 query 上最大化成功率，同时控制对 base 强能力的损伤。每次 teacher 调用（包括失败）均计费。
max   E_{x~D_test}[Success(πθ,x)]    s.t.  TeacherOutputTokens ≤ B,  Damage(πθ,πθ0) ≤ ε
论文定位应明确为“budgeted black-box agent distillation”。Data acquisition 是 supervision policy；capability atom 是 distillation coordinate；最终指标是 teacher-token efficiency、target gain 与 retention，而不是单纯的数据选择准确率。
2. 与 2025–2026 SOTA 的关系：我们真正要占的位置
工作
核心启发
我们的区别/使用方式
SCoRe / Student-Centered Distillation [1]
学生先走，teacher 纠正 earliest error；后接 short-horizon RL。
已有 first-error correction，但没有 capability residual、signed transfer 与“只蒸馏纠正差分”的统一目标。
SOD [2]
错误工具调用后 teacher signal 会随 divergence 级联失真；逐步重权。
支持 step reliability，但其核心坐标仍是 step divergence，不知道“缺什么能力”。
TurnOPD [3]
full-horizon OPD 尾轮低信号，浅层 token 吃掉 KL；做 turn-aware budget。
支持 sparse supervision，但未按 capability residual 控制梯度密度。
Unmasking OPD [4]
用 targeted rollout 估 ideal per-node success gradient；错误 rollout 上 teacher signal 更有用。
提供 utility-gradient 诊断；我们把 utility 思想变成训练事件与 atom 定义。
OPD² [5]
蒸馏 teacher 与 teacher-base 的 log-prob delta，而非 teacher 本身。
需 teacher/base logits；我们的差分是 black-box、student-state-conditioned 的 fail→correct 行为差分，必须主动区分。
ROPD [6]
从 teacher-student 对比诱导 rubric，黑盒兼容并做 on-policy optimization。
可作为无强 verifier 时的奖励层；我们更强调可复用 capability residual 与局部 intervention。
LOPD [7]
让 self-teacher 的 privileged context 从经验中端到端学习。
提示 teacher 应随成熟能力退场；可作为后续 self-teacher 方案。
STAT [8]
teacher 语义技能表 + Missing-Skill-Profile，用于重加权/合成。
我们的 skill/atom 不由 teacher 命名，而由 student-specific correction update 发现，并直接进入蒸馏。
RODS [9]
奖励方差识别动态 capability boundary，并在线合成结构匹配任务。
支持动态账本与 boundary-aware generation；我们把 boundary 分解到 atoms。
OGS [10]
把梯度正交性从 optimizer 移到 data selection，降低遗忘。
支持训练前冲突过滤；我们进一步估计 signed atom transfer 并按风险选择保护剂量。
PROOF-Gen [11]
teacher 失败 trace + evaluator feedback 可通过 scenario-specific reflection 大量修复。
说明 teacher fail 不等于不可教；应有 direct / repairable / unresolved 三级 teachability。
COVERT [12]
oracle-preserving distractor/ambiguity/noisy-output augmentation 提升 tool robustness。
可作为 robustness gap 的低成本 supervision，而不是一般示范。
Skill Self-Play [13]
动态 skill controller 条件化生成，skill library 与 solver 共演化。
支持 skill/atom-conditioned generation 与能力边界共同演化。
GiGPO [14]
共享状态形成 step-level group，做细粒度 advantage；ALFWorld/WebShop 有明显收益。
支持 turn/decision-level credit，适合估 utility-weighted DCA。

3. 新 Capability Atom：Differential Capability Atom（DCA）
3.1 定义哲学：atom 不是“真正技能”，而是可操作的学生适应坐标
不再声称“学生会什么只能由梯度看出来”。新的定义采用 operational validity：一个 atom 是反复出现的、student-specific 的 utility-improving adaptation direction；它只有在能够稳定预测跨样本迁移、行为改善与干扰时，才被纳入账本。
建议 Definition
A Differential Capability Atom is a recurrent, student-specific direction of utility-improving behavioral correction, discovered from behaviorally credited local interventions and validated by its ability to predict transfer and interference across tasks.

3.2 基本数据单位：不是整条 trajectory，而是局部 matched intervention
对 student rollout 中一个候选关键状态 h_t，保留学生实际动作 a_t^-；teacher 或可验证纠正机制给出 a_t^+。若替换后 continuation 的成功概率显著提高，则形成一个有效蒸馏事件 e=(h_t,a_t^-,a_t^+,ΔU_t)。
ΔU_t = P(success | do(a_t^+), h_t) − P(success | do(a_t^-), h_t)
事件的差分指纹使用同一 prefix 下的正确/错误动作更新差，而不是正确轨迹的绝对梯度：
z_t = Normalize( RP( AdamWhite( ΔU_t · [∇ log πθ(a_t^+|h_t) − ∇ log πθ(a_t^-|h_t)] ) ) )
这会抵消共享 prompt、格式、API schema 和长前缀带来的大量共同成分，更接近“从错变对所需的局部参数变化”。若 a^- 不可明确定位，可退化为 utility-weighted positive gradient；若没有可靠 counterfactual，则该事件不用于发现核心 atoms，只可作为普通训练数据。
3.3 Multi-resolution：turn × layer/block，而不是单一扁平向量
长 agent trajectory 往往同时包含 trigger、执行、状态跟踪、组合、termination 等能力。一个轨迹一个向量会把多种变化糊在一起。推荐对关键 turn 按参数 block 计算/压缩，保留 layer profile，并用 group-sparse dictionary 得到 soft loading。
z_{t,l} ≈ Σ_a α_{t,a} · d_{a,l},    with sparse signed α and unit-norm atoms
工程上可先选少量代表 block（例如 attention output、MLP output 或 LoRA-adapted modules）以控制反传成本；随机投影维度应通过稳定性曲线选择，而不是固定写死。
3.4 Atom 对象与弱类型
每个 atom 不只是向量 d_a，而是一个带元数据的对象 A_a=(d_a, layer-profile, role, reliability, transfer-row)。role 建议只作为弱标签，用于解释与生成，不参与强制发现。
字段
含义
推荐估计
d_a
差分 update-space 主方向
signed sparse coding / online dictionary
layer profile
能力变化主要落在哪些 block
各 block loading 与谱统计
role
trigger / execute / compose / boundary-verify
行为模式 + 少量 teacher/规则弱标注
q_a
atom 的 construct validity
bootstrap 稳定性 × transfer 预测 × intervention 选择性
M_:a
学习 a 对其他 atoms 的 signed transfer
gradient prior + 小规模行为干预校准

3.5 Atom 的三重准入标准
Stability：更换 support bootstrap、随机投影 seed、邻近 checkpoint 后仍能匹配到相近 atom；不稳定 factor 不进入核心账本。
Predictive transfer：一个 correction 对 atom a 的 loading 越高，给相似 atom 题带来的 post-update gain 应越大。
Interventional validity：对高 loading correction 做局部更新，应优先改变对应行为，而不是无选择地改变所有任务。
建议把“解释性”降为次要指标。JOIN、tool schema 等人可读现象可以做 qualitative evidence，但论文的核心 validity 必须是稳定性、迁移预测和干预选择性。
4. Signed Capability Graph 与动态账本
4.1 从“供给非负”升级为 signed transfer
v5 把数据对目标能力的贡献 s_a≥0，与干扰 I(x) 分成两套量。新版本直接学习 signed transfer matrix M：M_ba 表示一次主要学习 atom a 的更新，对 evaluation atom b 的期望行为变化。允许正迁移、无关和负迁移。
M_ba = E[ Δ Success_b | update dominated by atom a ]
梯度 cosine 只做低成本先验，真正的 M 应用少量可验证 micro-update / checkpoint delta 做校准。这样“干扰”不是额外理论，而是 capability graph 的负边。
4.2 账本不再只有一个 δ：至少区分三类 gap
账本状态
定义直觉
主要蒸馏方式
Frontier gap
大 K rollout 也几乎不会；真正缺能力
full demo / repaired demo / synthetic teaching task
Access gap
pass@K 显著高于 pass@1；会但不稳定/概率质量没集中
local correction / preference / on-policy self-distillation
Robustness gap
clean 会，但 distractor、歧义、脏输出、silent/boundary 条件下失败
oracle-preserving augmentation + boundary negative pairs
Composition gap
单 atoms 会，组合或 junction 失败
composition-junction correction / multi-atom blueprint

L_a = (δ_a^front, δ_a^access, δ_a^robust, δ_a^comp, T_a^direct, T_a^repair, uncertainty_a)
因此 p≈0 不再自动意味着“买完整示范”，p≈0.5 也不自动意味着“最值钱”。关键是 evaluation mass × reducibility × teacher solvability × student uptake。
5. 蒸馏核心：Capability-Residual Correction Distillation（CRCD）
5.1 核心原则一：Distill the correction, not the trajectory
学生先在部署路径上 rollout。对失败轨迹只定位最早“真正改变后续成败”的 consequential divergence。Teacher 的默认职责不是重写整条轨迹，而是在该 student-visited state 上给最小纠正动作或 criterion。随后由学生自己 continuation；只有纠正被 verifier 证实能提升结果，才成为蒸馏事件。
“earliest error”与“earliest consequential error”要区分：表面第一个不一致不一定影响成败。推荐两阶段定位：先用规则/teacher/trajectory diff 提候选，再用替换动作后的短 continuation 做 causal validation。
5.2 局部双目标：decision boundary + exact execution
纯 CE 容易复制 teacher style，纯 preference 又可能学不准 JSON、参数与调用格式。因此推荐局部 preference + corrected-span CE。Preference 使用 base-reference centering 可减少无关概率漂移。
L_pref = −log σ( β[(logπθ(a+)−logπθ(a−)) − (logπ0(a+)−logπ0(a−))] )
L_local = − Σ_{token ∈ corrected span} log πθ(token | h_t, previous corrected tokens)
L_corr = L_pref + η · L_local
对于只需要“不调用/停止/追问”的 boundary 行为，preference 比完整 CE 更重要；对于复杂参数填写，local CE 权重可以更高。η 可按 atom role 自适应，而不是一个全局常数。
5.3 核心原则二：Distill only missing capabilities
每个蒸馏事件 e 有 soft atom loading α_e。它是否值得学，不由题级 (1−p̂) 决定，而由它落到的能力是否仍有 residual 决定。
R_e = Σ_a α_ea · δ_a(relevant gap type)
如果 teacher 给了与学生不同的动作，但该 correction 只落在已饱和 atom 上，则 R_e≈0，不应因为“teacher 更强”就强制模仿。这里把 teacher authority 改造成 student-need-conditioned supervision。
5.4 核心原则三：只在 teacher signal 有正效用且可靠时反传
最终事件权重建议由四项组成：
w_e,t = U_e × R_e × Q_e,t × m_e,t
因子
含义
可实现估计
U_e
intervention utility：纠正是否真的改变成功概率
K 个短 continuation 的 Beta 后验差；廉价版用单次 verifier + confidence
R_e
capability residual：补的是不是仍欠的能力
atom loading × 当前 ledger gap
Q_e,t
teacher reliability：这个 turn 的 teacher signal 还能不能信
修复验证、prefix divergence、离关键分叉距离、teacher consistency
m_e,t
supervision density mask：哪些 token/turn 真值得产生梯度
critical span + recovery window；行为重新稳定后归零

L_CRCD = Σ_e Σ_t w_e,t · L_corr(e,t) + λ_e · L_retain(e)
5.5 Sparse Capability Distillation：把 teacher-token budget 进一步变成 gradient budget
近期结果表明 dense on-policy self-distillation 也可能造成更大 parameter/response drift [15]；TurnOPD 也显示 full-horizon 训练会把监督浪费在低价值尾轮 [3]。因此 teacher 说出的 token 不等于都应该反传。默认只训练关键 correction span 与必要 recovery window。
m_t = 1 at critical turn;  γ^(t−t*) in verified recovery window;  0 otherwise
当后续状态已经重新回到学生可自解区域时，teacher supervision 应停止，后续成功轨迹可转为 self-anchor，而不是继续 CE teacher 的风格。
5.6 Risk-constrained retention：Protect only at-risk capabilities
不再对所有训练样本统一加 λ retention。先用 signed graph 预测一次 correction 对 mastered atoms 的负迁移风险：
H_e = Σ_{b∈mastered} [ −(M · α_e)_b ]_+
λ_e = f(H_e)
H_e≈0 时不加保护；H_e 高时才启用 self-distillation anchor、base-KL、样本冲突过滤，必要时再做子空间投影。优先级推荐：训练前过滤/降权 → selective anchor/KL → optimizer-time projection。
5.7 Atom maturity：外部 teacher 应逐 atom 退场
同一训练过程中不同 atoms 可处在不同成熟阶段。监督形式不是固定 epoch schedule，而是按 ledger 路由：
状态
默认 supervision
teacher 角色
Frontier 高
full/repaired demo 或 teaching task
知识/策略来源
Access 高
local consequential correction
纠错器
Robustness 高
paired augmentation / boundary negative
边界定义者或无需 teacher
已掌握
verified self rollout / self-teacher
尽量退场
稳定饱和
no distillation
不再花预算

5.8 Composition Distillation：蒸馏 capability transition
当单个 atoms 已会而多轮组合仍失败时，不应该继续重复基础示范。维护 composition gap，定位“junction state”——例如 state tracking 已完成但下一工具选择错误——只蒸馏从 capability a 到 b 的过渡。
δ_ab^comp high  ⇒  train only at junction states where a is satisfied but b fails
这是 agent 场景区别于普通单轮 KD 的潜在强贡献，建议在核心机制稳定后作为第二阶段增强。
6. End-to-End 方法：从 support 到最终证书
下面给出完整闭环。每一步都列出可选实现与推荐默认项；论文主算法可用“推荐默认”形成一条干净主线，其他选项用于消融或 benchmark-specific adaptation。
步骤
可选实现/含义
推荐默认
Step 0 — Harness 与服务同一性
确认 support probe、teacher 生成、训练数据渲染、测试路径与官方 harness 完全一致；任何 prompt/template/tool schema 差异会让账本失真。
必须先完成。
Step 1 — Student behavioral probing
对 support 做 K-rollout 或 sequential Beta probing；记录 pass@1/pass@K、失败轨迹、关键状态、类别/轮深/成本。
默认自适应 K：决策区间稳定即停；强制保留少量大 K 样本用于 frontier/access 分解。
Step 2 — 候选 consequential divergence 定位
规则 diff、环境错误、teacher critique、trajectory alignment 均可提候选；再用 replacement + continuation 判断是否真影响结果。
推荐“廉价候选 + 小 K causal continuation”。
Step 3 — 获取最小纠正
teacher 给 action/span；若 direct fail，使用 trace+verifier feedback 做 reflective repair；若仍失败，标 unresolved-under-budget。
默认 local correction；仅 frontier gap 很高时允许 full demo。
Step 4 — DCA 指纹与 atom discovery
raw gradient / normalized gradient / differential gradient / utility-weighted differential gradient；hard cluster / sparse dictionary / structured dictionary。
推荐 utility-weighted, Adam-whitened, normalized differential turn-level fingerprint + signed sparse dictionary。
Step 5 — Atom validity 与 graph calibration
bootstrap matching、held-out transfer、micro-update intervention；用 gradient conflict 作 M 的先验，行为 probe 校准 signed transfer。
达不到 validity gate 的 atom 降级为普通 feature，不用于强路由。
Step 6 — 建立动态 ledger
对每 atom 记 frontier/access/robustness/composition gap、teacher direct/repair solvability、uncertainty、mastery。
每个行为 probe checkpoint 刷新；不建议每 step 重建 dictionary。
Step 7 — 选择 query × supervision mode
比较 full demo、local correction、repair、self-anchor、synthesis、augmentation 的预期 residual reduction / teacher token / risk。
MVP 用规则路由；完整版用 UCB/Thompson 估 value-of-supervision。
Step 8 — 形成蒸馏事件池
事件带 h, a−, a+, corrected span, ΔU, atom loading, ledger snapshot, teacher reliability, risk。
训练数据单位从 trajectory row 改为 intervention event；trajectory 只是上下文容器。
Step 9 — CRCD 训练
局部 pref + local CE；乘 U×R×Q×m；高风险事件才加 retention。
先做固定 batch 训练，再升级在线刷新；避免一次引入太多 moving parts。
Step 10 — 行为探针与账本刷新
每 N update 做小型 deployment-path probe；保存最好 checkpoint；更新 residual、risk calibration 与 teacher routing。
N 由行为变化速度决定；BFCL MVP 可每 0.25–0.5 epoch。
Step 11 — 动态生成/增强
对仍欠且可教的 atoms 生成；frontier 生成 teaching task，robustness 做 paired oracle-preserving augment，composition 做 multi-atom blueprint。
先 blueprint/checker，再 surface dialogue；生成题必须有可执行验证。
Step 12 — 停止与证书
当所有可教 residual 的单位成本价值≤0，或预算耗尽即停；按类别/atom 对 base 做 Δ 与置信区间。
预算是上界；发布 gate 需保证 base 强区不出现超噪声掉点。

7. Supervision Policy：每一步“可以买什么”
把 acquisition 写成蒸馏监督策略，而不是单纯挑 query。决策对象是 (x,m)，其中 m 是 supervision mode。
m ∈ {self, local-correction, full-demo, repair-demo, synthesis, robustness-augmentation, composition-task}
V(x,m) = [ E(residual reduction via M·α) + ξ·InformationGain − RiskPenalty ] / E(teacher output tokens)
MVP 不需要直接学习 V。可先用可解释规则：frontier→demo/repair；access→local correction；robustness→augmentation；composition→junction task；mastered→self/no-distill。完整版再引入 UCB/Thompson，用少量探索预算学习 teacher uptake 与 correction value。
8. 关键设计选择与推荐决策
组件
候选
推荐
理由
Atom 指纹
整轨迹绝对梯度；turn 梯度；normalized；correct−wrong differential；utility-weighted differential
utility-weighted differential turn fingerprint
最接近“从错变对”的局部适应，减少共享前缀与长度偏差
Atom 粒度
query / trajectory / turn / token
turn + corrected span；必要时 block-level
agent 错误通常发生在决策轮；token 太细成本高
Factorization
k-means / spherical k-means / sparse dictionary / mixture encoder
signed sparse dictionary，后续可尝试 learned encoder
保留 soft loading 与可组合性
Critical error
teacher 指第一个错；规则第一个错；causal replace-test
候选定位 + causal continuation
避免把无关差异误当关键能力
Distill loss
full CE；local CE；pairwise pref；local CE+pref；rubric-RL
local CE + base-centered pairwise preference
既学决策边界又学精确工具格式
权重
1−p；cluster fail；step divergence；U×R×Q×m
U×R×Q×m
分别对应有用、缺口、可信、密度
Retention
全局 λ；self-anchor；sample filter；projection
risk-triggered anchor + conflict filter
先用低成本方法；高风险才做 optimizer surgery
Teacher fail
discard；重试；reflective repair
reflective repair 后再标 unresolved
PROOF-Gen 表明 near-miss 失败常可恢复
动态性
atoms 固定；ledger 动态；atoms+ledger 都动态
ledger 每 probe 动态；atoms 低频刷新
避免训练非平稳性过强
无强 verifier
teacher judge；rubric；learned PRM
ROPD-style rubric 作为可插拔后备
不让主方法依赖一个额外 reward model

9. 最小可实现版本（MVP）与完整版
9.1 MVP：先证明蒸馏机制，而不是一次实现所有图结构
只在 BFCL 做：student rollout → earliest consequential correction → local CE + pairwise preference。
DCA 先用 normalized differential gradient + sparse dictionary；不先做复杂 prerequisite graph。
Residual 先用 atom-level access/failure mass；frontier/access 用 pass@1/pass@8 或 pass@16 近似。
Retention 先用 risk-triggered self-anchor；risk 先以 atom/gradient conflict + category calibration 近似。
训练期间 3–5 个 behavior probe checkpoint，保留最好而不是固定 3 epoch。
MVP 的成功标准是证明“局部纠正 + capability residual gating”本身优于完整轨迹蒸馏，并显著减少 Live/Irrelevance 这类已掌握能力的损伤。
9.2 完整版：闭环 capability-state distillation
utility-weighted DCA + signed transfer graph + frontier/access/robustness/composition ledger；
query × supervision-mode value function与信息增益；
teacher direct→repair→unresolved 三级 teachability；
atom-conditioned blueprint generation 与 boundary negative pairs；
per-atom teacher retirement 与 composition-junction distillation；
低频 atom refresh + 高频 ledger refresh。
10. 实验总目标与研究问题（RQs）
RQ
问题
RQ1
DCA 是否比 raw/cluster gradient 表征更能预测真实行为迁移？
RQ2
蒸馏局部 fail→correct 差分是否比完整 teacher trajectory 更高效、更少遗忘？
RQ3
Capability residual gating 是否提供超越题级 p̂ / step divergence 的增益？
RQ4
Sparse/reliability-aware supervision 是否能减少 cascade noise 与 parameter drift？
RQ5
Signed atom transfer 是否能预测并防止 Live/Irrelevance 等 base 强能力损伤？
RQ6
按 frontier/access/robustness/composition 路由 supervision 是否比固定 demo→guided 更省 teacher token？
RQ7
Atom-conditioned generation 是否比 category/seed generation 更能补未饱和能力？
RQ8
这些结论能否跨 BFCL、ALFWorld、AppWorld、τ² 复现？

11. Phase 0：实验前置与不可跳过的 sanity checks
检查
内容
通过条件
Harness identity
同一 query 在 probe/train/eval 的 system prompt、tool schema、renderer、termination 规则一致。
否则整个 ledger 方向无效。
Base replication
至少两次复现实验 base 总分与细分类别，确认方差。
当前 BFCL base 46.27 作为内部参照；先建立 CI。
Teacher direct quality
每类抽样 direct pass、token cost、失败类型。
决定 repair 与 budget policy。
Render/data audit
每个 intervention event 能重放；a−/a+ 与环境状态一致。
防止“离线正确、harness 错位”。
Budget accounting
teacher output、repair、user-sim/τ² 双边成本统一记账。
所有方法按同一 paid-token 口径比较。

12. Phase 1：决定论文生死的 P0 实验
12.1 P0-A：Atom representation ladder
在同一批 verified correction events 上比较四种表征，训练器与数据完全不变：
Arm
指纹
A1
整轨迹 raw gradient
A2
turn-level normalized/preconditioned gradient
A3
matched correct−wrong differential gradient
A4
utility-weighted differential gradient（推荐）

评价不只看 clustering。必须看：bootstrap matching/stability；held-out correction transfer AUROC；predicted atom loading 与实际 post-update gain 的 Spearman；top-loading micro-update 的选择性。
预期结论/Go-NoGo
支持核心假设的结果：A3/A4 在 transfer prediction 和 intervention selectivity 上持续优于 A1/A2；A4 最好。若 A3/A4 不能提升真实迁移预测，则应弱化“capability atom”主张，保留为训练 feature，而不要继续构造复杂 graph。

12.2 P0-B：蒸馏对象 ladder（最关键）
Arm
训练信号
B0
当前 full teacher trajectory SFT
B1
student-guided / corrected full trajectory CE
B2
只 CE earliest consequential corrected span
B3
只做 a+ > a− 的 local preference
B4
local preference + corrected-span CE（核心）
B5
B4 + DCA residual gating

主指标：BFCL overall、MT/stateful、Live、Irrelevance、teacher tokens、trained tokens、base damage。B4 如果只提高 target 但仍严重伤 Live/Irrel，则说明 sparse correction 本身不够，需要 signed-risk retention；B5 若比 B4 进一步提升，才证明 atoms 不只是解释工具。
预期结论/Go-NoGo
最希望看到 B4 > B2/B3 > B0/B1，且 B4 的 retention 明显更好；B5 再提供 ≥1pp 绝对收益，或在同分情况下减少 ≥20% teacher/training token，或显著降低 damage。若 B4 不优于 B1，先检查 consequential-error 定位与 pair construction，不要马上归因 atom。

12.3 P0-C：Residual gating 对比
Arm
权重
C0
uniform
C1
1−p̂
C2
cluster failure rate
C3
SOD-style step divergence / reliability
C4
atom residual only
C5
intervention utility × atom residual × reliability（推荐）

若 C4/C5 不优于 C1/C3，说明 atom ledger 没有提供足够新信息；这比最终总分更重要，因为它直接回答“为什么需要 atoms”。
13. Phase 2：Retention、teacher routing 与数据闭环
13.1 P1-D：Signed transfer calibration
先不直接上 gradient surgery。对一批 correction events 计算 predicted conflict/risk，做小步真实更新，再测 mastered atom/category 的行为变化。画 predicted risk quantile → actual damage 曲线，并报告 Spearman/AUROC。
若 risk calibration 有效，再比较：无保护；全局 λ；conflict-filter；risk-triggered self-anchor；risk-triggered anchor+filter；最后才是 projection。
预期结论
最理想的证据不是“projection 又涨了 0.5 分”，而是 risk score 对 actual damage 有清晰单调性；随后 selective protection 用更少 retention loss 达到和/或超过全局 λ 的保护效果。

13.2 P1-E：Frontier / Access 路由
对 support 子集测 pass@1 与 pass@K，将题/atom 分成 frontier、access、mastered；再比较固定 full-demo、固定 local correction、p̂-threshold routing、gap-type routing。
预期：access gap 应更适合 local correction / self-on-policy；frontier gap 才需要 full/repaired demo。若 routing 只省 token 不降分，也构成重要贡献；如果还能提高分数，则说明 supervision type 与 capability state 的匹配成立。
13.3 P1-F：Teacher failure repair
每类抽 direct-fail 题，使用 trace+verifier feedback 做 1–N 次 reflective repair，记录 recovered pass、额外 teacher tokens、学生 uptake。只在 repair value positive 时进入训练。
预期：Web Search/τ² 等可能存在大量 near-miss；若 repair 后 teacher 可解但 student uptake 仍低，问题在 distillation；若 teacher repair 也失败，则才写 unresolved-under-budget certificate。
14. Phase 3：Generation、Robustness 与 Composition
14.1 P2-G：生成条件消融
Arm
生成条件
G0
类别/seed 形状（现有）
G1
欠供 atom
G2
atom + gap type + boundary seed
G3
G2 + verified blueprint + paired negative/counterexample

主指标不只看生成题通过率，还要看生成数据的 atom residual coverage、训练后的 held-out atom gain，以及对 saturated atoms 的无效供给率。
14.2 P2-H：Composition-junction distillation
先筛出“单 atom mastery 高、组合任务失败”的样本。只在 junction state 蒸馏 transition，与重复单技能样本、多轮完整 trajectory 做对比。ALFWorld/AppWorld 特别适合，因为状态依赖更清晰。
15. 四台 benchmark 的具体实验策略
Benchmark
最适合验证什么
执行重点
BFCL
先做全部 P0；重点看 single vs MT、Live/Web Search/Irrelevance、boundary behavior。
最适合验证 local correction、risk retention、boundary negative；现有最好 finetune 仍低 base，先解决 damage。
ALFWorld
环境 verifier 强、状态可重放，适合 causal correction 与 GiGPO-style utility credit。
重点验证 turn-level DCA、composition-junction、课程/深度。
AppWorld
demo 极长，完整 teacher trajectory 成本高；官方 scaffold 下 base 非零后才有 retention 账。
重点验证 teacher token 节省与 local correction；这是 budget 故事的强场景。
τ²
teacher 与 user-sim 都有成本，且 teacher fail/near-miss 可能重要。
重点验证 repair、双边成本账本、robustness/boundary 与黑盒 supervisor 路由。

16. 实验调度：按“信息价值”而不是模块完整度推进
顺序
实验
它回答的关键问题
P0-1
BFCL harness/base 方差 + 现有数据重审计
所有后续可信
P0-2
B0–B5：full vs local differential distillation
确认蒸馏主线
P0-3
A1–A4：DCA validity
确认 atom 主线
P0-4
C0–C5：residual/reliability gating
确认 atoms 进入 loss 是否有增益
P1-1
signed risk calibration + selective anchor
解决 Damage
P1-2
frontier/access routing + teacher repair
解决预算效率与 hard cases
P1-3
把核心方法迁移 ALFWorld/AppWorld
跨台验证
P2-1
atom-conditioned generation + robustness pairs
闭环 acquisition
P2-2
composition distillation + τ²
增强 agent-specific novelty
P3
LOPD/rubric learned criterion 等高级扩展
仅在主线已正后进入

17. 预期实验结论：应该出现什么模式，什么结果会否定我们
假设
支持它的预期模式
否定/回退条件
DCA 有效
utility-weighted differential atoms 对 held-out transfer/gain 的排序明显优于 raw gradient cluster；跨 seed/checkpoint 可匹配。
若没有：atom 退化为辅助 feature，不再声称 latent capability ontology。
局部差分蒸馏有效
local pref+CE 在相同 teacher budget 下优于 full/corrected trajectory，并降低 Live/Irrel 损伤。
若没有：检查关键错误定位、a+/a− 对是否真的 causal；可能需要短 horizon RL 而非 preference。
Residual gating 有效
atom residual 比 1−p 或 step divergence 更能挑出“值得学”的 correction。
若没有：ledger 状态估计太噪；改成 reducible-error predictor，而不是强行用 atom。
Sparse supervision 有效
更少 trained tokens 仍达到更高或相同成功率，参数/行为 drift 更小。
若没有：任务需要后续恢复策略；扩大 recovery window 而不是回到全轨迹。
Signed risk 可预测
risk quantile 与实际 base damage 单调；selective anchor 比全局 λ 更高效。
若没有：不用 atom risk 做硬 gate，退到 behavior anchor + checkpoint selection。
Routing 有效
frontier→demo、access→correction/self 的路由减少 teacher 输出并保持/提高分数。
若没有：gap taxonomy 不能预测最佳 supervision；用 learned bandit 替代规则路由。
生成闭环有效
atom+gap generation 的 residual coverage 与 held-out gain 高于类别生成，饱和供给更少。
若没有：生成器只在表面模仿 atom seed；加强 blueprint/verifier 约束。
跨台成立
核心 B4/B5 + selective retention 至少在 3/4 benchmark 有一致方向。
若只 BFCL 有效：论文定位应缩窄为 tool-call boundary distillation。

18. 建议的定量 Go / No-Go Gates
以下不是对结果的承诺，而是提前定义的内部决策阈值，避免看到结果后移动目标。可根据 base 方差与运行成本微调。
Gate
建议阈值
Core distillation gate
B4 相比 strongest full-trajectory baseline：overall ≥ +2pp，或同分下 teacher/training token 降 ≥30%，且 base damage 不更差。
Atom utility gate
B5/C4-C5 相比无 atom 的 B4/C3：≥ +1pp，或 budget AUC 提升 ≥20%，或 damage 明显下降。
Atom validity gate
predicted transfer/gain Spearman > 0.30 且 top-vs-bottom atom-loading quartile 有显著 post-update gain 差异。
Risk gate
negative-transfer predictor 对 harmful update AUROC 目标 >0.70；否则不用于硬过滤。
Teacher-efficiency gate
完整方法相对 full-demo pipeline 的 paid teacher output token 目标下降 30–70%，以 AppWorld/τ² 为重点。
Publishability gate
最终方法至少匹配/超过 base overall，并在 target adaptation 与 retention 间形成清晰 Pareto 改善；最好在 ≥3/4 benchmark 出现一致方向。

19. BFCL 的具体预期路径（基于当前现象，不是结果承诺）
当前内部参照为 base 46.27，而现有最好 finetune 约 38.30，主要风险是 adaptation 获得部分 MT/stateful 增益的同时损伤 Live/Web Search/Irrelevance 等强区。新方法首先要“止血”，其次才是继续拉目标能力。
第一阶段：B4（local pref+CE）应比 full trajectory 显著减少不必要梯度密度，目标是先把 finetune 与 base 的差距明显缩小，而不是追求一次过 46。
第二阶段：B5（residual gating）应减少在 saturated/不可吸收原子上的训练，把增益集中到 MT/stateful/关键 boundary gaps。
第三阶段：risk-triggered self-anchor/冲突过滤负责恢复 Live/Irrelevance；若只靠加大 λ 才能保住，说明 signed risk 还没学到。
第四阶段：frontier/access routing 与 teacher repair 决定 Web Search/Memory 等低 p 区是否继续投预算；只有 repairable 且 student uptake>0 才保留。
最终目标：在同一 teacher-token 预算下形成“overall 上升 + base 强类别不显著掉点”的 Pareto frontier，而不是单臂总分偶然变好。
20. 主要风险与设计防线
风险
防线
Atom 不稳定/随 checkpoint 旋转
低频刷新、bootstrap matching、只让高 reliability atoms 进入硬决策；其余作为连续 feature。
Differential pair 不是 causal
必须 replacement+continuation 验证；“teacher 不同意”不能自动成为 correction。
Preference 学不到精确 API 格式
保留 corrected-span CE；η 按 execute atom 增强。
Full demo 对 frontier 仍必要
不强行 all-local；gap-type routing 允许 cold-start demonstration。
Teacher 失败被误判不可教
direct→reflective repair→unresolved；每级都记 cost 与 success。
Signed graph 太贵
先 gradient prior + sparse calibration；只对 mastered/high-mass atoms 估负边。
方法组件太多导致审稿人看不懂
主论文只保留三核心：local correction distillation、capability residual gating、risk-constrained sparse distillation。
与 OPD² “delta”混淆
全文避免把主方法叫 Delta Distillation；明确 OPD² 是 teacher-vs-teacher-base logits，我们是 black-box student error-vs-correction intervention。

21. 最终论文应讲的三个贡献
Contribution 1 — Student-state correction distillation
学生先走；teacher 只在经 causal 验证的 consequential divergence 上提供最小纠正。训练优化 correct-vs-actual-error 的局部行为差分，而非复制完整专家轨迹。

Contribution 2 — Capability-residual distillation
用 Differential Capability Atoms 把每次纠正分解到 student-specific adaptation coordinates；蒸馏强度由对应能力的剩余 frontier/access/robustness/composition deficit 决定。

Contribution 3 — Risk-constrained sparse distillation
只在高 utility、可信、未饱和的局部 token/turn 上反传；并只在 signed capability graph 预测会伤已掌握能力时启动 retention。

Acquisition、repair、generation、curriculum、teacher retirement 都应作为这三项的配套机制，而不是与它们并列成八个贡献。
22. 一页算法规格（推荐主算法）
输入：support S、student πθ0、black-box teacher T、verifier V、budget B。
在 deployment path 上 probe student；估 pass@1/pass@K，并收集失败轨迹。
对失败轨迹提出 earliest-error 候选；用局部 replacement+continuation 找 consequential divergence。
在该 student state 请求最小 teacher correction；direct fail 时做 reflective repair；验证纠正是否提高成功概率。
由 verified (h,a−,a+) 计算 normalized utility-weighted differential fingerprints；学习/更新 soft DCA dictionary。
对高 reliability atoms 估 frontier/access/robustness/composition residual，并校准 signed transfer 风险。
对候选 (query, supervision mode) 估 residual reduction / paid teacher token；选择正边际价值事件。
形成 event pool：保存 ΔU、α、residual、Q、risk 与 corrected span。
训练：local preference + corrected-span CE；乘 U×R×Q×m；只对高风险事件加 self-anchor/KL/filter。
周期性 behavior probe；保存最好 checkpoint；刷新 ledger，低频刷新 atoms；成熟 atoms 从 external teacher 退到 self/no-distill。
对仍有正价值 residual 的 atoms 做 blueprint synthesis / robustness pairs / composition tasks；重复。
当 max supervision value≤0 或预算耗尽，停止；输出 overall、atom/category Δ、CI、teacher-token frontier 与不可解决凭证。
23. 参考文献与 SOTA 依据（截至 2026-09-02）
[1] Lyu et al. Student-Centered Distillation Narrows the Agentic Gap Between Small and Large LLMs (SCoRe). ICML 2026. arXiv:2509.14257. https://arxiv.org/abs/2509.14257
[2] Zhong et al. SOD: Step-wise On-policy Distillation for Small Language Model Agents. arXiv:2605.07725. https://arxiv.org/abs/2605.07725
[3] Zhou et al. TurnOPD: Making On-Policy Distillation Turn-Aware for Efficient Long-Horizon Agent Training. arXiv:2607.05804. https://arxiv.org/abs/2607.05804
[4] Armandpour et al. Unmasking On-Policy Distillation: Where It Helps, Where It Hurts, and Why. 2026. arXiv:2605.10889. https://arxiv.org/abs/2605.10889
[5] Heo et al. On-Policy Delta Distillation (OPD²). arXiv:2607.15161. https://arxiv.org/abs/2607.15161
[6] Fang et al. Rubric-based On-policy Distillation (ROPD). arXiv:2605.07396. https://arxiv.org/abs/2605.07396
[7] Latent On-Policy Self-Distillation (LOPD). arXiv:2608.13040. https://arxiv.org/abs/2608.13040
[8] He et al. Skill-Targeted Adaptive Training (STAT). ICLR 2026. arXiv:2510.10023. https://arxiv.org/abs/2510.10023
[9] Fang et al. RODS: Reward-Driven Online Data Synthesis for Multi-Turn Tool-Use Agents. arXiv:2606.19047. https://arxiv.org/abs/2606.19047
[10] Zhang et al. Training Data Selection with Gradient Orthogonality for Efficient Domain Adaptation (OGS). arXiv:2602.06359. https://arxiv.org/abs/2602.06359
[11] Ta, Zhu, Shayandeh. PROOF-Gen: From Optimized Data to Better Distillation. arXiv:2608.23911. https://arxiv.org/abs/2608.23911
[12] Xu et al. Controllable and Verifiable Tool-Use Data Synthesis for Agentic Reinforcement Learning (COVERT). arXiv:2604.09813. https://arxiv.org/abs/2604.09813
[13] Huang et al. Skill Self-Play: Pushing the Frontier of LLM Capability with Co-Evolving Skills. arXiv:2607.22529. https://arxiv.org/abs/2607.22529
[14] Feng et al. Group-in-Group Policy Optimization for LLM Agent Training (GiGPO). NeurIPS 2025. arXiv:2505.10978. https://arxiv.org/abs/2505.10978
[15] Wang et al. Denser ≠ Better: Limits of On-Policy Self-Distillation for Continual Post-Training. arXiv:2607.01763. https://arxiv.org/abs/2607.01763
24. 当前最推荐的下一步
如果下一轮只允许做三件事，建议严格按以下顺序：
1. 先把现有 BFCL 数据重构成 intervention events，跑 B0–B5，验证“完整轨迹 → 局部 correction → local preference+CE → atom residual gating”的增益阶梯。
2. 同一批事件上跑 A1–A4，验证 differential utility atom 能否预测真实 post-update transfer；这是 capability atom 是否值得继续做成论文核心的生死实验。
3. 只有上述两项正后，再做 signed-risk calibration + selective self-anchor，把 BFCL 的 Damage 问题从“全局 λ 调参”升级成可预测、可选择的保护。
North-star result
在相同 paid teacher output-token 预算下，CRCD 比 strongest full-trajectory / step-aware black-box distillation 更高成功率；同时对 base 强能力的损伤更小。更重要的是，DCA residual 与 signed-risk 能分别预测“哪次纠正会带来收益”和“哪次纠正会造成损伤”。这两条预测关系一旦成立，整套方法就不再是组件组合，而是一个可验证的 distillation principle。

