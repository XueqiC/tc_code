# RTD unified replacement：P0 方法与数学契约

状态：P0 CPU 实现；P1–P3 全部 PLANNED。未进行 GPU、Gemma 权重加载、API 或 benchmark 实验。本文不把历史 RTD 成绩归给新机制。任务边界为任务书 §2、§3.1、§6、§7/P0。

## 1. 设定与现有实现核查

给定支持任务、完整交互状态、已购黑盒教师文本和固定 agent harness，学习产物是学生参数。训练回报为从任务起点随机执行的完整终局回报；官方部署解码得分另报。教师不提供 logits，也不提供参数梯度。所有梯度均由学生对固定文本求导。

新包位于 `src/bfas/rtd/unified/`，没有修改任何既有 v1.1 源文件。`UnifiedEngine` 仅接受 `method: rtd_unified` / `protocol_version: unified-p0-1`；旧 CLI、α gate、d 求解、回报梯度、bank 和 persistence 保持原行为。新配置为 `configs/rtd/unified_bfcl_gemma4.yaml`，学生为 `google/gemma-4-12B-it`。

**损失口径核查结果：**本工作树的 `alpha_d.components` 调用 `source_gradient_pair` 与 `-backend.score_behavior`，两者均求和；`transport.positive_mixture_loss` 也明确注明不做 token-length normalization。`fingerprint.py`、`behavior/microupdate.py` 则存在按实际长度平均的路径。因此“现有 v1.1 hard loss 都按每序列长度平均”并不是这份代码的事实。新包保留显式 `per_sequence_mean`，作为所要求新配置的默认值，同时提供 `total_token_nll`，不暗改旧臂。P1 必须匹配口径，旧 v1.1 原样臂和匹配口径的归因臂应分开标注。

## 2. 不可变对象与版本

`TeachingProblem` 固定以下内容：

- θ 和来源快照的有序参数布局、原始 dtype、内容 hash，以及包含来源后端约定的 `source_policy_hash`；
- 冻结的正对角 P、H=P⁻¹、η、λ、loss hash、采样/曝光/optimizer 版本；
- 每个预定曝光槽的完整 state hash、父任务、权重 w_i、m=2 个 draw ID、样本内容 hash、行为 hash、硬梯度与保留梯度；
- 每份已购证据的 query/version/state、教师文本 hash、完整 token 行为 hash、θ/loss 绑定和教师梯度；
- 参考 a_ref、基础增量、列方向 U、完整 K、PSD 数值修复记录、反馈版本/hash、h、ε，以及可重投影的轨迹统计；
- x 的 box 和每个来源槽共享的多教师版本 simplex。

底层 `Array` 保存 float64 字节和 shape，返回不可写的 bytes-backed NumPy view；不能用 `setflags(write=True)` 修改。参数/tensor 出口都是新拷贝。仅用 `dataclass(frozen=True)` 而继续暴露 mutable tensor 不足以满足本契约。工厂验证内层父任务、已购权限、梯度版本、m、draw ID 唯一性和冻结权重和为一；不对添加证据后的旧权重重新归一化。

默认第一份教师每个来源的 a_ref=.5，表示跨状态统一的**通用初始化超参数**，不是基准/错误类型规则。后购教师版本参考系数为零，旧参考不变。无可用反馈时保存 `unmeasured`，零 h 表示参考启动问题，不能称为“测得零标准误”；采购标签要求已测反馈。

这里 U 是全部 trainable 参数上的 dense CPU 参考表示；P0 不把它冒充可直接训练 12B 模型的高效存储。模型身份须由实际加载后的参数与后端生成，配置中的模型名称不能替代 checkpoint hash。

## 3. 经验目标、连续系数与冻结求导

在同一状态从冻结 p_t 独立新采两个完整行为，包括真实 EOS，或生成 cap 时的真实截断结果。不会为了不同、格式或正确性拒绝采样。令 a_ij∈[0,1]：

\[
\widehat q_i={1\over m}\sum_j[(1-a_{ij})\delta_{Y_{ij}}+a_{ij}\delta_{y_i^T}],\qquad m=2.
\]

实现按完整 token 行为身份合并相同结果，而不是把重复候选当不同类别。教师文本 hash 单独保留，以便审计不同文本/版本。

\[
\alpha_i=(a_{i1}+a_{i2})/2,\qquad d_i=(a_{i1}-a_{i2})/2.
\]

这只是解释性重参数化。如果 Y_i2=y_i^T，教师行为总质量为 α_i+(1−a_i2)/2，不能把 α 称为固定的教师行为总质量。

给定损失 ℓ_θ(y)，G_ij=∇ℓ_θ(Y_ij)，G_T=∇ℓ_θ(y_T)。求导时冻结 p_t、样本、教师文本、w、a、P、η 和反馈。目标生成过程随 θ 改变的总导数不属于这里的梯度。

\[
g_{raw}=G_{avg}+{1\over m}\sum_j a_j(G_T-G_j),\quad
g_{CV}=G_{soft}+{1\over m}\sum_j a_j(G_T-G_j).
\]

`estimate` 提供 raw/CV；实际更新使用相同代数在问题中构造 U。a 无论来自 QP 或未来网络，在学生更新处均停梯度。有限批次的 CV 可以对应带符号梯度组合；它不一定是本次非负 q̂ 的精确梯度。

## 4. 前缀测度和随机长度：必须补上的修正

令 L 是含实际采样终止 token 的随机动作长度，h_t 是第 t 个到达的来源前缀，
g_t(v)=−∇log π_θ(v|h_t)，s_t=Σ_v p_t(v|h_t)g_t(v)。采样 cap 是在动作前固定的有限停止规则；不插入未采到的 EOS。

### 4.1 总 token NLL

G_hard,Σ=Σ_{t≤L}g_t(Y_t)，S_Σ=Σ_{t≤L}s_t。事件“到达第 t 个前缀”对此前历史可测，故逐前缀条件期望给出

\[
\mathbb E[S_\Sigma-G_{hard,\Sigma}]=0.
\]

当当前策略等于快照时，每个前缀的 logit 导数为 π−p=0。实现使用精确解析 cotangent，避免相等 logits 下数值相减引入虚假梯度。相同参数是充分条件；不能声称参数相等是必要条件，因为参数/策略可能不可辨识。

### 4.2 每序列实际长度平均：可执行的匹配保留估计

这里 G_hard=G_hard,Σ/L。一般不能用 S_Σ/L 替换它，因为 1/L 依赖当前 token 及未来 token。我们用动作采样前固定的 c>0：

\[
G_{soft,len}=cS_\Sigma+(1/L-c)G_{hard,\Sigma}.
\]

于是逐样本

\[
G_{soft,len}-G_{hard,len}=c(S_\Sigma-G_{hard,\Sigma}),
\]

右侧无随机长度乘子，其期望为零。配置 c=1/1024，对 512/1024 的 cap 均合法；这是影响方差的通用控制变量尺度，不改变目标期望，也不使用实现后的输出长度选择 c。硬来源和教师均各自按**自身**实际 token 数归一化。不同 L 的样本不在 batch 内先合并 token 再平均。

实现名 `length_corrected_prefix`。这包含硬梯度残差，准确名称是“带长度修正的软保留估计”，不是纯 KL。`soft_retention_loss` 的标量是用于产生所定义梯度的 surrogate，不能作为 KL 数值报告。它同时对任意冻结系数选择器保持 §3 的无条件恒等式。

### 4.3 纯软 Rao–Blackwell 选项

若能提供所有下一 token 的条件逆长度
ρ(h_t,v)=E[1/L | h_t,Y_t=v]，则

\[
G_{soft,RB}(Y)=\sum_{t\le L}\sum_v p_t(v|h_t)\rho(h_t,v)g_t(v)
\]

也满足 E G_soft,RB=E G_hard,len。这是对 token 及未来长度共同积分，不能用实现后的 1/L 代替 ρ。`conditional_length` 接受这个显式条件量；缺失时拒绝执行。P0 有有限树精确枚举测试；没有为大词表免费提供精确后缀期望，也不把未来近似器标为无偏 oracle。

### 4.4 任务书数学修正

**随机实际长度平均时，“匹配 hard 梯度”和“快照处保留梯度恒零”一般不可同时满足。**考虑根节点以 .5 输出 EOS（L=1），以 .5 输出继续 token，随后必在第二个位置结束或 cap（L=2）。根节点的 continue logit 在快照处为 0，故硬 NLL 根梯度分别为 +.5、−.25；总体梯度 .5×.5+.5×(−.25)=.125。

任何期望匹配的 soft 估计也必须有 .125 的期望，不能逐样本恒零。普通 prefix KL/L 恰为零，因而有偏。`length_corrected_prefix` 在 c=.5 时分别给 .25、0，期望 .125；纯软 RB 根梯度在每条行为上都是 .125。

所以任务书的 a=0 零梯度/不动条件只能在总 NLL、采样前确定的缩放，或确实固定长度的匹配情形下使用。P0 没有用错误恒等式让测试“通过”：既测试适用条件下的零梯度，也测试随机长度下的反例。保留每序列口径意味着接受这个非零性。未经说明切回 total NLL 也不合适。

## 5. 仿射教学更新与联合 QP

\[
b_{ij}=-(\eta w_i/m)P(G_i^T-G_{ij}),\qquad
\theta^+(a)=\theta-\eta P\sum_iw_iG_{soft,i}+Ua.
\]

raw 对照仅将基础项换为 G_avg。所有方向含具体 teacher–student 差；不做正交化、聚类或逐列筛选。相同学生梯度仍可有非零教师方向；学生梯度零但教师非零时同样可学；G_ij=G_T 时对应列为零。

共同参考 θ_ref=θ⁺(a_ref)，在它上面测 ĥ=∇J 的估计；x=a−a_ref。求

\[
F(x)=\widehat h^TUx-\frac\lambda2x^TKx-\sum_\ell\epsilon_\ell|x_\ell|,
\quad K=U^THU,\quad -a_{ref}\le x\le1-a_{ref}.
\]

H=P⁻¹，λ 按原单位配置，**不沿用旧 d 的 mean-diagonal 隐式单位变换**。K 不是回报 Hessian。构造时检查对称性与最小特征值；仅在 relative `psd_tolerance` 范围内对称化/将负特征值裁为零，记录最小特征值、非对称残差和修复范数，显著非 PSD 则报错。

引入 t≥x、t≥−x、t≥0，最小化
½xᵀλKx−(Uᵀĥ)ᵀx+εᵀt。这是凸 QP；t≤1 可无损加入，因为规范解 t=|x|≤1。实现用 SciPy SLSQP 求这个**二次目标+线性约束**，再独立恢复非负对偶乘子并检查原始可行性、stationarity、complementarity。所有矩阵项来自 `TeachingObjective`。只接受 KKT 残差达配置容差的解，不以优化器 success 字符串代替证据。为数值稳定除以一个正的共同目标尺度；报告该尺度以及归一化 KKT 残差，乘子恢复为原单位。数值失败抛错。

不能根据 |z_i|≤ε_i 单独冻结坐标；零坐标条件依赖 z_i−λΣ_j K_ij x_j。参考可行且无信息时同一目标保持 x=0，a_ref 的教师教学仍然存在，不伪造正反馈。

数学上的最优解集合在交换来源后等变；唯一解的系数相应等变。重复方向时系数不唯一，测试对称实例的系数与实际更新，文档不把任意数值 solver 的平局选择宣称为新的唯一真值。

## 6. 回报、trial、commit 与接口

对每个反馈任务 t，有独立完整轨迹 τ_tk 和预先固定的任务权重 ω_t：

\[
\widehat h=\sum_t{\omega_t\over n_t}\sum_k(R_{tk}-b_{tk})
\nabla\sum_{u\in\text{学生动作}}\log\pi_{\theta_{ref}}(y_u|s_u).
\]

`collect_task_feedback` 包装 v1.1 的 `collect_feedback` / `reinforce_gradient`。同任务 LOO baseline 不含当前轨迹回报；也支持固定独立零 baseline。完整 prompt 中工具/环境文本是条件，不是被计分的动作。由同一后端生成和校验 likelihood，temperature=1、top_p=1，拒绝 off-policy 身份。上式针对训练随机策略，不针对官方 greedy Overall。`FeedbackStatistic.project` 复用 v1.1 的重拟合 LOO jackknife；不等任务权重以任务均值和独立任务方差组合。n=2 的 SE 是配对贡献代理，不能称为严密置信界。

ε 来自这些反馈标准误，标签严格为 **uncertainty regulariser / 不确定性正则**。它不覆盖反馈陈旧、来源采样、表示近似或选择后的同时误差。

| 语义接口 | P0 行为 |
|---|---|
| build_prequery_features | 只接受 PublicQuerySpec 和显式不可变购前 context；字段 allowlist |
| purchase_or_reveal | 调用原 SealedReplayBroker/Ledger；先公开 cap 预留，后揭示实际支出；同包一次收费 |
| build_teaching_problem | 固定 θ/p_t/P/loss/曝光/来源/已购文本，建立基线、U、K、可行域 |
| solve_exact | 同一个 F 的 auxiliary-variable QP + KKT 检查 |
| predict_control | 明确 stub，返回 a_ref；未训练 |
| commit_distillation | 从原 θ 安装一次被冻结 P 的蒸馏更新；不调用 Adam/RL step |
| collect_task_feedback | 上述完整任务 score-function 梯度，反馈父组与内层/确认集隔离 |
| build_acquisition_labels | 同参考、同 h、同旧权重下追加已购方向并重新求解同一 F |
| score_query_batch | 明确 PLANNED stub；无预测分数，不制造排序 |
| validate_and_report | 校验版本、系数、参数增量、F 与单次更新计数；没有虚构 benchmark 分数 |

`TeachingObjective` 是唯一 objective/update 定义。solver 的 quadratic/auxiliary callback、trial、commit、采购反事实、未来控制器 decision regret 都调用它。trial 先保存模型参数/缓冲区/梯度/模式、optimizer state、Python/NumPy/Torch RNG、显式 Generator 和传入的 caller state；异常也恢复。模型只正式提交一次蒸馏。参考 trial 不成为下一步的起点。无独立 α 元目标、无反传经过解出的 a。

实际增量与仿射预测的浮点舍入差记录为 `increment_error`；双精度测试对照独立 hard/soft/teacher 梯度累积。非线性有状态 Adam 不在本恒等式的适用范围内。P0 不声称 BF16 参数增量在实数意义下精确。

## 7. 采购嵌套与多个教师版本

候选槽和免费学生保留基线在采购前存在，w 不变。新证据只开放方向。每个来源支持 a_ijv≥0、Σ_v a_ijv≤1：

\[
\widehat q_i={1\over m}\sum_j[(1-\sum_va_{ijv})\delta_{Y_{ij}}+
\sum_va_{ijv}\delta_{y_i^{Tv}}].
\]

新增版本的 a_ref=0。扩展旧解为 (a_old,0) 时，参考、实际增量、目标质量和不确定性惩罚恢复旧问题。旧方向按原顺序累积，跳过精确零系数，因此追加零列使用量不会改变旧浮点参数增量。K 的构造/修复仅可能产生已检查容差内的数值差；F 的零使用等价性按该容差验证。

购后标签为 F*_new−F*_old，保存购前特征 hash、ledger hash、θ/反馈/optimizer 版本。它包括新教师能带来的全部替换增益，即使两条学生梯度相同也不自动为零。没有把“改 d 的收益”当成全部教学价值。旧问题可行解嵌入新问题，因此精确局部最优值不下降；真实未来任务成功率没有这个保证。

如果添加新状态时强制加一个非零 KL，即使 a=0 也改变基线，就不满足零使用契约；本工厂拒绝未预设槽的证据。接口按可信、协作进程的信息边界设计，不声称防御恶意 Python 反射；不能给预测器任意 broker 或 hidden payload context。

## 8. 已证明的恒等式清单

**I1 合法性与重复合并。**每个来源份额非负且和为 1/m；合并同一行为只相加，故 q̂≥0、Σq̂=1。多版本 simplex 用同一证明。

**I2 重参数化与来源交换。**a1=α+d、a2=α−d；交换来源和对应系数保持 q̂、raw/CV 梯度、Ua 和实际更新。对置换矩阵 Π，U→UΠ、K→ΠᵀKΠ、x→Πᵀx，F 与可行域不变。

**I3 原始经验梯度。**冻结 q̂ 后展开 Σ_y q̂(y)∇ℓ(y)，得到 g_raw；适用于本文两种 ℓ，不限总序列 NLL。

**I4 自适应选择下的无条件期望。**逐样本 g_CV−g_raw=G_soft−G_avg，不含 a。若 E G_soft=E G_avg，则对来源和反馈/选择过程共同平均，任意可测、冻结的样本依赖 a 都有 E g_CV=E g_raw。不要求 a 与样本独立；不推导给定样本/控制的条件无偏性。

**I5 两种长度口径的匹配。**§4 的可测前缀条件期望证明 total NLL 的匹配；固定 c 差恒等式证明 length-corrected 的匹配；对 (token,未来长度) 的条件期望证明纯软 ρ 形式的匹配。相同策略处 total 梯度逐前缀为零；固定 L 且 c=1/L 时 mean 也成立。

**I6 旧新估计器差。**m=2 时 g_new−g_old=α(G_soft−G_avg)，g_old−g_raw=(1−α)(G_soft−G_avg)。自适应乘子破坏旧证明；没有修改任务书这两个正确的系数公式。

**I7 仿射对应及自然退化。**把 g_CV 代入固定线性一步 θ−ηPΣw g 即得基础项+Ua；相对同参考为 Ux。相同/零/教师相同梯度退化直接由列差得到，不需要额外规则。

**I8 端点值、期望、方差。**a=0：raw=G_avg，CV=G_soft；a=1：raw=G_T，CV=G_T+G_soft−G_avg。教师固定时后一式方差为 Var(G_soft−G_avg)，均值为 G_T。没有普遍的 CV 方差更低结论。

**I9 QP 等价性与最优性。**ε≥0 时存在 t=|x| 的最优 auxiliary 解；H PSD 给 K PSD；凸问题的原/对偶可行性、stationarity、complementarity 证明该局部 QP 的全局最优性。实现只声称达到记录的数值容差。

**I10 新证据零使用嵌套。**固定基线/旧权重/参考/反馈，添加系数零的方向和 simplex 版本后旧 F 与增量恢复；可行集嵌套推出局部 F* 非减。无“真实成功率非减”的结论。

**I11 合法 baseline 的 score-function 恒等式。**对独立于当前轨迹动作的 baseline，条件于其他轨迹，b E[∇log P_θ(τ)]=0；因而同任务独立 LOO 不改期望梯度。环境转移不含待学习参数，轨迹 score 只需学生动作项。有限时域/可交换微分积分和独立 rollout 是假设。

## 9. 明确反例与受限比较

| 编号 | 构造 | 精确结论 |
|---|---|---|
| C1 自适应旧 weighted-soft | 两动作 A/B 各 .5，梯度 −.5/+.5，教师 A，soft=0；仅当第一来源 B 时 α=1，d=0 | 枚举四对：E raw=E CV=−3/8；E old=−1/4 |
| C2 CV 批次不是正目标；端点有噪声 | 同 C1 的 pair (B,B)，a=(1,1) | CV=−1，超出所有正混合梯度区间 [−.5,.5]；全一端点 raw 方差 0、CV 方差 1/8；全零端点 raw 方差 1/8、CV 方差 0 |
| C3 随机长度与零保留冲突 | §4.4 的 L∈{1,2} 树，快照相同 | mean hard/匹配 soft 期望根梯度 1/8；naive KL/L 为 0 |
| C4 teacher 前缀不能替代 source 前缀 | 源只有 .5 到第二前缀，教师总到第二前缀；当前第二 logit=−.7，源=0 | 教师前缀 soft 的第二坐标是匹配期望的两倍 |
| C5 逐坐标原始 z 筛除错误 | K=[[1,1],[1,2]]、z=(.8,0)、ε=(0,.1)、a_ref=(.5,.5)、λ=1 | x=(.5,−.2) 最优；第二坐标尽管 |z2|≤ε2，仍应通过交叉项抵消更新代价 |
| C6 scalar 受限比较 | b1=(1,1)、b2=(1,−1)、h=(0,1)、H=I、λ=1、参考 .5 | scalar x1=x2 的最优 F=0；完整解 x=(.5,−.5) 得 F=.5，并保留第一坐标 |
| C7 当前回报作 baseline 有偏 | Bernoulli(.5) 动作，R=动作，score=动作−.5 | 真梯度与两独立轨迹 LOO 期望=.25；baseline=自己的 R 给恒零 |
| C8 “a=0”不等于可忽略新增基线 | 旧增量为0，新增槽强制 soft=1、ηP w>0，即使 a=0 | 新增量=−ηP w≠0，违反嵌套；本实现拒绝这类新增槽 |
| C9 固定 α 不固定 teacher 行为质量 | 来源 (A,T)，a=(.5,.5) | α=.5，而 q̂(T)=.75 |

C6 对该局部 F 的一般充分条件是：ε=0 时，在共同参考附近有可行来源差方向落在某个保留观测的零空间、与 h 正对齐，而 scalar 限制的方向空间没有这样的对齐。ε>0 时还要求该方向的线性对齐严格超过 εᵀ|x|，且 scalar 可行方向没有正的惩罚后线性收益。足够小步长下这个正的一阶净收益压过二次项。比较类明确限制为同批次、同估计器、a_i1=a_i2 的 state-scalar；不证明所有其他学习算法都做不到。

## 10. 条件性局部收益界和不声称的内容

若在整个可行局部区域，J 对 H 范数满足下侧平滑界，λ≥L_H，且**同时**有 |b_lᵀ(ĥ−h)|≤ε_l，则

\[
J(\theta_{ref}+Ux)-J(\theta_{ref})\ge F(x).
\]

证明：平滑界的线性项为 hᵀUx，逐项同时误差用 εᵀ|x| 控制，二次惩罚用 λ 覆盖。由于误差界同时有效，才能在看过反馈选择 x 后继续使用；单坐标经验 SE 不够。因此 P0 只称 F 为局部代理，不把正 F 报成置信保证。

若实际位移为 Ux+e、‖e‖_H≤ρ 且 ‖h‖_{H,*}≤M，再减
(M+L_H‖Ux‖_H)ρ+L_Hρ²/2。近似解相对代理 optimum 损失 δ 时再减 δ；未被 ε 覆盖的对齐误差 ζ 给额外 ζᵀ|x|，几何误差 E_K 给额外 λ‖E_K‖₂‖x‖²/2。近似梯度/sketch 可通过方向误差进入 e 或 ζ，须避免重复计数。来源采样方差、有限反馈与选择过拟合须实测，独立确认不能省略。

不声称：全局真实任务最优；每步成功率单调；CV 必然降方差；非负每批次 CV 监督目标；对目标生成器总导数；有状态 Adam 更新无偏；SE 是选择后置信界；Gram 是任务 Hessian；采购代理等于完整信息价值或可作为全局停止证书；任何 benchmark 性能提高；Gemma 适配/显存已验证；QP 或 a1/a2 重命名单独具有方法新颖性。

## 11. P0 验证、审计与复现

测试位于 `tests/test_unified_*.py`，覆盖概率/重复、任意选择器枚举、所有端点的值/均值/方差、前缀/随机长度、全交叉 QP 与 KKT、来源交换/退化、同起点 trial/commit、异常恢复、动作-only LOO、隐藏字段、账本、双教师版本和单步 engine。

```bash
export BFCL_PROJECT_ROOT=/tmp/rtd-unified-p0-bfcl
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PYTHONPATH=src:. /home/xueqi/hq/projects/tc-alignment/.venv/bin/python -m bfas.rtd.unified --config configs/rtd/unified_bfcl_gemma4.yaml
PYTHONPATH=src:. /home/xueqi/hq/projects/tc-alignment/.venv/bin/python -m pytest -q tests
```

CLI 只验证和描述 P0 配置，不下载模型、不启动训练。编程入口：构造 `TeachingContext` → 用 `score_sources` / `score_teacher` 取得新梯度 → `build_teaching_problem` → `UnifiedEngine.run_step`，注入兼容的 CPU/未来生产 backend 和完整任务 rollout。实际训练前必须独立完成 P1 的 Gemma/harness/tokenizer 身份适配，不能把旧 Qwen runtime 当成已兼容的 Gemma loader。

预算审计见 `RTD_UNIFIED_P0_BUDGET_MANIFEST.json`：复核本地 v1.1 public certificate、公开目录和 aggregate audit，420 包/55,370 内容 tokens，21,203 exact、34,167 estimated；整数 cap 5,537/13,843。本轮无真实购买、无新 API 成本；历史 1,233,607 演示输出和 142,727 生成输出保留各自口径。任务书中的另一池 16,748 不并入。student_init_hash/teacher 逐包身份尚未为新实验冻结，manifest 用 null/PLANNED 明示，不能填模型名替代 hash。

CPU suite 的实测结果和耗时见实验计划的 P0 记录。GPU/rollout/Gram/反馈/commit 的生产成本剖析仍为 PLANNED；不拿 CPU QP 毫秒数推测总体加速。

本次 CPU 环境：PyTorch 2.13.0+cu130、NumPy 2.5.2、SciPy 1.18.0；测试未执行 GPU 实验。BFCL 官方支持的 `BFCL_PROJECT_ROOT` 指向 writable scratch，避免在只读共享环境创建 `.file_locks`；数据目录仍由 PACKAGE_ROOT 决定。OMP/MKL 单线程仅为 CPU 小矩阵测试降低并行开销。独立 worktree 的旧 AppWorld suite 还需要主 checkout 中被 `logs/` 规则忽略的原始测试 fixture，恢复记录和 hash 见实验计划。另修正一个 pre-P0 HEAD 也失败的历史 manifest-byte 测试：先断言 Azure BFCL merge 未改冻结 scoring projection，再仅归一历史 raw adapter hash；保留原 manifest/梯度 oracle，不改任何 v1.1 runtime。复现证据见实验计划。


## 12. P1 preparation 接线（未运行性能实验）

`tools/rtd_experiment.py` 按 unified 配置派发到 `unified.experiment.P1Experiment`，复用 v1.1 的持久化/账本/benchmark harness。D1 调用原 α/d mixin；D2/D3 与归因臂调用同一 `TeachingObjective` 与 exact solver。D0 通过 `appworld_train.rtd_sft_kl_gradient` 的 AW_DISTILL=rtd_sft_kl 窗口入口提供教师 SFT+source-prefix KL。完整配置、匹配/口径说明和命令见 `RTD_UNIFIED_P1_PREP_ZH.md`。

新增限制为 homogeneous equality E(a−a_ref)=0。D2 对同状态坐标设置相等；fixedmean 对每曝光槽设置系数和固定；nocross 仅替换求解器所用的 Gram，TeachingProblem 保留完整 K；shuffle 在 exact 求解之后随机置换同槽 source pairing。限制与干预均记录在 manifest/campaign，不能把打乱后的系数或对角代理 optimum 称为原完整问题的 optimum。

streamed head VJP 提供 total hard/soft 梯度，再逐来源执行 §4.2 长度修正；D0 的常规 KL 按来源自身长度平均，不声称等于 corrected soft retention。teacher 长度包含真实 native turn/handoff terminator，已有 terminator 不额外加 EOS。所有提交仍为一次冻结 P 的蒸馏更新，没有额外 backbone RL step。CPU 验证不构成 Gemma GPU 显存、总体速度或任务收益的证据。
