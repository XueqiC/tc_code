# RTD v1：完整方法与「总—分—总」执行任务书

日期：2026-09-06。状态：待检验的新方法提案，不是已获实验支持的结论。本文可以整体交给 coding agent 执行。

## 0. 本次要完成的工作

开发并测试 **Return-guided Repair-Transport Distillation，RTD**：以训练后小模型 agent 的任务回报为共同目标，学习如何把学生输出分布中的质量转移到已购买的教师行为，并用同一训练响应决定下一笔教师预算花在哪里。

采用「总—分—总」：先交付完整闭环和完整基线的结果，再交换组件，最后把有效变体装回完整算法。原子、条件对照或任何局部指标的正结果，都不再是运行完整 pipeline 的前置条件。数据泄漏、错误求导、不完整评测等有效性问题仍须先修。

研究对象始终是固定 agent 系统 H 中的 LLM 骨干。最终交付训练后的模型参数 theta；部署时不需要教师、采购器、门控器或额外 agent 模块。

本文提出具体的默认算法和可被替换的求解器。ICLR 是研究目标，不能预先宣称达到录用标准。候选贡献是下面三者的共同作用，而不是给已有元学习、监督学习或概率混合重新命名：

1. 文本黑盒条件下，保持完整输出空间的、依赖学生来源行为的教学目标；
2. 用实际训练更新后的完整 agent 回报学习教学强度；
3. 将新增证据挤占旧证据训练曝光的机会成本纳入购买价值，并在购买前预测它。

## 1. 从最新结果出发，但不把局部失败写成普遍规律

| 当前事实，来自用户提供的报告 | 本版设计回应 |
|---|---|
| 20,991 个候选对主要来自 263 个状态，204 个状态来自一个种子函数 | 有效样本数按父任务、函数和真实请求统计，不能把组合对数当独立教学信息 |
| 严格判据找到绑定混淆 6 对、调用/弃答 49 对 | 不再用混淆判据筛采购候选；未满足该判据的错误仍进入学习 |
| A/B/C 已训完、尚无最终分数 | 完成现有评测并独立归档；不得预设联合目标有效或无效 |
| 稀疏字典和参数编码没有行为预测增量 | 第一版不学习 capability dictionary，不以低秩、稀疏或可命名原子为假设 |
| 一阶指纹能预测局部似然变化 | 保留梯度计算工具，但将目标换成训练后真实任务回报的敏感度 |
| 零分支效用的示范仍能教会 ALFWorld 行为 | 不以 Q、Delta U、成功过滤或阈值决定是否允许教学 |
| 增加事件、增加重复曝光的作用混淆 | 购买价值显式扣除被替代的旧训练曝光；同时报告教师预算与学生计算量 |
| BFCL、AppWorld 曾出现调用/弃答漂移和循环 | 用真实完整 rollout 回报指导目标，而不预设一条边界方向可以修复所有问题 |
| 新教师调用受 429/403 阻塞 | 先跑严格封存响应的回放闭环；真实在线获取作为同接口的后续运行模式 |

既有条件对照实验继续完成：官方评测 41241770、留出评测 41241772/3 的状态以实际队列查询为准，不重复提交正在运行的作业。46 训练对、9 留出对的协议与取消比例上限的修订均保留。9 对来自一个函数，不能支持跨函数总体置信结论。

## 2. 问题定义和资源边界

对目标任务分布 T，给定小模型 pi_theta0、固定 harness H、m 个支持父任务及明确声明的任务池/环境访问。目标为：

\[
\max_{\mathcal A,\phi}\quad
J_{\mathcal T}(\theta_{\mathrm{final}})
=\mathbb E_{x\sim\mathcal T,\,\tau\sim\mathcal H[\pi_{\theta_{\mathrm{final}}}](x)}R(\tau),
\qquad
\theta_{\mathrm{final}}=\operatorname{Train}_{E}(\theta_0,\mathcal D_{\mathcal A},\phi),
\]

\[
\sum_{q\in\mathcal A}c(q)\le B_T.
\]

B_T 是教师预算；E 是预先声明的学生训练曝光，另记录完整 GPU、rollout 与验证开销。R 使用该任务官方终局回报或官方主要成功指标，不设计基于错误类型的奖励。

跨任务通用表示同一程序分别将学生专门化到不同目标任务；不要求一组 LoRA 参数或隐空间方向横跨全部 benchmark。允许 harness 适配器解析各环境的标准动作和评测结果；采购器、目标和学习规则不接收 benchmark ID、错误分类、人工能力标签或手写阶段规则。

### 2.1 不能隐藏在 few-shot 中的资源

每次运行的 manifest 必须分别列出：

- 支持父任务数量 m，以及这些父任务的来源；
- 可用但无教师标签的任务实例、环境和生成器；
- 是否允许在这些任务上反复取得环境回报，以及实际次数；
- 既有教师证据和其历史成本；
- 元学习反馈、开发分析、最终确认与认证分别使用哪些实例；
- 是否存在跨任务预训练 controller。本版默认没有。

默认将支持父任务按哈希分成两折，轮换监督折与回报反馈折。每个反馈块的监督数据不能来自该块反馈折的父任务；后续轮换可以使用另一折，须明确这是 meta-training，不能称为 untouched validation。缓存不足以构建合法划分时，给出实际 m 并标记 exploratory，不从认证集补数据。

永久保留 owned 证据，但每轮构造 D_inner(r)：候选、pending、教师监督、来源 replay、预条件器和特征标准化都只使用当轮 inner 父任务。另一折以前已被模型学过的历史不抹除，因此本轮反馈不是 fresh validation。先确定折角色，再构建本轮统计；本轮没有合法可购包时选择继续训练，不从反馈折补。

BFCL 官方 65 题校准集不得进入训练、梯度、采购或超参数选择。历史反复使用的开发/valid_unseen 数据继续标为开发数据。首轮缓存试验若使用超出 m 的旧池，就标记为历史池方法开发；不能将其包装成已完成严格 few-shot 实验。

## 3. 第一块：用可训练的证据传输替代未成立的原子编码

完整 agent state 为 s=(x, observation history, action history)。y 是模型一次完整动作输出，包括输出终止；环境返回的 observation 不属于被蒸馏 token。多轮教师请求形成多个真实 (s,y) 记录，不把未知未来 observations 当成模型输出。

冻结当前来源策略 p_t。默认每个购买轮冻结一次来源策略；该轮内来源样本、特征、预条件器以及求导中的归一化均不对控制器反传。

对有已购教师证据的状态，令 nu_D(.|s) 为缓存教师文本的经验分布。一个回答时是点质量，多个回答时默认均匀；不要求 teacher logits。

定义来源依赖的门控：

\[
a_\phi(s,y)=\sigma(\phi^\top\chi(s,y)).
\]

特征 chi 默认是：固定初始学生隐藏表示的 32 维随机投影、当前来源输出的对数概率与长度，以及常数项。采用训练侧统计量标准化；固定投影 seed；不采用类别、函数名 one-hot、正确性或配对混淆标签。隐藏表示可编码自然语言语义，但不能将人工错误类型额外注入。采购特征见第 6 节。

教学核为：

\[
K_{\phi,\mathcal D}(y'\mid s,y)
=(1-a_\phi(s,y))\mathbf1[y'=y]
+a_\phi(s,y)\nu_{\mathcal D}(y'\mid s).
\]

没有该状态教师证据时 K 为 identity。这是教师数据可用性的定义，不是任务类型规则。门控不能凭空为没有教师响应的状态生成教师修复。

对应完整动作空间的目标：

\[
q_{\phi,\mathcal D}(y'\mid s)
=\sum_y p_t(y\mid s)K_{\phi,\mathcal D}(y'\mid s,y).
\tag{1}
\]

这没有在有限候选集合中重新归一化学生分布。未被转移的来源质量仍保留在原行为上。

第一版保留原框架第一块的功能——识别何种证据能改变学习——但不坚持把对象命名为能力原子。可记录某个教学强度对反馈任务回报的响应 h_ij；它是局部训练响应，不预设低秩、稳定字典或跨 benchmark 参数方向。

## 4. 第二块：以正权重完整分布目标蒸馏小模型

### 4.1 可直接实现的单一损失

从冻结 p_t 随机采样来源 y^S，而不是反复复制 greedy 输出。对同一真实 state 读取已购 y^T。默认每个监督状态两个独立来源样本，共享教师回答时不重复计算教师成本。

\[
\mathcal L_D(\theta,\phi)
=\mathbb E_{s,y^S,y^T}
\left[(1-a_\phi(s,y^S))\ell_\theta(s,y^S)
+a_\phi(s,y^S)\ell_\theta(s,y^T)\right],
\quad
\ell_\theta(s,y)=-\log\pi_\theta(y\mid s).
\tag{2}
\]

两项都非负。直接计算期望混合权重，不采样一个 Bernoulli 后对离散选择错误反传。候选输出必须匹配完整 state；不能把另一个 state 的回答直接当作本 state 的教师目标。

pi 是真正的完整输出概率，含终止 token；不能以每 token 平均 log-prob 替换公式中的序列概率。长输出成本通过资源预算和采样分布处理，不通过悄悄改变行为概率定义处理。

**首版曝光单位固定为完整动作来源槽位。** 一个槽位含一个来源样本及其对应的教师项；混合权重之和为 1。所有同池组件对照使用同一槽位列表、同一两侧文本、同一计算顺序，因而能严格匹配两侧实际 forward/backward token。不同采购池不保证动作长度相同，须另外做学生 token/compute 匹配比较，不能将等步数称为等 token。

学生 rollout 状态和已购教师 bridge 状态都可以进入监督状态分布。旧池 L_D 默认在当轮 inner 父任务内均匀抽任务，再在其可用的来源/已购状态记录中均匀抽状态；新包 L_q 则在该包状态中均匀抽槽位；去除重复完整 state hash。购买当步严格使用第 5 节的 .75 L_D+.25 L_q，而不是直接合池后均匀抽样；随后的 replay 步才按更新后的 D_inner 抽样。未经购买的候选教师状态、教师前缀与隐藏响应不能进入该分布。状态分布是经验近似，不自动等于 q 的真实 occupancy。

没有任何可用教师项且当前学生就是来源策略时，执行精确 no-op，而不是对少量自采样文本反复 SFT。已发生更新后 identity 项仍有保持来源分布的作用。有限来源采样不能保证每一步不漂移；将此列为实测量。

### 4.2 元学习更新：使用实际 agent 回报

为使第一版推导和代码完全一致，实际学生优化器先采用固定预条件的一步更新：

\[
\theta^+(\phi)=\theta-\eta P\nabla_\theta\mathcal L_D(\theta,\phi).
\tag{3}
\]

默认 P 为训练侧来源 rollout 梯度二阶矩的 RMS 对角预条件器，经数值阻尼、平均对角归一化后整轮冻结。它只负责数值尺度，不称为自然梯度，也不声称任意 LoRA 重参数化不变性。P=I 是同一理论下的简化变体。

学习率 eta 由预训练 pilot 的共同数值程序选定：使用固定 alpha=.5 的监督更新及训练侧来源样本，将初始步幅匹配到预设 KL 尺度；只调数值幅度，不看官方最终分数或错误类型。整个正式轮次固定 eta；计算门控或插入导数时不能随 alpha/epsilon 重新校准。配置需保存 pilot 候选、选择结果及实际 KL，不把 KL 近似当任务回报保证。

pilot 只能使用合法已购证据。首轮 D 为空时，先按采购先验完成首次选择；其 identity 参考更新不依赖 eta，空选择仍是 no-op。首次买到合法包后才进行数值校准，并在计算该块导数前冻结 eta；不能为 pilot 提前解封整个 teacher bank。后续某轮 inner 无可用已购证据时沿用上一轮 eta，不借用反馈折标签。pilot 的所有处理和试更新计算单独计入账本。

更新后的学生从完整任务起点运行，而不是只从教师前缀续跑。用随机 policy rollout 估计：

\[
\widehat g_J
=\frac1N\sum_{n=1}^{N}(R_n-b_n)
\nabla_{\theta^+}\sum_{t}\log\pi_{\theta^+}(y_{n,t}\mid s_{n,t}).
\tag{4}
\]

b_n 采用同一任务另外 rollout 的 leave-one-out 均值，或与该动作无关的基线。排除环境 observations，包含实际采样的所有动作 token 与终止。默认 temperature=1、top_p=1；如果实际采样分布不同，score 必须匹配该分布。训练采样与计分后端不一致造成的误差需记录。

该估计针对随机执行 policy 的任务回报；官方 greedy 评测是独立终点，不声称前者是 greedy 指标的无偏梯度。

对自由门控 a_i，固定来源、P、eta 与曝光权重 mu_i，有：

\[
\frac{\partial J(\theta^+)}{\partial a_i}
=-\eta\mu_i
\left\langle\nabla_{\theta^+}J,
P(g_i^T-g_i^S)\right\rangle.
\tag{5}
\]

式 (5) 是一次实际更新映射的链式法则，不需要将环境回报对动作可微化。它测量教师相对来源行为的**教学更新作用**；不会把 gT·gT 的自相似当成能力证明。

gT、gS、gq、gD 都在该块更新前的当前 theta 计算，不能复用轮初 fingerprint；冻结的是来源策略而非损失梯度。VJP 输入 gJ、reward、baseline 和来源样本必须 stop-gradient，避免多反传一项并非本目标的 Hessian。采购用的 gJ_ref 和门控用的 gJ_actual 取值点不同，见第 5.1 节。

代码用 VJP 计算整个线性门控的梯度，不必存储「全部示范 × 全部参数」矩阵。每个反馈块更新：

\[
\phi\leftarrow\phi+\gamma
\left[(\partial_\phi\theta^+)^\top\widehat g_J-\lambda_\phi\phi\right].
\tag{6}
\]

初始化 phi=0，即 a=.5，正则收缩到该初值。在反馈回报全零且无可识别回报梯度时，门控仍有普通教师监督的启动路径；此时要报告控制器没有学到收益排序。不能改用 Delta U、错误类型或 teacher likelihood 偷渡新的目标。

本次 theta+ 提交成为实际学生；更新后的 phi 用于下一个训练块。不要在观察一次 rollout 后选最好的虚拟学生，并把挑选收益忽略不报。

### 4.3 实际更新而不是形式推导

第一版每轮 12 个实际单步块，共 3 轮；每块 8 个来源槽位，可用梯度累积完成，长上下文不裁掉语义片段。第 1、4、7、10 步是决策窗口：每窗口选择一个新请求包或继续训练；其余 8 步只 replay 当前证据，门控不变。每个决策窗口取得一次实际更新后回报来更新门控，并为购买插入价值额外取得参考更新回报；二者同点时可以复用。每次评估抽 4 个反馈父任务、每题 2 次随机完整 rollout；不足 4 个则使用实际可用数量并记录。此数值配置三个 benchmark 相同。

每轮来源策略冻结，因此式 (5) 在轮内仍成立，但 q 是该轮来源分布的投影目标。更新跨出有效局部尺度时重新冻结来源/刷新数值几何属于预设轮次程序；不按某个能力名称选择刷新。

这些是第一版起点，不是最优剂量承诺。正式结果需记录总动作槽位、两侧训练 token、更新次数、回报 rollout 数和 GPU 时间。36 步的比较不能替代更强训练器的公平调优。

## 5. 第三块的理论起点：购买价值必须包括训练机会成本

设 L_D 是当前证据下的单位曝光损失，L_q 是一个新教师请求包解封后可构成的单位曝光损失。**包**是一个实际教师请求及其依赖、重试和输出形成的整体，不是任意拆出的有效事件。

在固定训练曝光下插入新包：

\[
\mathcal L_{\epsilon,q}
=(1-\epsilon)\mathcal L_D+\epsilon\mathcal L_q.
\tag{7}
\]

两项均按完整动作来源槽位取平均，不按包内事件数求和。epsilon 是新包占下一训练块的曝光份额，定义与实际 sampler 一致。

\[
v_q=\left.\frac{d}{d\epsilon}
J\!\left(\theta-\eta P\nabla\mathcal L_{\epsilon,q}\right)
\right|_{\epsilon=0}
=-\eta\left\langle g_J,P(g_q-g_D)\right\rangle,
\tag{8}
\]

其中 g_J 必须在基准更新 theta_D+=theta-eta P g_D 上估计。

g_q 表示新包教学作用，-g_D 表示被它替代的旧训练曝光。式 (8) 不是任意 sample score，而是与式 (3) 相同训练算子下的局部插入导数。它不等于整个后续学习过程的真实查询 VOI，不包含所有长期顺序解锁；有限 epsilon 的误差要测。

式 (8) 扣除了继续训练旧数据本来会得到的局部收益。但必须注意：当所有候选共享 gD、gJ 和 epsilon 时，这只是候选间的共同常数，不改变「已经决定购买后」的候选排序。

因此把继续训练已有证据作为显式选项 q=empty：L_empty=L_D，v_empty=0，c_empty=0。机会成本影响的是「购买还是继续训练」。它不是候选内部排名的新技巧，也不能单独被声称为原创理论。

第一版每个决策窗口最多一个新包，epsilon=.25，严格执行 .75 L_D+.25 L_q；空选项执行 L_D。两种选择都消耗一个窗口和同样的 8 个实际动作槽位。不能抽到空选项就免费重抽直到购买。包大小通过可提供的状态、行为和教师成本体现，预算可以合理地不花完。

### 5.1 标签究竟如何得到

每个决策窗口固定保存同一个起点 theta：

1. 从旧 D_inner 构造 8 个参考槽位，在 theta 求 gD，得到虚拟参考 theta_D+；用反馈折完整 rollout 估计 gJ_ref。D 为空时参考是 identity/no-op。
2. 用购买前特征和当前 posterior 选择一个新包或 empty。选择新包后解封、扣账，暂放 pending；在同一个起点 theta 求其 gq，用 gJ_ref 计算式 (8)。
3. 真正提交 theta_mix+=theta-eta P[(1-epsilon)gD+epsilon gq]；empty 时提交 theta_D+。不得先提交参考更新，再从参考点多训练一步却称为式 (7)。
4. 从 theta_mix+ 取得新的实际回报梯度 gJ_actual，用它对该次实际混合更新做式 (5)/(6)，phi 留给下一块。empty 时参考与实际模型相同，可复用 rollout。
5. 合并已购包到 D；更新采购 posterior。后续决策窗口使用已揭示的标签，形成轮内学习—再购买。

gD 与 gq 分别由固定的 8 个来源槽估计，实际混合更新使用这两个估计的 .75/.25 加权组合，不另外重采 6+2 个样本。这样式 (8) 的参考和实际插入共享同一个有限样本映射。加权曝光相当于 6 个旧槽加 2 个新槽，但原始处理的是两组共 16 槽；gD 的参考计算可复用。将原始处理 token、梯度计算和权重后的曝光分别记账，不能声称只花了 8 槽计算。

每个 pending 包以请求 ID 计一条观测，不把一个包的响应复制成几十条独立标签。虚拟参考计算与额外 rollout 全部计入学生开销，不能藏进固定训练步数。

对已经训练过的旧包在共同 D 下重测式 (8)，得到的是**额外曝光/重加权价值**。可以作辅助特征校准，但不要与真正的新增包标签混称。若做 leave-one-package-out，必须重算该基准更新及其 g_J；只从梯度里减一项而复用旧 g_J 是近似。

每轮优先用当前学生取得的新标签拟合；历史标签如保留，必须加入学生进度特征并单独报告其陈旧性。默认轮间 posterior 回到同一收缩先验，再用本轮新增包标签拟合。数据少时主要探索是允许且应如实报告的结果。

## 6. 第三块的可执行采购器

### 6.1 请求空间

真实在线请求为 q=(s,L)，s 来自当前学生访问的有效完整状态，L 是最多执行的教师动作轮数。L 默认取整数 1 到 8；每一步依据环境真实返回继续教师动作，不虚构未来 observation。总 output-token cap 是另一项预算参数，按 provider 的实际计费边界执行。

所有合法状态均有机会成为候选，不使用「只有错题」「只有首分歧」「只有正 Delta U」「必须 kappa>0」「必须调用/弃答配对」等过滤。为有限内存可以对状态 reservoir sampling，但不能按手写错误类型改变概率。

候选特征 chi_pre 只能来自购买前：完整 state 的冻结投影、当前学生输出及其概率/长度、L、已购买状态的投影覆盖、当前训练进度和支持集回报摘要。不读取未买的教师文本、teacher gradient、真实响应长度、checker 对教师输出的结果或 hidden cost。

缓存回放只能提供缓存中**实际存在的请求规格**。历史长输出不能先看内容再假装按短前缀价格购买；全包付费后可以在蒸馏阶段选择可用前缀。缓存请求若包含教师前缀，必须先拥有并计入该前缀的成本依赖；拥有前，该完整 state 不能暴露给 selector。

### 6.2 学习购买前的价值分布

第一版使用强收缩 Bayesian linear regression：

\[
v_q=\beta^\top\chi_{\rm pre}(q)+\xi_q,
\quad \beta\sim\mathcal N(0,\lambda_A^{-1}I).
\]

观测是第 5 节的局部插入价值，不是教师分支救活率。采用请求级样本和回报估计误差；样本不足时固定先验噪声尺度并记录，不用少数回报反转拟合复杂 encoder。

预期成本另建回归器，初期用已知 provider output cap 与已购请求长度分布，随后只用已解封实际 usage 更新；历史池无法精确恢复的成本带 exact/estimated 标记。

每个决策窗口从 posterior 抽一次 beta，得到价值样本；把 empty 的价值和成本固定为零，解：

\[
\max_{p\in\Delta}\sum_qp_q\epsilon_q\tilde v_q-\tau D_{\rm KL}(p\Vert p_0)
\quad\text{s.t.}\quad\sum_qp_q\widehat c_q\le b,
\]

\[
p_q\propto p_{0,q}\exp\{(\epsilon_q\tilde v_q-\lambda_c\widehat c_q)/\tau\}.
\tag{9}
\]

默认先验 p0 给 empty 一半概率，其余一半在去重后的合法请求间均匀分配；这是跨任务固定的数值探索先验，不是按错误类别路由。它避免候选组合数膨胀就机械地压低继续训练的概率。R0 使用相同先验与预算程序。

lambda_c 由一维凸对偶求解以满足期望成本约束；tau 是全任务共用的探索温度。b 定义为本轮剩余可用预算除以剩余决策窗口数，不能直接填整个剩余预算。默认使用先验预测标准差对价值尺度归一化、预期成本按合法非空候选均值归一化，令无量纲 tau=1，ridge precision=1。保留全部归一化元数据。

真实预算由独立 ledger 约束：候选请求的可计费上界及尚未拥有的依赖链超过剩余授权预算时不能发出；不能以期望约束替代硬预算。上界来自归档请求的 max-output、动作上限与有限重试上限，或同规格统一的公开保守 cap，不能由隐藏真实 usage 反推。如果因此所有请求不可行，本窗口选择 empty，报告预算点不可执行采购。隐藏真实 usage 由 broker 在选择后扣账；超出预留上界属于账本错误，不能静默超支。

每轮 4 个决策窗口，每窗口最多一个未购请求包；共 3 轮，最多购买 12 个新包，实际可更少。empty 可以在不同窗口再次选择；新包无放回，若确需重问同一 state，必须建立新的真实请求 ID 并重新计费。原始 prior 导致近随机探索不等于算法失效。

式 (9) 是局部收益模型上的随机预算分配，不是全局最优采购解。第一版按固定预算/轮次终止；不把 max predicted value <= 0 称为停止证书，也不对负价值包硬阈值删除。

## 7. 理论能够支持什么

### 7.1 概率质量守恒和 identity

K 的每行非负且和为 1，因此式 (1) 是完整动作空间上的合法分布。a 恒为零时 q=p_t。通过来源—目标耦合，有：

\[
\operatorname{TV}(q(\cdot\mid s),p_t(\cdot\mid s))
\le\mathbb E_{y\sim p_t}a(s,y).
\]

这是目标分布的性质，不是有限样本训练后的无漂移保证，也不是所有非目标行为的逐项保留保证。迁移门控的均值控制一个上界，不保证 student fitting error 足够小。

### 7.2 一个解释而不是另一种实现

式 (2) 可写成来源交叉熵加上加权 teacher/source 差；来源交叉熵在 population 上对应 forward KL(p_t || pi_theta) 加常数。不能将其称为 reverse KL。

代码必须保留正权重混合形式，不能重复优化「精确全分布 KL + 固定有限负来源 CE」形成的有符号伪目标；后者可能与本方法的 population 等式不再一致。

### 7.3 训练响应与采购的一致性

式 (5) 和 (8) 是同一实际单步训练映射的两个方向导数：一个改变同一来源的质量去向，另一个改变固定曝光里训练哪份证据。

在 J 对参数 L-smooth、P/eta/样本固定的局部区域：

\[
|J(\theta^+(\epsilon))-J(\theta_D^+)-\epsilon v_q|
\le \frac{L\eta^2\epsilon^2}{2}\|P(g_q-g_D)\|^2.
\]

这是条件性的局部截断界；L 通常未知，因此不能把它直接变成可计算的安全阈值。实际用少量同起点插入实验检验误差。采用多步 Adam 时必须对真实 unroll 求导，不能沿用单步等式假称精确。

### 7.4 为什么终点必须仍是 student agent

对同一有限时域环境中的因果动作 policy q 与 pi_theta，完整轨迹 KL 满足链式分解。R 在 [0,1] 时，Pinsker 给出：

\[
J(\pi_\theta)-J(p_t)
\ge J(q)-J(p_t)
-\sqrt{\tfrac12D_{\rm KL}(P_q\Vert P_{\pi_\theta})}.
\]

这解释教学目标的潜在收益必须能被学生拟合兑现。该不等式是标准工具的应用，不作为原创定理；黑盒 q 的密度和完整 occupancy 通常不能精确取得，本版不会将该界冒充数值 certificate。经验状态分布训练还存在 occupancy mismatch。

### 7.5 明确不承诺

不证明有限 few-shot 可识别任意未知任务分布，不证明全局最优购买，不证明一定存在可复用原子，不证明任意 teacher text 有益，不证明元回报梯度有足够信噪比。跨任务学习只能依靠共享基座先验、声明过的环境反馈及可检验的泛化。

## 8. 第四块：认证实际学生

冻结最终参数、解码和评分流程以后，在独立数据上评测完整 H[pi_theta]。

第一版输出以下结果，而不是人为 atom certificate：

1. 官方完整任务回报与相对 base 的逐任务修复/损伤表；
2. 独立抽样单位下的总体成功率区间；若 IID Bernoulli 假设成立可用单侧 Clopper–Pearson 下界，否则按父任务/场景聚合后用适当的组级区间；
3. 预先指定的子群诊断，不在看过失败后反复造组并称同时保证；
4. 若研究选择性风险，必须先冻结接受规则，报告接受覆盖率与对应风险界；第一版不需要为完整闭环另造选择性 controller。

BFCL 65 题只能给有限精度的证据；AppWorld 19 场景各 3 变体时，统计单位是场景而非 57 个独立样本。单训练 seed 不产生跨训练种子置信结论；保持用户已决定的单种子设置。

## 9. 先总：第一轮完整 pipeline 怎么运行

### 9.1 三种运行模式必须分开

| 模式 | 用途 | 允许的结论 |
|---|---|---|
| fixed_evidence | 同一批已购文本上的训练器比较 | 蒸馏算子与剂量比较 |
| sealed_replay | teacher cache 由 broker 封存，选请求后解封 | 有限缓存支持上的回顾性采购闭环与预算比较 |
| online | 当前学生新状态上的真实文本教师请求 | 在线状态覆盖、变长购买与真实 teacher-token efficiency |

目前默认 sealed_replay。完整走完候选 → 购买 → 目标 → 训练 → 完整任务反馈 → 再购买 → 最终评测。不能因 API 受限就只跑固定池训练器，但也不能把回放称为完成真实在线 few-shot acquisition。

真实 online 沿用相同 broker 接口；仅使用已有授权的 provider 和预算。429/403 按 provider 的错误处理执行，不能通过轮换凭据绕过限制，也不能无上限重试。若没有可用调用，记录 unavailable 并继续回放工作。

### 9.2 请求级成本

历史 BFCL 成本必须保留：演示输出 1,233,607 tokens 精确计数，生成池约 143k 估计。它们不是本轮新花费，但在端到端成本表中不能变为零。

一次请求拆出的多个事件只付一次整包价格；失败尝试、discarded output、计费 reasoning、teacher verification 均计入对应真实账单。用已有教师前缀建立状态的依赖成本一起计入。区分 input、output、reasoning 和总货币成本，避免将 provider 计费口径混为一谈。

cache broker 应保存 public manifest 与 sealed payload 两份。public 部分只含真实可预先知道的请求信息和成本上界；selector 无读取 sealed 文件/teacher embedding 的路径。审计日志保存 reveal 的时间顺序。

以合法包的 bank 成本 B_bank 为单位，首轮完整试验取累计 10%、25%、50% 三个预算上限，最多 3 轮，每轮 4 个决策窗口。它们是可用上限，不能强制花满；窗口有限、选择 empty 或昂贵包都可能产生余额。横轴按实际花费，另报告上限。百分比是缓存研究配置，不授权新增货币支出，不能拆包凑数。

### 9.3 首先跑两个完整臂

- **R0：随机采购/继续训练 + 固定 a=.5 的混合目标。** 与 RTD 相同来源、状态接口、学生优化器、轮数、先验 p0 和资源上限；采购在同预算程序下不使用学习价值，支持同样的 L 集合。
- **R1：完整 RTD。** 回报学习门控 + 曝光守恒插入价值 + Bayesian 采购；其余相同。

这两个臂必须都走完整个闭环。R0 是结构对照，不是最终强基线。即使 R1 暂时低于 R0，仍交付完整轨迹和结果；再按组件交换定位问题，不退回无限诊断。

首份结果同时附一个复用现有强配置、在同一合法已购集合上训练的 text-only CE 或 base-centred pairwise 终点评测。无需在首轮启动全部强对照，但仅胜过 R0 时结论只能是「优于固定混合闭环」，不能写为提高了现有最强学生。

先在 BFCL 跑完两个臂；随后在 ALFWorld 用同一程序跑两个臂，不要求 BFCL 的门控、排序或局部指标先显著。AppWorld 在程序和配置冻结后迁移一次完整闭环，不继续扩旧 pairwise-spread 配方。

若历史池缺少独立父任务、真实可重放请求或成本依赖，仍可完成合法子集的闭环，但报告明确限定。需要新覆盖的部分留给真实 online，不能从认证数据填补。

### 9.4 完整运行伪代码

```text
输入：固定 harness、theta0、支持父任务两折、合法候选接口、预算 ledger
初始化：phi=0；采购先验；theta=theta0；D=空或明确计费的 seed evidence
for 购买轮 r = 1..3:
    先轮换 inner/feedback 父任务折，创建 D_inner 与合法候选视图
    再冻结来源 p_r、inner 特征统计和预条件器；固定该轮 eta
    生成本轮合法来源状态，或读取依赖已拥有的 sealed 请求目录
    for 实际更新 step = 1..12:
        if step 在预定窗口 {1,4,7,10}:
            保存起点 theta；用旧 D_inner 计算虚拟参考 theta_D+
            从参考模型的完整任务回报计算 gJ_ref
            用购买前特征与 posterior 在新包/empty 中选择一次
            if 选择新包:
                broker 解封扣账；同起点求 gq 和 insertion 标签
                实际提交同起点 .75 gD + .25 gq 的正混合更新
                取得实际模型完整回报 gJ_actual，更新门控
                合并新包到 owned/D_inner；更新 posterior
            else:
                提交参考模型，复用其 gJ 更新门控
        else:
            用 D_inner 执行一次正混合 replay 更新；保持门控
    保存 checkpoint、来源状态、完整 ledger 和元学习反馈轨迹
    运行预定完整开发评测，不据此临时增加事件轮次或调整错误规则
冻结选定配置；进行独立确认/认证；输出三个预算点的完整结果
```

每窗口只提交一个学生更新。虚拟参考和额外回报是附加计算，单独记账，不伪称为额外提交的训练步；D 为空的 no-op 窗口仍消耗决策机会并记录计算，但不伪造有监督训练曝光。

## 10. 再分：用最小交换矩阵定位贡献

### 10.1 采购与蒸馏的 2×2

先保存 R0 和 R1 的真实购买 ledger。默认取各自最终已购完整集合，从相同 theta0 按相同两折与曝光日程重训全部四格，冻结证据可用集，允许各训练器按相同规则重采来源。不要把原始自适应 R0/R1 直接当两个对角线、只补非对角线。逐次 reveal 回放可作以后另一个实验，不与此固定集合矩阵混写。

| 固定购买 ledger | D0 固定 a=.5 | D1 学习门控 |
|---|---|---|
| A0 随机采购 | Y00 | Y01 |
| A1 RTD 采购 | Y10 | Y11 |

Y01-Y00、Y11-Y10 是给定证据下的蒸馏收益；Y10-Y00、Y11-Y01 是给定训练器下的采购收益；交互项是 Y11-Y10-Y01+Y00。

全在线 R0/R1 包含适应性状态改变；该固定 ledger 矩阵研究组件条件作用。两者不能互相替代。无需交互项大于零才承认两个组件分别有价值。

### 10.2 每次只换一个组件

| 组件 | 默认 | 第一替换 | 第二替换 | 理论关系 |
|---|---|---|---|---|
| 教学强度 | 来源依赖线性 gate | 单一可学习 scalar gate | 固定 a=.5 | 同一 K 的嵌套限制，检验内容信息是否超过全局剂量 |
| 来源目标 | 真实来源质量保留 | a=1 教师监督 | 打乱 gate 特征对应 | 同一目标族的端点/破坏结构对照 |
| 更新响应求解 | 固定 P 一步实际更新 | P=I 一步 | 3 步 functional AdamW 真正 unroll | 同一外层 J，不得把一步导数套到 Adam |
| 采购决策 | posterior sampling | posterior mean | 随机 | 同一预算请求空间；检验信息是否可在购买前使用 |
| 插入价值 | g_q-g_D，含 empty | 只用 g_q，含 empty | 相同标签随机置换 | 前两者只在购买/复用分配上有区别；共同基准下非空候选排序恒等 |
| 教师长度 | 联合选择 s,L | 固定 L=1 | 固定 L=4 | 同一请求空间的限制；缓存不支持时只在线测试 |

所有变体保存母版理论目标、替换接口和新近似。如果某变体改用 teacher likelihood 外层靶，只能标为 surrogate ablation，不能继续声称直接优化任务回报。不要一次运行全表笛卡尔积。

第一优先替换是 **learned scalar gate**：它直接检验 learned feature gate 是否只是换一种全局强度调节。第二优先是采购 2×2。第三才是多步 solver 或更复杂 encoder。

### 10.3 必須加入的强对照

完整探索阶段可以先跑 R0/R1；在宣称性能或方法贡献之前，至少补：

1. 随机学生状态 + 短教师 continuation + 调好的 text-only SFT；
2. 同池、调好的 base-centred pairwise；
3. teacher SFT/固定混合 + 同等环境反馈的直接 policy-gradient 更新，或等计算 RL 基线；
4. 使用干净支持反馈的普通元重加权，而不做来源质量传输。

第 3 项不可省略：RTD 消耗真实环境回报和额外 rollout，必须排除收益只是更多反馈/计算。对照可以各自合理调优，但 RTD 不得按 benchmark 手工切 loss。优先复用现有强训练配置，同时单独给出相同优化器的归因对照。

不同采购池动作长度不同，给出两个视角：固定教师预算 + 固定完整动作槽位；固定教师预算 + 固定学生训练 token/compute。第二视角可用带曝光校正的采样实现，必须写明其采样分布，不能无声换成逐输出长度归一化概率。

## 11. 最后总：如何根据结果更新整个方法

| 观察 | 可支持的解释 | 下一次完整版本只改什么 |
|---|---|---|
| R1 > R0；2×2 两条主效应均正 | 目标与采购各有收益 | 保持骨架，测试固定长度和更大预算 |
| Y01>Y00，但 A1 对 A0 没有增益 | 蒸馏有价值，购买前预测尚弱 | 保留 D1，换采购估计器/请求覆盖；不改蒸馏定义 |
| Y10>Y00，但 D1 无增量 | 新证据更好，门控未增加收益 | 保留 A1，使用 scalar/固定 gate 对比，再改求解器 |
| learned scalar = feature gate | 当前证据只支持全局剂量信息 | 第一块收缩为可学习传输强度，不强称 capability geometry |
| gate 降低局部损失却未提高完整回报 | surrogate/外层估计或 occupancy 有问题 | 检查实际回报 VJP、支持覆盖、来源刷新；不换成错误规则 |
| 学习效应只有随机执行得分，greedy 无提升 | 采样 policy 与部署目标不同 | 报告目标错配；下一版统一部署采样或研究有明确定义的 greedy 代理 |
| ALFWorld 回报全零、gate 不动，但普通监督开始有效 | 教师 bootstrap 存在，元学习暂未启动 | 继续预定完整轮次；不声称 gate 已识别证据 |
| a 接近 0，R1 仅仅保持 base | 学会少更新，不是高性能蒸馏 | 与 scalar、KL、低 lr 强对照比较；不能据此完成方法主张 |
| acquisition 只偏向便宜包 | 价值模型弱或成本占主导 | 对比便宜优先和等曝光，检验每包价值校准 |
| 回放胜出、online 不胜出 | 缓存条件选择偏差/新状态预测失效 | 修前瞻特征与在线估计，不能以回放替代在线结论 |

每次修改形成 v1.1/v1.2，保留相同外层 J、请求成本和曝光定义；组件分析完成后必须再跑新的完整组合。没有理论关联的技巧只能列为独立 baseline。

## 12. 三个 benchmark 的具体问题

**BFCL。** 首先完整回放，以官方 checker 判定完整模型输出。采购/门控不接收错误类型。分解调用率、调用数、参数值、类型与格式仅用于事后解释；不能把调用率变化本身当条件决策改善。当前几乎单函数的生成池不足以支撑泛化采购，应把父函数覆盖不足写进结论，并在真正可用的任务池上补在线请求。

**ALFWorld。** 用同一算法检验低初始能力时正教师监督能否启动，以及回报反馈出现后是否能提高数据利用效率。既有强 CE 是必须正面对比的性能锚点，不能因新方法形式更统一而省略。不能预期「只有边界原子」，不使用类别特定 loss。完整任务 success 是主指标，一阶 log-likelihood 迁移只是诊断。

**AppWorld。** 在冻结的 RTD 程序下跑一轮完整配置迁移。状态来自实际执行，全阶段由 sampler/请求器处理，无 early/spread 手写路由。使用官方执行和终局评测；循环、终止次数、state-based tests 仅为诊断。旧 A=12/57 不作已建立正锚点，固定同机比较并按场景聚合。若 cache 不支持在线实际失败状态，标记请求覆盖限制，不用最近邻教师回答替代。

所有 benchmark 保持单训练 seed；各臂使用相同设备/软件、相同任务清单与评测参数。bootstrap 用于任务采样的不确定性，不声称重复训练稳定性。

## 13. coding agent 实现任务书

你需要在现有仓库实现并跑完本文件定义的 RTD v1。先核查仓库 AGENTS、当前分支与作业状态；复用已有训练、checker、事件 schema、缓存和完整性检查。下列路径是建议的新模块，不表示当前已存在。

### T0：锁定协议并接收既有结果

- 保存 `docs/rtd_v1_protocol.md`、数据访问表和冻结配置。
- 检查并接收既有 A/B/C 评测；不覆盖其 checkpoint 或配置。
- 核对父任务/函数/state/request ID；以 prompt/state 内容哈希处理旧 ID 复用。
- 在正式训练前写明 m、两折映射、teacher bank 包边界、预算与 3 轮曝光安排。

### T1：请求包与封存 broker

建议 `src/bfas/rtd/broker.py`、`ledger.py`。提供：

```python
list_candidates(student_snapshot, owned_ids, remaining_budget) -> PublicQuerySpec[]
acquire(query_id) -> PurchasedEvidencePackage
assert_no_hidden_access(selector_trace) -> AuditResult
```

PublicQuerySpec 不含隐藏教师文本、真实未公开 usage 或教师验证结果。返回包记录请求依赖、完整 usage、成本置信标签和真实响应。没有合法请求重放记录时返回 unavailable，不能造记录。

### T2：来源传输目标

建议 `transport.py`、`features.py`。实现完整 state 校验、随机来源采样、冻结特征、正混合 CE、identity/no-op、完整输出 log-prob。保留目标构造与 student loss 的分离接口。

### T3：真实更新与回报 VJP

建议 `functional_step.py`、`return_gradient.py`。实现实际固定 P 一步更新、完整任务随机 rollout、与生成一致的 likelihood 重算、门控 VJP、lambda 正则和固定反馈间隔。P/eta/source/normalization stop-gradient 必须显式，不能依赖隐式 optimizer 状态。

不将旧三步 Adam 的参数差直接当作 -eta P g，也不把旧 Fisher/sketch 的成功当作新回报梯度有效。

### T4：插入价值与采购

建议 `insertion.py`、`acquisition.py`。对 pending 请求包计算曝光守恒导数，Bayesian 线性 posterior、成本模型、对偶求解和随机选择。实现 `pending_new` 与 `reweight_existing` 两种标签类型，禁止混写。

### T5：总—分—总 runner

建议 `tools/rtd_experiment.py`：

```text
audit / smoke / run / resume / replay-ledger / swap-component / evaluate / report
```

实现 R0/R1 完整三轮、固定 ledger 2×2 与单组件覆盖配置。幂等 resume，checkpoint/数据/配置/硬件 hash 一并保存；不能因目录同名便假定评测属于本模型。官方评测所有期望 task ID 完整后才能生成 aggregate。

### T6：有意义的正确性检查

先做小规模数学和数据正确性检查，通过后直接完整运行，不另设性能 Go gate：

1. 有限动作 toy 上 q 的归一化、a=0 identity、a=1 teacher 端点及候选外质量；
2. 正混合损失的 MC 期望与显式 q 交叉熵一致；
3. 双精度小模型上 gate VJP 和 insertion 导数与中心差分一致；
4. 当 g_q=g_D 时固定曝光插入导数为零；不扣 g_D 的对照产生错误增量；
5. 来源采样策略、teacher-forced score、EOS 和 masks 一致；
6. 未购买 payload 不可进入特征；同包拆事件只计一次，依赖先购买；
7. no-op/恢复初态、参数更新与评测完整性检查。

toy 测试只验证实现，不证明 benchmark 性能或原创新定理。

### T7：必须交付的文件和表

- canonical method、完整配置和不可变 protocol；
- 每次真实 request/reveal/usage、每次训练曝光、回报 rollout 的 ledger；
- R0/R1 三轮预算曲线与完整 final checkpoint；
- 已有 A/B/C 的独立结果及其解释边界；
- 固定 ledger 2×2；首个 scalar gate 变体；
- 当前最强 text-only baseline 和同反馈 RL/meta-reweighting 对照的计划及结果状态；
- 逐任务修复/损伤表、缺失任务清单、置信区间单位；
- 明确分开的 observation / hypothesis / implementation issue。

不可用的 online teacher 应报告阻塞，但继续完成已授权的代码、数学检查、缓存闭环和分析。不要用「等待原子显著」代替完整运行。

## 14. 建议冻结配置模板

以下是初始数值方案，不是已调好的最优设置。pilot 只解决可测性、内存和步幅，修改须写入版本；同一组件比较保持一致。

```yaml
method: rtd_v1
mode: sealed_replay
training_seed: 0
same_hardware_required: true
harness_frozen: true
teacher_access: text_only
support_split: parent_hash_two_fold
certificate_isolation: strict

rounds: 3
decision_steps_per_round: [1, 4, 7, 10]
max_new_packages_per_decision: 1
replay_is_explicit_choice: true
replay_prior_mass: 0.5
budget_checkpoints_bank_fraction: [0.10, 0.25, 0.50]
online_max_teacher_turns: 8
replay_request_lengths: recorded_only

source_refresh: round
source_samples_per_state: 2
source_temperature: 1.0
source_top_p: 1.0
loss: positive_source_teacher_mixture
behavior_probability: complete_action_with_termination
exposure_unit: complete_source_action_slot
slots_per_step: 8
committed_steps_per_round: 12

gate: linear_sigmoid
gate_hidden_projection_dim: 32
gate_initial_logit: 0.0
gate_ridge: 1.0
gate_learning_rate: 0.1
gate_gradient_scale: running_rms_from_past_feedback_only

optimizer: fixed_preconditioned_single_step
preconditioner: train_only_rms_diagonal
preconditioner_refresh: round
preconditioner_damping_relative: 0.01
preconditioner_mean_diagonal: 1.0
step_size_selection: frozen_training_only_kl_pilot
pilot_kl_target_per_step: 0.005
meta_feedback_steps: [1, 4, 7, 10]
insertion_reference: virtual_same_start
extra_reference_compute_accounting: separate
meta_tasks_per_feedback: 4
rollouts_per_meta_task: 2
reward: official_terminal_metric
baseline: leave_one_out_same_task

acquisition: bayesian_linear_posterior_sampling
acquisition_prior_precision: 1.0
acquisition_prior_noise_variance: 1.0
acquisition_entropy_temperature: 1.0
acquisition_value: exposure_conserving_insertion
insertion_fraction: 0.25
acquisition_posterior_refresh: round
hard_cost_reservation: true
```

若初始 gate 梯度尺度为零，running RMS 只用数值 epsilon 防除零；不得根据成功率切换另一套训练目标。若任务数不足、包太贵或上下文超出模型上限，记录实际资源限制；上下文处理须与固定 harness 协议一致，不能只为某个臂裁掉关键观察。

## 15. 文献关系与可检验 novelty

元梯度赋权已经存在。[Learning to Reweight Examples](https://arxiv.org/abs/1803.09050) 使用验证梯度决定训练样本权重；不能把式 (5) 的梯度对齐形式单独当原创。

近期 [BLADE](https://arxiv.org/abs/2606.18650) 已研究 LLM 训练的双层自适应数据选择。因此「bilevel + model-specific data selection」也不是足够的 novelty claim。本版差异候选是黑盒未购买请求、来源质量传输、完整 agent 回报与固定训练曝光的共同建模，仍需正面对照。

[A Few Teacher Steps Go a Long Way](https://arxiv.org/abs/2607.04574) 已验证学生访问状态上的短、不过滤教师 continuation 在预算匹配下可以很强。本版必须打这个基线，不能把 on-policy states 或短监督长度本身据为己有。

[FutureBridge-OPD](https://arxiv.org/abs/2608.01953) 已通过教师 bridge 后的学生续跑评价指导的后续作用。本版要检验的区别是证据**经过参数学习后**的完整 agent 回报，并非仅评估当前学生接手一次 bridge 的效果。

[TurnOPD](https://arxiv.org/abs/2607.05804) 与 [TIDE](https://arxiv.org/abs/2608.09836) 是必须核查的相邻 agent/distillation 工作；在实现比较前逐条标注教师访问权限、预算和训练目标，不把需要 teacher logits 的原版直接放入 text-only 同权限表。

本版有希望成为论文中心的可证伪陈述是：

> 在少量目标任务反馈与有限文本教师预算下，学习学生来源行为向教师行为的传输，并以固定训练曝光下的任务回报插入价值购买后续证据，能提高蒸馏后小模型 agent 的 in-task performance / teacher-token efficiency。

支撑它需要：完整方法优于强基线；同反馈/计算不能解释收益；来源依赖信息优于 scalar 剂量；购买前预测改善真实购买；至少两个任务环境中用同一程序复现。某组件被更简单变体替代并不自动否定整篇论文，但应相应缩小贡献。

ICLR 关注技术正确、证据支持和有意义的新知识，并不要求每个模块都发明一套全新理论。[ICLR 2026 Reviewer Guide](https://iclr.cc/Conferences/2026/ReviewerGuide) 不构成预先确认本方法达到标准的依据。本轮应交付一套能完整被检验、能逐块改进的方法，而不是预设实验会证明它。
