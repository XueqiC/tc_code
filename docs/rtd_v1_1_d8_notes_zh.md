# RTD v1.1 D8：rev 3.1 的状态 α + 联合来源选择 d

本次以完整 `RTD_V1_1_METHOD_ZH.md`（含 §2.3、§3–3.4、§6.1、§7、附录 C/D）为规范，在 **D1 + D2/D3 的 M1 合并树**上实现。只修改 `/home/xueqi/hq/projects/tc-alignment-v11a`；共享 `data/`、`envs/`、Python 环境只读；没有 GPU、在线教师、正式训练、官方评测或 commit。已有的 `PROJECT_STATE.md` 用户修改保留。

## 1. 文件与责任

| 文件 | 本次变化 |
|---|---|
| `src/bfas/rtd/alpha_d.py` | 配置/臂验证、状态门控、合法 q、单槽估计器、成对来源对象、同批次参考、CPU 保存的方向、H-Gram、联合坐标求解器、逐轨迹反馈重投影、分块 α VJP。 |
| `src/bfas/rtd/experiment_alpha_d.py` | 新版本 runner：监督记录映射、先冻结后重采、4 状态 microbatch、同批次参考与提交、独立 α 反馈、另行记账的历史采购标签、步级曝光及恢复。 |
| `src/bfas/rtd/experiment.py` | 仅在 v1.1 且显式 `gate_mode` 时分派 D8；33 维状态 α（32 维状态投影 + 截距）或标量；反馈可输出逐轨迹梯度；新增曝光不变量与提交后临时状态清理。 |
| `src/bfas/rtd/experiment_v11.py` | 复用 D2/D3 的选择、窗口预算、reserve/reveal 与逐事务存盘。D8 在全部采购完成后采样；原 D1 包采样/校准代码原序移入 `batch_prepare_package`，诊断路径保留。 |
| `src/bfas/rtd/return_gradient.py` | `reinforce_gradient(..., trajectory_scores=list)` 可选输出未乘回报的逐轨迹 score 梯度，CPU/detach 保存。未传该参数时，历史梯度运算顺序、结果与返回字段不变。 |
| `src/bfas/rtd/source_estimator.py` | 配置入口识别 `alpha_d`；hard2/soft/CV 的既有梯度实现不变。D8 直接复用 D1 `source_gradient_pair`。 |
| `src/bfas/rtd/config_v11.py`、`cli.py` | 新键验证、V0/V1/V2 臂约束、完整/缩减曝光 manifest、return objective/分块导数/采购参考声明。旧配置不插入新默认值。 |
| `configs/rtd/v1_1_bfcl.yaml` | 用 rev 3.1 替换 M1 的 CV 占位配置：两轮、40 单元、K≤20、microbatch=4、状态 α + 联合 d。四份 v1.0 C25 YAML 未修改。 |
| `tests/test_rtd_v11_alpha_d.py` | D8 数学、D1 实际组件、执行顺序、曝光、重采、反馈重投影、warm-up、配置/manifest 与恢复测试。 |
| `tests/test_rtd_v11_merge.py` | 仅更新规范 YAML 断言；原 soft/CV 混合梯度与恢复测试继续保留。 |
| `tests/test_rtd_v11_bank.py` | 按 rev 3.1 更新规范 YAML 的缩减曝光断言：`slots_per_step=8` 现在是合法且明确标记的随机曝光变体；其余预算、bank、旧 manifest 回归不变。 |

## 2. 公式 → 代码

### 注入量、目标与估计器

`state_features(FullState, initial_state_projection)` 的输入类型拒绝 `SourceSample`。完整 χ 只含发行初始学生在 **状态 prompt** 上的冻结隐藏投影与截距，不含来源动作的长度、似然、梯度投影、类别或错误类型。未使用 D1 的 35 维来源特征标准化器来产生 α；该旧标准化器仍供历史采购特征/诊断路径使用。标量 χ=[1]。

```text
injection:       α_i = sigmoid(χ_i · φ)，或 fixed_alpha
target_weights:  [(1−α_i−d_i)/2, (1−α_i+d_i)/2, α_i]
target_distribution: 上述权重乘 δ_Y1、δ_Y2、ν_T
estimate:        ĝ_i = (1−α_i) Sbar_i + α_i T_i − d_i(H_i1−H_i2)/2
```

`components` 分别对两条来源调用 D1 `source_gradient_pair`，返回硬 NLL 梯度 H1/H2 与两个完整词表软来源梯度的均值；教师通过原 `score_behavior` 的完整动作 NLL 求导。保留 D1 的 EOS/cap 边界与逐位置完整词表计算，无格式过滤。

α 与 w 在所有本步来源抽样前冻结。每个状态的每次曝光都独立调用冻结 `p_t` 两次，即使同状态在本步重复出现也不复制来源对。`sample_state(refresh=True, cache=False)` 不读取轮内来源缓存作为提交目标。日志同时保留 draw ID、RNG 前后摘要、完整样本 hash 与 token hash；RNG 必须前进，返回对象不能来自缓存。独立抽样的 token 可以相同，**不要求 token hash 不同**。恢复重算未提交阶段会留下重复计算日志，但不会重复提交或把旧步来源当作新抽样。

对相同快照，软项在每个前缀精确零梯度；整体为 `αT−d(H1−H2)/2`，通常仍会移动学生。`α=d=0` 保证整个更新为零；不把退化模型里偶然的梯度抵消解释为门控性质。理论无偏性固定 α/w，仅替换与当前样本无关的软来源基线；d 可以同时依赖 Y1/Y2。

### 同批次更新与分块 α 导数

`build_reference` 在同一 start 上按每 4 状态计算梯度，所有项乘预先冻结的 w（和为 1），最后只构造一次虚拟参考：

```text
theta0 = θ − ηP Σ_i w_i[(1−α_i)Sbar_i + α_iT_i]
v_i    = (ηw_i/2)P(H_i1−H_i2)
actual = BatchReference.selected(d) = theta0 + Σ_i d_i v_i
```

方向与单槽基线梯度 detach 后放在 CPU；不会同时保留多个 LM 计算图。H-Gram 与点积逐参数块在 CPU float64 累加。此实现需要存储 O(E×LoRA 参数数) 的 CPU 方向，加上逐轨迹反馈统计；不声称这是压缩草图或已经测得生产吞吐。D1 的模型前向仍是一条动作图一次。

`selected(d)` 对 α/d 停梯度。`blocked_alpha_vjp` 在本步 **实际已提交** θS(d) 上的独立回报梯度处，重新评分相同来源对和教师记录，计算

```text
∂θS(d)/∂α_i = −ηw_i P(T_i−Sbar_i)
∂J/∂φ = Σ_i [gJ(actual)ᵀ ∂θS(d)/∂α_i] α_i(1−α_i)χ_i
```

φ 使用原 `GateController` 更新。这里固定求解后的 d，不求 `∂d*(α)/∂α`，也不声称穿过参考回报梯度对 α 的依赖。非线性外层回报的中心差分测试固定同一个 d。任意浮点参数上的仿射等式按精度误差检查；二进制可精确表示的 tiny 参数另用 `torch.equal` 检验严格等式。

### 联合 d 求解

```text
H = P^{-1}
K = gram(v, P)，K_ij = v_iᵀHv_j
z_i = gJ(theta0)ᵀv_i
maximize zᵀd − d_lambda/2 * dᵀKd − epsilonᵀ|d|
subject to |d_i| ≤ min(α_i, 1−α_i)
```

`solve_d` 是全坐标循环的精确一维更新（soft-threshold 后投影到 box），以整个目标的 proximal/KKT 残差检验收敛。矩阵必须对称半正定；λ=0、零方向、α 端点都支持；达到迭代上限仍未收敛直接报错，不提交未验证解。

没有按原始 z 做预筛，也没有 `|z_i|≤epsilon_i ⇒ d_i=0` 的断言。零点为内部点时应检查 `|z_i−λΣ_{j≠i}K_ij d_j|≤epsilon_i`。测试包含 K=[[1,.9],[.9,1]]、z=[1,0]、ε=[.1,.1]：第二坐标虽原始 z=0，仍取负值抵消第一方向成本。交换一个来源对会翻转该方向、相应 K 行列、z/d 符号，提交保持不变。

### 反馈刷新、误差与年龄

决策步在 theta0 上刷新一次 source-selection 回报梯度。`FeedbackStatistic` 保存 CPU 上的聚合梯度、逐轨迹未乘回报的 score 梯度、reward、task ID、baseline 类型、参数 hash 和刷新步号；非决策步不生成新反馈，沿本步 **新** v 重算 z 与 ε 并重新联合求解。

- `loo`：同任务至少三条轨迹时逐条删除，并重新计算剩余轨迹的 same-task LOO baseline，然后用 jackknife 方差估计均值误差。只有两条轨迹时无法删除后再拟合 LOO，明确使用 paired-contribution 误差代理。
- `split_half`：任务内按抽样次序分成两半，每半使用另一半的平均 reward 作独立于本半动作的 baseline；两半投影均值差换算为标准误代理。
- 单任务只有一条轨迹（smoke）时 ε 不可估计；日志为 null、solver reason=`insufficient_trajectories`，该步 d=0。
- 这些都是给定任务集的 Monte Carlo 误差估计，**不覆盖反馈陈旧偏差**、来源方向抽样噪声或未观测 reward 模式；不宣称是可靠置信界。对称性、方向缩放、重投影测试不等于噪声界实证验证。

## 3. 曝光单位与冷启动边界

默认每次提交 **40 条教师监督记录 + 80 个来源动作**，最多 20 个新包各提供一条记录，其余从合法、已付费、当前 inner fold 的旧池有放回抽取；4 状态一 microbatch，共 10 次后构造和提交一步。每条记录 w=1/本步记录数；D9 只优化 d，不改变位置份额。

adapter 可实现 `support.supervision_records(package)`，必须按固定顺序返回 `Behavior`。默认是已有 `package.behaviors` 的顺序：BFCL 已验证状态续写记录、ALFWorld episode 内各 command/state 的监督记录。新购买 episode 本步只选一条，不将整个 episode 冒称为一条动作；提交后全部已付记录进入旧池。记录包含 package ID、record index、完整 state hash、教师文本、权重和重复次数。

**空旧池不能制造监督记录**。E=40、K≤20 与首窗旧池为空无法同时满足；满额配置遇到此情况在提交前明确报错，已支付包仍由账本及 `revealed` recovery 保存。需要 runner 的已有且已付费 inner bootstrap 池；不得为补池免费读取 sealed bank。此保护同样适用于折旋转后合法旧池为空的情况。D8 没有偷偷把新包重复称为旧池，也没有把 source-only no-op 称为教师曝光。

显式 `slots_per_step<40` 是 **`random exposure`**。若空旧池只有少于该上限的已购监督记录，只训练这些真实记录；若购买数超过实际曝光容量，随机选出训练包，其余仍付费、入池、另列 `unexposed_purchased_packages`，不伪造标签。没有一条教师记录则不提交。manifest 始终不作“本步所有实购包均已训练”的全局承诺，实际完整情况由每步记录给出。低配 CPU 集成测试使用这种明示变体；40 单元测试使用正常账本购买的 bootstrap 旧证据。

## 4. 固定执行顺序与恢复

1. `alpha_step_start → batch_reference → batch_selected`：历史采购后验选择，D2/D3 Q/K 硬账本 reserve/reveal，逐请求保存。D8 不在选择前调用当前窗回报或旧 `remeasure_owned`。
2. `alpha_revealed`：全部揭示后选择本步监督记录，固定 w/状态 α，再重新抽两条来源。首个有证据的训练 KL pilot 有独立 role，不使用回报。
3. 在本批构造 theta0；决策步使用 `source_selection_feedback` 的温度 1 rollout；非决策步重投影最近反馈；求解 d。
4. 同一 start 上安装 `theta0+Σd_i v_i`，保存实际学生后进入 durable `actual` 阶段。theta0 从不先安装成一个学生优化步。
5. `alpha_actual`：在已提交学生上另外生成 `alpha_post_commit_feedback`，计算固定 d 的分块 α 更新。即使 d=0 导致两个参数 hash 相同，也不按 hash 复用两批反馈。
6. 决策窗另行计算 `acquisition_surrogate_feedback`，更新历史采购标签，最后 `feedback → committed` 保存轨迹并关账本窗口。

采购参考保留为 **另一个 `InsertionReference`**：从本步 start、旧池的均匀教师注入基线建立 old-only 参考，使用新包基线梯度形成 D2 一阶可加插入标签。它在采购选择之后测量，供以后窗口的历史后验；即使碰巧与 theta0 hash 相同也有独立 role/反馈成本。当前并未把它冒称为 D9 完整集合 Fhat 或独立配对收益。旧 D2 drift/可靠性测量流程没有偷接到 d 的参考上；D9 后续应在独立采购参考角色中恢复这些诊断。

持久化状态分别保存 `d_reference`（本批）、`reference`（采购）、`d_feedback`（可重投影历史）及角色分明的反馈。只有真正提交的学生参数写入 backbone；未提交阶段恢复时 RNG、来源快照和账本事务从 checkpoint 还原。`actual` 阶段恢复直接还原已提交参数，后续 φ/标签阶段不重复安装一步。提交后清除本批 pairs、directions、d 解与梯度，长期只保留正权重曝光记录、φ/后验与可重投影反馈；无持久化 signed-CE 训练数据。

## 5. 配置

| 键 | 默认/约束 |
|---|---|
| `gate_mode` | `learned_alpha`；`fixed_alpha` / `learned_alpha`，同时作为 D8 显式入口。 |
| `fixed_alpha` | .5；合法区间 [0,1]。V0 必须 .5。 |
| `gate` | 原 `linear_sigmoid` 表示本次 33 维状态门控；`scalar_sigmoid` 是单截距。 |
| `source_estimator` | D8 为 `alpha_d`；两个来源样本；不与 `cv_cs_mode` 或旧 `gate_override` 混用。 |
| `d_mode` | `learned`；可选 `zero`。 |
| `d_lambda` | 1.0，有限非负。 |
| `d_warmup_windows` | 1；从运行开始累计决策窗口计数，前若干窗口及其非决策步 d=0。 |
| `z_error_mode` | `loo` / `split_half`。 |
| `d_solver_tolerance` / `d_solver_max_iterations` | 1e-8 / 10000。 |
| `d_redundancy_cosine_threshold` | .9；统计 H-cosine 超过该阈值的无序方向对比例，零方向对 cosine 记 0。 |
| `exposure_slots_per_window` | D8 固定 E=40。 |
| `max_new_packages_per_window` | 默认20，0≤K≤20。 |
| `slots_per_step` | 默认40；1–39 明确标记 `random exposure`。 |
| `microbatch_states` | 固定4。 |

V0 必须 `fixed_alpha=.5 + d_mode=zero`，采购执行器按 V0 使用相同成本约束下的随机选择；V1/V2 必须 `learned_alpha + learned`。CLI 接受 V0/V1/V2 并验证这些组合。规范 YAML 是学习机制配置，运行 V0 时须显式提供对应配置。**仅选择 V1 名称不等于已经完成 V0 曝光重放**：manifest 写 `v1_exposure_replay_implemented=False`，D7 负责重放。D8 不宣称 V1−V0 隔离 d；应称为自适应蒸馏整体效应，同 α 的 d 对照才隔离来源选择。

## 6. 日志字段

| 事件/字段 | 含义 |
|---|---|
| `alpha_d_window_order` | 进入历史采购时的轮/步与已有后验观测数。 |
| `alpha_d_exposure_frozen` | `before_source_sampling=True`、逐记录身份、state features、w/α；role 区分 pilot/commit。 |
| `alpha_d_source_pair` | 两个 draw ID、sample/action hashes、来源快照、RNG 前后摘要、`cache_reused=False`。 |
| `alpha_d_microbatch` | 每批4状态、累计状态数；此时 `committed=False`。 |
| `alpha_d_reference` | start hash、theta0 hash、`same_batch=True`、source-selection role。 |
| `alpha_d_solver` | `z_hat`、`epsilon_hat`、完整 K、K trace/Frobenius/diagonal/cosine 冗余比例、`d_star`、w/α、边界活跃坐标、目标值、迭代数、proximal 残差、warmup、`feedback_age` 与反馈参数 hash。 |
| 同上不确定性字段 | `epsilon_reprojected=True`、`uncertainty_estimator`、`uncertainty_covers_staleness_bias=False`。 |
| `alpha_d_student_commit` | 实际参数 hash；α/d 均 stop-gradient。 |
| `alpha_d_gate_update` | 实际学生上的 VJP、next φ、`blocked_d_fixed=True`、`differentiates_through_d_solver=False`。 |
| `alpha_d_acquisition_reference` | 采购参考 hash、另列 source-selection hash、独立反馈 role、历史一阶插入标签。 |
| `alpha_d_step` / trajectory | 真实 `exposure_units`、`source_actions`、microbatch 数、训练/未训练新包、`exposure_records`、成本/预算、整个 α/d 控制记录。 |
| `return_objective` | 始终 `temperature_1_stochastic_policy_expected_return`；不是官方 greedy 分数。 |

## 7. D7 / D9 / D6 接口

- **D7**：使用 `ExposureRecord` 的 package/record/state 身份接入 V0 完整曝光重放，复用购买时间、w 与重复次数；在来源抽样之前构造记录与状态 χ/α。`alpha_freeze`/`alpha_draw_pairs` 分开，支持先校验重放再从每臂自己的冻结学生重采。`build_reference(pairs, weights, alpha, ..., step)` 接收显式冻结非均匀权重；当前 runner 默认均匀权重。不要调用旧 `draw_slots` 构建 α/d 提交，也不要将轮内 `source_cache` 当成重放来源动作。满额运行需预先安排合法已付 bootstrap 旧池。
- **D9**：复用 `BatchReference.theta0/start_hash/directions/weights/alpha/baseline_gradients`、`gram`、`solve_d`；完整更新应使用 `theta0−theta_ref+Σd_i v_i`。替换 `alpha_acquisition_labels` 的历史可加近似，在独立采购参考处构造包含教师注入的集合 Fhat/删包旧池补位代理；禁止用来源选择的 theta0 回报替代采购参考。当前 `batch_rows/labels/value_statistics` 与 D2 持续后验保持兼容；购买前训练上下文、完整 Fhat 与漂移重测仍由 D9 接入。
- **D6**：固定同一 `BatchReference` 可调用 `selected(d*)` 与 `selected(zeros)` 做同 α 对照；来源交换需同时翻转 v/z/K 行列及 d，不能只改文本标签。`FeedbackStatistic.project(new_directions, mode)` 支持诊断的误差重投影；`gram_statistics` 提供联合方向冗余。任何机制效果/修复损伤或 z 预测有效性验证必须另取反馈或留出任务，不能用选 d 的 rollout 自证。现有 D1 来源方差 helper 保留，α/d 单槽梯度可经 `components`/`estimate` 取得。

## 8. CPU 验证

```bash
PY=/home/xueqi/hq/projects/tc-alignment/.venv/bin/python
PYTHONPATH=src:. CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  BFCL_PROJECT_ROOT=/tmp/rtd-v11-d8-bfcl \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  "$PY" -m pytest tests/test_rtd_*.py -q -p no:cacheprovider
```

BFCL 使用官方 `BFCL_PROJECT_ROOT` 将锁文件写到 `/tmp`；没有修改共享 BFCL 环境。ALFWorld 沿用合并树的临时 fixture，不放宽 symlink 路径保护。环境失败若出现将明确报告，不隐藏。

专项结果：**34 passed**（20.22 s）。覆盖有限动作枚举无偏性、教师质量/合法性、软项身份零梯度与整体非零、加权更新、严格可表示仿射等式、联合抵消坐标、收敛/失败、来源交换、误差新方向重投影、固定 d 的非线性 VJP 差分、40/80/10 曝光、单步冷启动保护、缩减曝光、V0/manifest、逐轨迹重建、adapter、跨窗口 warm-up 与全部 durable phase 恢复。

最终 D8 + bank + M1 集成验证：**62 passed，47.17 s**，日志 `/tmp/rtd-v11-d8-integration-final.log`。

最终完整 `tests/test_rtd_*.py`：**1014 passed，6 warnings，437.47 s，exit code 0**；没有 skip/xfail，没有环境失败。包括 v1.0 完整 manifest/window/梯度字节基准、原 D1/M1 诊断及恢复，以及真实 ALFWorld CPU fixture。完整日志 `/tmp/rtd-v11-d8-full-final.log`。六条警告来自既有 tensor 标量转换和 PEFT 配置提示。

首轮完整测试为 1006 passed、1 failed：当时收集的旧测试仍要求拒绝 `slots_per_step=8`，与 rev 3.1 的随机曝光要求冲突；已改为验证明确标记的合法变体。一次中间专项因编辑引入的缩进错误在收集阶段终止；修复后重新执行上述 62 项集成和 1014 项完整套件，没有隐藏失败。

四份 `v1_bfcl_c25*.yaml` 通过与 HEAD 原字节对比；`git diff --check` 通过。最终 `git status --short` 和 `git diff --stat` 已执行，所有 D8 修改留在 worktree，未暂存、未提交。
