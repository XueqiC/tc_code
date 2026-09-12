# RTD v1.1 D9：rev 3.1 反馈恢复、联合控制与完整更新采购代理

规范为完整 `RTD_V1_1_METHOD_ZH.md` rev 3.1，尤其 §2.3、§3.4、§4.2、§6.1–6.2；已依次阅读 D8、D7、D2/D3 笔记。操作仅在 `/home/xueqi/hq/projects/tc-alignment-v11a`，`data/`、`envs/` 与共享 Python 环境只读。仅 CPU tiny-model/现有测试，没有 GPU、生产训练、在线教师、官方评测、暂存或提交。开始时已有的 `PROJECT_STATE.md` 修改未动。

## 1. 先关闭 D7 的版本分歧

在实现 D9 前，先恢复并测试了 rev 3.1 的三个角色：

| 角色 | 参数点 | 用途 |
|---|---|---|
| `acquisition_reference_feedback` | 旧证据虚拟更新 θ̄⁺ | 独立插入控制标签与完整更新采购代理 |
| `same_batch_reference_feedback` | 本步实际证据的 θS(0) | ẑ、沿 v 的 ε̂、联合 d 求解 |
| `post_commit_feedback` | 同批次 θS(d*) | 分块 α VJP；d* 固定、停梯度 |

采购仍先使用历史后验；实际记录、w、状态 α 在来源抽样前冻结。旧池与实际批次的参考都从同一个 start 构造，单次 backbone 提交只有 `BatchReference.selected(d*)`。旧证据 rollout 先记账，同批次 rollout 使用另一个确定性随机流；两者不能互换。反馈 helper 同时核对参数 hash 与角色，即使 θ 值偶然相同，也不能用别的角色冒充。非决策步还核对历史 `FeedbackStatistic.feedback_role`，拒绝 D7 的旧参考统计被静默用作 d 反馈。

非决策步不新增控制反馈，保留最新同批次逐轨迹 score/reward/task 统计，在本步新方向上重算 ẑ 和 ε̂，并记录 `feedback_age`。这仍不覆盖陈旧偏差。α 的 VJP 使用已提交学生上的独立反馈，固定 d，不穿过求解器。

manifest 改为 `distillation_protocol=alpha_d_rev31_same_batch`、trajectory schema 5，分别列出三个反馈角色；D7 笔记开头已说明版本分歧在此关闭。

**V0 参数回归有修改前基准**：修改源码前，用原 D7 HEAD 的 CPU TinyLM V0 跑两轮24步，保存每步参数 hash。恢复后24个参数 hash 与完整曝光日程逐项一致；最终套件继续用 `tests/fixtures/rtd_v11/v0_rev2_parameter_hashes.json` 检查。新增同批次反馈使用独立随机流，不推进原来源/采购反馈的随机流；V0 的 d=0 提交因而保持不变。

## 2. 一个求解器、两种来源控制

扩展现有 `src/bfas/rtd/joint_surrogate.py`，没有新建平行实现。`control` 继续调用 D8 的 box/L1 小 QP 求解器 `alpha_d.solve_d`：全坐标精确一维更新、投影、全目标 proximal/KKT 残差、失败即拒绝提交。

```text
v_i = (η w_i / 2) P (G_i1 − G_i2)
K_ij = v_iᵀ P⁻¹ v_j
joint:       max ẑᵀd − λ/2 dᵀKd − ε̂ᵀ|d|
independent: 同上，但求解时只保留 diag(K)
|d_i| ≤ min(α_i, 1−α_i)
```

实际批次的全部曝光单元一起进入控制器，包括实购教师记录、旧已购池和无教师参考状态；最后一类 α=0，因此其 d 合法边界也为0。w、α 不参加优化。没有按原始 ẑ 预筛；原始 ẑ=0 的坐标仍可能通过抵消其他更新而取非零值。

`acquisition_value_mode` 只接受 `joint` / `independent`。V2 和 V1 默认 joint；V0 为 independent 且 d=0。V2 可显式设置 independent 作为控制配置，manifest 保留实际开关。采购随机/回放由原 `acquisition` / `ledger_replay` 控制，不再用 `random`、`replay` 冒充 value mode。

每步同时求出联合 d 与独立 d，计算但只提交配置选中的一个；这不增加来源抽样、教师购买或学生更新预算。`joint_vs_independent_control` 记录同一 start、θS(0)、实购池、记录、draw IDs、w、α、ẑ、ε̂、K、反馈批次身份及每臂一次更新。差异只有求解时是否保留 K 的交叉项。

日志同时区分：

- `predicted_joint_gain`：联合最优解在完整 K 下的收益。
- `predicted_independent_joint_gain`：独立解在完整 K 下的收益。
- `sum_independent_gains`：忽略交叉项时各坐标独立收益之和。

独立解是联合问题的合法点，联合最优值应不低于前者；同向重复方向的联合收益可能低于可加预测。正交测试严格相等，共线测试有冗余折扣；不将“联合控制更好”误写成“联合收益必高于可加预测”。日志包含完整 K、trace、Frobenius 范数、对角线、H-cosine 超过阈值的无序对比例、d、三项概率份额及冻结曝光权重。

## 3. 完整更新 F̂ 与删包补位

`set_solution` / `set_value` 保留 D7 的完整更新定义，并匹配实际一步梯度累加顺序及参数精度：

```text
δ0 = θ_A(0) − θ_ref
Δ_A(d) = δ0 + Σ_i d_i v_i
F̂(A) = max_d [g_refᵀΔ_A(d) − λ/2 ||Δ_A(d)||²_(P⁻¹) − Σ_i ε̂_i |d_i|]
```

展开后交给同一 d 求解器的线性系数为 `g_refᵀv_i − λ δ0ᵀHv_i`，基线常数仍计入 F̂。不能丢掉 δ0，否则 d=0 时教师注入的价值也会消失。必要测试中，固定全部 d=0 后，有益/有害教师更新的标签分别为 +0.5 / −0.5。

`marginal_values` 在实购包上计算 `F̂(D)−F̂(D\{q})`，同包的所有曝光单元一起删除。补位记录在实际来源抽样与反馈前从合法旧池确定；补回各槽时使用该槽原来的 w、补位记录预先冻结的 α/梯度/来源，不改变其他槽。非均匀权重、重复包曝光、正交可加情况均有测试。所有集合比较强制使用相同 reference hash 和相同回报梯度；不调用 candidate 环境评测。

**两个参考点的数学约定**：实际 d 必须使用 §3.4 的 `gJ(θS(0))`。采购 F̂ 按独立采购参考使用 `gJ(θ̄⁺)`；两者复用同一优化形式，不声称实际 d 与采购代理的最优 d 一定数值相同。若把 F̂ 的共同参考设置为完整集合 θS(0)，其完整集合 δ0=0、交叉偏移=0，与 D8 d 控制器严格相同，专项对此检验参数/目标结果相等。保持旧证据采购梯度时，一般不能同时要求两个最优 d 相等；日志分开保留它们，未以旧梯度替换实际 d 反馈。

新标签名称是 `acquisition_surrogate_full_update_with_teacher_injection`，不是实际任务收益。原独立插入标签完整保存在 `independent_insertion_control` 事件与步记录 `independent_control_labels`。持续后验接口仍使用单位曝光 value；联合包边际值除以该包实际总 share 后拟合，`position_value` 恢复为真实包边际值，避免二次缩放。零权重单元不会产生虚假的正曝光后验观测。

当前标签 covariance 沿用 D3 的可加反馈噪声代理，明确标为 `additive_feedback_noise_proxy`，不声称覆盖联合求解器不确定性。ε̂ 的 L1 正则作用于 d；完整教师基线的反馈噪声由上述标签 covariance 近似处理。这些是局部代理，不是统计保证。

## 4. 购买前上下文、收缩与封存边界

`PrePurchaseFeatures.from_public` 保留旧路径39维输入；α/d 路径为49维，新增字段顺序由 `TRAINING_CONTEXT_FEATURES` 固定：轮进度、步进度、已购包数、已购状态数、已购状态投影均值范数、状态 α 的 mean/std/min/max、反馈年龄。规模字段归一化，α 只从状态特征计算；不使用当前来源动作、候选教师文本、类别或错误类型。

`acquisition_pre_purchase_context` 在购买前持久化上下文及所有候选特征。新增标签用当时保存的 `batch_rows` 拟合，不以提交后 α 或新实购集合重新制作训练特征。价值后验使用49维，成本模型明确剥离新增10维、继续使用原39维，避免改变V0的成本预测和预算选择；有模型 precision/information 与成本预测严格相等测试。两模型跨轮保存，resume 保留相同维度、观测与随机状态。

α/d 后验每窗拟合前保存对本窗已购标签的预测；至少4个窗口、8个标签且有正的窗外解释方差，才放松95%的后验**均值**收缩，否则 `shrinkage_factor=.05`。原始 information 单独持久化，model information 为其收缩版本；precision 保留 D3 的共享噪声 GLS 更新。D3 漂移降权同步缩放原始 information，后续拟合不会撤销降权。已测试4窗8标签仍无预测能力时保持强收缩。这是保守的预测启发式，不是有效性认证，也没有用本窗拟合后的残差自证预测能力。

`supervision_records` 和 `alpha_draw_pairs` 在梯度计算前核对 ledger ownership；`marginal_values` 还核对实购 ID 集。旧池由已付记录/参考状态构造。泄漏测试在真实 CPU runner 的每次梯度入口检查 ownership，保留一个未购候选并确认其梯度、代理标签、后验观测均未出现；伪造未付教师记录在抽样前被拒绝。

## 5. 实际配对收益仅验证

控制反馈与验证反馈不共用采样。每个学习臂决策窗额外做两类固定规模诊断：

1. 若本窗有联合采购标签，按 query ID 选第一个已购包，独立执行完整集合与删包补位集合的一步更新，验证该边际 F̂。选包不依赖预测值；不是逐候选采购评测。
2. 在同一个 `BatchReference`、同一反馈估计、同一已购池、w/α/来源与更新预算上，独立执行 joint 与 independent d 的一步更新。

两次更新分别由 `execute_update` 从同一冻结 start 执行，不安装到 backbone。每个学生都从任务起点重新执行反馈任务，各自使用专用随机流与不同 batch ID；任务可相同，Monte Carlo 批次与选 d 的批次独立。参数 hash 恰好相同时仍分别执行/采样。验证只计算任务回报，不产生训练回报梯度。

`realised_paired_gain_validation` 保存两边任务回报、参数/batch身份、预测、实测差及误差，明确 `validation_only=True`、`used_for_posterior=False`、`selection_feedback_reused=False`。测试把实测收益改为极端值后，后验仍只接收原采购代理标签。

这些验证 rollout 以 `validation_*` 单列计算，属于诊断，不计入三个训练反馈角色；没有隐藏在每步80条来源动作里。默认每窗至多4个验证批次：一个已购包对与一个控制器对，各2批。V0不运行这些机制验证。不存在正式 benchmark 改善、显著性或生产吞吐结论。

## 6. 恢复与验证

复用原 durable `revealed → actual → feedback → committed` 阶段。`actual` checkpoint 保存已提交参数、本批参考、全部方向、两类反馈统计和冻结购买前特征；恢复直接继续反馈/标签/验证，不重复提交 backbone。`feedback` checkpoint 保存 α、代理标签和验证结果，后验更新在原提交阶段进行；已提交后清除本批图、方向与采购统计，仅保留历史同批次可重投影反馈和正曝光记录。

前置恢复：D7/D8 专项 **51 passed**；当时新增恢复专项 **4 passed**。D9 专项 **23 passed，34.16秒**；D7/D8/D9 集成中间版本 **71 passed，119.65秒**。最终 D9 文件共25项，新增的低预测能力/漂移与成本隔离测试均由最终完整套件覆盖。最后的后验/manifest 专项 **38 passed，41.02秒**，成本隔离与24步V0回归 **2 passed，15.89秒**。

完整 CPU 命令：

```bash
PY=/home/xueqi/hq/projects/tc-alignment/.venv/bin/python
PYTHONPATH=src:. CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  BFCL_PROJECT_ROOT=/tmp/rtd-v11-d9-bfcl-final \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  "$PY" -m pytest tests/test_rtd_*.py -q -p no:cacheprovider
```

BFCL锁文件仅写 `/tmp`；ALFWorld使用已有临时fixture，未修改共享环境与路径保护。最终完整 `tests/test_rtd_*.py`：**1056 passed，6 warnings，510.95秒，exit code 0**；没有skip/xfail或环境失败。日志 `/tmp/rtd-v11-d9-full-final.log`。六条警告来自既有tensor标量转换和PEFT配置提示，没有新增警告。

此前首轮完整套件 **1054 passed，6 warnings，562.24秒**，日志 `/tmp/rtd-v11-d9-full.log`；随后补正manifest标签、漂移information同步降权与成本特征隔离，并增加两项测试，再运行了上述最终完整套件。

开发中首次集成为54通过、1失败：旧V0配置测试需要显式 independent；同时修复 NumPy 对 Tensor 的弃用警告。新增专项首次为16通过、4失败：测试断言需要 detach，手动恢复调度需要将 `feedback` 映射到 `feedback_commit`；修复后全部通过，没有 skip/xfail。结束执行 `git diff --check`、`git status --short`、`git diff --stat`，所有修改留在当前 worktree。
