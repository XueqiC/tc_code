# RTD v1.1 D1：软来源与控制变量估计器

实现范围：仅 `tc-alignment-v11a` worktree；CPU 数学/runner 测试，无训练运行、无 GPU、无提交。`data/`、`envs/` 及共享 Python 环境只读。配置仍从现有 BFCL v1.0.1 模板显式选择新估计器；不改冻结 YAML、预算、采购规则或官方评测协议。

## 1. 每个文件的变化

| 文件 | 变化 |
|---|---|
| `src/bfas/rtd/source_estimator.py` | 配置验证、`SourceControl` 独立抽样来源记录、可微的估计器系数、D6 方差诊断 helper。 |
| `src/bfas/rtd/source_scoring.py` | 两次 teacher-forced 主干前向；在输出 head 前截取隐藏状态，按位置投影完整词表；计算并返回硬/软来源梯度。 |
| `src/bfas/rtd/runtime.py` | `streamed_gradient` / `streamed_gate_vjp` 增加显式估计器选项；新更新系数与新 VJP 共用定义；逐槽梯度范数与 c_s 日志。原 hard2 函数体运算及旧 VJP 保留。 |
| `src/bfas/rtd/experiment.py` | 原写死的 2 改为读取 `source_samples_per_state`，默认仍 2。CV 每次 `draw_slots` 刷新来源，保存 old/new 控制样本；pilot、参考/实际更新及实际反馈 VJP 使用对应记录。 |
| `src/bfas/rtd/cli.py` | 验证新配置，允许改变来源样本数。hard2 的显式默认选项规范化为历史省略形式，保持旧配置与 manifest 声明。 |
| `tests/test_rtd_v11_source_estimator.py` | 本文第 6 节的 CPU 数学、内存接口、BFCL 回归和恢复测试。 |

`transport.py` 与 `functional_step.py` 无需修改：继续使用现有 `SourceSample` / `(source, teacher)`、完整动作边界、同状态检查以及 `FrozenStep.update(parameters, gradient)`。`slot_loss` 仍是旧 hard2 的损失 oracle；soft/CV 调用新的流式梯度接口，不能拿旧 `slot_loss` 替代其更新算子。

## 2. 目标及控制系数

令 Y 来自本轮冻结来源 p_t，H 为该采样完整动作的 NLL 梯度，S 为相同采样前缀上的完整词表软 CE 梯度，T 为同状态教师完整动作的 NLL 梯度。动作包含实际采到的 EOS；达到长度/context cap 时不补 EOS。只对动作预测位置求和，不作长度平均。

软来源目标为：

\[
L_{src}^{soft}=E_{Y\sim p_t}\sum_j KL(p_t(\cdot|s,Y_{<j})\Vert\pi_\theta(\cdot|s,Y_{<j})).
\]

CE 与 KL 只相差对 θ 为常数的来源熵，梯度相同。每个位置的精确 logit 导数是 q−p；实现直接将此向量作为当前 head 的 cotangent，再经隐藏状态反传。p 与 q 用各自**完整词表**的 log_softmax 计算。因而当前参数等于来源快照时，每个位置的残差及梯度精确为零，不靠事后清零、不靠候选集合归一化。前缀仍来自原始温度 1/top_p 1 采样。

固定/标量门控的软更新为 `(1−a)S+aT`。一般来源依赖门控的 CV 更新为：

\[
g_{CV}=(1-a)H+aT-c_s(H-S)=(1-a-c_s)H+c_sS+aT.
\]

独立条件积分给出 `E[H|s]=E[S|s]`；因此只要 c_s 与当前来源抽样独立，条件控制项均值为零。实现选择估计 `E[1−a(s,Y)|s]`，**不声称这是最优协方差系数，也不保证降方差**：

| `cv_cs_mode` | 定义与限制 |
|---|---|
| `loo`（CV 默认） | 同一状态先抽 m 个独立来源，对当前原始 draw i 使用 `c_i = mean_{k≠i}(1−a(s,Y_k))`。m≥2。按原始抽样下标排除，不能从有放回抽取的训练曝光槽反建 LOO 集合。 |
| `independent` | 另外抽 m 个独立来源 Z，使用 `mean_k(1−a(s,Z_k))`；每状态共 2m 个来源抽样。当前 draw 不得出现在辅助池中。 |
| `fixed_one_minus_a` | 取 c_s=1−a，仅适用于固定/标量门控；硬项系数直接成为 0，与 soft 更新逐字节一致。拒绝完整来源依赖门控。 |

辅助来源同样使用冻结 p_t；特征及标准化统计量冻结，而辅助门控对 φ **不 detach**。不同独立抽样可以产生相同 token 序列，故不能按文本去重或排除所有相同文本。`SourceControl` 保存样本对象、冻结特征和排除下标；检查同完整状态、来源快照和原始 draw 身份。独立性是采样调用的契约，runner 通过独立 `sample_action` 调用实现，不能把当前 draw 复制一份冒充独立样本。

无教师时有效 a=0，c_s=1，soft/CV 都使用 S；当前等于来源的原有 exact-noop 分支仍可跳过更新。一般完整门控不能直接把硬来源替换为软来源；runtime 与 runner 会拒绝该组合。

## 3. 流式计算与元梯度

每个来源槽先后进行冻结快照和当前参数的两次 teacher-forced 主干前向，均使用同一 prompt 与采样 action IDs。临时 head pre-hook 截取输入并立即结束该次前向，避免分配 `sequence × vocabulary` logits；`finally` 总会移除 hook。不调用 generate、不安装/覆盖 resident 学生权重、不修改冻结来源。

保存的是两个隐藏状态序列及硬/软隐藏 cotangent。默认 `position_batch_size=1`，每批只投影动作预测位置，计算完整词表 p/q，并立刻释放该批词表张量和 head 图。来源/当前 LM 主干不为每个 token 重算；当前主干图被硬、软两次 backward 共用。head 若有 LoRA 参数，其直接梯度也纳入结果。返回的 LoRA 梯度均 detach。教师项另走原有完整动作评分。

模型接口要求 `get_output_embeddings()` 返回逐位置 head，且其输出即评分 logits（允许浮点类型转换）。当前 Qwen3.5/PEFT CPU 测试覆盖实际路径和非重入 checkpoint replay。带额外 post-head 非线性变换的模型需单独适配；已明确拒绝 `final_logit_softcapping`。该实现需要标准 head 接口，不对缺少此接口的模型静默退回整段词表分配。

新 VJP 对同一系数函数 `(1−a−c_s, c_s, a)` 求导。设 w 为实际更新处的反馈梯度，对每槽先计算 H/S/T 与 `(ηP)w` 的三个内积，再用 autograd 对小型系数图求导并取负号/槽均值。ηP 的乘法顺序与 `FrozenStep.update` 一致，包括混合精度情况。此路径包含 `∂c_s/∂φ` 及 LOO 跨 draw 贡献；不复用旧 `a(1−a)χ(T−H)` VJP，也不需要 LM 二阶反向。

CV 只作为**刷新样本的单次梯度估计器**：每次抽训练槽时，为访问到的状态建立新的独立来源组，随后有放回选曝光槽；本次更新内固定样本。参考/实际/pilot 各自保存自己的组，恢复和后续 VJP 不重新抽样。来源权重仍整轮冻结为 `state['source']` / `source_id`。固定 head/projection、标准化与 RMS 几何仍使用原有整轮冻结规则。soft/hard2 延续原来源缓存行为。

## 4. 配置与日志

```yaml
# 默认省略这些新项等价于 hard2；原 source_samples_per_state: 2 保留。
source_estimator: cv                 # hard2 | soft | cv
source_samples_per_state: 2          # 正整数；loo 至少 2；hard2 也支持 8
cv_cs_mode: loo                      # loo | independent | fixed_one_minus_a
```

R0 固定 .5 可选 `source_estimator: soft`；R1 的 soft 必须选择 `gate: scalar_sigmoid`，或使用既有固定 gate override。scalar 特征必须是常数 1 的单列截距。新选项在 BFCL CLI 和共用 `RTDExperiment` 中验证；本次不扩展另有冻结 schema 的 ALFWorld config validator。

soft/CV 的每槽 `source_estimator_slot` 记录：估计器、c_s 模式和数值、state/source 身份、slot_index、槽均值系数、源 KL、动作位置数、position batch 大小、两次主干前向计数，以及：

- `hard_gradient_norm`：完整混合更新 `(1−a)H+aT` 的 L2 范数。
- `soft_gradient_norm`：完整混合更新 `(1−a)S+aT` 的 L2 范数。
- `cv_gradient_norm`：本槽实际 CV 更新的 L2 范数；soft 模式下为其固定系数退化形式。
- `hard_source_gradient_norm` / `soft_source_gradient_norm`：不混入教师的 H/S 范数。

hard2 保留原 compute 事件和原评分/求导顺序，不为了诊断增加来源快照前向。代码修改会正常改变真实 checkout 的 `rtd_source` 内容哈希，不能伪装为旧实现；兼容承诺是相同 provenance 输入下的完整 BFCL manifest 声明、原默认配置字节及参考梯度字节一致。测试用固定合成 BFCL checkout 锁定全 manifest 哈希，并锁定修改前 main HEAD 的配置/梯度哈希。

## 5. D2 / D6 接口

D2 继续传入非空 `targets: Sequence[(SourceSample, Behavior|None)]`、逐槽 chi、φ，以及当前参数。新关键字：

```python
options = dict(
    source_estimator='cv',
    source_parameters=round_source_parameters,  # 与每槽 source_id 一致
    cv_cs_mode='loo',
    source_controls=controls,                   # 每槽一个 SourceControl
)
g = streamed_gradient(targets, chi, phi, backend, start, gate=gate, **options)
updated = step.update(start, g)
vjp = streamed_gate_vjp(targets, chi, phi, backend, start, step,
                       feedback_at_actual, gate=gate, **options)
```

D2 为同一次更新保存/复用 targets、chi、controls、来源快照和 start。做新的 E=40 曝光混合时，梯度和 VJP 必须乘**同一组曝光份额**，feedback 在最终实际更新处求。`RTDExperiment.estimator_options(controls)` 提供相应 options；当前 old/new controls 在恢复状态里独立保存。D2 合并/拆分槽时也须对齐 controls，不能按文本重建、不能重采后算 VJP。`functional_step` 和插入价值继续接收显式梯度，D1 不改 0.75/0.25 旧曝光规则。

D6 可调用 `source_variance_diagnostic(state, backend, parameters, source_parameters, phi, feature_fn, generator=..., K=..., teacher=..., gate=..., source_samples_per_state=..., cs_mode=...)`。要求 K≥2；feature_fn 使用已冻结统计量，不能在本批来源上重新拟合；state/teacher 由调用者保证来自已授权/已购数据。需要 benchmark 特定 action cap 时由调用者包在 `backend.action_limit(category)` 中。

helper 每轮重新抽 m 个来源（independent 模式再抽 m 个辅助来源），三个估计器共用该批来源、教师、θ/φ及流式 H/S/T。返回各次**平均槽梯度**的 L2 范数、范数均值及总体方差 `var(norms, correction=0)`，不把它称为梯度协方差迹。一般完整门控下 soft 只是有偏诊断对照，输出 `soft_preserves_target=False`。`diagnostic_gradients=[]` 可选参数让 `streamed_gradient` 返回比较所用的聚合 H/S/CV 混合梯度到该列表，供后续扩展；默认只返回实际梯度。任何方差下降结论都须来自 D6 实测。

## 6. 验证

专项测试覆盖：逐前缀快照相等梯度精确为零（FP32/FP64，含 EOS/cap）；与完整词表 CE 的梯度对照；枚举最多两 token 的完整动作分布及 100,000 次 Monte Carlo 验证 E[H]=E[S]；枚举独立来源对验证 CV 无偏及一般直接 soft 的偏差；严格 LOO 排除与辅助门控求导；固定/标量 CV 与 soft 字节一致；新更新后非线性回报的 VJP 与中心差分（所有系数模式及 FP32/FP64 对角）；head trainables、checkpoint replay、两次主干前向与单位置词表投影；真实 CPU 小型 Qwen3.5/PEFT；方差诊断及逐槽日志；错误配置；BFCL 历史完整 manifest/参考梯度；CV 刷新、恢复及实际门控更新。

使用的命令（禁止生成共享环境 pycache）：

```bash
export PY=/home/xueqi/hq/projects/tc-alignment/.venv/bin/python
export PYTHONPATH=src:.
export CUDA_VISIBLE_DEVICES=''
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export BFCL_PROJECT_ROOT=/tmp/rtd-v11-d1-bfcl
mkdir -p "$BFCL_PROJECT_ROOT"  # BFCL 的锁/输出目录，避免向只读 envs 写锁
"$PY" -m pytest tests/test_rtd_v11_source_estimator.py -q
"$PY" -m pytest tests/test_rtd_*.py -q
```

专项最终版本 **34 项通过**；D1 + malformed rollout + return gradient + 旧 score-consistency 精简 runner 回归共 **89 项通过**。最终完整 `tests/test_rtd_*.py` 为 **921 passed、1 failed、6 warnings**（412.82 秒）；唯一失败为下述修改前也存在的 ALFWorld symlink 集成限制。完整日志：`/tmp/rtd-v11-d1-full-final.log`；修改前复现日志：`/tmp/rtd-v11-d1-alfworld-baseline.log`。`git diff --check` 通过。

隔离环境的已确认限制：原 ALFWorld `test_real_environment_two_packages_replay_deterministically` 会在 `_Sources.path` 抛出 `archive input escapes root`，因为它拒绝解析后位于 checkout 外的 `envs/` symlink。本次用 `git show HEAD:.../alfworld_bank.py` 提取修改前的 `_Sources` 在同一测试中复现了完全相同的失败；没有放宽此检查、没有改共享环境。默认 BFCL 的 17 项只读锁文件失败已通过上面的官方 `BFCL_PROJECT_ROOT` 环境变量解决；这只是测试会话设置，不改 BFCL 源码/manifest。
