# RTD v1.1 D11：联合 d 与完整更新采购代理的尺度归一化

依据 `RTD_V1_1_METHOD_ZH.md` §3.3–§3.4、§4.2 及 D8/D9 实现。
工作仅在 `/home/xueqi/hq/projects/tc-alignment-v11a`、`rtd-v11-a`；
`data/`、`envs/` 和指定 Python 环境只读，不运行 GPU，不暂存、不提交。

## 1. 原因与归一化目标

任务提供的真实 smoke 证据指出，`v_i=(ηw_i/2)P(G_i1−G_i2)` 的学习率与
曝光权重使 `K_ij=v_iᵀP⁻¹v_j` 对角线约为 `1e-13…1e-11`。
原始线性项随方向尺度一次变化，二次项随其平方变化；因此固定原始 λ=1
会使二次项对控制几乎不起作用。本次没有访问或修改其他 worktree 的 smoke 文件。

在决策窗首步的实际完整曝光集合上，取

```text
I = {i : K_ii > 0}
s = mean_{i∈I} K_ii                 （I 为空时 s=1，明确记录 fallback）
K̃ = K/s,  z̃ = ẑ/√s,  ε̃ = ε̂/√s
d* = argmax [z̃ᵀd − (λ/2)dᵀK̃d − ε̃ᵀ|d|]
     subject to |d_i| ≤ min(α_i, 1−α_i)
```

非零判断是严格的 `K_ii>0`，不用绝对阈值；极小非零方向仍参与均值。
α 端点、原始 ẑ 大小和 ε̂ 大小均不参与尺度筛选，也不筛掉求解器坐标。
λ=1 表示**参考集合非零方向的平均归一化自身曲率为1**；实际线性收益仍由
反馈梯度与方向的对齐决定，不能承诺每个坐标的二次成本等于它的实际线性收益。
这仍是局部代理的超参标定，不是对未知平滑常数的估计。

若所有 `v_i→c v_i`（c>0），则 `K→c²K`、`ẑ→c ẑ`、`ε̂→c ε̂`、
`s→c²s`，三个归一化输入完全不变，所以 QP、收敛判断和 d* 不变。
退化多解时保留原坐标顺序与零初值，使用同一个确定性求解器。

## 2. 等价式与 λ 的单位（修正任务描述中的代数冲突）

上式可以等价写成

```text
z̃ᵀd − (λ_effective/2)dᵀKd − ε̃ᵀ|d|,
λ_effective = λ/s.
```

这里 K 是原始矩阵，但线性项仍是**归一化**的 z̃、ε̃。
把整个目标乘以正数 √s，最优 d 不变，得到真正的原始单位等价问题：

```text
ẑᵀd − (λ_original_units/2)dᵀKd − ε̂ᵀ|d|,
λ_original_units = λ/√s.
```

因此，“归一化 ẑ/ε̂ 后等价于原始目标取 λ/s”不成立；后者若保留原始
线性项，会失去所要求的尺度不变性。例如单坐标 `K=v²,ẑ=.2v,ε̂=0`，
归一化后内部最优解为 `.2`；原始线性项加 `λ/s` 则给出 `.2v`。
实现优先采用尺度不变目标，并同时记录上述两个不同单位下的系数。
`v→10⁻⁶v` 时，`lambda_effective` 增大 `10¹²` 倍，
`lambda_original_units` 增大 `10⁶` 倍。

联合零坐标条件及 box 保持原来的 KKT 形式。在允许零为内部点的坐标上：

```text
|z̃_i − λ_effective Σ_{j≠i} K_ij d_j| ≤ ε̃_i
等价于
|ẑ_i − λ_original_units Σ_{j≠i} K_ij d_j| ≤ ε̂_i.
```

不能在第二式里把 `lambda_original_units` 换成 λ/s，也不能据原始
`|ẑ_i|≤ε̂_i` 预筛坐标。LOO 删除轨迹后重新拟合 baseline、沿新方向重投影
ε̂、warm-up、固定 α/w、来源交换对称性与停梯度逻辑均沿用 D8/D9。

## 3. 一个窗口、一套标定；完整 F̂ 保留教师注入

`DCalibration` 在决策窗首步的完整实际证据上创建，并随状态持久化。
窗口内非决策步继续重采来源、重投影 ẑ/ε̂，但复用该窗口的 s；新决策窗
重新标定。全零参考的 `s=1` fallback 也冻结到窗口结束。

联合与独立控制使用同一个 s；独立臂仅删除 K 的非对角项。
采购沿用独立的旧证据参考点及反馈角色，但复用实际控制的 s。
完整集合与全部删包/旧池补位版本共享 s，不在每次删除后重新求均值。
独立调用 `marginal_values` 时，则先从传入的完整集合标定一次。

```text
δ0 = θ_A(0) − θ_ref
Δ_A(d) = δ0 + Σ_i d_i v_i
F̂_s(A) = max_d [g_refᵀΔ_A(d)/√s
                 − λ/(2s) ||Δ_A(d)||²_H − ε̂ᵀ|d|/√s]
```

展开后 QP 的归一化线性系数是
`g_refᵀv_i/√s − (λ/s)δ0ᵀHv_i`；基线常数仍为
`g_refᵀδ0/√s − λ/(2s)||δ0||²_H`。
实现将原始单位的 `ẑ−(λ/√s)cross` 交给统一控制器，由它归一化一次，
避免 cross 重复缩放。即使强制所有 d=0，完整教师基线仍计入 F̂。
若 `θ_ref=θ_A(0)`，δ0=0，F̂ 的完整集合解与同输入 d 控制严格一致。
不同反馈参考点仍不保证两个 d 数值相同。

采购后验使用归一化边际标签，对既有可加反馈噪声 covariance 同步除以 s，
保留其不覆盖联合求解不确定性的限制。独立插入控制标签保持原始单位。
实际配对回报是原始回报单位，因此验证把归一化预测乘 √s 后再计算预测误差；
日志另存 `surrogate_prediction` 和 `prediction_scale`。原始 ẑ 的方向探针
不再缩放。骨干仍只提交实际蒸馏更新，没有改变 η、P、w 或加入额外更新。

## 4. 配置、日志与兼容

- `d_lambda` 保持默认 `1.0`；新增 `d_lambda_normalisation`，只允许
  `mean_diagonal`（默认）或 `none`。`none` 取 s=1，恢复历史数值目标。
- `K_scale`、`K_scale_fallback`、`K_scale_nonzero_directions`、
  `d_lambda`、`lambda_effective`、`lambda_original_units` 与 `objective_units`
  出现在控制、求解器和采购诊断中，包括 warm-up/配置零/反馈不足的分支。
- `K_statistics.trace/frobenius_norm/diagonal` 报告 K/s 的统计；
  `K_statistics.raw_trace` 仍保留原始 trace，另有字符串
  `raw_trace_scientific`，使后续定点舍入展示也不吞掉这个证据。
  完整 `K` 仍为原始矩阵，配合 s 可重建归一化矩阵。
- `z_hat_normalised`、`epsilon_hat_normalised` 记录求解单位；原始反馈 ẑ/ε̂
  仍保留，无法估计的误差仍为 null。
- 冻结控制 archive 重放原窗口 s；D11 之前缺少归一化配置/日志的 archive
  使用历史 `none`。原 v1.0 无 d，未修改其 YAML、梯度或配置默认值。

## 5. CPU 验证

新增 `tests/test_rtd_v11_d_normalisation.py` 覆盖：内部解在方向缩小 `10⁻⁶`
后不变及系数比率、两个单位下的真实等价式、无预筛联合零条件、全零 fallback、
极小非零方向与 α 端点、真实 tiny 模型的共线方向使默认 joint/independent
产生不同更新、默认 λ 的 d=0 教师正负价值、固定删包标定、完整更新展开、
窗口内刷新/跨窗口重标定、resume、后验噪声单位和配对验证单位。

```bash
PY=/home/xueqi/hq/projects/tc-alignment/.venv/bin/python
PYTHONPATH=src:. CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  BFCL_PROJECT_ROOT=/tmp/rtd-v11-d11-all-bfcl \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  "$PY" -m pytest tests/test_rtd_v11_*.py -q -p no:cacheprovider
```

测试只使用 CPU tiny fixtures；BFCL 可写锁根设为 `/tmp`，无共享环境写入。
最终完整 `tests/test_rtd_v11_*.py`：**219 passed，1 warning，211.57秒，exit code 0**；
日志 `/tmp/rtd-v11-d11-all.log`。既有警告来自 acquisition 测试把有梯度的 tensor
转为 Python 标量。无 skip/xfail 或环境失败，包括 v1.0 字节基准、V0 的24步
参数 hash 回归、warm-up、恢复、LOO 与来源交换对称性测试。

源码完成后的 D8/D9/conventions 前置专项为 **83 passed，141.95秒**。
全套运行期间为新增测试补充了完整更新交叉项的独立 QP 展开断言；随后再次运行
D11 专项：**15 passed，5.37秒**，日志 `/tmp/rtd-v11-d11-final-new.log`。
首次 D11 专项有一条 fixture 断言失败：tiny bank 到第二窗已无可购买包，
因此该窗无采购代理；修正断言后，仍强制检查首窗采购代理及次窗重新标定。

`git diff --check` 通过；四份 `v1_bfcl_c25*.yaml` 无差异。
结束执行 `git status --short --branch` 和 `git diff --stat`；全部修改留在当前
worktree，新增笔记与专项测试未暂存，无 commit。
