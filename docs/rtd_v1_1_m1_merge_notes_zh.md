# RTD v1.1 M1 合并说明

## 1. 交付状态与输入核验

**这是已完成冲突处理的 D1 + D2/D3 合并候选，不是完整 D8/rev 3 实现。原 worktree 的 Git 元数据只读，因此原分支尚未进入 merge 状态；合并暂存在下述 `/tmp` 元数据中，未提交。**

- 本工作区：`/home/xueqi/hq/projects/tc-alignment-v11a`，分支 `rtd-v11-a`，开始时 HEAD 为 `46055e8`，包含 D1 `cc389ff`。
- 合并来源：`rtd-v11-b` = `d9954f0`，包含 D2/D3 `24f5fca` 和两轮配置修正 `d9954f0`。
- 已阅读 D1、D2/D3 说明及 HEAD 中的 `RTD_V1_1_METHOD_ZH.md` rev 3。**输入中没有 `docs/rtd_v1_1_d8_notes_zh.md`、D8 源码或测试**；本 worktree 文件和可见 refs 的提交记录均未找到 D8 交付。已请求 D8 的 commit/ref，不能把 D1 CV 门控冒称为 α+d。
- 初始 `PROJECT_STATE.md` 有一行未暂存用户修改；保持原样，不纳入合并暂存。
- 工作期间观察到方法文档被外部更新并在原 index 暂存为 rev 3.1。这不是本次修改，未覆盖或纳入临时 merge index。下文以用户指定的 rev 3 为验收基准，并单列 rev 3.1 对后续工作的影响。

原命令已实际执行：

```bash
git merge --no-commit --no-ff rtd-v11-b
```

在写 `ORIG_HEAD` 时失败：

```text
fatal: update_ref failed for ref 'ORIG_HEAD': cannot lock ref 'ORIG_HEAD':
Unable to create '/home/xueqi/hq/projects/tc-alignment/.git/worktrees/tc-alignment-v11a/ORIG_HEAD.lock': Read-only file system
```

该路径是本 worktree 的共享 Git 管理目录；当前权限禁止写入，也禁止提权。为继续完成可审阅代码，使用 `git clone --shared --no-checkout` 在 `/tmp/rtd-v11-m1-git.mIvW6q/.git` 建立独立元数据，并把 **work-tree 显式指向本工作区**，然后执行同一 merge 命令。没有创建另一份源码 checkout，也没有写其他 worktree、共享 `data/`、`envs/` 或 Python 环境。

临时 index 中的合并可用下列命令审阅：

```bash
git --git-dir=/tmp/rtd-v11-m1-git.mIvW6q/.git \
  --work-tree=/home/xueqi/hq/projects/tc-alignment-v11a status
git --git-dir=/tmp/rtd-v11-m1-git.mIvW6q/.git \
  --work-tree=/home/xueqi/hq/projects/tc-alignment-v11a diff --cached --stat
```

最终暂存补丁另导出为 `/tmp/rtd-v11-m1-resolution.patch`。它与临时 `MERGE_HEAD` 便于后续迁移；**它们不会使原 index 自动暂存，也不会解除原 Git 元数据的只读限制**。没有执行 `git commit`。

## 2. 每个文本冲突及解决理由

实际只有两个冲突文件、三个冲突块，没有测试文件冲突。

| 文件 / 位置 | 两侧差异 | 解决与理由 |
|---|---|---|
| `src/bfas/rtd/cli.py`，`load_config` 的 mutable keys | D1 允许配置 `source_samples_per_state`；D2/D3 增加 v1.1 专属 rounds/E/K/quota 配置验证 | 合并两个 key 集；来源样本数可配置，batch keys 仍只在 `protocol_version=1.1.0` 下接受。保留 D1 的 `validate_source_config` 和显式 hard2 默认归一化；同时保留 v1.1 bank、证书、manifest 与两轮 coordinator。 |
| `src/bfas/rtd/experiment.py`，构造函数 | D1 验证估计器与 gate 组合；D2/D3 设置 `self.v11` | 依次保留两项。soft/full linear gate 与不合法 CV 模式继续拒绝；v1.0/v1.1 phase dispatch 不被覆盖。 |
| `src/bfas/rtd/experiment.py`，`draw_slots` | D1 为 CV 保存独立来源组和 `SourceControl`；D2/D3 提供显式 `count` | 保留 D1 groups/controls 逻辑，循环次数改为 `self.slots if count is None else count`。默认 E 个旧槽，新包 `count=1`，单槽也保留完整辅助来源组；LOO 仍按原 draw 下标排除，不能从一个曝光槽重建。 |

自动合并的交叠点也已核对：

- `step_start` 同时保留 D1 的 `old_controls`、显式 `old_gradient` 与 D2/D3 的 `exposure_slots=E`。
- `RTDExperiment` 继承 batch mixin；v1.1 分派批量 phases，v1.0 继续使用原 single-package phases。没有把旧路径的 `.75/.25` 或 8 槽改为 v1.1 规则。
- 后验采用 `ValuePosterior` / `LegacyValuePosterior` 版本分流；v1.1 跨轮保留模型、历史与成本观测，round checkpoint 仍保存 `batch_state`。
- 求解集合和 RNG 先持久化，然后固定顺序逐包 reserve/reveal/settle；hard-cap 拒绝不重抽计划；实际损失用最终实购集合，窗口 Q/K 与全局账本约束保留。
- 新 bank/config/certificate/recorded-cost half-up 预算、两轮 runner/coordinator 和回归 fixture 均保留。

## 3. 无文本冲突但必须修复的接口缺口

仅去掉冲突标记会让 D1 在 batch 路径中失效。本次补充如下连接：

1. `experiment_v11.py::batch_selected` 在每包抽样后立即保存 `batch_controls[query_id]`，随后才运行可能另抽样的 KL pilot。防止 pilot 覆盖本次 pending 来源的控制组；子事务 checkpoint 同时保存 targets/chi/controls/RNG。
2. `batch_revealed` 将对应 controls 传入实际新包梯度。旧梯度仍使用 `old_controls`，无教师 exact-noop 保留。
3. `batch_gate_vjp` 调用 D1 `estimator_options(controls)`，传递 `source_estimator`、本轮冻结参数、`cv_cs_mode` 与相同 controls。修复原 batch VJP 静默落回 hard2 的问题。
4. 实际梯度与 VJP 使用同一权重：旧槽均值乘 `1-m/E`，每个实购新包槽乘 `1/E`；gate 反馈仍检查位于实际更新参数。
5. 已购包漂移重测也消费自身 `draw_controls`；它不借用旧提交或 pending 包的控制组。
6. 每个 v1.1 step 重置 per-package controls；已提交后释放 controls，防止跨窗遗留与无用 checkpoint 膨胀。v1.0 状态 schema 不增加这些 batch 字段。

新增 `tests/test_rtd_v11_merge.py` 覆盖 soft/scalar、CV LOO、CV independent、CV fixed-one-minus-a 的混合更新与有限差分 VJP，以及三个 CV 模式在首包已付、revealed、actual、feedback 四个断点的恢复。检查 pending 与 pilot controls 分离、唯一事务序列、学生参数和后验一致。独立运行的账本时间戳及其 hash chain 不要求相等；所有事务语义字段要求相等，各自 hash chain 仍由正常恢复审计验证。

没有删除或改弱两侧测试：D1 专项测试原样保留，D2/D3 的三个新测试文件和原测试变更原样引入。D2/D3 对 ALFWorld 测试 fixture 的只读 symlink 适配也保留，不放宽生产路径保护。

## 4. 配置、bank 与 v1.0 隔离

`configs/rtd/v1_1_bfcl.yaml` 是本候选的统一 v1.1 配置入口：

```yaml
protocol_version: 1.1.0
rounds: 2
budget_checkpoints_bank_fraction: [0.10, 0.25]
slots_per_step: 40
exposure_slots_per_window: 40
max_new_packages_per_window: 20
replay_bank_path: data/rtd/v1_1_bfcl
acquisition_posterior_refresh: persistent
source_estimator: cv
source_samples_per_state: 2
cv_cs_mode: loo
gate: linear_sigmoid
```

**最后四项目前只能采用实际可用的 D1 接口；D8 gate/solver keys 缺失，不能宣称已经配置了 rev 3 主机制。** YAML 顶部明确写出该限制。D1 的 soft/scalar 与 hard2 诊断选项继续可用。v1.1 loader 保留 D2/D3 支持的显式两/三轮变体；正式入口 YAML 为两轮。D2/D3 说明第 5 节的“三轮默认”是 `24f5fca` 时的文字，已由 `d9954f0` 的两轮 YAML 取代。

只读执行 `load_config` + `bank_audit(build=False)` 验证现存 bank：420 包，recorded 内容预算分母 55,370，两个累计 ceiling 为 5,537 / 13,843；证书 SHA256 为 `caa729a786671562434dbe60f9c3cbf77996ae26b249352da48d955c71d1f25d`。只读 public/integrity/audit，不重建 bank、不揭示未购响应。

四份 `configs/rtd/v1_bfcl_c25*.yaml` 在合并前后 SHA256 完全一致，包括 linear/scalar 和两个 frozen 配置。既有 v1.0 manifest/window 字节回归与 D1 hard2 回归均纳入完整套件；真实源码 identity 随代码变化是正常行为，没有伪造旧 hash。

## 5. 与 rev 3 的语义差距及后续归属

本次以 rev 3 §2.3/§3.3/§3.4 判定语义，**不把 D2/D3 的旧数据采购参考重新命名为同批次 d=0 参考**。缺少 D8 输入意味着下列核心工作仍未交付：

| 归属 | 合并代码的实际行为 | 必须补齐的 rev 3 要求 |
|---|---|---|
| D8 输入恢复 / D8b | 只有 D1 soft/CV + 原 sigmoid；没有 α/d 分解或 d 求解器 | 引入 D8 实现与测试，再落地状态 α、成对来源估计器 `(1−α)Gsoft+αGT−d(G1−G2)/2`、合法边界、停梯度和分块 α 更新。保留 teacher-mass、无偏性、对称性等测试。 |
| D8b / D7，§2.3 | 先抽 E 个旧来源，再采购新包；每新包另抽一个来源槽；旧均值事后乘 `1-m/E`。linear gate 特征含当前来源长度/似然 | 采购完成后先固定实际证据、w 与只用状态特征的 α，再抽来源；权重和 α 在本批来源采样前冻结。当前 CV 独立控制组只实现 D1 估计器，不等于这个冻结契约。 |
| D8b / D7，§3.3–3.4 | `reference_feedback` 在旧数据虚拟更新处取得并供采购插入标签；`actual_feedback` 在最终混合更新处取得，可按 hash 复用 | 为同一实际批次另建 `theta_S(0)`，在那里取得 d 的反馈；固定相同证据/w/α/来源后提交 `theta_S(d)`，再用独立记账的 actual 反馈更新 α。采购历史参考与来源选择参考必须分开保存、验证身份和计费。现有两种反馈角色不能冒充 D8 的两个取值点。 |
| D8b / D7，非决策步 | CV 每次 draw 刷新组；soft/hard2 仍用轮内缓存；非决策步没有 d 或反馈年龄 | 主机制每步重采来源对，最近反馈重投影到本步 v，求解 d 并记录反馈年龄；冻结 p_t 参数与刷新样本是不同概念。 |
| D9 | 标签为共享旧数据参考的一阶 `v_q/E`，后验持久化；没有 Gram 交叉项或联合 d 优化 | 冻结 w，仅联合解 d；构造 F-hat 的删包代理边际标签，加入当前 α 分布/已购集合摘要/反馈年龄等购买前上下文；用独立反馈检验代理，保留可加近似的明确命名。 |
| D7 | batch executor 的求解→账本事务→实购集合保留；fixed_evidence 是最终集合重训 | V1 必须复用 V0 的购买时间、状态、包权重和重复次数；各臂从各自冻结学生重采来源。现有 fixed_evidence 不能冒充完整曝光重放。 |
| D5 | 基础 spend/exposure/covariance/reliability 日志可用；D1 梯度诊断 helper 可用 | 汇总每轮失败率/方差、α/d 分布、反馈年龄、两参考点算力账、z 与独立配对收益关系、修复/损伤与正式预算曲线；不能仅凭测试通过宣称机制收益。 |
| D6 | 有数学/事务测试，无完整三组机制干预 runner | 同 α 的 d vs 0、打乱来源对应、联合 vs 独立控制；保留来源估计器对照。只用合法已购 inner 数据，实际收益检验使用另批反馈。 |

工作期间外部暂存的 **rev 3.1** 进一步澄清：E=40 是 **40 个监督记录单元 / 80 个来源动作**，4 状态 microbatch 累计 10 次；联合 d 禁止按原始 z 单坐标预筛；非决策步不确定性必须沿新 v 重新估计；F-hat 包含完整教师注入更新，删包以预先确定的旧池单元补位，强制 d=0 时仍能区分教师价值。当前合并没有偷偷改曝光单位或实现这些新要求，后续 D8b/D7/D9 必须统一落实。

## 6. CPU 验证

测试使用指定共享 Python（只读），无 GPU，BFCL 文件锁和测试产物放 `/tmp`：

```bash
PY=/home/xueqi/hq/projects/tc-alignment/.venv/bin/python
PYTHONPATH=src:. CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  BFCL_PROJECT_ROOT=/tmp/rtd-v11-m1-bfcl \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  "$PY" -m pytest tests/test_rtd_*.py tests/test_checker_bridge*.py -q -p no:cacheprovider
```

完整结果：**999 passed，6 warnings，428.02 秒，exit code 0；零跳过**。包括 RTD 全部测试、两个 checker bridge 测试文件、17 个新增合并集成测试，以及真实 ALFWorld CPU fixture。无需使用用户允许的 symlink-limitation skip。日志：`/tmp/rtd-v11-m1-full.log`。

专项首次执行的 12 个失败均来自新增恢复测试误比较独立运行的时间戳/hash chain；学生轨迹在该断言之前已相等。已改为逐事务语义比较，没有修改产品代码以迎合时间戳断言。最终完整套件包含该修正。

最终 `diff --cached --check` 通过，临时 index 无未解决条目，`MERGE_HEAD=d9954f093ce2f50f804d596390ca136e4144d3ae`；仅暂存 M1 与 incoming D2/D3 文件。D1 runtime/scorer/estimator/专项测试及 incoming 的 8 个测试/fixture 文件逐字节核对通过。原 `PROJECT_STATE.md` 补丁保持一致，外部 rev 3.1 方法文档也未修改。测试通过验证的是上述合并候选，不代表缺失的 D8 或 rev 3 主机制已经交付。
