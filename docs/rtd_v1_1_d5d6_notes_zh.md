# RTD v1.1 D5+D6：rev 3.1 指标与冻结窗口配对干预

规范为完整 `docs/RTD_V1_1_METHOD_ZH.md` rev 3.1，尤其 §6.2–6.3；已通读 D1、D2/D3、D7、D8、D9 笔记。D7 历史 rev 2 参考点描述不再适用。工作只在 `/home/xueqi/hq/projects/tc-alignment-v11a`；`data/`、`envs/`、共享 Python 环境只读。没有 GPU、生产训练、官方评测、在线教师、暂存或提交。

## 1. 共用更新与三个反馈角色

D5/D6 复用 D9 的 `joint_surrogate.validate_pair`，其从 D9 原配对循环提取：两分支各调用一次 `execute_update`，同一冻结 start，通过 `BatchReference.selected(d)` 形成实际参数，再执行独立完整任务 rollout。原训练 runner 也调用这个 helper。没有另写联合求解器；`joint_surrogate.control` 继续调用 `alpha_d.solve_d`，独立控制仅去掉 K 的非对角项。

| 反馈角色 | 参数点 | 用途 |
|---|---|---|
| `acquisition_reference_feedback` | 旧池虚拟参考 | D9 采购 F̂、独立插入标签 |
| `same_batch_reference_feedback` | 本批 θS(0) | ẑ、ε̂、d 求解 |
| `post_commit_feedback` | 本批 θS(d*) | 固定 d 的分块 α VJP |
| `validation_*` | 独立执行的一步学生 | 仅诊断，不进入上述三个角色或后验 |

归档核对同批反馈的角色、参数 hash、逐轨迹统计 batch hash；对照评测复用 `AlphaDExperimentMixin.alpha_validation_return`。验证 RNG 使用独立域和分支身份，不推进训练 RNG。即便两分支参数完全相同，也分别 rollout，保留 Monte Carlo 噪声；不强行把实测差清零。JSON 记录 `validation_only=True`、`used_for_posterior=False`、`selection_feedback_reused=False`。报告拒绝被标记为复用选择反馈的验证记录。

`independent_insertion_control` 直接保留 D9 冻结标签；F̂ 的完整集合 d、删包 d、边际值直接来自 D9 `acquisition_surrogate`。删包学生复用 `replacement_batch`，按原槽 w 用预定旧池证据补回。**删包对比的 α 可以随被替换的教师记录改变**，它是采购代理验证，不冒称同 α 的核心机制隔离。

## 2. D5 指标

新模块 `src/bfas/rtd/metrics_v11.py` 实现算术与流式诊断，`experiment_metrics_v11.py` 在决策窗和轮末接入。

### 固定集与解析失败

运行开始，在 manifest 的 `fixed_task_set_v11` 中冻结父任务内容 hash、官方 task ID、task-start 状态定位、**完整 FullState 内容 hash**、槽位顺序、两折和整体 hash。选择只依赖公开支持父任务内容 hash，不依赖采购、来源输出或表现。恢复和评测核对同一任务身份，缺失状态报错，不换任务。

**现有 BFCL support 为22/18个父任务；其中7个 memory/web 任务没有已有的可执行起始状态适配器，实际 runnable 状态为18/15。** 只在运行开始按公开适配器可用性排除这7个父任务，名单保存到 `excluded_unavailable_parents`；冻结以后缺失任何状态仍报错，不替换。默认 `short_fold: repeat`：每折固定20槽，短折按同一内容排序循环补足；真实 BFCL 两折分别是18个独立状态+2个重复槽、15个独立状态+5个重复槽。manifest/JSON/Markdown 显式记录 `unique_states`、`repeated_slots`；这些重复槽不是额外独立任务。需要严格20个不同状态时配置 `short_fold: error`，在运行开始拒绝短折。没有修改 support 划分、引入校准/认证任务或读取封存教师内容来凑数。

每轮末每槽独立抽 K=4 条 T=1、top_p=1 动作：80条/折、160条/轮。`CheckerBridge.check_syntax` 暴露**已有** `bfcl_decode.guard_rtd_decoding` 的 Qwen 语法检查，支持 direct/独立 syntax subprocess；官方 worker 的原评分分派及冻结 scoring projection 保持不变，没有放宽身份保护或修改历史 evidence，也没有新造解析规则。纯合法文本允许通过语法检查，语法通过不等于完成任务；空输出、破损 tool-call framing、非法 JSON/调用结构按已有 guard 失败。checker 基础设施异常向上抛出，不当模型解析失败。与训练日志的 malformed/truncated 数量分开。

### 梯度方差

在固定的当前 inner-fold 状态、冻结教师记录、α、w、start、来源快照和预条件器上，重采 R 次，报告 **一步预条件更新位移 L2 范数** 的均值、全部 R 个范数及总体方差 `var(norms, ddof=0)`；不是梯度协方差迹。反馈折的固定状态仅评测，不进入梯度、预条件器或标准化。

每次每状态8条新来源，2硬/软共享前2条，8硬使用全部8条。三者复用 D1 `source_gradient_pair` 与 D8 `estimate`；均固定 d=0、同 α/教师目标。语法过滤使用前2条的合法硬样本均值；全失败时用同批软来源补位，保留教师注入和原曝光槽。它明确是**有偏诊断**，不进入主方法。

另报 `alpha_d`：前2条构造 `build_reference`，在新的方向上重投影冻结的同批反馈统计，再经同一控制器求 d。非同参数点的历史反馈是近似诊断，日志明确不覆盖反馈陈旧偏差。所有额外来源、梯度和 rollout 计算单列，不藏进每步40单元/80动作。

### 修复、损伤、ẑ 与 F̂

每个决策窗从冻结 start 测固定集 greedy 成功，再比较 learned d 和 d=0 的独立一步学生；两边 α、w、教师和来源完全相同。记录 failed→success 的修复、success→failed 的损伤、保留、仍失败与净修复；同一重复槽在前后保持对应。greedy 使用原任务执行器/官方 checker，HF 使用 KV-cache greedy generation；不拿 T=1 回报充当 greedy 成功。

每窗额外选一个允许非零 d 的坐标（按状态 hash/曝光下标确定，不看 ẑ），实施 `d_i=min(.1,α_i,1−α_i)` vs d=0。报告 `d_i ẑ_i` 与实际 ΔJ 的配对预测误差，以及 **ẑ_i 与 ΔJ/d_i 的 Pearson 相关**。D6 可用 `--z-probes` 增加预定坐标数。α 端点不可探测，直接无可用相关样本；不伪造零斜率。

F̂ 的验证复用 D9 每窗按 query ID 预选的一个已购包：完整集合 vs 删包补位，预测是原代理边际值，实测是独立反馈完整任务回报差。不是似然改善，也不拿拟合反馈自证。少于2对、恒定预测或恒定收益时相关系数为 `null`，附原因；没有显著性或机制有效结论。

### 联合/可加、账本与官方 Overall

分别报告：

1. D9 `predicted_joint_gain − sum_independent_gains`：代理内部的联合/可加差。
2. 独立验证 learned d vs d=0 的实测 ΔJ 减去可加预测：包含有限更新误差和 rollout 噪声；输出 mean error/MAE/RMSE。
3. D9 joint vs independent 配对：预测两控制器差 vs 实测两学生差。

采购以 **committed trajectory/steps** 为权威，每窗列购买数、计划购买数、窗口/累计支出、剩余授权、窗口授权、K/预测成本/硬 cap/停止原因。重复 compute 尝试不能多算窗口。报告直接沿用 v1 collector 的 bounded `Inputs`、`official_scores`、`ledger_charges`；官方分数读取绑定 campaign 的完整 CSV 和 evaluation 完成状态，不用本地 greedy 诊断代替官方 Overall。

## 3. D6 runner 与归档

启用规范 YAML 的 `metrics_v11` 后，每个决策窗保存：

```text
<run>/controls/windows/r1-s01.pt
<run>/controls/windows/r1-s01.json  # SHA256、manifest/start/feedback/evidence/geometry binding
<run>/metrics/round-1.json
<run>/metrics/round-2.json
```

PT 包含完整有限更新所需的 LoRA start/source、BatchReference、来源对、w/α、FrozenStep、逐轨迹同批反馈、实购池、预定旧池补位和 D9 代理/验证结果。只加载本 runner 生成的本地 hash 绑定归档。CLI 读取归档，不实例化恢复训练器、不采购、不安装更新到 backbone，也不修复输入 journal。GPU 装载只移动模型大小的 start/source/theta0 与预条件器；O(E×参数)方向、基线梯度与逐轨迹统计始终留在 CPU。发布 JSON 描述文件前中断留下的孤立 PT 可从 durable 窗口状态重建。原 D9 运行未保存这些窗口中间张量时，只有 JSON 日志不足以精确重建其冻结窗口；需要新运行生成归档，不能假装从已清理的 round-state 恢复同一来源证据。

执行顺序：

1. learned d vs d=0，同 α。
2. 所有 Y1/Y2 交换后重新计算参考/方向/Gram，并重新求解；检查 d 反号、更新保持相同。与主对照共用原 D8/D9 算子。
3. learned d vs shuffled pairing：按状态 hash 排序交替定义本地来源的正/反规范顺序，得到原 Y1 的平衡规范下标0/1，再按 seed 排列这组原分配；重复曝光同一状态使用相同分配。保留每条状态自己的两条完整动作，保持教师目标、α、w 和**带符号 d 的整个集合**。改变的是哪条本地行为接收 Y1 系数；不跨状态移动文本，不为打乱重新求 d。恒等排列不改变任何来源；单状态或无实际方向交换时明确记录 `changed=False` / `no_orientation_changed`，不制造一个虚假打乱效果。预测差复用 D9 `objective` 在原/打乱方向上评估同一个 d。
4. joint controller vs independent control：同 pool、同反馈统计和每臂一步，只改变 K 非对角项。
5. D9 F̂ 配对和预定 z 坐标验证。
6. **最后**执行2硬/软/8硬/语法过滤对照及 R 次来源方差诊断。每个来源估计器从共同 start 执行一步。

每个已评测分支包含固定集解析失败、greedy 修复/损伤、held-out feedback T=1 回报、实际更新范数、教师质量和参数/反馈批次身份。完整 source-resampling 方差共用冻结窗口起点并单列，不声称是在各分支更新后的 checkpoint 又训练了一步。输出 `controls.json`、`controls.md` 和实际计算 `compute.jsonl`。

## 4. 配置与命令（GPU 未执行）

规范 YAML 默认开启指标和窗口归档，历史没有 `metrics_v11` 的配置继续保持旧行为：

```yaml
metrics_v11:
  enabled: true
  variance_resamples: 4
  archive_windows: true
  short_fold: repeat
```

HPG 从本隔离工作树提交，固定一张 B200、两处预先指定的晚期窗口（V2 r1/s10、r2/s10），**总计约4–6 GPU·h 是排程估计，未实测吞吐**。脚本申请6小时、128GB CPU内存供 O(E×LoRA参数) 方向/反馈归档；更复杂完整任务可能超过估计，不能删任务来凑时长。所有可写缓存在当前 checkout 的 `.cache/rtd-v11-controls`，共享 `data/`/`envs/` 仍只读。

```bash
mkdir -p logs
# 训练先按 D7 的 V0/V2/V1 流程产生 rev3.1 窗口归档。
# RTD_PYTHON 可指定 HPG 上已有的相容 Python 环境。
sbatch scripts/rtd_v11_controls_hpg.slurm \
  results/rtd_v1_1/V2 results/rtd_v1_1_controls

# 在已经分配的一张 B200 内，只执行单个冻结窗口：
PYTHONPATH=src:. "$RTD_PYTHON" tools/rtd_v11_controls.py \
  --window results/rtd_v1_1/V2/controls/windows/r1-s10.pt \
  --out results/rtd_v1_1_controls/r1-s10 --resamples 4 --seed 0 --z-probes 4
```

两者是替代启动方式，输出目录必须新建，不重复执行同一路径。不要把含大量中间张量的全部窗口归档当成已压缩数据；按需要另行管理磁盘保留，本实现不自动删除归档。

CPU 报告命令（只读输入；新建 Markdown/JSON）：

```bash
PY=/home/xueqi/hq/projects/tc-alignment/.venv/bin/python
PYTHONPATH=src:. CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  "$PY" tools/rtd_v11_report.py --root "$PWD" \
  --runs results/rtd_v1_1 --campaigns results/bfcl_std_hpg \
  --base-overall 46.06 \
  --controls results/rtd_v1_1_controls/r1-s10/controls.json \
  --controls results/rtd_v1_1_controls/r2-s10/controls.json \
  --out docs/rtd_v1_1_report_zh.md
```

报告固定 V0/V1/V2 和10%/25%两点，缺失条目显示未完成；V1−V0 为自适应蒸馏整体效果、V2−V1 为自适应采购整体效果。CLI 当前生产支持 BFCL；公共 Python API 可注入其他具备 task-start feedback 与 syntax checker 的适配器，未声称已完成 ALFWorld/AppWorld 的生产 D6 CLI 集成。

## 5. CPU 验证

```bash
PY=/home/xueqi/hq/projects/tc-alignment/.venv/bin/python
PYTHONPATH=src:. CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  BFCL_PROJECT_ROOT=/tmp/rtd-v11-d5d6-bfcl-full \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  "$PY" -m pytest tests/test_rtd_*.py -q -p no:cacheprovider
```

BFCL lock 仅写 `/tmp`；ALFWorld 使用既有临时 fixture；没有修改环境路径保护。新增专项首次 **15 passed，30.61秒**；D5/D6+D9 集成 **40 passed，63.18秒**。覆盖真实 checker bridge 语法、160条独立T=1采样、固定名单/短折说明、方差/相关/修复算术、来源打乱与对称性、同α、D9执行/求解复用、反馈角色拒绝、归档篡改、resident参数/RNG不变、轮末及恢复、官方CSV读取和报告新文件保护。

首轮完整套件为 **1066 passed、5 failed、6 warnings，600.94秒**：4项历史 checker 身份回归及1项 raw source manifest 字节 oracle。修复采用独立 syntax worker，官方评分函数/分派和旧 scoring hash 不变；历史 oracle 仅把新增诊断造成的 raw checker 源码哈希归一化到原值，运行 manifest 仍保存真实源码。身份+专项 **93 passed，38.68秒**；另两项 subprocess/历史manifest 检查 **2 passed，5.57秒**。新增真实轮指标报告测试首次因 tiny fixture 仍携带旧三点 fractions 而失败；修正 fixture 为两点，未放宽报告校验。最终新增专项 **16 passed，18.12秒**。第二次完整运行发现上述 tiny fixture 问题，结果为1071 passed、1 failed、6 warnings（554.61秒）；修正后第三次完整 `tests/test_rtd_*.py` 为 **1072 passed、6 warnings，591.64秒，exit code 0**，没有 skip/xfail。完整日志 `/tmp/rtd-v11-d5d6-full-verified.log`；六条警告仍来自既有 tensor 标量转换及 PEFT 配置提示。完整集结束前，收尾补上固定状态内容 hash 绑定；随后全部 `tests/test_rtd_v11_*.py` 重跑为 **183 passed、1条既有警告，200.82秒**，日志 `/tmp/rtd-v11-d5d6-v11-final.log`。实际 BFCL support 的只读核验进一步确认18/15个可执行状态；增加运行开始排除不可执行适配器后，最终D5/D6专项 **16 passed，19.84秒**，日志 `/tmp/rtd-v11-d5d6-fixed-final.log`。该检查未执行模型、教师或官方任务评测。HPG 脚本只做 `bash -n`；没有提交 GPU 作业。

最终 `git diff --check` 通过；结束执行 `git status --short` 与 `git diff --stat`。改动及新增文件均留在本 worktree，未暂存、未提交；未改动 `data/`、`envs/` 或其他 worktree。
