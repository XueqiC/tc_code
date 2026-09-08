# RTD v1.1 D7：冻结实现约定

范围仅为 `/home/xueqi/hq/projects/tc-alignment-v11a`。`data/`、`envs/` 及共享 Python 环境只读；本次只有 CPU 测试，没有训练作业、GPU、在线教师、官方评测或提交。开始时已有的 `PROJECT_STATE.md` 修改保留。

已完整阅读方法文件及 D1、D2/D3、D8 笔记。**版本差异明确处理**：工作树的 `RTD_V1_1_METHOD_ZH.md` 已是 rev 3.1，§3.4 将 d 的反馈点改为同批次 θS(0)；此次 D7 指令明确要求 rev 2 的 **旧证据虚拟参考 θ̄⁺ 同时供应插入值与 z**。本次按此次指令执行，manifest 标记 `alpha_d_d7_rev2_frozen`。D8 的同批次 θS(0) 仍用于构造实际梯度的仿射分解，但不再是回报评估点。不将此实现冒称为满足 rev 3.1 的同批次回报下界前提。方法文档本身保持原文，D8 的冷启动拒绝、三批反馈及未实现 V1 重放说明由本笔记取代。

## 1. 约定 → 代码 → 测试

| 冻结约定 | 实现 | 主要测试 |
|---|---|---|
| 三臂只有一个 ĝd 算子 | `alpha_d.components/estimate/build_reference/blocked_alpha_vjp`；臂只决定采购与 α/d 的生产方式 | `test_all_three_arms_execute_one_estimator_and_prefix_implementation`；D8 枚举无偏、同 α 对照、交换对称性、VJP 差分 |
| 每步从本轮冻结学生重采两动作；T=1、top_p=1；禁止缓存 | `alpha_draw_pairs` 强制 `refresh=True, cache=False`，检查缓存对象、重复对象、RNG 前进和来源快照；记录 draw ID、sample/action hash、RNG 前后摘要 | D7 缓存拒绝、三臂共享路径、24 步 V0/V1；D8 非决策步重采与恢复 |
| 完整词表软项的位置与 EOS/cap 规则一致 | `source_scoring.sampled_prefix_positions` 是硬/软共用入口，实际 EOS 纳入，截断不补 EOS | D7 EOS/cap 位置；D1 全词表独立 oracle、逐前缀零梯度、真实 Qwen/PEFT CPU |
| E=40 单元、80 来源动作、K≤20、4 状态×10 后提交 | `alpha_revealed` 固定记录/w/α 后才抽样；`build_reference` 每4状态累计；`supervision_records` 固定 episode→有序记录 | D7 满40冷启动/零采购；D8 40/80/10、adapter、非决策步；D2 预算/K |
| 两个回报点分离 | 旧池 d=0 `InsertionReference` 上的 `virtual_reference_feedback` 供应插入值/z；已提交学生上的 `alpha_post_commit_feedback` 供应门控 VJP；核对参数 hash 与对象身份 | D7 不同参数状态、计算角色分离、两种反馈替换拒绝；即便同 hash 也重新 rollout |
| 先批量混合提交，再入累计池 | ledger 的 paid 集与 `state['owned']` 的训练池分开；`alpha_feedback_commit` 才将新包入池；`alpha_d_pool` 记录前后成员 | D7 V0/V2 首窗到非决策步的 pool transitions |
| V1 重放完整曝光，不重放来源 | `conventions.py` 的原子 `exposure_schedule.json` 导出、身份/hash验证；`batch_reference` 直接取 V0 实购集合；记录/w在采样前恢复 | D7 24 步、两轮、非均匀权重/重复次数逐项相等，样本 hash 序列不同；完成后恢复；文件篡改/未完成拒绝 |
| 求解器不凌驾于硬账本 | V0/V2 共用 `batch_selected`，按 query ID 排序逐包 reserve→reveal/settle（原子释放差额）；每笔更新公开候选/余额，不补买计划外项 | D7 两臂 optimistic-cost→hard-cap-tail，损失只含实购集；D2 原逐事务崩溃恢复 |
| 失败计费准确 | `Ledger.online_request`：本地 precheck 在调用前；收费 receipt 在任务验证前 settle；消费状态未知的异常保留 reservation 等待核对 | D7 fake online broker：本地失败零调用/零账，收费输出后任务失败仍扣17、恢复后不退款 |
| 10%/25% 为授权上限，实际支出作横轴 | 原累计授权/actual-spend报告保留，manifest 明写不强制花完；`acquisition_execution` 记录 planned/acquired/drops | D7 零采购与 cap-tail 未花满；D2 预算证书/账本；完整报告回归 |
| 冷启动 old pool = paid records + reference states | `alpha_old_pool` 按确定顺序合并；reference record 的 teacher=None、α=0，只含软来源项；不免费解封 bank | D7 满40首窗、K=0 identity、V1首窗重放；D8原拒绝测试改为合法reference单元测试 |
| 三臂及措辞作为配置值 | `conventions.ARMS/arm_config`，CLI `--arm` / YAML `arm`；`--replay-schedule` 向 coordinator worker 传递，resume核对同一文件hash | D7 映射/manifest/launcher；D1 v1.0 hard2字节回归；D2旧诊断改用显式soft |
| 评测与认证分开 | manifest/报告/新v1.1官方结果标记 `development evaluation`；`certification_metadata` 只建立冻结checkpoint、held-out数据、bootstrap标签，不产生统计界 | D7 manifest/认证plumbing；全套官方评测身份/恢复回归 |

测试文件：`tests/test_rtd_v11_conventions.py`；相应更新 `test_rtd_v11_alpha_d.py`、`test_rtd_v11_source_estimator.py`、`test_rtd_v11_posterior.py` 及共用 CPU fixture。D1 的旧 manifest 字节 oracle 仍保留原摘要；报告文件新增措辞引起的**实际 source hash**单独验证后，只在该 oracle 中归一化为原报告源码 hash。运行 manifest 始终记录真实源码身份，没有伪造旧 provenance。

## 2. 曝光、来源与参考计算

一个单元 = 一个状态监督记录 + 两个来源动作。无教师的 reference record 也是合法单元，`teacher_evidence_units` 与 `reference_pool_units` 分别计数，和为 `exposure_units`。默认每步40/80/10；配置 `slots_per_step<40` 始终标记 **`random exposure`**。兼容字段 `source_only_placeholders` 仅是 reference 单元计数的别名，不能将其当作教师曝光。

旧池在步开始时仅包含以前提交后入池的付费记录，以及合法 inner fold 的参考状态。新 episode 当步最多选一条 adapter 固定映射的监督记录；提交后该 episode 的全部已付记录才可进入后续抽样。每条记录携带 query ID、record index、完整 state hash、teacher是否存在、w、是否新包、重复总次数。不存在用未付费的教师响应填满40单元的路径。

旧证据虚拟参考也使用40条预先确定的旧池记录，其来源采样和流式梯度计入独立 `alpha_d_old_virtual_reference` 计算。**每次实际提交仍是40单元/80来源动作；额外参考与pilot计算不能隐藏在该80中。** 两个rollout批次分别记录 `virtual_reference_feedback` 与 `alpha_post_commit_feedback`。相同参数hash（例如K=0、α=d=0）也不复用rollout。非决策步沿新方向重投影最近反馈，继续记录反馈年龄和误差不覆盖陈旧偏差。

来源/反馈 RNG 使用 `digest([training_seed, arm, 'source_and_feedback'])` 派生独立流；训练初始化seed仍为0。各臂均使用同一T=1/top_p=1 categorical sampler。独立抽样允许token序列偶然相同，不要求每一对token hash都不同；测试核验整条样本hash序列不同及每次真实抽样的RNG/draw身份。

## 3. V0 导出与 V1 重放

每个 committed 阶段原子重写运行目录下的 `exposure_schedule.json`，恢复会从durable step重建相同文件，不追加重复步。文件只包含曝光和采购，不包含α、d、来源文本、来源token或rollout。

身份绑定 bank public/integrity hash、数据与基座hash、初始参数hash、support parents、预算上限、训练seed、轮数、E/K。V1要求完整V0日程：正式模式24步，smoke模式1步。购买时仍经过自身硬账本；无法获得V0计划中的实购包则拒绝继续，不悄悄换包。入池前后成员、记录教师标记、重复次数、权重和窗口都逐项检查。文件SHA256进入V1 config/manifest；恢复时变动会被拒绝。

V1使用自己的冻结学生、状态α和d；旧参考记录也随日程重放，来源则重新采样。pilot属于独立训练校准计算，不是提交曝光。V0/V1额外计算按各自journal实际记录。

## 4. V2 联合代理与计费边界

`joint_surrogate.py` 在同一旧证据虚拟参考梯度上计算完整更新代理，包含教师注入的基线位移、来源方向的Gram交互项以及误差惩罚。已购包LOO用预先抽定的旧池记录补回同一位置，保持w；不增加逐候选环境rollout。V2后验学习这些联合代理边际值，公开候选仍由D2预算greedy近似选择。该实现不声称知道未购教师梯度或能精确求解未知候选间的相互作用。

所有d强制为0时，代理仍区分有益/有害教师更新，并对重复方向计入联合二次成本。标签明确写 `joint_full_update_surrogate_with_teacher_injection`。当前标签协方差沿用D2的**可加反馈噪声代理**，不声称覆盖联合求解器不确定性；实际配对收益检验和更完整D9诊断仍是另一个任务。

硬账本`settle`在一个durable事件里记录真实消费并释放cap差额；仅剩消费额影响后续授权。新online hook本次仅由fake broker测试，无生产网络调用。在线provider adapter必须将已收费错误响应转换为包含usage的receipt；receipt结算后，即使任务验证失败也保持账单。没有receipt的异常不推断免费，不释放尚待核对的reservation。

## 5. 标签与 manifest

- `V1−V0`：**自适应蒸馏整体效果**，α与d都改变；**同α的learned d vs d=0**才隔离核心机制。
- J：`temperature_1_stochastic_policy_expected_return`。官方greedy score是独立指标。
- 基座：**发行方原始 checkpoint,未进行本项目 bank 适配**；记录实际基座hash，拒绝CRCD路径作为v1.1基座。
- manifest明确BFCL m=40；ALFWorld训练父组135，fold0 inner79/feedback56、fold1 inner56/feedback79，包数63/44，排除4个probe父组。实际support存在时另记父hash角色集合；ALFWorld manifest入口直接从验证后的support生成`parent_group_roles_by_fold`。
- 每轮官方评测标记 **development evaluation**。认证仅是冻结后独立held-out数据上bootstrap bounds的单独标签和身份plumbing，`bounds=None`，本次没有产生认证结论。

## 6. HPG 启动与24小时续跑

从本工作树根目录提交，脚本固定B200×1、CPU8、内存64G、24h。HF cache环境由 `tools/aw_hpg_common.sh` 设置；运行期缓存改到当前checkout的 `.cache/rtd-v11`，不写 `envs/` symlink。默认 `RTD_COMMAND=auto`：已有manifest则resume，否则run。可以显式使用 `run|resume|smoke`；resume默认只读已保存config，适配原有依赖作业relay。脚本支持 `RTD_EXTRA_ARGS`（按空白分词、不eval）以及额外位置参数。

三臂共用 `configs/rtd/v1_1_bfcl.yaml`，无须手动编辑V0的α/d配置：

```bash
mkdir -p logs
RTD_V0_JOB=$(sbatch --parsable --export=ALL,RTD_RUN_DIR=results/rtd_v1_1/V0 scripts/rtd_v11_run_hpg.slurm V0)
sbatch --export=ALL,RTD_RUN_DIR=results/rtd_v1_1/V2 scripts/rtd_v11_run_hpg.slurm V2
sbatch --dependency=afterok:${RTD_V0_JOB} --export=ALL,RTD_RUN_DIR=results/rtd_v1_1/V1 scripts/rtd_v11_run_hpg.slurm V1 --replay-schedule results/rtd_v1_1/V0
```

24小时后继续同一运行目录，示例V0 relay：

```bash
sbatch --dependency=afterany:${RTD_V0_JOB} --export=ALL,RTD_COMMAND=resume,RTD_RUN_DIR=results/rtd_v1_1/V0 scripts/rtd_v11_run_hpg.slurm V0
```

V1/V2同理，替换对应作业ID与运行目录；V1恢复从manifest读取已绑定schedule，无须再传路径。若V0需要relay，V1的afterok依赖应指向完成V0的最后一个relay作业。脚本不自动提交额外作业；本次未执行上述sbatch命令。

## 7. CPU 验证

```bash
PY=/home/xueqi/hq/projects/tc-alignment/.venv/bin/python
PYTHONPATH=src:. CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  BFCL_PROJECT_ROOT=/tmp/rtd-v11-d7-bfcl \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  "$PY" -m pytest tests/test_rtd_*.py -q -p no:cacheprovider
```

BFCL锁文件只写`/tmp`；ALFWorld使用原有临时fixture，未改路径保护。

最终完整 `tests/test_rtd_*.py`：**1031 passed，6 warnings，489.61秒，exit code 0**。没有skip/xfail或环境失败。完整日志：`/tmp/rtd-v11-d7-full-final.log`。六条警告来自已有tensor标量转换与PEFT配置提示。

最后的空采购窗口候选日志调整另通过D7/D2/D8专项：**74 passed，1 warning，95.90秒**，日志 `/tmp/rtd-v11-d7-executor-final.log`。此前D7/D8/D1及恢复专项 **86 passed**；旧恢复/配置兼容专项 **51 passed**。CLI还以替代campaign核验了V0/V1/V2参数映射，没有加载生产模型或启动评测。

首轮完整套件为 **1024 passed、6 failed**：失败全部来自新增配置入口对旧`resume_config`短路行为和一个旧CLI测试替身的影响。已恢复旧配置完全相同时的原样读取，以及R0/R1入口的原调用形式；随后上述51项回归和完整1031项均通过。初次D8专项中与新约定冲突的冷启动/参考点旧断言已按约定更新；未隐藏或跳过失败。

`git diff --check`通过；最终已执行`git status --short`与`git diff --stat`。所有修改留在当前工作树，不暂存、不提交；原有`PROJECT_STATE.md`修改未动。
