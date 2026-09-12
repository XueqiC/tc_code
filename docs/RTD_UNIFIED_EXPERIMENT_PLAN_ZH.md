# RTD unified replacement：P1–P3 实验计划

状态：**P1 preparation 启动工程已实现；P1、P2、P3 性能实验及所有实验臂结果仍为 PLANNED**。P1 的配置、D0–D3/归因 presets、共同 recorded exposure、统一 smoke/stage profiling 与命令见 `RTD_UNIFIED_P1_PREP_ZH.md`。只有 CPU 验证与只读 bank 审计；没有 GPU、API、Gemma 或 benchmark 性能运行。

P1 preparation 的本轮最终 CPU 验证：**2,436 passed、4 skipped、6 warnings，931.44 s，退出码 0**；独立 focused suite 55 passed。GPU/API 实验 0；不据此填写任何任务收益或 GPU 速度。

## P0 实际交付记录

- 已实现：独立不可变 TeachingProblem、raw/CV 与长度修正、完整交叉项 auxiliary QP、单一共享 objective、单步 trial/commit、v1.1 回报适配、公开特征/账本、嵌套采购标签、多教师 simplex。
- 数学纠正：随机实际长度平均不保证快照处零梯度；不能直接用 prefix KL/L 声称匹配 hard 梯度。旧 v1.1 α/d 实际代码使用 total NLL，不能混称同口径。证明与数值反例见 `RTD_UNIFIED_METHOD_ZH.md` I1–I11、C1–C9。
- CPU 验证：51 项 unified 测试单独通过（8.36 s）；最终完整命令 `PYTHONPATH=src:. /home/xueqi/hq/projects/tc-alignment/.venv/bin/python -m pytest -q tests` 得到 **2,387 passed、4 skipped、6 warnings，884.06 s（14:44）**，退出码 0，包括全部 51 项 unified 测试。环境设置见下；warnings 来自既有 tensor-to-scalar 与 PEFT fixture 路径。不以 toy 测试声称机制性能。
- worktree 测试准备：原 AppWorld 测试依赖的 `tests/fixtures/appworld_official/tasks/50e1ac9_1/logs/lm_calls.jsonl` 被现有 `logs/` 忽略规则排除。本次从主 checkout 原样复制，SHA256 为 `419da65489aec088fba0afc6a1bcf0f59b2b5329a34173e5dbad9cb093056278`，未修改旧测试/adapter；该本地 fixture 仍为 ignored，不混入 P0 提交。
- 只读共享 BFCL 环境不能在包目录创建 lock，故使用其既有 `BFCL_PROJECT_ROOT=/tmp/rtd-unified-p0-bfcl` 配置，仅迁移输出/锁；54 项 malformed/return-gradient 测试在此配置下通过（53.09 s）。完整 suite 另固定 `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`；这些环境设置不改变测试集合或断言。
- 首次完整运行得到 2,386 passed、4 skipped、1 failed（872.51 s）；唯一失败为既有历史 manifest-byte oracle。在 `/tmp` 解包未修改的 pre-P0 HEAD 后，单独运行该测试也得到同一 `e251f010…` hash。原因是此前 Azure BFCL merge 改了 adapter 的 operational 文件内容，冻结 scoring projection 仍为 `f6d6d341…`。仅在该旧测试中先断言这个 scoring hash，再把 raw adapter hash 归一到历史 `c3300f2` 的 `fd260576…`，继续验证原有 `428e3d9b…` manifest oracle；没有更新生产 hash、旧梯度 oracle 或任何 v1.1 runtime。修正后的 oracle 与 identity 检查共 79 项通过（8.36 s），随后最终全量复跑通过，见上方 CPU 验证行。
- 预算：已核验 v1.1 公共 cap certificate；详细内容成本/历史成本分离见 `RTD_UNIFIED_P0_BUDGET_MANIFEST.json`。本轮真实教师购买、API 调用、GPU 实验均为 0。
- 预测 stub：`predict_control` 返回 a_ref；`score_query_batch` 无预测分数。生产流式梯度、模型/harness 适配和训练调度尚未实现为 unified benchmark runner。

## P1：固定合法教师池，蒸馏机制验证（PLANNED）

先 BFCL 与 ALFWorld，统一数学程序；AppWorld 长程验证在核心确认后进行。相同初始化、合法已购池、固定曝光计划、机器、优化预算与调参预算。按父任务/轨迹/模板隔离教学、反馈与最终确认；反馈轮换折是训练用途，不能称 untouched test。探索遵循当前单训练 seed 0；最终复现使用用户后续冻结的种子安排，分别报告训练方差与评测重复方差。

Gemma-4-12B-it 在本分支只是明确配置的 student。PLANNED 前置工程：核验 checkpoint/tokenizer/chat template/动作终止/工具 schema、backend 实际采样与 scoring、LoRA 布局、logit transforms、权重许可证与本地可用性、显存；全部写入 identity。旧 Qwen 结果不能当新学生 base。不为任意臂按错误类型切换 loss。

| 核心臂 | 蒸馏程序 | 用途 | 状态 |
|---|---|---|---|
| D0 | 充分调优的教师 SFT+软 KL；同时报告已调好的 pairwise 锚点 | 强标准蒸馏对照 | PLANNED |
| D1 | 原 RTD v1.1 α gate+d 分块程序 | 可复现历史程序；原样保留 total NLL | PLANNED |
| D2 | 新估计器，约束每状态 a_i1=a_i2；跨状态标量联合求解 | 检验效果是否只来自剂量/教学重加权 | PLANNED |
| D3 | 新估计器、每来源自由系数、完整 K、exact local QP | 统一来源替换 | PLANNED |

损失口径必须预先登记。D0/D2/D3 的 matched 对比共用 per_sequence_mean（带正确长度修正）或另行固定 total NLL；默认配置是前者。D1 原样 total NLL 结果仅作原方法整体对照；需要纯机制归因时，另标 D1-compatible-loss 适配臂或整套 total-NLL matched 矩阵，不能悄悄改旧实现。KL 的前缀与来源 sampling measure 一致；不能用教师前缀替代。a_ref=.5、λ、η、P 更新周期、ε计算和 tolerance 都在数据/评分前冻结。

先匹配小预算闭环，再逐一运行归因，避免全表笛卡尔积：

| 配对归因 | 固定条件与比较 | 状态 |
|---|---|---|
| 联合几何 | D3 vs 去交叉项/预登记分组 K；都在完整 F 下评价 | PLANNED |
| 估计器 | D3-CV vs D3-raw；匹配 loss normalization、采样、曝光、反馈 | PLANNED |
| 来源对应 | D3 vs 打乱来源—系数对应；保持系数分布和教师证据 | PLANNED |
| 剂量/选择性 | D3 vs 固定每状态平均替换量的受限版本；报告 α 与 teacher 行为真实质量 | PLANNED |
| 通用回报修正 | 同反馈/计算预算的元重加权或受约束回报修正 | PLANNED |
| 参考点 | 可增加蒸馏前回报参考对照；不能声称覆盖全部 RL baseline | PLANNED |

通用强对照须先写方程。例如对固定基础增量，选择 Δ=Bc，最大化 ĥᵀΔ−λ‖Δ‖²_H/2−εᵀ|c|，并用相同可行系数集。如果 B=U、c=x，代数上就是 D3；报告等价性，不重复起不同名称的臂。非等价重加权基线可在相同 trainable loss 梯度族与曝光约束下优化状态权重；写明它能否改变 teacher–student 差和是否执行额外 RL step。不能仅与没有反馈的弱基线比较。

PLANNED 主报告：完整任务收益—保留损伤曲线、独立任务上的修复/损伤、α与实际教师行为质量、来源梯度/更新增量方差、a=1 端点噪声、格式漂移、联合/独立控制的实际参数偏差、代理与独立确认集回报增益的对应。不用参与控制的同一批反馈证明控制准确。

PLANNED 判读规则：

- D3 只胜 D1、不胜 D2：支持重加权，尚不支持来源选择。
- 只胜高剂量不稳定臂：支持稳定性；进一步匹配剂量/η后判断。
- 与通用回报修正等价或表现相同：不宣称特殊知识传输机制。
- full K 无增量：如实简化；不预设交叉项必须有优势。
- 两条学生梯度相同但教师可学：记录剂量效果；不人为制造“选择性成功”。
- CI 宽、有效父组少、结果矛盾：INCONCLUSIVE，不当普遍证伪/确认。
- 确有失败机制：登记少数统一目标内的估计器、步长/信任域或反馈频率变体，附计算成本，不按错误类别切 loss。

## P2：完整预算流程与采购（PLANNED）

固定蒸馏器后，比较随机购买、正确记账的便宜优先、原 Bayesian scalar-value、购前教学响应预测+同目标价值，以及购后真实响应 oracle（仅诊断上界）。先离线 sealed replay；prospective 只在另行已有明确授权的 API 支出内执行。本次配置不授权新 API。AppWorld 在 BFCL/ALFWorld 核心验证后追加。

统一采购定义 A(Q)=E[F*(D∪Evidence(Q))−F*(D)|购前信息]，所有比较共享 θ、h、参考、总更新计划、旧曝光权重。候选槽及免费 retention 基线预先冻结；购买只开放 teacher 方向；新增版本系数可零且 Σ_v a_ijv≤1。任何改变参考的变体必须保留完整位移与基础增益，不能把 teacher gain 在重新中心化中丢掉。

PLANNED 预测器：冻结学生表示+小网络/集成或核模型，只看购前 state/history、学生行为/不确定性、请求上限、当前 θ/P/进度/预算/已购集合摘要。输出方向—回报对齐、方向尺度、已有方向关系、成本及不确定性的联合低维统计。Gram 从共享预测因子构造，不把独立预测元素默认为 PSD。输出真实长度、teacher text/gradient、teacher checker 结果不可提前进入输入。

保存购前特征再揭示付费响应；标签带 θ、反馈、曝光、参考版本。旧证据重算不增加 teacher bill，但计入算力；标签陈旧单独记录。预测验证按父任务分组，保留随机采购数据或选择概率。批量采购联合求解或逐次重定价，不永久按一个 value/token 标量排序。可研究同目标边际/对偶筛选，再对小候选集精确重解。

PLANNED 预算 manifest 至少记录 benchmark、teacher_id、student_init_hash、pool/split版本、cost_regime、整数 cap、实际支出、exact/estimated、估算方法、全部失败/重试、生成池和历史成本、剩余额度。BFCL 55,370 内容分母、13,843 cap、另一池 16,748 采集成本和百万级历史输出严格分开。封存回放只能声称模拟访问协议下效率；prospective 含全部计费尝试。warm-start 记录累计证据，不当作未训练 base。环境、学生 token 和 GPU 单列。

PLANNED 报告：任务表现 vs 实际 teacher 支出曲线、达到相同表现的成本、预测误差/decision regret、重复证据边际值、代理与实际学习收益、总训练曝光。oracle 有效但 predictor 无效则改 acquisition estimator；oracle 也无优势则先查价值目标、候选池和蒸馏器，不靠加深网络掩盖。

## P3：计算近似与部署成本（PLANNED）

先测时间和峰值显存，再决定加速位置。分项为学生采样、teacher/replay、source/teacher/soft 梯度、sketch/metric/Gram、反馈完整 rollout、QP、控制预测、正式 commit、保存恢复/调度。20 状态×2≈40 个变量不能先认定 QP 是主瓶颈。

PLANNED 三方式：exact；amortized-only；amortized 初始化+预先冻结迭代数/容差的修正。真实问题修正与预测问题修正分别标注。轻量共享 head+来源等变集合汇聚，输入已拥有的状态/来源/教师和批次/反馈摘要，不训练第二个大型 LLM 或全局 capability atoms。

PLANNED 标签损失用同一 `TeachingObjective.regret`、实际更新偏差、预测收益与可行性；不只回归 a 的 MSE，因为重复方向系数不唯一。反馈/几何刷新窗口预登记，记录年龄和漂移；减少反馈改变信息质量，不能叫纯实现等价优化。

PLANNED 报告两种预算视角：相同骨干训练量下的任务表现与总耗时；相同总 GPU/环境预算下的任务表现。还需反馈 rollout 数、F regret、参数增量误差、控制器训练/标签/校正成本、摊销回本点、不同学生阶段的稳定性。仅 QP 变快而总时间不变不算主要加速；性能换速度则报告 Pareto 曲线。

## 共同评测与最终判断（PLANNED）

- BFCL 官方 Overall，先验证全量生成完整性；temperature .001 与严格 greedy 0 分开；不同硬件 base 不是训练 seeds。
- ALFWorld valid_seen 用开发；反复使用的 valid_unseen 不能重新称 untouched test。低初始能力/已有能力学生可互补验证，但同数学程序。
- AppWorld 官方完整任务 TGC；pooled unit-test pass ratio 仅诊断。dev40/max24 与历史 dev57 不直接合并。
- 独立确认集不参与特征归一化、采购、控制器训练、调参或 checkpoint 选择。P1 preparation 按本次用户协议单列 calibration split，仅用于预登记的 D0 SFT/KL 权重选择；该 split 是开发用途，不能再称 untouched confirmation。训练反馈、calibration 与独立确认按父任务分开。
- P1–P3 最终可能支持选择性、剂量、稳定性、采购或计算中的某一部分；按证据缩小主张。P0 正确性不是性能有效或创新性证据。

## 复现入口

当前可执行：

```bash
export BFCL_PROJECT_ROOT=/tmp/rtd-unified-p0-bfcl
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PYTHONPATH=src:. /home/xueqi/hq/projects/tc-alignment/.venv/bin/python -m bfas.rtd.unified --config configs/rtd/unified_bfcl_gemma4.yaml
PYTHONPATH=src:. /home/xueqi/hq/projects/tc-alignment/.venv/bin/python -m pytest -q tests
```

P1 的 run/smoke/resume/evaluate 启动路径、D0 baseline entry、每个 arm 的命令及 matched 协议见 `RTD_UNIFIED_P1_PREP_ZH.md`；生产 GPU 运行尚未执行。下一项是先在已授权单卡上执行 ALFWorld D3 一窗口 smoke，查看阶段成本，再运行同合法固定池的小预算 D0–D3。P2/P3 仍为 PLANNED。
