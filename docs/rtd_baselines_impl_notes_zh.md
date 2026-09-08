# C27 stage 1：隔离强基线实现

本阶段只新增文件。实现前完整阅读 `rtd_strong_baselines_plan_zh.md`，并核对 RTD v1 规范 §10.1、§10.3。没有读取活动 `results/rtd_v1/*` ledger，没有连接 hpg、启动生产训练、申请教师、启动 BFCL campaign，或改写现有 RTD／adapter／配置文件。测试从 `tests/fixtures/rtd_baselines/fixture.json` 构造真实 C25 格式的合成 bank、ledger、compute、trajectory 和 checkpoint，全部运行产物写入 pytest 的 `/tmp` 目录。

## 已实现的入口与训练定义

新包 `src/bfas/rtd/baselines/` 和 `tools/rtd_baselines.py` 提供独立 `prepare / tune / run`。旧 `tools/rtd_experiment.py run` 不接受这些基线配置；不把 B3/B4 伪装成 R0/R1。

| 配方 | 实际执行 |
| --- | --- |
| B1 | 如实标记 `B1-replay-L1`。合法历史完整状态、每轮冻结学生随机采两条 source；监督只算完整教师动作 CE。强配方 pT，无 success filter、source identity loss、额外 demo 或 anchor。 |
| B2 | 同完整 state **及 parent** 的教师 chosen／冻结学生 rejected；空文本 EOS rejected 保留。调用 `arms.pair_unit_margins`（`pair_unit.py` 也复用该函数），再取 logistic loss。reference 始终 theta0；不用 CC 双条件 squared hinge。AdamW、weight decay=.01、beta=.1，无 ref-free、span-only、额外权重或 anchors；实现不读取遗留 AW 环境变量。 |
| B3-SFT / B3-mix | 分别 pT 教师 CE／p0 固定 .5 `positive_mixture_loss`。反馈步在当前 theta 完成监督与所有配额组的新 on-policy rollout，调用公开 `reinforce_gradient`，以 `g_sup - stop(gJ)` 一次提交，方向为回报上升。两种目标均有独立配置和网格。 |
| B4 | 每个已购 query 一个 scalar，softplus 后按整个合法 inner 池的 p 质量归一化到 mean-one；query 只作索引。只有 weighted teacher CE，无 gate feature、source loss、gT-gS 或 source-quality transport。训练侧 source RMS P 和 .5／.005 KL pilot 冻结每轮实际一步。先得到 theta+，再在该点取得真实回报；teacher-only streaming reduction 调公开 `gate_vjp`，不调用 `streamed_gate_vjp`。控制器使用过去反馈 RMS／ridge；反馈折坐标不更新。 |

生产模型通过现有 `runtime.load_backend` 加载本地 BF16 base、FP32 LoRA 坐标和相同 score/generation 实现。新 manifest 校验完整 effective runtime、base、tokenizer、data、硬件类和已审计的 evaluation harness；不会用当前 YAML 默认值填补旧来源配置。若旧配置不满足本阶段声明的 BFCL 512/1024、32768 等设置，须另定并完整重跑可比较协议，不能无声覆盖。

## 曝光与资源账本

`pool.py` 实现 parent → 去重 full state → 两条 source → 已付费教师经验项的 p0，以及 pT 条件化，公开 teacher-available 总质量、parent mass、query mass。包内 `(state,text)` alias 不增质量；不同付费 query 的相同回答保留经验重数。教师费只在 query 层计一次。

V-S 为每轮 96、合计 288 完整监督槽；36×8 或 B1/B2 调优的 72×4，反馈窗口映射到相同 0/24/48/72 槽边界。日志区分 nominal weighted slots、raw slots、实际教师／source 分支和 importance-weight sum。`p0_attribution` 可以运行无教师 CE=0、仍保留分母的分布对照；它本身**不提供外部固定两侧文本／顺序／数值 P 的严格归因证书**。

V-T 在代码 docstring、每轮 `exposure-N.json` 和哈希链 journal 中写出原式：

`Z = sum_j p_j/c_j;  q_i = (p_i/c_i)/Z;  rho_i = p_i/q_i = Z*c_i`

每个 draw 使用 `rho_i * L_i`，按**预定 draw 数**平均；B4 再乘 w。完整 logprob 始终求和，不除长度，不裁 rho，不做 minibatch self-normalization。c 使用相同 tokenizer 的完整 prompt+action 输入长度，同时保存 action-only 计数；B2/mix 包含两侧，B1/B4 只有教师侧。c/q/rho 和全部 draw 在每轮训练前冻结；不截动作凑整数、不按实际 token 提前停止、不重抽低成本样本。

T_update 以 `pair_unit` 的一次 forward+backward 完整 input exposure 口径计数，另分别给 forward_tokens/backward_tokens，避免把同一输入混成双倍 token。B3 的 feedback score/backprop 进入 T_update，先按参照实际反馈长度预留；B4 外层 gJ/VJP、source P、pilot、reference、generation 各自计账。B3 同配额反馈若耗尽目标，明确报 infeasible，不减 rollout。结束时核验每轮误差不超过一个完整监督微批，以及最终总目标误差 ≤1%；不满足写 `matched=false`。V-T 同 token 不宣称同 FLOPs 或 GPU 时间。

`ResourceJournal` 继承 `ComputeJournal`，保留 hash chain、嵌套 scope、失败工作、峰值显存。GPU 时间只汇总最外层完成 scope，CPU／reserved 秒独立列出。`audit.json.resources` 汇总教师账单、新教师 token=0、学生 token、提交／曝光、反馈与调优 rollout、T_update、T_aux 和 GPU 秒。多 trial 计每次学生计算；调优总表的教师实际证据成本只计一次，同时另存跨 trial 分摊总额。

## prepare 的完成门与格式

来源必须同时有 manifest、teacher.jsonl、trajectory、三份 round checkpoint、最终 audit 和 compute.jsonl。纯只读解析 teacher／compute hash chain，验证 reservation/reveal/dependency 转移，拒绝 torn 尾行，不调用会恢复写入的 `Ledger.resume`。严格要求三轮 36 committed steps、12 windows、fold 0/1/0、最终审计、无 reservation/pending/unmerged、owned/spend/policy 绑定及 checkpoint artifact hashes。

反馈表从成功 compute scope 的 `feedback_rollout` 和 `return_gradient`、trajectory 的 `truncation.actual_feedback_reused`、`feedback_reused` 提取；没有 parent/count 或组数不完整会拒绝。重试的失败工作保留在来源计算统计，已提交配额只选最后成功组。baseline 的 reference/actual 标签只标来源资源组，另记录真实 sampling parameter hash；每个实际组独立 LOO 后按 rollout 数平均，reused actual 不重复采样／计费。调用 `BFCLSupport.feedback` 原公开路径，自任务初态获得原 `bfcl_task_rollout` 的终局反馈，旧回答／reward 不作为 baseline 训练输入。

`prepare` 仅解封 reveal 已拥有的包，核对成本／usage／dependency／integrity；校验 bank/support/official data 与来源 data hash 一致，导出 source hashes、ledger event digest/尾 sequence、原 identity、支持父任务、合法状态／教师 token 目录、同池账单和预算。调用前后再检查来源文件哈希，不对增量活动运行“取并集修复”。V-T 教师长度在 prepare 计算；baseline 自己的 source 文本和长度只能在各轮冻结采样后计算。

配置中的 A0/A1 路径和硬件 hash **没有填造**，`bfcl_suite.yaml` 是会拒绝执行的待绑定模板。生成的 `${pool}_${recipe}_${view}.yaml` 自包含，实际读取 suite 指定 recipe 配置并绑定证据 hash，不隐式继承旧 RTD YAML。调用约定：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. .venv/bin/python tools/rtd_baselines.py prepare \
  --config configs/rtd/baselines/bfcl_suite.yaml --out configs/rtd/baselines/generated
PYTHONPATH=src:. .venv/bin/python tools/rtd_baselines.py tune \
  --config configs/rtd/baselines/bfcl_tuning.yaml --prepared configs/rtd/baselines/generated \
  --out results/rtd_baselines/tuning
PYTHONPATH=src:. .venv/bin/python tools/rtd_baselines.py run \
  --config configs/rtd/baselines/generated/A1_b3_sft_slots.yaml \
  --selection results/rtd_baselines/tuning/selection.json --run-dir results/rtd_baselines/A1_b3_sft_slots_s0
```

这些是后续执行契约，本阶段没有执行。输出目录必须全新，禁止重用或覆盖源 run、bank、data、logs、results/rtd_v1。默认调优配置完整执行 B1 六个、B2 六个 trial；`bfcl_tuning_all.yaml` 额外支持计划全部 24 trial。全部 seed=0、每 trial 从 theta0 起步。独立 development RNG=20000，固定使用来源每轮首窗 reference 的 parent 配额表作为额外轮末支持反馈，先逐任务平均再平均三轮。并列依次选少步、低 lr／eta multiplier、低 meta_lr，并显式记录不可辨识；不访问官方认证集来选参。

## 官方 checkpoint 兼容性与后续集成清单

新 runner 保存真实 `round-N/lora/{adapter_config.json,adapter_model.safetensors,...}`、tokenizer、`round_state.pt`、checkpoint.json、manifest、trajectory、teacher/compute journal。round_state 包含真实参数、theta0、轮初 source/cache、P/eta、optimizer/controller、u、RNG 和 reference cache；72 步如实写成每轮 24 个提交。测试调用原 `identity.verified_checkpoint` 验证绑定。生产导出用已加载 PEFT 模型的 `save_pretrained`，可供原 `evaluation._flatten_adapter` 加载；CPU 合成 adapter 测试验证目录及哈希，不声称已做 4B 合并或官方 BFCL 认证。

已实现 B1–B4 的 prepare/tune/run 和原官方 evaluate **无需修改任何现有文件**。后续正式终点评测可直接：

```bash
PYTHONPATH=src:. .venv/bin/python tools/rtd_experiment.py evaluate \
  --run-dir results/rtd_baselines/A1_b3_sft_slots_s0 --round 3
```

剩余集成项及精确文件范围如下；均未在本阶段执行：

| 项目 | 是否需要改现有文件／原因 |
| --- | --- |
| 最终 A0/A1 binding、matched hardware、共同可行 V-T 预算、正式调优／训练／认证 | 不需改现有源码；等待完成来源后只生成新配置和新结果。参照反馈本身超过低 T_update 时，须为所有臂声明共同可行预算并重跑参照。 |
| D0/D1 的外部同反馈表、固定两侧文本／顺序／P 的严格归因、完整四格 suite | 尚未实现；可继续在新 `baselines/` 文件中做调度，保持 RTD 公式。若决定复用 `RTDExperiment` 的状态机直接注入，才需在 `src/bfas/rtd/experiment.py` 的 `choose_feedback_tasks`／`draw_slots` 接口接纳冻结外部表，并在 `src/bfas/rtd/cli.py:load_config` 显式验证新键；本阶段不要求或执行该方案。不能把当前默认 fixed-evidence 运行当同反馈 D1。 |
| full-bank B1 正式采购→导出→训练协调器 | 本阶段提供 `acquisition.py` 的 sealed `random_control=True` 整包窗口和公开历史成本 uniform F 分配原语、两份独立采购 schema/config，并测试预算与概率；尚无 full-bank 生成训练配置的 CLI 协调器。可只新增文件，无需编辑现有采购器。采购配置不是可直接 `run` 的已准备训练配置。 |
| 任意新学生 occupancy 的在线 B1 | 本阶段只支持诚实 replay-L1；需要真实 stateful teacher 请求路径、完整依赖和明确预算。可新增 backend；现有 `new_teacher_calls=false` 不改，不伪造 bridge。 |
| 原 `tools/rtd_experiment.py report` 汇总基线 | 原 `src/bfas/rtd/evaluation.py:report` 依赖 RTD 专用 old/new exposure 字段，不直接支持新 audit。当前使用新资源 audit；若要求旧 report 统一汇总，需给该函数新增 baseline manifest 分支，或在新包新增报告入口（无需改现有文件的推荐实现）。评分／evaluate 无需改。 |
| 长任务恢复／HPG 作业 | round_state 保留恢复所需核心训练状态，但当前 CLI 不实现中断 resume；不得伪称自动恢复。完整恢复协调器和 `scripts/rtd_baseline_hpg.slurm` 可新增，不把 B3 传给旧 RTD_ARM 脚本。 |

没有强基线性能结论。本阶段验收仅是独立实现、合成 CPU 正确性和格式兼容；最终 evidence、同机正式认证、D0/D1、full-bank 正式端点及严格在线覆盖仍是后续执行工作。

## 验证

使用 `PYTHONDONTWRITEBYTECODE=1` 防止导入已有模块时写入字节码；测试命令为：

```bash
PYTHONPATH=src:. .venv/bin/python -m pytest tests/test_rtd_baselines_*.py -q -p no:cacheprovider
```

覆盖预算／曝光算术、q/rho 分布与完整动作期望、36/72 步分配、只读 prepare 成功与拒绝、B2 reference/空动作、B3 回报上升和相同 RTD 父任务／组数／LOO、B4 teacher-only 有限差分 VJP／禁用 transport、五个真实 tiny 训练循环、V-T 日志、官方 checkpoint hash gate、调优网格／独立支持评估／tie-break，以及非嵌套资源汇总。

最终实测：**55 passed in 73.51s**，pytest 退出码 0；`prepare/run --help` 的新入口也已检查。与任务开始时的受保护已跟踪文件 SHA256 快照比较，没有变化。工作期间其他并行任务提交了 C26 文件并更新 `PROJECT_STATE.md`；这些既有／并发改动保留，本阶段没有编辑它们。
