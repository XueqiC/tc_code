# C26-F：停跑后 ALFWorld RTD 分派集成计划

依据：`docs/rtd_alfworld_readiness_zh.md` §7.1–§7.4。C26-E 只新增文件，未接通训练。准备入口的 `run` / `resume` 返回退出码 2，并打印 `not yet integrated: run/resume require the C26-F edits`。本计划不授权在活跃工作树上修改源码或运行 GPU smoke。

## 1. C26-E 已冻结的接缝

`benchmarks/registry.py:get_benchmark(config)` 按 `benchmark` 返回不可变 providers；未指定时保持 BFCL。BFCL 十个 provider 都是原实现的直接 import 对象。ALFWorld 使用新模块或薄适配器；通用 `SealedReplayBroker` 继续复用。合同测试用 `inspect.signature` 读取 BFCL 实际函数，逐一比较全部参数名、位置/关键字种类、默认值和注解，没有设置伪造的 `__signature__`。

| provider | BFCL 原对象 | ALFWorld 对象 / 参数转换 |
|---|---|---|
| `bank_builder` | `bank.build_bfcl_bank(root, directory, *, entries=None, renderer=render_prompt)` | `alfworld_bank_builder` → C26-A `build_alfworld_bank`；显式拒绝 BFCL entries/renderer。只生成 inventory；不可当 C26-B verified bank |
| `broker_builder` | `broker.SealedReplayBroker(directory, ledger, *, inner_parent_hashes)` | 同一个通用 broker；verified bank 已保存完整 behaviors |
| `support_protocol` | `experiment.BFCLSupport(root, config)` | `ALFWorldExperimentSupport(root, config)`；提供 parents/states/categories/entries/unavailable 和相同签名的 feedback |
| `action_limit` | `TorchPolicyBackend.action_limit(self, category)` | `alfworld_action_limit(self, category)`；只接受 agent_action=256，异常后恢复旧 cap |
| `feedback_rollout` | `bfcl_task_rollout(entry, category, truth, backend, parameters, generator, *, checker=None)` | `alfworld_feedback_rollout`；entry 可为 reset FullState；truth 必须为空；checker 是显式 `ALFWorldFeedbackContext` |
| `official_evaluation` | `evaluation.evaluate(root, directory, round_number, *, port=None, base_evaluation=None, lock_timeout=None, lock_log_interval=None)` | `alfworld_evaluate`；经过共同硬件/harness/source guard，再调用 C26-D make_manifest/evaluate |
| `harness_identity` | `identity.evaluation_harness_identity(root, config)` | `alfworld_harness_identity` → C26-D，解析本地模型、环境和 valid_seen 数据目录 |
| `cap_policy` | `caps.public_cap(request_class, *, limits=None, evidence=())` | `alfworld_cap_policy`；limits 仅允许 ALFWorld `CapConfiguration` |
| `affordability` | `caps.affordability(records, payloads=None)` | C26-A/B 的 `alfworld_caps.affordability` |
| `scoring_projection` | `scoring_scope.scoring_projection(name, content)` | C26-D `alfworld_identity.scoring_projection` |

`ALFWorldExperimentSupport.feedback_context(round_number, backend, journal)` 提供已有 tokenizer 的 renderer、RealStepper factory、训练 support 和当前轮次；创建 context 不启动环境。feedback 返回通用 `TaskRollout`，现有实验的 `reinforce_gradient` 保留 LOO 和相同的 checked_score_action。环境错误由 `as_task_rollout` 抛出，不能变成 reward=0。

每次 rollout 从实验保存的 torch generator 抽一个 63-bit episode index，交给 C26-C 的 `episode_seed`，派生 seed 写入 `alfworld_episode`。不添加未保存的计数器。恢复未完成 phase 时重置为该 phase 保存的 RNG；已完成 phase 不重新采样。source RNG、numpy task/slot RNG 的保存路径继续使用原 StateStore。

`alfworld_config.manifest_section(root, config)` 是 CPU section，不是完整 run manifest：提供 config/hash、base/tokenizer/harness、sealed bank/support/140-task valid_seen data_hash、bank audit、可用 public-cap 预算、估计和缺失费用明细、score consistency 和 resources。完整 run 的 arm、smoke、hardware/source、initial_parameter_hash、StateStore 绑定必须由原 CLI 补齐。它逐次检查 sealed bytes、外部 support 一致性、当前训练 world、C26-B 环境/tokenizer。数据/环境在 checkout 内使用冻结相对路径；可通过隔离目录下的只读链接访问同一资产。

生产 selector 目前仍只有 broker/bank/ledger 三个名称的 import denylist。C26-E 测试用 test-only guard 执行新增 denylist 的规范；它不是生产补丁。新增 registry 自身在 selector context 下拒绝 provider resolution。不能把测试通过写成“现有 selector 已禁止所有 ALF import”。

## 2. 十个原有文件的精确修改范围

以下均是待审查 diff，C26-E 没有修改这些文件。避免递归分派：BFCL registry 指向原函数，原函数只在 `benchmark == alfworld` 时调用 ALF provider；BFCL 分支继续执行其原函数体，不能反调自己的 registry entry。

### 1）`src/bfas/rtd/cli.py`

- `load_config(path)`：先读 benchmark；ALFWorld 立即交 `alfworld_config.load_config`；BFCL 原有 canonical/mutable 检查、defaults、旧配置恢复不变。科学版本继续 `protocol_version: 1.0.1`；ALF 增加单独 `benchmark_schema_version`。
- `data_identity(root, config, bank)`：ALF 分派到新模块同签名函数；BFCL 仍 hash BFCL data。
- `bank_audit(config, *, build=False)`：ALF 调 `alfworld_config.bank_audit(ROOT, config)`，打印 ALF affordability 字段；禁止把 `--build-bank` 的 C26-A inventory 当 verified bank 自动使用。新 B bank 只能走显式环境验证/封包命令。BFCL build 行为不变。
- `make_manifest(config, arm, audit, *, smoke=False)`：ALF 用 `manifest_section` 构造科学 section，再由现有代码补 `version/arm/smoke/hardware/hardware_hash/rtd_source/backend/backend_rationale`；不放入 BFCL 精确历史用量字段。不要让 BFCL 样式的 resources 覆盖 ALF estimated/unknown 明细。fixed-source 检查仍在统一出口执行。
- `run_command`：`providers = get_benchmark(config)`；`support = providers.support_protocol(ROOT, config)`；backend 继续 `runtime.load_backend`；实验仍只有 `RTDExperiment(...)`。support_access 标签按 benchmark 写出，ALF 不启动 CheckerBridge。BFCL 的 `_checker_context` 保留；ALF run 外层用 nullcontext，实际每个 feedback window 显式构造支持 context。
- smoke override：ALF `parents_per_fold=4, slots=8, rollouts=2, windows=1, baseline=leave_one_out_same_task`，独立标记 partial smoke。不能继承 BFCL smoke 的 K=1/zero baseline。正式 YAML 始终 3×12。
- `run_campaign` / `main evaluate` / `main report`：调用统一 evaluation dispatcher；子进程仍使用已有训练 worker 参数、GPU UUID、原环境、through_round、锁和 resume。原目录/manifest 不可覆写。新准备入口在 C26-F 后只能委托此入口，不能另建 loop。
- `resume_config`：保留“saved config 完整 hash + 显式 supplied config 相等”规则。ALF defaults 不能渗入 saved config。组件 swap 仍只能改变一个组件；C26-E scalar YAML 与主 YAML 只有 gate 不同。

### 2）`src/bfas/rtd/experiment.py`

- `RTDExperiment.__init__`：保存 `get_benchmark(config)`；broker 构造换 `providers.broker_builder`。不改变 ledger、预算授权、save 时点、恢复指针、optimizer 参数和 phase 编号。
- `sample_state`：ALF `providers.action_limit(backend, support.categories[tid])`；BFCL 使用原 backend action_limit。source 样本数取配置（冻结为 2），temperature/top_p 均为 1。source cache、score journal-before-enforce、hidden projection 全部复用。
- `pool`：ALF 对 reset/teacher states 分派到 `support.protocol.guard_states`，传当前 round 和已拥有 packages 的完整映射。guard 必须在 source、P、standardizer 或 chi 使用之前；pending/revealed 新包仅在 broker 已完成 acquisition/ownership 后通过，不能给未购买前缀免费 normalizer/source 访问。跨折依赖、probe/calibration 必须拒绝。
- `choose_feedback_tasks`：ALF 对全部合法 feedback parents 均匀、无放回抽 `meta_tasks_per_feedback=4`；固定 selected trial；每题 `rollouts_per_meta_task=2`；结果仍保存为 `(parent, count)`。不能用 usable teacher packages 筛掉失败/no-payload parents。BFCL 单轮/多轮组保留，只把 8/4、4/2 读成原 config 对应键和相同默认。
- `feedback`：ALF 每个调用先 `context = support.feedback_context(s['round'], backend, journal)`，传给现有 `support.feedback(parent, backend, parameters, sampling_rng, context)`；reference 和 actual 都走此函数。返回 `TaskRollout` 后仍走原 `reinforce_gradient`。ALF smoke 也用 LOO；M/K、identifiable、全零、malformed、truncation、token/episode 诊断均持久化。
- `slots` / `round_start` / `feedback` 的 smoke 分支：仅把 benchmark-specific smoke override 读出来；ALF 每折四个按 hash 固定的 reset parents、八槽、K=2。恢复仍用保存的集合/抽样，不能重选。`assert_run_invariants` 保持 smoke 一步、正式 36 步/12窗口的检查。
- `round_start`、`step_start`、`reference`、`selected`、`revealed`、`actual`、`feedback_commit`、`committed`、`round_end`、`run` 的数学及转换顺序不改。pilot 继续已拥有内折 evidence 和 source；eta 未选中时保留 initial_eta；actual-only commit、虚拟同起点 reference、epsilon=.25、reference 复用均不改。

### 3）`src/bfas/rtd/return_gradient.py`

- `TorchPolicyBackend.action_limit(self, category)`：注册的 `agent_action` 分派到 ALF provider；BFCL single_turn/multi_turn 分支保留原行为和签名。不能让所有非 multi_turn 都默认为 single_turn。
- `TorchPolicyBackend.__init__`：校验所用 action class/caps；backend_id 继续绑定 effective caps。避免无意迁移 BFCL backend_id。未注册类别明确拒绝。
- `bfcl_task_rollout`、`ActionTrace`、`TaskRollout`、`reinforce_gradient`、GateController/VJP 数学不动。ALF adapter 已返回通用 TaskRollout，不把 parsed command 重分词作为 logprob，不把 observations 加入 action loss。

### 4）`src/bfas/rtd/runtime.py`

- `load_backend`：按 registry/config 选择 action caps；ALF `max_action_tokens=256`，`action_caps={'agent_action':256}`；BFCL 原 512/1024 和 override 行为保留。传入 backend 的 score tolerance / memory policy 不分叉。
- `HFGenerateBackend` 的 action-limit policy 使用上面统一接口。`sample_action`、teacher-forced CE、KV categorical、logits/score dtype、checkpointing、streamed_gradient/VJP、单 action graph 顺序不动；不得创建第二个 resident model 或另一个采样/scoring 后端。

### 5）`src/bfas/rtd/evaluation.py`

- `evaluate` 入口从 training manifest 读取 benchmark；ALF return `providers.official_evaluation(...)`，BFCL 继续原函数体。ALF wrapper 保留 common `guard_hardware` / `guard_harness` / `record_code_drift`，然后 C26-D `make_manifest(..., run_directory=..., round_number=...)` 验证 checkpoint。
- ALF campaign 位于训练目录的独立 sibling `<run-name>-alfworld-evaluations/round-N`；完成后 `evaluation-N.json` 写原生 ALF 结果，不伪造 overall_accuracy_percent。端口参数对 TextWorld pipe backend 没有效果；显式记录这个事实。
- `report(directories, output)`：按 manifest benchmark 分派结果验证/metric extraction。ALF 调 C26-D `validate_evaluation` / `campaign_identity` / `checked_expectations`，核对 round_checkpoint、140题、artifact hashes、base class，再输出 success_rate (0–1)、success_percent (0–100)、各类 success、repair/damage。现有 BFCL `identity.checkpoint`/resultdir/scoredir 字段不能套给 ALF。
- ledger/checkpoint owned-set/spend 和外层 compute 统计沿用现有报告骨架；额外列出 ALF estimated cost、missing responses/attempts、valid_seen exploratory、identifiable feedback、cap/fallback/ACTION、episode steps。正式比较包含 R0/R1、历史强 CE（仅标注历史）、同合法 owned-set 强 CE，不用 success 提升作实现门槛。

### 6）`src/bfas/rtd/evaluation_lock.py`

- `tag_lock_path(root, tag)`：增加 keyword-only `benchmark='bfcl'`；默认 BFCL 锁路径和 tag bytes 完全保持；ALF 对应 C26-D 的 tag namespace。统一检查 tag 不允许路径穿越。
- `evaluation_lock` / `wait_settings`：按 benchmark/tag 日志、timeout/log_interval 接通 ALF wrapper 的同名参数；C26-D 目前只接 lock_timeout，C26-F 要把 log interval 传至同一 wait policy，不能 silently pretend it took effect。
- `reserve_port` / `port_lock_path`：继续以物理 port 全局排他；benchmark 只能影响 campaign/tag 名称，不能给同一个 port 建不同互不排他的锁。ALF TextWorld 无 HTTP server，不分配无用端口。

### 7）`src/bfas/rtd/identity.py`

- `evaluation_harness_identity(root, config)`：ALF 调 registry provider；BFCL 原内容投影和 metadata 不变。
- `saved_identities` / `guard_harness`：ALF 先验证原 manifest config/hash/harness，再检查当前 ALF provider；首轮 F 集成严格拒绝任何漂移。不能查用 `legacy_identities` 中的 BFCL/C25 legacy audit 为 ALF 开口子。后续显式 continuity 更新扩展必须用 `alfworld_identity.audited_harness_hashes` 的 FULL manifest-bound 链，返回兼容 common guard 的 `evaluation_harness/harness_hash/rtd_source` 字段；campaign supplement 与 training manifest 的 binding 不可混用。
- `audited_harness_hashes`：ALF supplement 分派到 ALF 链验证；无 supplement 时只接受当前原 manifest hash。绝不接受另一 run 的历史 hash 集合。
- `validate_resume`：仍逐字段比较科学 config/data/base 和 tolerance，继续 hardware guard、source drift journal、显式 acknowledge 规则；不重写旧 manifest。`record_code_drift`、`verified_checkpoint` 保持通用校验和调用顺序。
- `audit_legacy`：明确只允许 BFCL；ALF 不进入 BFCL 旧数据/权重推断路径。`source_identity` 的扫描范围不缩小；因此所有 edits/新文件必须在隔离 checkout 上完成并一次性冻结后启动新 run。

### 8）`src/bfas/rtd/scoring_scope.py`

- `scoring_projection(name, content)` 原函数及签名保留；新增 `benchmark_scoring_projection(name, content, *, benchmark="bfcl")`，内部调 `get_benchmark({"benchmark": benchmark}).scoring_projection(name, content)`。`scoring_hash(name, content, *, benchmark="bfcl")` 使用这个 dispatcher；默认 BFCL 原 selector/evidence AST 及 hash 规则保持。
- C26-D `SCOPES` 独立维护；mixed operational/scientific edits 必须有 AST scope 测试，缺定义/重复定义直接失败。registry dispatch 自身至少纳入 run source identity；与科学评分有关的调用点进入 ALF projection。
- 原 projection provider 的两个参数保持不变；common identity/report 中需要 projection 的调用明确传 benchmark 给新增 dispatcher。不能靠 `__signature__` 或 `**kwargs` 掩盖差异。

### 9）`src/bfas/rtd/caps.py`

- `public_cap` / `affordability` 保留原签名、默认常量、输出结构和数值。新增 `affordability_for(config, records, payloads=None)` 与 `cap_policy_for(config)`，分别返回 registry affordability 结果和 cap provider；common CLI/report 的 class policy 入口显式使用这两个函数，避免给旧函数增加隐式全局 benchmark 状态。
- 新 ALF 路径的 `cap_audit` 始终调用 C26-A/B 实现，cap 来自公开配置；usable denominator 从 verified records 得出；unavailable 112次不进入预算分母，历史 gap 的 cost 保持 unknown。
- CLI 不再用 BFCL `PUBLIC_CLASS_CAPS` 对 ALF 校验，也不要求 ALF affordability 存在 demo/item capacity 字段。hidden text、won、实际 usage、tokenizer recount 不得影响公共 cap。

### 10）`src/bfas/rtd/selector.py`

- `_check_import(name, fromlist=())`：原 broker/bank/ledger 集合外，拒绝 `alfworld_bank`、`alfworld_state`、`alfworld_support`、`alfworld_rollout`、`alfworld_evaluation`、`alfworld_identity`、`alfworld_config` 和 `registry`（后两者提供间接 privileged 能力）。必要时对原始 ALF adapter 做相同 deny。
- 保留 `public_only` 对 `__import__` 和 `importlib.import_module` 双入口的检查，以及 `_audit` 的 open/listdir/subprocess/socket 拒绝。测试包含 cached/uncached、fromlist、relative 四种路径；不能只测首次 import。
- 把 C26-E test-only guard 测试切换到真实原生 guard，去掉 fixture patch；保留 registry 自身的防护和 selector 静态无 privileged imports 检查。明确仍是 cooperative guard，不声称 OS sandbox。

## 3. rai smoke 的执行顺序与资源

所有命令在已应用 C26-F 的独立 checkout 中执行；新 run 输出目录必须为空且不在任何活跃 rai/hpg run 下。不要访问 BFCL C25 bank 或旧日志。C26-E 可立即执行的只有：

```bash
PYTHONPATH=src:. .venv/bin/python tools/rtd_alfworld_experiment.py audit
PYTHONPATH=src:. .venv/bin/python tools/rtd_alfworld_experiment.py smoke-plan
```

smoke-plan 从 verified bank 输出两条真实 train query/task IDs、各自完整 demo 命令数、support hash、manifest hash，以及固定每折四个父组。它不会生成模型、初始化环境、创建 run 或探测 GPU。source 深状态数由实际采购决定，计划明确给出公式，不能虚报一个事先固定的总 token 量。

1. CPU replay：对计划里的两个 train 游戏（一折一个），分别 fresh reset 两次、逐条执行完整已留存成功命令，逐 state 比较 task/world/environment/hash、完整 history、goal、admissible、done/won。只读 sealed payload；不重封 bank。每题另起 reset，把固定无效 raw action 送入现有 parser，再执行固定 look 路径到 terminal 或 40 步；保留 fallback/失败结果，不能因失败删样。给 RealStepper 30秒单操作 / 120秒 episode deadline，任何基础设施异常停止验收并报告 reason。
2. CPU 测试通过且 rai 确有一张空闲 GPU 后，绑定一个 UUID；只启动一个 Qwen/Qwen3.5-4B、本地 LoRA rank16/alpha32，BF16 base + FP32 trainables。内存声明 peak38GB、reserve2GB、state estimate16GB、max_state_batch_size1；source temp1/top_p1，action256、context32768。
3. 使用修改后的原入口（C26-E 当前不能执行以下命令）：

```bash
# 仅 C26-F 后；调用前由操作者设置 CUDA_VISIBLE_DEVICES 为 rai 的空闲 GPU UUID。
PYTHONPATH=src:. .venv/bin/python tools/rtd_experiment.py smoke \
  --config configs/rtd/v1_alfworld_c26.yaml --arm R1 \
  --run-dir /tmp/rtd-alfworld-c26f-rai-window
```

4. 只执行 round1/step1 的 reference→selected→revealed→actual→feedback→committed→round_end；每折四父组、8槽、M4/K2。最多两条 feedback 分支 × 8 episodes =16 episodes，40 actions/episode，最多163,840 feedback action tokens。reset source 为4×2=8 actions，最多2,048 tokens；购入的每个独立 teacher state 再有2个 source actions。pilot eta 用 YAML 中七个训练候选及 initial_eta，不能利用 valid_seen 做选择。
5. 保留 teacher ledger、compute/source/episode/score diagnostics、recovery、step checkpoint/trajectory 和 manifest。检查 budget/ownership/fold/8-slot exposure、EOS/cap、每个 action 的完整生成 ID、原生 CE 与 tolerance、LOO identifiable block、单次 commit。window 允许全零回报/零 gate 梯度；不重复直到“成功”。15分钟 smoke deadline 如超时，报告资源限制并保留 recovery，不能缩 K 或改 LOO 偷过。
6. 释放训练模型进程后，再启动相同硬件 class 的独立官方 campaign，完整 greedy valid_seen 140题；最多5,600 actions / 1,433,600 action tokens。部分 smoke checkpoint 必须标 smoke，不能称完成一轮训练。执行/复用必须通过 C26-D checkpoint + config/tokenizer/world/harness/hardware + 全题 artifact 校验。

## 4. 集成后验收门槛

先在 CPU、tmp_path fixtures 运行，禁用 CUDA；测试不扫描活跃 run dirs：

```bash
PYTHONPATH=src:. .venv/bin/python -m pytest tests/test_rtd_alfworld_config.py tests/test_rtd_alfworld_contract.py -q -p no:cacheprovider
PYTHONPATH=src:. .venv/bin/python -m pytest tests/test_rtd_transport.py tests/test_rtd_functional_step.py tests/test_rtd_return_gradient.py tests/test_rtd_checks.py tests/test_rtd_acquisition.py -q -p no:cacheprovider
PYTHONPATH=src:. .venv/bin/python -m pytest tests/test_rtd_*.py -q -p no:cacheprovider
```

- 保留并运行现有 `test_rtd_evaluation_identity_reuse.py`、`test_rtd_manifest_tolerance.py`；不得覆盖用户未提交文件。真实环境 tests 必须按原测试的 opt-in 开关单独执行，不能让广义 CPU 回归自动访问生产资产。
- 新建 `tests/test_rtd_alfworld_resume.py`：在 reference/selected/revealed/actual/feedback/committed 每个 durable save 后中断，对比无中断轨迹、参数、source/feedback RNG、task/rollout seeds、P/eta/normalizer、owned/fold、spend、pending labels；不得重复 charge/commit。环境重新创建必须用已保存 seed/动作重建，不能重新抽一个随机 teacher prefix。
- 原生 selector 测试必须在没有 test-only patch 的情况下拒绝所有新增 privileged imports/data reads；普通 public candidate 选择可用，隐藏文本/won/usage 扰动不影响 public view。
- 配置：原 BFCL defaults/hash/round-trip、ALF/ scalar identity、未知字段、wrong split/140 count/40 steps/256 cap、memory/tolerance 漂移、另一模型/tokenizer、support/protected parent 变更均符合冻结规则。旧 BFCL manifests/protocol_version 不迁移。
- 数学：完整正混合、同任务 LOO（全0/全1→0、混合可识别）、source normalization、finite-difference insertion/VJP、gq=gD 零值、EOS 和无 EOS cap、observation mask、失败样本保留/基础设施异常中止。
- 官方 campaign：少题/重题/wrong split、bool score 类型、错 adapter/config/tokenizer/hardware/step cap、缺 base class、artifact 损坏全部拒绝 aggregate/reuse；report 不能把 0–1 success_rate 当百分比。锁 timeout、同 tag 恢复、同 port 排他性、code drift 和 manifest-bound identity continuity 都需回归。
- 完成 CPU 与两 train replay / rai 单窗口 correctness 后才运行正式 R0/R1 3×12、每轮 eval，随后同 owned-set CE、固定 ledger 2×2/单组件变体；不设置 success 必须提高的 gate。C26-E validator 暂只接受 sealed_replay，fixed_evidence 衍生配置须在 C26-F/后续单独接入原固定 ledger 验证，不擅自删该标记绕过校验。

## 5. C26-E 验证记录

隔离目录：`/tmp/tc-alignment-c26e`。真实只读 CPU audit 输出：`/tmp/c26e_audit.json`。真实 sealed bank 为 `data/rtd/v1_alfworld_c26`：219 attempts，107 usable，112 unavailable，m=135；usable public-cap 总额140,247,040，预算14,024,704 / 35,061,760 / 70,123,520。历史 output estimate189,541；usable estimate36,294；失败 estimate153,247；另有1次已知 missing attempt，完整 input/reasoning/retry 成本未知。

C26-E 的 CPU tests 使用合成 bank/world/model bytes；没有加载 4B、运行 GPU、启动真实环境或改动 sealed bank。指定两文件测试最终通过95项（20.25秒）；可应用 patch 为 `/tmp/c26e-new-files.patch`；原工作树的 `PROJECT_STATE.md` 在本任务开始前已修改；工作期间该文件及 `notes/exp_log.md` 又收到并行运行进度更新。本任务未写这两个文件，也不回滚他人的更新。
