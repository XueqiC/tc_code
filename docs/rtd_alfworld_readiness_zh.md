# RTD v1 ALFWorld sealed_replay 就绪审计与实施计划

审计日期：2026-09-07。本次只读代码、历史数据和日志，只写本文；未启动 teacher、环境 episode、模型推理、训练、评测、测试、集群作业或同步。token 统计只在 CPU 上读取本地 tokenizer。以下数字是本机快照，HPG 产物只在明确说明时引用实验日志，未远程验证。

依据：`docs/RTD_END_TO_END_METHOD_AND_EXECUTION_V1.md` §2、§4.2–4.3、§9、§12 ALFWorld、§13；`docs/2026-09-06-rtd-v1-execution-plan-zh.md`；`docs/rtd_v1_protocol.md`；`docs/rtd_v1_data_access.md`；`configs/rtd/v1_bfcl_c25.yaml`。协议中的 C25 及后续修订覆盖旧 BFCL 实施说明；BFCL 单轮 8×4 的资源特例不自动迁移到 ALFWorld。

## 1. 结论与当前缺口

可以准备 **历史命令证据上的 exploratory sealed_replay**，但当前不能直接把 BFCL YAML 改成 `benchmark: alfworld` 开跑，也不能声称已具备完整原始 teacher 请求回放。

- 原始历史账本有 **219 次 episode 尝试、142 个 task ID**；成功示范 **107 个、1,377 个命令 turn**。所有记录的 teacher 都是 `gpt-5.4`。
- 可保存 219 个历史尝试条目；其中 107 个有精简的命令 payload，可作为待核验包；112 个失败尝试没有响应文本，只能留在成本审计，当前不能作为可重放包。已完成状态重放验证的 RTD ALFWorld 包数量目前是 **0**。
- `tokens_spent` 合计 **189,541**，是旧代码的字符估算，不是精确 API usage。完整原始回复、reasoning、逐调用 usage、失败轨迹及完整 retry 边界均缺失。不能由保留下来的短命令反推真实付费输出。
- 385 行事件来自上述 107 个示范；85 行 consequential 子池只覆盖 **60 个 task ID**。它们不是 385/85 个新 teacher 请求，不能拿事件数当支持父任务数或采购次数。
- 可复用 TextWorld worker、官方命令解析和 RTD 数学内核；缺 bank 重建、完整状态审计、ALFWorld support/feedback/evaluation、配置与身份分派。
- 建议使用全部合法历史支持范围，报告 `m=135` 个训练父组、138 个 task ID，标 `exploratory`。这是从 142 个历史 task ID 合并同游戏 trial 后的 139 组，再排除 4 个既有 probe 父组；详细折映射见 §5。不能沿用 BFCL 的 `m=40`。
- 必须列入强 CE 锚点：`alfabl_CE_conseq_s0`，valid_seen **109/140 = 77.86%**；base **10/140 = 7.14%**。现有记录不是新 RTD 的公平同预算结果。

## 2. 教师材料清点、请求边界与 token 成本

### 2.1 逐来源库存

下表“新增包”指相对于原始 ledger 的新增真实 teacher 请求；同一材料的副本、事件、utility、fingerprint 不产生新包。行数已由本机 JSON/JSONL 读取核对。

| 来源 | 本机数量及覆盖 | 可封存包及请求定义 | 成本与可用性 |
|---|---|---|---|
| `data/alf_records` | **不存在** | 0 | 请求中这个目录不能作为已找到的档案引用。实际评测记录在下一行。 |
| `results/alf_records/*.jsonl` | 30 个评测文件；base/CE/A/C56 等核对文件各 134 行，字段为 `task_id, won, steps` | 0 teacher 包 | valid_unseen 学生评测结果，没有 teacher continuation 或 usage；不能流入采购、训练或反馈折。 |
| `data/teacher_ledger/alfworld.jsonl` | 219 行、142 task ID；107 success、112 failure；1,377 个保存的 demo turn | 219 个历史 episode 尝试条目，107 个命令证据候选包，112 个 unavailable；每包从该 train task 的 reset 状态开始，包含同一次 attempt 的全部已保存命令 | 189,541 历史估计 tokens；成功部分 36,294，失败部分 153,247。尚无 provider 精确输出计数。 |
| `results/bfas/alfworld/collect_shared/demos.json` | `demos=107`、`completed_task_ids=142`、`complete=true`；demo task 集与 ledger 的成功集一致 | **新增 0**；为上述成功 episode 的缓存副本。封存前仍须比对 turn 内容，不能仅凭 task ID 合并 | 不重复收费；不能以 `complete=true` 推断原始 usage/失败响应完整。 |
| `results/bfas/alfworld/collect_s0/{collection,unguided,guided}.json` | `p_hats=142`；guided 的 rounds 映射覆盖 96 题；本机 `collection.json` 的 `teacher_demos` 为空 | **新增 0**；学生 unguided/guided 回放，不是原始 teacher 调用 | 历史环境/学生计算需另记。不能把 guided 成功当作无教师成本数据，或把旧 p-hat 当本轮反馈。 |
| `results/bfas/alfworld/ours_s0/pool.jsonl` | **4,864 行，全为 `teacher=self`，68 个 task ID**；旧 T8 文档记为 205 条成功 self trajectory | **新增 0**；其中 guided trajectory 有示范依赖 | 不是 4,864 行 teacher 数据。不得当 RTD 本轮冻结来源样本；需按当前轮策略重新采样并记录 IDs、EOS、logprob。 |
| `data/alf_sft/events_v1.jsonl`、`pool_A_all.jsonl` | 各 385 行、107 task ID；旧 utility 分类为正 85、零 268、负 32 | **新增 0**；385 个派生事件归入 107 个 episode 包 | `tools/alf_event_mine.py` 无 teacher API；重放 demo 前缀后替换学生回复的 ACTION。不能给每个事件分摊一个伪“请求价格”。 |
| `pool_B_first.jsonl` | 107 行、107 task ID | 新增 0；同包的首个被探测分歧 | 不是独立请求类，也不应成为 RTD 手写首分歧策略。 |
| `pool_C_conseq.jsonl`、`pool_D_weighted.jsonl`、`pool_events_pref_v1.jsonl` | 各 85 行、60 task ID；T8 记为相同内容 | 新增 0；是 A 的同源子集 | 只作历史 CE 锚点/审计；主 bank 不用 ΔU>0 过滤或据此决定允许教学。 |
| `pool_A_85.jsonl`、`pool_zero85.jsonl` | 分别 85 行/62 task ID、85 行/59 task ID | 新增 0 | 随机/零 utility 历史对照；不扩展 teacher 覆盖。 |
| `pool_E_negteacher.jsonl`、`events_negative.jsonl` | 93 行/63 task ID，32 行/26 task ID；E 含 8 个交换正负答案的行 | 新增 0 | 交换后的学生答案不因此变成 teacher 输出；不能照搬为 RTD 正教师目标。 |
| `events_v1_k8.jsonl` | 268 行、94 task ID | 新增 0 | `alf_event_reestimate.py` 只增加学生分支 rollout；旧 K=3 的零桶重估，不能收费为新 teacher 文本。 |
| `event_value_v1.jsonl`、`event_value_ce.jsonl` | 各 385 行、107 task ID | 新增 0 | 旧 reach/edit/value 诊断；不能作为新 gate/selector 特征或官方终局奖励。 |
| `events_r2ce.jsonl`、`r2/pool_A_all.jsonl` | 各 183 行、76 task ID | 新增 0；复用同一批 107 demos，在 CE 学生状态上重挖 | 额外成本是学生/环境计算。必须标出生成这些学生状态的 CE checkpoint 及其训练证据依赖；默认不混入首轮主 bank。 |
| `r2/pool_B_first.jsonl`、`r2/pool_C_conseq.jsonl`、`r2/pool_D_weighted.jsonl` | 76/76、25/24、25/24（行/task ID） | 新增 0 | 同源派生子集。 |
| `r2/pool_union_C.jsonl`、`r2/pool_E_negteacher.jsonl`、`r2/events_negative.jsonl` | 110/76、36/25、16/11 | 新增 0 | union 的 110 行也不是 110 包；r2 E 中有 11 个反向偏好行。 |
| `events_smoke.jsonl` | 4 行、2 task ID | 新增 0 | 测试性事件不加入正式库存分母。 |
| `probe_alf_base_correct_v1.jsonl` | 4 行、4 个非 demo train task ID，但都在历史 142 demand 中 | 0 teacher 包 | 旧 base-correct probe；保留排除清单，不用于新 RTD 教师监督、反馈或统计量拟合。 |
| `data/events_unified/alfworld_v1.jsonl` | 385 个标准化事件 | 新增 0 | 原始事件的 schema 转换，不能升级为真实 teacher continuation。 |
| `data/atoms/alfworld_v1_K32.npz` | `Z:385×32`、`U:256×32` | 0 teacher 包 | 历史 PCA loadings；含 teacher/student 差分证据，不能购买前使用，RTD 也不需要这个原子层。 |
| `data/fingerprints/alfworld_v1_v1.{npz,index.jsonl}`、`fisher_alfworld_v1_v1.npz` | `psi:385×256` 及索引/Fisher | 0 teacher 包 | 诊断，不复用为 RTD 的初始学生 hidden-state projection 或 train-only P。 |
| `results/analysis/transfer_matrix_alfworld_v1.npz`、`*_eval*.json` | 数组预分配 385×385，**实际只有 120 行全有限，即测量 120×385** | 0 teacher 包 | T9 三步 pairwise 的局部似然迁移；不等于 success 回报梯度，不能作采购标签。 |

### 2.2 真实 teacher 路径与哪些内容已经丢失

没有找到独立的 `src/alfworld_teacher.py`。实际链路为：

`tools/alf_collect_azure.sh` → `bfas.run` / `src/bfas/ledger.py` → `ALFWorldAdapter.teacher_episode()` → `_episode(teacher_config=...)` → `src/appworld_teacher.py:generate_reply()`。

`teacher_episode()` 是历史账本的 episode 尝试边界，不是一次 HTTP 请求。一场最多 40 步；每步给 teacher 当前 goal、observation、admissible commands、最近历史及 ReAct 示例，收到文本后解析命令并执行。因此完整包可以包含多个真实 teacher turn；不能把新 online 默认的 8 turn 上限追溯套到旧 episode，也不能把一场拆成任意长度的新可购请求。

关键丢失发生在两处：

1. `ALFWorldAdapter._demo_turns()` 将 `target` 换成解析后的**命令**，并保存 deployment turn。teacher 分支保存的 deployment prompt 是旧 bare prompt；真实 teacher 调用用的是 ReAct messages。虽然 demo turn 有 `prompt/target/context`，它不等于原始 API request/response 档案。
2. `teacher_episode()` 仅把 `response_texts` 暂交给 ledger 计数；`src/bfas/ledger.py:estimate_response_tokens()` 用 `Σ ceil(len(response)/4)`，`append_episode()` 不保存这些原始文本。失败尝试连 demo 都没有；`strip_think()`、空回复重试和 API 内部 reasoning 的成本不能从保留命令恢复。

219 行均有正 `tokens_spent`，但这不证明成本完整。成功记录的估计范围是 25–2,011 tokens/episode，失败为 276–2,366。成功父题上的失败记录为 7 次、9,420 estimated tokens，须保留在历史支出。另有一个 task 的 ledger 仅存 `attempt_index=1`，缺 0：`look_at_obj_in_light-Pillow-None-DeskLamp-314/trial_T20190908_115833_802764`。不能把缺失尝试补成零元，也不能推定它曾经成功/失败；记录 `historical_attempt_gap`。独立 reset 的前次失败不是天然的后次状态依赖，但它仍是未完整恢复的历史成本。

本机 `data/azure_usage.jsonl` 有 1,382 行，仅 `t/model/total_tokens`：910 行 gpt-5.6-luna、472 行 claude-sonnet-4-6，**没有 gpt-5.4**，也没有 task/request ID、input/output/reasoning 拆分。它不能给本批 ALFWorld 补上精确账单。当前 teacher client 支持 `AZURE_USAGE_LOG`，不代表当年每次调用都留下了可归属记录。

### 2.3 本次 tokenizer 复算：只衡量保留文本

使用本地 `Qwen/Qwen3.5-4B` snapshot `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` 的 `tokenizer.json`，通过 `tokenizers.Tokenizer.encode(..., add_special_tokens=False)` 计数；不加载模型、不添加 EOS、不截断。它给出这些字符串在**学生 tokenizer 下的精确长度**，作为 teacher 成本时仍只是可见文本估计，不能代替 provider usage。

| 文本 | prompt tokens 合计 | 保存的正目标 tokens 合计 | 旧 student `_rejected` tokens 合计 | 长度提示 |
|---|---:|---:|---:|---|
| 107 demos / 1,377 commands | 672,908 | **8,327** | 无 | 每命令均值 6.05；每包命令合计中位 60、最大 212；保存 prompt 最大 1,079。prompt 是 deployment 版本，不是真实 teacher 输入账单。 |
| A / 385 events | 231,012 | **8,944** | 97,200 | prompt 最大 1,263；目标中位 8、均值 23.23、最大 265；旧学生中位 256。 |
| C / 85 events | 48,875 | **3,094** | 21,179 | prompt 最大 1,134；目标均值 36.40、最大 265；旧学生均值 249.16。 |
| r2 / 183 events | 116,679 | **1,528** | 1,927 | prompt 最大 1,103；目标中位 7。 |
| 4 个 probe | 2,920 | 853（学生输出） | 无 | 0 teacher token；不进入本轮训练。 |

不能相加得到教师总支出：C 是 A 子集，event response 又可能混入学生原有 thought。保留 **189,541 历史字符估计**作为旧账本总量，另列上述文本长度；成功包揭封可扣对应历史 `tokens_spent` 并带 `cost_confidence=estimated`，不能把 36,294 改成 8,327 以制造效率提升。input、reasoning、discarded output、完整 retry 及货币成本应记 `unknown`，不是 0。

### 2.4 封包建议与缺口处理

建议先实现 `alf_demo_episode` 这一类：每个历史 attempt 一个 opaque SHA-256 query ID，绑定 ledger 文件 hash、task/attempt/timestamp 和行号；完整性 hash 与 payload 留在 sealed 区，不能成为 selector 特征。候选公开的是可预先知道的 train reset 请求状态、固定请求类、统一 cap 和来源特征。不能公开成功分数、响应长度、动作序列、teacher-derived embedding。

成功包内保存整条命令序列、原始 ledger 行、所有同源事件 aliases、精简/缺失字段说明。构造 teacher-prefix state 时，从固定环境 reset 逐条执行包内命令，保存实际 observation、admissible commands 和**完整有序历史**。用 `FullState.create()` 校验历史 teacher 目标与本轮 source sample 位于同一状态；仅 prompt 字符串相等不够。旧事件有 381 个不同 prompt、385 个 `_traj`；不同父任务或不同 history 不得按 prompt 单独合并。

本次没有运行上述环境重建；因此 107 是**候选上限**，不是已验证可用量。三种缺口需分别处理：

- 请求/状态可还原、仅 usage 不精确：可按 estimated 的历史命令证据回放，明确 `payload_kind=extracted_teacher_commands`；不声称保留原始 teacher 推理全文。
- 仅能看到合成 event response，无法恢复其 episode/命令/状态来源：unavailable，不能伪装成“学生现场问 teacher 后收到这段回复”。`corrected_reply()` 生成的混合 thought+ACTION 应单独标派生变换；主 RTD 目标优先使用已归属的 teacher 命令文本，锚点 CE 可保留历史配方。
- 失败响应/状态缺失：保留账本与 unavailable 原因，计入完整历史成本，不纳入 usable bank 分母。不能为凑 219 包发明失败文本，不能按隐藏成功标签给 selector 排序。仅成功 payload 留存造成的幸存者偏差必须在结果中声明。

同一包揭封一次后可在多个已验证状态使用，不重复计费。包内 teacher 前缀不是免费公开状态；需先拥有整个包。如果将来恢复逐 turn 原始请求，可另做有向依赖链，深处请求必须依赖产生其状态的此前 teacher 包，检查所有依赖的折和预算。不要把现有 1,377 个命令直接拆成 1,377 个廉价采购机会。

`PublicQuerySpec.L` 应表达**请求预先声明的最大 teacher turn 数**：当前 episode 方案拟定 L=40，实际 3–40 turns 放 sealed。当前文件没有完整的逐请求配置 envelope，40 只能标历史程序重建约定，不能以实际响应长度充当 public L。若严格要求逐请求的 `recorded_only` 证据，仍须恢复旧启动参数/原始 envelope，或把该运行标成有声明近似的历史回放；不能写成已满足严格原始请求 replay。

### 2.5 ledger 的 public per-class caps

BFCL C25 的模式可复用：**先按统一 public cap 预留，揭封后按记录成本扣账并释放余量；分母为 usable public cap 之和；横轴为实际扣账的 exact/estimated token**。不能把 BFCL 的 DeepSeek 65,536/8,192 直接解释为 gpt-5.4 的真实限制。

本计划建议 C26 冻结以下**回顾性约定**，不是已恢复的 provider 账单保证：

| 请求类 | 建议 public output cap | 依据、适用范围 |
|---|---:|---|
| `alf_demo_episode` | **1,310,720 = 2,048 × 40 × 2 × 8** | 本机 `appworld_teacher.MAX_COMPLETION_TOKENS=2048`；ALF episode 默认 40 steps；空内容最多再调用一次；client 的最大 retry 循环次数 `max(CHAT_COMPLETION_RETRIES+1, RATE_LIMIT_RETRIES+1)=8`。所有 episode 同 cap，包括失败条目；缺历史配置绑定，所以标 `class_uniform_public_fallback`。 |
| `alf_teacher_turn`（未来恢复原始 turn 档案时才启用） | **32,768 = 2,048 × 2 × 8** | 一个环境 turn 的完整尝试/空回复重试组。当前可验证此类包数 0，不计入本轮分母，不从 episode 拆包。 |
| event、r2 event、atom、T9、self rollout、evaluation record | 不设可购类 | 全是已有包别名或学生/环境产物，新增 teacher 包数 0。没有 ALFWorld `generator_item` 库存。 |

上述乘数来自当前源代码，旧配置、quota recovery 的额外探测、返回丢失的已处理请求都没有完整绑定，不能宣传成真实历史总账的硬上界。新的 cap provenance 不能照搬 `caps.py` 里“project lead 已决定”的措辞；它是本文的实施建议，待 C26 配置冻结。封存时若记录成本超过 cap，必须 fail closed；缺失成本继续标缺失，不倒推一个刚好可买的 cap。

若 107 个候选全部通过状态审计、无额外排除，示例 `B_bank=140,247,040`，10/25/50% 上限分别 **14,024,704 / 35,061,760 / 70,123,520**。这是条件计算；正式 manifest 以实际 usable 集合重算，112 个不可重放尝试不加入分母。

此 convention 很宽：全部 107 个成功包的历史估计也仅 36,294 tokens，而每臂最多 12 次新选择。因此很可能由决策窗口/包覆盖限制结果，三个 cap 点不会形成有辨识力的成本曲线。必须报告实际花费、剩余预算、empty 次数、每折可购量和 cap/estimated-cost 比；不声称已测出精确 teacher-token efficiency，也不为花满预算而拆包。预期成本约束 `b=remaining_budget/remaining_windows` 不是单包硬 cap。

## 3. 环境、随机完整任务反馈与官方评测

### 3.1 可复用接口及限制

`src/bfas/adapters/alfworld.py`：

- `DATA=envs/alfworld/data/json_2.1.1`，`ENV_PYTHON=envs/alfworld/.venv/bin/python`；两者本机存在。按 `_game_ids()` 相同过滤逻辑只读计数：train **3,553** 个 task ID（1,465 个游戏名父组），valid_seen **140**（137 组），valid_unseen **134**（47 组）。这只是文件可用性，不是环境启动测试。
- `_EnvBridge(split, task_id)` 启动独立 Python TextWorld worker。先读 ready，再读初始 state；`step(command)` 通过带 request ID 的 JSON line 往返，返回 `observation/admissible/done/won`，`close()` 回收进程。
- `_worker()` 用 `AlfredTWEnv`、batch_size=1、`domain_randomization=False`；内部 `max_nb_steps_per_episode=50`。外层 `_episode()` 默认 **40** actions，正式配置应显式固定 40，不能依赖两处默认值恰好一致。
- `_goal_line()` / `_obs_with_goal()` 保留首个 observation 中的目标；固定渲染使用当前 observation 尾部 2,000 字符、最近 8 条 history、每条反馈前 300 字符。完整 state 归档须保留未裁剪历史；policy prompt 仍遵循已冻结的历史 harness 窗口，不能静默切换。
- ReAct student 默认开启，包含 `TEACHER_REACT_INSTRUCTION` 和六类 `TEACHER_REACT_EXAMPLES`。这些固定示例属于已有 harness 先验，须在 manifest 公开、hash 并让所有臂及 CE/base 一致；不能把它说成无示例 zero-shot。task type 仅用于既有环境提示与事后分组，不输入 gate、采购器或 loss。
- `_teacher_command()` 取最后一个 ACTION；`src/alfworld_eval.py:pick_command()` 依次做匹配、大小写/子串处理，失败回退 `look`。这会把不完整思考映射成一个环境动作。它是现有评分路径的一部分，不能在 RTD 悄悄改成失败奖励或过滤样本。
- `_server_reply()` 在 ReAct 下最多 **256** tokens，裸命令模式 32；但 `BFAS_NO_SERVER=1` 的 `_episode()` 分支固定传 32，且只保存文本、没有生成 token IDs/logprobs。直接调用 `adapter.rollout()` 不能满足 RTD §4.2。
- `_EnvBridge._read()` 目前没有请求 deadline，worker 异常需传播；新的封装须提供有记录的 timeout/清理机制，不能把基础设施故障转成 reward=0。新实现不应直接重用会全量覆盖输出的 `evaluate()` 写文件流程。

### 3.2 RTD full-task stochastic rollout 的接法

新增 `alfworld_rollout.py` 提供类似 `alfworld_task_rollout(task_ref, backend, parameters, generator, env_factory)` 的接口：

1. 每条反馈 trajectory 从 **train 完整任务 reset** 开始，无示范 prepend、无 teacher prefix、无从分歧点续跑。给同一实际 task 重复 K=2 次，以 fresh worker/reset 隔离环境；连续采样使用可恢复 RNG，不能每条重置成相同 seed。
2. 每一步用冻结的 ALFWorld ReAct renderer 得到 prompt，调用现有 `HFGenerateBackend.sample_action()`，`temperature=1, top_p=1, top_k=0`，保存完整 `ActionTrace`：prompt/action IDs、实际 EOS 或 cap termination、逐 token generation score、policy/backend identity。
3. 用原始生成文本走既有 command parser，保存 raw text、执行 command、完整 observation/history。**REINFORCE 计分针对生成的全部文本 tokens**，包括 thought 和实际 EOS；解析出的短命令不是重新计分的行为。环境 observation、prompt/example tokens 均不计为动作；未采样 EOS 不补 EOS。
4. 执行到 `done` 或固定 40-step horizon；返回 `TaskRollout(reward=float(won), from_task_start=True)`，奖励严格是环境 success 的 0/1。记录 horizon、action cap 和 parser fallback 次数作诊断；fallback `look` 后仍可能成功，不套用 BFCL “malformed 必须 reward=0”的规则。worker crash/timeout 是失败的测量，需恢复/报错，不是学生失败。
5. 调用 `return_gradient.py:reinforce_gradient()`，同一 task 的 LOO baseline；按 **rollout 数**归一化，累加所有 action 的完整序列 logprob，不按 action 长度或成功子集归一化。生成分数与 teacher-forced 分数逐 token 校验，复用当前 `ScoreTolerance` 的完整日志和拒绝规则，不因 ALFWorld 稀疏回报放宽阈值。
6. 虚拟参考 `theta_D+` 上的 `gJ_ref` 服务 insertion/acquisition；实际提交 `theta+` 上的 `gJ_actual` 服务 gate。两点相同时才复用。`reward/baseline/gJ/source/P/eta/normalization` stop-gradient，复用 `runtime.streamed_gate_vjp()`；不能使用旧 T9 三步 Adam 参数差替代固定 P 的一步导数。

来源采样是另一件事：在合法 inner reset 或已购包中恢复出的相同完整状态上，每轮冻结 source policy、每状态采 2 次全动作。旧 `_rejected` 多数在 256 tokens 截断，不可重标成完整 EOS source，也不能作为本轮 off-policy 样本直接塞入梯度。

### 3.3 官方评测及强 CE 锚点

正式开发端点固定 **valid_seen 的完整 140 个 task ID、greedy temperature=0、ReAct、256 tokens/action、40 steps**，主指标 `successes/140`。`ALFWorldAdapter.evaluate()` 默认是 valid_unseen，且 `BFAS_ALFWORLD_EVAL_GAMES` 可缩小清单；新 campaign 必须显式锁 split 与完整 IDs，并拒绝缺失、重复、额外记录以及错误 denominator。

| 锚点/历史结果 | checkpoint 或证据路径 | valid_seen 140 | valid_unseen 134 | 解释 |
|---|---|---:|---:|---|
| Base | `Qwen/Qwen3.5-4B`；`scripts/alf_valseen_hpg.slurm`；`notes/exp_log.md:650–651` | **10/140，7.14%** | 12/134，8.96% | 同一 ReAct scaffold。 |
| **Strong CE-C** | **`results/appworld_students/alfabl_CE_conseq_s0/hub_merged`**（脚本使用）；原始 checkpoint 在同目录 `adapter/` | **109/140，77.86%** | **100/134，74.63%** | 85 consequential events ×4 epochs，44 optimizer steps；`exp_log.md:582,650`。 |
| 最佳历史 pairwise C56 | `results/appworld_students/alfctl_C_56step_s0/hub_merged` | 38/140，27.14% | 38/134，28.36% | C 池约 55 steps；不是 strong CE 替代品。 |
| CE-A | `alfce_CE_A_s0`；`results/alf_records/ce_CE_A_s0.jsonl` | **无已核实分数** | 105/134，78.36% | 385 行、49 steps；不能把 78.36% 填进 valid_seen 栏。 |
| `crcd_alf3_C_rho1_s0` | `results/bfas/alfworld/crcd_alf3_C_rho1_s0/eval_metrics.json` | 20/140，14.29% | 未据此声称结果 | 85 行、44 steps，旧 mirror；其余四个计划臂被取消，不能列为已跑结果。 |

CE-C 的 `adapter/` 本机仍存在，含约 8.41 GB `model.safetensors`、config、tokenizer 和 chat template；**缺的是 `hub_merged/`**。本次只检查文件存在/大小，没有遍历权重 hash 或重做 merge。valid_seen 的 CE/base 数字来自 HPG 实验日志及脚本，相关 `vs_*_s0/eval_metrics.json` 本机未见；表中成功整数由日志百分比和140反推，待原始记录复核。valid_unseen CE/base/C56/A 的 134 行记录已逐行核对成功数。后续须校验保留权重内容、用新目录导出并在同一硬件/同一 harness 重评。

§9 要求的公平强对照应另外建立：从相同 base 出发，在某个完成的 R0/R1 **实际 owned 包集合**上用历史强 CE 配方训练，并记录相同 teacher 历史支出、曝光和训练算力，给出 valid_seen 端点。历史 85 行是按 ΔU 选择的事件集合，既不等于 RTD 最多 12 包，也不等于其命令目标；两个比较都要报告，不能混写“同数据”。若只胜 R0 而远低于 CE-C，结论仅限固定混合闭环对照。

新评测须绑定 checkpoint 文件 hash、base/tokenizer/chat-template hash、数据清单、环境/解析/示例/step cap、解码参数和 hardware class；逐 task 保存结果后完整性通过才 aggregate。复用已有结果必须验证整套身份与 artifacts hash，不能仅凭目录名、aggregate 数字或旧 BFCL identity supplement。输出逐 task repair/damage、成功率及区间；valid_seen 140 条存在 137 个游戏名组，至少报告按游戏父组聚合的任务采样不确定性。

valid_seen 是开发数据。valid_unseen 虽名为 unseen，历史已被反复用于比较、挖结论，不是 untouched certificate；134 个 task ID 只有 47 个游戏名组，也不能当 134 个独立场景认证。独立最终确认/认证样本及其未触碰证据仍缺。保持单训练 seed，不宣称跨训练种子稳定性。

## 4. `src/bfas/rtd/*.py` 的 BFCL 绑定逐项审计

下面区分算法复用与实际耦合；“有 BFCL 依赖”不等于整文件都要重写。函数/符号是后续 integration 的定位依据。

| 现有模块/位置 | BFCL 专用内容或隐藏假设 | ALFWorld counterpart / 后续动作 |
|---|---|---|
| `bank.py`：`build_bfcl_bank`, `parent_hash`, `full_state` | BFCLAdapter entries；BFCL support/calibration、DeepSeek attempt、生成池、FC render、function/question/GT schema | 新 `alfworld_bank.py` 与 `alfworld_state.py`。原 builder 保留，不扩成夹杂两个 benchmark 的大函数；调用方改 registry 分派。 |
| `caps.py`：`PUBLIC_CLASS_CAPS`, `public_cap`, `limits_from_metadata`, `archived_cap_evidence`, `affordability` | 固定两类 demo/generator、DeepSeek URLs、BFCL 脚本证据、两类列统计 | 新 `alfworld_caps.py`；公共 audit 接口接收 class/cap policy。不能让 ALF 包通过 BFCL 常量一致性检查。 |
| `experiment.py`：`BFCLSupport` | 用官方 entries/ground truth；`parents[h]→单个 tid`；initial question/function state；memory/web unavailable；CheckerBridge | 新 `ALFWorldSupport`，parent→trial 清单及选定 task，完整 reset state、环境 feedback；同 task 两次 LOO，不能同父组不同 trial 混作同 task。 |
| `experiment.py`：imports、`sample_state`, `choose_feedback_tasks`, smoke `round_start` | 顶层引入 BFCL render/parent hash/rollout；`category.startswith('multi_turn')` 决定 cap 与反馈；**硬编码 8×4 / 4×2，未真正读取 YAML 的反馈剂量**；smoke 限 single-turn | 把 support/rollout/action-limit/feedback task sampler 作为依赖注入；ALF 固定 4×2。类别仅诊断，不进入控制器。其余 phase/resume/slot/insertion 程序复用。 |
| `return_gradient.py`：`bfcl_task_rollout`, `bfcl_decode` import | BFCL handler inference、函数执行全局状态清理、AST/relevance/multi-turn checker | 新 `alfworld_rollout.py`，不用 checker bridge。保留通用 ActionTrace、TaskRollout、LOO、REINFORCE、VJP；BFCL rollout 留在自身路径。 |
| `return_gradient.py`：`TorchPolicyBackend.action_limit`、TaskRollout malformed 约束 | action class 仅 single/multi；BFCL malformed action 强制 reward=0 | action-limit 交给 benchmark policy；ALF parser fallback 独立诊断，不能标成这个 BFCL malformed flag。通用终止/完整 task 校验保留。 |
| `runtime.py`：`load_backend` | 按 benchmark 读取 action caps，但 fallback 为 BFCL 512/1024；继承的 action_limit 仍解析 single/multi | 明确 ALF `agent_action=256` 或等价声明，经注册的 cap resolver；保留 HF 生成/功能计分一致性、FP32 LoRA、逐 action streaming 与 checkpointing。 |
| `cli.py`：`load_config`, `bank_audit`, `data_identity`, `make_manifest`, `run_command`, `_checker_context`, parser | canonical 是 `v1_bfcl_c25.yaml`，benchmark/m/caps/历史成本等被冻结；builder 固定 BFCL；数据 hash 固定 leaderboard；固定 BFCLSupport + CheckerBridge；模型路径工具名为 bfcl；help/default config 也是 BFCL | 新 ALF CLI/config validator/manifest audit，最终在原 CLI 用 benchmark registry 分派。通用模型本地 snapshot 查找可复用，但不启动 BFCL campaign/checker。 |
| `evaluation.py`：`official_expectations`, `validate_evaluation`, `_completed_campaign_identity`, `evaluate`, `report` | BFCL all-category/prereq 清单、`BFCL_v4_*`、CSV overall、`results/bfcl_std`、`BFCLSTD_*`、merge/campaign 脚本/成功日志标记；报告字段 accuracy | 新 `alfworld_evaluation.py`：140 IDs、bool won、success rate、ALF 输出目录、同类硬件 base comparison、身份严格复用；共享预算/repair-damage 报告需 metric-aware 分派。 |
| `evaluation_lock.py`：lock path | lease 机制可复用，但路径固定 `results/bfcl_std/.locks` | 新 ALF campaign/tag namespace；若两者共享同一服务端口空间，端口锁必须共同互斥，不能因目录不同同时占一个端口。 |
| `identity.py`：`LEADERBOARD`, `EVALUATION_TOOLS`, `CONTENT_VERSION`, `_evaluation_harness_paths`, `evaluation_harness_identity`, legacy 分支 | 评分库存与版本均为 BFCL，要求其 package/data 存在；`source_identity` 扫整个 src/tools/scripts | 新 ALF environment/scoring identity，原 guard 通过 benchmark identity provider 选择。把 ALF library/TextWorld、game bytes、prompt/parser、模型渲染、horizon 纳入绑定。 |
| `scoring_scope.py`：`PYTHON_SCOPES`, `SHELL`, `scoring_projection` | BFCL checker verdict block、campaign shell regex、BFCL evaluation/adapter selector、BFCL export AST | 新 ALF scoring scope，独立版本；通过 dispatcher 接入，**不重定义已有 BFCL v3/v4 的语义**。 |
| `identity_update.py` | C25 历史证据、BFCL `bfcl_eval/` 文件集合与 v3 bridge 特例 | 保留为 BFCL legacy 工具；ALF 新 run 不调用它。不需要为了 ALF 改旧 evidence/hash 常量，未来 ALF migration 如需要应新建。 |
| `bfcl_decode.py` | BFCL failed-decode handler monkeypatch 与错误类别 | 整体 BFCL 专用，ALF 不 import/复用，原文件无需修改。 |
| `features.py` | **没有 BFCL 专用特征**：32 维 frozen theta0 hidden projection + source full logprob + source length + intercept | 直接复用；renderer/source state 由 ALF 提供。不能用 atoms/T9/ΔU/类别补维度；normalizer 只用当轮 inner。 |
| `selector.py` | 数据结构通用；import guard 按名字片段 `broker/bank/ledger` 拒绝 | 新名 `alfworld_bank` 不会自动匹配 `bank` 片段；扩展显式特权模块拒绝清单并测试已加载模块路径，公共类型不变。现机制是合作式防漏，不是恶意 Python 的 OS sandbox。 |
| `broker.py`, `ledger.py`, `transport.py` | 核心通用，少量 `cc_pairs.digest` 依赖不等于 BFCL checker；FullState 必须 task dict + 非空有序消息 history | 原样复用 ownership、依赖、fold、reserve/reveal、state equality、正混合。provider 不详字段留 payload 的 usage/provenance，不能伪造精确值。 |
| `functional_step.py`, `insertion.py`, `acquisition.py` | 无 BFCL 奖励/任务 schema；通用固定 P 一步、曝光守恒、Bayesian posterior | 无必要代码修改；沿用数值测试。 |
| `scoring.py`, `checkpointing.py`, `memory.py`, `persistence.py`, `hardware.py`, `__init__.py` | 无 ALF 应替换的 BFCL评分逻辑；硬件同类校验/compute journal 已通用 | 保留，使用现有接口。hardware 的 rai/HPG class 行为继续有效，不因 ALF 放宽。 |

需要同时注意 RTD 目录之外的绑定：`src/bfas/adapters/alfworld.py`、`src/alfworld_eval.py`、`tools/bfas_eval_ckpt.py`、`tools/bfcl_hub_merge_export.py`、`src/bfas/adapter.py`。可只读复用解析、环境 worker 和导出能力；不要运行含 `rm -rf` 的旧 `alf_*slurm` 或 lane/waiter 脚本来启动新 campaign。

## 5. 支持父任务、两折与实际资源口径

### 5.1 m 与可复现的拟议两折

历史来源 `results/bfas/alfworld/ours_s0/support_split.json`：177 个 support task ID = **142 demand +35 calibration**，seed=50；142 demand 与 ledger 的 task 集完全相同，和这 35 calibration 无 task ID 交集。root `configs/support_split.json` 的 50/40/10 不是这批 ALFWorld 支持清单，不能套用。

同一 `task_id` 形如 `<game-family-and-goal-and-room>/trial_...`。142 个 ID 只有 **139 个首段游戏父组**；107 个 demo ID 只有 **104 组**。建议以首段作为保守的 train family/goal/room grouping，把各 trial 的具体 world 文件独立 hash 绑定；后续若内容审计发现跨名字的相同游戏还要合并。不能只 hash trial ID 随机分折，也不能将 task type 的六类当成 m=6。

本次只读计算的拟议 fold 算法如下；它不是现成的已冻结 support 文件：

```python
parent_game = task_id.split("/", 1)[0]
group_bytes = json.dumps(
    {"benchmark": "alfworld", "split": "train", "parent_game": parent_game},
    sort_keys=True, separators=(",", ":")
).encode("utf-8")
parent_hash = hashlib.sha256(group_bytes).hexdigest()
fold = int(parent_hash, 16) % 2
```

这里 benchmark 字符串只用于审计身份，不输入 gate/acquisition 特征。文件内容另存每个 game/traj 的 SHA-256 及环境数据版本，guard 时两者都检查；首段名称 hash 本身不是内容完整性保证。

| 范围 | task ID 数 | 父组 m | fold 0 / fold 1 | 用途 |
|---|---:|---:|---|---|
| 历史 demand 全量 | 142 | 139 | 81 /58 组，83 /59 IDs | 资源审计上限；包含 4 个旧 probe 父组。 |
| **建议 C26 正式支持范围** | **138** | **135** | **79 /56 组** | 排除 4 个 probe 父组；另永远排除 35 calibration IDs/对应组。 |
| 有成功 demo payload 的子集 | 107 | 104 | 61 /43 组，63 /44 demos | 不是单独的反馈集；4 probes 都在非 demo demand 中，故此数未变。 |

建议不把“只有 teacher 成功的 104 组”选为 feedback 分布，这会把 teacher 成功筛选带入回报测量。反馈从合法 135 组抽取，包含没有可重放 teacher payload 的组；反复环境回报访问均计数。3,553 个 train games 在本方案中只是可用环境库存，不是额外获准使用的反馈/调参池。多个 trial 属同一父组时，固定规则选 task，或在父组内按冻结 RNG 抽 task；同一反馈块该 task 的两条 rollout 必须完全相同起点。

轮次固定：r1 inner0/feedback1；r2 inner1/feedback0；r3 inner0/feedback1。先分角色，再创建候选、pending、D_inner、source replay、P、projection 统计及 standardizer。owned 永久保存，但另一折的证据当轮不能训练，也不能通过 prefix 依赖绕行；轮换后的反馈是 meta-training，绝非 untouched validation。无可购包就选 empty，不从另一折或 calibration 补。

还发现 139 个历史游戏名组中有 **7 个名字也出现在 valid_seen**（旧 calibration 有 1 个），与 valid_unseen 为 0。名字相同不证明 world bytes 或具体 trial 相同，需在状态审计中比对。valid_seen 本来就是 seen 环境开发端点；此处不声称父组独立认证。若未来要求连开发端点也按父组彻底排除，须重新冻结 support/bank/m/折与分母，不能在看到分数后删除。新 manifest 必须写明这点。

### 5.2 反馈与曝光数量

ALFWorld 建议冻结 `meta_tasks_per_feedback=4`、`rollouts_per_meta_task=2`，符合主规范 §4.3 和执行备注中多步任务 4×2 的口径。不要把每步 action 当一次完整任务，不继承 BFCL 单轮 8×4。

| 计算 | 一轮一臂 | 三轮一臂 | R0+R1 三轮 |
|---|---:|---:|---:|
| committed steps / 有效 source slots 上限 | 12 /96 | 36 /288 | 72 /576 |
| 决策窗口 / 新选择上限 | 4 /4 | 12 /12 | 24 /24 |
| 参考反馈（每窗口 4×2） | 32 episodes | 96 | 192 |
| 不同实际更新点的额外反馈 | 0–32 | 0–96 | 0–192 |
| **完整随机反馈合计** | **32–64** | **96–192** | **192–384** |
| 每轮官方 valid_seen | 140 greedy episodes | 420 | 840 |

R0 也遵循同样计算程序，不因 gate 固定就少算反馈造成不公平。实际与参考点一致/empty 时可复用，严格记录身份和省下的 compute。source 每状态 2 个动作样本不在上述反馈数中：总数为 `2 × Σ_round(本轮实际访问的不同合法 source states)`，包括当轮新购买后首次见到的状态。teacher prefix 的环境 replay、pilot、P、feature 提取、虚拟参考梯度、失败基础设施重试另计。

288 是加权曝光的动作槽位上限，不是 288 个整场示范。来源与教师两侧分别记录 token exposure；pending 插入遵守 `.75 g_D + .25 g_q`。首轮 D 为空时先依采购先验买包，合法已购后才 pilot；identity/no-op 不伪记为监督更新。

官方 R0/R1 6 个端点共 840 episodes；加同机 base 和历史 CE 各一次为 1,120；再加同 owned-set 强 CE 一个端点为 **1,260**。后续 2×2 的 4 个端点另加 560。以上都是环境计算，sealed_replay 新 teacher calls/tokens 均为 0，历史支出不归零。

## 6. rai / HPG 内存与时间：只给有边界的估算

本次没有 GPU profiling。可引用的实测证据：T8 旧 mirror 在 rai A100 的峰值 allocated **10.6 GB（C）/11.6 GB（A）**，约 **34 s/step**，不是 RTD 完整反馈；旧 ALF mining 是 385×2×3=2,310 个分支 continuation、77.6k student calls、10.4h，使用并行 vLLM，不能换算成串行 HF+反传吞吐；HPG 旧 CE 的 140 题 valid_seen 约 4 分钟；当前 BFCL RTD rai 曾达 **33.0 GiB allocated**，B200 一轮训练日志约 52 分钟，均不能直接当 ALF 预测。

4B BF16 base 权重约 8.4 GB；RTD 保留多个 FP32 LoRA snapshot/梯度/P，逐 action graph 和 checkpointing 能限制图驻留，但 generation 的 vocab scores、eager attention、上下文长度仍决定峰值。沿用 C25 LoRA r16/alpha32 作为声明，实际 trainable_numel 必须由加载后记录，不拿旧 r8 或“21.2M”口头值代替。

| 设备 | 建议首个 smoke 资源声明 | 每臂三轮训练+反馈的规划范围 | 每个 140 题 greedy 评测规划范围 |
|---|---|---|---|
| rai RTX 6000 Ada 48 GB | state batch=1、单 action 反传；预算 38 decimal GB、reserve=2 GB；短 prompt 典型峰值暂估 16–30 GB | **6–20 h** | 10–40 min |
| rai A100 80 GB | 同一算法配置；先 batch=1，可在不改样本顺序/数学结果下依据内存策略调度 | **4–14 h** | 8–30 min |
| rai RTX PRO 6000 Blackwell 约98 GB | 同上；不能按最大显存推断一定最快 | **3–10 h** | 5–25 min |
| HPG B200 | 1 GPU、14 CPU、100 GB host RAM 起步；沿用 streamed LoRA/checkpointing | **2–8 h** | 4–15 min |

这些是调度用的宽区间，**不是测得的吞吐或完工承诺**；模型冷启动、export、排队、环境启动、IO 和异常恢复另计。建议首次完整臂申请 24h 可恢复 allocation，先用短 smoke 测量 generation/score/env/peak，再仅修正 walltime 申请，不看 success 调算法。

可复算的时间模型比单个小时数更可靠：`T_feedback = N_episode × mean_actions × (t_generate + t_score_grad + t_env) + N_episode × t_reset`。按每臂最多 192 episodes、平均 30 actions、每 action 256 tokens，约 **1.47M** feedback generation tokens；40 actions 上界为 **1,966,080**，且每个动作还要重算似然/梯度。假设有效每 action 耗时 1/2/4/8 秒，仅 5,760 个反馈 actions 就分别约 1.6/3.2/6.4/12.8h。再加 source、pilot、P、reference/gate，旧“ALF eval 4 分钟”显然不能替代 RTD 预算。

保持训练与 vLLM 评测分阶段，先释放训练进程再启动评测服务；GPU_UTIL 是服务预留比例，不等于训练峰值保证。长 rollout 增加时间和日志，不应把 40 个 action 的计算图全部留在显存。全长 state 留在 CPU 归档；不得为适配 48GB 静默截断任务语义。出现超 context 要报错并记录，不减长动作 EOS 或抛弃低回报 trajectory。

协议已有 hardware class：Ada 50,865,307,648 bytes、A100 85,094,825,984、PRO 101,971,722,240；用户常用的 48/80/98 GB 名称不能替代 manifest 的精确 bytes。三张 rai 卡是**不同比较类**。R0/R1、同 owned CE、base 应在同一 GPU class/软件环境运行；不能把 Ada R0 与 A100 R1 当主配对结论。HPG B200 容量/driver 按实际 allocation 记录，不假设；设备 UUID 换同类可以审计，跨类不能复用训练身份或基准分数。

## 7. 分阶段 Codex 实施：先仅新增文件，后停跑集成

### 7.1 活跃运行期间的边界

即使不修改旧文件，向现有 `src/`、`tools/`、`scripts/` 增加 `.py/.sh/.slurm` 也会改变 `identity.py:source_identity()`，因为它递归扫描整个树。历史日志已出现源码漂移导致停在 round guard 的事故。因此下面的实现任务必须在**独立 checkout/拷贝的项目目录**完成；不要把新文件同步进当前工作树或 HPG 活跃路径。若不能提供独立目录，就等运行结束再实施。本次只写 docs，未执行这些任务。

“仅新增文件”阶段不把方法复制成第二套未经验证的训练器，不 monkeypatch 正在运行的全局模块；先完成独立的 benchmark 边界和 fixtures，完整 RTD runner 接入明确留到 integration 阶段。

### 7.2 新文件交付顺序

| 阶段 | 只新增的建议文件 | 具体交付 / 测试门槛 |
|---|---|---|
| C26-A：档案与状态审计 | `src/bfas/rtd/benchmarks/__init__.py`（无副作用）、`alfworld_state.py`、`alfworld_caps.py`、`alfworld_bank.py`；`tools/rtd_alfworld_bank.py`；`tests/test_rtd_alfworld_bank.py` | CPU inventory/alias/成本分类；success command payload 与 raw API 区分；219/107/112 与 189,541 对账；cap provenance、missing attempt、protected groups；fixture 上 state/history/hash 校验。真实环境重放另设显式命令。 |
| C26-B：support 与封包验证 | `src/bfas/rtd/benchmarks/alfworld_support.py`；`configs/rtd/v1_alfworld_support_c26.json`；`tests/test_rtd_alfworld_support.py`、`tests/test_rtd_alfworld_state.py` | 固定 m、trial→parent、两折及 exclusions；游戏/环境内容 hashes；有 deadline 的 reset/prefix replay；逐源包对应报告及 unavailable 明细。bank 写到全新 `data/rtd/v1_alfworld_c26`，只在隔离环境/停跑后生成，不能本任务提前创建。 |
| C26-C：完整随机任务反馈 | `src/bfas/rtd/benchmarks/alfworld_rollout.py`；`tests/test_rtd_alfworld_rollout.py` | 注入 backend/env factory；从 reset 运行；raw sampled text→原 parser→env；生成 ID/EOS/截断/observation masks；失败不丢样；基础设施异常不得伪作零奖励；接通通用 LOO/REINFORCE。 |
| C26-D：官方 campaign 与身份 | `src/bfas/rtd/benchmarks/alfworld_evaluation.py`、`alfworld_identity.py`；`tools/rtd_alfworld_evaluate.py`；`tests/test_rtd_alfworld_evaluation.py`、`tests/test_rtd_alfworld_identity.py` | 140 IDs 全量校验、greedy、strict checkpoint/hash reuse、同类硬件、逐题恢复、base repair/damage、ALF metric；只使用新输出目录，不覆盖历史强 CE。 |
| C26-E：可审查集成准备 | `src/bfas/rtd/benchmarks/registry.py`、`alfworld_config.py`；`configs/rtd/v1_alfworld_c26.yaml`；`tools/rtd_alfworld_experiment.py`；`tests/test_rtd_alfworld_config.py`、`tests/test_rtd_alfworld_contract.py` | 暴露 support/rollout/caps/identity/evaluation factory；新 CLI 先支持 audit/fixture smoke，依赖未集成时明确拒绝 full run；准备下节旧文件的具体 integration diff。不得把半接通工具标为完整闭环。 |
| C26-F：停跑后的集成、smoke、完整臂 | 新增 `tests/test_rtd_alfworld_resume.py`、`scripts/rtd_alfworld_run_hpg.slurm`；同时应用下节已审查 edits | 先 CPU 数学/泄漏/身份回归，再真实环境 smoke 和 4B 单窗口 smoke；正确性通过后固定配置跑 R0/R1 三轮，含 round eval；接同 owned-set CE、固定 ledger 2×2/单组件变体。不设置 success 必须提高的 Go gate。 |

新增 YAML 至少显式列出以下差异；这不是声称当前 `cli.load_config()` 可以读它：

```yaml
benchmark: alfworld
mode: sealed_replay
experiment_scope: exploratory
student: Qwen/Qwen3.5-4B
training_seed: 0
teacher_access: text_only
new_teacher_calls: false
new_teacher_tokens: 0
support_parent_tasks_m: 135  # 在真正内容审计/排除后冻结；变更须更新全套 identity
support_manifest: configs/rtd/v1_alfworld_support_c26.json
replay_bank_path: data/rtd/v1_alfworld_c26
replay_public_cap_output_tokens_by_class: {alf_demo_episode: 1310720}
replay_cap_scope: class_uniform_public_fallback
budget_basis: usable_public_cap_sum
meta_tasks_per_feedback: 4
rollouts_per_meta_task: 2
max_action_tokens_by_benchmark:
  alfworld: {agent_action: 256}
alfworld_train_split: train
alfworld_evaluation_split: valid_seen
alfworld_expected_eval_tasks: 140
alfworld_max_episode_steps: 40
alfworld_student_react: true
evaluation_temperature: 0.0
max_context_tokens: 32768
max_state_batch_size: 1
memory_peak_budget_gb: 38
memory_reserve_gb: 2
```

其余数学协议：3 rounds、每轮12 committed steps、窗口1/4/7/10、每步8 slots、source2/temperature1/top_p1、phi0→a=.5、P/eta train-only KL pilot、epsilon=.25、positive mixture、LOO、预算10/25/50%、R0/R1 同程序、score tolerance，均应完整写入最终 YAML。去掉 BFCL 的 `meta_tasks_multi_turn/rollouts_multi_turn`、demo/generator 两类 cap、`historical_demo_output_tokens_exact:1233607` 和 `historical_generation_output_tokens_estimated:142727`，换成 ALF 的估计/缺失明细。科学配置版本/benchmark schema 明确冻结，不重写旧 BFCL run 的 `protocol_version` 或 manifests。

### 7.3 停跑后预计必须编辑的现有 RTD 文件（精确清单）

建议集成方案需要编辑 **10 个**已有模块，作用限定如下：

1. `src/bfas/rtd/cli.py`：配置/manifest/data identity/bank audit/support/checker/evaluation/report 分派；保留 BFCL 默认路径与旧配置再现。
2. `src/bfas/rtd/experiment.py`：support protocol 注入、反馈 sampler 的配置读取、action-limit 分派、generic smoke；保持已存在 phase state machine、插入/提交语义。
3. `src/bfas/rtd/return_gradient.py`：action-limit API 接受注册的 action class，避免所有非 multi-turn 默认 single-turn；通用 REINFORCE 公式不改。
4. `src/bfas/rtd/runtime.py`：backend 接受注册 cap policy；日志/模型/scoring backend 保持一致。
5. `src/bfas/rtd/evaluation.py`：ALF campaign 与 metric/report 分派；BFCL 原实现作为已冻结路径保留。
6. `src/bfas/rtd/evaluation_lock.py`：增加默认向后兼容的 benchmark/tag namespace 参数，端口排他性统一。
7. `src/bfas/rtd/identity.py`：ALF harness/data identity provider 与 guard 路由；不把新增 ALF run 送入 BFCL legacy audit。
8. `src/bfas/rtd/scoring_scope.py`：新 benchmark projection 的分派；旧 BFCL selector/evidence/hash 语义保持。
9. `src/bfas/rtd/caps.py`：让通用 affordability/audit 接收注册 class policy，BFCL 默认常量不变；ALF cap audit 实现在新模块。
10. `src/bfas/rtd/selector.py`：显式拒绝新的 privileged bank/state-reconstruction 模块导入，补漏读测试。

`bank.py`、`bfcl_decode.py`、`identity_update.py` 可保留 BFCL 专用；`features.py`、broker/ledger/transport、functional_step/insertion/acquisition 和 memory/scoring/persistence/hardware 等无需为了 benchmark 切换修改。若最后选择完全独立的新入口绕过其中某个 dispatcher，须从 integration diff 清单中删去相应编辑，并证明仍经过同一 guard/resume/report 合同；不能绕过校验来“避免改文件”。

### 7.4 有意义的测试与验收

CPU fixture 测试写入 `tmp_path`，不访问/修改现有 bank、run 或 protected data：

- 一 episode 的多个 event aliases 只扣一次；同名不同 request 不合并；失败缺 payload 为 unavailable；cannot reserve/reveal 跨折或缺依赖；cap 仅由配置生成，改变 hidden text/usage/won 不改变 public view。
- 缺 goal、错 world hash、乱序 history、teacher prefix 未拥有、两个 trial 分到不同 fold 均拒绝；prompt 相同但完整 state 不同不合并。probe/calibration/反馈折不进 inner/P/normalizer。
- 2–3 步可枚举模拟环境上核对 REINFORCE 的 task-start return gradient；有 EOS、无 EOS cap、parser fallback、后续成功、worker exception；排除 observations，禁止以已执行 command 重分词替换 sampled IDs；同任务 LOO 全0或全1时为0，混合时可识别。
- 复用 `tests/test_rtd_transport.py`、`tests/test_rtd_functional_step.py`、`tests/test_rtd_return_gradient.py`、`tests/test_rtd_checks.py`、`tests/test_rtd_acquisition.py`，检验归一化、identity、正混合、VJP/插入中心差分及 `gq=gD` 零插入值。插入有限差分及等梯度零值测试实际位于 `test_rtd_functional_step.py` 和 `test_rtd_checks.py`；无需为只改配置写镜像测试。
- 官方记录少1题、重复、错 split、错权重/config/tokenizer/step cap、缺 base class 或 artifact 损坏都拒绝 aggregate/reuse。环境 score bool 类型严格；百分比和0–1 success rate 单位不混用。
- 在 reference/selected/revealed/actual/feedback/committed 中断后恢复，不重复收费/提交，恢复同 source RNG、反馈 task/rollout、P/eta、owned、fold。重开 env 时只重演已记录动作，不重新随机抽前缀。
- 集成后运行现有 `tests/test_rtd_*.py` 及 checker 回归，包括未提交但当前已存在的 `test_rtd_evaluation_identity_reuse.py`、`test_rtd_manifest_tolerance.py`；不覆盖这些用户工作。新/旧 benchmark 都应能通过 config identity、数据隔离、hardware/resume/evaluation completeness。

真实 smoke 在停跑/隔离环境中显式执行：先至少两个 train 游戏重放成功 demo 和失败动作路径，核对 reset/prefix determinism；再 4B 一个决策窗口、每折至少两个父组，用正式 K=2 跑 LOO、score consistency、ledger、checkpoint/eval identity。smoke 只验实现与资源，不因为 reward=0 或 R1 没赢就无限增诊断。正确性通过后照规范交付完整三轮轨迹。

## 8. 主要风险及结果中必须暴露的限制

**低初始能力与稀疏信号。** greedy base 的 7.14% 不是随机 policy 的成功概率，但作数量级示例，若 p=1/14、8 条独立 rollout，则一组全零概率约 `(13/14)^8=55.3%`。LOO 还会把同一 task 的两次全1变为零 advantage；每 task 的非零概率仅 `2p(1-p)≈13.3%`，四题至少一组混合结果约 43.4%。实际任务异质性和相关性可能更差。所有回报零时 phi 的收益项为零，a=.5 的教师监督仍能 bootstrap；必须报告 identifiable feedback blocks、全零比例、gradient norm 与 gate 变化，不能换成 ΔU/teacher likelihood 奖励。

**动作长度和格式是主要混淆。** C 的 teacher 派生目标平均36.4 tokens，而旧学生249.16，多为256-token reasoning 截断。T9 120×385 的 mean transfer 98.6% 为正、rank-1 share约0.997，和 command length/teacher base likelihood 强相关；`crcd_alf3_C_rho1_s0` 仍只有14.29%，离77.86% CE很远。这表明局部格式/似然迁移不能当完整成功改进。RTD 仍使用完整序列概率和固定槽位曝光，不因长度差改成 per-token objective；同时报告两侧 tokens、cap率、ACTION率、fallback率、episode steps，全部仅作诊断。

**包覆盖和早期 bootstrap。** 107 个完整示范包最多买12个，85/385事件数不能掩盖父组覆盖不足。包内深状态必须有前缀依赖且由环境还原；当前 base 自由执行未必可达。旧日志 reachability 记 consequential85 中25可达；本机池按 `_prefix_len==0` 统计是21/85（A为102/385），不能直接把日志的25解释成当前文件的零深度数。需重建 event key/定义再解释差异，不能把 reachability 作为购买前过滤。r2 evidence 还携带旧 CE 训练依赖，默认只作诊断，不能免费给新 R0/R1。

**请求/成本完整性。** 可见命令、字符估计和真实 API 账单是三件不同的东西；成功缓存偏差、缺失败响应、缺一次 attempt、缺输入/reasoning/重试会限制端到端效率结论。输出“可重放历史命令包上的估计支出曲线”可以；输出“精确成本的在线 few-shot acquisition”不成立。

**评测与资源公平。** 历史 strong CE 更强且剂量/目标/事件筛选不同，必须正面对比，不能只展示 R1-R0；同一合法 owned 集合的强 CE 才是采购/蒸馏比较。valid_seen 的开发复用、valid_unseen 的历史触碰、父组相关性、单 seed、不同 GPU class 必须分别说明。没有独立认证集时只交付开发端点和其明确的统计范围。

## 9. 审计复现线索

关键输入 SHA-256（仅 hash 原始文件，不改动）：

| 文件 | SHA-256 |
|---|---|
| `data/teacher_ledger/alfworld.jsonl` | `b296dda679aeb994c1cdd0c15f9b1d3b62b080016845aa43a0ca2823ffea7db5` |
| `data/alf_sft/pool_A_all.jsonl` | `e6c41336fb6335feba710646214c1c73d16f9ab6d68894c763e2c330a4d3f3f0` |
| `data/alf_sft/pool_C_conseq.jsonl` | `79c4dd5cc8b6a6d799d629ed5baee18c5389b773a5e5d6ccac30700590aeeb97` |
| `data/alf_sft/events_r2ce.jsonl` | `a892a9481b7995d607f9bb5db3af83b4c1810dbae3b26fa10931ef518a3be0db` |

历史解释参考 `docs/2026-09-04-alf3-mirror-prep.md`，以及 `notes/exp_log.md` 的 ALF teacher goal-fix、alf_events_v1、CE ladder、valid_seen、crcd_alf3、T9 记录。goal 修复前的 0/20 teacher 及旧2B裸prompt产物被历史日志声明作废，不能复活入 bank。`tools/alf_reachability.py`、`alf_event_analyze.py`、`alf_event_reestimate.py`、`alf_events_to_pools.py` 和各 `alf_*lane/waiter` 是历史采样/分析/调度路径，本文仅读取，不执行。
