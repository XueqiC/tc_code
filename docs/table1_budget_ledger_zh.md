# Table 1 统一预算账本与封存池审计（2026-09-08 冻结协议）

已生成 CPU 审计、随机采购与七臂池。现有记录只能支持明确标注的缓存内容回放，不能证明完整历史 API 成本已经恢复，也不能据此认证正在运行的 V0/V1/V2。dDPO 缺排名调用成本，PBSD-agent 缺消费 c 的训练入口，ALFWorld/AppWorld 缺完整同池 BB-OPD 状态对应；manifest 保留这些状态，不能把退化池当成完成的基线。

## 1. 边界与计费口径

- 当前 native 格式工作仅在 `/home/xueqi/hq/projects/tc-alignment-audit2`、`table1-native` 分支进行；不提交，不调用 GPU、环境或 teacher API。`data/`、`envs/`、`.venv` 是共享符号链接。
- `data/` 只读与写 `data/table1_pools/` 的要求冲突，可审阅池改放当前 worktree 的 **results/table1_audit/pools/**。工具拒绝写 data、envs、虚拟环境、控制目录和其他 worktree。
- 本分支缺少指定的 `docs/2026-09-08-table1-audit-and-plan-zh.md`、`src/bfas/rtd/bank_v11.py`、`tools/rtd_v11_build_bank.py`。实际读取方法 §2、旧 bank 实现、v1.1 封存银行、cap certificate、账本与本地配置；未去其他 worktree 取代码。
- 冻结定义：`C_m = Σ c(q)`，用于取得证据的教师调用只付一次，失败、重试、弃用输出也收费。切段、重复 epoch、c 中重复引用不再计费。未知费用不能用 0 或学生答案 token_hint 代替。
- **sealed replay spend** 是本次已购包记录成本；**historical pool-building** 是归档建池记录总额，单列且不与已购成本相加。`complete=false` 表示仍缺完整 provider 账单；estimated 数字不构成数学下界保证。
- 库存审计是获授权的离线特权阶段，读取全库内容/验证信息以生成包与对应关系，不能供选择器使用。采购只读冻结 public.json 的 ID、成本与预算参数；订单冻结后才揭示已购包。journal 的“未读未购内容”适用于采购和池构建，不伪称审计本身没有读全库。
- 公开逐包记录成本是严格前缀停止所需的信息例外；原 RTD 只公开类别 cap、真实记录成本封存。该差异可能改变采购排序和停止。
- 三次训练共享 acquisition seed 0：先按包 ID 排序，再用 `random.Random(0).shuffle`；训练 seed 0/1/2 仅改变训练。下一包超预算立即停止，不跳过、不用余额挑便宜包。
- manifest 指定 `Qwen/Qwen3.5-4B`、`adapter=null`。这是后续训练要求；本次没有加载模型，不声称已核实运行中 checkpoint。

## 2. 库存与历史成本

| benchmark | 候选包 | bank 可用包 | 可训练正行 | 候选成本 | 可用内容成本 | 失败成本 | 历史已知建池总额 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| bfcl | 420 | 420 | 371 | 55370 | 55370 | 8737 | 1376334 |
| alfworld | 219 | 107 | 1377 | 189541 | 36294 | 153247 | 189541 |
| appworld | 88 | 78 | 1482 | 90533 | 83552 | 6981 | 90533 |

BFCL：698 个封存归档包中 420 个可采购（84 demo attempts + 336 generator items）。55,370 = 21,203 exact + 34,167 estimated；15 次 demo 失败成本 6,140，另有 34 个 generator 条目的独立求解验证失败、成本 2,597。共 49 个失败包、8,737，采购支付但不产正例。69 个成功 demo 加 302 个通过归档验证的 generator 条目，共 371 正行。generator 验证读取原始 verified/reproduced（243 个 reproduced=true、59 个 verified=true、34 个 verified=false），不将 bank 可用性误当成功。34 个旧 verified task ID 是合并库任务级信息；demo 逐尝试结果来自 a1/a2/a3 对应官方 score 文件，先核对分类总数/错误数，再取补集。

BFCL 历史建池已知总额 1,376,334 = 1,233,607 exact demos + 142,727 estimated generation。generator 成本来自归档文件估计，按作者字段字节分摊；item 无法恢复为独立 provider call ID，也缺批次、失败草稿/修复/验证费用。因此是内容回放代理成本，不能声称满足完整逐调用 C_m。另一个 `teacher_ledger/bfcl.jsonl` 有 171 行、16,748，属于另一批采集；相同 task_id/attempt_index 不证明与 a1/a2/a3 是同一调用，不混入银行总额。

ALFWorld：保留 GPT-5.4；219 次 episode attempts = 107 可用 + 112 失败，189,541 = 36,294 可用 + 153,247 失败。B=9,074 保留用户暂定绝对值。每 attempt 成本是多个历史续写的聚合估计，原始逐 API 子调用与失败轨迹缺失；标注 **estimated-budget replay**。失败包进入随机序列，不能从“可用 107 包”开始采购再宣称统一失败计费。

AppWorld：冻结现有 78 个 DeepSeek-V4-pro 成功 episode / 1,482 行（原混合池 128 episodes / 2,506 行）。读取 `appworld_traces/deepseek-v4-pro/train.jsonl` 与 `train_debug.jsonl`：两条 debug 成功轨迹与 train 完全相同，去重；另有 10 次失败、6,981 估算 tokens。冻结候选为 **88 attempts**（成功内容仍是原 78 episodes），成本 **90,533 = 83,552 + 6,981**；提议 **B=22,633**（25%，正整数 half-up）。只冻结 78 个成功包的 25% 为 20,888，但会排除已知失败尝试，本工具不采用该幸存者池。

AppWorld 每个保留 assistant turn 建立带来源行号/turn_index 的合成 call ID，按 `max(len(content)//4,1)` 估计；episode 一次采购、包括其所有保留输出，多个训练行不重复收费。这仍非 provider usage：无 debug 的失败、底层重试/丢弃输出不可恢复；metrics 是一次运行摘要，不能用来杜撰缺失调用。`teacher_ledger/appworld.jsonl` 的 42 行是 GPT-5.4（4,791,617），与 DeepSeek 池不混用。

**包边界限制：**本次保留 BFCL item、ALFWorld/AppWorld episode attempt 的现有银行/采集边界。若严格要求每包等于一次 API 续写而非 episode acquisition bundle，ALFWorld 聚合记录不足以实现；AppWorld 需另冻结带历史前缀依赖的逐调用协议。所有完整 API 账单缺项都保留为 unknown，不编造调用。

各 ledger.json 给出全部包、teacher_calls、来源、attempt_index、verified、成本 confidence、失败归因及未入候选的历史包。failure_cost_attribution 给出每任务全体记录与失败小计；已购失败归其原任务，不将整任务历史失败再次加到成功包。

## 3. 采购结果

正例列是训练行数；BFCL tasks 指官方父任务覆盖数，生成任务另有 legacy_task_id；其余是任务 ID 数。三训练种子订单相同。

| benchmark | B | 训练 seeds | 已购包 | tasks | 正行 | C_m | 余额 | 失败尝试 | 已购失败成本 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bfcl | 13843 | 0/1/2（共享顺序） | 103 | 22 | 94 | 13776 | 67 | 9 | 2075 |
| bfcl | 5537 | 0/1/2（共享顺序） | 35 | 15 | 30 | 5471 | 66 | 5 | 1797 |
| alfworld | 9074 | 0/1/2（共享顺序） | 8 | 8 | 47 | 9013 | 61 | 5 | 7280 |
| appworld | 22633 | 0/1/2（共享顺序） | 22 | 21 | 368 | 22107 | 526 | 3 | 3236 |

acquired_B<cap>_seed<k>.json 保存完整冻结顺序、逐步累计成本、下一个超预算包/成本与精确余额。任务、正例统计仅在采购完成后写入，不参与选择。

## 4. 旧池对应与缓存排名

合法对应要求 task_id、teacher、turn_index、messages、prompt、response 一致，额外字段逐项登记。内容相等不证明两个采集文件共享同一历史 provider 调用。BFCL 另按封存原始 result 的序列化核实；仅归档响应对应、没有合法训练行时单列 archived_response_package_ids，不允许复用。每条旧行都有物理行号、完整行 hash、对应 package_ids、逐预算 purchased_sets。最后一列是缺少合法训练行对应；详细账本进一步区分“没有封存响应对应”与“响应对应但训练不可用”。

| benchmark | 旧池文件 | 总行 | 合法内容对应 | rank 行 | 其他不可复用行 |
| --- | --- | --- | --- | --- | --- |
| bfcl | pool_bfcl_ds_sft.jsonl | 23 | 23 | 0 | 0 |
| bfcl | pool_bfcl_ds_ddpo.jsonl | 30 | 0 | 5 | 25 |
| bfcl | pool_bfcl_ds_pbsd.jsonl | 25 | 0 | 0 | 25 |
| bfcl | pool_bfcl_ds_bbopd.jsonl | 25 | 0 | 0 | 25 |
| bfcl | pool_bfcl_star.jsonl | 88 | 0 | 0 | 88 |
| appworld | pool.jsonl | 2506 | 1482 | 0 | 1024 |
| appworld | pool_agentkd.jsonl | 2506 | 1482 | 0 | 1024 |
| appworld | pool_bbopd.jsonl | 551 | 28 | 0 | 523 |
| appworld | pool_ddpo.jsonl | 2537 | 1482 | 31 | 1024 |
| appworld | pool_pbsd.jsonl | 2506 | 1482 | 0 | 1024 |
| appworld | pool_selfmix.jsonl | 2017 | 1033 | 0 | 984 |
| appworld | pool_selfmix_9b.jsonl | 2008 | 1002 | 0 | 1006 |
| appworld | pool_selfmix_cum.jsonl | 1934 | 942 | 0 | 992 |
| appworld | pool_selfmix_pref.jsonl | 2017 | 1033 | 0 | 984 |
| appworld | pool_selfmix_r2.jsonl | 1915 | 942 | 0 | 973 |
| appworld | pool_selfmix_v2pref.jsonl | 2021 | 1033 | 0 | 988 |
| appworld | pool_selfmix_x4.jsonl | 2074 | 1033 | 0 | 1041 |
| appworld | pool_star.jsonl | 19 | 0 | 0 | 19 |
| appworld | pool_v21_r1.jsonl | 400 | 0 | 0 | 400 |
| appworld | pool_v31_best.jsonl | 272 | 0 | 0 | 272 |
| appworld | pool_v32_adv.jsonl | 689 | 0 | 0 | 689 |
| appworld | pool_v3_r1.jsonl | 673 | 0 | 0 | 673 |

旧 BFCL SFT 的 23 行均可核实；旧 dDPO/PBSD/BB-OPD 的 25 条 demo 与当前 23 条 SFT 任务无交集，不能直接作为同池全量基线。BFCL 5 条 rank 所指任务不在已购任务集。AppWorld 混合教师行不能因 task_id 相同映射为 DeepSeek。

排名调用是额外教师证据。bfcl_pair_pools.py、aw_ddpo_rank.py 保存的是教师选中的学生回答；token_hint 是该学生回答长度，不是教师返回排名索引的开销。账本没有可关联的排名 call ID/费用。下表“任务合格”只是必要条件，仍因缺费用而不能复用；其余按要求标记 **requires new teacher calls**。本次无新调用，pool_ddpo 是明确标注的 SFT fallback，不是完成的 dDPO 基线。

| benchmark | B | 缓存 rank | 已购任务合格 | 合格但成本缺失 | requires new teacher calls | 实际复用 |
| --- | --- | --- | --- | --- | --- | --- |
| bfcl | 13843 | 5 | 0 | 0 | 5 | 0 |
| bfcl | 5537 | 5 | 0 | 0 | 5 | 0 |
| alfworld | 9074 | 0 | 0 | 0 | 0 | 0 |
| appworld | 22633 | 31 | 7 | 7 | 24 | 0 |

## 池行格式：native-fc v2

**BFCL 默认格式为 `native-fc-v2`**；原 CLI 拼写 `--row-format native-fc` 现在是 v2 别名。`tools/table1_pool_from_sealed.py --benchmark bfcl` 写当前 worktree 的 `results/table1_audit/pools_native/bfcl/`，允许将该目录原 `native-fc` 升级为 v2。legacy 目录仍为 `results/table1_audit/pools/`，不能被 native 覆盖；ALFWorld/AppWorld 默认仍为 legacy。本次只在 `/home/xueqi/hq/projects/tc-alignment-audit2`、`table1-native` 分支写入，不提交、不调用 GPU/API；`data/`、`envs/` 只读。

### 真实评测证据与修复原因

用户提供的新运行证据：native v1 学生 Overall=16.7，典型输出是 `<think>\n\n` 后 EOS。v1 的 prompt 已包含 `<think>\n\n</think>\n\n`，response 只有 continuation，导致关闭 think block 和紧接的 tool-call 边界不作为完整目标接受监督。真实评测输入却止于 `<|im_start|>assistant\n`，模型必须自行生成这个 block。本次仅做 CPU 格式/训练边界验证，没有重新训练或声称分数已恢复。

只读复制自 `/home/xueqi/hq/projects/tc-alignment-table1/results/table1_audit/bfcl/` 的 `ddpo_samples_B13843.jsonl` 包含 **88 个学生样本、22 个不同任务、每题四次相同 prompt**；`ddpo_rank_ledger.jsonl` 包含 31 次已发生的排名调用。副本位于本 worktree 同名相对路径，源路径和逐文件 SHA-256 记录在 `results/table1_audit/native_fc_v2_evidence_sources.json`。没有修改源 worktree，也没有重新采样或购买排名。

### 精确 prompt、target 与 RTD 的区别

- **`messages=[]`**。`bfcl_evaluation_prompt()` 在输入副本上执行 BFCL 官方 `add_language_specific_hint_to_function_doc()` → `_func_doc_language_specific_pre_processing()`，再调用 `QwenFCHandler._pre_query_processing_prompting()`、`add_first_turn_message_prompting()`、`_format_prompt()`。用 `__new__` 避开构造 API client；只运行纯预处理/模板函数。
- **prompt = 官方语言预处理后的 handler 模板，止于 `<|im_start|>assistant\n`**，不追加 think suffix。Python 描述追加 ` Note that the provided function is in Python 3 syntax.`；Java/JavaScript 使用各自官方提示，并执行参数描述、类型和 items/properties 的相应转换。
- **response = `<think>\n\n</think>\n\n` + sealed `behavior.text`**。teacher continuation 原字节、调用顺序、嵌套参数和自然语言均保留；不重选 ground_truth 候选、不编造拒答、不附加手工 EOS。teacher/tool continuation 仍是 `<tool_call>\n{"name": ..., "arguments": ...}\n</tool_call>` 原生输出形式。
- `bfcl_native_prompt()` 仍先用 RTD 原来的 `thinking_off(handler._format_prompt(...))` 验证 sealed prompt。`bfcl_inputs()` 从 generator 对象或 sealed `<tools>` JSON 恢复原始工具键序，再与排序过的 `task_json` 做对象核验；不能直接将 task_json 重序列化后当作部署字节。`read_bank_payload()` 仅揭示已购 package，并核对冻结源 SHA-256；失败包仍计费但不产正例。task_id、teacher、turn_index、token_hint 原值不变。

**22/22 记录 prompt 全部逐字节相等，Java 无剩余差异。** 其中 14 个不同官方任务也出现在已购正行中（重复 demo 共 22 行）。对旧 RTD prompt，仅去掉函数描述语言 note 再补 think suffix，13/14 任务相等。唯一额外差异是 `simple_java_6`：`refreshMetadata`、`append`、`keepState` 从 `boolean` 变为 `string`，每个参数追加 ` This is Java boolean type parameter in string representation.`。v2 执行完整预处理后该例也完全相等。另一个 recorded Java 任务 `simple_java_4` 的 `any` 参数同样按官方逻辑变为 string；它没有已购 SFT 正行，但独立官方输入测试覆盖其记录 prompt。

可用于论文的精确定义：**“Native-fc v2 uses the evaluation handler's language-preprocessed function documentation and assistant-start prompt. Relative to RTD's sealed rendering, it moves the empty think block from the masked prompt to the supervised response and adds language-specific documentation/schema preprocessing; the teacher continuation is unchanged.”** RTD `bank.full_state()` / `return_gradient.bfcl_task_rollout()` 的原有 `thinking_off` 边界与本次真实 BFCL CLI 评测不同，不能再称 v2 与 RTD 的 prompt/response 切分完全相同。

用户说明记录输出已经过 vLLM reasoning parser，看不到模型生成的空 think block。vendored Qwen handler 的 `_parse_query_response_prompting()` 也会在 `</think>` 处分割，仅将其后的内容写入 `model_responses`，reasoning 另存；`decode_ast()` 提取 `<tool_call>`。因此保存的 continuation 不是完整的模型发射序列。测试以完整 block + 工具调用/自然语言/空 continuation 调用该 parser，核对清理结果。

### 空调用与训练 labels

B13843 每 seed 有 **13** 条、B5537 有 **5** 条成功 generator 的 sealed continuation 为空且 `ground_truth=[]`。保留 **`_native_fc_empty_response=true` 的语义：教师 continuation 为空**。**v2 的 response 并非空字符串，而是 think prefix 单独组成的目标 `<think>\n\n</think>\n\n`；encode 随后追加一个 EOS。** 这些行也必须学习 `</think>`，不再是 v1 的 EOS-only 目标。`load_pool()` 验证显式标记与 v2 的空 messages、assistant-start prompt、prefix-only response 一致；保留 v1 已标注空目标的加载兼容，普通空 response 仍拒绝。

真实 `load_pool()` → `prompt_token_ids()` → `encode(..., device='cpu')` 使用本地 **Qwen/Qwen3.5-4B tokenizer，`local_files_only=True`**。messages 为空时不 re-template，prompt/response 各自以 `add_special_tokens=False` 编码。prompt 按 4096/tail 截断，response 按 `MAX_RESPONSE_TOKENS=512` 截断，再追加一次 EOS。labels 精确等于 `[-100] * len(prompt_ids) + response_ids + [EOS]`；**完整 think prefix 的 token，包括 `</think>` 和后续换行，都属于 labels，而非 prompt mask**。另覆盖 32-token prompt 截断。测试不加载模型权重、禁止 socket connect 和 CUDA 初始化；只验证训练输入和损失边界，不宣称所有 trainer 截断配置与 RTD 相同。

### 七臂与排名回放

SAD 的 `_sad_spans` 在 v2 response 上重新按 trainer 的代码围栏规则计算，字符偏移包含新增 prefix；不改变 SAD 的损失定义。PBSD-insp `_rejected` 将旧 decoded FC 列表转为原生调用，并补模型 think prefix；已有完整 think 发射保留，尾部 EOS 去掉交给 encode 添加。当前已购 BFCL 缓存仍为 **0 对**，测试用 fixture 验证转换、嵌套参数、失败文本和 EOS，不修复失败调用。

PBSD-agent 的 `c.state_prompt` **逐条等于所属 row.prompt**，`c.response` 同样包含 think prefix。按 `(task_id, prompt)` 分组防止复用 task_id 的不同生成问题串用证据。`c` 仍缺训练消费者，STaR teacher pool 仍为空。

若本地存在 `bfcl/ddpo_rank_ledger.jsonl`，默认重建器自动执行纯 CPU `replay_native_rankings()`，复用 `table1_ddpo_rank.render_ranked_pools()` 的费用/状态计算，不调用 `sample`、`rank`、client 或服务。每个排名候选及其顺序、上下文、best/worst 与 preference_row 均与原始 samples/ledger 核对；B5537 只使用自身已购父任务，可从 B13843 样本文件读取这些任务。

**排名行 response/chosen 与 `_rejected` 都是 think prefix + 记录的学生样本原字节。必须补 prefix**：样本虽称 raw output，但已经过 parser，偏好损失应覆盖模型自行发射的序列，边界必须与 SFT 一致。排名行不调用 decoded-list 转换，不修复 malformed JSON，不替换成 teacher 答案。原始 samples/ledger 不变；pool 中的 `messages=[]`、prompt 为记录的官方 prompt。rank 行 token_hint 保留原记录，仅为训练提示而非费用。

本次 B13843 的 31 个排名 attempt 中，16 个成功、15 个 parse_failed；全部计费 **1,083 tokens**，不能只算最后成功回复。B5537 的任务子集包括 10 个成功、10 个 parse_failed，共 **680 tokens**。三个 seeds 共享相同 call IDs；较小 cap 的回放是同一批证据的子集，不是再次购买。新的 dDPO `C_m` 分别为 **14,859 / 6,151**，超 cap **1,016 / 614**，manifest 明示 `exceeds_cap=true`。`ready` 仅表示已购任务的排名证据可用，不表示满足原总费用 cap。

### 输出与复现

六个目录的逐池行数如下；seed 0、1、2 各有一份，表内不合并训练行。

| cap | seed | sft | sad | bbopd | pbsd_insp | ddpo | star | pbsd_agent |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 13843 | 0 | 94 | 94 | 94 | 94 | 110 | 0 | 94 |
| 13843 | 1 | 94 | 94 | 94 | 94 | 110 | 0 | 94 |
| 13843 | 2 | 94 | 94 | 94 | 94 | 110 | 0 | 94 |
| 5537 | 0 | 30 | 30 | 30 | 30 | 40 | 0 | 30 |
| 5537 | 1 | 30 | 30 | 30 | 30 | 40 | 0 | 30 |
| 5537 | 2 | 30 | 30 | 30 | 30 | 40 | 0 | 30 |

共 **42 个池、36 个非空池、2,310 行**，比 v1 多 78 条跨 seed 排名行。六个 manifests 为 `row_format='native-fc-v2'`。SFT/SAD/BB-OPD/PBSD-insp/PBSD-agent 的 C_m 仍分别为 B13843=13,776、B5537=5,471；STaR=0。除新增排名证据、费用及其 provenance 外，采购 IDs、顺序、种子、失败计费、source row hashes、acquisition 文件和 legacy 池均保持原值。manifest 为排名 ledger 和 sample 文件记录 SHA-256；重建可重复运行，已存在但不兼容的会计信息仍拒绝覆盖。

```bash
export PYTHONPATH=src:.
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=''
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
PY=.venv/bin/python
$PY tools/table1_pool_from_sealed.py --benchmark bfcl
$PY -m pytest -q -p no:cacheprovider tests/test_table1_native_fc.py tests/test_table1_row_format.py tests/test_table1_audit.py tests/test_appworld_train_truncation.py tests/test_table1_ddpo_rank.py --junitxml=results/table1_audit/native_fc_v2_cpu_tests.xml
```

v2 本次 CPU 回归 **209 passed，0 failures/errors/skipped，177.22 秒**，见 `results/table1_audit/native_fc_v2_cpu_tests.xml`；逐池计数、hash、费用、输入保护与 22 prompt 对照见 `native_fc_v2_validation.json`。默认重建再次运行后，42 个池与六个 manifests 全部逐字节不变。旧 `native_fc_cpu_tests.xml` / `native_fc_validation.json` 是 v1 历史记录，不作为 v2 验证结论。

## 5. 七臂池与训练入口

下表保留 legacy/排名补采前快照；当前 BFCL native-fc v2 的行数、dDPO 费用和状态以上节六目录表及 manifests 为准。

| benchmark | B | arm | rows | C_m | PBSD pairs | 状态 |
| --- | --- | --- | --- | --- | --- | --- |
| bfcl | 13843 | bbopd | 94 | 13776 | 0 | ready |
| bfcl | 13843 | ddpo | 94 | 13776 | 0 | sft_fallback_missing_rank_costs |
| bfcl | 13843 | pbsd_agent | 94 | 13776 | 0 | evidence_only_consumer_missing |
| bfcl | 13843 | pbsd_insp | 94 | 13776 | 0 | ready |
| bfcl | 13843 | sad | 94 | 13776 | 0 | ready |
| bfcl | 13843 | sft | 94 | 13776 | 0 | ready |
| bfcl | 13843 | star | 0 | 0 | 0 | no_teacher_pool_required |
| bfcl | 5537 | bbopd | 30 | 5471 | 0 | ready |
| bfcl | 5537 | ddpo | 30 | 5471 | 0 | sft_fallback_missing_rank_costs |
| bfcl | 5537 | pbsd_agent | 30 | 5471 | 0 | evidence_only_consumer_missing |
| bfcl | 5537 | pbsd_insp | 30 | 5471 | 0 | ready |
| bfcl | 5537 | sad | 30 | 5471 | 0 | ready |
| bfcl | 5537 | sft | 30 | 5471 | 0 | ready |
| bfcl | 5537 | star | 0 | 0 | 0 | no_teacher_pool_required |
| alfworld | 9074 | bbopd | 0 | 9013 | 0 | requires_new_teacher_calls |
| alfworld | 9074 | ddpo | 47 | 9013 | 0 | sft_fallback_missing_rank_costs |
| alfworld | 9074 | pbsd_agent | 47 | 9013 | 0 | evidence_only_consumer_missing |
| alfworld | 9074 | pbsd_insp | 47 | 9013 | 0 | ready |
| alfworld | 9074 | sad | 47 | 9013 | 0 | ready |
| alfworld | 9074 | sft | 47 | 9013 | 0 | ready |
| alfworld | 9074 | star | 0 | 0 | 0 | no_teacher_pool_required |
| appworld | 22633 | bbopd | 5 | 22107 | 0 | partial_context_matches |
| appworld | 22633 | ddpo | 368 | 22107 | 0 | sft_fallback_missing_rank_costs |
| appworld | 22633 | pbsd_agent | 368 | 22107 | 0 | evidence_only_consumer_missing |
| appworld | 22633 | pbsd_insp | 368 | 22107 | 13 | ready |
| appworld | 22633 | sad | 368 | 22107 | 0 | ready |
| appworld | 22633 | sft | 368 | 22107 | 0 | ready |
| appworld | 22633 | star | 0 | 0 | 0 | no_teacher_pool_required |

- **SFT：**只输出已购非失败内容；BFCL 默认使用上节 `native-fc-v2`，空调用监督完整 think prefix 后接 EOS；其他 benchmark 仍用 legacy。旧 BFCL legacy 池独立保留，失败包不产正例，教师内容和费用不变。封存 package.response_rendering 描述的是修复前源行，当前训练格式以 manifest.row_format 为准。
- **SAD：**同 SFT prompt/response，加 `_sad_spans` 字符边界，按现有 action_spans 的代码围栏规则；训练器仍自行算 mask。BFCL 无代码围栏响应按现有 trainer 规则属于非 action 部分，本次未另造 BFCL mask。
- **BB-OPD：**单轮 BFCL 与 SFT 逐行相同。AppWorld 只取旧 on-policy 上下文/响应与已购 demo 完全一致的行，是部分内容对应，不代表完整 BB-OPD 或证明当前学生访问这些状态。ALFWorld 无对应缓存，空池。补齐需另冻结 on-policy 状态与费用。
- **PBSD-insp：**只对已购行附加旧缓存的学生失败首轮 `_rejected`，其他正例训练；不做新采样。BFCL 本次 0 对，AppWorld 本次 13 对。
- **dDPO：**原未知成本缓存仍不复用；BFCL v2 使用只读复制的已付费 ledger，追加学生 rank 行并计入全部尝试费用，详见上节。其他池保留原状态。
- **STaR：**teacher pool 为空、C_m=0、调用/证据 ID 列表为空，不使用公共基线采购的教师证据。不能将空池交给拒绝空输入的 SFT loader；学生自训练另行进行。
- **PBSD-agent：**保留 trainer 基础字段，加 c，只引用相同实际任务的已购 demos（不将 generator 父任务当作生成任务本身）。当前 appworld_train 没有消费 c 的入口；agentkd 的 `_thought` 不是此算法。标为 evidence_only_consumer_missing，不伪称已经可训练。

标准字段为 task_id、teacher、turn_index、messages、prompt、response、token_hint。manifest 给出各臂 call ID 集合、C_m、历史总额、hash、基座、旧行字段差异。除 STaR 外，C_m 包括全部已购包，即使该臂最终未使用某包。

对已采购的非空训练池，使用 `AW_POOL_PATH` 和 **--selection full**。src/appworld_train.py 的 --budget 按学生 tokenizer 计算被选训练行 response tokens；不能代替教师调用费用，也不应再按行重做预算选择。此次只检查输入格式，不执行训练。

### 5.1 BFCL dDPO 排名补采工具（历史接口说明）

`tools/table1_ddpo_rank.py` 的 sample/rank 子命令及下列 GPU/API 命令是历史采集接口说明，本次没有运行。已有真实记录从 table1 worktree 只读复制后，由 v2 默认池重建器回放到 `pools_native`；以下 rank 子命令本身仍写 legacy `pools`，不承担 v2 target 转换。5 条未知成本旧缓存 rank 不复用。

在当前 worktree、`table1-audit` 分支运行：

```bash
export PYTHONPATH=src:.
export PYTHONDONTWRITEBYTECODE=1
PY=.venv/bin/python
$PY tools/table1_ddpo_rank.py sample --cap 13843 --repeats 4 --gpu <uuid> --port auto
$PY tools/table1_ddpo_rank.py rank --cap 13843 --key-file ~/.ollama_api_key --max-calls 40 --max-output-tokens 64
```

`sample` 使用三个 manifest 共同的 `tasks_covered`（22 个官方父任务），并核验三份 SFT 内容 hash、采购 ID、初始 checkpoint 和费用一致。94 行中包含 `gen_` / `oos_` 生成任务；它们不是额外要采样的官方 task ID。初始模型固定为 `Qwen/Qwen3.5-4B`、无 adapter，temperature=1.0，默认每题 4 次，共 88 行。只调用本地 GPU 服务，不加载教师 key、不调用教师。模型/tokenizer 从已有本地缓存加载，离线模式禁止自动下载。

旧 K 次 unguided 采样入口是 `tools/bfcl_roll_pass.sh base <gpu> <port> 4`（旧 temperature=0.7）；`bfcl_teacher_demos.sh` 是教师示范采集，`bfcl_demo_pool.py` 和 `bfcl_gen_rows.py` 是行构建。本工具复用旧入口的 vLLM + 官方 BFCL CLI 路径，改为 temperature=1.0，并用 `BFCL_PROJECT_ROOT` 隔离写目录。实际命令如下（`RUN=results/table1_audit/bfcl/ddpo_sample_B13843`，运行时展开为绝对路径；`PORT` 为自动选择的空闲端口，`r=0..3`）：

```bash
CUDA_VISIBLE_DEVICES=<uuid> envs/vllm-serve/.venv/bin/vllm serve Qwen/Qwen3.5-4B \
  --served-model-name Qwen/Qwen3.5-4B --host 127.0.0.1 --port "$PORT" \
  --gpu-memory-utilization 0.85 --max-model-len 32768
BFCL_PROJECT_ROOT="$RUN" LOCAL_SERVER_ENDPOINT=127.0.0.1 LOCAL_SERVER_PORT="$PORT" \
  envs/bfcl/.venv/bin/bfcl generate --model Qwen/Qwen3.5-4B-FC --run-ids \
  --skip-server-setup --temperature 1.0 --num-threads 4 --include-input-log \
  --result-dir "$RUN/result_r$r"
BFCL_PROJECT_ROOT="$RUN" envs/bfcl/.venv/bin/bfcl evaluate --model Qwen/Qwen3.5-4B-FC \
  --result-dir "$RUN/result_r$r" --score-dir "$RUN/score_r$r" --partial-eval
```

用户只需执行上面的 `sample` 子命令；服务启动、端口、清理与子进程环境均由它管理。入口 Python 为 `$PY`，BFCL/vLLM 子命令使用旧脚本对应的专用环境。完整展开命令保存在 `RUN/run.json` 并输出到终端。selection、result、score、锁、日志和运行缓存都放在 `results/table1_audit`，不改共享 `data/`、`envs/` 或旧 selection 文件，不加载共享 `.env`。每次 repeat 先检查全部结果，再用现有 `extract_verdicts` 核对官方 score 总数和失败补集；缺失 checker 文件不被当作通过。最终 `bfcl/ddpo_samples_B13843.jsonl` 包含 `task_id, sample_index`（0-based）、`response, verified`，另存官方 handler 的 `prompt, messages` 供排名偏好行复现上下文。重复运行保留已完成 repeat 和官方部分结果。

`rank` 对每个有至少两个不同 response 的已购父任务调用一次 `deepseek-v4-pro`，强制 think=false、temperature=0，使用旧 `RANK_PROMPT` / `last_two_numbers`，展示全部去重候选及完整官方 prompt，不按 checker 正误预筛选。不把生成子任务的 prompt 当作父任务上下文，也不把 SFT 教师答案当候选。成功排名后以学生 best/worst 组成旧格式的 `teacher='rank'` 行：`task_id, teacher, turn_index, messages, prompt, response, _rejected, token_hint`；`token_hint` 仍只是学生答案的训练提示长度，不用于排名费用。

教师客户端原先只返回文本，Ollama usage 被丢弃。本次为 `appworld_teacher.generate_reply` 添加可选 `usage_out`，在解析 content 前暴露 provider usage；原调用者仍获得字符串。排名优先按 `completion_tokens` / `output_tokens` 记 `tokens_spent`、`confidence='exact'`，不使用输入或 total tokens。若没有 usage，用本地 Qwen tokenizer 对教师回复计数并标 `estimated`；空回复但 usage 缺失至少记 1，不能推断免费。索引解析失败不产偏好行，仍将调用、原回复和费用写入账本，且不再次请求该任务；只有 provider 明确报告 output=0 时才可零收费跳过。没有返回文本也没有 usage 的传输/配额错误记 `tokens_spent=null, confidence='unknown'`，manifest 标 `incomplete_ranking_costs`，保留未知 call ID，不能假称已精确结算。fallback tokenizer 失败时也保留回复并标未知费用。

新账本为 `results/table1_audit/bfcl/ddpo_rank_ledger.jsonl`，保留原 BFCL ledger 的 `task_id, teacher, attempt_index, temperature, verified, tokens_spent, purpose='rank', timestamp`，补充 `call_id`、provider usage、候选、回复、解析状态和 preference row。`verified=false` 表示排名调用不是通过官方 checker 的教师示范；学生样本的官方 verdict 在 samples 文件中。三个训练 seed 共享同一次排名及 call ID，不三次付费。

`--max-calls` 是该已购任务集**跨恢复运行累计的 HTTP attempt 上限**，包含失败和 429 重试；调大此参数可继续尚未处理的任务。关闭客户端隐藏重试及备用 key 轮换；只使用指定 key-file，`OLLAMA_BASE_URL` 如未设置默认 `https://ollama.com`。每次运行遇到 429 至多重试两次（该任务本轮总计三次，间隔 1/2 秒），耗尽后终止整个运行；配额恢复后可重跑，之前的尝试仍计入累计 max-calls。其他 API 错误立即终止，若该任务曾发生非 429 API 错误，需先核对并结算该记录再继续。已返回排名（包括解析失败）不重呼叫。文件锁防止两个进程同时购买；每次请求前 fsync 写 `ddpo_rank_attempts.jsonl`，回复结算后 fsync 追加 ledger。若进程在请求与记账之间崩溃，下一次会拒绝自动重试、列出未结算 call ID，需根据真实 provider 记录补齐 ledger 后恢复；不能删除意图日志来把潜在收费调用变成免费。

每次正常停止（含 max-calls/配额停止）均重建三个 seed 的 `pool_ddpo.jsonl = 原审计 pool_sft + 每个成功排名任务一行`，同时更新 `arms.ddpo` 的 `ranking_call_ids, teacher_call_ids, pool_sha256, rows, ranking_cost, C_m`。完成且至少有一个偏好行时 `status` / `trainer_status='ready'`；未处理完为 `partial_ranking`，没有可用偏好为 `no_rankable_preferences`，未知费用为 `incomplete_ranking_costs`。manifest 同时列出解析失败、候选不足、尚未处理和未知费用的任务/调用。非 ready 返回退出码 2；部分结果仍保存。

**费用不受剩余 67 tokens 截断：**`C_m(ddpo) = 13,776 + Σ所有相关排名 attempt 的 output tokens`，失败/弃用回复同样收费；B=13,843 限制原证据采购，不阻止已授权的排名补采。超过 cap 时 `exceeds_cap=true`、`over_cap_tokens>0`、`remaining_budget<0`，不丢任务以适配预算。22 个可排名任务、无重试且每次最多 64 output tokens 时排名上限为 1,408，总额至多 15,184；真实是否超 cap 以 provider usage 为准。顶层采购 C_m/余额、其他臂和历史账本保持原值；未知费用时所列 C_m 仅含已知费用，`ranking_cost_complete=false`。旧 legacy 重建器仍拒绝覆盖不兼容的排名会计信息；v2 默认重建器会读取排名账本并重新生成 native dDPO 行，账本是恢复来源。

CPU 验证命令：`PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_table1_ddpo_rank.py tests/test_table1_audit.py`。覆盖去重/恢复、跨运行调用上限、失败与零/未知 usage、429 有界重试、旧 rank 行格式、三个 manifest 及超 cap 费用、官方 score 完整性、采样命令隔离，以及真实 94 行/22 个官方父任务的只读核验。

## 6. RTD 交叉核对与 Table 1 合规性

| 项目 | 核实事实 | 影响 |
| --- | --- | --- |
| BFCL 分母和证书 | audit_v11：55,370，5,537/13,843/27,685；demo 2048 / generator 512。public/index/core/set hash、逐包 integrity、实际成本不超类上界均核验 | 内容预算一致；不等价完整历史调用账单 |
| Ledger | ledger.py 按 recorded cost、exact/estimated 结算；同 query_id 一次计费；无 verified 免单分支 | 进入 settle 的失败调用会收费 |
| Broker | broker.py 先按 public cap reserve，读 payload、检查非空 behaviors，再 settle；异常 release。ALFWorld 112 失败包 public unavailable，不会提供 | 端到端失败采购收费不一致；不能只看 Ledger 宣布合规。BFCL 可解析的错误 demo 与未通过原始验证的 generator 仍有 behaviors，RTD 可能付费训练；本工具付费但不产正例 |
| cap 与记录成本 | RTD 用 cap 做可买性/预留；本工具用逐包 recorded cost 做严格前缀 | 余额低于 2048/512 时 RTD 可能提前失去候选，批量预留也改变停止和已购集 |
| cost model | 本地 acquisition.py 拟合 recorded/cap，exact noise=1、estimated noise=4，预测裁至 [0,cap]；仅学已购样本 | cap 变化影响初始预测、尺度、价值/成本排序；缺失 v1.1 代码，不能假定运行实现相同 |
| ALFWorld 库与预算 | c26 219 条仅 107 可用；旧审计 usable_public_cap_sum=140,247,040，episode cap=1,310,720 | 与本次 219 attempts/B=9,074 不同，需统一失败候选政策与绝对 cap |
| 随机顺序与窗口 | 本工具全池固定前缀；旧 RTD 有 inner fold、dependency、offer/window 和 cap 限制 | 都称 seed 0 不保证同序列/停止；V1 复用 V0 曝光也不证明符合本次前缀协议 |
| 可执行配置 | 本地 v1_bfcl_c25.yaml 为 1.0.1、usable_public_cap_sum、每窗最多 1 包；缺 v1.1 配置/构建代码 | 方法文档与数据证书可核实，运行实现不能认证；未访问其他 worktree/远端 |
| 初始 checkpoint | 方法 §2 与本工具要求 base Qwen3.5-4B、无 adapter | 当前没有运行中 manifest/checkpoint 副本，V0/V1/V2 实际初始化仍需运行记录 |

BFCL v1.1 证书的内容分母与已结算成本符合指定回放口径，但调用完整性、公开成本信息、cap 停止、失败可买性、随机顺序与实际初始化仍有差异/证据缺口。ALFWorld 旧池排除失败，AppWorld 新冻结池尚未接入 RTD。因此不给运行中 V0/V1/V2 签“完整冻结协议合规”；用户可结合运行 manifest 决定作为缓存内容实验披露，或统一差异后比较。本次未修改 RTD 代码。

## 7. 复现与 CPU 验证

在当前 worktree 执行。PYTHONDONTWRITEBYTECODE 防止写共享依赖 bytecode，无需模型、tokenizer 下载或环境初始化。

```bash
export PYTHONPATH=src:.
export PYTHONDONTWRITEBYTECODE=1
PY=.venv/bin/python
$PY tools/table1_budget_audit.py
$PY tools/table1_random_acquisition.py
$PY tools/table1_pool_from_sealed.py
$PY -m pytest -q -p no:cacheprovider tests/test_table1_audit.py
$PY tools/table1_budget_audit.py --report-only
git status --short
git diff --stat
```

测试覆盖：排序与随机确定性、三训练种子共享顺序、严格前缀、cap/余额/零成本包、失败收费无正例、切段不重计、STaR 零成本、缺费用 rank 不免费、PBSD/c 同任务限制、篡改采购拒绝、仅已购文件可读、只读路径保护、旧池 prompt/response 完全一致与其他字段逐项比较、SAD 与实际 trainer 函数一致。随机采购测试将 sealed reader 和文件读取均 monkeypatch 为抛异常，仍能完成采购。真实归档对应测试在本次生成结果上执行。

results/ 被 .gitignore 忽略；产物在磁盘保留供审阅，未暂存/提交。ledger.sources 记录输入 SHA-256；再次审计若源文件变化则拒绝覆盖冻结库存。

初次预算审计 CPU 验证：**31 tests，0 failures，0 errors，0 skipped**。当时非空池仅通过实际 trainer 的 load_pool 字段验证（只抽取该函数，不导入 GPU 栈），不包含 encode。JUnit 记录：results/table1_audit/cpu_tests.xml；本次真实 load_pool + encode 的 163 项回归见“池行格式”。
