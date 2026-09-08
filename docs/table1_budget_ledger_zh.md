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

## 池行格式

**BFCL 默认改为 `native-fc`，入口为 `tools.table1_row_format.render_package()` → `render_native_row()`。** `tools/table1_pool_from_sealed.py --benchmark bfcl` 默认写当前 worktree 的 **`results/table1_audit/pools_native/bfcl/`**；原 `results/table1_audit/pools/bfcl/` 保留 `legacy-messages-v1`，可用 `--row-format legacy-messages-v1` 显式重建。不同格式禁止覆盖同一已有目录。ALFWorld/AppWorld 默认仍为 legacy，未重建或修改其池。本次在 `/home/xueqi/hq/projects/tc-alignment-audit2`、`table1-native` 分支操作；不提交、不调用 GPU/API、不运行环境，`data/`、`envs/` 只读。

### 为什么采用 native

用户提供的训练/评测证据：旧 `legacy-messages-v1` 格式的 94 行训练 3 epochs 后，Qwen3.5-4B 对 BFCL 各条目都输出 `"<think>\n[]"`，Overall 约 11%；旧 23 行池只是训练不足，未暴露同样严重的退化。采用 native rendering 的 RTD 保持约 46–47。这些是已有运行证据，本次 CPU 工作没有重新训练或复测分数。

代码可直接确认两处不一致：legacy 把函数文档塞进 system message，`prompt_token_ids()` 因 `messages` 非空重新调用 `tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)`，**没有传 `tools`**；部署则使用 Qwen FC handler 的工具列表模板。legacy 目标为 `'[{"f": "{json}"}]'`，而 FC handler 的 `_extract_tool_calls()` / `decode_ast()` 解码的是 `<tool_call>` 标签内的 `{name, arguments}` JSON。这不是 `messages=[]` 行被重复套模板的问题：trainer 对空 messages 本来就直接消费 prompt。改用 native 是对齐真实训练/部署条件和输出语言，不能从 CPU 测试推断新的模型分数。

### 精确 prompt 与 teacher continuation

- 原始审计 snapshots 保持封存；仅按冻结的 purchased IDs 读取 `data/rtd/v1_1_bfcl/sealed/<id>.json`，由 `read_bank_payload()` 核验 `ledger.sources` 的 SHA-256。失败包仍计费但不生成正行，采购顺序、调用、成本均不变。
- **`messages=[]`**。`prompt` 是 sealed `behavior.state.prompt` 的原始字节，包含工具列表、实际题目/历史以及 `<|im_start|>assistant\n<think>\n\n</think>\n\n` 后缀。`bfcl_native_prompt()` 用 vendored **`QwenFCHandler._format_prompt(messages, functions)`** 和现有 **`bfas.cc_pairs.thinking_off()`** 重新构造并逐字节核对，不复制模板，也不初始化 API client。
- 精确代码边界：`src/bfas/rtd/bank.py:build_bfcl_bank()` 内的 `full_state()` 使用 `thinking_off(render_prompt(...))`；`bfas.cc_pairs.render_prompt()` 调用 `BFCLAdapter._render()` → `QwenFCHandler._format_prompt()`。**当前 envs 内 Python `_format_prompt()` 本身止于 assistant marker**；它的模板说明包含 non-thinking 分支，但 Python 实现没有自行追加 think-empty。`src/bfas/rtd/return_gradient.py:bfcl_task_rollout()` 的评测 query 明确调用 `thinking_off(handler._format_prompt(...))`，这才是与 sealed RTD 一致的 non-thinking 部署 prompt。测试保留并明确验证这个区别，不宣称裸 `_format_prompt()` 单独返回完整后缀。
- `task_json` 排序过对象键，不能用它重序列化工具列表来追求字节一致。`bfcl_inputs()` 从 generator 的 `historical_response.function` 或 demo sealed prompt 的 `<tools>` JSON 恢复原始顺序；与 `task_json.function` 及 `history_json` / `task_json.question` 做对象核对。
- **`response=behavior.text`**，与 RTD 的 `score_behavior()` 直接编码的文本一致。demo 的 bank helper 是 `bfas.rtd.bank._response_text()` → `bfas.cc_pairs.render_calls()`；generator 是银行已有的 `render_truth()` 结果，重建池不重新选择 ground_truth 的可接受值。单次调用为 `<tool_call>\n{"name": "f", "arguments": {...}}\n</tool_call>`；多个调用按原序以一个换行连接。irrelevance demo 的自然语言逐字保留。response 没有手工 EOS，也不附加 think 前缀。
- `task_id`、`teacher`、`turn_index`、**`token_hint` 原值**保留。token_hint 是旧行的长度提示，trainer 的真实 response token 统计独立计算；它不代表教师费用。

### 空调用的显式例外

B13843 每 seed 有 **13** 条、B5537 有 **5** 条成功 generator 条目的 `ground_truth=[]`，其 sealed native continuation 是空字符串。RTD 的 `render_calls([])` 就是 `''`，后续 `action_ids()` / `score_behavior()` 监督 EOS。归档没有这些题目的自然语言答案；不能编造拒答，也不能声称旧审计补入的 `'[]'` 与 RTD 文本相同。

这些行保留空 response 并加 **`_native_fc_empty_response=true`**。renderer 只在封存空调用证据成立时标注；`load_pool()` 只对显式布尔标记、`messages=[]` 且有正确 native 后缀的行放行，普通空回答仍报错。`encode()` 不改：这类目标只有它追加的一个 EOS。其余 B13843 的 81 / B5537 的 25 行都必须有 EOS 之前的非空 response tokens。**“与 RTD 完全一致”和“每一行 response tokens 都非空”对这 13/5 条无法同时满足**；本次明确保留 RTD 的 EOS-only 例外，行数和已购费用不变。

### Trainer、SAD 与配对字段

真实 `load_pool()` → `prompt_token_ids()` → `encode(..., device='cpu')`：空 messages 禁止 re-template；prompt 和 response 分别以 `add_special_tokens=False` 编码。按 `AW_MAX_PROMPT_TOKENS=4096`、`AW_TRUNCATE_SIDE=tail` 截断 prompt，response 截至 `MAX_RESPONSE_TOKENS=512`，末尾追加一次 EOS。输入严格为 `prompt_ids + response_ids + EOS`，labels 为 `[-100] * len(prompt_ids) + response_ids + EOS`；工具说明与 think-empty 前缀全部属于 prompt mask。另测试 32-token 截断边界。这里保证文本与 RTD 相等，不宣称 trainer 的 4096/512 截断配置等于 RTD 的全部训练配置。

`_sad_spans` 在 native response 上按真实 `appworld_train.action_spans()` 的代码围栏规则重新计算，不沿用旧字符偏移；该 trainer 规则不会把 XML tool calls 自动算作代码 action，本次不改 SAD 的损失定义。`pbsd_insp._rejected` 若有精确已购状态对应的缓存，使用 RTD `render_calls()` 格式转换旧 decoded FC 列表，native 输出/自然语言保留，不重新采样或修复失败动作。当前 BFCL 已购缓存仍为 **0 对**；测试用显式 fixture 覆盖多个调用、嵌套参数、原生失败文本和尾部 EOS 的处理。

`pbsd_agent.c` 的 `state_prompt/response` 使用同一 native rendering，且 **c.state_prompt 必须等于所属行 prompt**。银行内某些 generator task_id 在不同问题间复用，native 模式按 `(task_id, prompt)` 分组，只引用确切同状态的已购证据，避免把同名不同题的 c 混入；所有引用都保留 package_id/turn_index。`c` 仍没有现成训练消费者，dDPO 仍为缺排名费用的 SFT fallback，STaR 仍为空 teacher pool。

### 输出、回归与不变量

两个预算 × 三 seeds × 七臂，共 **42 个 native 池、36 个非空池、2,232 行**（含臂/seed 重复）。B13843 六个非 STaR 臂各 94 行、C_m=13,776；B5537 各 30 行、C_m=5,471；每个 STaR 池 0 行、C_m=0。六个 manifests 均为 `row_format='native-fc'`。只有格式及 `arms.*.pool_sha256` 改变；`row_origins.row_sha256` 始终指向不可变封存源行，采购 IDs、种子、费用、失败统计、排名/基线状态不变。重建器同时与源 manifest 核对这些不变量，未改变的 acquisition 不重写。

`tests/test_table1_native_fc.py` 从三个不同已购任务取 demo、generator、自然语言 irrelevance，使用 envs 内 handler 自己的预处理/首轮/格式化函数（只读，不 query），独立从官方任务文件或已购 generator 对象取输入，并通过 RTD 自己的 `render_prompt()`、`_response_text()` / `render_truth()` 检查 prompt/response 字节相等。另对全部 94 已购正行验证 sealed bytes；对六组七臂逐行走真实 trainer，**使用本地 Qwen/Qwen3.5-4B tokenizer，`local_files_only=True`**，验证不 re-template、非空目标及 EOS-only 例外、一次 EOS 和精确 prompt mask。测试禁止 socket connect、模型权重加载和 CUDA 初始化；legacy 全池回归继续保留。

```bash
export PYTHONPATH=src:.
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=''
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
PY=.venv/bin/python
$PY tools/table1_pool_from_sealed.py --benchmark bfcl --row-format native-fc
$PY -m pytest -q -p no:cacheprovider tests/test_table1_native_fc.py tests/test_table1_row_format.py tests/test_table1_audit.py tests/test_appworld_train_truncation.py tests/test_table1_ddpo_rank.py --junitxml=results/table1_audit/native_fc_cpu_tests.xml
```

本次 CPU 回归 **177 passed，0 failures/errors/skipped（181.70 秒）**。逐池行数、字节/hash 与费用核验记录在 `results/table1_audit/native_fc_validation.json`；JUnit 在 `results/table1_audit/native_fc_cpu_tests.xml`。42 个 native 池/六个 manifest 全部校验通过；106 个预先固定 hash 的 legacy 池及 BFCL 审计/采购文件逐字节未变。legacy 之前的 163-test 记录保留在 `row_format_cpu_tests.xml`，不把旧结果冒充本次验证。

## 5. 七臂池与训练入口

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

- **SFT：**只输出已购非失败内容；BFCL 默认使用上节 `native-fc`，空调用按封存 RTD 目标监督 EOS；其他 benchmark 仍用 legacy。旧 BFCL legacy 池独立保留，失败包不产正例，教师内容和费用不变。封存 package.response_rendering 描述的是修复前源行，当前训练格式以 manifest.row_format 为准。
- **SAD：**同 SFT prompt/response，加 `_sad_spans` 字符边界，按现有 action_spans 的代码围栏规则；训练器仍自行算 mask。BFCL 无代码围栏响应按现有 trainer 规则属于非 action 部分，本次未另造 BFCL mask。
- **BB-OPD：**单轮 BFCL 与 SFT 逐行相同。AppWorld 只取旧 on-policy 上下文/响应与已购 demo 完全一致的行，是部分内容对应，不代表完整 BB-OPD 或证明当前学生访问这些状态。ALFWorld 无对应缓存，空池。补齐需另冻结 on-policy 状态与费用。
- **PBSD-insp：**只对已购行附加旧缓存的学生失败首轮 `_rejected`，其他正例训练；不做新采样。BFCL 本次 0 对，AppWorld 本次 13 对。
- **dDPO：**缺少可恢复排名调用成本，0 行复用，SFT fallback 状态保留。
- **STaR：**teacher pool 为空、C_m=0、调用/证据 ID 列表为空，不使用公共基线采购的教师证据。不能将空池交给拒绝空输入的 SFT loader；学生自训练另行进行。
- **PBSD-agent：**保留 trainer 基础字段，加 c，只引用相同实际任务的已购 demos（不将 generator 父任务当作生成任务本身）。当前 appworld_train 没有消费 c 的入口；agentkd 的 `_thought` 不是此算法。标为 evidence_only_consumer_missing，不伪称已经可训练。

标准字段为 task_id、teacher、turn_index、messages、prompt、response、token_hint。manifest 给出各臂 call ID 集合、C_m、历史总额、hash、基座、旧行字段差异。除 STaR 外，C_m 包括全部已购包，即使该臂最终未使用某包。

对已采购的非空训练池，使用 `AW_POOL_PATH` 和 **--selection full**。src/appworld_train.py 的 --budget 按学生 tokenizer 计算被选训练行 response tokens；不能代替教师调用费用，也不应再按行重做预算选择。此次只检查输入格式，不执行训练。

### 5.1 BFCL dDPO 排名补采工具（待用户运行）

`tools/table1_ddpo_rank.py` 支持初始学生采样、排名和显式解析失败重试。本次修复仅修改工具、文档并运行 CPU fake-client 测试，**没有运行 GPU 或教师 API，也没有改写真实运行账本或池**。§4–§5 的表格仍是补采前快照。5 条旧缓存 rank 不复用。

用户提供的真实运行证据位于 `/home/xueqi/hq/projects/tc-alignment-table1/`：`results/table1_audit/bfcl/ddpo_rank_ledger.jsonl` 共 16 次调用，15 次 `parse_failed` 的 `response=""`、每次 `tokens_spent=64`，另一次 `ranked` 花费 63；合计排名费用 1,023，`C_m=14,799`。`logs/table1_ddpo_rank.log` 和 `logs/table1_ddpo_rank2.log` 显示，第二次仅改为 `--max-output-tokens 1024` 没有新增调用，仍为 `pending_task_ids=[]`：旧实现将解析失败也视作已完成。以下重试开关解决此恢复问题；这些外部证据仅作只读核验。

在当前 worktree、`table1-audit` 分支运行：

```bash
export PYTHONPATH=src:.
export PYTHONDONTWRITEBYTECODE=1
PY=.venv/bin/python
$PY tools/table1_ddpo_rank.py sample --cap 13843 --repeats 4 --gpu <uuid> --port auto
$PY tools/table1_ddpo_rank.py rank --cap 13843 --key-file ~/.ollama_api_key --max-calls 40
```

已有 samples 和失败账本时，无需重新采样；由用户在持有该账本的运行目录执行重试命令（默认输出预算 512，也可显式提高为 1024）：

```bash
$PY tools/table1_ddpo_rank.py rank --cap 13843 --key-file ~/.ollama_api_key \
  --retry-parse-failed --max-calls 15 --max-output-tokens 1024
```

该命令的 `--max-calls 15` 允许本轮至多 15 次新调用；对应上述证据会跳过已有的 1 个 ranked 任务，重试另外 15 个任务。更改输出预算本身不会解除失败去重，必须传 `--retry-parse-failed`。工具输出固定在其 worktree 的 `results/table1_audit`，不跨 worktree 写账本。

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

`rank` 对每个有至少两个不同 response 的已购父任务请求 `deepseek-v4-pro`，强制 think=false、temperature=0，并向当前 Ollama 兼容后端传 `reasoning_effort="none"`（[Ollama 支持的请求字段](https://docs.ollama.com/api/openai-compatibility)）。此参数通过 `appworld_teacher.generate_reply` 的可选参数传递；其他调用者和其他后端不自动添加该字段。`--max-output-tokens` 默认从 64 提高到 **512**：真实运行中即使 think=false，64-token completion 预算仍全部耗在隐藏推理上，正文为空；额外关闭 reasoning 并留出预算可减少此类失败，但仍按实际回复解析与记账。

请求使用旧 `RANK_PROMPT` / `last_two_numbers`，展示全部去重候选及完整官方 prompt，不按 checker 正误预筛选。不把生成子任务的 prompt 当作父任务上下文，也不把 SFT 教师答案当候选。成功排名后以学生 best/worst 组成旧格式的 `teacher='rank'` 行：`task_id, teacher, turn_index, messages, prompt, response, _rejected, token_hint`；`token_hint` 仍只是学生答案的训练提示长度，不用于排名费用。

教师客户端通过 `appworld_teacher.generate_reply` 的可选 `usage_out`，在解析 content 前暴露 provider usage；原调用者仍获得字符串。排名优先按 `completion_tokens` / `output_tokens` / `eval_count` 记 `tokens_spent`、`confidence='exact'`，不使用输入或 total tokens。若没有 usage，用本地 Qwen tokenizer 对教师回复计数并标 `estimated`；空回复但 usage 缺失至少记 1，不能推断免费。空字符串、纯空白和索引解析失败均记 `parse_failed`，不产偏好行，仍将调用、原回复和费用写入账本；只有 provider 明确报告 output=0 时才可零收费跳过。默认恢复跳过解析失败；传 `--retry-parse-failed` 后，对最新 attempt 为 `parse_failed` 或回复为空字符串/纯空白的任务，本轮各新增至多一次调用。按 `attempt_index` 判定最新 attempt；任一历史 attempt 已 `ranked` 的任务始终跳过，即使之后存在失败记录或原始回复字段为空。每次重试使用新 `call_id`、`attempt_index=max(该任务历史索引)+1`，追加 ledger；历史失败记录保留且继续计费。

没有返回文本也没有 usage 的传输/配额错误记 `tokens_spent=null, confidence='unknown'`，manifest 标 `incomplete_ranking_costs`，保留未知 call ID，不能假称已精确结算。`response=null` 表示未收到回复，不作为空字符串解析失败处理。fallback tokenizer 失败时也保留回复并标未知费用。

新账本为 `results/table1_audit/bfcl/ddpo_rank_ledger.jsonl`，保留原 BFCL ledger 的 `task_id, teacher, attempt_index, temperature, verified, tokens_spent, purpose='rank', timestamp`，补充 `call_id`、provider usage、候选、回复、解析状态和 preference row。`verified=false` 表示排名调用不是通过官方 checker 的教师示范；学生样本的官方 verdict 在 samples 文件中。三个训练 seed 共享同一次排名及 call ID，不三次付费。

`--max-calls` 是**本次 invocation 的 HTTP attempt 上限**，新任务、解析失败重试、API 失败和 429 重试共用此上限。历史调用不占用本轮次数，但仍计入总费用；下次运行重新计算调用额度。关闭客户端隐藏重试及备用 key 轮换；只使用指定 key-file，`OLLAMA_BASE_URL` 如未设置默认 `https://ollama.com`。普通未完成任务遇到 429 至多重试两次（该任务本轮总计三次，间隔 1/2 秒）；显式解析失败重试只新增一次 attempt，即使再次失败或遇到 429，也不会在同一轮再请求该任务。429 重试耗尽或其他 API 错误终止本轮；配额恢复后可重跑。历史非 429 API 错误通常需先核对结算；最新状态符合显式解析/空回复重试条件时可按上述规则追加 attempt，旧费用和未知费用记录仍保留。

文件锁防止两个进程同时购买；每次请求前 fsync 写 `ddpo_rank_attempts.jsonl`，回复结算后 fsync 追加 ledger。若进程在请求与记账之间崩溃，下一次会拒绝自动重试、列出未结算 call ID，需根据真实 provider 记录补齐 ledger 后恢复；不能删除意图日志来把潜在收费调用变成免费，`--retry-parse-failed` 也不绕过此检查。

每次正常停止（含 max-calls/配额停止，甚至零次新调用）均从**完整 ledger** 重建三个 seed 的 `pool_ddpo.jsonl = 原审计 pool_sft + 每个成功排名任务一行`。费用累加所有相关任务的历史 attempt，重复的相同 `call_id` 只计一次；每个任务只采用 `attempt_index` 最大的 ranked 结果，更晚失败不覆盖已有成功。重建会替换旧池中的 rank 行，保证每个 seed 对每个 ranked 任务恰有一行，不因重复恢复而重复追加；同步重算 `arms.ddpo` 的 `ranking_call_ids, teacher_call_ids, pool_sha256, rows, ranking_cost, C_m, rank_pairs` 及 `ranking.new_ddpo.ranked_tasks`。

完成且至少有一个偏好行时 `status` / `trainer_status='ready'`；未处理完为 `partial_ranking`，没有可用偏好为 `no_rankable_preferences`，未知费用为 `incomplete_ranking_costs`。默认恢复仍可跳过解析失败后 ready；开启重试时，尚未重试或再次解析失败的任务列入 `pending_task_ids`，不能误报全部完成。manifest 同时列出解析失败、候选不足、尚未处理和未知费用的任务/调用。非 ready 返回退出码 2；部分结果仍保存。

**费用不受剩余 67 tokens 截断：**`C_m(ddpo) = 13,776 + Σ所有相关排名 attempt 的 output tokens`，失败/弃用回复同样收费；B=13,843 限制原证据采购，不阻止已授权的排名补采。超过 cap 时 `exceeds_cap=true`、`over_cap_tokens>0`、`remaining_budget<0`，不丢任务以适配预算。如果 22 个父任务均有不同候选、无重试且每次最多 512 output tokens，排名上限为 11,264，总额至多 25,040；真实证据中只有 16 个任务进入排名。上述重试的费用从已花费的 1,023 继续累加：`ranking_cost=1,023+Σ新 attempt output tokens`，不能只记最新成功。真实是否超 cap 以 provider usage 为准。顶层采购 C_m/余额、其他臂和历史账本保持原值；未知费用时所列 C_m 仅含已知费用，`ranking_cost_complete=false`。完成后不要再次运行旧 `table1_pool_from_sealed.py` 覆盖新 dDPO manifest；排名账本是恢复来源。

CPU 验证命令：`PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_table1_ddpo_rank.py tests/test_table1_audit.py`。fake client 覆盖仅重试解析失败/空回复、ranked 永不重呼叫、attempt 递增、历史费用累加、本轮调用上限（新任务和重试共用）、429 有界重试、512 默认值与 reasoning-off 请求字段、完整 ledger 修复陈旧 manifest/重复 rank 行，以及三个 seed 每个 ranked 任务恰有一行。另覆盖零/未知 usage、官方 score 完整性、采样命令隔离，以及真实 94 行/22 个官方父任务的只读核验。

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
