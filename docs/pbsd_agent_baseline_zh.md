# Table 1：PBSD — agent adaptation

实现入口：`AW_DISTILL=pbsd_agent` → `src/bfas/pbsd_agent.py`。本文描述用户冻结的 agent 版本；旧 `AW_DISTILL=pbsd` 的行为、pool 构造器、`bfas.arms` 和其他训练 arm 均保持原样。主 trainer 只有两个新模式分支，共 8 行插入。

## 算法与训练边界

对每个选中的任务起始状态 `s`，读取同一 task ID、同一状态的已购买且验证通过的 DeepSeek demonstration。`c` 只包含这些 demo 的第一段 assistant 输出，用明确的 evidence 边界包装。工具描述仍来自原始 `s`，不从别的任务取资料。

- Student：当前 LoRA policy `πθ(·|s)`。输入使用原 trainer 的 `prompt_token_ids()`：有 `messages` 时使用原 chat template，否则使用已序列化的 `prompt`。
- Contextual teacher：初始学生底座 `πθ0(·|s,c)`。复用同一个 PEFT 模型，在 `disable_adapter()`、`eval()`、`no_grad()` 中采样和评分；底座全部冻结，没有 EMA、同步 LoRA 或另一个外部教师。
- `y+`：从 contextual teacher 本地采样，默认 temperature 1.0；DeepSeek 文本只作为 `c`，不会被直接指定为 target。生成结果不经过正确性过滤。
- `y−`：从当前学生本地采样，temperature 固定 1.0；不要求失败，不读取 pool 的 `_rejected`。默认每个 optimizer step 都刷新。
- 两边使用新建的 generation config：`top_p=1`、`top_k=0`、无 repetition penalty、无 beam search；action 上限复用 `trainer.MAX_RESPONSE_TOKENS=512`。

每个 pair 的 loss 是：

```text
sp = log πθ(y+ | s)       tp = log πθ0(y+ | s,c)
sn = log πθ(y− | s)       tn = log πθ0(y− | s,c)
L  = -logsigmoid(β * (sp - tp - sn + tn))
```

四项都调用已有 `completion_log_prob()`，采用 causal shift、response-only mask 和序列 log-prob 求和，无长度归一化、SFT 混合项、EMA gate 或 verifier weight。`tp`、`tn` 显式 detach；采样也没有计算图。teacher 工作先于学生 autograd 图构建；两个学生 forward 到 backward 之间不切换 adapter，兼容 gradient checkpointing。

编码直接连接「对应分支的 prompt IDs + 原始采样 action IDs」，prompt、工具描述、起始 observation、`c` 的 labels 全部为 `-100`。EOS 只有被模型实际生成时才计入；达到 action cap 时评分该输出前缀，不人为补 EOS。这样避免 decode/re-encode 或追加 EOS 改变采样对象。

BFCL single-turn 的状态就是已有 demo pool 中的 task prompt；在线采样走本地当前模型，作用对应 `bfcl_task_rollout()` 中对当前状态的 `sample_action()`。这里不执行 BFCL handler/checker，因为所需负样本是任意当前 action，且本版本不训练后续环境状态。沿用 baseline 的 chat template，不额外套用 RTD 的 thinking-off 转换。

AppWorld / ALFWorld 只取每个任务最小 `turn_index` 的 demo 第一 assistant turn；已有 AppWorld 的第一索引可能是 2（消息位置），所以不能硬编码为 0。有 `messages` 时禁止先前 assistant/tool 消息；原始 prompt 也检查已序列化的 assistant/tool 历史。若同一 task 的起始 demo prompt 不一致，则报错。将来训练后续 turn 需要用当前 policy 执行环境得到 `s_t`，以完整环境状态/历史指纹索引已购买的同状态证据；不能把任务起始 demo 或其他轨迹的第 t 个 turn 当作状态匹配证据。

## 与论文的关系和差异

公式、contextual reference、teacher 正样本和当前学生负样本对应 [论文 v2 §3.2 与 Algorithm 1](https://arxiv.org/html/2605.05040v2#S3.SS2)。冻结初始 teacher 与[附录 F.3](https://arxiv.org/html/2605.05040v2#A6.SS3)一致。

| 项目 | 本实现 | 原因 / 对比 |
|---|---|---|
| 模型、任务 | Qwen3.5-4B；BFCL / AppWorld / ALFWorld task-start | Table 1 的 agent 设置；论文使用 Qwen3 系列和数学/ToolAlpaca |
| privileged context | 同任务已购买 DeepSeek demo 第一 assistant turn | 用户冻结的 evidence 约定；普通工具描述双方均可见 |
| 优化配置 | rank 16、alpha 32、dropout 0、AdamW、LR 1e-4、8 units/step、3 epochs | 复用此仓库其他 arms；论文 F.3 为 rank 64、alpha 128、LR 5e-6、batch 32、500 steps |
| 生成 | Transformers 本地采样；512 action tokens；temperature 默认 1.0 | 与本仓库 action cap 及用户默认一致；论文训练配置为 1024、1.1，使用 vLLM |
| 刷新 | 正/负默认每步生成；可显式缓存正样本或延长刷新周期 | 默认保持在线构造；非默认缓存属于计算消融，必须连同配置报告 |
| prompt 长度 | 保留完整 `s` 和 `c`，超限报错 | baseline `encode()` 可截断 prompt；本实现避免丢失工具、状态或已付费证据 |
| EOS | 只评分实际采样 EOS，截断时不补 | 原 `encode()` 对作者提供的文本补 EOS；在线采样需要保持同一 action 的概率定义 |
| 预算 | 只按实际 demo acquisition 输出 usage 扣 evidence budget，本地采样/评分另记 | 用户要求分离已购买 evidence 与训练 compute |
| 多轮范围 | 仅 task-start / 第一 assistant turn | 后续状态证据尚未定义；不能宣称完成了全轨迹在线 PBSD |

不承诺两次独立刷新后的随机 loss 单调下降，也不将 teacher-generated 正样本解释为必然正确。

## 数据与预算

`--selection full` 在这个新模式中仍受 `--budget B` 约束：按 pool 首次任务出现顺序取完整 evidence unit 的最大前缀，遇到下一个 unit 超预算就停止；`random` 用 `seed` 打乱任务 units 后使用同一规则。一个 unit 的多个同状态 demo 一起购买、一起入 context，不拆分。重复训练/采样不会再次扣 acquisition cost。

输入沿用 trainer JSONL 字段：`task_id, teacher, turn_index, prompt, response, token_hint`，可带 `messages`。`teacher` 必须为 `ds` 或包含 `deepseek`，显式 `verified=false` 被拒绝。输入应来自已验证 demo pool；缺失 `verified` 的历史 curated pool 仍按其 producer 的验证约定读取。`token_hint` 不能当作 API usage。

成本可以来自行内 `evidence_output_tokens`、`output_tokens`、`output_token_count`、`completion_tokens`、`usage.output_tokens` 或 `usage.completion_tokens`。必须是非负整数（合计必须为正）；历史数组按记录值逐项相加；冲突成本报错。sealed 的 `cost` 需要 `cost_confidence=exact` 或明确 `cost_basis=output_tokens`。估算成本不接受。

缺少行内成本时，`AW_PBSD_COST_RECORDS` 可指向 JSON / JSONL / sealed directory。通过 **task ID + 完全相同的 response 文本** 匹配；记录还带 prompt/turn 时同时校验。sealed 支持 `historical_response` 内的 `task_id/id` 和 `response/result`；成本来自外层 exact cost/usage 或内层 recorded usage。不同匹配记录给出不同成本时拒绝猜测。全 episode aggregate 不能伪装成第一 turn 的 usage；这种情况应从原始逐调用记录导出准确的 first-turn cost。

对 `data/bfcl_sft/pool_bfcl_ds_sft.jsonl`，默认读取原 producer 使用的只读归档：

```text
envs/bfcl/gorilla/berkeley-function-call-leaderboard/result_demos_deepseek_v4_pro_FC/**/*_result.json
```

23 行全部匹配，合计 **4,768 output tokens**。该成本可以包含 API 内部 reasoning 等实际收费输出，不能用最终展示 response 的学生 tokenizer token 数替代。GPU check 的固定四任务为 `irrelevance_16`、`live_multiple_207-91-1`、`parallel_80`、`simple_python_31`，成本分别为 447、79、142、110，共 **778**。

模型和 tokenizer 强制 `local_files_only=True`；训练/check 的 offline guard 拦截 IP socket、DNS 和 UDP，吞掉异常的网络尝试也会使运行失败。没有 teacher API、subprocess、环境执行或 pool 写入；需要的 HF checkpoint 必须预先存在于本机缓存。

## 配置和产物

| manifest 配置 | 环境变量 | 默认 |
|---|---|---:|
| `pbsd_beta` | `AW_PBSD_BETA` | 0.1 |
| `pbsd_positive_temperature` | `AW_PBSD_POSITIVE_TEMPERATURE` | 1.0 |
| `pbsd_negative_refresh_steps` | `AW_PBSD_NEGATIVE_REFRESH_STEPS` | 1 |
| `pbsd_positive_refresh_steps` | `AW_PBSD_POSITIVE_REFRESH_STEPS` | 1；0 表示每个 unit 只生成一次 |
| `pbsd_max_context_tokens` | `AW_PBSD_MAX_CONTEXT_TOKENS` | 32768，含 teacher context 和 action 预留 |
| epochs / LR | `AW_EPOCHS` / `AW_LR` | 3 / 1e-4 |
| gradient checkpointing | `AW_GRAD_CKPT` | 0；1 启用 |
| exact archived costs | `AW_PBSD_COST_RECORDS` | BFCL 指定池有默认归档，其余无 |

刷新 bucket 是零起始 optimizer step 的 `step // refresh_steps`。缓存按 task-state-evidence unit 和分支区分：某个 unit 下次被选入 batch 时，如果进入新 bucket 才重新采样；未使用的任务不产生额外 compute。默认的 1 始终使用本次更新之前的当前学生；值大于 1 时允许同组更新复用 behavior sample，应如实作为非默认配置报告。

如果显式设置 `AW_MAX_PROMPT_TOKENS`，将它用作完整学生状态长度上限，超限报错；不执行 head/tail 截断。未设置时不继承 baseline 的默认 640 截断。teacher 的额外 evidence 使用独立的完整 context cap。`AW_TRUNCATE_SIDE` 不改变此行为。

输出与其他 arms 一致：`results/appworld_students/<tag>/adapter` 是 **merge_and_unload 后的完整模型**，含 tokenizer，非只有 PEFT delta。另存 `selection_manifest.json` 和 `pbsd_agent_journal.jsonl`。拒绝覆盖已有 tag。

journal 每步保存每个 pair 的四项 log-prob、loss、正负 action SHA256、generation ID、sample step、实际 output token 数、是否 cap 截断，以及 student/teacher prompt hash。manifest 保存逐 evidence row 的 pool index、response hash、cost 和 cost source。compute 累计分别记 teacher/student 的生成次数、prompt/output tokens 和评分 input/output tokens；GPU 诊断评分另有 `diagnostic_*` 字段。`evidence_output_tokens` 始终是一次性的 acquisition 总额；`teacher_api_calls=0`。

上下文隔离检查验证实际 forward 的 conditioning IDs 与「仅由原始 state 编码」的 hash 一致。Teacher prefill 必须匹配独立构造的 `(s,c)` IDs。KV/recurrent cache 只在单次 generate 内使用；后续 decode forward 追踪该分支的 prefill，绝不把 teacher cache 给 student。这里隔离的是 **conditioning context**：模型可能自行生成与 demo 相同的回答，生成的 `y+` 作为监督输出进入 student scoring 是算法本身，不是把 `c` 注入学生 prompt。

## 本 worktree 的 CPU 与 GPU 命令

```bash
cd /home/xueqi/hq/projects/tc-alignment-pbsd
PY=.venv/bin/python
export PYTHONPATH=src:.
export PYTHONDONTWRITEBYTECODE=1

# CPU，无模型下载：tiny Llama + 真实 PEFT；另有旧 arms loss / 源码 hash。
"$PY" -m pytest tests/test_pbsd_agent.py tests/test_appworld_train_truncation.py \
  tests/test_pair_unit.py tests/test_bfas_conformance.py -q

# GPU：四个 BFCL support-demand tasks × 两步；check 默认缓存正样本。
AW_GRAD_CKPT=1 "$PY" tools/pbsd_agent_check.py --seed 0 \
  --student Qwen/Qwen3.5-4B --tag pbsd_agent_check_s0

# 检查训练默认的“正样本也每步刷新”；使用新 tag。
AW_GRAD_CKPT=1 "$PY" tools/pbsd_agent_check.py --seed 0 \
  --positive-refresh-steps 1 --tag pbsd_agent_check_refresh_s0

# Table 1 BFCL：完整 23 行已购买 evidence，预算 4768。
AW_DISTILL=pbsd_agent AW_POOL_PATH=data/bfcl_sft/pool_bfcl_ds_sft.jsonl \
  AW_GRAD_CKPT=1 AW_MAX_PROMPT_TOKENS=4096 \
  "$PY" src/appworld_train.py --selection full --budget 4768 --seed 0 \
  --student Qwen/Qwen3.5-4B --tag bfcl_pbsd_agent_s0
```

GPU check 每个 pair 打印四项 log-prob 和 action hash，更新后逐项用 `no_grad` 重算 teacher score，要求 bitwise 相同；确认实际 forward hash、LoRA-only gradients、每步 negative action hash 不同、positive generation ID 与配置一致、证据成本准确。检查两种下降：第二批在线 pair 的平均 pre-update loss 小于第一批，以及第一批固定 pairs 在两次更新后的 loss 低于更新前。任何失败退出非零并保留诊断；不重采样、不调参重试、不按 loss 选择样本。随机 sample 重复或在线 loss 上升都可能导致真实运行失败，需要查看报告，不能将 CPU 测试通过写成 GPU 检查通过。

输出格式兼容 `tools/bfcl_std_campaign.sh` 的 adapter / merge-export 约定。当前分支存在 `scripts/bfclv2_hpg.slurm`；用户提到的 `tools/bfclb2_fill_lane.sh` 不存在。旧 campaign 会向 vendored `envs/bfcl/...` 写临时 generation/score，所以本任务不启动它，保持 `envs/` 只读；严格只读的评测须另行准备可写 harness 工作副本。没有修改现有 campaign 或其他 worktree。

AppWorld 例子：从**既有已购买原始记录**制作当前 worktree 中的输入文件，保留任务初态的 prompt/messages、第一 assistant response、DeepSeek provenance 和准确逐调用成本。示例路径是用户准备的文件，不是声称当前仓库已存在：

```bash
AW_DISTILL=pbsd_agent AW_POOL_PATH=results/pbsd_inputs/appworld_deepseek.jsonl \
  AW_PBSD_COST_RECORDS=results/pbsd_inputs/appworld_exact_costs.jsonl \
  AW_GRAD_CKPT=1 AW_MAX_PROMPT_TOKENS=8192 \
  "$PY" src/appworld_train.py --selection full --budget 20000 --seed 0 \
  --student Qwen/Qwen3.5-4B --tag appworld_pbsd_agent_s0
```

也可以用已有 `data/appworld_sft/pool.jsonl`，但必须补充可以准确匹配第一 turn 的已记录 usage。缺少准确成本时读取器会直接报错，不把 `token_hint` 当预算，也不会重新调用教师。

ALFWorld 使用同样的行 schema；task-start prompt 应含初始 observation、目标和原本可见的工具/命令说明，response 仅是该状态的第一 assistant action。已看到的 `data/alf_sft/pool_B_first.jsonl` 是 `demo_replay` event pool；不能通过改 teacher 标签把它当作 DeepSeek 证据。现存部分 ALFWorld sealed episode 只有 estimated usage，亦不能用于此精确成本基线。

```bash
AW_DISTILL=pbsd_agent AW_POOL_PATH=results/pbsd_inputs/alfworld_deepseek.jsonl \
  AW_PBSD_COST_RECORDS=results/pbsd_inputs/alfworld_exact_costs.jsonl \
  AW_GRAD_CKPT=1 AW_MAX_PROMPT_TOKENS=8192 \
  "$PY" src/appworld_train.py --selection full --budget 20000 --seed 0 \
  --student Qwen/Qwen3.5-4B --tag alfworld_pbsd_agent_s0
```

如果 pool 行已带准确 `evidence_output_tokens`，省略 `AW_PBSD_COST_RECORDS`。本次实现不购买新证据，也不写 `data/`、`envs/`，未提交 git commit。

本次 CPU 验证：上述四个 test 文件共 99 项通过；六组旧 arm loss hashes 另外在原始 `HEAD` trainer 上重算一致。真实 Qwen3.5-4B tokenizer 离线检查 23 个 BFCL units，student / teacher 最大 prompt 长度分别为 3306 / 3492 tokens。当前 sandbox 无 GPU，未运行真实 4B 两步数值检查，也没有生成 Table 1 分数。
