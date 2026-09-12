# RTD v1.1 D12：训练期 HF 批量采样

本次只修改 `/home/xueqi/hq/projects/tc-alignment-v11a`，分支 `rtd-v11-a`；不提交。`data/`、`envs/`、共享 `.venv` 及生产 worktree 只读。没有 GPU 训练、GPU 性能测量、教师调用或正式评测。

## 1. 接入范围

`runtime.HFGenerateBackend` 通过 `generation_batch.HFGenerationBatchMixin` 增加：

```python
backend.sample_actions(prompt_ids, n, parameters, generator)
backend.sample_actions_batch(prompts, n_per_prompt, parameters, generator,
                             max_batch_tokens=16384)
```

兼容已格式化的 prompt 字符串及原始 token IDs；返回 `ActionTrace`，第二个接口按原 prompt 顺序返回每个 prompt 的动作元组。同 prompt 用展开的输入行在一次 `generate()` 中抽样，保持 `num_return_sequences=1`，等价于重复输入后的多返回序列。超过显存 token 预算的请求拆为多个调用，不删减 prompt、K 或动作上限。

| 消费位置 | 批量范围 | 保留的检查 |
|---|---|---|
| `sample_state` 的来源 Y1/Y2（含 round setup） | 同状态两动作 | 逐动作 teacher-forced 重评分、失败前写 journal、source/policy 身份 |
| `alpha_draw_pairs` 的 pilot、commit、old virtual reference | 多状态 × 每状态两动作，按长度分桶 | 付费/inner-fold 校验先于生成；仍逐单元消费，执行原 D7 cache、RNG、draw ID、来源快照检查 |
| 三个反馈角色 | 每个任务 K 个 task-start 动作一起生成 | 每条完整 task-start rollout 仍独立执行原 BFCL runner/checker；角色和参数 hash 检查不变 |
| `alpha_validation_return` | 同上，使用原 validation 独立 RNG | 不供应 α/d/采购标签，检查完整任务及所选参数点 |
| 固定任务集 parse battery | 40 个槽位 × K=4，跨状态分桶 | 原独立诊断 RNG、固定任务集身份、syntax checker、action hash、失败率分母 |
| `TorchPolicyBackend.source_sampler` 的 HF 子类路径 | 同一 SamplingRequest 的多动作 | 原 frozen snapshot 验证 |

多轮任务只有起始状态在抽样前已知。其后动作取决于各自环境返回，继续走独立 `sample_action`，不并发执行 BFCL 环境、不重放环境前缀、不假定分叉后的 prompt 相同。三角色之间以及 validation 之间不共享已生成动作。CPU 实际检查了全部 **33 个可运行 BFCL support** 的起始 prompt，与原 runner 首次查询 **逐字一致**。

来源和 parse 的预取队列只活在一次采样 scope 内。每个元素只能消费一次，消费时核对 prompt token、policy、动作 cap 和 RNG ticket；用完、出错或退出 scope 后立即清空。它不进入 source cache、StateStore 或后续 commit step，KV cache 也只属于当次 `generate()`。

## 2. 配置、批次和内存

`v1_1_bfcl.yaml` 启用：

```yaml
generation_batch: {prompts_per_batch: 8, max_batch_tokens: 16384}
```

缺少该键就保持旧单动作路径；v1.0 拒绝该键。仅 v1.1 显式提供空映射时使用上述默认值。旧配置不会自动加入 batching，也不会替换已经保存的 manifest。启用后的后端身份绑定批次配置、RNG 规则、torch/transformers 版本；实际采样元数据仍记录完整实现和 dtype。

批次规则：

1. 先按原请求顺序分配 RNG tickets，再按 `(effective_action_limit, prompt_length, original_index)` 稳定排序。
2. 最多包含 8 个 prompt 请求；重复槽位也算独立请求。同请求的 n 行尽量留在同一批。
3. 预算是 **展开行数 ×（最长 prompt 长度 + 动作 cap）**，包括所有 padding 和所有返回序列；最长/最短 prompt 长度比不超过 2。
4. effective cap 不同的行分批。因此每行都是原来的 `min(max_action_tokens, max_context_tokens - 原prompt长度)`，不被同批其他 prompt 缩短。
5. 单条已超过预算的完整 prompt 单独运行，保留原 context/cap 语义，不能通过静默截断规避内存问题。任意接近 32k 的长 prompt 不属于本默认值的 <40 GB 保证范围。

内存选择针对当前真实 support，而非所有可能的 32k 输入：本地 Qwen 配置 vocab=248320、hidden=2560、32 层，其中 8 层 full attention，16 个 query heads、4 个 KV heads、head_dim=256。BF16 基座约 8 GB，FP32 LoRA 和功能快照另计。按真实 40 槽位的 prompt/cap 做 CPU 规划：80 动作 **13 次调用**，160 个 parse 动作 **24 次调用**；最大的 prompt 是 7076 tokens。保留的 FP32 generation scores 上界分别约 **8.14 GB / 10.17 GB**；长 prompt 会缩到两行，避免扩大 eager attention 的二次内存项。这是将峰值控制在 40 GB 以下的保守配置依据，**没有 GPU 实测，不能把估算当测量**。benchmark 同时报告 allocated/reserved 峰值，并对 batched generation 的 allocated 峰值 ≥40 GB 返回非零。

## 3. Qwen3.5 左 padding 安全性

实际只读检查版本：`torch 2.13.0+cu130`、`transformers 5.14.1`。检查文件：

`/home/xueqi/hq/projects/tc-alignment/.venv/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py`

以及同安装的 `masking_utils.py`、`generation/utils.py`。

结论：当前实现的 **文本输入、一次完整 prefill、随后逐 token decode** 支持左 padding。本次不是带 padding 的 cached chunk continuation，也不处理多模态输入。

- `create_recurrent_attention_mask`（masking_utils:1447）在首次 prefill 将二维 mask 传给 linear-attention 层；已有 recurrent cache 时返回 None。
- `apply_mask_to_padding_states`（modeling_qwen3_5:207）在 batch>1、sequence>1 时将 padding hidden states 清零。单行批次由 planner 直接使用其真实长度，因此没有需要该分支处理的单行左 padding。
- `Qwen3_5GatedDeltaNet.forward`（:448）先做上述清零。QKV、a/b/z 投影和 depthwise convolution 都无 bias；左侧零前缀的 q/k/v 为零。卷积初始状态为零，循环初始状态为零，即使 g 有衰减项，零前缀也不会产生有内容的循环状态。真实 token 前的卷积上下文与未 padding 时的隐式零边界一致。
- conv/recurrent cache 分 batch 行保存。prefill 后每行末尾都是真实 prompt token；后续活动行输入是真实生成 token，因此 cached decode 不再需要对这些行乘 padding mask。
- full-attention 层使用 `create_causal_mask` 屏蔽 padding keys。HF `_prepare_position_ids_for_generation`（generation/utils:723）用 `attention_mask.cumsum(-1)-1` 生成真实位置，后续从各行最后位置增加，左 padding 不改变真实 token 的 RoPE 位置。
- HF `unfinished_sequences` 单独处理 EOS；结束行后续填充的 EOS 仅影响该行，其额外状态不参与其他行。返回时在首个 EOS 后裁掉填充及对应 score，真实 EOS 自身仍计分。达到 cap 的动作标记 truncated，不补 EOS。

CPU tiny Qwen3.5 含真实 linear/full attention、非零 PEFT LoRA，覆盖不等长左 padding、跨 delta-rule chunk 边界及 cached decoding；去 padding 后对每个采样 token 使用单行无 cache teacher forcing，FP32 最大误差 <1e-4。GPU FLA/causal-conv1d 路径仍需用附带脚本验证数值一致性；本次未运行这些 GPU kernels。

## 4. RNG、journal 与恢复

规则名：`ordered-int64-tickets-hf-batch-sha256-v1`。

每个真实动作从既有 durable sampling generator 消耗一次标量 `torch.randint(0, 2**63-1)`。ticket 含 seed、该流消费前后的 RNG digest，以及由规则名/前态/ticket 生成的 draw ID。master stream 的消费顺序是调用者的原始 prompt-major/sample-major 顺序，分桶不会重排该顺序。

每批用有序 ticket 列表的 SHA256 前 15 个 hex 位派生 HF seed；HF 在 `fork_rng` 中使用该 seed 的正常二维 `torch.multinomial`，不改变 temperature=1、top_p=1、top_k=0、typical_p=1、repetition_penalty=1。每个动作保存 ticket、batch seed、batch RNG 后态、批大小、padding 宽度和逐 token generation logprob；`generated_tokens` 增加每样本 hash 和 RNG 字段。

固定 `(初始seed, 原请求序列, draw位置, cap, 批次配置, 软件版本)` 可确定性重放。**不同批次布局不承诺同 seed 生成同 token**：HF 的二维 multinomial 消耗顺序与串行不同。各样本仍遵循相同 categorical 分布；CPU 测试用 tiny vocab 的两组各 512 次单 token 抽样检查批量/原单行路径与理论分布，并另外核对固定布局逐 token、逐字段重放。

预取用 sampling RNG 的克隆生成，再在原消费位置逐 ticket 推进真正的 durable RNG。因此原 `alpha_d_source_pair` 的 before/after、freshness、draw ID、两份 source sample hash、source ID 校验全部保留。崩溃后丢弃未完成 scope，依原 StateStore 的 durable phase/RNG 恢复；失败尝试留在 append-only journal，重新执行的动作与原尝试一致。新增测试注入“第一对已完成、第二对开始前崩溃”，通过真实 StateStore 恢复后核对第一对全部 D7 hash/digest 字段，并检查下一 commit step 获得新 draw IDs。

## 5. Score guard

`scoring.py` 未修改。仍是每动作 mean abs ≤0.05 nat、超过 1.0 nat 的 token 数 ≤2、任一 token abs ≤8.0 nat；结构、prompt 边界、EOS/cap、policy/backend identity、分数覆盖和 total 一致性也保留。所有 action score 仍从原 generation scores 计算，teacher forcing 独立执行，无重分词或后补 EOS。

benchmark 按每个动作调用相同 `score_diagnostic`，记录完整 diagnostic 后汇总 max abs、fraction >1 nat 和失败动作索引；不会用跨动作平均值掩盖某条动作超限。已保存的更严格 tolerance 也不会被 benchmark 放宽。

## 6. compute.jsonl 证据与调用数预期

只读生产文件 `tc-alignment-v11prod/results/rtd_v1_1/V0/compute.jsonl`，固定采用用户指出的**前 443 个 generation 调用**，避免把仍在增长的后续记录混入。核算为 **3195.918992 秒**；约占所述 8000 秒的 40%。

| 前缀中的 context | 原调用数 | 本次默认配置规划 |
|---|---:|---:|
| round setup source_sampling | 36 | 18 |
| alpha_d_pilot | 80 | 5 |
| alpha_d_commit_source_sampling | 80 | 14 |
| alpha_d_old_virtual_reference | 80 | 14 |
| acquisition_reference_feedback | 42 | 17 |
| same_batch_reference_feedback | 40 | 15 |
| post_commit_feedback | 41 | 16 |
| validation_learned_d_vs_zero_full | 44 | 19 |
| 合计 | **443** | **118** |

来源的 5/14/14 是对该前缀记录的原始 prompt IDs 和原 cap 执行实际 planner 得到的 CPU 计数；setup 仍逐状态、每次两动作。反馈每个角色有 34 个 task-start rollout（8×K4 + 1×K2），合并为 9 次起始调用，保留后续环境动作调用数，故每角色减少 25 次。同 prompt 合并单独预计 443→205；加来源跨 prompt 规划预计 443→118，减少约 **73.4%**。该前缀不含完整后续 validation/parse 等工作，不能把 118 当作所有决策步的固定总调用数。

这些数字固定旧 prompt、旧 cap 和旧多轮分支，仅说明 launch 数量。新抽样的 token 长度、EOS 时机、多轮分支以及学习轨迹可能不同，实际 GPU 耗时不能按 7.2秒×剩余调用数线性推算，也不能预先声称整步提速比例。

## 7. 用户运行 GPU benchmark

```bash
PY=/home/xueqi/hq/projects/tc-alignment/.venv/bin/python
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. \
  BFCL_PROJECT_ROOT=/tmp/rtd-v11-d12-bfcl \
  "$PY" tools/rtd_v11_generation_bench.py \
  --out results/rtd_v1_1/generation_bench_auto
```

继承用户设置的单张 `CUDA_VISIBLE_DEVICES`。默认加载配置中的本地真实 Qwen3.5-4B + 初始 LoRA；可用 `--model /local/checkpoint` 指定本地基座，或 `--window results/rtd_v1_1/V0/controls/windows/r1-s01.pt` 加载经过现有 hash 验证的已归档 frozen source LoRA。无教师采购、训练提交或权重下载。

对相同 40 个真实 support 槽位各采两条动作，两种模式各 80 条。该 manifest 的 40 个 parent 中 7 个 agentic parent 没有可运行 adapter，因此沿用 D5/D6 固定任务集的每 fold 20 槽位/显式重复规则，报告 33 个唯一状态、重复项和排除项，不能声称 40 个不同的可运行任务。

输出 `report.json` 和 hash-chained `compute.jsonl`，报告 wall time、实际 action tokens/s、generation 调用数、allocated/reserved 峰值、独立 rescoring 的时间/峰值、logprob agreement 及 complete/truncated 数。嵌套 phase 会重置 CUDA 峰值计数，因此脚本使用 journal 的祖先峰值累计器，不能读最后一次调用重置后的瞬时计数冒充全程峰值。生成计时不包含 rescoring。

`--kernels torch` 强制使用本地 torch convolution/delta-rule/gated-norm fallback；`--kernels auto` 使用环境已有 kernels。可用 `--order batched-first` 做反向顺序复测，每次选择新输出目录。任一模式逐动作 score guard 失败，或 batched generation allocated 峰值不小于 40 GB，脚本返回非零。

## 8. CPU 验证

```bash
PY=/home/xueqi/hq/projects/tc-alignment/.venv/bin/python
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. CUDA_VISIBLE_DEVICES='' \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  BFCL_PROJECT_ROOT=/tmp/rtd-v11-d12-bfcl \
  "$PY" -m pytest tests/test_rtd_v11_*.py -q -p no:cacheprovider
```

最终完整 v1.1 套件：**232 passed，1 个已有 warning，226.12 秒**。日志 `/tmp/rtd-v11-d12-final.log`。随后后端 identity/显式关闭、manifest 与原 failing-score journal 路径专项：**16 passed，15.77 秒**，日志 `/tmp/rtd-v11-d12-last-check.log`。

最后恢复了单动作旧路径在每次评分前退出 action-limit scope 的精确调用边界，并补核 benchmark 汇总：generation batching、score consistency、runtime memory 三组 **73 passed，19.23 秒**，日志 `/tmp/rtd-v11-d12-final-runtime.log`。

额外执行原 runtime memory、score consistency、checks、return-gradient 套件。首轮含两个失败：新增 StateStore fixture 少了 phase 字段，以及兼容 helper 给旧 monkeypatch 多传 temperature/top_p 关键字；两者已修复，原数值/恢复/内存路径其余 109 项通过。修复后 generation + score-consistency **50 passed**，并以上述最终完整套件和专项再次覆盖。不修改或跳过旧测试来掩盖失败。

GPU benchmark 的 CLI help 和 CPU agreement guard 已验证；真实 GPU 时间、吞吐、BF16/FLA 分数与内存峰值等待用户运行。
