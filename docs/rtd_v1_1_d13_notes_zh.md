# RTD v1.1 D13：无梯度 teacher-forced forward 批处理

仅修改 `/home/xueqi/hq/projects/tc-alignment-v11a`、分支 `rtd-v11-a`。`data/`、`envs/`、共享 Python 环境及生产 worktree 只读；没有 GPU 测量、训练、教师调用或 commit。

## 1. 接入范围及保留的梯度路径

本次采用任务允许的 **只批处理 no-grad forward** 分支。新实现位于 `src/bfas/rtd/forward_batch.py`，由 `runtime.HFGenerateBackend` 使用。

| 路径 | D13 行为 |
|---|---|
| `sample_state` 来源 Y1/Y2 的独立 teacher-forced 一致性检查 | 同 prompt 两行一起 forward；预算不足则拆批。 |
| `alpha_draw_pairs` 的 pilot / commit / old virtual reference 来源检查 | D12 生成预取结束后，对本 scope 的全部动作按长度分桶评分，再按原始顺序逐动作消费、记录、检查。 |
| `source_kl` 的 pilot 评估 | 每批分别在 frozen source 和 trial parameters 下 forward；完整词表条件 KL、逐动作求和及原动作顺序的均值保持。七个 trial 都走批处理。 |
| 独立 diagnostic / benchmark rescoring | `score_actions_batch` 或 `prefetch_scores` 批处理；D12 benchmark 的 `agreement` 也支持此路径。 |
| D1/D8 的 H1、H2、软项及教师梯度；blocked alpha VJP | 保持原单动作图及逐单位梯度。`source_scoring.py`、`source_estimator.py`、`alpha_d.py` 的数学实现未修改。 |
| REINFORCE 的 `checked_score_action` | 原评分 tensor 同时供应检查和梯度；不能用 detached 预取分数替代，继续单动作。 |
| RMS preconditioner | 统计量来自 **逐 source 的梯度平方**，不是独立的 no-grad LM 评估；保留逐 source 求导和原累加顺序。 |
| 初始 prompt 隐藏投影、greedy/parse 生成、环境续写 | 原路径。没有新增诊断采样或环境调用。 |

`source_teacher_forced_pair` 的“两次”指同一动作在 frozen/current 参数下的两个主干前向，不是 Y1/Y2。一对状态动作仍调用两次该函数。虽然其中 frozen forward 本身无梯度，本次仍让它与 current forward 保持相同单行布局：只批量 frozen 一侧可能使 BF16 下相等策略的 `q-p` 出现数值残差，破坏 D1 的精确零软梯度。

没有引入 per-row retained-graph backward 或 vmap。生产记录的单动作 `source_teacher_forced_pair` 峰值约 **23.474 GiB = 25.205 GB**，模型加载约 **8.497 GB**。仅用 `8.497 + 2×(25.205−8.497)` 作保守容量检查已约 **41.9 GB**，超过 40 GB；这不是精确显存模型，也不是双行实测。没有 GPU 时无法验证带 checkpoint replay 和多个 row backward 的峰值，因此不承诺把此图扩大到两行。本次仍只有一个 live gradient action graph，H1−H2、逐单位 v_i、教师项、估计器累加与更新 hash 计算均保持。

## 2. 配置、分桶及位置语义

规范 `v1_1_bfcl.yaml` 启用：

```yaml
generation_batch:
  prompts_per_batch: 8
  max_batch_tokens: 16384
  forward_prompts_per_batch: 8
```

新键默认 **0，关闭 forward 批处理**。缺少整个 `generation_batch` 时仍是原 v1.0 路径；旧 D12 配置缺少新键时只批量生成。v1.0 继续拒绝 `generation_batch`；旧配置规范化不会补入 `forward_prompts_per_batch: 0`。可显式设 0 回到 D12 的评分路径。

生成与 forward 共用 `length_bucketed_groups`：稳定长度排序、最多指定 prompt 请求数、长度比不超过 2、包括 padding 的 token 预算；超预算单行独立运行，不截断。生成仍按 effective action cap 分组，forward 使用完整 `prompt+action` 长度、`limit=0`。forward 相邻同 prompt 动作构成一个请求组，按需要拆分；不是按文本去重动作。分桶后用原 index 还原输出顺序。

输入从位置 0 开始放置完整 prompt 和 sampled action，末尾 **right padding**，attention mask 在真实长度内为 1，其后为 0。没有共享/复用 prompt KV cache；“同 prompt 一次 forward”是把两条完整序列作为同一次模型调用的两行。真实 token 的位置和 RoPE 与 batch 1 相同。

所有动作使用 `source_scoring.sampled_prefix_positions` 的验证和位置：prompt 长 p、动作长 n 时，取 logits `[p−1, p+n−1)`，对应 labels `[p, p+n)`。包括实际采到的 EOS；cap action 不补 EOS；不重新分词动作、不对长度平均、不评分 prompt 或 padding。CPU 测试覆盖 prompt 长度 1、不等长动作、混合 EOS/cap、超过 delta-rule 128-token chunk 边界及超过投影 chunk 边界。

通过临时 output-head pre-hook 截取主干隐藏状态，立即拷贝每行实际动作预测位置并释放 prompt/padding 的隐藏张量。head 每次最多投影 **32 个位置 × 完整词表**，避免分配 `batch × 7k × 248320` logits。词表大小 248320 时，一个此类 FP32 临时张量约 31.8 MB；KL 的多个词表临时张量只活在一个 chunk 中，不保存整批源分布。head 内的 LoRA 参数通过 `functional_call` 绑定，模型参数不安装/覆盖。此路径要求逐位置 output head，拒绝已知的 post-head softcapping。

本地 Qwen3.5 实现的 full attention 使用因果 mask；linear attention 对 padding hidden states 清零且状态按行独立，右侧 padding 不影响真实前缀。读取本地安装的 `modeling_qwen3_5.py` 确认 logits 直接来自 `lm_head`。GPU 的 FLA/causal-conv1d kernels 尚未在本次 sandbox 验证。

评分保持 native CE 和原逐动作 `values.sum()`。pilot 保持原表达式 `p*(expm1(logq−logp)−(logq−logp)).clamp_min(0)`；仅完整词表 reduction 按位置 chunk 累加，存在普通 FP32 reduction/batching 舍入差异，绝非更换 KL 定义。最后按原动作顺序 stack/mean，不按 batch 平均。

## 3. RNG、身份、journal、恢复

- forward 配置不进入 D12 的 **sampling identity**。`sampling_config()` 仍只包含原 `prompts_per_batch/max_batch_tokens`；生成实现、RNG tickets、batch seed、ActionTrace 字段和 sample/action/source hash 不因 forward 开关改变。配置及源码 provenance 正常反映新实现，不伪造整个 manifest/hash-chain 不变。
- shared planner 与本 worktree `HEAD` 的 D12 planner 做了 1000 组随机请求布局对照，结果逐组相同；另用真实 tiny Qwen 检验开关前后生成对象逐字段相等。
- 分数预取只保存 detached 标量/逐动作 token 分数，按 `(prompt IDs, action IDs, EOS, truncated)` 保存独立的消费队列；重复动作仍各消费一次。scope 绑定参数身份，错误参数、错误动作、重复消费、未完全消费都会报错。正常退出或异常后清空；不存入 StateStore、不跨步、不消耗 RNG。
- 原 `checked_score_action → score_diagnostic → journal → enforce` 顺序不变，失败动作仍先留完整诊断再报错。`source_sample`、`alpha_d_source_pair` 的相对顺序、draw IDs、采样对象/hash、RNG before/after 通过开关对照及 D7 中途失败恢复测试。
- `teacher_forced_forward` 的 compute 记录改为实际批量调用，附加 sequences、实际/填充 token 数、right padding 和 position chunk；pilot 的 forward_passes 是实际主干调用数。逻辑动作数不变，耗时/compute 条数/hash chain 自然变化。原 score diagnostic 的字段、dtype 元数据、EOS/mask 声明、容差及 hash 算法不变。
- 开着 autograd 的调用始终走原单动作评分，绝不从 no-grad 预取队列返回 detached 分数。批量 API 要求调用者显式处于 `torch.no_grad()`。

## 4. 生产证据和每步节省范围

按任务要求只读调用：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. CUDA_VISIBLE_DEVICES='' \
  /home/xueqi/hq/projects/tc-alignment/.venv/bin/python \
  tools/rtd_v11_phase_times.py \
  /home/xueqi/hq/projects/tc-alignment-v11prod3/results/rtd_v1_1/V0
```

第一次读取的 step 1 前缀为：generation **85 calls / 848.759 s**；source_teacher_forced_pair **240 / 1127.448 s**；pilot_conditional_kl **7 / 598.663 s**；teacher_forced_forward **440 / 500.856 s**。生产文件随后继续追加，后续增长不混入这里的基线。工具是 inclusive phase time，不能再把父级 phase 和这些子项相加。

从该前缀的 276 个来源诊断提取原 prompt/action IDs，在 CPU 上运行实际 D13 planner：

| no-grad 来源检查 | 原 forward 数 | D13 规划 forward 数 | 原评分时间 |
|---|---:|---:|---:|
| round setup（逐状态本地 Y1/Y2） | 36 | 18 | 39.689 s |
| pilot 80 actions | 80 | 7 | 65.863 s |
| commit 80 actions | 80 | 11 | 78.138 s |
| old virtual reference 80 actions | 80 | 14 | 81.579 s |
| 合计 | **276** | **50** | **265.269 s** |

setup 的全局离线 planner 虽可得到更少调用，实际接入仍逐状态，因此报告 18。每个 pilot trial 的 frozen/current 前向从 `80×2=160` 变为 `7×2=14`；七个 trial 合计从 **1120 降到 98** 次。独立试参仍各算 frozen side，没有跨 trial 储存完整词表概率。

直接覆盖的原 no-grad 耗时约 **863.932 s（14.4 min）**。若这些路径实测达到 2× / 4× 吞吐，对此记录前缀分别节省约 **432 s / 648 s（7.2 / 10.8 min）**；这只是基于证据的条件估算，不是 GPU 结果。简单按调用数线性缩放会给出约 761 s 的乐观节省，但忽略更大 batch 的单次成本，不能作承诺。**1127 s 的 source-gradient pair 成本、本来的 gradient teacher/REINFORCE/RMS 成本，以及 generation 不计入 D13 节省。**

默认 16384 padded tokens 限制长输入批大小，词表工作区有上述独立界限；仍需用真实 GPU 峰值检验 <40 GB。任意接近 context cap 的超预算单行仍运行完整输入，不属于该默认预算的普适显存保证。

## 5. GPU benchmark

新脚本复用 D12 的本地模型、LoRA、真实 40 support 槽位加载及 kernel 选项。40 槽位沿用 D5/D6 显式重复规则，不声称 40 个唯一可运行任务。两种评分方式使用 **同一组 80 动作、相同参数**。

```bash
PY=/home/xueqi/hq/projects/tc-alignment/.venv/bin/python
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. \
  BFCL_PROJECT_ROOT=/tmp/rtd-v11-d13-bfcl \
  "$PY" tools/rtd_v11_forward_bench.py \
  --out results/rtd_v1_1/forward_bench_auto
```

默认 fresh sampling；重用 D12 benchmark 动作时加：

```text
--actions results/rtd_v1_1/generation_bench_auto/compute.jsonl
```

重测 D13 已保存动作可加 `--actions .../actions.json`。若 D12 使用了 `--window`、自定义本地基座或生成 batch 配置，需提供相同参数；脚本验证原 backend/policy identity，拒绝把动作重新绑定到另一学生。`--window` 沿用原归档 hash 验证。无权重下载。

输出 `report.json`、精确动作 `actions.json`、逐 token 对照 `scores.json`、hash-chained `compute.jsonl`。报告 unbatched/batched wall time、实际 forward calls、allocated/reserved peak、每动作及全局 max abs logprob diff。计时包括评分和结果转 CPU，不含新动作生成及诊断 JSON 写入。峰值使用 journal 的祖先累计器，避免嵌套 phase 重置计数导致漏报。

默认严格 `--atol 1e-4`，任一 token 超限、非有限分数、覆盖不一致、原 per-action generation score guard 失败，或 batched allocated peak **≥40 GB**，均非零退出。FP32 tiny 测试使用 1e-4；真实 BF16 batching 可能超过这个严格阈值。可显式指定 GPU 对照容差（如 `--atol 0.05`），报告会记录该值，原 generation consistency guard 不会因此放宽。没有自动放宽容差。

可用 `--kernels torch` 验证 torch fallback，`--kernels auto` 使用本地已安装 kernels，`--order batched-first` 反向测序。每次使用新输出目录；只测本次无梯度评分，不把生成速度或梯度速度混入结论。

## 6. CPU 验证

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. CUDA_VISIBLE_DEVICES='' \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  BFCL_PROJECT_ROOT=/tmp/rtd-v11-d13-bfcl \
  /home/xueqi/hq/projects/tc-alignment/.venv/bin/python \
  -m pytest tests/test_rtd_v11_*.py -q -p no:cacheprovider
```

新增测试包含逐位置 FP32 1e-4、KL 对照及相等参数精确零、单 prompt 两动作合批、mask/cap/context/token budget、head 位置工作区、nonresident LoRA binding、checkpoint 下 H1/H2/软/教师梯度 **逐 tensor byte-identical**、forward 开关的 D12 identity/RNG/hash 对照、失败前 journal、队列单次消费、D7 midphase recovery、D12 动作导入及 benchmark 超限/显存阈值非零退出。D1 无偏性、D8 对称性、D9 测试文件未修改。

核心 D13 首轮修复后的 8 项：**8 passed / 7.89 s**。第一轮唯一失败是测试把 forward pre-hook 放在 PEFT 不经 `__call__` 的 base wrapper 上，未捕获调用；改挂顶层 PEFT module 后通过，没有放宽数值断言。D12 原套件 **13 passed / 14.50 s**。额外 runtime memory / score consistency / return gradient：**72 passed / 40.78 s**。

最终完整 `tests/test_rtd_v11_*.py`：**243 passed，1 个已有 warning，224.24 s**；包括全部 11 个新增 D13 测试。日志：`/tmp/rtd-v11-d13-full.log`、`/tmp/rtd-v11-d13-runtime.log`。CLI `--help` 和 `git diff --check` 已通过。GPU 吞吐、BF16/FLA 数值及显存峰值尚未测量。

## 6. GPU 基准结果与生产决定(2026-09-08,A100)

`tools/rtd_v11_forward_bench.py`(80 个真实动作,同参数):unbatched 80 次 forward 172.1 s → batched 13 次 53.0 s(3.25×),峰值 26.1 GB。但逐 token logprob 一致性 **未通过**:max |Δ| = 0.424 nat,62/80 个动作超过 1e-4 的目标(bf16 批量 forward 与单行 forward 的数值差异)。虽然仍在 score guard 的 1.0 nat/​token 容差之内,按"正确标准优先"的要求,**生产配置将 forward_prompts_per_batch 设为 0(关闭 D13 forward 批处理)**,只保留 D12 的生成批处理(其一致性检查通过)。代码保留,待有 fp32 评分或更严格的等价验证后再启用。
