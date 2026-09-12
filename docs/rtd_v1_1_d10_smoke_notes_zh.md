# RTD v1.1 D10：BFCL fold 修复与真实 smoke 阻塞记录

**状态：代码修复已实现；三个真实 GPU smoke 尚未完成。** 当前执行会话没有暴露 NVIDIA 设备，严格按指定 UUID 启动的 V0 在硬件身份检查处失败。不能把 CPU 测试当作真实 smoke，也没有生成任何训练收益或 GPU 性能结论。

操作仅限 `/home/xueqi/hq/projects/tc-alignment-v11run`，分支 `rtd-v11-run`。已读方法 rev 3.1 §2–§4、D7/D8/D9 笔记与 `scripts/rtd_v11_run_hpg.slurm`。`data/`、`envs/`、`.venv` symlink 目标只读；不修改其他 worktree，不暂存、不提交。开始时已有未跟踪的 `.venv` symlink，保持原样。

## 1. 失败与修复

| 失败 / 发现 | 原因、处理与边界 |
|---|---|
| 原始 `logs/rtd_v11_smoke_V0.log`：`fold_roles` 内 `int(h, 16)` 对 dict 抛出 `TypeError` | BFCL `parents` 是 `{official_id, parent_hash, fold, demo_demand, generator_seed}` 记录列表。`src/bfas/rtd/conventions.py` 对记录使用显式 `fold`，只把 `parent_hash` 字符串写入两个轮转角色。非法 fold 拒绝，不回退猜测。 |
| 首轮新专项测试：预期 20/20，实际 22/18；1 failed、6 passed、16 deselected，2.27 s | 本工作树真实 manifest 为 m=40、fold0=22、fold1=18，且40条显式 fold 当前都恰好等于哈希奇偶。保留原始数据；测试分别覆盖真实文件的22/18，以及复制其完整记录形状后仅在临时 fixture 设置的40条20/20分配。另翻转显式 fold，证明 helper 不重新按 parity 分折。 |
| 新 manifest 测试加载配置时找不到 fixture 下的 `configs/rtd/v1_bfcl_c25.yaml`；针对性复现2 failed、5 passed、17 deselected，2.16 s | 现有 `manifest_inputs` fixture 已把 `cli.ROOT` 改为临时目录。仅在加载真实配置期间用局部 monkeypatch 恢复工作树 ROOT，随后恢复 fixture 的模型/硬件环境来检查 manifest 输出。修复后折专项8 passed、16 deselected，2.46 s，见 `logs/rtd_v11_d10_fold_tests.log`。这是新测试设置错误，没有修改生产路径来迁就 fixture。 |
| 修复后用指定完整环境重跑 V0：`ValueError: exactly one CUDA GPU is required`，exit 1 | `make_manifest → hardware_identity` 在模型加载前拒绝继续。当前主机名 `sn4622122543`，`/dev/nvidia*` 不存在，`nvidia-smi` exit 9；指定 UUID 下 `torch.cuda.is_available() == False`、`device_count() == 0`。venv 为 `torch 2.13.0+cu130`，编译架构含 `sm_120`，因此没有换 torch/venv，也没有换 GPU或放宽硬件检查。继续运行需要执行环境实际暴露指定 GPU。 |

真实 BFCL support SHA256：`1e374eddc6024488a22dee8c8d4427f5bec803baa45a16f9fdeab53ea7f3c571`。没有为满足20/20描述而修改 manifest 或封存 bank；原有 `BFCLSupport` 的内容/折一致性检查不变。

ALFWorld 路径已核对：`alfworld_config.manifest_section` 把验证后的 `support['parents']` 传给 helper；该对象是以 parent hash 为键的 dict。`alfworld_support.validate_support` 核对每组 `fold=parent_fold(hash)`，`parent_hashes` 也按 parity 选择。helper 对字符串迭代项保留 `int(hash, 16) % 2`，原135父组的79/56语义不变。

原失败目录及日志在重跑前移动到：

```text
_trash/rtd_v11_d10_observed_fold_failure/smoke_V0/
_trash/rtd_v11_d10_observed_fold_failure/rtd_v11_smoke_V0.log
```

本次失败目录 `results/rtd_v1_1/smoke_V0/` 保留原样，仅含 `.lock`；本次完整异常见 `logs/rtd_v11_smoke_V0.log`。若重试，仍须先移动目录和日志到新的 `_trash/` 子目录，不能删除或覆盖旧证据。

## 2. 真实 smoke 事实与未产生的指标

bank 审计成功：420个可用包，recorded 内容成本分母55,370 token（exact 21,203、estimated 34,167），公共 cap 总和344,064，累计授权上限5,537 / 13,843。它们是 bank 审计值，**不是本次消费**。

| 指标 | V0 本次真实尝试 | V1 / V2 |
|---|---|---|
| 完成的 commit / exposure units / source actions | 0 / 0 / 0；未到训练阶段 | 尚未启动 |
| teacher-evidence units / reference units | 未产生曝光记录；两者都未执行 | 未产生 |
| purchases / spend | 未创建 ledger、未发生 reserve/reveal 或消费 | 未产生 |
| d solver stats（迭代、残差、目标、边界、warmup） | 未进入求解器，N/A | N/A |
| `acquisition_reference_feedback` 参数 hash | 未执行，N/A | N/A |
| `same_batch_reference_feedback` 参数 hash | 未执行，N/A | N/A |
| `post_commit_feedback` 参数 hash | 未执行，N/A | N/A |
| 各 phase wall time / peak GPU memory | 在 `ComputeJournal` 创建前失败，没有 phase 计时或 CUDA 峰值，N/A | N/A |
| trajectory / manifest / journal / exposure_schedule | 均未写出；只有目录锁文件 | 未产生 |

V1 必须重放完成的 V0 `exposure_schedule.json`；目前没有合法输入。V2 也需要相同的指定可见 GPU。本次没有绕过依赖、伪造身份、减到2个曝光单元、免费解封教师包或用测试替身跑生产命令。完成标准仍是每臂1个commit、40单元/80来源动作、硬账本实购、完整日程及轨迹/manifest/journal。

## 3. 命令

本次实际执行的失败归档：

```bash
mkdir -p _trash/rtd_v11_d10_observed_fold_failure
mv results/rtd_v1_1/smoke_V0 _trash/rtd_v11_d10_observed_fold_failure/smoke_V0
mv logs/rtd_v11_smoke_V0.log _trash/rtd_v11_d10_observed_fold_failure/rtd_v11_smoke_V0.log
```

本次实际执行的完整环境与 V0 命令（按任务原样）：

```bash
cd /home/xueqi/hq/projects/tc-alignment-v11run && ROOT=$PWD && export PYTHONPATH="$ROOT/src:$ROOT" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 PYTHONDONTWRITEBYTECODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True XDG_CACHE_HOME="$ROOT/.cache/rtd-v11" VLLM_CACHE_ROOT="$ROOT/.cache/rtd-v11/vllm" TRITON_CACHE_DIR="$ROOT/.cache/rtd-v11/triton" TORCH_HOME="$ROOT/.cache/rtd-v11/torch" TORCHINDUCTOR_CACHE_DIR="$ROOT/.cache/rtd-v11/torchinductor" CUDA_CACHE_PATH="$ROOT/.cache/rtd-v11/nv" NUMBA_CACHE_DIR="$ROOT/.cache/rtd-v11/numba" MPLCONFIGDIR="$ROOT/.cache/rtd-v11/matplotlib" CUDA_VISIBLE_DEVICES=GPU-97762062-28db-d85e-cb46-294b082f94df
.venv/bin/python tools/rtd_experiment.py smoke --arm V0 --run-dir results/rtd_v1_1/smoke_V0 --config configs/rtd/v1_1_bfcl.yaml > logs/rtd_v11_smoke_V0.log 2>&1
```

GPU访问恢复后，先以新 `_trash/` 目录保留当前失败，再执行同一环境及 V0。以下命令**尚未执行**，应在 V0 完成后，使用完全相同的环境和 GPU UUID：

```bash
.venv/bin/python tools/rtd_experiment.py smoke --arm V1 --replay-schedule results/rtd_v1_1/smoke_V0 --run-dir results/rtd_v1_1/smoke_V1 --config configs/rtd/v1_1_bfcl.yaml > logs/rtd_v11_smoke_V1.log 2>&1
.venv/bin/python tools/rtd_experiment.py smoke --arm V2 --run-dir results/rtd_v1_1/smoke_V2 --config configs/rtd/v1_1_bfcl.yaml > logs/rtd_v11_smoke_V2.log 2>&1
```

最终 CPU 验证命令：

```bash
PYTHONPATH=src:. CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  BFCL_PROJECT_ROOT=/tmp/rtd-v11-d10-bfcl-final \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m pytest tests/test_rtd_v11_*.py -q -p no:cacheprovider \
  > logs/rtd_v11_d10_tests.log 2>&1
```

第一次完整套件（测试 fixture 修复前）：172 passed、2 failed、1 warning，166.05 s；两项失败均是上文的临时 ROOT 配置缺失。日志保留为 `logs/rtd_v11_d10_tests_before_fixture_fix.log`。警告来自已有 `test_rtd_v11_acquisition.py` 的 requires-grad tensor 转 scalar。

最终完整 `tests/test_rtd_v11_*.py`：**174 passed、1 warning，176.32 s，exit 0**；没有 skip/xfail。完整日志：`logs/rtd_v11_d10_tests.log`。原 v1.0 `test_hard2_config_manifest_and_reference_gradient_are_byte_identical` 通过，ALFWorld 79/56角色断言通过，真实BFCL及40记录20/20 fixture 的 manifest 输出检查通过。警告仍是上述已有 tensor 转 scalar。

生产改动仅限 `fold_roles`；v1.0配置、运算及历史字节 oracle 均未修改。`git diff --check` 通过；所有 configs 与 ALFWorld support/config/state 文件均无 diff。最终 `git status --short` 为两项 tracked 修改（conventions 与测试）、新 D10 笔记，以及开始时已有的未跟踪 `.venv`。tracked `git diff --stat`：2 files changed、49 insertions、2 deletions；未跟踪笔记不在该 stat 中。没有暂存或提交。

## 4. D10b：可配置 smoke 时限与阶段耗时（2026-09-08；无 GPU 执行）

**后续 rai 真实尝试已通过 manifest / hardware identity、bank audit，并完成4笔 acquisition 与 pilot 的80次来源动作采样，但仍未完成 pilot 校准和第一个 commit。** 本节补充上文较早的设备不可见记录。此次只在 `tc-alignment-v11run` / `rtd-v11-run` 修改代码和文档、执行小规模 CPU 检查；`data/`、`envs/` 和 `.venv` 目标只读，没有操作其他 worktree，没有暂存或提交。开始时已有未跟踪 `.cache/`，保持原样。

原始证据保持原样：

```text
_trash/rtd_v11_d10_smoke_timeout/smoke_V0/compute.jsonl
_trash/rtd_v11_d10_smoke_timeout/rtd_v11_smoke_V0.log
```

compute.jsonl SHA256：`bbc0bc4ba69d1ad565138f8498493fde683dd3c82fb76cb1d492db2cfaadd584`。旧异常是 `TimeoutError('smoke exceeded 15 minutes; resume state retained')`，由 CLI 的900秒 SIGALRM 触发。日志有40个 `role=pilot` 的 `alpha_d_source_pair`（80个来源动作），但 `alpha_d_pilot` 的结束状态为 **failed**；822.913秒是超时前累计耗时，不是成功完成校准的耗时。校准的 reference 构建中断，不能把已有80次采样当作 pilot 或 commit 完成。

### 时限与 journal 修改

- CLI 增加正整数 `--smoke-deadline-seconds`，默认 **900**。同一值写入 `smoke_override.max_seconds`，用于从 `run_command` 开始计时的绝对 deadline、SIGALRM、journal 入口超时检查和最终 elapsed 检查；超时消息显示配置的秒数。默认 smoke 的原有 override 数值保持不变。
- `scripts/rtd_v11_run_hpg.slurm` 在 `RTD_COMMAND=smoke` 时传入 **7200**；run/resume 不自动加此参数。v1.1 的 E=40单元/80来源动作、独立 pilot、三种 feedback 角色，以及 V1/V2 的 validation 工作量显著高于 v1.0 的小型 smoke，900秒不能覆盖此次真实执行；7200秒仅放宽运行时限，不改变任何曝光、采样或反馈规则，也不保证共享 GPU 上一定按时完成。
- resume 仍要求配置与保存的 manifest 一致；恢复以7200秒启动的 smoke 时需显式传 `--smoke-deadline-seconds 7200`。本任务没有修改已归档900秒 manifest 或绕过 resume 身份/配置检查，没有执行真实 smoke 重跑。
- `ComputeJournal.measure()` 已统一把 `operation` 从 begin 写入 end；旧证据的 **226条 compute_end 均有 operation，且与 begin_sequence 指向的 begin 一致**。`alpha_d_pilot`、`generation`、`source_teacher_forced_pair`、`teacher_forced_forward`、`acquisition_reference_feedback`、`same_batch_reference_feedback`、`post_commit_feedback` 及 `validation_*` 已有对应 scope。此次为 alpha-d 和旧 v1.1 batch 分支的实际学生参数安装补充 `operation=commit` scope；alpha-d scope 还包含参数快照、commit 事件及保存到 `actual` 恢复点。v1.0 执行路径不增加此 scope。
- 新工具 `tools/rtd_v11_phase_times.py <run dir>` 仅以标准库只读解析 JSONL，按 operation 汇总次数、wall/gpu 秒数和 allocated/reserved 内存峰值；不实例化可能修复 torn journal 的 `ComputeJournal`。

实际执行的汇总命令：

```bash
PY=.venv/bin/python
PYTHONPATH=src:. PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' \
  "$PY" tools/rtd_v11_phase_times.py _trash/rtd_v11_d10_smoke_timeout/smoke_V0
```

| operation | n | wall s | gpu s | peak allocated GiB | peak reserved GiB |
|---|---:|---:|---:|---:|---:|
| alpha_d_pilot | 1 | 822.913 | 822.913 | 9.906 | 10.100 |
| generation | 84 | 488.136 | 488.136 | 8.631 | 8.766 |
| initial_hidden | 4 | 1.477 | 1.477 | 8.537 | 8.557 |
| model_load | 1 | 4.018 | 4.018 | 7.913 | 7.922 |
| round_sources_and_geometry | 1 | 42.037 | 42.037 | 9.036 | 9.225 |
| source_preconditioner | 1 | 13.530 | 13.530 | 9.036 | 9.225 |
| source_sampling | 1 | 27.089 | 27.089 | 8.537 | 8.576 |
| source_teacher_forced_pair | 30 | 101.714 | 101.714 | 9.773 | 9.932 |
| teacher_forced_forward | 103 | 97.863 | 97.863 | 9.694 | 9.758 |

计数包含失败尝试；表中唯一失败项是 `alpha_d_pilot`。wall 是包含子 scope 的耗时，嵌套行不能直接相加；CUDA journal 的 gpu 秒数等于同步边界下的 wall 秒数，**不是 CUDA kernel 活跃时间或 GPU 利用率**。内存列对各次记录取最大值，GiB=2^30 bytes。旧尝试没有进入三种 feedback、validation 或 commit，因此表中没有这些行，不能填0或推断完成。V0 本身也按现有协议不执行 paired validation。

### 观测吞吐与 batch size 的决定位置（只报告）

84次 `generation` 共记录 **6,394 action tokens / 488.136秒 = 13.10 tokens/s**；平均 **76.12 tokens/调用、5.81秒/调用**。其中 source sampling 为4次、162 tokens、15.857秒；pilot 为80次、6,232 tokens、472.279秒，即77.90 tokens/调用、5.90秒/调用、13.20 tokens/s。此约13 tokens/s 是各次单序列调用的累计 token/累计 generation scope 时间，包含该 scope 的调用开销，不能代表整步吞吐。按此次运行背景，GPU 当时与另一用户的作业共享；本任务未重新测量独占 GPU 性能。

source sampling 和 pilot generation 的 **实际 HF generation batch size 都是1**，决定位置为：

- `src/bfas/rtd/runtime.py` → `HFGenerateBackend.sample_action()`：`num_return_sequences=1`，`input_ids=torch.tensor([prompt_ids])`，attention mask 形状 `(1, len(prompt_ids))`，每次 `model.generate()` 只有一个 prompt。
- `src/bfas/rtd/experiment.py` → `sample_state()`：按 `source_samples_per_state` 顺序调用 `sample_action()`，当前为2次独立调用；`pool()` 在 `batches(..., 'source_states')` 内仍逐状态调用 `sample_state()`。
- `src/bfas/rtd/experiment_alpha_d.py` → `alpha_calibrate()` / `alpha_draw_pairs()`：pilot 顺序遍历40个 exposure records，每条执行 `sample_state(..., refresh=True, cache=False)` 获取2个新动作，没有把80个动作组成一次 generation batch。
- `src/bfas/rtd/memory.py` → `MemoryPolicy` / `memory_batch_size()` / `memory_batches()` 决定外层状态调度块大小；它与 HF generation batch 是不同层次。旧 journal 的两条 `source_states` memory_batch 均为1，四条 `preconditioner_sources` 也均为1。首条状态调度记录虽设置 `max_batch_size=2`，但可用预算28,560,522,240 bytes、单状态估计16,000,000,000 bytes，只能选1。

没有修改这些生成/调度文件、配置、RNG、协议或 batching；上述位置供后续任务评估。

### CPU 验证

首批 **27 passed，7.30秒**：CLI 默认/覆盖/非法值，900与7200秒的 config/journal/alarm 一致性及消息，journal 过期检查，只读 helper 的累计耗时/失败计数/内存最大值，launcher 的 smoke 参数、现有 HF generation/phase-memory 检查，以及真实 CPU executor 的 pilot、三种 feedback、四种 validation 和 commit 的 compute_end 名称。包含两个 v1.0 回归：`test_hard2_config_manifest_and_reference_gradient_are_byte_identical` 与 `test_v10_one_window_byte_identical_to_prechange_cpu_fixture`。日志：`/tmp/rtd-v11-d10b-tests.log`。

```bash
PY=.venv/bin/python
export PYTHONPATH=src:. PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=''
export BFCL_PROJECT_ROOT=/tmp/rtd-v11-d10b-bfcl XDG_CACHE_HOME=/tmp/rtd-v11-d10b-cache
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
"$PY" -m pytest -q -p no:cacheprovider \
  tests/test_rtd_v11_smoke_deadline.py \
  tests/test_rtd_v11_alpha_d.py::test_runner_acquire_freeze_sample_same_batch_commit_then_alpha_feedback \
  tests/test_rtd_v11_conventions.py::test_launcher_argv_cache_and_resume_relay_without_launching_gpu \
  tests/test_rtd_capped_sampling.py \
  tests/test_rtd_v11_source_estimator.py::test_hard2_config_manifest_and_reference_gradient_are_byte_identical \
  tests/test_rtd_v11_acquisition.py::test_v10_one_window_byte_identical_to_prechange_cpu_fixture
```

追加 commit/recovery 检查 **10 passed，17.66秒**（同一 CPU 环境）：`test_resume_preserves_same_draws_single_commit_and_posterior` 的6个恢复点，以及 `test_batch_estimator_commit_and_vjp_use_identical_exposure_and_controls` 的4个旧 v1.1 estimator 组合。日志：`/tmp/rtd-v11-d10b-recovery-tests.log`。合计 **37 passed**，没有启动模型服务或真实 GPU 工作。归档 helper 输出如上；`bash -n scripts/rtd_v11_run_hpg.slurm`、`git diff --check` 通过。最终使用 `git status --short` / `git diff --stat` 检查；新增 helper 和专项测试未暂存，因而不计入普通 tracked diff stat。
