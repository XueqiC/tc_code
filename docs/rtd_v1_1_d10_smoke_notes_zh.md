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
