# RTD v1.1 D14：流式曝光日程回放

本次仅改执行与完整性记录，不改 α/d、采购机制、损失、采样或预算。工作范围为当前 `tc-alignment-v11a` / `rtd-v11-a`；`data/`、`envs/` 和共享 Python 环境只读。仅执行 CPU 测试，没有 GPU、训练作业提交或 git commit。

## 1. 完整模式与 streaming 模式

D7 §3 的默认行为保留：`--replay-schedule <V0 run dir>` 等价于 complete 模式，开始时必须已有完整 V0 日程，config/manifest 绑定整个文件 SHA256。旧的无逐步 hash 日程仍可在 complete 模式使用；完整日程若携带逐步 hash，也会验证其内容。

新增入口：

```text
--replay-schedule <V0 run dir> --replay-mode streaming
--replay-poll-seconds 60
--replay-timeout-seconds 129600
```

YAML 对应 `replay_mode`、`replay_poll_seconds`、`replay_timeout_seconds`。只用于 V1；轮询/超时必须为正的有限秒数。省略 CLI 参数时使用配置值，resume 默认完整读取保存的配置；显式重复传入的参数必须与保存配置一致。Coordinator 会把 streaming 参数传给首次训练 worker，后续 worker 从 manifest 恢复。

V0 的目录存在即可启动 V1，不必等首次导出。V1 先保存可恢复的初始状态，然后等待有效的身份块。身份仍绑定 bank public/integrity、数据、基座、初始学生参数、support parents、预算、seed、rounds、E/K 与 smoke 模式。预期身份计算一次，后续每次读到的快照必须保持相同身份。

在 V1 的 `(round, step)` 开始、采购和当步来源采样之前，只等待对应的已提交 V0 步。V0 导出一步，V1 就可以消费一步，不等后续 23 步。V1 可与 V0 的下一步并行；这允许一拍滞后，不要求严格同步或限制 V0 领先步数。

曝光 payload 进入原有回放路径：采购窗口、实购集合、自身硬账本、入池前后成员、state/teacher/new 标记、package 权重、重复次数和 reference records 检查均保留。来源动作仍来自 V1 自己的冻结学生和独立 RNG。日程不含 α、d、来源文本/token 或 rollout。

## 2. 逐步完整性与恢复

每个 V0 导出 step 新增 `content_hash`：使用现有 `persistence.digest`，对删除 `content_hash` 后的完整 step 对象计算 SHA256，即排序键、禁止 NaN 的 JSON 表示。hash 包括采购、曝光和参考记录；不包含整个日程的 `complete` 标志或后续步。

hash 只加在导出层，不写入 durable training row。V0 恢复时立即从 durable committed steps 重建日程，旧步 hash 不变，也能重建丢失的辅助 JSON；后续导出只扩展这个前缀。V2 的导出内容保持原样。V1 导出的曝光日程同样携带 step hash，方便逐项比较 V0/V1。

Streaming 读取规则：

- 从一次 `read_bytes()` 返回的同一份字节解析 JSON 和计算最终文件 SHA256，不将不同原子替换版本的读/hash 混用。
- 文件尚不存在、JSON 截断、UTF 解码失败均记等待并重试。合法 JSON 中身份不符、步骤缺失/重复/乱序、hash 缺失或内容不符是硬错误；不会当作暂时未完成而绕过。
- 步序列必须是预期 `12 × rounds`（smoke 为 1 步）的连续前缀；采购仅允许出现在 1/4/7/10 步。
- 消费前先 fsync `replay_step_consumed` journal 事件，再原子更新 manifest，然后返回原始曝光 payload 供训练。跨崩溃已领取的步保持相同 hash，允许从其 durable phase 重做未完成部分。
- 每次快照核对所有已消费 hash，V0 重启后若已消费步被改写或从有效文件中消失，立即拒绝。resume 在任何训练工作之前还会核对 checkpoint 中已提交/正在执行的曝光与消费记录。
- journal 是 hash chain；manifest 是其原子投影。崩溃导致 journal 领先 manifest 时可补齐投影；manifest 领先 journal 或已记录内容相冲突会拒绝恢复。

结束时必须再次取得 `complete=true`、全部步骤齐全且每个已消费 hash 完全相同的最终文件。通过后才发布最后一轮 checkpoint；避免 coordinator 以“checkpoint 已存在”为由跳过最终检查。已经完成的 V1 再次 resume，包括仅剩 evaluation/report 的 coordinator，也会重新验证最终日程和整个文件 hash。

不一致通过 `replay_schedule_diff` 写入 `compute.jsonl`，含原因、字段/步号、expected 与 actual；随后抛出 `ValueError`，不继续训练或发布最终结果。若有效文件删除已消费步，也按不一致处理。格式变化不影响逐步 hash，但在最终整文件绑定后仍会触发文件 hash 不一致。

## 3. Manifest 与 checkpoint 绑定

Streaming V1 的 config 不保存开始时的整个文件 hash。manifest 记录：

| 字段 | 内容 |
|---|---|
| `replay_mode` | `streaming` |
| `replay_schedule_identity` | 初始化时固定的预期身份 |
| `replay_consumed_steps` | 按顺序保存 `{round, step, content_hash}` |
| `replay_schedule_hash` | 运行中为 `null`，最终校验通过后为完整文件 SHA256 |

`manifest_hash` 仅对 streaming V1 排除两个可变进度字段：`replay_consumed_steps` 与 `replay_schedule_hash`。固定身份、模式和整个 config 仍参与 checkpoint 绑定；进度由消费 journal 与恢复校验约束。StateStore、round checkpoint、checkpoint 验证、controls archive/配套报告和 report 使用同一个绑定函数，所以第二轮进度变化不会使第一轮 checkpoint 或之前的 controls 报告失效。controls 归档仍通过整个 `.pt` 文件 SHA256 保护其保存时的完整 payload（包括当时的 manifest）。

其他运行，包括 complete V1、V0、V2 和 legacy，仍逐字使用原来的整个 manifest digest。`experiment_alpha_d.py` 和 launcher 文件没有修改。共享执行入口的新增分支仅供 streaming V1；V0 的例外仅是要求的导出 hash 与恢复时重导出，不改变 durable training rows。

## 4. 等待计费、超时与 HPG 启动

每次等待写入 `compute_begin` / `compute_end`，`operation=replay_schedule_wait`、`compute_type=idle`，记录原因、目标步、时间戳与实际 `wall_seconds`；`gpu_seconds=0`。它是独立的外层计费事件，不使用会把 sleep 计为 GPU 计算的 CUDA measure。失败/中断的等待同样保留 journal；stdout 会显示等待原因。

默认轮询间隔 60 秒。默认总体超时为每次执行尝试的 36 小时墙钟 deadline，所有逐步等待、断读重试和最终 complete 等待共用，不因新一步而重置；训练耗时也经过这个 deadline。剩余超时不足一次轮询时缩短最后一次 sleep。超时会写 `replay_schedule_timeout` 并明确提示 V0 可能停止、等待的文件/步以及可恢复的 V1 状态。resume 是新的执行尝试，重新计时并先检查已有消费记录；coordinator 的每个训练 worker 各自属于一次执行尝试。

从本 worktree 根目录提交，V0 使用支持逐步 hash 的导出版本。以下只是启动说明，本次未执行 sbatch：

```bash
mkdir -p logs results/rtd_v1_1/V0
sbatch --export=ALL,RTD_RUN_DIR=results/rtd_v1_1/V0 \
  scripts/rtd_v11_run_hpg.slurm V0

RTD_EXTRA_ARGS="--replay-schedule $PWD/results/rtd_v1_1/V0 --replay-mode streaming --replay-poll-seconds 60 --replay-timeout-seconds 129600" \
  sbatch --export=ALL,RTD_RUN_DIR=results/rtd_v1_1/V1 \
  scripts/rtd_v11_run_hpg.slurm V1
```

V1 不带 `afterok` 依赖。`RTD_EXTRA_ARGS` 仍按 launcher 原有规则用空白分词、不 eval；路径避免包含空格，或用 launcher 位置参数直接传递已引用的路径。launcher 的 GPU/CPU/内存/24h 时限和 auto/run/resume 行为均未改变。

如果 V0 死亡，V1 在下一次缺步/最终完成等待时持续轮询，达到本次执行的总体 deadline 后失败退出。HPG 脚本的 24h 作业时限可能早于默认 36h deadline，届时先由 Slurm 结束作业，durable 状态仍可恢复。V0 续跑并有进展/完成后重新提交 V1：

```bash
sbatch --export=ALL,RTD_RUN_DIR=results/rtd_v1_1/V1 \
  scripts/rtd_v11_run_hpg.slurm V1
```

auto 检测到 manifest 后使用 resume，不必再提供回放参数。已有的 complete V1 不在原目录切换 streaming；需要单独的新 V1 运行目录。D14 之前启动且仍持有旧代码的 V0 进程不会被热更新；其无逐步 hash 日程不能用于 streaming。使用 D14 导出需在正常续跑时加载新代码，现有代码漂移检查仍可能要求 `--acknowledge-code-drift`，不会自动绕过。

## 5. CPU 验证

新文件 `tests/test_rtd_v11_streaming_replay.py` 覆盖：24 步两轮 lag-1 实际 tiny engine、非均匀权重与重复曝光、V1 独立来源及与 complete V1 的相同最终参数、torn read、首个导出缺失、共享总体 timeout、等待零 GPU 计费、最终屏障、逐步/身份/最终文件分歧、各 durable phase 恢复、journal/manifest 崩溃窗口、V0 重新导出、历史 complete 模式、CLI/coordinator/launcher 参数传递、已完成 coordinator 的恢复验证，以及 controls archive/报告跨进度更新保持有效。

```bash
PY=/home/xueqi/hq/projects/tc-alignment/.venv/bin/python
PYTHONPATH=src:. CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  BFCL_PROJECT_ROOT=/tmp/rtd-v11-d14-full-final-bfcl \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  "$PY" -m pytest tests/test_rtd_*.py -q -p no:cacheprovider
```

最终完整回归：**1166 passed，6 warnings，683.93 秒，exit code 0**，无 skip/xfail。日志 `/tmp/rtd-v11-d14-full-final.log`。六条警告来自已有 tensor 标量转换与 PEFT 配置提示。D14 新文件共 34 项测试。

已完成的验证与发现：

- conventions/smoke deadline 回归：**38 passed**，日志 `/tmp/rtd-v11-d14-existing.log`。
- 新 streaming 专项首轮：**29 passed**；补充 coordinator/launcher/controls 后，与 metrics/controls、evaluation resume、manifest tolerance 一起验证：**121 passed，4 warnings**，135.73 秒，日志 `/tmp/rtd-v11-d14-final-targeted.log`。
- 首轮全套：**1164 passed，1 failed，6 warnings**，672.27 秒，日志 `/tmp/rtd-v11-d14-full.log`。唯一失败是 D12 测试仍期待生产 `forward_prompts_per_batch=8`；仓库 HEAD 的 YAML 已明确因 bf16 benchmark 关闭此路径、设为 0，HEAD loader 也确实规范化为省略关闭字段。本次仅修正旧测试期望，并加断言确认有效值为 0；没有修改 YAML、generation/forward 实现或生产行为。修正后的 generation 文件：**13 passed**，日志 `/tmp/rtd-v11-d14-generation-final.log`。
- 对当前 worktree 的 `git show HEAD:<path>` 源码使用内存 import loader 做独立前后对照，没有检出/修改其他 worktree。V0 全 24 步、V2 前 4 步（包含 1/4 两个采购窗口）的 training rows JSON **逐字节相同**，最终参数 hash、采购账单和配置相同；V2 日程文件逐字节相同，V0 导出仅增加逐步 hash。脚本 `/tmp/rtd_v11_d14_byte_oracle.py`，通过日志 `/tmp/rtd-v11-d14-byte-current.log`。
- 对照最初尝试运行无额外约束的 tiny V2 24 步，在 **原始 HEAD** 第 1 轮第 10 步遇到现有 `illegal alpha/d probability transfer`；保留 `/tmp/rtd-v11-d14-byte-head.log`，未改此机制问题。上述 V2 字节一致性证据只覆盖前 4 步，不声称该合成 24 步运行成功。

不暂存、不提交；结束时执行 `git diff --check`、`git status --short` 和 `git diff --stat`。
