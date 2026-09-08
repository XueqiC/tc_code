# C26-I：ALFWorld evaluation campaign 锁粒度热修复

本次仅修改 `/home/xueqi/hq/projects/tc-alignment-alf`，分支 `alfworld-c26f`，起始 HEAD `a8a41bdc7459a40a3510e4437f007e947ddfcdb5`。仅 CPU fixtures；没有修改其他 worktree、`data/`、`envs/` 或远端运行，没有提交。

## 问题与修复

用户提供的 HPG 现场现象：同一个 root 下，`results/c26f/hpg-R0`、`hpg-R1`、`hpg-R1s` 同时运行。三者都用 `round-1` 作为 tag，旧代码将 ALFWorld 锁绑定到 tag，导致等待 `764484/alfworld/round-1`，独立 arm 的评估被串行化，单次约 4.7 小时。本次没有重新读取远端日志，这些耗时与 PID 来自用户报告。

registry 实际写入的目录分别是 `<run dir>-alfworld-evaluations/round-N`，因此锁应保护这个目录。新公式为：

```text
campaign = (Path(output_root) / tag).resolve()
key = hashlib.sha256(str(campaign).encode()).hexdigest()
lock = root / "results/alfworld_std/.locks" / key / ".lock"
```

这里使用路径字符串 UTF-8 字节的 SHA-256。旧 ALFWorld 代码实际调用 `persistence.digest(tag)`，它先 JSON 序列化 tag；虽然与直接 SHA-256(tag) 的值不同，碰撞原因相同：同 root、同 tag 只有一把锁。

- 不同绝对 campaign 目录可以并发，包括不同父目录下同名 run。
- 同一目录的相对路径、绝对路径、symlink 别名得到同一把锁；同目录同 tag 的 resume/reuse 仍互斥。
- ALFWorld 的 `tag_lock_path` 必须提供 `output_root`，不能静默退回旧 tag-only 锁。tag 的路径穿越校验保留。
- holder JSON 保留 `pid`、`hostname`，tag 改为 `alfworld/<run dir name>/<tag>`，例如 `alfworld/hpg-R0/round-1`。独立 evaluator 没有训练目录时使用 output root 名称。训练 journal 的等待标签也带 run 名。
- lease 覆盖原有 guard、恢复、复用、完整评估与落盘区间；永久 lock inode、超时、轮询、等待统计与继承 descriptor 行为保留。
- BFCL tag hash、namespace、port lease 与调用行为保持原样。

## Hash 与审计链：此 checkout 的评分 hash 实际不变

核验发现，任务描述中“修改 evaluator 文件必然改变 harness hash”的前提不适用于当前 ALFWorld checkout。`alfworld_identity.py::evaluation_harness_identity()` 对 evaluator 使用已有的 `scoring_projection()`，仅绑定 episode、评分、模型/环境构造和科学 dispatch；`evaluate()` 的锁、输出路径、日志属于已经排除的运行逻辑。公共 `evaluation_lock.py` 不在 ALFWorld scoring inventory 中。此次没有修改 inventory、projection 规则或 identity 模块。

完整旧 scoring inventory 与修复后逐项相等：

| 项目 | 修复前后相同的 SHA-256 |
|---|---|
| evaluator scoring projection | `f3d9b3e168634e11df36d711817493d196b32e75969401f0a25bffa6aa5f6b14` |
| registry scoring projection | `deddd0a2d6c5f030a905dac75faec0c18a88ac286391494bde2e4c229e5c6ed8` |
| 原样保留的 ALFWorld identity 模块字节 | `f44c617dfa6f11b060f11bb16fefd562e80f2a07f7034b36da76e0b2423539c5` |

`configs/rtd/identity_evidence/c26i.json` 登记四个生产文件修复前后的 **源码字节 hash**、完整旧 scoring inventory、起始 revision，以及从该 revision 提取的旧 `evaluate()` coordinator。测试执行这个旧 coordinator 生成历史 campaign，再用新 evaluator 复用。该文件是审阅/回归证据，不是任意旧身份的授权列表。

对固定 data/model/tokenizer/environment/config，`evaluation_harness_hash_before == evaluation_harness_hash_after`。完整 harness hash 还绑定本机资产，不能把 CPU fixture 的 hash 当作 HPG 的 hash。因此本次没有人为制造一个新评分版本，也没有对未知历史 hash 添加白名单。已完成 `evaluation-N.json` / `campaign.json` 按原 identity 复用，旧 manifest 通过原 `guard_manifest` / `guard_harness`；C26-I 的 identity 链保持原端点。

每次 campaign 通过 `guard_manifest` 后，在 `audit.jsonl` 中追加 `identity_audit`，包括：

```text
reason = "C26-I lock granularity, no scoring change"
method = "guard_manifest/audited_harness_hashes"
manifest_hash, previous_harness_hash, current_harness_hash
audited_hashes, supplement_hash, campaign_directory, lock
gpu_seconds = gpu_reserved_seconds = 0
```

该记录说明已验证的 manifest/chain 和锁位置，不等于评估已经完成。C26-I 自身前后 hash 相同；显式传入既有 ALFWorld supplement 时，日志保留 `audited_harness_hashes()` 验证后的 newest-first 链和 supplement digest。历史评分复用仍追加 `evaluation_reuse_via_audited_identity`，保留原 scored identity 与完成文件字节。

v1.0.6/v1.0.7 文档中的 C25q/C25r/C25r-b 是 BFCL 的显式 identity migration / completed reuse 路径；当前 ALFWorld training 分支明确不接受 BFCL legacy supplement。原有 ALFWorld campaign 的 full-manifest-bound supplement 校验、多级链、断链拒绝、未知 hash 拒绝均保留并复验。没有改变 config/data/model/tokenizer/hardware/checkpoint/完整140题/artifact/aggregate 的任何 guard，也没有让训练自动接受评分漂移。训练源码 hash 会变化，生产 resume 仍使用既有 `--acknowledge-code-drift`；它不授权其他 identity 漂移。

## CPU 验证

新增测试 `tests/test_rtd_alfworld_lock_hotfix.py`：两个真实完整140题 fake-backend campaign 在 barrier 内同时持锁，`timeout=0` 且等待回调必须未调用；同名 run 不同父目录也并发。另有跨进程 flock、同 campaign/别名互斥、永久 inode、BFCL 原路径、旧 coordinator 完成文件复用、训练 receipt 复用、旧 manifest resume，以及伪造未知 hash 拒绝。

既有 `test_rtd_alfworld_evaluation.py` 的多次显式审计链复用、未审计历史评分拒绝、未完成历史 campaign 拒绝重生成及损坏数据检查继续执行；只调整新增 audit 日志和新锁参数的断言。操作字段不影响评分的既有 identity 测试增加替换确实发生的断言。

本地执行命令：

```bash
cd /home/xueqi/hq/projects/tc-alignment-alf
PY=/home/xueqi/hq/projects/tc-alignment/.venv/bin/python
export PYTHONPATH=src:. PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p results/c26i/bfcl-test-runtime results/c26i/tmp results/c26i/cache
export BFCL_PROJECT_ROOT="$PWD/results/c26i/bfcl-test-runtime"
export TMPDIR="$PWD/results/c26i/tmp" XDG_CACHE_HOME="$PWD/results/c26i/cache"
"$PY" -m pytest tests/test_rtd_*.py -q -ra -p no:cacheprovider \
  > results/c26i/pytest-full.log 2>&1
```

`BFCL_PROJECT_ROOT` 是已有的 BFCL 输出/锁目录设置，避免在只读 `envs/` 写 `.file_locks`。没有修改 BFCL 环境包。环境 opt-in 测试保持已有 `@pytest.mark.skipif`，reason 为 `opt in with RTD_ALFWORLD_REAL_TESTS=1; real read-only assets`，没有新增 skip/xfail。

最终完整执行 `tests/test_rtd_*.py`：**961 passed，4 skipped，0 failed，6 warnings，695.11 秒（11分35秒），退出码0**，共965项，日志 `results/c26i/pytest-full.log`。4项 skip 均为上述真实环境 opt-in；6项 warning 来自既有 torch 标量转换及 tiny PEFT 配置/保存检查。没有省略其他测试文件或增加环境之外的跳过条件。

定向 hotfix 测试 **8 passed，22.39 秒**，日志 `results/c26i/pytest-hotfix.log`；独立 CLI 修正后定向验证 **1 passed，6.60 秒**，日志 `results/c26i/pytest-cli.log`。这些测试均包含在最终完整套件中，不叠加为独立通过数。首次组合运行漏导入依赖 fixture，在新训练 receipt 测试 setup 报错；补齐 import 后上述8项全部通过，首次日志保留，不冒充成功验收。该组合还发现独立 CLI 的 tag 校验调用需要传入 output_root，现已修正；首次全量运行遇到同一错误后中断，保留为 `results/c26i/pytest-full-initial.log`，最终验收重新执行完整套件。

报告4个 Bash 命令块通过 `bash -n`，内嵌 HPG Python 通过编译检查；从 `/tmp` 以绝对路径实际检查现有 verify/audit 与 identity 工具的 CLI 参数。没有执行远端命令或读取 HPG 成果。rsync 清单与全部10个修改/新增文件逐项一致。最终 `git diff --check` 通过；`git status --short` 和 `git diff --stat` 分别保存为 `results/c26i/git-status.txt`、`results/c26i/git-diff-stat.txt`。默认 diff stat 只统计7个已跟踪修改文件，另3个新增文件由 status 列出；全部保持未提交。

## HPG 验证与启用（本次未远端执行）

同步前让旧 campaign writer 正常结束或退出；同一个 campaign 不应同时存在持旧 tag-only 锁和新目录锁的进程。已运行/等待中的 Python 进程保留旧模块，需重新启动才应用修复。不要删除 `.lock`；删除永久 inode 会破坏互斥。

以下在 HPG **从 `/tmp` 执行**。先将 `HPG_ALF_ROOT`、`HPG_RTD_PYTHON` 设置为实际绝对路径；本地 `/home/xueqi/.../.venv/bin/python` 不代表 HPG 安装路径。所有项目、bank、run 和工具参数均使用绝对路径。下方 bank 路径取两份现有 `*_hpg.yaml` 的 `replay_bank_path`；若某 run 保存了不同路径，以其原 manifest 绑定为准，不能重封银行来改变绑定。

```bash
cd /tmp
ALF_ROOT="${HPG_ALF_ROOT:?Set the absolute HPG project path}"
PY="${HPG_RTD_PYTHON:?Set the absolute HPG Python path}"
case "$ALF_ROOT" in /*) ;; *) exit 2 ;; esac
case "$PY" in /*) ;; *) exit 2 ;; esac
export ALF_ROOT PYTHONPATH="$ALF_ROOT/src:$ALF_ROOT"
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=''
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

# 现有 verify 工具的只读 audit 子命令；不重新 replay/seal bank。
"$PY" "$ALF_ROOT/tools/rtd_alfworld_verify.py" audit \
  --bank "$ALF_ROOT/data/rtd/v1_alfworld_c26_hpg"

# 现有 v1.0.7 identity 工具：只检查已完成 receipt 的链成员关系。
for arm in hpg-R0 hpg-R1 hpg-R1s; do
  run="$ALF_ROOT/results/c26f/$arm"
  if [ -f "$run/evaluation-1.json" ]; then
    "$PY" "$ALF_ROOT/tools/rtd_check_evaluation_identity.py" "$run" --round 1 || exit 1
  fi
done

# CPU 回归从 /tmp 使用绝对测试路径；运行产物置于 /tmp。
export BFCL_PROJECT_ROOT=/tmp/rtd-c26i-bfcl-runtime
mkdir -p "$BFCL_PROJECT_ROOT"
"$PY" -m pytest "$ALF_ROOT/tests/test_rtd_alfworld_lock_hotfix.py" \
  "$ALF_ROOT/tests/test_rtd_alfworld_identity.py" \
  "$ALF_ROOT/tests/test_rtd_alfworld_evaluation.py" -q -ra -p no:cacheprovider
```

bank audit 校验 sealed 文件，不证明生产模型推理完成；identity 工具只检查保存的链成员关系，不重新 hash 当前源码与全部 campaign artifacts。缺少 `evaluation-1.json` 的 arm 尚不能计为“已验证复用”。正常 evaluation/resume 仍须通过完整 guards。以下额外只读检查重算当前 harness guard，打印三把新锁和现有 owner：

```bash
"$PY" - <<'PY'
import json, os
from pathlib import Path
from bfas.rtd.identity import guard_harness, audited_harness_hashes
from bfas.rtd.benchmarks.alfworld_evaluation import tag_lock_path
root = Path(os.environ['ALF_ROOT']).resolve()
locks = []
for arm in ('hpg-R0', 'hpg-R1', 'hpg-R1s'):
    run = root / 'results/c26f' / arm
    manifest = json.loads((run / 'manifest.json').read_text())
    identities = guard_harness(root, run, manifest)
    output = run.with_name(run.name + '-alfworld-evaluations')
    lock = tag_lock_path(root, 'round-1', output_root=output)
    locks.append(lock)
    print(json.dumps(dict(run=str(run), lock=str(lock),
        audited_hashes=audited_harness_hashes(manifest, identities),
        holder=json.loads(lock.read_text()) if lock.is_file() else None)))
assert len(set(locks)) == 3
PY
```

holder JSON 在释放后仍保留，单独存在不证明锁仍被持有。新进程启动后，核对 `audit.jsonl` 的 C26-I `identity_audit`、各 arm 的不同 lock 路径、正确 owner tag，以及已完成 campaign 的 reuse 记录。CPU barrier fixture 证明锁并发；本次没有测量 HPG 三臂真实 GPU 吞吐或完成时间。

## 精确 rsync-safe 文件清单

仅同步以下10个文件；不使用 `--delete`，不同步 `results/`、`data/`、`envs/`、`.git` 或旧运行 manifest/supplement。

```text
configs/rtd/identity_evidence/c26i.json
docs/rtd_alfworld_c26i_lock_hotfix_zh.md
src/bfas/rtd/benchmarks/alfworld_evaluation.py
src/bfas/rtd/benchmarks/registry.py
src/bfas/rtd/evaluation_lock.py
tests/test_rtd_alfworld_evaluation.py
tests/test_rtd_alfworld_identity.py
tests/test_rtd_alfworld_integration.py
tests/test_rtd_alfworld_lock_hotfix.py
tools/rtd_alfworld_evaluate.py
```

从本 worktree 本地执行 dry run，远端目标必须指向已经确认的对应 ALF checkout：

```bash
cd /home/xueqi/hq/projects/tc-alignment-alf
cat > /tmp/c26i-rsync-files.txt <<'FILES'
configs/rtd/identity_evidence/c26i.json
docs/rtd_alfworld_c26i_lock_hotfix_zh.md
src/bfas/rtd/benchmarks/alfworld_evaluation.py
src/bfas/rtd/benchmarks/registry.py
src/bfas/rtd/evaluation_lock.py
tests/test_rtd_alfworld_evaluation.py
tests/test_rtd_alfworld_identity.py
tests/test_rtd_alfworld_integration.py
tests/test_rtd_alfworld_lock_hotfix.py
tools/rtd_alfworld_evaluate.py
FILES
rsync -anvi --files-from=/tmp/c26i-rsync-files.txt ./ \
  "${HPG_ALF_DEST:?Set host:/absolute/path/to/the/ALF/checkout}/"
```

审阅 dry run 后实际同步使用同一清单去掉 `-n`。生产目录若已存在本 checkout 没有的其他评分 hotfix，应先对照源码与 manifest 链；C26-I 证据不授权覆盖或接受未知评分版本。
