# C26-F：ALFWorld RTD v1 分派集成验收报告

日期：2026-09-07。工作目录：`/home/xueqi/hq/projects/tc-alignment-alf`；分支：`alfworld-c26f`。本次接续已有未提交改动，未提交 git，未修改活跃工作树的源码、运行目录或日志。`data/`、`envs/` 仅经只读链接读取；所有验收产物在本 worktree 的 `results/c26f/`。使用指定的共享 Python 可执行文件，设置 `PYTHONDONTWRITEBYTECODE=1`，未加载 GPU 模型、运行 GPU smoke 或提交 Slurm 作业。

## 1. 实现与计划对应

ALFWorld 已接入原 `tools/rtd_experiment.py` 的 audit/smoke/run/resume/evaluate/report；训练仍只有 `RTDExperiment`，生成和原生 CE 仍使用同一个 HF 模型。正式配置保持 `protocol_version: 1.0.1`、3 轮 × 12 步、每轮 4 个决策窗口、M=4/K=2、8 槽、同任务 LOO、epsilon=.25。正式两份 YAML 的科学字段没有改写；scalar 文件仅 gate 不同。

| 文件 | 最终改动与约束 |
|---|---|
| `src/bfas/rtd/cli.py` | 按 benchmark 分派配置、data identity、bank audit、manifest、support 和 evaluator。ALF 保留 estimated/unknown 成本；拒绝将 inventory `--build-bank` 当 verified bank。ALF 使用 nullcontext，BFCL CheckerBridge 保留。新增显式 smoke 父组数选项，resume 使用保存配置并拒绝漂移。scalar swap 只改 gate；ALF 拒绝 BFCL 的 `update-identity` 路径。 |
| `src/bfas/rtd/experiment.py` | 保存 providers、复用通用 broker；ALF source 使用 agent_action cap，pool 在 source/P/normalizer/chi 前校验 reset 和完整已购映射。恢复时按 ledger reveal 顺序恢复已付费包，不重复收费。所有合法 feedback parents 均可抽样，含失败/no-payload parents；保存 `(parent,K)`、episode seed 和诊断。ALF smoke 保持 LOO/8 槽，插入参考仍走正式数学检查。 |
| `src/bfas/rtd/return_gradient.py` | 注册 agent_action=256，校验 caps 和类别，异常后恢复 action cap；BFCL 原 backend identity 构造和 single/multi cap 行为保留。未修改 `bfcl_task_rollout`、`ActionTrace`、`TaskRollout`、`reinforce_gradient` 或 Gate/VJP 数学。 |
| `src/bfas/rtd/runtime.py` | 在模型分配前检查 benchmark/caps；ALF 设置 `max_action_tokens=256`、`action_caps={'agent_action':256}`。复用 score tolerance、memory policy、LoRA、streamed gradient 和单模型路径。 |
| `src/bfas/rtd/evaluation.py` | ALF 分派到原生 campaign；report 校验完整 checkpoint、140题和逐题 artifact hashes，输出 success_rate、success_percent、分类成功、repair/damage（有合法 base 时）、估计成本与缺失计数、可识别反馈、失败/fallback/ACTION/cap/token/episode、wall-time 及通用 code_drift journal。BFCL 报告字段及评分读取保留；不同 benchmark 不组成配对比较。 |
| `src/bfas/rtd/evaluation_lock.py` | `tag_lock_path(..., benchmark='bfcl')` 的默认路径/hash 不变；ALF 使用 C26-D namespace。拒绝路径穿越；物理 port 仍全局互斥。 |
| `src/bfas/rtd/identity.py` | ALF harness 分派，先验证保存 config/hash/harness binding；首轮 F 严格拒绝 harness 漂移，不读取 BFCL legacy supplement。ALF audited hashes 使用自身 full-manifest-bound 链验证。保留通用 hardware/source/resume/checkpoint 校验。`audit_legacy` 仅 BFCL。 |
| `src/bfas/rtd/scoring_scope.py` | 保留旧双参数 projection，增加 benchmark dispatcher 和默认 BFCL 的 hash 参数；旧 BFCL AST/scoring hash 保持。 |
| `src/bfas/rtd/caps.py` | 增加 `affordability_for` / `cap_policy_for`；原 public_cap/affordability 的签名、常量和输出不变。 |
| `src/bfas/rtd/selector.py` | 将 ALF bank/state/support/rollout/evaluation/identity/config/registry 和原始 adapter 加入真实 import denylist；保留 import/importlib、read/listdir/subprocess/socket guard。仍是 cooperative guard，不声称 OS sandbox。 |
| `src/bfas/rtd/benchmarks/alfworld_config.py` | 仅允许显式、完整且有类型校验的 smoke_override：每折 2 或 4 父组，8 槽、K=2、1 窗口、LOO、900秒。正式默认值保持。 |
| `src/bfas/rtd/benchmarks/alfworld_evaluation.py` | 原生锁委托统一 ALF tag path，传递 timeout、log interval 和 wait callback，支持实际等待统计。 |
| `src/bfas/rtd/benchmarks/alfworld_identity.py` | 把 registry 的科学适配调用纳入 ALF AST projection；缺失/重复定义失败，锁与日志参数不改变评分投影。 |
| `src/bfas/rtd/benchmarks/registry.py` | ALF evaluation 依次执行 data/hardware/harness/source/checkpoint guard，使用 sibling campaign，记录无效 port 参数及等待/评估时间，校验原生 receipt 后复用。 |
| `tools/rtd_alfworld_experiment.py` | CPU audit/smoke-plan 保留；run/resume 委托统一 CLI，没有另建训练 loop。 |
| `scripts/rtd_alfworld_run_hpg.slurm` | 1×B200、8 CPU、64G RAM、48h；显式三个 ALF 数据根，保留 Slurm GPU visibility；缓存/TMPDIR/日志在 results；支持 RTD_CONFIG/RTD_RUN_DIR/RTD_PYTHON/RTD_EXTRA_ARGS 和 run/resume/smoke。 |

C27 已知失败的修复位于 `choose_feedback_tasks`：局部 `config = getattr(self, 'config', {})`。仅有 state/support/RNG 的旧调用继续使用 BFCL 原 8/4 tasks、4/2 rollouts 默认值，不要求新增 config/providers 属性；没有改动或削弱 `tests/test_rtd_baselines_runner.py`。

全量测试还发现 `sample_state` 的同类部分构造对象兼容性问题，也改为局部缺省配置；保留2次source采样默认值和失败likelihood检查先写journal再抛错的顺序，没有改动 `tests/test_rtd_score_consistency.py`。

统一 ALF campaign 输出为 `<run-name>-alfworld-evaluations/round-N/`，训练目录中的 `evaluation-N.json` 保存相同的原生结果。TextWorld 使用 pipe，不启动 HTTP 服务或预留端口。每轮训练 worker 退出后 coordinator 才评估；run/resume 继续保留 UUID、原环境、through_round、独占锁和原 manifest。

## 2. 测试与 CPU 验收

| 文件 | 验证内容 |
|---|---|
| `tests/test_rtd_alfworld_integration.py`（新增） | 注入 backend/env 的共享 CLI smoke/resume/evaluate/report，每折2/4父组；完整 R0/R1/R1s 三轮 campaign；worker 参数、折轮换、全量评估复用；未购买/cross-fold 状态拒绝；失败 parent 合法性；strict identity；AST scope；action cap；锁参数；scalar swap；Slurm 三命令真实 shell 参数/环境验证。 |
| `tests/test_rtd_alfworld_resume.py`（新增） | reference/selected/revealed/actual/feedback/committed 六个 durable save 后分别中断；比较参数、三种 RNG、source cache、episode/rollout seed、P/eta/normalizer、owned/spend、labels/trajectory；另测 durable charge 与 phase save 之间崩溃，无重复 charge/commit。 |
| `tests/rtd_alfworld_runner_fixtures.py`（新增） | 可枚举 CPU policy 和可保存的 LoRA fixture；只替代模型/环境边界，实际执行 source、LOO、pilot、VJP、ledger、StateStore。 |
| `tests/test_rtd_bfcl_c26f_regression.py`（新增） | 分别执行旧版和新版函数构建 R0/R1、smoke/non-smoke manifest，除 rtd_source 外逐字段相等；检查全部 BFCL scoring projection hashes。 |
| `tests/fixtures/rtd_bfcl_c26f_before.json`（新增） | 保存集成前 commit `b5fc5285efaf184949944e2e41535df7d4434c22` 的 manifest/hash oracle，及原样提取并带 SHA256 的旧 load_config/data_identity/make_manifest/harness/projection/hash 函数。测试不需要 git 或生产 BFCL 资产；同时比较固定输出 oracle，避免新旧路径一起漂移。 |
| `tests/test_rtd_alfworld_cpu_replay.py`（新增） | opt-in 真实两 train 游戏各 fresh reset 重放两次，逐状态比对 sealed verification；另 fresh reset 固定无效 raw action→原 parser→look 到 terminal/40步。 |
| `tests/test_rtd_alfworld_config.py` | 原准备入口拒绝训练的断言改为真实委托断言；验证共享 loader 和原配置合同。 |
| `tests/test_rtd_alfworld_contract.py` | 移除 test-only import guard fixture；cached/uncached/fromlist/relative 四种路径均验证原生 guard；评价 wrapper 同时检查 wait interval/callback。 |
| `tests/test_rtd_alfworld_evaluation.py` | 真实环境检查须显式 `RTD_ALFWORLD_REAL_TESTS=1`；保留原完整性/错 split/重复/类型/base class/损坏/恢复测试。 |
| `tests/test_rtd_alfworld_rollout.py` | 同样增加真实环境 opt-in；保留完整动作、EOS/cap、observation mask、LOO、失败保留/基础设施中止检查。 |
| `tests/test_rtd_alfworld_state.py` | 同样增加真实环境 opt-in；保留状态 identity、deadline 和 CPU 数据根检查。 |

执行环境及完整套件命令（线程数限制只影响 CPU 测试调度）：

```bash
cd /home/xueqi/hq/projects/tc-alignment-alf
PY=/home/xueqi/hq/projects/tc-alignment/.venv/bin/python
export PYTHONPATH=src:. PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p results/c26f/tmp results/c26f/cache results/c26f/bfcl-test-runtime
export TMPDIR="$PWD/results/c26f/tmp" XDG_CACHE_HOME="$PWD/results/c26f/cache"
export BFCL_PROJECT_ROOT="$PWD/results/c26f/bfcl-test-runtime"
$PY -m pytest tests/test_rtd_*.py tests/test_checker_bridge*.py -q -p no:cacheprovider \
  --basetemp=results/c26f/pytest-final > results/c26f/pytest-full.log 2>&1
```

完整套件：**912 passed，4 skipped，0 failed，6 warnings，632.51秒（10分32秒），退出码0**，共916项。4项skip均为显式opt-in的真实环境检查；本次另行启用两游戏真实CPU replay并通过。warnings来自既有torch标量转换和tiny PEFT配置检查。另行运行的 BFCL oracle + 未改 C27 失败用例：**6 passed，4.30秒**；共享 ALF 集成：**16 passed，202.12秒**；真实两游戏 CPU replay：**1 passed，24.06秒**。这些检查与完整套件有重叠，不累加为独立测试总数。

`BFCL_PROJECT_ROOT` 是原官方 BFCL 已支持的输出/锁位置环境变量；数据/function-doc 路径仍取原 package。首次完整回归的17个多轮用例因尝试在只读 `envs/` 创建 `.file_locks` 被沙箱拒绝；未写入成功。通过上述环境变量把锁移到results，不改BFCL源码、数据、锁实现或任何测试断言。另一个失败是上述sample_state兼容性问题。首次失败日志保留为 `pytest-full-before-fixes.log`，最终结果以 `pytest-full.log` 为准。

修复后的定向组合（malformed_rollouts、return_gradient、score_consistency、共享CLI两种smoke、BFCL oracle）为 **101 passed，68.39秒**，日志 `fixes.log`。所有报告中的bash命令块通过 `bash -n` 语法检查，GPU/Slurm命令未实际执行。

集成测试的 smoke 在进程内通过现有 `runtime.load_backend` 和 feedback context 边界注入 CPU fixture，没有新增或假称存在命令行 `--fake-backend`。运行的是实际 CLI、verified synthetic bank、support、source、broker、gate、LOO、checkpoint 和 report。完整 synthetic R0/R1/R1s 均完成36步/12窗口：分别购买6/8/8包、spend=2034/2733/2733；这些数值仅用于验收，不能解释为模型实验结果。

真实 audit 均退出0：

```bash
PYTHONPATH=src:. "$PY" tools/rtd_experiment.py audit \
  --config configs/rtd/v1_alfworld_c26.yaml > results/c26f/audit-main.log
PYTHONPATH=src:. "$PY" tools/rtd_experiment.py audit \
  --config configs/rtd/v1_alfworld_c26_scalar_gate.yaml > results/c26f/audit-scalar.log
```

同时用准备入口额外检查完整 CPU manifest section（本地4B权重只做字节 hash，不实例化模型），两者均 passed，输出 `manifest-section-main.json` / `manifest-section-scalar.json`。验证了当前 train world、environment、tokenizer、support、sealed bytes 和完整140题 valid_seen。主/scalar config_hash 分别为 `88c78e47d1bb2f5cabef0ad6745461cbeb89102676eedd999aea4413ba9c2ca8` / `8ab22cc2aca82a9404b7c3b9dba224f45482a26f715ee2c7ba5cbbb23f7be677`；二者 evaluation harness hash 相同。

银行保持219 attempts、107 usable、112 unavailable、135 parents；usable cap sum=140,247,040，三轮预算14,024,704 / 35,061,760 / 70,123,520。历史输出估计189,541；usable估计36,294，失败估计153,247；112 missing responses及1次已知 missing attempt，input/reasoning/discarded/retry成本仍 unknown。audit 中的 C26-A/B 历史 limitations 是冻结银行原文，其中“未来 C26-F 才接入”的句子描述封包时状态；本次没有改写封包来更新叙述。

真实 replay 命令与结果：

```bash
RTD_ALFWORLD_REAL_TESTS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  PYTHONPATH=src:. "$PY" -m pytest tests/test_rtd_alfworld_cpu_replay.py \
  -q -p no:cacheprovider --basetemp=results/c26f/pytest-real-replay \
  > results/c26f/real-replay.log 2>&1
```

| train task / query | 完整成功命令 | 两次重放 | 独立失败路径 |
|---|---:|---|---|
| `pick_and_place_simple-ToiletPaper-None-Toilet-419/trial_T20190908_002434_086261`；`01a3d1eff4d8566e2824a6388a09f188ffad1f17aba654b0dd7ad4d1d4efffa2` | 16 | 与 sealed states/transcript 完全一致 | parser fallback 后40步，won=false |
| `pick_and_place_simple-Newspaper-None-GarbageCan-214/trial_T20190907_181248_411320`；`00400e2dfbce6c70f1ec8393c8b1d36bc58050fab0852364c026436e94762a93` | 4 | 与 sealed states/transcript 完全一致 | parser fallback 后40步，won=false |

每个环境操作30秒、episode 120秒 deadline；无基础设施异常。状态记录位于 `results/c26f/pytest-real-replay/test_two_train_demos_twice_and0/two-train-replay.json`，未重封银行。

## 3. rai 的 GPU smoke 命令（本次未执行）

由操作者先确认空闲卡并设置 `RAI_GPU_UUID` 为该卡的完整 UUID；固定使用同一个硬件 class。以下目录必须是新目录。本次用户要求每折**2父组**，所以显式传 `--smoke-parents-per-fold 2`，覆盖原计划默认每折4父组；manifest 中 `smoke=true` 且保存完整 `smoke_override`，正式 YAML 不改。

```bash
cd /home/xueqi/hq/projects/tc-alignment-alf
PY=/home/xueqi/hq/projects/tc-alignment/.venv/bin/python
export PYTHONPATH=src:. PYTHONDONTWRITEBYTECODE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${RAI_GPU_UUID:?Set the verified idle rai GPU UUID}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export ALFWORLD_DATA="$PWD/envs/alfworld/data" ALFWORLD_DATA_ROOT="$PWD/envs/alfworld/data"
export ALFRED_DATA="$PWD/envs/alfworld/data/json_2.1.1"
export BFAS_ALFWORLD_MAX_STEPS=40 BFAS_ALFWORLD_STUDENT_REACT=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export XDG_CACHE_HOME="$PWD/results/c26f/rai-cache" TMPDIR="$PWD/results/c26f/rai-tmp"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton" TORCH_HOME="$XDG_CACHE_HOME/torch"
export TORCHINDUCTOR_CACHE_DIR="$XDG_CACHE_HOME/torchinductor" CUDA_CACHE_PATH="$XDG_CACHE_HOME/nv"
export NUMBA_CACHE_DIR="$XDG_CACHE_HOME/numba" MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$XDG_CACHE_HOME" "$TMPDIR"
SMOKE_RUN="$PWD/results/c26f/rai-R1-window-p2-k2"
"$PY" tools/rtd_experiment.py smoke --config configs/rtd/v1_alfworld_c26.yaml \
  --arm R1 --smoke-parents-per-fold 2 --run-dir "$SMOKE_RUN"

# 仅在同一 smoke 中断后恢复；保持源码和保存配置，不能换 K/parents/LOO。
"$PY" tools/rtd_experiment.py resume --arm R1 --run-dir "$SMOKE_RUN"

# smoke 完成、训练进程退出后独立评估；不要对未完成 checkpoint 执行。
"$PY" tools/rtd_experiment.py evaluate --run-dir "$SMOKE_RUN" --round 1 \
  --evaluation-lock-timeout 21600 --evaluation-lock-log-interval 60
"$PY" tools/rtd_experiment.py report --run-dir "$SMOKE_RUN" --out "$SMOKE_RUN/report"
```

这里仍是 `Qwen/Qwen3.5-4B`，LoRA rank16/alpha32，BF16 base + FP32 trainables；peak38GB/reserve2GB/state estimate16GB/batch1，action256/context32768，source temperature=1/top_p=1。只执行 round1/step1 一次 reference→selected→revealed→actual→feedback→commit，8槽、M=2/K=2/LOO，允许全零回报或零 gate 梯度。最多2条反馈分支 × 4 episodes=8 episodes、320 actions、81,920 feedback action tokens；inner两父组的reset source为4 actions、1,024 tokens上限，另加每个新独立已购 teacher state 的2条 source actions。若实际/参考同一点则按现有规则复用。

CLI 的900秒 smoke deadline 保留；超时保留 recovery 并报告限制，不能减K、换零baseline或反复抽样直到成功。后续独立官方评估仍是完整140题，最多5,600 actions / 1,433,600 action tokens，保存 partial-smoke checkpoint 标记，不能称为完成一轮训练。

## 4. HPG 提交命令与时间假设（本次未执行）

以下部署约定：将冻结后的隔离 checkout 放到 HPG 的 `/blue/fsu-compsci-dept/xc25.fsu/hq/tc-alignment-alf`；只读使用旁边已有环境的 Python。这个远端路径需要在部署时存在，本次没有连接 HPG 核验或同步。不得将补丁部署到正在运行 BFCL 的目录。Slurm 会在脚本启动前打开日志，因此先建立 `results/c26f/hpg/`。

```bash
cd /blue/fsu-compsci-dept/xc25.fsu/hq/tc-alignment-alf
mkdir -p results/c26f/hpg
export RTD_PYTHON=/blue/fsu-compsci-dept/xc25.fsu/hq/tc-alignment/.venv/bin/python
export RTD_EXTRA_ARGS='--evaluation-lock-timeout 21600 --evaluation-lock-log-interval 60'

sbatch --account=fsu-compsci-dept --qos=fsu-compsci-dept --partition=hpg-b200 \
  --nodes=1 --ntasks=1 --gres=gpu:b200:1 --cpus-per-task=8 --mem=64G --time=48:00:00 \
  --job-name=c26f_R0 --output=results/c26f/hpg/%x_%j.out \
  --export=ALL,RTD_COMMAND=run,RTD_ARM=R0,RTD_CONFIG=configs/rtd/v1_alfworld_c26.yaml,RTD_RUN_DIR=results/c26f/hpg-R0 \
  scripts/rtd_alfworld_run_hpg.slurm

sbatch --account=fsu-compsci-dept --qos=fsu-compsci-dept --partition=hpg-b200 \
  --nodes=1 --ntasks=1 --gres=gpu:b200:1 --cpus-per-task=8 --mem=64G --time=48:00:00 \
  --job-name=c26f_R1 --output=results/c26f/hpg/%x_%j.out \
  --export=ALL,RTD_COMMAND=run,RTD_ARM=R1,RTD_CONFIG=configs/rtd/v1_alfworld_c26.yaml,RTD_RUN_DIR=results/c26f/hpg-R1 \
  scripts/rtd_alfworld_run_hpg.slurm

# R1s 是 scalar gate 的实验标签；CLI arm 仍为 R1。
sbatch --account=fsu-compsci-dept --qos=fsu-compsci-dept --partition=hpg-b200 \
  --nodes=1 --ntasks=1 --gres=gpu:b200:1 --cpus-per-task=8 --mem=64G --time=48:00:00 \
  --job-name=c26f_R1s --output=results/c26f/hpg/%x_%j.out \
  --export=ALL,RTD_COMMAND=run,RTD_ARM=R1,RTD_CONFIG=configs/rtd/v1_alfworld_c26_scalar_gate.yaml,RTD_RUN_DIR=results/c26f/hpg-R1s \
  scripts/rtd_alfworld_run_hpg.slurm
```

三条命令各申请1 GPU/8 CPU/64G/48h，输出目录分离。恢复时重用对应那条命令、同配置/arm/run-dir，把 `RTD_COMMAND=run` 改为 `RTD_COMMAND=resume`。HPG smoke 同理将其改为 `smoke`、换全新目录，并令 `RTD_EXTRA_ARGS='--smoke-parents-per-fold 2'`。额外参数按空白分词，不执行 shell 内容，值本身含空格时应直接调用 Python CLI。

采用用户提供的最新 round1/2 BFCL 速度约 **5 committed steps/h on B200**：12步约2.4h，36步约7.2h，不能再沿用旧 readiness 的52分钟/轮或2–8h/臂估算。ALFWorld feedback 是40步上限的整场episode；正式每臂至多192 feedback episodes=7,680 actions / 1,966,080 generated action tokens，另有每轮 source/P/pilot/VJP 和3×140题greedy evaluation。

本次没有 GPU 吞吐测量。调度假设为：ALF 训练总成本相对这份 BFCL 基准为 **2–4倍**，即14.4–28.8h/臂；串行 HF greedy 每140题预留 **2–4h**，三次共6–12h；模型加载、环境启动与IO另预留2–4h。由此 R0、R1、R1s **各预期约22–45h**，统一申请48h。这个2–4倍是假设，不是把BFCL每步直接乘40；反馈动作时间、早停、完整上下文和已购source深度都会改变比率。R0也保留同样反馈程序，R1s不预设更快。排队/锁等待及故障重试不计入该范围，三臂串行约66–135h、三作业总预约上限144 GPU小时。应先用单窗口实测修正资源/时间申请；任何回报升降都不能作为调参或验收门槛。

## 5. 验收边界与交付状态

CPU 集成与两真实train replay已验；rai单窗口模型正确性、真实GPU峰值/吞吐、正式R0/R1/R1s和每轮140题模型评估留待上述命令执行。正式比较应同时列R0/R1、仅作历史参照的强CE、同合法owned-set强CE；报告保留历史锚点但不把其伪作同预算结果。C26-F没有解除ALF `sealed_replay` 配置冻结，后续fixed-evidence/同owned-set CE/2×2实验需独立接入固定ledger验证，不能删标记绕过validator。C26-D已有的显式campaign continuity链也不等于本次授权训练harness漂移。

最终工作树状态为20个已跟踪文件修改、8个未跟踪新文件；`git diff --stat` 为20 files changed、424 insertions、120 deletions（git默认不把未跟踪文件计入diff stat）。完整输出保存于 `results/c26f/git-status.txt`、`results/c26f/git-diff-stat.txt`；`git diff --check` 已通过。所有源码、测试、脚本和本报告保持未提交。

冻结源码清单为 `results/c26f/source-identity.json`：`rtd-source-v2`，337文件，hash=`4592f4405b4ca2a778a84e0d1015d1881aaf099d8aefcc8b893e257f6299ff8b`。新run须在代码停止修改后启动；本报告的CPU manifest section没有冒充完整GPU run manifest。
