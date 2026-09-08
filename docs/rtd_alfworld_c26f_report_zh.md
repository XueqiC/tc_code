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

## 6. C26-G：episode deadline 语义修复（2026-09-07）

本节补充 C26-F 冻结后的首次真实 GPU smoke 失败及修复；上文的 120 秒 deadline 和“GPU smoke 未执行”是 C26-F 当时状态。此次修改在隔离工作树 `/home/xueqi/hq/projects/tc-alignment-alf`、分支 `alfworld-c26f`、起始 HEAD `1be9dff` 完成；live tree 与只读 `data/`、`envs/` symlink 目标未写入，本次没有使用 GPU，也没有提交。

根因已用 `results/c26f/rai_smoke.log` 与 `results/c26f/rai-R1-window-p2-k2/compute.jsonl` 核对：round 1 / step 1 的 `reference_feedback` 中，sequence 110 的 `alfworld_episode` 保存 `EnvironmentUnavailable: ALFWorld worker timeout (read=30.0s, episode deadline)`；已采样 8 个 action，成功完成 7 次环境 step，action/prompt token 数为 1680/4622。随后 `as_task_rollout()` 抛出 `IncompleteFeedbackError`。旧桥接创建时执行 `deadline = monotonic() + 120`，把两次环境调用之间的学生采样也计入环境期限。C26-B 的脚本命令没有模型生成，因此未暴露这个错误；失败不能解释为任务回报 0。

训练与完整 greedy evaluation 现在共用 `BoundedEnvBridge` 的计时、发送、读取和清理实现。两份 YAML 均显式声明以下可配置键，`alfworld_config.py` 与 evaluation config 校验均拒绝非正值、非有限数、布尔值和非数值；允许正整数或小数。registry → `RealStepper` → bridge 与 evaluation factory 使用同一配置映射，evaluation harness identity 绑定有效值，修改限制会改变身份。

| 配置键 | 默认值 | 语义与理由 |
|---|---:|---|
| `alfworld_worker_read_timeout_s` | 30 s | 保留每次命令 IO 的上限；发送与接收共享期限，worker 不读取 stdin 时也能超时。 |
| `alfworld_env_time_budget_s` | 600 s | 累计桥接内部环境 wall time，包括启动握手、reset 读取、成功及失败的 step IO；两次调用之间的模型采样不扣预算。正常每步环境耗时低于 1 秒，600 秒为启动和环境波动留出余量。 |
| `alfworld_episode_wall_guard_s` | 3600 s | 从创建 worker 起计算的宽松整集 wall guard，在每次 IO 入口和等待期间检查，作为兜底保护。不会异步中断正在执行的模型生成。 |

默认规划估算为 **40 steps ×（generation 约 3–6 s + env <1 s）≈ 120–280 s/episode**，另加启动与清理。仅生成就可能达到原 120 秒期限，因此不能把原值简单当作环境慢。3–6 秒是规划假设，并非此次测得的 GPU 吞吐；旧 rai smoke 日志中的部分生成实际约 8–20 秒，进一步说明要将生成时间与环境预算分开。CLI 现有 900 秒 smoke 总期限是另一个约束，本次保持不变；多个 full episodes 仍可能超过它。

每次尝试均追加 `alfworld_episode`，包含 `env_seconds`、`generation_seconds`、`wall_seconds`、`n_env_calls` 与从 1 起的 `attempt`。真实 worker 的 `n_env_calls` 包括一次 ready 握手、一次 reset state 读取及每个成功/失败 step；嵌套的 step/read 只计一次。`generation_seconds` 是环境调用之间的外部时间，包含生成、prompt 渲染、解析及状态处理，不是纯 GPU kernel 时间；`wall_seconds` 覆盖整次尝试并包含清理，因此不要求前两项精确相加。计时不参与随机轨迹相等性判定，避免影响既有 seed/resume 验证。

仅来自环境操作的 `EnvironmentUnavailable`（包括桥接转换的启动/管道 IO 故障）触发**最多一次**重试。重试先清理旧 worker，再完整 reset/replay 相同任务、相同 seed、同一合法 owned prefix 与 policy；不额外抽取 registry 的外部 seed，不因正常失败回报、parser fallback、模型异常或状态合同错误而重抽。第一次失败记录保留 `reward=None`，并追加 `alfworld_episode_retry`，记录失败原因、seed/prefix、失败与重试 attempt、`max_attempts=2`；失败尝试的 token 开销保留在日志中，不进入梯度。第二次环境失败仍按既有路径触发 `IncompleteFeedbackError`，不补零、不丢任务、不减少 K。greedy evaluation 同样记录尝试与一次重试，持续失败仍向上抛出环境异常，不写完成题目或 aggregate。

旧离线 replay 调用的 `episode_timeout=` 参数保留为累计环境预算的兼容别名；新训练与评估只通过三项显式配置控制。本次未修改已有 smoke manifest、recovery、bank 或日志，也未把旧运行自动迁移到新源码身份；修复后的 GPU 验证应使用新的运行目录。

CPU 验证使用 `PY=/home/xueqi/hq/projects/tc-alignment/.venv/bin/python`、`PYTHONPATH=src:.`、`PYTHONDONTWRITEBYTECODE=1`、`CUDA_VISIBLE_DEVICES=''` 和 `-p no:cacheprovider`。新增测试以虚拟时钟和 fake worker 检查长生成不消耗环境预算、单次/累计环境超时、wall guard、真实管道背压时发送也有界、相同 seed/prefix 的一次重试、失败后排除反馈、配置验证与评估配置传递。完整测试计数及最终 git 状态见下方验收补记。

C26-G 验收补记：新增定向测试 **44 passed**；修正既有断言后的定向组合 **52 passed**。评估中断 fixture 现在连续两次失败以验证重试耗尽，评估复用断言计入新增的 140 条 episode 日志；恢复测试逐项检查计时非负、调用次数，并仅从轨迹一致性比较中排除三个耗时字段。身份测试继续验证修改 evaluation split 会使 harness guard 拒绝，只更新重构后的参数位置。

第一组指定命令 `PYTHONPATH=src:. $PY -m pytest tests/test_rtd_alfworld_*.py tests/test_rtd_bfcl_c26f_regression.py -q -p no:cacheprovider`：**406 passed，4 skipped，0 failed，438.56 秒，退出码 0**。日志：`results/c26g/pytest-alfworld.log`。4 项 skip 是未启用的真实环境 opt-in 测试；本次未重新执行真实模型 GPU smoke。最终验收同时设置 `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1` 限制 CPU 线程数。首次探索性运行的断言失败日志单独保留在 `pytest-alfworld-initial.log`，该运行已中断，不计入通过数；以上计数来自修正后重新执行的完整第一组。

全量首次运行未设置 `BFCL_PROJECT_ROOT`，得到 **939 passed，17 failed，4 skipped，6 warnings，670.75 秒**；17 个失败均为既有 BFCL 多轮测试尝试在只读 `envs/` 下创建 `.file_locks`，写入被沙箱拒绝。该限制及正确设置已在 C26-F §2 记录；C26-G 随后补上 `BFCL_PROJECT_ROOT=$PWD/results/c26g/bfcl-test-runtime`，重新运行原完整套件。只移动官方支持的运行输出/锁目录，未修改 BFCL 源码、数据或对应断言。首次日志保留为 `results/c26g/pytest-full-readonly-locks.log`；最终验收以 `results/c26g/pytest-full.log` 为准。

补上运行目录后，原 17 个 BFCL 失败用例定向复验为 **17 passed，37 deselected，21.64 秒**（`results/c26g/pytest-bfcl-locks.log`）。完整指定套件 `PYTHONPATH=src:. $PY -m pytest tests/test_rtd_*.py tests/test_checker_bridge*.py -q -p no:cacheprovider` 随后独立重跑完成：**956 passed，4 skipped，0 failed，6 warnings，684.27 秒（11分24秒），退出码 0**，共 960 项，日志 `results/c26g/pytest-full.log`。6 项 warning 来自既有 torch 标量转换和 tiny PEFT 配置，4 项 skip 为真实环境 opt-in；各组检查有重叠，不相加为独立通过数。

最终 `git diff --check` 通过。C26-G 的 `git status --short` 与 `git diff --stat` 分别保存在 `results/c26g/git-status.txt`、`results/c26g/git-diff-stat.txt`；默认 diff stat 不计尚未跟踪的新测试文件。工作树保留未提交状态；本次未使用 GPU，未改动 live tree、只读数据/环境或既有 smoke 产物。

## 7. C26-H：非首选 trial 的购买候选缺少 source features（2026-09-07）

本节在隔离目录 `/home/xueqi/hq/projects/tc-alignment-alf`、分支 `alfworld-c26f`、起始 HEAD `5b8b0ff` 上完成。仅 CPU fake backend 验证；没有写入 live tree、`data/` / `envs/` symlink 目标，没有提交、部署或重新执行 GPU 作业。以下更新取代上文对 C26-F fixture 覆盖范围过宽的解释。

### 根因与复现证据

用户提供的 HPG R0 journal 摘要为 `source_sample=158`、`generated_tokens=413`、`score_consistency=413`、`feedback_rollout=8`、`alfworld_episode=8`，随后在 round 1 / step 1 的 `reference()` → `PrePurchaseFeatures.from_public()` 抛出 `ValueError('missing source features; run legal source sampling first')`；R1、R1s 和 rai smoke 同样失败。HPG bank/journal 本次不在本地，以上远端计数按用户提供的证据记录，未声称重新读取远端产物。

本地只读检查 `data/rtd/v1_alfworld_c26/public/{support,reset_states,requests}.json`：135 个训练 parent、138 个训练 trial、142 条 public reset、107 个可用 package。**3 个可用 package 的 reset hash 不属于其 parent 的 selected trial；fold 0 有 2 个，fold 1 有 1 个。三者均能在 public reset 表找到精确匹配。** 例如 query `01a3d1eff4d8566e2824a6388a09f188ffad1f17aba654b0dd7ad4d1d4efffa2` 使用 `pick_and_place_simple-ToiletPaper-None-Toilet-419/trial_T20190908_002434_086261`，该 parent 的 selected trial 则为 `trial_T20190908_002335_756186`。

`freeze_support()` 为 feedback 固定每个 parent 字典序最小的 trial；`ALFWorldExperimentSupport.states` 只保存这些 selected reset，`pool()` 因而只对这些 reset 和已购 teacher states 采样。银行 public request 的 `state_hash` 正确绑定**各购买请求自己的完整 reset**。`reference()` 用 `source_cache` 生成 `StudentSnapshot.state_features`，broker 按 `spec.state_hash` 查询，非首选 trial 得到空 `PublicFeatures()`。这不是 source sampling 没运行，也不是 `features.py` 漏掉统一记录调用；是 **parent → selected reset** 与 **request → exact trial reset** 两种索引覆盖范围不同。即使两个 trial 的 prompt 一样，也不能复用对方的 state hash。

新增 tiny fixture 有 4 个 parent（每折 2 个）、每个 parent 两个 trial，只有 `trial-2` 有可用 package；public reset 全部经 fake 环境重放后 seal，调用真实 `audit_verified_bank()`。只替换生产 135-parent / 本机环境安装审计入口为已通过真实小银行审计的结果，以及 backend/environment 两个外部边界；broker、providers、RTDExperiment、所有数学与持久化均实际运行。修复前，三个 arm 都在已经完成 source sampling 和 reference feedback 后，于上述同一调用链抛出**完全相同错误**：`results/c26h/pytest-reproduce.log`，**3 failed，16 deselected，3.59 秒**。保留失败日志，最终测试本身要求成功完成窗口，不使用 xfail 或吞掉异常。

### 修复、特征定义与隔离

生产代码只增加两处 ALF 分支：

- `registry.py` 的 ALF support 保存训练 trial 的 public reset hash 索引，并提供 `candidate_states()`。只接受精确 hash；先验证完整 reset、训练任务/世界绑定与当前 candidate fold，再返回确定顺序的去重状态。未知 hash、feedback fold、teacher prefix 均明确拒绝；不按 parent/prompt 回退，也不读 sealed payload 来补状态。
- `experiment.py::reference()` 在原候选特征构造前，先由原 broker 按 inner fold、ownership、availability、dependency 和 hard cap 列出合法候选，对缺少 source/projection cache 的 public reset 使用原 `sample_state()` 分批采样，记录 `candidate_source_sampling` / `source_sample` / `score_consistency`。然后完整执行原 `feature()` → `StudentSnapshot` → broker → `from_public()` → acquisition 路径。已有 source/projection 不重采；下一轮 source cache 按原规则清空，initial projection 继续缓存。

特征仍为 `features.py` 的 frozen initial-student 32 维投影，加当前 frozen source 的第一个完整 action 的 logprob 和长度；购买向量继续加 public L、coverage、progress、support_return、intercept，共 39 维。没有加入 teacher command、success、历史实际 cost、hidden reason 或 sealed state。新增候选采样只补购买特征，不将未购买轨迹加入 teacher pool，不重拟合本轮 normalizer / RMS P，不改变 parent/slot 抽样池。选中后，已付款 package 的完整 states 才按原 `pool()` ownership guard 进入 pending sampling / frozen chi transform / pilot / insertion。

BFCL 的原有行、provider 对象、broker、`features.py`、acquisition 和数学实现没有修改；runner 新逻辑严格位于 `benchmark == 'alfworld'` 分支。BFCL frozen manifest/scoring oracle 测试文件和 oracle 本身未改动。额外 AST 核验确认：去掉唯一新增 ALF 分支后，整个 `experiment.py` 与 HEAD 的 AST 完全一致；registry 的 ALF evaluation scoring projection 与 HEAD 一致。只读银行核验的逐 request 证据保存在 `results/c26h/public-reset-audit.json`。

### 为什么已有验收没有发现

旧 `sealed_campaign` 集成 fixture 虽有 135 个 parent 和 219 次请求，却每个 parent **只有一个 trial**；因此每个可购 reset hash 都天然等于 `support.states[parent]` 的 hash。旧完整 36-step campaign 和六阶段 resume 的确执行过数学路径，但没有覆盖非首选 trial。较早 support contract fixture 有同 parent 两个 trial，只检查分组、fold、prefix guards 和 broker 合同，没有把这种结构送入 `RTDExperiment.reference()` 的购买特征构造。真实 CPU demo replay 验证状态重放，也不执行 acquisition。接口签名、属性存在、单次 source event，以及单独的多 trial 合同都不能证明所有候选的 feature table 完整。

新增验收固定 seed 0/1，覆盖 R0/R1/R1s 的购买与 empty 分支，明确核对 39 维特征、相同 prompt 下不同 trial 的完整 state/source 绑定、候选特征构造与选择期间无 sealed 读取、normalizer/P 未变、posterior beta、插入标签、8 槽 × .25 暴露、KL pilot、actual 参数变化、LOO feedback、gate VJP、posterior/cost 拟合和 commit。启动时既有 privileged bank audit 不属于 selector/feature 计算。R1/R1s 的购买用例有非零 insertion value；不要求所有反馈或 gate 梯度非零，零回报仍合法。另有 unknown hash / feedback state / teacher prefix 在采样前拒绝的负例。

`test_rtd_alfworld_resume.py` 对旧 fixture 和新多 trial fixture 都执行 reference、selected、revealed、actual、feedback、committed 六个 checkpoint 的中断/恢复，以及 ledger 已 durable settle、phase 尚未 save 的崩溃恢复。比较范围新增 candidate specs、selected features、reference、step rule 和 selector audit，保留三类 RNG、source/projection、参数、P/eta/normalizer、posterior/cost、owned/spend 和轨迹。恢复后只收费/提交一次；从 complete 再 resume 不消耗 RNG、不重复 episode。

### experiment.py 全阶段 dispatch 审计

表中“真实失败运行”仅指用户本次提供的 round 1 / step 1 故障证据；CPU 覆盖不等于已经获得真实模型结果。

| 阶段 / 所调用路径 | ALFWorld 分派与本次审计结果 | 真实失败运行覆盖边界 |
|---|---|---|
| `__init__` / resume、`save` / `transition` | registry support/broker；ALF 按 durable ledger 顺序重建 paid packages，再恢复 active fold、模型和 RNG。新旧 fixture 检查 checkpoint 与 ledger 超前恢复；没有新 state schema。 | 初始化已到达；远端 mid-window 恢复未在本次执行。 |
| `round_start` | parent hash 两折轮换、source snapshot refresh、posterior reset；`pool → sample_state → feature → FrozenStandardizer / rms_diagonal` 使用合法 selected resets + owned states。action cap 走 ALF provider。保留该 geometry 范围。 | round 1 source 和 geometry 已经过；首次无 inner teacher 时 pilot 合法延后。 |
| `step_start` | 8 槽，same-start reference / exact-noop，ALF feedback context → 完整 episode → `TaskRollout` → 同任务 K=2 LOO。不会进入 BFCL truth/checker。 | reference feedback 已完成 8 episodes；真实 HF generation/scoring、TextWorld worker 和 wall-time 只在真实运行才能核验。 |
| `reference` | 本次补全所有合法购买 trial 的 source features；原 coverage / 39 维向量 / posterior sampling 或 R0 random / public selector audit 不变。负例在采样前拒绝。 | 故障点；补丁后的 candidate sampling、posterior choice 和 selected save 尚无真实 GPU 验证。 |
| `selected` | 原 hard-cap reserve/reveal、request→ledger link；ALF paid mapping + protocol guard 校验完整 prefix，pending source、frozen chi 和首次 KL pilot。 | 故障运行未到达；新 fixture 已实际购买非首选 trial 并跑完。 |
| `revealed` | 共用 streamed gradient、same-start `InsertionReference.insert` / `empty`；标签绑定 query/round/start/reference，8 槽和 epsilon=.25 未变。 | 未到达；CPU 验证非零 insertion value 和 actual 更新。 |
| `actual` | 相同参数复用 reference feedback，否则重新走 ALF full-episode feedback；R0 固定 gate，R1/R1s 共用相应 VJP/controller。 | 未到达；CPU 覆盖 fresh/reused feedback、linear/scalar/fixed gates。 |
| `feedback` → `feedback_commit` | actual-only `commit_step`，合并 owned，按已保存 public features 和 revealed cost 更新 posterior/cost model；预算、fold、曝光、audit invariants。 | 未到达；CPU 核对实际参数、成本、标签和唯一 commit。 |
| `committed` | 落盘 step artifact、释放有限更新对象、按 1/4/7/10 调度下一步；replay 非决策步沿原路径执行。 | 未到达；既有三臂完整 CPU campaign 覆盖 36 步 / 12 窗口。 |
| `round_end` | 共用 LoRA/tokenizer + round state 原子落盘、hash、trajectory、下一轮折轮换。官方 evaluation 在 CLI worker 外由 ALF provider 调度，不在 experiment 内调用 BFCL evaluator。 | 未到达；CPU campaign 覆盖三轮 worker/resume 和完整 synthetic 140-task evaluation/report；真实模型保存/评估未重跑。 |
| `initialize_fixed` | 当前 ALF config 只接受 `sealed_replay`，此分支不属于已支持入口；不能因共用 runner 有此方法就声称 ALF fixed-evidence 已验收。 | 不适用；保持既有不支持状态。 |
| 共用辅助路径 | `packages/pool/draw_slots/chi/gradient/calibrate/feedback/choose_feedback_tasks/scope/batches` 逐项检查：只有 ALF 专用 guard、cap、feedback context 需要分派，其余用共用数学/ledger/memory；`draw_slots` 的 2 samples 和正式 8 slots 与 frozen config 一致。 | 大模型 memory batching、long context、HF source KL/backward 与 GPU 数值/性能仍需要真实模型验证。 |

**剩余风险与运行边界：** 本次没有发现 supported `sealed_replay` 各阶段的其他 ALF dispatch 缺口，但新测试不证明真实 Qwen/LoRA 在 post-reference 阶段无 OOM、数值/score tolerance、长轨迹或 worker 故障。真实故障运行没有到达 reveal、pilot、insertion、actual feedback/VJP、commit、round checkpoint 或 evaluation；这些不可标为真实模型已通过。新增缺失 reset 每轮需要最多两条完整 source action（每个状态两条，非每个 request 两条），增加生成时间；15 分钟 smoke 总时限仍可能不足。生产 HPG bank 不在本地，必须在原已绑定 bank 上验证其 public resets 完整性；缺失/非法 reset 会显式失败，不读 sealed 来绕过。

恢复测试证明保存于 `reference` 的旧形状状态可由新增逻辑补候选特征；从保存的 `selected` 及以后恢复不重新做选择。此次补丁改变训练 source identity；生产 CLI 对旧运行仍要求现有 `--acknowledge-code-drift` 并检查 config/data/base/hardware/harness，不重写原 manifest、不自动接受漂移。本次没有替操作者执行远端恢复。由于新增采样消耗 saved sampling RNG，补丁后的后续采样按修复后的算法继续，不能声称等于不存在此修复的反事实随机轨迹。

### C26-H 验证记录

使用指定 `PY=/home/xueqi/hq/projects/tc-alignment/.venv/bin/python`、`PYTHONPATH=src:.`、`PYTHONDONTWRITEBYTECODE=1`、`CUDA_VISIBLE_DEVICES=''`、三个 CPU 线程变量均为 1，以及 `BFCL_PROJECT_ROOT=$PWD/results/c26h/bfcl-test-runtime`。后者只将 BFCL lock/output 放入可写目录，保持数据与环境只读。定向新增验收：**16 passed，24 deselected，18.45 秒**（`results/c26h/pytest-targeted.log`）。

第一组指定命令 `PYTHONPATH=src:. $PY -m pytest tests/test_rtd_alfworld_*.py tests/test_rtd_bfcl_c26f_regression.py -q -p no:cacheprovider`：**422 passed，4 skipped，0 failed，464.19 秒，退出码 0**。日志：`results/c26h/pytest-alfworld.log`。其中 BFCL regression 的 5 项原 oracle 检查全部通过；4 项 skip 是未启用的真实环境 opt-in 测试。

随后完整执行 `PYTHONPATH=src:. $PY -m pytest tests/test_rtd_*.py tests/test_checker_bridge*.py -q -p no:cacheprovider`：**972 passed，4 skipped，0 failed，6 warnings，678.16 秒（11分18秒），退出码 0**，共 976 项。日志：`results/c26h/pytest-full.log`。6 项 warning 来自既有 torch 标量转换与 tiny PEFT 配置/保存检查；4 项 skip 仍为真实环境 opt-in。两组测试包含重叠用例，不累加为独立通过数。

最终 `git diff --check` 通过；`git status --short` 和 `git diff --stat` 保存于 `results/c26h/git-status.txt`、`results/c26h/git-diff-stat.txt`。共修改 6 个已跟踪文件，其中生产代码仅 registry 增加 21 行、experiment 增加 14 行，其余为测试与本报告；未提交。真实模型 post-reference 窗口与 HPG/rai 恢复仍按上面的运行边界标为未执行。
