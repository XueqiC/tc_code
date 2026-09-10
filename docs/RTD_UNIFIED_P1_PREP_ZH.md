# RTD unified P1 preparation

本次交付为 P1 启动工程与 CPU 验证；**没有运行 GPU、Gemma、环境任务实验或 teacher API，没有性能/速度结论**。P1–P3 实验结果仍为 PLANNED。入口是 `tools/rtd_experiment.py` 的 run/smoke/resume/evaluate；D0 另有 `tools/rtd_unified_baseline.py`。

## 配置与 arm

`unified_alfworld_gemma4.yaml`、`unified_webshop_gemma4.yaml` 从各自 D16 v1.1 配置派生，复制 `unified_bfcl_gemma4.yaml` 的 unified 数学块。三者 student 为 google/gemma-4-12B-it、LoRA r16/alpha32、相同七类 target modules、两预算轮 10%/25%、每轮 12 次 commit、decision 1/4/7/10、E=40、K≤20、60 GB 显存预算、原 hard reservation / sealed replay / 按父任务轮换折。BFCL 明确使用 gemma4 call format。

ALFWorld 现有 bank 的 CPU 审计：107 个包，135 个父任务，36,294 estimated recorded output tokens 分母，整数累计 cap 3,629 / 9,074。它们是已有内容访问预算，不是新增 API 授权。新 bank 由 `tools/rtd_bank_build.py` 生成，启动会核验 certificate/student/support/harness；不得把 Qwen 渲染的旧 bank 当 Gemma bank。

| --arm | 实际控制/训练语义 |
|---|---|
| D0 | appworld_train 的 SFT + 冻结 round snapshot 的 source-prefix forward soft KL |
| D1 | 原 v1.1 state-only learned α gate + learned d solver；保留 warmup、blocked α VJP、mean-diagonal d 标度及 total NLL |
| D2 | CV，按状态共享一个系数；同状态两来源及重复曝光的系数一起约束相等，跨状态联合 QP |
| D3 | CV，全来源自由系数，完整 UᵀHU，exact local QP |
| D3-nocross | 仅控制器优化时 K←diag(K)；仍在未改动的完整 F 下报告价值 |
| D3-raw | Gavg 为基线，替换量仍为 Σ a(GT−Gj)/2；reference/commit 都使用 raw |
| D3-shuffle | 先求 D3，再按预登记 seed 对每个状态曝光内的两来源做随机置换；保持同份教师、系数分布及每状态剂量，不拒绝 identity permutation |
| D3-fixedmean | 每曝光槽 Σj aj=2a_ref，只有来源 split 自由；因此全局 Σa 也固定 |

D2/fixedmean 在原 auxiliary QP 中加入对 x=a−a_ref 的线性等式，复用独立 KKT 验证。没有把 D2 实现为逐状态独立求解。D3-nocross/shuffle 的评价始终使用完整原问题；shuffle 的 KKT 只属于打乱前的求解，不称打乱后系数为 exact optimum。slot 显式绑定 schedule 中那一份教师记录，重复 state 不会自动打开别的已购 teacher version。

## 匹配协议与 D0

所有 P1 臂从同一 issuer checkpoint + 相同 seed/LoRA 初始化出发。生产 run 必须提供同一完成的 V0 `exposure_schedule.json`；借用 D14 的 hash-bound replay、逐步 record/weight/repetition/pool/purchase 检查。不会按某臂的反馈重新采购或重排曝光。采样每次刷新，按 seed/round/step/role 使用共同 RNG seed；学生更新后分布当然可不同。feedback task 选择使用独立共同 seed，不能受某臂前面的 RNG 消耗影响。

三角色 acquisition_reference_feedback、same_batch_reference_feedback、post_commit_feedback 在每个 decision 都执行相同任务/rollout 预算。D2/D3 在自己的 a_ref trial 测 h；普通 commit 复用最近 decision 的反馈并重新投影、记录年龄。D1 保留原 α/d 参考点；D0 收集相同反馈但不用于系数学习。原 acquisition label fitting、额外 paired-validation/variance-resampling 在所有 P1 臂均关闭；完整确认评测走 evaluate，不能把训练反馈当独立确认。

优化匹配：24 次单步 commit，η=`initial_eta=1e-5` 在实验前固定，所有臂共用；不运行各臂独立的 η pilot。`p1.optimization=shared_fixed_eta` 与 manifest 的 `step_size_selection` 明确覆盖继承的 v1.1 pilot 设置。P 为各自当前 round 来源上的同一 train-only RMS 规则，round 内冻结，LoRA rank/modules/alpha 与精度相同。它是相同优化程序，不声称不同权重轨迹上的 P 数值相等。D1 的 total NLL 与 D0/D2/D3 的默认 per_sequence_mean 有既有口径差异；D1 是原程序整体对照，不能将其差异全部归于来源选择。

D0 的专用配置是 `configs/rtd/unified_alfworld_gemma4_d0.yaml`；也可在任意上述 unified config 上选 `--arm D0`。`src/appworld_train.py:rtd_sft_kl_gradient` 扩展 Table 1 baseline trainer 的窗口入口，mode 为 `AW_DISTILL=rtd_sft_kl`。它用同样的已购 rows/曝光权重作 teacher CE，并从同一冻结 round snapshot 新采两条 source，在其前缀上计算 forward KL；没有 teacher-prefix KL、prompt 截断、隐式 shuffle、额外 epoch 或未匹配的 Adam step。外层统一 shell 提交这一 baseline 梯度以匹配 frozen-P optimizer；原 Table 1 AW_DISTILL=ddpo / pair_unit 等入口不改。

D0 loss = Σi wi [cT NLL(teacher_i)/LiT + cKL/2 Σj KL(p_t || p_θ; source-prefix_ij)/Lij]，无教师的免费 reference slot 只有 KL。total-NLL matched 矩阵可把所有统一臂的 normalization 改成 total_token_nll，去掉长度分母；D1 本来就是 total NLL。D0 是常规纯软 KL，不声称其随机长度归一后与 hard-source NLL 期望一致；D2/D3 使用 P0 的正确 length-corrected CV。

匹配且不逐臂调的量：steps=24、lr/η=1e-5、rank=16、LoRA alpha/modules、P 规则、pool/exposure、反馈预算。**仅 calibration split 可调** D0 cT∈{.5,1}、cKL∈{.1,1,10}，默认 (1,1) 是预登记起点，尚未声称已充分调优。改值必须属于登记网格，写入 config/manifest/campaign identity；不访问 confirmation/test 做选参或选 checkpoint。用户要求的 calibration-only 调参是明确的开发用途，不能随后把该 split 称 untouched confirmation。已调好的 pairwise 锚点须另报其确切配置/证据/学生；没有捏造 Gemma 的已调 pairwise 结果。

## 运行命令（本轮未执行）

先选择一张已获授权的 GPU，设置 CUDA_VISIBLE_DEVICES 为该卡 UUID；各臂使用同一卡。以下从仓库根目录执行：

```bash
export PYTHONPATH=src:.
PY=/home/xueqi/hq/projects/tc-alignment/.venv/bin/python
CFG=configs/rtd/unified_alfworld_gemma4.yaml
OUT=results/rtd_unified/p1_alfworld
SCHEDULE=$OUT/V0/exposure_schedule.json
```

如已有匹配的完整 V0 schedule，直接将 SCHEDULE 指向它。否则先录制一份 D16 v1.1 V0 schedule（这是另一次 GPU 运行，本次未运行）：

```bash
$PY tools/rtd_experiment.py run --config configs/rtd/v1_1_alfworld.yaml --arm V0 --run-dir "$OUT/V0"
```

所有 P1 臂回放这同一份 schedule，分别独立初始化：

```bash
$PY tools/rtd_unified_baseline.py run --config configs/rtd/unified_alfworld_gemma4_d0.yaml --replay-schedule "$SCHEDULE" --run-dir "$OUT/D0"
$PY tools/rtd_experiment.py run --config "$CFG" --arm D1 --replay-schedule "$SCHEDULE" --run-dir "$OUT/D1"
$PY tools/rtd_experiment.py run --config "$CFG" --arm D2 --replay-schedule "$SCHEDULE" --run-dir "$OUT/D2"
$PY tools/rtd_experiment.py run --config "$CFG" --arm D3 --replay-schedule "$SCHEDULE" --run-dir "$OUT/D3"
$PY tools/rtd_experiment.py run --config "$CFG" --arm D3-nocross --replay-schedule "$SCHEDULE" --run-dir "$OUT/D3-nocross"
$PY tools/rtd_experiment.py run --config "$CFG" --arm D3-raw --replay-schedule "$SCHEDULE" --run-dir "$OUT/D3-raw"
$PY tools/rtd_experiment.py run --config "$CFG" --arm D3-shuffle --replay-schedule "$SCHEDULE" --run-dir "$OUT/D3-shuffle"
$PY tools/rtd_experiment.py run --config "$CFG" --arm D3-fixedmean --replay-schedule "$SCHEDULE" --run-dir "$OUT/D3-fixedmean"
```

D0 等价入口：`AW_DISTILL=rtd_sft_kl $PY tools/rtd_experiment.py run --config "$CFG" --arm D0 --replay-schedule "$SCHEDULE" --run-dir "$OUT/D0"`。BFCL/WebShop 分别改 CFG、OUT 和录制 schedule 的 v1_1 配置；BFCL 录制用 `v1_1_bfcl_gemma4.yaml`。各 benchmark 只使用自己的 bank/schedule，不能跨 benchmark 回放。

ALFWorld D3 精确 smoke 命令：

```bash
PYTHONPATH=src:. /home/xueqi/hq/projects/tc-alignment/.venv/bin/python tools/rtd_experiment.py smoke --config configs/rtd/unified_alfworld_gemma4.yaml --arm D3 --run-dir results/rtd_unified/p1_alfworld/D3_smoke --smoke-deadline-seconds 900
```

smoke 唯一允许不传 schedule，用于独立接线检查，不计为同池比较结果。它执行一个 decision/commit，E=2、K=1、每反馈角色 1 个 task × 2 个独立 rollout，采用合法的固定零 baseline，最多 900 秒；完整任务 horizon 与动作 cap 不改，超时保留恢复状态。CPU tiny smoke 验证不代替 Gemma 显存/运行时验收。

```bash
$PY tools/rtd_experiment.py resume --run-dir "$OUT/D3"
$PY tools/rtd_experiment.py evaluate --run-dir "$OUT/D3" --round 2
```

resume 不需要重传 config/arm；所有继承的 executor 默认值都在首次验证时物化，saved config 的 p1_runtime_defaults_frozen 阻止恢复时重新读取后来修改的 v1.1 YAML。saved manifest 绑定有效 preset、schedule 内容 hash、checkpoint/data/tokenizer/harness/source/hardware。smoke resume 的 deadline 需与原命令一致。campaign identity 明确含 arm，不能共用结果文件。

## profiling 与验证

每段 stage 输出如下（decimal GB，与 60 GB budget 一致）：

```text
[rtd-profile] stage=source_teacher_soft_gradients wall_seconds=1.234567 peak_gpu_gb=23.456789 peak_reserved_gb=25.000000 status=complete
```

九个 stage：student_sampling、teacher_replay、source_teacher_soft_gradients、sketch_metric_gram、feedback_rollouts、qp、controller、commit、save。每次实际进入都会输出 wall time 与局部 allocated/reserved peak；异常输出 failed；CPU 为 0 GB。compute.jsonl 保留有父子关系的细粒度计时和 stage 标签，勿把嵌套 inclusive 时间相加为总时间。反馈方向投影/标准误计入 sketch_metric_gram；同一个 immutable objective 的 Uᵀh 只计算一次，不在 QP 每次 callback 中重复大维投影。controller=exact-only 时只测小型结果装配，D0 QP 是显式空操作；没有假装训练 predictor。梯度使用现有 streamed vocabulary head，CPU 仍保存 dense trainable directions：大模型 host RAM 与总体耗时尚未实测。

CPU 检查覆盖新配置与拒绝非法参数、D2 joint tie 与受限 optimum、nocross 的独立对角解且完整 F 不变、shuffle permutation、fixedmean 每槽/全局和、streamed vs dense P0 梯度及随机长度、D0 独立 CE/KL 梯度、native EOS 长度、八臂真实 tiny smoke/resume、共同 V0 schedule/账本与篡改拒绝、CLI dispatch/deadline、profile 格式与异常记录。

```bash
export CUDA_VISIBLE_DEVICES=''
export BFCL_PROJECT_ROOT=/tmp/rtd-unified-p1-bfcl
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PYTHONPATH=src:. /home/xueqi/hq/projects/tc-alignment/.venv/bin/python -m pytest -q tests
```

最终 focused CPU 检查：55 passed，33.83 s（P1、P0 QP、P0 execution）。最终全 suite：**2,436 passed、4 skipped、6 warnings，931.44 s（15:31），退出码 0**。命令与环境设置见上，完整原始日志在本次 workspace 的 `/tmp/rtd-unified-p1-pytest.log`。GPU/API 运行次数为 0。
