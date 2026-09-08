# RTD v1.1 D2+D3 实现说明

本次只实现隔离 worktree 的批量采购、统一曝光、预算证书、持续后验与恢复；不启动 GPU、在线教师、正式训练或评测，不提交。`data/`、`envs/` 的共享 symlink 不写入。新 bank 由下述 CLI 后续生成，未在生产 data 目录执行构建。

## 1. 本次冻结的语义

以用户此次任务及 `2026-09-09-rtd-v1-1-detailed-plan-zh.md` §3–4 为准，取代较早 budget/quota 提案里的“两槽/包、整体 .25、40 次含 empty 子抽样”：

```
E = exposure_slots_per_window       # 默认 40
K = max_new_packages_per_window     # 默认 20，0 <= K <= E
m = 本窗实际成功购买数
L_S = (1-m/E) L_D + sum(L_q for q in S)/E
```

每个新包抽 **一个**完整动作槽；旧数据参考固定抽 E 槽，按平均梯度计算 `g_D`。剩余 E−m 槽使用同一个旧数据估计，不另重采。实际提交从同一个起点做一次混合更新，gate VJP 使用完全相同的 `(1-m/E), 1/E` 权重。非决策步也使用 E 个旧槽。

`S=[]` 执行旧数据参考更新；没有可产生梯度的旧证据且学生仍等于来源快照时，沿用合法 exact no-op。理论曝光槽始终 E 个（含 no-op）；有效梯度曝光另记，不能把 no-op 算作教师曝光。原始处理量最多 E+m 槽，权重后的有效槽数按 E−m/m 记；旧参考、pilot、诊断、额外 rollout 的计算另列。

v1.1 删除 `replay_prior_mass`，不再给每窗固定 50% empty。未填充槽自然归旧数据；后验样本的边际收益非正时可选空集。这只是局部采购决策，**不是**停止训练或全局最优性证书。v1.0 原 `.5` prior、`.25` 插入和 8 槽规则不变。

## 2. 采购与预算

窗口开始时取本轮剩余累计余额 R、含当前窗的剩余窗口数 W：

```
Q = min(R, max(ceil(R/W), 当前合法且全局可支付候选的最大公开 cap))
B_t = Q
```

每窗冻结购买前特征，抽一次后验 beta；用预测成本做 value/cost greedy，并与最佳单包比较，约束 `sum(c_hat) <= Q`、`|S| <= K`。这不是精确 knapsack。R0/`acquisition: random` 用随机次序，遵守相同成本、配额和曝光限制，没有 empty 硬币。候选仅含公开规格、学生特征、已购覆盖/历史反馈；不使用逐包隐藏成本。

计划选择集及 RNG 先落盘，再按该固定次序逐包 reserve/reveal。预测成本不代替公开 cap；实际余额无法预留时保留 `hard_cap_tail`，最多访问剩余计划项各一次，不重新抽 beta 或补买计划外包。日志区分 `planned_selected` 与最终 `selected`，并重算最终集合的 acquisition 可加预测。

全局 `S+H<=C` 不变；新增窗口满足 `S_window+H<=Q`，同时包含待结算请求的 K 上限。`reserve_chain` 先原子检查所有未付依赖的两层额度与数量；broker 仍只展示依赖已拥有的请求。正常 settle 按 actual 结算、隐式释放 cap−actual；显式 `release` 释放未完成预留。零成本包占 K，已购复用不重复收费。新增 `window_open/window_close` 不改变旧 ledger 事件格式。

预算口径为 `usable_recorded_output_tokens`，C25 固定分母 55,370，正整数 half-up：`(B*p+50)//100`。三点 5,537 / 13,843 / 27,685；两轮只取前两点。累计授权不清零已花成本。manifest 明确写 `cached-content cost; not the full teacher bill`，不把 generator estimated usage 当成完整 provider 账单。

### 新 bank CLI

```bash
PYTHONPATH=src:. CUDA_VISIBLE_DEVICES='' \
  /home/xueqi/hq/projects/tc-alignment/.venv/bin/python \
  tools/rtd_v11_build_bank.py \
  --source data/rtd/v1_bfcl_c25 \
  --out data/rtd/v1_1_bfcl
```

CLI 默认绑定原 C25 public SHA256 `2d27d0f4dcfd1076ac2fbdba6ff3d2789caf5971c14a5da61634d710d00084e8` 和总成本 55,370；拒绝已有输出目录、源目录及嵌套覆盖。构建时特权读取、验证全部封存 payload integrity，再对可用集合认证 demo 2,048 / generator 512，规则是类别最大 recorded cost 向上取 2 的幂。逐包 payload 和旧 audit **原字节复制**，不可用记录 cap/provenance 保持原样。

新 `public/cap_certificate.json` 只公开总量、类别统一上界/数量及集合哈希，不公开逐包 usage、分位表或教师内容。这是明确的数据依赖公开成本假设，不是独立 provider 保证。`cap_provenance` 保存旧 provenance；以 certificate `core_hash` 引用证书核心，完整证书另外绑定新 public 文件 hash，避免相互 hash 引用形成循环。manifest/data identity 再绑定完整 certificate 文件 SHA256。运行期证书校验只读 public 与 integrity，不打开未购响应。新审计写 `sealed/audit_v11.json`，不覆盖旧 audit。

只读核验现存 C25：420 可用包、55,370 tokens，其中 exact=21,203、estimated=34,167、支持父任务 m=40；public hash 匹配。测试银行全部为 tmp fixture。

## 3. 持续后验、不确定性与泄漏边界

`ValuePosterior.begin_round` 只更新 round ID，不重建模型。`CostRegressor.begin_round` 同样保留模型与观测。新增标签的特征包含当前训练进度和支持集回报；history、cost observations 均保存实际 observation round ID，旧轮特征不能冒充本轮新标签。新增包每个 request ID 只拟合一次。

标签 `value` 为单位曝光导数 `v_q=-eta<gJ_ref,P(gq-gD)>`；`position_share=1/E`、`position_value=value/E`、`position_variance=variance/E²`。同一参考 rollout 批次对所有已购包流式投影，只保留 rollout×package 标量矩阵，不保存每 rollout 的整模型梯度。分任务 bootstrap（256 次）重算 LOO baseline，保存完整标签协方差。后验批量更新使用 generalized least squares，保留同批标签相关性，避免当成 m 份独立反馈。

该方差是**给定任务集及包梯度样本后的反馈估计方差近似**，不是完整 VOI 方差。回报不可区分或某任务只有两个 rollout 时保留 prior noise 下限，不能把统一零回报解释为精确零价值；另有正数 `value_noise_floor` 和协方差特征值下限。来源槽抽样噪声不在此估计中。

`value_covariance.shared_gradient_projection_error` 记录“先累计 gJ 再求内积”与“逐 rollout 投影再累计”的数值残差；两者在实数中相同，BF16 加法顺序会产生舍入差异，不把 float64 容差硬套到低精度训练。

每轮第一窗最多重测 `drift_reference_packages=4` 个按 ID 固定排序、已购且属于当前 inner fold 的参照包。重测使用本轮共享参考，标签仍为 `reweight_existing`，与新增购买标签分开记录，不直接当作新购买再次拟合。超过前后估计方差的漂移按记录的 inflation factor 同时缩小 precision/information，保持 posterior mean、增加 covariance，不清空历史。

保持两折隔离：第二轮若仅拥有第一轮 fold-0 包，就记录 `no_legal_purchased_reference`，不读取反馈折教师来凑参照数；第三轮可重测第一轮包。明确扣账的校准包只有进入合法已购集合后才能供诊断；不新增免费 calibration 入口。

两组可靠性估计使用同一参考点、相同任务安排的两批独立生成 rollout，只在已购请求上执行；第二组不重复拟合 posterior。记录两组值、相关系数（样本不足或零方差为 null）、差值 z-score；相关并非多 seed 显著性证据。

每窗 `value_difference_significance` 记录 posterior 均值最高两候选之差、协方差导出的标准误及 1.96 阈值。`decision_changed` 比较“上一窗决策前模型”和“当前模型”，在**本窗相同候选、特征、成本、预算与标准正态随机向量**上重算选择，避免把库存变化误当后验影响。首窗无可比较模型则 null，随机控制臂为 false；实际 cap 拒绝单独记。它是模型内诊断，不是官方评测统计显著性。

## 4. 字段与恢复

`decision` 记录：`query_ids`、公开 caps、`expected_costs`、`sampled_values`、beta/standard-normal draw、Q/R/W、E/K、`planned_selected`/`selected`、`predicted_cost`、`predicted_additive_gain`、`budget_binding`、`stop_reason`、significance。停止原因包括 `package_limit`、`predicted_budget`、`hard_cap_tail`、`nonpositive_sampled_gain`、`no_feasible_candidates`、`candidate_exhaustion`。

提交后的 `batch_window` 及 `steps/*.json` / trajectory 保存：

- `predicted_additive_gain`：已付标签的 `sum(v_q/E)`，共同参考的局部可加近似。
- `acquisition_predicted_additive_gain`：采购时 posterior sample 对最终购买集的预测。
- `realised_joint_update_gain`：actual 平均 rollout reward − reference 平均 reward；相同模型复用反馈时为 0。
- `joint_vs_additive_error`：realised − 共享参考标签可加预测；另记 `joint_vs_acquisition_prediction_error`，不混淆两种误差。
- 每包标签、完整 covariance、reliability、round drift、包数、实际支出/余额、raw/weighted 槽、`new_slots_by_query` 的完整来源/教师记录。

联合收益差包含有限插入误差与 rollout 噪声，不能当成精确 Taylor 残差。`compute.jsonl` 的诊断 rollout、流式重评分、pilot 等实际成本保留；失败/恢复重复计算不从 compute 日志删除。已提交轨迹是窗口计数权威，D5 不应把重复 compute 事件累计成多个提交。

恢复沿用 `reference → selected → revealed → actual → feedback → committed`。`selected` 内每完成一个包保存一次当前事务 index/query、计划集合、已付 pending 集、特征、成本 posterior、来源槽和全部 RNG。ledger 超前只接受该**当前 query** 的事务，不能凭整张计划表接受未来 reveal；window-open 必须与持久化 Q/K/id 一致，window-close 只能出现在已提交阶段。

Recovery 保存整个 state；`round_state.pt` 保存原来源/后验状态，并增加 `batch_state` 保存其余新状态、漂移历史/标志、上一决策模型、RNG、窗口记录。官方 checkpoint artifact hash 继续绑定 round_state。两轮 runner/coordinator 在第 2 轮结束，不创建或评测第 3 轮。

## 5. 配置与文件改动

新示例 `configs/rtd/v1_1_bfcl.yaml` 默认三轮。两轮设 `rounds: 2`、`budget_checkpoints_bank_fraction: [0.10, 0.25]`。`exposure_slots_per_window: 40`、`max_new_packages_per_window: 20` 可省略；`slots_per_step` 若提供必须等于 E。其他新键为 `drift_reference_packages: 4`、`value_noise_floor: 1.0e-8`；持续后验配置声明 `acquisition_posterior_refresh: persistent`。旧 `insertion_fraction`、`max_new_packages_per_decision`、`replay_prior_mass`、`acquisition_entropy_temperature` 在新配置中拒绝，避免静默忽略旧数学。

| 文件 | 改动与责任 |
|---|---|
| `acquisition.py` | 持续 ValuePosterior、LegacyValuePosterior、跨轮成本观测、异方差/协方差更新、漂移 inflation、批量 greedy、显著性与同条件决策比较 |
| `insertion.py` | 旧标签/schema 保持；新增 BatchInsertionLabel 和共享参考 batch labels/update，1/E 明确入日志 |
| `experiment_v11.py` | opt-in batch phases、付费后反馈方差/可靠性、合法已购漂移、单次实际 commit 与窗口汇总 |
| `experiment.py` | 版本分流、E/显式 draw count、后验跨轮生命周期、round_state、两轮终止、batch invariants |
| `value_feedback.py` | paid guard、共享 rollout 方向投影、重算 LOO 的 bootstrap covariance、独立可靠性统计 |
| `ledger.py` | 可选窗口额度/K、chain 原子预检、window 事件重放；旧 reserve/reveal/release 格式不变 |
| `broker.py` | 候选同时检查全局及窗口公开 cap/K；读取仍在 reserve 后 |
| `selector.py` | `select_public_batch` 复用公开数据访问 guard，拒绝重复/未提供 ID |
| `caps.py` | 新版本常量及 recorded half-up 预算函数；旧 affordability/public caps 不变 |
| `bank_v11.py`、`tools/rtd_v11_build_bank.py` | 离线冻结 C25 转换、类别证书、不可覆盖 CLI、运行期不解封的验证 |
| `config_v11.py`、`cli.py`、新 YAML | 版本化默认值/验证、bank audit/manifest/data identity 证书绑定、两轮 coordinator |
| `persistence.py` | 当前子事务及 window-open/close 的严格 ledger 超前验证 |
| 三个 `test_rtd_v11_*.py` | 统一曝光/有限差分/VJP、持续后验/泄漏、bank/config/manifest、逐阶段及 reserve/reveal/close 崩溃恢复 |
| `test_rtd_acquisition.py` | 直接 ValuePosterior API 的测试改为跨轮保持；Legacy 路径另测 reset |
| `test_rtd_evaluation_resume.py` | 两轮 coordinator 的中断/完成复用测试 |
| `test_rtd_alfworld_state.py` | 真实 CPU integration 的两个 reset 输入复制到 tmp fixture，适配只读 envs symlink；生产路径保护与真实 world hash/replay 校验不变 |
| `tests/fixtures/rtd_v11/` | 修改前 tiny-model 单窗与 manifest 的字节基准（manifest 环境 identity 固定、tmp 路径归一化） |

`features.py` 已核对，无需改变公开特征维度/冻结投影或引入任何教师特征。旧配置使用 LegacyValuePosterior，旧 manifest、单窗字段及计算结果按回归基准保持；真实 `rtd_source` 内容哈希会随源码改变，这是正常身份审计，不伪装成旧源码。

## 6. D1 / D5 / D6 接口

- **D1**：`RTDExperiment.gradient(targets, chi, parameters)` 与 `BatchExperimentMixin.batch_gate_vjp(targets, chi, feedback)` 必须成对替换为软/CV 的同一个更新算子。批量层只组合梯度/VJP，不另外引入 RL 更新。`InsertionReference` 接收 `old_gradient` 和 per-package gradients，已适配不同来源估计器；保存同一 ref/start 和来源快照信息。本段不实现 D1 的软/CV 数学，新配置目前仍调用原 transport 算子，整合 D1 后再正式运行。
- **D5**：从已提交 trajectory/step rows 取每窗主指标；从 compute 取全部实际计算，包括 drift/reliability 额外 rollout。区分 acquisition 预测与已购插入预测，报告 basis/certificate、实际 spend/ceiling、cap-tail 和 K 的限制。旧 report 可消费新轨迹的基础 exposure/truncation 字段；新的汇总图/固定任务失败率留给 D5。新 `selected` 是列表，`labels` 是 request-id 字典，不能继续假定单 selected/label。
- **D6**：来源估计器对照仅传 ledger-owned 且合法 inner 的包；`require_paid` / `insertion_statistics` 可复用。明确扣账校准须先经过正常 broker 事务，不能为未购候选调用真实标签估计。二次独立反馈与共享参考只供诊断，第二组不重复拟合。正式四种来源估计器 runner 和 V0→V1 按窗 ledger 复用不在本 D2+D3 段新增；现有 fixed_evidence 是完整集合重训，不能冒充按窗采购复用。

## 7. 验证

CPU 命令：

```bash
PYTHONPATH=src:. BFCL_PROJECT_ROOT=/tmp/rtd-v11-bfcl-test-root \
  CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/xueqi/hq/projects/tc-alignment/.venv/bin/python \
  -m pytest -q tests/test_rtd_*.py
```

首次完整执行发现共享只读 envs 的两类环境问题：BFCL 默认 `.file_locks` 路径不可写；ALFWorld `_Sources` 拒绝越出 worktree 的 symlink。前者使用 BFCL 官方 `BFCL_PROJECT_ROOT` 临时输出开关，后者只修测试 fixture，不改生产安全检查；真实重放不匹配仍会失败。

最终完整 `tests/test_rtd_*.py`：**929 passed，6 warnings，489.75 秒**；无跳过，包含真实 ALFWorld CPU 重放。最后的反馈投影数值残差改动另跑全部三个 v1.1 测试文件：**40 passed，1 warning**。覆盖 gate VJP、两轮 campaign/report、40 槽/20 包、hard-cap-tail、逐事务恢复及 v1.0 字节基准。未运行生产模型、正式 BFCL 评测或 GPU 预检。
