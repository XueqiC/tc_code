# 统一 CRCD(原子发现 / 预算采集 / 约束镜像蒸馏 / 认证)——P0 审计与计划(2026-09-04 12:40)

对应用户 9/4 任务书(docs/2026-09-04-crcd-next-tasks-from-user.md)§15 的七项。

## 1. 仓库审计

**可复用组件(已存在、已验证)**
- 学生/训练:`src/appworld_train.py`(LoRA r=8 训练器;SFT/CE、base 中心 DPO、无参考成对、事件权重、风险触发锚点 λ_k、span-only;`encode()`/`completion_log_prob()`/`build_lora_model()` 可被新训练器复用)。
- 指纹:`src/features.py`(LoRA-B 梯度,Adam 白化,随机投影 64 维)、`tools/crcd_fingerprints.py`(A1/A2/A3/A4;A3 = 差分梯度);**尚无 Fisher 白化、无长度归一化、无版本化**。
- 事件挖掘:BFCL `tools/bfcl_event_mine_single.py`(单轮 + 弃答,GT checker,`--rates-key prompt`)、`tools/bfcl_event_mine.py`(有状态首分歧 + K 续跑);ALFWorld `tools/alf_event_mine.py`(demo 前缀回放、首分歧、2K 并行续跑 ΔU、≤4 探测)、`alf_event_reestimate.py`(K 自适应再估计)、`alf_reachability.py`(base/纠正后策略可达性)。
- 有效性门:`tools/crcd_validity_gate_v2.sh` + `crcd_gate_analyze_v2.py`(8 簇增益矩阵、列中心化 Spearman、符号预测、类别/随机基线)。
- 漂移诊断:`tools/crcd_drift_diag.py`(掌握态 KL、熵、首 token 调用概率、base 回复 log-prob)。
- 评测:BFCL 官方 `tools/bfcl_std_campaign.sh`;ALFWorld 官方 valid_unseen `tools/bfas_eval_ckpt.py` + `src/bfas/adapters/alfworld.py`(ReAct 部署脚手架,逐局 records.jsonl);AppWorld 官方脚手架 `src/bfas/adapters/appworld_official.py`(simplified_react_code_agent,dev 评测器);配对检验 `tools/alf_paired.py`。
- 启动:`scripts/*_hpg.slurm`(缓存/HOME 已绕开 hpg $HOME),`tools/*_lane.sh`(rai)。
- 数据:BFCL 事件池 r1/r2/r3/r4、锚点、桶;ALFWorld events_v1(385)、events_v1_k8、event_value_{v1,ce}、round-2 events_r2ce(183);teacher ledger(ALFWorld 107 条验证 demo;AppWorld 40/42 验证 demo)。
- 结果台账:`notes/exp_log.md`(60+ 官方臂,含步数/事件数/池)、`results/alf_records/`(逐局)、`results/analysis/*.json`。

**缺失(相对任务书)**
- Fisher 白化差分指纹、统一事件 schema、稀疏字典 + 摊销后验 q_φ(z|s,y^S)、原子残差 r_k、预算采集(离线回放 + VOI)、镜像目标 + 原始-对偶训练器、离原子位移惩罚、认证(原子级置信界、选择性部署)。
- AppWorld:适配器每步记录不全(缺 snapshot/重建 id、逐步 token、API 调用数);无 train/dev 划分审计;teacher(deepseek via ollama)周上限未恢复,Azure 两 key 周耗尽。
- 记录字段:现有事件缺 git commit / 每事件重复次数 / 精确 teacher token(BFCL 用 GT 无 teacher token;ALFWorld demo token 在 ledger 里有)。

**模型与 teacher 配置**:学生 Qwen/Qwen3.5-4B(LoRA r=8);teacher:deepseek(ollama,主协议,当前周上限)、gpt-5.4(Azure,消融,两 key 周耗尽);GT checker(BFCL)零 teacher 成本。

**算力与存储**:rai 5 卡(今日空 GPU1/2/4:98G/49G/98G;GPU0/3 组员在用);hpg 两账户(dept 24×B200 / 组 8×B200),今日可用;rai 盘 97%(项目 ~110G),hpg /blue 项目 142G,$HOME 38%。

## 2. 方法一致性审计(现有代码/文档 vs 任务书)

| 任务书要求 | 现状 | 差距/处理 |
|---|---|---|
| 指纹 = F^{-1/2}(g^T − g^S),长度归一化,固定 base | A3 = Adam 白化(非 Fisher)、未长度归一化、投影 64 | T1 重做:Fisher 对角(学生 rollout 估计)、长度归一、投影 256、版本化;A1/A2/A3 保留为基线 |
| 原子 = 稀疏字典列,事件 = 稀疏组合;不用硬簇 | 门 v2 用 k-means 硬簇 + 质心余弦 | T2 实现软稀疏编码 + PCA/k-means/随机/语义/类别基线;门改为连续一阶迁移预测 |
| ΔU 只作后验随机变量;不硬过滤、不加权、不反向 | 训练器有 `AW_DDPO_WEIGHT`(du/tanh/pos/conf)与 E 负教师池 | 标记为基线/作废:文档已写 NO;新方法不用 |
| 主蒸馏 = 约束镜像投影 + 原始-对偶;不用 CE/DPO 逐事件开关 | 现有:CE、base-DPO、风险触发锚点(λ 按边界侧) | T3 新训练器 `tools/crcd_mirror_train.py`;现有锚点 λ 机制是"K=2 边界原子"的特例,可作对照 |
| 记号 y^S/y^T,不用 y⁺/y⁻ | 事件字段 `response`/`_rejected`,文档里 a⁺/a⁻ | schema 转换器改名;方法文档重写记号表 |
| 记录 git commit/seed/事件数/步数/重复/teacher token/rollout 数/ckpt | exp_log 有步数/事件数/池;缺 commit、重复次数、rollout 数、精确 token | 统一 schema + 结果记录器补齐(T1 schema;训练器输出补字段) |
| ALFWorld:开发用 train/valid_seen,valid_unseen 只做最终确认 | 迄今所有 ALFWorld 官方分都在 valid_unseen(已反复使用) | 如实披露;新组件在 train/valid_seen 上调;冻结后再上 valid_unseen 三种子 |
| BFCL:留一个未触碰的校准子集 | 无 | 从生成池划出校准子集(不动官方评测集) |
| AppWorld:功能式 API、逐步记录、train/dev 划分审计、绝不看 test | 官方脚手架可跑;记录不全;无审计 | T4 审计 + 冒烟;适配器补字段(下一步) |

**过时/须标记的主张**(任务书 §14):"相对纠正普遍优于 CE"(错,ALFWorld 反转)、"任务族决定算子"(BFCL 证伪)、"ΔU 数值可作权重"(NO)、"负教师反向"(有害)、"首分歧可靠"(NO)、"指纹簇已是原子"(条件性,未验证)、"事件效率 = teacher token 效率"(BFCL 用 GT,零 token,不能推)。这些已在 docs/2026-09-03-crcd-next-stage-response-zh.md 里标注,将写入方法文档。

## 3. 实现计划(文件与依赖)

P0(今日并行):
1. `src/bfas/events_schema.py` + `tools/events_convert.py`(T1)→ 统一事件;依赖:无。
2. `src/bfas/fingerprint.py`(T1)→ Fisher 白化 ψ、版本化、.npz;依赖:1。
3. `src/bfas/atoms.py` + `tools/atoms_fit.py` + `tools/atoms_transfer_eval.py`(T2)→ 字典、基线、连续迁移门;依赖:先用现有 A3 + 门 v2 增益矩阵,再用 2 的 ψ。
4. `src/bfas/mirror.py` + `tools/crcd_mirror_train.py` + `tests/test_mirror.py`(T3)→ 镜像目标、原始-对偶、离原子惩罚、探针 KL、冒烟;依赖:复用 appworld_train 的编码/logprob。
5. `tools/appworld_audit.py` + `tools/appworld_smoke.sh` + `docs/2026-09-04-appworld-split-audit.md`(T4);依赖:无。
6. 方法文档 `docs/METHOD.md`(我)、实验计划(本文件 §4)、结果台账格式补字段(exp_log 追加字段规范)。
P1:统一事件上重算 ψ → 字典 → 迁移门(BFCL/ALFWorld);ALF-1/BF-1 回归从台账核对;ALF-3/BF-3 用镜像训练器对照 CE/成对(同池同预算三种子);原子残差探针与对偶轨迹图;AppWorld AW-1(需 teacher)。
P2:离线回放采集(ALFWorld demo 池、BFCL 事件池)、AppWorld 在线采集(需 teacher)、认证校准、最终冻结评测。

## 4. 实验矩阵(P1 先行;全部单种子探索、主对比三种子)

| 编号 | benchmark / 划分 | 池与预算 | 臂 | 指标 | 预期 / 判定 |
|---|---|---|---|---|---|
| ALF-1 回归 | valid_unseen 134(已用,披露) | 现有台账 | base 8.96;CE_C 71.1±2.2;CE_B 24.9±3.2;CE_zero 50.0±5.4;成对 25.4±1.7;round-2 CE 55.2 / 成对 67.9 | 成功率、配对、按类别 | 已复现自台账,不重跑 |
| ALF-2 原子 | train 事件 385 → train/dev 切分 | ψ(Fisher)| 稀疏字典 vs PCA / 原始 / k-means / 语义 / 类别 / 随机 / A1–A3 | 留出重构覆盖、迁移预测(Spearman/NDCG/AUROC/偏相关) | 须优于最强基线;否则降级为诊断 |
| ALF-3 低技能镜像 | 同 C 池 85 事件、同 44 步、同 token | CE / 成对 / 全局镜像 / 全 CRCD / 无对偶 / 无探针 | valid_seen 开发,冻结后 valid_unseen 三种子 | 全 CRCD 接近 CE(≥65)且 ≫ 成对;否则"获取极限"主张假 |
| ALF-4 胜任后 | round-2 残差 25 + 累计 110 | 累计投影(从 base)/ 适应基座压力测试(从 CE 模型)× {CE, 成对, 全局镜像, 全 CRCD} | 同上 + 每事件重复计数 | CRCD 自动保守(≥ 成对的鲁棒性);对偶随原子满足下降 |
| BF-1 回归 | 官方 | 台账 | base 46.06;CE 8–19;成对 47.37±0.23;C3 47.91±0.36;自适应 47.26±0.32 | 分轴 + 掌握态 KL | 已复现自台账 |
| BF-2 原子 | r2/r3 事件(生成池,零 teacher token) | ψ(Fisher) | 同 ALF-2 + 偏相关控制类别/边界侧 | 门 v2 类比 | 须超过类别基线并能预测负迁移 |
| BF-3 镜像 | r3 联合池 324 | 全局镜像 / 全 CRCD / 无离原子惩罚 / 无对偶 / 仅后验均值 vs CE / 成对 / CE→成对 / KL-CE | 官方分轴 + KL + 位移 | 无 CE 式崩溃;Overall ≥ 最强成对;Irrel/Rel 不单轴大跌 |
| BF-4/5 | 同上 | 对偶轨迹图;每事件重复 1/3/5/8 | λ_k/J_k/r_k 轨迹;安全区宽度 | 对偶只在违约原子上升;CRCD 安全区更宽 |
| AW-1 | train/dev 试点(≤ 20 题) | base + teacher 同脚手架 | TGC/SGC/需求通过率/交互/token | teacher 须显著强于学生(阻塞:teacher 额度) |
| AW-2/3 | train 事件试点 | 同 ALF-2/3 | 同上 | 同上 |

## 5. 成本估计

- GPU:P0 今日 rai 三卡 ~6 小时(指纹 ~600 事件 ×2 梯度 ≈ 1.5h;冒烟 <1h;AppWorld 冒烟 <1h)。P1:ALF-3/ALF-4 各 6 臂 × 3 种子 ≈ 36 训练 + 36 评测 ≈ 36 B200 小时;BF-3/4/5 ≈ 20 臂 ≈ 40 B200 小时;指纹/字典 CPU/GPU 小时级。
- 环境 rollout:ALFWorld 续跑已存(385 事件 × 6);新采集每轮 ~2 万次调用(GPU 10h);AppWorld 每题 ~10 交互。
- teacher token:P0/P1 为 0(用现有 demo 与 GT);AppWorld AW-1 试点 20 题 × ~3k token ≈ 6 万 teacher output token;AW-4 三档预算需另议(deepseek 周上限恢复后)。

## 6. 立即执行的 P0(无新 teacher 消耗、不碰 test)

已并行启动:T1 指纹 + schema(GPU1)、T2 字典 + 迁移门(CPU)、T3 镜像 + 原始-对偶 + 冒烟(GPU2)、T4 AppWorld 审计 + base 冒烟(GPU4)。我这边:方法文档、台账字段规范、BFCL 校准子集划分、ALFWorld 开发/确认划分声明。

## 7. 需要用户决定的事项

1. **ALFWorld 划分**:valid_unseen 已被反复使用。建议:新组件在 train(挖掘)+ valid_seen(开发)上调,valid_unseen 只做冻结后的三种子确认——是否同意用 valid_seen 做开发集(需跑 base/CE/成对的 valid_seen 参考分,约 3 B200 小时)?
2. **AppWorld teacher**:deepseek 周上限、Azure 周耗尽。AW-1 起步需 ~6 万 teacher token;等 ollama 周期重置(时间未知)还是先用已归档的 40 条 gpt-5.4 演示做离线试点(零新 token,但 teacher 标记为 gpt-5.4)?
3. **BFCL 的 teacher token 口径**:~~BFCL 用 GT checker(零 teacher token)~~ **已更正(13:10)**:生成池由 deepseek 合成、演示精确 1,233,607 token,BFCL 与其他 benchmark 同口径记账;官方题 y^T 改用演示,calibration 泄漏已清;主结果重跑 crcd_r3_union_t_s0。
4. **算力**:主对比三种子 ≈ 80 B200 小时;hpg 白天可用则本周内完成,否则只夜间跑要 3–4 天。
