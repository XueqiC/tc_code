# 2026-09-04 全天实验总结(统一 CRCD 第一天)

写于 2026-09-05 00:30。范围:9/4 12:30 收到统一 CRCD 任务书之后到现在的全部工作。所有实验单种子(用户决定),所有 BFCL 分数为官方 v4 全量评测,ALFWorld 为 valid_seen(开发划分)140 局,AppWorld 为官方 ReAct 脚手架 dev57。

## 0. 一句话结论

- **成立的**:指纹的一阶迁移预测(ψ·ψ 预测实测单事件迁移,ρ 0.67 / 符号 AUROC 0.77,显著超类别标签,对 lr 稳健,优势在 teacher 侧);ALFWorld 开发划分与 valid_unseen 排序一致;BFCL 记账一致版主结果 46.74(+0.68 over base),同池上 deepseek 演示目标优于 GT 目标(+0.70)。
- **不成立/负的**:稀疏原子字典(覆盖率 = PCA,迁移预测无增量,BFCL 与 ALFWorld 都是);镜像蒸馏在 BFCL 竞争力 base 上三臂全负且随对偶压力单调恶化(24.0 / 35.5 / 44.7),诊断为 lr 20 倍 + 压力过大,对照在跑;等 token 预算下"残差/token"采集规则没赢随机(No-Go);AppWorld 首分歧事件上 CE 把学生训成循环(0/57)。
- **修正的**:BFCL 从来不是"零 teacher token"(演示 1,233,607 精确 + 生成池 ≈143k 估计);单轮池混入 calibration 题(泄漏)已清;早上建的校准集全是生成题,已换成 65 道官方题。

## 1. 决定与记账

| 项 | 决定 |
|---|---|
| 种子 | 三个 benchmark 全部单种子 |
| BFCL teacher token 口径 | 与论文统一:生成池由 deepseek-v4-pro 合成(脚本原来丢了 usage,已改为逐次记录),演示精确计数;GT checker 只作验证器;官方题 y^T 只能来自演示 |
| 代码 | 从 9/5 00:45 起全部交 Codex(gpt-6-astra @ xhigh),`tools/codex.sh`;`ops/codex_task.sh` 默认已改 |
| git | checkpoint 5be0483 |

BFCL 记账重建(`tools/bfcl_teacher_tokens.py` → `results/analysis/bfcl_teacher_tokens.json`):演示 40 题 × 3 次尝试 = 1,233,607 output token(34 题验证通过,≈36k/条含推理);生成池全部 ≈143k(重新分词 ×1.3,下界)。

## 2. P0 基础设施(4 条并行子代理,全部完成)

**T1 统一事件 schema + Fisher 白化指纹**(`src/bfas/events_schema.py`、`src/bfas/fingerprint.py`)。ψ = F^{-1/2}(g^T − g^S),LoRA 参数上的长度归一化梯度差,对角经验 Fisher 来自学生 rollout,256 维 sketch,单位范数。产出 bfcl_r2(210)、bfcl_r3(148)、bfcl_r3t(302,teacher 版池)、alfworld(385)四套,每套标了 calibration 事件索引。诊断:ψ 与未白化差余弦 0.75,事件间 0.13,同状态不同学生样本 0.69。

**T2 稀疏字典 + 迁移门**(`src/bfas/atoms.py`)。FISTA 稀疏编码 + 单位球字典更新,模型选择用分组留出重构 + 1-SE 规则(K=32)。结果:留出覆盖率稀疏 0.518 ≈ PCA 0.523(A3 特征),Fisher ψ 上 0.554 ≈ 0.553,ALFWorld 上 0.841 = 0.841;对 gate-v2 臂级增益的迁移预测,稀疏字典离对角 Spearman +0.08(A3)/ +0.07(Fisher)= 随机字典,不超类别/同侧标签。判定:稀疏原子不成立,测试床功效低。

**T3 镜像蒸馏 + 原对偶训练器**(`src/bfas/mirror.py`、`tools/crcd_mirror_train.py`,12 测试)。目标 q_i ∝ π0·exp(Λ_i Q_i/η),Λ_i 由 λ_k 与加载 z̄ 得到,损失 KL(q‖π_θ) 在候选集上 + off-atom 位移罚 + probe KL。smoke:满足的原子 λ→0、违反的单调上升,Λ 与 teacher 目标质量相关 0.74,log-odds 恒等式误差 1e-16;发现 off-atom 罚项经 Adam 耦合会震荡,改为解耦近端步。

**T4 AppWorld 审计 + smoke**。train 30 场景×3 / dev 19×3 场景不相交;推荐主协议 = app 家族情节(spotify、phone+venmo),test_normal 冻结后只跑一次;环境两遍字节级一致;离线评测器可用。发现 T4 自己的 smoke 脚本漏了 `enable_thinking=false`(已补);9/1 的 dev 14% 不受影响。

## 3. BFCL

### 3.1 BF-1 记账一致版(主锚点)
`tools/bfcl_pool_teacherize.py`:r3 联合池 324 → 302 行(去 8 行 calibration、去 14 行无演示官方题、15 行官方题 y^T 换成 deepseek 演示,273 行生成题标 teacher 编写)。base 学生,pref-only 1 epoch = 38 步。

| | teacher 版 | 同池 GT 目标对照 | 旧 oracle-GT 324 行 | base |
|---|---|---|---|---|
| Overall | **46.74** | 46.04 | 47.82 | 46.06 |
| NL / Live | 82.75 / 78.98 | 82.96 / 78.90 | 82.75 / 78.83 | 79.58 / 77.72 |
| MT / Mem | 51.88 / 25.16 | 52.00 / 23.44 | 52.12 / 30.11 | 50.12 / 25.59 |
| Irrel / Rel | 78.74 / 81.25 | 78.66 / 81.25 | 78.98 / 81.25 | 82.70 / 75.00 |

同池上演示目标 +0.70;与 324 行版的差来自去掉的 22 行/少 3 步(或 Memory 噪声)。

### 3.2 BF-2b 单事件迁移矩阵(T6,正结果)
每个事件单独做 3 步 base-centred pairwise 更新(lr 1e-4;5e-6 低于 bf16 测量底噪),量全部 210 事件上 log π(y^T)−log π(y^S) 的变化。120 行(后补齐 210 行)。自效应全正(中位 0.84);T 对称(0.87)、近秩 1(simple_python 块)。预测实测迁移(25,080 对,prompt 分组 bootstrap):ψ·ψ 离对角 ρ 0.674 [.60,.73]、符号 AUROC 0.774;稀疏字典 0.686 / 0.782;PCA-32 0.527;类别标签 0.608 / 0.605;配对差 ψ−类别 Δρ +0.065 [+.015,+.110]、ΔAUROC +0.169;双中心化后 ψ = 类别(交互结构是类别块)。lr 5e-5 复制(无饱和)排序不变;侧分解:学生侧 push-down 是类别块 + 长度效应(类别赢),teacher 侧迁移 ψ 保持优势(AUROC 0.654 vs 0.556)。方法文档已改:容量子空间 = 白化梯度子空间,字典只作可选基。

### 3.3 BF-3 镜像蒸馏真实子空间版(T7 准备,负)
K=32 PCA 加载,同 302 行池,38 步,每 4 步对偶更新。rho1(ρ=1,λ 顶 5)**24.03**;rhoJ0(ρ=J0+0.3,λ≈4)**35.50**;k1(K=1 全局)**44.74**;三臂 Rel 全塌到 62.5、Irrel 升到 88 → 往弃答塌,损伤随每事件位移 Λ 单调。诊断:镜像配置 lr=1e-4(SFT 默认),pairwise 锚用 5e-6,差 20 倍;K=1 位移只有 0.017 仍掉 1.3 → 优化本身在伤。对照在跑:gentle(λ≤1 + 32 probe 信任域,lr 1e-4)、lrmatch(同 + lr 5e-6)。顺带修了两个 bug:池里 `_traj` 不唯一(302 行 80 个)导致加载错位;缓存键撞。

### 3.4 BF-6 采集离线回放(T11 + 7 臂)
在 302 行池上按 teacher token 成本(每行 3–1,220,演示行最贵)回放 7 种策略。回放本身就发现:池里 ΔU 全 >0(硬过滤=随机),288 行 turn 0(首分歧≈随机),等 token 下成本主导(残差/token 93% = 最便宜优先),等行数下残差规则=覆盖规则(最贵)。

| 策略 | b25(≈4k token) | u50(151 行/19 步) |
|---|---|---|
| random | **47.77**(87 行/11 步) | 46.95 |
| residual/token | 46.97(200/25) | 评测中 |
| cheapest-first | 46.89(243/31) | — |
| consequential-first | 45.46(184/23) | 47.02(只花 57% token) |

等 token 下随机最好、分数与步数反相关(剂量混淆);等行等步下后果性优先 = 随机但省 43% token。残差规则目前 No-Go。

### 3.5 Block IV 认证(T10)
覆盖证书:K=32 子空间对 r3t 指纹的分组交叉拟合覆盖 0.603 [.573,.631];ALFWorld ψ 投 BFCL 基 0.11 = 随机。原子证书(Hoeffding+Bonferroni):BF-3 两臂无原子可认证满足。风险证书:官方单轮 3641 题学生错误率 0.2035 vs base 0.1994,McNemar p=0.28;parallel 显著变好(p=1e-4),live_irrelevance 显著变坏(64 vs 5,p=4e-14);官方校准集 v2(65 题)学生 0.138 vs base 0.169,类别级证书在 τ=0.2/0.3/0.4 均在名义内。

## 4. ALFWorld

- 开发划分参考(valid_seen 140):base 7.14 / CE 后果性 77.86 / pairwise 27.14(valid_unseen 8.96 / 74.63 / 28.36),排序一致。CE 分类别:heat 50、two_obj 62.5 最弱,其余 80–89。
- ψ 字典拟合:稀疏 = PCA(K=32 0.841;K=4 已 0.761),ALFWorld 分歧梯度近一维(T8:PC1 69%)。
- ALF-3 镜像蒸馏 5 臂已提交 hpg(C_rho1 / C_rhoJ0 / C_k1 / A_rho1 / C_strong=gt_teacher κ=20),valid_seen 评测,排队中。base 几乎总偏好 y^S(p(T)≈0.13),存的 ΔU 当效用时 λ=5 只能移 2.1 nat,C_strong 测 CE 式装入。
- ALFWorld 单事件迁移矩阵(T9)测量中。

## 5. AppWorld

- 事件挖掘器(T5,`tools/appworld_event_mine.py`,18→67 测试):官方脚手架回放演示前缀,每决策点采学生动作,分歧处两支各 K 次续跑到终局(新环境重放前缀),官方评测器 success 做 Q,ΔU 全保留;按 API 调用效果判等;thinking 关闭核实。
- AW-1 base 半边:40 条演示,K=3、4 探针,160 事件(ΔU>0 21 / <0 25 / =0 114),0 teacher token,~4.6 h。事故:排队脚本 grep "FINAL" 命中交接注释提前触发并杀了 vllm,4 题补挖。
- 臂(官方 dev57,base 14.0%):CE 后果性 **0/57**、CE 全池 **3/57**——学生退化成循环查 API 文档(49.7 步/题,只有 3 题调 complete_task);根因是事件全在 t≤2,CE 学成"永远先查文档"(首分歧陷阱)。pairwise 两臂 OOM 后带梯度检查点重跑中。
- 修法(Codex 首批任务):`--probe-select spread/late`(探针沿演示均匀分布,一致状态也记录)、自适应输出上限(late 状态 28k prompt 撑爆 32k)。spread 重挖 17/40(52 事件,14 后果性),完成后自动跑 4 臂。

## 6. 文档与台账
METHOD.md(状态标签、记账章节、Block I 结论更新)、审计文档、regression_anchors JSON、exp_log(每条含 commit/事件数/步数/token)、PROJECT_STATE(含 ⟳ 重启清单)。专题文档:transfer-matrix、certification、acquisition-replay、appworld-event-mining、appworld-split-audit、bf3/alf3-mirror-prep。

## 7. 明天优先级
1. 读 BF-3 对照(gentle / lrmatch):决定镜像蒸馏是超参问题还是方法问题。
2. ALF-3 五臂 + ALFWorld 迁移矩阵:看 Block I/III 在不会做的 base 上是否反转。
3. AppWorld spread 事件的 4 臂;若 pairwise 也不行,AppWorld 需要 teacher 半边(等 deepseek 配额)。
4. BF-6 u50 residual 出分后写 Go/No-Go 结论;若 No-Go,采集块按"后果性优先 + 低剂量"写。
5. 把 BF-2b 的迁移预测改成 teacher 侧靶,作为论文 Block I 的主证据。
