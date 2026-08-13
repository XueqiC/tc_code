# SpecDistill v3 — Teacher-in-the-loop 完整方法设计(蒸馏向补救)

日期:2026-08-12 · 状态:设计稿,待 PI 拍板后实施

## 出发点:v2 的三个遗留问题,一个机制统一解决

1. **teacher 角色太弱**("这不就是 LLM training 吗"):v2 里 teacher 只是语料来源,选完就没它事了。
2. **few-shot query 质量敏感**:k 条 query 差 → 需求分布估偏 → atom 覆盖不全,后面全错。
3. **供给受限档位吃亏**:D_atom 在 0.8B×10k 只有 0.56(组合版 0.77)——池子里没货时只能买剩货。

统一解法:**把 teacher 变成"按单供货方"** ——账本哪里缺,就让 teacher 生产哪种轨迹。三个问题分别对应:teacher 深度进环(1)、缺失需求由 teacher 探针补全和验伪(2)、缺货直接下采购单(3)。

## 完整流程(五步,全部内生,零手工开关)

### 第 0 步 输入
k 条 few-shot query、black-box teacher API(只出文本、按 token 计费)、base student(white-box)、总预算 b。**b 现在覆盖所有向 teacher 购买的 token**(初始池 + 补货),预算语义更纯粹:b = 花在 teacher API 上的总 token 数,同时封顶学生训练算力(epoch 固定)。

### 第 1 步 字典:免费学,不花预算
梯度指纹(student 侧 Adam 预条件梯度)可以对**任何**文本算——公开语料、gold 数据、已有轨迹都行,不需要 teacher。稀疏字典(K=64, L1)在这些免费指纹上学,得到 capability atoms。
**要点:atom 词汇表的覆盖度由字典语料决定,与 k 条 query 无关**——query 差不会让 atoms 缺失,只会让"需求估计"缺失。这是补救可行的前提。

### 第 2 步 需求补全:query 质量补救的核心(新机制)
raw 需求 = k 条 query 的稀疏码均值。三个修正,全部自监督:

**(a) 共现补全**。atoms 在字典语料里有共现结构(比如 JOIN 几乎总和 WHERE 过滤同现)。若 raw 需求激活了 A 而与 A 强共现的 B 需求≈0,大概率是 k 条 query 漏了任务的一个侧面。做法:在 atom 共现图上做一步有界扩散,把需求"补圆"。
**扩散强度不是手设的**:留一法自监督——藏起第 i 条 query,看剩 k−1 条经扩散能否恢复它的 atom 分布,选恢复误差最小的强度。query 本身就当验证集用,零新增监督。

**(b) 不确定度记账**。bootstrap/留一给每个 atom 账户配置信区间;只被 1-2 条 query 撑起的"单锚 atom"自动标为低置信(λ 实验早证明单锚危险,同一机制)。

**(c) teacher 探针验伪(蒸馏向补救)**。对补全新增的和低置信的账户,不直接花大钱,先小额下探针单:按该 atom 载荷加权取 pool prompt / query 变体,让 teacher 作答,把回答的指纹再过字典——落回同一 atom → 账户确认;散掉 → 判为噪声,退账。**geometry proposes, behavior/teacher confirms**,与分离原则一致。探针花费记入 b(很小,每 atom 若干百 token)。

### 第 3 步 出清 + buy-vs-ask(teacher 进入市场)
账本开好后逐轮花钱,每轮比较两个选项:
- **buy**:池中最优剩余轨迹的边际收益 = Σ_a min(账户余额, 供给_a) / token 价
- **ask**:向 teacher 定向下单的期望收益——按残余账本构造 prompt(载荷加权),期望供给由**已见 teacher 批次的 atom 组成统计**估计(在线更新,不手设)
剩货收益跌破下单期望收益就 ask;新轨迹入池参与后续轮次。终止:预算耗尽或账清。
供给受限档位(如 0.8B×10k)会自然表现为"早早转入 ask 模式"——这正是 v2 输掉的格子的救法。

### 第 4 步 认证训练(沿用 v2 已验证组件)
选好/补好的语料一次性训练,行为验证停(校准 query 上探 runnable-rate,留最优 checkpoint);conformal 边界与证书**仍只对原始 k 条 query 校准**——补全只改供给侧(训练什么),不扩规格(承诺什么),证书永远相对给定 spec 保守诚实。部署时 gate 把界外请求路由回 teacher。

## 规格不变量(重申,防止方法漂移)
- k 条 query 是唯一边界/证书依据,任何补全、探针、采购都不得注入新规格。
- teacher 只在供给侧出现:生产轨迹、回答探针。它不定义任务。

## 卖点重述
- **对 reviewer**:teacher-in-the-loop closed market——需求定账、存货先清、缺口定向采购、探针验伪;蒸馏不再是"选数据",是带预算约束的主动询价过程。black-box API 设定(按 token 计费)让每个决策都有真实货币语义。
- **query 鲁棒性**:样本复杂度上 atoms 把边界估计从高维均值压到几十个权重;共现补全+探针验伪把"k 条 query 采样噪声"变成可检测、可补救的量。敏感度曲线(质量退化 vs 表现)是主打图。

## 验证实验(优先级序)
| id | 问题 | 设计 | 预期 |
|---|---|---|---|
| E6a | 补全能救漏覆盖吗 | 多侧面任务,故意从 k 中删掉某侧面的所有 query;对比 raw 需求 / +共现补全 / +探针验伪 的分侧面 exec acc | raw 丢掉该侧面,补全恢复大半 |
| E6b | ask 能救供给受限吗 | 0.8B×10k(v2 输掉的格子)开 buy-vs-ask | 追平/反超 0.77 |
| E6c | 质量敏感度曲线 | k 减半/改写扰动/混 1-2 条离题;三条线 LESS vs atom vs atom+补全探针 | 我们最平 |
| E6d | AppWorld 定向补货 | 残余账本驱动 vs 均匀加厚,同 token 预算 | 定向更高 TGC |

## 实施顺序
1. sqljoin 判定格落地 → 确认 D_atom 内生过滤是否成立(决定第 3 步的基线形态)
2. E6b 最小实现(buy-vs-ask 只需在现有 clearing loop 加一个分支 + teacher 批量下单脚本,teacher API 现成)
3. E6a/E6c(共现图 + LOO 扩散 ~100 行;探针脚本与 E6b 共用)
4. AppWorld E6d(排在 4B/9B 基线之后)

## 插件式集成视图(PI QA 2026-08-12 深夜)
v3 = v2 链条 + 三个插件,骨架不变:
- 共现表 = 字典步骤的零成本副产品(池码统计)
- 补全+验伪 = 开账步骤的前处理;探针轨迹本身入池(验伪花费即预购,不浪费)
- buy-vs-ask = 采购循环里多一个候选动作;池从静态货架变为可进货货架
teacher 出场点从 ① 扩展到 ①③④⑥;全链无新增手设参数。

## E6f 真 few-shot 协议 + k-sweep(PI 质疑 2026-08-13 凌晨)
PI 指出:当前受控实验的池 prompt 来自 benchmark train 分布 —— 模拟捷径,
设定名不副实(对比仍公平:baseline 共享同一池)。
真 few-shot 协议(E6f):输入仅 k 条 + teacher API + 公共语料。
- 字典/共现表:公共语料(任务无关资产)
- 池:teacher 从 k 条自举(解 k 条 + self-instruct 变体),全记入 b
- 其余照旧。
差异化优势放大:LESS/SmartAD 无池起步只能均匀生成再排序;
我们按账本定向生成。主实验 = 同 b 下定向自举 vs 均匀自举。
k-sweep(5/10/25/50)把 few-shot 声明变成实测样本复杂度曲线。
受控套件保留现协议作可比赛道;E6f 为 v3 主战场。
优先级:atomg 判定 → E6b → E6f/k-sweep。

## 干净 few-shot 蒸馏 problem setting(PI 定调 2026-08-13 凌晨,待最终拍板)
给定:k 条 query(唯一任务信息)· 单 black-box teacher(API 计费)·
white-box base student · 预算 b(teacher API 总 token)· 免费资产(公开
语料+本地算力)。不给:现成池 / benchmark train 集 / 额外标注 ——
每条 trace 都从 b 购买(解 k 条、变体、探针、补货全计费)。
求:(student, 边界, 证书);Q1 conformal 覆盖,Q2 界内逼近 teacher、
零界外浪费。评价:同 (teacher,k,b) 比花钱效率。
单 teacher 定调:主表 deepseek-v4-pro;多 teacher 降为 ablation /
混合价格市场 future work(deepseek-only 子池重跑选择即可复用数据)。
Baseline 适配:LESS/SmartAD 无池起步 = 均匀自举再排序 vs 我们定向自举。
实验技巧:缓存超池 + 付费采样接口(信息等价于真 API,现有 952 池 /
AppWorld 128eps 直接复用为缓存)。

## 最终方法形态(干净设定版,PI 确认 setting 后 2026-08-13)
核心反转:无货架可挑,方法核心动作 = 决定下一笔钱问 teacher 什么。
0(免费) 字典+共现表 → ① 开账(码均值→扩散补漏→置信区间→W=需求×b)
→ ② 种子单(解 k 条,兼验伪) → ③ 定向采购循环:下单规划(min 算术
用于计划买什么)→ 回货记账(失败也计费;执行验证+边界门,跑偏变体
不入库记损耗)→ 穿插探针 → ④ 账清即停(省钱=战绩,B* 可比)
→ ⑤ 认证训练(固定 epoch+行为停) → ⑥ 交付(conformal+证书+gate)。
对照:baseline 无池起步只能均匀自举再挑;我们下单前钱已知去向。
新诊断指标:损耗率(不合格/超供 token 占比)。
叙事:v2 在货架上把钱花对;v3 按账本向工厂下生产订单 ——
账本从挑货的秤升级为生产计划表。

## 蒸馏侧设计(PI 要求补强 2026-08-13;术语规范:弃用市场比喻)
1. Demand-weighted token-level loss:span 级 CE 加权(atom 组成 x 需求
   对齐度),界外夹带内容 loss 降权而非整条二值取舍。
2. Interleaved acquisition-training:SFT 若干步 <-> 当前 student 上重算
   k 条指纹/需求(已吸收技能需求自然衰减)-> 下轮采样只补未吸收技能。
   分离原则:梯度决定采什么,行为探针决定何时停。
3. DAgger 式纠错采样:学生本地 rollout,执行反馈定位错误中间状态,
   发 teacher 要延续演示(计费),进下轮 SFT。black-box 下唯一可行的
   on-policy 蒸馏(GKD/MiniLLM 需 logits 已被设定排除)。覆盖 vs 纠错
   的预算分配由需求残差决定。
验证:E7a 加权 loss vs 平权(同语料);E7b 交替 vs 一次性(同预算);
E7c 纠错采样 @ AppWorld。
方法两条腿:数据获取侧(需求估计->定向采样)+ 蒸馏侧(1-3)。

## AgentOPSD (arXiv 2608.05987) insights 吸收(2026-08-13)
论文:turn 级信用 = 对最终成功信念的边际修正(log-odds 递归贝叶斯),
self-teacher(skill 条件化)likelihood 差做证据,reshape GRPO advantage;
ALFWorld 7B 89.1%。
吸收两条:
1. E7a 定型为 TURN 级 demand-weighted loss(their ablation: turn 89.1 >
   token 85.9 > trajectory):每 turn 指纹(per-step stacking 基建现成)
   -> 码 -> 需求对齐度 = turn 权重。
2. 权重乘历史衰减项(gamma 递归累积,被前文覆盖的重复套路降权):
   静态对齐度 -> 历史感知边际贡献,与 buy-vs-ask 边际逻辑同构。
定位:他们 self-teacher RL(无外部 teacher/预算),我们 black-box teacher
预算化蒸馏;SkillBank 文本 skill vs 我们梯度空间 atoms(接 Fig1a 故事)。
Related work 新增簇:OPSD/RLSD/SDAR/StepOPSD/AgentOPSD(蒸馏信号做
信用分配)。附带 ALFWorld/WebShop 参考数字(Line-1 对表)。

## 定稿快照:v3 完整方法四阶段(2026-08-13 发 PI 版)
阶段0 任务无关预备(免费):公开语料指纹 -> 字典(K atoms)+共现统计
阶段1 需求估计(全部 k):码均值 -> 共现扩散(留一定强度)-> bootstrap CI -> w·b
阶段2 种子生成(计费):teacher 解 k 条,执行验证,探针确认低置信分量
阶段3 定向获取-训练交替循环:
  (a) 定向合成(欠供 atoms,载荷加权 self-instruct;录取=执行+边界分,
      不合格记 overhead)
  (b) SFT:turn 级加权 loss(需求对齐 x 历史衰减边际新颖度)
  (c) on-policy 纠错(DAgger,black-box 兼容):cal 条 rollout 失败态
      -> teacher 延续演示
  (d) 需求刷新(当前 student 上重算 k 条指纹,已吸收自然衰减)
  停:行为探针收敛且需求残差平;省下的 b = B* 战绩
阶段4 认证部署:fit 子空间+cal conformal 阈值,探针选 checkpoint,
  gate 界外回退 teacher。
原则:梯度定获取与权重,行为定停止与证书;k 条唯一规格。
指标:in-task 官方分、B*(tau)、off-task 残留、overhead 率。

## 阶段0取消:全任务本地化(PI 定调 2026-08-13)
字典改在线:v0 只在 k 条指纹上拟合(atom 数由能量准则定,不手设);
每轮买回 trace 增量更新;新 atom 诞生信号 = 新 trace 重建残差高(内生)。
查漏升级:不再靠公开语料共现先验 —— teacher 解 k 条的回答码里持续
出现、query 需求侧很弱的 atom = 任务隐式技能(候补需求,过探针/行为
验证再入账)。信息源 = 老师做题实际用到什么。
代价(论文写明):跨任务摊销取消;k 条与 teacher 解答均不触及的技能
不可恢复(信息论边界)。
新四阶段:① 初始化(k->字典v0+需求v0) ② 种子生成(解k条,回答揭示
隐式需求) ③ 获取-训练交替(定向合成/turn级加权SFT/DAgger纠错/需求
刷新+字典增长) ④ 认证部署。方法全程只见 k 条 + 自购 trace。
实现迁移:现行池上拟合已接近;主要改动 = 增量更新 + 残差触发加atom。

## Agent 蒸馏 vs 普通 LLM 蒸馏:六条本质差异(2026-08-13,文献综合)
1. 交互闭环->误差复利(covariate shift):C2M/SOD 解;我们 DAgger 纠错,
   预算按需求残差分配覆盖vs纠错
2. 监督单位=决策非 token(格式脆断,全对或全废):StructuredAD span,
   AgentOPSD turn>token;我们 turn 级加权 loss
3. 成功信号稀疏但可执行验证(免费判官,文本蒸馏没有):我们用在
   录取过滤/行为停/证书三处
4. 能力=离散技能组合:kang2025 工具外挂/AMD 记忆库/SkillBank 文本表示;
   我们 atoms = 技能的梯度空间操作化(定向采样的前提)
5. 错误恢复在成功轨迹中稀缺,student 最需要:纠错采样从真实失败态收集
6. 能力会执行、越界有后果 + 预算化获取:边界+conformal 证书+gate ——
   文献空白,我们的护城河
Problem setting 改写:经典 few-shot 式 support set S={q_i}~P_T,
除 S 零任务信息;baseline 同协议。落点 method.tex。
文献:kang2025 (2505.17612, first-thought prefix / self-consistent action),
C2M (2509.14257, student rollout + teacher correction + RL),
StructuredAD (2505.13820), SOD (2605.07725), AMD (2608.07169),
AgentOPSD (2608.05987)。

## 四个升级点(2026-08-13,对照六差异查缺)
E8a 可学性条件生成:生成 prompt 要求 teacher 最简/规范/少风格噪声
    (first-thought prefix 启发);指标 = 同 atom 覆盖的 token 花费、
    每块钱需求清偿率
E8b 二级加权 loss:turn 权重(需求对齐x边际新颖) x span 类型
    (动作 span 严格 / 推理 span 轻量,StructuredAD 启发)
E8c SFT 后短 RFT 打磨:calibration query 上环境成功率作奖励,零 teacher
    预算(E4c SFT->RFT 基建复用);C2M 证明超越纯模仿上限
E8d 推理期免费武器:self-consistent action generation(执行表决)+
    工具调用约束解码(封死格式报废)
优先级:E8a/E8d 最便宜见效快,atomg 判定后先做。

## 蒸馏侧定稿(method.tex 已重写)+ 实验矩阵对齐(2026-08-13 晚)
蒸馏三创新(全部 atom 分辨率,统一原则:指挥花钱的词汇表同时指挥学习):
1. 需求加权 loss(权重来源=配额,非启发式)
2. 技能分辨吸收监控:per-atom 需求衰减曲线;饱和->降权(治 M17 surplus),
   停滞->触发定向补采;行为探针仍是唯一裁判
3. 技能分辨纠错:失败 rollout 特征过字典命名缺陷技能,纠错 prompt 按
   atom 合成、预算按 atom 失败率分配(超越状态级 DAgger)
实验矩阵:今晚受控套件五臂(uniform/emb/atom/atom2=+补全+加权/base,
0.8B x {2k,10k} x 3 seeds,指标含浪费率与 B*);明天 AppWorld:
吸收监控消融(饱和降权 vs 固定;停滞触发 vs 不触发)+
纠错消融(atom 级 vs 状态级 DAgger vs 无)。
