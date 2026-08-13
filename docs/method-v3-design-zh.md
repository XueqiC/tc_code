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
