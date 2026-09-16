# 零教师候选挖掘:U0 自身 828 个执行状态上"证据已在上下文、学生答错"的盘点(2026-09-16)

范围:用户 11:45 CDT 批准的零教师盘点。不训练、不采购、不改任何训练材料;所有数字都是描述性的。
材料:U0(react7 bank + pure CE,seed 0,贪心,max 7 步)在 200 个 support 父任务上的全部 828 个执行状态
(`results/repair_states/all_u0_states.jsonl`)。工具:`tools/cr_decision_value.py answer-now --no-teacher --dataset`
(dv 树 07f558bc),LONI 作业 1030229(存档思路模式)与 1030230(重新生成思路模式),各 828 状态,gpu4 各约 30 分钟。
输出:`results/repair_astar_r7_s200_u0/mine_{archived,fresh}/`;分层统计 `mining_tiers.json`;人工标注 `results/repair_states/mined_annotations.json`。

## 1. 做了什么

在每个状态上,把学生的思路保留(存档模式)或重新生成(fresh 模式),然后强制它立刻 `Finish[...]`,用 gold 只做打分。
再用 HotpotQA 的 supporting facts(只做标签,不进提示)描述"证据是否已在上下文里",分四个由松到紧的代理:

| 代理 | 定义 | 说明 |
| --- | --- | --- |
| T2 标题被提及 | 每个 supporting 页面的标题字符串出现在之前任一 observation 里(工具字段 `all_supporting_titles_seen`) | 第一跳页面提到第二跳实体名就会触发,不等于取回了该页面 |
| T3 页面已取回 | 每个 supporting 页面都被某次 Search 实际取回(查询等于标题,或页面首句含标题主干) | 主干启发式有少量误报(Sugarloaf、Matecumbe) |
| T4 句子可见 | 每条 supporting 句子的实词在同一条 observation 里召回 ≥ 0.7 | 当前 Wikipedia 与 2017 快照不同,漏报多 |
| G gold 可见 | SQuAD 归一化后 gold 是某条 observation 的子串 | 工具原字段是区分大小写的精确匹配,带引号的 gold 会漏 |

另外用 forced F1 ≥ 0.5 标出"近似命中"(York vs York, North Yorkshire 这类 EM 过严的情况),
并对 T3 集合的 31 个父任务与 T2 集合的 20 个抽样状态逐条阅读标注。

## 2. 分层计数(存档思路模式;fresh 模式几乎相同)

| 层 | 状态 | 父任务 | 其中自主运行 EM=0 的父任务 | 近似命中(F1≥0.5) |
| --- | ---: | ---: | ---: | ---: |
| T1 强制作答错 | 580 | 169 | 88 | 44 状态 / 32 父任务 |
| T2 = T1 且所有 supporting 标题被提及 | 120 | 53 | 36 | 13 / 12 |
| T3 = T1 且所有 supporting 页面已取回 | 79 | 40 | 33 | 13 / 12 |
| T4 = T1 且所有 supporting 句子可见 | 20 | 15 | 15 | 9 / 9 |
| G = T1 且 gold 可见 | 97 | 47 | 31 | 19 / 14 |
| T3 且非近似命中 | 66 | 31 | 24 | — |
| T4 且非近似命中 | 11 | 6 | 6 | — |

两种思路模式:强制答案 828 个状态里 762 个完全相同;对错 2×2 = 两模式都错 574 / 都对 244 / 仅存档对 6 / 仅 fresh 对 4。
重新生成思路不改变任何一层的父任务数(T2 仍是 53 个父任务,STRICT 集 = 两模式都错且标题被提及 118 状态 / 53 父任务)。
这与 242 个采购状态上的结论一致:旧思路不是限制。

## 3. 逐条阅读:T3 且非近似命中的 31 个父任务

| 类别 | 父任务数 | 例子 |
| --- | ---: | --- |
| A 上下文支持,学生确实理解错 | 3 | Rich Hall 页面明写 "and Saturday Night Live",学生追着搜索建议 "Live at the Apollo";Skintight 页面明写 vocalist Liv Kristine,学生答成 Theatre of Tragedy 的男主唱;南通/离石两页都取回,问哪个在江苏,答离石 |
| B 学生答案可辩护,EM 判错(同义、单复数、冗长) | 7 | vocalist vs "singer, songwriter"(当前页面就写 singer);author vs "American writers";magazine vs magazines;band vs "they are both American rock bands";public airport vs "public use airports";director vs "filmmaker and actor";"Moon in Guanche" vs "Guanche religion" |
| C gold 可疑或题目病句 | 2 | "what does Burnham Pavilions and Zaha Hadid have in common" gold=architect;"What song was presented by …" gold=Dancing Stars(是电视节目) |
| D 第二跳尚未取回:学生的下一步动作是对的,只是被迫提前作答 | 9 | Iris Murdoch、Sarah Brightman、Wanda Jackson、Columbia College、Matecumbe、Aleksandrov 等;其中 6 个自主运行最终 EM=1 |
| E 检索落到错误页面或消歧义页 | 4 | Search[Trigg Hound] 重定向到 American Foxhound;Search[USS Intrepid] 是消歧义页;Search[Woman]/Search[Bustle] 落到通用词条;Kevin McCarthy 同名 |
| F 正确页面已取回,但所需句子不在返回的首段,或当前 Wikipedia 已变 | 6 | B*Witched 解散句在页面深处;Tom Kitt 当前首段没有出生日期;Osawatomie 首段只有 2020 年人口(gold 是 2010 年);Rocky 票房;Curaçao 纹章 "trading bond";Steve Israel 的 DCCC 主席句不在首段 |

T2 抽样的 20 个状态里,T3 之外的 8 个:1 个是 A 类(Entr'acte 页面明写 "synonymous to an intermission",学生答 intermède),
其余是 D/E/F 类(Deadpool 2 页面未取回、Tower City station 未取回、Iron City 未取回、Paterno 名字被 observation 截断、
Magdalen 2014 年捐赠数字已不在当前页面)。近似命中的 F1 判据对数字会误判(£150 million vs £180.8 million 的 F1 = 0.5),
所以 B 类要以阅读为准。

**汇总:"证据已在上下文、学生答错"的父任务 = 4 个(SNL、Liv Kristine、南通、intermission),
T2\T3 未抽样部分按抽样比例再加 1–2 个,合计约 4–6 / 200 ≈ 2–3%。**
这与 242 个采购状态上的读出(4 个上下文支持父任务)量级一致。

## 4. 自主运行的失败分解(同一批 200 个父任务,U0 贪心)

| 失败类 | 父任务 | 占 200 |
| --- | ---: | ---: |
| 7 步内没有 Finish(循环搜索、重复查询、消歧义失败) | 40 | 20.0% |
| 有 Finish 但错,F1 ≥ 0.5(近似命中:April 6, 2016 vs 2016;York;Sam Endicott;Mother Jones;2240 feet …) | 15 | 7.5% |
| 有 Finish 但实质错 | 33 | 16.5% |
| 自主 EM=1 | 112 | 56.0% |

实质错的 33 个里,证据可见而理解错的是上面那 4–6 个;其余是检索没取到所需句子(F 类)、落错页面(E 类)、
以及在证据不足时猜答案。近似命中的 15 个在 dev-500 评测里同样按 EM 计 0。

## 5. 结论(不预先决定教什么)

1. 在 U0 自己的轨迹上,"证据已在上下文、学生答错"这一类只有约 2–3% 的父任务。以 ±1–2 pp 的 dev-500 噪声,
   这一类不足以支撑一个机制的训练与检验,也不值得为它投入教师预算。
2. 学生真正丢分的地方按大小排:7 步内没结束(20%)、检索深度不够(所需句子不在返回首段,需要 Lookup 或换查询)与落错页面、
   答案形式与 gold 不一致(7.5%,实质已答对)。这三类都不需要教师来发现;第三类甚至是评测侧的问题。
3. 两种思路模式几乎完全一致(762/828 强制答案相同),"换一段思路再作答"没有任何增益,与采购状态上的结论相同。
4. 代理的局限:标题被提及不等于页面已取回(120 → 79 状态);句子级可见受当前 Wikipedia 漂移影响,只能作下界;
   F1 近似命中对数字无效;强制作答不是自主作答。

## 6. 材料与可复现

- 分层与人工标注:`results/repair_astar_r7_s200_u0/mining_tiers.json`、`results/repair_states/mined_annotations.json`、
  `results/repair_states/mined_cases.jsonl`(120 个 T2 状态含 supporting facts 与可见 observation)、`mined_sample20.json`。
- 脚本(scratchpad,零 GPU):`mining_tiers.py`(四代理 + 近似命中 + 两模式 2×2)、`mining_report.py`(工具字段汇总)。
- 教师消耗:0;训练:0;LONI:2 张 gpu4 × 约 30 分钟。
