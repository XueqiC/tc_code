# 教师接管筛选实验:24 个冻结失败状态上的 U0 / 教师 1 步交还 / 教师 3 步交还 / 教师做完后缀(2026-09-16,14:07 CDT)

范围:用户 12:51 CDT 的指令。一次教师采集,零训练;只回答三个问题:在相同工具和 7 步上限下教师能教会学生什么、需要多长的交互片段、
以及每次恢复依赖了什么新信息。所有数字都是描述性的,24 题只是筛选实验。

## 1. 设置

- **选题(教师调用前冻结,`results/repair_states/takeover_selection.json`)**:support 里自主运行 EM=0 的父任务;三层:E 错误页/消歧义页 4(全部),
  F 正确页面已取回但所需句子不在返回首段或当前 Wikipedia 已变 7(全部),U 未完成、重复或无进展交互 13(从 30 个候选里 seed 20260916 随机抽)。
  每题一个真实状态 = 第一个故障信号(Could not find / 消歧义页 / 重复查询)之后一步,上限第 5 步,即剩余 3–6 步;状态在当前 Thought 生成之前。不按教师是否成功换题。
- **教师接管**:gpt-5.6-luna 官方 API(flex),从同一状态用同一 episode runner 续跑:看到任务、工具说明、few-shot 示例、真实转录(学生此前的 Thought/Action/Observation),
  不看 gold,不看学生当前 Thought;外加一段固定的回复格式说明(sha256 记入 identity;没有它教师会回裸字符串)。同一查询复用冻结缓存;新查询实时取回并写入缓存,再同步给学生侧。
  硬上限 30,000 计费输出 token。工具 `tools/cr_takeover.py`(dv b9810f47 / b62926c2),两处冻结缺陷的处理见 PROJECT_STATE。
- **学生分支(LONI,冻结 U0_s0,贪心)**:k0 = U0 从该状态自行继续;k1 / k3 = 继承教师前 1 / 3 步(含真实 observation 与已消耗步数)后交还 U0;总上限仍是 7 步。
  教师在 k 步内已 Finish 或步数用尽时不可交还,分支继承教师结果并标记(k1 有 5 个,k3 有 10 个)。学生续跑先按离线缓存跑了一遍,发现新查询不在缓存会被截断(13/57 个续跑),
  改为工具在线后重跑;两次结果的 EM 逐题相同,下面用在线结果。
- **读出**:G(k) = 平均 EM(教师 k 步后交还)− 平均 EM(U0 自行继续),父任务配对 bootstrap;两类成功分开:教师直接 Finish 答对 vs 交还后学生自己 Finish 答对。

## 2. 主结果

| 分支 | EM | 其中学生自己 Finish 答对 | 其中教师已 Finish(继承) | G(k) | 95% CI | 好/平/差 |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| k0 U0 自行继续 | 2/24 | 2 | — | — | — | — |
| k1 教师 1 步交还 | 5/24 | 2 | 3 | +0.125 | [−0.04, +0.29] | 4 / 19 / 1 |
| k3 教师 3 步交还 | 8/24 | 2 | 6 | +0.25 | [+0.08, +0.42] | 6 / 18 / 0 |
| 教师做完后缀 | 12/24 | — | 12 | +0.42 | [+0.21, +0.63] | 10 / 14 / 0 |

- 教师采集:24/24 都到 Finish,86 步 86 次调用,0 个无效动作,13,041 计费输出 token(每题约 540);无采购失败。
- k0 是 U0 的新贪心样本,不等于存档轨迹:与存档前 1–2 步逐字相同后分叉(vLLM 贪心不完全确定),24 题里这次自己答对 2 题(SNL、Peter Brown/Oz)。
- **交还后学生自己答对、且 k0 没答对的父任务只有 2 个**:k1 的 intermission,k3 的 Osawatomie。G(k) 的其余部分都是教师已在 k 步内 Finish 的继承结果。

## 3. 教师的 12 个成功靠什么(逐条阅读,`results/repair_states/takeover_teacher_annotations.json`)

| 来源 | 父任务数 | 例子 |
| --- | ---: | --- |
| 通过工具交互取得了回答所需的新证据 | 4 | Osawatomie:Search 失败 → Search[Osawatomie, Kansas] → **Lookup[2010]** 取到 "4,447";Rich Hall:在已取回页面里 **Lookup[live]** 取到 SNL;Entr'acte:直接搜题目里的实体;Peter Brown:用 Similar 列表消歧义到 Peter Brown (Oz) → Pirates in Oz → 插画师 |
| 取得部分证据,最终仍靠知识 | 3 | USS Intrepid(消歧义到 CV-11,但退役年份句没取到);Woman/Bustle(Bustle 2013 取到,Woman 页没取到);B*Witched(成员页取到,解散句没取到) |
| 没有新证据,凭知识作答 | 4 | Trigg Hound(1 步)、Steve Israel/DCCC(1 步)、Tom Kitt 生年(页面首段没有)、That's All Right(搜到邮票页) |
| 凭学生已有上下文作答 | 1 | UWF 校园面积 |

教师的 12 个失败:证据没取到而凭知识猜错 7(Denny Laine、Rocky 票房、Hosty Duo 城市、Gingrich 的书、Curaçao 纹章、Jessica Drake 前夫的节目、Paddington vs Paddington 2),
近似命中 3(Tony Haygarth、Cable、£180 million),如实答 Unknown 1,gold 本身有问题 1(Dancing Stars)。工具的 gold 可见性统计(正规化后 gold 出现在教师新观察里)给出 6 / 6,
比阅读多算了 B*Witched 与 Woman(gold 字符串出现在页面里,但支撑句没有)。

## 4. 交还是否有效,取决于教师的成功来源

| 教师成功来源 | 可交还分支里学生的结果 |
| --- | --- |
| 取得新证据(4) | **决定性检索步已在交还前缀里时 4/4 学生自己答对**:SNL k1 ✓(Lookup 结果在第 1 步)、intermission k1 ✓、Osawatomie k3 ✓(k1 只含失败的搜索,✗)、Peter Brown k3 ✓(k1 只含错的插画师页,✗) |
| 部分证据 / 纯知识(8) | 可交还的分支里学生 0 次答对:USS Intrepid k1 ✗ k3 ✗、Woman ✗ ✗、B*Witched ✗ ✗、Tom Kitt ✗ ✗、That's All Right k1 ✗;1 步知识 Finish(Trigg、Israel、UWF)无法交还 |

也就是说:**教师相对学生的可迁移优势 = 检索交互(在已取回页面里 Lookup、用题目里的实体换查询、用 Similar 列表消歧义),在这 24 个失败状态里出现 4 次(17%);
教师另外 8 个成功来自知识,交还无法传递。** 交还所需的片段长度就是"到决定性检索步为止":1 步(Lookup/换查询直接命中)或 3–4 步(先消歧义再取页)。

## 5. 对照用户的决策表

| 观察 | 本次结果 |
| --- | --- |
| 教师一步后 U0 就经常恢复成功 | 否:k1 可交还 19 个里学生自己答对 2 个(其中 1 个 k0 也对) |
| 一步帮助很小,连续几步后 U0 能恢复 | 只在证据型案例里:k3 多恢复 Osawatomie、Peter Brown(需 3–4 步取到决定性观察) |
| 只有教师完成整个后缀才成功 | 教师完整 12/24,但其中 8 个是知识型,不是"后缀太长" |
| 教师主要靠直接给答案获得成功 | 大体如此:8/12 依赖知识;检索型 4/12 |
| 教师在相同协议下也很少成功 | 否:12/24(近似命中另 3) |

## 6. 限制

- 24 题、单个教师样本、单个学生贪心样本;G(k) 的 CI 宽;k0 与存档轨迹不同(贪心不确定性),两者都报。
- 三层都是检索相关的失败类,4/24 不能外推到全部 88 个失败父任务。
- 教师在事实不在首段时倾向于猜而不是继续用 Lookup(7 个知识猜错),这既限制了教师完整后缀的上限,也说明"检索深度"技能在教师身上也不稳定。
- 教师只看学生 prompt + 格式说明;换成显式的"先 Lookup 再答"指令会改变教师行为,本次没做。

## 7. 文件

- 教师:`tc-alignment-dv/results/takeover/`(takeover.jsonl、branch_k1/k3.jsonl、ledger.jsonl、receipt.json);学生:`results/takeover_{k0,k1,k3}_online/`(离线一遍在 `takeover_{k0,k1,k3}/`);
  报告:`results/takeover_report_online/report.{json,md}`;选题与标注:`results/repair_states/takeover_selection.json`、`takeover_states.jsonl`、`takeover_teacher_annotations.json`。
- LONI 作业:tk_k0/k1/k3 1030412–4(离线),tk_k0on/k1on/k3on 1030434–6(在线)。教师消耗 13,041 计费输出 token;训练 0。

## 附:逐题表(k0/k1/k3 单元格:EM 后缀 S = 学生自己 Finish,× = 步数用尽或答错未 Finish,† = 继承教师结果)

| 父任务 | 层 | 起步 | k0 U0 | k1 | k3 | 教师完整 | 教师成功来源 | 说明 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5a7470ca | E | s4 (3 步已用) | 0× | 1† | 1† | 1 (1 步) | knowledge | 1 step: Finish from knowledge (Colonel Trigg, Kentucky); no new observation |
| 5a7a81b8 | E | s3 (2 步已用) | 0× | 0× | 0× | 1 (4 步) | partial | resolved the disambiguation to USS Intrepid (CV-11) (new page) and Lookup[Decommissioned] gave the post-war se |
| 5ac26049 | E | s3 (2 步已用) | 0× | 0× | 0× | 1 (5 步) | partial | Search[Bustle (magazine)] gave founded 2013 (new evidence); Woman magazine page never retrieved; 'Woman first' |
| 5ac2c2c9 | E | s5 (4 步已用) | 0× | 0† | 0† | 0 (1 步) | near-miss | 1 step: answered Cable from knowledge (gold 'Nathan Summers / Cable'); Deadpool 2 page never retrieved |
| 5a79f16c | F | s4 (3 步已用) | 0× | 0× | 0× | 1 (4 步) | partial | retrieved Sinéad O'Carroll's page (member of B*Witched) but the split sentence never appeared; the 'decided to |
| 5a8787b4 | F | s3 (2 步已用) | 0× | 0× | 0× | 1 (5 步) | knowledge | retrieved Tom Kitt (musician) page (no birth year in the live lead), Lookup[born] empty; 1974 from knowledge |
| 5a8d7a25 | F | s3 (2 步已用) | 0× | 0× | 1S | 1 (4 步) | evidence | Search[Osawatomie] failed → Search[Osawatomie, Kansas] → Lookup[2010] returned 'As of the census of 2010, ther |
| 5ab55d01 | F | s2 (1 步已用) | 0× | 0× | 0× | 0 (6 步) | wrong-knowledge | Rocky franchise page retrieved twice, Lookup[worldwide box office] empty; guessed $1 billion (gold $1.4 billio |
| 5ac2773a | F | s3 (2 步已用) | 0S | 0S | 0× | 0 (5 步) | wrong-knowledge | the 'trading bond' sentence never retrieved; paraphrased from knowledge |
| 5ae0636a | F | s2 (1 步已用) | 0× | 0× | 0× | 0 (6 步) | near-miss | Magdalen page retrieved, Lookup[endowment]/[2014] did not contain the 2014 figure; answered £180 million (gold |
| 5ae73164 | F | s5 (4 步已用) | 0S | 1† | 1† | 1 (1 步) | knowledge | 1 step: Finish New York from knowledge (DCCC chair); the DCCC sentence was never retrieved |
| 5a79e3b5 | U | s3 (2 步已用) | 0S | 0S | 1† | 1 (2 步) | knowledge | search hit the postage-stamp page (irrelevant); answered That's All Right from knowledge |
| 5a7fc8b8 | U | s3 (2 步已用) | 0S | 0† | 0† | 0 (1 步) | near-miss | answered Tony Haygarth; gold is the full legal name George Anthony David Haygarth (EM strictness) |
| 5a8105cf | U | s3 (2 步已用) | 0× | 0× | 0× | 0 (5 步) | wrong-knowledge | no discography retrieved; guessed Master Suite (gold Wings On My Feet) |
| 5a850b0e | U | s2 (1 步已用) | 0× | 0× | 0× | 0 (6 步) | gave-up | 'Fear (song)' disambiguation never resolved; teacher answered Unknown honestly |
| 5a8ff2c3 | U | s4 (3 步已用) | 1S | 1S | 1† | 1 (2 步) | evidence | Lookup[live] inside the already-retrieved Rich Hall page returned the Saturday Night Live sentence → Finish (L |
| 5a908fc8 | U | s4 (3 步已用) | 0× | 1† | 1† | 1 (1 步) | existing-context | 1 step: answered from the campus size already visible in the student's history (UWF 1,600 acres) plus judgemen |
| 5ab323da | U | s2 (1 步已用) | 0× | 0× | 0† | 0 (2 步) | wrong-knowledge | retrieved Paul King (director) page; answered Paddington (gold Paddington 2); defensible but not EM |
| 5ab57c5b | U | s2 (1 步已用) | 0× | 1S | 1† | 1 (2 步) | evidence | Search[Entr'acte] (the entity named in the question) returned 'synonymous to an intermission' → Finish (better |
| 5ab6f4b9 | U | s2 (1 步已用) | 1S | 0× | 1S | 1 (5 步) | evidence | failed searches → used the Similar list → Search[Peter Brown (Oz)] identified Pirates in Oz (1931) → Search[Pi |
| 5abca037 | U | s2 (1 步已用) | 0× | 0× | 0× | 0 (6 步) | wrong-knowledge | Hosty Duo found only as a song credit in Music of Oklahoma; guessed Oklahoma City population (gold 110,925 = N |
| 5ac1ad29 | U | s4 (3 步已用) | 0S | 0S | 0× | 0 (4 步) | wrong-knowledge | four failed searches; picked 'Skin in the Game' from a Similar list (gold The Unwinding) |
| 5ae43a2e | U | s3 (2 步已用) | 0× | 0× | 0× | 0 (5 步) | wrong-knowledge | identified Brad Armstrong (ex-husband) via knowledge; hosted show never retrieved; guessed The Sex Factor (gol |
| 5ae7888f | U | s5 (4 步已用) | 0× | 0× | 0† | 0 (3 步) | gold-artifact | gold 'Dancing Stars' is a TV show, not a song; teacher guessed Rise Like a Phoenix |

k0: records 24 | live 24 | EM(all) 2.0/24 | EM(live) 2.0/24 | student Finish 7 | terminations {'step_cap': 17, 'finish': 7}

k1: records 24 | live 19 | EM(all) 5.0/24 | EM(live) 2.0/19 | student Finish 5 | terminations {'finish': 5, 'step_cap': 13, 'format_failure': 1}

k3: records 24 | live 14 | EM(all) 8.0/24 | EM(live) 2.0/14 | student Finish 2 | terminations {'step_cap': 12, 'finish': 2}
