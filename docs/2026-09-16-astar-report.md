# 2026-09-15/16 夜间报告:A*/B* 检验——材料有教学差异、动作监督强度严格一致时,学生前缀是否改善自主执行

> 时间为 America/Chicago(CDT)。承接 `docs/2026-09-15-repair-distillation-report.md`(三臂 A/B/C 的"未检出稳定优势、也未建立等价")与你 21:28 的决定。
> 数字来自 LONI 树 `tc-hotpotqa-repair` 的冻结根 `runs/conditional_response/repair_astar_r7_s200_u0` 与 rai 的采购账本。

## 0. 一页结论

【待填:结果】

## 1. 命题与设计

命题(你的 §4):**在同一批确有教学差异的决策状态上、监督相同的正确推理与动作时,仅将动作训练前缀换成学生自己产生的思路,是否改善自主执行?**

| 项目 | A* | B* |
|---|---|---|
| 正确推理监督 | 学 s → r^T | 完全相同 |
| 动作监督 | 学 (s, r^T) → a^T | 学 (s, r^S) → a^T |
| 动作、终止标记与出现次数 | 相同 | 相同 |
| 归一化分母与系数 | 相同 | 相同 |
| 学生错误 Thought | 不作监督目标 | 只作输入(零权重,不进分母) |

实现:A* 一条序列(目标 `Thought N: r^T` + `\nAction N: a^T` + 边界 + 格式尾);B* 两个视图(推理视图目标 `Thought N: r^T`;动作视图掩码 `Thought N: r^S`,目标 `\nAction N: a^T` + 边界 + 格式尾)。
A* 作批次参照,B* 同包同批;逐包、逐批 reasoning / action / boundary / formatting 目标 token 数与全局分母恒等(代码断言,写入 freeze_receipt.equality_invariant);
λ=0 纯 CE、token 比例权重;两臂都从 U0 seed-0 LoRA 起、AdamW 新状态、lr 1e-5、clip 1;两种子只改批次顺序;对照 = U0 原样合并同批评测。
剂量按新银行曝光定:3 遍规则(与 U0 主干一致)→ 20 步(每遍目标 token 3,272 = reasoning 2,272 + action 856 + 边界/格式 144;20 步 × ~530 = 10,789),不复制 113 步。

## 2. 材料:独立教师 + 差异提名(零测试信息)

1. **候选**(预注册):排除已用的 32 父任务,seed 20260917 打乱余下 168 个 support-200 父任务取前 140,每父任务从其 U0 greedy 执行步(含 Finish)均匀抽 2 步 → 280 状态。
2. **教师独立作答**(不给学生 Thought、不给拟议动作;react7 两行格式):8 状态干跑 ok 后全量;280 中 242 ok、1 格式失败、1 供应商错误、**36 未买(达 cap 24,000 即停)**;
   计费 output 24,000(推理 10,480)、prompt 272,535、≈ $0.041;今日教师累计 29,207 / 30,000。教师动作 = 学生拟议动作 **62%**(修正提示时 85%:看到学生方案后的认可倾向是真的);对任务有用 72%。
3. **提名与分层**(冻结规则;差异只提名,不作对错标签):243 包 → 72 提名 = 72 选中 / 52 父任务;标签 missed_finish 27、wrong_entity 17、other_difference 15、repetition 8、premature_finish 5;
   排除 same 151、equivalent_query 14、legal_alternative 5;分层 teacher-Finish 32 / continue 40;步位置 1–7 都有;每父任务 ≤ 2。选中包教师 output 9,973。
4. **留出集**(local-check):seed 20260918,排除训练父任务 → 64 状态 / 32 父任务,无重叠。

## 3. gap 打分(描述性;U0 上逐包,nats / action+边界 token)

| 分组 | n | ℓ_T 均值 / 中位 | ℓ_S 均值 / 中位 | d = ℓ_S − ℓ_T 均值 / 中位(q05, q95) |
|---|---|---|---|---|
| 全部 | 72 | 0.153 / 0.026 | 1.239 / 1.104 | **1.086 / 0.993**(−0.02, 2.31) |
| missed_finish | 27 | 0.102 / 0.020 | 1.565 / 1.496 | 1.463 / 1.494(0.69, 2.37) |
| wrong_entity | 17 | 0.091 / 0.013 | 1.114 / 1.278 | 1.023 / 1.086(−0.04, 1.96) |
| other_difference | 15 | 0.329 / 0.250 | 0.871 / 0.919 | 0.543 / 0.403(−0.05, 1.14) |
| repetition | 8 | 0.166 / 0.032 | 0.827 / 0.677 | 0.660 / 0.564(0.01, 1.71) |
| premature_finish | 5 | 0.093 / 0.040 | 1.670 / 1.858 | 1.577 / 1.626(0.79, 2.32) |
| 教师决策 finish | 32 | 0.116 / 0.016 | 1.444 / 1.365 | 1.327 / 1.325(0.23, 2.34) |
| 教师决策 continue | 40 | 0.182 / 0.037 | 1.075 / 0.927 | 0.893 / 0.783(−0.04, 2.18) |

读法:随机材料的前置检查里每 token S−T 只有 0.20;这批状态上同一动作在学生自己的思路下难 5 倍以上——"有教学差异"成立;A* 的动作项照旧几乎免费(ℓ_T 中位 0.03)。

## 4. 训练(20 步各;每步分母 A*/B* 完全相同)

批均每 token CE:A* 0.81 → 0.65(s0)、0.78 → 0.66(s1);B* 1.08 → 0.78、1.06 → 0.81;B* 高出 ≈ 0.27 ≈ 动作占比 26% × d 1.09,与 gap 一致;anchor ≡ CE(λ=0);3 遍剂量下没有记忆到零。

## 5. 结果

【待填:结果】

## 6. 判读与分析

【待填:结果】

## 7. 过程与文件

- 工具:`tools/hotpotqa_repair_pool.py`(--prompt-mode independent、candidates、nominate;repair 树 d392f6b2)、`tools/cr_repair_cell.py --design astar` + `cr_repair_gap.py` + local v2 + analyze v2(hq 33b93f3a);
  LONI 树代码 = 冻结 v6 清单(pin_check 75/75)+ 新工具;作业 freeze 1028325 / gap 1028326 / train 1028327[0-3] / evaluate+local 1028328[0-4] / analyze 1028329。
- 账本:`tc-alignment-buy/envs/hotpotqa/teacher_pool_scan/{astar_dry8,astar_v1}`;候选/提名/留出:`results/repair_states/{astar_candidates.jsonl,astar_packages_all.jsonl,astar_nominate/,astar_heldout/}`;
  结果:`results/repair_astar_r7_s200_u0/`(gap_summary、frozen_protocol、runs/*/evaluation.json、local_check.json、analysis/)。
- 失误:全量采购在我的 shell 10 分钟上限处被杀(228/280 已买),`--resume` 续完,无重复计费;干跑 8 状态在全量里被再买一次(≈ 600 token)。
