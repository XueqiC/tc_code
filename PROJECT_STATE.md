# tc-alignment — state

> Claude: 每完成一段工作就更新本文件。重启或上下文压缩后,先读这里再干活。

## Research question / goal
**(2026-08-09 用户重构后的核心问题,目标 ICLR)**

给定一个基于大 LLM 的 agent 和 k 个 few-shot query(k ∈ 几 ~ 几百),要一个方法:
1. **边界估计**:从这些 query 准确判断 topic/task 的边界 T(哪些请求属于该任务域);
2. **匹配蒸馏**:蒸馏出小 LLM,使基于它的 agent 的能力边界 C **恰好**满足 T——
   任务(T)与能力(C)的边界对齐/匹配问题。
约束:agent 中的 LLM 用 **code** 完成任务,边界的表示与蒸馏都应建立在代码行动空间上
(候选表示:query 在 teacher 上诱导的代码足迹——tool/API 闭包、组合结构、数据流深度)。

用户已定调(2026-08-09 晚):
- 方法不必沿用原稿 IR,可整体重新设计;
- 边界估计走 **gradient → spectral space → stack** 的路线;
- code vs JSON 非重点,重点是把它做成一个研究问题;
- **capability shaping** 的故事方向认可。
方法草图见 docs/2026-08-09-tc-boundary-gradient-spectral-sketch.md:
梯度空间统一 T 与 C(query→teacher 轨迹→LoRA 梯度特征;T=谱子空间+conformal
半径;C=残差梯度小的区域;蒸馏=按投影能量比选数据 ± 更新投影;逐层/逐步
stack 成层级边界)。诚实问题:C⊆T 受 base 先验限制(capability vs behavior)。

### 背景:此前的 ICLR 草稿(docs/iclr-draft-intro-and-methodology.md)
原 working title: *Which Tasks Survive Distillation: Action-Space Notation as a
Training-Time Control on Agent Capability*。
原核心问题:蒸馏轨迹里工具调用的**记法**(code vs JSON)是不是训练时决定
"学生能做哪些任务"的变量?两个学生 IRT 对齐 ability 后比 per-task 成败模式,
检验差异是否沿三个结构坐标组织:tool closure、composition shape、dataflow depth。
重构后,IR + 结构坐标机制可能复用为"定义/度量边界 T 与 C 的坐标系"。

设计要点(详见 docs/iclr-draft-intro-and-methodology.md):
- 单一操纵变量:teacher 轨迹先解析成 notation-neutral IR,再由 IR 分别渲染成
  code / JSON 两个训练语料;render↔parse round-trip 机器验证等价(这是论文卖点之一)。
- JSON 记法用独立的 from_steps 引用表承载依赖,不内联 marker(避免把字面量误读成依赖边)。
- 两个学生:同 base、同 size、同超参同 seed,只差语料记法。
- 对照:接口扰动(改名+改 schema 重测)、无预算上限的 scaffolding baseline(带预注册停止规则)、
  floor exclusion(双零任务剔除报 censored)、truncation 断言(无截断)。
- 已量化的 confound:同一轨迹 JSON 比 code 贵 ~1.40× tokens,且随 composition shape 变化
  (branching 1.48× vs independent 1.34×)——与结构坐标共线。因此预注册**单侧**主张:
  code 赢是保守可报的,JSON 赢不可主张。
- 任务边界:按 topic 声明 tool scope(不做 per-request 预测——Barber et al. 不可能性),
  topic-conditional conformal coverage(Gibbs et al. 2025)。
- 次要仪器:gradient spectra(LESS 式 L2 归一 + format-permutation null 检验)。

## Open problems(draft 里自己标注的)
1. **动机的实证前提未支撑**:「依赖链式请求是蒸馏后退化最集中处」需要一个可引用的报告,
   或我们自己在公开 tool-use benchmark(StableToolBench / BFCL 系)上量一次。
2. **训练侧长度 confound 未解决**:per-task length control 尚待设计(评测侧已用不设上限解决)。

## Current status
- [x] scaffolded
- [x] 读完 intro+methodology 草稿,已存 docs/
- [x] 2026-08-09 用户重构了核心问题(T-C 边界对齐,见上)
- [x] 方法草图 v1:docs/2026-08-09-tc-boundary-gradient-spectral-sketch.md(已发 Discord)
- [x] novelty 扫描 v1:docs/2026-08-09-novelty-scan.md —— 结论:问题表述
  (distill-to-specification, 双侧 capability shaping)在 25-26 文献里无先例;
  最近邻 = SmartAD(ACL26,容量迁就非边界对齐)、SCOPE(conformal OOD 拒绝,
  serving 侧)、Learnware(spec 检索,我们是其生成式对偶)、LESS 谱系
  (梯度选数据,无任务 spec 角色)。风险:agent distillation 赛道拥挤,
  写作重心必须是问题表述+可度量性+保证。
- 草稿提到的另外两个文件(problem-setting-and-method.md、
  2026-08-06-action-space-capability-shape-design.md)尚未提供

## Running jobs
(none — 格式: host | pid或jobid | 在跑什么 | 启动时间 | 预计结束)

## Key results so far
(none — 1.40×/1.48×/1.34× token 比例与 round-trip 捕获的 3 个缺陷来自草稿所述前期工作,
 工件不在本仓库)

## Next steps
(待与用户确认优先级)
- 候选 A:先补动机实证——在公开 benchmark 上测一次蒸馏后 shape-correlated 退化(draft note 要求)
- 候选 B:搭 IR + 双 renderer/reader + round-trip 检查的代码骨架
- 候选 C:设计 per-task length control(§2.4.2 遗留的开放问题)
- 需要用户提供:teacher/student 选型、tool 环境与任务面板选择、以及草稿引用的另两份文档
