# 论文一 Few Shot 黑盒 Agent Distillation 研究交接文档

**版本：2026-10-04。** 工作题目：*Sufficient Experience for Black-Box Agent Distillation*。题目用于说明研究对象，尚不是最终论文标题。本文是截至该日期的完整交接基线；后续经过验证的实验记录可以更新它。没有用户确认的新约束不能被自动加入。

> 状态：研究问题、文献边界与实验执行草案。本文没有报告任何新实验结果；没有模型训练、基线复现或假设验证已经完成的含义。接手者应先完成接口检查和发现实验，再决定最终方法与论文主张。
>
> 适用对象：未参与此前讨论的研究者或 coding agent。本文独立定义研究目标、设置、资源、比较对象、实验协议和执行要求。研究方向二是利用真正的世界模型提供蒸馏学习环境；论文一应能独立成立，不以论文二成功为前提。

## 1. 已经确定的目标与不可误读的设置

我们研究从强教师向小模型迁移闭环 Agent 能力。目标是在一个固定的工作流／任务类型族中，使学生独立完成任务、合理获取信息、处理执行反馈，并在必要时恢复。商业部署的重复任务需求是直觉来源；论文的核心应是能够解释和提升能力迁移的基础发现。

研究 ambition 是 ICML oral／best-paper 水平的问题深度，但不预设会获奖，也不要求刻意制造反直觉结果。我们应在已有成果上推进下一项能力，而不是为了避开近邻工作不断缩小问题。

| 项目 | 当前设置 | 不允许的误读 |
|---|---|---|
| 初始经验 | 给定 K 条可信、高质量、真实执行过的种子轨迹 | K 不是最多可访问的训练任务数量，也不是训练期间总轨迹数量 |
| 经验扩展 | 可以自由生成新任务、新教师输出和新轨迹，也可以新增真实环境交互 | 不得把后续所有经验称为只有 K-shot；必须披露经验来源与规模 |
| 教师 | 黑盒 GPT-6-luna，使用输出与合法工具调用 | 不假设可访问参数、隐藏状态、梯度、完整 token 分布或未提供的 logprob |
| 学生 | 用户指定 `google/gemma-4-12B-it`，官方模型与文档已核实 | 公开存在不代表账户访问、下载、微调和集群运行已验证 |
| 模型来源 | 不使用中国模型 checkpoint，包括间接依赖的教师、奖励模型和生成模型 | 可以引用、讨论并用允许的模型复现中国团队发表的方法；不能按作者姓名判断模型来源 |
| 训练 | 只在现有 checkpoint 上 post-training | 不训练新的基础模型；从零构造的小型解析环境是诊断工具，不是另起一套模型预训练 |
| 资源 | 学术集群，包含 HiPerGator B200、LONI；约四个月 | 不假设无限 GPU、独占集群或尚未确认的配额 |
| 优先级 | 先发现现有蒸馏方法的真实能力障碍，再形成方法 | 当前不把预算优化、最少教师调用、成本调度作为主贡献 |

因此，暂时使用“由少量可信经验启动的 Agent 蒸馏”这一工作定义。若文章使用 few-shot，必须在摘要和实验设置中说明：few 指可信种子，新增任务、教师调用和真实交互并未被限制为 K。最终命名应服从实验实际，避免与标准固定支持集 few-shot 学习混淆。

**已排除的旧设定：** K 个任务是全部可用训练任务；教师必须提供 logits；以教师调用预算优化为主；把 LLM 文本模拟器等同于 JEPA；沿用其他模型／项目的旧实验数字。研究目标仍然较广，实验可以先小规模启动。本文的 benchmark 选择、K 取值、适配方式和时间表都是建议默认值，不是用户已经确定的结果。

## 2. 一个核心问题，以及两个待检验机制

**核心问题：给定少量可信经验，什么样的学习经验足以让受限学生形成自主成功策略？**

教师能完成任务，不意味着模仿其成功轨迹就能让学生学会。需要调查至少两个机制：

1. **相关知识没有被有效吸收或调用。** 轨迹包含动作前提和后果，但学生在新实例、新历史或错误恢复阶段不能使用它们。
2. **学生需要更明确的信息来作决策。** 同一观察足够强教师完成推断，却不足以让学生提取关键事实、预测动作后果或保留长期依赖。解决方式可能是更好的教学，也可能是学生学会合法地重新观察、查询和验证。

这两类解释不是已经成立的结论，也不是互斥标签。学生还可能因为教师给错指导、任务覆盖不足、格式错配、优化不足、环境接口缺陷而失败。研究首先需要区分它们。

希望推进的能力是：从少量可靠起点出发，系统能够发现学生缺什么，扩展出真正有用的新任务与经验，并让学生形成自己可实现的策略。最终学生应脱离教师独立执行；允许使用的环境工具仍按部署接口开放。

## 3. 为什么这个问题重要，以及怎样避免已有故事

真实 Agent 学习同时涉及事实、策略和信息获取。比如一次操作失败后，系统是否已经改变状态；一次提交能否撤销；某项前提能否从当前观察推出，还是需要查询。教师可能依靠长上下文、复杂计算或隐含知识正确处理这些问题；学生即使模仿了动作序列，也可能不知道何时重查状态、何时改变路线。

若只是增加成功轨迹，经验扩展可能继续复制教师的解决方式，而没有补上学生的决策需要。相反，若给学生更多环境信息便能成功，应该研究这些信息是需要进入参数、需要在当前上下文中明确表达，还是必须在执行时重新获取。三者具有不同的能力迁移含义。

目前不能把“学生状态指导”“更强教师未必更适合学生”“交互减少分布偏移”“教师在学生历史上会不可靠”作为新发现。这些已有直接近邻。我们的机会在于形成一个经控制实验支持的解释：**在允许大量合成新任务和轨迹时，哪些经验仍然缺失；这些缺失如何随学生的可实现策略变化；怎样产生能使学生真正学会的经验。**

## 4. 数学对象：经验充分性与学生相关的环境歧义

令 E 为真实环境，q 为固定任务族中的任务实例，T 为教师，Π_s 为学生在合法部署接口下可实现的完整策略类。策略包括环境工具调用、获取信息、内部状态更新和最终行动。J_E(π) 是在固定评测任务分布上的期望真实回报。

D_0 是 K 条种子轨迹。D 必须包含此后实际获得的全部信息：新教师输出、新环境观察、验证标签，以及允许使用的已有材料。教师、学生及任何外部生成模型的预训练先验不能被假装不存在。理论可以将它们作为固定先验 I_0；控制实验则在检查点冻结后随机分配特定环境机制，避免该机制被提前记住。

定义与当前证据相容的环境集合 V(D; I_0)，并定义达到目标的学生策略集合：

\[
G_\varepsilon(E,\Pi_s)
=\{\pi\in\Pi_s:J_E(\pi)\geq J_E(\pi_T)-\varepsilon\}.
\]

当前证据支持的共同成功策略为：

\[
\mathcal I_\varepsilon(D,\Pi_s;I_0)
=\bigcap_{E\in\mathcal V(D;I_0)}G_\varepsilon(E,\Pi_s).
\]

我们真正想隔离的信息障碍须同时满足：每个相容环境单独都有合格学生策略，即对所有 E 都有 G_ε(E, Π_s) 非空；但这些集合的交集为空。前者排除“学生在某个环境里本来就不可能成功”的解释。若教师回报目标不可实现，也可先用明确且学生可达的目标 ρ(E) 替代教师差值目标，不能通过选择过高阈值制造不可辨识性。

这个定义表达的是信息条件，不是一个现成训练算法：

- 若交集非空，存在一种对当前不可区分环境都足够好的学生策略。实验失败仍可能来自优化、表示使用、数据组织或教师错误；不能由此声称必须增加真实信息。
- 若交集为空，则没有一个策略能对全部相容环境达到所述目标。但要推出统计下界，还需构造观测分布不可区分的环境、明确的回报间隔，以及对随机策略也成立的策略类条件。仅仅写下空交集不是完整定理。
- 一个值得检验的现象是：较强策略类中交集非空，而受限学生类中为空。教师能用复杂策略绕开某个未知事实，学生可行的替代策略却依赖它。**容量限制可能改变所需的环境信息。**

理论必须保留以下限制：

1. 参数更少不自动意味着严格嵌套的策略类。一个 12B 模型或某种 LoRA 设置训练失败，也不证明对应策略在数学上不可表示。理论采用明确的记忆、观测、计算或决策结构限制；模型大小、上下文长度等只是对应的经验代理。需要说明时域以及是否允许随机化。
2. 若学生可以在执行时合法查询后再行动，这个策略必须属于 Π_s。否则所谓“训练时必须获得新事实”的下界可能是人为禁止信息获取造成的。
3. 新教师输出可能引入预训练中已有、D_0 中没有的知识。只有在给定全部先验与输入后，生成过程与未知环境机制条件独立时，才能使用数据处理论证。
4. 信息充分不等于优化可达，也不等于有限训练一定成功。必须分别讨论存在性、可识别性和可学习性。
5. 若教师目标本身超出任何学生策略可达到的水平，G 可能为空。需同时报告该策略类的可行参考；不能把所有容量差距都解释为证据问题。
6. 这里的 V 通常不能在真实 LLM 实验中精确枚举。它适用于解析环境和受控变体；真实 benchmark 使用可测干预及其预测，不宣称完成了精确版本空间识别。

7. 对有噪声的观察与随机教师输出，V 应由预先指定的统计置信集合或似然条件定义，不能简单用“该句输出概率非零”判断相容。所有下界都依赖证据和查询接口：允许无限新增真实交互时，不能仅凭 K 个种子推出不可学习。

## 5. 最接近的研究与尚待建立的贡献

文献核查截至 2026-10-04。以下列出主张最相关的原始来源，不声称穷尽全部 SOTA，也不将不同模型、数据规模和接口上的成绩直接排序。2026 年 arXiv 条目若没有单列出版信息，按预印本处理；实施时重新核对版本和代码。

| 研究基础 | 已经做到什么 | 我们必须推进的下一步 |
|---|---|---|
| [Self-Instruct](https://aclanthology.org/2023.acl-long.754/)，[AgentGen](https://arxiv.org/abs/2408.00764)，[TaskCraft](https://arxiv.org/abs/2506.10055) | 从种子扩展任务；后两者涉及 agent 环境、任务与验证 | 生成更多任务本身不是创新；解释哪些新增经验真正解除学生的能力障碍 |
| [Scaling Web Agent Training through Automatic Data Generation and Fine-grained Evaluation](https://arxiv.org/abs/2602.12544) | 种子任务、教师生成、真实 web 执行、细粒度评价和微调组成完整链路 | 必须比较经过真实执行验证的扩展 SFT，不能只比较种子 SFT |
| [DAgger](https://proceedings.mlr.press/v15/ross11a.html)，AISTATS 2011 | 在学生访问的状态上获得专家监督 | 学生状态纠正与分布偏移已有基础；需要明确还缺少什么经验 |
| [SmartAD](https://aclanthology.org/2026.findings-acl.1349/)，Findings ACL 2026 | 利用学生 NLL 选择更适合其容量的教师轨迹并进行分段加权 | 不能把“适合学生的教学”作为未经研究的主张 |
| [DART-SD](https://arxiv.org/html/2608.18524v1) | 教师执行图、Critical Topological Breakpoint、恢复后缀和局部 masked SFT | “只教关键恢复段”已有直接近邻；需检验其仍缺失的策略判别信息 |
| [Graph-Conditioned On-Policy Agent Distillation from Off-the-Shelf Teachers](https://arxiv.org/abs/2609.37522) | 执行图与学生 hindsight 改进教师监督 | 最新强近邻；原方法需教师 logprobs，不能伪称黑盒原样复现 |
| [Harness-Zero](https://arxiv.org/abs/2609.24974) | 蒸馏 harness 提供的指导，并在部署时撤掉该指导 | “把外部辅助内化进学生”本身不足以成为新贡献 |
| [Distilling Realizable Students from Unrealizable Teachers](https://arxiv.org/abs/2505.09546)，[When Does On-Policy Interaction Help?](https://arxiv.org/abs/2607.29617) | 不可实现教师、受限学生、交互和表示条件之间的理论关系 | 新理论须说明先验、证据扩展与策略相关信息带来何种额外条件或预测 |
| [Causal Confusion in Imitation Learning](https://proceedings.neurips.cc/paper_files/paper/2019/hash/947018640bf36a2bb609d3557a285329-Abstract.html)，NeurIPS 2019 | 模仿中的因果混淆及干预动机 | “观察相关性不够，需要干预”不能单独作为新发现 |

目前尚未建立的潜在贡献是：在强的任务扩展与交互蒸馏之上，发现可重复的剩余障碍；提出能预测哪类经验有效的学生相关条件；据此产生现有方法不能稳定获得的独立行动能力。这是待验证方向，不是“已证明无人研究”。遇到近邻应更新能力目标和比较基线，保留基础问题的高度。

## 6. Baseline 与黑盒可访问性

初轮发现实验从 base student、teacher、Seed-SFT、grounded Augmented-SFT、DAgger-hard 开始。它们负责检查问题是否存在，不能被称为完整 SOTA 比较。SmartAD 和 DART-SD 应在机制结论形成之前加入，特别是在轨迹选择、纠正或恢复上声称贡献时。

| 方法 | 本项目中的实现 | 黑盒条件与定位 |
|---|---|---|
| Base student 与 teacher | 固定 prompt、观察权限、动作接口和评测协议 | 测量真实 headroom；教师不必处处强于学生 |
| Seed-SFT | 只用 K 条可信种子；训练 assistant 输出 | 朴素起点；不限制其他方法新增任务 |
| Grounded Augmented-SFT | 生成新任务，实际执行和验证，再训练正确轨迹 | 必须的强基线；同池随机筛选与质量筛选分开记录 |
| DAgger-hard | 学生 rollout，教师在学生历史上给出采样动作／纠正，聚合后 SFT | 黑盒 LLM 适配，不能把教师预测的工具结果当作真实执行 |
| SmartAD | 每任务正确教师候选池，学生 NLL 选择，加权 SFT | 原理兼容黑盒；使用允许的模型做论文级重实现 |
| DART-SD | 执行图与 CTB 定位，生成恢复后缀，只训练指定局部 token | 原文生成接口和 CE 不要求教师 logits；商业教师迁移、状态图和 AugGen 适配需披露 |
| [Black-Box On-Policy Distillation of Large Language Models](https://arxiv.org/html/2511.10643v3) | 判别器与强化学习路线的扩展比较 | 可用黑盒输出；原工作不是现成闭环 agent baseline，迁移不能冒充原设定 |
| [ReOPD](https://arxiv.org/html/2607.04763v3)、[FutureBridge-OPD](https://arxiv.org/html/2608.01953v2)、GC-OPD | 作为强近邻审计；必要时另实现硬标签变体 | 原方法依赖教师 logprobs；纯 sampled-output 主设定下不能原样运行 |
| [One-Shot OPD](https://arxiv.org/html/2609.04172v1) | 边界比较：极少提示下的 on-policy 蒸馏 | 少提示不等于少教师 token 标签；原 logit 监督与外部环境覆盖问题不同 |
| 仅环境反馈的 RL 对照 | 若方法主要收益来自新增验证反馈，加入同接口 reward-only 训练 | 区分教师知识迁移与一般交互学习；具体算法须经过收敛和稳定性检查 |
| 真值与可学习性上界 | 提供正确事实或学生可执行策略，确认相同学生能否学会 | 仅诊断；必须撤除辅助再测独立行为 |

SmartAD 的复现细节应对照论文：每任务从 10 条正确候选中选择；NLL 先按 assistant turn 内平均，再按 turn 平均；reasoning/action/final 的权重为 1/1.5/2。原 CodeAct 划分移植到其他 harness 时需记录映射。若初轮缩小候选数，称为适配而非精确复现。全部失败的候选生成尝试也计入日志。未核实官方完整代码，不承诺可直接安装运行。

DART-SD 的全文为 *Diamond-topology Aware Retrieval and Tuning for Self-Distillation of Multi-Turn Tool-Calling Agents*。公式 15 的 mask 只覆盖 CTB 之后、最终答案步骤之前新生成的 assistant response steps；保留前缀、user／tool observations 和最终答案全部为零。不能套用 assistant-all mask。AugGen 细节及图状态等价判断存在重实现工作，应登记与原文差异，避免靠隐藏状态让方法获益。

比较公平不等于所有实验都锁死同一成本。发现阶段应允许各 baseline 合理训练至稳定，检查 learning curves、数据混合和学习率，避免制造共同失败。机制归因再做同经验池、同有效 token／训练曝光、同真实转移数等相应匹配对照。增加新任务和真实反馈是合法方法能力，不能在主设定里禁止后再宣称胜出。

## 7. 实验资源与尚未完成的可运行性验证

### 7.1 模型和训练运行时

准确教师 ID `gpt-6-luna` 已有 [OpenAI 官方 API 页面](https://developers.openai.com/api/docs/models/gpt-6-luna)。账户权限、图文／工具调用、实际服务配置与当前集群网络尚未做 canary 验证。主实验只请求采样文本与动作，不依赖教师 logprobs、隐藏推理或内部表示；可见的简短解释也不自动等于真实推理过程。

[Google 官方 Gemma 4 12B 模型卡](https://huggingface.co/google/gemma-4-12B-it)与 [Gemma4 Unified Transformers 文档](https://huggingface.co/docs/transformers/model_doc/gemma4_unified)已确认该准确 checkpoint。使用它自己的 processor、chat template 和 `Gemma4UnifiedForConditionalGeneration` 架构支持，固定 `enable_thinking` 与工具格式。[官方 QLoRA 教程](https://ai.google.dev/gemma/docs/core/huggingface_text_finetune_qlora)提供 post-training 起点。接手时仍需锁权重 revision、依赖版本和许可记录。

建议默认用 HF Transformers 做 BF16 LoRA，单张 B200 或 80GB 级 GPU、4k–8k 序列、microbatch 1、gradient checkpointing 起步，再根据真实轨迹长度 profile。这是未经实测的启动配置，不是显存保证。过长轨迹不能为适配配置而悄悄截掉关键工具反馈。量化、LoRA rank 与 full tuning 都可能影响机制结论，必要时加入较强适配控制。

[vLLM 支持表](https://docs.vllm.ai/en/latest/models/supported_models/)列有对应架构，但不能据此保证当前版本动态加载 LoRA。先合并 adapter 后 rollout，或保持 HF 推理；通过 save/reload/merge 一致性检查后再扩展吞吐。

### 7.2 Benchmark 和状态重放

| 资源 | 建议用途 | 必须保留的边界 |
|---|---|---|
| [ALFWorld](https://github.com/alfworld/alfworld) | 首个闭环语言环境：探索、动作前提和恢复 | 导出当前版本数据 manifest；不要沿用其他项目 task/trial 数。剔除 `extra.expert_plan`、PDDL facts、`policy_commands` 等 oracle；`admissible_commands` 对所有方法一致 |
| [AppWorld](https://github.com/StonyBrookNLP/appworld) | 第二个工具工作流环境：跨 app 状态和副作用 | train 教学，dev 调参与逐项诊断，test_normal/test_challenge 只做封存聚合评估；按 scenario 关联分组 |
| [ScienceWorld](https://github.com/allenai/ScienceWorld) | 如果核心机制是测量、隐藏性质和信息获取，可作为第二主环境 | 固定代码、task 与 variation；核验 reset 和 UUID／对象顺序可复现 |
| [BFCL V4](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard) 多轮 | 工具状态与参数／缺失信息的辅助验证 | 未核实通用官方训练划分；用独立生成训练实例或清楚命名自定义研究 split。官方成功兼含 state 与 response-path 检查，不能只看终态 |

建议先跑 ALFWorld，再根据 pilot 机制在 AppWorld 和 ScienceWorld 中选择一个主验证环境；BFCL 多轮作为接口能力补充，不要求首轮同时跑四个完整基准。初始 K 默认可选 10，随后 K∈{5,10,20}，至少规划 3 组 support sampling seeds。均为待 pilot 修订的默认方案。K 始终数可信轨迹；同时报告唯一任务实例数、模板／scenario 数，不能把同一实例的多个轨迹当成多个独立任务。

ALFWorld 可通过 reset＋动作前缀重放建立分支起点，但要验证环境状态和可见历史一致；没有 native clone 不等于无法研究分支。AppWorld 的数据库 save/load 不自动恢复 Python locals、解释器、时钟和 RNG，应使用新 kernel＋前缀重放或经验证的完整快照。所有分支能力仅用于 train/dev 研究，不赋予最终 agent 任意真实世界回滚权限。

### 7.3 计算量估算

当前没有该模型对和 harness 的实测 GPU-hours，不能给出可信总量承诺。先测短训练的秒／update、有效 tokens/s、峰值显存，以及 rollout／评测吞吐；再按实际实验矩阵外推：

```text
GPU_hours = Σ_jobs(number_of_GPUs × measured_wall_seconds / 3600)
estimated_training_hours = updates × measured_seconds_per_update × GPUs / 3600
estimated_rollout_hours = episodes × measured_GPU_seconds_per_episode / 3600
```

包含 adapter 合并、推理、验证、分支和失败重试；排队时间、CPU 环境时间与教师 API 调用分别报告。训练与推理重叠执行时不要重复计算同一 GPU 占用。4 个月是日历计划，不等于 4 个月独占集群；配额和并行数以实际可用情况为准。

## 8. 发现实验：先区分原因，再发明方法

### 8.1 先证明基础链路可以学会

在少量公开训练任务上检查：学生 base 能执行；教师明显有可学习的行为；高质量轨迹 SFT 能降低验证损失，且不会因格式或训练错误破坏行为。若不能，优先排查接口、截断、标签、chat template、动作解析、terminal handling、过拟合与学习率。不能立即归因于知识或环境信息缺口。

最小技术验收包括一个完整 assistant→tool→tool result→assistant 回合、一次 backward/update、保存重载、adapter 合并前后的固定输入一致性，以及 10–20 个固定开发实例的端到端 smoke episodes。普通 SFT 的 user／tool observation token 不参与 loss；DART-SD 使用其更严格的局部 mask。统一 thinking 开关，区分 parser error、服务错误、环境错误和真正策略失败。

另做“学生能学会”的正对照：给同一个学生一条已知正确、在其合法接口下可执行的简化策略，充分训练后撤除策略提示，再测新实例。若该控制也失败，先调查适配容量、优化和策略复杂度；不能直接归因于经验的信息缺口。提示中临时提供策略后成功，只证明辅助有效，不证明能力已进入参数。

### 8.2 固定任务族内的经验扩展控制

建立共同候选池，至少区分四种来源：

| 条件 | 新任务覆盖 | 新的真实环境信息 | 要回答的问题 |
|---|---:|---:|---|
| 原任务扩写与重采样 | 低 | 无新增机制事实 | 是否只是曝光与优化不足 |
| 大量新任务和教师轨迹 | 高 | 受当前可得事实限制 | 覆盖扩展本身能否解决问题 |
| 新任务＋相关机制的可信反馈 | 高 | 有 | 是否存在真正的环境事实缺口 |
| 新任务＋等量无关反馈 | 高 | 有但无关 | 收益是否只来自更多文本、步骤或监督 |

不得虚构“没有新增信息”的组：如果教师调用真实工具、使用检索、观察新环境结果，产生的信息必须入账。生成任务文本与执行该任务是两个事件，分别记录。

### 8.3 知识吸收与信息表达的交叉干预

对固定学生与相同任务，交叉比较“原始／针对性 post-training”与“原始观察／明确表达已有决策相关信息”。明确表达只能使用动作前合法可得的信息。再测试去掉辅助表达后能否独立成功，以及允许重新查询时能否恢复。

知识探针检查事实判断、动作前提、可选动作后果与失败后状态；行为测试检查实际闭环执行。探针只用于机制分析，不以“会回答知识问答”替代 Agent 能力。

在 train/dev 中重放到同一决策历史，实际执行不同候选动作，并固定后续 continuation policy，才能尽量隔离当前动作的作用。设置等长度无关事实、已有事实重述，以及相关的新执行反馈，分开信息增量和表达变化。所有“相关事实”必须有执行证据；教师关于动作后果的预测单列为未经验证标签，并与环境验证标签对照。

### 8.4 容量改变知识需求的受控环境对

构造至少一组满足以下结构的机制变体：教师可采用共同成功策略；受限学生可行的替代策略依赖某个机制事实；种子经验对该事实没有判别力。对照一个机制已在种子中可辨认的版本，以及一个学生无需知道该事实也可成功的版本。

应先从自然 benchmark 的真实失败中找到机制，再用受控变体验证因果解释。若只有人为设计的世界支持假设、真实任务没有对应现象，不足以支撑广泛的 Agent 蒸馏主张。

这只是设计目标，不是已完成 benchmark。所有环境对需要证据检查：种子信息是否真相同、教师能否从命名或生成文本猜出机制、学生能否通过合法查询绕开歧义、策略类是否真的受到指定限制。只要任何一点不满足，就不能把结果解释为信息论障碍。

### 8.5 核心假设与否证条件

| 待检验假设 | 强证据 | 会否证或削弱它的结果 |
|---|---|---|
| 更多生成经验仍可能缺少关键判别信息 | 扩展覆盖后行为平台期；相关可信事实解除平台期，无关事实无效 | 调好标准 SFT／OPD 或增加同信息经验就解决问题 |
| 学生能力改变必要信息 | 同机制的事实干预，对不同受控学生限制产生可预测差异 | 效应只随总体任务难度变化，不能由策略需求解释 |
| 更强教师轨迹可能更少包含学生必要信息 | 控制任务与成功率后，复杂教师路线绕开某事实，显式查询路线提供该事实 | 差异被轨迹质量、长度、教师错误或输出格式解释 |
| 教学可使必要信息内化或使学生主动获取 | 辅助信息撤除后仍成功，或学生学会在合法条件下自行查询 | 方法仅在持续喂入隐藏真值时有效 |

否证不是项目失败。它决定下一步应研究优化、任务覆盖或策略构造。禁止将未得到的反直觉现象写进论文标题和摘要。

理论目标也必须经发现实验约束：在一个明确的受限设置中，争取同时给出必要信息的下界和可达到的构造／算法上界，并导出能够预测未参与理论设定选择的实验条件的结论。空交集定义、一个不可辨识反例或泛泛的 no-free-lunch 论证，本身都不足以构成新理论贡献。本文没有声称这些界已经证明，也没有声称该方向无人研究。

## 9. 方法应由发现实验推出

暂不承诺一个多模块系统。可执行的研究路线是：从可信轨迹识别任务结构；收集学生真实失败；提出能区分失败原因的教学干预；生成新任务、解释或对照行动；在真实环境验证经验；post-train 学生；在未用于生成的实例和状态上评估。

可能成立的方法方向包括：

- 若知识已经在经验中但未被调用，学习把前提、状态变化和后果与决策绑定，并通过变化表述及闭环迁移检验，而非只重复动作 token。
- 若学生依赖不同的信息粒度，让教师提供学生能够执行的路线、显式检查点或合法查询策略，再逐步撤除训练辅助。
- 若缺少判别事实，通过可验证的新交互产生对照经验，使学生区分“动作执行了什么”与“我希望它执行什么”。

这些是探索分支，不是三项必须拼装的方法部件。最终方法应对应最有力、可重复的机制；同时应保留通向更广泛自主能力的路径。不能为了快速显著增益把任务压缩成一个手工规则有效的特例。

## 10. 数据划分、真值和来源追踪

数据至少分成种子集、可扩展训练池、开发集、封存测试集。任务家族可以相同，实例、布局、实体、机制分配或随机初始化按研究声明独立划分。测试不要求跨任意新领域，但必须检验同任务族内真实泛化。

还要区分两种合法协议：仅 K 个可信种子加自由生成；以及 K 个可信种子加外部未标注训练任务池。后者同样可以研究，但外部任务池及工具文档属于额外可用资源，不能隐藏。生成的新任务必须对应可执行初态、可达目标与合法工具 schema，不能仅凭教师输出一句新 instruction 就视为新环境实例。先用发现集建立机制，再在独立开发实例确认；不因观察到某个效果反复改封存测试。

1. 测试任务和测试状态不能用于教师生成、失败归因、提示调整或 checkpoint 选择。
2. 所有合成任务保存 parent seed、生成 prompt、teacher model/version、时间、随机 seed、原始输出与过滤理由。
3. “可信”必须有具体定义：在固定 harness 中可执行，终止条件与关键中间事实经验证；教师自评成功不能单独充当真值。
4. 新环境反馈保存环境版本、状态快照引用、动作、可见观察和隐藏验证标签。隐藏标签与模型可见信息存放在不同字段；训练加载器只读取显式白名单。
5. 失败与被拒绝样本也保存，避免只统计成功轨迹掩盖生成困难。
6. 相同轨迹的重试、不同表述和重复训练曝光不能算成新的独立任务。
7. 所有基线共享等价的格式、权限和终止语义；若研究信息干预，只有被干预的字段可以改变。

## 11. 指标与统计解释

主指标是真实环境中的独立任务成功率或预先定义回报。配套报告首次成功、从可恢复失败状态继续执行的成功率、非法动作与无效循环、需要查询时的查询行为，以及不需要查询时能否正常完成。

机制指标包括知识探针正确率、同一状态不同动作的后果区分、对机制交换的响应、辅助信息撤除后的保持、训练 loss 与真实执行的关系。不得只凭注意力图或相关性宣称因果机制。

用相同测试实例和环境 seed 比较策略；保存逐实例配对结果。报告置信区间及多训练 seed 的变化。是否使用 McNemar、paired bootstrap 或分层 bootstrap 取决于独立单位；多个 trial 属于同一任务时不能假装独立。测试 seed 数及最小有意义效应由 pilot 方差和资源决定，正式实验前冻结。

新增任务数、可信转移数、教师调用／tokens、训练曝光、GPU-hours、wall time 全部记录。它们用于排除混杂和描述实验，不应取代能力发现成为文章主故事。

## 12. 实现契约与最小配置

建议目录：`configs/`、`harness/`、`teachers/`、`data/manifest/`、`data/raw/`、`data/verified/`、`train/`、`eval/`、`analysis/`、`reports/`。用户数据与权重路径使用配置，不硬编码个人账号或集群路径。

```yaml
project: sufficient_experience_agent_distillation
status: discovery_not_run
models:
  teacher:
    requested_name: gpt-6-luna
    provider_model_id: gpt-6-luna
    account_canary_passed: false
    access: black_box_outputs_only
  student:
    requested_repo: google/gemma-4-12B-it
    revision: null          # 下载并核验后锁定
  checkpoint_origin_policy: no_chinese_checkpoints
training:
  mode: post_training_only
  adaptation: bf16_lora_proposed_pending_profile
  thinking_mode: fixed_before_comparison
setting:
  task_family: fixed_workflow_family
  seed_trajectory_count: 10   # 建议初轮默认值，非用户固定要求
  support_sampling_seeds: [0, 1, 2]
  new_tasks_allowed: true
  new_teacher_outputs_allowed: true
  new_real_interactions_allowed: true
  budget_optimization_is_main_objective: false
environment:
  name: alfworld
  version: null
  action_format: null
  legal_information_queries: []
  train_mechanism_split: null
evaluation:
  sealed_manifest: null
  privileged_fields_allowed: false
  checkpoint_selection_split: development_only
```

接口契约：

- `Teacher.generate(history, task, visible_context)` 返回原始文本、解析动作和调用元数据，不假定 logits。
- `Environment.step(action)` 返回模型可见观察、奖励、terminated、truncated 和审计信息；不可把隐藏状态默认拼进 prompt。
- `Environment.snapshot/restore` 仅在环境真实支持并经测试时提供；不支持 native snapshot 时使用经过校验的 reset＋前缀重放；仍不能恢复时明确报错，不伪造反事实分支。
- `Experience.verify` 输出可执行性、成功判断、事实来源、失败原因；环境规则无法判断时使用 unknown，而非硬标正确。
- `Dataset.build` 输出每条 token/action 标签的来源，并检查 role 边界、截断和 loss mask。
- `Evaluator.run` 禁止写入训练池或触发在线更新，完整记录动作与返回值。

建议逐事件 JSONL：

```json
{
  "run_id": "...", "episode_id": "...", "event_id": "...",
  "split": "train", "task_family": "...", "task_id": "...",
  "parent_seed_ids": ["..."], "mechanism_id": "audit_only",
  "student_revision": "...", "teacher_provider_id": "...",
  "environment_revision": "...", "policy_condition": "...",
  "history_ref": "...", "visible_context_ref": "...",
  "action_raw": "...", "action_parsed": "...",
  "observation_ref": "...", "verification_ref": "...",
  "feedback_source": "real_environment",
  "new_environment_information": "unknown_until_audit",
  "teacher_tokens_in": 0, "teacher_tokens_out": 0,
  "environment_steps": 1, "latency_seconds": 0.0,
  "terminated": false, "truncated": false,
  "status": "ok", "error_type": null
}
```

数据处理必须区分真实反馈、教师推断、人工／解析规则验证与 WM 合成。缓存键包含模型 revision、prompt、任务、历史和采样设置；不能把不同干预条件误命中到同一缓存。

## 13. 四个月推进路线

| 阶段 | 交付内容 | 进入下一阶段前需要的证据 |
|---|---|---|
| 第 1–2 周 | 模型与接口审计、harness、manifest、base／teacher／seed SFT、资源 profiling | 真实闭环可运行；数据无泄漏；基本蒸馏链路有效或故障被解释 |
| 第 3–5 周 | 强扩展 SFT、黑盒互动基线、机制探针、首个受控环境对 | 找到可重复的剩余失败，并排除至少主要格式、优化、覆盖混杂 |
| 第 6–9 周 | 交叉干预、策略可行性分析、方法原型与理论反例／条件 | 方法解决被诊断的机制；不能仅依赖持续注入隐藏真值 |
| 第 10–13 周 | 第二环境验证、多个训练 seed、辅助撤除、机制交换、最强 baseline 复核 | 结论跨实例与相关条件稳定，主张与证据范围一致 |
| 第 14–16 周 | 封存评测、理论完善、复现包、论文写作与文献更新 | 所有主表可由 manifest 重建；无未标记诊断上界或未披露信息 |

阶段安排不是硬性停止门。若发现结果已扎实，应继续向更自主的信息获取、更可迁移的经验组织或更复杂工作流推进，而不是仅优化叙述。若核心假设被否证，应更新科学问题与路线，并保留负结果和完整原因。

## 14. 与论文二的接口，以及接手者第一批动作

论文一应输出三类可复用材料：学生需要的经验类型、经过真实环境验证的干预／分支、以及能预测学习效果的机制指标。论文二研究预训练 WM 能否生成具有同等学习价值的经验。论文一不需要先造 WM，也不应把文本工具模拟器称为 JEPA 或视觉世界模型。

接手后按顺序执行：

1. 读取现有 repository 的 AGENTS.md、环境与作业配置；核对是否已有真实进度。本文不能覆盖之后产生的实验记录。
2. 验证指定教师 API 与学生 checkpoint；记录可访问性、来源、许可、模板和版本。不可静默替换模型。
3. 运行一个完整环境 episode；验证动作解析、查询权限、重置、失败副作用和 terminal 语义。
4. 导出数据 manifest 并冻结开发／测试边界；明确 K 是种子轨迹数。
5. 建立统一 base、teacher、seed SFT 和扩展 SFT 入口，做小规模训练与评测 smoke test。
6. 记录资源 profiling；再安排集群实验，不凭空给出 GPU-hours 总量。
7. 阅读并核对最近邻方法的真实信息接口，区分忠实复现与黑盒改编。
8. 从失败样本选择首个可干预机制，先写出替代解释与否证条件，再生成额外经验。
9. 向研究负责人汇报可运行栈、已观察事实、剩余不确定性和下一组具有区分力的实验。

首批交付物应包括 `environment_manifest`、`data_manifest`、`model_provenance`、一份完整可重放的 episode、可检查的训练样本与 loss mask、base／teacher／SFT 的逐实例结果，以及 `profiling_report`。这些是建议的产物名称，本文不表示代码或结果已经存在。账户密钥通过安全环境配置，不写入报告或聊天；API alias 若没有固定 snapshot，记录调用时间与返回的模型标识，并在结果中披露可能漂移。

当前最需要产出的不是一个提前命名的复杂算法，而是一组能说明“学生为什么没有学会，以及什么经验让它真正学会”的可靠发现。在这项发现之上，我们再构造能够推进自主能力的方法。
