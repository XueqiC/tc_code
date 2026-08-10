# Novelty 定位:25–26 年最近邻文献扫描

日期:2026-08-09。方法草图见 2026-08-09-tc-boundary-gradient-spectral-sketch.md。

## 五个相邻文献簇与我们的差异

### 1. Agent distillation(最拥挤,必须正面区分)
- **Distilling LLM Agent into Small Models with Retrieval and Code Tools**
  (NeurIPS 2025, arXiv:2505.17612)— 把完整任务求解行为(含 code 工具)蒸给小模型;
  first-thought prefix + self-consistent action generation。
- **SmartAD: Capacity-Aligned Agent Distillation**(ACL 2026 Findings)—
  已读摘要确认:"capacity-aligned" = 按学生 NLL 选学生学得动的 teacher 轨迹
  + 分段加权 loss(action/decision span 权重高)。是"迁就学生容量",
  **不是**任务边界对齐。
- Structured Agent Distillation(arXiv:2505.13820)、SOD 逐步 on-policy 蒸馏
  (arXiv:2605.07725)、curriculum turn-level on-policy(arXiv:2606.15912)。

**共同点**:目标全是"学生在给定 benchmark 上分数最大化"。
**没人做的**:(a) 从 few-shot query 估计任务域 T 并给出带保证的边界;
(b) 双侧目标——域内覆盖 + 域外抑制(capability shaping);
(c) 把"学生能力边界"本身作为被控制、被度量的对象。
→ 我们的定位:把 distillation 从 performance maximization 改写成
**boundary-constrained specification satisfaction**。

### 2. 梯度特征做数据选择 / 梯度子空间
- LESS 谱系(GrADS 2025;gradient-orthogonality domain adaptation,
  arXiv:2602.06359;dual scoring, arXiv:2605.06166)。
- 子空间用于**效率**:GaLore、SubTrack++(arXiv:2502.01586)、PESO
  (arXiv:2512.02216)、GEMS 多子空间 tuning(arXiv:2601.09496)。
- Spectral Lens(arXiv:2605.05683):梯度/激活谱作为优化诊断。

**差异**:现有工作把梯度子空间当作压内存的工具或"对固定 val 集有用"的选数据
准则;没人把**谱子空间当作任务的 specification**——一个从 few-shot query 学出、
带 conformal 成员保证、并且训练侧和评测侧共用的任务域表示。角色转换是新的。

### 3. OOD 检测 / conformal 拒绝
- 梯度基 OOD:GradNorm、GradOrth、GAIA(见 ACM CSUR 2025 survey)。
- conformal:Polysemantic Dropout(EMNLP 2025, arXiv:2509.04655)、
  **SCOPE: Sequential Conformal Probing for OOD Rejection in LLM Services**
  (arXiv:2606.21255)— 最接近我们"边界判别器"这半边,但它是 serving 时
  在激活空间做过滤,与训练/蒸馏无关,也不度量模型能力。
- EigenTrack(arXiv:2509.15735):激活谱特征做 OOD/幻觉追踪。

**差异**:OOD 工作只回答"这个输入在不在分布内";我们要求边界与**能力**同空间
——同一个 U_T 既判 query 也量 student 学没学会,且反向驱动训练。

### 4. Learnware / capability specification(思想上最亲缘)
- **Learnware of Language Models**(arXiv:2505.13425):~100 个专才 8B SLM
  + 每个模型带 capability specification,用户按 spec 挑模型(隐私保护匹配)。

**差异**:learnware 是从现有模型池**检索**匹配任务的模型;我们是给定任务
**合成**恰好匹配的模型。可以把我们定位成 learnware 的"生成式对偶":
spec 不是事后描述,而是蒸馏的目标约束。这是很好的 related-work 支点。

### 5. Unlearning / capability control(C⊆T 方向的工具箱)
- SSPU:SAE 子空间引导投影 unlearning(arXiv:2505.24428);
  geometric unlearning(arXiv:2511.17100, 2605.01735);G-effect 梯度视角
  (unlearning objectives);WMDP 一系。

**差异**:unlearning 是从大模型里**删除**指定能力;我们是在蒸馏时**从头塑形**。
但域外抑制(C⊆T)可直接借它们的投影/正则工具;"capability vs behavior"
的坦白讨论也可引它们的评测协议(jailbreak 鲁棒性等)。

## Novelty 主张(当前判断)
新对象 + 新组合,单点技术均有前身:
1. **问题表述本身可能是主要贡献**:distill-to-specification —— 给定 few-shot
   query 定义的任务域,产出能力边界恰好匹配的小 agent(双侧:覆盖+抑制)。
   在 agent distillation 文献里没有先例。
2. **T 与 C 同空间可度量**:梯度谱子空间同时作为任务 spec、边界判别器
   (conformal 保证)、能力度量(残差梯度)、训练目标(投影选数据/投影更新)
   ——四位一体是新的。
3. 风险:agent distillation 赛道拥挤,审稿人第一反应会是"又一篇蒸馏"。
   写作上必须把重心放在问题表述 + 可度量性 + 保证,方法数字只是支撑。
   另需在投稿前复查 SCOPE、SmartAD 的后续引用链,确认没有更近的新工作。

## 待深挖(下一轮)
- SmartAD 全文细读(已存 PDF 文本提取件);Learnware 全文(spec 具体构造)。
- "capability elicitation / sandbagging / evaluation-aware" 文献(安全侧对
  capability boundary 的定义方式)是否有可借用的形式化。
- ICLR 2026 有无 task-vector-as-boundary 类工作(proceedings 尚在放出)。
