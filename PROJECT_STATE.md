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

## Infra(2026-08-09 晚设置)
- GitHub 私有 repo:https://github.com/XueqiC/tc-alignment(gh CLI 在
  ~/hq/tools/gh,配置在 ~/hq/tools/gh-config,repo 局部 credential.helper 已接好;
  push 用 `cd projects/tc-alignment && git push`)
- hpg blue 路径实际为 `/blue/yd24f.fsu/xc25.fsu/hq`(CLAUDE.md 与 sync 脚本已修正)
- hpg 工具:uv 装在 /blue/.../hq/tools(UV_CACHE_DIR 也在 /blue,别用 home)
- 组 blue 存储 8T 已用 93%(剩 ~640G),大数据集落盘前先看容量
- 用户名下 7 个死 pending job(DependencyNeverSatisfied)已按其明确授权 scancel
- rai:runyang 占 4 张卡(vLLM);用户想让出——已告知需其本人处理,我们不动
- hpg-b200 分区快照(08-09 晚):464 GPU,空闲仅 ~12;两账户队列基本无我方任务

## Running jobs
- 定时:cron 219d967b 每 30min(:13/:43)向 Discord 报进度(用户要求)
-(rai GPU 提示:CUDA_DEVICE_ORDER=PCI_BUS_ID 必须设,否则 CUDA 按算力
  排序会落到别人的卡上)

## Pilot(exp_id: pilot-boundary-v1)
- 数据:data/pilot/{gsm8k-code,gsm8k-cot,pandas,sql,alpaca}.jsonl 各 120 条;
  gsm8k-code 由 CoT 的 <<a op b=c>> 注释机械转 Python(变量替换+执行验证),
  与 gsm8k-cot 共享完全相同的 prompt(困难对,query-only 特征按构造不可分)
- 特征:Qwen2.5-1.5B-Instruct + LoRA r8,lora_B 梯度(init 时 = 全梯度的 A^T
  随机投影,LESS 技巧),每模块再 JL 到 64 维,196 模块拼接 → 12544 维,L2 归一
- 对比:同一子空间+conformal 流程下 grad vs BGE emb(query+traj)vs emb(query)
- 预注册标准见 docs/2026-08-09-pilot-design.md;结果 → results/pilot/summary.md

## HPG readiness(2026-08-09 验证)
- venv:/blue/.../hq/tc-alignment/.venv,torch 2.11.0+cu128 + numpy,python 3.12(uv)
- 冒烟测试 job 39036449 COMPLETED:B200 178GB sm_100 正常,SMOKE_OK
- 模板:scripts/smoke_hpg.slurm(10min 1×B200)可复制改成真实验 job

## Key results so far
- **pilot-boundary-v1(2026-08-09)**:梯度谱子空间边界估计强烈成立。
  困难对(gsm8k-code 边界 vs gsm8k-cot 探针,题面相同)hard-AUROC:
  grad **1.000** / emb-traj 0.687 / emb-query 0.404(≈按构造的随机);
  跨域 grad 全 1.0、误纳 0%、conformal 覆盖 93-97%(名义 90%);
  k=5 即达 1.0,emb-traj 随 k 增大反而降(0.80→0.60)。
  图:results/figs/pilot_{pca,roc_hard,k_curve}.png;表:results/pilot/summary.md
- **pilot-controls-v1**:两个替代解释均排除——768 同维投影 hard-AUROC 0.999;
  模板剥离(去 def solution()/return,裸赋值)后 1.000、拒绝率 100%。
  已知小问题:splits 用了内建 hash(d)(进程间加盐)→ 跑批间 split 不稳定,
  正式实验前改成稳定 hash。剩余 caveat:域间本身较易分;下一轮要加
  域内细分(如 pandas 内 filtering vs groupby)与参考模型稳健性检验。
- **e1-intra-v1(2026-08-10 凌晨)**:域内细分是当前表示的分辨率极限——
  grad AUROC 仅 0.62-0.75,emb-traj 在 3/4 对上更高(但细分标签按关键词
  定义,天然偏袒 surface 方法,测试本身有循环性)。方法启示:粗边界 grad
  压倒性成立;细边界需 (a) per-step 梯度 stacking,(b) 行为定义的细分标签,
  (c) 子空间能量阈值调优。诚实纳入论文,不隐藏。
- **pilot-distill-v1(2026-08-09)**:C 边界在同一梯度空间可度量,且蒸馏自然塑形。
  90 条 gsm8k-code 伪轨迹 SFT(LoRA r16, 33 步)后:
  行动空间完全切换(format 0%→100%,gen 不再出 CoT);exec_acc 0%→37%,
  但 CoT-any 67%→37% —— **行为切换远快于能力迁移**(capability vs behavior 直接证据);
  loss:in-T 1.67→0.17,out-T 全部变差(cot 0.64→1.11,alpaca 1.50→2.22);
  残差梯度中位数:in-T 5.1→2.6,out-T 全部上升 —— 蒸馏把学生"削尖"进 T,
  域外自发退化(对 C⊆T 方向是利好信号)。图:results/figs/pilot_distill.png
- (草稿前期工作:JSON/code 1.40× token 比等,工件不在本仓库)

## PI 拍板(2026-08-09 深夜,均已吸收进 docs/2026-08-09-progress-recap-zh.md)
- 环境面板:**AppWorld(主)+ BFCL v4 + τ²-bench 域切片**(SOTA 对齐)
- teacher:ollama cloud **deepseek-v4-pro + kimi-k2.7-code**(bulk),
  备选 qwen3.5:397b(同族对照);gpt-5.4-mini(Azure)只做 probe
  (3 key × 100k tok/周 = 300k/周,不够 bulk);兜底 = hpg 自托管 teacher
- 学生线:**Qwen3.5 0.8/2/4/9B** + 1 个跨家族对照点
- 模型刷新(2026-08-10 用户要求最新款):teacher = **Kimi K3**(7/27 开源)
  + DeepSeek-V4-Pro + k2.7-code(对照);27B 参考模型换 **Qwen3.6-27B**
- **base vs instruct(已确认 2026-08-10)**:双轨——主线 Qwen3.5-*-Base*
  (能力归因干净,R1-Distill 谱系做法),instruct 平行副线 1-2 尺寸
  (先验挤占现象 + 部署现实);pilot 已有数据归入 instruct 副线
- 实验代码默认委派 codex(gpt-5.6-sol,reasoning xhigh,ops/codex_task.sh)
- hpg 算力:用户明确要求"可能的话多申请几个 B200 把实验做好"(2026-08-10)——
  E3/E4 sweep 可放开到多卡多任务(仍守 ≤4 pending 默认;sweep 时经用户点头放宽)
- API 凭证在 ~/hq/secrets/llm_apis.env(600,不进 git;已提醒用户轮换)

## 完整实验记录
docs/2026-08-09-pilot-report-zh.md(三实验全记录+图+证据链+insight+局限+
W1-W6 执行计划,PI 已确认精彩)——新会话想快速恢复上下文先读它和本文件。

## Next steps(W1,2026-08-10 开工)
1. codex:框架加固(stable-hash split、verifier 沙箱、feature 缓存)
2. codex:AppWorld + BFCL v4 接入骨架;我设计 topic 划分
3. E1 扩展:域内细分困难对 + 参考模型稳健性(rai)
4. E2 gate:残差梯度 vs 执行成功率相关性(rai)
5. teacher 轨迹池 v0:双 teacher 各 ~1k 条(限速分批)
- 时间线:ICLR 2027 投稿 ~9 月底,6 周计划见 method-design-v1 §8
- 旧候选(动机实证 / IR renderer / length control)已被新问题表述取代或吸收
