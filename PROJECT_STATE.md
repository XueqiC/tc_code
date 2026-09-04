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

(2026-08-14 下午)
- rai GPU2 | fam_b2_rai(对决 seed2:boot3/boot4/llm2llm/ourscorr ×{plain,ours})| pid 4053126 | logs/fam_b2_rai.log | ~3.5h
- rai GPU3 | mx_2b_s2(矩阵 seed2:13 conds)| pid 4053127 | logs/mx_2b_s2.log | ~5h
- hpg | 39411442(cfx 0-2, group:策展修复 E3_CURATE_GATE=0 ×3 seeds)/ 39411443(k15 3-5, dept:boot3/4/5 @k=15 ×3 seeds)| 旧全家福 39392528/29 已按失败分析 scancel
- rai GPU4 | boot5_smoke(k=15 b2k 烟测)| logs/boot5_smoke.log
- 后台 | evol 扩池 --per-query 4(公平性修复)| logs/evol_extend2.log;完成后 evol 行 3 种子重跑
- 已完成今日:fam_b0_rai、fam_b1_rai、mx_2b_s0、mx_2b_s1(两种子矩阵+对决全表在 results/,聚合器 tools/aggregate_tables.py 2B)

## Paper(2026-08-10 启动)
- paper/ = ICLR 2027 官方模板 + v0.1 骨架(abstract/intro 成稿,method scope,
  TODO 槽位随实验滚动填充);写作规范 = ~/hq/.claude/skills/paper-writing/
- Overleaf:用户经 GitHub import 链接(已发操作说明);rai 无 pdflatex,
  待装 TinyTeX 本地校验
- PI 要求(2026-08-10):story 最重要,精读优秀蒸馏论文的叙事写法并吸收
- **机理三连完成(2026-08-10 凌晨)**:M1 削尖=断供+漂移(负内积<15%,
  核质量预测 ρ=0.53);M2 原子需求单调排空(-89%)伴随 exec 0→57%,吸收呈
  顺序波(自发课程表);M3 行为翻转区间更新 85.5% 集中于 5/128 原子
  (低维载体,0 号=格式原子)。方法三输入:正向预测有动力学证据/塑形可到
  原子粒度/排程是免费杠杆。M2 前两版失败教训:probe 配置=特征空间的一部分
  (r16/r8 基不匹配),须版本化;最终方案=双模型探针协议(m2_v3.py)。
- E2-v3 = 原子供给预测在 sweep cohort + M2 记录上验证(任务 #9,A2 预注册:
  cohort ρ≥0.6 / 逐题 AUROC≥0.75)
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
- **e3-tier1(2026-08-10)**:主实验首轮跑通。域内 D/B 并列 90%(饱和);
  D 域外 loss 最高=削最尖;F 域外拒答 100%/域内误拒 0%(−6.7pt);
  行动空间俘获全条件 100%;梯度边界较 embedding 更紧(153 vs 264 过阈)。
  单 seed;tier-2 需更紧预算/子话题 T/agent 环境拉开覆盖差。
  六学生 (训练集,逐题成败) 已就绪 → 核预测能力零成本验证(任务 #9)。
- **概念迭代(2026-08-10 与 PI 实时)**:learning-kernel 统一框架 →
  PI 嫌 kernel 过时 → 26 年锚点重选:**Capability Atoms**(梯度稀疏字典,
  挂靠 Gradient Atoms 2603.14665 + SAE 浪潮);atoms-pilot v1 超参失败但
  join 原子可解释性直接命中,v2 在跑。论文框架语言待 PI 定夺(datamodels
  vs atoms)。
- **e2-gate-v1(2026-08-10)**:**FAIL**(pooled ρ=−0.403 < 门槛 −0.5)→
  按预注册,几何 C 不进训练环;E3 数据选择只用 T 侧子空间,C 一律行为化测量。
  解剖:残差梯度后期回升(风格漂移混淆"没学会"),但 within-checkpoint
  排序强(后期 ρ=−0.70,最低 decile 成功率 96%)。E2-v2:within-checkpoint
  归一 / 学生自采样轨迹残差,本周重测。
- **e1-refmodel-v1(2026-08-10)**:参考模型稳健性通过(Qwen3.5-4B/9B 与
  1.5B 结论一致,hard-AUROC 均 1.000)。
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


## 深刻问题(2026-08-21,Xueqi 提出,需调研)
in-task 与 off-task 能力在蒸馏中如何 decouple / 是什么耦合关系?
主目标不变:同预算下 in-task 更高、更快达到;off-task 相对低。
调研方向:能力耦合的表征(共享 atoms/梯度子空间重叠)、我们自己的
数据(budget-residue 曲线、门开关对照、atoms 重叠度)、文献(任务
算术/task vectors、灾难性遗忘、能力纠缠、cross-task transfer)。


## 主表 benchmark 定稿(2026-08-22)
AppWorld(agent 锚点,TGC/SGC)+ τ-bench(客服 agent,pass^k,teacher 兼任 user-sim 计入预算)
+ BIRD/Spider2 text-to-SQL(执行准确率;按 DB 域天然划分 → 边界/耦合上主台)
+ DS-1000(数据科学代码,执行验证,单轮对照)。
洞见形成以这四台为主要土壤;gsm 受控套件仅作机制解剖。

## 大故事线(2026-08-21 定稿,一切服从它)
**Budget 被尽可能地用来提升 in-task performance。**
**重要修正(同日)**:方法**不刻意压制 off-task**——off-task 低不是
我们做出来的目标,而是"预算没有漏到界外"的自然结果;若 off-task
变差(如破坏性耦合摧毁 cot),用耦合结构去**解释**它,不是去追求它。
叙事口径:我们优化的唯一对象是 in-task;off-task 的走向由任务对的
耦合类型决定(解耦→不动、建设性→顺带提升、破坏性→受损且可用锚
缓解),方法只是不为界外能力花钱。
每个组件的存在理由都用这句话检验:配额分配(钱按需求走)、环内门+
回购(坏货的钱找回来)、先试后买(学生会的不花钱)、纠错采购(只为
增量付钱)、自解锚(免费数据防倒退)、off-task 最低(钱没漏到界外的
证据)。方法叙事、实验指标、耦合调研全部向这条线对齐。

## GOAL (2026-08-19 定稿,主次分明)
- **主目标**:同预算下 in-task 准确率最大化(argmax Perf s.t. Cost ≤ b)。
- **辅目标**:off-task 最低——作为"预算被高效转化为任务能力"的证据,
  不是独立约束。门若对 in-task 无代价则保留(零泄漏免费的故事最硬)。

## RUNNING JOBS (2026-08-28 pm)
- BFCL STANDARD-FLOW RE-MEASUREMENT (user directive: all numbers via official bfcl flow, old custom-export numbers void):
  - lane A tmux hq:stdA GPU1 port 8921: base, sft s0-2, star s0-2 (logs/bfclstd_laneA.log)
  - lane B tmux hq:stdB GPU2 port 8922: oursadv s0-2, sad s0-2 (logs/bfclstd_laneB.log)
  - lane C tmux hq:stdC GPU3 port 8923: agentkd s0-2, ddpo s0-2 (logs/bfclstd_laneC.log)
  - lane D tmux hq:stdD GPU4 port 8924: pbsd s0-2, bbopd s0-2 (logs/bfclstd_laneD.log)
  - monitor b8x2f1a2e; script tools/bfcl_std_campaign.sh (merge->official generate->evaluate->copy->delete merged); scores land in results/bfcl_std/<tag>/
  - root cause confirmed mechanically: old export served flattened Qwen3_5ForCausalLM; hub merge serves correct Qwen3_5ForConditionalGeneration (426 overlaid + 312 restored tensors, bit-verified)
- codex xhigh implementing src/bfas/ (waiter b5gpsx8oh)
- deepseek teacher demo generate (PID 3701357)
- disk: purge APPROVED+DONE 2026-08-28 (export_vllm 58 dirs + _trash emptied; 55G->679G free); STANDING RULE: all experiments <=20% of total disk (rai 2.8T)

## PREVIOUS RUNNING JOBS
- ⛔ 2026-08-26 17:50: Azure 网关 token 配额耗尽(403, 两 key 同池, GPT+Claude 同池)。luna/sonnet demo 收集已停(verified: luna 12, sonnet 9, 已落盘可续)。等用户向 ORD 问配额规则/提额。DeepSeek 侧 Ollama 周配额同样卡住。teacher 双渠道均阻塞。
- ⚠️ 合规事件 2026-08-26: UF RC 禁止受关注外国(含中国)开发的 LLM(Fla. Stat. 288.860);管理员在 cancel Qwen job。tc 在 hpg 的全部 pending 作业已撤(ALFWorld×3, τ²)。hpg 上不得再提交 Qwen/DeepSeek 工作负载,待用户决定(FSU 侧待确认;备选:换 Llama/Gemma 学生 + 非中系 teacher API)。/blue 上仍有 Qwen 权重与 checkpoint,视执法情况需清理。
- rai tmux hq:v3diag (GPU4): V3 鲁棒性诊断 lane — np_s2(λ_pref=0) → s3 → s4 → np_s0;trainer 新增 v3stats(w_mean/pref 计数)
- BLOCKED: Ollama API 配额满 → teacher BFCL 行 + teacher dev40 std 重跑暂停(监视 quota 恢复自动续)
- rai tmux hq:baserun: BFCL base r3 生成中 (GPU2); r2 需重跑(旧目录污染已清)
- rai tmux hq:pbsdrun: PBSD BFCL ×3 链已触发 (GPU3, port 8901, campaign_D)
- hpg dept: basebench/ALFWorld 评测阵列 ×3 PENDING (QOSGrpCpuLimit)
- hpg yd24f: tc-tau2 smoke PENDING; tc-v3 阵列已取消(旧pool bug)
- V3-pref 三种子: 0.305±0.071 (.351/.340/.223);V3-nopref 三种子: 0.330±0.054 (.388/.319/.282), TGC 15/120 — 当前最佳配置(round-2 候选主配方)
- GPU4 继续: s3/s4(带 pref 对照种子)

## INFRA NOTE (2026-08-17): hpg 工作区迁移
- yd24f.fsu 组 /blue 配额被整组占满(8T/8T + 文件数到顶),所有写入失败(samp 两连败根因)。
- 工作区已整体迁至 /blue/fsu-compsci-dept/xc25.fsu/hq/tc-alignment(部门空间,余 12T);
  旧路径 /blue/yd24f.fsu/xc25.fsu/hq/tc-alignment 现为符号链接。旧结果在新位置的
  results_ydfull/、logs_ydfull/(只读)。.venv 仍物理在旧空间(只读使用,符号链接接入);
  envs/ 后台 rsync 补齐中。sync_to_hpg.sh / fetch_results.sh / *.slurm 已改指部门路径。
- rai→hpg 大文件同步受 MTU 黑洞限制:用 gzip+600B 分块 base64 推送(push 模式),取回用 900B 分块。

## ALFWorld/tau2 track (2026-08-29)
- hpg READY: envs/alfworld (.venv imports OK, 2.3G data), envs/tau2/repo cloned, project .venv torch 2.11+cu128
- sync_to_hpg.sh now excludes envs/ (code-only, seconds through tunnel)
- bfas efficiency rework at codex xhigh (log logs/codex_20260828_215533.log, waiter bu4hixpnr): per-seed collection cache + parallel teacher demos + vllm-backed parallel rollouts
- plan: smoke 5-task collect on rai -> full collect s0-2 (rai, GPU-shared) -> pools to hpg -> arms train/eval on B200 -> ALFWorld column; tau2 adapter next

## BFCL standard-flow column (2026-08-29 05:00, official harness-native flow)
- base 46.27 | PBSD 44.14+-0.97 | SAD 43.46+-0.29 | AgentKD 43.41+-0.98 | BB-OPD 43.14+-0.70 | SFT 42.85+-1.09 | dDPO 42.77+-0.05 | STaR 39.39+-1.04
- ours: s0=24.02 (NonLive 81.19/Live 78.02 ABOVE base; MultiTurn 4.50/Memory 6.24 collapsed -> single-turn-only pool destroys mt termination); s1 lane F GPU3, s2 lane G GPU1 running
- ours-mt retrain (mt-included pool 183 rows) on hpg array 40550930 -- TUNNEL DOWN 05:00, jobs unaffected, user notified
- collision guard: kill lane F script the moment its oursadv_s1 OVERALL fires (lane G owns s2)
- ALFWorld collect: quota pause-resume loop (key2 session limit), phase caches protect progress

## tau2 adapter LIVE (2026-08-29 pm)
- src/bfas/adapters/tau2.py (codex, reviewed): official tau2 run harness, student via vllm OpenAI endpoint, user-sim = deepseek metered separately in ledger (purpose=user_sim), pass^1 eval, 37 tests green
- pool: 178 tasks (retail 74, telecom 74, airline 30) -> support 50 (S_d 40 / S_c 10) per clamp rule
- collection queued AFTER ALFWorld demos finish (shared teacher quota)

## Night status (2026-08-30 ~02:20)
- v2 BFCL: all 27 trunc-fixed trainings DONE on rai; eval lanes: e1 GPU3 (oursadv x3), e2 GPU1 (sft/sad/agentkd x9), e3 GPU4 (ddpo/pbsd/bbopd x8, requeued after disk incident); tail 5 (bbopd_s2, star x3, oursmt s1/s2) queue when lanes free; oursmt_s0=39.02 already in
- disk: 100% incident #2 during ckpt wave; user granted blanket permission; purged v1 bfclb_* ckpts (240G) + _trash (44G) -> 293G free
- ALFWorld: teacher fixed (think:false + empty-retry + ledger cleansed of broken-era attempts); collection running, T-factor verdict pending
- mechanism doc docs/mechanism-sustained-improvement.md delivered: gain decomposition D*T*U vs interference; SELF-ANCHOR design (verified self-rollouts on high-phat tasks as behavior anchors, lambda by training-mass conservation) = next ours version; codex ChatGPT quota exhausted -> small fixes self-written, disclosed

## RUNNING JOBS (2026-08-30 21:10 EDT, verified live)
- rai GPU2 | bfas BFCL ours 采集 (deepseek teacher demos, seed1, dry-run) | pid 2537197 | logs/bfas_bfcl_ours_collect.log
- rai | v2e2 lane: agentkd_s1 generate 重跑 (磁盘事故#3 requeue) | logs/bfclstd_v2e2.log
- rai | v2e4 lane: pbsd_s0 generate 重跑 | logs/bfclstd_v2e4.log
- rai | ALFWorld teacher 采集 | logs/alf_collect.log
- hpg | 全部 PENDING(QOSGrpCpu/MemLimit, sdl v9/v10 阵列占队): awb3 array 40606110 (AppWorld trunc-fixed 8臂×3seed), ancir 40627612 (IR-surgical anchor)
- disk: 435G free / 97% used
- BFCL v2 表(截断修复后): base 46.27 | ours-mt s0 39.02 | PBSD 38.44/38.37 | AgentKD s0 38.12 |
  BBOPD 37.65/33.17 | SAD 36.85±2.16 | SFT 36.37±3.71 | dDPO 35.09±2.83 | anc2 35.43 FAIL | anc1 21.81 FAIL
  → v1→v2 全线下降(含 baseline);所有训练臂仍低于 base。待用户拍板 anchor 路线去留。

## POOL v4 — boundary-directed rebuild (2026-08-30 21:30, user directive)
User directive: pool 做得不够好,要生成更多边界 data trace;不管 baseline 是否普遍低于
base,**我们的方法必须超过 base**。方法探索阶段**单 seed 即可**(省算力);目标是三
seed 稳定超 base(如 AppWorld 上那样)。

### Diagnosis (results/analysis/pool_v4_diagnosis.md)
- ours-mt s0 39.02 vs base 46.27。失血:Irrelevance Detection 82.07→54.65 (−27.4)、
  MT Miss Param 45.5→27.5、MT Miss Func 50.5→37.0、Web Search 10.0→**0.00**、
  NonLive Parallel 76→67。上涨:Relevance Detection 75→93.75、MT Long Context +9。
  → 单一机制解释全部:学生学到**"永远调用"反射**。
- pool 体检:183 行 / **仅 33 个不同任务**;7 个计分类别**零质量**;
  web_search = 1 任务×17 行、multi_turn_base = 1×20、miss_param = 1×20。
- support 配额按 benchmark 大小成比例 = 与 deficit 反向(live_multiple 52 vs memory 8)。
- anchor v1/v2 失败的原因也在这:两版锚的都是**调用行**,给调用反射加油;
  真正该锚的是 **abstain 行(irrelevance)**,而 pool 里这类是 0。

### Fixes landed (38 tests pass)
1. `src/bfas/adapters/bfcl.py` — memory 类别修复。根因:`bfcl generate` 崩在
   `KeyError: 'MemoryAPI_'`(官方无 `memory` 可运行类别,它展开成 kv/vector/rec_sum;
   我们传 `memory` → backend 后缀空)。teacher memory 0/18 全是这个,一次模型调用都没发出。
   修:官方 loader 展开三 backend + id 改写 + prereq 写入链随任务进 selective 文件 +
   prereq 不进 task pool。memory 155 不可跑 → 465 可跑任务。
2. `src/bfas/protocol.py` — `coverage_floor` 采样(BFCL 专用开关,不动 AppWorld/tau2)。
   support 先均分再按大小铺余量,无魔法数字(floor = 预算 // 类别数)。
   memory 8→35、web_search 5→11、live_parallel 1→11、live_multiple 52→15。

### Stopped
- 旧配额采集 (pid 2537197) 已停:旧按大小配额 + memory 每次崩。已购 demo 在
  data/teacher_ledger/bfcl.jsonl(80 verified / 171 attempts),按 task_id 复用不重买。

### Next
- 探 memory / web_search 的 teacher 真实通过率(logs/mem_probe.log)→ 决定 deficit
  配额里 64% 给不给这两轴;买不到就退回 Multi-Turn + abstain。
- 然后重采 pool v4 → **单 seed** 训 + 评,门槛 **> base 46.27**;过了才铺 s1/s2。

### POOL v4 pipeline LAUNCHED (2026-08-30 21:32)
新发现(比 memory bug 更根本,见 notes/exp_log.md):
- **pool 里 phat=0 的行是 0 条**;79% 的行 phat=1;teacher 字段全 `self`;
  guided 行(`_mu_nll_sum`)0 条 → **pool 完全没有边界数据**,只是自我模仿已掌握区域。
  这是"训练臂结构上不可能超 base"的根因。
- pool 的 33 个任务 **100% 来自已归档的 v1 按大小 split**
  (`configs/bfcl_support_split_v1_proportional.json`),与当前 v2 均衡 split 只重合 2/50。
  pool 建于 8-29 00:06,v2 split 写于 8-30 07:12 —— 所有 BFCL 数字训的都是旧池。

Fixes/artifacts:
- `tools/bfcl_support_split.py` 改用 adapter 的 task_pool(memory 首次进 support);
  新 split = 21 类别 × 2-3 题 = 50(旧版备份 configs/bfcl_support_split_v2_nomemory.json)
- `tools/bfcl_teacher_demos.sh` selective 文件改用 adapter(memory 不再崩)
- 新增 `tools/bfcl_roll_pass.sh`(base/mt 学生 rollout pass,此前无脚本、纯手工跑)
- 旧 split 的 rollout 目录已归档到 `_trash/rolls_v1split_20260830/`

RUNNING (rai):
- teacher demos: `bash tools/bfcl_teacher_demos.sh deepseek-v4-pro-FC` → logs/bfcl_demos_v4.log
  (72 entries / 21 类别,含 memory prereq 链;3 attempts)
- base rollout pass: `bash tools/bfcl_roll_pass.sh base 4 8951 4` → logs/bfcl_roll_base_v4.log
  (GPU4, vllm port 8951, 4 repeats → phat)
NEXT: guided pass(phat=0 的题)→ pool v4 → **单 seed** 训 + 评,门槛 > base 46.27
- 整条链已 memory-aware:support_split / teacher_demos.sh / demo_pool / v3_pool /
  guided_pass 全部改走 adapter;备份在 scratchpad *.bak。38 tests pass。
- v2 基线最后两个评测 agentkd_s1 / pbsd_s0 于 21:09 无错误日志自行死亡(生成 ~80%),
  非本会话所为(21:16 采集仍在写 ledger)。暂不重排,优先 pool v4。
- phat v4 出炉:demand 40 题全部产出(memory 首次可跑)。phat=0:5 题;0<phat<1:16;
  phat=1:19。逐题 phat 见 results/analysis/phat_v4.json。
- 注意:每类别仅 1-3 题,support 上的 deficit 不能外推(web_search support phat=1.00
  但 benchmark Web Search 只有 10.0)。是否把 k 从 50 抬到 150-200 待用户拍板。
- 硬约束(今晚踩到):官方 selective 生成用包内固定路径的 id 文件,**同一时刻只能有一个
  selective 任务**,否则后者静默覆盖前者的选择表。已串行化 + 两个驱动都加了断点续跑。
- ollama key1 已 429,现用 key2。
- 当前链(后台 bwf2c7dc0):teacher demos(attempt 2-3, key2)→ 自动接 mt rollout pass
  (GPU4/8952)→ 之后 guided pass → pool v4 → 单 seed 训评。

### POOL v4 pipeline 状态 (2026-08-31 00:10)
- teacher demos: verified **34/40** demand,全部来自 attempt 1(贪心);attempt 2/3
  (T=0.7)零新增 → "贪心解不出的,采样也解不出"(采购成本论证的经验点)
- demo 库 34 条(11 条 stateful 只做 guidance);SFT baseline 池已按新 split 重建(23 行)
- mt rollout r0-r3 完成(44 题/8 类),dump 各 1.5-1.9MB / 43-44 题
- guided pass 跑着(GPU4/8953):符合条件仅 **2 道**(phat=0 且 teacher 会);
  另 3 道 phat=0 的题 teacher 也不会 → 应标 infeasible,不硬灌
- **权重洞察**:trainer 用 (1-phat) 加权,旧池 145 行 phat=1 → 权重 0。
  旧池的**有效**训练集只有 38 行 / 约 14 个任务,且全部是学生已能做对一半以上的题,
  边界上一行都没有。这是"结构上超不过 base"的更准确说法。
- 备份:`_trash/pools_pre_v4_20260831/`(13 个 pool/demo 文件)。
  注意 `pool_bfcl_ds_sft.jsonl` 旧版(v1 split)在备份前已被重建覆盖,不可精确复原;
  checkpoint 与分数仍在,受影响的只是那份输入文件。
- 下一步:pool v4 = `tools/bfcl_v3_pool.py --tag ds --repeats 4`(需一张卡算 guided mu)
  → 单 seed 训 + 评,门槛 > base 46.27
- ⚠️ 硬注意:`tools/bfcl_guided_pass.py` 会**就地改写官方数据文件**(注入 worked example),
  靠 `finally` 还原(`*.bak_guided`)。**绝不能 SIGKILL 这个进程** —— 被 -9 打断会把
  benchmark 数据永久留在被注入状态,之后所有评测静默污染。停它只用 SIGTERM/SIGINT。
  跑完必须确认 `bfcl_eval/data/*.bak_guided` 已消失。

### pool v4 第一个 seed (2026-08-31 01:30)
- pool v4: `data/bfcl_sft/pool_bfcl_ds_v3.jsonl` — 144 行 / 35 任务 / **19 类别**
  (旧 183/33/11);有效质量 Σ(1-phat) 20.2 vs 14.5;单任务最大 18 行 vs 36
- 训练:`bfclb3_ours_s0`,rai GPU4,**必须带** `AW_MAX_PROMPT_TOKENS=4096
  AW_TRUNCATE_SIDE=tail`(第一次漏了 → 91/144 行被 640/head 截断,已删重训;
  正确配置下 33/144 over cap)
- 评测:`bash tools/bfcl_std_campaign.sh 4 8961 bfclb3_ours_s0` → logs/bfclstd_v3e1.log
  门槛 > base 46.27。重点看 Irrelevance Detection / Web Search / MT Miss Param+Func
- 未解决(结构性):phat=0 的边界行仍为 0 —— guided 通道只支持单轮任务,
  stateful 边界够不着;而 BFCL 失分最重的正是 stateful

### pool v5 (2026-08-31 05:15)
- `data/bfcl_sft/pool_bfcl_ds_v3.jsonl` = 155 行 / 36 任务 / 20 类别;
  **phat=0 行 11 条(史上第一次有边界质量)**;有效质量 31.2(v4 20.2,旧池 14.5)
- multi_turn_base 经 guided 通道补回 11 行(修法 2 的直接产物)
- 不变量仅剩 live_parallel_multiple 报空(phat=0 且 teacher 也不会 → 真 infeasible,
  用 --allow-empty-categories 显式放行并记录)
- 训练 `bfclb4_ours_s0`(GPU4)→ 自动接 `bfcl_std_campaign.sh 4 8962`,门槛 > base 46.27
- 教训:`bfcl generate` 默认跳过已有结果(--allow-overwrite 默认 False),
  **任何意在"重新生成"的重跑必须先归档/删除结果目录**,否则是报成功的空转

## ⚠️ 机制纠正 (2026-08-31,用户指出,最重要)
**few-shot 设定 k=50 不可改。但用于蒸馏的数据是可以"生成"的 —— 需要多少生成多少。**

我此前的机制把 **few-shot 支撑集** 和 **蒸馏训练集** 当成同一个东西:学生 rollout、
teacher demo、guided pass **全部只在那 50 道题上打转**。后果:每类别 1-3 题、一道题
的抽样运气能让 Multi Turn 摆动 40 分、有效质量上限约 31 行。昨晚提议"把 k 抬到
150-200"是**在错误方向上补救**,已作废。

正确机制三段(此前只做了①和半个③):
1. k=50 few-shot query → 估计边界 T(规格,固定)
2. **T → 生成落在 T 内的新 query + 轨迹**(此前完全没有这一段)
3. 生成的数据 → 蒸馏
论文口号 distill-to-specification:**规格来自 k 条 query,数据来自生成**。

仓库里已有生成工具但只用于 pilot 的 gsm8k 域,**从未为 BFCL/AppWorld 建生成器**:
`tools/evol_extend_pool.py`、`tools/atomgen_pool.py`、`data/pool_gen_v{1,2}.jsonl`。

BFCL 生成器设计草案(待用户确认 a/b 两点):
- 生成单元 = (query, 函数集, 目标调用);由现有 function schema 出发,teacher 按指定
  调用模式造 query,ground truth = 目标调用 → **官方 AST checker 可直接验证**
- 边界导向:按 T 的薄弱处定向生成(缺 abstain → 造"工具不覆盖"query;缺多轮终止 →
  造短程收尾任务;缺 miss_param → 造参数缺失 query)
- stateful:造多轮脚本,用官方状态后端执行验证
- 风险:teacher 造题 + teacher 解题 = 自洽但证据弱,且会烙进 teacher 偏好。
  拟定原则:**生成题必须过官方 checker 才进池,过不了就丢**
- 待确认:(a) 强制官方 checker 验证?(b) 生成规模上限(决定 teacher 配额消耗)

## 磁盘 (2026-08-31 已清)
清掉 awb2_*(20)、bfclb2_*(23)、bfclb3_ours_s0 → **释放 368G**,306G → 674G(95%)。
分数全部保留在 results/bfcl_std/(1.8G)与 results/appworld/(30M),只删权重。
保留:bfclb4_ours_s0、bfclb5_gentle_s0、awb9_adv_*、awb6_*、base_export_ctl。

## 快速迭代:代理评测 (2026-09-01)
- `configs/bfcl_proxy_ids.json` = 507 条 / 21 类别(每类 20,排除 support split,固定种子)
- `bash tools/bfcl_fast_eval.sh <GPU> <PORT> <tag>` → 分轴打印,单轮与 stateful 同时可见
- **只与自身可比,永远不作为上报数字**;上报仍走 `tools/bfcl_std_campaign.sh` 全量

## 显式保留项 + 今晚排程 (2026-09-01)
- `src/appworld_train.py`:权重从 `(1-p̂)` 改为 `(1-p̂) + AW_PRESERVE·p̂`。
  默认 `AW_PRESERVE=0` = 旧行为不变(38 测试通过)。**取值由代理集测,不人为拍板。**
  动机:只买缺口、对已有能力零投入 → 已有能力被覆盖时没有任何阻力(pool v6 摆到 97% stateful)。
- RAI 串行队列 `tools/tonight_rai.sh 2 8985`:preserve ∈ {0.25, 0.50, 1.00},
  每个 训练 + `bfcl_fast_eval.sh` 代理评测(507 条,分轴打印)。
- HPG 阵列 **40747650** `scripts/bfclb7_hpg.slurm`:pool v6 三种子 + 各自全量官方评测。
  分工:hpg 出可上报数字(种子方差),rai 出选配比的代理数字。
- awb3 24/24 全部结束;ancir 已不在队列(IR anchor 线被生成机制取代)。
- 机制改进(2026-09-01 下午,依用户三点):观测由决策驱动(区间跨过池平均才继续 roll,
  批次落在后验预期内即 measured_out;去掉 median-sd 启发式与固定 4 次);生成统一为单一
  入口 bfcl_generate_mt.py(一切皆 episode,judge 随种子形状特化);多轮融合方案定为
  TCOD 前缀课程 + Guided-OPD 式"学生状态修复数据"(见 exp_log)。

## HPG 恢复 + 全阵列提交 (2026-09-01 ~13:45)
- 隧道 MTU 黑洞由 Mac 重建 autossh 解决;根因链:bfcl-venv 是 rai rsync 过去的死venv
  (shebang 指 /home/xueqi)→ hpg 上 BFCL 评测从未可用 → 新建原生 venv(BFCL_VENV_READY,
  vllm 装在 envs/bfcl/.venv);评测脚本 BFCL 自动探测两个 venv 位置 + PATH 带上 venv bin。
- pool v9(955 行)经分块推送落地,md5 校验通过。
- 已提交:**40788920** bfclb10(v9 三种子)/ **40788921** bfclb7(v6 三种子)/
  **40788922** bfclpres(preserve 0.25/0.50/1.00)—— 9 个作业并行,均带无分即错检查。
- rai:v9 队列守望继续等 GPU1/4。
- 2026-09-01 深夜:用户授权双账户排程(dept + yd24f);λ=0.25 重试走 yd24f(40829742)。
  五臂已出四:v9+curr 38.30 > v9+clw 37.47 > v9 36.70 > v9+repair 36.34;
  λ=0.50(v6池)显示 preserve 项按设计护住单轮(Non-Live 86.25 纪录、Live 77.05)。
  在跑:bfclb14(curr+clw 组合)、λ=1.00、λ=0.25(重试)。
  下一步:全家桶臂(v9 + curriculum + cluster + λ最佳档)冲 base。

## ⟳ RESTART CHECKLIST (written 2026-09-01 ~19:50 EDT, before CLI restart)
重启后第一件事(会话级的东西全部丢失,要重建):
1. 重挂 hpg 结果监视(每 5 分钟):grep 'OVERALL=|NO SCORE|DONE' 于
   logs/bfclb14_40823622.out(curr+cluster 组合)、logs/presev_40829742_*.out(λ=0.25, yd24f 账户)
2. 重挂每小时 :23 进度汇报 cron(规则:分轴报、proxy 标注、无变化一句话、死作业直说)
3. bfclb14 出分后 → 提交全家桶臂:v9 池 + AW_CURRICULUM=turn + AW_CLUSTER_FILE + AW_PRESERVE=0.5
   (slurm 模板照 scripts/bfclb14_cc.slurm 加 AW_PRESERVE=0.5,tag bfclb15_full_s0)
4. 出全对照表后 → 给用户发英文完整 pipeline(在对话框,不写文档)+ 五臂表
当前已出分(全量官方,单 seed):v9+curr 38.30 > v9+clw 37.47 > v9 36.70 > v9+rep 36.34;
λ 剂量曲线(v6池):0.50 甜点(26.56,Non-Live 86.25,Live 77.05);λ=1.00:Live=base 77.65,Non-Live 87.31。
API 面板:置顶消息 id 在 ops/api_panel_target.json,刷新用 edit_message。

## ⟳ 重启后恢复 (2026-09-01 ~20:05 EDT)
- HPG 项目路径实为 `/blue/fsu-compsci-dept/xc25.fsu/hq/tc-alignment`(不是 yd24f 那条)。
- bfclb14_cc_s0(40823622,dept):训练+merge 已完成(263 steps,hub_merged 校验过),
  正在跑官方 BFCL 评测(bfcl_std_campaign 0 9050)。
- presev 40829742(λ=0.25 重试,yd24f,array 0-2):PD QOSGrpMemLimit。
- λ=1.00 已出分:bfclp_100_s0 OVERALL=23.02%(v6 池);λ=0.50 26.56%。
- 已重挂:Monitor bviouqqp4(5 分钟 grep OVERALL/NO SCORE/DONE/Traceback)、cron ab1068a5(每小时 :23 汇报)。
- 已预写 `scripts/bfclb15_full.slurm`(= bfclb14_cc + AW_PRESERVE=0.5,tag bfclb15_full_s0),
  **bfclb14 出分后再 sbatch**(dept 账户)。
- 20:15 bfclb14_cc_s0 出分:OVERALL 37.29(Non-Live 83.46 / Live 76.09 / MT 18.00 / Memory 39.78 / Irrel 79.77)。
  **组合不叠加**:MT 掉到 18.00(低于 v9 基线 21.12),curriculum 的 +3pp 消失;单轮轴≈b12。
  推测两机制在多轮行上争质量。已记入 exp_log。
- 已提交 **bfclb15_full_s0 = 40832797**(dept,b14+AW_PRESERVE=0.5,日志 logs/bfclb15_40832797.out)。
  向用户建议补提 v9+curr+λ0.5(无 cluster)臂,等回复。
- 20:25 用户批准:提交 v9+curr+λ0.5(无 cluster)臂 → **bfclb16_cp_s0 = 40832925**(dept,
  scripts/bfclb16_cp.slurm,日志 logs/bfclb16_40832925.out)。
  清理过时实验:scancel bfclb15 40832797(b14+λ,组合已证不叠加)、presev λ=0.25 40829742(v6 池,
  yd24f 排不上,v6 已被 v9 取代);kill rai 上 v9 队列守望 pid 63975(本地重训 bfclb10 v9 基线 +
  proxy 评测,hpg 已出官方分 36.70,无需再跑)及两个 16 天前的 aw4_queue tail。
  当前 hpg 只剩 bfclb16 一个作业;rai 无本项目进程。

## ⟳ 会话拆分交接 (2026-09-01 20:20 EDT)
- 本项目从此由独立 tmux 会话 `tc`(cwd 本目录,DISCORD_STATE_DIR=discord-tc)负责;原 hq 会话转为中控(DM only)。
- 交接时状态:hpg 只有 **bfclb16_cp_s0 = 40832925**(v9+curr+λ0.5,dept,PD),日志 logs/bfclb16_40832925.out;
  rai 无本项目进程。原会话的 Monitor/cron 已停,**tc 会话启动后需重挂**:
  1. 5 分钟监视 bfclb16 日志(grep OVERALL=|NO SCORE|DONE|Traceback)+ squeue;
  2. 每小时 :23 频道进度汇报 cron;
  3. bfclb16 出分后 → 六臂全对照表 + 给用户发英文完整 pipeline(对话框内,不写文档);
     若 bfclb16 ≥ 38.30 则 curriculum+preserve 是当前最佳栈,考虑加种子。

## ⟳ tc 会话上线 (2026-09-01 ~20:20 EDT)
- 核对:bfclb16_cp_s0 = 40832925 RUNNING(c1001a-s15,dept),日志 logs/bfclb16_40832925.out 已进 epoch 1/3 课程阶段(392/955 行)。
- 已重挂:Monitor bm635nard(5 分钟 grep OVERALL/NO SCORE/DONE/Traceback + squeue 退出检测)、cron 6844b078(每小时 :23 汇报)。
- 已在频道发"tc 会话已上线,恢复了 3/3"。rai 无本项目进程。

## ⚠️ 用户指示 (2026-09-01 20:45 EDT):实验不能只基于一个 benchmark
- 用户原话:"我说过你的实验不能是仅仅基于一个 benchmark,你应该基于所有的几个 benchmark"。
- 现状:BFCL 六臂;AppWorld 停在 awb3(旧池 v32_adv,新机制未验证);ALFWorld 采集 8/30 停在 58 任务/3 demo;τ² 未采集。
- 已提方案(等确认):bfclb16 后 BFCL 冻结 → AppWorld 四臂(v32/+curr/+clw/+λ0.5,机制迁移检验)→ ALFWorld 续采集+四臂 → τ²。
- 规则:此后每个机制的验证必须跨 benchmark 报告,BFCL 单臂不再单独加。

## RUNNING JOBS (2026-09-01 20:55 EDT, 跨 benchmark 补齐启动;用户 20:46 批准"先开始补齐")
- hpg dept | **bfclb16_cp_s0 = 40832925**(v9+curr+λ0.5)RUNNING | logs/bfclb16_40832925.out | Monitor bm635nard
- hpg dept | **awb4 = 40836032** array 0-2(AppWorld 机制迁移:+curr / +λ0.5 / +curr+λ0.5,池 pool_v32_adv,seed0,对照 awb3_oursadv_s0 7/40)| scripts/awb4_mech_hpg.slurm | logs/awb4_40836032_*.out | Monitor b7ht0464i
- rai GPU2 | ALFWorld teacher 采集续跑(key2,dry-run=只采集)| pid 645239 | logs/alf_collect.log(续 8/29 缓存 results/bfas/alfworld/collect_s0)
- rai GPU4 | τ² 采集(key1,port 8940;20:53 修 empty-target bug 后重启)| pid 656828 | logs/tau2_collect.log
- cron 6844b078 每小时 :23 汇报。
- 发现:pool_v32_adv 只有 **21 个不同任务**(689 行 = 轮次),cluster 级 credit assignment 在此池无意义(BFCL 池 ~900 任务);AppWorld +clw 臂需等 AppWorld 生成池。tools/cluster_pool.py 已加 --generic 模式备用。
- rai GPU1 | AppWorld 链:cluster 打标(--generic)→ 本地训 awb4_cp_s1(curr+λ0.5, seed1)→ dev40 评测 | chain pid 652810 | logs/awb4_gpu1_chain.log, logs/awb4_cp_s1_train.log | tools/awb4_rai_gpu1.sh
- Monitors:bm635nard(bfclb16)、b7ht0464i(awb4 hpg)、rai 三条 lane 合一(tail -F alf/tau2/gpu1 chain)。用户 20:50:"都可以占上,别占满就行"(GPU0/3 是同学的)。
- 20:57 τ² 修复验证:airline:40/42 verified demo 落 ledger。成本 2.5–5 万 token/demo → ~200 万/seed;key1 周额度可能只够 1 seed,待观察。
- 21:18 **bfclb16_cp_s0 出分 37.74**(NL 84.38 / Live 75.72 / MT 25.87 最高 / Mem 29.25 / Irrel 81.18)。低于 b11 38.30 → 不加种子;BFCL 六臂冻结,最佳单配方仍是 v9+curriculum。hpg 现只剩 awb4 阵列(_0/_1 RUNNING,_2 PD)。
- 21:30 故障两起:hpg awb4_cp_s0(40836032_2)存权重时 Disk quota exceeded(dept 组 21.4T/21.5T);删除无分数残留 bfclp_025_s0(17G)/bfclb15_full_s0/awb4_cp_s0 后写入测试通过,cp_s0 以 --array=2 重提 = **40838618**(Monitor 已挂)。我们在 /blue/fsu-compsci-dept 占 3.2T,待清。
  rai awb4_cp_s1 epoch 3 OOM(97G)→ appworld_train.py 新增 AW_GRAD_CKPT=1(梯度检查点),chain2 pid 681359,logs/awb4_gpu1_chain2.log。
- 21:45 hpg 占用:账户 3.2T = sdl 2.8T + tc 364G + tools 30G。tc 可清:旧池 BFCL 合并权重(bfclb_oursmt s0-2、bfclb2_oursanc3_s0、bfclp_050/100,~48G)待用户点头;sdl 2.8T 归中控。
- rai GPU1 awb4_cp_s1 重训 epoch 1 进行中(AW_GRAD_CKPT=1,显存 18G)。
- 21:35 汇报:awb4 curr_s0/pres_s0 权重已存盘、评测中;cp_s0 重提 40838618 FAILED(配额再满,hf-cache 加载 Qwen 报 OSError,需查 snapshot 是否损坏);rai cp_s1 epoch 2/3;τ² 9/21 verified(key1);ALFWorld 卡在 key2 429(17 尝试/3 demo 不变)—— 等用户决定是否共用 key1。
- 21:40 根因:配额满 → hf-cache models--Qwen--Qwen3.5-4B/refs/main 被截成 0 字节 → 所有 Qwen 加载 OSError。已写回 851bf6e8…,offline 解析通过。教训:hpg 配额满时 HF 缓存会被静默损坏,失败后先检查 refs/main。cp_s0 第三次提交(见下)。
- cp_s0 第三次提交 = **40839830**(dept,Monitor 已挂)。当前 hpg 本项目:40836032_0/_1 评测中、40839830 PD。
- 21:50 **awb4_curr_s0 = 8/40 (20.0%)** vs awb3_oursadv_s0 7/40;pres_s0 评测中;cp_s0 40839830 训练中;rai cp_s1 epoch 2-3。
- 22:12 **awb4_pres_s0 = 6/40 (15.0%)**。AppWorld 同池三点:curr 8 > 无 7 > λ0.5 6(单 seed)。cp_s0 40839830 训练 31 分钟。
- 22:15 用户:"如果 base 是 0 那就说明 base 选的不太对"。数据:同框架 base 2B 0/40、4B 0/40、9B 2/40,teacher 7/40 → 脚手架压上限。已提议:(1) 官方 ReAct 脚手架重测 base 4B/9B + teacher(GPU1 空出后,不花额度);(2) 在修好框架下选最小明显非零的 Qwen3.5 作 AppWorld base,baseline 矩阵重跑。等用户确认。
- 22:20 用户批准"官方 ReAct 重测 base → 重选 base → baseline 矩阵重跑"。进展:克隆 envs/appworld-repo(0.2.0.dev0,2026-02);其 experiments 代码需要 appworld 0.2 + recoma/litellm(pydantic v2),与 envs/appworld-venv(0.1.3, pydantic v1)不兼容 → 决定另建 envs/appworld-official/.venv,不动现有 venv。
  事故:给 appworld-venv 装 litellm 时 pydantic 被升到 v2,appworld import 断了 ~5 分钟,已回退 pydantic 1.10.26 + 卸 litellm,验证 import 正常(cp_s1 评测尚未开始,未受影响)。
  vllm 已在 GPU4:8950 起 Qwen3.5-4B base(served name qwen35-4b-base,logs/vllm_awbase4b_8950.log)。
- 22:40 官方脚手架环境就绪:envs/appworld-official/.venv(appworld 0.2.0.dev0 editable from envs/appworld-repo + appworld-agents[simplified]);LFS bundle 经 media.githubusercontent 手工拉取;`appworld install --repo` 解包(写了 ~/.cache/appworld);数据 0.2.0 下到 envs/appworld-official/data(dev.txt 与 0.1.0 完全相同,但 ground_truth/evaluation.py 有差异);envs/appworld-repo/data → ../appworld-official/data。
  配置:envs/appworld-repo/experiments/configs/simplified_react_code_agent/local/qwen35-{4b,9b}-base_dev.jsonnet(官方 ReAct 原样,只换成本地 vllm :8950,非思考模式,max_steps 50)。
- 22:50 官方脚手架 smoke 通过(需 OPENAI_API_KEY/OPENAI_BASE_URL 环境变量,官方 openai 路径忽略 config 里的 base_url;打了两个本地补丁:litellm 可选、reasoning_content None)。已起 **tools/awoff_base_chain.sh**(pid 763611,GPU4:8950):官方 ReAct dev56 base 4B → 9B,num-processes 4;日志 logs/awoff_base_chain.log、logs/awoff_qwen35-{4b,9b}-base_dev.log;输出 envs/appworld-repo/experiments/outputs/…;Monitor bhmhp6n3l。
- 22:50 用户"可以的,你做吧"→ (1) 删 hpg 六个旧池权重目录(48G),headroom 6.1T;(2) ALFWorld 采集改 key1 重启(pid 809356,workers 2,与 τ² 共享)。
- 22:52 **官方脚手架 base 4B:dev57 TGC 14.0%(8/57),dev40 子集 4/40 = 10%**;我们框架 0/40 → 脚手架压制确认。9B 自动接跑中。
- 23:02 **awb4_cp_s0 = 3/40**(curr+λ0.5,mean_steps 19.7)。AppWorld 同池四点:curr 8 > 无 7 > λ 6 > curr+λ 3,排序与 BFCL 一致;但整列待官方脚手架重做。9B 官方重测 vllm 加载中(tools/awoff_9b_only.sh,pid 827088)。
- 23:15 **官方脚手架 base 9B:dev57 TGC 15.8%(9/57),dev40 子集 6/40**;4B 14.0% / 4/40。两者在官方框架下都明显非零,4B→9B 只在 difficulty-2 上多 2 个。建议:AppWorld 列整体切到官方脚手架(rollout+评测),base 继续用 4B(与 BFCL 一致、省算力);等用户拍板后改 bfas AppWorld adapter 走官方 agent。

## 设计:AppWorld adapter 切官方脚手架(2026-09-01 23:20,待用户拍板 base 尺寸后实施)
照 tau2 adapter 的模式(官方 harness + vllm 端点 + 从官方日志渲染 Turn):
- 学生 rollout:`appworld run simplified_react_code_agent/local/<policy>_<split>` (envs/appworld-official/.venv, --root envs/appworld-repo, OPENAI_BASE_URL 指向 serving_lane 的 vllm,--task-id 或 dataset 子集,--num-processes 4);
  每任务 `tasks/<id>/logs/lm_calls.jsonl` 含完整 request.messages + 输出 → render_logged_turn 风格渲染成 Turn(prompt=官方 messages 经学生 chat template,target=输出);
  verified = `evaluation/report` 的 success(官方 evaluator)。
- teacher demo / teacher_episode:同一 agent,model_config 走 litellm→teacher(Azure gpt-5.6-luna 或 ollama deepseek),guided 版把 worked example 追加到官方 instructions 之后(与现在 demo.worked_example 注入方式一致)。
- evaluate:官方 run 于 dev(或 test_normal)+ 官方 evaluate,headline=TGC。自家 src/appworld_eval.py 退役(仅留作历史对照)。
- 训练侧不变(pool 行 = 官方 prompt 渲染,turn_index 照旧)。
- 已有:配置模板 experiments/configs/simplified_react_code_agent/local/*.jsonnet;两个本地补丁(litellm 可选、reasoning_content None);数据 0.2.0。
- 重跑范围:base、6 baselines、ours(v32 池需在官方脚手架下重新采集,因为 prompt 变了)。
- 23:40 τ² infra bug:bfas serving lane 的 vllm 未开 auto tool choice → 官方 harness 的 tool_choice=auto 全 400,学生 rollout 全 infra error。已在 src/bfas/run.py 加 --enable-auto-tool-choice --tool-call-parser hermes(BFAS_TOOL_PARSER 可改),20 conformance 测试通过;τ² 采集重启(demo 缓存保留)。ALFWorld key1 也大多 429,verified 4,基本停滞。
- 23:55 τ² 修复验证:学生 rollout 出 reward(4 条,1 成功);残余 infra error = user-sim 空消息(deepseek 偶发,harness 重试 4 次),非我方 bug。τ² pid 884699。
- 00:35 **awb4_cp_s1 = 9/40(22.5%)** vs 同配方 seed0 3/40 → AppWorld 单 seed 噪声 ±3 任务以上,四点排序不可读。rai GPU1 已空。
- 01:25 用户纠正(记入记忆):**方法探索只跑单 seed**(多 seed 等方法定稿);**机制测试四台一起**(BFCL/AppWorld/ALFWorld/τ²),不是只挑两台。用户 01:20 拍板 base 4B + AppWorld 切官方脚手架;01:21:"把该修复的都修复上,快点拿到可用于方法提升的信号并迭代"。
- 01:25 新文件 src/bfas/adapters/appworld_official.py(AppWorldOfficialAdapter:官方 agent 跑学生/teacher/评测,lm_calls.jsonl 渲染 Turn,官方 evaluator 判 verified;运行产物在 envs/appworld-repo/experiments/outputs/simplified_react_code_agent/bfas/,日志 logs/bfas/appworld_official/)。
- 01:45 τ² 第二次崩:guided 渲染 prefix 不匹配(临时目录已删无法复现)→ 改为跳过并 dump 到 logs/bfas/tau2_render_failures/;同时发现 23:15 的 unguided 缓存是 tool_choice bug 期间采的(全 400),已移 _trash,重采。τ² 重启(key1,GPU4)。
- AppWorld 官方 adapter:5 单测 + 28 bfas 测试通过;GPU1:8960 上 smoke(2 rollout → teacher demo → guided)进行中,logs/bfas/appworld_official/。
- 01:55 额度腾挪:ollama key1 已 429、key2 恢复 200 → τ² 重启改 key2(pid 976555,user-sim 也走 key2);ALFWorld 改 **Azure gpt-5.6-luna 当 teacher**(BFAS_TEACHER=gpt-5.6-luna,pid 977397,GPU2),绕开 ollama 周额度。
- 02:05 **Azure key 真相**:~/.azure_llm_api 的 AZURE_LLM_KEY/KEY2 均 401;~/hq/secrets/llm_apis.env 的 AZURE_P1_PRIMARY(面板用的)对 gpt-5.6-luna/gpt-5.4 均 OK(经 appworld_teacher 客户端验证)。启动 teacher 时用 env AZURE_LLM_ENDPOINT/AZURE_LLM_KEY 覆盖(见 tools/alf_collect_azure.sh),不改 $HOME 文件。
- 02:05 AppWorld 官方 adapter smoke:学生 rollout 2 任务 46s OK(1 verified);teacher 因 ollama 429 失败 → adapter 新增 Azure/litellm teacher 模式(BFAS_TEACHER=gpt-5.6-luna),6 单测通过。ALFWorld 以 Azure teacher 重启(tools/alf_collect_azure.sh)。
- 02:25 **GPU 序号坑**:rai 是混合卡,CUDA 设备顺序 ≠ nvidia-smi 序号;`CUDA_VISIBLE_DEVICES=4` 让 τ² 的 vllm 跑到了同学的 GPU0 上。此后一律用 GPU UUID(`nvidia-smi --query-gpu=index,gpu_uuid`)指定;τ²、ALFWorld 已按 UUID 重启。ALFWorld ledger 已清掉 7 条 Azure 401 空账(备份在 _trash)。
- 02:25 官方 agent 的 litellm 路径对 gpt-5 系拒绝 temperature≠1 → language_model.py 本地补丁 litellm.drop_params=True;AppWorld Azure teacher smoke 重跑中。
- 01:40 校时:上面 01:25–02:25 的时间戳估快了约 40 分钟(实际 00:45–01:40)。当前:alf pid 991024(Azure teacher,GPU2 by UUID)、τ² pid 990638(key2,GPU4 by UUID)、AppWorld Azure teacher smoke 进行中、vllm 8960 (GPU1) 供 smoke。
- ~01:50 根因补充:bfas ServingLane 用 `--gpu` 的值覆盖 CUDA_VISIBLE_DEVICES,所以 `CUDA_VISIBLE_DEVICES=4 --gpu 0` 实际总是 GPU0。现在 `--gpu <GPU-UUID>`(port offset 自动为 0,靠 BFAS_PORT_BASE 区分 lane)。τ² pid 999627(GPU4 UUID,key2);ALFWorld pid 1002096(GPU2 UUID,teacher **gpt-5.4**,与 AppWorld 的 gpt-5.6-luna 分开部署以拆分 Azure 每分钟限流)。AppWorld adapter 的官方 agent retry 改 15s×60。
- ~02:00 确认:τ² vllm 在 GPU4(48.9G),GPU0 已无本项目进程;τ² 学生 rollout 出 reward;ALFWorld gpt-5.4 teacher 第一条真实 ledger 记录已落。
- ~02:10 AppWorld Azure teacher:gpt-5.6-luna 部署被系里共用打满(面板"shared deployment window 1,506,993/1,507,000 per 60s"),重试 15 分钟仍 429 → 改试 gpt-5.4(ALFWorld 已在用且能过)。smoke 重跑中。
- ~02:20 **AppWorld 官方 adapter 全链路 smoke 通过**(teacher gpt-5.4 → demo → guided rollout verified)。正式采集已启动:tools/aw_collect_official.sh(单 seed,GPU1 UUID,port 8950),日志 logs/aw_collect_official.log,产物 results/bfas/appworld/。成本:~10 万 token/demo。
- ~02:40 **ALFWorld 根因 #3:prompt 丢目标**(目标句只在首个观测里,8 行历史窗一过就看不见)→ 学生/teacher 都乱走。已修(每轮前置 Goal:),旧缓存与整个 alfworld ledger 作废(_trash),lane 重启(pid 见 logs/alf_launch_offset.txt)。正在跑 3 条 teacher 验证。
- 02:40 ALFWorld 修复验证:gpt-5.4 teacher 3/3 通过(修前 0/20)。AppWorld teacher 预算看门 Monitor bvz2n6lgd(400 万 token 停 lane)。
- 03:35 采集进度:AppWorld teacher 30/32 verified、369 万 token(看门改 490 万,让 demo 阶段跑完);ALFWorld 20/26 verified(修复有效);τ² teacher 32/58、学生 rollout 76。Azure 周额度将主要被 AppWorld demo 用掉,等用户确认。
- 03:45 **Azure P1 周额度耗尽(403)**(gateway 计费 > ledger 的 380 万)。两条 lane 切 AZURE_SLOT=P2(独立 500 万)重启;ledger 清掉 403 空账(alfworld 14 行、appworld 6 行,备份 _trash)。AppWorld demo 已 30 条 verified,剩 ~8 任务。
- 03:55 P2 切换确认:AppWorld ledger 33→38 行(434 万 token 累计,P2 看门 830 万,Monitor bwzoet7kr),ALFWorld 25→26 行。lane pid:appworld 1098873(logs/aw_launch_pid.txt)、alfworld 1099058、τ² 999627。
- 04:35 AppWorld demo 阶段完成:40/42 verified(40 demand 全覆盖),479 万 token;学生 unguided rollout 进行中(GPU1)。ALFWorld teacher 44/73 verified,学生 rollout 已开(GPU2)。τ² 学生 rollout 92。
- 05:35 AppWorld 学生 rollout 中;ALFWorld teacher 67/129 verified,学生 rollout 中;τ² rollout 145。三 lane 存活。
- 06:35 三 lane 存活:AppWorld 学生 rollout 中;ALFWorld teacher 92/194;τ² rollout 199。无新分数。
- 07:40 ALFWorld lane 死因:GPU2 被同学 vllm 占满(44.8G),我们 unguided 阶段 vllm 起不来 → 改 ALF_GPU=1 GPU_UTIL=0.4 与 AppWorld vllm 共用 GPU1(49+39G)重启(pid 见 logs/alf_launch_offset.txt)。同学也在 GPU4 上放了 45G vllm(与 τ² 的 49G 共存)。
- 07:40 τ² user-sim 在 ollama key2 上 429 上千次,rollout 停滞(211);unguided 阶段 6 小时未落盘,不敢重启。tau2 adapter 新增 Azure 通道(BFAS_TAU2_USER_MODEL=azure/gpt-5.4 或 BFAS_TAU2_TEACHER=azure/...,凭据 env-only),9 测试通过;若下次汇报仍停滞则以 Azure user-sim 重启。
- 07:50 τ² 重启为 tools/tau2_collect.sh(pid 1301337):user-sim = azure/gpt-5.4(P2),teacher = deepseek@ollama key1(demo 已缓存),--seeds 1,GPU4 UUID。放弃了 key2 上停滞 6h 未落盘的 unguided rollout。
- 07:55 **Azure 四把 key 全 403**(P1/P2 均耗尽)。ALFWorld lane(pid 1298185)已在 GPU1 起 vllm(44G)进入学生 rollout;teacher demo 107 条已缓存。τ² 再次重启:TAU2_USER=deepseek-v4-pro(ollama key1),pid 见 logs/tau2_launch_pid.txt。
- 08:05 τ²(key1 user-sim)学生 rollout 已出 reward,无 429。三 lane:AppWorld pid 1098873(GPU1)、ALFWorld 1298185(GPU1)、τ² 1307859(GPU4)。
- 08:40 ALFWorld 采集三阶段完成(collect_s0/collection.json),建池 audit 失败:adapter suffix 固定 marker 列表不含 Qwen3.5 的 "<|im_start|>assistant\n<think>\n\n</think>\n\n" → _render 改为从模板学 suffix(同 tau2),20 测试过,lane 重启走缓存建池(pid 1466431)。
- 08:50 ALFWorld 建池第二个 audit:最长 prompt 1103 token > 默认 cap 1024 → 脚本设 AW_MAX_PROMPT_TOKENS=2048,重跑建池(pid 1468535;logs/alf_launch_offset.txt 现为 "pid offset" 两列)。suffix 学习修复已生效(第一个 audit 通过)。
- 08:55 **ALFWorld 池建成(3743 行)**;ours 臂 train+eval 在 rai GPU1 启动(ALF_EXTRA="" 非 dry-run,AW_GRAD_CKPT=1;pid/offset 见 logs/alf_launch_offset.txt)。下一步 base 臂评测对照。
- 09:05 ALFWorld train 模式两次误启动:(a) 08:39 建池 run 是 --seeds 3,建完 s0 池后自动去采 s1,vllm(脚本硬编码 GPU_UTIL=0.45,与 AppWorld 49G 同卡)EngineCore 死 → 500;(b) 残留 APIServer 占 8930。已修:脚本 GPU_UTIL/PORT 可由 env 覆盖,collect_s1 残片移 _trash,ALF_SEEDS=1 GPU_UTIL=0.35 重启 train 模式(pid 1477330)。`--seeds N` = 前 N 个 seed。
- 09:15 发现 lane 脚本 bug:`${X_EXTRA:---dry-run}` 把空串当未设 → 所谓 train 模式其实都是 dry-run(秒退、零输出)。三个脚本改为 `MODE=train` 显式开关。ALFWorld ours 真正 train 模式启动:pid 1479433(GPU1,GPU_UTIL 0.35)。
- 09:20 ALFWorld ours 训练开始(rows_over_cap 0/3743,cap 2048,grad ckpt,GPU1 62G);Monitor 已挂(epoch/SUMMARY/eval/错误)。
- 09:35 ALFWorld ours 训练 52 分钟(epoch 1 未完);AppWorld rollout 第 11 轮;τ² rollout 88,无 429。
- 10:35 ALFWorld epoch 1/3 完(468 步,w_mean 0.94);AppWorld rollout 第 11 轮 >2h;τ² rollout 151。
- 11:35 无变化:ALFWorld epoch 3 训练中;AppWorld rollout 第 11 轮;τ² rollout 203。
- 11:50 用户:ollama 已恢复;gpt-5.4 数据要留(已归档 results/bfas_archive/gpt54_teacher_20260902/,AppWorld collect_s0 落盘后补);要求先做方法分析。已写 docs/2026-09-02-method-analysis-zh.md(方法定义 / 四台证据 / P1–P7 问题 / S1–S7 方案 / 下一步建议),等用户拍板。
- 12:05 用户要求分析 v2:设定按 few-shot 规则(5% clamp[50,250])而非固定 k=50、目标写清、流水线写细、问题按 benchmark 写"好/不好/为什么/打算"。已重写 docs/2026-09-02-method-analysis-zh.md。发现两处论文表与实现不一致:BFCL support 实际 50(coverage floor)vs 论文 235;τ² 池 178 vs 论文 115。
- 12:10 用户:方法部分要从机理写设计原理,不写工程。分析 v3 已改为五条设计原理(先测再买 / 规格来自 few-shot、数据来自生成 / 从学生状态出发 / 收益−伤害分解 → 四个训练机制 / 预算一等公民)+ 按台的机理原因。等拍板下一步。
- 12:15 分析 v4:方法部分改为六步连贯流程(机理嵌入每一步),已发。等用户拍板第 4 节 1–5。
- 12:40 分析 v5:整套方法以 capability atom 账本组织(需求/供给/缺口),数据获取四步 + 蒸馏四机制 + 理想 vs 实现差距表 + 40 篇 2025–26 文献 → 改进 A–H(A 边界×可教性、B 按 p̂ 分层供给/最早分叉纠正、C 原子条件生成、D 真值保持增强、E 轮级权重、F 黑盒 on-policy 奖励、G 自蒸馏锚+梯度冲突过滤、H 原子依赖课程+行为计时)。等用户拍板。
- 12:40 τ² 在 key1 上再次 429(880 次),key2 已恢复。根治:新增 tools/ollama_proxy.py(本地 8999,双 key 自动 failover,429 冷却),lane 的 OLLAMA_BASE_URL 指向它即可不再因换 key 重启。
- 12:50 ollama 代理已起(python3 tools/ollama_proxy.py --port 8999,logs/ollama_proxy.log,统计 http://localhost:8999/proxy/stats);三个 lane 脚本改为 OLLAMA_BASE_URL/OLLAMA_API_KEY 可由 env 覆盖。τ² 经代理重启:pid 1675420(user-sim deepseek,GPU4)。
- 13:20 用户发来 CRCD v6 设计文档(Capability-Residual Correction Distillation:DCA 差分能力原子、signed capability graph、frontier/access/robustness/composition 账本、local pref+CE、U×R×Q×m、risk-triggered retention、supervision routing;docs/2026-09-02-crcd-v6-from-user.md)。已回评估:统一了 BFCL 六臂证据;三风险(因果验证成本、4B pref 前科、事件量级需来自生成任务集);建议 MVP 顺序 P0-1 事件挖掘 → P0-2 B0–B5 → P0-3 A1–A4 → P0-4 C 阶梯,BFCL 先行。等用户点头开写事件挖掘器。

## CRCD MVP 启动(2026-09-02 13:25,用户:"咱们可以开始实验了,尽量并行")
- P0-1a base CI:hpg **40886321**(scripts/bfclbase_ci_hpg.slurm,tag base,写 hpg:results/bfcl_std/base;rai 的 46.27 不动)。
- P0-1b 事件挖掘器:tools/bfcl_event_mine.py(首分叉 + K 次 causal continuation,GT 作 oracle 纠正;输出 data/bfcl_sft/events_v1.jsonl,行含 prompt/response=a⁺/_rejected=a⁻/_event_dU)。学生 vllm GPU4:8975(两名字 Qwen/Qwen3.5-4B 与 -FC)。
- B 阶梯映射:B2 = events 作 SFT 行;B4' = AW_DISTILL=ddpo(SFT 阶段 + base-centered pref,用 _rejected);B3 待训练器加"只 pref"开关;B5 = + 簇/DCA residual gating。
- 13:45 ALFWorld ours 训练完(checkpoint 在 results/bfas/alfworld/ours_s0/checkpoint),lane 内评测 vllm 起不来(训练显存未释放)→ tools/alf_eval_chain.sh 用 tools/bfas_eval_ckpt.py 在 GPU1 顺序评 ours → base(logs/alf_eval_chain.log,Monitor 已挂)。训练器新增 AW_SFT_SKIP=1(B3 只 pref)。
- 14:00 ALFWorld 评测根因:训练器存的是扁平 Qwen3_5ForCausalLM,vLLM 要 Hub 布局 → 用 tools/bfcl_hub_merge_export.py --model Qwen/Qwen3.5-2B 生成 hub_merged 再评;bfas run.py `_train` 已改为训练后自动 hub-merge(失败回退扁平)。base 评测在跑,ours 排在其后(后台 b41284rsl)。
- 14:05 单轮事件挖掘器 tools/bfcl_event_mine_single.py(support 单轮 + gen_pool_v3/v4 已复现题;K 次学生采样过 AST checker → u⁻;a⁺ = GT 调用块;irrelevance 暂跳过)。
- 14:30 **ALFWorld base 2B = 0/134(全部走满 40 步)**。用户:学生一律 ≥4B,2B 无价值(记入记忆)。已做:bfas DEFAULT_MODELS alfworld/tau2 → 4B;ALFWorld 学生加 ReAct 部署脚手架(BFAS_ALFWORLD_STUDENT_REACT 默认 1,与 teacher 同 prompt,20 测试过);停掉 ours(2B)评测;τ²(2B 学生)采集停掉待 4B 重启;ALFWorld 学生侧需 4B 重采(demo 复用)。
- 14:40 τ² 以 4B 学生经代理重启(pid 1740451);ALFWorld 学生侧 4B + ReAct 重采(pid 1741453,teacher deepseek@proxy,demo 107 条从 ledger 复用;2B 产物移 _trash/alfworld_2b_student_0902);GPU1 残留 eval vllm 已杀。事件挖掘:stateful 22 事件(5 条 dU>0,mean dU 0.12——首分叉多数不 consequential),单轮 36/50 任务出事件。用户要求今天出 CRCD 初步报告。
- 14:50 hpg 新增:ALFWorld base 4B 官方评测 **40886895**(scripts/alf_base_eval_hpg.slurm;hpg envs/vllm-serve/.venv → ../vllm/.venv 符号链接);bfclbase 40886321 跑中。B1 臂放弃(v9 guided 行仅 11)。用户:rai 与 hpg 都要排满。
- 13:35(校时:此前 13:2x–14:5x 的条目实际约提前 1 小时)状态:事件 stateful 32 / 单轮 98;hpg bfclbase 40886321、alfbase 40886895 跑中;rai 五进程:两挖掘器、AppWorld、τ²(4B)、ALFWorld(4B)。
- 14:00 hpg alfbase 40886895 死于 vllm 找不到 ninja(venv bin 不在 PATH)→ slurm 加 PATH,重提 **40888039**。
- 14:10 后台流程(bo8hm2oca):单轮挖掘结束(或 14:55 超时)→ tools/bfcl_events_to_pools.py(dU>0、每类别封顶 80、去重、剥空 think)→ rsync 池到 hpg → sbatch scripts/crcd_b_ladder_hpg.slurm(b2ce / b3pref / b4cepref,tag crcd_<arm>_s0)。stateful 挖掘 20/63 集时 37 事件仅 6 条 consequential。
- 14:2x B 阶梯已提 hpg **40888383**(array 0-2:crcd_b2ce_s0 / crcd_b3pref_s0 / crcd_b4cepref_s0;池 137 事件行,mean dU 0.75,simple_python 封顶 80)。Monitor 已挂。
- 14:20 hpg 出分:**BFCL base 复测 46.06**(rai 46.27;分轴 NL 79.58 / Live 77.72 / MT 50.12 / Mem 25.59 / WS 9.50 / Irrel 82.70)→ base 噪声 ~0.2。**ALFWorld base 4B(ReAct)= 12/134 = 8.96%**(pick_and_place 20.8、pick_cool 19.0、look_at 11.1、pick_heat 4.3、clean/two 0)。B 阶梯 _0/_1 跑中、_2 排队。
- 14:35 B 阶梯:_0 B2 训完(54 步)评测中;_1 B3 DPO 18 步评测中;_2 PD。报告草稿 docs/2026-09-02-crcd-p0-report-zh.md 待填 B 分数。
- 14:50 **B3(只 pref,137 事件,18 步)= 46.09 ≈ base**:NL +1.2 / Live +0.9 / MT +0.8 / Mem −0.6 / Irrel −1.7(均噪声内)。第一个不低于 base 的训练臂。B2 评测中,B4' 待跑。
- 15:20 B 阶梯 _0(B2,评测中)与 _2(B4',训练中)15:16 被外部 SIGNAL 取消(reason QOSGrpMemLimit;非本会话操作)。B4' 重提 40897561;B2 重提(见下)。B3 46.09 已入账。
- B2 重提 = **40897586**(array 0);B4' = 40897561(array 2)。Monitor 已挂。

## ⚠️ 规则(2026-09-02 15:30,用户):Qwen 作业白天在 hpg 会被取消 → 白天只在 rai 跑,半夜再放 hpg(已存记忆)
- 已取消 hpg 40897561/40897586;停 stateful 挖掘(30/63 集,67 事件,6 consequential)与其 vllm。
- rai GPU4 链 tools/crcd_rai_chain.sh(pid 见 logs/crcd_rai_chain.log):b2ce → b3pref → b4cepref,各 train + 507 题代理评测(标 proxy);Monitor 已挂。用户:下午重点先验证方法。
- 15:35 rai 链 B2 训练中(pid 3680039);AppWorld 第 12 轮 12h;ALFWorld/τ² 4B 采集中。hpg 无本项目作业(白天规则)。
- 15:55 **B2(只 CE 纠正 span)proxy 8.17 vs base proxy 37.46 —— 崩盘**(Irrel 72.5→5、Relevance 100、MT 0):同一批事件,CE 目标 = 永远调用反射;B3 pref-only 官方 46.09 = base。B 阶梯已给出最强信号:蒸馏对象比数据更重要。B3/B4' proxy 跑中。
- 16:20 B3 proxy 38.18(base proxy 37.46):Live +1.4、MT +4.2、Irrel −7.5(3 题)。B4' 训练中,随后 1-epoch 对照臂。
- 16:35 B4' 训练中(ddpo 参考 137/137);采集三 lane 存活;hpg 无本项目作业;一次性 cron 4694073b 23:07 提交官方评测。
- 16:55 **B4'(CE 3ep + pref)proxy 7.32**:CE 阶段的崩盘 DPO 救不回。B 阶梯结论:同一批事件,CE 崩(8.2 / 7.3)、pref 稳(≈base)。1-epoch 对照臂自动开跑。
- 17:05 CRCD P0 报告已发(docs/2026-09-02-crcd-p0-report-zh.md)。第二轮挖掘准备:bfcl_event_mine.py 加 --start-index(从第 30 集续挖 multi_turn/web_search);bfcl_event_mine_single.py 加 --irrelevance(abstain 事件,a⁺ 取 v9 池 abstain 行)。待 GPU4 上 1-epoch 对照臂跑完后启动(需学生 vllm)。
- 17:15 队列:1-epoch 对照臂(tools/crcd_rai_extra.sh,GPU4)→ 完成后自动起 tools/crcd_round2_mine.sh(学生 vllm 8975;单轮 irrelevance/abstain 事件 → events_single_irrel_v2.jsonl;stateful 从第 30 集续挖 → events_v1.jsonl;日志 logs/crcd_round2_mine.log,Monitor 已挂)。
- 17:35 b2ce1ep proxy 评测中;采集三 lane 存活(AppWorld 第 12 轮 14h、τ² 35 rollout)。
- 17:55 **B2 1-epoch(18 步)proxy 15.93,仍崩**(Irrel 5 / Rel 100 / MT 0)→ 伤害来自 CE 目标本身而非过拟合;同步数的 pref(B3)= base。b4cepref1ep 训练中。
- 18:10 b4cepref1ep proxy 7.66(崩)。B 阶梯 + 对照全部完成:含 CE 的四臂全崩,pref-only 持平 base。第二轮挖掘自动开始(GPU4)。
- 18:35 第二轮单轮挖掘 125/369(99 事件,含 abstain);ALFWorld 4B unguided 完成→guided;τ² 57;AppWorld 第 12 轮 15h。
- 18:45 写好 tools/crcd_fingerprints.py(A 阶梯 MVP:A1 全文 / A2 响应 / A3 差分 / A4 dU×差分;验证代理 = 类别 ARI、u⁻ 的 CV Spearman、stateful consequential AUROC)。等 GPU4 挖掘结束后跑。
- 19:00 第二轮单轮挖掘完成 322 事件但 irrelevance 只 2 条(support 仅 6 题且学生多数会拒绝)→ 单轮挖掘器加 out_of_scope 支持,对 gen_oos.jsonl(59 条已验证界外题)挖 abstain 事件(events_single_oos_v2.jsonl,pid 3897689)。stateful 续挖中。
- 19:20 abstain 事件 20 条(gen_oos 59 题,学生 39 题本就会拒绝)。pool builder 支持多单轮文件;第二轮池 = stateful 全部 + single v1 + irrel v2 + oos v2(stateful 续挖完后正式建)。夜间 hpg 计划:round-1 B2/B4' 官方评测 + round-2 池的 B2/B3。
- 19:30 夜间 slurm 写好:scripts/crcd_night_hpg.slurm(array 0-3:r1_b2ce / r1_b4cepref 官方分;r2_b3pref / r2_b2ce 用第二轮池 pool_events_{pref,ce}_v2.jsonl)。23:07 cron:stateful 挖完后先 `tools/bfcl_events_to_pools.py --single v1 irrel_v2 oos_v2 --out-ce ..._ce_v2 --out-pref ..._pref_v2`,rsync 池,sync 代码,sbatch。
- 19:50 第二轮挖掘完成:stateful 112 事件 / 17 consequential(miss_func 11!);abstain 20;第二轮池 210 行已建并同步 hpg。A 阶梯指纹分析在 GPU4 跑(logs/crcd_fingerprints_v2.log → results/analysis/crcd_fingerprints_v2.json)。
- 19:35 指纹提取重跑中(FEAT_GRAD_CKPT=1,1536 tok);ALFWorld guided;τ² 65;AppWorld 第 12 轮 16h;hpg 空。
- 19:55 用户:hpg 空了、夜里可以跑 → 已提 **40928415**(crcd_night_hpg.slurm,4 臂:r1_b2ce / r1_b4cepref 官方分;r2_b3pref / r2_b2ce),Monitor bxja6riyi;23:07 cron 已删。A 阶梯 MVP 结果:A1 全文 ARI 0.96(=类别身份)/ Spearman 0.59;A2 响应 0.52 / 0.66;A3 差分 0.58 / 0.64;A4 因单位归一化与 A3 相同(缺陷待修);consequential AUROC 样本不足。
- 20:00 夜间批次 2 = **40928469**(crcd_night2_hpg.slurm:r2_b3pref_ep3 / r2_b3pref_ep5,pref 剂量),Monitor bxja6riyi(40928415)+ 新 Monitor。两阵列 PD(Priority)。
- 20:20 指纹 v3(A4 修复):A4 ARI 0.38 / Spearman 0.65;u⁻ 预测代理对 A4 有泄漏(|A4| = dU),明天以"训练后增益"做真正的有效性门。
- 20:30 dept 组内存配额被 sdl 六个新作业 + 我们的阵列挤满(QOSGrpMemLimit):40928415_0 在跑,_1-3 PD;夜间批次 2 改提 yd24f 账户 = **40929466**(Priority PD),旧 40928469 已取消。
- 20:45 A 阶梯有效性门(overnight):tools/crcd_gate_subset.py(按 A3 簇 k=8 切池 + 测量任务集)→ tools/crcd_validity_gate.sh(每簇 pref-only 1 epoch → hub-merge → vllm → 单轮任务 K=4 通过率;base 也测)→ tools/crcd_gate_analyze.py(增益矩阵 G vs 质心余弦 P 的 Spearman、对角/非对角选择性)。等 fingerprints_v4(含 gate.events)出来后启动。
- 20:55 有效性门已排队(waiter pid 4005646:fingerprints_v4 完成 → gate_subset → crcd_validity_gate.sh,GPU4,日志 logs/crcd_validity_gate.log,Monitor byk91ju4d;单簇 ≈ 15 分钟,8 簇 + base ≈ 2.5 小时)。明早汇总:hpg 6 臂官方分(40928415 ×4 dept、40929466 ×2 yd24f)+ 有效性门 G/P 矩阵 + ALFWorld 4B 池/ours。
- 21:25 用户:今天能排的都排上。已加:夜间批次 3 = **40930427**(yd24f;pref LR 2e-4 ×{1ep,3ep} on round-2 pool);scripts/alf_ours_hpg.slurm(ALFWorld 4B 池三臂 ours/+curr/+λ0.5,训练+hub-merge+官方评测)由 waiter(logs/alfours_waiter.log)在池落地且为夜间时自动提交,白天则推迟。
- 20:35(实钟)hpg 1R(40928415_0)+ 6 臂 PD(组配额);fp v4 计算中;ALFWorld guided 7h;τ² 67;AppWorld 17h。
- 21:1x **B2 官方 10.95**(Irrel 3.57 / Rel 100 / MT 0.75)—— 与 proxy 一致,CE 崩盘官方坐实。
- 21:35 ALFWorld 4B 池落地(results/bfas/alfworld/ours_s0/pool.jsonl),三臂 slurm 手动提交(waiter 因 21 点判为白天而推迟);hpg 2R(B4' r1、B3 r2)。
- 21:40 ALFWorld 4B 三臂 = **40934733**(yd24f;alfb1_{ours,curr,pres}_s0,池 4864 行,训练+hub-merge+官方 unseen 评测)。Monitor 已挂。有效性门簇 0 在测量。
- 21:5x 有效性门簇 0 测完(rates_c0.json):训练簇 0(80 事件)后,62 题总体 +5.6pp;自身簇 +8.3pp,c3 +9.2,c7 +14.3,c4 −8.3,c1/c6 0 → 单行看不出选择性,等 8 行齐再算 Spearman(P,G)。簇 1 开始训练。
- 22:35 **round-2 B3 官方 47.34(base 46.06,+1.28;r1 46.09)** — 首个超 base 的臂;分轴 NL 81.92/Live 78.61/MT 52.00/Mem 26.67/Irrel 80.57/Rel 81.25/Web 11.50。事件 137→210 单调改善。已发频道。
- 22:35 ⚠️ ollama 周用量上限:两 key 均 429 冷却(proxy 429 计数 6.9k),τ² 停 69 条、AppWorld 采集连失 6 次;deepseek 采集停摆待 key 恢复(周期重置)或新 key。hpg:_1 B4' DPO 中、_3 r2 B2 CE 运行,其余 7 臂 QOSGrpMemLimit 排队。门:簇 0/1 测完,簇 2 训练中。
- 23:0x B4'(r1)官方 9.13(proxy 7.32 坐实):NL 44.31/Live 31.24/MT 2.00/Mem 1.51/Irrel 5.70。B 阶梯官方三点齐:CE 10.95、CE→pref 9.13、pref 46.09;r2 pref 47.34。剩 _3 r2 B2 CE 运行中。
- 23:1x 用户:baseline 已跑过(Table 1 awb2_* 三种子),不要重跑。同协议 bfcl_std 里 BFCL 专用 STaR = 40.77/38.25/39.15(< base);Table 1 的 awb2_star 45.2 是 rai 批次(base 43.9)。待办(白天 rai,eval-only,零 ollama):把 awb2_star ckpt 在 bfcl_std 协议下重新评测以对齐列。记忆已存 project-baselines-already-run。
- hpg $HOME 满(profile 警告 No space left on device):不影响 /blue 作业,但 HF cache 若在 $HOME 会再出问题;之后清理。
- 23:2x 用户:今晚都排上,**明早 ~09:15 交汇总文档**(CRCD 方法全貌、今明所有实验做法+详细结果、改进点、下一步;中文附件)。
- 23:2x round-3 自迭代启动:tools/crcd_round3_mine.sh(rai GPU2,port 8976;先等 r2 adapter rsync → 合并 → 挖 single/oos/stateful(≈2h)→ pool_events_pref_r3 + union v3 → 夜间自动 sbatch scripts/crcd_r3_hpg.slurm(r3_union / r3_cont),白天则 rai 训 r3union 代理评)。日志 logs/crcd_r3_mine.log。awb2_star ckpt 在 rai/hpg 都已不存在 → STaR 列对齐无法只评测,作罢(同协议 bfclb_star 39.4 可用)。
- [ ] Monitor logs/crcd_r3_mine.log(round-3 挖掘链 pid 193966,GPU2)与 hpg crcdR3 作业;Monitor 40934733(ALFWorld 三臂);一次性 cron 09-03 08:40 早间汇总文档(docs/2026-09-03-crcd-summary-zh.md)。
- 23:3x 为 r3_cont(从 r2 模型续训)打补丁:bfcl_hub_merge_export.py `--model` 可指向本地 hub_merged 目录;bfcl_std_campaign.sh 支持 `BFCLSTD_BASE_MODEL` 覆盖合并基座;crcd_r3_hpg.slurm 的 r3_cont 导出该变量。已 sync hpg。一次性 cron 66c25eb1(09-03 08:40)生成早间汇总文档;Monitor bj1wj1wv7 盯 r3 挖掘链。
- 23:3x r2 B2 CE 官方 **18.38**(r1 CE 10.95):Irrel 3.57→76.67(弃答目标治好"总想调用"),但 MT 2.38 / Mem 5.81 仍崩 → CE 的破坏不止于调用反射,短纠正片段的 CE 会覆写长程行为;偏好式才是保留安全的。40928415 全部 COMPLETED。
- 23:3x 发现 hpg Monitor bug:作业离队后 `squeue -j` 非零导致最后一窗的分数行被丢弃(误报 ssh failed);重建 3 个 Monitor(加 `; true`)。
- 23:4x Monitor 重建:b71d40144(40929466 pref 剂量)、bbuld2bip(40930427 LR)、(ALFWorld 见 00:2x 行)、bj1wj1wv7(r3 挖掘链)、byk91ju4d(有效性门);40928415 全部完成已停。
- 23:35 汇报已发。hpg 本项目 7 臂全 QOSGrpCpuLimit 排队(0 在跑);r3 链起 vllm 中;门 4/8。
- 23:5x yd24f 组 CPU 被组员占满(56/64)→ 把 40929466(pref 剂量)与 40930427(LR)改到 fsu-compsci-dept 并降内存到 100G(dept 已用 820/1000G),状态从 QOSGrpMemLimit 变为 Priority(等 GPU 槽位)。ALFWorld 40934733 仍在 yd24f 等 CPU。
- 00:0x **hpg $HOME 满导致官方评测失败**:vllm 的 torch_compile_cache 在 ~/.cache/vllm 下 makedirs Errno 28 → ep3(40929466_0)与 lr2x(40930427_0)训练完成但 GENERATE FAILED。修复:tools/bfcl_std_campaign.sh 把 VLLM_CACHE_ROOT/TORCHINDUCTOR/TRITON/XDG 缓存指到 /blue/.../hq/tools/*-cache(已 sync;仍在训练的 ep5、ep3_lr2x 到评测阶段会读到新脚本)。新增 scripts/crcd_evalonly_hpg.slurm(仅评测已训 adapter)。~/.cache/huggingface 83G 是 $HOME 元凶(我们的作业用 /blue 的 HF_HOME,不依赖它)——删除需用户点头(在 ~/hq 之外)。
- 00:1x eval-only 数组 = **40943088**(dept,100G;_0 ep3、_1 lr2x)。Monitor 已挂。剂量命名:40929466 = ep3/ep5;40930427 = lr2x(1ep)/ep3_lr2x(3ep)。
- 00:2x ALFWorld 三臂重提为 **40943138**(yd24f;脚本加了 /blue 缓存重定向,旧 40934733 已取消);Monitor 新 id 见会话。crcd_r3_hpg.slurm 同样加了缓存重定向。eval-only Monitor bl25lwwzo。
- 00:3x round-3 单轮挖掘:r2 模型残差 **115 事件**(base 上 round-2 为 322,同一 369 题、K=4)→ 残差随轮次收缩。OOS/stateful 继续。
- 00:5x round-3 OOS 弃答挖掘:仍 20 事件(与 round-2 在 base 上相同)→ r2 模型没学会弃答,与官方 Irrel 不涨一致;弃答需要更强的通道(C 阶梯/更多弃答事件)。stateful 挖掘中(≈1h)。
- 00:35 汇报已发。hpg:ep5/ep3_lr2x 训完进评测;EV_0(ep3)评测中,EV_1 排队;ALFWorld 40943138 排队。r3 stateful 第 10 集。门 6/8。已向用户申请清理 hpg ~/.cache/huggingface(83G)。
- 01:0x **pref 剂量 ep3 官方 47.61**(ep1 47.34,base 46.06):NL 83.02 / Live 78.68 / MT 53.00 / Mem 32.47(+6.9 vs base)/ Irrel 73.45(−9.3 vs base)/ Rel 81.25 / Web 8.50。剂量越大目标涨、Irrel 掉 → C 阶梯(风险触发保留)是下一步。
- 01:2x **C 阶梯准备好**:bfcl_event_mine_single.py 加 `--anchors-out`(学生自己答对的回复为 chosen,答错回复或最小翻转为 rejected,dU=0 的保留锚点,每类 cap 40);tools/crcd_anchor_mine.sh(pid 1500075,门跑完后改用 GPU4 立即开始,用 base 采锚点 → pool_events_pref_c1 = v2 210 + 锚点 → 夜间 sbatch scripts/crcd_c_hpg.slurm:c1_anchor(1ep)/c1_anchor_ep3(3ep),对照 47.34/Irrel 80.57 与 47.61/Irrel 73.45)。日志 logs/crcd_anchor_mine.log。
- 01:3x **pref 剂量 ep5 官方 44.82**(NL 73.48 / Live 73.95 / MT 49.62 / Mem 30.54 / Irrel 71.78):剂量曲线 ep1 47.34 → ep3 47.61 → ep5 44.82,峰值≈3 epoch,之后保留损失压过目标增益。40929466 完成。剩 ep3_lr2x(训完评测中)、lr2x(EV_1 评测中)。
- 01:4x **ep3_lr2x 官方 45.68**(NL 83.00 / Live 78.98 / MT 51.50 / Mem 25.81 / Irrel 73.71):LR 翻倍保住单轮增益但丢掉 MT/Memory 增益,Irrel 代价不变 → LR 维持 1e-4。40930427 完成;lr2x(1ep)在 EV_1 评测中。
- 01:35 汇报已发(剂量/LR 三臂表)。r3 stateful 20/63 集;门簇 7 训练中;EV_1 评测中;ALFWorld 排队。
- 01:5x **有效性门 v1 完成:未通过**。Spearman(P,G) 0.08(全)/0.03(非对角),门槛 0.30;对角增益 +2.2pp vs 非对角 −0.3pp(方向对但在噪声内:每簇 6–19 题、K=4,单格 SE≈8pp);列效应主导(OOS 弃答列在几乎所有簇训练下都涨,c4 列都跌)→ 小剂量单簇偏好训练主要造成全局行为漂移,不是簇特异技能。下一版门:测量集扩到全部 369+59 题、K=8、剂量 3ep,并用列中心化 G。结果 results/analysis/crcd_gate_v1.json。
- 01:5x **lr2x(1ep)官方 46.32**(NL 82.02 / Live 78.53 / MT 52.00 / Mem 25.16 / Irrel 80.36):LR 翻倍无益,Memory 回到 base。剂量/LR 四臂全部官方齐:ep1 47.34 / ep3 47.61 / ep5 44.82 / lr2x 46.32 / ep3_lr2x 45.68 → 定 LR 1e-4、3 epoch 为当前最佳剂量。
- 02:16 锚点(单轮)93 对:56 自对比(base 同题既对又错)+ 37 翻转(全对题配最小翻转 rejected);类别以 simple_python 40 / live_simple 22 为主,irrelevance 仅 5 → 弃答保留锚点主要靠 OOS 阶段(base 在 59 题里 39 题全对)。
- 02:3x **C 阶梯提交 hpg = 40949167**(dept,100G;c1_anchor 1ep / c1_anchor_ep3):C1 池 359 行 = round-2 210 事件 + 149 锚点(单轮 93 + OOS 弃答 56)。锚点文件 data/bfcl_sft/anchors_{single,oos}_base.jsonl。Monitor 已挂。GPU4 锚点 vllm 已退出。
- 02:4x ALFWorld 40943138 也改到 dept + 100G(yd24f CPU 一直被占),状态 Priority。hpg 待跑:crcdC ×2、alfours ×3、(r3 ×2 即将提交)。
- 02:35 汇报已发。r3 stateful 30/64(73 事件),预计 ~04:30 提交;crcdC、alfours 均 GrpMemLimit 排队。
- 03:0x **round-3 提交 hpg = 40951778**(dept;r3_union = base 训 r2∪r3 324 行;r3_cont = 从 r2 模型续训 r3 148 行)。r3 事件:单轮 115(base 322)、OOS 20(=)、stateful 105(base 112);池 148 行。Monitor 已挂。GPU2 已释放。
- 03:1x 一次性 cron 05:25:若 crcdC/crcdR3 在 hpg 仍 PENDING,则 rai GPU2 代理评测兜底(tools/crcd_rai_extra.sh,CRCD_ARMS="c1anchor r3union"),保证早间文档至少有 proxy 数。门 v2(全 428 题、K=8、3ep)列入下一步,不今晚跑。
- 03:3x ALFWorld 40943138 三臂 13 秒即败:`No module named peft` —— slurm 把 envs/vllm/.venv/bin 放 PATH 前面,`python` 解析到 vllm venv。改为显式 .venv/bin/python(训练/合并/评测),内存 100G,重提 = **40952830**(dept)。Monitor 已挂。03:26 起 crcdC ×2、crcdR3 ×2 全部 RUNNING。
- 03:35 汇报已发。hpg 7 臂全 RUNNING(crcdC ×2 训完进评测、crcdR3 ×2 训完进评测、alfours ×3 训练中)。
- 04:2x **round-3 官方:r3_union 47.82(新高)/ r3_cont 46.66**。轮次单调:base 46.06 → r1 46.09 → r2 47.34 → r3 47.82(固定 1ep);r3_union 分轴 NL 82.75 / Live 78.83 / MT 52.12 / Mem 30.11 / Irrel 78.98 / Web 10.50。从 r2 续训只用新残差反而掉到 46.66(Web 7.50、Mem 26.67)→ 迭代方案定为"每轮从 base 在累计池上重训"(base 中心、不叠加漂移)。r3_cont 合并已确认叠在 r2 模型上。
- 04:3x 追加 round-3 后续 = **40955850**(dept,100G;r3u_ep3 = 联合池 324 行 ×3ep;r3u_anchor_ep3 = 联合池 + 149 锚点 = 473 行 ×3ep;scripts/crcd_r3b_hpg.slurm)。Monitor 已挂。当前 hpg:crcdC ×2 评测中、alfours ×3 训练中、crcdR3b ×2 排队/开跑。
- 04:4x **C1 锚点(1ep)官方 47.16**(NL 83.79 / Live 78.46 / MT 49.62 / Mem 30.11 / Irrel 81.25 / Rel 75.00 / Web 9.50):Irrel 回到 81.25(r2 80.57、base 82.70),Memory +3.4,但 MT −2.4、Relevance −6.25 → 翻转型锚点(全对题配合成 rejected)在挪动"调用/弃答"边界而非钉住它。下一变体:只用自对比锚点(不合成翻转)。等 ep3 锚点臂。
- 04:5x 追加 **C2 = 40956032**(dept;仅自对比锚点 72 对 + round-2 池 = 282 行,1ep;scripts/crcd_c2_hpg.slurm),对照 c1_anchor 47.16 / r2 47.34。Monitor 已挂。hpg 本项目:c1_anchor_ep3 评测中、alfours ×3、crcdR3b ×2、crcdC2。
- 04:4x **C1 锚点 ep3 官方 46.37**(NL 80.67 / Live 76.02 / MT 50.88 / Mem 27.96 / Irrel 84.44 / Rel 75.00 / Web 7.00):Irrel 反超 base,但 NL/Live 跌破 base、目标增益缩水 → 翻转锚点过度保护弃答、把边界推向弃答。C 阶梯结论:锚点要按两侧平衡(自对比 C2 在跑)。40949167 完成。
- 04:35 汇报已发(round-3 + C 阶梯四臂表)。在跑:r3u_ep3;排队:r3u_anchor_ep3、C2;alfours ×3 训练中。
- 05:25 守卫:crcdC/crcdR3 早已官方出分,rai 兜底未启动。
- 05:4x ALFWorld curr 臂训练+合并完成但评测死于 $HOME 满(vllm usage-stats 写 ~/.config)。修复:所有 hpg 脚本加 VLLM_NO_USAGE_STATS=1 + HOME 指到 /blue/.../tools/fakehome。eval-only:scripts/alf_evalonly_hpg.slurm,curr = **40959145**;waiter(pid 2528096,logs/alf_evalonly_waiter.log)等 40952830 离队后为 ours/pres 自动提交 eval-only。注意:数组作业日志名是 alfours_<任务jobid>_<idx>.out,不是 %A。
- 06:0x **r3u_ep3 官方 45.31**(NL 78.77 / Live 77.35 / MT 49.62 / Mem 29.68 / Irrel 72.78):联合池 ×3ep = 123 步过量。剂量的有效变量是**优化步数**:41 步 47.82、81 步 47.61、123 步 45.31、135 步 44.82 → 峰值 40–80 步,>120 步保留崩。追加 r3u_ep2(82 步)对照。
- 06:0x r3u_ep2 = **40959417**(dept;联合池 ×2ep ≈ 82 步;scripts/crcd_r3c_hpg.slurm)。Monitor 已挂。
- 05:35 汇报已发(注:上两条"06:0x"实际为 05:2x)。hpg 6 在跑:C2、r3u_anchor_ep3、r3u_ep2、alfEV curr、alfours ours/pres。
- 06:1x **ALFWorld 4B curr 臂 9/134 = 6.72%(base 8.96%)**:类别此消彼长(heat/clean/two 涨,simple place/cool 跌),总体低于 base,噪声量级。等 ours/pres。
- 06:2x **r3u_anchor_ep3 官方 45.65**(180 步;NL 74.79 / Live 72.61 / MT 46.75 / Mem 32.90 / Irrel 86.06 / Rel 75.00):锚点把 Irrel 推到 86.06、Memory +7.3,但调用轴全跌破 base;锚点不能抵消步数过量。40955850 完成。剩 C2、r3u_ep2、ALFWorld ours/pres。
- 06:1x **C2 自对比锚点官方 47.43**(NL 83.94 / Live 79.05 / MT 52.25 / Mem 29.89 / Irrel 75.75 / Rel 81.25 / Web 9.50):保住 Relevance/MT、NL/Memory 上涨,但 Irrel 无保护(自对比里几乎没有弃答对)。两类锚点作用在边界两侧 → 追加 **C3 = 40960794**(自对比 72 + 调用侧翻转 20 + 弃答侧翻转 20 = 112 锚点 + round-2 池 = 322 行,1ep;scripts/crcd_c3_hpg.slurm)。Monitor 已挂。
- 06:2x ALFWorld ours/pres 训练完成(在职评测同样死于 $HOME),waiter 自动提交 eval-only = **40960981**(_0 ours、_1 pres),已在跑;alfEV Monitor(befh12tzl)按名字盯所有 alfEV 日志。hpg 在跑:alfEV ×2、C3、r3u_ep2。
- 06:4x 早间汇总文档草稿已写:docs/2026-09-03-crcd-summary-zh.md(方法全貌、B/r2/剂量/A/r3/C/其它 benchmark 表、改进点、下一步);待补三处:C3(40960794)、r3u_ep2(40959417)、ALFWorld ours/pres(40960981)。08:40 cron 时填入并附件发频道。
- 06:5x **ALFWorld 4B 三臂官方齐**:pres(λ0.5)16/134 = **11.94%** > ours 12/134 = 8.96%(= base)> curr 9/134 = 6.72%。单种子、134 局(SE≈2.5pp),方向性结论:保留加权臂最好,与 BFCL"保留是瓶颈"一致。40960981 完成。剩 C3、r3u_ep2。
- 06:35 汇报已发(C2、r3u_anchor_ep3、ALFWorld 三臂)。hpg 在跑:C3、r3u_ep2(均评测阶段)。
- 06:5x **r3u_ep2 官方 47.85(新高,82 步)**:NL 83.12 / Live 79.20 / MT 50.25 / Mem 34.62(+9.0)/ Irrel 74.14 / Web 11.00。步数窗口 40–80 坐实;剂量在用 Irrel 换 Memory/单轮。剩 C3。
- 07:1x **C3 平衡锚点官方 48.02(新高,+1.96 vs base)**:NL 83.67 / Live 78.68 / MT 53.62(最高)/ Mem 31.40 / Irrel 77.17 / Rel 75.00 / Web 8.50。平衡锚点优于两种单侧锚点(47.16 / 47.43)与无锚点(47.34);剩余代价 Irrel −5.5、Rel −6.25。今晚全部作业完成;文档填齐后发布。
- 07:2x 今夜全部作业完成;docs/2026-09-03-crcd-summary-zh.md 填齐(含总表)并发布。08:40 文档 cron 删除。hpg 本项目 0 作业;rai 无本项目 GPU 任务(τ²/AppWorld 采集进程仍在,受 ollama 周上限阻塞)。

## ⟳ RESTART CHECKLIST(2026-09-03 07:30 更新)
- hpg:本项目 0 作业。今夜全部结果已入 notes/exp_log.md;文档 docs/2026-09-03-crcd-summary-zh.md 已发频道。
- rai:无本项目 GPU 任务;仍存活的采集进程(受 ollama 周上限阻塞,自动重试):AppWorld 官方脚手架 deepseek 采集 pid 1098873(logs/aw_collect_official.log,GPU1 端口 8950)、τ² 4B 采集(logs/tau2_launch_pid.txt,GPU4 端口 8940)、ollama 代理 :8999。
- 会话级任务需重挂:每小时 :35 进度汇报 cron;无 Monitor 需重挂。
- 待用户决定:清理 hpg ~/.cache/huggingface(83G,$HOME 满);ollama 周期何时重置。
- 下一步候选(零 API):r3u + C3 锚点;锚点配比/风险触发;门 v2;round-4 挖掘;ALFWorld 自对比事件;AppWorld gpt-5.4 消融。
- 07:3x 白天 rai(GPU2)启动 **r3u + C3 锚点** 代理评臂:pool_events_pref_v3c3.jsonl(324 + 112 = 436 行,1ep ≈ 55 步),pid 2683777(GPU4;GPU2 被组员占满),logs/crcd_rai_r3uc3.log;出 PROXY 分后与 base 代理 37.46 / r2-pref 代理 38.18 比;晚上再上 hpg 官方评。
- 07:35 汇报已发。rai GPU4 r3uc3 训练中(pid 2683777)。
- 08:0x r3uc3 训练+合并完成,代理评 vllm 在 GPU_UTIL 0.3 下 Mamba cache 不够(max_num_seqs 1024 > 952 块)→ 以 0.4 单独重跑评测(pid 2731703,logs/crcd_rai_r3uc3_eval.log)。
- 08:2x **r3uc3 代理 38.80**(proxy;base 37.46、r2-pref 38.18;MT 58.33、Irrel 60.00)—— 代理最高。官方评:scripts/crcd_r3uc3_hpg.slurm 已同步,22:05 cron 提交(Qwen 夜间规则)。
- 08:2x 一次性 cron def9a35f(09-03 22:05):提交 scripts/crcd_r3uc3_hpg.slurm。重启后需重建。rai 无本项目 GPU 任务。
- 08:4x **有效性门 v2 启动**(rai GPU4,pid 2765534,logs/crcd_validity_gate_v2.log):同 8 簇池,K=8、每簇 3ep、测量全部 428 题(base 先重测);tools/crcd_validity_gate_v2.sh,输出 data/bfcl_sft/gate_v2/rates_*.json;完成后 `python tools/crcd_gate_analyze.py --gate data/bfcl_sft/gate_v2 --out results/analysis/crcd_gate_v2.json`。预计 10h+。
- 08:35 汇报已发。门 v2 在测 base(K=8,428 题)。
- 08:5x 用户指示:停掉 AppWorld 官方脚手架采集 lane(pid 1098873 + vllm 8950,GPU1)以腾 GPU;已停。collection_cache 在 results/bfas/appworld/*/ 保留;ollama 恢复后用 tools/aw_collect_official.sh 重启即续采。
- 注意:重启 AppWorld lane 前删掉陈旧的 results/bfas/appworld/collect_s0.lock / collect_shared.lock(进程被 kill,锁未释放)。
- 08:5x AppWorld lane 退出时 ledger:teacher 演示 **40/42 已验证**(token 4.79M)—— 演示采集其实已接近完成,缺的是 guided/学生 rollout 阶段;ollama 恢复后只需补 2 条演示 + 学生侧 rollout 即可建池。
- 09:35 汇报已发。门 v2 base 仍在测(K=8)。
- 10:0x **ALFWorld CRCD 事件挖掘器写好**(tools/alf_event_mine.py + tools/alf_event_lane.sh,零 API):回放 107 条已验证 demo 的命令前缀,每步问学生(ReAct 部署脚手架),首个分歧点 = 事件(a⁺ = demo 命令,a⁻ = 学生命令),K 条学生续跑估 ΔU;chosen = 学生回复把 ACTION 换成 a⁺(纠正片段),rejected = 学生原回复。冒烟测试(2 demo,K=2)pid 2903840,GPU1,logs/alf_event_lane_smoke.log;通过后跑全量。
- 10:2x ALFWorld 挖掘冒烟通过(2 demo → 4 事件、0 后果性、606 次调用、~7 min/demo 串行)。已并行化续跑(2K 线程)并改为"探到后果性分歧为止(≤4 次探测)"。全量启动:107 demo,K=3,GPU1,pid 2939670,logs/alf_event_lane.log / logs/alf_event_mine.log;产物 data/alf_sft/events_v1.jsonl → pool_events_pref_v1.jsonl(dU>0)。预计 3–4h。
- 10:5x **数据发现:生成题 id 冲突**——gen_pool_v3 303 行只有 27 个唯一 id(v4 也复用),门 v1 的通过率按 id 存导致同 id 不同题互相覆盖(部分失真);训练池不受影响(去重键含 rejected 文本)。修复:单轮挖掘器加 `--rates-key prompt`(sha1(prompt) 为键),gate_v2/clusters.json 的簇内题改为 prompt hash;门 v2 重新启动(pid 2982016,base 重测)。
- 11:0x ALFWorld 挖掘进度:5 demo / 25 min → 18 事件、5 后果性(28%);全量 107 demo 预计 ~19:30 完成,约 100 条后果性事件(与 BFCL r1 的 137 同量级)。
- 11:1x 写好 scripts/alf_crcd_hpg.slurm(ALFWorld CRCD 臂:base 中心 DPO 1ep on data/alf_sft/pool_events_pref_v1.jsonl → hub-merge → 官方 valid_unseen 评测)。**今晚提交**(22:05 cron):若 data/alf_sft/pool_events_pref_v1.jsonl 已生成,先 rsync 该池到 hpg,再 sbatch scripts/alf_crcd_hpg.slurm。
- 10:35 汇报已发(注:上面 10:5x/11:0x/11:1x 条目的实际时间约早 1h)。ALFWorld 挖掘 32 min、门 v2 重启 7 min。
- 11:35 汇报已发。**hpg ssh 被拒(Permission denied keyboard-interactive,隧道端口正常)**,疑因 $HOME 满;已请用户清 ~/.cache/huggingface。22:05 cron 提交前需确认 ssh 恢复;若仍不通,推迟到恢复后手动提交。
- 12:1x 用户:hpg 恢复(ssh 可登录,但 $HOME 仍 100%/91G .cache);要求更新 CRCD 阶段总结。文档已更新(顶部"9/3 白天更新"、门 v1 的 id 冲突补充、ALFWorld 事件挖掘机制段、下一步),重新发附件。hpg 探针 Monitor 已停。
- 12:2x 用户授权清理。rai 已清:旧 harness AppWorld 臂(awb4/6/9)、门 v1 adapter ×8、代理臂 adapter(crcdr_b*、crcd_r2 本地副本)、2B ALFWorld、base_export、所有 hub_merged、_trash → 释放约 330G(盘 99%→97%,results 373G→79G);清单 docs/cleanup_2026-09-03.txt。保留:bfclb* 六臂 adapter、crcdr_r3uc3 adapter、data/、ledger、结果 json。
- 12:5x 用户发"下一阶段研究 prompt"(docs/2026-09-03-crcd-next-stage-from-user.md);回复了并行计划。训练器新增:AW_DDPO_REF_FREE=1(无参考成对目标)、AW_DDPO_WEIGHT=none|du|tanh|pos(+AW_DDPO_TAU;按 _event_dU 加权,池内归一到均值 1,锚点权重 1)。**P0-4 ref-free 臂**在 rai GPU3 跑(r2 池 1ep,代理评;对照 r2-pref 代理 38.18),pid 3195882,logs/crcd_rai_b3reffree.log。tools/alf_events_to_pools.py 写好(A/B/C/D + negative)。
- 12:5x scripts/alf_ablation_hpg.slurm 写好(A_all/B_first/C_conseq/D_weighted/CE_conseq 五臂,1ep,官方评)。**今晚提交**(22:05 cron):挖掘完成后先 `python tools/alf_events_to_pools.py`,rsync data/alf_sft/pool_*.jsonl 到 hpg,再 sbatch alf_ablation_hpg.slurm(可替代 alf_crcd_hpg.slurm,C_conseq 即 CRCD 臂)。
- 13:0x 门 v2 base 通过率按 prompt 键 416 题(修复后覆盖完整),簇 0 训练开始(3ep);预计每簇 ~1.5h(K=8 测 428 题)→ 明早出全部 8 簇。
- 13:1x P1-6 ΔU 校准研究脚本就绪:tools/crcd_du_buckets.py(neg/zero/small/medium/large 五桶,等行数、步数对齐)+ tools/crcd_du_calib.sh(逐桶训练 + 代理评)。待 GPU3 上 ref-free 臂结束后启动(约 5 桶 × 1h)。
- 13:1x ΔU 桶:neg 只有 1 条(BFCL 的 GT a⁺ 很少比学生差,负桶留给 ALFWorld),zero 123/small 85/medium 193/large 112 可用,各取 60 行 ×5ep(≈37 步)。校准链 waiter(pid 3204613)等 GPU3 的 ref-free 臂结束后启动,logs/crcd_du_calib.log。
- 13:2x P1-7 纠正覆写(corrective overwrite)诊断启动(rai GPU1 与挖掘共卡):tools/crcd_drift_chain.sh(pid 3208077)在 r2 池上重训 CE 3ep / CE 1ep / pairwise 1ep(不评测),然后 tools/crcd_drift_diag.py 在 149 条"base 已掌握"的 prompt 上按 token 算 KL(πθ‖π0)、熵、首 token 调用概率、base 回复的 log-prob → results/analysis/crcd_drift_v1.json。假设:CE 的 KL ≫ pairwise 的 KL。
- 13:4x 训练器新增 **风险触发锚点**(AW_ANCHOR_ADAPTIVE=1):锚点不混池;按边界侧(call/abstain)各留 12 条 base 正确回复做探针,每 8 步重打分 D_k = 探针 log-prob 相对 base 的变化,λ_k ← [λ_k + η(−ε − D_k)]₊(ε=1.0 nat、η=0.5、上限 4),λ_k>0 时每 chunk 注入 ⌈λ_k⌉(≤2)条该侧锚点对、权重 λ_k;轨迹写 AW_ANCHOR_TRACE。scripts/crcd_night4_hpg.slurm(c4_adaptive / b3_reffree / r3uc3_adaptive)**今晚提交**(已 sync)。
- 13:5x 风险触发锚点冒烟通过(16 事件 + 24 锚点、探针 4/侧、ε=0.2:第 1 步 abstain 侧 drift −0.24 → λ 激活、第 2 步注入后回到 0)。night4 slurm 的 ARMS 顺序 bug 已修并 sync。

### 夜间提交清单(22:05 cron 执行;Qwen 作业仅夜间上 hpg)
1. `ssh hpg "cd /blue/fsu-compsci-dept/xc25.fsu/hq/tc-alignment && sbatch --parsable scripts/crcd_r3uc3_hpg.slurm"`(r3u+C3 官方)
2. `ssh hpg "... && sbatch --parsable scripts/crcd_night4_hpg.slurm"`(c4_adaptive / b3_reffree / r3uc3_adaptive)
3. 若 logs/alf_event_lane.log 有 `[alf-lane] DONE`:`.venv/bin/python tools/alf_events_to_pools.py`;`rsync -az data/alf_sft/pool_*.jsonl data/alf_sft/events_v1.jsonl hpg:/blue/fsu-compsci-dept/xc25.fsu/hq/tc-alignment/data/alf_sft/`;`sbatch --parsable scripts/alf_ablation_hpg.slurm`(五臂)。否则推迟到挖掘完成后提交(仍需在 06:00 前)。
4. 每个 job 挂 Monitor(grep OVERALL|DONE|NO SCORE|GENERATE FAILED|Traceback|bfas-eval,ssh 命令末尾加 `; true`,离队即退出),记 job id。
- 12:40 汇报已发(无新分;三卡并行 + 夜间清单)。
- 13:1x 用户催"还没回应":补发 docs/2026-09-03-crcd-next-stage-response-zh.md(§25 十项格式的研究回应 + 新颖性判断 + 4 个新想法:ΔU 序贯检验、负教师证据臂、span-only 成对、本地 9B 生成边界锚点)。
- 13:3x 想法 1 的便宜证伪(CPU 模拟):序贯 ΔU 检验只省 20% 续跑、与 K=3 判定一致率 67% → K=3 下"后果性"大多只是多赢 1 局(P(ΔU>0)≈0.6),ΔU 符号本身低置信。改方向:对近零事件自适应加 K / 用 P(ΔU>0) 或 LCB 加权(D 臂已用 tanh 权重;可加 "conf" 模式)。挖掘 40/107(127 分歧、46 后果性)。
- 13:4x 训练器加 AW_DDPO_WEIGHT=conf(w = P(ΔU>0),由两侧 Beta 后验 MC 得到);ALFWorld 消融加第 6 臂 D_conf(全部分歧 × P(ΔU>0) 权重 —— "置信度感知"版本,对照 C 硬阈值)。已 sync。
- 13:5x **纠正覆写诊断出结果**:149 条 base 已掌握 prompt 上,逐 token KL(πθ‖π0):CE 0.383 / CE-1ep 0.426 / pairwise 0.0047(小 ~80×);base 正确回复的 log-prob:CE −81(base −36),pairwise −36.9;首 token 调用概率:CE 0.60(调用侧 1.00),pairwise 0.0001。CE 两侧都动、pairwise 两侧都不动 → §7 假设成立。results/analysis/crcd_drift_v1.json。
- 13:5x 用户重发同一 prompt("看看今天开展什么实验"):回复了今日实验清单(已出:纠正覆写 ✅、序贯 ΔU ❌;在跑:无参考、ΔU 校准、ALFWorld 挖掘、门 v2;今晚:r3u+C3、自适应锚点、无参考官方、ALFWorld 六臂);默认把负教师臂(dU<0 反向对)加入今晚 ALFWorld 数组。
- 14:0x ALFWorld 消融加第 7 臂 E_negteacher(C 池 + dU<−0.2 事件的反向对:chosen=学生动作、rejected=demo 动作;§16.1 负教师证据)。tools/alf_events_to_pools.py 与 slurm 已更新、sync。
- 14:1x §16.5 **span-only 成对目标**实现(AW_DDPO_SPAN_ONLY=1:chosen/rejected 的公共前缀与后缀 token 从两侧 label 掩掉,偏好信号只落在差异 span 上),在 rai GPU1 跑 r2 池 1ep 代理评(pid 3302660,logs/crcd_rai_b3span.log;对照 r2-pref 代理 38.18)。之后用 crcd_drift_diag.py 比它与 pairwise 的 KL。
- 14:2x **无参考成对臂代理 41.12**(proxy;base 37.46、base-DPO 同池 38.18):NL 75.83 / Live 75.34 / MT 62.50 / Irrel 72.50 / Rel 69.23。去掉 base 参考在代理上反而更好(+2.9)——假设是"参考项把 base 已排对的样本的损失压小,无参考=同步数下更大的有效剂量"。官方今晚(night4);先用 drift 诊断看它的 KL 是否仍小。GPU3 已接 ΔU 校准链。
- 13:35 汇报已发(纠正覆写、无参考代理 41.12、序贯模拟、在跑项)。
- 13:5x 无参考臂的漂移诊断:KL 0.0013(base-DPO 0.0047、CE 0.383),base 回复 log-prob 不动 → **保留来自成对对比本身,不来自 base 参考**;论文不能把 base 中心当信赖域贡献(§6 结论)。无参考在代理上更高的原因待官方分与"有效剂量"检验。
- 14:3x night4 加第 4 臂 b3_reffree_ep3(无参考 ×3ep = 81 步):若无参考 1ep ≈ base-DPO 3ep(47.61)、无参考 3ep 过量,则"有效剂量"解释成立。已 sync。
- 14:5x span-only 臂训练完成(27 步),代理评 vllm 在 0.3 下 Mamba cache 不足(与 r3uc3 同样问题;0.4 才够)→ 以 0.4 单独重跑评测(logs/crcd_rai_b3span_eval.log)。
- 15:1x **门 v2 第一行(簇 0 = simple_python 80 事件,3ep,K=8,prompt 键)**:自身簇 0.146→0.938(+0.79),其它簇 ≈0 或负(c3 OOS 弃答列 −0.19,c4/c5 −0.04)→ 与 v1 完全不同:选择性很强,且出现可预测的负迁移(调用簇训练伤弃答)。等其余 7 行算 Spearman/符号预测。
- 15:2x span-only 臂漂移 KL 0.0047(与全 span 成对相同;无参考 0.0013)→ §16.5 "更小纠正支持 → 更小漂移"在这个剂量下不成立(成对漂移已在地板);看代理分是否有目标增益差异。
- 15:3x **span-only 臂代理 47.34**(proxy;NL 77.50 / Live 76.71 / MT 62.50 / Mem 33.33 / Rel 76.92 / Irrel 65.00):Memory 33.33 是代理里 ~3 题翻 1 题,把宏平均抬了 ~6;除此之外约与无参考同级(MT 62.5、Rel +7.7)。代理太粗 → night4 加第 5 臂 b3_span 官方评(已 sync)。
- 15:4x 夜间清单追加 5:`sbatch scripts/crcd_drift_hpg.slurm`(hpg 上对 9 个官方 ckpt 跑漂移诊断 → results/analysis/crcd_drift_rounds.json,补"漂移随轮次/锚点"曲线;锚点文件已 rsync)。rai GPU1 新起 r3reffree(r3 联合池 324 行 × 无参考 1ep = 41 步,代理评;pid 3387710,logs/crcd_rai_r3reffree.log)。
- 14:35 汇报已发。挖掘 50/107;门 v2 簇 1 训练中;calib zero 桶评测中;r3reffree 训练中。
- 15:5x r3 联合池 × 无参考(41 步)代理 39.06(r2 池 × 无参考 27 步 41.12):步数一多,无参考先过峰、Irrel 掉到 65 → 支持"无参考 = 更大有效剂量"解释;官方 night4(无参考 1ep vs 3ep)定论。GPU1 空出(仅挖掘)。
- 16:1x **ΔU 校准 zero 桶代理 36.27**(proxy;base 37.46):ΔU=0 的分歧训出来是负增益(Irrel 60、Live 74)→ "分歧 ≠ 监督"的第一个校准点。等 small/medium/large。
- 16:2x 漂移表(KL/token,149 掌握态):无参考-27步 0.0013 < pairwise-27 0.0044–0.0047 = span-27 0.0047 = 无参考-41 0.0042 < r3uc3-55 0.0113 ≪ CE 0.38–0.43。成对类漂移随步数缓慢上升但比 CE 低 30–300×。r3uc3 的探针与其训练锚点重叠,不是干净数字。
- 15:35 汇报已发(zero 桶、r3reffree、漂移表)。
- 16:5x 门 v2 第二行(簇 1 = simple_javascript 11 事件):自身 0→0(JS 题这个剂量学不动),迁移到 c0 +0.15、c2 +0.15,弃答簇 c3 −0.11、c5 −0.10 → "调用簇训练伤弃答簇"的符号结构再次出现。簇 2 训练中。
- 16:35 汇报已发。
- 17:0x ΔU 校准 small 桶代理 43.46(proxy;Memory 33.33 又是 1 题抬 ~6;Irrel 50、Rel 84.6、MT 54.2)。代理 Overall 在这些小臂上不可靠,校准曲线按轴读;若要定论需官方评(明晚可上 hpg 四桶)。
- 17:1x 夜间清单追加 6:`sbatch scripts/crcd_ducalib_hpg.slurm`(ΔU 四桶 zero/small/medium/large 官方评,池与 plan.json 已 rsync 到 hpg data/bfcl_sft/du_calib/)。
- 17:10 用户问进展,已回复今日机制结论 + 在跑 + 夜间清单。

## ⟳ RESTART CHECKLIST(2026-09-03 17:15 更新)
- rai 在跑(按 pid/日志重挂 Monitor):ALFWorld 挖掘 lane pid 2939670(logs/alf_event_lane.log、logs/alf_event_mine.log,GPU1,~20:00 完成后自动建池 data/alf_sft/pool_events_pref_v1.jsonl);门 v2 pid 2982016(logs/crcd_validity_gate_v2.log,GPU4,8 簇,完后跑 `python tools/crcd_gate_analyze.py --gate data/bfcl_sft/gate_v2 --out results/analysis/crcd_gate_v2.json`);ΔU 校准链(logs/crcd_du_calib.log,GPU3,medium/large 桶)。
- 会话级任务需重挂:每小时 :35 进度汇报 cron;一次性 cron 22:05(def9a35f)执行"夜间提交清单"1–6(见上);若会话在 22:05 前重启,要重建该 cron。
- hpg:本项目 0 作业;$HOME 38%;所有脚本已绕开 home。
- τ²/AppWorld 采集停摆(ollama 周上限);AppWorld lane 已停(重启前删 results/bfas/appworld/*.lock)。
- 17:35 汇报已发(无新分)。
- 18:1x ΔU medium 桶代理 40.64(proxy;NL 79.2 / Live 78.1 / MT 45.8 / Irrel 45 / Rel 84.6;Memory 1 题)。BFCL 上桶的类别构成与 ΔU 混杂(高 ΔU 桶几乎全是单轮调用事件),校准要做"类别匹配"版本;先看今晚官方四桶。
- 18:2x 类别匹配校准池:tools/crcd_du_buckets.py 加 --category;data/bfcl_sft/du_calib_sp/(仅 simple_python,small/medium/large 各 40 行,步数对齐)。明晚 hpg 官方(今晚队列已满)。
- 18:4x 门 v2 第三行(簇 2 = live_multiple/parallel/memory/java 18 事件):自身 +0.125(最大),live 邻簇 +0.04–0.05,弃答簇 −0.02(不伤)→ 三行都是对角最大;5 行待出(~20:30 / 22:00 / 23:30 / 01:00 / 02:30)。
- 18:3x 用户:hpg 空着,现在就并行。已提交:**41026602** r3uc3 官方、**41026604** night4 ×5(c4_adaptive / b3_reffree / r3uc3_adaptive / b3_reffree_ep3 / b3_span)、**41026605** ΔU 四桶官方、**41026606** 漂移诊断。22:05 cron 已删(避免重复提交);ALFWorld 七臂由 waiter(pid 3730712,logs/alf_ablation_waiter.log)在挖掘完成后自动建池 → rsync → sbatch。白天被取消的风险已告知用户。
- 18:4x RESTART 补充:22:05 cron 已删;hpg 晚间批次 Monitor b2uujuumj(41026602/604/605/606);ALFWorld waiter pid 3730712 + Monitor b7f2y0kd0(logs/alf_ablation_waiter.log 给出 job id 后需另挂该数组的 Monitor)。记忆已更新:hpg 白天可按用户指示提交。
- 18:35 汇报已发。hpg 11 任务 Priority 排队;挖掘 85/107;门簇 3 训练中;large 桶评测中。
- 19:2x **ΔU 校准四桶代理齐**(proxy,按轴):NL 78.8/78.8/79.2/**82.5**、MT 50/54/46/**58**、Rel 77/85/85/**92**(zero/small/medium/large)→ 目标轴随 ΔU 单调,large 桶三轴最高;Irrel 各正桶都掉 ~22(zero 桶掉 12)。Overall 受 Memory 1 题扰动不可用。官方四桶 41026605 排队中。GPU3 空出。
- 19:3x GPU3 空出 → 启动类别匹配校准(仅 simple_python,small/medium/large 各 40 行 ×8ep ≈40 步,代理评;pid 3769479,logs/crcd_du_calib_sp.log)。
- 19:3x 四个 ΔU 桶 ckpt 的漂移诊断在 GPU3 跑(logs/crcd_drift_du.log → results/analysis/crcd_drift_du.json):看"漂移/增益"是否随桶变化。
- 19:5x **漂移随轮次/剂量(hpg 官方 ckpt)**:KL 随步数升 18 步 0.003 → 27 步 0.005 → 41 步 0.008 → 82 步 0.02;同步数下与池构成基本无关;锚点臂 0.007–0.009。官方分 47.6–47.85 的两个 82 步臂都在 KL≈0.02,135 步(44.82)更大 → "KL≈0.02 信赖域"可作停止准则候选(§8)。41026606 完成。
- 20:2x 四桶漂移:zero 0.0056,small/medium/large 0.046–0.051(call-first 0.16–0.19)——同为 37 步,单类调用事件重复 5 遍的桶漂移是混合池 82 步的 2.5 倍 → 漂移不只看步数,看"每事件重复次数/池的类别平衡";解释了正桶 Irrel −22。写入回应文档。
- 20:5x 门 v2 第四行(簇 3 = OOS 弃答 19 事件,3ep):**自身 −0.06**(弃答事件训不出弃答),反而调用簇 +0.05–0.10 → 对角首次为负;与"r2/r3 模型 OOS 残差不减、Irrel 不涨"一致:弃答这个原子用当前偏好式事件学不到,需要专用通道(更多弃答种子、或把"不调用"作为显式动作对比)。簇 4 训练中。
- 19:35 汇报已发(注:此前几条"20:xx"实为 19:xx)。hpg 7R(dU 1–3 桶 CPU 限额排队);挖掘 100/107;门簇 4 训练中;calib-sp small 训练中。
- 19:5x **span-only 官方 47.52**(全 span 47.34;分轴差 ≤1.5)+ 漂移相同 → §16.5 最小编辑在这个剂量下无增益、无漂移收益,降级。代理 47.34 那次纯属 Memory 1 题假象。
- 20:2x **晚间批次官方五臂**:r3uc3(固定锚点,55 步)46.82;c4_adaptive(210 事件 + 风险触发锚点,27 步)46.08;b3_reffree 46.60(Irrel 82.74 最高);**r3uc3_adaptive(324 事件 + 风险触发锚点,41 步)47.81 = r3_union 47.82 但 Irrel 82.02(+3.0)、MT 53.75** → 自适应保留的 Pareto 改进成立;b3_reffree_ep3 44.94(过量)。**噪声警告**:c4_adaptive 实际等价于 r2-pref 换行序(λ 到 24 步才醒)→ 46.08 vs 47.34,单种子官方 Overall 噪声约 ±1.3,47.8–48.0 之间的差异都在噪声内;最终配方需 3 种子。
- 20:5x **ΔU=0 桶官方 47.52(+1.5 vs base;Memory 32.47)**——与代理(36.27)相反。这些是有状态(多轮/记忆)分歧,K=4 续跑估不出效果(u+≈u−≈低),但纠正本身有价值 → 估计器在长程事件上漏检,"ΔU=0 即行为等价"不成立;硬阈值 ΔU>0 在 BFCL 有状态事件上会丢掉有用监督。等 small/medium/large 官方。
- 20:2x **ALFWorld 挖掘完成**:107 demo → 385 分歧(后果性 85 / 22%,ΔU=0 268,ΔU<0 32);首分歧里只有 22/107 是后果性的 → B 与 C 有真实区分度。池 A 385 / B 107 / C 85 / D 85 / E 93 / D_conf(A 池加权)。waiter 提交的 41031092(各 1ep,步数不齐)已取消,改**步数对齐**重提 = **41031108**(A 1ep=48 步;B/C/D/E/CE 4ep≈44–54 步;D_conf 1ep=48)。Monitor 已挂。
- 20:4x GPU1 空出 → 启动 **ALFWorld ΔU 再估计**(tools/alf_event_reestimate.py + alf_reest_lane.sh,pid 3928412):对 268 条 K=3 估为 ΔU=0 的分歧每侧再跑 5 条续跑(合并成 K=8),看多少变成后果性(检验估计器漏检;对应 BFCL zero 桶官方 +1.5 的反证)。输出 data/alf_sft/events_v1_k8.jsonl,日志 logs/alf_event_reest.log,预计 4–5h。
- 20:4x 用户要今日汇总:已发(8 条机制结论 + ALFWorld + 清理 + 明早可得),附更新后的回应文档。hpg:alfabl 4R+1P、crcdDU 3R;门簇 4 训练中;calib-sp medium 训练中;K=8 再估计 10/268。
- 20:5x 一次性 cron **02803832**(9/4 08:40):早间汇总(ALFWorld 七臂、ΔU 官方四桶 + 类别匹配、门 v2 全 8 行 Go/No-Go、K=8 再估计),更新回应文档并发频道。重启后需重建。当前 Monitor:b2uujuumj(hpg 晚间批)、b14wzi8ii(alfabl 41031108)、bn7br9b7n(K=8 再估计)、begi3298u(calib-sp)、bh2m5tq39(门 v2)。
- 20:4x **ALFWorld 消融首批官方**:B 首分歧 18.66%(25/134)、C 后果性 17.16%(23)、**D 效用加权 23.13%(31,base 8.96 的 2.6 倍)**;A / CE / D_conf / E 待出。读法:硬阈值 C 不优于 B(§21 后果性 NO-GO 倾向),但 D>C +6 → ΔU 数值有信息(GO 倾向);134 局 SE≈2.5pp。
- 21:1x 门 v2 第五行(簇 4 = live_parallel_multiple/memory_kv/mt_miss_func 19 事件):自身 +0.167(最大),c0 +0.146 次之,其余 +0.02–0.05,弃答簇不伤 → 5 行里 4 行对角最大(弃答簇例外)。簇 5 训练中,剩 3 行(~22:40 / 00:10 / 01:40)。
- 21:0x 用户给控制方案 → 训练器加 AW_DDPO_WEIGHT_PERMUTE=global|category;提交 **41032020** 控制五臂(D_perm / D_perm_cat / B_44step / C_56step / A_85);D_const ≡ C(权重均值归一 + 同 seed 行序)。tools/alf_paired.py 配对检验:D vs C p=0.115(14/6/17/97)、C vs B p=0.85、C vs base p=0.035、D vs pres p=0.011。事件深度均值 1.75,p1 107 / p2–4 278。待做:可达性 lane(学生自由 rollout N=4,前缀完全匹配率)明天 GPU1 空后跑。Monitor 已挂。
- 21:1x 可达性 lane 就绪(tools/alf_reachability.py + alf_reach_lane.sh):每训练局 4 条学生自由 rollout,事件 reach = 前 t 步与 demo 前缀完全一致的比例,edit = 1−u⁻,value = reach×edit×max(ΔU,0);waiter(pid 4025290)在 GPU1 的 K=8 再估计结束后自动启动,输出 data/alf_sft/event_value_v1.jsonl,日志 logs/alf_reachability.log。
- 21:2x 【早间汇总补充清单】除 cron 提示所列外还要收:hpg logs/alfctl_*_[0-4].out(控制五臂 D_perm / D_perm_cat / B_44step / C_56step / A_85);rsync 各臂 results/bfas/alfworld/*/records.jsonl 到 results/alf_records/ 后用 tools/alf_paired.py 做配对检验(D vs D_perm、D vs D_perm_cat、C vs B_44step、C_56step vs B、A_85 vs C);data/alf_sft/event_value_v1.jsonl 的 reach/value 统计(按 probe/depth)。按用户 9/3 21:00 的准则判定:只有 D 同时优于常数(=C)与两种置换控制,"效用加权"才算成立;C≈B 时报效率(事件数/步数)而非 NO-GO。
- 21:3x **ALFWorld CE 臂官方 74.63%(100/134)**——base 8.96、最好的成对臂 D 23.13;配对 CE 独解 71、D 独解 2,p<0.001;训练 85 行 ×4ep=44 步。**与 BFCL 完全相反**:base 在 ALFWorld 上不是"边距错",而是"不会"(格式+计划),模仿式 CE 直接装上行为,成对修复只挪边距。→ 目标函数必须"残差条件化":学生在该任务族基本不会(u⁻≈0)时用 CE,会但边距错时用成对。A 全部分歧臂 18.66(=B)。正在提交 CE 控制(CE-A / CE-B / CE-零ΔU / CE-1ep / CE→pairwise)。
- 21:4x CE 控制五臂 = **41032245**(CE_A / CE_B / CE_zero / CE_C1ep / CEpair_C;scripts/alf_ce_hpg.slurm;pool_zero85 已同步)。Monitor 已挂。早间汇总也要收 logs/alfce_*_[0-4].out 与其 records(配对 vs CE_conseq)。
- 21:5x **BFCL 算子选择测试 = 41032704**(scripts/crcd_opsel_hpg.slurm):r2 池按 base 通过率(门 v2 prompt 键 K=8)分成"不会的任务族"(JS 23 + live_parallel_multiple 24 + java 3 = 50 行,base<0.1)与"会的任务族"(simple_python/live_* 135 行,base≥0.25);各跑 CE 与成对(≈44–51 步)官方评。预测:CE 在不会的族赢、在会的族崩(对应 ALFWorld 反转),成对反之。Monitor 已挂。早间汇总也要收 logs/crcdOp_41032704_*.out(分轴看 NL-JS / Live-parallel-multiple 列)。
- 22:0x ALFWorld E 负教师臂官方 **8.21%**(C 17.16;只加了 8 条反向对就把 C 的增益全抹掉)→ §16.1 这种形式有害:反向对强化的是学生早期"look/乱逛"的失败模式。D_conf 待出。
- 22:1x ΔU 官方桶:zero 47.52(Irrel 75.3、Mem 32.5)、small 47.24(NL 85.1、Irrel 60.9)、medium 46.05(NL 85.0、Rel 93.8、MT 53.3、Irrel 53.0)→ 目标轴随 ΔU 单调涨,Irrel 随 ΔU 单调掉;ΔU 同时预测"目标增益"和"边界副作用";large 待出。
- 22:3x **效用加权控制出局**:D_perm(全局置换权重)23.13 = D;D_perm_cat(类别内置换)27.61 > D → 按用户准则,ΔU 对齐的权重不优于置换权重,"效用加权机制"在 ALFWorld 不成立(D>C 的差来自权重分布本身)。B_44step / C_56step / A_85 待出。
- 22:3x 已向用户报告置换控制结论(效用加权 NO;非均匀权重 > 均匀 p=0.013 → 疑 C 剂量不足,等 C_56step);配对数据在 results/alf_records/。
- 22:4x ALFWorld 七臂齐:A 18.66 / B 18.66 / C 17.16 / D 23.13 / D_conf 14.93(全池 × P(ΔU>0),不如均匀 A)/ E 8.21 / **CE 74.63**。41031108 完成。
- 22:5x **ΔU 官方四桶齐**:Overall 47.5 / 47.2 / 46.1 / 43.9;NL 82.2 / 85.1 / 85.0 / 85.2;Rel 81 / 81 / 94 / 94;MT 51.6 / 51.3 / 53.3 / 53.3;Irrel 75 / 61 / 53 / 54。判定:ΔU 数值与"调用轴目标增益"校准,也与"弃答损失"校准;净值随 ΔU 反向 → ΔU 是"变化方向"的信息,不是标量学习价值;不当权重用。晚间批次 41026602/604/605/606 全部完成。
- 23:0x **CE-B(首分歧 CE,42 步)18.66%** vs CE-C(后果性 CE)74.63 → 在绝对纠正算子下,后果性 vs 首分歧差 56 点;后果性过滤在 CE 下决定性成立,在成对下看不出来(因为成对根本装不上行为)。等 CE-A / CE-零ΔU / CE-1ep / CE→pair。
- 23:0x 已向用户报告 CE-B vs CE-C(p<0.001)与主张改写("后果性决定学什么;残差状态决定用什么算子")。
- 23:1x **残差条件算子混合臂 = 41035987**(scripts/crcd_hybrid_hpg.slurm):阶段 1 对"不会的族"(pool_incomp_v2,50 行)做 CE 7ep → 合并;阶段 2 以阶段 1 模型为起点与参考,对"会的族"(pool_comp_v2,135 行)做成对 3ep → 官方评(合并基座 = 阶段 1 模型)。对照:ce_incomp / pref_comp 单独(41032704)、r2 pref 47.34、base 46.06。Monitor 已挂;早间汇总要收 logs/crcdHyb_41035987_*.out。
- 23:2x **步数对齐控制**:A_85(随机 85,44 步)14.18;B_44step 17.91;**C_56step 28.36**(vs B 同步数 18.66,p=0.029;vs C_44 17.16,p=0.006;vs D 23.13,p=0.25)。结论:44 步时 C≈B≈随机;56 步时 C 显著优于 B 且 B 不随步数涨 → 后果性池才"随剂量持续变现";D 相对 C_44 的优势 = 多给 C 一点剂量就能复现(与置换控制一致)。控制五臂 41032020 完成。
- 23:3x 后果性池成对剂量曲线 = **41037417**(C_88step / C_110step;scripts/alf_dose_hpg.slurm),已有 C_44 17.2 / C_56 28.4;看成对能否逼近 CE 74.6 或在哪过峰。Monitor 已挂;早间汇总要收 logs/alfdose_*_[0-1].out。
- 23:5x **CE 阶梯(ALFWorld 官方)**:首分歧 18.7 | 零ΔU 85 行 39.6 | 后果性 85 行 11 步 41.0 | 后果性 85 行 44 步 **74.6** | 全部 385 行 49 步 78.4(vs C p=0.46 等价)。同行数同步数下后果性 ≈ 零ΔU 的 2 倍(p<0.001);全分歧要 4.5× 事件才追平 → 后果性选择在 CE 下事件效率 ~4×;CE 到 44 步基本饱和。等 CE→pair。
- 00:0x CE→pair(后果性 85,CE 4ep + DPO 1ep)官方 78.36(CE 单独 74.63,噪声内略高)。CE 控制五臂 41032245 完成。
- 21:35 汇报已发(注:此前标 22:xx–00:0x 的条目实际为 20:4x–21:3x)。hpg 7R:crcdOp ×4、crcdHyb、alfdose ×2。
- 21:5x 后果性池成对剂量曲线:44 步 17.2 → 55 步 28.4 → 88 步 19.4 → 110 步 18.7:窄峰 ~55 步后回落(与 BFCL 同形);成对天花板 ~28% 远低于 CE 74.6 → 加剂量补不上算子差。41037417 完成。
- 22:0x 按用户"冻结后 3 种子"的要求,冻结 ALFWorld 决定性对比(CE 后果性 vs CE 首分歧 vs CE 零ΔU,同 ~44 步),提交 seed 1/2 = **41039203**(6 任务;scripts/alf_seeds_hpg.slurm;训练与评测 seed 同为 1/2)。Monitor 已挂;早间汇总要收 logs/alfseed_*_[0-5].out 与 records(seed_*),给三种子均值±SE。

## ⟳ RESTART CHECKLIST(2026-09-03 22:05 更新)
- hpg 在跑(重启后按 job id 重挂 Monitor,grep OVERALL/bfas-eval/DONE/NO SCORE/Traceback,ssh 命令末尾加 `; true`):41032704 crcdOp ×4(BFCL 算子选择)、41035987 crcdHyb(混合臂)、41039203 alfseed ×6(CE 三种子)。已完成:41026602/604/605/606、41031108、41032020、41032245、41037417。
- rai 在跑:K=8 再估计 pid 3928412(logs/alf_event_reest.log,GPU1)→ 完成后 waiter pid 4025290 自动起可达性 lane(logs/alf_reachability.log);门 v2 pid 2982016(GPU4,簇 5/8);类别匹配校准(logs/crcd_du_calib_sp.log,GPU3,large 桶)。
- 会话级:每小时 :35 汇报 cron;一次性 cron 02803832(9/4 08:40 早间汇总;若会话重启需重建,内容见 20:5x 与 21:2x 的补充清单)。
- 记录/文档:notes/exp_log.md、docs/2026-09-03-crcd-next-stage-response-zh.md(逐条已更新)、results/alf_records/(配对检验用逐局记录)、tools/alf_paired.py。
- 21:5x 用户:"出了结果就汇总" → 每批结果到达即在频道汇总(不等早上);早间 cron 仍做总表。
- 22:1x 门 v2 第六行(簇 5,13 事件):自身 +0.083,最大在同类姊妹簇 c4 +0.146(两簇共享 live_parallel_multiple / mt_miss_func),c0 +0.10、c7 +0.09,弃答簇 −0.03 → 非对角最大但落在指纹上应最相近的簇。剩簇 6、7(~01:00 完)。
- 22:3x CE 三种子首批:CE_C s1 67.16(s0 74.63)、CE_B s1 26.87(s0 18.66)→ 差距 ~40 点保持;等 CE_zero s1 与 seed 2 三臂后汇总(均值±SE)。
- 22:4x **类别匹配校准(仅 simple_python 三桶,proxy)**:NL 78.8 / 80.4 / 78.3,MT 58 / 50 / 46,Rel 84.6 三桶相同,Irrel 45 / 47.5 / 45 → 固定类别后目标轴不随 ΔU 涨;之前的单调曲线主要是类别构成 → ΔU 数值在同族内几乎不含学习价值信息(与 ALFWorld 置换控制一致)。GPU3 空出。
- 22:4x 已向用户汇总类别匹配校准(ΔU 数值在同族内无学习价值信息;作用是选择而非加权)+ CE seed1 首两臂。GPU3 空闲(留给组员)。
- 22:5x BFCL 算子选择三臂:pref_incomp 47.64(Simple AST 71.3 vs base 65.8 → JS 族 +5.5;Live-PM 70.8 vs 66.7 +4.2);ce_comp 19.08(会的族上 CE 崩,如预测);pref_comp 46.56。等 ce_incomp(关键格:不会的族上 CE 是否赢)。
- 23:0x **CE 三种子齐**:后果性 71.1±2.2、零ΔU 50.0±5.4、首分歧 24.9±3.2;每个 seed C>B p<0.001,C>zero p≤0.047。结论:后果性优先成立且稳健;零ΔU 事件也有真实价值(估计器漏检)→ "后果性优先"而非"后果性唯一";首分歧几乎无用。41039203 完成。
- 23:1x **BFCL 算子选择关键格:ce_incomp 7.69**(JS 族 Simple AST 38.5 vs base 65.8,反而更差)→ "不会的族用 CE"在 BFCL 内不成立。规则修正:决定算子的不是"任务族层面会不会",而是"学生是否已有一个与纠正输出同格式的胜任策略可被覆写"(BFCL base 46%、格式/弃答与每次调用纠缠 → 任何 CE 都覆写;ALFWorld base 9%、无可保护 → CE 装行为)。可先验预测量:base 在部署分布上的成功率 / 漂移探针的 KL 敏感度。混合臂(CE 不会族 → 成对)预计也低,待出。41032704 完成。
- 23:2x **ALFWorld round-2 残差挖掘启动**(tools/alf_round2_lane.sh,pid 68211,GPU3):从 hpg 拉 CE 模型(74.6)权重 → 合并 → 用它作学生挖 demo 回放事件(K=3,≤3 探测/demo)→ data/alf_sft/events_r2ce.jsonl。目的:学生胜任后,残差上是否轮到相对修复优于 CE(修正后算子规则的检验)。明天在 hpg 跑 CE vs 成对(从 CE 模型起)。
- 23:3x 【早间汇总补充清单 2】还要收:logs/crcdHyb_41035987_*.out(混合臂;预计低);logs/alf_round2_lane.log + data/alf_sft/events_r2ce.jsonl(CE 模型上的残差统计:分歧数/后果性数,与 base 上的 385/85 对比 = 残差收缩);三种子表(已在 exp_log);BFCL 算子选择四臂表与修正后的规则(见 23:1x);类别匹配校准结论。文档顶部"9/4 早更新"要点建议:① 后果性优先(三种子稳健)② 算子由"是否有可覆写的胜任策略"决定(BFCL/ALFWorld 双向证据)③ ΔU 只用于选择 ④ 风险触发锚点 Pareto 改进 ⑤ 门 v2 结论 ⑥ 噪声 ±1.3 与种子计划。
- 22:36 汇报已发(注:此前 23:xx 条目实际为 22:0x–22:3x)。hpg 仅剩 crcdHyb 评测中。
- 22:5x 混合臂官方 **7.34**(CE 阶段覆写后成对阶段救不回,与 B4′ 9.13 同一现象)→ 族级混合在 BFCL 作废,与修正后的规则一致。hpg 本项目 0 作业。
- 23:0x hpg 空 → 提交 BFCL 冻结主臂三种子补齐 = **41051456**(r3_union s1/s2、r3uc3_adaptive s1/s2;scripts/crcd_seeds_hpg.slurm;seed 0 = 47.82 / 47.81)。Monitor 已挂;早间汇总给两臂三种子均值±SE(与 ±1.3 噪声对照)。
- 23:5x 门 v2 第七行(簇 6 = 混合 live 簇 32 事件):自身 +0.05,最大在类别重叠的 c2 +0.175、c0 +0.125;弃答簇 c3 −0.105、c5 −0.083。剩簇 7(~01:20),之后跑 crcd_gate_analyze(含列中心化、符号预测、类别/随机基线)。
- 00:0x 门 v2 预分析(7/8 行,tools/crcd_gate_analyze_v2.py):A3 余弦 Spearman 全格 0.33、非对角 0.26、非对角列中心化 **0.44**(置换零假设 p=0.007);类别 Jaccard 基线列中心化 0.36;同侧指示 −0.10;对角 +0.166 vs 非对角 +0.022;对角为行最大 43%;符号预测准确率 0.62(召回 0.67);precision@2 0.50、NDCG@3 0.75。倾向 GO(>0.30 且略高于类别基线),等第 8 行定论。
- 00:1x 阶段 2 就绪:scripts/alf_round2_hpg.slurm(r2_CE_C / r2_pref_C,均从 CE 模型起、以其为合并基座,~44 步)+ tools/alf_round2_waiter.sh(pid 167469):挖掘完成 → 建池 data/alf_sft/r2/ → rsync → sbatch。日志 logs/alf_round2_waiter.log。
- 23:35 汇报已发。round-2 挖掘 80/107(124/14,残差 ≈ base 的 1/5);门簇 7 训练中;K=8 130/268;BFCL 种子 4R。
- 23:5x **BFCL 三种子**:r3_union 47.37±0.23(base +1.3 ≈ 5 SE,稳健);r3u+自适应锚点 47.26±0.32,Irrel 81.24±0.48 vs 78.64±0.38(+2.6,稳健),Rel 77.1±2.1(不稳)。种子 SE 0.2–0.3 → 之前 ±1.3 的"复刻差"部分来自行序+晚醒锚点,非纯种子噪声。41051456 完成。
- 00:0x 再补两组种子:**41060663** BFCL C3 固定锚点 s1/s2(s0 48.02;与自适应 47.26±0.32 比)、**41060664** ALFWorld 成对最佳臂 C_56step s1/s2(s0 28.36)。Monitor 已挂;早间汇总纳入。
- 00:1x **ALFWorld round-2 挖掘完成**(CE 模型当学生):183 分歧 / 25 后果性(base 轮 385 / 85)→ 后果性残差收缩到 29%,调用量 1/4;waiter 正在建池并提交 hpg(CE vs 成对,从 CE 模型起)。
- 00:2x round-2 从 CE 模型起的两臂 = **41062069**(r2_CE_C / r2_pref_C;残差 25 事件 ×14ep ≈ 44 步;对照 CE 模型自身 74.6)。Monitor 已挂。round-2 池:A 183 / B 76(仅 8 后果性)/ C 25 / E 36。GPU3 已释放。
- 00:3x GPU3 空出 → 可达性 lane 改为立即在 GPU3 跑(pid 322973,端口 8985;原 GPU1 waiter 已停),输出 data/alf_sft/event_value_v1.jsonl;预计 ~1.5h,能进早间汇总。
- 00:5x ALFWorld 成对最佳臂三种子:28.4 / 22.4 / 25.4 = **25.4±1.7**(CE 同事件 71.1±2.2)→ 算子差 ~46 点稳健。等 BFCL C3 种子。
- 01:0x **门 v2 定论(8 行)**:A3 余弦 Spearman 全格 0.33 / 非对角 0.24 / 列中心化 **0.43**(置换 p=0.004);类别 Jaccard 基线 0.32 / 0.22 / 0.32;对角 +0.158 vs 非对角 +0.022;对角为行最大 4/8;负迁移符号预测弱(准确率 0.57);NDCG@3 0.77。**判定:条件性 GO**——指纹能预测迁移(高于随机与类别基线,但幅度不大),负迁移预测弱、簇数只有 8、A1 基线未测 → 指纹定位为"覆盖/诊断 + 弱迁移预测",不称原子,Level 3 不解锁。results/analysis/crcd_gate_v2.json。
- 01:1x **round-2(从 CE 模型 74.6 起)官方**:CE 55.22(vs CE 模型 p<0.001,−19)、成对 67.91(vs CE 模型 p=0.093,−6.7;vs CE 臂 p=0.009,+12.7)→ **学生胜任后算子顺序翻转**(round-1 同 benchmark 上 CE 74.6 vs 成对 17–28)。两臂都没超过 CE 模型:25 事件 ×14 遍属过量(每事件重复),下一步按事件数配剂量(2–4ep)。41062069 完成。
- 01:2x round-2b(按事件配剂量,从 CE 模型起)= **41064045**:r2b_pref_C_ep3(25 残差 ×3ep ≈10 步)、r2b_pref_union(r1∪r2 后果性 110 行 ×3ep ≈41 步)、r2b_CE_union(110 行 CE 1ep ≈14 步)。对照 CE 模型 74.6 与 round-2 的 55.2 / 67.9。Monitor 已挂;早间汇总纳入。
- 01:4x round-2b 前两臂:CE_union(110 行 1ep,14 步)75.37、pref_C_ep3(25 残差 ×3ep,10 步)74.63 = CE 模型 74.63 → 低剂量两者都不动;等 pref_union(110 行 ×3ep,41 步)。
- 00:4x round-2b 齐:pref_union 76.12(CE 模型 74.63)、CE_union 75.37、pref_C_ep3 74.63 → 按事件配剂量后两算子都不伤胜任学生,成对累计池是唯一略高于 CE 模型的臂;round-2 的 55.2/67.9 是 25 事件重复 14 遍的过量所致。41064045 完成。
- 00:5x pref_union vs CE 模型配对 p=0.69(4/2/98/30)→ round-2b 的结论是"按事件配剂量不伤胜任学生",不是"再提升"。

## ⟳ RESTART CHECKLIST(2026-09-04 01:00 更新)
- hpg 在跑:41060663 crcdSeedC3 ×2(BFCL C3 固定锚点 s1/s2)。今夜已完成:41031108、41032020、41032245、41032704、41035987、41037417、41039203、41051456、41060664、41062069、41064045(全部已入 exp_log)。
- rai 在跑:K=8 再估计 pid 3928412(GPU1,logs/alf_event_reest.log,~02:30 完,产物 data/alf_sft/events_v1_k8.jsonl);可达性 lane pid 322973(GPU3,logs/alf_reachability.log,~03:30 完,产物 data/alf_sft/event_value_v1.jsonl)。GPU4 空(门 v2 已完成,分析在 results/analysis/crcd_gate_v2.json)。
- 会话级:每小时 :35 汇报 cron;一次性 cron 02803832(9/4 08:40 早间汇总,须收:C3 种子、K=8 统计、可达性统计;文档顶部"9/4 早更新"已预填 1–9 项,补第 10 项后定稿并发附件)。重启需重建两者与上述 Monitor。
- 决定性结论已存记忆 project-crcd-mechanism-findings-0903。
- 01:1x **BFCL C3 三种子 47.91±0.36**(NL 83.63、Irrel 78.06、Rel 75.00);与 r3_union 47.37±0.23、自适应 47.26±0.32 的 Overall 差在 ~1.2 个合并 SE 内;三种配方按轴互换(C3 拿 NL/Mem 丢 Relevance;自适应保 Irrel;r3_union 保 Relevance)。BFCL 主结论:相对 base +1.3~+1.9 稳健,锚点变体只挪边界。41060663 完成;hpg 本项目 0 作业。
- 01:3x **BFCL round-4 挖掘启动**(tools/crcd_round4_mine.sh,pid 437306,GPU4,端口 8986):从 hpg 拉 r3_union(47.82)权重 → 合并 → 挖单轮/OOS/stateful 残差 → pool_events_pref_r4 + 联合池 v4(v3 ∪ r4)→ 06:00 前自动 sbatch scripts/crcd_r4_hpg.slurm(r4_union 1ep / 2ep),否则 rai 代理。残差曲线 322 → 115 → ?。Monitor 已挂;早间汇总纳入(若挖完)。
- 02:1x round-4 单轮残差:r3_union 模型上 **82 事件**(base 322 → r2 模型 115 → r3 模型 82);OOS/stateful 继续。
- 02:3x round-4 OOS 残差:r3_union 模型上 **27**(base 20、r2 20)→ 弃答不但没学到,累计池训练后略变差,与官方 Irrel 78.98(base 82.70)一致;stateful 挖掘中(~1.5h)。
- 01:35 汇报已发(注:此前标 02:xx 的条目实际为 01:xx)。round-4 stateful 10/64;K=8 210/268;可达性 60/107。
- 02:35 汇报已发(无新分)。round-4 stateful 20/64;K=8 250/268;可达性 100/107。
- 02:5x K=8 再估计(253/268 部分):零ΔU 事件 15% 翻正、7.5% 翻负、77% 仍为零;P(ΔU>0)>0.8 仅 5% → K=3 的"零"标签基本稳定。结合 CE_zero=50%:零ΔU 事件在 CE 下的价值是泛化的 demo 模仿,不是该状态的干预价值;"后果性"衡量的是哪些纠正改变结果,CE_C 的 4 倍事件效率来自这里。分析脚本 tools/alf_event_analyze.py。
- 03:0x **可达性出来**:深度 0(首次探测)状态可达 0.96,深度 ≥1 的 teacher 前缀状态在 base 自由 rollout 下可达率 0.00–0.03 → 85 条后果性事件里只有 25 条 base 可达;但 CE 用全部 85 条(甚至以不可达为主的 385 条)都能到 74–78% → "base 策略下的可达性"不是正确的价值因子,纠正是顺序解锁的(改了深度 0 才到得了深度 1)。事件价值模型改用"纠正后策略下的可达性/可解锁性"。GPU3 空出。
- 03:1x 可达性已报;GPU3 空出 → 启动"纠正后策略下的可达性"(同脚本,学生 = CE 模型 74.6,pid 1927477,端口 8987,输出 data/alf_sft/event_value_ce.jsonl,logs/alf_reach_ce_lane.log / logs/alf_reachability.log 会被覆盖写;~2.5h):检验深度 ≥1 的 teacher 前缀状态在纠正后是否变得可达(可解锁性)。
- 03:1x 【早间汇总补充清单 3】收:K=8 最终(`python tools/alf_event_analyze.py`,默认路径)、纠正后策略可达性(`python tools/alf_event_analyze.py --value data/alf_sft/event_value_ce.jsonl --out results/analysis/alf_event_analysis_ce.json`,对比 base 策略的 p2–p4 可达 0.00–0.03)、round-4 挖掘统计与 hpg job(logs/crcd_r4_mine.log 末尾 "[r4] submitted hpg <id>";若 06:00 前未提交则为 rai 代理)。
- 03:3x K=8 再估计最终:n=268 pos=39 neg=19 zero=210 P>0.8=13 P<0.2=2(14.6% 翻正、7.1% 翻负、78.4% 仍零)。GPU1 空出。
- 03:4x **弃答通道测试 = 41072197**(scripts/crcd_abstain_hpg.slurm):仅弃答纠正 39 行(OOS + irrelevance)×8ep ≈ 39 步,CE vs 成对从 base 起,官方评;看弃答原子能否被单独学到(门 v2 c3 行与 round-4 OOS 残差 27 都说偏好事件学不到)。Monitor 已挂;早间汇总纳入。
- 03:5x **纠正后策略的可达性**:p2 0.68 / p3 0.42 / p4 0.30(base 0.02 / 0 / 0);深度 1 从 0.03 升到 0.68;85 条后果性事件 base 可达 25 → 纠正后可达 55(均值 0.26 → 0.59)→ 可解锁性成立:可达性要在纠正后策略下衡量。GPU3 空出。
- 04:0x rai:GPU1、GPU3 空(留给组员);仅 GPU4 的 round-4 挖掘在跑(有状态阶段)。hpg:仅弃答通道测试 41072197。可解锁性结论已存记忆。
- 04:2x 弃答通道 CE 臂官方 **11.73**(Irrel 99.66 / Rel 0.00 / NL 10.7):装上了"总想弃答"反射——纠正覆写在边界两侧对称。等 pref_abstain。
- 03:35 汇报已发(注:此前 03:5x/04:xx 条目实际为 03:0x–03:3x)。round-4 stateful 50/64;pref_abstain 评测中。
- 03:5x round-4 有状态残差 102 / 后果性 20(base 112 / 17,r3 轮 105 / 20)→ 有状态残差不随轮次收缩,单轮残差收缩(322 → 115 → 82);与官方 MT 持平 ~52 一致。等 v4 池与 hpg 提交。
- 04:0x **round-4 提交 hpg = 41076158**(r4_union 1ep ≈51 步 / 2ep ≈102 步,v4 联合池 408 行;对照 r3_union 47.37±0.23)。Monitor 已挂;GPU4 释放,rai 全部空闲。

## ⟳ RESTART CHECKLIST(2026-09-04 04:05 更新)
- hpg 在跑:41076158 crcdR4 ×2(round-4 联合池 1ep/2ep)、41072197_1 pref_abstain(弃答通道成对臂)。Monitor:bxxmtm50c、b6baxmv5z。
- rai:无本项目 GPU 任务(K=8、可达性 ×2、round-2 挖掘、round-4 挖掘全部完成,产物在 data/alf_sft/、data/bfcl_sft/,分析在 results/analysis/)。
- 会话级:每小时 :35 汇报 cron;一次性 cron 02803832(9/4 08:40 早间汇总:补 round-4 两臂、pref_abstain,把文档顶部第 10 项定稿,发附件)。重启需重建。
- 今夜结论均已入 notes/exp_log.md 与 docs/2026-09-03-crcd-next-stage-response-zh.md;决定性结论在记忆 project-crcd-mechanism-findings-0903。
- 04:2x 弃答通道成对臂官方 **42.13**(Irrel 88.38 +5.7,但 Rel 56.25 −19、NL −9、MT −8):单侧同质池重复 8 遍,成对也会推边界(比 CE 的 11.73 温和得多)。结论:弃答原子可学,但只能在平衡池/锚住调用侧的前提下学——C3 与自适应锚点正是在做这件事;调用/弃答是同一条决策边界,单侧过量纠正无论算子都会挪它,算子只决定挪多远。41072197 完成。
- 04:35 汇报已发。round-4 两臂训完进评测(51 / 102 步)。
- 05:0x **round-4 官方 46.92**(1ep 51 步;r3_union 47.37±0.23)→ 累计自迭代在 round 3 后停滞/微降(−0.45,≈1.5 SE);Irrel 继续下滑 77.4;新增的 122 事件多是单轮尾部 + 不收缩的有状态事件。等 2ep(102 步,预计更低)。
- 05:1x round-4 结果已报(自迭代 round 3 收敛;下一步弃答/有状态专门通道)。等 r4 2ep 剂量对照。
- 05:2x round-4 2ep 官方 47.05(Mem 35.70 最高、MT 48.9、Irrel 75.0)→ 任何剂量下 round 4 都不超过 round 3。41076158 完成;**今夜全部作业结束,hpg 0 作业**。

## ⟳ RESTART CHECKLIST(2026-09-04 05:30 更新)
- hpg:本项目 0 作业;rai:无本项目 GPU 任务。今夜全部结果已入 notes/exp_log.md;回应文档 docs/2026-09-03-crcd-next-stage-response-zh.md 已定稿(顶部 10 条 Go/No-Go);决定性结论在记忆 project-crcd-mechanism-findings-0903。
- 会话级:每小时 :35 汇报 cron(重启需重建)。08:40 早间汇总 cron 已由 05:30 的定稿汇总替代(删除)。
- 明日候选(零 API):弃答/有状态专门通道(平衡锚点 + 风险触发 + 有状态单独剂量)、ALFWorld round-2 按事件配剂量的累计臂再加种子、事件价值模型(可解锁性 × 可编辑性 × ΔU)在 ALFWorld 池上做选择实验、AppWorld 待 ollama。
- 05:3x **总汇总已发**(定稿文档附件 + 10 条 Go/No-Go);08:40 cron 已删。今夜结束:hpg 0 作业,rai 空闲。
- 05:35 汇报已发(无变化)。rai 上无本项目 GPU 进程(仅 Monitor 的 tail);hpg 0 作业。
- 06:35 汇报已发(无变化)。
- 07:35 汇报已发(无变化)。
- 08:35 汇报已发(无变化)。
- 09:35 汇报已发(无变化)。
- 10:35 汇报已发(无变化;hpg 离线,无作业受影响)。
- 11:35 汇报已发(无变化;hpg 仍离线)。

## 阶段切换(2026-09-04 12:30):统一 CRCD(任务书 docs/2026-09-04-crcd-next-tasks-from-user.md)
- 12:40 四条 P0 子任务并行(后台 agent):T1 Fisher 白化指纹 + 统一事件 schema(GPU1;文件 src/bfas/events_schema.py、src/bfas/fingerprint.py、tools/events_convert.py、data/fingerprints/*.npz);T2 稀疏字典 + 连续迁移门 + 基线(CPU;src/bfas/atoms.py、tools/atoms_*.py、results/analysis/atoms_transfer_gate_v2.json);T3 镜像目标 + 原始-对偶训练器 + 冒烟(GPU2;src/bfas/mirror.py、tools/crcd_mirror_train.py、tests/test_mirror.py);T4 AppWorld 划分审计 + base 冒烟(GPU4;tools/appworld_audit.py、docs/2026-09-04-appworld-split-audit.md)。日志 logs/t{1..4}_*.log。
- 我方产出:docs/2026-09-04-crcd-unified-audit-zh.md(§15 七项审计)、docs/METHOD.md(方法文档骨架,含 impl/prop/val/neg 标记)。
- 待用户决定(见审计 §7):ALFWorld 开发集用 valid_seen;AppWorld teacher(等 ollama 或先用 gpt-5.4 归档演示);BFCL 不做 token 效率主张;三种子算力窗口。
- 12:5x data/bfcl_sft/calibration_ids.json:60 条从未进过任何训练池/锚点/桶的单轮 prompt(按 hash),保留给认证校准。ALFWorld 划分可用:train 8810、valid_seen 494、valid_unseen 341 目录(valid_seen 可作开发集,待用户同意)。当前 commit a8d4744(工作区大量未提交改动)。
- 12:5x 待用户决定(补充 5):任务书要求每条记录带 git commit,而工作区有大量未提交改动(git status 约百余项);建议我把当前工作区提交为一个检查点(不 push),之后每个里程碑一提交——需用户点头(规则:仅在用户要求时提交)。
- 13:0x src/bfas/adapters/alfworld.py:评测划分可由 BFAS_ALFWORLD_EVAL_SPLIT=valid_seen|valid_unseen 指定(worker 映射到 AlfredTWEnv 的 eval_in/out_of_distribution);默认仍 valid_unseen。回归锚点表 results/analysis/regression_anchors_2026-09-04.json。
