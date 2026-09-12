# Table 1 核查与补测计划(teacher = deepseek-v4-pro,student = Qwen3.5-4B)

日期:2026-09-08。目标:Overleaf 主表(AppWorld / BFCL v4 / τ²-bench / ALFWorld × {Teacher, Base student, 7 baselines}),每格按标准协议、三种子(论文表要求 mean±std over 3 seeds)。

## 1. 现有数字逐格核查

### 1.1 Base student
| 列 | 表中 | 核查结论 |
|---|---|---|
| AppWorld | 0.248±.035(dev40,pooled partial) | dev40、max_steps 24、greedy;pooled partial = 通过的单元测试数/总测试数(pooled);TGC 0–1/40。协议与 awb3 一致,**保留**。 |
| BFCL v4 | 0.439±.004 | 早期 rai 战役(43.55/43.92/44.26),Multi-Turn 45–46、Web 6.5–7.5;后来同模型标准战役 **46.27(rai)/46.06(hpg)/46.25(rai Blackwell)**,差异全在 Multi-Turn(→50–51)和 Web(→10)。**表中 0.439 作废,用标准战役 46.27(rai);不重测。** |
| ALFWorld | – | 已有标准结果:valid_seen 140 题 greedy **7.14%**(hpg 41098042)。**直接填。** |
| τ²-bench | – | 无。 |

### 1.2 Teacher(deepseek-v4-pro)
| 列 | 表中 | 核查结论 |
|---|---|---|
| AppWorld | 0.569 | teacher_dev40_eval:model=api:deepseek-v4-pro,success 20/40(TGC 0.5),pooled partial 0.569。**保留。** |
| BFCL v4 | – | 未测。teacher-as-agent 跑 BFCL v4 全量 5217 条(含多轮/记忆/联网)估算 40–60M token,**超过一个 ollama 周额度**。建议:引用 BFCL 官方 leaderboard 上 DeepSeek-V4-pro 的分数(标准评测,零 API),表注说明;若 leaderboard 无该模型,再议。 |
| ALFWorld | – | 未测。valid_seen 140 题 × ≤40 步,估算 4–6M token,可做。 |
| τ²-bench | – | 见 §3。 |

### 1.3 七个 baseline
**AppWorld 列(表中 awb2)**:awb2 训练有 truncation bug(92% 行被 640-token 截头),权重已删。修好后的 **awb3**(hpg 40606110,8 臂 × 3 种子,dev40,同协议)已完成,结果(pooled partial / success):

| 臂 | pooled partial(3 seeds) | mean | success |
|---|---|---|---|
| Vanilla SFT | .239/.197/.239 | 0.225 | 0/40 ×3 |
| Structured AD(sad) | .207/.213/.218 | 0.213 | 0/40 ×3 |
| Agent distillation | .213/.197/.197 | 0.202 | 0/40 ×3 |
| BB on-policy | .197/.197/.197 | 0.197 | 0/40 ×3 |
| dDPO | .309/.255/.213 | 0.259 | 3/0/0 |
| PBSD | .234/.213/.223 | 0.223 | 0/40 ×3 |
| STaR | .282/.266/.213 | 0.254 | 2/1/0 |
| (ours-adv,参考) | .436/.399/.479 | 0.438 | 7/2/7 |

**但** awb3 的训练池是三教师混合(data/appworld_sft/episodes.jsonl:deepseek 78 / kimi 11 / qwen397b 39),不是纯 deepseek。若主表 teacher 必须是 deepseek-v4-pro,AppWorld 的 7 个 baseline 要用 deepseek-only 池重训(deepseek 的 episode 已缓存,**不需要新 API**;BB-OPD 的 on-policy relabel 需要少量 API)。

**BFCL 列(表中 awb2)**:同样作废。标准战役下有两批 BFCL 专用 baseline(池 = deepseek demos,data/bfcl_sft/pool_bfcl_ds_*):
- bfclb(v1,训练截断 bug):SFT 42.85 / SAD 43.46 / AgentKD 43.41 / dDPO 42.77 / PBSD 44.14 / BB-OPD 43.14 / STaR 39.39 —— **作废**(训练缺陷,非评测缺陷)。
- bfclb2(truncation-fixed,**有效**):SFT 37.87/32.11/39.13 = 36.37±3.74;SAD 38.09/34.34/38.12 = 36.85±2.17;dDPO 33.14/33.79/38.34 = 35.09±2.83;AgentKD s0 38.12;BB-OPD s0/s1 37.65/33.17;PBSD s1/s2 38.44/38.37。
- **缺**:AgentKD s1/s2、BB-OPD s2、PBSD s0、STaR s0/s1/s2 = 7 格。权重已删 → 重训(池已缓存,零 API)+ 标准战役评测。**已于今天在 rai GPU3/GPU4 起两条 lane(tools/bfclb2_fill_lane.sh)。**

**ALFWorld 列**:无 baseline。现有 CE 锚点 77.86% 是 gpt-5.4 的 107 条 demo 上的 SFT(teacher 不符)。deepseek 的 ALFWorld demo 采集当时因周额度中止(且需 think:false 修复,已修)。
**τ²-bench 列**:无。

## 2. 补测清单与 API 预算(deepseek-v4-pro @ ollama)

| 项 | 需要 | API token(估) | GPU | 备注 |
|---|---|---|---|---|
| A. BFCL 7 个缺格 | 重训 + 标准评测 | 0 | rai 2 lane,~1 天 | 已启动 |
| B. ALFWorld deepseek demo | 142 train 任务 × 3 次尝试 | ~1.5M | – | think:false;去重按状态 hash |
| C. ALFWorld teacher 行 | valid_seen 140 题 | ~5M | – | 一次性 |
| D. ALFWorld 7 baseline × 3 seeds | bfas 管线(alfworld adapter)训练 + 140 题 greedy 评测 | BB-OPD relabel ~0.5M | hpg B200 ≈ 21×(1h 训练 + 4.7h 评测)≈ 120 GPU·h | 评测锁已按 campaign 分开 |
| E. AppWorld deepseek-only 重训 | 7 × 3(awb3 配方,池过滤为 deepseek) | BB-OPD relabel ~0.3M | hpg ≈ 24 作业 × ~1.5h | 仅当你要求纯 deepseek |
| F. BFCL teacher 行 | teacher-as-agent 5217 条 | **40–60M** | – | 建议引用 leaderboard |
| G. τ²-bench 整列 | 见 §3 | **≥40M/行** | – | 建议放弃 |

不含 F、G:**≈7M token**,一个周额度可覆盖;顺序 B → C → D 的 relabel → E 的 relabel,先用 key1,key2 备用。

## 3. τ²-bench 可行性
账本:58 个 teacher episode 共 3.3M token(中位 47k/episode),user simulator 33M token(338 次有记录调用,中位 39k,p90 204k)。一个 τ² episode ≈ 150k token(user-sim 占大头,且 user-sim 也要一个强模型)。标准 τ²(airline 50 / retail 114 / telecom 114,pass^k)哪怕 1 trial,**一个模型一行 ≈ 278 × 150k ≈ 42M token**,9 行 ≈ 375M。一个周额度做不到一行。结论:主表去掉 τ² 列(或改为附录小规模、非标准子集——但你要求标准)。

## 4. AppWorld 指标
官方指标是 TGC/SGC(dev40 上所有 baseline 都是 0/40 或接近,base 0–1/40,teacher 20/40);pooled partial 是单元测试通过率的软指标。建议主表报 pooled partial 并在表注给 TGC,或两者并列;需要你定。

## 5. 需要你决定
1. τ² 列:放弃(推荐)/ 附录小规模。
2. BFCL teacher 行:引用 leaderboard(推荐)/ 花 40–60M token 自测。
3. AppWorld baseline:接受 awb3 混合教师池(零成本)/ 重训 deepseek-only(E,hpg 24 作业)。
4. AppWorld 指标:pooled partial 主 + TGC 注(推荐)/ 只 TGC。
5. ALFWorld:按 B→C→D 全做(推荐,≈7M token + 120 GPU·h)。

## 6. 立即执行(不等决定,零 API)
- A(BFCL 7 格)已在 rai 跑。
- Overleaf:BFCL base 改 46.27,ALFWorld base 填 7.14;其余等补测。
