# CRCD 初步结果报告(P0,BFCL 先行)— 2026-09-02

## 0. 一句话

今天把 CRCD 的 P0 骨架在 BFCL 上跑通:学生 rollout → 首分叉定位 → K 次 continuation 的因果验证 → 事件池 →
B 阶梯(只 CE 纠正 span / 只 pref / CE+pref)三臂跑完(B3 官方,其余 proxy);同时补齐了两个 base 参照
(BFCL base 复测 46.06,ALFWorld base 4B 9.0%)。两个最重要的发现:①**同一批事件,CE 崩盘、preference 持平 base**——蒸馏对象决定伤害;②**stateful 任务的"最早分叉"绝大多数不是
consequential**——纠正那一轮后 K 次 continuation 的成功率几乎不变(45 个事件里只有 6 个 dU>0),
文档里"earliest error ≠ consequential error"在真实数据上成立,而且比预想的更极端。

## 1. 参照(P0-1)

| 项 | 结果 | 说明 |
|---|---|---|
| BFCL base(4B,官方全量)第 2 次 | **46.06** | 第 1 次 46.27;分轴 NL 79.58 / Live 77.72 / MT 50.12 / Memory 25.59 / Web Search 9.50 / Irrel 82.70 |
| → base 噪声 | ~0.2 | 以后 0.5 以内的总分差当噪声;分轴噪声更大(Memory/Web Search 题少) |
| ALFWorld base(4B,ReAct 部署脚手架,官方 unseen 134 局) | **12/134 = 9.0%** | pick_and_place 20.8、pick_cool 19.0、look_at 11.1、pick_heat 4.3、clean 0、two_obj 0;平均 38 步 |
| ALFWorld base(2B,裸 prompt) | 0/134 | 作废;学生一律 ≥4B(用户规则) |

## 2. 事件挖掘(P0-1)

**单轮(support 单轮 + 生成单轮题,K=4 采样,AST checker 判定)**:363 题 → 319 题有可纠正事件,41 题学生 4/4 全对,3 题无真值。
学生通过率分布:0/4 通过 98 题、1/4 98 题、2/4 87 题、3/4 36 题;平均 dU(=1−u⁻)0.64。
类别:simple_python 266(生成池偏科)、live_parallel_multiple 13、simple_javascript 13、live_simple 11、live_multiple 10、其余零星。

**stateful(生成 stateful 集,63 集 × 3 rollout,首分叉 + K=4 continuation)**(仍在挖,至 45 事件):
- 首分叉几乎都在第 0 轮(41/45):学生第一步就和真值动作不同、状态分叉;
- **consequential(dU>0)仅 6/45,neutral 39/45,无负值**;u⁺ 均值 0.51,u⁻ 均值 0.44——
  就是说把第一步换成真值后,学生后面的 continuation 一半以上还是失败;分叉那一步不是成败的关键;
- 目前覆盖的是 memory 三类(rec_sum 28、vector 14、kv 3),multi_turn / web_search 集在后面。
含义:(a) 训练信号应只用 consequential 事件,否则在"改了也没用"的轮上花梯度(正是文档 5.1 的主张);
(b) memory 任务的真正缺口在后面的轮(检索后如何用),需要"最早 *consequential* 分叉"定位——
把候选从第一个分叉扩到后续分叉逐个做 continuation 检验(明天实现);(c) 今天的 B 阶梯池里 stateful 只有 2 行,
结论只对单轮类别有效。

**池(B 阶梯)**:137 行(dU>0、按 (task, turn, a⁻) 去重、每类别封顶 80 防 simple_python 主导、a⁻ 剥掉空 think 块),平均 dU 0.75。

## 3. B 阶梯(P0-2,单 seed;B3 官方全量,B2/B4' 为 rai 代理分,官方分今晚 hpg 补)

| 臂 | 训练信号 | Overall | Non-Live | Live | MT | Memory | Irrel | vs base(46.06) |
|---|---|---|---|---|---|---|---|---|
| base | — | 46.06 | 79.58 | 77.72 | 50.12 | 25.59 | 82.70 | — |
| B0 v9 全轨迹 SFT(已有) | 955 行(生成+demo+guided) | 36.70 | 82.75 | 73.35 | 21.12 | 31.61 | 82.29 | −9.4 |
| B2 只 CE 纠正 span(proxy) | 137 事件行,3 epoch | **8.17 proxy**(base proxy 37.46) | 42.5 | 34.3 | 0.0 | 0.0 | 5.0 | 崩盘:永远调用反射 |
| B3 只 local pref(base-centered) | 同上 + a⁻,18 步 | **46.09**(proxy 38.18 vs base proxy 37.46) | 80.75 | 78.61 | 50.88 | 24.95 | 81.02 | +0.0 官方;proxy Live +1.4、MT +4.2、Irrel −7.5(3 题) |
| B4' CE(3 ep)→ pref(两阶段,proxy) | 同上 | **7.32 proxy** | 39.2 | 31.5 | 0.0 | 0.0 | 2.5 | CE 阶段崩盘,pref 救不回 |
| B2 对照:只 CE,1 epoch(18 步,proxy) | 同上 | **15.93 proxy** | 43.8 | 43.8 | 0.0 | 33.3 | 5.0 | 仍崩:伤害来自 CE 目标本身,非过拟合 |
| B4' 对照:CE 1 ep → pref(proxy) | 同上 | **7.66 proxy** | 38.8 | 32.9 | 0.0 | 0.0 | 5.0 | 仍崩 |

**读法。** 同一批 137 个事件,只换蒸馏对象:
- **CE 复制纠正 span(B2)= 崩盘**:Irrelevance 72.5→5、Relevance 100、MT 0。事件里全是"该调用什么"的正目标
  (irrelevance 事件这轮没挖),CE 把"永远调用"烙死——和 pool v4 时代的失败同一机理,只是池更小、更纯、更快。
- **base-centered 局部 preference(B3)= 零伤害**:官方 46.09 = base;proxy 上 Live +1.4、MT +4.2。
- **CE 之后再 pref(B4')= 仍崩**:DPO 修不回 CE 阶段的伤。
结论(P0-2 的第一个硬信号):**蒸馏对象决定 Damage**。复制目标(哪怕只是纠正 span)会把 teacher 行为的形状整体压给学生;
preference 只移动决策边界,不改变学生的行为分布——这正是文档"distill the correction, not the trajectory"的反面证据加正面证据。
已做的对照:1-epoch CE(18 步)仍崩(proxy 15.9)——与同步数的 pref-only(=base)直接对照,伤害来自 CE 目标而非训练量;待做:加 irrelevance/abstain 事件后的 CE、更大事件池下的 pref 是否产生真实增益。

## 4. 其他三台今日状态

- AppWorld(官方脚手架,4B):demo 40/40,学生 unguided rollout 第 12 轮(慢,共卡),池未落地。
- ALFWorld:2B 结果全部作废;学生侧 4B + ReAct 重采中(107 条 demo 复用);base 4B 已出(上表)。
- τ²:学生 4B 重采中(经 ollama 双 key 代理,零 429);demo 32/58。

## 5. 明天(按信息价值)

1. 第二轮事件挖掘:最早 **consequential** 分叉(逐分叉做 continuation 检验),覆盖 multi_turn / web_search,
   加 teacher 纠正(非 GT)通道;B 阶梯第二轮含完整 stateful。
2. A 阶梯(P0-3):同一批事件上 A1–A4 指纹,用"训练后 held-out 事件增益"做标签(需要一批 micro-update)。
3. C 阶梯(P0-4):uniform / 1−p̂ / 簇 / dU / U×R。
4. AppWorld 池落地即出 ours/base;ALFWorld/τ² 池重采完即出 ours。
