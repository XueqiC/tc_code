# 方法逐步细节(精确到张量与公式,以 gsm8k-code 为例)

以当前实现(src/e3_tier1.py + src/boundary.py)为准;与设计不一致处已标注。

## 第 0 步 数据角色
| 数据 | 数量 | 用途 |
|---|---|---|
| few-shot query(spec) | k=50 | 需求账本(全 50);内部劈 35 fit + 15 calibration(固定种子) |
| 候选池 pool | 952 行 | 3 teacher × {gsm-code, pandas, sql} + gold {alpaca, gsm-cot};字典、供给、被选对象 |
| test 集 | 独立 | 只评测,任何步骤不可见 |
| 预算 b | 例 20k | 允许进入训练集的 response token 总数 |

## 第 1 步 梯度指纹(每条文本一个向量)
对池里每行和每条 spec query:
1. student 挂 LoRA(r=16, α=32),**只有 96 个 lora_B 矩阵开梯度**
2. 文本按 chat template 编码(prompt 截断 + response 截断),前向算 response 段的交叉熵,反向 → 96 个 lora_B 梯度矩阵
3. **LESS 式 warmup 提供 Adam 预条件**:先在池的 5%(48 行)上训 4 个 epoch,存 4 个 checkpoint 的 Adam 二阶矩 exp_avg_sq。每个 checkpoint 下重算梯度并做 g / (√exp_avg_sq + 1e-8) —— 含义:除掉"这参数平时波动多大",突出"相对于日常波动,这次想动多少"
4. **随机投影降维**:每个模块一个固定高斯矩阵(randn/√proj_dim),把展平的梯度投到低维,拼接 → 该 checkpoint 的特征;4 个 checkpoint 取均值 → 最终指纹 x(全体堆叠成矩阵 X,行数 = 文本数)

## 第 2 步 稀疏字典 → capability atoms
1. 输入:X(当前实现 = spec 50 + pool 952 共 1002 行;**设计口径应为 pool-only,下轮修正**)
2. 解 min_{C,D} ‖X − C·D‖² + λ‖C‖₁:K=64 行字典 D(每行 = 一个 atom,梯度空间单位方向),lasso_lars,α=0.05,交替优化
3. 码取绝对值:codes = |C|,每行文本 → 64 维稀疏非负向量
4. **支撑集**:spec 需求按大小排序,取累计质量 90% 的 atom 集合(支撑 atoms);之外的 atom 视为界外方向

## 第 3 步 需求账本
demand = spec 50 条码的均值(限支撑集);账本 W = demand / Σdemand × b。
例:b=20k,JOIN 型 atom 占 45% → 该账户 9000 token。

## 第 4 步 入场券(conformal 边界门)⚠️ 本次发现的缺陷所在
- **设计**:只有"像这个任务"的轨迹才有资格进市场
- **当前代码**:admitted = {i : (1 + #{cal 分数 ≤ score_i})/(n_cal+1) ≥ α/2}。**n_cal=15、α/2=0.05 时 (1+0)/16=0.0625 ≥ 0.05 恒成立 → 全池放行,门空转**。这就是 sqljoin 残留 0.900 的机械根源:gsm 轨迹畅通入场,靠共享 atom(通用代码)把供给卖进合法账户
- **修正**(下轮实现):改用 D_less_bnd 已验证的门 —— score_i ≥ calibration 分数的 α/2 分位(sqljoin 上该阈值 0.723,只放行 55/952)。分数定义见第 7 步,同一套边界,产物复用

## 第 5 步 市场出清(在入场者中花预算)
每个入场轨迹 i:supply_i = (code_i / Σcode_i) × tokens_i,限支撑集——"它的 token 按码比例摊到各 atom"。
贪心循环:
```
while 有剩余轨迹且预算未尽:
    对每个 i:gain_i = Σ_a min(W_a, supply_i[a]) / tokens_i   # 性价比
    选 gain 最大者;超预算则弃;否则买入:W ← max(W − supply_i, 0)
若账清而预算有余:剩余入场轨迹按供给密度(Σsupply/tokens)降序补买
```
账户填满后同类供给自动不计入 gain(min 的作用)→ 边际递减内生。

## 第 6 步 训练 + 行为停
选出的行按 chat template 编码,LoRA SFT 固定 3 epoch(lr 按学生规模缩放,梯度累积);每 8 个 optimizer step 在 **calibration 中 10 条**上探 runnable-rate,保留最优 checkpoint,结束时恢复(只选 checkpoint,不动边界)。

## 第 7 步 边界 + 证书(交付物)
1. **fit 35 条**的指纹行归一化(unit),不减均值直接 SVD,保留累计能量 90% 的前 r 个右奇异向量 → 子空间 U
2. 任意文本的边界分数 s(x) = ‖xU‖ / ‖x‖(指纹落在任务子空间里的范数占比,∈[0,1])
3. **calibration 15 条**的分数取 α 阶次序统计量(k=⌊α(n+1)⌋)→ 阈值 t。conformal 保证:新 in-task query 被判"界内"的概率 ≥ 1−α,无分布假设
4. 部署 gate:请求指纹 s ≥ t → student;否则拒绝/路由 teacher。证书 = "在边界内覆盖 ≥1−α" + 行为探针通过率

## 第 8 步 评测
test 集官方指标(gsm 执行准确率 / sqljoin 有效率 + 界外残留 / AppWorld TGC),3 种子。

## v3 增补(设计已定,未实现)
- 查漏:池码共现表 P(b|a) → 需求扩散一步,γ 由 spec 内部留一预测误差自选
- 探针:扩散新增账户先由 teacher 小额验伪(回答码落回该 atom 才开户)
- buy-vs-ask:每轮 gain 与"定向补货期望收益"比较,决定买存货还是问 teacher
