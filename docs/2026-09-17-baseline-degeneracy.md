# 基线在我们的材料上退化到什么程度(2026-09-17,纯 CPU 静态分析)

> 目的:在为 SmartAD / SAD / Kang 重跑主表基线之前,先确定它们在**我们实际拥有的材料**上
> 是否还与纯 CE 有区别。结论直接决定哪些格值得花 GPU。
> 全部结论由 LONI 上真实的 `training_rows.json`(ALFWorld 1,535 行、HotpotQA 2,079 行)算出,未训练、未消耗 GPU。

## 1. 三种基线各自的作用点

| 方法 | 采集侧 | 损失侧 |
|---|---|---|
| SmartAD | 每个任务从已购轨迹中选一条(按 base 学生的 macro-mean turn NLL) | 分段加权 CE:reason 1.0 / action 1.5 / final 2.0 / observation 0,按权重和归一 |
| SAD | — | 把 token 分成 {reason} 与 {action, final} 两组,**两组各自取均值后再平均** |
| Kang | 一致性投票(n=3)决定查询哪些状态 + 前缀模式 | — |

## 2. 三处退化(全部可由材料直接判定)

### 2.1 每个任务只有一条已验证轨迹 → SmartAD 的选择是 1 选 1

| 银行 | 任务数 | (task, package) 对 | 每任务轨迹数分布 |
|---|---:|---:|---|
| ALFWorld 142 | 104 | 104 | {1: 104} |
| HotpotQA 904 | 614 | 614 | {1: 614} |

**SmartAD 的采集侧在两个基准上都是空操作。** 这与 2026-09-14 的记录一致,但这次是从最终训练行直接确认的。

### 2.2 ALFWorld 银行没有 reason 段 → SAD 与 SmartAD 的损失都退化成纯 CE

| 银行 | reason 字符 | action 字符 | final 字符 | SmartAD 在多少行里用到 >1 种权重 | SAD 在多少行里真有两组 |
|---|---:|---:|---:|---:|---:|
| ALFWorld | **0** | 24,304(90.1%) | 2,678(9.9%) | **0 / 1,535(0.0%)** | **0 / 1,535(0.0%)** |
| HotpotQA | 266,145(79.4%) | 50,938(15.2%) | 18,211(5.4%) | 2,079 / 2,079(100%) | 2,079 / 2,079(100%) |

ALFWorld 的每一行要么全是 action、要么全是 final,**没有任何一行混合**。
SmartAD 按权重和归一,行内权重恒定时归一化直接把权重消掉;SAD 的 {reason} 组为空,只剩一组。
**因此在 ALFWorld 上,SmartAD 的损失与 SAD 的损失彼此相同,且都等于按行宏平均的纯 CE。**

在 HotpotQA 上两者都真实生效:SAD 的等效逐字符权重是 **reason ×0.63、action+final ×2.42**;
SmartAD 的等效逐字符权重是 reason ×0.885、action ×1.327、final ×1.769。

### 2.3 预算 = 100% support 时,采集侧方法无从选择

主表现有的格用的是 `--support-tasks`(ALFWorld 142、HotpotQA 904)= **全部 support**,
此时 Kang 的一致性投票没有任何取舍空间,采集侧退化为恒等。
**Kang 只有在预算 < support 时才有意义。**

## 3. 唯一剩下的差别:宏平均 vs 微平均

我们现有的 CE 格用 `--method sft`,其损失按**批内总生成 token 数**归一(微平均);
SmartAD/SAD 这条代码路径按**行**取平均(宏平均)。两者只差行长度加权。

| 银行 | 每行字符数 min / 中位 / max | 变异系数 |
|---|---|---:|
| ALFWorld | 4 / 15 / 43 | 0.349 |
| HotpotQA | 66 / 160 / 311 | 0.204 |

ALFWorld 行长差异不小(4 到 43 字符),所以宏、微并不重合;但这是**归一化口径**的差别,
不是这些方法发表时主张的机制。

## 4. 对重跑计划的结论

| 格 | 建议 | 理由 |
|---|---|---|
| ALFWorld × SmartAD | **不跑** | 损失与 SAD 逐字相同,采集为空操作;跑出来只是另一个归一化口径的 CE |
| ALFWorld × SAD | **最多跑一格**,并改名为"宏平均 CE"对照 | 它测的是宏 vs 微归一化,不是发表的机制 |
| ALFWorld × Kang | **不跑(在预算 = 100% 下)** | 采集侧无取舍空间 |
| HotpotQA × SmartAD | **跑** | 100% 的行都用到多种权重,机制真实生效 |
| HotpotQA × SAD | **跑** | 等效权重 reason ×0.63 / action+final ×2.42,与 CE 差别显著 |
| HotpotQA × Kang | 需要先定一个 **预算 < support** 的点 | 否则采集侧退化 |

**这同时是一个论文级结论,而不只是省 GPU**:
ALFWorld 这一列**在结构上无法区分分段加权类基线**,因为教师银行里根本没有可见推理段可供加权。
这不是这些方法失效,而是**材料不具备让它们生效的结构**。报告 ALFWorld 列时必须写明这一点,
否则读者会把"三条基线数字相同"误读成实现错误或抄袭。

## 5. 本文没有做的事

- 没有训练、没有评测,全部结论来自静态材料分析,GPU 消耗为 0。
- 没有断言 SmartAD/SAD 在 HotpotQA 上会赢或会输,只断言它们在那里**有区别**。
- 宏 vs 微的实际数值差距**未测**,只给出行长分布作为上界线索。

---

## 6. 补充:Kang 在我们的材料上根本跑不起来(2026-09-17 04:10 CDT 追查)

上面第 4 节说"Kang 需先定一个预算 < support 的点"。追进代码后,情况比那更硬:

### 6.1 忠实版 Kang(first-thought prefix)在我们的银行上直接抛错

`src/bfas/rtd/baselines/paper_train.py::PaperTrainer.__init__`:

```python
if method == KANG_PREFIX and any(r.acquisition_method != KANG_PREFIX for r in self.rows):
    raise ValueError("kang_first_thought_prefix requires newly acquired prefixed trajectories; "
                     "use kang_action_list_summary for legacy data")
```

我们两个银行里**每一行的 `acquisition_method` 都是 `None`**(已在训练行上确认)。
**忠实版 Kang 需要用它自己的前缀重新向教师采集**,也就是一笔新的教师开销。

### 6.2 能跑的那个版本,代码自己声明是偏离的

`paper_fidelity.py` 对 `kang_action_list_summary` 的标注:

> `implementation_basis="local_adaptation"`, `fidelity="deviating"`,
> `limitation="NOT the published mechanism: retrospective action-list summary rewrites the first training target."`

**所以不花新钱能跑的 Kang,只有代码自己承认"不是发表的机制"的那个改写版。**

### 6.3 HotpotQA 上 Kang 没有实现

`install_alfworld_kang` 与 `install_bfcl_kang` 存在,**没有 HotpotQA 版本**;
`paper_evaluation.py` 只在 ALFWorld 分支安装投票。HotpotQA 的 Kang 行需要先写实现。

### 6.4 SAD 的配置自证了第 2.2 节的结论

`paper_train.py` 里 SAD 的配置是
`sad=dict(reason_coefficient=.5, act_coefficient=.5, absent_span="renormalize present groups")`。
**"缺失的段组就在present的组上重新归一"** —— ALFWorld 没有 reason 段,于是只剩一组、归一化后就是纯 CE。
这不是我的推断,是实现自己的既定行为。

### 6.5 Kang 这一行的三个选项(需要你定)

| 选项 | 代价 | 得到什么 |
|---|---|---|
| A. 只报 `kang_action_list_summary` | 0 新开销;ALFWorld 可跑,HotpotQA 仍需实现 | 一行明确标注"偏离发表机制"的对照 |
| B. 为忠实 FTP 重新采集 | 一笔新教师采购(量级与已花的 665k token 相当);HotpotQA 还要先写实现 | 忠实的 Kang 行 |
| C. 主表不放 Kang,在正文说明原因 | 0 | 少一条基线,但不会有"偏离版冒充忠实版"的风险 |

我的建议是 **C 或 A**,并且无论哪个都必须在表注里写明限制。
在预算已经紧张、且 Kang 的忠实版还要 HotpotQA 新实现的情况下,B 的性价比最低。

---

## 7. 数值验证:不是读代码推断,是实测相等

用部署树里真实的 `span_ce` 函数,喂随机 logprob,对比三种损失(CPU,`CUDA_VISIBLE_DEVICES=''`):

**ALFWorld 形状的行(单一段类型,占我们全部 1,535 行的 1,535 行):**

| 段类型 | token 数 | SAD | SmartAD | 宏平均 CE | 最大两两差 |
|---|---:|---:|---:|---:|---:|
| action | 4 | 0.371246576309 | 0.371246576309 | 0.371246576309 | **0.00e+00** |
| action | 11 | 0.422877983613 | 0.422877983613 | 0.422877983613 | **0.00e+00** |
| action | 43 | 0.494947185350 | 0.494947185350 | 0.494947185350 | **0.00e+00** |
| final | 4 | 0.455620467663 | 0.455620467663 | 0.455620467663 | **0.00e+00** |
| final | 11 | 0.556312853640 | 0.556312853640 | 0.556312853640 | **0.00e+00** |
| final | 43 | 0.486196770224 | 0.486196770224 | 0.486196770224 | **0.00e+00** |

**逐位相同**,不是"近似相等"。

**HotpotQA 形状的行(reason 与 action/final 混合,占全部 2,079 行的 2,079 行):**

| 组成 | SAD − CE | SmartAD − CE |
|---|---:|---:|
| reason 30 / action 8 / final 4 | −0.019466 | −0.008952 |
| reason 60 / action 10 / final 2 | +0.004948 | −0.000145 |

两者都真实偏离 CE,且彼此不同。

**结论**:第 2.2 节不再是代码阅读得出的推断,而是可复现的数值事实
(脚本 `/work/xueqic/equiv_test.py`,随机种子 0)。
