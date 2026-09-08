"""Chinese report and privileged post-purchase lineage joins. Never selector input."""
from pathlib import Path

from tools.table1_common import BENCHMARKS, DEFAULT_OUT, ROOT, output_path, read_json, write_json


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                      '| ' + ' | '.join(['---']*len(headers)) + ' |'] +
                     ['| ' + ' | '.join(str(x).replace('|', '\\|') for x in row) + ' |' for row in rows])


def write_report(out=DEFAULT_OUT, pool_root=None):
    out = Path(out)
    pool_root = Path(pool_root or out/'pools')
    ledgers = {b: read_json(out/b/'ledger.json') for b in BENCHMARKS}
    acquisitions, manifests = [], []
    for benchmark, ledger in ledgers.items():
        purchased = {}
        for path in sorted((out/benchmark).glob('acquired_B*_seed*.json')):
            a = read_json(path)
            if a['public_sha256'] != ledger['protocol']['public_sha256']:
                raise ValueError('stale acquisition in report directory')
            purchased[path.name] = set(a['purchased_ids'])
            if a['training_seed'] == 0:
                acquisitions.append(a)
                manifests.append(read_json(pool_root/benchmark/f'B{a["cap"]}_seed0/manifest.json'))
        for row in ledger['old_pool_rows']:
            row['purchased_sets'] = {name: sorted(ids & set(row['package_ids'])) for name, ids in purchased.items()}
        for row in ledger['cached_rankings']:
            row['purchased_sets'] = {name: dict(
                task_eligible=bool(ids & set(row['candidate_package_ids'])), reusable=False,
                status='missing ranking call cost' if ids & set(row['candidate_package_ids']) else 'requires new teacher calls')
                for name, ids in purchased.items()}
        write_json(out/benchmark/'ledger.json', ledger)
    inventory = []
    for b, l in ledgers.items():
        t = l['totals']
        inventory.append([b, t['candidate_packages'], t['usable_packages'], t['positive_rows'],
                          t['candidate_recorded_cost'], t['usable_content_cost'],
                          t['failed_candidate_cost'], t['historical_pool_building']['known_total']])
    acquisition_rows = [[a['benchmark'], a['cap'], '0/1/2（共享顺序）', a['packages_purchased'],
                        a['task_count'], a['positives'], a['C_m'], a['remaining_budget'],
                        a['failed_attempts'], a['failed_attempt_cost']] for a in acquisitions]
    rank_rows = [[m['benchmark'], m['cap'], m['ranking']['total_cached'],
                  m['ranking']['purchased_task_eligible'], m['ranking']['missing_ranking_call_cost'],
                  m['ranking']['requires_new_teacher_calls'], m['ranking']['reused']] for m in manifests]
    arm_rows = [[m['benchmark'], m['cap'], arm, v['rows'], v['C_m'], v['pbsd_pairs'], v['trainer_status']]
                for m in manifests for arm, v in m['arms'].items()]
    old_rows = [[b, s['path'].split('/')[-1], s['rows'], s['matched_rows'], s['ranking_rows'], s['no_counterpart_rows']]
                for b, l in ledgers.items() for s in l['old_pool_summary']]
    text = '''# Table 1 统一预算账本与封存池审计（2026-09-08 冻结协议）

已生成 CPU 审计、随机采购与七臂池。现有记录只能支持明确标注的缓存内容回放，不能证明完整历史 API 成本已经恢复，也不能据此认证正在运行的 V0/V1/V2。dDPO 缺排名调用成本，PBSD-agent 缺消费 c 的训练入口，ALFWorld/AppWorld 缺完整同池 BB-OPD 状态对应；manifest 保留这些状态，不能把退化池当成完成的基线。

## 1. 边界与计费口径

- 仅在 `/home/xueqi/hq/projects/tc-alignment-audit`、`table1-audit` 分支工作；不提交，不调用 GPU、环境或 teacher API。`data/`、`envs/`、`.venv` 是共享符号链接。
- `data/` 只读与写 `data/table1_pools/` 的要求冲突，可审阅池改放当前 worktree 的 **results/table1_audit/pools/**。工具拒绝写 data、envs、虚拟环境、控制目录和其他 worktree。
- 本分支缺少指定的 `docs/2026-09-08-table1-audit-and-plan-zh.md`、`src/bfas/rtd/bank_v11.py`、`tools/rtd_v11_build_bank.py`。实际读取方法 §2、旧 bank 实现、v1.1 封存银行、cap certificate、账本与本地配置；未去其他 worktree 取代码。
- 冻结定义：`C_m = Σ c(q)`，用于取得证据的教师调用只付一次，失败、重试、弃用输出也收费。切段、重复 epoch、c 中重复引用不再计费。未知费用不能用 0 或学生答案 token_hint 代替。
- **sealed replay spend** 是本次已购包记录成本；**historical pool-building** 是归档建池记录总额，单列且不与已购成本相加。`complete=false` 表示仍缺完整 provider 账单；estimated 数字不构成数学下界保证。
- 库存审计是获授权的离线特权阶段，读取全库内容/验证信息以生成包与对应关系，不能供选择器使用。采购只读冻结 public.json 的 ID、成本与预算参数；订单冻结后才揭示已购包。journal 的“未读未购内容”适用于采购和池构建，不伪称审计本身没有读全库。
- 公开逐包记录成本是严格前缀停止所需的信息例外；原 RTD 只公开类别 cap、真实记录成本封存。该差异可能改变采购排序和停止。
- 三次训练共享 acquisition seed 0：先按包 ID 排序，再用 `random.Random(0).shuffle`；训练 seed 0/1/2 仅改变训练。下一包超预算立即停止，不跳过、不用余额挑便宜包。
- manifest 指定 `Qwen/Qwen3.5-4B`、`adapter=null`。这是后续训练要求；本次没有加载模型，不声称已核实运行中 checkpoint。

## 2. 库存与历史成本

'''
    text += table(['benchmark', '候选包', 'bank 可用包', '可训练正行', '候选成本', '可用内容成本', '失败成本', '历史已知建池总额'], inventory)
    text += '''

BFCL：698 个封存归档包中 420 个可采购（84 demo attempts + 336 generator items）。55,370 = 21,203 exact + 34,167 estimated；15 次 demo 失败成本 6,140，另有 34 个 generator 条目的独立求解验证失败、成本 2,597。共 49 个失败包、8,737，采购支付但不产正例。69 个成功 demo 加 302 个通过归档验证的 generator 条目，共 371 正行。generator 验证读取原始 verified/reproduced（243 个 reproduced=true、59 个 verified=true、34 个 verified=false），不将 bank 可用性误当成功。34 个旧 verified task ID 是合并库任务级信息；demo 逐尝试结果来自 a1/a2/a3 对应官方 score 文件，先核对分类总数/错误数，再取补集。

BFCL 历史建池已知总额 1,376,334 = 1,233,607 exact demos + 142,727 estimated generation。generator 成本来自归档文件估计，按作者字段字节分摊；item 无法恢复为独立 provider call ID，也缺批次、失败草稿/修复/验证费用。因此是内容回放代理成本，不能声称满足完整逐调用 C_m。另一个 `teacher_ledger/bfcl.jsonl` 有 171 行、16,748，属于另一批采集；相同 task_id/attempt_index 不证明与 a1/a2/a3 是同一调用，不混入银行总额。

ALFWorld：保留 GPT-5.4；219 次 episode attempts = 107 可用 + 112 失败，189,541 = 36,294 可用 + 153,247 失败。B=9,074 保留用户暂定绝对值。每 attempt 成本是多个历史续写的聚合估计，原始逐 API 子调用与失败轨迹缺失；标注 **estimated-budget replay**。失败包进入随机序列，不能从“可用 107 包”开始采购再宣称统一失败计费。

AppWorld：冻结现有 78 个 DeepSeek-V4-pro 成功 episode / 1,482 行（原混合池 128 episodes / 2,506 行）。读取 `appworld_traces/deepseek-v4-pro/train.jsonl` 与 `train_debug.jsonl`：两条 debug 成功轨迹与 train 完全相同，去重；另有 10 次失败、6,981 估算 tokens。冻结候选为 **88 attempts**（成功内容仍是原 78 episodes），成本 **90,533 = 83,552 + 6,981**；提议 **B=22,633**（25%，正整数 half-up）。只冻结 78 个成功包的 25% 为 20,888，但会排除已知失败尝试，本工具不采用该幸存者池。

AppWorld 每个保留 assistant turn 建立带来源行号/turn_index 的合成 call ID，按 `max(len(content)//4,1)` 估计；episode 一次采购、包括其所有保留输出，多个训练行不重复收费。这仍非 provider usage：无 debug 的失败、底层重试/丢弃输出不可恢复；metrics 是一次运行摘要，不能用来杜撰缺失调用。`teacher_ledger/appworld.jsonl` 的 42 行是 GPT-5.4（4,791,617），与 DeepSeek 池不混用。

**包边界限制：**本次保留 BFCL item、ALFWorld/AppWorld episode attempt 的现有银行/采集边界。若严格要求每包等于一次 API 续写而非 episode acquisition bundle，ALFWorld 聚合记录不足以实现；AppWorld 需另冻结带历史前缀依赖的逐调用协议。所有完整 API 账单缺项都保留为 unknown，不编造调用。

各 ledger.json 给出全部包、teacher_calls、来源、attempt_index、verified、成本 confidence、失败归因及未入候选的历史包。failure_cost_attribution 给出每任务全体记录与失败小计；已购失败归其原任务，不将整任务历史失败再次加到成功包。

## 3. 采购结果

正例列是训练行数；BFCL tasks 指官方父任务覆盖数，生成任务另有 legacy_task_id；其余是任务 ID 数。三训练种子订单相同。

'''
    text += table(['benchmark', 'B', '训练 seeds', '已购包', 'tasks', '正行', 'C_m', '余额', '失败尝试', '已购失败成本'], acquisition_rows)
    text += '''

acquired_B<cap>_seed<k>.json 保存完整冻结顺序、逐步累计成本、下一个超预算包/成本与精确余额。任务、正例统计仅在采购完成后写入，不参与选择。

## 4. 旧池对应与缓存排名

合法对应要求 task_id、teacher、turn_index、messages、prompt、response 一致，额外字段逐项登记。内容相等不证明两个采集文件共享同一历史 provider 调用。BFCL 另按封存原始 result 的序列化核实；仅归档响应对应、没有合法训练行时单列 archived_response_package_ids，不允许复用。每条旧行都有物理行号、完整行 hash、对应 package_ids、逐预算 purchased_sets。最后一列是缺少合法训练行对应；详细账本进一步区分“没有封存响应对应”与“响应对应但训练不可用”。

'''
    text += table(['benchmark', '旧池文件', '总行', '合法内容对应', 'rank 行', '其他不可复用行'], old_rows)
    text += '''

旧 BFCL SFT 的 23 行均可核实；旧 dDPO/PBSD/BB-OPD 的 25 条 demo 与当前 23 条 SFT 任务无交集，不能直接作为同池全量基线。BFCL 5 条 rank 所指任务不在已购任务集。AppWorld 混合教师行不能因 task_id 相同映射为 DeepSeek。

排名调用是额外教师证据。bfcl_pair_pools.py、aw_ddpo_rank.py 保存的是教师选中的学生回答；token_hint 是该学生回答长度，不是教师返回排名索引的开销。账本没有可关联的排名 call ID/费用。下表“任务合格”只是必要条件，仍因缺费用而不能复用；其余按要求标记 **requires new teacher calls**。本次无新调用，pool_ddpo 是明确标注的 SFT fallback，不是完成的 dDPO 基线。

'''
    text += table(['benchmark', 'B', '缓存 rank', '已购任务合格', '合格但成本缺失', 'requires new teacher calls', '实际复用'], rank_rows)
    text += '''

## 5. 七臂池与训练入口

'''
    text += table(['benchmark', 'B', 'arm', 'rows', 'C_m', 'PBSD pairs', '状态'], arm_rows)
    text += '''

- **SFT：**只输出已购非失败内容，统一使用上节 `legacy-messages-v1`，不再混入 RTD chat/tool-call 行。原始空调用列表仍序列化为 `[]`，失败包不产正例，教师内容和费用不变。封存 package.response_rendering 描述的是修复前源行，当前训练格式以 manifest.row_format 为准。
- **SAD：**同 SFT prompt/response，加 `_sad_spans` 字符边界，按现有 action_spans 的代码围栏规则；训练器仍自行算 mask。BFCL 无代码围栏响应按现有 trainer 规则属于非 action 部分，本次未另造 BFCL mask。
- **BB-OPD：**单轮 BFCL 与 SFT 逐行相同。AppWorld 只取旧 on-policy 上下文/响应与已购 demo 完全一致的行，是部分内容对应，不代表完整 BB-OPD 或证明当前学生访问这些状态。ALFWorld 无对应缓存，空池。补齐需另冻结 on-policy 状态与费用。
- **PBSD-insp：**只对已购行附加旧缓存的学生失败首轮 `_rejected`，其他正例训练；不做新采样。BFCL 本次 0 对，AppWorld 本次 13 对。
- **dDPO：**缺少可恢复排名调用成本，0 行复用，SFT fallback 状态保留。
- **STaR：**teacher pool 为空、C_m=0、调用/证据 ID 列表为空，不使用公共基线采购的教师证据。不能将空池交给拒绝空输入的 SFT loader；学生自训练另行进行。
- **PBSD-agent：**保留 trainer 基础字段，加 c，只引用相同实际任务的已购 demos（不将 generator 父任务当作生成任务本身）。当前 appworld_train 没有消费 c 的入口；agentkd 的 `_thought` 不是此算法。标为 evidence_only_consumer_missing，不伪称已经可训练。

标准字段为 task_id、teacher、turn_index、messages、prompt、response、token_hint。manifest 给出各臂 call ID 集合、C_m、历史总额、hash、基座、旧行字段差异。除 STaR 外，C_m 包括全部已购包，即使该臂最终未使用某包。

对已采购的非空训练池，使用 `AW_POOL_PATH` 和 **--selection full**。src/appworld_train.py 的 --budget 按学生 tokenizer 计算被选训练行 response tokens；不能代替教师调用费用，也不应再按行重做预算选择。此次只检查输入格式，不执行训练。

## 6. RTD 交叉核对与 Table 1 合规性

| 项目 | 核实事实 | 影响 |
| --- | --- | --- |
| BFCL 分母和证书 | audit_v11：55,370，5,537/13,843/27,685；demo 2048 / generator 512。public/index/core/set hash、逐包 integrity、实际成本不超类上界均核验 | 内容预算一致；不等价完整历史调用账单 |
| Ledger | ledger.py 按 recorded cost、exact/estimated 结算；同 query_id 一次计费；无 verified 免单分支 | 进入 settle 的失败调用会收费 |
| Broker | broker.py 先按 public cap reserve，读 payload、检查非空 behaviors，再 settle；异常 release。ALFWorld 112 失败包 public unavailable，不会提供 | 端到端失败采购收费不一致；不能只看 Ledger 宣布合规。BFCL 可解析的错误 demo 与未通过原始验证的 generator 仍有 behaviors，RTD 可能付费训练；本工具付费但不产正例 |
| cap 与记录成本 | RTD 用 cap 做可买性/预留；本工具用逐包 recorded cost 做严格前缀 | 余额低于 2048/512 时 RTD 可能提前失去候选，批量预留也改变停止和已购集 |
| cost model | 本地 acquisition.py 拟合 recorded/cap，exact noise=1、estimated noise=4，预测裁至 [0,cap]；仅学已购样本 | cap 变化影响初始预测、尺度、价值/成本排序；缺失 v1.1 代码，不能假定运行实现相同 |
| ALFWorld 库与预算 | c26 219 条仅 107 可用；旧审计 usable_public_cap_sum=140,247,040，episode cap=1,310,720 | 与本次 219 attempts/B=9,074 不同，需统一失败候选政策与绝对 cap |
| 随机顺序与窗口 | 本工具全池固定前缀；旧 RTD 有 inner fold、dependency、offer/window 和 cap 限制 | 都称 seed 0 不保证同序列/停止；V1 复用 V0 曝光也不证明符合本次前缀协议 |
| 可执行配置 | 本地 v1_bfcl_c25.yaml 为 1.0.1、usable_public_cap_sum、每窗最多 1 包；缺 v1.1 配置/构建代码 | 方法文档与数据证书可核实，运行实现不能认证；未访问其他 worktree/远端 |
| 初始 checkpoint | 方法 §2 与本工具要求 base Qwen3.5-4B、无 adapter | 当前没有运行中 manifest/checkpoint 副本，V0/V1/V2 实际初始化仍需运行记录 |

BFCL v1.1 证书的内容分母与已结算成本符合指定回放口径，但调用完整性、公开成本信息、cap 停止、失败可买性、随机顺序与实际初始化仍有差异/证据缺口。ALFWorld 旧池排除失败，AppWorld 新冻结池尚未接入 RTD。因此不给运行中 V0/V1/V2 签“完整冻结协议合规”；用户可结合运行 manifest 决定作为缓存内容实验披露，或统一差异后比较。本次未修改 RTD 代码。

## 7. 复现与 CPU 验证

在当前 worktree 执行。PYTHONDONTWRITEBYTECODE 防止写共享依赖 bytecode，无需模型、tokenizer 下载或环境初始化。

```bash
export PYTHONPATH=src:.
export PYTHONDONTWRITEBYTECODE=1
PY=.venv/bin/python
$PY tools/table1_budget_audit.py
$PY tools/table1_random_acquisition.py
$PY tools/table1_pool_from_sealed.py
$PY -m pytest -q -p no:cacheprovider tests/test_table1_audit.py
$PY tools/table1_budget_audit.py --report-only
git status --short
git diff --stat
```

测试覆盖：排序与随机确定性、三训练种子共享顺序、严格前缀、cap/余额/零成本包、失败收费无正例、切段不重计、STaR 零成本、缺费用 rank 不免费、PBSD/c 同任务限制、篡改采购拒绝、仅已购文件可读、只读路径保护、旧池 prompt/response 完全一致与其他字段逐项比较、SAD 与实际 trainer 函数一致。随机采购测试将 sealed reader 和文件读取均 monkeypatch 为抛异常，仍能完成采购。真实归档对应测试在本次生成结果上执行。

results/ 被 .gitignore 忽略；产物在磁盘保留供审阅，未暂存/提交。ledger.sources 记录输入 SHA-256；再次审计若源文件变化则拒绝覆盖冻结库存。
'''
    validation_path = out/'validation.json'
    if validation_path.exists():
        validation = read_json(validation_path)
        text += f'\n初次预算审计 CPU 验证：**{validation["tests"]} tests，{validation["failures"]} failures，{validation["errors"]} errors，{validation["skipped"]} skipped**。当时非空池仅通过实际 trainer 的 load_pool 字段验证（只抽取该函数，不导入 GPU 栈），不包含 encode。JUnit 记录：results/table1_audit/cpu_tests.xml；本次真实 load_pool + encode 的回归见“池行格式”。\n'
    report_path = ROOT/'docs/table1_budget_ledger_zh.md'
    # These implementation notes are maintained with their respective tools;
    # refreshing accounting tables must not erase the repaired row contract or
    # the existing dDPO instructions.
    if report_path.exists():
        previous = report_path.read_text(encoding='utf-8')
        for start, end in (
            ('## 池行格式\n', '## 5. 七臂池与训练入口\n'),
            ('### 5.1 BFCL dDPO 排名补采工具（待用户运行）\n', '## 6. RTD 交叉核对与 Table 1 合规性\n'),
        ):
            if start in previous:
                section = previous.split(start, 1)[1].split(end, 1)[0]
                text = text.replace(end, start + section + end, 1)
    output_path(report_path).write_text(text, encoding='utf-8')
