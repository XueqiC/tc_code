#!/usr/bin/env python3
"""Read six RTD v1 sealed-replay archives; create a Chinese report, JSON and PNG.

No runner/ledger imports: their recovery readers can repair journals in place.
All outputs must be new files. By default the figure lives beside --out; an
explicit --figure results/figs/rtd_v1_budget_curve.png selects that destination.
Example (from the repository root)::

    PYTHONPATH=src:. .venv/bin/python tools/rtd_v1_collect_report.py \
        --base-overall 46.74 --out docs/rtd_v1_final_report_draft_zh.md
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import tempfile


ARMS = ("R0", "R1", "R1s")
MACHINES = ("rai", "hpg")
ROUNDS = (1, 2, 3)
MISSING = "(未完成)"
BANK_CONTENT = 55_370
AXES = {
    "Overall": "Overall Acc", "Non-Live AST": "Non-Live AST Acc",
    "Live": "Live Acc", "Multi-Turn": "Multi Turn Acc", "Memory": "Memory Acc",
    "Irrelevance": "Irrelevance Detection", "Relevance": "Relevance Detection",
    "Web Search": "Web Search Acc",
}


class Inputs:
    """Bound each journal read to its initial size; record the exact read digest."""

    def __init__(self):
        self.files = []
        self.warnings = []

    def chunks(self, path, *, lines=False):
        try:
            stream = path.open("rb")
        except FileNotFoundError:
            self.files.append({"path": str(path), "missing": True})
            return
        with stream:
            before = os.fstat(stream.fileno())
            remaining = before.st_size
            digest = hashlib.sha256()
            while remaining:
                block = stream.readline(remaining) if lines else stream.read(remaining)
                if not block:
                    break
                remaining -= len(block)
                digest.update(block)
                yield block
            after = os.fstat(stream.fileno())
        changed = (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns)
        self.files.append(dict(path=str(path), size=before.st_size,
                               mtime_ns=before.st_mtime_ns, sha256=digest.hexdigest(),
                               changed_during_read=changed, unread_bytes=remaining))
        if changed or remaining:
            self.warnings.append(f"{path}: 读取期间发生变化；这是逐文件快照，请稍后重新采集。")

    def json(self, path):
        raw = b"".join(self.chunks(path))
        if not raw:
            return None
        try:
            return json.loads(raw)
        except (ValueError, UnicodeError) as exc:
            self.warnings.append(f"{path}: JSON 不完整或无效（{exc}）。")
            return None

    def journal(self, path):
        for number, raw in enumerate(self.chunks(path, lines=True), 1):
            # The runner only admits newline-terminated journal records.
            if not raw.endswith(b"\n"):
                self.warnings.append(f"{path}:{number}: 忽略未结束的末行，原文件未修复。")
                continue
            if raw.strip():
                try:
                    yield json.loads(raw)
                except (ValueError, UnicodeError) as exc:
                    raise ValueError(f"{path}:{number}: invalid journal record") from exc


def percentage(value):
    if value is None or str(value).strip().upper() in {"", "N/A", "NA", "—"}:
        return None
    result = float(str(value).strip().removesuffix("%").strip())
    if not math.isfinite(result) or not 0 <= result <= 100:
        raise ValueError(f"invalid percentage: {value!r}")
    return result


def selection_stats(selection):
    """None is the empty query ID (acquisition.BudgetedSelector.choose)."""
    ids, probabilities = selection["query_ids"], selection["probabilities"]
    if len(ids) != len(probabilities) or ids.count(None) != 1 or len(set(ids)) != len(ids):
        raise ValueError("selection needs unique query_ids and exactly one null empty option")
    if any(not isinstance(q, str) for q in ids if q is not None):
        raise ValueError("package query_ids must be strings")
    p = [float(v) for v in probabilities]
    if any(not math.isfinite(v) or v < 0 for v in p) or not math.isclose(sum(p), 1., abs_tol=1e-8):
        raise ValueError("selection probabilities must be finite, nonnegative and sum to one")
    package_p = [v for q, v in zip(ids, p) if q is not None]
    n, mass = len(package_p), sum(package_p)

    def entropy_ratio(values, support_size):
        if support_size <= 1:
            return None  # H/log(1) is undefined, not evidence of concentration.
        return -sum(v * math.log(v) for v in values if v > 0) / math.log(support_size)

    return dict(candidate_pool_size=n, option_count=len(ids), empty_probability=p[ids.index(None)],
                entropy_ratio=entropy_ratio(p, len(ids)),
                package_entropy_ratio=entropy_ratio([v / mass for v in package_p], n) if mass else None,
                selected=selection.get("selected"))


def summarize(values):
    values = [v for v in values if v is not None]
    return dict(n=len(values), min=min(values), mean=sum(values) / len(values), max=max(values)) if values else None


def read_compute(inputs, path):
    tags, counts, observed, links = {}, {}, set(), {}
    if not path.exists():
        # Still record the missing input.
        list(inputs.journal(path))
        return tags, counts, observed, links
    for event in inputs.journal(path):
        r = event.get("round")
        if r not in ROUNDS:
            continue
        observed.add(r)
        kind = event.get("kind")
        if kind in {"evaluation_begin", "evaluation_reused"} and event.get("tag"):
            tags[r] = event["tag"]  # Last attempt, never search by arm/glob.
        if kind == "request_reveal_link":
            links[event["query_id"]] = r
        if kind == "feedback_rollout":
            row = counts.setdefault(r, dict(feedback_rollouts=0, malformed=0, truncated=0,
                                           malformed_recorded=0, truncated_recorded=0))
            rollout = event["rollout"]
            row["feedback_rollouts"] += 1
            for metric in ("malformed", "truncated"):
                row[metric] += int(bool(rollout.get(metric, False)))
                row[metric + "_recorded"] += int(metric in rollout)
    return tags, counts, observed, links


def official_scores(inputs, campaign_root, evaluation, event_tag, r):
    evaluation = evaluation or {}
    identity = evaluation.get("campaign_identity") or {}
    direct_tag = identity.get("tag")
    # Actual evaluation.py stores output_directory even for reused old campaigns.
    output_tag = Path(evaluation["output_directory"]).name if evaluation.get("output_directory") else None
    if direct_tag and output_tag and direct_tag != output_tag:
        raise ValueError(f"round {r}: campaign tag disagrees with output_directory")
    tag = direct_tag or output_tag or event_tag
    source = "campaign_identity.tag" if direct_tag else "output_directory" if output_tag else "compute event.tag"
    result = dict(tag=tag, tag_source=source if tag else None, csv=None,
                  complete=False, scores={key: None for key in AXES})
    if not tag:
        return result
    if Path(tag).name != tag or tag in {".", ".."} or "\\" in tag:
        raise ValueError(f"invalid campaign tag: {tag!r}")
    path = campaign_root / tag / "data_overall.csv"
    result["csv"] = str(path)
    raw = b"".join(inputs.chunks(path))
    if not raw:
        return result
    rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))
    if len(rows) != 1:
        raise ValueError(f"{path}: expected one official model row, found {len(rows)}")
    result["scores"] = {axis: percentage(rows[0].get(column)) for axis, column in AXES.items()}
    saved = percentage(evaluation.get("overall_accuracy_percent"))
    overall = result["scores"]["Overall"]
    if saved is not None and overall is not None and not math.isclose(saved, overall, abs_tol=.005):
        raise ValueError(f"{path}: Overall disagrees with evaluation-{r}.json")
    result["complete"] = bool(evaluation and evaluation.get("validation", {}).get("complete") is True
                              and overall is not None)
    return result


def ledger_charges(events, ceilings, checkpoints, links):
    """Round ownership binds historical reveals; authorization binds live work."""
    ownership = {}
    for r, checkpoint in sorted(checkpoints.items()):
        for q in checkpoint.get("owned", []):
            ownership.setdefault(q, r)
    current, charges, seen, started = 1, [], set(), set()
    for event in events:
        if event.get("kind") == "authorize":
            budget = event["budget"]
            if budget not in ceilings:
                raise ValueError(f"unknown authorized budget: {budget}")
            current = ceilings.index(budget) + 1
            started.add(current)
        if event.get("kind") != "reveal":
            continue
        q, cost = event["query_id"], event["cost"]
        if q in seen or not isinstance(cost, (int, float)) or not math.isfinite(cost) or cost < 0:
            raise ValueError("duplicate reveal or invalid recorded cost")
        if event.get("confidence") not in {"exact", "estimated"}:
            raise ValueError("reveal missing exact/estimated confidence")
        seen.add(q)
        r = ownership.get(q, links.get(q, event.get("round", current)))
        started.add(r)
        charges.append(dict(round=r, query_id=q, cost=cost, confidence=event["confidence"]))
    for r, checkpoint in checkpoints.items():
        paid = [c for c in charges if c["round"] <= r]
        if (set(checkpoint.get("owned", [])) != {c["query_id"] for c in paid}
                or checkpoint.get("actual_spend") != sum(c["cost"] for c in paid)):
            raise ValueError(f"round {r}: checkpoint ownership/spend disagrees with reveal ledger")
    return charges, started


def rollout_metric(metric, r, counts, steps, report_rows):
    """Report rows are cumulative; missing preceding totals cannot imply zero."""
    sampled = committed = None
    sampled_source = committed_source = None
    coverage = counts.get(r, {})
    total = coverage.get("feedback_rollouts", 0)
    if total and coverage[metric + "_recorded"] == total:
        sampled, sampled_source = coverage[metric], "compute.jsonl"
    decisions = [s for s in steps if s.get("decision")]
    field = "truncation" if metric == "truncated" else "malformed_feedback"
    key = metric + "_rollouts"
    recorded = [s[field][key] for s in decisions if key in s.get(field, {})]
    if decisions and len(recorded) == len(decisions):
        committed, committed_source = sum(recorded), "trajectory.json"
    for category in ("sampled", "committed"):
        if (sampled if category == "sampled" else committed) is not None:
            continue
        report_key = f"{category}_{metric}_rollouts"
        now = report_rows.get(r, {}).get(report_key)
        previous = 0 if r == 1 else report_rows.get(r - 1, {}).get(report_key)
        if now is not None and previous is not None:
            if now < previous:
                raise ValueError(f"round {r}: decreasing cumulative {report_key}")
            if category == "sampled":
                sampled, sampled_source = now - previous, "report/budget_curve.json 差分"
            else:
                committed, committed_source = now - previous, "report/budget_curve.json 差分"
    return dict(sampled=sampled, committed=committed, sampled_source=sampled_source,
                committed_source=committed_source, feedback_rollouts=total,
                flags_recorded=coverage.get(metric + "_recorded", 0),
                sampled_recorded_lower_bound=coverage.get(metric, 0),
                committed_recorded_lower_bound=sum(recorded),
                committed_windows=len(decisions), windows_recorded=len(recorded))


def collect_arm(inputs, directory, campaign_root, machine, arm):
    manifest = inputs.json(directory / "manifest.json")
    config = (manifest or {}).get("config", {})
    if manifest and (config.get("mode") != "sealed_replay" or config.get("benchmark", "bfcl") != "bfcl"):
        raise ValueError(f"{directory}: expected BFCL sealed_replay")
    trajectory = inputs.json(directory / "trajectory.json") or {}
    checkpoints = {c["round"]: c for c in trajectory.get("checkpoints", [])}
    steps = trajectory.get("steps", [])
    teacher_path = directory / "teacher.jsonl"
    teacher = list(inputs.journal(teacher_path))
    tags, counts, observed, links = read_compute(inputs, directory / "compute.jsonl")
    report = inputs.json(directory / "report/budget_curve.json") or {}
    # Per-run reports normally contain one arm; never mix a multi-run report.
    report_rows = {row["round"]: row for row in report.get("rows", [])
                   if Path(row.get("run", str(directory))).name == directory.name}
    ceilings = (manifest or {}).get("budget_ceilings", [])
    charges, started = ledger_charges(teacher, ceilings, checkpoints, links) if teacher_path.exists() else ([], set())
    started |= observed | set(checkpoints) | {s["round"] for s in steps}
    windows_per_round = len(config.get("decision_steps_per_round", [])) or None
    hardware = (manifest or {}).get("hardware", {})
    hardware = hardware.get("hard", hardware)
    result = dict(machine=machine, arm=arm, run=str(directory), manifest_present=manifest is not None,
                  seed=config.get("training_seed"), hardware=hardware,
                  planned_windows=(windows_per_round * config.get("rounds", 3)) if windows_per_round else None,
                  max_packages_per_window=config.get("max_new_packages_per_decision"),
                  config=config, rows=[])
    for r in ROUNDS:
        checkpoint = checkpoints.get(r)
        evaluation = inputs.json(directory / f"evaluation-{r}.json")
        if evaluation:
            bound = evaluation.get("identity", {}).get("checkpoint")
            if bound and (bound.get("round") != r or (checkpoint and bound != checkpoint)):
                raise ValueError(f"{directory}: evaluation-{r}.json belongs to another checkpoint")
        official = official_scores(inputs, campaign_root, evaluation, tags.get(r), r)
        if evaluation:
            started.add(r)
        round_steps = [s for s in steps if s["round"] == r]
        selections = []
        for s in round_steps:
            if s.get("decision") and "probabilities" in s.get("selection", {}):
                selections.append(dict(step=s["step"], **selection_stats(s["selection"])))
        paid = [c for c in charges if c["round"] <= r]
        new = [c for c in charges if c["round"] == r]
        have_spend = teacher_path.exists() and r in started
        spend = sum(c["cost"] for c in paid) if have_spend else None
        ceiling = checkpoint.get("authorized_budget") if checkpoint else ceilings[r - 1] if len(ceilings) >= r else None
        row = dict(machine=machine, arm=arm, round=r, complete=official["complete"],
                   training_complete=checkpoint is not None, started=r in started,
                   official=official, packages_purchased=len(new) if have_spend else None,
                   cumulative_packages=len(paid) if have_spend else None,
                   query_ids=[c["query_id"] for c in new] if have_spend else None,
                   cumulative_query_ids=[c["query_id"] for c in paid] if have_spend else None,
                   actual_spend=spend, cap_ceiling=ceiling,
                   spend_ceiling_ratio=spend / ceiling if spend is not None and ceiling else None,
                   spend_bank_percent=100 * spend / BANK_CONTENT if spend is not None else None,
                   actual_exact=sum(c["cost"] for c in paid if c["confidence"] == "exact") if have_spend else None,
                   actual_estimated=sum(c["cost"] for c in paid if c["confidence"] == "estimated") if have_spend else None,
                   windows=selections, decision_windows=len([s for s in round_steps if s.get("decision")]),
                   selection_summary={key: summarize([w[key] for w in selections]) for key in
                                      ("candidate_pool_size", "empty_probability", "entropy_ratio", "package_entropy_ratio")},
                   rollouts={metric: rollout_metric(metric, r, counts, round_steps, report_rows)
                             for metric in ("malformed", "truncated")})
        result["rows"].append(row)
    return result, manifest


def base_score(manifests, fallback):
    recorded = []
    for path, manifest in manifests:
        for scope, obj in (("manifest", manifest), ("config", manifest.get("config", {}))):
            for key in ("base_overall", "base_overall_accuracy_percent", "base_student_overall"):
                if obj.get(key) is not None:
                    recorded.append((percentage(obj[key]), f"{path}:{scope}.{key}"))
            for key in ("base_student", "base_evaluation"):
                if isinstance(obj.get(key), dict):
                    value = obj[key].get("overall_accuracy_percent", obj[key].get("overall"))
                    if value is not None:
                        recorded.append((percentage(value), f"{path}:{scope}.{key}"))
    if recorded:
        values = {value for value, _ in recorded}
        if len(values) != 1 or None in values:
            raise ValueError("conflicting/invalid recorded base Overall values")
        return dict(overall=recorded[0][0], sources=[source for _, source in recorded])
    if fallback is None or percentage(fallback) is None:
        raise ValueError("No base Overall in manifests/configs; supply --base-overall (e.g. 46.74)")
    return dict(overall=percentage(fallback), sources=["--base-overall"])


def machine_effects(arms):
    indexed = {(a["machine"], a["arm"]): a for a in arms}
    effects = []
    for arm in ARMS:
        rai, hpg = indexed["rai", arm], indexed["hpg", arm]
        for left, right in zip(rai["rows"], hpg["rows"]):
            known = (left["training_complete"] and right["training_complete"]
                     and left["cumulative_query_ids"] is not None and right["cumulative_query_ids"] is not None)
            identical = left["cumulative_query_ids"] == right["cumulative_query_ids"] if known else None
            available = identical is True and left["complete"] and right["complete"]
            effects.append(dict(arm=arm, round=left["round"], identical_purchase_prefix=identical,
                                packages=left["cumulative_packages"] if known else None,
                                rai_overall=left["official"]["scores"]["Overall"],
                                hpg_overall=right["official"]["scores"]["Overall"],
                                hpg_minus_rai=round(right["official"]["scores"]["Overall"] -
                                                    left["official"]["scores"]["Overall"], 6) if available else None))
    return effects


def collect(root, *, base_overall=None):
    root = Path(root).resolve()
    inputs, arms, manifests = Inputs(), [], []
    for machine in MACHINES:
        for arm in ARMS:
            directory = root / ("results/rtd_v1" if machine == "rai" else "results/rtd_v1_hpg") / (
                f"rai_{arm}" if machine == "rai" else arm)
            campaigns = root / ("results/bfcl_std" if machine == "rai" else "results/bfcl_std_hpg")
            result, manifest = collect_arm(inputs, directory, campaigns, machine, arm)
            arms.append(result)
            if manifest:
                manifests.append((str(directory / "manifest.json"), manifest))
    base = base_score(manifests, base_overall)
    seeds = sorted({a["seed"] for a in arms if a["seed"] is not None})
    gpu_models = sorted({a["hardware"]["gpu"] for a in arms if a["machine"] == "rai" and a["hardware"].get("gpu")})
    config_differences = {}
    for arm in ARMS:
        configs = [a["config"] for a in arms if a["arm"] == arm]
        config_differences[arm] = {k: {machine: cfg.get(k) for machine, cfg in zip(MACHINES, configs)}
                                   for k in sorted(set().union(*configs)) if configs[0].get(k) != configs[1].get(k)}
    return dict(schema_version=1, generated_at=datetime.now(timezone.utc).isoformat(),
                root=str(root), base=base, arms=arms, machine_effects=machine_effects(arms),
                scope=dict(expected_arms=6, available_arms=len(manifests), expected_rounds=18,
                           completed_rounds=sum(r["complete"] for a in arms for r in a["rows"]),
                           completed_arms=sum(all(r["complete"] for r in a["rows"]) for a in arms),
                           seeds=seeds, seed_count=len(seeds), recorded_bank_content=BANK_CONTENT,
                           rai_gpu_models=gpu_models, rai_gpu_model_count=len(gpu_models),
                           config_differences=config_differences),
                inputs=inputs.files, warnings=inputs.warnings)


def number(value, digits=2):
    return MISSING if value is None else f"{value:.{digits}f}"


def table(headers, rows):
    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ")
    return "\n".join("| " + " | ".join(map(cell, row)) + " |" for row in
                     [headers, ["---"] * len(headers), *rows])


def score_cell(row, axis="Overall"):
    value = number(row["official"]["scores"][axis])
    return value + (f" {MISSING}" if value != MISSING and not row["complete"] else "")


def overall_table(report):
    return table(["机器", "臂", "r1 Overall", "r2 Overall", "r3 Overall"],
                 [["基座学生", "base", number(report["base"]["overall"]), "—", "—"]] +
                 [[a["machine"], a["arm"], *[score_cell(r) for r in a["rows"]]] for a in report["arms"]])


def range_cell(stats, *, percent=False):
    if stats is None:
        return MISSING
    scale = 100 if percent else 1
    return (f"{stats['mean'] * scale:.3f}" + ("%" if percent else "") +
            f" [{stats['min'] * scale:.3f}, {stats['max'] * scale:.3f}]")


def rollout_cell(row, metric, kind):
    diagnostic = row["rollouts"][metric]
    if diagnostic[kind] is not None:
        return number(diagnostic[kind], 0)
    recorded = diagnostic["flags_recorded" if kind == "sampled" else "windows_recorded"]
    if recorded:
        return f"≥{diagnostic[kind + '_recorded_lower_bound']} {MISSING}"
    return MISSING


def render_markdown(report, figure_link):
    rows = [r for a in report["arms"] for r in a["rows"]]
    scope = report["scope"]
    lines = ["# RTD v1.0 BFCL 六臂汇总（sealed_replay）", "",
             f"采集时间：{report['generated_at']}。官方评测完成 {scope['completed_rounds']}/18 个轮次，"
             f"三轮均完成 {scope['completed_arms']}/6 臂。缺失或未确认完成统一标记 {MISSING}。", "",
             "## 官方 Overall（%）", "", overall_table(report), "",
             "基座来源：" + "；".join(report["base"]["sources"]) + "。基座分轴未提供。", "",
             "## 官方分轴（6 臂 × 3 轮，%）", "",
             table(["机器", "臂", "轮", *AXES],
                   [["基座学生", "base", "0", number(report["base"]["overall"]), *(["未记录"] * 7)]] +
                   [[r["machine"], r["arm"], r["round"], *[score_cell(r, axis) for axis in AXES]] for r in rows]), "",
             "数值只取对应机器 campaign 的 data_overall.csv，不从轴分数重算 Overall。"
             "缺少 evaluation-N.json 的完整验证标记时，已有 CSV 数值仍附未完成标记。", "",
             "## 采购与预算", "",
             "购买数只计 teacher.jsonl 的 reveal；reserve、release 与 empty 不计购买。"
             "累计支出与 exact/estimated 均以 recorded 输出 token 计，不等同完整在线教师账单。"
             "未开始轮次不沿用上一轮支出；正在进行轮次显示已记录的部分值。", "",
             table(["机器/臂", "轮/评测状态", "本轮包", "累计包", "累计支出", "cap ceiling", "支出/ceiling", "exact", "estimated", "支出/55,370"],
                   [[f"{r['machine']}/{r['arm']}", f"r{r['round']}" + ("" if r["complete"] else MISSING),
                     *[number(r[k], 0) for k in ("packages_purchased", "cumulative_packages", "actual_spend", "cap_ceiling")],
                     number(100 * r["spend_ceiling_ratio"], 4) + "%" if r["spend_ceiling_ratio"] is not None else MISSING,
                     number(r["actual_exact"], 0), number(r["actual_estimated"], 0),
                     number(r["spend_bank_percent"], 3) + "%" if r["spend_bank_percent"] is not None else MISSING] for r in rows]), "",
             "## 选择分布与反馈 rollout", "",
             "候选包数排除 query_id=null 的 empty；概率按 query_ids 配对。均值为各已提交决策窗口的等权平均，"
             "方括号为最小/最大值。全分布熵比 = H(p)/ln(K+1)，含 empty；"
             "包条件熵比 = H(p(package | 非 empty))/ln(K)。K≤1 或包概率质量为零时，条件熵比未定义。"
             "JSON 保留逐窗口指标。", "",
             table(["机器/臂", "轮", "已记录分布/计划窗", "候选包数", "empty 概率", "全分布熵比", "包条件熵比"],
                   [[f"{a['machine']}/{a['arm']}", f"r{r['round']}" + ("" if r["complete"] else MISSING),
                     f"{len(r['windows'])}/{a['planned_windows'] // 3 if a['planned_windows'] else '?'}",
                     *[range_cell(r["selection_summary"][k], percent=k == "empty_probability") for k in
                       ("candidate_pool_size", "empty_probability", "entropy_ratio", "package_entropy_ratio")]]
                    for a in report["arms"] for r in a["rows"]]), "",
             "下表均为本轮计数：sampled 包含失败/重复尝试，committed 取提交窗口已去除实际反馈复用的总数。"
             "report/budget_curve.json 的累计字段先做相邻轮差分；缺失旧版本标志或前一轮总数时不推断为零。"
             "只有部分事件记录标志时显示已知下界（≥），同时标记未完成；具体字段覆盖数和来源见 JSON。", "",
             table(["机器/臂", "轮", "malformed sampled", "malformed committed", "truncated sampled", "truncated committed"],
                   [[f"{r['machine']}/{r['arm']}", f"r{r['round']}" + ("" if r["complete"] else MISSING),
                     *[rollout_cell(r, metric, kind) for metric in ("malformed", "truncated")
                       for kind in ("sampled", "committed")]] for r in rows]), "",
             "## 机器效应：相同购买序列的描述性差值", "",
             "逐轮比较截至该轮的累计 reveal query_id 有序序列；两个已完成训练轮次序列一致才纳入。"
             "相同前缀不代表三轮完整轨迹相同。差值为 hpg − rai（百分点）。购买相同不控制采样、"
             "训练数值、软件或配置，不能将差值归因于硬件，也不能据此估计统计噪声带。", "",
             table(["臂", "轮", "累计序列一致", "累计包", "rai Overall", "hpg Overall", "hpg−rai"],
                   [[e["arm"], e["round"], "是" if e["identical_purchase_prefix"] is True else
                     "否" if e["identical_purchase_prefix"] is False else MISSING,
                     number(e["packages"], 0), number(e["rai_overall"]), number(e["hpg_overall"]),
                     number(e["hpg_minus_rai"]) if e["identical_purchase_prefix"] is not False else "不纳入"]
                    for e in report["machine_effects"]]), "",
             "## 实际支出曲线", "", f"![Overall vs actual recorded spend]({figure_link})", "",
             "每个机器/臂一条线，颜色区分 R0/R1/R1s，圆点为 rai，方点为 hpg；仅画已完成且支出可核对的点。"
             "横轴是累计实际 recorded 支出（exact + estimated），不使用 cap ceiling；缺轮留断点。", "",
             "## 范围限制（自动统计）", "",
             f"- 已收到 {scope['available_arms']}/6 臂；已完成 {scope['completed_rounds']}/18 评测轮次。"
             f"记录到 {scope['seed_count']} 个训练种子：{scope['seeds']}。"
             "单种子不能给出跨种子方差、置信区间或显著性结论。",
             "- v1.0 配置为三轮共 12 个决策窗口，每窗口至多 1 包；逐臂实际配置和已提交窗口数如下。",
             "- recorded bank 内容分母固定为 55,370 token；预算 ceiling 来自公开 cap，二者不可混用。"
             "estimated 历史成本不包含不可恢复的拒稿、修复和验证等账单。",
             f"- rai 已记录 {scope['rai_gpu_model_count']} 种 GPU：" + "；".join(scope["rai_gpu_models"]) +
             "。rai 使用混合硬件，臂间差值不构成同硬件对照；相同 GPU 型号也不自动保证其他条件一致。",
             "- 只读取归档证据，未重新运行官方评测、重算模型/数据哈希或修复日志。"
             "并行运行中的输入是逐文件快照，不能保证跨文件同一时刻。", "",
             table(["机器/臂", "seed", "已提交窗/计划窗", "每窗最多包", "最新记录轮", "累计包", "支出/55,370"],
                   [[f"{a['machine']}/{a['arm']}", a["seed"] if a["seed"] is not None else MISSING,
                     f"{sum(r['decision_windows'] for r in a['rows'])}/{a['planned_windows'] or '?'}",
                     a["max_packages_per_window"] if a["max_packages_per_window"] is not None else MISSING,
                     latest["round"] if latest else MISSING,
                     number(latest["cumulative_packages"], 0) if latest else MISSING,
                     number(latest["spend_bank_percent"], 3) + "%" if latest else MISSING]
                    for a in report["arms"]
                    for latest in [next((r for r in reversed(a["rows"]) if r["actual_spend"] is not None), None)]]), "",
             "跨机器配置不同的字段（完整值见 JSON）："]
    lines += [f"- {arm}：" + ("、".join(fields) or "未发现") for arm, fields in scope["config_differences"].items()]
    lines += ["", "## 读取提示与来源", "", "JSON sidecar 保存各轮 campaign tag、CSV 路径、逐窗指标、"
              "购买序列、字段来源和输入文件读取 SHA-256。"]
    lines += [f"- {warning}" for warning in report["warnings"]] or ["未检测到读取期间变更或无效 JSON；缺失文件清单见 JSON。"]
    return "\n".join(lines) + "\n"


def plot_budget_curve(report, stream):
    # Keep matplotlib's font cache out of the repository and live run dirs.
    with tempfile.TemporaryDirectory(prefix="rtd-report-mpl-") as cache:
        previous = os.environ.get("MPLCONFIGDIR")
        os.environ["MPLCONFIGDIR"] = cache
        try:
            import matplotlib
            matplotlib.use("Agg")
            from matplotlib import pyplot as plt
            fig, ax = plt.subplots(figsize=(9, 5.4), layout="constrained")
            colors = {"R0": "#0072B2", "R1": "#D55E00", "R1s": "#009E73"}
            for arm in report["arms"]:
                rows = arm["rows"]
                valid = [r["complete"] and r["actual_spend"] is not None for r in rows]
                ax.plot([r["actual_spend"] if ok else math.nan for r, ok in zip(rows, valid)],
                        [r["official"]["scores"]["Overall"] if ok else math.nan for r, ok in zip(rows, valid)],
                        color=colors[arm["arm"]], marker="o" if arm["machine"] == "rai" else "s",
                        linestyle="-" if arm["machine"] == "rai" else "--",
                        label=f"{arm['machine']} {arm['arm']}")
                for r, ok in zip(rows, valid):
                    if ok:
                        ax.annotate(f"r{r['round']}", (r["actual_spend"], r["official"]["scores"]["Overall"]),
                                    xytext=(4, 6), textcoords="offset points", fontsize=8)
            ax.axhline(report["base"]["overall"], color="#666666", linewidth=1, linestyle=":",
                       label=f"Base {report['base']['overall']:.2f}%")
            ax.set(xlabel="Cumulative recorded spend (output tokens; exact + estimated)",
                   ylabel="Official BFCL Overall (%)", title="RTD v1.0 sealed replay — available completed rounds")
            ax.grid(alpha=.2)
            ax.legend(fontsize=9, ncols=2)
            fig.savefig(stream, format="png", dpi=180)
            plt.close(fig)
        finally:
            if previous is None:
                os.environ.pop("MPLCONFIGDIR", None)
            else:
                os.environ["MPLCONFIGDIR"] = previous


def write_report(report, out, figure):
    out, figure = Path(out).resolve(), Path(figure).resolve()
    sidecar = out.with_suffix(".json")
    root = Path(report["root"])
    targets = [out, sidecar, figure]
    if out.suffix != ".md" or figure.suffix != ".png" or len(set(targets)) != 3:
        raise ValueError("outputs require distinct .md, .json, and .png paths")
    for target in targets:
        if target.exists():
            raise FileExistsError(f"new files only; output already exists: {target}")
        for protected in ("results", "data", "logs", ".git", ".agents", ".codex"):
            if target.is_relative_to((root / protected).resolve()):
                # This one figure path is allowed only when explicitly selected.
                if target == figure and target == root / "results/figs/rtd_v1_budget_curve.png":
                    continue
                raise ValueError(f"read-only input directory: {target}")
    image = io.BytesIO()
    plot_budget_curve(report, image)
    report["outputs"] = dict(markdown=str(out), json=str(sidecar), figure=str(figure))
    markdown = render_markdown(report, Path(os.path.relpath(figure, out.parent)).as_posix())
    contents = [markdown.encode("utf-8"), (json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8"), image.getvalue()]
    for target, content in zip(targets, contents):
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(content)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1], help="repository/archive root")
    parser.add_argument("--out", type=Path, required=True, help="new Chinese Markdown output (JSON sidecar shares stem)")
    parser.add_argument("--figure", type=Path, help="new PNG path; defaults beside --out to keep results read-only")
    parser.add_argument("--base-overall", type=float, help="base score in percent; used only when manifests/configs omit it")
    args = parser.parse_args(argv)
    try:
        report = collect(args.root, base_overall=args.base_overall)
        write_report(report, args.out, args.figure or args.out.with_name("rtd_v1_budget_curve.png"))
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f"rtd_v1_collect_report: {exc}\n")
    print(overall_table(report))
    print(f"\nMarkdown: {args.out}\nJSON: {args.out.with_suffix('.json')}\nFigure: {report['outputs']['figure']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
