#!/usr/bin/env python3
"""Read-only rev 3.1 report: V0/V1/V2, 10%/25%, official Overall and D5/D6.

Reuse the v1 collector's bounded readers, official CSV validation and ledger
accounting. Committed trajectory rows are authoritative; repeated compute events
are counted as compute, never duplicate purchase windows or evidence of gains.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from bfas.rtd.metrics_v11 import paired_correlations, window_metrics, prediction_errors
from tools.rtd_v1_collect_report import (Inputs, base_score, ledger_charges, number, official_scores,
                                         read_compute, score_cell, table)

ARMS = ('V0', 'V1', 'V2')
BUDGETS = (.10, .25)


def collect(root, *, runs=None, campaigns=None, base_overall=None, controls=()):
    root = Path(root).resolve()
    runs = Path(runs) if runs is not None else root/'results/rtd_v1_1'
    campaigns = Path(campaigns) if campaigns is not None else root/'results/bfcl_std_hpg'
    inputs, arms, manifests, fixed_hashes = Inputs(), [], [], set()
    for arm in ARMS:
        directory = runs/arm
        manifest = inputs.json(directory/'manifest.json')
        config = (manifest or {}).get('config', {})
        if manifest:
            if manifest['arm'] != arm or config.get('protocol_version') != '1.1.0':
                raise ValueError('v1.1 report requires the requested V arm manifest')
            if config.get('benchmark', 'bfcl') != 'bfcl':
                raise ValueError('official Overall table requires BFCL archives')
            if config.get('rounds') != 2 or config.get('budget_checkpoints_bank_fraction') != list(BUDGETS):
                raise ValueError('rev 3.1 report requires two budget points 10%/25%')
            manifests.append((str(directory/'manifest.json'), manifest))
            if manifest.get('fixed_task_set_v11'):
                fixed_hashes.add(manifest['fixed_task_set_v11']['hash'])
        trajectory = inputs.json(directory/'trajectory.json') or {}
        # Live partial runs have durable step files before trajectory is published.
        committed = {(s['round'], s['step']): s for s in trajectory.get('steps', [])}
        if len(committed) != len(trajectory.get('steps', [])):
            raise ValueError('duplicate trajectory step')
        for path in sorted((directory/'steps').glob('r*-s*.json')):
            s = inputs.json(path)
            if s is not None:
                committed.setdefault((s['round'], s['step']), s)
        steps = list(committed.values())
        windows = window_metrics(steps)
        checkpoints = {c['round']: c for c in trajectory.get('checkpoints', [])}
        tags, _, _, links = read_compute(inputs, directory/'compute.jsonl')
        ledger = list(inputs.journal(directory/'teacher.jsonl'))
        charges, _ = ledger_charges(ledger, (manifest or {}).get('budget_ceilings', []), checkpoints, links) if ledger else ([], set())
        validations = [v for s in steps for v in s.get('realised_paired_gain_validation', [])]
        metrics, rows = {}, []
        for r, budget in enumerate(BUDGETS, 1):
            evaluation = inputs.json(directory/f'evaluation-{r}.json')
            if evaluation and evaluation.get('identity', {}).get('checkpoint'):
                if evaluation['identity']['checkpoint'] != checkpoints.get(r):
                    raise ValueError('official evaluation belongs to another checkpoint')
            official = official_scores(inputs, campaigns, evaluation, tags.get(r), r)
            metric = inputs.json(directory/'metrics'/f'round-{r}.json')
            if metric:
                if (metric['round'] != r or metric['arm'] != arm or
                        metric['fixed_task_set_hash'] != manifest['fixed_task_set_v11']['hash']):
                    raise ValueError('round metrics identity mismatch')
                if r in checkpoints and metric.get('parameter_hash') != checkpoints[r]['parameter_hash']:
                    raise ValueError('round metrics belong to another checkpoint')
                metrics[str(r)] = metric
            paid = [c for c in charges if c['round'] <= r]
            latest = max((s for s in steps if s['round'] <= r), key=lambda s: (s['round'], s['step']), default=None)
            spend = sum(c['cost'] for c in paid) if ledger else latest['actual_spend'] if latest else None
            round_validations = [v for s in steps if s['round'] == r for v in s.get('realised_paired_gain_validation', [])]
            additive, realised = [], []
            for s in steps:
                if s['round'] != r:
                    continue
                for v in s.get('realised_paired_gain_validation', []):
                    if v['comparison'] == 'learned_d_vs_zero' and s.get('alpha_d', {}).get('sum_independent_gains') is not None:
                        additive.append(s['alpha_d']['sum_independent_gains']); realised.append(v['realised_paired_gain'])
            denominator = (manifest or {}).get('budget_denominator')
            rows.append(dict(arm=arm, round=r, budget_fraction=budget, actual_spend=spend,
                spend_bank_percent=100*spend/denominator if spend is not None and denominator else None,
                official=official, complete=official['complete'], metrics=metric,
                correlations=paired_correlations(round_validations), joint_vs_additive=prediction_errors(additive, realised)))
        arms.append(dict(arm=arm, manifest=manifest, run=str(directory), rows=rows, windows=windows,
                         correlations=paired_correlations(validations)))
    if len(fixed_hashes) > 1:
        raise ValueError('V arms use different fixed task sets')
    diagnostics = []
    for path in controls:
        report = inputs.json(Path(path))
        if report is None or report.get('schema') != 'rtd-v11-controls-rev31-1':
            raise ValueError('invalid paired controls report')
        arm = next((a for a in arms if a['arm'] == report['arm']), None)
        if arm is None or arm['manifest'] is None:
            raise ValueError('controls require their source arm manifest')
        from bfas.rtd.persistence import manifest_hash
        if report['manifest_hash'] != manifest_hash(arm['manifest']):
            raise ValueError('controls belong to a different run manifest')
        diagnostics.append(report)
    base = base_score(manifests, base_overall)
    return dict(schema='rtd-v11-report-rev31-1', created_at=datetime.now(timezone.utc).isoformat(), root=str(root),
        arms=arms, base=base, controls=diagnostics, inputs=inputs.files, warnings=inputs.warnings,
        scope='single seed; development evaluation; actual spend; cached content cost',
        expected_arms=list(ARMS), budget_points=list(BUDGETS))


def render_markdown(report):
    lines = ['# RTD v1.1 rev 3.1 指标报告', '',
        '单种子开发评测；10%/25% 为累计授权上限，横轴按实际支出。缓存内容成本不等于完整教师账单。', '',
        table(['臂', '10% Overall', '10% 实际支出', '25% Overall', '25% 实际支出'],
            [['发行方基座', number(report['base']['overall']), '—', '—', '—']] +
            [[a['arm'], *[cell for r in a['rows'] for cell in (score_cell(r), number(r['actual_spend'], 0))]] for a in report['arms']]), '',
        'V1−V0：自适应蒸馏整体效果（α 与 d 都改变）；同 α 的 d 对照才隔离来源选择。V2−V1：自适应采购整体效果。', '']
    rows, variance, correlations, windows, repairs = [], [], [], [], []
    for arm in report['arms']:
        for r in arm['rows']:
            m = r['metrics'] or {}
            parse = m.get('parse', {})
            rows.append([arm['arm'], r['round'], number(parse.get('failure_rate'), 5),
                parse.get('samples', '(未完成)'), number(r['joint_vs_additive']['mae'], 5)])
            for name, v in m.get('gradient_variance', {}).get('estimators', {}).items():
                variance.append([arm['arm'], r['round'], name, v['R'], number(v['variance'], 10)])
            for kind, value in r['correlations'].items():
                c = value.get('correlation', value)
                correlations.append([arm['arm'], r['round'], kind, c['n'], number(c['pearson'], 5), c.get('reason') or '—'])
        for w in arm['windows']:
            windows.append([arm['arm'], f"{w['round']}/{w['step']}", w['purchases'], w['spend'],
                w['cumulative_spend'], w['remaining_authorization'], ', '.join(map(str, w['binding_reasons'])),
                number(w['joint_minus_additive'], 6)])
            for pair in w['realised_paired_gain_validation']:
                if pair['comparison'] == 'learned_d_vs_zero':
                    for side in ('full', 'control'):
                        counts = pair[side].get('repair_damage')
                        if counts:
                            repairs.append([arm['arm'], f"{w['round']}/{w['step']}", side, counts['tasks'],
                                            counts['repaired'], counts['damaged'], counts['net_repair']])
    lines += [table(['臂', '轮', '固定集解析失败率', 'T=1 样本数', '实测联合−可加 MAE'], rows), '',
        table(['臂', '轮', '估计器', 'R', '更新方向范数总体方差'], variance), '',
        table(['臂', '轮/步', '购买数', '窗口支出', '累计支出', '剩余授权', '绑定/停止原因', '预测联合−可加'], windows), '',
        table(['臂', '轮/步', '同α分支', '固定槽数', '修复', '损伤', '净修复'], repairs), '',
        table(['臂', '轮', '预测与独立实测', 'n', 'Pearson', '不可估计原因'], correlations), '',
        'ẑ 使用独立坐标干预的 ΔJ/d 验证；F̂ 使用 D9 完整更新/删包补位配对收益。反馈批次与选择 d 的批次分开。',
        '联合−可加区分代理内部差值与实测收益预测误差；误差包含有限更新与 rollout 噪声。', '']
    for arm in report['arms']:
        fixed = (arm['manifest'] or {}).get('fixed_task_set_v11')
        if fixed:
            lines.append(f"{arm['arm']} 固定集：" + '；'.join(
                f"fold{f} 20槽 / {fixed['folds'][f]['unique_states']}独立状态 / {fixed['folds'][f]['repeated_slots']}重复槽" for f in ('0', '1')) + '。')
    for diagnostic in report['controls']:
        from tools.rtd_v11_controls import render_markdown as render_controls
        lines += ['', render_controls(diagnostic)]
    lines += ['', '未完成指标保留为空；不从训练 malformed 计数推断固定集解析失败率。JSON 保留输入哈希、每窗 D9 标签与配对诊断。']
    lines += [f'- {warning}' for warning in report['warnings']]
    return '\n'.join(lines)+'\n'


def write_report(report, out):
    out = Path(out)
    if out.suffix != '.md':
        raise ValueError('report output must be .md')
    paths = [out, out.with_suffix('.json')]
    inputs = {Path(row['path']).resolve() for row in report['inputs']}
    for path in paths:
        if path.exists() or path.resolve() in inputs:
            raise FileExistsError('report outputs must be new files, separate from inputs')
    texts = [render_markdown(report), json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n']
    for path, text in zip(paths, texts):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x') as stream:
            stream.write(text)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--runs', type=Path)
    parser.add_argument('--campaigns', type=Path)
    parser.add_argument('--base-overall', type=float, help='explicit publisher/base Overall, normally 46.06')
    parser.add_argument('--controls', type=Path, action='append', default=[])
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = collect(args.root, runs=args.runs, campaigns=args.campaigns,
            base_overall=args.base_overall, controls=args.controls)
        write_report(result, args.out)
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f'rtd_v11_report: {exc}\n')
    print(args.out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
