#!/usr/bin/env python3
"""Run rev 3.1 paired controls on a locally archived frozen window (one GPU).

CPU tests inject the tiny backend through run_controls; this CLI loads the same
production backend as training. No teacher purchase, training commit or RL step.
"""
import argparse
import json
from pathlib import Path

from bfas.rtd.controls_v11 import load_window, place_window, run_controls
from bfas.rtd.persistence import ComputeJournal, atomic_json, file_hash, tree_hash, digest
from tools.rtd_v1_collect_report import number, table


def render_markdown(report):
    rows = []
    for pair in report['pairs']:
        for side in ('full', 'control'):
            r = pair[side]
            m = r['metrics']
            counts = m['repair_damage']
            rows.append([pair['comparison'], side, number(r['mean_return'], 4),
                number(m['parse']['failure_rate'], 4), counts['repaired'], counts['damaged'],
                number(m['teacher_mass'], 5), number(pair['realised_paired_gain'], 5)])
    for row in report['source_estimator_diagnostics']:
        m, counts = row['metrics'], row['metrics']['repair_damage']
        rows.append(['source estimator (last)', row['estimator'], number(row['mean_return'], 4),
            number(m['parse']['failure_rate'], 4), counts['repaired'], counts['damaged'],
            number(m['teacher_mass'], 5), '—'])
    variance = report['gradient_variance']['estimators']
    return '\n'.join([
        '# RTD v1.1 rev 3.1 配对干预', '',
        f"{report['arm']} r{report['round']} s{report['step']}；冻结 start `{report['start_hash']}`。",
        '每个分支从同一检查点执行一步；独立温度1反馈只用于验证。greedy 修复/损伤按固定槽位计数。',
        '同 α 的 learned d vs d=0 隔离来源选择；V1−V0 是自适应蒸馏整体效果。', '',
        table(['比较', '分支', '留出 J', '解析失败率', '修复', '损伤', '教师质量', '配对 ΔJ'], rows), '',
        '来源重采样的更新位移范数总体方差（correction=0）；共同冻结窗口起点，完整 R 个范数见 JSON。', '',
        table(['估计器', 'R', '平均范数', '范数方差'],
              [[name, v['R'], number(v['mean_norm'], 8), number(v['variance'], 10)] for name, v in variance.items()]), '',
        'F̂ 删除包时使用 D9 的预定旧池补位；这类采购对比不属于同 α 的机制隔离。',
        '语法过滤是有偏诊断：保留合法硬样本，全失败时用软来源；教师目标和质量保留。', '',
        table(['预测验证', 'n', 'Pearson'], [[k, v.get('n', v.get('correlation', {}).get('n')),
            number(v.get('pearson', v.get('correlation', {}).get('pearson')), 5)] for k, v in report['correlations'].items()]), '',
        f"固定槽位独立状态数：fold0={report['fixed_task_set']['folds']['0']['unique_states']}，"
        f"fold1={report['fixed_task_set']['folds']['1']['unique_states']}。不足20时重复项明确保存在 manifest。", '',
        '缺少变化/恒定收益时相关系数为 null；不将它解释为零相关或机制有效。', ''])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--window', type=Path, required=True, help='controls/windows/rN-sNN.pt and adjacent hash descriptor')
    parser.add_argument('--out', type=Path, required=True, help='new output directory')
    parser.add_argument('--resamples', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--z-probes', type=int, default=4)
    args = parser.parse_args(argv)
    if args.resamples < 2 or args.z_probes < 1:
        parser.error('resamples >= 2 and z-probes >= 1 required')
    # Validate frozen archive before loading a production model or creating output.
    payload = load_window(args.window)
    if args.out.exists():
        parser.error('output directory already exists; choose a new directory')
    from bfas.rtd.cli import ROOT, BFCLSupport, _checker_context
    from bfas.rtd.runtime import load_backend
    from bfas.rtd.identity import evaluation_harness_identity
    config, manifest = payload['manifest']['config'], payload['manifest']
    if config.get('benchmark', 'bfcl') != 'bfcl':
        parser.error('CLI currently uses BFCL support; other adapters use run_controls directly')
    if tree_hash(Path(manifest['model_path'])) != manifest['base_checkpoint_hash']:
        parser.error('frozen base model content changed')
    if digest(evaluation_harness_identity(ROOT, config)) != manifest['harness_hash']:
        parser.error('frozen evaluation harness changed')
    args.out.mkdir(parents=True)
    journal = ComputeJournal(args.out/'compute.jsonl', cuda=True)
    backend = load_backend(config, manifest, journal)
    # Restore the archived start functionally; backbone installation is unnecessary.
    payload = place_window(payload, device='cuda:0')
    support = BFCLSupport(ROOT, config)
    with _checker_context() as checker:
        report = run_controls(payload, backend, support, checker, journal,
            R=args.resamples, seed=args.seed, z_probes=args.z_probes)
    report['input'] = dict(window=str(args.window.resolve()), sha256=file_hash(args.window))
    atomic_json(args.out/'controls.json', report)
    (args.out/'controls.md').write_text(render_markdown(report))
    print(args.out/'controls.md')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
