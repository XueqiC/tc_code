#!/usr/bin/env python3
"""Union frozen banks, optionally balancing whole packages by supervised tokens."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]

from bfas.rtd.baselines.pi1 import encode_teacher_turn, load_bank, load_config
from bfas.rtd.benchmarks.alfworld_support import audit_verified_bank, seal_verified_bank, FrozenRenderer
from tools.alf_bank_subset import read_json, register_config, training_support, usable_packages
from tools.alfworld_teacher_pool import bank, write_json


def mix_packages(left, right, *, mixing='all', seed=0):
    """Exact subset-sum; integer tolerance, deterministic seeded tie breaking.

    Retain the smaller side in full. Among reachable larger-side token sums
    within [ceil(.95*T), floor(1.05*T)], minimize (abs(sum-T), sum). Seeded
    package order breaks equal-sum ties. Never split or duplicate a package.
    """
    if mixing not in ('all', 'tokens-1:1') or type(seed) is not int:
        raise ValueError('unknown mixing rule or noninteger seed')
    for costs in (left, right):
        if any(not isinstance(q, str) or type(n) is not int or n < 1 for q, n in costs.items()):
            raise ValueError('positive integer per-package supervised token counts required')
    if mixing == 'tokens-1:1' and set(left) & set(right):
        raise ValueError('mixing requires distinct source package IDs; overlapping banks are ambiguous')
    selected = [sorted(left), sorted(right)]
    totals = [sum(left.values()), sum(right.values())]
    if mixing == 'tokens-1:1':
        if not all(totals):
            raise ValueError('token balancing requires nonempty material on both sides')
        larger = 0 if totals[0] >= totals[1] else 1
        target = totals[1 - larger]
        costs = (left, right)[larger]
        ids = sorted(costs)
        random.Random(seed).shuffle(ids)
        lower, upper = (95 * target + 99) // 100, 105 * target // 100
        mask = (1 << (upper + 1)) - 1
        reachable = [1]
        for q in ids:
            prior = reachable[-1]
            reachable.append(prior | ((prior << costs[q]) & mask))
        candidates = [n for n in range(lower, upper + 1) if (reachable[-1] >> n) & 1]
        if not candidates:
            raise ValueError('no whole-package subset can meet the +/-5% token ratio; use all or collect more material')
        total = min(candidates, key=lambda n: (abs(n - target), n))
        kept = []
        for i in range(len(ids) - 1, -1, -1):
            if not (reachable[i] >> total) & 1:
                kept.append(ids[i])
                total -= costs[ids[i]]
        assert total == 0
        selected[larger] = sorted(kept)
    selected_totals = [sum(costs[q] for q in ids) for costs, ids in zip((left, right), selected)]
    return dict(kind=mixing, seed=seed, tolerance='95*T <= 100*selected_larger <= 105*T',
                algorithm='exact subset-sum; nearest total, lower total, seeded package order',
                source_supervised_tokens=totals, selected_supervised_tokens=selected_totals,
                included_package_ids=selected,
                package_supervised_tokens=[dict(sorted(left.items())), dict(sorted(right.items()))])


def package_costs(source, renderer, config):
    audit = audit_verified_bank(source)
    summary = read_json(source / 'sealed/audit.json')
    config = dict(config, sealed_manifest_sha256=audit['manifest_sha256'], support_size=summary['m'],
                  demonstrations=summary['usable_packages'], supervised_turns=summary['command_turns'])
    rows, _ = load_bank(source, config, renderer)
    costs = {}
    for row in rows:
        costs[row.package_id] = costs.get(row.package_id, 0) + len(encode_teacher_turn(
            renderer.adapter._tokenizer, row, config['max_context_tokens'])['target_ids'])
    return costs


def materialize_union(left, right, output, *, mixing='all', seed=0, renderer=None,
                      template=None, model_path=None):
    left, right, output = Path(left).resolve(), Path(right).resolve(), Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    if any(output.resolve().is_relative_to(p) for p in (left, right)):
        raise ValueError('output must be outside both source banks')
    sources = [left, right]
    hashes = [bank.file_hash(p / 'sealed/manifest.json') for p in sources]
    packages = [usable_packages(p) for p in sources]
    for q in packages[0].keys() & packages[1].keys():
        if (left / 'sealed' / f'{q}.json').read_bytes() != (right / 'sealed' / f'{q}.json').read_bytes():
            raise ValueError('overlapping package IDs have different bytes')
    supports = [read_json(p / 'public/support.json') for p in sources]
    # A common reset/request identity is essential when sealed bytes stay unchanged.
    def frozen(support):
        return {k: v for k, v in support.items() if k not in (
            'manifest_hash', 'training_selection', 'training_task_ids', 'fold_task_counts')}
    if frozen(supports[0]) != frozen(supports[1]):
        raise ValueError('union requires the same frozen support, worlds, and request environment')
    resets = [read_json(p / 'public/reset_states.json') for p in sources]
    if resets[0] != resets[1]:
        raise ValueError('union reset states differ')
    config = load_config(template or ROOT / 'configs/rtd/pi1_alfworld_k32_kang.yaml')
    if renderer is None:
        from tools.bfcl_hub_merge_export import _snapshot_for_model
        renderer = FrozenRenderer(model_path or _snapshot_for_model(config['student']))
    costs = [package_costs(p, renderer, config) for p in sources]
    rule = mix_packages(*costs, mixing=mixing, seed=seed)
    payloads, origins = {}, {}
    for source, inventory, ids in zip(sources, packages, rule['included_package_ids']):
        for q in ids:
            payloads[q], origins[q] = inventory[q], source
    if not payloads:
        raise ValueError('union has no usable packages')
    payloads = dict(sorted(payloads.items()))
    support = training_support(supports[0], payloads)
    records = [bank.public_record(q, support['tasks'][p['provenance']['task_id']]['request'])
               for q, p in payloads.items()]
    historical = [read_json(p / 'sealed/audit.json')['historical_inventory'] for p in sources]
    # Retain complete cost provenance; never prorate historical purchases.
    summary = dict(source_files=[dict(path=path, sha256=sha) for path, sha in sorted({
        (f['path'], f['sha256']) for h in historical for f in h['source_files']})],
        source_inventories=historical)
    archive = bank.Archive(records, payloads, {}, [a for p in sources for a in read_json(p / 'sealed/event_aliases.json')], summary)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f'.{output.name}-', dir=output.parent) as tmp:
        staged = Path(tmp) / 'bank'
        seal_verified_bank(staged, archive, support, payloads, resets[0])
        for q, source in origins.items():
            shutil.copyfile(source / 'sealed' / f'{q}.json', staged / 'sealed' / f'{q}.json')
            if bank.file_hash(staged / 'sealed' / f'{q}.json') != read_json(source / 'sealed/manifest.json')['artifacts'][f'sealed/{q}.json']:
                raise ValueError('source package changed during copying')
        manifest_path = staged / 'sealed/manifest.json'
        manifest = read_json(manifest_path)
        manifest['artifacts'] = {name: bank.file_hash(staged / name) for name in manifest['artifacts']}
        manifest['derivation'] = dict(sources=[dict(bank=str(p), sealed_manifest_sha256=sha) for p, sha in zip(sources, hashes)],
                                      mixing=rule, included_package_ids=sorted(payloads), new_teacher_calls=0, new_teacher_tokens=0)
        write_json(manifest_path, manifest)
        audit = audit_verified_bank(staged)
        for source, sha in zip(sources, hashes):
            audit_verified_bank(source, expected_manifest_sha256=sha)
        output.mkdir(exist_ok=False)
        try:
            staged.replace(output)
        except BaseException:
            output.rmdir()
            raise
    return dict(audit=audit, mixing=rule)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--left', type=Path, required=True)
    parser.add_argument('--right', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mixing', choices=('all', 'tokens-1:1'), default='all')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--template', type=Path)
    parser.add_argument('--model-path', type=Path)
    args = parser.parse_args(argv)
    if any(args.config.resolve().is_relative_to(p.resolve()) for p in (args.left, args.right, args.output)):
        raise ValueError('registration must be outside frozen banks')
    if args.config.exists() or args.config.is_symlink():
        raise FileExistsError(args.config)
    result = materialize_union(args.left, args.right, args.output, mixing=args.mixing, seed=args.seed,
                               template=args.template, model_path=args.model_path)
    result['registration'] = register_config(args.output, args.config, template=args.template, model_path=args.model_path)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
