#!/usr/bin/env python3
"""Materialize trainer pools by revealing only a validated purchased prefix.

BFCL defaults to native-fc-v2 in results/table1_audit/pools_native; legacy pools
remain in pools. data/ is read-only. No model, teacher API, or environment runs.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

from tools import table1_common as io
from tools.table1_random_acquisition import acquire
from tools.table1_row_format import (
    NATIVE_ROW_FORMAT, NATIVE_THINK_PREFIX, ROW_FORMATS, bfcl_evaluation_prompt,
    bfcl_native_rejected, read_bank_payload,
    render_package, row_format_for,
)


def action_spans(response):
    """Same fence boundaries as appworld_train.action_spans, without torch imports."""
    spans, cursor = [], 0
    while True:
        opening = response.find('```', cursor)
        if opening < 0:
            return spans
        start = opening + 3
        newline, closing = response.find('\n', start), response.find('```', start)
        if closing < 0:
            return spans
        if newline >= 0 and closing >= newline:
            start = newline + 1
            closing = response.find('```', start)
            if closing < 0:
                return spans
        if start < closing:
            spans.append([start, closing])
        cursor = closing + 3


def build_pools(snapshots, acquisition, protocol, rendered_rows=None, *, row_format=None):
    owned = acquisition['purchased_ids']
    if set(snapshots) != set(owned) or len(set(owned)) != len(owned):
        raise ValueError('exactly the owned packages must be revealed')
    charges, row_origins, base, ranks = [], [], [], {}
    pbsd, bbopd = [], []
    tasks, positive_tasks, failed_ids = set(), set(), []
    for q in owned:
        s = snapshots[q]
        p = s['package']
        if p['package_id'] != q or io.cost_total(p['teacher_calls']) != p['recorded_cost']:
            raise ValueError('package call cost/identity mismatch')
        if p['failed_attempt'] and s['rows']:
            raise ValueError('failed attempt must not yield a positive')
        tasks.add(p['task_id'])
        if p['failed_attempt']:
            failed_ids.append(q)
        charges.extend(p['teacher_calls'])
        for rank in s['rank_candidates']:
            ranks[rank['row_sha256']] = rank
        rows = s['rows'] if rendered_rows is None else rendered_rows[q]
        if len(rows) != len(s['rows']):
            raise ValueError('row rendering changed purchased positive count')
        for i, row in enumerate(rows):
            positive_tasks.add(p['task_id'])
            base.append(deepcopy(row))
            row_origins.append(dict(package_id=q, package_row_index=i, task_id=p['task_id'],
                                    row_sha256=io.digest(s['rows'][i])))
            paired = deepcopy(row)
            if str(i) in s['pbsd_rejected']:
                rejected = s['pbsd_rejected'][str(i)]
                paired['_rejected'] = (bfcl_native_rejected(rejected)
                                       if row_format == NATIVE_ROW_FORMAT else rejected)
            pbsd.append(paired)
            if acquisition['benchmark'] == 'bfcl' or i in s['bbopd_matches']:
                bbopd.append(deepcopy(row))
    spend = io.cost_total(charges)
    if (spend != acquisition['C_m'] or spend > acquisition['cap'] or
            sum(s['package']['recorded_cost'] for s in snapshots.values()) != spend):
        raise ValueError('C_m mismatch, shared calls counted twice, or cap exceeded')
    sad = []
    for row in base:
        tagged = deepcopy(row)
        tagged['_sad_spans'] = dict(action=action_spans(row['response']),
                                    coordinates='character offsets [start,end)',
                                    rule='appworld_train.action_spans; complement is reasoning')
        sad.append(tagged)
    # c is data for the PBSD-agent evidence consumer; it is not an agentkd thought prefix.
    # Generator archives can reuse a task_id for different questions. Native
    # evidence must share the row's exact state, not just that historical alias.
    def evidence_key(row):
        return (row['task_id'], row['prompt']) if row_format == NATIVE_ROW_FORMAT else row['task_id']

    demos = defaultdict(list)
    for row, origin in zip(base, row_origins):
        demos[evidence_key(row)].append(dict(package_id=origin['package_id'],
            state_prompt=row['prompt'], response=row['response'], turn_index=row['turn_index']))
    agent = []
    for row, origin in zip(base, row_origins):
        r = deepcopy(row)
        r['c'] = deepcopy(demos[evidence_key(row)])
        agent.append(r)
    # No archived ranking has a recoverable ranking-call ID and output cost.
    # Task overlap is necessary but insufficient to use extra teacher evidence.
    pools = dict(sft=base, sad=sad, bbopd=bbopd, pbsd_insp=pbsd,
                 ddpo=deepcopy(base), star=[], pbsd_agent=agent)
    manifest = dict(schema='table1-pool-manifest-v1', benchmark=acquisition['benchmark'],
        cap=acquisition['cap'], training_seed=acquisition['training_seed'],
        order_seed=acquisition['order_seed'], order_policy=acquisition['order_policy'],
        initial_checkpoint=io.INITIAL_CHECKPOINT, public_sha256=acquisition['public_sha256'],
        budget_label=protocol.get('label', 'recorded-cost replay'),
        full_historical_call_accounting_complete=protocol['historical_pool_building']['complete'],
        acquisition_sha256=io.digest({k: acquisition[k] for k in
            ('public_sha256', 'purchased_ids', 'purchases', 'cap', 'order_seed', 'training_seed')}),
        purchased_ids=owned, packages_purchased=len(owned), tasks_covered=sorted(tasks),
        positive_tasks=sorted(positive_tasks), positives=len(base),
        failed_attempt_ids=failed_ids, failed_attempt_cost=sum(snapshots[q]['package']['recorded_cost'] for q in failed_ids),
        unique_charged_calls=len({c['call_id'] for c in charges}),
        historical_pool_building=protocol['historical_pool_building'],
        sealed_replay_spend=spend, remaining_budget=acquisition['cap']-spend,
        training_selection='--selection full (budget enforced at acquisition; do not re-budget student response tokens)',
        row_origins=row_origins,
        ranking=dict(total_cached=protocol['cached_ranking_rows'], purchased_task_eligible=len(ranks),
                     reused=0, missing_ranking_call_cost=len(ranks),
                     requires_new_teacher_calls=protocol['cached_ranking_rows']-len(ranks),
                     eligible_rows=list(ranks.values()),
                     decision='exclude unknown-cost extra teacher calls; token_hint measures student answer, not ranking'),
        legacy_equality_checks=[dict(package_id=q, **m) for q, s in snapshots.items() for m in s['legacy_matches']],
        journal=dict(sealed_reader_ids=owned, unpurchased_content_read=False,
                     new_teacher_calls=0, gpu_used=False, data_writes=0), arms={})
    notes = dict(
        sft='purchased positive rows; failed attempts still charged',
        sad='same prompt/response as SFT; supplemental character-span tags; trainer recomputes masks',
        bbopd=('single-turn BFCL equivalence: identical to SFT' if acquisition['benchmark']=='bfcl' else
               'only exact cached on-policy context/response counterparts; missing contexts require new teacher calls'),
        pbsd_insp='cached failed student first turn attached only to exact purchased state; otherwise positive-only',
        ddpo='SFT-only fallback: no cached ranking has an accounted teacher-call cost; not a completed dDPO baseline',
        star='STaR uses no teacher evidence; empty teacher pool, do not pass it to the SFT trainer',
        pbsd_agent='evidence c contains only purchased demos for the same task; existing appworld_train does not consume c')
    for arm, rows in pools.items():
        manifest['arms'][arm] = dict(rows=len(rows), C_m=0 if arm=='star' else spend,
            remaining_budget=acquisition['cap'] if arm=='star' else acquisition['cap']-spend,
            note=notes[arm], pool_sha256=io.digest(rows),
            teacher_call_ids=[] if arm=='star' else sorted({c['call_id'] for c in charges}),
            teacher_evidence_package_ids=[] if arm=='star' else owned,
            pbsd_pairs=sum('_rejected' in r for r in rows),
            trainer_status=('evidence_only_consumer_missing' if arm=='pbsd_agent' else
                            'sft_fallback_missing_rank_costs' if arm=='ddpo' else
                            ('partial_context_matches' if rows else 'requires_new_teacher_calls')
                            if arm=='bbopd' and acquisition['benchmark'] != 'bfcl' else
                            'no_teacher_pool_required' if arm=='star' else 'ready'))
    return pools, manifest


def manifest_without_row_format(manifest):
    """Fields that a rendering-only operation is forbidden to change.

    row_origins hashes continue to identify the immutable sealed source rows;
    arms.*.pool_sha256 identifies the rendered training rows.
    """
    result = deepcopy(manifest)
    result.pop('row_format', None)
    for arm in result['arms'].values():
        arm.pop('pool_sha256', None)
    return result


def replay_native_rankings(directory, pools, manifest):
    """Read-only ledger/sample replay; never invoke sampling or rank APIs.

    Larger-cap samples can cover smaller-cap purchased parents. Charge every
    relevant attempt, including parse failures, using the existing rank ledger
    accounting. Keep parsed student continuations byte-for-byte after prefix.
    """
    from tools import table1_ddpo_rank as ddpo

    ledger = directory / 'ddpo_rank_ledger.jsonl'
    if not ledger.exists():
        return pools, manifest
    tids = set(manifest['tasks_covered'])
    calls = {q: c for q, c in ddpo.read_calls(ledger).items() if c['task_id'] in tids}
    samples = {}
    sample_hashes = {}
    for path in sorted(directory.glob('ddpo_samples_B*.jsonl')):
        sample_hashes[path.name] = io.file_hash(path)
        for _, row in io.read_rows(path):
            if row['task_id'] not in tids:
                continue
            key = (row['task_id'], row['sample_index'])
            if key in samples and samples[key] != row:
                raise ValueError(f'conflicting recorded student sample: {key}')
            samples[key] = row
    grouped = defaultdict(list)
    entries, _ = ddpo.task_entries(tids)
    expected_prompts = {t: bfcl_evaluation_prompt(e['question'][0], e['function'], t)
                        for t, e in entries.items()}
    for key in sorted(samples):
        row = samples[key]
        if (row.get('messages') != [] or row['prompt'] != expected_prompts[row['task_id']]
                or not isinstance(row['response'], str)):
            raise ValueError('student sample differs from evaluation prompt/text contract')
        group = grouped[row['task_id']]
        if row['response'] not in group:
            group.append(row['response'])
    if set(grouped) != tids:
        raise ValueError('ranking replay needs recorded samples for all purchased tasks')
    for call in calls.values():
        tid = call['task_id']
        if (call['candidates'] != grouped[tid]
                or call['context'] != dict(messages=[], prompt=expected_prompts[tid])):
            raise ValueError('ranking candidates/context differ from recorded student samples')
        if call['status'] == 'ranked':
            pair = call['best_worst']
            if (len(pair) != 2 or pair[0] == pair[1]
                    or any(type(i) is not int or not 1 <= i <= len(grouped[tid]) for i in pair)
                    or call['preference_row'] != ddpo.preference(
                        tid, call['context'], grouped[tid], pair)):
                raise ValueError('ranking preference differs from recorded chosen/rejected samples')

    def emission_row(row):
        for field in ('response', '_rejected'):
            # These fields are parser outputs, not decoded FC lists. Do not
            # normalize JSON, repair errors, or infer a teacher continuation.
            if row[field].endswith(('<|im_end|>', '<|endoftext|>')):
                raise ValueError('recorded rank continuation contains a terminal EOS')
            row[field] = NATIVE_THINK_PREFIX + row[field]
        return row

    rendered, _ = ddpo.render_ranked_pools(
        [(None, manifest, pools['sft'])], tids, calls,
        {t for t, answers in grouped.items() if len(answers) >= 2}, row_renderer=emission_row)
    _, manifest, pools['ddpo'] = rendered[0]
    manifest['ranking']['new_ddpo']['row_format'] = NATIVE_ROW_FORMAT
    manifest['ranking']['new_ddpo']['ledger_sha256'] = io.file_hash(ledger)
    manifest['ranking']['new_ddpo']['sample_file_sha256'] = sample_hashes
    manifest['ranking']['new_ddpo']['target_rule'] = (
        'empty think prefix + recorded parsed student continuation, for both chosen and rejected')
    return pools, manifest


def materialize(directory, acquisition_path, pool_root=None, row_format=None):
    directory = Path(directory)
    public = io.read_json(directory / 'public.json')
    protocol = io.read_json(directory / 'protocol.json')
    acquisition = io.read_json(acquisition_path)
    original_acquisition = deepcopy(acquisition)
    expected = acquire(public, acquisition['cap'], acquisition['training_seed'], acquisition['order_seed'])
    for key in ('benchmark', 'purchased_ids', 'purchases', 'C_m', 'remaining_budget', 'public_sha256', 'frozen_order'):
        if acquisition[key] != expected[key]:
            raise ValueError(f'acquisition does not match frozen public prefix: {key}')
    if protocol['public_sha256'] != expected['public_sha256']:
        raise ValueError('protocol/public inventory mismatch')
    integrity = io.read_json(directory / 'integrity.json')
    snapshots = {}
    costs = {p['package_id']: p['recorded_cost'] for p in acquisition['purchases']}
    for q in acquisition['purchased_ids']:
        s = io.read_sealed(directory, q, set(acquisition['purchased_ids']))
        if io.digest(s) != integrity[q] or s['package']['recorded_cost'] != costs[q]:
            raise ValueError('purchased snapshot integrity/cost mismatch')
        snapshots[q] = s
    benchmark = acquisition['benchmark']
    row_format = row_format_for(benchmark, row_format)
    if pool_root is None:
        pool_root = io.DEFAULT_OUT / ('pools_native' if row_format == NATIVE_ROW_FORMAT else 'pools')
    sources = (io.read_json(directory / 'ledger.json')['sources']
               if benchmark != 'appworld' else {})
    rendered = {}
    for q, snapshot in snapshots.items():
        payload = (read_bank_payload(benchmark, q, set(acquisition['purchased_ids']), sources)
                   if benchmark != 'appworld' else None)
        rendered[q] = render_package(benchmark, snapshot, payload, row_format=row_format)
    pools, manifest = build_pools(snapshots, acquisition, protocol, rendered, row_format=row_format)
    manifest['row_format'] = row_format
    _, sealed_manifest = build_pools(snapshots, acquisition, protocol)
    if manifest_without_row_format(manifest) != manifest_without_row_format(sealed_manifest):
        raise ValueError('rendering changed sealed manifest/accounting')
    base_manifest = deepcopy(manifest)
    if row_format == NATIVE_ROW_FORMAT:
        pools, manifest = replay_native_rankings(directory, pools, manifest)
    destination = Path(pool_root) / acquisition['benchmark'] / f'B{acquisition["cap"]}_seed{acquisition["training_seed"]}'
    if (destination / 'manifest.json').exists():
        old = io.read_json(destination / 'manifest.json')
        if old.get('row_format') != row_format and not (
                old.get('row_format') == 'native-fc' and row_format == NATIVE_ROW_FORMAT):
            raise ValueError('row format differs; use a sibling pool directory to preserve existing pools')
        if manifest_without_row_format(old) not in (
                manifest_without_row_format(base_manifest), manifest_without_row_format(manifest)):
            raise ValueError('rendering would change manifest/accounting; refusing to overwrite')
    for arm, rows in pools.items():
        io.write_rows(destination / f'pool_{arm}.jsonl', rows)
    io.write_json(destination / 'manifest.json', manifest)
    # Reporting fields are attached only after all purchases were frozen and paid.
    acquisition.update(tasks_covered=manifest['tasks_covered'], task_count=len(manifest['tasks_covered']),
                       positives=manifest['positives'], failed_attempts=len(manifest['failed_attempt_ids']),
                       failed_attempt_cost=manifest['failed_attempt_cost'], ranking=base_manifest['ranking'])
    acquisition['journal'] = expected['journal'] + [dict(stage='post_purchase_report',
        sealed_reader_ids=acquisition['purchased_ids'], unpurchased_content_read=False)]
    if acquisition != original_acquisition:
        io.write_json(acquisition_path, acquisition)
    print(f'{acquisition["benchmark"]} B={acquisition["cap"]} seed={acquisition["training_seed"]}: '
          f'{len(pools["sft"])} positives, {len(manifest["tasks_covered"])} tasks, '
          f'{manifest["arms"]["ddpo"].get("rank_pairs", 0)} recorded rank pairs, '
          f'dDPO C_m={manifest["arms"]["ddpo"]["C_m"]}')
    return manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--audit-root', type=Path, default=io.DEFAULT_OUT)
    ap.add_argument('--out', type=Path, help='pool root (default: pools_native for native-fc-v2; pools for legacy)')
    ap.add_argument('--benchmark', choices=io.BENCHMARKS)
    ap.add_argument('--row-format', choices=ROW_FORMATS,
                    help='default: native-fc-v2 for BFCL; native-fc is a v2 alias; legacy-messages-v1 otherwise')
    args = ap.parse_args()
    if args.row_format in (NATIVE_ROW_FORMAT, 'native-fc') and args.benchmark != 'bfcl':
        ap.error('native-fc-v2 requires --benchmark bfcl')
    for benchmark in [args.benchmark] if args.benchmark else io.BENCHMARKS:
        directory = args.audit_root / benchmark
        for path in sorted(directory.glob('acquired_B*_seed*.json')):
            materialize(directory, path, args.out, row_format=args.row_format)


if __name__ == '__main__':
    main()
