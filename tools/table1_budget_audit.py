#!/usr/bin/env python3
"""Privileged, read-only source audit and frozen Table 1 replay inventory.

Writes only within results/ (or /tmp). This stage may inspect all archived
content; its public projection discloses IDs/costs only to acquisition.
Historical API completeness is never inferred from retained training rows.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

from tools.table1_common import (BENCHMARKS, DEFAULT_OUT, ROOT, INITIAL_CHECKPOINT,
                                digest, file_hash, read_json, read_rows,
                                row_identity, write_json)


class Sources:
    def __init__(self):
        self.files = {}

    def track(self, path):
        path = Path(path)
        name, checksum = str(path.relative_to(ROOT)), file_hash(path)
        if name in self.files and self.files[name] != checksum:
            raise ValueError(f'source changed during inventory: {name}')
        self.files[name] = checksum
        return path

    def verify(self):
        for name, checksum in self.files.items():
            if file_hash(ROOT/name) != checksum:
                raise ValueError(f'source changed during inventory: {name}')

    def json(self, path):
        return read_json(self.track(path))

    def rows(self, path):
        return list(read_rows(self.track(path)))


def call_record(q, task, teacher, attempt, verified, cost, confidence, source,
                boundary='archived attempt; provider subcall boundaries may be missing'):
    return dict(call_id=q, task_id=task, teacher=teacher, attempt_index=attempt,
                verified=verified, recorded_cost=cost, cost_confidence=confidence,
                source=source, boundary=boundary)


def trainer_row(task, teacher, index, prompt, response, messages=None):
    return dict(task_id=task, teacher=teacher, turn_index=index, prompt=prompt,
                messages=messages or [], response=response, token_hint=max(len(response)//4, 1))


def bfcl_verdict(sources, provenance, cache):
    """Official score files enumerate errors; validate totals before taking complement."""
    path = ROOT / provenance['path'].replace('result_demos_', 'score_demos_').replace('_result.json', '_score.json')
    if path not in cache:
        if not path.is_file():
            cache[path] = None
        else:
            rows = sources.rows(path)
            header = rows[0][1]
            failed = {r['id'] for _, r in rows[1:] if 'id' in r}
            result_rows = sources.rows(ROOT / provenance['path'])
            ids = {r['id'] for _, r in result_rows}
            valid = (len(ids) == header.get('total_count') and failed <= ids and
                     len(failed) == header.get('total_count', 0)-header.get('correct_count', 0))
            cache[path] = (ids, failed) if valid else None
    value = cache[path]
    return (provenance['task_id'] not in value[1]) if value and provenance['task_id'] in value[0] else None


def old_paths(benchmark):
    if benchmark == 'bfcl':
        return [ROOT / f'data/bfcl_sft/pool_bfcl_ds_{arm}.jsonl'
                for arm in ('sft', 'ddpo', 'pbsd', 'bbopd')] + [ROOT / 'data/bfcl_sft/pool_bfcl_star.jsonl']
    if benchmark == 'appworld':
        return sorted((ROOT / 'data/appworld_sft').glob('pool*.jsonl'))
    return []


def bank_inventory(benchmark, sources):
    name = 'v1_1_bfcl' if benchmark == 'bfcl' else 'v1_alfworld_c26'
    bank = ROOT / 'data/rtd' / name
    requests = sources.json(bank / 'public/requests.json')
    bank_audit = sources.json(bank / 'sealed/audit.json')
    integrity = sources.json(bank / 'sealed/integrity.json')
    from bfas.cc_pairs import digest as bank_digest  # stdlib-only archive hash
    snapshots, packages, score_cache = {}, [], {}
    old_sft = sources.rows(ROOT / 'data/bfcl_sft/pool_bfcl_ds_sft.jsonl') if benchmark == 'bfcl' else []
    old_by_task = defaultdict(list)
    for _, r in old_sft:
        old_by_task[r['task_id']].append(r)
    for request in requests:
        q = request['spec']['query_id']
        path = bank / 'sealed' / f'{q}.json'
        payload = sources.json(path)
        if bank_digest(payload) != integrity[q]:
            raise ValueError(f'bank integrity mismatch: {q}')
        p = payload['provenance']
        kind = p['kind']
        task = p.get('task_id', p.get('parent_id'))
        legacy_id = p.get('legacy_id', task)
        available = request.get('unavailable_reason') is None
        historical_response = payload['historical_response']
        if kind == 'demo_attempt':
            verified = bfcl_verdict(sources, p, score_cache)
            verification_source = 'official per-attempt score file (validated error-list complement)'
        elif kind == 'generator_item':
            verified = historical_response.get('verified', historical_response.get('reproduced'))
            verification_source = 'archived independent teacher solve: verified / reproduced'
        else:
            verified = payload.get('success')
            verification_source = 'historical episode verified flag'
        if verified is not None and type(verified) is not bool:
            raise ValueError('recorded verification must be boolean or unknown')
        teacher = ('deepseek-v4-pro-FC' if kind == 'demo_attempt' else
                   'deepseek-v4-pro' if kind == 'generator_item' else
                   payload['historical_response']['teacher'])
        # ALF failures are retained as purchasable no-positive attempts, never filtered by outcome.
        candidate = available if benchmark == 'bfcl' else True
        cost, confidence = payload['cost'], payload['cost_confidence']
        attempt = p.get('attempt_index', p.get('attempt'))
        source = dict(path=p.get('path', p.get('ledger_path')), line=p.get('line'))
        call = call_record(q, task, teacher, attempt, verified, cost, confidence, source,
                           p.get('boundary', 'archived attempt; provider subcall boundaries may be missing'))
        rows = []
        rendering = 'sealed_behavior_text'
        if available and verified is not False:
            for i, behavior in enumerate(payload.get('behaviors', [])):
                state, response = behavior['state'], behavior['text']
                if kind == 'demo_attempt':
                    raw = payload['historical_response']['result']
                    response = raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False)
                    matches = [r for r in old_by_task[task] if r['response'] == response]
                    if matches:
                        row = dict(matches[0])
                        rendering = 'exact legacy row; response matches this attempt raw result'
                    elif old_by_task[task]:
                        # Same official single-turn context; response stays this attempt's raw output.
                        row = dict(old_by_task[task][0])
                        row.update(response=response, token_hint=max(len(response)//4, 1))
                        rendering = 'legacy official task prompt with this attempt raw result'
                    else:
                        # No legacy template exists: preserve the frozen RTD rendering consistently.
                        row = trainer_row(legacy_id, teacher, i, state['prompt'], behavior['text'])
                else:
                    if kind == 'generator_item' and not response:
                        if payload['historical_response'].get('ground_truth') != []:
                            raise ValueError('unexplained empty teacher response')
                        response = '[]'
                        rendering = 'empty teacher ground_truth list serialized as []; sealed RTD text is empty'
                    row = trainer_row(legacy_id, teacher, i, state['prompt'], response)
                if not row['response']:
                    raise ValueError('trainer requires a non-empty response')
                rows.append(row)
        package = dict(package_id=q, task_id=task, legacy_task_id=legacy_id, kind=kind,
                       candidate=candidate, bank_available=available,
                       unavailable_reason=request.get('unavailable_reason'),
                       teacher=teacher, attempt_index=attempt, verified=verified,
                       verification_source=verification_source,
                       recorded_attempt_count=historical_response.get('attempts'),
                       recorded_cost=cost, cost_confidence=confidence, teacher_calls=[call],
                       positive_rows=len(rows), failed_attempt=verified is False,
                       dependencies=request.get('dependencies', []), source=source,
                       response_rendering=rendering,
                       public_class_cap=request['spec']['cost_upper_bound'],
                       missing_costs=payload.get('usage', {}).get('missing_costs', []))
        if kind == 'demo_attempt':
            raw = payload['historical_response']['result']
            package['archived_response_sha256'] = digest(raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False))
        packages.append(package)
        if candidate:
            snapshots[q] = dict(package=package, rows=rows, pbsd_rejected={}, legacy_matches=[],
                                rank_candidates=[], bbopd_matches=[])
    if benchmark == 'bfcl':
        historical = dict(known_total=bank_audit['historical_demo_output_tokens_exact'] +
                          bank_audit['historical_generation_output_tokens_estimated'],
                          exact=bank_audit['historical_demo_output_tokens_exact'],
                          estimated=bank_audit['historical_generation_output_tokens_estimated'],
                          complete=False,
                          missing=['generator rejected drafts, repair and verification calls',
                                   'provider retry/subcall boundaries and reasoning boundaries'],
                          scope='bank source inventory; auxiliary teacher_ledger is a separate collection')
        certificate = sources.json(bank / 'public/cap_certificate.json')
        v11 = sources.json(bank / 'sealed/audit_v11.json')
        if bank_digest(certificate['core']) != certificate['core_hash']:
            raise ValueError('certificate core hash mismatch')
        if bank_digest(sorted(p['package_id'] for p in packages if p['candidate'])) != certificate['core']['usable_set_hash']:
            raise ValueError('certificate usable set mismatch')
        if file_hash(bank / 'public/requests.json') != certificate['public_sha256']:
            raise ValueError('certificate public hash mismatch')
        if file_hash(bank / 'sealed/integrity.json') != certificate['core']['sealed_integrity_sha256']:
            raise ValueError('certificate integrity-index hash mismatch')
        available = [p for p in packages if p['candidate']]
        if (len(available) != 420 or sum(p['recorded_cost'] for p in available) != 55370 or
                any(p['recorded_cost'] > certificate['core']['class_caps'][p['kind']] for p in available)):
            raise ValueError('BFCL certificate inventory or cost mismatch')
        crosscheck = dict(certificate=certificate, bank_audit_v11=v11,
                          certificate_public_and_integrity_hashes_valid=True,
                          available_class_caps_valid=True,
                          current_bank_builder_present=(ROOT/'src/bfas/rtd/bank_v11.py').exists(),
                          current_build_tool_present=(ROOT/'tools/rtd_v11_build_bank.py').exists())
    else:
        historical = dict(known_total=sum(p['recorded_cost'] for p in packages), exact=0,
                          estimated=sum(p['recorded_cost'] for p in packages), complete=False,
                          missing=['raw responses of failures', 'complete retry and per-call usage boundaries'],
                          scope='all 219 archived episode attempts, not a complete provider invoice')
        crosscheck = dict(bank_cap_audit=bank_audit['cap_audit'],
                          failed_attempts_absent_from_RTD_available_pool=112)
    return packages, snapshots, historical, crosscheck


def appworld_inventory(sources):
    base = sources.rows(ROOT / 'data/appworld_sft/pool.jsonl')
    episodes = sources.rows(ROOT / 'data/appworld_sft/episodes.jsonl')
    ds = [r for _, r in base if r['teacher'] == 'deepseek-v4-pro']
    ds_episodes = [r for _, r in episodes if r['teacher'] == 'deepseek-v4-pro']
    if len(ds) != 1482 or len(ds_episodes) != 78:
        raise ValueError('DeepSeek sub-pool differs from frozen 78 episodes / 1482 rows')
    tasks = {r['task_id'] for r in ds}
    by_task = defaultdict(list)
    for row in ds:
        by_task[row['task_id']].append(row)
    packages, snapshots, seen = [], {}, {}
    for name in ('train.jsonl', 'train_debug.jsonl'):
        path = ROOT / 'data/appworld_traces/deepseek-v4-pro' / name
        for line, trace in sources.rows(path):
            if trace['task_id'] not in tasks:
                continue
            content_id = digest(trace)
            source = dict(path=str(path.relative_to(ROOT)), line=line)
            if content_id in seen:
                if seen[content_id]['source']['path'] == source['path']:
                    raise ValueError('ambiguous duplicate attempts within one trace source')
                seen[content_id]['duplicate_source_copies'].append(source)
                continue
            q = digest(['appworld-episode-attempt', content_id])
            calls = []
            for i, turn in enumerate(trace['turns']):
                if turn['role'] == 'assistant':
                    calls.append(call_record(digest([q, 'assistant-call', i]), trace['task_id'],
                        trace['teacher'], trace['attempt'], trace['success'],
                        max(len(turn['content'])//4, 1), 'estimated', dict(**source, turn_index=i),
                        'retained assistant turn; floor(chars/4), minimum 1; missing provider retries'))
            rows = []
            if trace['success']:
                for row in by_task[trace['task_id']]:
                    i = row['turn_index']
                    messages = [{'role': t['role'], 'content': t['content']} for t in trace['turns'][:i]]
                    if (i < len(trace['turns']) and trace['turns'][i]['content'] == row['response']
                            and messages == row['messages']):
                        rows.append(dict(row))
            cost = sum(c['recorded_cost'] for c in calls)
            package = dict(package_id=q, task_id=trace['task_id'], legacy_task_id=trace['task_id'],
                           kind='episode_attempt', candidate=True, bank_available=bool(rows),
                           unavailable_reason=None if rows else 'failed or unused trace attempt',
                           teacher=trace['teacher'], attempt_index=trace['attempt'],
                           verified=trace['success'], recorded_cost=cost, cost_confidence='estimated',
                           teacher_calls=calls, positive_rows=len(rows), failed_attempt=not trace['success'],
                           dependencies=[], source=source, duplicate_source_copies=[])
            seen[content_id] = package
            packages.append(package)
            snapshots[q] = dict(package=package, rows=rows, pbsd_rejected={}, legacy_matches=[],
                                rank_candidates=[], bbopd_matches=[])
    if sum(p['positive_rows'] for p in packages) != 1482:
        raise ValueError('AppWorld trace-to-pool correspondence is incomplete or ambiguous')
    historical = dict(known_total=sum(p['recorded_cost'] for p in packages), exact=0,
                      estimated=sum(p['recorded_cost'] for p in packages), complete=False,
                      missing=['non-debug failed attempts', 'provider retries and exact token usage',
                               'discarded/empty outputs'],
                      scope='78 frozen DeepSeek episodes plus all recoverable failed attempts for those tasks')
    return packages, snapshots, historical, dict(
        retained_episodes=78, retained_rows=1482, retained_row_token_hints=sum(r['token_hint'] for r in ds),
        metrics=sources.json(ROOT / 'data/appworld_traces/deepseek-v4-pro/metrics.json'),
        freeze_rule='78 existing DS episodes plus deduplicated debug attempts for the same tasks; no outcome filtering')


def cross_reference(benchmark, sources, packages, snapshots):
    """Exact row payload matching; task-only matches never authorize teacher evidence."""
    index = defaultdict(list)
    by_task = defaultdict(list)
    archived = defaultdict(list)
    for package in packages:
        if 'archived_response_sha256' in package:
            archived[package['legacy_task_id'], package['archived_response_sha256']].append(package['package_id'])
    for q, snapshot in snapshots.items():
        by_task[snapshot['package']['legacy_task_id']].append(q)
        for i, row in enumerate(snapshot['rows']):
            index[row_identity(row)].append((q, i))
    mappings, ranks, summaries = [], [], []
    for path in old_paths(benchmark):
        matched = rank_count = count = 0
        for line, row in sources.rows(path):
            count += 1
            source = dict(path=str(path.relative_to(ROOT)), line=line)
            ranking = row.get('teacher') in {'rank', 'student_sample'} and '_rejected' in row
            matches = index.get(row_identity(row), []) if not ranking else []
            entry = dict(**source, task_id=row['task_id'], teacher=row.get('teacher'),
                         row_sha256=digest(row), row_fields=sorted(row),
                         package_ids=sorted({q for q, _ in matches}),
                         status='exact_row_counterpart' if matches else 'no_sealed_counterpart')
            entry['counterpart_scope'] = 'content equivalence; not proof that two historical collections share a provider call'
            archived_matches = archived.get((row['task_id'], digest(row['response'])), []) if not ranking else []
            entry['archived_response_package_ids'] = archived_matches
            if archived_matches and not matches:
                entry['status'] = 'archived_response_counterpart_without_eligible_exact_training_row'
            if ranking:
                rank_count += 1
                candidate_ids = sorted(by_task.get(row['task_id'], []))
                rank = dict(**source, task_id=row['task_id'], row_sha256=digest(row),
                            candidate_package_ids=candidate_ids,
                            ranking_call_id=None, recorded_cost=None, cost_confidence=None,
                            teacher='deepseek-v4-pro',
                            status='cached_ranking_output_without_call_cost' if candidate_ids else 'requires new teacher calls',
                            reason='student chosen-response token_hint is NOT the teacher ranking output cost')
                ranks.append(rank)
                entry.update(status=rank['status'], referenced_task_package_ids=candidate_ids)
                # Metadata only. Unknown-cost rankings cannot be revealed as free evidence.
                for q in candidate_ids:
                    snapshots[q]['rank_candidates'].append(rank)
            elif matches:
                matched += 1
                for q, i in matches:
                    baseline = snapshots[q]['rows'][i]
                    differences = sorted(k for k in set(row) | set(baseline) if row.get(k) != baseline.get(k))
                    snapshots[q]['legacy_matches'].append(dict(**source, row_index=i,
                        old_row_sha256=digest(row), differing_fields=differences,
                        prompt_response_equal=True))
                    if 'pbsd' in path.name and row.get('_rejected'):
                        snapshots[q]['pbsd_rejected'][str(i)] = row['_rejected']
                    if 'bbopd' in path.name:
                        snapshots[q]['bbopd_matches'].append(i)
            mappings.append(entry)
        summaries.append(dict(path=str(path.relative_to(ROOT)), rows=count, matched_rows=matched,
                              ranking_rows=rank_count, no_counterpart_rows=count-matched-rank_count))
    return mappings, ranks, summaries


def audit_benchmark(benchmark, out=DEFAULT_OUT):
    sources = Sources()
    packages, snapshots, historical, crosscheck = (
        appworld_inventory(sources) if benchmark == 'appworld' else bank_inventory(benchmark, sources))
    mappings, ranks, summaries = cross_reference(benchmark, sources, packages, snapshots)
    aux = []
    path = ROOT / f'data/teacher_ledger/{benchmark}.jsonl'
    for line, r in sources.rows(path):
        aux.append({k: r.get(k) for k in ('task_id', 'teacher', 'attempt_index', 'verified',
                                         'tokens_spent', 'purpose', 'timestamp')} | {'line': line})
    # In BFCL this is a different, later collection: same task/attempt number is not call identity.
    aux_policy = ('same ledger already represented by sealed packages; do not add again' if benchmark == 'alfworld'
                  else 'separate collection; no proven source identity with selected bank; not added to bank total')
    by_task = defaultdict(list)
    for p in packages:
        by_task[p['task_id']].append(p)
    attribution = []
    for task, group in sorted(by_task.items()):
        failures = [p for p in group if p['failed_attempt']]
        attribution.append(dict(task_id=task, package_ids=[p['package_id'] for p in group],
            historical_recorded_cost=sum(p['recorded_cost'] for p in group),
            failed_attempt_ids=[p['package_id'] for p in failures],
            failed_attempt_recorded_cost=sum(p['recorded_cost'] for p in failures),
            rule='each purchased attempt charged once to its task; historical failures not silently added twice'))
    candidates = sorted((p for p in packages if p['candidate']), key=lambda p: p['package_id'])
    total = sum(p['recorded_cost'] for p in candidates)
    caps = [5537, 13843] if benchmark == 'bfcl' else [9074] if benchmark == 'alfworld' else [(total+2)//4]
    directory = Path(out) / benchmark
    public = dict(schema='table1-public-v1', benchmark=benchmark,
                  packages=[dict(package_id=p['package_id'], recorded_cost=p['recorded_cost']) for p in candidates])
    protocol = dict(benchmark=benchmark, caps=caps, initial_checkpoint=INITIAL_CHECKPOINT,
        acquisition_order_seed=0, training_seeds=[0, 1, 2], order_policy='shared',
        candidate_total_recorded_cost=total, public_sha256=digest(public),
        historical_pool_building=historical, cached_ranking_rows=len(ranks),
        cost_scope='sealed-pool replay spend; historical pool-building total disclosed separately',
        label='estimated-budget replay' if benchmark != 'bfcl' else 'mixed exact/estimated cached-content replay',
        public_cost_exception='per-package recorded costs published for strict-prefix stopping; RTD only publishes class caps',
        candidate_rule=('all 420 RTD-available packages, including unsuccessful demo attempts' if benchmark == 'bfcl' else
                        'all 219 attempts; failures purchasable without positives' if benchmark == 'alfworld' else crosscheck['freeze_rule']))
    totals = dict(candidate_packages=len(candidates), candidate_recorded_cost=total,
                  usable_packages=sum(p['bank_available'] for p in candidates),
                  usable_content_cost=sum(p['recorded_cost'] for p in candidates if p['bank_available']),
                  positive_packages=sum(bool(p['positive_rows']) for p in candidates),
                  positive_rows=sum(p['positive_rows'] for p in candidates),
                  positive_content_cost=sum(p['recorded_cost'] for p in candidates if p['positive_rows']),
                  failed_candidate_attempts=sum(p['failed_attempt'] for p in candidates),
                  failed_candidate_cost=sum(p['recorded_cost'] for p in candidates if p['failed_attempt']),
                  candidate_cost_by_confidence={c: sum(p['recorded_cost'] for p in candidates if p['cost_confidence']==c)
                                                for c in ('exact', 'estimated')},
                  historical_pool_building=historical)
    ledger = dict(schema='table1-audit-v1', benchmark=benchmark, protocol=protocol,
                  totals=totals, packages=packages, failure_cost_attribution=attribution,
                  old_pool_rows=mappings, old_pool_summary=summaries, cached_rankings=ranks,
                  auxiliary_teacher_ledger=dict(path=str(path.relative_to(ROOT)), records=aux,
                      recorded_total=sum(r['tokens_spent'] or 0 for r in aux), policy=aux_policy),
                  rtd_crosscheck=crosscheck, sources=sources.files,
                  audit_journal=dict(stage='privileged_offline_inventory', read_sealed_content=True,
                      influences_order=False, new_teacher_calls=0, gpu_used=False, data_writes=0))
    if benchmark == 'bfcl':
        ledger['old_verified_task_ids'] = sources.json(ROOT / 'data/bfcl_demos_ds_verified.json')
    previous_path = directory / 'ledger.json'
    if previous_path.exists() and read_json(previous_path)['sources'] != sources.files:
        raise ValueError('source inventory changed; use a fresh --out instead of silently refreezing')
    sources.verify()
    for q, snapshot in snapshots.items():
        write_json(directory / 'sealed' / f'{q}.json', snapshot)
    write_json(directory / 'integrity.json', {q: digest(s) for q, s in snapshots.items()})
    write_json(directory / 'ledger.json', ledger)
    write_json(directory / 'protocol.json', protocol)
    write_json(directory / 'public.json', public)
    print(f'{benchmark}: candidates={len(candidates)}, recorded={total}, caps={caps}, '
          f'positive_rows={totals["positive_rows"]}, historical_known={historical["known_total"]}')
    return ledger


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--benchmark', choices=BENCHMARKS)
    ap.add_argument('--out', type=Path, default=DEFAULT_OUT)
    ap.add_argument('--report-only', action='store_true')
    ap.add_argument('--pool-root', type=Path)
    args = ap.parse_args()
    if args.report_only:
        from tools.table1_report import write_report
        write_report(args.out, args.pool_root)
        return
    for benchmark in [args.benchmark] if args.benchmark else BENCHMARKS:
        audit_benchmark(benchmark, args.out)


if __name__ == '__main__':
    main()
