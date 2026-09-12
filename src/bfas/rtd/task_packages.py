"""Per-task attempt ledgers and frozen attempt purchases; no runtime raw pool."""
from collections import defaultdict
import json
import random

ATTEMPT_LEDGER = 'task_attempts.json'
PURCHASE_BASIS = 'recorded_teacher_attempt'
ORDER_RULE = 'sorted query IDs; random.Random(0).shuffle; stop before first overflow'


def frozen_attempt_order(query_ids):
    order = sorted(query_ids)
    if len(set(order)) != len(order):
        raise ValueError('duplicate attempt IDs')
    random.Random(0).shuffle(order)
    return order


def attempt_summary(row, *, confidence):
    from .bank_build import ledger_cost
    if row.get('usage_status') == 'reported':
        confidence = 'exact'
    tokens, confidence = ledger_cost(row, default_confidence=confidence)
    usage = row.get('usage', {})
    if 'completion_tokens' in usage and usage['completion_tokens'] != tokens:
        raise ValueError('attempt ledger tokens differ from completion usage')
    if row.get('usage_status') == 'estimated':
        confidence = 'estimated'
    tid = row['task_id']
    if not isinstance(tid, str) or not tid:
        raise ValueError('attempt requires a task ID')
    return dict(task_id=tid, attempt_index=row['attempt_index'], tokens=tokens,
                verified=row['verified'], cost_confidence=confidence)


def build_attempt_ledger(records, payloads):
    """Summarize the immutable attempts, including unavailable/protected rows.

    Existing sealed IDs, costs, rendering, and the usable fraction denominator
    remain archival identities. Task groups express parentage, not joint charges.
    """
    grouped, seen = defaultdict(list), set()
    for record in records:
        q = record.spec.query_id
        payload = payloads[q]
        attempts = payload.get('historical_attempts', [payload['historical_response']])
        summaries = []
        for row in attempts:
            summary = attempt_summary(row, confidence=payload['cost_confidence'])
            key = summary['task_id'], summary['attempt_index']
            if key in seen:
                raise ValueError('duplicate task attempt in sealed inventory')
            seen.add(key)
            summaries.append(dict(summary, query_id=q))
        tid = payload['provenance']['task_id']
        if (not summaries or any(r['task_id'] != tid for r in summaries)
                or sum(r['tokens'] for r in summaries) != payload['cost']):
            raise ValueError('sealed task/cost differs from attempt ledger')
        selected_index = payload['historical_response']['attempt_index']
        grouped[tid].append((record, selected_index, summaries))
    tasks = []
    for tid, members in sorted(grouped.items()):
        parents = {r.parent_hash for r, _, _ in members}
        if len(parents) != 1 or any(r.dependencies for r, _, _ in members):
            raise ValueError('task ledger requires one parent and no cross-package dependencies')
        usable = [(r, i) for r, i, _ in members if r.unavailable_reason is None]
        selected, _ = min(usable or [(r, i) for r, i, _ in members], key=lambda v: (v[1], v[0].spec.query_id))
        attempts = sorted([a for _, _, rows in members for a in rows], key=lambda a: a['attempt_index'])
        tasks.append(dict(task_id=tid, query_id=selected.spec.query_id, parent_hash=selected.parent_hash,
            query_ids=sorted(r.spec.query_id for r, _, _ in members), attempts=attempts,
            tokens=sum(a['tokens'] for a in attempts), usable=bool(usable),
            cost_confidence='estimated' if any(a['cost_confidence'] == 'estimated' for a in attempts) else 'exact'))
    order = [t['task_id'] for t in tasks]
    random.Random(0).shuffle(order)
    return dict(version=2, purchase_unit='teacher_attempt', cost_basis=PURCHASE_BASIS,
        purchase_seed=0, order_rule=ORDER_RULE, task_order=order, tasks=tasks,
        attempt_order=frozen_attempt_order(payloads))


def validate_attempt_ledger(value, records):
    """Validate only certificate-bound public metadata, never teacher payloads."""
    # Existing v1 files remain byte-identical archival ledgers. Their obsolete
    # task purchase metadata never controls runtime purchase order or prices.
    legacy = (value['version'] == 1 and value['purchase_unit'] == 'task'
        and value['cost_basis'] == 'all_recorded_task_attempts'
        and value['order_rule'] == 'sorted task IDs; random.Random(0).shuffle; stop before first overflow')
    current = (value['version'] == 2 and value['purchase_unit'] == 'teacher_attempt'
        and value['cost_basis'] == PURCHASE_BASIS and value['order_rule'] == ORDER_RULE
        and value['attempt_order'] == frozen_attempt_order(records))
    if not (legacy or current) or value['purchase_seed'] != 0:
        raise ValueError('unknown attempt ledger protocol')
    seen, tids = set(), []
    for task in value['tasks']:
        tids.append(task['task_id'])
        ids = task['query_ids']
        if (not ids or ids != sorted(set(ids)) or seen.intersection(ids)
                or not set(ids) <= records.keys() or task['query_id'] not in ids
                or any(records[q].parent_hash != task['parent_hash'] or records[q].dependencies for q in ids)):
            raise ValueError('task purchase inventory/parent mismatch')
        seen.update(ids)
        attempts = task['attempts']
        indices = [a['attempt_index'] for a in attempts]
        if (not attempts or indices != sorted(set(indices))
                or any(type(i) is not int or i < 0 for i in indices)
                or {a['query_id'] for a in attempts} != set(ids)
                or any(a['task_id'] != task['task_id'] or type(a['tokens']) is not int or a['tokens'] < 0
                       or type(a['verified']) is not bool or a['cost_confidence'] not in {'exact', 'estimated'}
                       for a in attempts)
                or type(task['tokens']) is not int or task['tokens'] != sum(a['tokens'] for a in attempts)
                or type(task['usable']) is not bool
                or task['usable'] != any(records[q].unavailable_reason is None for q in ids)
                or task['usable'] != (records[task['query_id']].unavailable_reason is None)
                or task['cost_confidence'] != ('estimated' if any(a['cost_confidence'] == 'estimated' for a in attempts) else 'exact')):
            raise ValueError('invalid task attempt accounting')
    order = sorted(tids)
    random.Random(0).shuffle(order)
    if seen != records.keys() or len(set(tids)) != len(tids) or value['task_order'] != order:
        raise ValueError('task purchase order/inventory mismatch')
    return value


def attempt_packages(value, records):
    """Flatten validated parent groups without combining sibling attempts."""
    validate_attempt_ledger(value, records)
    packages = {}
    for task in value['tasks']:
        for attempt in task['attempts']:
            q = attempt['query_id']
            if q in packages:
                raise ValueError('sealed package contains multiple attempts; rebuild the bank')
            packages[q] = dict(attempt, parent_hash=task['parent_hash'],
                usable=records[q].unavailable_reason is None)
    return packages


def rebuild_task_accounting(bank):
    """Reindex an existing certified bank offline, with a byte-preservation gate.

    The archived ledger rows already reside in sealed/. No tokenizer, teacher,
    environment replay, or raw-pool reconstruction is needed for this rebuild.
    """
    from dataclasses import replace, asdict
    from pathlib import Path
    import shutil
    from tempfile import TemporaryDirectory
    from .bank_build import validate_state_certificate
    from .broker import RequestRecord
    from .selector import PublicQuerySpec, PublicFeatures
    from .persistence import atomic_json, digest, file_hash
    from ..cc_pairs import digest as payload_digest
    bank = Path(bank).resolve()
    cert = validate_state_certificate(bank)
    if cert['core']['benchmark'] not in {'alfworld', 'bfcl', 'hotpotqa'}:
        raise ValueError('unsupported task bank')
    records, payloads = [], {}
    integrity = json.loads((bank/'sealed/integrity.json').read_text())
    for row in json.loads((bank/'public/requests.json').read_text()):
        fields = dict(row['spec'])
        features = fields.pop('features')
        record = RequestRecord(PublicQuerySpec(**fields, features=PublicFeatures(**features)),
            row['parent_hash'], tuple(row['dependencies']), row['unavailable_reason'])
        records.append(record)
        q = record.spec.query_id
        payloads[q] = json.loads((bank/'sealed'/f'{q}.json').read_text())
        if payload_digest(payloads[q]) != integrity[q]:
            raise ValueError('sealed integrity mismatch before accounting rebuild')
    summary = build_attempt_ledger(records, payloads)
    validate_attempt_ledger(summary, {r.spec.query_id: r for r in records})
    mutable = {'public/requests.json', 'public/cap_certificate.json',
               'public/' + ATTEMPT_LEDGER, 'sealed/audit_v11.json'}
    before = {str(p.relative_to(bank)): file_hash(p) for p in bank.rglob('*') if p.is_file()}
    protected = {name: sha for name, sha in before.items() if name not in mutable}
    with TemporaryDirectory(prefix='.task-accounting-', dir=bank.parent) as scratch:
        candidate = Path(scratch)/'candidate'
        shutil.copytree(bank, candidate)
        atomic_json(candidate/'public'/ATTEMPT_LEDGER, summary)
        core = cert['core']
        core['public_artifacts'][ATTEMPT_LEDGER] = file_hash(candidate/'public'/ATTEMPT_LEDGER)
        core_hash = digest(core)
        for i, r in enumerate(records):
            if r.unavailable_reason is None:
                provenance = json.loads(r.spec.cap_provenance)
                provenance['certificate_core_hash'] = core_hash
                records[i] = replace(r, spec=replace(r.spec, cap_provenance=json.dumps(provenance, sort_keys=True)))
        atomic_json(candidate/'public/requests.json', [asdict(r) for r in records])
        atomic_json(candidate/'public/cap_certificate.json', dict(core=core, core_hash=core_hash,
            public_sha256=file_hash(candidate/'public/requests.json')))
        audit = json.loads((candidate/'sealed/audit_v11.json').read_text())
        audit['cap_certificate_sha256'] = file_hash(candidate/'public/cap_certificate.json')
        atomic_json(candidate/'sealed/audit_v11.json', audit)
        validate_state_certificate(candidate)
        changed = [name for name, sha in protected.items() if file_hash(candidate/name) != sha]
        changed += [name for name, sha in before.items() if file_hash(bank/name) != sha]
        if changed:
            raise ValueError('bank/support/folds changed; refusing rebuild: ' + ', '.join(changed))
        # All unchanged public artifacts (including the entire support/folds and
        # reset states) and sealed payload bytes have passed before replacement.
        previous = Path(scratch)/'previous'
        bank.rename(previous)
        try:
            candidate.rename(bank)
        except BaseException:
            previous.rename(bank)
            raise
    return dict(bank=str(bank), support_sha256=protected['public/support.json'],
        preserved_artifacts=protected, support_and_folds_byte_identical=True,
        budget_denominator=core['budget_denominator'], tasks=len(summary['tasks']),
        attempts=sum(len(t['attempts']) for t in summary['tasks']),
        certificate_sha256=file_hash(bank/'public/cap_certificate.json'))
