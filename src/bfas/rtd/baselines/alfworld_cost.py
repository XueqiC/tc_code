"""Read-only accounting of ALFWorld collection ledgers, never public cost caps."""
from decimal import Decimal
import json
from pathlib import Path

from ..persistence import file_hash


def read_json(path):
    return json.loads(Path(path).read_text())


def collection_cost(bank, *, method, collection=None):
    """Charge each request once at its last ledger state, including failed buys.

    Reservations are updates to request IDs, not extra calls. An unresolved
    reservation cannot be presented as ACTUAL cost: fail instead of guessing.
    Monetary cost is an estimate at the collector's recorded prices.
    """
    bank = Path(bank).resolve()
    collection = Path(collection or str(bank) + '.collection').resolve()
    identity = read_json(collection/'identity.json')
    expected = {'smartad': 'smartad', 'sad': 'plain', 'ce': 'plain', 'kang': 'kang-ftp'}[method]
    if identity.get('method', 'plain') != expected:
        raise ValueError('method and collection identity differ')
    ledger, usage = collection/'teacher_ledger.jsonl', collection/'usage.jsonl'
    ledger_hash, usage_hash = file_hash(ledger), file_hash(usage)
    audit = read_json(bank/'sealed/audit.json')
    sources = audit['historical_inventory']['source_files']
    if len(sources) != 1 or sources[0]['sha256'] != ledger_hash:
        raise ValueError('bank and teacher ledger differ; require a frozen collection')
    support = read_json(bank/'public/support.json')
    task_ids = set(support['training_task_ids'])
    if task_ids != set(support['historical_task_ids']):
        raise ValueError('K=32 collection must contain training tasks only')
    latest = {}
    for line in usage.read_text().splitlines():
        call = json.loads(line)
        previous = latest.get(call['id'])
        if previous and any(previous.get(k, 'trajectory') != call.get(k, 'trajectory')
                            for k in ('task_id', 'attempt_index', 'phase')):
            raise ValueError('request ID reused across tasks/attempts/phases')
        latest[call['id']] = call
    calls = list(latest.values())
    if not calls or any(c['task_id'] not in task_ids for c in calls):
        raise ValueError('empty or out-of-support usage ledger')
    for call in calls:
        if call['status'] != 'reported':
            raise ValueError('actual cost unavailable: unresolved request reservation')
        u = call['usage']
        if (any(type(u[k]) is not int or u[k] < 0 for k in
                ('prompt_tokens', 'cached_tokens', 'completion_tokens'))
                or u['cached_tokens'] > u['prompt_tokens']):
            raise ValueError('invalid reported usage')
        if call.get('phase', 'trajectory') not in {'cot', 'trajectory'}:
            raise ValueError('unknown purchase phase')
    rates = [Decimal(str(x)) for x in identity['prices']]
    if len(rates) != 3 or any(not r.is_finite() or r < 0 for r in rates):
        raise ValueError('invalid recorded prices')

    def total(items):
        counts = {k: sum(c['usage'][k] for c in items)
                  for k in ('prompt_tokens', 'cached_tokens', 'completion_tokens')}
        usd = ((counts['prompt_tokens'] - counts['cached_tokens'])*rates[0]
               + counts['completion_tokens']*rates[1] + counts['cached_tokens']*rates[2])/1000000
        return dict(**counts, tokens=counts['prompt_tokens']+counts['completion_tokens'],
                    calls=len(items), estimated_usd=float(usd), estimated_usd_decimal=str(usd))

    # The episode ledger aggregates all request phases for that attempt. Use it
    # as a cross-check, not a second bill (Kang planning would be counted twice).
    attempts = {}
    for line in ledger.read_text().splitlines():
        row = json.loads(line)
        key = (row['task_id'], row['attempt_index'])
        if key in attempts or key[0] not in task_ids:
            raise ValueError('duplicate or out-of-support episode')
        subtotal = total([c for c in calls if (c['task_id'], c['attempt_index']) == key])
        if (row['usage'] != {k: subtotal[k] for k in row['usage']}
                or row['tokens_spent'] != subtotal['completion_tokens']):
            raise ValueError('episode and request ledger sums disagree')
        attempts[key] = subtotal
    if {(c['task_id'], c['attempt_index']) for c in calls} != set(attempts):
        raise ValueError('request ledger has attempts absent from episode ledger')
    phases = {p: total([c for c in calls if c.get('phase', 'trajectory') == p])
              for p in ('cot', 'trajectory')}
    if expected != 'kang-ftp' and phases['cot']['calls']:
        raise ValueError('planning charges in a non-Kang bank')
    if expected == 'kang-ftp' and not phases['cot']['calls']:
        raise ValueError('Kang requires its planning purchases')
    if file_hash(ledger) != ledger_hash or file_hash(usage) != usage_hash:
        raise ValueError('collection changed during accounting')
    return dict(method=method, bank=str(bank), bank_audit_sha256=file_hash(bank/'sealed/audit.json'),
        **total(calls), phase_costs=phases,
        per_task={t: total([c for c in calls if c['task_id'] == t]) for t in sorted(task_ids)},
        per_attempt={t: {str(a): cost for (tid, a), cost in sorted(attempts.items()) if tid == t}
                     for t in sorted(task_ids)},
        scope='all purchases for frozen support, including failed and unselected candidates',
        usage_basis='last reported state per request ID; prompt + completion; cached input is a subset',
        prices_per_million=[str(r) for r in rates], currency='USD (estimate at recorded prices)',
        sources={str(ledger): ledger_hash, str(usage): usage_hash,
                 str(collection/'identity.json'): file_hash(collection/'identity.json')},
        new_teacher_calls=0)
