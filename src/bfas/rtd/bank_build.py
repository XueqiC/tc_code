"""Offline v1.1 bank certificates and ledger-backed state packages (D16).

This module is privileged ingestion. No teacher is called. Historical failed
attempts remain sealed and paid in the inventory, outside the usable denominator.
"""
from collections import Counter
from dataclasses import asdict, replace
import json
from pathlib import Path

from .bank_v11 import CERTIFICATE
from .broker import RequestRecord, seal_bank
from .caps import V11_BUDGET_BASIS, recorded_budget_ceilings
from .persistence import atomic_json, digest, file_hash
from .selector import PublicFeatures, PublicQuerySpec


def privileged():
    from . import selector
    if selector._active.get() is not None:
        raise PermissionError('selector cannot inspect or build a bank')


def seal_v11(directory, records, payloads, *, benchmark, student, public, audit, inputs,
             teacher_accounting=None):
    privileged()
    # Real paid pools carry task/attempt identities. Older synthetic/legacy
    # banks without a ledger keep their original request protocol.
    if benchmark in {'alfworld', 'bfcl', 'hotpotqa'} and all(
            'task_id' in p.get('provenance', {}) for p in payloads.values()):
        from .task_packages import ATTEMPT_LEDGER, build_attempt_ledger
        public = dict(public, **{ATTEMPT_LEDGER: build_attempt_ledger(records, payloads)})
    maxima, counts, confidence = {}, Counter(), Counter()
    for row in records:
        if row.unavailable_reason is not None:
            continue
        p = payloads[row.spec.query_id]
        kind = json.loads(row.spec.cap_provenance)['request_class']
        cost = p['cost']
        if type(cost) is not int or cost < 0 or p['cost_confidence'] != row.spec.cost_confidence:
            raise ValueError('invalid recorded output cost/confidence')
        maxima[kind] = max(maxima.get(kind, 0), cost)
        counts[kind] += 1
        confidence[p['cost_confidence']] += cost
    if not counts or sum(confidence.values()) <= 0:
        raise ValueError('bank needs usable recorded output costs')
    caps = {k: 1 << max(0, (v - 1).bit_length()) for k, v in maxima.items()}
    records = [replace(r, spec=replace(r.spec, cost_upper_bound=caps[json.loads(r.spec.cap_provenance)['request_class']]))
               if r.unavailable_reason is None else r for r in records]
    directory = seal_bank(directory, records, payloads)
    for name, value in public.items():
        atomic_json(directory / 'public' / name, value)
    atomic_json(directory / 'sealed/audit.json', audit)
    usable = [r for r in records if r.unavailable_reason is None]
    core = dict(version='rtd-v1.1.0-state-bank', protocol_version='1.1.0', benchmark=benchmark,
        student=student, inputs=inputs, sealed_integrity_sha256=file_hash(directory/'sealed/integrity.json'),
        public_artifacts={name: file_hash(directory/'public'/name) for name in public},
        audit_sha256=file_hash(directory/'sealed/audit.json'),
        usable_set_hash=digest(sorted(r.spec.query_id for r in usable)), available_packages=len(usable),
        class_counts=dict(counts), class_caps=caps, all_actual_within_class_cap=True,
        cap_basis='sealed_bank_class_content_envelope', cap_rule='next_power_of_two_of_usable_class_maximum',
        public_cost_assumption='bank-dependent class bounds and total; not a provider guarantee',
        budget_basis=V11_BUDGET_BASIS, budget_denominator=sum(confidence.values()),
        rounding='positive_integer_half_up', cost_scope='cached-content cost; not the full teacher bill',
        online_authorized=False)
    if teacher_accounting is not None:
        core['teacher_accounting'] = teacher_accounting
    core_hash = digest(core)
    for i, row in enumerate(records):
        if row.unavailable_reason is None:
            provenance = json.loads(row.spec.cap_provenance)
            provenance.update(basis=core['cap_basis'], certificate_core_hash=core_hash,
                              output_token_cap=row.spec.cost_upper_bound, online_authorized=False)
            records[i] = replace(row, spec=replace(row.spec, cap_provenance=json.dumps(provenance, sort_keys=True)))
    atomic_json(directory/'public/requests.json', [asdict(r) for r in records])
    atomic_json(directory/CERTIFICATE, dict(core=core, core_hash=core_hash,
                                         public_sha256=file_hash(directory/'public/requests.json')))
    summary = dict(version=core['version'], budget_basis=core['budget_basis'], cost_scope=core['cost_scope'],
        budget_denominator=core['budget_denominator'], recorded_bank_usage=core['budget_denominator'],
        budget_ceilings=recorded_budget_ceilings(core['budget_denominator'], rounds=2),
        bank_public_cap_sum=sum(r.spec.cost_upper_bound for r in usable),
        available_cost_by_confidence=dict(confidence), available_packages=len(usable),
        cap_certificate_sha256=file_hash(directory/CERTIFICATE), public_cost_assumption=core['public_cost_assumption'])
    if teacher_accounting is not None:
        summary['teacher_accounting'] = teacher_accounting
    atomic_json(directory/'sealed/audit_v11.json', summary)
    return summary


def validate_state_certificate(bank, *, benchmark=None, student=None):
    """Public/index validation only: never open an unpurchased response."""
    privileged()
    bank = Path(bank)
    cert = json.loads((bank/CERTIFICATE).read_text()); core = cert['core']
    if (core['version'] != 'rtd-v1.1.0-state-bank' or cert['core_hash'] != digest(core)
            or cert['public_sha256'] != file_hash(bank/'public/requests.json')
            or core['sealed_integrity_sha256'] != file_hash(bank/'sealed/integrity.json')
            or core['audit_sha256'] != file_hash(bank/'sealed/audit.json')
            or core['budget_basis'] != V11_BUDGET_BASIS
            or core['cap_basis'] != 'sealed_bank_class_content_envelope'
            or core['cap_rule'] != 'next_power_of_two_of_usable_class_maximum'
            or core['rounding'] != 'positive_integer_half_up'
            or core['online_authorized'] is not False or core['all_actual_within_class_cap'] is not True
            or type(core['budget_denominator']) is not int or core['budget_denominator'] <= 0
            or (benchmark is not None and core['benchmark'] != benchmark)
            or (student is not None and core['student'] != student)):
        raise ValueError('v1.1 state bank certificate binding mismatch')
    for name, sha in core['public_artifacts'].items():
        if Path(name).name != name or file_hash(bank/'public'/name) != sha:
            raise ValueError('bank support/reset artifact mismatch')
    rows = json.loads((bank/'public/requests.json').read_text())
    usable = [r for r in rows if r['unavailable_reason'] is None]
    counts = Counter()
    for r in usable:
        spec = r['spec']; p = json.loads(spec['cap_provenance']); kind = p['request_class']
        cap = core['class_caps'][kind]
        if (type(cap) is not int or cap < 1 or cap & (cap-1) or spec['cost_upper_bound'] != cap
                or p['certificate_core_hash'] != cert['core_hash'] or p['basis'] != core['cap_basis']):
            raise ValueError('uncertified public class cap')
        counts[kind] += 1
    if (dict(counts) != core['class_counts'] or len(usable) != core['available_packages']
            or digest(sorted(r['spec']['query_id'] for r in usable)) != core['usable_set_hash']):
        raise ValueError('certified usable set changed')
    return cert


def state_record(q, state, kind, cost, confidence, *, dependencies=(), unavailable=None, parent=None):
    return RequestRecord(PublicQuerySpec(q, state.state_hash if state else digest(['unavailable', q]),
        PublicFeatures(), 1, max(1, cost), confidence, json.dumps(dict(request_class=kind))),
        state.parent_hash if state else parent, tuple(dependencies), unavailable)


def allocate_cost(total, targets, tokenizer):
    """Integer allocation of recorded episode usage; never label it per-turn exact."""
    if type(total) is not int or total < 0 or not targets:
        raise ValueError('recorded episode cost and targets required')
    weights = [max(1, len(tokenizer.encode(t, add_special_tokens=False))) for t in targets]
    costs = [total*w//sum(weights) for w in weights]
    for i in range(total-sum(costs)):
        costs[i] += 1
    return costs


def ledger_cost(row, *, default_confidence):
    total = row['tokens_spent']
    confidence = row.get('cost_confidence', default_confidence)
    if (type(total) is not int or total < 0 or type(row['verified']) is not bool
            or type(row['attempt_index']) is not int or row['attempt_index'] < 0
            or row.get('purpose', 'teacher') != 'teacher'
            or confidence not in {'exact', 'estimated'}):
        raise ValueError('invalid recorded ledger spend, verification, or teacher attempt')
    return total, confidence


def read_pool(path):
    path = Path(path)
    if path.suffix == '.jsonl':
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    value = json.loads(path.read_text())
    if isinstance(value, dict):
        return [dict(v, task_id=k) for k, v in value.items()]
    return value


def load_student_tokenizer(config):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(config['student'], local_files_only=True, trust_remote_code=False)
