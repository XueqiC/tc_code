"""Privileged offline C25 conversion; never called by an acquisition policy.

Only a total and two uniform caps become public. Payload bytes, unavailable
records, dependency graphs and all historical cost fields are preserved.
"""
import copy
import json
from pathlib import Path
import shutil

from ..cc_pairs import digest as payload_digest
from .caps import V11_CLASS_CAPS, V11_BUDGET_BASIS, C25_RECORDED_OUTPUT_TOKENS, recorded_budget_ceilings
from .persistence import atomic_json, digest, file_hash

C25_PUBLIC_SHA256 = '2d27d0f4dcfd1076ac2fbdba6ff3d2789caf5971c14a5da61634d710d00084e8'
CERTIFICATE = 'public/cap_certificate.json'


def build_v11_bank(source, destination, *, expected_total=C25_RECORDED_OUTPUT_TOKENS,
                   expected_source_hash=C25_PUBLIC_SHA256):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.exists() or destination == source or destination.is_relative_to(source) or source.is_relative_to(destination):
        raise ValueError('new isolated bank directory required; source is immutable')
    source_public = file_hash(source / 'public/requests.json')
    if expected_source_hash is not None and source_public != expected_source_hash:
        raise ValueError('frozen C25 public input hash mismatch')
    original = json.loads((source / 'public/requests.json').read_text())
    records = copy.deepcopy(original)
    integrity = json.loads((source / 'sealed/integrity.json').read_text())
    if len({r['spec']['query_id'] for r in records}) != len(records) or set(integrity) != {r['spec']['query_id'] for r in records}:
        raise ValueError('request/integrity identity mismatch')
    total, maxima, counts, confidence = 0, dict.fromkeys(V11_CLASS_CAPS, 0), dict.fromkeys(V11_CLASS_CAPS, 0), dict(exact=0, estimated=0)
    usable_ids = []
    for row in records:
        spec, q = row['spec'], row['spec']['query_id']
        if len(q) != 64 or any(c not in '0123456789abcdef' for c in q):
            raise ValueError('invalid request identity')
        payload = json.loads((source / 'sealed' / (q + '.json')).read_text())
        if payload_digest(payload) != integrity[q]:
            raise ValueError('sealed payload integrity mismatch')
        if row['unavailable_reason'] is not None:
            continue
        kind = json.loads(spec['cap_provenance'])['request_class']
        cost = payload['cost']
        if kind not in V11_CLASS_CAPS or type(cost) is not int or not 0 <= cost <= V11_CLASS_CAPS[kind]:
            raise ValueError('recorded cost exceeds certified class cap')
        if payload['cost_confidence'] != spec['cost_confidence'] or spec['cost_confidence'] not in confidence:
            raise ValueError('recorded cost confidence mismatch')
        usable_ids.append(q)
        total += cost
        maxima[kind] = max(maxima[kind], cost)
        counts[kind] += 1
        confidence[spec['cost_confidence']] += cost
    if total != expected_total:
        raise ValueError('usable recorded output token denominator mismatch')
    if any(counts[k] and 2 ** max(0, (maxima[k]-1).bit_length()) != cap for k, cap in V11_CLASS_CAPS.items()):
        raise ValueError('class cap must be the next power of two of the frozen content maximum')
    core = dict(version='rtd-v1.1.0-bfcl-bank', protocol_version='1.1.0',
        source_public_sha256=source_public, sealed_integrity_sha256=file_hash(source/'sealed/integrity.json'),
        usable_set_hash=digest(sorted(usable_ids)), available_packages=len(usable_ids),
        class_counts=counts, class_caps=V11_CLASS_CAPS, all_actual_within_class_cap=True,
        cap_basis='sealed_bank_class_content_envelope', cap_rule='next_power_of_two_of_usable_class_maximum',
        public_cost_assumption='approved bank-dependent class bounds and total; not a provider guarantee',
        budget_basis=V11_BUDGET_BASIS, budget_denominator=total, rounding='positive_integer_half_up',
        cost_scope='cached-content cost; not the full teacher bill', online_authorized=False)
    core_hash = digest(core)
    for row in records:
        if row['unavailable_reason'] is not None:
            continue
        spec = row['spec']
        old = json.loads(spec['cap_provenance'])
        kind = old['request_class']
        spec['cost_upper_bound'] = V11_CLASS_CAPS[kind]
        spec['cap_provenance'] = json.dumps(dict(request_class=kind, basis=core['cap_basis'], protocol_version='1.1.0',
            output_token_cap=V11_CLASS_CAPS[kind], certificate_core_hash=core_hash,
            historical_cap_provenance=old, public_cost_assumption=core['public_cost_assumption'],
            online_authorized=False), sort_keys=True)
    # Check everything above before creating output. Never rewrite payload JSON.
    destination.mkdir(parents=True, exist_ok=False)
    shutil.copytree(source/'sealed', destination/'sealed')
    (destination/'public').mkdir()
    atomic_json(destination/'public/requests.json', records)
    certificate = dict(core=core, core_hash=core_hash, public_sha256=file_hash(destination/'public/requests.json'))
    atomic_json(destination/CERTIFICATE, certificate)
    summary = dict(version=core['version'], budget_basis=V11_BUDGET_BASIS, cost_scope=core['cost_scope'],
        budget_denominator=total, budget_ceilings=recorded_budget_ceilings(total),
        bank_public_cap_sum=sum(r['spec']['cost_upper_bound'] for r in records if r['unavailable_reason'] is None),
        recorded_bank_usage=total, available_cost_by_confidence=confidence, available_packages=len(usable_ids),
        cap_certificate_sha256=file_hash(destination/CERTIFICATE), public_cost_assumption=core['public_cost_assumption'])
    # Preserve the source audit too: the v1.1 audit is a distinct file.
    atomic_json(destination/'sealed/audit_v11.json', summary)
    return summary


def validate_v11_certificate(bank, *, expected_total=C25_RECORDED_OUTPUT_TOKENS):
    """Validate public binding without opening any unpurchased response file."""
    bank = Path(bank)
    certificate = json.loads((bank/CERTIFICATE).read_text())
    core = certificate['core']
    if (certificate['core_hash'] != digest(core) or certificate['public_sha256'] != file_hash(bank/'public/requests.json')
            or core['sealed_integrity_sha256'] != file_hash(bank/'sealed/integrity.json')
            or core['class_caps'] != V11_CLASS_CAPS or core['budget_basis'] != V11_BUDGET_BASIS
            or core['budget_denominator'] != expected_total or core['all_actual_within_class_cap'] is not True
            or core['cap_basis'] != 'sealed_bank_class_content_envelope'
            or core['cap_rule'] != 'next_power_of_two_of_usable_class_maximum'
            or core['rounding'] != 'positive_integer_half_up'):
        raise ValueError('v1.1 cap certificate/bank binding mismatch')
    records = json.loads((bank/'public/requests.json').read_text())
    usable = [r for r in records if r['unavailable_reason'] is None]
    if len(usable) != core['available_packages'] or digest(sorted(r['spec']['query_id'] for r in usable)) != core['usable_set_hash']:
        raise ValueError('certified usable set changed')
    for row in usable:
        spec = row['spec']
        provenance = json.loads(spec['cap_provenance'])
        if (spec['cost_upper_bound'] != V11_CLASS_CAPS[provenance['request_class']]
                or provenance['certificate_core_hash'] != certificate['core_hash']
                or provenance['basis'] != core['cap_basis']):
            raise ValueError('nonuniform or uncertified public cap')
    return certificate
