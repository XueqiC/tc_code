"""Convert immutable C26-B verified episodes into student-rendered v1.1 banks."""
from dataclasses import asdict
from copy import deepcopy
import json
from pathlib import Path

from ...adapters.alfworld import ALFWorldAdapter
from ...cc_pairs import digest as payload_digest
from ..bank_build import privileged, load_student_tokenizer, state_record, seal_v11
from ..persistence import file_hash, digest
from ..transport import FullState
from .alfworld_support import audit_verified_bank, prompt_messages, validate_support
from .alfworld_state import canonical_hash


def frozen_conversion_support(source_support, manifest_path):
    """Allow collector identity changes only; never select or refold support."""
    support = validate_support(json.loads(Path(manifest_path).read_text()))
    normalized = deepcopy(source_support)
    if normalized['environment'] != support['environment']:
        environment = normalized['environment']
        if (environment.get('collection') != 'alfworld_teacher_pool' or
                environment.get('historical_environment_hash') != support['environment']['environment_hash']):
            raise ValueError('collector environment does not descend from frozen support')
        normalized['environment'] = support['environment']
        for task in normalized['tasks'].values():
            task['request']['environment_hash'] = support['environment']['environment_hash']
            task['request_hash'] = canonical_hash(task['request'])
    normalized['manifest_hash'] = support['manifest_hash']
    different = sorted(k for k in normalized.keys() | support.keys() if normalized.get(k) != support.get(k))
    if different:
        raise ValueError('source differs from frozen support beyond collector identity: ' + ', '.join(different))
    return support


def convert_alfworld_bank(source, destination, *, config, tokenizer=None, ledger=None):
    privileged()
    source = Path(source)
    # The legacy verifier binds replay/order/world bytes and the historical bill.
    verified = audit_verified_bank(source)
    source_support = json.loads((source/'public/support.json').read_text())
    support = frozen_conversion_support(source_support, config['support_manifest'])
    adapter = ALFWorldAdapter()
    adapter._tokenizer = tokenizer or load_student_tokenizer(config)
    def render(value):
        old = FullState(**value)
        request, history = json.loads(old.task_json), json.loads(old.history_json)
        task = source_support['tasks'][request['task_id']]
        if request != task['request'] or old.parent_hash != task['parent_hash']:
            raise ValueError('source state differs from audited support')
        request = support['tasks'][request['task_id']]['request']
        prompt = adapter._render(prompt_messages(request, history))
        return FullState.create(request, history, prompt, old.parent_hash)
    resets = {tid: asdict(render(s)) for tid, s in json.loads((source/'public/reset_states.json').read_text()).items()}
    integrity = json.loads((source/'sealed/integrity.json').read_text())
    rows = json.loads((source/'public/requests.json').read_text())
    records, payloads = [], {}
    for row in rows:
        q = row['spec']['query_id']
        p = json.loads((source/'sealed'/f'{q}.json').read_text())
        if payload_digest(p) != integrity[q]:
            raise ValueError('source sealed integrity changed during conversion')
        behaviors = [dict(state=asdict(render(b['state'])), text=b['text']) for b in p['behaviors']]
        # Preserve the paid episode boundary. D12 maps its ordered state records
        # through supervision_records; converting is not another teacher purchase.
        state = FullState(**behaviors[0]['state']) if behaviors else None
        records.append(state_record(q, state, 'alf_demo_episode', p['cost'], p['cost_confidence'],
            dependencies=row['dependencies'], unavailable=row['unavailable_reason'], parent=row['parent_hash']))
        # verification and historical_response remain the original collector
        # evidence. Only runtime requests/states are rebound and student-rendered.
        payloads[q] = dict(p, behaviors=behaviors,
            request_state_hash=support['tasks'][p['provenance']['task_id']]['request_hash'],
            provenance=dict(p['provenance'], source_request_state_hash=p['request_state_hash'],
            source_bank_integrity_sha256=file_hash(source/'sealed/integrity.json'),
            rendering_student=config['student']))
    audit = json.loads((source/'sealed/audit.json').read_text())
    binding = dict(source_support_manifest_hash=source_support['manifest_hash'],
        frozen_support_manifest_hash=support['manifest_hash'],
        frozen_support_sha256=file_hash(config['support_manifest']),
        source_environment=source_support['environment'],
        task_ids_added=[], task_ids_dropped=[], packages_dropped=0,
        changed_request_identities=sum(source_support['tasks'][t]['request_hash'] != v['request_hash']
                                       for t, v in support['tasks'].items()),
        verification_scope='original collector evidence; runtime states rebound to frozen requests')
    audit.update(support_manifest_hash=support['manifest_hash'], support_binding=binding)
    inputs = dict(source_manifest=file_hash(source/'sealed/manifest.json'),
                  source_integrity=file_hash(source/'sealed/integrity.json'), verification=verified,
                  support_binding=binding)
    if ledger is not None:
        inputs['ledger_sha256'] = file_hash(ledger)
        # Ledger identity is archival evidence; only the audited costs are reused.
        ledger_lines = Path(ledger).read_text().splitlines()
        for p in payloads.values():
            line = p['provenance']['line']
            if json.loads(ledger_lines[line-1]) != p['historical_response']:
                raise ValueError('supplied ALFWorld ledger differs from verified archive')
    return seal_v11(destination, records, payloads, benchmark='alfworld', student=config['student'],
        public={'support.json': support, 'reset_states.json': resets}, audit=audit, inputs=inputs)
