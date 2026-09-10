"""Convert immutable C26-B verified episodes into student-rendered v1.1 banks."""
from dataclasses import asdict
import json
from pathlib import Path

from ...adapters.alfworld import ALFWorldAdapter
from ...cc_pairs import digest as payload_digest
from ..bank_build import privileged, load_student_tokenizer, state_record, seal_v11
from ..persistence import file_hash, digest
from ..transport import FullState
from .alfworld_support import audit_verified_bank, prompt_messages


def convert_alfworld_bank(source, destination, *, config, tokenizer=None, ledger=None):
    privileged()
    source = Path(source)
    # The legacy verifier binds replay/order/world bytes and the historical bill.
    verified = audit_verified_bank(source)
    adapter = ALFWorldAdapter()
    adapter._tokenizer = tokenizer or load_student_tokenizer(config)
    def render(value):
        old = FullState(**value)
        request, history = json.loads(old.task_json), json.loads(old.history_json)
        prompt = adapter._render(prompt_messages(request, history))
        return FullState.create(request, history, prompt, old.parent_hash)
    support = json.loads((source/'public/support.json').read_text())
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
        payloads[q] = dict(p, behaviors=behaviors, provenance=dict(p['provenance'],
            source_bank_integrity_sha256=file_hash(source/'sealed/integrity.json'),
            rendering_student=config['student']))
    audit = json.loads((source/'sealed/audit.json').read_text())
    inputs = dict(source_manifest=file_hash(source/'sealed/manifest.json'),
                  source_integrity=file_hash(source/'sealed/integrity.json'), verification=verified)
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
