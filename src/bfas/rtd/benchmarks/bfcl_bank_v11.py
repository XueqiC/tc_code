"""GPT-5.4 ledger/pool ingestion using the native student row converter."""
from collections import defaultdict
from dataclasses import asdict
import json
from pathlib import Path

from ..bank import parent_hash
from ..bank_build import (privileged, read_pool, load_student_tokenizer, ledger_cost,
                          state_record, seal_v11)
from ..persistence import digest, file_hash
from ..student import bfcl_row
from ..transport import FullState


def build_bfcl_pool_bank(root, directory, *, pool, ledger, config, entries=None, tokenizer=None):
    privileged()
    from ...adapters.bfcl import BFCLAdapter
    entries = entries if entries is not None else BFCLAdapter()._load_entries()[0]
    tokenizer = tokenizer or load_student_tokenizer(config)
    support = json.loads((Path(root)/config['support_manifest']).read_text())
    parents = {p['official_id']: p['parent_hash'] for p in support['parents']}
    for tid, h in parents.items():
        if parent_hash(entries[tid]) != h:
            raise ValueError('frozen BFCL parent identity changed')
    grouped = defaultdict(list)
    for row in read_pool(pool):
        grouped[str(row['task_id'])].append(row)
    records, payloads, used = [], {}, set()
    rows = read_pool(ledger)
    historical = 0
    for i, paid in enumerate(rows):
        if str(paid.get('teacher', '')).removeprefix('azure/').removesuffix('-FC') != 'gpt-5.4':
            raise ValueError('BFCL pool requires GPT-5.4 ledger')
        tid = str(paid['task_id'])
        total, confidence = ledger_cost(paid, default_confidence='exact')
        historical += total
        q = digest(['bfcl-paid-episode', file_hash(ledger), i])
        if not paid['verified'] or tid not in parents or tid not in grouped:
            records.append(state_record(q, None, 'demo_attempt', total, confidence,
                parent=parents.get(tid, digest(['excluded', tid])), unavailable='failed or absent/protected demo'))
            payloads[q] = dict(cost=total, cost_confidence=confidence, usage=dict(output_tokens=total),
                provenance=dict(task_id=tid, ledger_row=i), historical_response=paid, behaviors=[])
            continue
        if tid in used:
            raise ValueError('ambiguous successful ledger attempt')
        used.add(tid)
        source_rows = grouped[tid]
        turns = paid['demo']['turns']
        if len(turns) != len(source_rows):
            raise ValueError('pool/ledger step count differs')
        rendered = []
        for turn, row in zip(turns, source_rows):
            # Compare after canonical rendering too: pool may already be Gemma.
            paid_row = dict(task_id=tid, prompt=turn['prompt'], response=turn['target'])
            context = turn.get('context')
            if isinstance(context, dict):
                paid_row['_render_context'] = context
            elif isinstance(context, list) and not tid.startswith(('multi_turn', 'memory', 'web_search')):
                paid_row['_render_context'] = dict(messages=entries[tid]['question'][0], functions=entries[tid]['function'])
            rendered_paid = bfcl_row(config, paid_row, tokenizer)
            result = bfcl_row(config, row, tokenizer)
            if (result['prompt'], result['response']) != (rendered_paid['prompt'], rendered_paid['response']):
                raise ValueError('pool target/context differs from paid teacher output')
            rendered.append(result)
        # BFCL keeps the paid request/episode boundary, and D12 enumerates states.
        behaviors = []
        for row in rendered:
            ctx = row['_render_context']
            state = FullState.create(dict(function=ctx['functions'], question=entries[tid]['question'][0]),
                                     ctx['messages'], row['prompt'], parents[tid])
            behaviors.append(dict(state=asdict(state), text=row['response']))
        first = FullState(**behaviors[0]['state'])
        records.append(state_record(q, first, 'demo_attempt', total, confidence))
        payloads[q] = dict(cost=total, cost_confidence=confidence, usage=dict(output_tokens=total),
            provenance=dict(task_id=tid, ledger_row=i, verified=True, rendering_student=config['student']),
            historical_response=paid, behaviors=behaviors)
    if set(grouped) - {str(r['task_id']) for r in rows if r['verified']}:
        raise ValueError('pool includes unpurchased teacher evidence')
    return seal_v11(directory, records, payloads, benchmark='bfcl', student=config['student'],
        public={'support.json': support}, audit=dict(m=len(parents), historical_output_tokens=historical,
        historical_attempts=len(rows)), inputs=dict(pool=file_hash(pool), ledger=file_hash(ledger)))
