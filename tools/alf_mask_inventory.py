#!/usr/bin/env python3
"""Read-only, CPU inventory of authored tokens masked by SmartAD/SAD markers."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT)]


def inventory_rows(rows, tokenizer, max_context_tokens=32768):
    from bfas.rtd.baselines.alfworld_training import encoded_spans
    affected = []
    authored_tokens = boundary_tokens = supervised_tokens = 0
    for row in rows:
        encoded, kinds = encoded_spans(tokenizer, row, max_context_tokens)
        count = len(tokenizer.encode(row.target, add_special_tokens=False))
        masked = kinds[:count].count('observation')
        authored_tokens += count
        boundary_tokens += len(encoded['target_ids']) - count
        supervised_tokens += sum(k != 'observation' for k in kinds)
        if masked:
            affected.append(dict(package_id=row.package_id, task_id=row.task_id,
                                 turn_index=row.index, masked_authored_tokens=masked))
    return dict(packages=len({r.package_id for r in rows}), rows=len(rows),
        authored_tokens=authored_tokens, native_boundary_tokens=boundary_tokens,
        supervised_tokens=supervised_tokens,
        masked_authored_tokens=sum(r['masked_authored_tokens'] for r in affected),
        affected_rows=len(affected), affected_packages=len({r['package_id'] for r in affected}),
        any_authored_token_masked=bool(affected), affected=affected,
        rule='current encoded_spans/token_kinds observation-marker mask on teacher_react_turns',
        model_calls=0, teacher_calls=0)


def inventory_bank(bank, renderer, max_context_tokens=32768):
    from bfas.rtd.baselines.pi1 import load_bank
    from bfas.rtd.persistence import file_hash
    bank = Path(bank).resolve()
    support = json.loads((bank/'public/support.json').read_text())
    records = json.loads((bank/'public/requests.json').read_text())
    usable = [r for r in records if r['unavailable_reason'] is None]
    turns = sum(len(json.loads((bank/'sealed'/f"{r['spec']['query_id']}.json").read_text())
                    ['teacher_react_turns']) for r in usable)
    config = dict(sealed_manifest_sha256=file_hash(bank/'sealed/manifest.json'),
                  support_size=support['m'], demonstrations=len(usable), supervised_turns=turns)
    rows, identity = load_bank(bank, config, renderer)
    return dict(bank=identity, **inventory_rows(rows, renderer.adapter._tokenizer, max_context_tokens))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bank', type=Path, action='append', required=True)
    parser.add_argument('--model-path', type=Path, required=True, help='cached tokenizer snapshot only')
    parser.add_argument('--max-context-tokens', type=int, default=32768)
    args = parser.parse_args(argv)
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    for key in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_DATASETS_OFFLINE'):
        os.environ[key] = '1'
    from bfas.rtd.benchmarks.alfworld_support import FrozenRenderer
    from bfas.rtd.benchmarks.alfworld_identity import tokenizer_identity
    renderer = FrozenRenderer(args.model_path)
    print(json.dumps(dict(tokenizer=tokenizer_identity(args.model_path),
        banks=[inventory_bank(bank, renderer, args.max_context_tokens) for bank in args.bank]), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
