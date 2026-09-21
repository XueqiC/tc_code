#!/usr/bin/env python3
"""Mint the frozen 2026-09-21 K=32 ALFWorld collection source, from /tmp.

No teacher, model, downloader, or GPU is used. The local tokenizer renders reset
prompts; ALFWorld/TextWorld run only through RealStepper's existing CPU worker.
Inputs are read-only. Every output path is new, including on retries.

The legacy bank schema requires a ledger-shaped record even before collection.
Each sealed record is explicitly an uncollected, zero-cost synthetic slot, never
a demonstration or failed teacher attempt. Its query identity uses the frozen
support file hash and 1-based support position; there is no teacher ledger.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))

from bfas.adapters import alfworld as adapter
from bfas.rtd.benchmarks import alfworld_bank as bank
from bfas.rtd.benchmarks.alfworld_caps import CapConfiguration
from bfas.rtd.benchmarks.alfworld_state import (
    _observed, canonical_hash, parent_fold, parent_hash, validate_full_state,
)
from bfas.rtd.benchmarks.alfworld_support import (
    VERSION, ROUND_FOLDS, FrozenRenderer, RealStepper, _signed,
    environment_identity, seal_verified_bank, validate_support,
)
from bfas.rtd.transport import FullState

SUPPORT = Path('configs/alfworld_support_k32_20260921.json')
MANIFEST = Path('results/manifest/alfworld_manifest_20260921.json')
OUTPUT = Path('data/rtd/v1_alfworld_k32_20260921')
SUPPORT_SHA256 = 'f11d713f9dbf4d9997b61961e70be8762ba1289de8bae276a2e62adce353c57d'
SPLIT_SIZES = dict(train=3553, valid_seen=140, valid_unseen=134)
UNAVAILABLE = 'uncollected support slot; no teacher attempt or demonstration'


def _ids_hash(ids):
    """Use the existing file hasher for the exporter's newline-joined encoding."""
    with tempfile.NamedTemporaryFile(prefix='alf-k32-ids-', dir='/tmp') as stream:
        stream.write('\n'.join(ids).encode('utf-8'))
        stream.flush()
        return bank.file_hash(stream.name)


def read_inputs(sources, support_path, manifest_path):
    frozen = sources.json(str(support_path))
    manifest = sources.json(str(manifest_path))
    entries = frozen['support']
    ids = [entry['game_id'] for entry in entries]
    for tid in ids:
        parent_hash(tid)
    if frozen['k'] != 32 or len(ids) != 32 or len(set(ids)) != 32:
        raise ValueError('frozen support must contain exactly 32 unique game IDs')
    if _ids_hash(ids) != frozen['support_sha256'] or frozen['support_sha256'] != SUPPORT_SHA256:
        raise ValueError('frozen support SHA256/order mismatch; never regenerate or reorder it')
    if ids != sorted(ids):
        raise ValueError('frozen support order incompatible with validate_support')
    if Counter(entry['task_type'] for entry in entries) != frozen['quota'] or len(frozen['quota']) != 6:
        raise ValueError('frozen support task-type quota mismatch')
    if sources.path(frozen['source_manifest']) != sources.path(str(manifest_path)):
        raise ValueError('frozen support source_manifest differs from supplied manifest')
    data = sources.path('envs/alfworld/data/json_2.1.1')
    if Path(manifest['data_root']).resolve() != data:
        raise ValueError('loader manifest data root differs from the existing adapter layout')
    for split, count in SPLIT_SIZES.items():
        item = manifest['splits'][split]
        split_ids = item['game_ids']
        if (item['loader_error'] is not None or item['loader_games'] != count
                or len(split_ids) != count or len(set(split_ids)) != count
                or _ids_hash(split_ids) != item['game_ids_sha256']
                or Path(item['split_dir']).resolve() != data / split):
            raise ValueError(f'loader manifest inventory/hash mismatch: {split}')
        for tid in split_ids:
            parent_hash(tid)
    train = manifest['splits']['train']
    if (frozen['source_manifest_train_sha256'] != train['game_ids_sha256']
            or frozen['train_pool'] != train['loader_games'] or not set(ids) <= set(train['game_ids'])):
        raise ValueError('frozen support is not bound to the loader train inventory')
    for entry in entries:
        traj = sources.json(f"envs/alfworld/data/json_2.1.1/train/{entry['game_id']}/traj_data.json")
        if traj['task_type'] != entry['task_type'] or adapter._category(entry['game_id']) != entry['task_type']:
            raise ValueError('support task type differs from the training world')
    return frozen, manifest


def build_support(sources, frozen, manifest, environment):
    """Adapt the fixed selection to the existing signed support schema."""
    ids = [entry['game_id'] for entry in frozen['support']]
    tasks, groups, worlds = {}, defaultdict(list), defaultdict(set)
    for tid in ids:
        h = parent_hash(tid)
        request = bank._reset_request(sources, tid, CapConfiguration(), environment['environment_hash'])
        tasks[tid] = dict(parent_hash=h, fold=parent_fold(h), request=request,
                          request_hash=canonical_hash(request), excluded=False)
        groups[h].append(tid)
        worlds[request['world_hash']].add(h)
    if any(len(hashes) > 1 for hashes in worlds.values()):
        raise ValueError('identical world bytes across parent names; grouping must be revised')
    parents = {h: dict(parent_game=tids[0].split('/')[0], task_ids=sorted(tids),
                       selected_task_id=min(tids), fold=parent_fold(h))
               for h, tids in sorted(groups.items())}
    names = {tid.split('/')[0] for tid in ids}
    overlaps = {
        split: dict(parent_names=sorted(names & {t.split('/')[0] for t in manifest['splits'][split]['game_ids']}),
                    comparisons=[], world_comparisons_performed=False, independent_certificate=False)
        for split in ('valid_seen', 'valid_unseen')
    }
    return validate_support(_signed(dict(
        version=VERSION, benchmark='alfworld', split='train', scope='exploratory', frozen=True,
        historical_task_ids=ids, historical_parent_groups=len(groups), tasks=tasks, parents=parents,
        m=len(parents), training_task_ids=ids,
        # K=32 has no designated calibration/probe tasks. Do not inherit C26's
        # unrelated historical exclusions or remove any of these frozen games.
        calibration_task_ids=[], probe_task_ids=[], protected_parent_hashes=[],
        fold_parent_counts={str(f): sum(parent_fold(h) == f for h in parents) for f in (0, 1)},
        fold_task_counts={str(f): sum(t['fold'] == f for t in tasks.values()) for f in (0, 1)},
        rounds={str(r): dict(inner=i, feedback=f) for r, (i, f) in ROUND_FOLDS.items()},
        fold_rule='existing alfworld_state.parent_fold(parent_hash(task_id))',
        feedback_task_rule='uniform parents; lexicographically first trial; same trial for both rollouts',
        source_files=sources.files, environment=environment, evaluation_overlap=overlaps,
        collection_source=dict(kind='uncollected_frozen_support', k=32,
                               support_sha256=frozen['support_sha256'], frozen_selection=frozen,
                               teacher_attempts=0, teacher_calls=0, teacher_tokens=0),
    )))


def collection_slots(support, support_path, support_file_hash):
    records, payloads, requests = [], {}, {}
    for position, tid in enumerate(support['historical_task_ids'], 1):
        request = support['tasks'][tid]['request']
        row = dict(task_id=tid, attempt_index=0, timestamp='uncollected:2026-09-21',
                   teacher=None, verified=False, tokens_spent=0,
                   purpose='collection_source_placeholder', synthetic=True, teacher_attempted=False)
        q = bank.query_id(support_file_hash, row, position)
        payloads[q] = dict(
            query_id=q, request_state_hash=canonical_hash(request), dependencies=[],
            status='unavailable', unavailable_reason=UNAVAILABLE, exclusion_reasons=[],
            success=False, payload_kind=None, commands=[], behaviors=[], cost=0,
            cost_basis='no_teacher_calls', cost_confidence='exact',
            usage=dict(teacher_calls=0, prompt_tokens=0, completion_tokens=0, monetary_cost=0),
            provenance=dict(kind='alf_demo_episode', synthetic=True, ledger_path=None,
                            ledger_sha256=support_file_hash, line=position, task_id=tid,
                            attempt_index=row['attempt_index'], timestamp=row['timestamp'], teacher=None,
                            source_path=str(support_path),
                            identity_basis='frozen support file SHA256 and 1-based support entry; no ledger',
                            state_validation='CPU reset only; no teacher trajectory'),
            historical_response=row, raw_ledger_line=json.dumps(row, ensure_ascii=False, allow_nan=False),
            historical_events=[],
        )
        requests[q] = request
        records.append(bank.public_record(q, request))
    return bank.Archive(records, payloads, requests, [], dict(
        source_files=list(support['source_files'].values()), collection_slots=len(records),
        teacher_attempts=0, new_teacher_calls=0, new_teacher_tokens=0,
        record_semantics=UNAVAILABLE,
    ))


def mint_source(root=ROOT, *, support_path=SUPPORT, manifest_path=MANIFEST,
                out=OUTPUT, tokenizer=None, timeout=30.0, reset_timeout=120.0,
                stepper_factory=None, renderer=None):
    root = Path(root).resolve()
    if Path.cwd().resolve() == root:
        raise ValueError('run the minting tool from /tmp, not the project root')
    if not math.isfinite(timeout) or timeout <= 0 or not math.isfinite(reset_timeout) or reset_timeout <= 0:
        raise ValueError('worker timeouts must be finite and positive')
    # Refuse dangling symlinks too; resolve() alone would hide an occupied path.
    output = root / out
    if os.path.lexists(output):
        raise FileExistsError(output)
    output = output.resolve()
    for protected in (root / 'data/rtd/v1_alfworld_c26', root / 'envs', root / 'src',
                      root / 'configs', root / 'results/manifest'):
        if output.is_relative_to(protected.resolve()):
            raise ValueError(f'output is inside read-only input/source tree: {protected}')
    if stepper_factory is None and (root != adapter.ROOT.resolve()
                                   or adapter.DATA.resolve() != root / 'envs/alfworld/data/json_2.1.1'):
        raise ValueError('--root must match the existing adapter root and data directory')
    os.environ.update(CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
                      HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
                      OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    sys.dont_write_bytecode = True
    sources = bank._Sources(root)
    support_path, manifest_path = sources.path(str(support_path)), sources.path(str(manifest_path))
    frozen, manifest = read_inputs(sources, support_path, manifest_path)
    if tokenizer is None:
        from bfas.rtd.benchmarks.alfworld_config import load_config, model_directory
        tokenizer = model_directory(root, load_config(root / 'configs/rtd/v1_alfworld_c26.yaml'))
    tokenizer = Path(tokenizer).resolve()
    environment = environment_identity(root, tokenizer)
    implementation_hash = bank.file_hash(__file__)
    support = build_support(sources, frozen, manifest, environment)
    if renderer is None:
        renderer = FrozenRenderer(tokenizer)
    resets = {}
    for index, tid in enumerate(support['historical_task_ids'], 1):
        request = support['tasks'][tid]['request']
        worker = (stepper_factory(request) if stepper_factory else
                  RealStepper(environment_hash=environment['environment_hash'], timeout=timeout,
                              reset_timeout=reset_timeout))
        try:
            _, obs = worker.reset(request)
            if obs.index != 0 or obs.world_hash != request['world_hash'] or obs.done or obs.won:
                raise ValueError(f'invalid initial reset observation: {tid}')
            history = [_observed(obs)]
            state = FullState.create(request, history, renderer(request, history), parent_hash(tid))
            validate_full_state(state, expected_world_hash=request['world_hash'])
            resets[tid] = asdict(state)
        finally:
            worker.close()
        print(f'[{index}/32] captured reset: {tid}', file=sys.stderr, flush=True)
    # Recheck every input after the worker run; never seal mixed source versions.
    if environment_identity(root, tokenizer) != environment or bank.file_hash(__file__) != implementation_hash:
        raise ValueError('environment or minting implementation changed during reset capture')
    for entry in sources.files.values():
        if bank.file_hash(root / entry['path']) != entry['sha256']:
            raise ValueError(f"source changed during reset capture: {entry['path']}")
    support_file_hash = sources.files[str(support_path)]['sha256']
    archive = collection_slots(support, support_path.relative_to(root), support_file_hash)
    archive.summary['minting_implementation_sha256'] = implementation_hash
    # seal_verified_bank delegates to seal_bank's mkdir(exist_ok=False) and
    # exclusive file opens. No rename/replace or existing-output cleanup occurs.
    # A partial write stays occupied and is deliberately refused on retries.
    seal_verified_bank(output, archive, support, archive.payloads, resets)
    from bfas.rtd.benchmarks.alfworld_config import audit_verified_bank
    from tools.alfworld_teacher_pool import load_source
    audit = audit_verified_bank(output)
    if load_source(output) != support:
        raise ValueError('collector did not load the exact frozen support')
    return dict(output=str(output), support_sha256=frozen['support_sha256'],
                reset_states=len(resets), fold_parent_counts=support['fold_parent_counts'],
                fold_task_counts=support['fold_task_counts'],
                fold_game_ids={str(f): [tid for tid in support['historical_task_ids']
                                       if support['tasks'][tid]['fold'] == f] for f in (0, 1)},
                teacher_calls=0, gpu_used=False, audit=audit)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--support', type=Path, default=SUPPORT)
    parser.add_argument('--manifest', type=Path, default=MANIFEST)
    parser.add_argument('--out', type=Path, default=OUTPUT)
    parser.add_argument('--tokenizer', type=Path, help='local tokenizer directory; defaults to the C26 config cache')
    parser.add_argument('--timeout', type=float, default=30.0)
    parser.add_argument('--reset-timeout', type=float, default=120.0)
    args = parser.parse_args(argv)
    result = mint_source(args.root, support_path=args.support, manifest_path=args.manifest,
                         out=args.out, tokenizer=args.tokenizer, timeout=args.timeout,
                         reset_timeout=args.reset_timeout)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
