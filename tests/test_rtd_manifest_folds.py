"""CPU manifest regressions for frozen support folds and legacy derivation."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path

import pytest

from bfas.rtd import cli
from bfas.rtd.bank_build import seal_v11, state_record
from bfas.rtd.conventions import fold_roles, resolve_parent_folds
from bfas.rtd.transport import FullState
from test_rtd_manifest_tolerance import manifest_inputs

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('benchmark', ['alfworld', 'bfcl', 'webshop'])
@pytest.mark.parametrize('missing', ['all', 'one'])
def test_unified_manifest_bank_missing_folds_matches_explicit(
        manifest_inputs, monkeypatch, benchmark, missing):
    # Real public/sealed files and certificates; external identities use CPU
    # fixtures. No backend, environment rollout, or teacher is needed.
    with monkeypatch.context() as m:
        m.setattr(cli, 'ROOT', ROOT)
        config = cli.load_config(ROOT/f'configs/rtd/unified_{benchmark}_gemma4.yaml', arm='D3')
    config['student'] = manifest_inputs.config['student']
    monkeypatch.setattr(cli, 'evaluation_harness_identity', lambda *a: {'fixture': benchmark})
    hashes = [f'{i:064x}' for i in range(4)]
    parents = [dict(parent_hash=h, official_id=f'task-{i}', fold=i % 2) for i, h in enumerate(hashes)]
    records, payloads = [], {}
    for i, h in enumerate(hashes):
        state = FullState.create({'task_id': f'task-{i}'}, [{'role': 'user', 'content': 'reset'}], 'reset', h)
        q = f'{i+10:064x}'
        records.append(state_record(q, state, 'demo_episode', 4, 'exact'))
        payloads[q] = dict(cost=4, cost_confidence='exact',
                           behaviors=[dict(state=asdict(state), text='action')])
    manifests = []
    for legacy in (False, True):
        rows = deepcopy(parents)
        if legacy:
            for p in rows if missing == 'all' else rows[:1]:
                del p['fold']
        if benchmark == 'alfworld':
            rows = {p['parent_hash']: dict(selected_task_id=p['official_id'],
                        **({'fold': p['fold']} if 'fold' in p else {})) for p in rows}
        bank = manifest_inputs.root/('legacy' if legacy else 'explicit')
        seal_v11(bank, records, payloads, benchmark=benchmark, student=config['student'],
            public={'support.json': dict(parents=rows)}, audit={'m': 4}, inputs={})
        config.update(replay_bank_path=str(bank), support_manifest=str(bank/'public/support.json'))
        before = (bank/'public/support.json').read_bytes()
        manifest = cli.make_manifest(dict(config), 'D3', cli.bank_audit(config), smoke=True)
        assert (bank/'public/support.json').read_bytes() == before
        assert manifest['arm'] == manifest['distillation_protocol'] == 'D3'
        manifests.append(manifest)
    expected = {str(f): dict(inner_parent_groups=hashes[f::2],
                             feedback_parent_groups=hashes[1-f::2]) for f in (0, 1)}
    explicit, legacy = manifests
    assert explicit['parent_group_roles_by_fold'] == legacy['parent_group_roles_by_fold'] == expected
    assert 'parent_group_fold_derivation' not in explicit
    assert legacy['parent_group_fold_derivation'] == dict(
        rule='int(parent_hash, 16) % 2', parent_groups=hashes if missing == 'all' else hashes[:1])


def test_conventions_hashes_records_and_keyed_support_share_folds():
    hashes = ['00', '01', '02', '03']
    records = [dict(parent_hash=h, official_id=f'game-{h}/trial-1') for h in hashes]
    keyed = {p['parent_hash']: dict(selected_task_id=p['official_id']) for p in records}
    before = deepcopy(keyed)
    explicit = [p | {'fold': int(p['parent_hash'], 16) % 2} for p in records]
    assert fold_roles(hashes) == fold_roles(records) == fold_roles(keyed) == fold_roles(explicit)
    assert fold_roles({p['parent_hash']: p['official_id'] for p in records}) == fold_roles(explicit)
    resolved, derived = resolve_parent_folds(keyed)
    assert derived == hashes and [p['official_id'] for p in resolved] == [p['official_id'] for p in records]
    assert keyed == before
    # Explicit assignments stay authoritative, even when they differ from parity.
    keyed['00']['fold'] = 1
    assert fold_roles(keyed)['1']['inner_parent_groups'] == ['00', '01', '03']
    assert resolve_parent_folds(keyed)[1] == ['01', '02', '03']


@pytest.mark.parametrize('parent', [{'parent_hash': None}, {'parent_hash': 'not-hex'}])
def test_conventions_missing_fold_requires_usable_parent_hash(parent):
    with pytest.raises(ValueError):
        fold_roles([parent])
