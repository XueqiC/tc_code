"""Offline targeted acquisition: frozen support, purchase isolation, and D0 union."""
import json
from pathlib import Path

import pytest

from test_alf_repair_pipeline import (
    Renderer, cfg_for, d0, offline_cpu, source, Teacher, factory, FakeStepper,
)
from test_rtd_alfworld_support import write_json
from bfas.rtd.baselines.pi1 import load_bank
from bfas.rtd.benchmarks.alfworld_support import freeze_support
from tools import alf_targeted_tasks as selector, alfworld_teacher_pool as pool
from tools import alf_bank_union as union
from tools.alf_bank_subset import read_json, usable_packages


OPTIONS = dict(method='smartad', candidates_per_task=2, attempts_per_task=4,
               attempt_start=6, temperature=.7, sweep_filter=True)


@pytest.mark.parametrize('workers', [1, 3])
def test_only_tasks_purchases_and_export_union_with_d0(d0, tmp_path, workers):
    original = read_json(d0 / 'public/support.json')
    selected = original['historical_task_ids'][:1]
    task_file, out = tmp_path / 'tasks.json', tmp_path / 'targeted'
    write_json(task_file, selected)
    teacher = Teacher()
    summary = pool.collect(d0, out, only_tasks=task_file, workers=workers,
                           stepper_factory=FakeStepper, adapter_factory=factory(teacher), **OPTIONS)
    assert summary['tasks'] == summary['tasks_attempted'] == summary['tasks_at_target'] == 1
    assert summary['historical_tasks'] == 2 and summary['verified'] == 2
    assert summary['shortfall'] == 0 and summary['stop_reason'] == 'complete'
    assert len(teacher.calls) == 4 and all(c[2]['temperature'] == .7 for c in teacher.calls)
    directory = out.with_name(out.name + '.collection')
    assert read_json(directory / 'identity.json')['only_task_ids'] == selected
    rows = pool.read_records(Path(summary['ledger']))
    calls = [json.loads(line) for line in (directory / 'usage.jsonl').read_text().splitlines()]
    assert {r['task_id'] for r in rows + calls} == set(selected)
    assert {r['attempt_index'] for r in rows + calls} == {6, 7}
    assert set(read_json(directory / 'candidate_sets.json')['tasks']) == set(selected)
    support = read_json(out / 'public/support.json')
    for field in ('historical_task_ids', 'historical_parent_groups', 'parents', 'm', 'fold_parent_counts'):
        assert support[field] == original[field]
    assert support['training_task_ids'] == selected
    assert support['training_selection'] == 'usable_packages'
    assert sorted(read_json(out / 'public/reset_states.json')) == original['historical_task_ids']
    assert pool.audit_verified_bank(out)['passed']
    assert {p['provenance']['task_id'] for p in usable_packages(out).values()} == set(selected)
    retry = Teacher()
    assert pool.collect(d0, out, only_tasks=task_file, workers=workers, stepper_factory=FakeStepper,
                        adapter_factory=factory(retry), **OPTIONS) == summary
    assert retry.calls == []
    before = [{str(p.relative_to(b)): p.read_bytes() for p in b.rglob('*.json')} for b in (d0, out)]
    combined = tmp_path / 'union'
    report = union.materialize_union(d0, out, combined, mixing='all', renderer=Renderer())
    assert report['audit']['passed'] and report['audit']['usable_packages'] == 4
    assert read_json(combined / 'public/support.json') == original
    derivation = read_json(combined / 'sealed/manifest.json')['derivation']
    assert derivation['compatibility']['statement'] == 'code-identity-only difference'
    expected = [row for b in (d0, out) for row in load_bank(b, cfg_for(b), Renderer())[0]]
    assert load_bank(combined, cfg_for(combined), Renderer())[0] == sorted(
        expected, key=lambda row: (row.package_id, row.index))
    assert before == [{str(p.relative_to(b)): p.read_bytes() for p in b.rglob('*.json')} for b in (d0, out)]
    write_json(task_file, original['historical_task_ids'])
    with pytest.raises(ValueError, match='resume'):
        pool.collect(d0, out, only_tasks=task_file, adapter_factory=factory(retry), **OPTIONS)
    assert retry.calls == []


@pytest.mark.parametrize('selection', [['outside/trial'], {}, [123], ['duplicate', 'duplicate']])
def test_only_tasks_rejects_invalid_ids_before_purchase(source, tmp_path, selection):
    task_file, out = tmp_path / 'tasks.json', tmp_path / 'targeted'
    write_json(task_file, selection)
    teacher = Teacher()
    with pytest.raises(ValueError, match='--only-tasks'):
        pool.collect(source, out, only_tasks=task_file, adapter_factory=factory(teacher))
    assert teacher.calls == []
    assert not out.exists() and not out.with_name(out.name + '.collection').exists()


@pytest.mark.parametrize('empty', [False, True])
def test_only_tasks_failed_or_empty_selection_preserves_history(source, tmp_path, empty):
    support = read_json(source / 'public/support.json')
    selected = [] if empty else support['historical_task_ids'][:1]
    task_file, out = tmp_path / 'tasks.json', tmp_path / 'targeted'
    write_json(task_file, selected)
    teacher = Teacher(error=ValueError('stub failure'))
    summary = pool.collect(source, out, only_tasks=task_file, stepper_factory=FakeStepper,
                           adapter_factory=factory(teacher), **OPTIONS)
    assert summary['verified'] == 0 and summary['attempts'] == (0 if empty else 4)
    rows = pool.read_records(Path(summary['ledger']))
    assert {r['task_id'] for r in rows} == set(selected)
    assert {r['attempt_index'] for r in rows} == (set() if empty else {6, 7, 8, 9})
    assert len(teacher.calls) == len(rows)
    assert (summary['tokens'] > 0) is (not empty)
    exported = read_json(out / 'public/support.json')
    assert exported['historical_task_ids'] == support['historical_task_ids']
    assert exported['training_task_ids'] == []
    assert pool.audit_verified_bank(out)['passed']


def test_only_tasks_cli_dry_run_is_read_only(source, tmp_path, capsys):
    task_file, out = tmp_path / 'tasks.json', tmp_path / 'targeted'
    write_json(task_file, read_json(source / 'public/support.json')['historical_task_ids'][:1])
    before = {str(p): p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    pool.main(['--source', str(source), '--out', str(out), '--only-tasks', str(task_file),
               '--dry-run', '--method', 'smartad', '--candidates-per-task', '2',
               '--attempts-per-task', '4', '--attempt-start', '6', '--temperature', '.7', '--sweep-filter'])
    plan = json.loads(capsys.readouterr().out)
    assert plan['dry_run'] and plan['teacher_calls'] == 0
    assert plan['tasks'] == 1 and plan['historical_tasks'] == 2
    assert plan['attempt_indices'] == [6, 7, 8, 9] and plan['temperatures'] == [.7] * 4
    assert plan['max_candidates'] == 2 and plan['max_attempts'] == 4 and plan['sweep_filter']
    assert before == {str(p): p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}


@pytest.fixture
def typed_support(tmp_path):
    root = tmp_path / 'archive-types'
    types = [*selector.DEFAULT_TYPES, *selector.DEFAULT_TYPES, 'pick_and_place_simple',
             'pick_cool_then_place_in_recep_extra']
    tids = [f'{kind}-Apple-None-Table-{i}/trial-1' for i, kind in enumerate(types)]
    write_json(root / pool.bank.SUPPORT, dict(demand=tids, calibration=[]))
    write_json(root / pool.bank.PROBE, [], lines=True)
    write_json(root / pool.bank.LEDGER, [dict(task_id=t) for t in tids], lines=True)
    for tid in tids:
        world = root / 'envs/alfworld/data/json_2.1.1/train' / tid
        write_json(world / 'game.tw-pddl', dict(grammar={'task': [{'rhs': 'Your task is to: put apple on table.'}]}))
        write_json(world / 'traj_data.json', dict(task_id=tid))
        write_json(world / 'initial_state.pddl', dict(room=tid))
    environment = dict(fixture='CPU only')
    environment['environment_hash'] = pool.canonical_hash(environment)
    return freeze_support(root, environment)


def test_targeted_selector_default_types_and_cli(typed_support, tmp_path, capsys):
    support_file, out = tmp_path / 'support.json', tmp_path / 'tasks.json'
    write_json(support_file, typed_support)
    selector.main(['--support', str(support_file), '--output', str(out)])
    selected = read_json(out)
    assert selected == sorted(set(selected)) and len(selected) == 6
    output = capsys.readouterr().out
    for kind in selector.DEFAULT_TYPES:
        assert f'{kind}: 2' in output
    assert all(t.split('-', 1)[0] in selector.DEFAULT_TYPES for t in selected)
    with pytest.raises(FileExistsError):
        selector.main(['--support', str(support_file), '--output', str(out)])
    custom = tmp_path / 'custom.json'
    selector.main(['--support', str(support_file), '--output', str(custom),
                   '--task-types', 'pick_and_place_simple'])
    assert len(read_json(custom)) == 1
    for invalid in (['pick_cool'], ['unknown'], ['pick_and_place_simple'] * 2):
        with pytest.raises(ValueError):
            selector.targeted_tasks(typed_support, invalid)
