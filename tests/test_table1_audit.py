"""CPU-only accounting and source-isolation tests, plus real archive correspondence."""
from collections import Counter
from pathlib import Path

import pytest

from tools import table1_common as io
from tools.table1_budget_audit import call_record, trainer_row
from tools.table1_pool_from_sealed import build_pools, materialize, action_spans
from tools.table1_random_acquisition import acquire


def public(costs):
    return dict(schema='table1-public-v1', benchmark='bfcl', packages=[
        dict(package_id=f'{i:064x}', recorded_cost=c) for i, c in enumerate(costs)])


def fixture_packages(costs=(7, 11, 13), failed=(1,)):
    p = public(costs)
    acquisition = acquire(p, sum(costs))
    snapshots = {}
    for i, entry in enumerate(p['packages']):
        q, cost = entry['package_id'], entry['recorded_cost']
        rows = [] if i in failed else [trainer_row('task', 'teacher', 0, 'prompt', 'answer')]
        package = dict(package_id=q, task_id='task', legacy_task_id='task',
            recorded_cost=cost, failed_attempt=i in failed, teacher_calls=[
                call_record(q, 'task', 'teacher', i, i not in failed, cost, 'exact', {})])
        snapshots[q] = dict(package=package, rows=rows, pbsd_rejected={}, rank_candidates=[],
                            legacy_matches=[], bbopd_matches=[])
    protocol = dict(historical_pool_building={'known_total': 999, 'complete': False},
                    cached_ranking_rows=0, public_sha256=io.digest(p))
    return p, acquisition, snapshots, protocol


def test_deterministic_sorted_order_and_shared_training_seeds():
    p = public([1]*20)
    a = acquire(p, 10)
    p['packages'].reverse()
    for seed in [0, 1, 2]:
        b = acquire(p, 10, training_seed=seed)
        assert a['frozen_order'] == b['frozen_order']
        assert a['purchased_ids'] == b['purchased_ids']
    assert acquire(p, 10, order_seed=1)['frozen_order'] != a['frozen_order']


@pytest.mark.parametrize('cap', [0, 1, 5, 7, 11, 20, 30, 100])
def test_cap_never_exceeded_and_strict_prefix(cap):
    p = public([0, 7, 11, 3, 19])
    result = acquire(p, cap)
    assert result['C_m'] <= cap
    assert result['remaining_budget'] == cap-result['C_m']
    assert result['purchased_ids'] == result['frozen_order'][:len(result['purchased_ids'])]
    if result['next_package_id'] is not None:
        assert result['next_package_cost'] > result['remaining_budget']


def test_no_skip_of_expensive_package():
    p = public([1]*6)
    order = acquire(p, 6)['frozen_order']
    for entry in p['packages']:
        if entry['package_id'] == order[1]:
            entry['recorded_cost'] = 100
    result = acquire(p, 5)
    assert result['purchased_ids'] == order[:1]
    assert result['remaining_budget'] == 4


def test_acquisition_does_not_read_sealed_or_any_files(monkeypatch):
    def deny(*args, **kwargs):
        raise AssertionError('sealed or filesystem read during acquisition')
    monkeypatch.setattr(io, 'read_sealed', deny)
    monkeypatch.setattr(Path, 'open', deny)
    result = acquire(public([3, 4, 5]), 10)
    assert result['journal'][0]['sealed_reader_calls'] == 0


def test_private_fields_rejected():
    p = public([1])
    p['packages'][0]['verified'] = True
    with pytest.raises(ValueError, match='private fields'):
        acquire(p, 1)


@pytest.mark.parametrize('cost', [-1, True, 0.5, None])
def test_invalid_costs_fail_closed(cost):
    with pytest.raises(ValueError):
        acquire(public([cost]), 10)


def test_failures_charged_segments_not_recharged_and_star_zero():
    _, acquisition, snapshots, protocol = fixture_packages()
    q = next(q for q, s in snapshots.items() if s['rows'])
    snapshots[q]['rows'] *= 3
    pools, m = build_pools(snapshots, acquisition, protocol)
    assert m['sealed_replay_spend'] == 31
    assert m['failed_attempt_cost'] == 11
    assert len(pools['sft']) == 4
    assert m['historical_pool_building']['known_total'] == 999
    for arm in io.ARMS:
        assert m['arms'][arm]['C_m'] == (0 if arm=='star' else 31)
    assert pools['star'] == []
    assert pools['bbopd'] == pools['sft']
    for sft, sad in zip(pools['sft'], pools['sad']):
        assert sft == {k: v for k, v in sad.items() if k != '_sad_spans'}


def test_unknown_cost_rankings_not_free():
    _, acquisition, snapshots, protocol = fixture_packages()
    protocol['cached_ranking_rows'] = 2
    rank = dict(row_sha256='rank', task_id='task', recorded_cost=None)
    for s in snapshots.values():
        s['rank_candidates'] = [rank]
    pools, m = build_pools(snapshots, acquisition, protocol)
    assert pools['ddpo'] == pools['sft']
    assert m['ranking']['purchased_task_eligible'] == 1
    assert m['ranking']['reused'] == 0
    assert m['ranking']['requires_new_teacher_calls'] == 1


def test_only_exact_purchased_states_receive_student_failures_and_c():
    _, acquisition, snapshots, protocol = fixture_packages()
    valid = [s for s in snapshots.values() if s['rows']]
    valid[0]['pbsd_rejected']['0'] = 'student failure'
    valid[1]['rows'][0]['task_id'] = 'different-generated-task'
    pools, _ = build_pools(snapshots, acquisition, protocol)
    assert sum('_rejected' in r for r in pools['pbsd_insp']) == 1
    assert all(len(r['c']) == 1 for r in pools['pbsd_agent'])


def test_unique_call_accounting_and_conflicts():
    c = dict(call_id='q', recorded_cost=7)
    assert io.cost_total([c, c]) == 7
    with pytest.raises(ValueError, match='conflicting'):
        io.cost_total([c, dict(call_id='q', recorded_cost=8)])


def test_failed_payload_cannot_yield_positive():
    _, a, snapshots, protocol = fixture_packages()
    next(s for s in snapshots.values() if s['package']['failed_attempt'])['rows'] = [
        trainer_row('task', 'teacher', 0, 'prompt', 'bad')]
    with pytest.raises(ValueError, match='failed attempt'):
        build_pools(snapshots, a, protocol)


def test_reader_checks_ownership_before_open(monkeypatch, tmp_path):
    monkeypatch.setattr(Path, 'open', lambda *a, **kw: pytest.fail('opened unowned file'))
    with pytest.raises(PermissionError):
        io.read_sealed(tmp_path, '0'*64, set())


def test_materialize_only_reads_paid_snapshots_and_rejects_tamper(tmp_path, monkeypatch):
    p, _, snapshots, protocol = fixture_packages()
    acquisition = acquire(p, 7)
    directory = tmp_path/'audit'
    io.write_json(directory/'public.json', p)
    io.write_json(directory/'protocol.json', protocol)
    io.write_json(directory/'ledger.json', {'sources': {}})
    io.write_json(directory/'integrity.json', {q: io.digest(s) for q, s in snapshots.items()})
    path = directory/'acquired.json'
    io.write_json(path, acquisition)
    seen = []
    def reader(directory, q, owned):
        assert q in acquisition['purchased_ids'] and q in owned
        seen.append(q)
        return snapshots[q]
    monkeypatch.setattr(io, 'read_sealed', reader)
    # Synthetic accounting fixture has no raw bank state. Real source rendering
    # and its ownership/hash guards are exercised in test_table1_row_format.py.
    monkeypatch.setattr('tools.table1_pool_from_sealed.read_bank_payload', lambda *a: None)
    monkeypatch.setattr('tools.table1_pool_from_sealed.render_package', lambda b, s, p, **kw: s['rows'])
    materialize(directory, path, tmp_path/'pools')
    assert seen == acquisition['purchased_ids']
    acquisition['C_m'] += 1
    io.write_json(path, acquisition)
    with pytest.raises(ValueError, match='prefix'):
        materialize(directory, path, tmp_path/'pools')
    acquisition['C_m'] -= 1
    io.write_json(path, acquisition)
    snapshots[acquisition['purchased_ids'][0]]['rows'][0]['response'] = 'tampered'
    with pytest.raises(ValueError, match='integrity'):
        materialize(directory, path, tmp_path/'pools')


def test_output_refuses_readonly_data_and_other_worktrees(tmp_path):
    for path in (io.ROOT/'data/table1_pools/no.json', io.ROOT/'envs/no.json',
                 io.ROOT/'.venv/no.json', io.ROOT.parent/'other/no.json'):
        with pytest.raises(ValueError):
            io.output_path(path)
    link = tmp_path/'linked'
    link.symlink_to(io.ROOT/'data', target_is_directory=True)
    with pytest.raises(ValueError):
        io.output_path(link/'no.json')


def test_sad_spans_match_trainer_without_importing_gpu_stack():
    import ast
    source = (io.ROOT/'src/appworld_train.py').read_text()
    function = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name=='action_spans')
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), '<trainer-action-spans>', 'exec'), namespace)
    for response in ('plain', '```python\na=1\n```', 'x```a```y```\nb\n```', '```unterminated'):
        assert action_spans(response) == [list(s) for s in namespace['action_spans'](response)]


def test_real_old_pool_prompt_response_and_fields():
    checked = Counter()
    cache = {}
    for benchmark in io.BENCHMARKS:
        directory = io.DEFAULT_OUT/benchmark
        if not (directory/'ledger.json').exists():
            pytest.skip('run tools/table1_budget_audit.py to enable real-archive test')
        for path in (directory/'sealed').glob('*.json'):
            snapshot = io.read_json(path)
            for match in snapshot['legacy_matches']:
                name = match['path']
                if name not in cache:
                    cache[name] = dict(io.read_rows(io.ROOT/name))
                old = cache[name][match['line']]
                new = snapshot['rows'][match['row_index']]
                assert old['prompt'] == new['prompt']
                assert old['response'] == new['response']
                actual = sorted(k for k in set(old)|set(new) if old.get(k) != new.get(k))
                assert actual == match['differing_fields']
                assert io.digest(old) == match['old_row_sha256']
                checked[benchmark] += 1
    assert checked['bfcl'] > 0 and checked['appworld'] >= 1482


def test_real_manifests_call_accounting_and_trainer_loader():
    """Use the trainer's actual load_pool function without importing torch/peft."""
    import ast
    import json
    from typing import Any
    function = next(n for n in ast.parse((io.ROOT/'src/appworld_train.py').read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name == 'load_pool')
    namespace = dict(Path=Path, Any=Any, json=json)
    exec(compile(ast.Module(body=[function], type_ignores=[]), '<trainer-load-pool>', 'exec'), namespace)
    checked = 0
    for benchmark in io.BENCHMARKS:
        directory = io.DEFAULT_OUT/benchmark
        if not (directory/'ledger.json').exists():
            pytest.skip('run the three Table 1 tools for integration tests')
        ledger = io.read_json(directory/'ledger.json')
        packages = {p['package_id']: p for p in ledger['packages']}
        order = None
        for path in sorted(directory.glob('acquired_B*_seed*.json')):
            a = io.read_json(path)
            if order is None:
                order = a['frozen_order']
            assert order == a['frozen_order']
            calls = [c for q in a['purchased_ids'] for c in packages[q]['teacher_calls']]
            assert io.cost_total(calls) == a['C_m'] <= a['cap']
            assert a['remaining_budget'] == a['cap']-a['C_m']
            pool_dir = io.DEFAULT_OUT/'pools'/benchmark/f'B{a["cap"]}_seed{a["training_seed"]}'
            manifest = io.read_json(pool_dir/'manifest.json')
            assert manifest['failed_attempt_cost'] == sum(packages[q]['recorded_cost'] for q in a['purchased_ids']
                                                         if packages[q]['failed_attempt'])
            assert manifest['initial_checkpoint'] == io.INITIAL_CHECKPOINT
            for arm, info in manifest['arms'].items():
                assert info['C_m'] == (0 if arm == 'star' else a['C_m'])
                if a['training_seed'] == 0:
                    rows = [r for _, r in io.read_rows(pool_dir/f'pool_{arm}.jsonl')]
                    assert len(rows) == info['rows']
                    assert io.digest(rows) == info['pool_sha256']
                    if rows:
                        loaded = namespace['load_pool'](pool_dir/f'pool_{arm}.jsonl')
                        assert len(loaded) == len(rows)
                        for raw, parsed in zip(rows, loaded):
                            parsed.pop('_pool_index')
                            assert raw == parsed
            checked += 1
    assert checked == 12


def test_real_input_hashes_remain_unchanged():
    sources = {}
    for benchmark in io.BENCHMARKS:
        path = io.DEFAULT_OUT/benchmark/'ledger.json'
        if not path.exists():
            pytest.skip('run tools/table1_budget_audit.py for input hash checks')
        sources.update(io.read_json(path)['sources'])
    assert sources
    assert all(io.file_hash(io.ROOT/path) == expected for path, expected in sources.items())


def test_bfcl_preserves_recorded_generator_verdicts():
    path = io.DEFAULT_OUT/'bfcl/ledger.json'
    if not path.exists():
        pytest.skip('run the audit for real generator verdict checks')
    ledger = io.read_json(path)
    counts = Counter()
    for package in ledger['packages']:
        if package['kind'] != 'generator_item' or not package['candidate']:
            continue
        raw = io.read_json(io.ROOT/'data/rtd/v1_1_bfcl/sealed'/f'{package["package_id"]}.json')['historical_response']
        expected = raw.get('verified', raw.get('reproduced'))
        assert package['verified'] is expected
        if expected is False:
            assert package['positive_rows'] == 0
        counts[expected] += 1
    assert counts == {True: 302, False: 34}


def test_cli_refuses_to_replace_frozen_order(tmp_path, monkeypatch):
    import sys
    from tools.table1_random_acquisition import main
    directory = tmp_path/'bfcl'
    io.write_json(directory/'public.json', public([1]*6))
    io.write_json(directory/'protocol.json', {'caps': [3]})
    args = ['table1_random_acquisition.py', '--audit-root', str(tmp_path), '--benchmark', 'bfcl']
    monkeypatch.setattr(sys, 'argv', args)
    main()
    main()  # byte-stable replay of the same frozen order is allowed
    monkeypatch.setattr(sys, 'argv', args + ['--order-seed', '1'])
    with pytest.raises(ValueError, match='frozen acquisition'):
        main()


def test_source_changes_during_inventory_rejected(tmp_path, monkeypatch):
    from tools import table1_budget_audit as audit
    monkeypatch.setattr(audit, 'ROOT', tmp_path)
    source = tmp_path/'source.json'
    source.write_text('{}')
    sources = audit.Sources()
    sources.json(source)
    source.write_text('{"changed":true}')
    with pytest.raises(ValueError, match='source changed'):
        sources.json(source)
    with pytest.raises(ValueError, match='source changed'):
        sources.verify()
