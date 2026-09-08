"""D5/D6 CPU arithmetic, frozen interventions and reporting with TinyLM."""
from dataclasses import replace
import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from bfas.behavior.deltas import tensor_state_hash
from bfas.rtd import alpha_d
from bfas.rtd.controls_v11 import (flip_pairs, load_window, run_controls, save_window, shuffled_pairing,
                                  validate_window, window_payload, place_window)
from bfas.rtd.functional_step import lora_parameters
from bfas.rtd.metrics_v11 import (correlation, freeze_tasks, gradient_variance, greedy_success, norm_variance,
    paired_correlations, parse_battery, parse_rate, prediction_errors, repair_damage, task_rows, window_metrics)
from bfas.rtd.persistence import ComputeJournal, atomic_json
from test_rtd_checks import toy_bank
from test_rtd_v11_alpha_d import engine, pair_problem
from test_rtd_v11_joint_controller import reveal
from tools.rtd_v11_controls import render_markdown as render_controls
from tools.rtd_v11_report import collect, render_markdown, write_report


class Syntax:
    checker_version = 'tiny-syntax-v1'
    def check_syntax(self, text):
        return dict(valid=bool(text))


def enabled_engine(tmp_path, toy_bank, **kw):
    kw.setdefault('budget_checkpoints_bank_fraction', [.1, .25])
    e = engine(tmp_path, toy_bank, metrics_v11=dict(enabled=True, variance_resamples=2), **kw)
    e.checker = Syntax()
    return e


def test_fixed_twenty_per_fold_is_deterministic_explicit_and_manifest_bound(toy_bank, tmp_path):
    e = enabled_engine(tmp_path/'run', toy_bank)
    fixed = e.manifest['fixed_task_set_v11']
    parents = [dict(parent_hash=h, official_id=t) for h, t in e.support.parents.items()]
    assert freeze_tasks(parents[::-1], states=e.support.states) == fixed
    assert len(task_rows(fixed, e.support)) == 40
    assert fixed['samples_per_state'] == 4 and fixed['temperature'] == 1
    assert all(f['unique_states'] == 2 and f['repeated_slots'] == 18 for f in fixed['folds'].values())
    assert json.loads((e.directory/'manifest.json').read_text())['fixed_task_set_v11'] == fixed
    with pytest.raises(ValueError, match='too small'):
        freeze_tasks(parents, short_fold='error')
    broken = copy.deepcopy(fixed); broken['folds']['0']['states'].reverse()
    with pytest.raises(ValueError, match='changed'):
        task_rows(broken, e.support)
    original = e.support.states[parents[0]['parent_hash']]
    e.support.states[parents[0]['parent_hash']] = replace(original, prompt='changed prompt')
    with pytest.raises(ValueError, match='full-state content changed'):
        task_rows(fixed, e.support)
    e.support.states[parents[0]['parent_hash']] = original
    del e.support.states[parents[0]['parent_hash']]
    with pytest.raises(ValueError, match='never substitute'):
        task_rows(fixed, e.support)
    # Before run start, unavailable initial-state adapters are excluded publicly.
    smaller = freeze_tasks(parents, states=e.support.states)
    assert smaller['excluded_unavailable_parents'] == [parents[0]['parent_hash']]
    assert all(r['parent_hash'] != parents[0]['parent_hash'] for f in smaller['folds'].values() for r in f['states'])


def test_parse_battery_uses_checker_160_fresh_T1_draws_and_independent_rng(toy_bank, tmp_path):
    e = enabled_engine(tmp_path/'parse', toy_bank)
    before = e.sampling_rng.get_state().clone()
    calls, draw_count = [], []
    sample = e.backend.sample_action
    def draw(*a, **kw):
        draw_count.append(kw)
        return sample(*a, **kw)
    class Alternating:
        def check_syntax(self, text):
            calls.append(text)
            return dict(valid=len(calls) % 2 == 0)
    e.backend.sample_action = draw
    result = parse_battery(e.manifest['fixed_task_set_v11'], e.support, e.backend, e.state['parameters'],
                           Alternating(), identity='test')
    assert len(calls) == len(draw_count) == 160
    assert all(kw == dict(temperature=1., top_p=1.) for kw in draw_count)
    assert result['failure_rate'] == .5 and result['failures'] == 80
    assert all(f['samples'] == 80 and f['failure_rate'] == .5 for f in result['by_fold'].values())
    assert torch.equal(e.sampling_rng.get_state(), before)


@pytest.mark.parametrize('text,valid', [('', False), ('plain assistant response', True),
    ('<tool_call>\n{"name":"f","arguments":{}}\n</tool_call>', True),
    ('<tool_call>\n{"name":"f","arguments":[]}\n</tool_call>', False),
    ('<tool_call>broken</tool_call>', False)])
def test_checker_bridge_syntax_reuses_rtd_decoding_guard(text, valid):
    from tools.behavior_atom.checker_bridge import CheckerBridge
    with CheckerBridge() as checker:
        assert checker.check_syntax(text)['valid'] is valid


def test_syntax_bridge_subprocess_is_separate_from_official_worker():
    import sys
    from tools.behavior_atom.checker_bridge import CheckerBridge
    with CheckerBridge(python=sys.executable) as checker:
        checker._checker = None
        assert checker.check_syntax('plain response')['valid'] is True
        assert checker.check_syntax('<tool_call>broken</tool_call>')['valid'] is False
        assert checker._proc is None
        assert checker._syntax_bridge._proc is not None
    assert checker._syntax_bridge._proc is None


def test_metric_arithmetic_and_degenerate_correlations():
    assert parse_rate([dict(success=True), dict(success=False)])['failure_rate'] == .5
    v = norm_variance([1., 2., 3.])
    assert v['variance'] == pytest.approx(2/3) and v['correction'] == 0
    assert correlation([1., 2., 3.], [6., 4., 2.])['pearson'] == pytest.approx(-1.)
    assert correlation([1.], [2.])['pearson'] is None
    assert correlation([1., 1.], [2., 3.])['reason'] == 'constant_prediction_or_gain'
    error = prediction_errors([1., 4.], [2., 2.])
    assert error['mean_error'] == -.5 and error['mae'] == 1.5 and error['rmse'] == pytest.approx(2.5**.5)
    counts = repair_damage(dict(a=False, b=True, c=True, d=False), dict(a=True, b=False, c=True, d=False))
    assert counts == dict(tasks=4, before_success=2, after_success=2, repaired=1, damaged=1,
                          retained=1, still_failed=1, net_repair=0)
    with pytest.raises(ValueError):
        repair_damage({'a': False}, {'b': True})
    with pytest.raises(ValueError):
        norm_variance([1.])
    with pytest.raises(ValueError):
        correlation([1., float('nan')], [2., 3.])


def test_shuffling_preserves_local_sources_teachers_signed_gate_values_and_alpha():
    b, start, source, pairs, step = pair_problem()
    second = pairs[1]
    state = replace(second.record.state, prompt='prompt another state')
    teacher = replace(second.record.teacher, state=state)
    second = replace(second, record=replace(second.record, state=state, teacher=teacher), sources=tuple(
        replace(src, behavior=replace(src.behavior, state=state)) for src in second.sources))
    pairs = (pairs[0], second)
    shuffled, mapping = shuffled_pairing(pairs, seed=42)
    assert shuffled_pairing(pairs, seed=42) == (shuffled, mapping)
    assert mapping['cross_state_assignment'] and any(mapping['swapped_exposures'])
    assert sorted(mapping['y1_index_by_state']) == sorted(mapping['assignment_pool'])
    unchanged, identity_mapping = shuffled_pairing(pairs, seed=0)
    assert identity_mapping['state_permutation'] == [0, 1] and unchanged == pairs
    assert not identity_mapping['changed']
    a, w, d = [.2, .7], [.3, .7], [.1, -.2]
    ref = alpha_d.build_reference(pairs, w, a, b, start, source, step)
    alternative = alpha_d.build_reference(shuffled, w, a, b, start, source, step)
    torch.testing.assert_close(ref.theta0['lora_transition'], alternative.theta0['lora_transition'])
    assert torch.equal(ref.alpha, alternative.alpha) and torch.equal(ref.weights, alternative.weights)
    assert torch.equal(alpha_d.target_weights(ref.alpha, torch.tensor(d)),
                       alpha_d.target_weights(alternative.alpha, torch.tensor(d)))
    for old, new in zip(pairs, shuffled):
        assert old.record is new.record and {id(x) for x in old.sources} == {id(x) for x in new.sources}
        assert all(src.behavior.state == old.record.state for src in new.sources)
    swapped = flip_pairs(pairs, [True, True])
    symmetric = alpha_d.build_reference(swapped, w, a, b, start, source, step)
    for n in start:
        torch.testing.assert_close(ref.selected(d)[n], symmetric.selected(-np.asarray(d))[n])


def test_equal_alpha_gradient_variance_resamples_and_keeps_teacher_when_filter_rejects():
    from bfas.rtd.controls_v11 import sampler
    from types import SimpleNamespace
    b, start, source, pairs, step = pair_problem()
    records = [p.record for p in pairs]
    support = SimpleNamespace(parents={r.state.parent_hash: 'task' for r in records}, categories={'task': 'simple'})
    payload = dict(source=source, window_id='variance')
    draw = sampler(payload, b, support, identity='variance')
    calls = []
    def counted(*args):
        calls.append(args[1:]); return draw(*args)
    class Reject:
        def check_syntax(self, text):
            return dict(valid=False)
    first = []
    v = gradient_variance(records, [.25, .75], [.2, .7], b, start, source, step,
        Reject(), counted, R=3, on_first=lambda batches: first.append(batches))
    assert len(calls) == 3*8*2 and len(first) == 1
    assert v['rejected_hard2_by_repeat'] == [4, 4, 4]
    assert v['estimators']['soft'] == v['estimators']['syntax_filtered_hard2']
    for batch in first[0].values():
        np.testing.assert_allclose(batch.alpha, [.2, .7])
        np.testing.assert_allclose(batch.weights, [.25, .75])


def test_tiny_controls_reuses_D9_updates_solver_feedback_and_outputs_json_md(toy_bank, tmp_path, monkeypatch):
    from bfas.rtd import joint_surrogate
    e = enabled_engine(tmp_path/'frozen', toy_bank, smoke=False, d_warmup_windows=0)
    reveal(e); e.actual()
    path = save_window(e)
    path.with_suffix('.json').unlink()  # interrupted before descriptor publication
    assert save_window(e) == path
    payload = load_window(path)
    placed = place_window(payload, device='cpu')
    assert placed['batch'].directions is payload['batch'].directions
    assert placed['batch'].baseline_gradients is payload['batch'].baseline_gradients
    assert placed['statistic'] is payload['statistic']
    assert all(v.device.type == 'cpu' for direction in placed['batch'].directions for v in direction.values())
    assert tensor_state_hash(placed['start']) == payload['batch'].start_hash
    assert payload['independent_insertion_control'] == e.state['independent_control_labels']
    calls, solves = [], []
    execute, solve = joint_surrogate.execute_update, joint_surrogate.solve_d
    def counted(batch, d, start, step):
        calls.append(tensor_state_hash(start)); return execute(batch, d, start, step)
    def solver(*a, **kw):
        solves.append(True); return solve(*a, **kw)
    monkeypatch.setattr(joint_surrogate, 'execute_update', counted)
    monkeypatch.setattr(joint_surrogate, 'solve_d', solver)
    original = tensor_state_hash(lora_parameters(e.backend.model))
    training_rng = e.sampling_rng.get_state().clone()
    report = run_controls(payload, e.backend, e.support, e.checker,
        ComputeJournal(tmp_path/'controls-compute.jsonl'), R=2, z_probes=1)
    assert calls and set(calls) == {payload['batch'].start_hash} and solves
    assert tensor_state_hash(lora_parameters(e.backend.model)) == original
    assert torch.equal(e.sampling_rng.get_state(), training_rng)
    kinds = [r['comparison'] for r in report['pairs']]
    assert kinds[:4] == ['learned_d_vs_zero', 'source_swap_symmetry', 'learned_d_vs_shuffled_pairing', 'joint_vs_independent_control']
    assert 'acquisition_surrogate' in kinds and 'z_direction' in kinds
    ids = []
    for pair in report['pairs']:
        assert pair['updates_per_arm'] == 1 and pair['start_hash'] == payload['batch'].start_hash
        for side in ('full', 'control'):
            row = pair[side]; ids.append(row['batch_id'])
            assert row['metrics']['parse']['samples'] == 160
            assert row['metrics']['repair_damage']['tasks'] == 40
            assert row['batch_id'] != payload['selection_feedback_batch']
        if pair['equal_alpha']:
            assert pair['full']['metrics']['teacher_mass'] == pair['control']['metrics']['teacher_mass']
    assert len(ids) == len(set(ids))
    assert [r['estimator'] for r in report['source_estimator_diagnostics']] == ['hard2', 'soft', 'hard8', 'syntax_filtered_hard2']
    assert '同 α' in render_controls(report)
    json.dumps(report, allow_nan=False)
    # Role must be checked even when the parameter hash is unchanged.
    for role in ('acquisition_reference_feedback', 'post_commit_feedback'):
        broken = dict(payload, statistic=replace(payload['statistic'], feedback_role=role))
        with pytest.raises(ValueError, match='same_batch_reference_feedback'):
            validate_window(broken)
    path.write_bytes(path.read_bytes()+b'tampered')
    with pytest.raises(ValueError, match='hash/schema'):
        load_window(path)


def test_enabled_tiny_run_round_metrics_and_resume_preserve_durable_outputs(toy_bank, tmp_path):
    e = enabled_engine(tmp_path/'round', toy_bank)
    e.run()
    result = json.loads((e.directory/'metrics/round-1.json').read_text())
    assert result['parse']['samples'] == 160
    assert result['gradient_variance']['source_draws'] == 2*8*20
    assert set(result['gradient_variance']['fixed_states']) <= {e.support.states[h].state_hash for h in e.state['inner']}
    assert (e.directory/'controls/windows/r1-s01.pt').exists()
    before = (e.directory/'metrics/round-1.json').read_bytes()
    restored = enabled_engine(e.directory, toy_bank, resume=True)
    restored.run()
    assert (e.directory/'metrics/round-1.json').read_bytes() == before
    assert restored.state['steps'] == e.state['steps']
    # Real committed metric rows render through the read-only report collector.
    runs = tmp_path/'report-runs'; runs.mkdir()
    (runs/'V2').symlink_to(e.directory, target_is_directory=True)
    report = collect(tmp_path, runs=runs, base_overall=46.06)
    assert report['arms'][2]['rows'][0]['metrics']['parse']['samples'] == 160
    md = render_markdown(report)
    assert 'alpha_d' in md and '同α分支' in md


def test_window_accounting_counts_commits_not_compute_attempts():
    def row(r, step, spend, selected):
        return dict(round=r, step=step, decision=True, selected=selected, actual_spend=spend,
            authorized_budget=100*r, remaining_budget=100*r-spend, window_id=f'{r}-{step}',
            selection=dict(stop_reason='hard_cap_tail', budget_binding=True))
    rows = [row(1, 1, 20, ['q']), row(1, 4, 20, []), row(2, 1, 35, ['r'])]
    stats = window_metrics(rows)
    assert [s['spend'] for s in stats] == [20, 0, 15]
    assert [s['remaining_authorization'] for s in stats] == [80, 80, 165]
    with pytest.raises(ValueError, match='duplicate'):
        window_metrics(rows+[rows[0]])


def report_fixture(root):
    from tools.rtd_v1_collect_report import AXES
    import csv
    parents = [dict(parent_hash=f'{i:064x}', official_id=f'task{i}') for i in range(40)]
    fixed = freeze_tasks(parents)
    for arm in ('V0', 'V1', 'V2'):
        run = root/'results/rtd_v1_1'/arm
        run.mkdir(parents=True)
        atomic_json(run/'manifest.json', dict(arm=arm, budget_denominator=55370, budget_ceilings=[5537, 13843],
            fixed_task_set_v11=fixed, config=dict(protocol_version='1.1.0', rounds=2, benchmark='bfcl',
                budget_checkpoints_bank_fraction=[.1, .25], base_overall=46.06)))
        for r in (1, 2):
            tag = f'{arm}-r{r}'
            campaign = root/'results/bfcl_std_hpg'/tag
            campaign.mkdir(parents=True)
            with (campaign/'data_overall.csv').open('w') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(AXES.values()))
                writer.writeheader(); writer.writerow({v: 50+r for v in AXES.values()})
            atomic_json(run/f'evaluation-{r}.json', dict(campaign_identity=dict(tag=tag), validation=dict(complete=True)))
    return root


def test_report_two_budgets_official_scores_missing_metrics_and_new_outputs(tmp_path):
    root = report_fixture(tmp_path)
    report = collect(root)
    assert len(report['arms']) == 3 and all(len(a['rows']) == 2 for a in report['arms'])
    assert report['arms'][0]['rows'][0]['official']['scores']['Overall'] == 51
    md = render_markdown(report)
    assert all(arm in md for arm in ('V0', 'V1', 'V2')) and '10% Overall' in md and '25% Overall' in md
    assert 'r3' not in md and '(未完成)' in md and '46.06' in md
    out = tmp_path/'report.md'
    write_report(report, out)
    assert out.read_text() == md and json.loads(out.with_suffix('.json').read_text())['budget_points'] == [.1, .25]
    with pytest.raises(FileExistsError):
        write_report(report, out)


def test_report_rejects_mixed_fixed_tasks_and_invalid_official_csv(tmp_path):
    root = report_fixture(tmp_path)
    path = root/'results/rtd_v1_1/V1/manifest.json'
    m = json.loads(path.read_text()); m['fixed_task_set_v11']['hash'] = 'wrong'; atomic_json(path, m)
    with pytest.raises(ValueError, match='different fixed task sets'):
        collect(root)
