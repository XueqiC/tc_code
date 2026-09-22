"""CPU oracles for the additive fidelity variants and immutable offline receipts."""
import json
from types import SimpleNamespace

import pytest
import torch

from test_alf_baseline_mechanisms import (
    BoundaryTokenizer, config, row, selection_args, scorer,
)
from test_baseline_run_training import Backend
from test_alfworld_teacher_pool import source
from test_rtd_alfworld_state import FakeStepper, render
from bfas.rtd.baselines import pi1
from bfas.rtd.baselines.alfworld_cli import main as baseline_main, parser
from bfas.rtd.baselines.alfworld_curriculum import exposure_schedule, trajectory_costs
from bfas.rtd.baselines.alfworld_selection import (
    SELECTION_RULE, read_selection, recompute_selection, score_statistics, select_candidates,
)
from bfas.rtd.baselines.alfworld_training import K32PaperTrainer, encoded_spans, prepare_training
from bfas.rtd.baselines.paper_losses import span_ce
from bfas.rtd.benchmarks import alfworld_bank as bank
from bfas.rtd.benchmarks.alfworld_support import seal_verified_bank, teacher_payload_fields, verify_package
from bfas.rtd.persistence import ComputeJournal, digest
from tools.alf_mask_inventory import inventory_bank
from tools.alf_smartad_recompute import main as recompute_main
from tools.alfworld_teacher_pool import kang_first_attempt_ids


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    import socket
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    monkeypatch.setattr(socket.socket, 'connect', lambda *a, **kw: pytest.fail('network access'))


def test_statistics_equal_for_equal_turn_lengths_and_reverse_unequal_ranking():
    equal = score_statistics([(4., 2), (10., 2)])
    assert equal['mean_nll'] == equal['turn_macro_mean_nll'] == 3.5
    a, b = score_statistics([(1., 1), (270., 90)]), score_statistics([(50., 20), (50., 20)])
    assert a['mean_nll'] > b['mean_nll']
    assert a['turn_macro_mean_nll'] == 2. < b['turn_macro_mean_nll'] == 2.5


@pytest.mark.parametrize('statistic,winner', [('token_mean', 'b'), ('turn_mean', 'c')])
def test_offline_selection_equals_direct_statistic_without_rescoring(tmp_path, statistic, winner):
    cs, tasks, identity, cost = selection_args()
    original = tmp_path/'original'
    select_candidates(cs, tasks, scorer, original, identity, cost)
    before = {p: p.read_bytes() for p in original.rglob('*') if p.is_file()}
    expected = select_candidates(cs, tasks, scorer, tmp_path/'direct', identity, cost, statistic=statistic)
    actual = recompute_selection(original/'scores', tmp_path/'offline', statistic=statistic)
    assert actual['tasks'] == expected['tasks']
    assert actual['training_rows'] == expected['training_rows']
    assert actual['teacher_data_cost'] == cost
    assert actual['rule'] == expected['rule']
    assert next(t for t in actual['tasks'] if t['task_id'] == 'task')['chosen_candidate_id'] == winner
    rows, _ = read_selection(tmp_path/'offline/selection.json', identity['student'])
    assert len(rows) == len(expected['training_rows'])
    assert {p: p.read_bytes() for p in original.rglob('*') if p.is_file()} == before


def test_recompute_cli_defaults_to_turn_mean_and_refuses_trajectory_totals(tmp_path, capsys):
    cs, tasks, identity, cost = selection_args()
    original = tmp_path/'original'
    select_candidates(cs, tasks, scorer, original, identity, cost)
    assert recompute_main(['--scores', str(original/'scores'), '--output', str(tmp_path/'new')]) == 0
    assert json.loads(capsys.readouterr().out)['statistic'] == 'turn_mean'
    for path in (original/'scores').glob('*.json'):
        score = json.loads(path.read_text())
        del score['turn_scores'], score['turn_macro_mean_nll']
        path.write_text(json.dumps(score))
    with pytest.raises(ValueError, match='only hold trajectory totals'):
        recompute_selection(original/'scores', tmp_path/'refused')
    assert not (tmp_path/'refused').exists()


@pytest.mark.parametrize('tamper', ['turn_score', 'rule', 'version', 'winner'])
def test_selection_revalidates_statistic_and_receipts(tmp_path, tamper):
    cs, tasks, identity, cost = selection_args()
    artifact = select_candidates(cs, tasks, scorer, tmp_path/'run', identity, cost, statistic='turn_mean')
    task = next(t for t in artifact['tasks'] if t['task_id'] == 'task')
    if tamper == 'turn_score':
        task['candidates'][0]['turn_scores'][0]['total_nll'] += 1
    elif tamper == 'rule':
        artifact['rule'] = SELECTION_RULE
    elif tamper == 'version':
        artifact['version'] = 'unknown'
    else:
        task['chosen_candidate_id'] = 'b'
    artifact['artifact_hash'] = digest({k: v for k, v in artifact.items() if k != 'artifact_hash'})
    path = tmp_path/'bad.json'
    path.write_text(json.dumps(artifact))
    with pytest.raises(ValueError):
        read_selection(path, identity['student'])


def test_legacy_v1_read_and_resume_unchanged(tmp_path):
    cs, tasks, identity, cost = selection_args()
    output = tmp_path/'legacy'
    artifact = select_candidates(cs, tasks, scorer, output, identity, cost)
    for name in ('manifest.json', 'selection.json'):
        value = json.loads((output/name).read_text())
        value.update(version='alfworld-smartad-selection-v1', rule=SELECTION_RULE)
        del value['statistic']
        if name == 'selection.json':
            for task in value['tasks']:
                for c in task['candidates']:
                    del c['turn_scores'], c['turn_macro_mean_nll']
            value['artifact_hash'] = digest({k: v for k, v in value.items() if k != 'artifact_hash'})
        (output/name).write_text(json.dumps(value))
    for path in (output/'scores').glob('*.json'):
        value = json.loads(path.read_text())
        del value['turn_scores'], value['turn_macro_mean_nll']
        path.write_text(json.dumps(value))
    (output/'candidates.json').unlink()
    before = (output/'selection.json').read_bytes()
    rows, _ = read_selection(output/'selection.json', identity['student'])
    assert len(rows) == len(artifact['training_rows'])
    select_candidates(cs, tasks, lambda r: pytest.fail('legacy rescore'), output, identity, cost, resume=True)
    assert (output/'selection.json').read_bytes() == before


def test_weight_sum_and_token_count_arithmetic_and_gradients():
    values = -torch.tensor([1., 2., 3., 4., 99.], dtype=torch.float64, requires_grad=True)
    kinds = ['reason', 'reason', 'reason', 'action', 'observation']
    assert float(span_ce(values, kinds, 'smartad').detach()) == 3.
    loss = span_ce(values, kinds, 'smartad_wsum')
    assert float(loss.detach()) == pytest.approx(12/4.5)
    assert torch.autograd.grad(loss, values)[0].tolist() == pytest.approx([-1/4.5]*3 + [-1.5/4.5, 0])


def test_weight_norm_training_matches_update_wide_autograd_oracle(tmp_path):
    cfg = config(pi1)
    rows = [row('a', target='THOUGHT: long reasoning\nACTION: look'),
            row('b', target='ACTION: x')]
    backend = Backend()
    backend.tokenizer = BoundaryTokenizer()
    encodings = [encoded_spans(backend.tokenizer, r, 32768) for r in rows]
    costs = [sum(k != 'observation' for k in kinds) for _, kinds in encodings]
    identity = dict(bank={}, teacher_data_cost={})
    manifest, plan = prepare_training(tmp_path/'run', rows, cfg, 0, 'smartad', identity, costs,
                                      smartad_variant='weight_norm')
    hp = manifest['hyperparameters']
    assert hp['smartad_variant'] == 'weight_norm' and hp['loss_variant'] == 'smartad_wsum'
    expected = backend.model.lora_logits.detach().clone().requires_grad_()
    opt = torch.optim.AdamW([expected], lr=hp['learning_rate'], betas=tuple(hp['adam_betas']),
                           eps=hp['adam_epsilon'], weight_decay=hp['weight_decay'], foreach=False, fused=False)
    for batch in plan['batches']:
        numerator, denominator = 0, 0
        for i in batch['indices']:
            encoded, kinds = encodings[i]
            weights = expected.new_tensor([{'reason': 1., 'action': 1.5, 'final': 2., 'observation': 0.}[k]
                                           for k in kinds])
            numerator -= (expected.log_softmax(0)[list(encoded['target_ids'])]*weights).sum()
            denominator += weights.sum()
        opt.zero_grad(set_to_none=True)
        (numerator/denominator).backward()
        torch.nn.utils.clip_grad_norm_([expected], hp['gradient_clip'])
        opt.step()
    trainer = K32PaperTrainer(backend, rows, cfg, 'smartad', tmp_path/'run',
        ComputeJournal(tmp_path/'run/compute.jsonl', cuda=False), manifest=manifest, plan=plan)
    result = trainer.train()
    assert torch.allclose(backend.model.lora_logits, expected, rtol=1e-5, atol=1e-9)
    assert result['supervised_tokens'] == 10*sum(costs)
    for endpoint in (3, 10):
        saved = json.loads((tmp_path/f'run/pass-{endpoint}/manifest.json').read_text())
        assert saved['hyperparameters'] == hp


def test_trajectory_cost_orders_complete_trajectories_by_spans_not_turn_count(tmp_path):
    rows = [row('long_single', target='THOUGHT: '+ 'x'*60+'\nACTION: look'),
            row('short_multi', index=1, target='b'), row('short_multi', index=0, target='a'),
            row('middle', target='ACTION: '+ 'x'*10)]
    tok, cfg = BoundaryTokenizer(), config(pi1)
    scores = trajectory_costs(rows, tok)
    assert scores['short_multi'] == dict(reason_tokens=0, action_tokens=2, cost=2.)
    costs = [len(tok.encode(r.target))+1 for r in rows]
    manifest, plan = prepare_training(tmp_path/'run', rows, cfg, 0, 'sad',
        dict(bank={}, teacher_data_cost={}), costs, sad_curriculum='trajectory_cost', trajectory_scores=scores)
    for order in plan['pass_orders']:
        assert order == [2, 1, 3, 0]
    assert plan['target_tokens'] == [3*sum(costs), 10*sum(costs)]
    assert plan['curriculum']['gamma'] == 0. and 'unavailable' in plan['curriculum']['entropy']
    assert manifest['hyperparameters']['sad_curriculum'] == 'trajectory_cost'
    legacy = exposure_schedule(rows, costs, cfg, 0, 'sad')
    assert legacy['pass_orders'][0][-2:] != [3, 0]
    weighted = trajectory_costs(rows, tok, alpha=2., beta=3.)
    for q, score in scores.items():
        assert weighted[q]['cost'] == 2*score['reason_tokens'] + 3*score['action_tokens']


@pytest.mark.parametrize('marker', [False, True])
def test_inventory_real_loader_on_synthetic_sealed_bank(source, tmp_path, marker):
    support = json.loads((source/'public/support.json').read_text())
    resets = json.loads((source/'public/reset_states.json').read_text())
    records = json.loads((source/'public/requests.json').read_text())
    payloads = {}
    authored_marker = 'Observation: authored note\n'
    for i, record in enumerate(records):
        qid = record['spec']['query_id']
        payload = json.loads((source/'sealed'/f'{qid}.json').read_text())
        episode = payload['historical_response']
        episode['demo']['teacher_commands'] = payload['commands']
        for j, (turn, command) in enumerate(zip(episode['demo']['turns'], payload['commands'])):
            turn['target'] = ('THOUGHT: plan\n' + (authored_marker if marker and i == j == 0 else '')
                              + 'ACTION: '+command)
        payload.update(teacher_payload_fields(episode), status='candidate', raw_ledger_line=json.dumps(episode))
        request = support['tasks'][payload['provenance']['task_id']]['request']
        payloads[qid] = verify_package(payload, request, FakeStepper(request), render)
    output = tmp_path/'synthetic'
    public = [bank.public_record(qid, support['tasks'][p['provenance']['task_id']]['request'])
              for qid, p in payloads.items()]
    archive = bank.Archive(public, payloads, {}, [], dict(source_files=[]))
    seal_verified_bank(output, archive, support, payloads, resets)
    class Renderer:
        adapter = SimpleNamespace(_tokenizer=BoundaryTokenizer())
        def __call__(self, request, history):
            return render(request, history)
    before = {p: p.read_bytes() for p in output.rglob('*') if p.is_file()}
    result = inventory_bank(output, Renderer())
    assert result['rows'] == 6 and result['packages'] == 3
    assert result['masked_authored_tokens'] == (len(authored_marker) if marker else 0)
    assert result['affected_rows'] == result['affected_packages'] == int(marker)
    assert result['authored_tokens'] + result['native_boundary_tokens'] == (
        result['supervised_tokens'] + result['masked_authored_tokens'])
    assert {p: p.read_bytes() for p in output.rglob('*') if p.is_file()} == before


def test_kang_first_attempt_never_substitutes_first_success():
    payloads = {q: dict(status=status, provenance=dict(attempt_index=attempt))
                for q, status, attempt in [('a', 'usable', 0), ('b', 'unavailable', 0),
                                           ('c', 'usable', 1), ('d', 'usable', 2)]}
    assert kang_first_attempt_ids(payloads) == ['a']


def test_cli_flags_and_wrong_method_rejected_before_gpu(tmp_path):
    argv = ['train', '--method', 'smartad', '--seed', '0', '--output', str(tmp_path/'run'),
            '--config', 'missing', '--model-path', 'missing', '--gpu-uuid', 'unused']
    assert parser().parse_args(argv + ['--smartad-variant', 'weight_norm']).smartad_variant == 'weight_norm'
    with pytest.raises(ValueError, match='require --method sad'):
        baseline_main(argv + ['--sad-curriculum', 'trajectory_cost'])
    argv[2] = 'sad'
    with pytest.raises(ValueError, match='requires --method smartad'):
        baseline_main(argv + ['--smartad-variant', 'weight_norm'])
