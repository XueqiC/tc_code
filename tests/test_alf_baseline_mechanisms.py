"""CPU-only oracles for selection, scheduling, accounting and shared training."""
from collections import Counter
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT)]

from bfas.rtd.baselines.alfworld_cli import main, safe_output, use_pi1_root
from bfas.rtd.baselines.alfworld_cost import collection_cost
from bfas.rtd.baselines.alfworld_curriculum import exposure_schedule
from bfas.rtd.baselines.alfworld_selection import read_selection, select_candidates
from bfas.rtd.baselines.alfworld_training import K32PaperTrainer, encoded_spans, prepare_training
from bfas.rtd.baselines.paper_data import TeacherRow
from bfas.rtd.baselines.paper_losses import span_ce
from bfas.rtd.persistence import ComputeJournal, digest, file_hash
from test_baseline_run_training import Backend, Tokenizer


def row(package='a', task='task', index=0, target='think\nACTION: look'):
    return TeacherRow(package, task, 'parent', index, 'prompt', target, 'alfworld', False)


def candidate(name, task='task', turns=1):
    return dict(candidate_id=name, package_id=name, task_id=task, attempt_index=0,
        verified=True, cost=dict(tokens=123), rows=[asdict(row(name, task, i)) for i in range(turns)])


def selection_args():
    candidates = [candidate('a'), candidate('b', turns=2), candidate('c'),
                  candidate('d', 'two'), candidate('e', 'two'), candidate('f', 'single')]
    tasks = ['task', 'two', 'single', 'empty']
    cost = dict(tokens=999, per_task={t: dict(tokens=100) for t in tasks})
    return candidates, tasks, dict(student=dict(base='frozen', tokenizer='frozen')), cost


def scorer(r):
    # a wins total NLL, c wins mean of turn means, b wins true token mean.
    return {('a', 0): (2., 1), ('b', 0): (9., 1), ('b', 1): (0., 99),
            ('c', 0): (1., 10), ('d', 0): (4., 2), ('e', 0): (4., 2),
            ('f', 0): (7., 1)}[(r.package_id, r.index)]


def test_selection_argmin_marks_shortfalls_and_is_byte_deterministic(tmp_path):
    cs, tasks, identity, cost = selection_args()
    a = select_candidates(cs, tasks, scorer, tmp_path/'a', identity, cost)
    b = select_candidates(list(reversed(cs)), list(reversed(tasks)), scorer, tmp_path/'b', identity, cost)
    assert a == b
    assert (tmp_path/'a/selection.json').read_bytes() == (tmp_path/'b/selection.json').read_bytes()
    groups = {t['task_id']: t for t in a['tasks']}
    assert groups['task']['chosen_candidate_id'] == 'b'
    assert groups['two']['chosen_candidate_id'] == 'd'
    assert groups['two']['candidate_count'] == 2 and groups['two']['shortfall'] == 1
    assert 'single candidate' not in groups['two']['reason']
    assert groups['single']['reason'] == 'single candidate, no choice'
    assert groups['empty']['chosen_candidate_id'] is None
    for t in a['tasks']:
        if t['candidates']:
            assert t['chosen_candidate_id'] == min(t['candidates'], key=lambda c: (c['mean_nll'], c['candidate_id']))['candidate_id']
    rows, artifact = read_selection(tmp_path/'a/selection.json', identity['student'])
    assert {r.package_id for r in rows} == {'b', 'd', 'f'}
    assert artifact['teacher_data_cost']['tokens'] == 999  # never selected-cost-only


def test_selection_resume_scores_only_unfinished_candidates_and_rejects_drift(tmp_path):
    cs, tasks, identity, cost = selection_args()
    calls = []
    def crash(r):
        calls.append(r.package_id)
        if r.package_id == 'b':
            raise InterruptedError
        return scorer(r)
    with pytest.raises(InterruptedError):
        select_candidates(cs, tasks, crash, tmp_path/'run', identity, cost)
    calls.clear()
    def resumed(r):
        calls.append(r.package_id)
        return scorer(r)
    select_candidates(cs, tasks, resumed, tmp_path/'run', identity, cost, resume=True)
    assert 'a' not in calls and calls.count('b') == 2
    select_candidates(cs, tasks, lambda r: pytest.fail('completed selection rescored'),
                      tmp_path/'run', identity, cost, resume=True)
    with pytest.raises(ValueError, match='identity differs'):
        select_candidates(cs, tasks, scorer, tmp_path/'run', dict(student='changed'), cost, resume=True)
    with pytest.raises(FileExistsError):
        select_candidates(cs, tasks, scorer, tmp_path/'run', identity, cost)
    with pytest.raises(ValueError, match='identity differs'):
        read_selection(tmp_path/'run/selection.json', dict(base='other'))


def test_selection_loader_rejects_resigned_non_argmin_and_changed_rows(tmp_path):
    cs, tasks, identity, cost = selection_args()
    artifact = select_candidates(cs, tasks, scorer, tmp_path/'run', identity, cost)
    task = next(t for t in artifact['tasks'] if t['task_id'] == 'task')
    task['chosen_candidate_id'] = 'a'
    artifact['artifact_hash'] = digest({k: v for k, v in artifact.items() if k != 'artifact_hash'})
    write_json(tmp_path/'bad.json', artifact)
    with pytest.raises(ValueError, match='argmin'):
        read_selection(tmp_path/'bad.json', identity['student'])
    task['chosen_candidate_id'] = 'b'
    artifact['training_rows'][0]['target'] = 'replaced'
    artifact['training_rows_hash'] = digest(artifact['training_rows'])
    artifact['artifact_hash'] = digest({k: v for k, v in artifact.items() if k != 'artifact_hash'})
    write_json(tmp_path/'bad.json', artifact)
    with pytest.raises(ValueError, match='scored candidate'):
        read_selection(tmp_path/'bad.json', identity['student'])


@pytest.mark.parametrize('bad', [(float('nan'), 2), (-1, 2), (1, 0), (1, 1.5)])
def test_selection_rejects_invalid_scores(tmp_path, bad):
    cs, tasks, identity, cost = selection_args()
    with pytest.raises(ValueError, match='NLL'):
        select_candidates(cs, tasks, lambda r: bad, tmp_path/'out', identity, cost)


def test_curriculum_changes_orders_preserves_exposure_and_is_seeded():
    rows = [row('long', index=i) for i in range(5)] + [row('short')] + [row('medium', index=i) for i in range(3)]
    costs = [i+2 for i in range(len(rows))]
    cfg = dict(exposure_passes=[3, 10], supervised_tokens_per_update=17)
    sad = exposure_schedule(rows, costs, cfg, 0, 'sad')
    assert sad == exposure_schedule(rows, costs, cfg, 0, 'sad')
    assert sad['pass_orders'] != exposure_schedule(rows, costs, cfg, 1, 'sad')['pass_orders']
    assert sad['pass_orders'] != exposure_schedule(rows, costs, cfg, 0, 'ce')['pass_orders']
    d = sad['curriculum']['row_difficulty']
    for order in sad['pass_orders']:
        assert sorted(order) == list(range(len(rows)))
        assert [d[i] for i in order] == sorted(d)
    flat = [i for batch in sad['batches'] for i in batch['indices']]
    assert Counter(flat) == {i: 10 for i in range(len(rows))}
    endpoints = {b['endpoint']: b['cumulative_tokens'] for b in sad['batches'] if b['endpoint']}
    assert endpoints == {3: 3*sum(costs), 10: 10*sum(costs)}


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def ledger_bank(tmp_path, method):
    bank = tmp_path/method
    collection = Path(str(bank)+'.collection')
    collection.mkdir(parents=True)
    write_json(collection/'identity.json', dict(method={'ce': 'plain', 'sad': 'plain', 'kang': 'kang-ftp'}.get(method, method),
                                               prices=['0.20', '1.20', '0.01']))
    calls = [dict(id=0, task_id='t', attempt_index=0, status='reported', phase='trajectory',
                  usage=dict(prompt_tokens=100, cached_tokens=20, completion_tokens=10)),
             dict(id=1, task_id='t', attempt_index=1, status='reported', phase='trajectory',
                  usage=dict(prompt_tokens=200, cached_tokens=0, completion_tokens=30))]
    if method == 'kang':
        calls.append(dict(id=2, task_id='t', attempt_index=0, status='reported', phase='cot',
                          usage=dict(prompt_tokens=50, cached_tokens=0, completion_tokens=5)))
    reservations = [dict(c, status='reserved', usage=dict(prompt_tokens=999, cached_tokens=0, completion_tokens=999)) for c in calls]
    (collection/'usage.jsonl').write_text('\n'.join(json.dumps(r) for r in reservations+calls)+'\n')
    episodes = []
    for a in (0, 1):
        usage = {k: sum(c['usage'][k] for c in calls if c['attempt_index'] == a)
                 for k in ('prompt_tokens', 'cached_tokens', 'completion_tokens')}
        episodes.append(dict(task_id='t', attempt_index=a, usage=usage,
                             tokens_spent=usage['completion_tokens'], verified=a == 0))
    (collection/'teacher_ledger.jsonl').write_text('\n'.join(json.dumps(r) for r in episodes)+'\n')
    write_json(bank/'sealed/audit.json', dict(historical_inventory=dict(source_files=[
        dict(path=str(collection/'teacher_ledger.jsonl'), sha256=file_hash(collection/'teacher_ledger.jsonl'))])))
    write_json(bank/'public/support.json', dict(training_task_ids=['t'], historical_task_ids=['t']))
    return bank, calls


@pytest.mark.parametrize('method', ['smartad', 'sad', 'ce', 'kang'])
def test_cost_manifests_equal_ledger_sums_include_unselected_failed_and_planning(tmp_path, method):
    bank, calls = ledger_bank(tmp_path, method)
    output = tmp_path/'cost-report'
    assert main(['cost', '--method', method, '--bank', str(bank), '--output', str(output)]) == 0
    manifest = json.loads((output/'manifest.json').read_text())
    cost = manifest['teacher_data_cost']
    for k in ('prompt_tokens', 'cached_tokens', 'completion_tokens'):
        assert cost[k] == sum(c['usage'][k] for c in calls)
    assert cost['tokens'] == sum(c['usage']['prompt_tokens']+c['usage']['completion_tokens'] for c in calls)
    assert cost['calls'] == len(calls)
    assert cost['per_task']['t']['tokens'] == cost['tokens']
    assert cost['phase_costs']['cot']['tokens'] == (55 if method == 'kang' else 0)
    assert cost['per_attempt']['t']['1']['tokens'] == 230  # failed attempt still costs money
    with pytest.raises(FileExistsError):
        main(['cost', '--method', method, '--bank', str(bank), '--output', str(output)])


def test_cost_rejects_unsettled_and_changed_collection(tmp_path):
    bank, calls = ledger_bank(tmp_path, 'sad')
    usage = Path(str(bank)+'.collection')/'usage.jsonl'
    with usage.open('a') as f:
        f.write(json.dumps(dict(calls[0], status='reserved'))+'\n')
    with pytest.raises(ValueError, match='actual cost unavailable'):
        collection_cost(bank, method='sad')


def test_annotate_existing_ce_run_in_new_directory_without_changing_original(tmp_path):
    bank, _ = ledger_bank(tmp_path, 'ce')
    write_json(bank/'sealed/manifest.json', dict(version='fixture'))
    receipt = dict(status='trained', seed=1, bank=dict(sealed_manifest_sha256=file_hash(bank/'sealed/manifest.json')))
    source = tmp_path/'existing-run/manifest.json'
    write_json(source, receipt)
    before = source.read_bytes()
    main(['cost', '--method', 'ce', '--bank', str(bank), '--run-manifest', str(source),
          '--output', str(tmp_path/'annotated-run')])
    result = json.loads((tmp_path/'annotated-run/manifest.json').read_text())
    assert source.read_bytes() == before
    assert result['status'] == 'trained' and result['seed'] == 1
    assert result['teacher_data_cost']['tokens'] == 340
    assert result['cost_annotation']['source_run_manifest_sha256'] == file_hash(source)
    write_json(source, dict(receipt, bank=dict(sealed_manifest_sha256='wrong')))
    with pytest.raises(ValueError, match='accounting bank differ'):
        main(['cost', '--method', 'ce', '--bank', str(bank), '--run-manifest', str(source),
              '--output', str(tmp_path/'bad-annotation')])


@pytest.fixture
def pi1():
    try:
        return use_pi1_root(os.environ.get('BFAS_PI1_ROOT'))
    except RuntimeError:
        pytest.skip('pi1 is not merged; set BFAS_PI1_ROOT to its local worktree for integration tests')


class BoundaryTokenizer(Tokenizer):
    all_special_tokens = ['<turn|>']
    all_special_ids = [254]
    unk_token_id = 253
    def encode(self, text, **kwargs):
        return [254] if text == '<turn|>' else super().encode(text, **kwargs)
    def convert_tokens_to_ids(self, text):
        return 254 if text == '<turn|>' else self.unk_token_id


def config(pi1):
    # Read the actual shared recipe from its owning checkout.
    root = Path(pi1.__file__).resolve().parents[4]
    return dict(pi1.load_config(root/'configs/rtd/pi1_alfworld_k32.yaml'),
                training_seed=0, training_device='cpu', supervised_tokens_per_update=10000)


@pytest.mark.parametrize('method', ['smartad', 'sad'])
def test_actual_training_uses_shared_encoder_loss_optimizer_schedule_and_cost(tmp_path, pi1, method, monkeypatch):
    cfg = config(pi1)
    b = Backend()
    b.tokenizer = BoundaryTokenizer()
    rows = [row('long', index=i, target='think\nACTION: look') for i in range(2)] + [row('short', target='ACTION: go')]
    costs = [sum(k != 'observation' for k in encoded_spans(b.tokenizer, r, 32768)[1]) for r in rows]
    bank, _ = ledger_bank(tmp_path, method)
    cost = collection_cost(bank, method=method)
    identity = dict(bank=dict(path=str(bank)), teacher_data_cost=cost)
    manifest, plan = prepare_training(tmp_path/'run', rows, cfg, 0, method, identity, costs)
    t = K32PaperTrainer(b, rows, cfg, method, tmp_path/'run',
        ComputeJournal(tmp_path/'run/compute.jsonl', cuda=False), manifest=manifest, plan=plan)
    monkeypatch.setattr(t, 'base_nll', lambda r: pytest.fail('training must not reselect'))
    expected = b.model.lora_logits.detach().clone().requires_grad_()
    hp = t.hp
    opt = torch.optim.AdamW([expected], lr=hp['learning_rate'], betas=tuple(hp['adam_betas']),
        eps=hp['adam_epsilon'], weight_decay=hp['weight_decay'], foreach=False, fused=False)
    # Independent full tensor autograd oracle for every scheduled optimizer update.
    for batch in plan['batches']:
        loss = 0
        for i in batch['indices']:
            encoded, kinds = encoded_spans(b.tokenizer, rows[i], 32768)
            loss += span_ce(expected.log_softmax(0)[list(encoded['target_ids'])], kinds, method)*costs[i]/batch['supervised_tokens']
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([expected], hp['gradient_clip'])
        opt.step()
    result = t.train()
    assert torch.allclose(b.model.lora_logits, expected, rtol=1e-5, atol=1e-9)
    assert result['supervised_tokens'] == 10*sum(costs)
    assert result['endpoints'] == [3, 10] and b.samples == 0
    for endpoint in (3, 10):
        m = json.loads((tmp_path/f'run/pass-{endpoint}/manifest.json').read_text())
        assert m['teacher_data_cost'] == cost
        assert m['supervised_token_count'] == endpoint*sum(costs)
        assert m['hyperparameters']['optimizer'] == 'AdamW'
        assert m['curriculum'] == plan['curriculum']
    with pytest.raises(FileExistsError):
        prepare_training(tmp_path/'run', rows, cfg, 0, method, identity, costs)


def test_encoder_import_tracks_owner_boundary_fix(pi1, monkeypatch):
    tok, r = BoundaryTokenizer(), row()
    original = pi1.encode_teacher_turn
    calls = []
    def wrapped(*args):
        calls.append(args)
        result = original(*args)
        return result
    monkeypatch.setattr(pi1, 'encode_teacher_turn', wrapped)
    encoded, kinds = encoded_spans(tok, r, 32768)
    assert len(calls) == 1 and encoded == original(tok, r, 32768)
    assert len(kinds) == len(encoded['target_ids'])
    assert encoded['labels'][:len(encoded['prompt_ids'])] == (-100,)*len(encoded['prompt_ids'])
    if encoded['target_ids'][-1] == 254:
        assert kinds[-1] == 'action'  # classified, never inserted locally


def test_shared_gpu_guard_checks_pci_order_and_uuid_without_gpu(pi1, monkeypatch):
    from types import SimpleNamespace
    from tools import alf_pi1_train as guard
    uuid = 'GPU-12345678-abcd-1234-abcd-123456789abc'
    calls = []
    def fake_inventory(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout=f'{uuid}, 00000000:61:00.0\n')
    monkeypatch.setattr(guard.subprocess, 'run', fake_inventory)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    selected = guard.select_device(uuid)
    assert selected['pci_bus_id'] == '00000000:61:00.0'
    assert os.environ['CUDA_DEVICE_ORDER'] == 'PCI_BUS_ID'
    assert calls[0][1] == '--id='+uuid
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')  # never expose a GPU in tests
    monkeypatch.setattr(torch.cuda, 'is_initialized', lambda: False)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 1)
    monkeypatch.setattr(torch.cuda, 'get_device_properties', lambda _: SimpleNamespace(uuid='wrong'))
    with pytest.raises(ValueError, match='UUID differs'):
        guard.verify_cuda_device(selected, config(pi1))


@pytest.mark.parametrize('method,bank,expected', [
    ('sad', '/home/xueqi/hq/projects/tc-alignment/data/rtd/v1_alfworld_k32_d0', 681640),
    ('ce', '/home/xueqi/hq/projects/tc-alignment/data/rtd/v1_alfworld_k32_d0', 681640),
    ('kang', str(ROOT/'artifacts/alfworld_k32_kang_ftp'), 794301),
])
def test_available_real_bank_cost_equals_reported_usage(method, bank, expected):
    if not Path(bank).is_dir():
        pytest.skip('local research bank not available')
    result = collection_cost(bank, method=method)
    latest = {}
    for line in Path(bank+'.collection/usage.jsonl').read_text().splitlines():
        call = json.loads(line)
        latest[call['id']] = call
    assert result['tokens'] == expected == sum(
        c['usage']['prompt_tokens']+c['usage']['completion_tokens'] for c in latest.values())


def test_cli_rejects_existing_or_protected_output_before_gpu(tmp_path):
    out = tmp_path/'exists'
    out.mkdir()
    with pytest.raises(FileExistsError):
        main(['select', '--output', str(out), '--config', 'missing', '--model-path', 'missing', '--gpu-uuid', 'invalid'])
    with pytest.raises(ValueError, match='outside data'):
        safe_output(ROOT/'artifacts/new-output', [])
    with pytest.raises(ValueError, match='frozen --bank'):
        main(['select', '--output', str(tmp_path/'new'), '--bank', str(tmp_path/'missing'),
              '--config', 'missing', '--model-path', 'missing', '--gpu-uuid', 'invalid'])
