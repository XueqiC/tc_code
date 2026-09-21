"""K=32 source contracts: synthetic CPU resets, real signing/sealing/auditors."""
from dataclasses import replace
import importlib.abc
import json
import os
from pathlib import Path
import socket
import sys
from types import SimpleNamespace

import pytest

from tools import alf_support_k32_source as source
from tools.alfworld_teacher_pool import collection_support, load_source
from bfas.rtd.benchmarks.alfworld_config import audit_verified_bank
from bfas.rtd.benchmarks.alfworld_state import Observation, parent_fold, parent_hash, validate_full_state
from bfas.rtd.benchmarks.alfworld_support import prompt_messages, validate_support
from bfas.rtd.transport import FullState

ROOT = Path(__file__).resolve().parents[1]


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) if isinstance(value, (dict, list)) else value)


@pytest.fixture(autouse=True)
def cpu_offline(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    def forbidden(*args, **kwargs):
        pytest.fail('source minting must not contact a teacher/network or start a real worker in unit tests')
    monkeypatch.setattr(socket.socket, 'connect', forbidden)
    import torch
    for name in ('init', 'is_available', 'device_count', 'get_device_properties', 'empty_cache'):
        monkeypatch.setattr(torch.cuda, name, forbidden)
    from bfas.rtd.benchmarks import alfworld_support
    monkeypatch.setattr(alfworld_support, 'BoundedEnvBridge', forbidden)
    import appworld_teacher
    monkeypatch.setattr(appworld_teacher, 'generate_reply', forbidden)

    class NoEnvironmentImports(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split('.')[0] in {'alfworld', 'textworld'}:
                pytest.fail('third-party environment imported outside existing worker')
    finder = NoEnvironmentImports()
    sys.meta_path.insert(0, finder)
    yield
    sys.meta_path.remove(finder)


class ResetOnly:
    def __init__(self, request):
        self.request, self.closed = request, False

    def reset(self, request):
        assert request == self.request
        return 0, Observation(0, 'Untruncated reset. ' * 200 + '\nYour task is to: ' + request['goal'],
                              ('look', 'inventory'), request['world_hash'])

    def step(self, *args):
        pytest.fail('source minting must only reset, never execute actions')

    def close(self):
        self.closed = True


def render(request, history):
    return '\n\n'.join(f"{m['role']}: {m['content']}" for m in prompt_messages(request, history))


@pytest.fixture
def campaign(tmp_path):
    root = tmp_path / 'project'
    support = json.loads((ROOT / source.SUPPORT).read_text())
    manifest = json.loads((ROOT / source.MANIFEST).read_text())
    support['source_manifest'] = str(root / source.MANIFEST)
    data = root / 'envs/alfworld/data/json_2.1.1'
    manifest['data_root'] = str(data)
    for split, item in manifest['splits'].items():
        item['split_dir'] = str(data / split)
    put(root / source.SUPPORT, support)
    put(root / source.MANIFEST, manifest)
    for entry in support['support']:
        tid = entry['game_id']
        world = data / 'train' / tid
        put(world / 'game.tw-pddl', dict(grammar={'task': [{'rhs': 'Your task is to: put apple on table.'}]}))
        put(world / 'traj_data.json', dict(task_id=tid, task_type=entry['task_type']))
        put(world / 'initial_state.pddl', 'unique fixture ' + tid)
    for name in ('src/bfas/adapters/alfworld.py', 'src/alfworld_eval.py', 'src/bfas/cc_pairs.py',
                 'tools/behavior_atom/gpu_driver.py', 'src/bfas/rtd/transport.py',
                 'src/bfas/rtd/benchmarks/alfworld_state.py'):
        put(root / name, (ROOT / name).read_text())
    env = root / 'envs/alfworld/.venv'
    put(env / 'bin/python', 'fixture interpreter, never executed')
    for package in ('alfworld', 'textworld'):
        site = env / 'lib/python3.11/site-packages'
        put(site / package / '__init__.py', '# fixture package, never imported')
        put(site / f'{package}-1.dist-info/METADATA', 'Version: 1\n')
        put(site / f'{package}-1.dist-info/RECORD', 'fixture dependency record')
    tokenizer = root / 'tokenizer'
    for name in ('tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja'):
        put(tokenizer / name, 'fixture renderer, no tokenizer instantiated')
    put(root / 'data/rtd/v1_alfworld_c26/sealed/sentinel.json', dict(read_only=True))
    c = SimpleNamespace(root=root, data=data, tokenizer=tokenizer, frozen=support,
                        out=root / source.OUTPUT, workers=[])
    def factory(request):
        worker = ResetOnly(request)
        c.workers.append(worker)
        return worker
    c.factory = factory
    return c


def mint(c, **kwargs):
    options = dict(out=c.out, tokenizer=c.tokenizer, stepper_factory=c.factory, renderer=render)
    options.update(kwargs)
    return source.mint_source(c.root, **options)


def snapshot(directory):
    return {str(p.relative_to(directory)): p.read_bytes() for p in directory.rglob('*') if p.is_file()}


def test_mint_passes_all_unchanged_contracts_and_preserves_inputs(campaign):
    c = campaign
    before = snapshot(c.root)
    result = mint(c)
    support = load_source(c.out)
    assert validate_support(support) == support
    assert audit_verified_bank(c.out)['passed']
    assert result['audit']['attempts'] == result['reset_states'] == 32
    assert result['audit']['usable_packages'] == result['teacher_calls'] == 0
    assert result['gpu_used'] is False
    assert len(c.workers) == 32 and all(w.closed for w in c.workers)
    assert result['support_sha256'] == source.SUPPORT_SHA256
    ids = [r['game_id'] for r in c.frozen['support']]
    assert support['historical_task_ids'] == support['training_task_ids'] == ids
    assert support['collection_source']['frozen_selection'] == c.frozen
    assert support['m'] == support['historical_parent_groups'] == 32
    assert support['fold_parent_counts'] == support['fold_task_counts'] == {'0': 16, '1': 16}
    assert {p.name for p in c.out.iterdir()} == {'public', 'sealed'}
    assert {p.name for p in (c.out / 'public').iterdir()} == {
        'support.json', 'requests.json', 'reset_requests.json', 'reset_states.json'}
    rows = json.loads((c.out / 'public/requests.json').read_text())
    requests = json.loads((c.out / 'public/reset_requests.json').read_text())
    resets = json.loads((c.out / 'public/reset_states.json').read_text())
    assert len(rows) == len(requests) == len(resets) == 32
    assert [requests[r['spec']['query_id']]['task_id'] for r in rows] == ids
    assert set(resets) == set(ids)
    sealed_ids = set()
    for i, row in enumerate(rows, 1):
        q = row['spec']['query_id']
        p = json.loads((c.out / 'sealed' / f'{q}.json').read_text())
        tid = requests[q]['task_id']
        sealed_ids.add(p['provenance']['task_id'])
        assert p['status'] == row['unavailable_reason'] == 'unavailable'
        assert p['unavailable_reason'] == source.UNAVAILABLE
        assert p['commands'] == p['behaviors'] == p['dependencies'] == []
        assert p['cost'] == 0 and p['success'] is False and p['payload_kind'] is None
        assert p['historical_response']['teacher_attempted'] is False
        assert p['historical_response']['synthetic'] is True and p['provenance']['ledger_path'] is None
        assert p['provenance']['line'] == i
        state = FullState(**resets[tid])
        assert validate_full_state(state, expected_world_hash=requests[q]['world_hash']) == state
        assert state.state_hash == row['spec']['state_hash']
        history = json.loads(state.history_json)
        assert len(history) == 1 and len(history[0]['content']) > 2000
        assert history[0]['admissible'] == ['look', 'inventory'] and 'won' not in history[0]
        assert state.prompt == render(requests[q], history)
        assert support['tasks'][tid]['fold'] == parent_fold(parent_hash(tid))
        assert support['tasks'][tid]['excluded'] is False
    assert sealed_ids == set(ids)
    # Exercise the collector's next transformation too, without purchase calls.
    collected = collection_support(support)
    assert validate_support(collected)['historical_task_ids'] == ids
    assert collected['parents'] == support['parents']
    for name, contents in before.items():
        assert (c.root / name).read_bytes() == contents
    other = c.root / 'data/rtd/second-source'
    mint(c, out=other)
    assert snapshot(c.out) == snapshot(other)


@pytest.mark.parametrize('kind', ['directory', 'file', 'dangling_symlink', 'existing_source'])
def test_existing_output_is_never_touched_even_with_invalid_inputs(campaign, kind):
    c = campaign
    c.out.parent.mkdir(parents=True, exist_ok=True)
    if kind == 'directory':
        put(c.out / 'sentinel', 'keep me')
    elif kind == 'file':
        c.out.write_text('keep me')
    elif kind == 'dangling_symlink':
        c.out.symlink_to(c.out.parent / 'missing-target')
    else:
        c.out = c.root / 'data/rtd/v1_alfworld_c26'
    before = snapshot(c.root)
    with pytest.raises(FileExistsError):
        mint(c, support_path='missing-input.json')
    assert snapshot(c.root) == before and c.workers == []
    if kind == 'dangling_symlink':
        assert c.out.is_symlink() and not c.out.exists()


@pytest.mark.parametrize('change,match', [
    ('duplicate', '32 unique'), ('reorder', 'SHA256/order'), ('hash', 'SHA256/order'),
    ('quota', 'quota'), ('train_hash', 'train inventory'),
    ('manifest_hash', 'inventory/hash'), ('loader_error', 'inventory/hash'),
    ('manifest_count', 'inventory/hash'), ('data_root', 'data root'),
    ('world_type', 'task type'), ('missing_world', ''),
])
def test_invalid_frozen_inputs_fail_before_workers_or_output(campaign, change, match):
    c = campaign
    path = c.root / source.SUPPORT
    frozen = json.loads(path.read_text())
    manifest_path = c.root / source.MANIFEST
    manifest = json.loads(manifest_path.read_text())
    world = c.data / 'train' / frozen['support'][0]['game_id']
    if change == 'duplicate':
        frozen['support'][1] = frozen['support'][0]
    elif change == 'reorder':
        frozen['support'].reverse()
    elif change == 'hash':
        frozen['support_sha256'] = '0' * 64
    elif change == 'quota':
        frozen['quota']['pick_and_place_simple'] += 1
    elif change == 'train_hash':
        frozen['source_manifest_train_sha256'] = '0' * 64
    elif change == 'manifest_hash':
        manifest['splits']['valid_seen']['game_ids'].reverse()
    elif change == 'loader_error':
        manifest['splits']['train']['loader_error'] = 'failed export'
    elif change == 'manifest_count':
        manifest['splits']['valid_unseen']['loader_games'] -= 1
    elif change == 'data_root':
        manifest['data_root'] = str(c.root / 'wrong-data')
    elif change == 'world_type':
        put(world / 'traj_data.json', dict(task_type='wrong'))
    elif change == 'missing_world':
        (world / 'initial_state.pddl').unlink()
    put(path, frozen)
    put(manifest_path, manifest)
    with pytest.raises((ValueError, FileNotFoundError), match=match):
        mint(c)
    assert not c.out.exists() and c.workers == []


@pytest.mark.parametrize('problem', ['reset_exception', 'wrong_world', 'wrong_index', 'done', 'won'])
def test_bad_reset_closes_worker_and_never_seals(campaign, problem):
    c = campaign
    class BadReset(ResetOnly):
        def reset(self, request):
            if problem == 'reset_exception':
                raise RuntimeError('worker failed')
            cursor, obs = super().reset(request)
            fields = dict(wrong_world=dict(world_hash='0' * 64), wrong_index=dict(index=1),
                          done=dict(done=True), won=dict(won=True))
            return cursor, replace(obs, **fields[problem])
    def factory(request):
        worker = BadReset(request)
        c.workers.append(worker)
        return worker
    with pytest.raises((RuntimeError, ValueError)):
        mint(c, stepper_factory=factory)
    assert len(c.workers) == 1 and c.workers[0].closed and not c.out.exists()


@pytest.mark.parametrize('changed', ['support', 'manifest', 'world', 'environment'])
def test_input_drift_during_capture_refuses_publication(campaign, changed):
    c = campaign
    first = c.frozen['support'][0]['game_id']
    paths = dict(support=c.root / source.SUPPORT, manifest=c.root / source.MANIFEST,
                 world=c.data / 'train' / first / 'initial_state.pddl',
                 environment=c.tokenizer / 'chat_template.jinja')
    def renderer(request, history):
        if request['task_id'] == first:
            path = paths[changed]
            path.write_bytes(path.read_bytes() + b'\n')
        return render(request, history)
    with pytest.raises(ValueError, match='changed during reset capture'):
        mint(c, renderer=renderer)
    assert len(c.workers) == 32 and all(w.closed for w in c.workers) and not c.out.exists()


def test_concurrent_creator_wins_and_is_not_overwritten(campaign):
    c = campaign
    def renderer(request, history):
        if not c.out.exists():
            put(c.out / 'sentinel', 'other creator')
        return render(request, history)
    with pytest.raises(FileExistsError):
        mint(c, renderer=renderer)
    assert snapshot(c.out) == {'sentinel': b'other creator'}
    assert all(w.closed for w in c.workers)


def test_auditor_rejects_tampered_reset(campaign):
    c = campaign
    mint(c)
    path = c.out / 'public/reset_states.json'
    path.write_bytes(path.read_bytes() + b'\n')
    with pytest.raises(ValueError, match='artifact integrity'):
        load_source(c.out)


def test_cli_uses_cpu_defaults_and_real_contracts(campaign, monkeypatch, capsys):
    c = campaign
    monkeypatch.setattr(source.adapter, 'ROOT', c.root)
    monkeypatch.setattr(source.adapter, 'DATA', c.data)
    def stepper(**kwargs):
        assert kwargs['timeout'] == 30.0 and kwargs['reset_timeout'] == 120.0
        assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
        assert os.environ['HF_HUB_OFFLINE'] == os.environ['TRANSFORMERS_OFFLINE'] == '1'
        class CLIReset(ResetOnly):
            def reset(self, request):
                self.request = request
                assert request['environment_hash'] == kwargs['environment_hash']
                return super().reset(request)
        worker = CLIReset(None)
        c.workers.append(worker)
        return worker
    monkeypatch.setattr(source, 'RealStepper', stepper)
    monkeypatch.setattr(source, 'FrozenRenderer', lambda path: render)
    assert source.main(['--root', str(c.root), '--tokenizer', str(c.tokenizer)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['output'] == str(c.out) and result['audit']['passed']
    assert len(c.workers) == 32 and all(w.closed for w in c.workers)


def test_reject_project_cwd_and_read_only_subtree(campaign, monkeypatch):
    c = campaign
    monkeypatch.chdir(c.root)
    with pytest.raises(ValueError, match='from /tmp'):
        mint(c)
    monkeypatch.chdir(c.root.parent)
    with pytest.raises(ValueError, match='read-only'):
        mint(c, out=c.root / 'data/rtd/v1_alfworld_c26/new-child')
    with pytest.raises(ValueError, match='finite and positive'):
        mint(c, timeout=float('nan'))
    assert c.workers == []
