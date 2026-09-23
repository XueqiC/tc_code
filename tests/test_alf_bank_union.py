"""Offline union identities: real sealing/auditing/loading, synthetic episodes."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]

from bfas.rtd.baselines.pi1 import load_bank
from bfas.rtd.benchmarks.alfworld_state import canonical_hash, replay_commands
from bfas.rtd.benchmarks.alfworld_support import (
    _signed, audit_verified_bank, prompt_messages, seal_verified_bank, teacher_payload_fields, verify_package,
)
from tools import alf_bank_union as union
from tools.alf_bank_subset import read_json, usable_packages
from test_alf_repair_pipeline import Renderer, cfg_for, offline_cpu
from test_alfworld_teacher_pool import source as source_fixture
from test_rtd_alfworld_state import FakeStepper, make_payload


def make_bank(directory, support, attempt, stepper=FakeStepper):
    """Seal fresh fixture evidence without using the union's identity rewriter."""
    records, payloads, resets = [], {}, {}
    for tid, item in support['tasks'].items():
        request = item['request']
        payload = make_payload(request)
        row = payload['historical_response']
        row['attempt_index'] = attempt
        row['demo']['teacher_commands'] = payload['commands']
        replay = replay_commands(request, payload['commands'], stepper(request), Renderer())
        for turn, state in zip(row['demo']['turns'], replay.states):
            turn['target'] = f"THOUGHT: solve it.\nACTION: {turn['target']}"
            turn['context'] = prompt_messages(request, json.loads(state.history_json), react=False)
        payload['provenance']['attempt_index'] = attempt
        q = union.bank.query_id(payload['provenance']['ledger_sha256'], row, payload['provenance']['line'])
        payload.update(query_id=q, raw_ledger_line=json.dumps(row), **teacher_payload_fields(row))
        payload = verify_package(payload, request, stepper(request), Renderer())
        assert payload['status'] == 'usable'
        records.append(union.bank.public_record(q, request))
        payloads[q] = payload
        resets[tid] = payload['verification']['states'][0]
    archive = union.bank.Archive(records, payloads, {}, [], dict(source_files=[]))
    seal_verified_bank(directory, archive, support, payloads, resets)
    assert audit_verified_bank(directory)['passed']
    return directory


def with_environment(support, revision):
    support = deepcopy(support)
    env = support['environment']
    env.pop('environment_hash')
    env.update(source_files={name: canonical_hash([name, revision]) for name in union.COLLECTOR_FILES},
               historical_environment_hash=canonical_hash(['historical', revision]))
    env['environment_hash'] = canonical_hash(env)
    for item in support['tasks'].values():
        item['request']['environment_hash'] = env['environment_hash']
        item['request_hash'] = canonical_hash(item['request'])
    return _signed(support)


@pytest.fixture
def banks(tmp_path):
    source = source_fixture.__wrapped__(tmp_path, SimpleNamespace(param=2))
    support = read_json(source / 'public/support.json')
    return tuple(make_bank(tmp_path / side, with_environment(support, index), index)
                 for index, side in enumerate(('left', 'right')))


@pytest.mark.parametrize('mixing', ['all', 'tokens-1:1'])
def test_code_identity_only_union_preserves_rows_and_records_derivation(banks, tmp_path, mixing):
    left, right = banks
    before = [{str(p.relative_to(b)): p.read_bytes() for p in b.rglob('*.json')} for b in banks]
    original_rows = [row for b in banks for row in load_bank(b, cfg_for(b), Renderer())[0]]
    out = tmp_path / 'union'
    report = union.materialize_union(left, right, out, renderer=Renderer(), mixing=mixing)
    assert report['audit']['passed']
    rows, identity = load_bank(out, cfg_for(out), Renderer())
    assert identity['demonstrations'] == 4 and len(rows) == 8
    assert rows == sorted(original_rows, key=lambda r: (r.package_id, r.index))
    support = read_json(out / 'public/support.json')
    assert support == read_json(left / 'public/support.json')
    derivation = read_json(out / 'sealed/manifest.json')['derivation']
    compatibility = derivation['compatibility']
    assert compatibility['statement'] == 'code-identity-only difference'
    assert compatibility['source_environment_hashes'] == [read_json(b / 'public/support.json')[
        'environment']['environment_hash'] for b in banks]
    assert compatibility['source_support_manifest_hashes'] == [read_json(b / 'public/support.json')[
        'manifest_hash'] for b in banks]
    assert {'task.request.world_files.game.tw-pddl', 'task.request.world_files.traj_data.json',
            'task.request.world_files.initial_state.pddl', 'task.request.world_hash',
            'task.request.goal', 'task.request.max_episode_steps', 'reset_state.prompt',
            'reset_state.history_json[0].content', 'reset_state.history_json[0].admissible'
            } <= set(compatibility['compared_fields'])
    assert derivation['identity_inheritance']['side'] == 'left'
    assert derivation['identity_inheritance']['support_manifest_hash'] == support['manifest_hash']
    assert derivation['new_teacher_calls'] == derivation['new_teacher_tokens'] == 0
    rebinding = derivation['identity_rebinding']['packages']
    assert set(rebinding) == set(usable_packages(right))
    for q, payload in usable_packages(right).items():
        result = read_json(out / 'sealed' / f'{q}.json')
        assert rebinding[q]['source_sha256'] == union.bank.file_hash(right / 'sealed' / f'{q}.json')
        assert rebinding[q]['union_sha256'] == union.bank.file_hash(out / 'sealed' / f'{q}.json')
        for key in ('teacher_react_turns', 'commands', 'historical_response', 'raw_ledger_line',
                    'provenance', 'usage', 'cost'):
            assert result[key] == payload[key]
        for old, new in zip(payload['verification']['states'], result['verification']['states']):
            assert {k: v for k, v in old.items() if k != 'task_json'} == {
                k: v for k, v in new.items() if k != 'task_json'}
    for q in usable_packages(left):
        assert (out / 'sealed' / f'{q}.json').read_bytes() == (left / 'sealed' / f'{q}.json').read_bytes()
    assert before == [{str(p.relative_to(b)): p.read_bytes() for p in b.rglob('*.json')} for b in banks]


@pytest.mark.parametrize('field', [
    'admissible', 'content', 'prompt', 'goal', 'max_episode_steps',
    'game.tw-pddl', 'traj_data.json', 'initial_state.pddl',
])
def test_incompatible_reset_names_task_and_field(banks, tmp_path, field):
    left, right = banks
    support = read_json(right / 'public/support.json')
    resets = read_json(right / 'public/reset_states.json')
    tid = sorted(support['tasks'])[0]
    request = support['tasks'][tid]['request']
    if field in union.WORLD_NAMES:
        request['world_files'][field] = 'f' * 64
        request['world_hash'] = canonical_hash(request['world_files'])
    elif field in ('goal', 'max_episode_steps'):
        request[field] = 'a different goal' if field == 'goal' else 39
    elif field == 'prompt':
        resets[tid]['prompt'] += ' changed'
    else:
        history = json.loads(resets[tid]['history_json'])
        history[0][field] += ['look'] if field == 'admissible' else ' changed'
        resets[tid]['history_json'] = json.dumps(history, sort_keys=True, ensure_ascii=False)
    support['tasks'][tid]['request_hash'] = canonical_hash(request)
    resets[tid]['task_json'] = json.dumps(request, sort_keys=True, ensure_ascii=False)
    union.write_json(right / 'public/support.json', _signed(support))
    union.write_json(right / 'public/reset_states.json', resets)
    out = tmp_path / 'refused'
    with pytest.raises(ValueError) as error:
        union.materialize_union(left, right, out, renderer=Renderer())
    assert tid in str(error.value) and field in str(error.value)
    assert not out.exists()


def test_valid_world_mismatch_refuses(banks, tmp_path):
    left, right = banks
    support = read_json(right / 'public/support.json')
    tid = sorted(support['tasks'])[0]
    request = support['tasks'][tid]['request']
    request['world_files']['initial_state.pddl'] = 'f' * 64
    request['world_hash'] = canonical_hash(request['world_files'])
    support['tasks'][tid]['request_hash'] = canonical_hash(request)
    other = make_bank(tmp_path / 'other-world', _signed(support), 2)
    with pytest.raises(ValueError) as error:
        union.materialize_union(left, other, tmp_path / 'refused', renderer=Renderer())
    assert tid in str(error.value) and 'world_files.initial_state.pddl' in str(error.value)


def test_valid_admissible_action_mismatch_refuses(banks, tmp_path):
    class DifferentActions(FakeStepper):
        def observation(self, index):
            value = super().observation(index)
            return replace(value, admissible=(*value.admissible, 'look')) if index == 0 else value

    left, right = banks
    support = read_json(right / 'public/support.json')
    other = make_bank(tmp_path / 'other-actions', support, 2, stepper=DifferentActions)
    with pytest.raises(ValueError) as error:
        union.materialize_union(left, other, tmp_path / 'refused', renderer=Renderer())
    assert sorted(support['tasks'])[0] in str(error.value) and 'admissible' in str(error.value)


def test_environment_sensitive_renderer_refuses(banks, tmp_path):
    class IdentityRenderer(Renderer):
        def __call__(self, request, history):
            return request['environment_hash'] + super().__call__(request, history)

    with pytest.raises(ValueError, match='rendered_prompt differs'):
        union.materialize_union(*banks, tmp_path / 'refused', renderer=IdentityRenderer())
    assert not (tmp_path / 'refused').exists()


def test_environment_behavior_metadata_is_not_ignored(banks):
    supports = [read_json(b / 'public/support.json') for b in banks]
    resets = [read_json(b / 'public/reset_states.json') for b in banks]
    supports[1]['environment']['tokenizer_files'] = {'tokenizer.json': 'f' * 64}
    with pytest.raises(ValueError, match='support.environment.tokenizer_files'):
        union.compatible_support(supports, resets)
