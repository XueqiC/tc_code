"""CPU-only REPAIR material oracles: no sockets, teacher service, or ALFWorld."""
from copy import deepcopy
from dataclasses import asdict
from decimal import Decimal
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]

from bfas.rtd.baselines.pi1 import encode_teacher_turn, load_bank, load_config
from bfas.rtd.benchmarks.alfworld_evaluation import Generation
from bfas.rtd.benchmarks.alfworld_state import Observation, canonical_hash, reconstruct_package, parent_hash
from bfas.rtd.benchmarks.alfworld_support import audit_verified_bank, FrozenRenderer
from tools import alf_bank_union as union, alf_pi1_support_rollouts as rollouts, alf_repair_collect as repair
from tools import alfworld_teacher_pool as pool
from tools.alf_bank_subset import read_json, register_config, sweep_statistics, usable_packages
from test_alf_pi1_train import Tokenizer
from test_alfworld_teacher_pool import source as source_fixture, Teacher, factory
from test_rtd_alfworld_state import FakeStepper


@pytest.fixture(autouse=True)
def offline_cpu(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    monkeypatch.setenv('BFAS_TEACHER_MIN_INTERVAL_S', '0')
    monkeypatch.delenv('BFAS_TEACHER', raising=False)
    for key in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_DATASETS_OFFLINE'):
        monkeypatch.setenv(key, '1')
    def forbidden(*args, **kwargs):
        pytest.fail('material test attempted network, real environment, or CUDA access')
    monkeypatch.setattr(socket.socket, 'connect', forbidden)
    monkeypatch.setattr(pool.appworld_teacher, 'generate_reply', forbidden)
    monkeypatch.setattr(pool.RealStepper, '__init__', forbidden)
    import torch
    for name in ('is_available', 'device_count', 'get_device_properties', 'init', 'empty_cache'):
        monkeypatch.setattr(torch.cuda, name, forbidden)


@pytest.fixture
def source(tmp_path):
    return source_fixture.__wrapped__(tmp_path, SimpleNamespace(param=2))


@pytest.fixture
def d0(source, tmp_path):
    out = tmp_path / 'd0'
    pool.collect(source, out, stepper_factory=FakeStepper, adapter_factory=factory(Teacher()))
    return out


class Renderer:
    adapter = SimpleNamespace(_tokenizer=Tokenizer())

    def __call__(self, request, history):
        return pool.render(request, history)


class Student:
    renderer = Renderer()

    def __init__(self, *, all_fail=False, late_repeat=False):
        self.all_fail, self.late_repeat = all_fail, late_repeat
        self.calls, self.closed = [], False

    def close(self):
        self.closed = True

    def generate(self, prompt, *, temperature, max_new_tokens):
        assert temperature == 0.0 and max_new_tokens == 256
        self.calls.append(prompt)
        if '(start)' in prompt:
            command = 'go to table 1'
        elif self.late_repeat and '> look\n' in prompt:
            command = 'go to table 1'
        elif not self.all_fail and 'task-fixture-success' in prompt:
            command = 'put apple 1 on table 1'
        else:
            return Generation('unparseable response', 10, False)
        return Generation('THOUGHT: act.\nACTION: ' + command, 10, False)


class StudentStepper(FakeStepper):
    def __init__(self, request, *, no_nothing=False):
        super().__init__(request)
        self.no_nothing = no_nothing
        self.location = False

    def observation(self, i):
        if i == 0:
            obs = super().observation(0)
            # Same frozen reset; identify successful task through a backend wrapper instead.
            return obs
        raise AssertionError('use step')

    def step(self, cursor, command):
        self.calls.append(command)
        if command == 'go to table 1':
            self.location = True
            text, done = 'Full observation ' + 'y' * 500, False
        elif command == 'put apple 1 on table 1' and self.location:
            text, done = 'Task complete.', True
        else:
            text, done = 'waiting' if self.no_nothing else 'Nothing happens. Invalid command.', False
        obs = Observation(cursor + 1, text, () if done else ('put apple 1 on table 1',),
                          self.request['world_hash'], done, done)
        self.observations.append(asdict(obs))
        return cursor + 1, obs


class TwoTaskStudent(Student):
    def __init__(self):
        super().__init__()
        self.episode = 0

    def generate(self, prompt, **kwargs):
        if '(start)' in prompt:
            self.episode += 1
        if self.episode == 2 and '(start)' not in prompt:
            assert kwargs == dict(temperature=0.0, max_new_tokens=256)
            return Generation('THOUGHT: finish.\nACTION: put apple 1 on table 1', 10, False)
        return super().generate(prompt, **kwargs)


def episode(commands, observations, success=False):
    return dict(status='completed', success=success, steps=len(commands), commands=commands, observations=observations)


@pytest.mark.parametrize('record,expected', [
    (episode(['a', 'b', 'a'], ['ok']*3), dict(k=2, rule='repeated_command', triggers=['repeated_command'])),
    (episode(['a', 'b'], ['ok', 'nOtHiNg HaPpEnS. more']), dict(k=1, rule='nothing_happens', triggers=['nothing_happens'])),
    (episode(['a'], ['Nothing happens.']), dict(k=0, rule='nothing_happens', triggers=['nothing_happens'])),
    (episode(['a', 'b', 'c', 'd', 'e'], ['ok']*5), dict(k=2, rule='midpoint', triggers=[])),
    (episode(['a', 'a'], ['ok', 'Nothing happens.']), dict(k=1, rule='repeated_command', triggers=['repeated_command', 'nothing_happens'])),
    (episode(['a'], ['prefix Nothing happens.']), dict(k=0, rule='midpoint', triggers=[])),
    (episode(['a', 'a'], ['Nothing happens.']*2, True), None),
])
def test_takeover_rule(record, expected):
    assert repair.takeover(record) == expected


@pytest.mark.parametrize('success,status', [(None, 'completed'), (False, 'failed_with_reason')])
def test_infrastructure_failure_cannot_become_a_repair(success, status):
    value = episode(['a'], ['ok'])
    value.update(success=success, status=status)
    with pytest.raises(ValueError):
        repair.takeover(value)


@pytest.mark.parametrize('commands,share,filtered', [
    ([], 0, False),
    (['go to cabinet 12', 'open drawer 3', 'close drawer 2'], 1, True),
    (['open cabinet 1', 'take apple 1'], .5, False),
    (['go to cabinet 1', 'close drawer 2', 'look'], 2/3, True),
    (['Open drawer 1', 'go to cabinet 1 extra', 'open drawer 1 extra', 'open desk 1'], 0, False),
])
def test_suffix_sweep_rule(commands, share, filtered):
    result = sweep_statistics(commands)
    assert result['sweep_share'] == share and result['sweep_filtered'] is filtered


def test_union_exact_arithmetic_seed_and_input_order():
    left, right = {f'p{i}': 25 for i in range(8)}, {'q': 100}
    selected = union.mix_packages(left, right, mixing='tokens-1:1', seed=9)
    assert selected == union.mix_packages(dict(reversed(list(left.items()))), right, mixing='tokens-1:1', seed=9)
    assert selected['selected_supervised_tokens'] == [100, 100]
    assert len(selected['included_package_ids'][0]) == 4
    assert selected != union.mix_packages(left, right, mixing='tokens-1:1', seed=10)
    assert union.mix_packages(right, left, mixing='tokens-1:1', seed=9)['selected_supervised_tokens'] == [100, 100]
    assert union.mix_packages(left, right)['selected_supervised_tokens'] == [200, 100]


@pytest.mark.parametrize('cost,accepted', [(94, False), (95, True), (105, True), (106, False)])
def test_union_tolerance_edges(cost, accepted):
    left, right = {'a': cost, 'b': 1000}, {'c': 100}
    if accepted:
        assert union.mix_packages(left, right, mixing='tokens-1:1')['selected_supervised_tokens'] == [cost, 100]
    else:
        with pytest.raises(ValueError, match='no whole-package subset'):
            union.mix_packages(left, right, mixing='tokens-1:1')


def cfg_for(path):
    summary = read_json(path / 'sealed/audit.json')
    return dict(load_config(ROOT / 'configs/rtd/pi1_alfworld_k32_kang.yaml'),
                sealed_manifest_sha256=pool.bank.file_hash(path / 'sealed/manifest.json'),
                support_size=summary['m'], demonstrations=summary['usable_packages'], supervised_turns=summary['command_turns'])


def run_repair(d0, tmp_path, *, student=None, stepper_factory=StudentStepper, teacher=None, **kwargs):
    rollout_dir = tmp_path / 'rollouts'
    backend = student or TwoTaskStudent()
    rollouts.rollout_support(d0, rollout_dir, backend, stepper_factory=stepper_factory, support_size=2)
    teacher = teacher or Teacher()
    collection, output = tmp_path / 'repair.collection', tmp_path / 'repair'
    result = repair.collect_repair(rollout_dir, d0, Path(str(d0) + '.collection'), collection, output,
        stepper_factory=stepper_factory, generate_reply=teacher, config_loader=lambda _: SimpleNamespace(),
        support_size=2, **kwargs)
    return rollout_dir, collection, output, result, teacher


def test_fake_backend_end_to_end_and_load_bank(d0, tmp_path):
    directory, collection, output, result, teacher = run_repair(d0, tmp_path)
    index = read_json(directory / 'index.json')
    records = [read_json(directory / e['file']) for e in index['tasks']]
    assert [r['success'] for r in records] == [False, True]
    assert records[0]['steps'] == 40 and records[0]['horizon_reached']
    assert records[1]['steps'] == 2 and not records[1]['horizon_reached']
    fallback = records[0]['turns'][1]
    assert {k: fallback[k] for k in ('parser_fallback', 'parser_candidate', 'parser_fallback_reason')} == dict(
        parser_fallback=True, parser_candidate='unparseable response', parser_fallback_reason='no_action_marker')
    assert records[1]['turns'][0]['parser_fallback'] is False
    assert 'parser_fallback_reason' not in records[1]['turns'][0]
    assert result['requests'] == result['usable_packages'] == len(teacher.calls) == 1
    payload = next(iter(usable_packages(output).values()))
    assert payload['student_prefix_commands'] == ['go to table 1']
    assert payload['commands'] == ['put apple 1 on table 1']
    assert payload['repair']['k'] == 1 and payload['repair']['remaining_horizon'] == 39
    assert payload['verification']['done'] and payload['verification']['won']
    assert len(payload['verification']['states']) == 3 and len(payload['behaviors']) == 1
    rows, _ = load_bank(output, cfg_for(output), Renderer())
    assert len(rows) == 1 and rows[0].target == teacher.replies[0]
    assert '> go to table 1' in rows[0].prompt
    encoded = encode_teacher_turn(Renderer.adapter._tokenizer, rows[0], 32768)
    assert set(encoded['labels'][:len(encoded['prompt_ids'])]) == {-100}
    assert encoded['labels'][len(encoded['prompt_ids']):] == tuple(rows[0].target.encode()) + (106,)
    support = read_json(output / 'public/support.json')
    assert support['m'] == len(support['historical_task_ids']) == 2
    assert support['training_task_ids'] == [records[0]['task_id']]
    assert (collection / 'usage.jsonl').is_file() and (collection / 'teacher_ledger.jsonl').is_file()
    merged = tmp_path / 'union'
    report = union.materialize_union(d0, output, merged, renderer=Renderer())
    merged_rows, identity = load_bank(merged, cfg_for(merged), Renderer())
    assert len(merged_rows) == 5 and identity['demonstrations'] == 3
    assert report['audit']['passed']
    for source in (d0, output):
        for q in usable_packages(source):
            assert (merged / 'sealed' / f'{q}.json').read_bytes() == (source / 'sealed' / f'{q}.json').read_bytes()
    assert read_json(merged / 'sealed/manifest.json')['derivation']['mixing']['kind'] == 'all'
    with pytest.raises(FileExistsError):
        repair.collect_repair(directory, d0, Path(str(d0) + '.collection'), collection, tmp_path / 'another', support_size=2)
    with pytest.raises(FileExistsError):
        rollouts.rollout_support(d0, directory, TwoTaskStudent(), support_size=2)
    with pytest.raises(FileExistsError):
        union.materialize_union(d0, output, merged, renderer=Renderer())


def test_nonadmissible_student_fallback_is_replayed_and_masked(d0, tmp_path):
    # First fallback has no Nothing-happens observation; repetition fires on the
    # next fallback, so a nonadmissible executed 'look' lies inside the prefix.
    stepper = lambda request: StudentStepper(request, no_nothing=True)
    _, _, output, result, _ = run_repair(d0, tmp_path, student=Student(all_fail=True), stepper_factory=stepper)
    assert result['usable_packages'] == 2
    for payload in usable_packages(output).values():
        assert payload['student_prefix_commands'] == ['go to table 1', 'look']
        assert payload['repair']['k'] == 2
        request = read_json(output / 'public/support.json')['tasks'][payload['provenance']['task_id']]['request']
        reconstructed = reconstruct_package(payload, request, stepper(request), pool.render,
            owned_ids={payload['query_id']}, inner_parent_hashes={parent_hash(request['task_id'])})
        assert len(reconstructed.behaviors) == 1 and reconstructed.won
    assert len(load_bank(output, cfg_for(output), Renderer())[0]) == 2


@pytest.mark.parametrize('usage', [True, False])
def test_failed_teacher_calls_remain_charged(d0, tmp_path, usage):
    def fail(config, messages, **kwargs):
        if usage:
            kwargs['usage_callback'](dict(prompt_tokens=100, completion_tokens=20))
        raise RuntimeError('stub teacher failed after purchase')
    _, collection, output, result, _ = run_repair(d0, tmp_path, teacher=fail)
    assert result['usable_packages'] == 0
    assert result['purchase_cost']['calls'] == 1
    assert result['purchase_cost']['tokens'] > 0
    assert result['purchase_cost']['uncertain_calls'] == (0 if usage else 1)
    rows = pool.read_records(collection / 'teacher_ledger.jsonl')
    assert len(rows) == 1 and not rows[0]['verified'] and rows[0]['tokens_spent'] > 0
    assert audit_verified_bank(output)['usable_packages'] == 0
    assert len(list((collection / 'packages').glob('*.json'))) == 1


def test_budget_blocks_before_purchase(d0, tmp_path):
    _, collection, output, result, teacher = run_repair(d0, tmp_path, max_tokens=1)
    assert teacher.calls == [] and result['purchase_cost']['calls'] == 0
    assert result['usable_packages'] == 0 and audit_verified_bank(output)['passed']
    assert read_json(collection / 'summary.json')['stop_reason'] != 'complete'


def test_add_demo_offset_temperature_and_optional_export_filter(source, tmp_path):
    teacher = Teacher()
    out = tmp_path / 'add-demo'
    pool.collect(source, out, stepper_factory=FakeStepper, adapter_factory=factory(teacher),
                 attempt_start=3, temperature=.7, sweep_filter=True)
    rows = pool.read_records(Path(str(out) + '.collection') / 'teacher_ledger.jsonl')
    assert {r['attempt_index'] for r in rows} == {3}
    assert {r['temperature'] for r in rows} == {.7}
    assert all(call[2]['temperature'] == .7 for call in teacher.calls)
    assert len(usable_packages(out)) == 2
    # Resuming honors the shifted range and does not repurchase successful tasks.
    pool.collect(source, out, stepper_factory=FakeStepper, adapter_factory=factory(teacher),
                 attempt_start=3, temperature=.7, sweep_filter=True)
    assert len(teacher.calls) == 4


def test_balanced_union_acceptance_and_duplicate_all_union(source, d0, tmp_path):
    second = tmp_path / 'second'
    pool.collect(source, second, stepper_factory=FakeStepper, adapter_factory=factory(Teacher()), attempt_start=3)
    out = tmp_path / 'balanced'
    report = union.materialize_union(d0, second, out, mixing='tokens-1:1', seed=23, renderer=Renderer())
    assert report['mixing']['selected_supervised_tokens'][0] == report['mixing']['selected_supervised_tokens'][1]
    assert len(load_bank(out, cfg_for(out), Renderer())[0]) == 8
    duplicate = tmp_path / 'deduplicated'
    union.materialize_union(d0, d0, duplicate, renderer=Renderer())
    assert len(load_bank(duplicate, cfg_for(duplicate), Renderer())[0]) == 4
    with pytest.raises(ValueError, match='overlapping'):
        union.materialize_union(d0, d0, tmp_path / 'invalid', mixing='tokens-1:1', renderer=Renderer())


def test_real_tokenizer_registration_and_cpu_preflight(d0, tmp_path):
    from tools.bfcl_hub_merge_export import _snapshot_for_model
    model = _snapshot_for_model('google/gemma-4-12B-it')
    _, _, output, _, _ = run_repair(d0, tmp_path)
    merged = tmp_path / 'union'
    union.materialize_union(d0, output, merged, model_path=model)
    for path in (output, merged):
        config = tmp_path / (path.name + '.yaml')
        report = register_config(path, config, model_path=model)
        cfg = load_config(config)
        rows, _ = load_bank(path, cfg, FrozenRenderer(model))
        assert len(rows) == cfg['supervised_turns']
        assert cfg['bank_supervised_tokens'] == report['bank_supervised_tokens'] > 0
        run = subprocess.run([sys.executable, str(ROOT / 'tools/alf_pi1_train.py'), '--config', str(config),
                              '--model-path', str(model), '--preflight'], cwd=ROOT,
                             env=dict(os.environ, CUDA_VISIBLE_DEVICES=''), capture_output=True, text=True, check=True)
        assert json.loads(run.stdout)['assertions_passed']


class SweepStepper(StudentStepper):
    """After the student's one-step prefix, teacher solves with a 2/3 sweep suffix."""
    def step(self, cursor, command):
        next_command = {'go to table 1': 'go to cabinet 1', 'go to cabinet 1': 'open drawer 2',
                        'open drawer 2': 'put apple 1 on table 1'}
        if command in next_command:
            self.calls.append(command)
            self.location = True
            obs = Observation(cursor + 1, 'Reached ' + command, (next_command[command],), self.request['world_hash'])
            self.observations.append(asdict(obs))
            self.next_command = next_command[command]
            return cursor + 1, obs
        if command == 'look':
            obs = Observation(cursor + 1, 'Nothing happens.', (self.next_command,), self.request['world_hash'])
            self.observations.append(asdict(obs))
            return cursor + 1, obs
        return super().step(cursor, command)


def test_sweep_filtered_repairs_stay_paid_in_collection(d0, tmp_path):
    _, collection, output, result, teacher = run_repair(d0, tmp_path, student=Student(all_fail=True),
                                                      stepper_factory=SweepStepper)
    assert result['requests'] == 2 and result['usable_packages'] == 0 and len(teacher.calls) == 6
    assert result['purchase_cost']['calls'] == 6
    for payload_path in (collection / 'packages').glob('*.json'):
        payload = read_json(payload_path)
        assert payload['verification']['success_reproduced']
        assert payload['unavailable_reason'] == 'sweep_filtered' and payload['behaviors'] == []
        assert payload['sweep']['sweep_share'] == 2/3
        assert payload['teacher_react_turns'] and payload['student_prefix_commands'] == ['go to table 1']
    for path in (collection / 'requests').glob('*.json'):
        request = read_json(path)
        assert request['sweep_filtered'] and request['sweep_share'] == 2/3
        assert not request['training_usable'] and request['unavailable_reason'] == 'sweep_filtered'
    assert read_json(output / 'public/support.json')['training_task_ids'] == []
    assert audit_verified_bank(output)['passed']


def test_zero_prefix_takeover_and_remaining_horizon(d0, tmp_path):
    class BadFirst(Student):
        def generate(self, prompt, **kwargs):
            assert kwargs == dict(temperature=0.0, max_new_tokens=256)
            return Generation('unparseable response', 10, False)
    class ResetFallbackStepper(StudentStepper):
        def step(self, cursor, command):
            if not self.location and command == 'look':
                return cursor + 1, Observation(cursor + 1, 'Nothing happens.', ('go to table 1',), self.request['world_hash'])
            return super().step(cursor, command)
    _, _, output, result, teacher = run_repair(d0, tmp_path, student=BadFirst(), stepper_factory=ResetFallbackStepper)
    assert result['usable_packages'] == 2 and len(teacher.calls) == 4
    for payload in usable_packages(output).values():
        assert payload['student_prefix_commands'] == [] and payload['repair']['k'] == 0
        assert payload['repair']['remaining_horizon'] == 40
        assert len(payload['teacher_react_turns']) == 2


def test_replay_divergence_refuses_purchase(d0, tmp_path):
    directory = tmp_path / 'rollouts'
    rollouts.rollout_support(d0, directory, TwoTaskStudent(), stepper_factory=StudentStepper, support_size=2)
    class Divergent(StudentStepper):
        def step(self, cursor, command):
            cursor, obs = super().step(cursor, command)
            return cursor, Observation(obs.index, 'different actual state', obs.admissible, obs.world_hash, obs.done, obs.won)
    teacher = Teacher()
    result = repair.collect_repair(directory, d0, Path(str(d0) + '.collection'), tmp_path / 'collection', tmp_path / 'repair',
        stepper_factory=Divergent, generate_reply=teacher, config_loader=lambda _: SimpleNamespace(), support_size=2)
    assert teacher.calls == [] and result['usable_packages'] == 0
    assert result['purchase_cost']['calls'] == 0


def test_add_demo_filter_is_opt_in_and_keeps_paid_rows(source, tmp_path):
    # Full demonstration: one nonsweep prefix + 2 sweep turns + one final action,
    # then add another sweep command to cross the strict >1/2 boundary.
    class FullSweep(SweepStepper):
        def step(self, cursor, command):
            if command == 'open drawer 2':
                return cursor + 1, Observation(cursor + 1, 'drawer opened', ('close drawer 2',), self.request['world_hash'])
            if command == 'close drawer 2':
                return cursor + 1, Observation(cursor + 1, 'drawer closed', ('put apple 1 on table 1',), self.request['world_hash'])
            return super().step(cursor, command)
    for enabled in (False, True):
        out = tmp_path / f'demo-{enabled}'
        teacher = Teacher()
        result = pool.collect(source, out, stepper_factory=FullSweep, adapter_factory=factory(teacher),
                              attempt_start=3, temperature=.7, sweep_filter=enabled)
        assert result['verified'] == (0 if enabled else 2)
        assert result['completion_tokens'] == 200
        if enabled:
            packages = [read_json(p) for p in Path(str(out) + '.collection').joinpath('packages').glob('*.json')]
            assert len(packages) == 2 and all(p['unavailable_reason'] == 'sweep_filtered' for p in packages)
            assert all(p['sweep']['sweep_share'] == .6 for p in packages)
        else:
            assert all('sweep' not in p for p in usable_packages(out).values())
