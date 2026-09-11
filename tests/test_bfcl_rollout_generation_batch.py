"""CPU BFCL callback/tool isolation and serial/lockstep sampling equivalence."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT)]

from bfas.rtd import bfcl_decode
from bfas.rtd.benchmarks.bfcl_rollout import bfcl_task_rollouts_lockstep
from bfas.rtd.experiment import BFCLSupport
from bfas.rtd.functional_step import lora_parameters, snapshot
from bfas.rtd.generation_batch import FirstActionBackend, GenerationBatch
from bfas.rtd.metrics_v11 import GreedyBackend, freeze_tasks, greedy_success, repair_damage
from bfas.rtd.return_gradient import bfcl_task_rollout
from bfas.rtd.runtime import HFGenerateBackend
from bfas.rtd.transport import FullState


class Tokenizer:
    eos_token_id, bos_token_id, unk_token_id = 3, 0, -1

    def encode(self, text, add_special_tokens=False):
        row, _ = json.JSONDecoder().raw_decode(text)
        return [0] * (1 + row['task'] % 4) + [8 + row['task'], 64 + row['step'], 1]

    def decode(self, ids, skip_special_tokens=False):
        return '' if ids and ids[0] == 6 else 'Done.'

    def convert_tokens_to_ids(self, token):
        return {'<turn|>': 4, '<|tool_response>': 5}[token]


class FakePolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_w = torch.nn.Parameter(torch.tensor(.25))
        self.calls, self.owner, self.probe = [], threading.get_ident(), None

    def generate(self, *, input_ids, attention_mask, generation_config):
        assert threading.get_ident() == self.owner and input_ids.device.type == 'cpu'
        assert not self.training and float(self.lora_w) == .75
        c = generation_config
        tasks, steps = input_ids[:, -3] - 8, input_ids[:, -2] - 64
        self.calls.append((len(tasks), input_ids.shape[1], c.max_new_tokens))
        if self.probe:
            self.probe(tasks, steps)
        sequences, scores = input_ids, []
        done = torch.zeros(len(tasks), dtype=torch.bool)
        for index in range(c.max_new_tokens):
            logits = torch.full((len(tasks), 128), -1000.)
            for row, task in enumerate(tasks.tolist()):
                if index == task % 4:
                    logits[row, c.eos_token_id[task % len(c.eos_token_id)]] = self.lora_w
                else:
                    logits[row, 2] = self.lora_w
                    logits[row, 6] = 0.
            scores.append(logits)
            token = (torch.multinomial(logits.softmax(-1), 1).squeeze(1)
                     if c.do_sample else logits.argmax(-1))
            token[done] = c.pad_token_id
            sequences = torch.cat((sequences, token[:, None]), dim=1)
            done |= torch.isin(token, torch.tensor(c.eos_token_id))
            if done.all():
                break
        return SimpleNamespace(sequences=sequences, scores=scores)


@pytest.fixture
def executor_stub(monkeypatch):
    # Import only vendored Python utilities; never construct a real environment.
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root/'envs/bfcl/gorilla/berkeley-function-call-leaderboard'))
    from bfcl_eval.eval_checker.multi_turn_eval import multi_turn_utils
    from bfcl_eval import utils
    monkeypatch.setattr(utils, 'populate_test_cases_with_predefined_functions', lambda tasks: tasks)
    active, peak, histories, lock = set(), [], [], threading.Lock()
    failure = dict(stage=None)

    class Handler:
        model_name_underline_replaced = 'stub'

        def _format_prompt(self, messages, functions):
            return json.dumps(messages)

        def _extract_tool_calls(self, text):
            return []

        def _parse_query_response_prompting(self, response):
            text = response.choices[0].text
            self._extract_tool_calls(text)
            return text

        def decode_ast(self, *args, **kwargs):
            return []

        def decode_execute(self, *args, **kwargs):
            return []

        def inference(self, task, **kwargs):
            key = self.model_name_underline_replaced + '_tool_instance'
            state = []
            setattr(multi_turn_utils, key, state)
            with lock:
                active.add(key); peak.append(len(active))
            try:
                if failure['stage'] == 'reset':
                    raise RuntimeError('injected reset failure')
                for step in range(task['horizon']):
                    message = dict(task=task['task'], step=step, observations=state[:])
                    data = dict(message=message, function=[])
                    response, _ = self._query_prompting(data)
                    parsed = self._parse_query_response_prompting(response)
                    if failure['stage'] == 'step':
                        raise RuntimeError('injected step failure')
                    # In-process tool state belongs to this episode alone.
                    state.append(response.choices[0].text)
                    assert getattr(multi_turn_utils, key) is state
                    with lock:
                        histories.append((task['id'], step, deepcopy(message), response.choices[0].text))
                    task['question'].append(parsed)  # rollout must copy its input
                return state, {}
            finally:
                with lock:
                    active.remove(key)

    class Checker:
        def check_multi_turn(self, task, result, truth, category):
            if failure['stage'] == 'checker':
                raise RuntimeError('injected checker failure')
            return dict(valid=len(result) == task['horizon'] and task['task'] % 2 == 0)

    monkeypatch.setattr(bfcl_decode, 'student_handler', lambda *args: Handler())
    return SimpleNamespace(**locals())


def make_case(*, count=8, budget=16384, call_format='qwen'):
    backend = HFGenerateBackend(FakePolicy().eval(), Tokenizer(), base_checkpoint_hash='fake',
        harness_hash='bfcl-stub', tokenizer_hash='fake', max_context_tokens=512, max_action_tokens=9,
        action_caps={'multi_turn': 3}, generation_batch=GenerationBatch(max_batch_tokens=budget))
    backend.student_config = dict(student_call_format=call_format)
    parameters = snapshot(lora_parameters(backend.model))
    parameters['lora_w'].data.fill_(.75)
    support = object.__new__(BFCLSupport)
    support.parents, support.entries, support.categories, support.states = {}, {}, {}, {}
    support._truth = {'multi_turn_base': {}}
    for task in range(count):
        parent, tid = f'{task:02x}', f'multi_turn_base_{task}'
        entry = dict(id=tid, task=task, question=[], function=[], horizon=(1, 3, 2, 5)[task % 4])
        support.parents[parent], support.entries[tid] = tid, entry
        support.categories[tid] = 'multi_turn_base'
        support._truth['multi_turn_base'][tid] = dict(ground_truth=[])
        prompt = json.dumps(dict(task=task, step=0, observations=[]))
        support.states[parent] = FullState.create(entry, [dict(role='user', content='start')], prompt, parent)
    requests = [support._feedback_request(p) for p in support.parents]
    generators = [torch.Generator().manual_seed(17+i) for i in range(count)]
    return SimpleNamespace(**locals())


def normalized(value):
    if isinstance(value, dict):
        return {k: normalized(v) for k, v in value.items() if k not in
            {'batch_size', 'padded_prompt_tokens', 'sampling_batch_size', 'sampling_prompt_width'}}
    if isinstance(value, (list, tuple)):
        return [normalized(v) for v in value]
    return value


def assert_clean(case, stub):
    assert not stub.active
    assert not any(k.startswith('stub_rtd_') for k in vars(stub.multi_turn_utils))
    assert not any(t.name.startswith('bfcl-rollout') for t in threading.enumerate())
    assert case.backend.max_action_tokens == 9
    assert float(case.backend.model.lora_w.detach()) == .25


@pytest.mark.parametrize('budget', [16384, 22, 1])
@pytest.mark.parametrize('call_format', ['qwen', 'gemma4'])
@pytest.mark.parametrize('greedy', [False, True])
def test_bfcl_rollout_independent_streams_exact_serial_lockstep(executor_stub, budget, call_format, greedy):
    s = make_case(budget=budget, call_format=call_format)
    backend = GreedyBackend(s.backend) if greedy else s.backend
    original, global_rng = deepcopy(s.requests), torch.get_rng_state().clone()
    expected = [bfcl_task_rollout(*r, backend, s.parameters, g, checker=executor_stub.Checker())
                for r, g in zip(s.requests, s.generators)]
    states = [g.get_state() for g in s.generators]
    serial_calls = len(s.backend.model.calls)
    histories = list(executor_stub.histories)
    executor_stub.histories.clear()
    s.backend.model.calls.clear()
    s.generators = [torch.Generator().manual_seed(17+i) for i in range(s.count)]
    actual = bfcl_task_rollouts_lockstep(s.requests, backend, s.parameters, s.generators,
                                        checker=executor_stub.Checker())
    assert normalized([asdict(r) for r in actual]) == normalized([asdict(r) for r in expected])
    assert all(torch.equal(g.get_state(), state) for g, state in zip(s.generators, states))
    assert torch.equal(torch.get_rng_state(), global_rng)
    assert s.requests == original
    assert sorted(histories) == sorted(executor_stub.histories)
    assert [len(r.actions) for r in actual] == [1, 3, 2, 5]*2
    assert any(a.truncated for r in actual for a in r.actions)
    if budget > 1:
        assert len(s.backend.model.calls) < serial_calls
    assert all(n == 1 or n*(width+cap) <= budget for n, width, cap in s.backend.model.calls)
    assert_clean(s, executor_stub)


@pytest.mark.parametrize('budget', [16384, 22])
def test_bfcl_rollout_keeps_prefetched_start_groups(executor_stub, budget):
    s = make_case(budget=budget)
    requests = [r for r in s.requests[:4] for _ in range(2)]
    rng = torch.Generator().manual_seed(1)
    starts = []
    for parent in list(s.support.parents)[:4]:
        with s.backend.action_limit('multi_turn_base'):
            starts.extend(s.backend.sample_actions(s.support.states[parent].prompt, 2, s.parameters, rng))
    state = rng.get_state().clone()
    expected = [bfcl_task_rollout(*r, FirstActionBackend(s.backend, action), s.parameters, g,
                                  checker=executor_stub.Checker())
                for r, action, g in zip(requests, starts, s.generators)]
    s.generators = [torch.Generator().manual_seed(17+i) for i in range(s.count)]
    actual = bfcl_task_rollouts_lockstep(requests, s.backend, s.parameters, s.generators,
                                        first_actions=starts, checker=executor_stub.Checker())
    assert normalized([asdict(r) for r in actual]) == normalized([asdict(r) for r in expected])
    # The harness may mark a sampled start malformed; its sampling record stays
    # exactly the original logical group, including physical start metadata.
    assert all(r.actions[0].generation_metadata == a.generation_metadata for r, a in zip(actual, starts))
    assert torch.equal(rng.get_state(), state)
    assert_clean(s, executor_stub)


@pytest.mark.parametrize('limit', [1, 3, 32])
def test_bfcl_rollout_greedy_provider_and_fixed_slots(executor_stub, monkeypatch, limit):
    monkeypatch.setenv('BFAS_DIAGNOSTIC_LOCKSTEP_EPISODES', str(limit))
    s = make_case(call_format='gemma4')
    fixed = freeze_tasks([dict(parent_hash=p, official_id=t) for p, t in s.support.parents.items()], states=s.support.states)
    reports = []
    for enabled in (False, True):
        executor_stub.peak.clear()
        monkeypatch.setenv('BFAS_DIAGNOSTIC_LOCKSTEP', str(int(enabled)))
        reports.append(greedy_success(fixed, s.support, s.backend, s.parameters,
                                     executor_stub.Checker(), identity='fixed'))
        assert max(executor_stub.peak) == (min(limit, s.count) if enabled else 1)
    assert reports[0] == reports[1] and len(reports[0]) == 40
    assert repair_damage(*reports)['net_repair'] == 0
    assert_clean(s, executor_stub)


@pytest.mark.parametrize('failure', ['reset', 'step', 'checker', 'generation', 'cancel'])
def test_bfcl_rollout_failure_cleanup(executor_stub, failure):
    s = make_case(budget=22)
    executor_stub.failure['stage'] = failure
    if failure in {'generation', 'cancel'}:
        def fail(tasks, steps):
            if len(s.backend.model.calls) == 2:
                raise (KeyboardInterrupt if failure == 'cancel' else RuntimeError)('injected generation failure')
        s.backend.model.probe = fail
    with pytest.raises(KeyboardInterrupt if failure == 'cancel' else RuntimeError):
        bfcl_task_rollouts_lockstep(s.requests, s.backend, s.parameters, s.generators,
                                    checker=executor_stub.Checker())
    assert_clean(s, executor_stub)


def test_bfcl_rollout_rejects_shared_stochastic_rng(executor_stub):
    s = make_case()
    with pytest.raises(ValueError, match='independent episode RNG'):
        bfcl_task_rollouts_lockstep(s.requests, s.backend, s.parameters, [s.generators[0]] * s.count)
    assert not s.backend.model.calls and not executor_stub.peak


@pytest.mark.parametrize('value', ['', '0', '-1', '1.5', 'many'])
def test_bfcl_rollout_invalid_diagnostic_limit_before_sampling(executor_stub, monkeypatch, value):
    monkeypatch.setenv('BFAS_DIAGNOSTIC_LOCKSTEP_EPISODES', value)
    s = make_case()
    with pytest.raises(ValueError, match='BFAS_DIAGNOSTIC_LOCKSTEP_EPISODES'):
        list(s.support.diagnostic_batch(s.support.parents, GreedyBackend(s.backend), s.parameters,
                                        s.generators[0], executor_stub.Checker()))
    assert not s.backend.model.calls and not executor_stub.peak


def test_bfcl_rollout_diagnostic_limit_inherits_feedback(executor_stub, monkeypatch):
    monkeypatch.delenv('BFAS_DIAGNOSTIC_LOCKSTEP_EPISODES', raising=False)
    monkeypatch.setenv('BFAS_FEEDBACK_LOCKSTEP_EPISODES', '3')
    s = make_case()
    list(s.support.diagnostic_batch(s.support.parents, GreedyBackend(s.backend), s.parameters,
                                    s.generators[0], executor_stub.Checker()))
    assert max(executor_stub.peak) == 3
    assert_clean(s, executor_stub)
