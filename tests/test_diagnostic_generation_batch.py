"""CPU argmax/episode equality through the production D16 dispatch and scheduler."""
from contextlib import contextmanager
from dataclasses import asdict
import json
import threading
from types import MethodType, SimpleNamespace

import pytest
import torch

from bfas.rtd.benchmarks.alfworld_state import Observation, parent_hash
from bfas.rtd.benchmarks.registry import (
    ALFWorldExperimentSupport, ALFWorldFeedbackContext, alfworld_action_limit,
)
from bfas.rtd.functional_step import lora_parameters, snapshot
from bfas.rtd.generation_batch import GenerationBatch
from bfas.rtd.metrics_v11 import GreedyBackend, freeze_tasks, greedy_success, repair_damage
from bfas.rtd.runtime import HFGenerateBackend
from bfas.rtd.transport import FullState


def render(request, history):
    return json.dumps(dict(task=int(request['task_id'].split('/')[0]), step=len(history)//2))


class Tokenizer:
    eos_token_id, bos_token_id, unk_token_id = 3, 0, -1

    def encode(self, text, add_special_tokens=False):
        row = json.loads(text)
        return [0]*(1+row['task'] % 4) + [8+row['task'], 64+row['step'], 1]

    def decode(self, ids, skip_special_tokens=False):
        return 'look' if ids else ''

    def convert_tokens_to_ids(self, token):
        assert token == '<turn|>'
        return 4


class FakeGreedyPolicy(torch.nn.Module):
    """Real generation boundary and argmax; batch-independent per-row logits."""
    def __init__(self):
        super().__init__()
        self.lora_w = torch.nn.Parameter(torch.tensor(.25))
        self.owner = threading.get_ident()
        self.calls = []
        self.probe = None

    def generate(self, *, input_ids, attention_mask, generation_config):
        assert threading.get_ident() == self.owner
        assert input_ids.device.type == attention_mask.device.type == 'cpu'
        c = generation_config
        assert not self.training and not c.do_sample and c.num_beams == 1
        assert c.use_cache and c.output_scores and c.return_dict_in_generate
        assert c.repetition_penalty == 1 and c.eos_token_id == [3, 4]
        assert float(self.lora_w) == .75  # evaluated virtual checkpoint is installed
        tasks, steps = input_ids[:, -3]-8, input_ids[:, -2]-64
        self.calls.append((len(tasks), input_ids.shape[1], c.max_new_tokens, tuple(steps.tolist())))
        if self.probe:
            self.probe(tasks, steps)
        sequences, scores = input_ids, []
        done = torch.zeros(len(tasks), dtype=torch.bool)
        for index in range(c.max_new_tokens):
            logits = torch.zeros((len(tasks), 128))
            for row, task in enumerate(tasks.tolist()):
                token = 3 + task % 2 if index == task % 3 else 2
                logits[row, token] = 2 + self.lora_w
            scores.append(logits)
            token = logits.argmax(-1)
            token[done] = c.pad_token_id
            sequences = torch.cat((sequences, token[:, None]), dim=1)
            done |= torch.isin(token, torch.tensor(c.eos_token_id))
            if done.all():
                break
        return SimpleNamespace(sequences=sequences, scores=scores)


def make_case(*, count=8, budget=16384, horizons=(1, 3, 2, 41), env_probe=None):
    support = object.__new__(ALFWorldExperimentSupport)
    support.parents, support.states = {}, {}
    for task in range(count):
        request = dict(task_id=f'{task}/trial')
        h = parent_hash(request['task_id'])
        support.parents[h] = request['task_id']
        support.states[h] = FullState.create(request, [dict(role='user', content='start')], render(request, []), h)
    support.protocol = object()
    workers, active, peaks, rendered = [], set(), [], []
    owner = threading.get_ident()

    class Env:
        def __init__(self):
            assert threading.get_ident() == owner
            self.closed = self.stepping = False
            self.commands = []
            active.add(id(self)); peaks.append(len(active)); workers.append(self)

        def reset(self, request):
            if env_probe:
                env_probe('reset', 0)
            self.task = int(request['task_id'].split('/')[0])
            self.horizon = horizons[self.task % len(horizons)]
            return 0, Observation(0, 'start', ('look',), 'world', done=False, won=False)

        def step(self, cursor, command):
            self.stepping = True
            try:
                if env_probe:
                    env_probe('step', cursor)
                assert cursor == len(self.commands)
                self.commands.append(str(command))
                index = cursor+1
                return index, Observation(index, f'step {index}', ('look',), 'world',
                    done=index == self.horizon, won=index == self.horizon and self.task % 2 == 0)
            finally:
                self.stepping = False

        def close(self):
            assert threading.get_ident() == owner and not self.stepping
            self.closed = True
            active.remove(id(self))

    def renderer(request, history):
        assert threading.get_ident() == owner
        rendered.append((request['task_id'], history))
        return render(request, history)

    backend = HFGenerateBackend(FakeGreedyPolicy().eval(), Tokenizer(), base_checkpoint_hash='fake',
        harness_hash='diagnostic', tokenizer_hash='fake', max_context_tokens=512, max_action_tokens=512,
        action_caps={'agent_action': 256}, generation_batch=GenerationBatch(prompts_per_batch=1,
                                                                         max_batch_tokens=budget))
    backend.student_config = dict(student_call_format='gemma4', benchmark='alfworld')
    backend.action_limit = MethodType(alfworld_action_limit, backend)
    counts = []

    @contextmanager
    def measured(operation, **values):
        assert operation == 'diagnostic_greedy_generation'
        counts.append(values)
        yield

    backend.measured = measured
    context = ALFWorldFeedbackContext(support.protocol, 1, renderer, Env, None)
    parameters = snapshot(lora_parameters(backend.model))
    parameters['lora_w'].data.fill_(.75)
    fixed = freeze_tasks([dict(parent_hash=h, official_id=tid) for h, tid in support.parents.items()],
                         states=support.states)
    return SimpleNamespace(**locals())


def assert_clean(s):
    assert not s.active and all(w.closed for w in s.workers)
    assert s.backend.max_action_tokens == 512
    assert float(s.backend.model.lora_w.detach()) == .25
    assert not any(t.name.startswith('alfworld-diagnostic') for t in threading.enumerate())


@pytest.mark.parametrize('limit,budget', [(None, 16384), (1, 16384), (3, 16384), (32, 530), (32, 1)])
def test_window_metrics_diagnostic_exact_serial_lockstep_records(monkeypatch, limit, budget):
    monkeypatch.delenv('BFAS_FEEDBACK_LOCKSTEP_EPISODES', raising=False)
    monkeypatch.delenv('BFAS_DIAGNOSTIC_LOCKSTEP_EPISODES', raising=False)
    if limit is not None:
        monkeypatch.setenv('BFAS_DIAGNOSTIC_LOCKSTEP_EPISODES', str(limit))
    serial, batched = make_case(budget=budget), make_case(budget=budget)
    rng = torch.Generator().manual_seed(19)
    before, global_before = rng.get_state().clone(), torch.get_rng_state().clone()
    parents = tuple(serial.support.parents)
    expected = [(p, serial.support.feedback(p, GreedyBackend(serial.backend), serial.parameters, rng, serial.context))
                for p in parents]
    actual = list(batched.support.diagnostic_batch(parents, GreedyBackend(batched.backend),
                                                  batched.parameters, rng, batched.context))
    assert [(p, asdict(r)) for p, r in actual] == [(p, asdict(r)) for p, r in expected]
    assert [len(r.actions) for _, r in actual] == [1, 3, 2, 40]*2
    assert [w.commands for w in serial.workers] == [w.commands for w in batched.workers]
    assert sorted(serial.rendered, key=lambda r: (r[0], len(r[1]))) == sorted(
        batched.rendered, key=lambda r: (r[0], len(r[1])))
    assert torch.equal(rng.get_state(), before) and torch.equal(torch.get_rng_state(), global_before)
    assert max(batched.peaks) == min(limit or 32, 8)
    if limit != 1 and budget > 1:
        assert len(batched.backend.model.calls) < len(serial.backend.model.calls)
        assert max(c['sequences'] for c in batched.counts) > 1
    assert all(n == 1 or n*(width+cap) <= budget for n, width, cap, _ in batched.backend.model.calls)
    for s in (serial, batched):
        assert_clean(s)


def test_window_metrics_diagnostic_fixed_slots_and_default_32(monkeypatch):
    monkeypatch.delenv('BFAS_DIAGNOSTIC_LOCKSTEP_EPISODES', raising=False)
    monkeypatch.delenv('BFAS_FEEDBACK_LOCKSTEP_EPISODES', raising=False)
    records = []
    for enabled in (False, True):
        monkeypatch.setenv('BFAS_DIAGNOSTIC_LOCKSTEP', str(int(enabled)))
        s = make_case(count=40, horizons=(1, 2))
        result = greedy_success(s.fixed, s.support, s.backend, s.parameters, s.context, identity='window')
        records.append(result)
        assert len(result) == 40
        assert len(s.workers) == len({r['parent_hash'] for f in s.fixed['folds'].values() for r in f['states']})
        assert max(s.peaks) == (32 if enabled else 1)
        assert_clean(s)
    assert records[0] == records[1]
    assert repair_damage(records[0], records[1])['net_repair'] == 0


def test_diagnostic_parallel_reset_and_step_rpcs(monkeypatch):
    monkeypatch.setenv('BFAS_DIAGNOSTIC_LOCKSTEP_EPISODES', '8')
    barrier = threading.Barrier(8, timeout=10)
    owner = threading.get_ident()

    def probe(stage, cursor):
        assert threading.get_ident() != owner
        barrier.wait()

    s = make_case(horizons=(2,), env_probe=probe)
    list(s.support.diagnostic_batch(tuple(s.support.parents), GreedyBackend(s.backend),
                                   s.parameters, torch.Generator(), s.context))
    assert_clean(s)


def test_diagnostic_concurrency_inherits_feedback_override(monkeypatch):
    monkeypatch.delenv('BFAS_DIAGNOSTIC_LOCKSTEP_EPISODES', raising=False)
    monkeypatch.setenv('BFAS_FEEDBACK_LOCKSTEP_EPISODES', '3')
    s = make_case(horizons=(1,))
    list(s.support.diagnostic_batch(tuple(s.support.parents), GreedyBackend(s.backend),
                                   s.parameters, torch.Generator(), s.context))
    assert max(s.peaks) == 3
    assert_clean(s)


@pytest.mark.parametrize('failure', ['reset', 'step', 'generation', 'cancel'])
def test_diagnostic_lockstep_failure_cleanup_without_retry(monkeypatch, failure):
    monkeypatch.setenv('BFAS_DIAGNOSTIC_LOCKSTEP_EPISODES', '8')

    def env_probe(stage, cursor):
        if stage == failure:
            raise BrokenPipeError('injected RPC failure')

    s = make_case(env_probe=env_probe)
    if failure in {'generation', 'cancel'}:
        def fail(tasks, steps):
            raise (KeyboardInterrupt if failure == 'cancel' else RuntimeError)('injected generation failure')
        s.backend.model.probe = fail
    with pytest.raises(KeyboardInterrupt if failure == 'cancel' else (BrokenPipeError, RuntimeError)):
        list(s.support.diagnostic_batch(tuple(s.support.parents), GreedyBackend(s.backend),
                                       s.parameters, torch.Generator(), s.context))
    assert len(s.workers) == 8  # original diagnostic failure propagates; no retry
    assert_clean(s)


@pytest.mark.parametrize('value', ['', '0', '-1', '1.5', 'many'])
def test_diagnostic_invalid_concurrency_before_io(monkeypatch, value):
    monkeypatch.setenv('BFAS_DIAGNOSTIC_LOCKSTEP_EPISODES', value)
    s = make_case()
    with pytest.raises(ValueError, match='BFAS_DIAGNOSTIC_LOCKSTEP_EPISODES'):
        list(s.support.diagnostic_batch(tuple(s.support.parents), GreedyBackend(s.backend),
                                       s.parameters, torch.Generator(), s.context))
    assert not s.workers and not s.backend.model.calls


def test_diagnostic_generation_batch_context_caps_and_first_stop():
    s = make_case()
    s.backend.max_context_tokens = 8
    s.backend.max_action_tokens = 3
    prompts = [render(dict(task_id=f'{i}/trial'), []) for i in range(8)]
    greedy = GreedyBackend(s.backend)
    expected = [greedy.sample_action(p, s.parameters, torch.Generator()) for p in prompts]
    actual = s.backend.greedy_actions(prompts, s.parameters, prompts_per_batch=32)
    assert list(actual) == expected
    assert any(a.truncated for a in actual)
    assert any(a.eos_token_id == 4 for a in actual)
    assert all(len(a.action_ids) == len(a.generation_token_logprobs) for a in actual)
