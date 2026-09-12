"""D12 prefetch equivalence for round-end/offline estimator diagnostics on CPU."""
from dataclasses import replace
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT)]

from bfas.behavior.deltas import tensor_state_hash
from bfas.rtd.controls_v11 import sampler
from bfas.rtd.generation_batch import GenerationBatch
from bfas.rtd.metrics_v11 import ESTIMATORS, estimator_batches, gradient_variance
from test_rtd_v11_alpha_d import pair_problem
from test_rtd_v11_source_estimator import Tokenizer as SourceTokenizer
from test_unified_generation_batch import FakeGenerationPolicy


class Tokenizer(SourceTokenizer):
    def encode(self, text, add_special_tokens=False):
        return super().encode(text, add_special_tokens) if isinstance(text, str) else list(text)


def case(budget):
    original, start, source, pairs, step = pair_problem()
    backend = FakeGenerationPolicy(original.model, Tokenizer(), base_checkpoint_hash='toy',
        harness_hash='estimator', tokenizer_hash='toy', max_action_tokens=2, max_context_tokens=20,
        action_caps={'single_turn': 2, 'multi_turn': 3},
        generation_batch=GenerationBatch(prompts_per_batch=8, max_batch_tokens=budget))
    records = [pairs[0].record, pairs[1].record, pairs[0].record]
    state = replace(records[1].state, parent_hash='other', prompt='prompt other')
    records[1] = replace(records[1], state=state, teacher=replace(records[1].teacher, state=state))
    support = SimpleNamespace(parents={r.state.parent_hash: r.state.parent_hash for r in records},
        categories={records[0].state.parent_hash: 'simple', 'other': 'multi_turn_base'})
    payload = dict(source=source, window_id='estimator')
    checker = SimpleNamespace(check_syntax=lambda text: dict(valid='2' in text))
    calls, actions = [], []
    generate = backend._generate_group
    def counted(rows, parameters):
        calls.append((len(rows), max(len(r['ids']) for r in rows), rows[0]['limit']))
        result = generate(rows, parameters)
        actions.extend(result)
        return result
    backend._generate_group = counted
    return SimpleNamespace(**locals())


def instrument(c, batched):
    draw = sampler(c.payload, c.backend, c.support, identity='variance')
    samples = []
    def sample(*args):
        result = draw(*args)
        samples.append((args[1:], result))
        return result
    if batched:
        sample.prefetch = draw.prefetch
    return sample, samples


@pytest.mark.parametrize('budget', [16384, 18, 1])
def test_controls_v11_metrics_v11_batched_estimator_exact_equivalence(budget):
    cases, outputs = [], []
    global_rng = torch.get_rng_state().clone()
    for batched in (False, True):
        c = case(budget)
        sample, samples = instrument(c, batched)
        resident = tensor_state_hash(c.backend.model.state_dict())
        first = []
        report = gradient_variance(c.records, [.2, .3, .5], [.2, .7, .2], c.backend,
            c.start, c.source, c.step, c.checker, sample, R=3,
            on_first=lambda batches: first.append(batches))
        # Compare the next draw as well: the durable diagnostic stream consumed
        # exactly the same tickets, despite prefetch using a generator clone.
        following = sample(c.records[0], 3, 0, 0)
        assert len(samples) == 73 and report['source_draws'] == 72
        assert getattr(c.backend, '_pending_generation', None) is None
        assert c.backend.max_action_tokens == 2
        assert resident == tensor_state_hash(c.backend.model.state_dict())
        assert all(n == 1 or n*(width+cap) <= budget for n, width, cap in c.calls)
        outputs.append((report, samples, following, first[0]))
        cases.append(c)
    assert outputs[0][:3] == outputs[1][:3]
    for name in ESTIMATORS:
        serial, batched = outputs[0][3][name], outputs[1][3][name]
        assert serial.start_hash == batched.start_hash
        assert torch.equal(serial.weights, batched.weights) and torch.equal(serial.alpha, batched.alpha)
        assert [tensor_state_hash(g) for g in serial.baseline_gradients] == [
            tensor_state_hash(g) for g in batched.baseline_gradients]
    assert torch.equal(torch.get_rng_state(), global_rng)
    if budget > 1:
        assert len(cases[1].calls) < len(cases[0].calls)
        assert max(n for n, _, _ in cases[1].calls) > 1


@pytest.mark.parametrize('failure', ['consumption', 'scoring'])
def test_controls_v11_metrics_v11_prefetch_cleanup_and_repeat_replay(monkeypatch, failure):
    c = case(16384)
    sample, samples = instrument(c, True)
    if failure == 'consumption':
        original = c.backend.sample_action
        def fail(*args, **kwargs):
            if len(samples) == 9:
                raise RuntimeError('injected consumption failure')
            return original(*args, **kwargs)
        monkeypatch.setattr(c.backend, 'sample_action', fail)
    else:
        def fail(*args):
            assert c.backend._pending_generation is None
            raise RuntimeError('injected scoring failure')
        monkeypatch.setattr('bfas.rtd.metrics_v11.source_gradient_pair', fail)
    with pytest.raises(RuntimeError, match='injected'):
        estimator_batches(c.records, [.2, .3, .5], [.2, .7, .2], c.backend, c.start, c.source,
                          c.step, c.checker, sample, repeat=0)
    assert c.backend._pending_generation is None and c.backend.max_action_tokens == 2
    monkeypatch.undo()
    replay, replayed = instrument(c, True)
    estimator_batches(c.records, [.2, .3, .5], [.2, .7, .2], c.backend, c.start, c.source,
                      c.step, c.checker, replay, repeat=0)
    assert replayed[:len(samples)] == samples
