"""D12: real tiny HF/PEFT CPU decoding, RNG recovery and runtime consumers."""
from dataclasses import asdict
import json
from pathlib import Path

import pytest
import torch

from bfas.rtd.cli import load_config
from bfas.rtd.functional_step import lora_parameters, snapshot
from bfas.rtd.generation_batch import GenerationBatch, feedback_rollouts, ticket
from bfas.rtd.persistence import ComputeJournal, digest
from bfas.rtd.return_gradient import IncompleteRolloutError, TaskRollout
from bfas.rtd.runtime import HFGenerateBackend
from bfas.rtd.scoring import ScoreTolerance, enforce_score_diagnostic, score_diagnostic


class Tokenizer:
    eos_token_id, bos_token_id = 3, 0

    def encode(self, text, add_special_tokens=False):
        return [int(x) for x in text.split()]

    def decode(self, ids, skip_special_tokens=False):
        return ' '.join(map(str, ids))


@pytest.fixture
def backend():
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
    torch.manual_seed(81)
    model = Qwen3_5ForCausalLM(Qwen3_5TextConfig(vocab_size=8, hidden_size=16,
        intermediate_size=32, num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
        head_dim=8, linear_key_head_dim=8, linear_value_head_dim=8, linear_num_key_heads=2,
        linear_num_value_heads=2, layer_types=['linear_attention', 'full_attention'],
        max_position_embeddings=512, eos_token_id=3, bos_token_id=0, attn_implementation='eager'))
    model = get_peft_model(model, LoraConfig(r=2, lora_alpha=4, lora_dropout=0.,
        task_type='CAUSAL_LM', target_modules=['q_proj', 'v_proj', 'o_proj'])).eval()
    # Nonzero LoRA verifies generation's temporary parameter installation too.
    with torch.no_grad():
        for p in lora_parameters(model).values():
            p.normal_(0, .05)
    return HFGenerateBackend(model, Tokenizer(), base_checkpoint_hash='tiny-qwen35',
        harness_hash='d12', tokenizer_hash='integer-vocab', max_action_tokens=8,
        max_context_tokens=512, generation_batch=GenerationBatch())


def rng(seed=9):
    return torch.Generator().manual_seed(seed)


def checked(backend, action, parameters):
    with torch.no_grad():
        score, values, metadata = backend.score_action(action, parameters, return_details=True)
    diagnostic = score_diagnostic(action, values, score, metadata, backend.score_tolerance,
                                  expected_prompt_ids=action.prompt_ids)
    enforce_score_diagnostic(diagnostic)
    assert diagnostic['max_abs_difference'] < 1e-4
    return diagnostic


def test_same_prompt_one_generate_independent_eos_and_scores(backend, monkeypatch):
    p = snapshot(lora_parameters(backend.model))
    calls = []
    generate = backend.model.generate
    def spy(**kw):
        calls.append(kw)
        return generate(**kw)
    monkeypatch.setattr(backend.model, 'generate', spy)
    actions = backend.sample_actions((0, 1, 2), 24, p, rng())
    assert len(calls) == 1 and calls[0]['input_ids'].shape == (24, 3)
    assert len({a.generation_metadata['rng_ticket']['draw_id'] for a in actions}) == 24
    assert any(a.truncated for a in actions) and any(not a.truncated for a in actions)
    for a in actions:
        assert len(a.action_ids) == len(a.generation_token_logprobs)
        assert 3 not in a.action_ids[:-1]
        assert a.truncated == (a.action_ids[-1] != 3)
        checked(backend, a, p)


def test_qwen35_left_padding_prefill_cache_and_depadding_fp32(backend, monkeypatch):
    p = snapshot(lora_parameters(backend.model))
    # Cross a delta-rule chunk boundary, with a real short row in each bucket.
    prompts = [(0, 1)*37, (2,)*65, (1,), (2, 0), (0, 1)*69, (2,)*127]
    calls = []
    generate = backend.model.generate
    def spy(**kw):
        calls.append((kw['input_ids'].clone(), kw['attention_mask'].clone()))
        return generate(**kw)
    monkeypatch.setattr(backend.model, 'generate', spy)
    actions = backend.sample_actions_batch(prompts, 2, p, rng())
    assert len(calls) < len(prompts)
    assert any(bool((mask == 0).any()) for _, mask in calls)
    for ids, mask in calls:
        assert bool((mask[:, 1:] >= mask[:, :-1]).all())
        assert bool((mask[:, -1] == 1).all())
    for prompt, group in zip(prompts, actions):
        assert len(group) == 2
        for a in group:
            assert a.prompt_ids == prompt
            checked(backend, a, p)


def test_context_caps_budget_order_and_no_forced_eos(backend):
    backend.max_context_tokens = 12
    p = snapshot(lora_parameters(backend.model))
    prompts = [(0, 1), (1,)*11, (2,)*9]
    groups = backend.sample_actions_batch(prompts, 3, p, rng(), max_batch_tokens=24)
    for prompt, actions in zip(prompts, groups):
        for a in actions:
            assert a.prompt_ids == prompt
            assert a.generation_metadata['effective_action_limit'] == min(8, 12-len(prompt))
            assert a.generation_metadata['batch_size'] * (
                a.generation_metadata['padded_prompt_tokens'] + a.generation_metadata['effective_action_limit']) <= 24
            checked(backend, a, p)
    state = rng()
    before = state.get_state().clone()
    with pytest.raises(IncompleteRolloutError):
        backend.sample_actions_batch([(1,), (2,)*12], 2, p, state)
    assert torch.equal(before, state.get_state())


def test_batch_sampling_law_matches_unbatched_tiny_vocabulary(backend):
    # HF multinomial consumes randomness differently for different row counts.
    # Compare independent empirical categorical laws, not seed-equal strings.
    backend.max_action_tokens = 1
    p = snapshot(lora_parameters(backend.model))
    n = 512
    batch = backend.sample_actions((0, 1), n, p, rng(3))
    backend.generation_batch = None  # original batch-1 HF implementation
    single_rng = rng(43)
    single = [backend.sample_action('0 1', p, single_rng) for _ in range(n)]
    with torch.no_grad():
        prob = backend._logits(p, torch.tensor([[0, 1]]))[0, -1].softmax(-1).double()
    counts = [torch.bincount(torch.tensor([a.action_ids[0] for a in group]), minlength=8).double()/n
              for group in (batch, single)]
    bound = 6 * (prob*(1-prob)/n).sqrt() + 1/n
    for frequency in counts:
        assert bool(((frequency-prob).abs() < bound).all())
    assert bool(((counts[0]-counts[1]).abs() < 6*(2*prob*(1-prob)/n).sqrt()+1/n).all())
    # IDs carry the enabled backend identity; scoring math is independent of it.
    for a in batch[:16] + tuple(single[:16]):
        checked(backend, a, p)


def test_rng_tickets_recovery_journal_and_parameter_restoration(backend, tmp_path):
    backend.journal = ComputeJournal(tmp_path/'compute.jsonl', cuda=False)
    original = snapshot(lora_parameters(backend.model))
    p = {k: v+.03 for k, v in original.items()}
    state = rng()
    global_state = torch.get_rng_state().clone()
    saved = state.get_state().clone()
    prompts = [(0, 1), (2, 1, 0), (1,)]
    first = backend.sample_actions_batch(prompts, 2, p, state)
    end = state.get_state().clone()
    state.set_state(saved)
    again = backend.sample_actions_batch(prompts, 2, p, state)
    assert first == again and torch.equal(end, state.get_state())
    assert torch.equal(global_state, torch.get_rng_state())
    for name, value in lora_parameters(backend.model).items():
        assert torch.equal(original[name], value)
    expected = rng()
    for a in [a for group in first for a in group]:
        assert ticket(expected) == a.generation_metadata['rng_ticket']
    rows = [json.loads(s) for s in backend.journal.path.read_text().splitlines()]
    generated = [r for r in rows if r['kind'] == 'generated_tokens']
    assert len(generated) == 12
    assert {r['sample_hash'] for r in generated} == {digest(asdict(a)) for group in first for a in group}
    assert all(r['rng_before'] != r['rng_after'] for r in generated)


def test_prefetch_consumes_each_pair_once_and_replays_after_failure(backend):
    p = snapshot(lora_parameters(backend.model))
    requests = [('0 1', 2, 8), ('2 1 0', 2, 8)]
    state = rng()
    before = state.get_state().clone()
    with pytest.raises(RuntimeError, match='crash'):
        with backend.prefetch_actions(requests, p, state):
            pair = backend.sample_actions('0 1', 2, p, state)
            assert not torch.equal(before, state.get_state())
            raise RuntimeError('crash before durable phase')
    assert backend._pending_generation is None
    state.set_state(before)  # existing StateStore phase recovery
    with backend.prefetch_actions(requests, p, state):
        assert backend.sample_actions('0 1', 2, p, state) == pair
        backend.sample_actions('2 1 0', 2, p, state)
    with pytest.raises(AssertionError, match='not fully consumed'):
        with backend.prefetch_actions(requests, p, state):
            pass


def test_feedback_samples_k_task_starts_and_runs_each_continuation(backend):
    from types import SimpleNamespace
    p = snapshot(lora_parameters(backend.model))
    class Support:
        parents = {'h': 'task'}
        categories = {'task': 'multi_turn'}
        states = {'h': SimpleNamespace(prompt='0 1')}
        def feedback(self, parent, b, parameters, generator, checker):
            a = b.sample_action('0 1', parameters, generator)
            continuation = b.sample_action('2 1 0', parameters, generator)
            return TaskRollout('task', (a, continuation), 0., b.identity(parameters))
    actions = list(feedback_rollouts(Support(), 'h', 4, backend, p, rng(), None))
    assert len(actions) == 4
    assert all(r.actions[0].generation_metadata['batch_size'] == 4 for r in actions)
    assert all(r.actions[1].generation_metadata['batch_size'] == 1 for r in actions)


def test_config_opt_in_identity_and_unchanged_outlier_guard(backend):
    root = Path(__file__).resolve().parents[1]
    v10 = load_config(root/'configs/rtd/v1_bfcl_c25.yaml')
    v11 = load_config(root/'configs/rtd/v1_1_bfcl.yaml', arm='V0')
    assert 'generation_batch' not in v10
    # Production disabled D13 forward batching after the bf16 benchmark;
    # normalization omits the explicit zero while retaining D12 generation.
    assert v11['generation_batch'] == dict(prompts_per_batch=8, max_batch_tokens=16384)
    assert GenerationBatch.from_config(v11).forward_prompts_per_batch == 0
    assert ScoreTolerance.from_config(v11) == ScoreTolerance(.05, 1., 2, 8.)
    legacy_args = dict(base_checkpoint_hash='tiny-qwen35', harness_hash='d12', tokenizer_hash='integer-vocab',
                       max_action_tokens=8, max_context_tokens=512)
    legacy = HFGenerateBackend(backend.model, backend.tokenizer, **legacy_args)
    explicit_off = HFGenerateBackend(backend.model, backend.tokenizer, generation_batch=None, **legacy_args)
    assert legacy.backend_id == explicit_off.backend_id != backend.backend_id
    p = snapshot(lora_parameters(backend.model))
    assert legacy.sample_action('0 1', p, rng()) == explicit_off.sample_action('0 1', p, rng())
    with pytest.raises(ValueError, match='1.1.0'):
        GenerationBatch.from_config(v10 | dict(generation_batch={}))
    for value in (0, -1, True, 1.2):
        with pytest.raises(ValueError):
            GenerationBatch(max_batch_tokens=value)
    p = snapshot(lora_parameters(backend.model))
    a = backend.sample_actions((0, 1), 1, p, rng())[0]
    values = torch.tensor(a.generation_token_logprobs)
    for delta, should_pass in ((.01, True), (8.01, False)):
        changed = values.clone(); changed[0] -= delta
        d = score_diagnostic(a, changed, changed.sum(), {}, backend.score_tolerance)
        assert d['passed'] == should_pass


def test_real_source_pair_checks_and_durable_midphase_resume(backend, tmp_path):
    from types import SimpleNamespace
    from bfas.rtd.alpha_d import ExposureRecord
    from bfas.rtd.experiment import RTDExperiment
    from bfas.rtd.experiment_alpha_d import AlphaDExperimentMixin
    from bfas.rtd.ledger import Ledger
    from bfas.rtd.persistence import StateStore
    from bfas.rtd.transport import FullState
    class Engine(AlphaDExperimentMixin):
        sample_state = RTDExperiment.sample_state
    e = Engine()
    e.backend, e.config = backend, dict(source_samples_per_state=2)
    e.sampling_rng = rng()
    e.journal = backend.journal = ComputeJournal(tmp_path/'compute.jsonl', cuda=False)
    e.ledger = Ledger(0, tmp_path/'teacher.jsonl')
    p = snapshot(lora_parameters(backend.model))
    states = [FullState.create({'q': str(i)}, [{'role': 'user', 'content': str(i)}],
                              prompt, str(i)) for i, prompt in enumerate(('0 1', '2 1 0'))]
    e.support = SimpleNamespace(parents={s.parent_hash: s.parent_hash for s in states},
        categories={s.parent_hash: 'single_turn' for s in states})
    e.state = dict(phase='revealed', round=1, step=1, source=p, source_id=backend.identity(p),
        inner={s.parent_hash for s in states}, source_cache={},
        projection_cache={s.state_hash: [] for s in states}, sampling_rng=e.sampling_rng.get_state())
    store = StateStore(tmp_path/'recovery', {'config': e.config})
    store.save(e.state, e.ledger)
    records = [ExposureRecord(None, 0, s, None) for s in states]
    sample_state = e.sample_state
    def crash(state, **kw):
        if state == states[1]:
            raise RuntimeError('injected source phase crash')
        return sample_state(state, **kw)
    e.sample_state = crash
    with pytest.raises(RuntimeError, match='injected'):
        e.alpha_draw_pairs(records, role='commit')
    failed_pair = [r for r in e.journal.events if r['kind'] == 'alpha_d_source_pair'][0]
    e.state = store.load(e.ledger)
    e.sampling_rng.set_state(e.state['sampling_rng'])
    e.sample_state = sample_state
    pairs = e.alpha_draw_pairs(records, role='commit')
    recovered = [r for r in e.journal.events if r['kind'] == 'alpha_d_source_pair'][-2:]
    for key in ('sample_hashes', 'action_hashes', 'draw_ids', 'rng_before', 'rng_after'):
        assert recovered[0][key] == failed_pair[key]
    assert recovered[0]['rng_after'] == recovered[1]['rng_before']
    assert len(pairs) == 2 and all(len(p.sources) == 2 for p in pairs)
    assert not e.state['source_cache']
    assert len([r for r in e.journal.events if r['kind'] == 'score_consistency']) == 6
    e.state['last_committed_draw_ids'] = [d for p in pairs for d in p.draw_ids]
    e.state['step'] = 2
    fresh = e.alpha_draw_pairs(records, role='commit')
    assert not set(e.state['last_committed_draw_ids']) & {d for p in fresh for d in p.draw_ids}


def test_benchmark_guard_fails_per_action_without_averaging_out_bad_tokens(backend):
    from dataclasses import replace
    from tools.rtd_v11_generation_bench import agreement
    p = snapshot(lora_parameters(backend.model))
    actions = backend.sample_actions((0, 1), 4, p, rng())
    assert agreement(backend, actions, p)['passed']
    a = actions[0]
    changed = list(a.generation_token_logprobs)
    changed[0] -= 8.1
    corrupted = replace(a, generation_token_logprobs=tuple(changed), generation_logprob=sum(changed))
    report = agreement(backend, (corrupted, *actions[1:]), p)
    assert not report['passed'] and report['failed_action_indices'] == [0]
    assert report['max_abs_diff'] > 8 and report['fraction_gt_1_nat'] > 0


def test_all_feedback_roles_and_validation_use_batched_k(backend, tmp_path):
    from contextlib import nullcontext
    from types import SimpleNamespace
    from bfas.rtd.experiment import RTDExperiment
    from bfas.rtd.experiment_alpha_d import AlphaDExperimentMixin
    class Support:
        parents = {'h': 'task'}
        categories = {'task': 'single_turn'}
        states = {'h': SimpleNamespace(prompt='0 1')}
        def feedback(self, parent, b, parameters, generator, checker):
            a = b.sample_action('0 1', parameters, generator)
            return TaskRollout('task', (a,), float(a.action_ids[0] == 1), b.identity(parameters))
    class Engine(AlphaDExperimentMixin):
        feedback = RTDExperiment.feedback
        def scope(self, role):
            return nullcontext()
    e = Engine()
    e.backend, e.support, e.checker = backend, Support(), None
    e.config, e.manifest = dict(training_seed=0), dict(arm='V0')
    e.device, e.sampling_rng = torch.device('cpu'), rng()
    e.alpha_d = e.v11 = True
    e.state = dict(round=1, step=1, window_id='window', smoke=False, feedback_tasks=[('h', 4)])
    e.journal = ComputeJournal(tmp_path/'compute.jsonl', cuda=False)
    p = snapshot(lora_parameters(backend.model))
    for role in ('acquisition_reference_feedback', 'same_batch_reference_feedback', 'post_commit_feedback'):
        result = e.alpha_feedback(p, role, isolated=role == 'same_batch_reference_feedback')
        assert result.metadata['feedback_role'] == role
        assert all(r.actions[0].generation_metadata['batch_size'] == 4
                   for r in e.state['feedback_rollouts'][role])
    result = e.alpha_validation_return(p, 'validation_test')
    assert len(result['rewards']) == 4
    rows = [r for r in e.journal.events if r['kind'] == 'validation_rollout']
    assert len(rows) == 4
    assert all(r['rollout']['actions'][0]['generation_metadata']['batch_size'] == 4 for r in rows)


def test_fixed_parse_battery_batches_160_fresh_actions(backend, tmp_path):
    from types import SimpleNamespace
    from bfas.rtd.metrics_v11 import freeze_tasks, parse_battery
    from bfas.rtd.transport import FullState
    states = {h: FullState.create({'q': h}, [{'role': 'user', 'content': h}], prompt, h)
              for h, prompt in [('00', '0 1'), ('01', '2 1 0')]}
    support = SimpleNamespace(states=states, parents={h: h for h in states},
                              categories={h: 'single_turn' for h in states})
    fixed = freeze_tasks([dict(parent_hash=h, official_id=h) for h in states], states=states)
    backend.journal = ComputeJournal(tmp_path/'compute.jsonl', cuda=False)
    p = snapshot(lora_parameters(backend.model))
    report = parse_battery(fixed, support, backend, p, SimpleNamespace(check_syntax=lambda text: {'valid': True}),
                          identity='independent-parse')
    assert report['samples'] == 160 and report['failure_rate'] == 0
    rows = backend.journal.events
    generated = [r for r in rows if r['kind'] == 'generated_tokens']
    assert len(generated) == len({r['draw_id'] for r in generated}) == 160
    assert len([r for r in rows if r['kind'] == 'compute_begin']) == 5
    assert backend._pending_generation is None


def test_generation_failure_restores_parameters_global_rng_and_hooks(backend, monkeypatch):
    original = snapshot(lora_parameters(backend.model))
    changed = {k: v+.03 for k, v in original.items()}
    global_rng = torch.get_rng_state().clone()
    generation_model = backend.model.get_base_model()
    hooks = dict(generation_model._forward_hooks)
    def fail(**kw):
        torch.rand(10)
        raise RuntimeError('decode failure')
    monkeypatch.setattr(backend.model, 'generate', fail)
    with pytest.raises(RuntimeError, match='decode failure'):
        backend.sample_actions((0, 1), 2, changed, rng())
    assert hooks == generation_model._forward_hooks
    assert torch.equal(global_rng, torch.get_rng_state())
    for n, p in lora_parameters(backend.model).items():
        assert torch.equal(original[n], p)


def test_feedback_generation_batch_matches_serial_hf_streams(backend, monkeypatch):
    """Real CPU HF multinomial, including EOS, padding and continuation tickets."""
    p = snapshot(lora_parameters(backend.model))
    prompts = ['0 1', '2 1 0', '1 0 2 1', '1']
    serial_rngs = [rng(i) for i in range(len(prompts))]
    batched_rngs = [rng(i) for i in range(len(prompts))]
    serial = [backend.sample_action(prompt, p, g) for prompt, g in zip(prompts, serial_rngs)]
    calls, generate = [], backend.model.generate
    def spy(**kwargs):
        calls.append(kwargs['input_ids'].shape[0])
        return generate(**kwargs)
    monkeypatch.setattr(backend.model, 'generate', spy)
    before = torch.get_rng_state().clone()
    batched = backend.sample_feedback_actions(list(zip(prompts, batched_rngs)), p)
    assert calls == [4]
    assert torch.equal(before, torch.get_rng_state())
    for a, b, sg, bg in zip(serial, batched, serial_rngs, batched_rngs):
        assert a.action_ids == b.action_ids and a.prompt_ids == b.prompt_ids
        assert torch.equal(sg.get_state(), bg.get_state())
        for key in ('rng_ticket', 'batch_seed', 'batch_rng_after'):
            assert a.generation_metadata[key] == b.generation_metadata[key]
        assert b.generation_token_logprobs == pytest.approx(a.generation_token_logprobs, abs=1e-6)
        checked(backend, b, p)


def test_feedback_generation_batch_fuses_original_start_rng_groups(backend, monkeypatch):
    p = snapshot(lora_parameters(backend.model))
    serial = [*backend.sample_actions('0 1', 2, p, rng(1)),
              *backend.sample_actions('2 0 1', 3, p, rng(2))]
    groups = [*backend.feedback_start_groups('0 1', 2, rng(1)),
              *backend.feedback_start_groups('2 0 1', 3, rng(2))]
    calls, generate = [], backend.model.generate
    def spy(**kwargs):
        calls.append(kwargs['input_ids'].shape[0])
        return generate(**kwargs)
    monkeypatch.setattr(backend.model, 'generate', spy)
    batched = backend.generate_feedback_groups(groups, p)
    assert calls == [5]
    for a, b in zip(serial, batched):
        assert a.action_ids == b.action_ids
        for key in ('rng_ticket', 'batch_seed', 'batch_rng_after'):
            assert a.generation_metadata[key] == b.generation_metadata[key]
        checked(backend, b, p)
