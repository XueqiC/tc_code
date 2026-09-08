from __future__ import annotations

import copy
import hashlib
import io
import json
import math
from pathlib import Path
import socket
from types import SimpleNamespace

import pytest
import torch
from peft import get_peft_model
from transformers import LlamaConfig, LlamaForCausalLM

import appworld_train as trainer
from bfas import pbsd_agent as p
from bfas.pbsd_evidence import CostIndex, digest, load_units, output_cost, select_units, task_start_units


class CharacterTokenizer:
    eos_token_id = 2
    bos_token_id = 1
    pad_token_id = 0

    def __call__(self, text, *, add_special_tokens=False, return_offsets_mapping=False):
        result = {'input_ids': [ord(c) % 256 + 3 for c in text]}
        if return_offsets_mapping:
            result['offset_mapping'] = [(i, i + 1) for i in range(len(text))]
        return result

    def apply_chat_template(self, messages, *, add_generation_prompt, tokenize):
        assert add_generation_prompt and not tokenize
        return ''.join(f'<{m["role"]}>{m["content"]}' for m in messages) + '<assistant>'


def row(task='task-a', cost=17, response='PRIVATE_EVIDENCE_A', **kw):
    return dict(task_id=task, teacher='deepseek-v4-pro', turn_index=0,
        prompt='normal task with tools', response=response, token_hint=99999,
        evidence_output_tokens=cost, verified=True, _pool_index=0, **kw)


def units(rows=None):
    return task_start_units(rows or [row()], Path('fixture.jsonl'), CostIndex())


@pytest.fixture
def engine(monkeypatch):
    for key in list(p.os.environ):
        if key.startswith('AW_'):
            monkeypatch.delenv(key)
    torch.set_num_threads(1)
    torch.manual_seed(7)
    model = get_peft_model(LlamaForCausalLM(LlamaConfig(vocab_size=260,
        hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=2048,
        eos_token_id=2, pad_token_id=0, bos_token_id=1, attention_dropout=0.)), trainer.lora_config())
    proxy = SimpleNamespace(prompt_token_ids=trainer.prompt_token_ids,
        prompt_truncation_config=trainer.prompt_truncation_config,
        completion_log_prob=trainer.completion_log_prob, MAX_RESPONSE_TOKENS=4)
    config = p.configuration({})
    tokenizer = CharacterTokenizer()
    contexts = p.prepare_contexts(proxy, tokenizer, units(), config)
    return p.Engine(proxy, model, tokenizer, contexts, config, audit=True)


@pytest.mark.parametrize('messages', [False, True])
def test_context_isolation_every_forward_and_mask(engine, messages):
    r = row()
    if messages:
        r['messages'] = [{'role': 'system', 'content': 'TOOLS'}, {'role': 'user', 'content': 'TASK'}]
    original = copy.deepcopy(r)
    context = p.prepare_contexts(engine.trainer, engine.tokenizer, units([r]), engine.config)[0]
    sample = engine.pair(context, 0)
    loss, record = engine.loss(sample)
    loss.backward()
    assert r == original
    assert context.student == tuple(trainer.prompt_token_ids(engine.tokenizer, context.unit.state))
    secret = engine.tokenizer(r['response'])['input_ids']
    def contains(sequence, needle):
        return any(list(sequence[i:i + len(needle)]) == needle for i in range(len(sequence)))
    assert not contains(context.student, secret)
    assert contains(context.teacher, secret)
    assert len(engine.forward_audit) >= 6
    for forward in engine.forward_audit:
        expected = record['teacher_prompt_sha256' if forward['teacher'] else 'student_prompt_sha256']
        assert forward['conditioning_sha256'] == expected
    for prompt in (context.student, context.teacher):
        ids, labels = p.encode_action(prompt, sample.positive.ids, 'cpu')
        assert (labels[:, :len(prompt)] == -100).all()
        assert labels[0, len(prompt):].tolist() == list(sample.positive.ids)
        assert ids.shape == labels.shape
    # Generated action tokens, never an authored demo target or synthetic EOS.
    assert sample.positive.ids != tuple(secret)
    assert len(sample.positive.ids) <= 4
    assert digest(context.student) != digest(context.teacher)


def test_hand_computed_four_log_probability_loss_and_detach():
    sp, tp, sn, tn = [torch.tensor(x, dtype=torch.float64, requires_grad=True)
                      for x in (-2., -1., -4., -2.5)]
    loss = p.preference_loss(sp, tp, sn, tn, beta=.1)
    expected = math.log1p(math.exp(-.1 * (-2. + 1. + 4. - 2.5)))
    assert float(loss.detach()) == pytest.approx(expected, abs=1e-14)
    loss.backward()
    assert sp.grad < 0 and sn.grad > 0
    assert tp.grad is None and tn.grad is None


def test_base_frozen_reference_stationary_after_lora_update(engine):
    context = engine.contexts[0]
    pair = engine.pair(context, 0)
    original = {n: v.detach().clone() for n, v in engine.model.named_parameters()}
    assert not pair.teacher_positive.requires_grad and pair.teacher_positive.grad_fn is None
    optimizer = torch.optim.AdamW(engine.trainable, lr=.01)
    loss, _ = engine.loss(pair)
    loss.backward()
    engine.check_gradients()
    assert any(v.grad is not None and v.grad.abs().sum() > 0 for v in engine.trainable)
    optimizer.step()
    for n, v in engine.model.named_parameters():
        if 'lora_' not in n:
            assert v.grad is None and torch.equal(v, original[n])
    assert any(not torch.equal(v, original[n]) for n, v in engine.model.named_parameters() if 'lora_' in n)
    with torch.no_grad():
        for sample, expected in [(pair.positive, pair.teacher_positive), (pair.negative, pair.teacher_negative)]:
            assert torch.equal(engine.score(context, sample.ids, True), expected)


@pytest.mark.parametrize('negative_interval,positive_interval', [(1, 0), (1, 1), (2, 2)])
def test_negative_refresh_current_student_and_positive_schedule(engine, negative_interval, positive_interval):
    engine.config.update(pbsd_negative_refresh_steps=negative_interval, pbsd_positive_refresh_steps=positive_interval)
    history, calls = [], []
    original_sample = engine.sample
    def sample(context, teacher, step):
        calls.append((teacher, step, [x.detach().clone() for x in engine.trainable]))
        return original_sample(context, teacher, step)
    engine.sample = sample
    optimizer = torch.optim.AdamW(engine.trainable, lr=.01)
    for step in range(3):
        pair = engine.pair(engine.contexts[0], step)
        assert engine.pair(engine.contexts[0], step) == pair
        optimizer.zero_grad()
        loss, _ = engine.loss(pair)
        loss.backward()
        optimizer.step()
        history.append(pair)
    for before, after in zip(history, history[1:]):
        for name, interval in [('negative', negative_interval), ('positive', positive_interval)]:
            a, b = getattr(before, name), getattr(after, name)
            expected_refresh = interval and b.step // interval != a.step // interval
            assert (a.generation_id != b.generation_id) == bool(expected_refresh)
    negative_calls = [x for x in calls if not x[0]]
    assert any(not torch.equal(a, b) for a, b in zip(negative_calls[0][2], negative_calls[1][2]))
    assert history[0].negative.ids != history[-1].negative.ids


def test_score_exact_generated_eos_and_truncation(engine):
    prompt = engine.contexts[0].student
    for action in ((5, 6, 2), (5, 6)):
        ids, labels = p.encode_action(prompt, action, 'cpu')
        with torch.no_grad():
            logits = engine.model(input_ids=ids).logits[0, len(prompt) - 1:-1].float()
            expected = logits.log_softmax(-1).gather(1, torch.tensor(action)[:, None]).sum()
            actual = trainer.completion_log_prob(engine.model, ids, labels)
        assert torch.allclose(actual, expected, atol=1e-6)
        assert ids[0, -len(action):].tolist() == list(action)


def test_budget_exact_per_unit_never_sampling_or_token_hint(engine):
    us = units([row(cost=17), row(task='task-b', cost=19, response='SECRET_B')])
    assert [u.cost for u in us] == [17, 19]
    assert select_units(us, 'full', 35, 0) == us[:1]
    assert select_units(us, 'full', 36, 0) == us
    with pytest.raises(ValueError, match='budget'):
        select_units(us, 'full', 16, 0)
    journal = io.StringIO()
    result = p.train(engine, seed=0, epochs=2, learning_rate=.001, batch_size=8, journal=journal)
    entries = [json.loads(s) for s in journal.getvalue().splitlines()]
    assert result['optimizer_steps'] == 2
    assert all(e['evidence_output_tokens'] == 17 and e['teacher_api_calls'] == 0 for e in entries)
    assert result['compute']['teacher_generation_calls'] == 2
    assert result['compute']['student_generation_calls'] == 2
    assert result['compute']['teacher_generation_output_tokens'] > 0
    assert len(entries[0]['pairs'][0].keys() & {'student_positive_logp', 'teacher_positive_logp',
        'student_negative_logp', 'teacher_negative_logp'}) == 4


def test_cost_resolution_sealed_exact_and_no_guesses():
    r = row()
    del r['evidence_output_tokens']
    record = dict(historical_response=dict(task_id=r['task_id'], response=r['response'], prompt=r['prompt'], turn_index=0),
                  usage={'output_tokens': 31}, cost=31, cost_confidence='exact')
    index = CostIndex([(record, 'sealed.json:1')])
    assert index.resolve(r, 'pool')[0] == 31
    for bad in [dict(r, task_id='different'), dict(r, response='different'), dict(r, prompt='different')]:
        with pytest.raises(ValueError, match='no exact cost'):
            index.resolve(bad, 'pool')
    with pytest.raises(ValueError, match='no exact cost'):
        CostIndex().resolve(r, 'pool')
    with pytest.raises(ValueError, match='conflicting'):
        output_cost(dict(output_tokens=10, usage={'completion_tokens': 20}))
    with pytest.raises(ValueError, match='not estimates'):
        output_cost(dict(cost=50, cost_confidence='estimated'))
    assert output_cost(dict(output_token_count=[10, 20]))[0] == 30


def test_evidence_task_start_matching_and_no_cross_task_context():
    rows = [row(), dict(row(response='LATER_SECRET'), turn_index=1,
        messages=[{'role': 'assistant', 'content': 'earlier action'}]), row(task='other', response='OTHER_SECRET')]
    us = units(rows)
    assert us[0].cost == 17 and len(us[0].rows) == 1
    assert 'PRIVATE_EVIDENCE_A' in p.evidence_context(us[0])
    assert 'LATER_SECRET' not in p.evidence_context(us[0])
    assert 'OTHER_SECRET' not in p.evidence_context(us[0])
    with pytest.raises(ValueError, match='not state-matched'):
        units([row(), dict(row(response='second demo'), prompt='other state')])
    with pytest.raises(ValueError, match='task-start'):
        units([dict(row(), messages=[{'role': 'assistant', 'content': 'previous'}])])
    with pytest.raises(ValueError, match='verified'):
        units([dict(row(), verified=False)])
    with pytest.raises(ValueError, match='DeepSeek'):
        units([dict(row(), teacher='other model')])


def test_no_silent_context_truncation(engine, monkeypatch):
    monkeypatch.setenv('AW_MAX_PROMPT_TOKENS', '3')
    with pytest.raises(ValueError, match='complete state'):
        p.prepare_contexts(engine.trainer, engine.tokenizer, units(), engine.config)
    monkeypatch.delenv('AW_MAX_PROMPT_TOKENS')
    with pytest.raises(ValueError, match='no evidence truncation'):
        p.prepare_contexts(engine.trainer, engine.tokenizer, units(), dict(engine.config, pbsd_max_context_tokens=10))


def test_network_denied_and_guard_restored():
    original = socket.getaddrinfo
    with pytest.raises(RuntimeError, match='forbids network'):
        with p.offline_only():
            socket.getaddrinfo('example.com', 443)
    assert socket.getaddrinfo is original
    with pytest.raises(RuntimeError, match='swallowed'):
        with p.offline_only():
            try:
                socket.create_connection(('127.0.0.1', 80))
            except RuntimeError:
                pass


def test_real_purchased_pool_read_only_costs():
    path = trainer.ROOT / 'data/bfcl_sft/pool_bfcl_ds_sft.jsonl'
    if not path.is_file():
        pytest.skip('read-only purchased evidence absent')
    before = path.read_bytes()
    us = load_units(trainer, path)
    assert len(us) == 23
    assert sum(u.cost for u in us) == 4768
    assert path.read_bytes() == before


def test_legacy_trainer_bytes_unchanged_except_opt_in_dispatch():
    source = Path(trainer.__file__).read_text()
    source = source.replace('    if raw_mode == "pbsd_agent":\n        from bfas.pbsd_agent import configuration\n        return configuration()\n', '', 1)
    source = source.replace('    if distillation is not None and distillation["mode"] == "pbsd_agent":\n        import sys\n        from bfas.pbsd_agent import run\n        run(args, trainer=sys.modules[__name__])\n        return\n', '', 1)
    # Frozen baseline at task start, covering every old arm and helper byte.
    assert hashlib.sha256(source.encode()).hexdigest() == 'e783c5e56082c44602dcf235de65ae88384a4b1056364ac58bd589b8f2ff1162'


class LegacyFixtureModel(torch.nn.Module):
    """Deterministic nonuniform logits, preserving old trainer control flow."""
    def __init__(self):
        super().__init__()
        self.lora_fixture = torch.nn.Parameter(torch.linspace(-.5, .5, 260, dtype=torch.float64))
        self.config = SimpleNamespace(use_cache=False)
        self.adapter_enabled = True

    def forward(self, input_ids, labels=None):
        scale = (input_ids.double() % 7 + 1).unsqueeze(-1) / 7
        values = self.lora_fixture if self.adapter_enabled else self.lora_fixture.detach() * .9
        logits = scale * values[None, None, :]
        loss = None if labels is None else torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, 260), labels[:, 1:].reshape(-1), ignore_index=-100)
        return SimpleNamespace(logits=logits, loss=loss)

    @p.contextmanager
    def disable_adapter(self):
        self.adapter_enabled = False
        try:
            yield
        finally:
            self.adapter_enabled = True


# SHA256 of little-endian float64 scalars passed to backward, through the
# actual pre-existing training loops (including SFT + frozen-reference DDPO).
LEGACY_LOSS_HASHES = {
    'sft': 'f4d7951917c5c4afd42970a1adb39870e69f60dd135676333187523113951360',
    'sad': 'ed0e56f7955fa94c25e6199b3a6867e5efb7d0c9a497a67be793824f8935129d',
    'ddpo': 'f2c5808027a75a4f4784c351f62b269bf1667de3fcaf497764650ea5b27c08c2',
    'pbsd': 'aa9a41ca6e083c55ff31d4c6fd4e8308e809972bb1bbb10b16ebb094ab00e78d',
    'agentkd': 'c80dd7efe2a2d26552f6bba9e9e72a65517cdf1414cc16d319be233f07978950',
    'smartad': '43003ec2ae8fd4396ae3e0506f9df3ef427db48b37a04320ddef20cb72da35f2',
}


@pytest.mark.parametrize('mode', list(LEGACY_LOSS_HASHES))
def test_legacy_loss_fixture_hash(monkeypatch, mode):
    import struct
    for name in list(p.os.environ):
        if name.startswith('AW_'):
            monkeypatch.delenv(name)
    monkeypatch.setenv('AW_EPOCHS', '1')
    monkeypatch.setenv('AW_LR', '0.001')
    if mode not in {'sft', 'smartad'}:
        monkeypatch.setenv('AW_DISTILL', mode)
    monkeypatch.setattr(trainer, 'GRADIENT_ACCUMULATION', 1)
    original_encode = trainer.encode
    monkeypatch.setattr(trainer, 'encode', lambda tokenizer, row: original_encode(tokenizer, row, device='cpu'))
    losses = []
    backward = torch.Tensor.backward
    def capture(value, *args, **kwargs):
        losses.append(float(value.detach()))
        return backward(value, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, 'backward', capture)
    model = LegacyFixtureModel()
    tokenizer = CharacterTokenizer()
    rows = [dict(row(response='Reason.\n```python\nx=1\n```'), _rejected='No.', _thought='Thought: yes.\n')]
    cfg = trainer.distillation_config()
    if mode == 'ddpo':
        trainer.cache_ddpo_reference_log_probs(model, tokenizer, rows)
    trainer.train_student(model, tokenizer, rows, seed=0, smartad=mode == 'smartad', distillation=cfg)
    if mode == 'ddpo':
        trainer.train_ddpo(model, tokenizer, rows, 0, 1, 5e-6, .1)
    hashed = hashlib.sha256(b''.join(struct.pack('<d', x) for x in losses)).hexdigest()
    assert hashed == LEGACY_LOSS_HASHES[mode], (mode, losses, hashed)


def test_checkpointed_backward_keeps_adapter_active(engine):
    engine.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    engine.model.enable_input_require_grads()
    journal = io.StringIO()
    result = p.train(engine, seed=0, epochs=2, learning_rate=.001, batch_size=4, journal=journal)
    assert result['optimizer_steps'] == 2
    assert all(x.requires_grad for x in engine.trainable)


def test_gpu_check_logic_on_four_tiny_tasks(engine):
    from tools.pbsd_agent_check import check
    rows = [dict(row(task=f'task-{i}', cost=10+i, response=f'PRIVATE_{i}'), _pool_index=i) for i in range(4)]
    engine.contexts = p.prepare_contexts(engine.trainer, engine.tokenizer, units(rows), engine.config)
    engine.config['pbsd_positive_refresh_steps'] = 0
    report = check(engine, learning_rate=.001, journal=io.StringIO())
    assert report['context_isolation'] and report['teacher_stop_gradient']
    assert report['teacher_recompute_bitwise_equal']
    assert report['evidence_output_tokens'] == 46
    assert report['compute']['teacher_generation_calls'] == 4
    assert report['compute']['student_generation_calls'] == 8
    assert report['audited_forwards'] > 32
    # This also tests the check's failure reporting; stochastic loss monotonicity
    # is deliberately NOT guaranteed by PBSD or manufactured by resampling.
    assert report['passed'] == (not report['failures'])


def test_run_dispatch_and_merged_save_convention(engine, tmp_path, monkeypatch):
    pool = tmp_path / 'pool.jsonl'
    pool.write_text(json.dumps(row()) + '\n')
    monkeypatch.setenv('AW_DISTILL', 'pbsd_agent')
    monkeypatch.setenv('AW_EPOCHS', '1')
    monkeypatch.setattr(trainer, 'POOL_PATH', pool)
    monkeypatch.setattr(trainer, 'OUTPUT_ROOT', tmp_path / 'outputs')
    # Only satisfy the CLI gate; the actual tiny model and optimizer stay CPU.
    cuda_gate = iter([True])
    monkeypatch.setattr(trainer.torch.cuda, 'is_available', lambda: next(cuda_gate, False))
    monkeypatch.setattr(trainer.torch.cuda, 'is_bf16_supported', lambda: True)
    monkeypatch.setattr(p, 'load_local_model', lambda *args: (engine.model, engine.tokenizer))
    monkeypatch.setattr(trainer, 'MAX_RESPONSE_TOKENS', 4)
    monkeypatch.setattr(engine.tokenizer, 'save_pretrained', lambda path: Path(path, 'tokenizer.json').write_text('{}'), raising=False)
    trainer.main(['--selection', 'full', '--budget', '17', '--seed', '0', '--student', 'tiny', '--tag', 'run'])
    output = tmp_path / 'outputs/run'
    manifest = json.loads((output / 'selection_manifest.json').read_text())
    assert manifest['evidence_output_tokens'] == 17
    assert manifest['optimizer_steps'] == 1 and manifest['status'] == 'complete'
    assert (output / 'adapter/model.safetensors').is_file()
    assert (output / 'adapter/tokenizer.json').is_file()
    assert (output / 'pbsd_agent_journal.jsonl').is_file()
