"""CPU-only RTD integration: paid episodes, cohort RNG, and startup contracts."""
from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
import socket
import sys
from types import SimpleNamespace

import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT)]
from bfas import hotpotqa as hp
from bfas.adapters.hotpotqa import HotpotQAAdapter
from bfas.rtd import cli, preflight
from bfas.rtd.bank_build import validate_state_certificate
from bfas.rtd.benchmarks.hotpotqa_bank import build_hotpotqa_bank
from bfas.rtd.benchmarks.hotpotqa_caps import action_limit
from bfas.rtd.benchmarks.hotpotqa_evaluation import validate_records
from bfas.rtd.benchmarks.hotpotqa_rollout import HotpotQAFeedbackContext
from bfas.rtd.benchmarks.hotpotqa_support import HotpotQASupport, parent_hash, reset_state
from bfas.rtd.benchmarks.registry import get_benchmark
from bfas.rtd.feedback_rng import FeedbackRNG, feedback_rng_identity, guard_feedback_rng_comparison
from bfas.rtd.generation_batch import feedback_rollout_tasks
from bfas.rtd.persistence import digest
from bfas.rtd.return_gradient import ActionTrace


class Wiki:
    closed = 0

    def reset(self):
        self.queries, self.cursor = [], 0

    def search(self, argument):
        self.queries.append({'query': argument, 'sha256': 'stub'})
        return 'A locally cached article.'

    def lookup(self, argument):
        self.cursor += 1
        return f'Result {self.cursor}: SYNTHETIC_GOLD.'

    def close(self):
        type(self).closed += 1


@pytest.fixture(autouse=True)
def no_network_gpu(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('HotpotQA RTD tests cannot use a model, GPU, network, or teacher')
    monkeypatch.setattr(socket.socket, 'connect', forbidden)
    from transformers import AutoModelForCausalLM
    monkeypatch.setattr(AutoModelForCausalLM, 'from_pretrained', forbidden)
    for name in ('is_available', 'device_count', 'set_device', '_lazy_init'):
        monkeypatch.setattr(torch.cuda, name, forbidden)
    monkeypatch.setattr(hp.Wikipedia, '_fetch', forbidden)


@pytest.fixture
def synthetic(tmp_path, monkeypatch):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast
    from bfas.ledger import demo_payload
    from bfas.adapter import Demo, Turn
    tok = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel(
        {'[UNK]': 0, '<bos>': 1, '<eos>': 2, '<turn|>': 3}, unk_token='[UNK]')),
        unk_token='[UNK]', bos_token='<bos>', eos_token='<eos>')
    tok.chat_template = "{{ bos_token }}{% for m in messages %}<|turn>{{ m['role'] }}\n{{ m['content'] }}<turn|>{% endfor %}{% if add_generation_prompt %}<|turn>model\n{% endif %}"
    model = tmp_path/'model'; tok.save_pretrained(model)
    questions = [dict(_id=tid, question=f'Synthetic question {i}?', answer='SYNTHETIC_GOLD', type='bridge')
                 for i, tid in enumerate(hp.load_manifest('train')['ids'])]
    monkeypatch.setattr(hp, 'load_questions', lambda split, **kw: questions)
    adapter = HotpotQAAdapter(offline=True); adapter._tokenizer = tok
    rows = []
    # Tiny paid pool, but every one of the real frozen 200 IDs has a reset.
    for question in questions[:2]:
        turns, replies = [], iter(['Action 1: search[article]', 'Action 2: finish[SYNTHETIC_GOLD]'])
        def generate(messages, stop, temperature):
            reply = next(replies)
            turns.append(Turn('portable archive', reply, deepcopy(messages)))
            return reply
        record = hp.run_episode(question, Wiki(), generate)
        demo = demo_payload(Demo(question['_id'], tuple(turns),
            f"Question: {question['question']}\n"+record['transcript'], record))
        rows.append(dict(task_id=question['_id'], teacher='openai/gpt-5.6-luna', attempt_index=0,
            temperature=0., verified=True, tokens_spent=11, purpose='teacher', timestamp='fixture',
            prompt_version=hp.PROMPT_VERSION, usage_status='reported',
            usage=dict(prompt_tokens=100, completion_tokens=11, cached_tokens=20), demo=demo))
    rows.append(dict(rows[0], task_id=questions[2]['_id'], verified=False, tokens_spent=7,
                     usage=dict(prompt_tokens=30, completion_tokens=7, cached_tokens=0)))
    rows[-1].pop('demo')
    pool, ledger = tmp_path/'demos.json', tmp_path/'teacher_ledger.jsonl'
    # Exercise a live pool whose demos.json lags the successful ledger.
    pool.write_text(json.dumps(dict(schema_version=1, prompt_version=hp.PROMPT_VERSION,
        teacher='openai/gpt-5.6-luna', demos={rows[0]['task_id']: rows[0]['demo']})))
    ledger.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    config = cli.load_config(ROOT/'configs/rtd/v1_1_hotpotqa_luna.yaml')
    bank = tmp_path/'bank'
    config.update(student=str(model), replay_bank_path=str(bank), support_manifest=str(bank/'public/support.json'))
    result = build_hotpotqa_bank(ROOT, bank, pool=pool, ledger=ledger, config=config, tokenizer=tok, wiki_factory=Wiki)
    return SimpleNamespace(root=tmp_path, config=config, bank=bank, result=result, tokenizer=tok,
                           questions=questions, pool=pool, ledger=ledger, rows=rows, adapter=adapter)


def test_hotpotqa_bank_build_exact_costs_folds_and_native_rendering(synthetic):
    c = synthetic
    cert = validate_state_certificate(c.bank, benchmark='hotpotqa', student=c.config['student'])
    assert cert['core']['budget_denominator'] == 22
    assert c.result['available_packages'] == 2
    assert c.result['teacher_accounting']['historical_output_tokens'] == 29
    assert c.result['teacher_accounting']['failed_output_tokens'] == 7
    rows = json.loads((c.bank/'public/requests.json').read_text())
    assert len(rows) == 3 and sum(r['unavailable_reason'] is not None for r in rows) == 1
    assert cert['core']['class_caps'] == {'hotpotqa_demo_episode': 16}
    for r in rows:
        payload = json.loads((c.bank/'sealed'/f"{r['spec']['query_id']}.json").read_text())
        assert payload['cost'] == payload['historical_response']['usage']['completion_tokens']
        assert payload['cost_confidence'] == 'exact' and r['dependencies'] == []
        assert len(payload['behaviors']) == (0 if r['unavailable_reason'] else 2)
        for behavior in payload['behaviors']:
            assert behavior['state']['prompt'].startswith('<bos>')
            assert behavior['state']['prompt'].endswith('<|turn>model\n')
            assert 'answer' not in json.loads(behavior['state']['task_json'])
    support = HotpotQASupport(ROOT, c.config)
    assert len(support.states) == 200
    assert {p['fold'] for p in support.manifest['parents']} == {0, 1}
    assert 'SYNTHETIC_GOLD' not in (c.bank/'public/reset_states.json').read_text()
    audit = json.loads((c.bank/'sealed/audit.json').read_text())
    assert len(audit['ledger_demos_not_yet_in_pool']) == 1
    assert audit['snapshot']['attempts.jsonl'] is None


@pytest.mark.parametrize('pool_attempt', ['earliest', 'later'])
def test_hotpotqa_bank_build_multiple_successes_charge_all_attempts(synthetic, pool_attempt):
    from bfas.adapter import Demo, Turn
    from bfas.ledger import demo_payload
    c = synthetic; rows = deepcopy(c.rows)
    first = rows[0]; first['attempt_index'] = 1
    later = deepcopy(first)
    later.update(attempt_index=4, tokens_spent=13)
    later['usage']['completion_tokens'] = 13
    turns = []
    replies = iter(['Action 1: search[later article]', 'Action 2: finish[SYNTHETIC_GOLD]'])
    def generate(messages, stop, temperature):
        reply = next(replies)
        turns.append(Turn('portable archive', reply, deepcopy(messages)))
        return reply
    record = hp.run_episode(c.questions[0], Wiki(), generate)
    later['demo'] = demo_payload(Demo(first['task_id'], tuple(turns),
        f"Question: {c.questions[0]['question']}\n"+record['transcript'], record))
    failed = dict(deepcopy(rows[2]), task_id=first['task_id'])
    # Selection follows attempt index even with out-of-order, gapped retries.
    rows = [later, *rows, failed]
    archive = json.loads(c.pool.read_text())
    archive['demos'][first['task_id']] = (first if pool_attempt == 'earliest' else later)['demo']
    c.pool.write_text(json.dumps(archive))
    c.ledger.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    bank = c.root/'retry-bank'
    result = build_hotpotqa_bank(ROOT, bank, pool=c.pool, ledger=c.ledger,
        config=c.config, tokenizer=c.tokenizer, wiki_factory=Wiki)
    requests = json.loads((bank/'public/requests.json').read_text())
    payloads = [json.loads((bank/'sealed'/f"{r['spec']['query_id']}.json").read_text()) for r in requests]
    payload = next(p for p in payloads if p['provenance']['task_id'] == first['task_id'])
    assert len(requests) == 3 and result['available_packages'] == 2
    assert payload['cost'] == 31 and payload['usage']['completion_tokens'] == 31
    assert payload['provenance']['attempt_index'] == 1
    assert payload['provenance']['ledger_rows'] == [4, 1, 0]
    assert payload['provenance']['charged_attempt_indices'] == [0, 1, 4]
    assert payload['historical_response'] == first
    assert payload['historical_attempts'] == [failed, first, later]
    assert [b['text'] for b in payload['behaviors']] == [t['target'] for t in first['demo']['turns']]
    assert sum(p['cost'] for p in payloads) == sum(r['tokens_spent'] for r in rows) == 49
    assert result['budget_denominator'] == 42
    assert result['available_cost_by_confidence'] == {'exact': 42}
    assert result['teacher_accounting']['verified_episodes'] == 3
    assert result['teacher_accounting']['verified_tasks'] == 2
    assert result['teacher_accounting']['later_verified_attempts'] == 1
    assert validate_state_certificate(bank)['core']['class_caps'] == {'hotpotqa_demo_episode': 32}
    assert (bank/'public/support.json').read_bytes() == (c.bank/'public/support.json').read_bytes()


@pytest.mark.parametrize('estimated_attempt', ['selected', 'failed', 'later_success'])
def test_hotpotqa_bank_build_estimated_usage_charged_and_propagated(synthetic, estimated_attempt):
    c = synthetic; rows = deepcopy(c.rows)
    estimated = rows[0]
    if estimated_attempt != 'selected':
        estimated = deepcopy(rows[0] if estimated_attempt == 'later_success' else rows[2])
        estimated.update(task_id=rows[0]['task_id'], attempt_index=1)
        rows.append(estimated)
    estimated['usage_status'] = 'estimated'
    c.ledger.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (c.root/'attempts.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    calls = []
    for i, r in enumerate(rows):
        call = dict(id=i, task_id=r['task_id'], attempt_index=r['attempt_index'], usage=r['usage'])
        calls.extend([dict(call, status='reserved'),
                      dict(call, status='estimated' if r is estimated else 'reported')])
    (c.root/'usage.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in calls))
    bank = c.root/'estimated-bank'
    result = build_hotpotqa_bank(ROOT, bank, pool=c.pool, ledger=c.ledger,
        config=c.config, tokenizer=c.tokenizer, wiki_factory=Wiki)
    requests = json.loads((bank/'public/requests.json').read_text())
    request = next(r for r in requests if r['spec']['cost_confidence'] == 'estimated')
    payload = json.loads((bank/'sealed'/f"{request['spec']['query_id']}.json").read_text())
    cost = sum(r['tokens_spent'] for r in rows if r['task_id'] == rows[0]['task_id'])
    assert request['unavailable_reason'] is None and payload['verification']['verified']
    assert payload['provenance']['attempt_index'] == 0
    assert payload['cost'] == payload['usage']['completion_tokens'] == cost
    assert payload['cost_confidence'] == 'estimated'
    assert result['available_cost_by_confidence'] == {'exact': 11, 'estimated': cost}
    assert result['teacher_accounting']['cost_confidence'] == 'estimated'
    assert result['teacher_accounting']['estimated_output_tokens'] == estimated['tokens_spent']
    assert result['teacher_accounting']['exact_output_tokens'] == sum(r['tokens_spent'] for r in rows if r is not estimated)
    assert result['teacher_accounting']['historical_output_tokens'] == cost+18
    assert result['budget_denominator'] == cost+11
    cert = validate_state_certificate(bank)
    assert cert['core']['teacher_accounting'] == result['teacher_accounting']
    assert (bank/'public/support.json').read_bytes() == (c.bank/'public/support.json').read_bytes()


@pytest.mark.parametrize('change', ['cost', 'context', 'target', 'pool', 'usage',
                                   'missing_task', 'unknown_task', 'negative_tokens', 'negative_usage'])
def test_hotpotqa_bank_build_rejects_corrupt_evidence(synthetic, change):
    c = synthetic; rows = deepcopy(c.rows)
    if change == 'cost': rows[0]['tokens_spent'] += 1
    elif change == 'context': rows[0]['demo']['turns'][0]['context'][0]['content'] += 'gold leak'
    elif change == 'target': rows[1]['demo']['turns'][-1]['target'] = 'finish[wrong]'
    elif change == 'missing_task': rows[2].pop('task_id')
    elif change == 'unknown_task': rows[2]['task_id'] = 'outside-support'
    elif change == 'negative_tokens': rows[2]['tokens_spent'] = rows[2]['usage']['completion_tokens'] = -1
    elif change == 'negative_usage': rows[2]['usage']['prompt_tokens'] = -1
    elif change == 'pool':
        value = json.loads(c.pool.read_text());value['teacher'] = 'wrong';c.pool.write_text(json.dumps(value))
    else:
        (c.root/'usage.jsonl').write_text(json.dumps(dict(id=0, task_id=rows[0]['task_id'], attempt_index=0,
            status='reported', usage=dict(prompt_tokens=1, completion_tokens=1, cached_tokens=0)))+'\n')
    c.ledger.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    with pytest.raises(ValueError):
        build_hotpotqa_bank(ROOT, c.root/'bad-bank', pool=c.pool, ledger=c.ledger,
                           config=c.config, tokenizer=c.tokenizer, wiki_factory=Wiki)
    assert not (c.root/'bad-bank').exists()


class Policy:
    backend_id = 'cpu-fake-policy'
    generation_batch = True
    max_action_tokens = 512
    max_context_tokens = 32768
    action_caps = {'agent_action': 100}

    def __init__(self, tokenizer):
        self.tokenizer, self.draws = tokenizer, []

    def identity(self, parameters):
        return 'policy'

    def action_limit(self, category):
        return action_limit(self, category)

    def sample_action(self, prompt, parameters, generator, **settings):
        assert settings == dict(temperature=1., top_p=1.)
        return self.sample_feedback_actions([(prompt, generator)], parameters)[0]

    def sample_feedback_actions(self, requests, parameters, *, on_batch=None, **settings):
        assert self.max_action_tokens == 100
        result = []
        for prompt, generator in requests:
            ticket = int(torch.randint(1, 10000, (1,), generator=generator))
            tail = prompt.rsplit('Question: ', 1)[-1]
            if 'Action 1:' not in tail and 'Need an action' not in tail:
                text = 'Need an action'  # action-only retry must preserve possible reward=1
            elif 'Observation 1:' not in tail:
                text = 'search[article]'
            elif 'Observation 2:' not in tail:
                text = 'Action 2: lookup[answer]'
            else:
                text = "Action 3: finish[SYNTHETIC_GOLD]"
            ids = tuple(self.tokenizer.encode(prompt, add_special_tokens=False))
            result.append(ActionTrace(ids, (ticket+10, 2), 2, text, -1., self.backend_id, 'policy',
                (-.5, -.5), dict(feedback_rng_version=2, ticket=ticket)))
        if on_batch:
            on_batch(range(len(result)), result)
        self.draws.append(len(result))
        return result


@pytest.mark.parametrize('cohort_limit', ['1', '2', '8'])
def test_hotpotqa_rollout_serial_lockstep_equal_with_v2_rng(synthetic, monkeypatch, cohort_limit):
    c = synthetic; support = HotpotQASupport(ROOT, c.config)
    backend = Policy(c.tokenizer); parameters = {'w': torch.zeros(1)}
    context = HotpotQAFeedbackContext(support, 1, c.adapter._render, None, Wiki, lambda _: c.questions)
    parents = [p for p in support.parents if int(p, 16) % 2 == 1][:2]
    tasks = [(p, 3) for p in parents]
    shared_rng = torch.Generator().manual_seed(123); initial = shared_rng.get_state().clone()
    rng = FeedbackRNG(0, 1, 1, 'reference')
    monkeypatch.setenv('BFAS_FEEDBACK_LOCKSTEP_EPISODES', cohort_limit)
    monkeypatch.setenv('BFAS_FEEDBACK_LOCKSTEP', '0')
    serial = list(feedback_rollout_tasks(support, tasks, backend, parameters, shared_rng, context, rng_context=rng))
    monkeypatch.setenv('BFAS_FEEDBACK_LOCKSTEP', '1')
    batch = list(feedback_rollout_tasks(support, tasks, backend, parameters, shared_rng, context, rng_context=rng))
    assert serial == batch
    assert torch.equal(initial, shared_rng.get_state())
    assert all(len(r.actions) == 4 for _, r in batch)  # fallback + search + lookup + finish
    assert backend.max_action_tokens == 512
    assert any(r.reward == 1 for _, r in batch)
    reordered = list(feedback_rollout_tasks(support, list(reversed(tasks)), backend, parameters,
        shared_rng, context, rng_context=rng))
    assert sorted(reordered, key=lambda x: x[0]) == sorted(serial, key=lambda x: x[0])


def test_hotpotqa_feedback_guards_and_cache_miss(synthetic):
    c = synthetic; support = HotpotQASupport(ROOT, c.config); backend = Policy(c.tokenizer)
    parameters = {'w': torch.zeros(1)}; generator = torch.Generator().manual_seed(1)
    context = HotpotQAFeedbackContext(support, 1, c.adapter._render, None, Wiki, lambda _: c.questions)
    inner = next(p for p in support.parents if int(p, 16) % 2 == 0)
    with pytest.raises(PermissionError): support.feedback(inner, backend, parameters, generator, context)
    parent = next(p for p in support.parents if int(p, 16) % 2 == 1)
    class Missing(Wiki):
        def search(self, argument): raise hp.OfflineCacheMiss('missing local snapshot')
    with pytest.raises(hp.OfflineCacheMiss):
        support.feedback(parent, backend, parameters, generator, replace(context, wiki_factory=Missing))
    assert backend.max_action_tokens == 512
    assert feedback_rng_identity(c.config) == {'feedback_rng_version': 2}
    with pytest.raises(ValueError, match='RNG versions differ'):
        guard_feedback_rng_comparison([dict(config=c.config), dict(config=c.config, feedback_rng_version=2)])


def test_hotpotqa_greedy_diagnostics_use_zero_temperature_and_both_folds(synthetic):
    c = synthetic; support = HotpotQASupport(ROOT, c.config)
    class Greedy(Policy):
        diagnostic_only = True
        def sample_feedback_actions(self, requests, parameters, *, on_batch=None, **kw):
            results = [ActionTrace(tuple(self.tokenizer.encode(p, add_special_tokens=False)),
                (10, 2), 2, 'finish[SYNTHETIC_GOLD]', -1., self.backend_id, 'policy', (-.5, -.5),
                dict(temperature=0., diagnostic_only=True)) for p, _ in requests]
            if on_batch: on_batch(range(len(results)), results)
            return results
    backend = Greedy(c.tokenizer); parameters = {'w': torch.zeros(1)}
    generator = torch.Generator().manual_seed(1); before = generator.get_state().clone()
    journal = SimpleNamespace(append=lambda *a, **kw: events.append(kw)); events = []
    context = HotpotQAFeedbackContext(support, 1, c.adapter._render, journal, Wiki, lambda _: c.questions)
    parents = list(support.parents)[:5]
    results = list(support.diagnostic_batch(parents, backend, parameters, generator, context))
    assert [p for p, _ in results] == parents and all(r.reward == 1 for _, r in results)
    assert all(e['temperature'] == 0 and e['diagnostic_only'] for e in events)
    assert torch.equal(before, generator.get_state())


@pytest.mark.parametrize('arm,name', [('V0', 'v1_1_hotpotqa_luna'), ('D3', 'unified_hotpotqa_gemma4_luna')])
def test_hotpotqa_preflight_synthetic_bank(synthetic, monkeypatch, arm, name):
    c = synthetic
    config = cli.load_config(ROOT/f'configs/rtd/{name}.yaml', arm=arm)
    config.update(student=c.config['student'], replay_bank_path=str(c.bank), support_manifest=c.config['support_manifest'])
    # Only environment/model identity inputs are fixture data; use real bank,
    # registry, support, tokenizer, renderer, ledger and executor constructors.
    monkeypatch.setattr(cli, 'evaluation_harness_identity', lambda *a: {'fixture': 'hotpotqa', 'expected': {}})
    monkeypatch.setattr(cli, 'data_identity', lambda *a: 'synthetic-data')
    result = preflight.preflight_config(config, arm)
    assert result['status'] == 'OK'
    assert result['rendered_states'] == result['support_states'] == 200
    assert result['available_packages'] == 2 and result['ledger_spent'] == 0


def test_hotpotqa_registry_and_frozen_p1_configs():
    for hot, alf in [('v1_1_hotpotqa_luna', 'v1_1_alfworld_luna'),
                     ('unified_hotpotqa_gemma4_luna', 'unified_alfworld_gemma4_luna'),
                     ('unified_hotpotqa_gemma4_luna_d0', 'unified_alfworld_gemma4_d0_luna')]:
        config = cli.load_config(ROOT/f'configs/rtd/{hot}.yaml')
        reference = cli.load_config(ROOT/f'configs/rtd/{alf}.yaml')
        for key in ('slots_per_step', 'exposure_slots_per_window', 'max_new_packages_per_window',
                    'rounds', 'memory_peak_budget_gb',
                    'training_seed', 'score_consistency_tolerance', 'rollouts_per_meta_task'):
            assert config[key] == reference[key]
        assert config['budget_checkpoints_tokens'] == [15000, 30000]
        assert 'budget_checkpoints_bank_fraction' not in config
        assert reference['budget_checkpoints_bank_fraction'] == [.1, .25]
        assert get_benchmark(config).support_protocol is HotpotQASupport
        assert config['support_parent_tasks_m'] == 200
        assert config['max_action_tokens_by_benchmark'] == {'hotpotqa': {'agent_action': 100}}


def test_hotpotqa_official_em_f1_validation(tmp_path):
    question = dict(_id='dev', question='Synthetic?', answer='alpha beta', type='bridge')
    record = hp.run_episode(question, Wiki(), lambda *a: 'finish[alpha]')
    cfg = dict(temperature=0., max_steps=7, max_tokens=100, task_ids=['dev'],
               prompt_version=hp.PROMPT_VERSION, wiki_version=hp.WIKI_VERSION)
    metrics = hp.compute_metrics([record], cfg) | dict(complete=True, requested_n=1, offline=True)
    (tmp_path/'records.jsonl').write_text(json.dumps(record)+'\n')
    result = validate_records(tmp_path, metrics, {'task_ids': ['dev']})
    assert result['em'] == 0 and result['f1'] == pytest.approx(2/3)
    for wrong in (metrics | {'em': 1}, metrics | {'complete': False}, metrics | {'f1': 0}):
        with pytest.raises(ValueError): validate_records(tmp_path, wrong, {'task_ids': ['dev']})


def test_hotpotqa_round_end_adapter_campaign_binds_em_f1_and_spend(tmp_path, monkeypatch):
    from contextlib import contextmanager
    from bfas.rtd.benchmarks import webshop_evaluation as campaign
    from bfas.rtd import identity, hardware, evaluation
    from bfas.rtd.persistence import tree_hash
    from bfas import run
    import bfas.adapters.hotpotqa as adapter_module
    root, directory, model = tmp_path, tmp_path/'run', tmp_path/'base'
    directory.mkdir(); model.mkdir(); (model/'config.json').write_text('{}')
    questions = [dict(_id=f'dev-{i}', question=f'Synthetic {i}?', answer='alpha beta', type='bridge') for i in range(500)]
    expected = dict(task_ids=[q['_id'] for q in questions], questions_hash=digest(questions))
    config = dict(benchmark='hotpotqa', student='configured-student')
    meta = dict(round=1, actual_spend=34, authorized_budget=100)
    manifest = dict(config=config, config_hash=digest(config), bank_path='bank', data_hash='data',
        model_path=str(model), base_checkpoint_hash=tree_hash(model), tokenizer_hash='tok', arm='V0')
    (directory/'manifest.json').write_text(json.dumps(manifest))
    monkeypatch.setattr(identity, 'verified_checkpoint', lambda *a: meta)
    monkeypatch.setattr(identity, 'guard_harness', lambda *a: dict(evaluation_harness={'expected': expected}, harness_hash='h'))
    monkeypatch.setattr(identity, 'record_code_drift', lambda *a, **kw: {})
    monkeypatch.setattr(cli, 'data_identity', lambda *a: 'data')
    monkeypatch.setattr(cli, 'hardware_identity', lambda: {'hard': {'gpu': 'stub'}})
    monkeypatch.setattr(hardware, 'guard_hardware', lambda *a, **kw: {'hard': {'gpu': 'stub'}})
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    monkeypatch.setattr(hp, 'load_questions', lambda split: questions)
    def export(manifest, checkpoint, out):
        out.mkdir(); (out/'model.safetensors').write_text('stub export')
    monkeypatch.setattr(evaluation, '_flatten_adapter', export)
    adapter = HotpotQAAdapter(offline=True)
    monkeypatch.setattr(adapter, 'prepare_renderer', lambda _: None)
    monkeypatch.setattr(adapter_module, 'HotpotQAAdapter', lambda **kw: adapter)
    @contextmanager
    def serve(*args):
        assert args[0] is adapter
        yield
    monkeypatch.setattr(run, 'serving_lane', serve)
    @contextmanager
    def client(*args):
        yield object()
    monkeypatch.setattr(hp, 'make_client', client)
    monkeypatch.setattr(hp, 'student_generator', lambda *a: lambda *args: 'finish[alpha]')
    monkeypatch.setattr(hp, 'Wikipedia', lambda **kw: Wiki())
    result = campaign.evaluate_adapter(root, directory, 1)
    assert result['overall_metric'] == 'em' and result['overall_accuracy_percent'] == 0
    assert result['f1'] == pytest.approx(2/3)
    assert result['checkpoint_spend'] == 34 and result['authorized_budget'] == 100
    assert result['validation']['n'] == 500 and result['identity']['checkpoint'] == meta
    assert campaign.evaluate_adapter(root, directory, 1) == result


def test_hotpotqa_blank_replies_keep_identical_supervision_and_paper_compatible_targets(synthetic):
    from bfas.rtd.student import teacher_tokens
    from bfas.rtd.transport import Behavior, FullState
    c = synthetic
    rows = deepcopy(c.rows); question = c.questions[1]
    replies = iter(['', 'search[article]', 'Action 2: finish[SYNTHETIC_GOLD]'])
    turns = []
    def generate(messages, stop, temperature):
        target = next(replies)
        turns.append(dict(context=messages, target=target, prompt='portable archive'))
        return target
    record = hp.run_episode(question, Wiki(), generate)
    rows[1]['demo'].update(turns=turns, worked_example=f"Question: {question['question']}\n"+record['transcript'])
    c.ledger.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    out = c.root/'blank-bank'
    build_hotpotqa_bank(ROOT, out, pool=c.pool, ledger=c.ledger, config=c.config,
                       tokenizer=c.tokenizer, wiki_factory=Wiki)
    payloads = [json.loads(p.read_text()) for p in (out/'sealed').glob('*.json') if len(p.stem) == 64]
    paid = next(p for p in payloads if p['provenance']['task_id'] == question['_id'])
    assert paid['cost'] == 11 and paid['provenance']['empty_targets_rendered_with_native_eos'] == [0]
    assert paid['historical_response']['demo']['turns'][0]['target'] == ''
    first = paid['behaviors'][0]
    assert first['text'] == c.tokenizer.eos_token and all(b['text'].strip() for b in paid['behaviors'])
    state = FullState(**first['state']); backend = SimpleNamespace(tokenizer=c.tokenizer, student_config=c.config)
    assert teacher_tokens(backend, Behavior(state, '')) == teacher_tokens(backend, Behavior(state, first['text']))
