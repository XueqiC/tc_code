"""D16 configured students and the frozen native FC deployment boundary."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

from test_bfcl_gemma4_rows import tokenizer, context, row_for
from bfas.rtd.bfcl_decode import student_handler, guard_rtd_decoding
from bfas.rtd.student import bfcl_row, termination_ids
from bfas.rtd.cli import load_config, validate_config
from bfas.rtd.hardware import host_class
from bfas.rtd.return_gradient import TorchPolicyBackend
from bfas.rtd.transport import FullState, Behavior
from bfas.rtd.persistence import digest, file_hash

ROOT = Path(__file__).resolve().parents[1]
CONFIG = dict(student='google/gemma-4-12B-it', student_call_format='gemma4', benchmark='bfcl')


@pytest.mark.parametrize('name', ['bfcl_gemma4', 'webshop', 'alfworld'])
def test_new_configs_preserve_v11_protocol(name):
    config = load_config(ROOT/f'configs/rtd/v1_1_{name}.yaml')
    assert config['student'] == CONFIG['student']
    assert config['rounds'] == 2 and config['budget_checkpoints_bank_fraction'] == [.1, .25]
    assert config['exposure_unit'] == 'state_teacher_record_plus_two_source_actions'
    assert config['source_temperature'] == 1. and config['same_hardware_required'] is True
    assert config['memory_peak_budget_gb'] == 60
    for key, value in [('source_temperature', .7), ('same_hardware_required', False), ('exposure_unit', 'tokens')]:
        with pytest.raises(ValueError):
            validate_config(config | {key: value})
    with pytest.raises(ValueError):
        validate_config(config | {'student_call_format': 'json'})
    from bfas.rtd.benchmarks.registry import get_benchmark
    assert callable(get_benchmark(config).feedback_rollout)


def test_bfcl_config_changes_only_requested_fields():
    import yaml
    old = yaml.safe_load((ROOT/'configs/rtd/v1_1_bfcl.yaml').read_text())
    new = yaml.safe_load((ROOT/'configs/rtd/v1_1_bfcl_gemma4.yaml').read_text())
    assert {k for k in old.keys() | new.keys() if old.get(k) != new.get(k)} == {
        'student', 'student_call_format', 'memory_peak_budget_gb', 'replay_bank_path'}


def test_gemma_render_decode_and_teacher_likelihood_boundary(tokenizer, context):
    from tools.bfcl_pool_render_gemma4 import render_row
    row = row_for(context, [{'places.find': {'query': 'Zürich "quoted"\\n東京'}}])
    result = bfcl_row(CONFIG, row, tokenizer)
    assert result == render_row(row, tokenizer, verify=True)
    handler = student_handler(CONFIG, tokenizer)
    errors = []
    guard_rtd_decoding(handler, lambda *e: errors.append(e), student_call_format='gemma4')
    assert handler.decode_ast(result['response'], None, False) == [{'places.find': {'query': 'Zürich "quoted"\\n東京'}}]
    assert not errors
    backend = object.__new__(TorchPolicyBackend)
    backend.tokenizer, backend.student_config = tokenizer, CONFIG
    backend.score_tokens = lambda prompt, action, parameters, **kw: (action, kw)
    state = FullState.create({'task': 'native'}, context['messages'], result['prompt'], digest('parent'))
    ids, kw = backend.score_behavior(Behavior(state, result['response']), {})
    assert ids == tokenizer.encode(result['response'], add_special_tokens=False)
    assert kw['eos_token_id'] == ids[-1] in termination_ids(backend)


@pytest.mark.parametrize('text', ['<|tool_call>call:places.find{q:<|"|>unfinished}', '<|tool_call>bad'])
def test_gemma_malformed_is_recorded_without_repair(tokenizer, text):
    handler = student_handler(CONFIG, tokenizer)
    errors = []
    guard_rtd_decoding(handler, lambda *e: errors.append(e), student_call_format='gemma4')
    response = SimpleNamespace(choices=[SimpleNamespace(text=text)],
                               usage=SimpleNamespace(prompt_tokens=5, completion_tokens=3))
    parsed = handler._parse_query_response_prompting(response)
    assert parsed['model_responses_message_for_chat_history'] == dict(role='assistant', content=text)
    with pytest.raises(ValueError):
        handler.decode_ast(text, None, False)
    assert [stage for _, stage in errors] == ['parse_response', 'decode_ast']


def test_blackwell_class_preserves_machine_and_gpu_constraints():
    gpu = 'NVIDIA RTX PRO 6000 Blackwell Workstation Edition'
    assert host_class('rai', gpu, {}) == host_class('rai.example', gpu, {}) == 'rai-rtx-pro-6000-blackwell'
    assert host_class('different', gpu, {}) == 'different'
    assert host_class('rai', 'NVIDIA A100', {}) == 'rai'


def test_paid_bfcl_bank_reuses_rendered_rows(tmp_path, tokenizer, context):
    from bfas.rtd.benchmarks.bfcl_bank_v11 import build_bfcl_pool_bank
    from bfas.rtd.bank_build import validate_state_certificate
    from bfas.rtd.bank import parent_hash
    tid = 'simple_python_0'
    entry = dict(id=tid, question=[context['messages']], function=context['functions'])
    row = dict(row_for(context, [{'places.find': {'query': 'Zürich'}}]), task_id=tid)
    paid = dict(task_id=tid, teacher='azure/gpt-5.4-FC', tokens_spent=33, attempt_index=0, verified=True,
        demo=dict(turns=[dict(prompt='legacy', target=row['response'], context=context)], worked_example=''))
    pool, ledger, support = (tmp_path/n for n in ('pool.jsonl','ledger.jsonl','support.json'))
    pool.write_text(json.dumps(row)+'\n'); ledger.write_text(json.dumps(paid)+'\n')
    support.write_text(json.dumps(dict(parents=[dict(official_id=tid, parent_hash=parent_hash(entry), fold=0)])))
    config = CONFIG | dict(support_manifest=str(support))
    out = tmp_path/'bank'
    result = build_bfcl_pool_bank(tmp_path, out, pool=pool, ledger=ledger, config=config,
                                 entries={tid: entry}, tokenizer=tokenizer)
    assert result['budget_denominator'] == 33
    cert = validate_state_certificate(out, benchmark='bfcl', student=config['student'])
    assert cert['core']['class_caps'] == {'demo_attempt': 64}
    assert json.loads((out/'public/support.json').read_text()) == json.loads(support.read_text())
    q = json.loads((out/'public/requests.json').read_text())[0]['spec']['query_id']
    b = json.loads((out/f'sealed/{q}.json').read_text())['behaviors'][0]
    rendered = bfcl_row(config, row, tokenizer)
    assert b['state']['prompt'] == rendered['prompt'] and b['text'] == rendered['response']
    # Bind unknown collection spend without relabeling the usable denominator.
    provenance = dict(ledger_sha256=file_hash(ledger), pool_cost_status='exact-plus-unknown-rerun',
        historical_cost_status='recorded-lower-bound-plus-unknown', recorded_output_tokens=33,
        missing_usage_attempts=[], adapter_rerun=dict(extra_calls='unknown', exact_output_tokens=None))
    sidecar = ledger.with_suffix('.provenance.json')
    sidecar.write_text(json.dumps(provenance))
    recorded = build_bfcl_pool_bank(tmp_path, tmp_path/'bound', pool=pool, ledger=ledger,
                                    config=config, entries={tid: entry}, tokenizer=tokenizer)
    bound = validate_state_certificate(tmp_path/'bound', benchmark='bfcl', student=config['student'])
    assert bound['core']['inputs']['ledger_provenance'] == file_hash(sidecar)
    assert recorded['budget_denominator'] == 33
    assert recorded['teacher_accounting']['pool_cost_status'] == 'exact-plus-unknown-rerun'
    provenance['ledger_sha256'] = 'wrong'
    sidecar.write_text(json.dumps(provenance))
    with pytest.raises(ValueError, match='provenance binding'):
        build_bfcl_pool_bank(tmp_path, tmp_path/'mismatch', pool=pool, ledger=ledger,
                            config=config, entries={tid: entry}, tokenizer=tokenizer)
    sidecar.unlink()
    paid['tokens_spent'] = -1
    ledger.write_text(json.dumps(paid)+'\n')
    with pytest.raises(ValueError, match='spend'):
        build_bfcl_pool_bank(tmp_path, tmp_path/'bad', pool=pool, ledger=ledger, config=config,
                             entries={tid: entry}, tokenizer=tokenizer)


def test_softcapped_head_gradients_match_native_forward():
    import torch
    from torch.func import functional_call
    from bfas.rtd.source_scoring import _position_vjps, cap_logits
    head = torch.nn.Linear(3, 5, bias=False).double()
    hidden = torch.randn(2,3, dtype=torch.float64, requires_grad=True)
    old = hidden.detach() + .3
    labels = torch.tensor([0,4])
    hard, soft, _ = _position_vjps(head, {}, {}, hidden, old, labels, softcap=.7)
    q = cap_logits(head(hidden), .7).log_softmax(-1)
    p = cap_logits(head(old), .7).softmax(-1).detach()
    expected_h, = torch.autograd.grad(-q.gather(1,labels[:,None]).sum(), hidden, retain_graph=True)
    expected_s, = torch.autograd.grad(-(p*q).sum(), hidden)
    torch.testing.assert_close(hard[0], expected_h, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(soft[0], expected_s, rtol=1e-12, atol=1e-12)


def test_gemma_official_checker_syntax_and_relevance(tokenizer, context):
    from tools.behavior_atom.checker_bridge import CheckerBridge
    row = bfcl_row(CONFIG, row_for(context, [{'places.find': {'query': 'Tokyo'}}]), tokenizer)
    with CheckerBridge(CONFIG['student']+'-FC', student_call_format='gemma4') as checker:
        assert checker.check_syntax(row['response'])['valid']
        assert not checker.check_syntax('<|tool_call>bad')['valid']
        verdict = checker.check_relevance(dict(id='irrelevance_99999', question=[context['messages']],
            function=context['functions']), 'No suitable tool.<turn|>', 'irrelevance')
        assert verdict['valid'] is True


def test_gemma_empty_native_abstention_remains_legal(tokenizer):
    handler = student_handler(CONFIG, tokenizer)
    errors = []
    guard_rtd_decoding(handler, lambda *e: errors.append(e), student_call_format='gemma4')
    assert handler.decode_ast('', None, False) == []
    assert handler.decode_ast('<turn|>', None, False) == []
    assert not errors
