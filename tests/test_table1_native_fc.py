"""Native BFCL bytes and real Qwen trainer boundaries; CPU and local files only."""
from copy import deepcopy
import json

import pytest

from tools import table1_common as io
from tools.table1_pool_from_sealed import build_pools, manifest_without_row_format, materialize
from tools.table1_row_format import (
    LEGACY_ROW_FORMAT, NATIVE_ROW_FORMAT, bfcl_native_rejected, read_bank_payload,
    render_package, row_format_for,
)


@pytest.fixture(autouse=True)
def cpu_offline(monkeypatch):
    import socket
    import torch
    from transformers import AutoModelForCausalLM

    def forbidden(*args, **kwargs):
        pytest.fail('native pool tests forbid network, model weights, and CUDA')

    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    monkeypatch.setenv('HF_HUB_OFFLINE', '1')
    monkeypatch.setenv('TRANSFORMERS_OFFLINE', '1')
    monkeypatch.setattr(socket.socket, 'connect', forbidden)
    monkeypatch.setattr(torch.cuda, '_lazy_init', forbidden)
    monkeypatch.setattr(AutoModelForCausalLM, 'from_pretrained', forbidden)


@pytest.fixture(scope='module')
def purchased():
    audit = io.DEFAULT_OUT / 'bfcl'
    acquisition = io.read_json(audit / 'acquired_B13843_seed0.json')
    sources = io.read_json(audit / 'ledger.json')['sources']
    owned = set(acquisition['purchased_ids'])
    return {q: (io.read_sealed(audit, q, owned), read_bank_payload('bfcl', q, owned, sources))
            for q in acquisition['purchased_ids']}


@pytest.fixture
def handler():
    from bfcl_eval.model_handler.local_inference.qwen_fc import QwenFCHandler

    # Pure formatting/decoding methods need no client or server construction.
    return QwenFCHandler.__new__(QwenFCHandler)


def example(purchased, case):
    for q, (snapshot, payload) in purchased.items():
        if not snapshot['rows']:
            continue
        text = payload['behaviors'][0]['text']
        kind = snapshot['package']['kind']
        if ((case == 'demo' and kind == 'demo_attempt' and text.startswith('<tool_call>'))
                or (case == 'generator' and kind == 'generator_item' and text)
                or (case == 'irrelevance' and kind == 'demo_attempt'
                    and 'irrelevance' in snapshot['rows'][0]['task_id'] and text)):
            return q, snapshot, payload
    pytest.fail(f'missing purchased {case} example')


def evaluation_task(snapshot, payload):
    """Independent input: official task file for demos, authored generator task.

    Do not recover expected tools from the native row or the sealed prompt.
    Only return the purchased task; no model/checker/environment is invoked.
    """
    if snapshot['package']['kind'] == 'generator_item':
        return deepcopy(payload['historical_response'])
    task_id = snapshot['rows'][0]['task_id']
    category = task_id.rsplit('_', 1)[0]
    path = (io.ROOT / 'envs/bfcl/gorilla/berkeley-function-call-leaderboard'
            / 'bfcl_eval/data' / f'BFCL_v4_{category}.json')
    return next(row for _, row in io.read_rows(path) if row['id'] == task_id)


@pytest.mark.parametrize('case', ['demo', 'generator', 'irrelevance'])
def test_purchased_prompt_matches_official_handler_and_rtd(purchased, handler, monkeypatch, case):
    from bfas.adapters.bfcl import BFCLAdapter
    from bfas.cc_pairs import normalize_truth, render_prompt, render_truth, thinking_off
    from bfas.rtd.bank import _response_text

    _, snapshot, payload = example(purchased, case)
    task = evaluation_task(snapshot, payload)
    row, = render_package('bfcl', snapshot, payload)
    # Same preprocessing/first-turn formatting functions as the official
    # inference_single_turn_prompting, stopping before any query/API method.
    inference = handler._pre_query_processing_prompting(deepcopy(task))
    inference = handler.add_first_turn_message_prompting(inference, deepcopy(task['question'][0]))
    raw_prompt = handler._format_prompt(inference['message'], inference['function'])
    assert raw_prompt.endswith('<|im_start|>assistant\n')
    # The vendored method itself stops before think; RTD's non-thinking eval
    # boundary in bfcl_task_rollout applies thinking_off to that exact string.
    deployment_prompt = thinking_off(raw_prompt)
    assert row['prompt'].encode('utf-8') == deployment_prompt.encode('utf-8')
    assert row['prompt'].endswith('<|im_start|>assistant\n<think>\n\n</think>\n\n')
    assert row['messages'] == []
    # Execute RTD bank's own renderer / continuation helpers on this task.
    monkeypatch.setattr(BFCLAdapter, '_handler', lambda self: handler)
    rtd_prompt = thinking_off(render_prompt(task['question'], task['function']))
    historical = payload['historical_response']
    rtd_response = (_response_text(historical['result']) if case != 'generator' else
                    render_truth(normalize_truth(historical['ground_truth']), historical['function'],
                                 historical.get('seed_category', 'simple_python')))
    assert row['prompt'].encode('utf-8') == rtd_prompt.encode('utf-8')
    assert row['response'].encode('utf-8') == rtd_response.encode('utf-8')
    assert (row['prompt'], row['response']) == (
        payload['behaviors'][0]['state']['prompt'], payload['behaviors'][0]['text'])
    decoded = handler.decode_ast(row['response'], 'Python', True)
    if case == 'irrelevance':
        assert not decoded and not row['response'].startswith('<tool_call>')
    else:
        assert decoded and row['response'].startswith('<tool_call>\n')
        assert row['response'].endswith('\n</tool_call>')
    if case == 'demo':
        assert len(decoded) > 1
        assert '\n</tool_call>\n<tool_call>\n' in row['response']


@pytest.fixture(scope='module')
def qwen_tokenizer():
    from transformers import AutoTokenizer
    from transformers.utils.hub import cached_file

    # Resolve the installed checkpoint cache only. Never download a tokenizer
    # or model; missing local Qwen files fail this test instead of skipping it.
    config = cached_file('Qwen/Qwen3.5-4B', 'tokenizer_config.json', local_files_only=True)
    from pathlib import Path
    tokenizer = AutoTokenizer.from_pretrained(
        Path(config).parent, local_files_only=True, trust_remote_code=False)
    assert tokenizer.eos_token == '<|im_end|>'
    return tokenizer


@pytest.mark.parametrize('case', ['demo', 'generator', 'irrelevance'])
def test_native_real_load_prompt_encode(purchased, qwen_tokenizer, tmp_path, monkeypatch, case):
    import appworld_train as trainer

    _, snapshot, payload = example(purchased, case)
    row, = render_package('bfcl', snapshot, payload)
    path = tmp_path / f'{case}.jsonl'
    io.write_rows(path, [row])
    loaded, = trainer.load_pool(path)
    assert loaded['_pool_index'] == 0 and loaded['messages'] == []

    def no_retemplating(*args, **kwargs):
        pytest.fail('messages=[] must use the deployed prompt verbatim')

    monkeypatch.setattr(qwen_tokenizer, 'apply_chat_template', no_retemplating)
    monkeypatch.setenv('AW_MAX_PROMPT_TOKENS', '4096')
    monkeypatch.setenv('AW_TRUNCATE_SIDE', 'tail')
    prompt = qwen_tokenizer(row['prompt'], add_special_tokens=False)['input_ids']
    response = qwen_tokenizer(row['response'], add_special_tokens=False)['input_ids']
    assert trainer.prompt_token_ids(qwen_tokenizer, loaded) == prompt
    assert response and qwen_tokenizer.eos_token_id not in response
    target = response[:trainer.MAX_RESPONSE_TOKENS] + [qwen_tokenizer.eos_token_id]
    ids, labels = trainer.encode(qwen_tokenizer, loaded, device='cpu')
    prompt = prompt[-4096:]
    assert ids.device.type == labels.device.type == 'cpu'
    assert ids.tolist() == [prompt + target]
    assert labels.tolist() == [[-100] * len(prompt) + target]
    assert (labels == -100).sum().item() == len(prompt)
    assert target.count(qwen_tokenizer.eos_token_id) == 1
    # Same assertion with actual truncation, retaining the think-empty suffix.
    monkeypatch.setenv('AW_MAX_PROMPT_TOKENS', '32')
    ids, labels = trainer.encode(qwen_tokenizer, loaded, device='cpu')
    assert ids.tolist() == [prompt[-32:] + target]
    assert labels.tolist() == [[-100] * 32 + target]


def test_every_purchased_native_row_keeps_sealed_text_and_metadata(purchased):
    count = 0
    for snapshot, payload in purchased.values():
        before = deepcopy((snapshot, payload))
        rows = render_package('bfcl', snapshot, payload)
        assert len(rows) == len(snapshot['rows'])
        for row, old, behavior in zip(rows, snapshot['rows'], payload['behaviors']):
            count += 1
            assert row['messages'] == []
            assert row['prompt'] == behavior['state']['prompt']
            assert row['response'] == behavior['text']
            for field in ('task_id', 'teacher', 'turn_index', 'token_hint'):
                assert row[field] == old[field]
        assert (snapshot, payload) == before
    assert count == 94


def test_native_rejects_prompt_drift(purchased):
    _, snapshot, payload = example(purchased, 'generator')
    payload = deepcopy(payload)
    payload['behaviors'][0]['state']['prompt'] += ' '
    with pytest.raises(ValueError, match='deployment rendering'):
        render_package('bfcl', snapshot, payload)


def test_native_cached_rejected_roundtrip(handler):
    calls = [{'f': json.dumps({'nested': [1, {'é': True}], 'text': '</tool_call>'})},
             {'g': '{"x": 0}'}]
    response = bfcl_native_rejected(json.dumps(calls))
    assert handler.decode_ast(response, 'Python', True) == [
        {'f': {'nested': [1, {'é': True}], 'text': '</tool_call>'}}, {'g': {'x': 0}}]
    assert bfcl_native_rejected(response + '<|im_end|>') == response
    assert bfcl_native_rejected('No matching tool.') == 'No matching tool.'
    assert bfcl_native_rejected('<tool_call>broken') == '<tool_call>broken'
    assert bfcl_native_rejected('[]') == ''
    assert bfcl_native_rejected('[{"name": "g", "arguments": {"x": 0}}]') == (
        '<tool_call>\n{"name": "g", "arguments": {"x": 0}}\n</tool_call>')


def test_native_supplemental_fields(purchased):
    import appworld_train as trainer

    audit = io.DEFAULT_OUT / 'bfcl'
    acquisition = io.read_json(audit / 'acquired_B13843_seed0.json')
    protocol = io.read_json(audit / 'protocol.json')
    snapshots = {q: deepcopy(s) for q, (s, _) in purchased.items()}
    rendered = {q: render_package('bfcl', snapshots[q], p) for q, (_, p) in purchased.items()}
    q, _, _ = example(purchased, 'demo')
    # Current purchased BFCL snapshots have zero cached pairs. Exercise the
    # exact-state attachment path with a synthetic failed student turn.
    snapshots[q]['pbsd_rejected']['0'] = '[{"f": "{\\"x\\": 0}"}]'
    # A fence in native text checks that spans are computed after rendering.
    rendered[q][0]['response'] += '\n```json\n{"x": 0}\n```'
    pools, manifest = build_pools(snapshots, acquisition, protocol, rendered, row_format=NATIVE_ROW_FORMAT)
    _, before = build_pools(snapshots, acquisition, protocol)
    assert manifest_without_row_format(manifest) == manifest_without_row_format(before)
    rejected = [r['_rejected'] for r in pools['pbsd_insp'] if '_rejected' in r]
    assert rejected == ['<tool_call>\n{"name": "f", "arguments": {"x": 0}}\n</tool_call>']
    for row in pools['sad']:
        assert row['_sad_spans']['action'] == [list(s) for s in trainer.action_spans(row['response'])]
    assert any(r['_sad_spans']['action'] for r in pools['sad'])
    for row in pools['pbsd_agent']:
        for c in row['c']:
            candidate = rendered[c['package_id']][c['turn_index']]
            assert (c['state_prompt'], c['response']) == (candidate['prompt'], candidate['response'])
            assert c['state_prompt'] == row['prompt']


def test_defaults_and_preserve_legacy_destination(tmp_path):
    assert row_format_for('bfcl') == NATIVE_ROW_FORMAT
    assert row_format_for('bfcl', LEGACY_ROW_FORMAT) == LEGACY_ROW_FORMAT
    assert row_format_for('alfworld') == row_format_for('appworld') == LEGACY_ROW_FORMAT
    with pytest.raises(ValueError, match='unsupported row format'):
        row_format_for('appworld', NATIVE_ROW_FORMAT)
    audit = io.DEFAULT_OUT / 'bfcl'
    source = audit / 'acquired_B5537_seed0.json'
    acquisition = tmp_path / source.name
    acquisition.write_bytes(source.read_bytes())
    root = tmp_path / 'pools'
    materialize(audit, acquisition, root, row_format=LEGACY_ROW_FORMAT)
    before = {p: p.read_bytes() for p in root.rglob('*') if p.is_file()}
    with pytest.raises(ValueError, match='sibling pool directory'):
        materialize(audit, acquisition, root)
    assert before == {p: p.read_bytes() for p in before}


@pytest.mark.parametrize('cap,count,empty_count', [(13843, 94, 13), (5537, 30, 5)])
def test_native_materialized_pools_real_trainer_all_rows(
        purchased, qwen_tokenizer, tmp_path, monkeypatch, cap, count, empty_count):
    import appworld_train as trainer
    from bfas.rtd.baselines.evidence import action_ids

    def no_retemplating(*args, **kwargs):
        pytest.fail('native pools must never be re-templated')

    monkeypatch.setattr(qwen_tokenizer, 'apply_chat_template', no_retemplating)
    monkeypatch.setenv('AW_MAX_PROMPT_TOKENS', '4096')
    monkeypatch.setenv('AW_TRUNCATE_SIDE', 'tail')
    audit = io.DEFAULT_OUT / 'bfcl'
    for seed in range(3):
        source = audit / f'acquired_B{cap}_seed{seed}.json'
        acquisition = tmp_path / source.name
        acquisition.write_bytes(source.read_bytes())
        root = tmp_path / 'pools_native'
        manifest = materialize(audit, acquisition, root)
        assert acquisition.read_bytes() == source.read_bytes()
        legacy = io.read_json(io.DEFAULT_OUT / 'pools/bfcl' / f'B{cap}_seed{seed}/manifest.json')
        assert manifest['row_format'] == NATIVE_ROW_FORMAT
        assert manifest_without_row_format(manifest) == manifest_without_row_format(legacy)
        for arm in io.ARMS:
            path = root / 'bfcl' / f'B{cap}_seed{seed}' / f'pool_{arm}.jsonl'
            assert io.digest([r for _, r in io.read_rows(path)]) == manifest['arms'][arm]['pool_sha256']
            if arm == 'star':
                assert path.read_bytes() == b''
                with pytest.raises(AssertionError, match='pool is empty'):
                    trainer.load_pool(path)
                continue
            rows = trainer.load_pool(path)
            assert len(rows) == count
            assert sum(r['response'] == '' for r in rows) == empty_count
            for row, origin in zip(rows, manifest['row_origins']):
                assert row['messages'] == []
                snapshot, payload = purchased[origin['package_id']]
                behavior = payload['behaviors'][origin['package_row_index']]
                assert (row['prompt'], row['response']) == (behavior['state']['prompt'], behavior['text'])
                prompt = qwen_tokenizer(row['prompt'], add_special_tokens=False)['input_ids']
                assert trainer.prompt_token_ids(qwen_tokenizer, row) == prompt
                response = qwen_tokenizer(row['response'], add_special_tokens=False)['input_ids']
                assert qwen_tokenizer.eos_token_id not in response
                if row['response']:
                    assert response and '_native_fc_empty_response' not in row
                else:
                    assert row['_native_fc_empty_response'] is True
                    assert payload['historical_response']['ground_truth'] == []
                    assert not response  # Explicit EOS-only exception, as in RTD.
                target = response[:trainer.MAX_RESPONSE_TOKENS] + [qwen_tokenizer.eos_token_id]
                assert action_ids(qwen_tokenizer, row['response']) == response + [qwen_tokenizer.eos_token_id]
                ids, labels = trainer.encode(qwen_tokenizer, row, device='cpu')
                assert ids.device.type == labels.device.type == 'cpu'
                assert ids.tolist() == [prompt[-4096:] + target]
                assert labels.tolist() == [[-100] * len(prompt[-4096:]) + target]
                assert target.count(qwen_tokenizer.eos_token_id) == 1
                if arm == 'sad':
                    assert row['_sad_spans']['action'] == [list(s) for s in trainer.action_spans(row['response'])]
                if arm == 'pbsd_agent':
                    for c in row['c']:
                        assert c['state_prompt'] == row['prompt']
                        assert c['response'] == purchased[c['package_id']][1]['behaviors'][c['turn_index']]['text']


def test_empty_native_requires_explicit_zero_call_provenance(purchased, tmp_path):
    import appworld_train as trainer

    snapshot, payload = next((s, p) for s, p in purchased.values()
                             if s['rows'] and p['behaviors'][0]['text'] == '')
    row, = render_package('bfcl', snapshot, payload)
    path = tmp_path / 'empty.jsonl'
    io.write_rows(path, [row])
    assert trainer.load_pool(path)[0]['response'] == ''
    for changes in ({'_native_fc_empty_response': False}, {'_native_fc_empty_response': 1},
                    {'messages': [{'role': 'user', 'content': 'question'}]}, {'prompt': 'unformatted'}):
        io.write_rows(path, [dict(row, **changes)])
        with pytest.raises(ValueError, match='response must be a non-empty string'):
            trainer.load_pool(path)
    payload = deepcopy(payload)
    payload['historical_response']['ground_truth'] = [{'f': {'x': [1]}}]
    with pytest.raises(ValueError, match='unexplained empty native'):
        render_package('bfcl', snapshot, payload)
