"""Native BFCL bytes and real Qwen trainer boundaries; CPU and local files only."""
from copy import deepcopy
import json

import pytest

from tools import table1_common as io
from tools.table1_pool_from_sealed import (
    build_pools, manifest_without_row_format, materialize, replay_native_rankings,
)
from tools.table1_row_format import (
    LEGACY_ROW_FORMAT, NATIVE_ROW_FORMAT, NATIVE_THINK_PREFIX, bfcl_evaluation_prompt,
    bfcl_native_prompt, bfcl_native_rejected, read_bank_payload,
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
    from bfcl_eval.utils import add_language_specific_hint_to_function_doc
    processed, = add_language_specific_hint_to_function_doc([deepcopy(task)])
    # Same preprocessing/first-turn formatting functions as the official
    # inference_single_turn_prompting, stopping before any query/API method.
    inference = handler._pre_query_processing_prompting(processed)
    inference = handler.add_first_turn_message_prompting(inference, processed['question'][0])
    raw_prompt = handler._format_prompt(inference['message'], inference['function'])
    assert raw_prompt.endswith('<|im_start|>assistant\n')
    assert row['prompt'].encode('utf-8') == raw_prompt.encode('utf-8')
    assert not row['prompt'].endswith(NATIVE_THINK_PREFIX)
    assert row['messages'] == []
    # Execute RTD bank's own renderer / continuation helpers on this task.
    monkeypatch.setattr(BFCLAdapter, '_handler', lambda self: handler)
    rtd_prompt = thinking_off(render_prompt(task['question'], task['function']))
    historical = payload['historical_response']
    rtd_response = (_response_text(historical['result']) if case != 'generator' else
                    render_truth(normalize_truth(historical['ground_truth']), historical['function'],
                                 historical.get('seed_category', 'simple_python')))
    assert (rtd_prompt, rtd_response) == (
        payload['behaviors'][0]['state']['prompt'], payload['behaviors'][0]['text'])
    # Precise paper comparison: RTD puts the empty-think suffix in its prompt
    # and omits language preprocessing. v2 adds the handler's language note
    # (and Java/JS schema conversion) and supervises that suffix in the target.
    assert rtd_prompt.endswith(NATIVE_THINK_PREFIX)
    assert row['prompt'] + NATIVE_THINK_PREFIX == thinking_off(
        render_prompt(processed['question'], processed['function']))
    assert row['prompt'] != rtd_prompt.removesuffix(NATIVE_THINK_PREFIX)
    assert row['response'] == NATIVE_THINK_PREFIX + rtd_response
    decoded = handler.decode_ast(row['response'], 'Python', True)
    if case == 'irrelevance':
        assert not decoded and not rtd_response.startswith('<tool_call>')
    else:
        assert decoded and row['response'].startswith(NATIVE_THINK_PREFIX + '<tool_call>\n')
        assert row['response'].endswith('\n</tool_call>')
    if case == 'demo':
        assert len(decoded) > 1
        assert '\n</tool_call>\n<tool_call>\n' in row['response']


RECORDED_SAMPLES = [r for _, r in io.read_rows(io.DEFAULT_OUT / 'bfcl/ddpo_samples_B13843.jsonl')]
RECORDED_TASK_IDS = sorted({r['task_id'] for r in RECORDED_SAMPLES})


@pytest.mark.parametrize('task_id', RECORDED_TASK_IDS)
def test_v2_prompt_equals_all_recorded_evaluation_prompts(purchased, task_id):
    from tools.table1_ddpo_rank import task_entries

    assert len(RECORDED_TASK_IDS) == 22
    task = task_entries([task_id])[0][task_id]
    expected = {r['prompt'].encode('utf-8') for r in RECORDED_SAMPLES if r['task_id'] == task_id}
    assert len(expected) == 1  # All four repeats recorded the same prompt.
    prompt = bfcl_evaluation_prompt(task['question'][0], task['function'], task_id)
    assert {prompt.encode('utf-8')} == expected
    for snapshot, payload in purchased.values():
        for row in render_package('bfcl', snapshot, payload):
            if row['task_id'] == task_id:
                assert row['prompt'].encode('utf-8') == prompt.encode('utf-8')


def test_java_residual_is_parameter_conversion_not_prompt_drift(purchased):
    from bfcl_eval.utils import _get_language_specific_hint

    shared, residual = [], []
    recorded = {r['task_id']: r['prompt'] for r in RECORDED_SAMPLES}
    for snapshot, payload in purchased.values():
        for row, behavior in zip(render_package('bfcl', snapshot, payload), payload['behaviors']):
            tid = row['task_id']
            if tid not in recorded:
                continue
            shared.append(tid)
            stripped = recorded[tid].replace(_get_language_specific_hint(tid.rsplit('_', 1)[0]), '')
            if stripped + NATIVE_THINK_PREFIX != behavior['state']['prompt']:
                residual.append(tid)
                old_tools = json.loads(behavior['state']['task_json'])['function']
                assert tid == 'simple_java_6'
                assert all(p['type'] == 'boolean' for p in old_tools[0]['parameters']['properties'].values())
                assert 'This is Java boolean type parameter in string representation.' in row['prompt']
                assert '"type": "boolean"' not in row['prompt']
            assert row['prompt'] == recorded[tid]
    assert len(set(shared)) == 14 and set(residual) == {'simple_java_6'}


@pytest.mark.parametrize('continuation', ['<tool_call>\n{"name": "f", "arguments": {}}\n</tool_call>',
                                       'No matching tool.', ''])
def test_handler_strips_model_emitted_think_block(handler, continuation):
    from types import SimpleNamespace

    parsed = handler._parse_query_response_prompting(SimpleNamespace(
        choices=[SimpleNamespace(text=NATIVE_THINK_PREFIX + continuation)],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=20)))
    assert parsed['model_responses'] == continuation
    assert parsed['reasoning_content'] == ''


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
    prefix_ids = qwen_tokenizer(NATIVE_THINK_PREFIX, add_special_tokens=False)['input_ids']
    assert target[:len(prefix_ids)] == prefix_ids
    assert qwen_tokenizer.convert_tokens_to_ids('</think>') in prefix_ids
    ids, labels = trainer.encode(qwen_tokenizer, loaded, device='cpu')
    prompt = prompt[-4096:]
    assert ids.device.type == labels.device.type == 'cpu'
    assert ids.tolist() == [prompt + target]
    assert labels.tolist() == [[-100] * len(prompt) + target]
    assert (labels == -100).sum().item() == len(prompt)
    assert target.count(qwen_tokenizer.eos_token_id) == 1
    # Truncating the prompt does not mask or remove the target's think block.
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
            assert row['prompt'] == bfcl_native_prompt(behavior['state'], payload['historical_response'])
            assert row['response'] == NATIVE_THINK_PREFIX + behavior['text']
            for field in ('task_id', 'teacher', 'turn_index', 'token_hint'):
                assert row[field] == old[field]
        assert (snapshot, payload) == before
    assert count == 94


def test_native_rejects_prompt_drift(purchased):
    _, snapshot, payload = example(purchased, 'generator')
    payload = deepcopy(payload)
    payload['behaviors'][0]['state']['prompt'] += ' '
    with pytest.raises(ValueError, match='RTD rendering'):
        render_package('bfcl', snapshot, payload)


def test_native_cached_rejected_roundtrip(handler):
    calls = [{'f': json.dumps({'nested': [1, {'é': True}], 'text': '</tool_call>'})},
             {'g': '{"x": 0}'}]
    response = bfcl_native_rejected(json.dumps(calls))
    assert handler.decode_ast(response, 'Python', True) == [
        {'f': {'nested': [1, {'é': True}], 'text': '</tool_call>'}}, {'g': {'x': 0}}]
    assert bfcl_native_rejected(response + '<|im_end|>') == response
    assert response.startswith(NATIVE_THINK_PREFIX)
    assert bfcl_native_rejected('No matching tool.') == NATIVE_THINK_PREFIX + 'No matching tool.'
    assert bfcl_native_rejected('<tool_call>broken') == NATIVE_THINK_PREFIX + '<tool_call>broken'
    assert bfcl_native_rejected('[]') == NATIVE_THINK_PREFIX
    assert bfcl_native_rejected('[{"name": "g", "arguments": {"x": 0}}]') == (
        NATIVE_THINK_PREFIX + '<tool_call>\n{"name": "g", "arguments": {"x": 0}}\n</tool_call>')


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
    assert rejected == [NATIVE_THINK_PREFIX + '<tool_call>\n{"name": "f", "arguments": {"x": 0}}\n</tool_call>']
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
    assert row_format_for('bfcl', 'native-fc') == 'native-fc-v2'
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
        # Ranking replay is additional paid evidence, separate from v2's
        # rendering-only invariant. Every other arm and acquisition is fixed.
        comparable = deepcopy(manifest)
        comparable['arms']['ddpo'] = legacy['arms']['ddpo']
        comparable['ranking'] = legacy['ranking']
        assert manifest_without_row_format(comparable) == manifest_without_row_format(legacy)
        for arm in io.ARMS:
            path = root / 'bfcl' / f'B{cap}_seed{seed}' / f'pool_{arm}.jsonl'
            assert io.digest([r for _, r in io.read_rows(path)]) == manifest['arms'][arm]['pool_sha256']
            if arm == 'star':
                assert path.read_bytes() == b''
                with pytest.raises(AssertionError, match='pool is empty'):
                    trainer.load_pool(path)
                continue
            rows = trainer.load_pool(path)
            rank_count = (16 if cap == 13843 else 10) if arm == 'ddpo' else 0
            assert len(rows) == count + rank_count
            assert sum(r.get('_native_fc_empty_response') is True for r in rows) == empty_count
            for row, origin in zip(rows, manifest['row_origins']):
                assert row['messages'] == []
                snapshot, payload = purchased[origin['package_id']]
                behavior = payload['behaviors'][origin['package_row_index']]
                assert row['prompt'] == bfcl_native_prompt(behavior['state'], payload['historical_response'])
                assert row['response'] == NATIVE_THINK_PREFIX + behavior['text']
                prompt = qwen_tokenizer(row['prompt'], add_special_tokens=False)['input_ids']
                assert trainer.prompt_token_ids(qwen_tokenizer, row) == prompt
                response = qwen_tokenizer(row['response'], add_special_tokens=False)['input_ids']
                assert qwen_tokenizer.eos_token_id not in response
                prefix_ids = qwen_tokenizer(NATIVE_THINK_PREFIX, add_special_tokens=False)['input_ids']
                assert response[:len(prefix_ids)] == prefix_ids
                if behavior['text']:
                    assert response and '_native_fc_empty_response' not in row
                else:
                    assert row['_native_fc_empty_response'] is True
                    assert payload['historical_response']['ground_truth'] == []
                    assert row['response'] == NATIVE_THINK_PREFIX
                    assert response == prefix_ids  # Every no-call row learns the closing tag.
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
                        assert c['response'] == NATIVE_THINK_PREFIX + purchased[c['package_id']][1]['behaviors'][c['turn_index']]['text']
            # Rank rows use the exact recorded context, and both sides of the
            # preference loss must include the model's own framing tokens.
            for row in rows[count:]:
                assert row['teacher'] == 'rank' and row['messages'] == []
                assert row['prompt'] in {r['prompt'] for r in RECORDED_SAMPLES}
                for field in ('response', '_rejected'):
                    response = qwen_tokenizer(row[field], add_special_tokens=False)['input_ids']
                    prefix_ids = qwen_tokenizer(NATIVE_THINK_PREFIX, add_special_tokens=False)['input_ids']
                    assert response[:len(prefix_ids)] == prefix_ids
                    prompt = trainer.prompt_token_ids(qwen_tokenizer, row)[-4096:]
                    target = response[:trainer.MAX_RESPONSE_TOKENS] + [qwen_tokenizer.eos_token_id]
                    ids, labels = trainer.encode(qwen_tokenizer, dict(row, response=row[field]), device='cpu')
                    assert ids.tolist() == [prompt + target]
                    assert labels.tolist() == [[-100] * len(prompt) + target]
                    assert target.count(qwen_tokenizer.eos_token_id) == 1


def test_empty_native_requires_explicit_zero_call_provenance(purchased, tmp_path):
    import appworld_train as trainer

    snapshot, payload = next((s, p) for s, p in purchased.values()
                             if s['rows'] and p['behaviors'][0]['text'] == '')
    row, = render_package('bfcl', snapshot, payload)
    path = tmp_path / 'empty.jsonl'
    io.write_rows(path, [row])
    assert trainer.load_pool(path)[0]['response'] == NATIVE_THINK_PREFIX
    for changes in ({'_native_fc_empty_response': False}, {'_native_fc_empty_response': 1},
                    {'messages': [{'role': 'user', 'content': 'question'}]}, {'prompt': 'unformatted'}):
        io.write_rows(path, [dict(row, **changes)])
        with pytest.raises(ValueError, match='invalid native empty-continuation marker'):
            trainer.load_pool(path)
    # A missing continuation is not an empty v2 target; v1 remains loadable.
    io.write_rows(path, [dict(row, response='')])
    with pytest.raises(ValueError, match='response must be a non-empty string'):
        trainer.load_pool(path)
    io.write_rows(path, [dict(row, prompt=row['prompt'] + NATIVE_THINK_PREFIX, response='')])
    assert trainer.load_pool(path)[0]['response'] == ''
    payload = deepcopy(payload)
    payload['historical_response']['ground_truth'] = [{'f': {'x': [1]}}]
    with pytest.raises(ValueError, match='unexplained empty native'):
        render_package('bfcl', snapshot, payload)


@pytest.mark.parametrize('cap,rank_count,rank_cost', [(13843, 16, 1083), (5537, 10, 680)])
def test_rank_replay_preserves_recorded_samples_and_all_attempt_costs(cap, rank_count, rank_cost):
    directory = io.DEFAULT_OUT / 'pools_native/bfcl' / f'B{cap}_seed0'
    source = io.DEFAULT_OUT / 'bfcl'
    manifest = io.read_json(directory / 'manifest.json')
    base = [r for _, r in io.read_rows(directory / 'pool_sft.jsonl')]
    tids = set(manifest['tasks_covered'])
    calls = [r for _, r in io.read_rows(source / 'ddpo_rank_ledger.jsonl') if r['task_id'] in tids]
    before = deepcopy((base, manifest, calls))
    pools, result = replay_native_rankings(source, dict(sft=base), manifest)
    assert (base, manifest, calls) == before
    assert pools['ddpo'][:len(base)] == base
    ranked = {r['task_id']: r for r in calls if r['status'] == 'ranked'}
    assert len(pools['ddpo']) == len(base) + rank_count
    for row in pools['ddpo'][len(base):]:
        call = ranked[row['task_id']]
        assert row['prompt'] == call['preference_row']['prompt']
        for field, index in zip(('response', '_rejected'), call['best_worst']):
            continuation = call['candidates'][index - 1]
            assert row[field] == NATIVE_THINK_PREFIX + continuation
            assert any(r['response'] == continuation and r['task_id'] == row['task_id']
                       for r in RECORDED_SAMPLES)
    arm = result['arms']['ddpo']
    assert arm['rank_pairs'] == rank_count
    assert arm['ranking_cost'] == sum(c['tokens_spent'] for c in calls) == rank_cost
    assert any(c['status'] == 'parse_failed' and c['tokens_spent'] > 0 for c in calls)
    assert arm['C_m'] == result['sealed_replay_spend'] + rank_cost
    assert arm['over_cap_tokens'] == arm['C_m'] - cap > 0
    assert arm['pool_sha256'] == io.digest(pools['ddpo'])
    assert arm['ranking_call_ids'] == sorted(c['call_id'] for c in calls)


@pytest.mark.parametrize('drift', ['context', 'candidates', 'preference', 'best_worst'])
def test_rank_replay_rejects_evidence_drift(tmp_path, drift):
    directory = io.DEFAULT_OUT / 'pools_native/bfcl/B13843_seed0'
    manifest = io.read_json(directory / 'manifest.json')
    base = [r for _, r in io.read_rows(directory / 'pool_sft.jsonl')]
    calls = [r for _, r in io.read_rows(io.DEFAULT_OUT / 'bfcl/ddpo_rank_ledger.jsonl')]
    ranked = next(c for c in calls if c['status'] == 'ranked')
    if drift == 'context':
        ranked['context']['prompt'] += NATIVE_THINK_PREFIX
    elif drift == 'candidates':
        ranked['candidates'][0] += ' changed'
    elif drift == 'preference':
        ranked['preference_row']['_rejected'] += ' changed'
    else:
        ranked['best_worst'] = [1, 1]
    io.write_rows(tmp_path / 'ddpo_rank_ledger.jsonl', calls)
    io.write_rows(tmp_path / 'ddpo_samples_B13843.jsonl', RECORDED_SAMPLES)
    with pytest.raises(ValueError, match='differ'):
        replay_native_rankings(tmp_path, dict(sft=base), manifest)
