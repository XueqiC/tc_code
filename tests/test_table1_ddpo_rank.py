"""CPU-only fake-provider tests. No GPU, credentials, downloads, or live API."""
import copy
import io as stdio
import json
from pathlib import Path
import urllib.error

import pytest

from tools import table1_common as io
from tools import table1_ddpo_rank as ddpo
import appworld_teacher as at


@pytest.fixture
def audit(tmp_path):
    root = tmp_path / 'audit'
    base = [dict(task_id=t, teacher='ds', turn_index=0, messages=[],
                 prompt='original ' + t, response='teacher answer', token_hint=4)
            for t in ('task_a', 'task_b')]
    for seed in range(3):
        directory = root / 'pools/bfcl' / f'B10_seed{seed}'
        io.write_rows(directory / 'pool_sft.jsonl', base)
        sft = dict(C_m=9, pool_sha256=io.digest(base), teacher_call_ids=['purchased'],
                   teacher_evidence_package_ids=['purchased'], rows=2)
        io.write_json(directory / 'manifest.json', dict(
            benchmark='bfcl', cap=10, training_seed=seed,
            initial_checkpoint=io.INITIAL_CHECKPOINT, tasks_covered=['task_a', 'task_b'],
            purchased_ids=['purchased'], sealed_replay_spend=9, remaining_budget=1,
            ranking=dict(reused=0, total_cached=5),
            arms=dict(sft=sft, ddpo=dict(sft, trainer_status='sft_fallback_missing_rank_costs'))))
    io.write_rows(root / 'bfcl/ddpo_samples_B10.jsonl', [
        dict(task_id=t, sample_index=i, response=f'student {i}', verified=i == 0,
             prompt='official ' + t, messages=[])
        for t in ('task_a', 'task_b') for i in range(3)])
    return root


class FakeClient:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.calls = []

    def __call__(self, messages, usage):
        self.calls.append(copy.deepcopy(messages))
        reply, reported = next(self.replies)
        usage.update(reported)
        if isinstance(reply, Exception):
            raise reply
        return reply


def run(audit, replies, cap=40, **kwargs):
    client = FakeClient(replies)
    result = ddpo.rank(audit, 10, cap, client, token_counter=lambda text: len(text.split()),
                       sleep=lambda seconds: None, **kwargs)
    return result, client


def ledger(audit):
    return ddpo.rows(audit / 'bfcl/ddpo_rank_ledger.jsonl')


def manifest(audit, seed=0):
    return io.read_json(audit / 'pools/bfcl' / f'B10_seed{seed}/manifest.json')


def test_dedupe_resume_hard_cap_and_all_seeds(audit):
    first, client = run(audit, [('1 3', {'completion_tokens': 6, 'total_tokens': 999})], cap=1)
    assert len(client.calls) == 1 and first['status'] == 'partial_ranking'
    assert first['C_m'] == 15 and first['exceeds_cap']
    final, client = run(audit, [('2 1', {'output_tokens': 7})], cap=1)
    assert len(client.calls) == 1 and final['status'] == 'ready'
    assert final['C_m'] == 22
    frozen = copy.deepcopy([manifest(audit, k) for k in range(3)])
    same, client = run(audit, [], cap=40)
    assert not client.calls and same == final
    assert [manifest(audit, k) for k in range(3)] == frozen
    for k in range(3):
        directory = audit / 'pools/bfcl' / f'B10_seed{k}'
        base, pool = ddpo.rows(directory / 'pool_sft.jsonl'), ddpo.rows(directory / 'pool_ddpo.jsonl')
        assert pool[:len(base)] == base
        prefs = pool[len(base):]
        assert len(prefs) == 2
        for row in prefs:
            assert set(row) == set(base[0]) | {'_rejected'}
            assert row['teacher'] == 'rank' and row['turn_index'] == 0
            assert row['response'].startswith('student') and row['_rejected'].startswith('student')
            assert row['response'] != row['_rejected']
            assert row['prompt'] == 'official ' + row['task_id']
            assert row['token_hint'] == max(len(row['response']) // 4, 1)
        m = manifest(audit, k)
        arm = m['arms']['ddpo']
        assert arm['status'] == arm['trainer_status'] == 'ready'
        assert arm['ranking_cost'] == 13 and arm['C_m'] == 22
        assert arm['remaining_budget'] == -12 and arm['over_cap_tokens'] == 12
        assert arm['pool_sha256'] == io.digest(pool)
        assert arm['teacher_call_ids'] == sorted(['purchased'] + [r['call_id'] for r in ledger(audit)])
        assert arm['ranking_call_ids'] == sorted(r['call_id'] for r in ledger(audit))
        assert m['remaining_budget'] == 1 and m['arms']['sft']['C_m'] == 9


@pytest.mark.parametrize('reply', ['', '   ', 'nonsense', '1 1', '0 3', '1 9', '1'])
@pytest.mark.parametrize('usage,expected,confidence', [
    ({'completion_tokens': 0}, 0, 'exact'),
    ({'completion_tokens': 8}, 8, 'exact'),
    ({}, None, 'estimated'),
])
def test_parse_failures_are_skipped_without_opt_in_and_charged(audit, reply, usage, expected, confidence):
    result, _ = run(audit, [(reply, usage)], cap=1)
    record = ledger(audit)[0]
    assert record['status'] == 'parse_failed'
    assert record['tokens_spent'] == (expected if expected is not None else max(1, len(reply.split())))
    assert record['confidence'] == confidence
    _, client = run(audit, [('1 2', {'completion_tokens': 1})], cap=2)
    assert len(client.calls) == 1
    assert 'official task_b' in client.calls[0][0]['content']
    assert manifest(audit)['arms']['ddpo']['skipped_parse_task_ids'] == ['task_a']
    assert result['C_m'] == 9 + record['tokens_spent']


@pytest.mark.parametrize('reply', ['', '   ', 'nonsense'])
def test_retry_only_failed_accumulates_charges_and_never_recalls_ranked(audit, reply):
    first, _ = run(audit, [(reply, {'completion_tokens': 64}),
                           ('1 2', {'completion_tokens': 63})])
    path = audit / 'bfcl/ddpo_rank_ledger.jsonl'
    original = path.read_bytes()
    old_calls = ledger(audit)
    same, client = run(audit, [])
    assert not client.calls and same == first

    result, client = run(audit, [('2 3', {'completion_tokens': 7})],
                         cap=1, retry_parse_failed=True)
    assert len(client.calls) == 1 and 'official task_a' in client.calls[0][0]['content']
    assert result['status'] == 'ready' and result['ranked_tasks'] == 2
    assert result['ranking_cost'] == 64 + 63 + 7 and result['C_m'] == 9 + 64 + 63 + 7
    assert path.read_bytes().startswith(original) and ledger(audit)[:2] == old_calls
    attempt = ledger(audit)[-1]
    assert attempt['attempt_index'] == old_calls[0]['attempt_index'] + 1
    assert len({r['call_id'] for r in ledger(audit)}) == 3
    assert ddpo.rows(audit / 'bfcl/ddpo_rank_attempts.jsonl')[-1]['call_id'] == attempt['call_id']
    for seed in range(3):
        m = manifest(audit, seed)
        arm = m['arms']['ddpo']
        assert arm['ranking_cost'] == 134 and arm['C_m'] == 143
        assert arm['rank_pairs'] == 2 and arm['skipped_parse_task_ids'] == []
        assert len(arm['ranking_call_ids']) == 3
        assert m['ranking']['new_ddpo']['ranked_tasks'] == ['task_a', 'task_b']
        pool = ddpo.rows(audit / 'pools/bfcl' / f'B10_seed{seed}/pool_ddpo.jsonl')
        assert len(pool) == arm['rows'] == 4 and arm['pool_sha256'] == io.digest(pool)
        assert [r['task_id'] for r in pool if r['teacher'] == 'rank'] == ['task_a', 'task_b']
        assert pool[-1] == old_calls[1]['preference_row']
    same, client = run(audit, [], cap=1, retry_parse_failed=True)
    assert not client.calls and same == result


def test_retry_cap_counts_new_attempts_and_pending_failures(audit):
    run(audit, [('', {'completion_tokens': 64})] * 2)
    result, client = run(audit, [('1 2', {'completion_tokens': 5})],
                         cap=1, retry_parse_failed=True)
    assert len(client.calls) == 1 and result['ranking_cost'] == 133
    assert result['status'] == 'partial_ranking' and result['pending_task_ids'] == ['task_b']
    final, client = run(audit, [('2 1', {'completion_tokens': 7})],
                        cap=1, retry_parse_failed=True)
    assert len(client.calls) == 1 and 'official task_b' in client.calls[0][0]['content']
    assert final['status'] == 'ready' and final['ranking_cost'] == 140
    assert [(r['task_id'], r['attempt_index']) for r in ledger(audit)] == [
        ('task_a', 0), ('task_b', 0), ('task_a', 1), ('task_b', 1)]


@pytest.mark.parametrize('cap', [1, 2])
def test_parse_retry_and_fresh_task_share_invocation_cap(audit, cap):
    run(audit, [('', {'completion_tokens': 64})], cap=1)
    result, client = run(audit, [('1 2', {'completion_tokens': 5})] * cap,
                         cap=cap, retry_parse_failed=True)
    assert len(client.calls) == cap
    assert result['ranking_cost'] == 64 + 5 * cap and result['C_m'] == 73 + 5 * cap
    assert result['pending_task_ids'] == (['task_b'] if cap == 1 else [])
    assert [(r['task_id'], r['attempt_index']) for r in ledger(audit)] == [
        ('task_a', 0), ('task_a', 1), ('task_b', 0)][:cap + 1]


@pytest.mark.parametrize('reply', ['', at.TeacherAPIError('HTTP 429', status_code=429)])
def test_parse_retry_buys_only_one_attempt_even_if_it_fails_again(audit, reply):
    run(audit, [('', {'completion_tokens': 64}), ('1 2', {'completion_tokens': 3})])
    result, client = run(audit, [(reply, {'completion_tokens': 8})], retry_parse_failed=True)
    assert len(client.calls) == 1 and len(ledger(audit)) == 3
    assert result['ranking_cost'] == 75 and result['pending_task_ids'] == ['task_a']
    assert result['status'] == 'partial_ranking' and ledger(audit)[-1]['attempt_index'] == 1


@pytest.mark.parametrize('status', ['legacy_empty', 'api_error'])
def test_empty_response_retried_even_without_parse_failed_status(audit, status):
    run(audit, [('', {'completion_tokens': 64}), ('1 2', {'completion_tokens': 3})])
    calls = ledger(audit)
    calls[0]['status'] = status
    calls[0]['attempt_index'] = 4  # Imported ledger may have gaps in attempt indices.
    calls[1]['response'] = ''  # A ranked result stays terminal even with empty raw text.
    io.write_rows(audit / 'bfcl/ddpo_rank_ledger.jsonl', calls)
    result, client = run(audit, [('1 3', {'completion_tokens': 2})], retry_parse_failed=True)
    assert len(client.calls) == 1 and ledger(audit)[-1]['attempt_index'] == 5
    assert result['status'] == 'ready' and result['ranking_cost'] == 69


def test_full_ledger_rebuild_uses_latest_ranked_preference_once_per_seed(audit):
    run(audit, [('1 2', {'completion_tokens': 3})] * 2)
    calls = ledger(audit)
    newer = copy.deepcopy(calls[0])
    newer.update(call_id='newer-ranking', attempt_index=2, tokens_spent=10,
                 response='3 1', best_worst=[3, 1])
    newer['preference_row'].update(response='student 2', _rejected='student 0')
    failed = dict(newer, call_id='later-failure', attempt_index=3, tokens_spent=64,
                  response='', status='parse_failed')
    failed.pop('preference_row')
    # Imported order differs from attempt order; a later failure cannot erase success.
    io.write_rows(audit / 'bfcl/ddpo_rank_ledger.jsonl', [newer, *calls, failed])
    for seed in range(3):
        directory = audit / 'pools/bfcl' / f'B10_seed{seed}'
        io.write_rows(directory / 'pool_ddpo.jsonl', [newer['preference_row']] * 3)
        m = manifest(audit, seed)
        m['arms']['ddpo'].update(ranking_cost=999, C_m=999)
        m['ranking']['new_ddpo']['ranked_tasks'] = []
        io.write_json(directory / 'manifest.json', m)
    result, client = run(audit, [], cap=1, retry_parse_failed=True)
    assert not client.calls and result['ranking_cost'] == 80 and result['C_m'] == 89
    for seed in range(3):
        directory = audit / 'pools/bfcl' / f'B10_seed{seed}'
        pool = ddpo.rows(directory / 'pool_ddpo.jsonl')
        assert pool == ddpo.rows(directory / 'pool_sft.jsonl') + [
            newer['preference_row'], calls[1]['preference_row']]
        m = manifest(audit, seed)
        arm = m['arms']['ddpo']
        assert arm['ranking_cost'] == 80 and arm['C_m'] == 89 and arm['rank_pairs'] == 2
        assert len(arm['ranking_call_ids']) == 4 and arm['skipped_parse_task_ids'] == []
        assert m['ranking']['new_ddpo']['ranked_tasks'] == ['task_a', 'task_b']


def test_empty_reply_unknown_usage_is_not_free(audit):
    run(audit, [('', {})], cap=1)
    assert ledger(audit)[0]['tokens_spent'] == 1
    assert ledger(audit)[0]['confidence'] == 'estimated'


def test_candidate_dedupe_and_all_candidates_in_prompt(audit):
    path = audit / 'bfcl/ddpo_samples_B10.jsonl'
    samples = [dict(task_id=t, sample_index=i, response=r, verified=True,
                    prompt='official ' + t, messages=[])
               for t, answers in [('task_a', ['a', 'a', 'b', 'c', 'd', 'e']),
                                  ('task_b', ['same', 'same'])]
               for i, r in enumerate(answers)]
    io.write_rows(path, samples)
    result, client = run(audit, [('<think>78 90</think>\n5 2', {'completion_tokens': 4})])
    assert result['status'] == 'ready' and len(client.calls) == 1
    assert '[5] e' in client.calls[0][0]['content']
    assert ledger(audit)[0]['preference_row']['response'] == 'e'
    assert ledger(audit)[0]['preference_row']['_rejected'] == 'b'
    assert manifest(audit)['arms']['ddpo']['fewer_than_two_distinct_task_ids'] == ['task_b']


def test_rate_limit_retries_are_bounded_and_count_toward_hard_cap(audit):
    error = at.TeacherAPIError('HTTP 429 quota', status_code=429)
    result, client = run(audit, [(error, {'completion_tokens': 0})] * 3)
    assert len(client.calls) == 3
    assert result['abort_reason'] and result['status'] == 'partial_ranking'
    assert {r['task_id'] for r in ledger(audit)} == {'task_a'}
    resumed, client = run(audit, [('1 2', {'completion_tokens': 2})] * 2)
    assert len(client.calls) == 2 and resumed['status'] == 'ready'
    assert len(ledger(audit)) == 5  # Quota recovered; prior attempts still counted.


def test_rate_limit_cannot_hide_extra_requests(audit):
    error = at.TeacherAPIError('HTTP 429', status_code=429)
    result, client = run(audit, [(error, {'completion_tokens': 0})], cap=1)
    assert len(client.calls) == len(ledger(audit)) == 1
    assert result['pending_task_ids'] == ['task_a', 'task_b']
    result, client = run(audit, [('1 2', {'completion_tokens': 3})], cap=1)
    assert len(client.calls) == 1 and result['ranking_cost'] == 3


def test_rate_limit_and_success_share_invocation_cap(audit):
    run(audit, [('', {'completion_tokens': 64})], cap=1)
    error = at.TeacherAPIError('HTTP 429', status_code=429)
    result, client = run(audit, [(error, {'completion_tokens': 1}),
                                 ('1 2', {'completion_tokens': 3})], cap=2)
    assert len(client.calls) == 2 and result['ranking_cost'] == 68
    assert [(r['task_id'], r['attempt_index']) for r in ledger(audit)] == [
        ('task_a', 0), ('task_b', 0), ('task_b', 1)]


def test_api_failure_with_usage_is_charged_and_unknown_is_explicit(audit):
    error = at.TeacherAPIError('malformed response')
    result, client = run(audit, [(error, {'completion_tokens': 11})])
    assert len(client.calls) == 1 and result['C_m'] == 20
    assert result['abort_reason']
    _, client = run(audit, [])
    assert not client.calls


def test_unknown_http_usage_is_not_zero(audit):
    result, _ = run(audit, [(at.TeacherAPIError('HTTP 402', status_code=402), {})])
    assert ledger(audit)[0]['tokens_spent'] is None
    assert result['status'] == 'incomplete_ranking_costs'
    assert not manifest(audit)['arms']['ddpo']['ranking_cost_complete']


def test_unsettled_intent_never_recalled(audit):
    io.write_rows(audit / 'bfcl/ddpo_rank_attempts.jsonl', [dict(
        call_id='interrupted-call', task_id='task_a', teacher=ddpo.TEACHER, purpose='rank')])
    client = FakeClient([])
    with pytest.raises(RuntimeError, match='unsettled ranking attempts'):
        ddpo.rank(audit, 10, 40, client)
    assert not client.calls
    arm = manifest(audit)['arms']['ddpo']
    assert arm['status'] == 'incomplete_ranking_costs'
    assert arm['unknown_ranking_call_ids'] == ['interrupted-call']


def test_interrupt_preserves_intent_and_marks_unknown_cost(audit):
    def interrupt(messages, usage):
        raise KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        ddpo.rank(audit, 10, 40, interrupt)
    assert ledger(audit) == []
    assert len(ddpo.rows(audit / 'bfcl/ddpo_rank_attempts.jsonl')) == 1
    assert manifest(audit)['arms']['ddpo']['status'] == 'incomplete_ranking_costs'
    with pytest.raises(RuntimeError, match='unsettled ranking attempts'):
        run(audit, [])


def test_noop_resume_does_not_read_missing_credentials(audit):
    run(audit, [('1 2', {'completion_tokens': 2})] * 2)
    result = ddpo.rank(audit, 10, 40, ddpo.TeacherClient('/nonexistent/key', 64))
    assert result['status'] == 'ready'


def test_tokenizer_error_preserves_response_and_unknown_charge(audit):
    def broken(text):
        raise OSError('no cached tokenizer')
    client = FakeClient([('1 2', {})])
    result = ddpo.rank(audit, 10, 1, client, token_counter=broken)
    assert result['status'] == 'incomplete_ranking_costs'
    assert ledger(audit)[0]['response'] == '1 2'
    assert ledger(audit)[0]['tokens_spent'] is None


def test_duplicate_ledger_call_is_not_double_charged(audit):
    run(audit, [('1 2', {'completion_tokens': 3})] * 2)
    ddpo.append(audit / 'bfcl/ddpo_rank_ledger.jsonl', ledger(audit)[0])
    result, client = run(audit, [], cap=1)
    assert not client.calls and result['ranking_cost'] == 6


def test_seed_mismatch_rejected_before_call(audit):
    path = audit / 'pools/bfcl/B10_seed2/manifest.json'
    m = io.read_json(path)
    m['tasks_covered'] = ['task_a']
    io.write_json(path, m)
    with pytest.raises(ValueError, match='three training seeds'):
        run(audit, [])


def test_generated_child_prompt_never_used_for_parent(audit):
    sample = dict(response='a')
    with pytest.raises(ValueError, match='official prompt'):
        ddpo.rank_context('parent', [sample, dict(response='b')],
                          [dict(task_id='gen_parent_0', prompt='wrong task')])


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps(self.payload).encode()


def test_existing_client_exposes_usage_and_disables_think_and_hidden_retries(monkeypatch, tmp_path):
    requests = []
    def open_request(request, **kwargs):
        requests.append(request)
        return FakeResponse(dict(usage={'completion_tokens': 5, 'prompt_tokens': 100},
                                 choices=[dict(message=dict(content='1 2'))]))
    monkeypatch.setattr(at.urllib.request, 'build_opener', lambda: type('Opener', (), {
        'open': staticmethod(open_request)})())
    monkeypatch.setattr(at, '_ollama_keys', lambda _: pytest.fail('must use only requested key'))
    monkeypatch.setenv('TEACHER_THINK', '1')
    monkeypatch.setenv('OLLAMA_BASE_URL', 'https://example.invalid')
    key = tmp_path / 'key'
    key.write_text('fake-key')
    client = ddpo.TeacherClient(str(key), 512)
    usage = {'stale': 999}
    assert client([dict(role='user', content='rank')], usage) == '1 2'
    assert usage == dict(completion_tokens=5, prompt_tokens=100)
    body = json.loads(requests[0].data)
    assert body['think'] is False and body['max_tokens'] == 512
    assert body['reasoning_effort'] == 'none'
    assert body['model'] == 'deepseek-v4-pro' and body['temperature'] == 0
    assert requests[0].get_header('Authorization') == 'Bearer fake-key'
    assert len(requests) == 1
    assert at.os.environ['TEACHER_THINK'] == '1'


@pytest.mark.parametrize('payload', [
    {'usage': {'completion_tokens': 7}, 'choices': []},
    {'usage': {'completion_tokens': 7}, 'choices': [{'message': {'content': None}}]},
])
def test_usage_survives_content_parser_errors(monkeypatch, payload):
    monkeypatch.setattr(at.urllib.request, 'build_opener', lambda: type('Opener', (), {
        'open': lambda *args, **kw: FakeResponse(payload)})())
    usage = {}
    with pytest.raises(at.TeacherAPIError):
        at.generate_reply(at.TeacherConfig('fake', 'fake', 'https://example.invalid', 'fake'),
                          [], usage_out=usage, max_retries=0, rotate_keys=False)
    assert usage == {'completion_tokens': 7}


def test_client_http_error_exposes_status_usage_without_retry(monkeypatch):
    calls = []
    def fail(*args, **kwargs):
        calls.append(1)
        raise urllib.error.HTTPError('https://example.invalid', 429, 'quota', {},
                                     stdio.BytesIO(b'{"usage":{"completion_tokens":0}}'))
    monkeypatch.setattr(at.urllib.request, 'build_opener', lambda: type('Opener', (), {
        'open': staticmethod(fail)})())
    usage = {}
    with pytest.raises(at.TeacherAPIError) as exc:
        at.generate_reply(at.TeacherConfig('fake', 'fake', 'https://example.invalid', 'fake'),
                          [], usage_out=usage, max_retries=0, rotate_keys=False)
    assert exc.value.status_code == 429 and len(calls) == 1
    assert usage == {'completion_tokens': 0}


def test_existing_text_only_client_interface_is_unchanged(monkeypatch):
    monkeypatch.setattr(at, '_ollama_keys', lambda _: ['fake'])
    def open_request(request, **kwargs):
        assert 'reasoning_effort' not in json.loads(request.data)
        return FakeResponse({'choices': [{'message': {'content': 'plain text'}}]})
    monkeypatch.setattr(at.urllib.request, 'build_opener', lambda: type('Opener', (), {
        'open': staticmethod(open_request)})())
    assert at.generate_reply(at.TeacherConfig('fake', 'fake', 'https://example.invalid', 'fake'), []) == 'plain text'


def test_rank_cli_retry_flag_and_default_completion_budget(monkeypatch):
    def fake_rank(audit, cap, max_calls, client, *, retry_parse_failed):
        assert audit == io.DEFAULT_OUT and cap == 13843 and max_calls == 2
        assert client.max_output_tokens == 512 and retry_parse_failed
        return {'status': 'ready'}
    monkeypatch.setattr(ddpo, 'rank', fake_rank)
    assert ddpo.main(['rank', '--max-calls', '2', '--retry-parse-failed']) == 0


def test_sampling_commands_and_isolated_environment(tmp_path, monkeypatch):
    monkeypatch.setenv('BFCL_DUMP_MESSAGES', '/forbidden/data/dump.jsonl')
    monkeypatch.setenv('REMOTE_OPENAI_TOKENIZER_PATH', '/adapter')
    monkeypatch.setenv('OPENAI_API_KEY', 'must-not-be-used')
    env = ddpo.sample_environment(tmp_path, 'GPU-fake', 9123)
    assert env['CUDA_VISIBLE_DEVICES'] == 'GPU-fake'
    assert env['BFCL_PROJECT_ROOT'] == str(tmp_path)
    assert env['PYTHONDONTWRITEBYTECODE'] == env['HF_HUB_OFFLINE'] == '1'
    assert 'BFCL_DUMP_MESSAGES' not in env and 'OPENAI_API_KEY' not in env
    assert 'REMOTE_OPENAI_TOKENIZER_PATH' not in env
    server, generate, evaluate = ddpo.sample_commands(tmp_path, 9123, 2)
    assert server[2] == 'Qwen/Qwen3.5-4B'
    assert not any('adapter' in arg or 'lora' in arg for arg in server)
    assert generate[generate.index('--temperature') + 1] == '1.0'
    assert '--skip-server-setup' in generate and '--run-ids' in generate
    assert '--include-input-log' in generate and '--partial-eval' in evaluate
    assert str(tmp_path / 'result_r2') in generate


def test_official_score_reconciliation_requires_complete_checker_output(tmp_path):
    result = tmp_path / 'result_r0/Qwen/BFCL_v4_simple_python_result.json'
    score = tmp_path / 'score_r0/Qwen/BFCL_v4_simple_python_score.json'
    io.write_rows(result, [dict(id=t, result='call()', inference_log=[
        dict(role='input', content={'formatted_prompt': 'official prompt'})])
        for t in ('simple_python_1', 'simple_python_2')])
    selection = {'simple_python': ['simple_python_1', 'simple_python_2']}
    with pytest.raises(ValueError, match='incomplete official scores'):
        ddpo.collect_repeat(tmp_path, 0, selection)
    io.write_rows(score, [dict(total_count=2, correct_count=1), dict(id='simple_python_2', valid=False)])
    samples = ddpo.collect_repeat(tmp_path, 0, selection)
    assert [s['verified'] for s in samples] == [True, False]
    assert all(s['prompt'] == 'official prompt' for s in samples)
    io.write_rows(score, [dict(total_count=2, correct_count=2), dict(id='simple_python_2', valid=False)])
    with pytest.raises(RuntimeError, match='reconciled'):
        ddpo.collect_repeat(tmp_path, 0, selection)


def test_real_audited_task_selection_is_22_official_parents():
    pools, tids = ddpo.purchased(io.DEFAULT_OUT, 13843)
    entries, selected = ddpo.task_entries(tids)
    assert len(pools[0][2]) == 94 and len(tids) == 22
    assert pools[0][1]['sealed_replay_spend'] == 13776
    assert set(entries) == set(tids) == {t for group in selected.values() for t in group}
    assert not any(t.startswith(('gen_', 'oos_')) for t in tids)
