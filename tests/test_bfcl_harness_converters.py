"""Tiny CPU archives exercise accounting and byte-preserving offline conversion."""
import json
import os

import pytest

from bfas.adapters.bfcl import BFCLAdapter, BFCLScoreError
from bfas.ledger import demo_payload, read_records
from tools.bfcl_ledger_from_harness import STEM, build_ledger, write_jsonl
from tools.bfcl_pool_from_harness import build_pool


@pytest.fixture
def archive(tmp_path, monkeypatch):
    adapter = BFCLAdapter()
    tasks = ['simple_python_0', 'simple_python_1', 'memory_kv_0', 'memory_kv_1']
    messages = [{'role': 'user', 'content': 'Find 東京'}]
    functions = [{'name': 'places.find', 'parameters': {'type': 'dict', 'properties': {}}}]
    adapter._entries = {t: dict(id=t, question=[messages], function=functions) for t in tasks}
    adapter._categories = {t: t.rsplit('_', 1)[0] for t in tasks}
    adapter._memory_prereqs = {t: ['memory_kv_prereq_0'] for t in tasks[2:]}
    monkeypatch.setattr(adapter, '_render', lambda m, f: json.dumps([m, f], ensure_ascii=False))
    monkeypatch.setattr(adapter, '_official_pass', lambda *a, **k: pytest.fail('teacher call'))
    merged = tmp_path/f'result_{STEM}'
    split = tmp_path/'split.json'
    split.write_text(json.dumps(dict(seed=50, demand=tasks, calibration=['protected_0'])))
    results = [dict(id=t, result='answer', output_token_count=cost)
               for t, cost in zip(tasks, [17, 11, [[2, 3]], [[5]]])]
    results.append(dict(id='memory_kv_prereq_0', result='write', output_token_count=[[7, 8], [9]]))
    for attempt in [1, 2, 3]:
        directory = tmp_path/f'result_{STEM}_a{attempt}'
        for cat in ['simple_python', 'memory_kv']:
            group = [r for r in results if r['id'].startswith(cat)]
            path = directory/f'BFCL_v4_{cat}_result.json'
            write_jsonl(path, group)
            os.utime(path, (1700000000 + attempt, 1700000000 + attempt))
            passed = [t for t in tasks if t.startswith(cat)] if attempt == 1 else []
            failed = [dict(id=t, valid=False) for t in tasks if t.startswith(cat) and t not in passed]
            write_jsonl(tmp_path/f'score_{STEM}_a{attempt}'/f'BFCL_v4_{cat}_score.json',
                        [dict(total_count=2, correct_count=len(passed)), *failed])
        if attempt == 1:
            write_jsonl(merged/'BFCL_v4_all_result.json', results[:-1])
    demo = demo_payload(adapter.demo_from_result(results[0], 0))
    cache = tmp_path/'demos.json'
    cache.write_text(json.dumps({tasks[0]: dict(task_id=tasks[0], raw=dict(checker_verified=True, attempt=1), **demo)}))
    return tmp_path, adapter, split, cache, merged


def convert(archive):
    root, adapter, split, cache, merged = archive
    rows, provenance = build_ledger(root, split, cache, adapter=adapter)
    ledger = root/'ledger.jsonl'
    write_jsonl(ledger, rows)
    return rows, provenance, ledger


def test_exact_nested_usage_shared_prereqs_verdicts_and_pool(archive):
    root, adapter, split, cache, merged = archive
    rows, provenance, ledger = convert(archive)
    assert len(read_records(ledger)) == 12
    assert [r['tokens_spent'] for r in rows[:4]] == [17, 11, 29, 5]
    assert sum(r['tokens_spent'] for r in rows) == 3 * 62
    assert [r['attempt_index'] for r in rows] == [0]*4 + [1]*4 + [2]*4
    assert [r['temperature'] for r in rows] == [0.001]*4 + [0.7]*8
    assert all(r['cost_confidence'] == 'exact' for r in rows)
    assert sum(r['verified'] for r in rows) == 4
    assert rows[0]['timestamp'] == '2023-11-14T22:13:21+00:00'
    assert rows[0]['provenance']['demo_source'] == str(cache)
    assert rows[1]['demo'] == demo_payload(adapter.demo_from_result(
        dict(id='simple_python_1', result='answer'), 0))
    assert all('demo' not in r for r in rows[4:])
    assert provenance['adapter_rerun']['extra_calls'] == 'unknown'
    assert len(provenance['verified_without_training_turns']) == 2
    pool, manifest = build_pool(merged, ledger, cache, adapter=adapter)
    assert len(pool) == 2 and len(manifest['excluded']) == 2
    assert pool[0]['response'] == rows[0]['demo']['turns'][0]['target']
    assert pool[0]['_render_context'] == rows[0]['demo']['turns'][0]['context']


def test_missing_usage_is_lower_bound_and_adapter_substitution_is_estimated(archive):
    root, adapter, split, cache, merged = archive
    path = root/f'result_{STEM}_a2/BFCL_v4_memory_kv_result.json'
    results = [json.loads(l) for l in path.read_text().splitlines()]
    del results[0]['output_token_count']
    results[0]['result'] = 'Error during inference: quota exceeded'
    write_jsonl(path, results)
    demos = json.loads(cache.read_text())
    demos['simple_python_0']['turns'][0]['target'] = 'different paid rerun response'
    cache.write_text(json.dumps(demos))
    rows, provenance, ledger = convert(archive)
    assert rows[0]['tokens_spent'] == 17 and rows[0]['cost_confidence'] == 'estimated'
    assert rows[6]['tokens_spent'] == 24 and rows[6]['cost_confidence'] == 'estimated'
    assert provenance['missing_usage_attempts'][0]['result_ids'] == ['memory_kv_0']
    pool, _ = build_pool(merged, ledger, cache, adapter=adapter)
    assert pool[0]['response'] == 'different paid rerun response'


@pytest.mark.parametrize('cost', [-1, True, 1.5])
def test_reject_invalid_token_counts(archive, cost):
    root, adapter, split, cache, merged = archive
    path = root/f'result_{STEM}_a1/BFCL_v4_simple_python_result.json'
    rows = [json.loads(l) for l in path.read_text().splitlines()]
    rows[0]['output_token_count'] = cost
    write_jsonl(path, rows)
    with pytest.raises(ValueError, match='token count'):
        convert(archive)


def test_reject_duplicate_and_unattributed_results(archive):
    root, adapter, split, cache, merged = archive
    path = root/f'result_{STEM}_a1/extra_result.json'
    write_jsonl(path, [dict(id='simple_python_0')])
    with pytest.raises(ValueError, match='duplicate'):
        convert(archive)
    write_jsonl(path, [dict(id='memory_kv_prereq_unowned', output_token_count=9)])
    with pytest.raises(ValueError, match='unattributed'):
        convert(archive)


def test_reject_incomplete_scores_and_unpaid_or_changed_pool(archive):
    root, adapter, split, cache, merged = archive
    rows, _, ledger = convert(archive)
    path = merged/'BFCL_v4_all_result.json'
    results = [json.loads(l) for l in path.read_text().splitlines()]
    results[0]['result'] = 'tampered'
    write_jsonl(path, results)
    with pytest.raises(ValueError, match='merged result differs'):
        build_pool(merged, ledger, cache, adapter=adapter)
    write_jsonl(path, results[:-1])
    with pytest.raises(ValueError, match='verified ids differ'):
        build_pool(merged, ledger, cache, adapter=adapter)
    score = root/f'score_{STEM}_a1/BFCL_v4_simple_python_score.json'
    write_jsonl(score, [dict(total_count=2, correct_count=1)])
    with pytest.raises(BFCLScoreError, match='reconciled'):
        convert(archive)
    score.unlink()
    rows, provenance, _ = convert(archive)
    assert not rows[0]['verified'] and not rows[1]['verified']
    assert len(provenance['missing_score_attempts']) == 2


def test_quota_error_cannot_be_a_demo_even_when_checker_passes_it(archive):
    root, adapter, split, cache, merged = archive
    path = root/f'result_{STEM}_a1/BFCL_v4_simple_python_result.json'
    results = [json.loads(l) for l in path.read_text().splitlines()]
    results[0]['result'] = 'Error during inference: OpenAI token quota is already exceeded'
    del results[0]['output_token_count']
    write_jsonl(path, results)
    rows, _, _ = convert(archive)
    assert rows[0]['provenance']['official_score_verified'] is True
    assert rows[0]['verified'] is False and 'demo' not in rows[0]
    assert rows[0]['cost_confidence'] == 'estimated'


def test_missing_web_score_is_not_inferred_from_merged_membership(archive):
    root, adapter, split, cache, merged = archive
    (root/f'score_{STEM}_a1/BFCL_v4_memory_kv_score.json').unlink()
    _, _, ledger = convert(archive)
    pool, manifest = build_pool(merged, ledger, cache, adapter=adapter)
    assert len(pool) == 2
    assert all('positive reconciled score' in item['reason'] for item in manifest['excluded'])
