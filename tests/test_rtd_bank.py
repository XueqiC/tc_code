"""Historical ingestion: hashes, exclusions, complete packages, frozen protocol."""
import hashlib
import json
from pathlib import Path
import sys

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    sys.path.insert(0, str(path))

from bfas.cc_pairs import digest, render_truth
from bfas.rtd.bank import build_bfcl_bank, parent_fold
from bfas.rtd.broker import SealedReplayBroker
from bfas.rtd.ledger import Ledger
from bfas.rtd.selector import StudentSnapshot
from bfas.rtd.caps import RequestLimits, public_cap, limits_from_metadata


def write(path, value, jsonl=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r) + '\n' for r in value) if jsonl else json.dumps(value))


def renderer(question, functions):
    return json.dumps([question, functions], sort_keys=True)


@pytest.fixture
def archive(tmp_path):
    root = tmp_path / 'archive'
    functions = [{"name": "call", "parameters": {"type": "dict", "properties": {"arg": {"type": "integer"}}, "required": ["arg"]}}]
    parent = {"id": "simple_python_1", "question": [[{"role": "user", "content": "seed"}]], "function": functions}
    rows = [{"id": "gen_simple_python_1_0", "seed_task": parent['id'],
             "question": [[{"role": "user", "content": question}]], "function": functions,
             "ground_truth": [{"call": {"arg": [1]}}], "seed_category": "simple_python"}
            for question in ['protected generated question', 'legal generated question']]
    oos = {**rows[1], "id": "oos_simple_python_1_0", "question": [[{"role": "user", "content": "another question"}]], "ground_truth": []}
    data = root / 'data/bfcl_sft'
    write(data / 'gen_pool_v3.jsonl', rows, True)
    write(data / 'gen_oos.jsonl', [oos], True)
    event = dict(task_id=rows[1]['id'], prompt=renderer(rows[1]['question'], functions),
                 response=render_truth(rows[1]['ground_truth'], functions), teacher="teacher_authored_gt",
                 _event_dU=99, _rejected='not source-policy sampling')
    write(data / 'pool_events_pref_v3t.jsonl', [event, event], True)
    write(data / 'demos_ds.json', {parent['id']: {'teacher_result': 'demo'}})
    write(data / 'calibration_ids.json', {'hashes': [hashlib.sha1(renderer(rows[0]['question'], functions).encode()).hexdigest()[:16]]})
    write(data / 'calibration_ids_official_v2.json', {'ids': ['official_protected']})
    write(root / 'configs/support_split.json', {'support': [parent['id']], 'calibration': ['not_support']})
    write(root / 'configs/bfcl_support_split.json', {'demand': [parent['id']], 'calibration': ['not_support']})
    write(root / 'results/analysis/bfcl_teacher_tokens.json', dict(
        demos_official_demand={'total_output_tokens': 17}, generation_total={'estimate': 35},
        generation_pools={'gen_pool_v3.jsonl': {'estimate': 30}, 'gen_oos.jsonl': {'estimate': 5}}))
    raw = root / 'envs/bfcl/gorilla/berkeley-function-call-leaderboard/result_demos_deepseek_v4_pro_FC_a1/model/result_result.json'
    write(raw, [{'id': parent['id'], 'result': 'FAILED ATTEMPT IS STILL EVIDENCE', 'output_token_count': 17, 'input_token_count': 42}], True)
    return root, {parent['id']: parent}, rows


def test_content_hash_exclusion_reused_ids_event_single_charging_and_exact_costs(archive, tmp_path):
    root, entries, rows = archive
    out = tmp_path / 'bank'
    summary = build_bfcl_bank(root, out, entries=entries, renderer=renderer)
    assert summary['m'] == 1 and summary['demand_attempts'] == 1
    assert summary['total_archived_packages'] == 4  # one real attempt, not three fabricated ones
    assert summary['available_packages'] == 3
    assert summary['historical_demo_output_tokens_exact'] == 17
    assert summary['event_rows'] == summary['event_alias_rows'] == 2
    assert summary['reused_generator_ids'] == 1
    h = summary['parents'][0]['parent_hash']
    assert summary['parents'][0]['fold'] == parent_fold(h)
    payloads = [json.loads(p.read_text()) for p in (out / 'sealed').glob('*.json')
                if p.stem not in {'audit', 'integrity', 'event_aliases'}]
    generated = [p for p in payloads if p['provenance']['kind'] == 'generator_item']
    assert sum(p['cost'] for p in generated) == 35
    assert all(p['cost_confidence'] == 'estimated' and p['usage']['output_tokens'] is None for p in generated)
    assert sum(len(p['provenance']['event_aliases']) for p in generated) == 2
    ledger = Ledger(1000000)
    broker = SealedReplayBroker(out, ledger, inner_parent_hashes={h})
    candidates = broker.list_candidates(StudentSnapshot('student', frozenset({h})), (), ledger.remaining)
    assert len(candidates) == 3
    assert all(c.features.projection is None for c in candidates)
    acquired = [broker.acquire(c.query_id) for c in candidates]
    assert any('FAILED ATTEMPT' in p.behaviors[0].text for p in acquired)
    assert not any('protected generated' in p.behaviors[0].state.prompt for p in acquired)
    # No generated question/response enters public pre-purchase state features.
    assert len({c.state_hash for c in candidates}) == 1
    assert ledger.spent == sum(p.cost for p in acquired)
    assert len([e for e in ledger.events if e['kind'] == 'reveal']) == 3


def test_modified_exact_cost_archive_fails_closed(archive, tmp_path):
    root, entries, _ = archive
    historical = root / 'results/analysis/bfcl_teacher_tokens.json'
    value = json.loads(historical.read_text()); value['demos_official_demand']['total_output_tokens'] += 1
    write(historical, value)
    with pytest.raises(ValueError, match='total differs'):
        build_bfcl_bank(root, tmp_path / 'bank', entries=entries, renderer=renderer)
    assert not (tmp_path / 'bank').exists()


def test_frozen_config_preserves_every_section_14_value():
    spec = (ROOT / 'docs/RTD_END_TO_END_METHOD_AND_EXECUTION_V1.md').read_text()
    template = yaml.safe_load(spec.split('## 14.')[1].split('```yaml\n')[1].split('```')[0])
    actual = yaml.safe_load((ROOT / 'configs/rtd/v1_bfcl.yaml').read_text())
    assert all(actual[k] == v for k, v in template.items())
    assert actual['mode'] == 'sealed_replay' and actual['meta_tasks_per_feedback'] == 4
    assert actual['new_teacher_calls'] is False


def test_configuration_cap_product_and_class_fallback_provenance():
    cap, provenance = public_cap('demo_attempt', limits=RequestLimits(2048, max_actions=2, max_attempts=3),
                                  evidence=[{'path': 'archived request configuration'}])
    assert cap == 12288 and json.loads(provenance)['basis'] == 'archived_configuration'
    cap, provenance = public_cap('generator_item', limits=RequestLimits(4096))
    assert cap == 4096 and json.loads(provenance)['limits']['max_attempts'] == 1
    cap, provenance = public_cap('generator_item')
    assert cap == 8192 and json.loads(provenance)['basis'] == 'class_uniform_public_fallback'
    with pytest.raises(ValueError, match='finite positive'):
        RequestLimits(2048, max_attempts=None)
    metadata = {'metadata': {'request_parameters': {'max_completion_tokens': 1024, 'max_actions': 2, 'max_retries': 2}},
                'output_token_count': 99999, 'result': {'max_tokens': 1}}
    assert limits_from_metadata(metadata, 'demo_attempt').cap == 6144
    metadata['output_token_count'] = 3
    assert limits_from_metadata(metadata, 'demo_attempt').cap == 6144
    assert limits_from_metadata({'result': {'max_tokens': 5}, 'function': {'max_tokens': 7}}, 'demo_attempt') is None
    assert limits_from_metadata({'request_config': {'max_tokens': 4096}}, 'demo_attempt') is None
    assert limits_from_metadata({'request_config': {'max_tokens': 4096}}, 'generator_item').cap == 4096


def test_bank_publishes_configuration_provenance_and_affordability_without_hidden_usage(archive, tmp_path):
    root, entries, _ = archive
    out = tmp_path / 'cap_bank'
    summary = build_bfcl_bank(root, out, entries=entries, renderer=renderer)
    public = json.loads((out / 'public/requests.json').read_text())
    for record in public:
        provenance = json.loads(record['spec']['cap_provenance'])
        assert provenance['basis'] == 'class_uniform_public_fallback'
        assert provenance['limits'] is None
        assert 'usage' not in provenance and 'response' not in provenance
    caps = sum(r['spec']['cost_upper_bound'] for r in public if r['unavailable_reason'] is None)
    assert [p['budget'] for p in summary['budget_affordability']] == [caps * f // 100 for f in (10, 25, 50)]
    assert [p['affordable_packages'] for p in summary['budget_affordability']] == [2, 2, 2]


def test_recovered_request_config_drives_bank_cap_and_affordability(archive, tmp_path):
    root, entries, _ = archive
    path = next((root / 'envs').rglob('*_result.json'))
    row = json.loads(path.read_text())
    row['metadata'] = {'request_parameters': {'max_completion_tokens': 17, 'max_retries': 0}}
    write(path, [row], True)
    out = tmp_path / 'configured'
    summary = build_bfcl_bank(root, out, entries=entries, renderer=renderer)
    records = json.loads((out / 'public/requests.json').read_text())
    demo = next(r for r in records if r['spec']['cost_confidence'] == 'exact')
    assert demo['spec']['cost_upper_bound'] == 17
    assert json.loads(demo['spec']['cap_provenance'])['basis'] == 'archived_configuration'
    assert summary['budget_affordability'][-1]['affordable_packages'] == 3
