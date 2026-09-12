"""C26-B immutable verified fixture -> configured-student v1.1 bank."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import pytest
import yaml

from test_rtd_alfworld_support import support_archive
from test_rtd_alfworld_state import FakeStepper, make_payload, render
from test_rtd_v11_webshop import Tokenizer
from bfas.rtd.benchmarks import alfworld_bank as bank
from bfas.rtd.benchmarks.alfworld_support import (
    ALFWorldSupport, _signed, verify_package, seal_verified_bank, prompt_messages,
)
from bfas.rtd.benchmarks.alfworld_bank_v11 import convert_alfworld_bank
from bfas.rtd.benchmarks.alfworld_state import canonical_hash
from bfas.rtd.bank_build import validate_state_certificate
from bfas.rtd.broker import SealedReplayBroker
from bfas.rtd.ledger import Ledger
from bfas.rtd.selector import StudentSnapshot
from bfas.rtd.transport import FullState
from bfas.rtd.persistence import file_hash
from bfas.rtd.cli import load_config
from bfas.rtd.conventions import fold_roles


@pytest.mark.parametrize('builder', ['direct', 'alfworld_cli', 'shared_cli'])
@pytest.mark.parametrize('collector_identity', [False, True])
def test_verified_v10_bank_converts_without_mutation(support_archive, tmp_path, monkeypatch, builder, collector_identity):
    root, support = support_archive
    frozen = deepcopy(support)
    if collector_identity:
        from tools.alfworld_teacher_pool import collection_support
        support = collection_support(support)
    tid = support['training_task_ids'][0]
    request = support['tasks'][tid]['request']
    payload = make_payload(request)
    q = payload['query_id']
    verified = verify_package(payload, request, FakeStepper(request), render)
    archive = bank.Archive([bank.public_record(q, request)], {q: payload}, {q: request}, [], {'source_files': []})
    source = tmp_path/'v10'
    seal_verified_bank(source, archive, support, {q: verified}, {tid: verified['verification']['states'][0]})
    before = {str(p.relative_to(source)): file_hash(p) for p in source.rglob('*') if p.is_file()}
    config = load_config(Path(__file__).resolve().parents[1]/'configs/rtd/v1_1_alfworld.yaml')
    frozen_path = tmp_path/'frozen_support.json'
    frozen_path.write_text(json.dumps(frozen))
    config['support_manifest'] = str(frozen_path)
    config_path = tmp_path/'config.yaml'
    config_path.write_text(yaml.safe_dump(config))
    out = tmp_path/'v11'
    if builder == 'direct':
        summary = convert_alfworld_bank(source, out, config=config, tokenizer=Tokenizer())
    else:
        from bfas.rtd.benchmarks import alfworld_bank_v11
        from tools import rtd_alfworld_bank_v11, rtd_bank_build
        monkeypatch.setattr(alfworld_bank_v11, 'load_student_tokenizer', lambda config: Tokenizer())
        # The shared entrypoint requires the exact archived ledger.
        ledger = tmp_path/'ledger.jsonl'
        ledger.write_text(json.dumps(verified['historical_response'])+'\n')
        args = ['--out', str(out), '--config', str(config_path), '--ledger', str(ledger)]
        if builder == 'alfworld_cli':
            rtd_alfworld_bank_v11.main(['--source', str(source), *args])
        else:
            rtd_bank_build.main(['--benchmark', 'alfworld', '--pool', str(source), *args])
        summary = json.loads((out/'sealed/audit_v11.json').read_text())
    assert summary['recorded_bank_usage'] == payload['cost'] == 40
    cert = validate_state_certificate(out, benchmark='alfworld', student=config['student'])
    assert cert['core']['class_caps'] == {'alf_demo_episode': 64}
    converted_support = json.loads((out/'public/support.json').read_text())
    assert converted_support == frozen
    binding = cert['core']['inputs']['support_binding']
    assert binding['source_environment'] == support['environment']
    assert binding['frozen_support_manifest_hash'] == frozen['manifest_hash']
    assert binding['changed_request_identities'] == (len(support['tasks']) if collector_identity else 0)
    assert all(p['fold'] == int(h, 16) % 2 for h, p in converted_support['parents'].items())
    roles = fold_roles(converted_support['parents'])
    assert set(roles['0']['inner_parent_groups'] + roles['1']['inner_parent_groups']) == set(support['parents'])
    assert not set(support['protected_parent_hashes']) & set(support['parents'])
    converted = json.loads((out/f'sealed/{q}.json').read_text())
    assert len(converted['behaviors']) == len(verified['behaviors']) == 2
    assert converted['cost'] == verified['cost'] and converted['cost_confidence'] == verified['cost_confidence']
    archived = json.loads((source/f'sealed/{q}.json').read_text())
    assert converted['verification'] == archived['verification']
    assert converted['historical_response'] == archived['historical_response']
    assert converted['request_state_hash'] == frozen['tasks'][tid]['request_hash']
    for new, old in zip(converted['behaviors'], verified['behaviors']):
        assert new['text'] == old['text']
        assert new['state']['history_json'] == old['state']['history_json']
        assert json.loads(new['state']['task_json']) == frozen['tasks'][tid]['request']
        messages = prompt_messages(json.loads(old['state']['task_json']), json.loads(old['state']['history_json']))
        assert new['state']['prompt'] == Tokenizer().apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    reset = json.loads((out/'public/reset_states.json').read_text())[tid]
    assert reset == converted['behaviors'][0]['state']
    # A support-only patch would pass certificate/preflight but fail these
    # production guards when a purchased teacher prefix is first consumed.
    guard = ALFWorldSupport(frozen)
    parent = frozen['tasks'][tid]['parent_hash']
    round_number = 1 if parent in guard.parent_hashes(1) else 2
    broker = SealedReplayBroker(out, Ledger(100), inner_parent_hashes={parent})
    candidates = broker.list_candidates(StudentSnapshot('fixture', frozenset({parent})), (), 100)
    assert [c.query_id for c in candidates] == [q]
    package = broker.acquire(q)
    states = [FullState(**b['state']) for b in converted['behaviors']]
    assert guard.guard_states(states, round_number, owned_packages={q: package}) == tuple(states)
    assert before == {str(p.relative_to(source)): file_hash(p) for p in source.rglob('*') if p.is_file()}


@pytest.mark.parametrize('change', ['world', 'inventory', 'calibration', 'order', 'lineage', 'fold'])
def test_conversion_rejects_nonidentity_support_changes(support_archive, tmp_path, change):
    from bfas.rtd.benchmarks.alfworld_bank_v11 import frozen_conversion_support
    from tools.alfworld_teacher_pool import collection_support
    _, frozen = support_archive
    source = collection_support(frozen)
    tid = source['training_task_ids'][0]
    if change == 'world':
        source['tasks'][tid]['request']['goal'] = 'a different goal'
        source['tasks'][tid]['request_hash'] = canonical_hash(source['tasks'][tid]['request'])
    elif change == 'inventory':
        del source['tasks'][tid]
    elif change == 'calibration':
        source['calibration_task_ids'] = []
    elif change == 'order':
        source['training_task_ids'].reverse()
    elif change == 'fold':
        source['tasks'][tid]['fold'] ^= 1
    else:
        source['environment']['historical_environment_hash'] = '0' * 64
    path = tmp_path/'frozen.json'
    path.write_text(json.dumps(frozen))
    with pytest.raises(ValueError, match='frozen support'):
        frozen_conversion_support(_signed(source), path)
