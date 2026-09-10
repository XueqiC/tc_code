"""C26-B immutable verified fixture -> configured-student v1.1 bank."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path

from test_rtd_alfworld_support import support_archive
from test_rtd_alfworld_state import FakeStepper, make_payload, render
from test_rtd_v11_webshop import Tokenizer
from bfas.rtd.benchmarks import alfworld_bank as bank
from bfas.rtd.benchmarks.alfworld_support import verify_package, seal_verified_bank, prompt_messages
from bfas.rtd.benchmarks.alfworld_bank_v11 import convert_alfworld_bank
from bfas.rtd.bank_build import validate_state_certificate
from bfas.rtd.persistence import file_hash
from bfas.rtd.cli import load_config


def test_verified_v10_bank_converts_without_mutation(support_archive, tmp_path):
    root, support = support_archive
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
    out = tmp_path/'v11'
    summary = convert_alfworld_bank(source, out, config=config, tokenizer=Tokenizer())
    assert summary['recorded_bank_usage'] == payload['cost'] == 40
    cert = validate_state_certificate(out, benchmark='alfworld', student=config['student'])
    assert cert['core']['class_caps'] == {'alf_demo_episode': 64}
    converted = json.loads((out/f'sealed/{q}.json').read_text())
    assert len(converted['behaviors']) == len(verified['behaviors']) == 2
    assert converted['cost'] == verified['cost'] and converted['cost_confidence'] == verified['cost_confidence']
    for new, old in zip(converted['behaviors'], verified['behaviors']):
        assert new['text'] == old['text']
        assert new['state']['history_json'] == old['state']['history_json']
        messages = prompt_messages(json.loads(old['state']['task_json']), json.loads(old['state']['history_json']))
        assert new['state']['prompt'] == Tokenizer().apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    assert before == {str(p.relative_to(source)): file_hash(p) for p in source.rglob('*') if p.is_file()}
