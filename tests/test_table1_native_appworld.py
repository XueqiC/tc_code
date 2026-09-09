"""Real AppWorld tokenizer, sealed ownership and deployment prompt regression."""
from copy import deepcopy

import pytest

from tools import table1_common as io
from tools.table1_appworld_format import (
    APPWORLD_NATIVE_ROW_FORMAT, THINK_OPEN, THINK_REMAINDER, base_tokenizer,
    evaluation_prompt, rejected_emission, render_rows, teacher_emission,
)
from tools.table1_pool_from_sealed import materialize
from tools.table1_validate_appworld import (
    purchased_prefix_checks, recorded_official_checks, trainer_checks,
)


@pytest.fixture(autouse=True)
def offline_cpu(monkeypatch):
    import socket
    import torch
    from transformers import AutoModelForCausalLM

    def forbidden(*args, **kwargs):
        pytest.fail('network, model weights and CUDA forbidden')

    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    monkeypatch.setenv('HF_HUB_OFFLINE', '1')
    monkeypatch.setenv('TRANSFORMERS_OFFLINE', '1')
    monkeypatch.setattr(socket.socket, 'connect', forbidden)
    monkeypatch.setattr(torch.cuda, '_lazy_init', forbidden)
    monkeypatch.setattr(AutoModelForCausalLM, 'from_pretrained', forbidden)


def test_recorded_base_dev_input_log_template_equality():
    checks = recorded_official_checks(base_tokenizer())
    assert len(checks) == 3
    assert len({c['prompt_sha256'] for c in checks}) == 3


def test_purchased_trace_and_every_pool_real_trainer():
    snapshots, checks = purchased_prefix_checks(base_tokenizer())
    assert len(checks) == 368
    results = trainer_checks(base_tokenizer(), snapshots)
    assert len(results) == 21 and sum(r['rows'] for r in results) == 5535


def test_emission_boundary_and_historical_turns():
    tokenizer = base_tokenizer()
    messages = [dict(role='user', content='task')]
    prompt = evaluation_prompt(messages)
    assert prompt.endswith(THINK_OPEN)
    response = teacher_emission('```python\nprint(1)\n```')
    assert response == THINK_REMAINDER + '```python\nprint(1)\n```'
    # Evaluator decode().strip() preserves the closing tag. The next-turn
    # template removes that reasoning block, as for the sealed code history.
    emitted = tokenizer.decode(tokenizer(response, add_special_tokens=False)['input_ids'],
                               skip_special_tokens=True).strip()
    a = messages + [dict(role='assistant', content=emitted), dict(role='user', content='observation')]
    b = messages + [dict(role='assistant', content='```python\nprint(1)\n```'), dict(role='user', content='observation')]
    assert evaluation_prompt(a) == evaluation_prompt(b)
    assert rejected_emission('reasoning\n</think>\n\ninvalid code') == 'reasoning\n</think>\n\ninvalid code'
    with pytest.raises(ValueError, match='framing'):
        teacher_emission('<think>already framed')
    with pytest.raises(ValueError, match='EOS'):
        rejected_emission('bad<|im_end|>')


def test_native_materialization_idempotent_and_immutable(tmp_path):
    directory = io.DEFAULT_OUT / 'appworld'
    original = directory / 'acquired_B22633_seed0.json'
    acquisition = tmp_path / original.name
    acquisition.write_bytes(original.read_bytes())
    before = acquisition.read_bytes()
    root = tmp_path / 'pools'
    manifest = materialize(directory, acquisition, root)
    assert acquisition.read_bytes() == before
    assert manifest['row_format'] == APPWORLD_NATIVE_ROW_FORMAT
    output = root / 'appworld/B22633_seed0'
    hashes = {p.name: io.file_hash(p) for p in output.iterdir()}
    materialize(directory, acquisition, root)
    assert hashes == {p.name: io.file_hash(p) for p in output.iterdir()}
    manifest['arms']['ddpo']['C_m'] += 1
    io.write_json(output / 'manifest.json', manifest)
    with pytest.raises(ValueError, match='manifest/accounting'):
        materialize(directory, acquisition, root)


def test_cached_rankings_and_prefix_tampering():
    directory = io.DEFAULT_OUT / 'appworld'
    acquisition = io.read_json(directory / 'acquired_B22633_seed0.json')
    ledger = io.read_json(directory / 'ledger.json')
    ranks = ledger['cached_rankings']
    assert len(ranks) == 31
    tasks = set(acquisition['tasks_covered'])
    assert sum(r['task_id'] in tasks for r in ranks) == 7
    assert all(r['ranking_call_id'] is None and r['recorded_cost'] is None for r in ranks)
    snapshot = io.read_sealed(directory, acquisition['purchased_ids'][0], set(acquisition['purchased_ids']))
    broken = deepcopy(snapshot)
    broken['rows'][1]['messages'][-2]['content'] = 'different teacher output'
    from tools.bfcl_demo_pool import serialize
    broken['rows'][1]['prompt'] = serialize(broken['rows'][1]['messages'])
    with pytest.raises(ValueError, match='consecutive'):
        render_rows(broken)
