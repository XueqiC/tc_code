"""Real sealed-pool rendering + load_pool/encode, entirely offline on CPU."""
from collections import Counter
import json

import pytest

from tools import table1_common as io
from tools.bfcl_demo_pool import serialize
from tools.table1_pool_from_sealed import build_pools, manifest_without_row_format, materialize
from tools.table1_row_format import (
    LEGACY_ROW_FORMAT as ROW_FORMAT, TEMPLATE_MARKERS, bfcl_response, read_bank_payload,
    render_legacy_row, render_package,
)


POOL_ROOT = io.DEFAULT_OUT / 'pools'
POOL_PATHS = sorted(POOL_ROOT.glob('*/*/pool_*.jsonl'))


@pytest.fixture(scope='module')
def tiny_tokenizer(tmp_path_factory):
    """Save/load a real local tokenizer, with Qwen-like special-token boundaries.

    Byte-level BPE needs no training, model weights, downloads or API. A think
    prefix emitted by the tokenizer belongs to the prompt, never to the labels.
    """
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    import appworld_train as trainer

    special = ['<eos>', '<pad>', '<unk>', '<|im_start|>', '<|im_end|>']
    vocab = {token: i for i, token in enumerate(special + sorted(pre_tokenizers.ByteLevel.alphabet()))}
    backend = Tokenizer(models.BPE(vocab=vocab, merges=[], unk_token='<unk>'))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, eos_token='<eos>', pad_token='<pad>', unk_token='<unk>',
        additional_special_tokens=special[3:], model_max_length=10**9,
    )
    tokenizer.chat_template = (
        "{% for message in messages %}"
        "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}"
        "{% endfor %}{% if add_generation_prompt %}"
        "{{ '<|im_start|>assistant\n<think>\n' }}{% endif %}"
    )
    path = tmp_path_factory.mktemp('table1-tiny-tokenizer')
    tokenizer.save_pretrained(path)
    return trainer.load_tokenizer(str(path))


@pytest.mark.parametrize('path', POOL_PATHS, ids=lambda p: str(p.relative_to(POOL_ROOT)))
def test_every_pool_real_load_and_encode_cpu(path, tiny_tokenizer, monkeypatch):
    import appworld_train as trainer
    import torch

    def forbidden(*args, **kwargs):
        pytest.fail('GPU/model loading is forbidden in a pool encoding test')

    monkeypatch.setattr(torch.cuda, '_lazy_init', forbidden)
    monkeypatch.setattr(trainer.AutoModelForCausalLM, 'from_pretrained', forbidden)
    monkeypatch.setenv('AW_MAX_PROMPT_TOKENS', '4096')
    monkeypatch.setenv('AW_TRUNCATE_SIDE', 'tail')  # scripts/bfclv2_hpg.slurm
    manifest = io.read_json(path.parent/'manifest.json')
    arm = path.stem.removeprefix('pool_')
    if manifest['arms'][arm]['rows'] == 0:
        assert path.read_bytes() == b''
        with pytest.raises(AssertionError, match='pool is empty'):
            trainer.load_pool(path)
        return
    rows = trainer.load_pool(path)
    assert len(rows) == manifest['arms'][arm]['rows']
    for index, row in enumerate(rows):
        assert row['_pool_index'] == index
        assert row['messages'] and row['prompt'] == serialize(row['messages'])
        assert not any(marker in row['prompt'] for marker in TEMPLATE_MARKERS)
        # Independently construct the expected prompt and target boundaries.
        prompt_text = tiny_tokenizer.apply_chat_template(
            row['messages'], add_generation_prompt=True, tokenize=False)
        assert prompt_text.endswith('<think>\n')
        prompt_ids = tiny_tokenizer(prompt_text, add_special_tokens=False)['input_ids'][-4096:]
        response_ids = tiny_tokenizer(row['response'], add_special_tokens=False)['input_ids']
        assert response_ids  # EOS alone must not satisfy this check.
        target = response_ids[:trainer.MAX_RESPONSE_TOKENS] + [tiny_tokenizer.eos_token_id]
        # A stale prompt cannot override messages on the trainer's real path.
        input_ids, labels = trainer.encode(
            tiny_tokenizer, dict(row, prompt='must not tokenize this fallback'), device='cpu')
        assert input_ids.device.type == labels.device.type == 'cpu'
        assert input_ids.tolist() == [prompt_ids + target]
        assert labels.tolist() == [[-100] * len(prompt_ids) + target]
        assert (labels == -100).sum().item() == len(prompt_ids)


def test_all_expected_pool_files_are_covered():
    assert len(POOL_PATHS) == 84
    assert {p.parent.parent.name for p in POOL_PATHS} == set(io.BENCHMARKS)


@pytest.mark.parametrize('cap,legacy_count,archive_exact', [(13843, 22, 15), (5537, 7, 3)])
def test_bfcl_legacy_rows_unchanged_and_exact_old_archive(cap, legacy_count, archive_exact):
    old = [r for _, r in io.read_rows(io.ROOT/'data/bfcl_sft/pool_bfcl_ds_sft.jsonl')]
    demos = io.read_json(io.ROOT/'data/bfcl_sft/demos_ds.json')
    verified = set(io.read_json(io.ROOT/'data/bfcl_demos_ds_verified.json'))
    assert len(old) == 23 and len(demos) == len(verified) == 34
    for seed in range(3):
        directory = POOL_ROOT/'bfcl'/f'B{cap}_seed{seed}'
        manifest = io.read_json(directory/'manifest.json')
        rows = [r for _, r in io.read_rows(directory/'pool_sft.jsonl')]
        sources = io.read_json(io.DEFAULT_OUT/'bfcl/ledger.json')['sources']
        counts = Counter()
        for row, origin in zip(rows, manifest['row_origins']):
            q = origin['package_id']
            snapshot = io.read_sealed(io.DEFAULT_OUT/'bfcl', q, set(manifest['purchased_ids']))
            sealed = snapshot['rows'][origin['package_row_index']]
            if not sealed['messages']:
                continue
            counts['legacy'] += 1
            # Whole row including whitespace within prompt/response is unchanged.
            assert row == sealed
            assert json.dumps(row, sort_keys=True, ensure_ascii=False).encode() == json.dumps(
                sealed, sort_keys=True, ensure_ascii=False).encode()
            prior = next(r for r in old if r['task_id'] == row['task_id'])
            assert row['prompt'].encode() == prior['prompt'].encode()
            assert row['messages'] == prior['messages']
            assert row['task_id'] in verified
            assert demos[row['task_id']]['question'] == row['messages'][1]['content']
            payload = read_bank_payload('bfcl', q, set(manifest['purchased_ids']), sources)
            raw = payload['historical_response']['result']
            response = raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False)
            assert row['response'].encode() == response.encode()
            if row == prior:
                counts['archive_exact'] += 1
                assert row['response'].encode() == prior['response'].encode()
        assert counts == {'legacy': legacy_count, 'archive_exact': archive_exact}


def test_real_pool_lineage_accounting_and_supplemental_fields():
    for directory in sorted(POOL_ROOT.glob('*/*')):
        manifest = io.read_json(directory/'manifest.json')
        benchmark = manifest['benchmark']
        audit = io.DEFAULT_OUT/benchmark
        acquisition = io.read_json(audit/f'acquired_B{manifest["cap"]}_seed{manifest["training_seed"]}.json')
        owned = set(acquisition['purchased_ids'])
        snapshots = {q: io.read_sealed(audit, q, owned) for q in acquisition['purchased_ids']}
        old_pools, old_manifest = build_pools(snapshots, acquisition, io.read_json(audit/'protocol.json'))
        assert manifest['row_format'] == ROW_FORMAT
        assert manifest_without_row_format(manifest) == manifest_without_row_format(old_manifest)
        sources = io.read_json(audit/'ledger.json')['sources']
        expected = {}
        for q, snapshot in snapshots.items():
            payload = read_bank_payload(benchmark, q, owned, sources) if benchmark != 'appworld' else None
            expected[q] = render_package(benchmark, snapshot, payload, row_format=ROW_FORMAT)
        new_pools, _ = build_pools(snapshots, acquisition, io.read_json(audit/'protocol.json'), expected)
        for arm, original in old_pools.items():
            rows = [r for _, r in io.read_rows(directory/f'pool_{arm}.jsonl')]
            assert rows == new_pools[arm]
            assert io.digest(rows) == manifest['arms'][arm]['pool_sha256']
            for before, after in zip(original, rows):
                assert before.keys() == after.keys()
                for key in before.keys() - {'messages', 'prompt', 'response', 'token_hint', 'c'}:
                    assert before[key] == after[key]  # Includes _sad_spans and _rejected.
                if 'c' in after:
                    assert len(before['c']) == len(after['c'])
                    for old_c, new_c in zip(before['c'], after['c']):
                        assert old_c.keys() == new_c.keys()
                        assert old_c['package_id'] == new_c['package_id']
                        assert old_c['turn_index'] == new_c['turn_index']
                        candidates = expected[new_c['package_id']]
                        assert any(r['task_id'] == after['task_id']
                                   and r['turn_index'] == new_c['turn_index']
                                   and r['prompt'] == new_c['state_prompt']
                                   and r['response'] == new_c['response'] for r in candidates)


def test_generator_calls_keep_concrete_arguments_and_order():
    calls = [dict(name='f', arguments={'nested': [1, {'é': True}], 'text': '</tool_call>'}),
             dict(name='g', arguments={'x': 0})]
    text = '\n'.join('<tool_call>\n' + json.dumps(c) + '\n</tool_call>' for c in calls)
    response = bfcl_response('generator_item', {}, text)
    parsed = json.loads(response)
    assert [{name: json.loads(args) for name, args in call.items()} for call in parsed] == [
        {c['name']: c['arguments']} for c in calls]
    assert '<think>' not in response and not response.startswith('<tool_call>')
    assert bfcl_response('generator_item', {'ground_truth': []}, '') == '[]'
    with pytest.raises(ValueError, match='unexplained empty'):
        bfcl_response('generator_item', {'ground_truth': [{'f': {}}]}, '')
    with pytest.raises(ValueError, match='framing'):
        bfcl_response('generator_item', {}, '<think>bad</think>')


def test_raw_sealed_reader_rejects_unowned_and_changed_sources(monkeypatch):
    monkeypatch.setattr(io, 'read_sealed', lambda *a: pytest.fail('unowned read'))
    with pytest.raises(PermissionError):
        read_bank_payload('bfcl', '0'*64, set(), {})
    monkeypatch.setattr(io, 'read_sealed', lambda *a: {})
    monkeypatch.setattr(io, 'file_hash', lambda *a: 'changed')
    name = f'data/rtd/v1_1_bfcl/sealed/{"0"*64}.json'
    with pytest.raises(ValueError, match='frozen source hash'):
        read_bank_payload('bfcl', '0'*64, {'0'*64}, {name: 'frozen'})


def test_render_rejects_pretemplated_messages():
    with pytest.raises(ValueError, match='applied chat template'):
        render_legacy_row({'response': 'answer'}, [{'role': 'user', 'content': '<think>'}], 'answer')


def test_materialize_preserves_manifest_and_is_idempotent(tmp_path):
    audit = io.DEFAULT_OUT/'bfcl'
    acquisition_path = audit/'acquired_B13843_seed0.json'
    # Work on an isolated acquisition copy; never rewrite audit inputs in tests.
    copy = tmp_path/acquisition_path.name
    copy.write_bytes(acquisition_path.read_bytes())
    materialize(audit, copy, tmp_path/'pools')
    directory = tmp_path/'pools/bfcl/B13843_seed0'
    before = {p.name: p.read_bytes() for p in directory.iterdir()}
    materialize(audit, copy, tmp_path/'pools')
    assert before == {p.name: p.read_bytes() for p in directory.iterdir()}
    manifest = io.read_json(directory/'manifest.json')
    manifest['arms']['ddpo']['C_m'] += 1
    io.write_json(directory/'manifest.json', manifest)
    with pytest.raises(ValueError, match='manifest/accounting'):
        materialize(audit, copy, tmp_path/'pools')
    assert before['pool_sft.jsonl'] == (directory/'pool_sft.jsonl').read_bytes()
