#!/usr/bin/env python3
"""CPU-only AppWorld prompt, provenance and real trainer boundary audit."""
from collections import Counter
from copy import deepcopy
import os
from pathlib import Path
from unittest.mock import patch

from tools import table1_common as io
from tools.table1_appworld_format import (
    APPWORLD_NATIVE_ROW_FORMAT, THINK_REMAINDER, base_tokenizer, evaluation_prompt,
)


def captured_evaluation_prompt(tokenizer, messages):
    """Run the actual HF evaluator up to generate(), capturing CPU input IDs.

    The model double returns EOS only; no weights, sampling, API or world.
    This is current-harness replay, not a newly recorded evaluation episode.
    """
    import torch
    import appworld_eval as evaluator

    class Capture:
        def parameters(self):
            yield torch.empty(0, device='cpu')

        def generate(self, **kwargs):
            self.ids = kwargs['input_ids'][0].tolist()
            return torch.tensor([self.ids + [tokenizer.eos_token_id]], device='cpu')

    model = Capture()
    with patch.dict(os.environ, {'APPWORLD_THINK': '0'}):
        evaluator.generate_reply(model, tokenizer, deepcopy(messages), 1, 0.0)
    return tokenizer.decode(model.ids, skip_special_tokens=False), model.ids


def recorded_official_checks(tokenizer):
    """Three real base-model dev calls; validate the shared template only.

    These input logs are official ReAct conversations, not awb3 transcripts.
    They do not certify awb3's message assembly or a historical server template.
    """
    from bfas.adapters.appworld_official import AppWorldOfficialAdapter, call_messages

    run = (io.ROOT / 'envs/appworld-repo/experiments/outputs'
           / 'simplified_react_code_agent/local/qwen35-4b-base_dev')
    path = run / 'tasks/0d8a4ee_1/logs/lm_calls.jsonl'
    adapter = AppWorldOfficialAdapter()
    adapter._tokenizer = tokenizer
    checks = []
    for line, call in io.read_rows(path):
        assert call['input']['model'] == 'qwen35-4b-base'
        assert not call['input'].get('tools')
        messages = call_messages(call)
        expected = adapter._render({'messages': messages})
        prompt = evaluation_prompt(messages, tokenizer)
        captured, ids = captured_evaluation_prompt(tokenizer, messages)
        assert prompt.encode('utf-8') == expected.encode('utf-8') == captured.encode('utf-8')
        checks.append(dict(path=str(path.relative_to(io.ROOT)), source_sha256=io.file_hash(path),
                           line=line, call_id=call['id'], messages_sha256=io.digest(messages),
                           prompt_sha256=io.digest(prompt), prompt_bytes=len(prompt.encode('utf-8')),
                           prompt_tokens=len(ids), equal=True))
        if len(checks) == 3:
            break
    assert len(checks) == 3
    return checks


def purchased_prefix_checks(tokenizer):
    import appworld_eval as evaluator
    from appworld_bridge import _supervisor_details

    audit = io.DEFAULT_OUT / 'appworld'
    acquisition = io.read_json(audit / 'acquired_B22633_seed0.json')
    owned = set(acquisition['purchased_ids'])
    ledger = io.read_json(audit / 'ledger.json')
    snapshots = {q: io.read_sealed(audit, q, owned) for q in acquisition['purchased_ids']}
    # Read only purchased source lines after validating the frozen file hashes.
    source_lines = {}
    for s in snapshots.values():
        source = s['package']['source']
        source_lines.setdefault(source['path'], set()).add(source['line'])
    traces = {}
    import json
    for name, wanted in source_lines.items():
        assert io.file_hash(io.ROOT / name) == ledger['sources'][name]
        with (io.ROOT / name).open(encoding='utf-8') as stream:
            for line, raw in enumerate(stream, 1):
                if line in wanted:
                    traces[name, line] = json.loads(raw)
    checks = []
    for q, snapshot in snapshots.items():
        package = snapshot['package']
        source = package['source']
        trace = traces[source['path'], source['line']]
        assert q == io.digest(['appworld-episode-attempt', io.digest(trace)])
        rows = snapshot['rows']
        if not rows:
            assert package['failed_attempt']
            continue
        spec_path = io.ROOT / 'envs/appworld-data/data/tasks' / package['task_id'] / 'specs.json'
        spec = io.read_json(spec_path)
        system = evaluator.make_system_prompt(dict(
            instruction=spec['instruction'], supervisor=_supervisor_details(spec)))
        assert rows[0]['messages'][0] == dict(role='system', content=system)
        assert len(rows) == sum(t['role'] == 'assistant' for t in trace['turns'])
        for row in rows:
            index = row['turn_index']
            assert row['messages'] == trace['turns'][:index]
            assert row['response'] == trace['turns'][index]['content']
            for observation in row['messages'][3::2]:
                content = observation['content']
                assert content.startswith('Execution output:\n')
                assert content.endswith('\n\nContinue the task.')
                output = content[len('Execution output:\n'):-len('\n\nContinue the task.')]
                assert evaluator.truncate_output(output) == output
            prompt = evaluation_prompt(row['messages'], tokenizer)
            captured, ids = captured_evaluation_prompt(tokenizer, row['messages'])
            assert prompt.encode('utf-8') == captured.encode('utf-8')
            assert ids == tokenizer(prompt, add_special_tokens=False)['input_ids']
            checks.append(dict(package_id=q, task_id=row['task_id'], turn_index=index,
                               prompt_sha256=io.digest(prompt), equal=True,
                               task_spec_sha256=io.file_hash(spec_path)))
    assert len(checks) == 368
    return snapshots, checks


def trainer_checks(tokenizer, snapshots):
    import appworld_train as trainer
    from tools.table1_pool_from_sealed import action_spans

    def forbidden(*args, **kwargs):
        raise AssertionError('native trainer must not apply a chat template')

    results = []
    root = io.DEFAULT_OUT / 'pools_native/appworld'
    counts = dict(sft=368, sad=368, bbopd=5, pbsd_insp=368, ddpo=368, pbsd_agent=368, star=0)
    reference = None
    with patch.object(tokenizer, 'apply_chat_template', forbidden), patch.dict(
            os.environ, {'AW_MAX_PROMPT_TOKENS': '4096', 'AW_TRUNCATE_SIDE': 'tail'}):
        for seed in range(3):
            directory = root / f'B22633_seed{seed}'
            manifest = io.read_json(directory / 'manifest.json')
            assert manifest['row_format'] == APPWORLD_NATIVE_ROW_FORMAT
            assert manifest['packages_purchased'] == 22 and len(manifest['tasks_covered']) == 21
            assert manifest['sealed_replay_spend'] == 22107 and manifest['remaining_budget'] == 526
            assert manifest['cost_confidence'] == 'estimated'
            assert manifest['ranking']['total_cached'] == 31
            assert manifest['ranking']['purchased_task_eligible'] == 7
            assert manifest['ranking']['missing_ranking_call_cost'] == 7
            assert manifest['ranking']['reused'] == 0
            hashes = {}
            for arm, count in counts.items():
                path = directory / f'pool_{arm}.jsonl'
                hashes[arm] = io.file_hash(path)
                assert manifest['arms'][arm]['rows'] == count
                assert manifest['arms'][arm]['C_m'] == (0 if arm == 'star' else 22107)
                if not count:
                    assert path.read_bytes() == b''
                    try:
                        trainer.load_pool(path)
                    except AssertionError as exc:
                        assert 'pool is empty' in str(exc)
                    else:
                        raise AssertionError('STaR must not be accepted by the SFT loader')
                    results.append(dict(seed=seed, arm=arm, rows=0))
                    continue
                rows = trainer.load_pool(path)
                assert len(rows) == count
                assert io.digest([{k: v for k, v in r.items() if k != '_pool_index'} for r in rows]) == manifest['arms'][arm]['pool_sha256']
                stats = Counter()
                for row in rows:
                    assert row['messages'] == [] and row['teacher'] == 'deepseek-v4-pro'
                    assert row['response'].startswith(THINK_REMAINDER)
                    full_prompt = tokenizer(row['prompt'], add_special_tokens=False)['input_ids']
                    assert trainer.prompt_token_ids(tokenizer, row) == full_prompt
                    prompt = full_prompt[-4096:]
                    stats['prompt_truncated'] += len(full_prompt) > 4096
                    for field in ('response', '_rejected'):
                        if field not in row:
                            continue
                        response = tokenizer(row[field], add_special_tokens=False)['input_ids']
                        assert tokenizer.eos_token_id not in response
                        target = response[:trainer.MAX_RESPONSE_TOKENS] + [tokenizer.eos_token_id]
                        ids, labels = trainer.encode(tokenizer, dict(row, response=row[field]), device='cpu')
                        assert ids.tolist() == [prompt + target]
                        assert labels.tolist() == [[-100] * len(prompt) + target]
                        assert target.count(tokenizer.eos_token_id) == 1
                        stats['encoded_targets'] += 1
                        stats['target_truncated'] += len(response) > trainer.MAX_RESPONSE_TOKENS
                    if '_sad_spans' in row:
                        assert row['_sad_spans']['action'] == action_spans(row['response'])
                        assert action_spans(row['response']) == [list(s) for s in trainer.action_spans(row['response'])]
                    if '_rejected' in row:
                        stats['pairs'] += 1
                        assert row['turn_index'] == 2
                        candidates = [s for s in snapshots.values() if s['package']['task_id'] == row['task_id']]
                        assert any(s['pbsd_rejected'].get('0') == row['_rejected'] for s in candidates)
                    for c in row.get('c', []):
                        source = snapshots[c['package_id']]['rows'][0]
                        assert source['task_id'] == row['task_id']
                        assert c['turn_index'] == source['turn_index'] == 2
                        assert c['response'] == THINK_REMAINDER + source['response']
                        assert c['state_prompt'] == row['prompt']
                    if arm == 'pbsd_agent':
                        assert row['c']
                assert stats['pairs'] == (13 if arm == 'pbsd_insp' else 0)
                results.append(dict(seed=seed, arm=arm, rows=count, **stats))
            if reference is None:
                reference = hashes
            assert hashes == reference
    return results


def main():
    os.environ.update(CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    import socket
    import torch
    from transformers import AutoModelForCausalLM

    def forbidden(*args, **kwargs):
        raise AssertionError('network, model weights and CUDA are forbidden')

    with patch.object(socket.socket, 'connect', forbidden), patch.object(torch.cuda, '_lazy_init', forbidden), patch.object(AutoModelForCausalLM, 'from_pretrained', forbidden):
        tokenizer = base_tokenizer()
        recorded = recorded_official_checks(tokenizer)
        snapshots, purchased = purchased_prefix_checks(tokenizer)
        pools = trainer_checks(tokenizer, snapshots)
    report = dict(
        schema='table1-native-appworld-validation-v1', row_format=APPWORLD_NATIVE_ROW_FORMAT,
        tokenizer=dict(path=tokenizer.name_or_path, template_sha256=io.digest(tokenizer.chat_template),
                       eos=tokenizer.eos_token, eos_id=tokenizer.eos_token_id),
        recorded_official_template_checks=recorded,
        recorded_evidence_scope='3 official ReAct base-model dev input calls; current base template replay only',
        awb3_recorded_evaluation_verified=False,
        limitation='awb3 records/input transcripts absent in this worktree; official ReAct messages differ',
        purchased_current_harness_checks=purchased, pools=pools,
        pool_files=len(pools), total_rows=sum(p['rows'] for p in pools),
        encoded_targets=sum(p.get('encoded_targets', 0) for p in pools),
        gpu_used=False, model_weights_loaded=False, network_calls=0)
    path = io.DEFAULT_OUT / 'native_appworld_validation.json'
    io.write_json(path, report)
    print(f'{path}: {len(recorded)} recorded template checks, {len(purchased)} purchased prefixes, '
          f'{len(pools)} pools, {report["encoded_targets"]} CPU encodings; awb3 archived-input evidence unavailable')


if __name__ == '__main__':
    main()
