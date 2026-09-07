"""C25d capped HF output, action caps, and sampling-memory scheduling; CPU only."""
from contextlib import nullcontext
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT)]

from bfas.rtd.cli import load_config
from bfas.rtd.functional_step import lora_parameters, snapshot
from bfas.rtd.memory import MemoryPolicy, memory_batch_size, memory_batches
from bfas.rtd.persistence import ComputeJournal
from bfas.rtd.runtime import HFGenerateBackend
from test_rtd_return_gradient import TinyLM, TinyTokenizer


@pytest.mark.parametrize('ids', [(1, 2), (1, 3), (), (1,), (3, 1)])
def test_fake_hf_capped_generation_is_scored_without_retry_or_forced_eos(ids, tmp_path):
    class FakeHF(TinyLM):
        def generate(self, *, input_ids, attention_mask, generation_config):
            self.calls += 1
            assert generation_config.max_new_tokens == 2
            assert generation_config.temperature == generation_config.top_p == 1
            assert generation_config.top_k == 0
            sequence = torch.cat((input_ids, input_ids.new_tensor([ids])), dim=1)
            scores = tuple(self(sequence[:, :i], attention_mask=None).logits[:, -1]
                           for i in range(input_ids.shape[1], sequence.shape[1]))
            return SimpleNamespace(sequences=sequence, scores=scores)
    tokenizer = TinyTokenizer()
    tokenizer.bos_token_id = 0
    model = FakeHF().eval()
    model.calls = 0
    journal = ComputeJournal(tmp_path/'compute.jsonl')
    b = HFGenerateBackend(model, tokenizer, base_checkpoint_hash='toy', harness_hash='toy',
        tokenizer_hash='toy', max_action_tokens=2, journal=journal)
    p = snapshot(lora_parameters(model))
    if ids not in {(1, 2), (1, 3)}:
        with pytest.raises(ValueError, match='malformed generation'):
            b.sample_action('prompt', p, torch.Generator().manual_seed(1))
    else:
        action = b.sample_action('prompt', p, torch.Generator().manual_seed(1))
        assert action.action_ids == ids and action.truncated == (ids[-1] != 3)
        assert action.text == ('12' if action.truncated else '1')
        score, diagnostic = b.checked_score_action(action, p)
        assert float(score.detach()) == pytest.approx(action.generation_logprob, abs=1e-12)
        assert len(diagnostic['teacher_forced_token_logprobs']) == 2
        assert diagnostic['eos']['action_positions'] == ([] if action.truncated else [1])
        assert journal.events[1]['kind'] == 'generated_tokens'
        assert journal.events[1]['truncated'] == action.truncated
    assert model.calls == 1


def test_action_cap_defaults_overrides_and_restoration(tmp_path):
    config = load_config(ROOT/'configs/rtd/v1_bfcl_c25.yaml')
    assert config['max_action_tokens_by_benchmark']['bfcl'] == {'single_turn': 512, 'multi_turn': 1024}
    config['max_action_tokens_by_benchmark'] = {'bfcl': {'single_turn': 7}, 'other': {'single_turn': 11}}
    path = tmp_path/'config.yaml'
    path.write_text(yaml.safe_dump(config))
    caps = load_config(path)['max_action_tokens_by_benchmark']['bfcl']
    assert caps == {'single_turn': 7, 'multi_turn': 1024}
    b = HFGenerateBackend(TinyLM().eval(), TinyTokenizer(), base_checkpoint_hash='toy', harness_hash='toy',
        tokenizer_hash='toy', max_action_tokens=512, action_caps=caps)
    p = lora_parameters(b.model)
    identity = b.identity(p)
    with b.action_limit('live_relevance'):
        assert b.max_action_tokens == 7
        with b.action_limit('multi_turn_base'):
            assert b.max_action_tokens == 1024 and b.identity(p) == identity
        assert b.max_action_tokens == 7
    assert b.max_action_tokens == 512
    with pytest.raises(RuntimeError):
        with b.action_limit('multi_turn_base'):
            raise RuntimeError('checker failure')
    assert b.max_action_tokens == 512
    for invalid in [0, -1, True, 1.5]:
        config['max_action_tokens_by_benchmark']['bfcl']['single_turn'] = invalid
        path.write_text(yaml.safe_dump(config))
        with pytest.raises(ValueError, match='positive integer action caps'):
            load_config(path)


def test_batches_use_visible_free_memory_maximum_and_stable_order(monkeypatch):
    from bfas.rtd import memory
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '2')
    readings = iter([(40_000_000_000, 48_000_000_000), (47_000_000_000, 48_000_000_000),
                     (47_000_000_000, 48_000_000_000)])
    devices, cleared, records = [], [], []
    def mem_get_info(device):
        devices.append(str(device))
        return next(readings)
    monkeypatch.setattr(torch.cuda, 'mem_get_info', mem_get_info)
    monkeypatch.setattr(memory, 'release_device_cache', lambda device: cleared.append(str(device)))
    batches = list(memory_batches(range(5), 'cuda:0', MemoryPolicy(), record=records.append))
    assert batches == [(0,), (1, 2), (3, 4)]
    assert devices == ['cuda:0']*3 and cleared == ['cuda:0']*6
    assert records[0]['available_bytes'] == 30_000_000_000
    assert records[1]['batch_size'] == 2
    monkeypatch.setattr(torch.cuda, 'mem_get_info', lambda device: (3_000_000_000, 48_000_000_000))
    size, row = memory_batch_size(8, 'cuda:0', MemoryPolicy())
    assert size == 1 and row['singleton_exceeds_estimate']
    # The driver reserve can be the tighter constraint than the peak target.
    size, row = memory_batch_size(8, 'cuda:0', MemoryPolicy(8, 100_000_000_000, 2_000_000_000, 500_000_000))
    assert size == 2 and row['available_bytes'] == 1_000_000_000
    def forbidden(*args):
        pytest.fail('CPU batching must not initialize CUDA')
    monkeypatch.setattr(torch.cuda, 'mem_get_info', forbidden)
    assert list(memory_batches(range(5), 'cpu', MemoryPolicy(max_batch_size=3))) == [(0, 1, 2), (3, 4)]
    assert list(memory_batches([], 'cpu', MemoryPolicy())) == []


@pytest.mark.parametrize('fail', [False, True])
def test_phase_peaks_are_local_and_step_max_survives_resets_and_failures(tmp_path, monkeypatch, fail):
    from bfas.rtd import memory
    monkeypatch.setattr(memory, 'release_device_cache', lambda device: None)
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda *a: None)
    peak = dict(allocated=0, reserved=0)
    monkeypatch.setattr(torch.cuda, 'reset_peak_memory_stats', lambda device: peak.update(allocated=0, reserved=0))
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda device: peak['allocated'])
    monkeypatch.setattr(torch.cuda, 'max_memory_reserved', lambda device: peak['reserved'])
    journal = ComputeJournal(tmp_path/'compute.jsonl', cuda=True)
    with pytest.raises(RuntimeError) if fail else nullcontext():
        with journal.measure_step(round=1, step=1, start_phase='round_start'):
            with journal.measure_phase('source_sampling', round=1):
                peak.update(allocated=20, reserved=25)
                with journal.measure_phase('generation'):
                    peak.update(allocated=10, reserved=15)
            with journal.measure_phase('source_preconditioner', round=1):
                peak.update(allocated=12, reserved=17)
                if fail:
                    raise RuntimeError('failed backward')
    phases = {r['operation']: r for r in journal.events if r['kind'] == 'subphase_memory'}
    assert phases['source_sampling']['peak_reserved_bytes'] == 25
    assert phases['generation']['peak_reserved_bytes'] == 15
    assert phases['source_preconditioner']['peak_reserved_bytes'] == 17
    assert phases['source_preconditioner']['status'] == ('failed' if fail else 'complete')
    ends = {r['operation']: r for r in journal.events if r['kind'] == 'compute_end'}
    assert ends['source_sampling']['peak_reserved_bytes'] == 25
    assert ends['source_preconditioner']['peak_reserved_bytes'] == 17
    assert journal.events[-1]['peak_allocated_bytes'] == 20
    assert journal.events[-1]['peak_reserved_bytes'] == 25
    assert journal.events[-1]['status'] == ('failed' if fail else 'complete')
