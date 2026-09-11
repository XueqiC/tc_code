"""Real ALFWorld first-window sampling with a fake CPU policy; no environment/API."""
from dataclasses import replace
import json
from pathlib import Path
import sys
from types import MethodType, SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT)]

from bfas.rtd import cli, runtime
from bfas.rtd.benchmarks.registry import ALFWorldExperimentSupport, get_benchmark
from bfas.rtd.caps import recorded_budget_ceilings
from bfas.rtd.experiment import RTDExperiment
from bfas.rtd.functional_step import snapshot
from bfas.rtd.generation_batch import action_cap, ticket
from bfas.rtd.persistence import digest
from bfas.rtd.return_gradient import TorchPolicyBackend
from bfas.rtd.unified.arms import ARMS
from bfas.rtd.unified.experiment import P1Experiment
from test_rtd_checks import TinyLM, TinyTokenizer


class BankTokenizer(TinyTokenizer):
    def encode(self, text, add_special_tokens=False):
        # Retain the whole bank prompt in the tiny model's three-token alphabet.
        return [ord(c) % 3 for c in text] if isinstance(text, str) else list(text)


class FakeGenerationPolicy(runtime.HFGenerateBackend):
    """Keep real D12 planning, queues, identity checks and scoring; fake decoding."""
    def _generate_group(self, rows, parameters):
        actions = []
        previous = self.max_action_tokens
        try:
            for row in rows:
                self.max_action_tokens = row['cap']
                rng = torch.Generator().manual_seed(row['ticket']['seed'])
                action = TorchPolicyBackend.sample_action(self, row['ids'], parameters, rng)
                actions.append(replace(action, generation_metadata=action.generation_metadata | dict(
                    rng_ticket=row['ticket'], batch_size=len(rows))))
        finally:
            self.max_action_tokens = previous
        return tuple(actions)


@pytest.mark.parametrize('arm', ARMS)
def test_unified_alfworld_first_round_generation_batch(tmp_path, monkeypatch, arm):
    config = cli.load_config(ROOT/'configs/rtd/unified_alfworld_gemma4.yaml', arm=arm)
    bank = ROOT/config['replay_bank_path']
    if not (bank/'public/support.json').exists():
        pytest.skip('real ALFWorld v1.1 bank is not installed')
    support = ALFWorldExperimentSupport(ROOT, config)
    _, _, settings = runtime.backend_config(config)
    backend = FakeGenerationPolicy(TinyLM().eval(), BankTokenizer(),
        base_checkpoint_hash='fake-cpu-policy', harness_hash='alfworld-sampling',
        tokenizer_hash='bank-character-mod3', **settings)
    backend.action_limit = MethodType(get_benchmark(config).action_limit, backend)
    # Round setup normally binds an environment factory; sampling needs no worker.
    monkeypatch.setattr(support, 'feedback_context', lambda *args: None)
    certificate = json.loads((bank/'public/cap_certificate.json').read_text())['core']
    manifest = dict(config_hash=digest(config), config=config, arm=arm, bank_path=str(bank),
        budget_ceilings=recorded_budget_ceilings(certificate['budget_denominator'], rounds=config['rounds']),
        hardware_hash='fake-cpu', smoke=True)
    e = P1Experiment(config, manifest, tmp_path/arm, backend, support, smoke=True)
    backend.journal = e.journal
    e.round_start()
    e.step_start()
    e.reference()
    while e.state['phase'] == 'selected':
        e.selected()
    assert e.state['phase'] == 'revealed'
    assert (e.state['round'], e.state['step']) == (1, 1)
    assert len(e.state['selected']) == e.max_new_packages == config['p1']['smoke_packages'] == 1
    # The former prefetch planner used this default instead of agent_action=256.
    assert backend.max_action_tokens == 512
    frozen = snapshot(e.state['source'])
    cache = dict(e.state['source_cache'])
    observed = {}

    class SamplingComplete(Exception):
        pass

    draw_pairs = e.alpha_draw_pairs

    def observe(records, *, role):
        pairs = draw_pairs(records, role=role)
        observed[role] = pairs
        if role == 'virtual_reference':
            raise SamplingComplete
        return pairs

    monkeypatch.setattr(e, 'alpha_draw_pairs', observe)
    # Exercise revealed -> P1 -> shared alpha/d -> sample_state -> D12 queue.
    # Stop before reference gradients or feedback can launch an environment.
    with pytest.raises(SamplingComplete):
        e.revealed()
    assert list(observed) == ['commit', 'virtual_reference']
    assert backend._pending_generation is None and backend.max_action_tokens == 512
    assert e.state['source_cache'] == cache
    assert e.config['source_samples_per_state'] == config['unified']['m'] == 2
    assert (e.config['exposure_slots_per_window'], e.config['max_new_packages_per_window']) == (40, 20)
    for name, value in frozen.items():
        assert torch.equal(e.state['source'][name], value)
    assert any(pair.record.teacher is not None for pair in observed['commit'])

    for role, pairs in observed.items():
        assert len(pairs) == e.slots == config['p1']['smoke_slots'] == 2
        assert all(len(pair.sources) == 2 for pair in pairs)
        rows = [r for r in e.journal.events if r['kind'] == 'alpha_d_source_pair' and r['role'] == role]
        assert len(rows) == len(pairs)
        actions = [r['action'] for r in e.journal.events if r['kind'] == 'source_sample'][-8:]
        actions = actions[:4] if role == 'commit' else actions[4:]
        expected_rng = torch.Generator().manual_seed(e.role_seed('sources/'+role))
        for pair, row in zip(pairs, rows):
            assert row['rng_before'] == digest(expected_rng.get_state().tolist())
            for source in pair.sources:
                action = actions.pop(0)
                assert source.frozen_snapshot_id == action['policy_id'] == e.state['source_id']
                assert tuple(action['prompt_ids']) == tuple(backend.tokenizer.encode(pair.record.state.prompt))
                assert action['generation_metadata']['max_action_tokens'] == 256
                assert action['generation_metadata']['batch_size'] >= 2
                assert action['generation_metadata']['rng_ticket'] == ticket(expected_rng)
            assert row['rng_after'] == digest(expected_rng.get_state().tolist())

    # The V0 shared sampler must give the same pairs for the same role stream,
    # even when the current trainable policy differs from the frozen source.
    e.state['parameters'] = {name: value + .1 for name, value in frozen.items()}
    for role, pairs in observed.items():
        e.sampling_rng.manual_seed(e.role_seed('sources/'+role))
        shared = RTDExperiment.alpha_draw_pairs(e, [p.record for p in pairs], role=role)
        assert shared == pairs


@pytest.mark.parametrize('benchmark,category,caps,expected', [
    ('bfcl', 'simple', {'single_turn': 512, 'multi_turn': 1024}, 512),
    ('bfcl', 'multi_turn_base', {'single_turn': 512, 'multi_turn': 1024}, 1024),
    ('bfcl', 'simple', {}, 4096),
    ('alfworld', 'agent_action', {'agent_action': 256}, 256),
    ('webshop', 'agent_action', {'agent_action': 128}, 128),
])
def test_generation_batch_cap_matches_registered_action_policy(benchmark, category, caps, expected):
    backend = SimpleNamespace(max_action_tokens=4096, action_caps=caps)
    backend.action_limit = MethodType(get_benchmark({'benchmark': benchmark}).action_limit, backend)
    assert action_cap(backend, category) == expected
    assert backend.max_action_tokens == 4096
