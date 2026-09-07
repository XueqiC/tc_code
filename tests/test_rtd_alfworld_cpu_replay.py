"""Explicit two-train-game CPU acceptance; sealed evidence is never resealed."""
from dataclasses import asdict
import json
import os

import pytest


@pytest.mark.skipif(os.environ.get('RTD_ALFWORLD_REAL_TESTS') != '1',
                    reason='opt in with RTD_ALFWORLD_REAL_TESTS=1; real read-only assets')
def test_two_train_demos_twice_and_fixed_failure_paths(tmp_path, monkeypatch):
    import torch
    from bfas.rtd.benchmarks import alfworld_config
    from bfas.rtd.benchmarks.alfworld_state import replay_commands
    from bfas.rtd.benchmarks.alfworld_support import FrozenRenderer, RealStepper
    from bfas.rtd.benchmarks.alfworld_rollout import parse_action
    from bfas.rtd.persistence import atomic_json, file_hash
    from tools.rtd_alfworld_experiment import ROOT, smoke_plan
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    monkeypatch.setattr(torch.cuda, '_lazy_init', lambda *a, **k: pytest.fail('CPU replay attempted CUDA'))
    config = alfworld_config.load_config(ROOT / 'configs/rtd/v1_alfworld_c26.yaml')
    plan = smoke_plan(ROOT, config)
    bank = ROOT / config['replay_bank_path']
    support = json.loads((bank / 'public/support.json').read_text())
    renderer = FrozenRenderer(alfworld_config.model_directory(ROOT, config))
    manifest_hash = file_hash(bank / 'sealed/manifest.json')
    records = []
    for task in plan['train_replays']:
        payload = json.loads((bank / f"sealed/{task['query_id']}.json").read_text())
        request = support['tasks'][task['task_id']]['request']
        attempts = []
        for _ in range(2):
            env = RealStepper(environment_hash=request['environment_hash'], timeout=30., episode_timeout=120.)
            try:
                result = replay_commands(request, payload['commands'], env, renderer)
                attempts.append(dict(replay=asdict(result), observations=env.observations))
            finally:
                env.close()
        assert attempts[0] == attempts[1]
        assert result.done and result.won
        assert [asdict(state) for state in result.states] == payload['verification']['states']
        assert result.transcript_hash == payload['verification']['transcript_hash']
        failure = []
        env = RealStepper(environment_hash=request['environment_hash'], timeout=30., episode_timeout=120.)
        try:
            cursor, observation = env.reset(request)
            raw = 'THOUGHT: deliberately invalid smoke output; no executable action.'
            command, fallback = parse_action(raw, observation.admissible)
            assert command == 'look' and fallback
            for i in range(40):
                cursor, observation = env.step(cursor, command if i == 0 else 'look')
                failure.append(asdict(observation))
                if observation.done:
                    break
        finally:
            env.close()
        records.append(dict(task=task, replay=attempts[0], repeated_exactly=True,
                            invalid_raw_action=raw, parser_fallback=fallback, failure_path=failure))
        # Retain earlier successful work if the next environment fails.
        atomic_json(tmp_path / 'two-train-replay.json', dict(gpu_used=False, records=records))
    assert file_hash(bank / 'sealed/manifest.json') == manifest_hash
