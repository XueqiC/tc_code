"""Real CPU startup constructors, sealed files and a local HF tokenizer."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT)]

from bfas.rtd import cli, preflight, runtime
from bfas.rtd.bank_build import seal_v11, state_record
from bfas.rtd.benchmarks import registry
from bfas.rtd.persistence import digest, file_hash
from bfas.rtd.return_gradient import TorchPolicyBackend
from bfas.rtd.transport import FullState
from bfas.rtd.unified.arms import ARMS
from test_rtd_checks import TinySupport
from test_rtd_manifest_tolerance import manifest_inputs

@pytest.mark.parametrize('name', ['alfworld', 'alfworld_d0', 'webshop', 'bfcl'])
def test_unified_backend_preflight_uses_v11_without_changing_frozen_identity(name):
    filename = ('unified_alfworld_gemma4_d0.yaml' if name == 'alfworld_d0' else
                f'unified_{name}_gemma4.yaml')
    for arm in ARMS:
        config = cli.load_config(ROOT/'configs/rtd'/filename, arm=arm)
        before = deepcopy(config)
        execution, lora, settings = runtime.backend_config(config)
        assert config == before and config['protocol_version'] == 'unified-p0-1'
        assert execution['protocol_version'] == '1.1.0'
        assert settings['generation_batch'].prompts_per_batch == 8
        assert settings['generation_batch'].forward_prompts_per_batch == 0
        assert settings['score_tolerance'].max_abs_outlier_tokens == 2
        assert lora.r == 16 and lora.lora_alpha == 32
        assert execution['p1'] == before['p1']
        assert execution['unified'] == before['unified']


@pytest.fixture
def local_startup(manifest_inputs, monkeypatch):
    c = manifest_inputs
    with monkeypatch.context() as m:
        m.setattr(cli, 'ROOT', ROOT)
        config = cli.load_config(ROOT/'configs/rtd/unified_bfcl_gemma4.yaml')
    config['student'] = c.config['student']
    canonical = c.root/'configs/rtd/v1_bfcl_c25.yaml'
    canonical.parent.mkdir(parents=True, exist_ok=True)
    canonical.write_bytes((ROOT/'configs/rtd/v1_bfcl_c25.yaml').read_bytes())
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast
    vocab = {'[UNK]': 0, '<s>': 1, '</s>': 2, '<turn|>': 3, '<|tool_response>': 4, 'reset': 5}
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel(vocab, unk_token='[UNK]')),
        unk_token='[UNK]', bos_token='<s>', eos_token='</s>')
    tokenizer.save_pretrained(config['student'])
    states = {f'{i:064x}': FullState.create({'id': i}, [{'role': 'user', 'content': 'reset'}],
              'reset', f'{i:064x}') for i in range(4)}
    support = TinySupport(states)
    support.unavailable = {}
    support.close = lambda: None
    records, payloads = [], {}
    for i, state in enumerate(states.values()):
        q = f'{i+10:064x}'
        records.append(state_record(q, state, 'demo_episode', 4, 'exact'))
        payloads[q] = dict(cost=4, cost_confidence='exact',
            behaviors=[dict(state=asdict(state), text='action')])
    parents = [dict(parent_hash=h, official_id=t, fold=int(h, 16) % 2) for h, t in support.parents.items()]
    bank = c.root/'bank'
    seal_v11(bank, records, payloads, benchmark='bfcl', student=config['student'],
        public={'support.json': dict(parents=parents)}, audit={'m': 4}, inputs={})
    config.update(replay_bank_path=str(bank), support_manifest=str(bank/'public/support.json'))
    monkeypatch.setattr(cli, 'evaluation_harness_identity', lambda *a: {'fixture': 'bfcl'})
    monkeypatch.setattr(registry, 'get_benchmark', lambda config: SimpleNamespace(
        support_protocol=lambda *a: support, action_limit=TorchPolicyBackend.action_limit))
    path = c.root/'config.yaml'
    path.write_text(yaml.safe_dump(config))
    return SimpleNamespace(config=config, path=path, bank=bank, support=support)


@pytest.fixture
def forbid_accelerators_and_network(monkeypatch):
    import socket
    from transformers import AutoModelForCausalLM
    def forbidden(*a, **kw):
        pytest.fail('preflight must never probe CUDA, load a model, or contact a network')
    monkeypatch.setattr(runtime, 'load_backend', forbidden)
    monkeypatch.setattr(AutoModelForCausalLM, 'from_pretrained', forbidden)
    monkeypatch.setattr(cli, 'hardware_identity', forbidden)
    for name in ('is_available', 'device_count', 'set_device', 'get_device_properties', 'manual_seed_all', '_lazy_init'):
        monkeypatch.setattr(torch.cuda, name, forbidden)
    monkeypatch.setattr(socket.socket, 'connect', forbidden)


@pytest.mark.parametrize('arm', ARMS)
def test_preflight_real_bank_manifest_tokenizer_cpu_only(local_startup, forbid_accelerators_and_network, arm):
    c = local_startup
    before = {p: p.read_bytes() for p in c.bank.rglob('*') if p.is_file()}
    config, selected = cli.startup_config(c.path, arm)
    result = preflight.preflight_config(config, selected)
    assert result['status'] == 'OK' and result['arm'] == arm
    assert result['support_states'] == result['rendered_states'] == 4
    assert result['ledger_spent'] == 0 and result['tokenizer'] == 'local cache'
    assert result['protocol_version'] == '1.1.0' and result['gpu_required']
    assert {p: p.read_bytes() for p in before} == before
    assert not (c.path.parent/'manifest.json').exists()
    assert not (c.path.parent/'teacher.jsonl').exists()


@pytest.mark.parametrize('arm', ['V0', 'D3'])
@pytest.mark.parametrize('form', ['bank_fraction', 'tokens'])
def test_preflight_budget_manifest_and_ledger(local_startup, forbid_accelerators_and_network, arm, form):
    from bfas.rtd.experiment import prepare_ledger
    c = local_startup
    config = deepcopy(c.config)
    if form == 'tokens':
        config.pop('budget_checkpoints_bank_fraction')
        # Deliberately exceed this tiny bank's total: caps are not spend targets
        # and must not be clamped or rescaled to the usable bank denominator.
        config['budget_checkpoints_tokens'] = [2500, 5000]
    if arm == 'V0':
        config, _, _ = runtime.backend_config(config)
    config = cli.validate_config(config, arm=arm)
    expected = [2500, 5000] if form == 'tokens' else [2, 4]
    audit = cli.bank_audit(config)
    assert audit['budget_ceilings'] == expected
    manifest = cli.make_manifest(config, arm, audit, smoke=True,
                                 hardware=dict(hard={'device': 'cpu'}, metadata={}))
    assert manifest['budget_checkpoint_form'] == form
    assert manifest['budget_ceilings'] == expected
    assert manifest['budget_rounding'] == ('none_absolute_tokens' if form == 'tokens' else 'positive_integer_half_up')
    ledger, _ = prepare_ledger(manifest, c.support)
    assert ledger.budget == expected[0] and ledger.spent == 0
    assert preflight.preflight_config(config, arm)['status'] == 'OK'


@pytest.mark.parametrize('mode', ['complete', 'streaming'])
@pytest.mark.parametrize('source_form', ['bank_fraction', 'tokens'])
def test_preflight_replay_budget_form_is_identity_even_at_equal_caps(local_startup, mode, source_form):
    from bfas.rtd.conventions import schedule_identity, hashed_step
    c = local_startup
    fraction = deepcopy(c.config)
    tokens = deepcopy(fraction)
    tokens.pop('budget_checkpoints_bank_fraction')
    tokens['budget_checkpoints_tokens'] = [2, 4]
    configs = dict(bank_fraction=fraction, tokens=tokens)
    manifests = {form: cli.make_manifest(config, 'D3', cli.bank_audit(config), smoke=True)
                 for form, config in configs.items()}
    assert manifests['tokens']['budget_ceilings'] == manifests['bank_fraction']['budget_ceilings']
    assert manifests['tokens']['config_hash'] != manifests['bank_fraction']['config_hash']
    assert manifests['tokens']['campaign_identity'] != manifests['bank_fraction']['campaign_identity']
    identities = {form: schedule_identity(manifests[form], config, c.support)
                  for form, config in configs.items()}
    # Fraction exports retain the historical identity structure for live V0.
    assert 'budget_checkpoints_tokens' not in identities['bank_fraction']
    assert identities['tokens']['budget_checkpoints_tokens'] == [2, 4]
    path = c.path.parent/'budget-schedule.json'
    path.write_text(json.dumps(dict(version='rtd-v11-exposure-v1', arm='V0', complete=True,
        smoke=True, identity=identities[source_form],
        steps=[hashed_step(dict(round=1, step=1, decision=True, selected=[], window_budget=0))])))
    for target_form, config in configs.items():
        c.path.write_text(yaml.safe_dump(config))
        target, arm = cli.startup_config(c.path, 'D3', path, replay_mode=mode)
        if target_form == source_form:
            assert preflight.preflight_config(target, arm)['status'] == 'OK'
        else:
            with pytest.raises(ValueError, match='identity'):
                preflight.preflight_config(target, arm)
            with pytest.raises(ValueError, match='config'):
                cli.resume_config(c.path, manifests[source_form])


def test_budget_manifest_legacy_fraction_resume_keeps_hard_guards(local_startup):
    from bfas.rtd.identity import validate_resume
    c = local_startup
    current = cli.make_manifest(c.config, 'D3', cli.bank_audit(c.config), smoke=True)
    saved = deepcopy(current)
    saved.pop('budget_checkpoint_form')
    validate_resume(c.path.parent, c.path.parent, saved, current)
    for change in ({'budget_checkpoint_form': 'tokens'}, {'budget_ceilings': [3, 4]}):
        with pytest.raises(ValueError, match='metadata changed'):
            validate_resume(c.path.parent, c.path.parent, saved, current | change)


@pytest.mark.parametrize('key,value,message', [
    ('generation_batch', {'prompts_per_batch': 0}, 'generation_batch requires positive integer'),
    ('score_consistency_tolerance', {'mean_abs': -1}, 'score tolerances'),
    ('rounds', 3, 'P1 requires two rounds'),
    ('max_context_tokens', 1, 'backend requires max_context_tokens'),
])
def test_preflight_reports_first_config_exception(local_startup, capsys, key, value, message):
    c = local_startup
    c.path.write_text(yaml.safe_dump(c.config | {key: value}))
    assert preflight.main(['--config', str(c.path), '--arm', 'D3']) == 1
    output = capsys.readouterr().out
    assert '[rtd-preflight] FAIL ValueError:' in output and message in output
    assert '[rtd-preflight] OK' not in output


def test_preflight_checks_certified_bank_before_backend(local_startup, monkeypatch, capsys):
    c = local_startup
    (c.bank/'public/requests.json').write_text('[]')
    monkeypatch.setattr(runtime, 'backend_config', lambda *a: pytest.fail('must fail bank audit first'))
    assert preflight.main(['--config', str(c.path), '--arm', 'D0']) == 1
    assert 'certificate binding mismatch' in capsys.readouterr().out


def test_preflight_propagates_executor_failure(local_startup, monkeypatch):
    from bfas.rtd import experiment
    def broken(*a, **kw):
        raise ValueError('executor fixture failure')
    monkeypatch.setattr(experiment, 'executor_config', broken)
    with pytest.raises(ValueError, match='executor fixture failure'):
        preflight.preflight_config(local_startup.config, 'D3')


def test_preflight_renderer_failure_closes_support(local_startup, monkeypatch, capsys):
    closed = []
    monkeypatch.setattr(local_startup.support, 'close', lambda: closed.append(True))
    def broken(*a, **kw):
        raise ValueError('renderer fixture failure')
    monkeypatch.setattr(preflight, 'prepare_renderer', broken)
    assert preflight.main(['--config', str(local_startup.path), '--arm', 'D3']) == 1
    assert 'FAIL ValueError: renderer fixture failure' in capsys.readouterr().out
    assert closed == [True]


@pytest.mark.parametrize('command', ['smoke', 'run'])
def test_preflight_is_first_launch_step_before_campaign_or_gpu(local_startup, monkeypatch, command):
    seen = []
    monkeypatch.setattr(cli, 'startup_config', lambda *a, **kw: (local_startup.config, 'D3'))
    def stop(config, arm, *, smoke):
        seen.append((arm, smoke))
        raise ValueError('preflight boundary')
    monkeypatch.setattr(preflight, 'preflight_config', stop)
    monkeypatch.setattr(cli, 'run_campaign', lambda *a: pytest.fail('campaign before preflight'))
    monkeypatch.setattr(runtime, 'load_backend', lambda *a: pytest.fail('model before preflight'))
    with pytest.raises(ValueError, match='preflight boundary'):
        cli.main([command, '--config', str(local_startup.path), '--arm', 'D3'])
    assert seen == [('D3', command == 'smoke')]


def test_backend_rejects_invalid_settings_before_cuda(monkeypatch):
    config = cli.load_config(ROOT/'configs/rtd/unified_alfworld_gemma4.yaml')
    config['generation_batch']['prompts_per_batch'] = 0
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: pytest.fail('CUDA before config checks'))
    with pytest.raises(ValueError, match='generation_batch requires positive integer'):
        runtime.load_backend(config, {}, None)


def test_preflight_completed_replay_checks_content_and_defers_only_initial_parameters(local_startup):
    from bfas.rtd.conventions import schedule_identity, load_schedule, hashed_step
    c = local_startup
    config, arm = cli.startup_config(c.path, 'D3')
    execution, _, _ = runtime.backend_config(config)
    manifest = cli.make_manifest(config, arm, cli.bank_audit(config), smoke=True)
    identity = schedule_identity(manifest, execution, c.support) | {'initial_parameter_hash': 'real-LoRA-hash'}
    step = hashed_step(dict(round=1, step=1, decision=True, selected=[], window_budget=0))
    schedule = dict(version='rtd-v11-exposure-v1', arm='V0', complete=True,
                    smoke=True, identity=identity, steps=[step])
    path = c.path.parent/'schedule.json'
    path.write_text(json.dumps(schedule))
    config, arm = cli.startup_config(c.path, 'D3', path)
    assert preflight.preflight_config(config, arm)['status'] == 'OK'
    with pytest.raises(ValueError, match='schedule identity differs'):
        load_schedule(config, manifest, c.support, smoke=True)
    schedule['steps'][0]['selected'] = ['tampered']
    path.write_text(json.dumps(schedule))
    config['replay_schedule_hash'] = file_hash(path)
    with pytest.raises(ValueError, match='step content hash changed'):
        preflight.preflight_config(config, arm)


@pytest.mark.parametrize('arm', ['V1', 'D0', 'D1', 'D2', 'D3', 'D3-nocross', 'D3-raw', 'D3-shuffle', 'D3-fixedmean'])
def test_preflight_streaming_constructs_reader_without_waiting(local_startup, monkeypatch, arm):
    from bfas.rtd.conventions import arm_config, schedule_identity, hashed_step
    from bfas.rtd.streaming_replay import StreamingSchedule
    c = local_startup
    execution, _, _ = runtime.backend_config(c.config)
    path = c.path.parent/'stream.json'
    config, arm = arm_config(execution if arm == 'V1' else c.config, arm, path, replay_mode='streaming')
    config = cli.validate_config(config)
    manifest = cli.make_manifest(config, arm, cli.bank_audit(config), smoke=True)
    identity = schedule_identity(manifest, config, c.support) | {'initial_parameter_hash': 'real-LoRA-hash'}
    monkeypatch.setattr(StreamingSchedule, 'wait', lambda *a: pytest.fail('CPU preflight must not poll'))
    result = preflight.preflight_config(config, arm)
    assert result['status'] == 'OK' and 'not yet published' in result['deferred'][0]
    path.write_text(json.dumps(dict(version='rtd-v11-exposure-v1', arm='V0', complete=False,
        smoke=True, identity=identity, steps=[])))
    assert preflight.preflight_config(config, arm)['deferred'] == []
    path.write_text(json.dumps(dict(version='rtd-v11-exposure-v1', arm='V0', complete=False,
        smoke=True, identity=identity, steps=[hashed_step(dict(round=2, step=1))])))
    with pytest.raises(ValueError, match='out-of-order steps'):
        preflight.preflight_config(config, arm)


def test_preflight_missing_local_tokenizer_is_an_actionable_cpu_failure(local_startup, capsys):
    model = Path(local_startup.config['student'])
    for name in ('tokenizer.json', 'tokenizer_config.json'):
        (model/name).unlink()
    assert preflight.main(['--config', str(local_startup.path), '--arm', 'D3']) == 1
    output = capsys.readouterr().out
    assert '[rtd-preflight] FAIL' in output and '[rtd-preflight] OK' not in output
