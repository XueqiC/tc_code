"""CPU acceptance of the shared CLI, evaluation dispatcher and native report."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import numpy as np
import torch
import yaml

from bfas.rtd import cli, evaluation, hardware, identity, runtime, scoring_scope
from bfas.rtd.benchmarks import alfworld_config, alfworld_evaluation, alfworld_identity, registry
from bfas.rtd.persistence import ComputeJournal, digest
from rtd_alfworld_evaluation_fixtures import campaign, FakeBackend, FakeEnv, ROOT, put
from test_rtd_alfworld_config import sealed_campaign
from test_rtd_alfworld_resume import integrated
from rtd_alfworld_runner_fixtures import RunnerBackend, feedback_context, multi_trial_window


@pytest.mark.parametrize('arm,gate', [('R0', 'linear_sigmoid'), ('R1', 'linear_sigmoid'), ('R1', 'scalar_sigmoid')])
@pytest.mark.parametrize('seed', [0, 1])
def test_nonselected_trial_full_decision_window(multi_trial_window, monkeypatch, arm, gate, seed):
    """C26-H: source sampling a parent's first trial cannot feature its other trials."""
    from bfas.behavior.deltas import tensor_state_hash
    from bfas.rtd.acquisition import PrePurchaseFeatures
    from bfas.rtd.features import FrozenProjection
    from test_rtd_alfworld_resume import comparable
    engine = multi_trial_window.engine('window', arm=arm, gate=gate, seed=seed)
    phases = []
    engine.after_save = phases.append
    engine.round_start()
    engine.step_start()
    assert engine.state['phase'] == 'reference'
    assert any(e['kind'] == 'source_sample' for e in engine.journal.events)
    assert sum(e['kind'] == 'feedback_rollout' for e in engine.journal.events) == 4
    hashes = {r.spec.state_hash for r in engine.broker._records.values()
              if r.parent_hash in engine.state['inner']}
    assert hashes.isdisjoint(engine.state['source_cache'])
    geometry = comparable((engine.state['standardizer'], engine.state['diagonal']))
    selected_resets = {s.state_hash for p, s in engine.support.states.items() if p in engine.state['inner']}
    # Before the fix this raises exactly:
    # ValueError('missing source features; run legal source sampling first').
    original_read = Path.read_text
    with monkeypatch.context() as patch:
        def public_read(path, *args, **kwargs):
            assert not path.is_relative_to(multi_trial_window.bank / 'sealed'), 'pre-purchase sealed read'
            return original_read(path, *args, **kwargs)
        patch.setattr(Path, 'read_text', public_read)
        engine.reference()
    s = engine.state
    assert set(s['source_cache']) == hashes | selected_resets
    assert comparable((s['standardizer'], s['diagonal'])) == geometry
    assert set(s['standardizer'].training_state_hashes) == selected_resets
    assert not engine.ledger.owned_ids and not engine.ledger.reservations
    for spec in s['candidate_specs'].values():
        sources = s['source_cache'][spec.state_hash]
        assert len(sources) == 2
        state = sources[0].behavior.state
        assert all(src.behavior.state == state and src.frozen_snapshot_id == s['source_id'] for src in sources)
        assert json.loads(state.task_json)['task_id'].endswith('trial-2')
        assert len(json.loads(state.history_json)) == 1
        first = engine.support.states[state.parent_hash]
        assert state.prompt == first.prompt and state.state_hash != first.state_hash
        initial_id = engine.backend.identity(s['initial'])
        hidden = engine.backend.initial_hidden(state.prompt, s['initial'], initial_snapshot_id=initial_id)
        projection = FrozenProjection(hidden.numel(), initial_snapshot_id=initial_id,
                                      seed=engine.config['gate_projection_seed'])(hidden, snapshot_id=initial_id)
        assert spec.features.projection == projection
        assert spec.features.source_logprob == sources[0].logprob
        assert spec.features.source_length == sources[0].length
        row = PrePurchaseFeatures.from_public(spec, round_id='r1', coverage=0., progress=0., support_return=0.)
        assert len(row.values) == 39 and np.isfinite(row.values).all()
    selection = s['selection']
    assert selection['query_ids'][0] is None and len(selection['probabilities']) == 3
    assert sum(selection['probabilities']) == pytest.approx(1.)
    assert (selection['beta'] is None) == (arm == 'R0')
    assert selection['pending_new_observations'] == 0
    assert s['audit_passed'] and engine.broker.assert_no_hidden_access(s['trace']).passed
    result = engine.run()
    assert result['passed'] and result['steps'] == result['decision_windows'] == 1
    assert phases == ['step_start', 'reference', 'selected', 'revealed', 'actual',
                      'feedback', 'committed', 'round_end', 'complete']
    step, = s['steps']
    q = step['selected']
    assert step['audit_passed'] and s['phase'] == 'complete'
    assert tensor_state_hash(s['parameters']) == step['actual_hash']
    assert not engine.ledger.reservations
    if q:
        assert engine.ledger.owned_ids == {q} == set(s['owned'])
        assert sum(e['kind'] == 'reveal' for e in engine.ledger.events) == 1
        assert step['actual_hash'] != step['start_hash']
        assert step['raw_new_slots'] == 8 and step['weighted_new_slots'] == 2
        assert step['label']['query_id'] == q and step['label']['label_type'] == 'pending_new'
        assert np.isfinite(step['label']['value']) and s['calibrated']
        assert set(s['posterior'].observations) == {q} == set(s['cost_model'].observations)
        assert not np.array_equal(s['posterior'].model.precision, np.eye(39))
        assert s['cost_model'].observations[q]['confidence'] == 'estimated'
        assert any(e['kind'] == 'kl_pilot' for e in engine.journal.events)
        assert any(len(json.loads(srcs[0].behavior.state.history_json)) > 1 for srcs in s['source_cache'].values())
        if arm == 'R1':
            assert step['label']['value'] != 0.
    else:
        assert step['exact_noop'] and step['actual_hash'] == step['start_hash']
        assert step['label'] is None and engine.ledger.spent == 0 and not s['owned']
        assert not s['posterior'].observations and not s['cost_model'].observations
        assert any(e['kind'] == 'feedback_reused' for e in engine.journal.events)
    if arm == 'R1':
        assert bool(q) == bool(seed)  # fixed seeds exercise both posterior outcomes
        gate_update, = [e for e in engine.journal.events if e['kind'] == 'gate_update']
        assert len(gate_update['vjp']) == (1 if gate == 'scalar_sigmoid' else 35)
    feedback = [e for e in engine.journal.events if e['kind'] == 'return_gradient']
    assert {e['role'] for e in feedback} == ({'reference_feedback', 'actual_feedback'} if q else {'reference_feedback'})
    assert all(e['metadata']['baseline'] == 'leave_one_out_same_task' and
               e['metadata']['tasks'] == 2 and e['metadata']['rollouts'] == 4 for e in feedback)


@pytest.mark.parametrize('fault', ['missing', 'feedback', 'prefix'])
def test_candidate_features_reject_illegal_states_before_sampling(multi_trial_window, monkeypatch, fault):
    from bfas.rtd.transport import FullState
    engine = multi_trial_window.engine('illegal')
    engine.round_start(); engine.step_start()
    spec = next(r.spec for r in engine.broker._records.values() if r.parent_hash in engine.state['inner'])
    if fault == 'missing':
        spec = replace(spec, state_hash='f' * 64)
        message = 'missing audited public reset'
    elif fault == 'feedback':
        spec = next(r.spec for r in engine.broker._records.values() if r.parent_hash in engine.state['feedback'])
        message = 'rejected'
    else:
        state = FullState(**multi_trial_window.payloads[spec.query_id]['behaviors'][1]['state'])
        engine.support._candidate_resets[state.state_hash] = state
        spec = replace(spec, state_hash=state.state_hash)
        message = 'never a teacher prefix'
    monkeypatch.setattr(engine.broker, 'list_candidates', lambda *a: [spec])
    monkeypatch.setattr(engine, 'sample_state', lambda *a: pytest.fail('sampled an illegal candidate'))
    before = engine.sampling_rng.get_state().clone()
    with pytest.raises(ValueError, match=message):
        engine.reference()
    assert torch.equal(before, engine.sampling_rng.get_state())
    assert not engine.ledger.owned_ids


@pytest.mark.parametrize('parents', [2, 4])
def test_shared_cli_cpu_smoke_resume_evaluate_report(sealed_campaign, monkeypatch, capsys, parents):
    c = sealed_campaign
    monkeypatch.setattr(cli, 'ROOT', c.root)
    monkeypatch.setattr(cli, 'hardware_identity', lambda: c.hardware)
    monkeypatch.setattr(hardware, 'hardware_identity', lambda: c.hardware)
    monkeypatch.setattr(registry.ALFWorldExperimentSupport, 'feedback_context', feedback_context)
    monkeypatch.setattr(cli, '_checker_context', lambda: pytest.fail('ALFWorld started CheckerBridge'))
    monkeypatch.setattr(torch.cuda, '_lazy_init', lambda *a, **k: pytest.fail('CPU smoke initialized CUDA'))
    backends = []
    def load(config, manifest, journal):
        # The existing runtime.load_backend seam is the only model substitution.
        # The fixture binds CPU accounting before any measured runner operation.
        journal.cuda = False
        backend = RunnerBackend()
        backends.append(backend)
        return backend
    monkeypatch.setattr(runtime, 'load_backend', load)
    config_path = c.root / 'config.yaml'
    config_path.write_text(yaml.safe_dump(c.config))
    directory = c.root / 'results' / 'cpu-cli-smoke'
    assert cli.main(['smoke', '--config', str(config_path), '--arm', 'R1', '--run-dir', str(directory),
                     '--smoke-parents-per-fold', str(parents)]) == 0
    saved_bytes = (directory / 'manifest.json').read_bytes()
    saved = json.loads(saved_bytes)
    assert saved['smoke'] and saved['config']['smoke_override']['parents_per_fold'] == parents
    assert saved['config']['smoke_override']['rollouts'] == 2
    assert 'historical_demo_output_exact' not in saved['resources']
    assert not saved['recorded_bank_usage']['provider_usage_exact']
    trajectory = json.loads((directory / 'trajectory.json').read_text())
    assert len(trajectory['steps']) == 1 and trajectory['steps'][0]['slots'] == 8
    assert cli.main(['resume', '--config', str(config_path), '--arm', 'R1', '--run-dir', str(directory)]) == 0
    assert (directory / 'manifest.json').read_bytes() == saved_bytes
    events = ComputeJournal(directory / 'compute.jsonl').events
    feedback = [e for e in events if e['kind'] == 'return_gradient']
    assert feedback and all(e['metadata']['tasks'] == parents and
        e['metadata']['rollouts'] == parents * 2 and e['metadata']['baseline'] == 'leave_one_out_same_task' for e in feedback)
    assert all(e['gpu_seconds'] == 0 for e in events if e['kind'] == 'compute_end')
    # Operational source drift may be logged during evaluation without changing
    # the frozen scoring harness or rewriting the native campaign receipt.
    put(c.root / 'src/c26f_operational.py', 'OPERATIONAL_LOGGING = True\n')
    original_evaluate = alfworld_evaluation.evaluate
    def evaluate_cpu(root, binding, **kwargs):
        return original_evaluate(root, binding, **kwargs,
            backend_factory=lambda _: FakeBackend(),
            env_factory=lambda tid: FakeEnv(tid, won=int(tid.rsplit('_', 1)[1]) < 109))
    monkeypatch.setattr(alfworld_evaluation, 'evaluate', evaluate_cpu)
    monkeypatch.setattr(evaluation, 'reserve_port', lambda *a, **k: pytest.fail('ALFWorld leased HTTP port'))
    capsys.readouterr()
    assert cli.main(['evaluate', '--run-dir', str(directory), '--round', '1', '--port', '12345',
                     '--evaluation-lock-timeout', '0', '--evaluation-lock-log-interval', '.01']) == 0
    metrics = json.loads(capsys.readouterr().out)
    assert metrics == dict(success_rate=109/140, success_percent=100*109/140, complete=True)
    receipt = (directory / 'evaluation-1.json').read_bytes()
    assert 'overall_accuracy_percent' not in json.loads(receipt)
    # Full reuse validates every artifact without constructing a policy model.
    monkeypatch.setattr(alfworld_evaluation, 'HFBackend', lambda _: pytest.fail('reuse loaded model'))
    assert evaluation.evaluate(c.root, directory, 1, lock_timeout=0) == json.loads(receipt)
    assert (directory / 'evaluation-1.json').read_bytes() == receipt
    output = directory / 'report'
    assert cli.main(['report', '--run-dir', str(directory), '--out', str(output)]) == 0
    row, = json.loads((output / 'budget_curve.json').read_text())['rows']
    assert row['success_percent'] == 100*109/140 and row['success_rate'] == 109/140
    assert row['historical_usage']['missing_responses'] == 112 and row['smoke']
    assert 'historical_demo_output_exact' not in row and 'official_accuracy_percent' not in row
    assert row['episode_steps'] > 0 and row['feedback_blocks'] == len(feedback)
    assert row['category_success']['pick_and_place_simple']['tasks'] == 140
    assert row['estimated_cost'] == row['actual_spend_x']
    assert any(e['kind'] == 'code_drift' and 'src/c26f_operational.py' in e['files']
               for e in row['code_drift'])
    transport = [e for e in ComputeJournal(directory / 'compute.jsonl').events if e['kind'] == 'evaluation_transport']
    assert transport[0]['requested_port'] == 12345 and transport[0]['port_used'] is False
    assert (directory / 'manifest.json').read_bytes() == saved_bytes
    artifact = next((directory.parent / (directory.name + '-alfworld-evaluations') / 'round-1/artifacts/tasks').glob('*.json'))
    artifact.write_text('{}')
    with pytest.raises(ValueError, match='artifacts'):
        evaluation.report([directory], directory / 'rejected-report')


def test_pool_rejects_unpurchased_and_cross_fold_before_source(integrated, monkeypatch):
    c = integrated
    engine = c.engine('guards')
    engine.round_start()
    engine.step_start(); engine.reference()
    candidates = engine.state['candidate_specs']
    q = next(iter(candidates))
    from bfas.rtd.broker import PurchasedEvidencePackage
    from bfas.rtd.transport import Behavior, FullState
    raw = json.loads((c.bank / f'sealed/{q}.json').read_text())
    package = PurchasedEvidencePackage(q, (), tuple(Behavior(FullState(**b['state']), b['text']) for b in raw['behaviors']),
        raw['cost'], raw['cost_confidence'], raw['usage'], raw['provenance'], raw['historical_response'])
    before = engine.sampling_rng.get_state().clone()
    with pytest.raises(ValueError, match='purchased'):
        engine.pool([package], package_only=True)
    assert torch.equal(before, engine.sampling_rng.get_state())
    purchased = engine.broker.acquire(q)
    assert engine.pool([purchased], package_only=True)
    wrong = next(iter(engine.state['feedback']))
    engine.support.states[next(iter(engine.state['inner']))] = engine.support.states[wrong]
    monkeypatch.setattr(engine, 'sample_state', lambda *a: pytest.fail('guard ran after source'))
    with pytest.raises(ValueError, match='rejected'):
        engine.pool()


def test_alfworld_harness_strict_no_bfcl_legacy_escape(integrated):
    c = integrated
    engine = c.engine('identity')
    saved = engine.manifest
    assert identity.guard_harness(c.root, engine.directory, saved) == saved
    legacy = c.root / 'configs/rtd/legacy_identities' / f'{digest(saved)}.json'
    put(legacy, dict(version='rtd-c25j-audited-identity-v1', harness_hash='foreign'))
    assert identity.saved_identities(c.root, engine.directory, saved) == saved
    with pytest.raises(ValueError, match='only supports BFCL'):
        identity.audit_legacy(c.root, engine.directory)
    with pytest.raises(ValueError, match='training harness is frozen'):
        cli.main(['update-identity', '--run-dir', str(engine.directory)])
    changed = deepcopy(saved['evaluation_harness']); changed['version'] = 'drift'
    with pytest.raises(ValueError, match='audited ALFWorld'):
        identity.guard_harness(c.root, engine.directory, saved, current=changed)
    with pytest.raises(ValueError, match='binding'):
        identity.audited_harness_hashes(saved, dict(saved, evaluation_harness=changed))


def test_common_alfworld_projection_mixed_scopes_fail_closed():
    name = 'src/bfas/rtd/benchmarks/registry.py'
    content = (ROOT / name).read_text()
    projection = lambda text: scoring_scope.scoring_hash(name, text, benchmark='alfworld')
    before = projection(content)
    assert projection(content.replace("'evaluation_transport'", "'transport_log'")) == before
    assert projection(content.replace('2**63 - 1', '2**62 - 1')) != before
    assert projection(content.replace("config['alfworld_data_root']", "config['wrong_data_root']")) != before
    for changed in (content.replace('def alfworld_feedback_rollout(', 'def missing_rollout('),
                    content + '\ndef alfworld_feedback_rollout(): pass\n'):
        with pytest.raises(ValueError, match='one definition'):
            projection(changed)


def test_native_wait_interval_reaches_common_policy(campaign, monkeypatch):
    from bfas.rtd import evaluation_lock as locks
    c = campaign
    observed = []
    original = locks.wait_settings
    def settings(timeout, log_interval):
        observed.append((timeout, log_interval))
        return original(timeout, log_interval)
    monkeypatch.setattr(locks, 'wait_settings', settings)
    with locks.evaluation_lock(locks.tag_lock_path(c.root, c.tag, benchmark='alfworld'), tag='held', timeout=0):
        with pytest.raises(TimeoutError):
            alfworld_evaluation.evaluate(c.root, c.manifest, **c.kwargs, lock_log_interval=.013)
    assert observed[-1] == (0, .013)
    for tag in ('../escape', '..', 'a/b', 'a\\b', None):
        for benchmark in ('bfcl', 'alfworld'):
            with pytest.raises(ValueError, match='tag'):
                locks.tag_lock_path(c.root, tag, benchmark=benchmark)


def test_action_cap_class_validation_and_runtime_binding(monkeypatch):
    from test_rtd_checks import TinyLM, TinyTokenizer
    from bfas.rtd.return_gradient import TorchPolicyBackend
    model = TinyLM().eval()
    kwargs = dict(base_checkpoint_hash='fixture', harness_hash='fixture', tokenizer_hash='fixture')
    backend = TorchPolicyBackend(model, TinyTokenizer(), action_caps={'agent_action': 256},
                                 max_action_tokens=256, **kwargs)
    with backend.action_limit('agent_action'):
        assert backend.max_action_tokens == 256
    for category in ('unknown', 'simple_python'):
        with pytest.raises(ValueError, match='class'):
            with backend.action_limit(category):
                pass
    for caps in ({'agent_action': 512}, {'agent_action': 256, 'single_turn': 512}, {'unknown': 256}):
        with pytest.raises(ValueError):
            TorchPolicyBackend(model, TinyTokenizer(), action_caps=caps, **kwargs)
    from contextlib import nullcontext
    import sys
    # Inspect the real loader's constructor arguments without a GPU allocation.
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 1)
    monkeypatch.setattr(torch.cuda, 'set_device', lambda _: None)
    monkeypatch.setattr(torch.cuda, 'manual_seed_all', lambda _: None)
    monkeypatch.setattr(torch, 'manual_seed', lambda _: None)
    monkeypatch.setattr(model, 'to', lambda _: model)
    monkeypatch.setattr(runtime, 'enable_gradient_checkpointing', lambda _: 0)
    monkeypatch.setitem(sys.modules, 'peft', SimpleNamespace(LoraConfig=lambda **k: k, get_peft_model=lambda m, c: m))
    monkeypatch.setitem(sys.modules, 'transformers', SimpleNamespace(
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=lambda *a, **k: model),
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: TinyTokenizer())))
    monkeypatch.setattr(runtime, 'HFGenerateBackend', lambda *a, **k: k)
    journal = SimpleNamespace(measure=lambda *a: nullcontext(), append=lambda *a, **k: None)
    result = runtime.load_backend(alfworld_config.default_config(),
        dict(model_path='fixture', base_checkpoint_hash='fixture', harness_hash='fixture', tokenizer_hash='fixture'), journal)
    assert result['max_action_tokens'] == 256 and result['action_caps'] == {'agent_action': 256}
    assert result['memory_policy'].max_batch_size == 1


def test_failed_no_payload_parents_remain_feedback_eligible(integrated):
    engine = integrated.engine('failed-parents')
    engine.round_start()
    available = {r.parent_hash for r in engine.broker._records.values() if r.unavailable_reason is None}
    missing = sorted(engine.support.protocol.parent_hashes(1, use='feedback') - available)
    assert len(missing) >= 4
    engine.state['feedback'] = set(missing)
    selected = engine.choose_feedback_tasks()
    assert len(selected) == 4 and all(p in missing and count == 2 for p, count in selected)


def test_alfworld_rejects_inventory_build_and_loo_smoke_drift(sealed_campaign, monkeypatch):
    c = sealed_campaign
    monkeypatch.setattr(cli, 'ROOT', c.root)
    with pytest.raises(ValueError, match='verified bank'):
        cli.bank_audit(c.config, build=True)
    override = dict(parents_per_fold=4, slots=8, rollouts=2, windows=1,
        baseline='leave_one_out_same_task', max_seconds=900)
    for key, value in [('slots', 2), ('rollouts', 1), ('baseline', 'smoke_zero'), ('windows', 2)]:
        with pytest.raises(ValueError, match='smoke'):
            alfworld_config.validate_config(dict(c.config, smoke_override=dict(override, **{key: value})))


def test_alfworld_component_swap_changes_only_gate(tmp_path):
    output = tmp_path / 'scalar.yaml'
    assert cli.main(['swap-component', '--config', str(ROOT / 'configs/rtd/v1_alfworld_c26.yaml'),
        '--component', 'gate', '--value', 'scalar_sigmoid', '--out', str(output)]) == 0
    assert cli.load_config(output) == cli.load_config(ROOT / 'configs/rtd/v1_alfworld_c26_scalar_gate.yaml')


@pytest.mark.parametrize('command', ['run', 'resume', 'smoke'])
def test_hpg_launcher_preserves_args_gpu_binding_and_writable_outputs(tmp_path, command):
    import os
    import subprocess
    import sys
    import shlex
    script = ROOT / 'scripts/rtd_alfworld_run_hpg.slurm'
    content = script.read_text()
    for directive in ('--mem=64G', '--cpus-per-task=8', '--gres=gpu:b200:1',
                      '--time=48:00:00', '--output=results/c26f/hpg/%x_%j.out'):
        assert '#SBATCH ' + directive in content
    subprocess.run(['bash', '-n', str(script)], check=True)
    # Exercise the real launcher/helper with a recorder in place of Python;
    # the synthetic allocation string is never passed to a GPU API.
    tools = tmp_path / 'tools'
    tools.mkdir()
    (tools / 'aw_hpg_common.sh').write_bytes((ROOT / 'tools/aw_hpg_common.sh').read_bytes())
    recorder = tmp_path / 'record.py'
    recorder.write_text('import json, os, sys\nprint(json.dumps(dict(args=sys.argv[1:], env=dict(os.environ))))\n')
    executable = tmp_path / 'python-recorder'
    executable.write_text('#!/usr/bin/env bash\nexec ' + shlex.quote(sys.executable) + ' ' +
                          shlex.quote(str(recorder)) + ' "$@"\n')
    executable.chmod(0o755)
    extra = '--evaluation-lock-timeout 0\n--evaluation-lock-log-interval .25'
    if command == 'smoke':
        extra += ' --smoke-parents-per-fold 2'
    env = dict(os.environ, SLURM_SUBMIT_DIR=str(tmp_path), SLURM_JOB_ID='12345',
        CUDA_VISIBLE_DEVICES='GPU-fixture-only', RTD_COMMAND=command, RTD_ARM='R1',
        RTD_CONFIG='configs/rtd/v1_alfworld_c26_scalar_gate.yaml', RTD_PYTHON=str(executable),
        RTD_RUN_DIR=str(tmp_path / 'results/R1s'), RTD_EXTRA_ARGS=extra)
    result = subprocess.run(['bash', str(script)], env=env, text=True, capture_output=True, check=True)
    recorded = json.loads(result.stdout)
    assert recorded['args'] == ['tools/rtd_experiment.py', command, '--arm', 'R1', '--config',
        env['RTD_CONFIG'], '--run-dir', env['RTD_RUN_DIR'], *extra.split()]
    values = recorded['env']
    assert values['CUDA_VISIBLE_DEVICES'] == 'GPU-fixture-only'
    assert values['ALFWORLD_DATA'] == values['ALFWORLD_DATA_ROOT'] == str(tmp_path / 'envs/alfworld/data')
    assert values['ALFRED_DATA'] == str(tmp_path / 'envs/alfworld/data/json_2.1.1')
    for key in ('XDG_CACHE_HOME', 'TRITON_CACHE_DIR', 'TORCH_HOME', 'CUDA_CACHE_PATH', 'TMPDIR'):
        assert Path(values[key]).is_relative_to(tmp_path / 'results')
    assert values['PYTHONDONTWRITEBYTECODE'] == '1' and not (tmp_path / 'envs').exists()


@pytest.mark.parametrize('arm,gate', [('R0', 'linear_sigmoid'), ('R1', 'linear_sigmoid'), ('R1', 'scalar_sigmoid')])
def test_shared_campaign_all_rounds_resume_workers_and_rotate_folds(sealed_campaign, monkeypatch, arm, gate):
    c = sealed_campaign
    monkeypatch.setattr(cli, 'ROOT', c.root)
    monkeypatch.setattr(cli, 'hardware_identity', lambda: c.hardware)
    monkeypatch.setattr(hardware, 'hardware_identity', lambda: c.hardware)
    monkeypatch.setattr(registry.ALFWorldExperimentSupport, 'feedback_context', feedback_context)
    monkeypatch.setattr(torch.cuda, '_lazy_init', lambda *a, **k: pytest.fail('CPU campaign initialized CUDA'))
    monkeypatch.setattr(cli, '_checker_context', lambda: pytest.fail('ALFWorld started BFCL checker'))
    loads, workers = [], []
    def load(config, manifest, journal):
        journal.cuda = False
        loads.append(manifest['config_hash'])
        return RunnerBackend()
    monkeypatch.setattr(runtime, 'load_backend', load)
    original_campaign = alfworld_evaluation.evaluate
    def evaluate_cpu(root, binding, **kwargs):
        return original_campaign(root, binding, **kwargs,
            backend_factory=lambda _: FakeBackend(), env_factory=lambda tid: FakeEnv(tid, won=False))
    monkeypatch.setattr(alfworld_evaluation, 'evaluate', evaluate_cpu)
    original_subprocess = cli.subprocess.run
    def worker(command, **kwargs):
        if len(command) > 1 and command[1] == str(c.root / 'tools/rtd_experiment.py'):
            workers.append(command)
            assert '--training-worker' in command and '--expected-gpu-uuid' in command
            assert kwargs['cwd'] == c.root
            return cli.subprocess.CompletedProcess(command, cli.main(command[2:]))
        return original_subprocess(command, **kwargs)
    monkeypatch.setattr(cli.subprocess, 'run', worker)
    config_path = c.root / 'config.yaml'
    config_path.write_text(yaml.safe_dump(dict(c.config, gate=gate)))
    directory = c.root / 'results/full-campaign'
    assert cli.main(['run', '--config', str(config_path), '--arm', arm, '--run-dir', str(directory)]) == 0
    saved = (directory / 'manifest.json').read_bytes()
    trajectory = json.loads((directory / 'trajectory.json').read_text())
    assert len(trajectory['steps']) == 36 and sum(s['decision'] for s in trajectory['steps']) == 12
    assert [s['round'] for s in trajectory['checkpoints']] == [1, 2, 3]
    assert [command[2] for command in workers] == ['run', 'resume', 'resume']
    assert len(loads) == 3
    assert all(s['slots'] == 8 and s['inner_fold'] == (s['round'] - 1) % 2 for s in trajectory['steps'])
    gradients = [e for e in ComputeJournal(directory / 'compute.jsonl').events if e['kind'] == 'return_gradient']
    assert {e['round'] for e in gradients} == {1, 2, 3}
    assert all(e['metadata']['tasks'] == 4 and e['metadata']['rollouts'] == 8 and
               e['metadata']['baseline'] == 'leave_one_out_same_task' for e in gradients)
    # Completed campaign resume revalidates all three 140-task artifacts and
    # checkpoints; it must neither launch another worker nor rewrite the run.
    assert cli.main(['resume', '--arm', arm, '--run-dir', str(directory)]) == 0
    assert len(workers) == len(loads) == 3 and (directory / 'manifest.json').read_bytes() == saved
