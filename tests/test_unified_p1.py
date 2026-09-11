"""P1 CPU contracts and real tiny-backend execution; no GPU/API experiment."""
from copy import deepcopy
from dataclasses import replace
import json
import re

import numpy as np
import pytest
import torch
import yaml

from bfas.rtd import cli
from bfas.rtd.persistence import digest, atomic_json
from bfas.rtd.unified.arms import ARMS, equality_rows, solve_arm, source_permutation
from bfas.rtd.unified.config import arm_config
from bfas.rtd.unified.experiment import P1Experiment
from bfas.rtd.unified.objective import TeachingObjective
from bfas.rtd.unified.profiling import STAGES, profile, profile_line
from bfas.rtd.unified.solver import solve_exact
from unified_fixtures import from_directions
from test_rtd_checks import toy_bank, tiny_backend, TinySupport


@pytest.mark.parametrize('benchmark', ['bfcl', 'alfworld', 'webshop'])
def test_benchmark_configs_share_budget_exposure_and_student(benchmark):
    config = cli.load_config(f'configs/rtd/unified_{benchmark}_gemma4.yaml')
    assert config['arm'] == 'D3' and config['memory_peak_budget_gb'] == 60
    assert config['student'] == 'google/gemma-4-12B-it'
    base = yaml.safe_load(open(f'configs/rtd/v1_1_{benchmark}.yaml'))
    for k in ('rounds', 'budget_checkpoints_bank_fraction', 'decision_steps_per_round',
              'committed_steps_per_round', 'slots_per_step', 'exposure_slots_per_window',
              'max_new_packages_per_window', 'hard_cost_reservation', 'new_teacher_calls',
              'new_teacher_tokens', 'source_temperature', 'source_top_p'):
        assert config[k] == base[k]
    for arm in ARMS:
        selected = cli.validate_config(config, arm=arm)
        assert selected['arm'] == arm
        assert selected['unified']['estimator'] == ARMS[arm].estimator
    for key, value in [('rounds', 3), ('memory_peak_budget_gb', 38), ('new_teacher_calls', True),
                       ('source_top_p', .9), ('optimizer', 'adam'), ('initial_eta', float('nan'))]:
        with pytest.raises(ValueError):
            cli.validate_config(config | {key: value})


@pytest.mark.parametrize('name', [
    'v1_1_alfworld_luna', 'unified_alfworld_gemma4_luna', 'unified_alfworld_gemma4_d0_luna'])
def test_score_consistency_luna_override_survives_frozen_p1_validation(name):
    from bfas.rtd.scoring import ScoreTolerance
    filename = f'configs/rtd/{name}.yaml'
    with open(filename) as stream:
        declared = yaml.safe_load(stream)['score_consistency_tolerance']
    config = cli.load_config(filename)
    # Operational Luna guard values may change independently of the validator.
    # Test preservation of the actual declaration, not a stale campaign value.
    expected = vars(ScoreTolerance()) | declared
    assert config['score_consistency_tolerance'] == expected
    assert cli.validate_config(config)['score_consistency_tolerance'] == expected
    # An omitted or partial declaration keeps all other strict defaults.
    config.pop('score_consistency_tolerance')
    assert cli.validate_config(config)['score_consistency_tolerance'] == vars(ScoreTolerance())
    assert cli.validate_config(config | {'score_consistency_tolerance': {'mean_abs': .08}})[
        'score_consistency_tolerance'] == dict(mean_abs=.08, max_abs=1., max_abs_outlier_tokens=2,
                                             max_abs_hard=8., min_tokens_for_mean=8)
    with pytest.raises(ValueError, match='finite and nonnegative'):
        cli.validate_config(config | {'score_consistency_tolerance': {'mean_abs': -1}})


def test_D2_matches_full_D3_with_tied_coefficients_and_scalar_grid():
    p = from_directions([[1., .8], [0., .6]], h=[.7, -.3], epsilon=[.03, .01])
    d2, a, _ = solve_arm(p, 'D2')
    tied = solve_exact(p, equality=np.array([[1., -1.]]))
    np.testing.assert_allclose(a, tied.coefficients.numpy(), atol=1e-8)
    assert a[0] == pytest.approx(a[1], abs=1e-10)
    objective = TeachingObjective(p)
    best = max(objective.value([v, v]) for v in np.linspace(0, 1, 2001))
    assert d2.value >= best-1e-10
    assert solve_exact(p).value >= d2.value-1e-10


def test_nocross_diagonal_controller_reports_full_objective():
    p = from_directions([[1., 1.], [0., 1.]], h=[.8, -.8], epsilon=[0., .1])
    _, a, _ = solve_arm(p, 'D3-nocross')
    # Independent 1D soft threshold with the true diagonal units.
    z, k = p.U.numpy().T@p.h.numpy(), np.diag(p.K.numpy())
    expected = .5+np.clip(np.sign(z)*np.maximum(abs(z)-p.epsilon.numpy(), 0)/k, -.5, .5)
    np.testing.assert_allclose(a, expected, atol=1e-8)
    assert abs(a[1]-.5) < 1e-8
    assert abs(solve_arm(p, 'D3')[1][1]-.5) > .1
    assert p.K.numpy()[0, 1] != 0  # immutable full F is not overwritten


def test_shuffle_is_source_permutation_preserves_coefficients_and_teacher_dose():
    p = from_directions([[1., .2, .7, -.5], [0., 1., .1, .2]], h=[.3, -.7])
    solved, a, permutation = solve_arm(p, 'D3-shuffle', seed=4)
    assert sorted(permutation) == list(range(4))
    np.testing.assert_array_equal(a, solved.coefficients.numpy()[permutation])
    for i, j in enumerate(permutation):
        assert p.coordinates[i].evidence_key == p.coordinates[j].evidence_key
        assert p.coordinates[i].slot_id == p.coordinates[j].slot_id
    assert sorted(a) == sorted(solved.coefficients.numpy())
    assert any(not np.array_equal(source_permutation(p, seed), np.arange(4)) for seed in range(10))


@pytest.mark.parametrize('reference', [.2, .5, .8])
def test_fixedmean_keeps_sum_per_slot_and_total(reference):
    p = from_directions([[1., .3, .8, -.2], [0., 1., .1, 1.]], h=[.7, -.8], a_ref=reference)
    _, a, _ = solve_arm(p, 'D3-fixedmean')
    assert sum(a) == pytest.approx(4*reference, abs=1e-9)
    np.testing.assert_allclose(equality_rows(p, 'fixed_mean')@(a-p.a_ref.numpy()), 0., atol=1e-9)
    assert not np.allclose(a, reference)


def test_profile_format_cpu_and_failed_stage(tmp_path, capsys):
    from bfas.rtd.persistence import ComputeJournal
    journal = ComputeJournal(tmp_path/'compute.jsonl')
    for stage in STAGES:
        with profile(journal, stage):
            pass
    with pytest.raises(RuntimeError), profile(journal, 'qp'):
        raise RuntimeError('failure probe')
    lines = capsys.readouterr().out.splitlines()
    pattern = r'\[rtd-profile\] stage=\w+ wall_seconds=\d+\.\d{6} peak_gpu_gb=0\.000000 peak_reserved_gb=0\.000000 status=(complete|failed)'
    assert len(lines) == 10 and all(re.fullmatch(pattern, line) for line in lines)
    assert lines[-1].endswith('status=failed')
    assert 'peak_gpu_gb=1.000000' in profile_line('commit', 1., 10**9)


def p1_engine(path, bank, arm='D3', *, resume=False, replay_schedule=None):
    config = cli.load_config('configs/rtd/unified_bfcl_gemma4.yaml', arm=arm)
    config.update(slots_per_step=2, max_new_packages_per_window=1)
    if replay_schedule:
        config, _ = arm_config(config, arm, replay_schedule)
    backend = tiny_backend()
    model = backend.model
    model.head = torch.nn.Identity()
    model.get_output_embeddings = lambda: model.head
    original = model.forward
    def forward(*args, **kwargs):
        out = original(*args, **kwargs); out.logits = model.head(out.logits); return out
    model.forward = forward
    directory, _, _, states = bank
    manifest = dict(config_hash=digest(config), config=config, arm=arm, bank_path=str(directory),
        budget_ceilings=[100000, 200000, 300000], bank_public_cap_sum=32768, hardware_hash='toy-cpu', smoke=True)
    path.mkdir(exist_ok=True)
    atomic_json(path/'manifest.json', manifest)
    return P1Experiment(config, manifest, path, backend, TinySupport(states), resume=resume, smoke=True)


@pytest.mark.parametrize('arm', list(ARMS))
def test_all_arms_tiny_smoke_one_commit_and_three_matched_feedback_roles(toy_bank, tmp_path, arm, capsys):
    e = p1_engine(tmp_path/arm, toy_bank, arm)
    result = e.run()
    assert result['passed'] and result['steps'] == result['decision_windows'] == 1
    assert len(e.state['steps']) == 1 and e.state['steps'][0]['slots'] == 2
    roles = [ev['role'] for ev in e.journal.events if ev['kind'] == 'feedback_rollout']
    assert len(roles) == 6
    assert set(roles) == {'acquisition_reference_feedback', 'same_batch_reference_feedback', 'post_commit_feedback'}
    assert len(e.ledger.owned_ids) == 1 and not e.state['posterior'].observations
    observed = json.loads((e.directory/'manifest.json').read_text())['score_consistency_observed']
    diagnostics = [ev for ev in e.journal.events if ev['kind'] == 'score_consistency']
    assert observed['checks'] == len(diagnostics) > 0
    assert observed['compared_tokens'] == sum(ev['n_tokens'] for ev in diagnostics)
    assert observed['max_abs_difference'] == max(ev['max_abs_difference'] for ev in diagnostics)
    output = capsys.readouterr().out
    assert all(f'stage={stage} ' in output for stage in STAGES)
    before = deepcopy(e.state['steps'])
    restored = p1_engine(tmp_path/arm, toy_bank, arm, resume=True)
    restored.run()
    assert restored.state['steps'] == before


def test_all_arms_replay_one_D14_schedule_and_detect_tampering(toy_bank, tmp_path):
    from test_rtd_v11_alpha_d import engine
    source = engine(tmp_path/'V0', toy_bank, arm='V0', slots_per_step=2, max_new_packages_per_window=1)
    source.run()
    expected = source.state['steps'][0]['exposure_schedule']
    for arm in ARMS:
        target = p1_engine(tmp_path/arm, toy_bank, arm, replay_schedule=source.directory)
        target.run()
        assert target.state['steps'][0]['exposure_schedule'] == expected
        assert target.ledger.charges == source.ledger.charges
    path = source.directory/'exposure_schedule.json'
    data = json.loads(path.read_text()); data['steps'][0]['selected'] = []
    atomic_json(path, data)
    with pytest.raises(ValueError, match='schedule changed'):
        target.run()


@pytest.mark.parametrize('normalization', ['per_sequence_mean', 'total_token_nll'])
def test_streamed_gradients_match_dense_P0_and_D0_source_prefix_KL(normalization):
    from test_rtd_v11_alpha_d import pair_problem
    from bfas.rtd.unified.immutable import Parameters
    from bfas.rtd.unified.problem import TeachingContext
    from bfas.rtd.unified.estimators import LossDefinition
    from bfas.rtd.unified.scoring import score_streamed, score_source_samples, score_teacher
    from bfas.rtd.functional_step import gradients
    from appworld_train import rtd_sft_kl_gradient
    backend, parameters, source, pairs, step = pair_problem()
    model = backend.model
    model.head = torch.nn.Identity(); model.get_output_embeddings = lambda: model.head
    original = model.forward
    def forward(*args, **kwargs):
        out = original(*args, **kwargs); out.logits = model.head(out.logits); return out
    model.forward = forward
    layout = Parameters.of(parameters)
    context = TeachingContext(layout, source, backend.identity(source), layout.flatten(step.diagonal),
        loss=LossDefinition(normalization, retention_scale=.13),
        owned_query_ids=('paid',), inner_parent_hashes=(pairs[0].record.state.parent_hash,))
    streamed, teacher = score_streamed(backend, pairs[0], context, parameters, source, slot_id='test', weight=1.)
    dense = score_source_samples(backend, pairs[0].sources, context, slot_id='test', weight=1.)
    dense_teacher = score_teacher(backend, pairs[0].record.teacher, context, query_id='paid', version='0')
    np.testing.assert_allclose(streamed.hard.numpy(), dense.hard.numpy(), atol=1e-12)
    np.testing.assert_allclose(streamed.soft.numpy(), dense.soft.numpy(), atol=1e-12)
    np.testing.assert_allclose(teacher.gradient.numpy(), dense_teacher.gradient.numpy(), atol=1e-12)
    loss = 0.
    for sample in pairs[0].sources:
        prompt = tuple(backend.tokenizer.encode(sample.behavior.state.prompt, add_special_tokens=False))
        ids = torch.tensor([prompt+sample.token_ids])
        new = backend._logits(parameters, ids)[0, len(prompt)-1:-1].log_softmax(-1)
        old = backend._logits(source, ids)[0, len(prompt)-1:-1].log_softmax(-1).detach()
        loss = loss+.7*(old.exp()*(old-new)).sum()/(2*(sample.length if normalization == 'per_sequence_mean' else 1))
    g = gradients(loss, parameters)
    expected = layout.flatten(g).numpy()+1.3*teacher.gradient.numpy()
    actual = rtd_sft_kl_gradient(backend, pairs[:1], [1.], parameters, source, mode='rtd_sft_kl',
        teacher_weight=1.3, kl_weight=.7, normalization=normalization)
    np.testing.assert_allclose(layout.flatten(actual).numpy(), expected, atol=1e-12)


def test_cli_unified_smoke_dispatch_deadline_and_manifest(toy_bank, tmp_path, monkeypatch):
    from types import SimpleNamespace
    from bfas.rtd import preflight, runtime
    from bfas.rtd.benchmarks import registry
    template = p1_engine(tmp_path/'template', toy_bank)
    template.support.unavailable = {}
    monkeypatch.setattr(runtime, 'load_backend', lambda *args: template.backend)
    monkeypatch.setattr(runtime, 'load_tokenizer', lambda *args: template.backend.tokenizer)
    # This dispatch fixture uses BFCL-shaped toy states under an ALF config;
    # real cached renderer/bank coverage lives in test_rtd_preflight.py.
    monkeypatch.setattr(preflight, 'prepare_renderer', lambda *args: len(template.support.states))
    monkeypatch.setattr(cli, 'bank_audit', lambda *args: {'available_packages': len(toy_bank[1])})
    monkeypatch.setattr(registry, 'get_benchmark', lambda config: SimpleNamespace(
        support_protocol=lambda *args: template.support))
    from contextlib import nullcontext
    monkeypatch.setattr(cli, '_checker_context', lambda *args: nullcontext())
    def manifest(config, arm, audit, smoke=False, hardware=None):
        return template.manifest | dict(config=config, config_hash=digest(config), arm=arm,
            hardware={'hard': {}, 'metadata': {'uuid': 'test', 'gpu': 'CPU', 'memory': 0}})
    monkeypatch.setattr(cli, 'make_manifest', manifest)
    # CLI's GPU accounting is replaced only in this explicit CPU fixture.
    from bfas.rtd.persistence import ComputeJournal
    monkeypatch.setattr(cli, 'ComputeJournal', lambda path, **kw: ComputeJournal(path, cuda=False,
        deadline=kw.get('deadline'), deadline_seconds=kw.get('deadline_seconds')))
    monkeypatch.setattr(cli, 'instance', lambda hardware: {'uuid': 'test'})
    monkeypatch.setattr(cli, 'device_class', lambda hardware: {'gpu': 'CPU', 'memory': 0})
    directory = tmp_path/'cli-smoke'
    assert cli.main(['smoke', '--config', 'configs/rtd/unified_alfworld_gemma4.yaml', '--arm', 'D3',
        '--run-dir', str(directory), '--smoke-deadline-seconds', '120']) == 0
    saved = json.loads((directory/'manifest.json').read_text())
    assert saved['arm'] == 'D3' and saved['config']['arm_preset']['trainer'] == 'unified'
    audit = json.loads((directory/'audit.json').read_text())
    assert audit['steps'] == 1 and audit['smoke_seconds'] < 120


def test_native_teacher_termination_not_counted_twice():
    from types import SimpleNamespace
    from bfas.rtd.student import teacher_tokens
    tokenizer = SimpleNamespace(eos_token_id=0, unk_token_id=-1,
        encode=lambda *a, **kw: [1, 7], convert_tokens_to_ids=lambda t: 7)
    backend = SimpleNamespace(tokenizer=tokenizer, student_config={'student': 'google/gemma-4-12B-it', 'benchmark': 'alfworld'})
    assert teacher_tokens(backend, SimpleNamespace(text='native turn')) == ((1, 7), 7)


def test_effective_defaults_are_frozen_for_resume(monkeypatch):
    from pathlib import Path
    from bfas.rtd.unified.config import runtime_config
    c = cli.load_config('configs/rtd/unified_bfcl_gemma4.yaml', arm='D1')
    assert c['p1_runtime_defaults_frozen'] and 'gate_projection_seed' in c
    before = runtime_config(c)
    original = Path.read_text
    def forbid_defaults(path, *args, **kwargs):
        if path.name == 'v1_1_bfcl.yaml':
            raise AssertionError('resume reread current v1.1 defaults')
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'read_text', forbid_defaults)
    assert runtime_config(c) == before
    assert cli.resume_config(None, dict(config=c, config_hash=digest(c))) == c


def test_arm_is_bound_into_manifest_and_campaign(tmp_path, monkeypatch):
    from bfas.rtd.unified.config import manifest_fields
    c = cli.load_config('configs/rtd/unified_bfcl_gemma4.yaml')
    manifest = dict(base_checkpoint_hash='issuer', tokenizer_hash='tokenizer', data_hash='bank',
                    hardware_hash='same-gpu', smoke=False)
    fields = []
    for arm in ARMS:
        selected, _ = arm_config(c, arm)
        row = manifest_fields(selected, manifest | {'config_hash': digest(selected)})
        assert row['distillation_protocol'] == arm and row['arm_components'] == selected['arm_preset']
        fields.append(row)
    assert len({r['campaign_identity'] for r in fields}) == len(ARMS)
    assert all(r['p1_matching'] == fields[0]['p1_matching'] for r in fields)
