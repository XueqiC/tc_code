"""C26-F before/after oracle captured before integration at the recorded commit.

Only fixture asset paths are relocated. Every manifest field except the run's
source inventory is compared, with real scoring projections through registry.
No production BFCL data, bank, model, GPU or active run is read.
"""
import hashlib
import json

import pytest

from bfas.rtd import cli, identity, scoring_scope
from bfas.rtd.benchmarks.registry import get_benchmark
from rtd_identity_fixtures import ROOT, hardware_fixture, put_tools


def legacy_namespace(module, frozen):
    """Execute verbatim pre-C26-F functions, without needing git at test time."""
    snapshot = frozen['source_functions'][module.__name__.rsplit('.', 1)[1]]
    assert hashlib.sha256(snapshot['source'].encode()).hexdigest() == snapshot['sha256']
    namespace = dict(vars(module))
    exec(compile(snapshot['source'], '<C26-F legacy oracle>', 'exec'), namespace)
    return namespace


@pytest.mark.parametrize('arm', ['R0', 'R1'])
@pytest.mark.parametrize('smoke', [False, True])
def test_bfcl_manifest_before_after_registry_every_field(tmp_path, monkeypatch, arm, smoke):
    frozen = json.loads((ROOT / 'tests/fixtures/rtd_bfcl_c26f_before.json').read_text())
    config = cli.load_config(ROOT / 'configs/rtd/v1_bfcl_c25.yaml')
    legacy_cli = legacy_namespace(cli, frozen)
    assert legacy_cli['load_config'](ROOT / 'configs/rtd/v1_bfcl_c25.yaml') == config
    if smoke:
        config = dict(config, smoke_override=dict(parents_per_fold=2, slots=2, rollouts=1,
            windows=1, baseline='action-independent zero', max_seconds=900))
    put_tools(tmp_path)
    files = {identity.LEADERBOARD + '/bfcl_eval/utils.py': 'fixture',
        identity.LEADERBOARD + '/bfcl_eval/data/tasks.json': '{}',
        'model/model.safetensors': 'fixture weights', 'model/tokenizer.json': '{}',
        'bank/public/requests.json': '[]', 'bank/sealed/integrity.json': '{}',
        config['support_manifest']: '{"parents":[]}'}
    for name, content in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    audit = dict(bank_path=str(tmp_path / 'bank'), bank_public_cap_sum=100,
        budget_ceilings=[10, 25, 50], recorded_bank_usage={}, available_packages=1, m=1)
    providers = get_benchmark(config)
    assert providers.harness_identity is identity.evaluation_harness_identity
    monkeypatch.setattr(cli, 'ROOT', tmp_path)
    monkeypatch.setattr(cli, 'hardware_identity', hardware_fixture)
    monkeypatch.setattr(cli, 'evaluation_harness_identity', providers.harness_identity)
    monkeypatch.setattr('tools.bfcl_hub_merge_export._snapshot_for_model', lambda _: tmp_path / 'model')
    legacy_scoring = legacy_namespace(scoring_scope, frozen)
    legacy_identity = legacy_namespace(identity, frozen)
    legacy_identity['scoring_hash'] = legacy_scoring['scoring_hash']
    legacy_cli.update(ROOT=tmp_path, hardware_identity=hardware_fixture,
        evaluation_harness_identity=legacy_identity['evaluation_harness_identity'])
    before = legacy_cli['make_manifest'](config, arm, audit, smoke=smoke)
    after = cli.make_manifest(config, arm, audit, smoke=smoke)
    assert before.keys() == after.keys()
    for key in before.keys() - {'rtd_source'}:
        assert before[key] == after[key], key
    assert after['rtd_source']['kind'] == 'rtd-source-v2'
    # Also retain the independently captured output oracle, so shared helper
    # changes cannot silently move both implementations together.
    if arm == 'R1' and not smoke:
        comparable = {k: v for k, v in after.items() if k != 'rtd_source'}
        relocated = json.loads(json.dumps(comparable).replace(str(tmp_path), '<ROOT>'))
        assert relocated == frozen['manifest']


def test_all_bfcl_scoring_projection_hashes_are_unchanged():
    frozen = json.loads((ROOT / 'tests/fixtures/rtd_bfcl_c26f_before.json').read_text())
    before = frozen['scoring_hashes']
    legacy = legacy_namespace(scoring_scope, frozen)
    assert set(before) == {*scoring_scope.PYTHON_SCOPES, scoring_scope.SHELL}
    provider = get_benchmark({}).scoring_projection
    for name, expected in before.items():
        content = (ROOT / name).read_text()
        assert legacy['scoring_hash'](name, content) == expected
        assert scoring_scope.benchmark_scoring_projection(name, content) == provider(name, content)
        assert scoring_scope.scoring_hash(name, content) == expected
        assert scoring_scope.scoring_hash(name, content, benchmark='bfcl') == expected
